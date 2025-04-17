import warnings

warnings.filterwarnings("ignore")  # Suppress all warnings

import json
import logging
import statistics
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import pyfolio as pf
import requests
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from pycoingecko import CoinGeckoAPI
from web3 import Web3

# Constants and mappings
SUPPORTED_POOL_TYPES = {
    "WEIGHTED": "Weighted",
    "COMPOSABLE_STABLE": "ComposableStable",
    "LIQUIDITY_BOOTSTRAPING": "LiquidityBootstrapping",
    "META_STABLE": "MetaStable",
    "STABLE": "Stable",
    "INVESTMENT": "Investment",
}
REQUIRED_FIELDS = (
    "chains",
    "graphql_endpoint",
    "current_positions",
    "coingecko_api_key",
)
BALANCER = "balancerPool"
FEE_RATE_DIVISOR = 1000000
DAYS_IN_YEAR = 365
PERCENT_CONVERSION = 100
TVL_WEIGHT = 0.7
APR_WEIGHT = 0.3
TVL_PERCENTILE = 50
APR_PERCENTILE = 50
SCORE_PERCENTILE = 80
CHAIN_URLS = {
    "mode": "https://1rpc.io/mode",
    "optimism": "https://mainnet.optimism.io",
    "base": "https://1rpc.io/base",
}

EXCLUDED_APR_TYPES = {"IB_YIELD", "MERKL", "SWAP_FEE", "SWAP_FEE_7D", "SWAP_FEE_30D"}
LP = "lp"
errors = []
WHITELISTED_ASSETS = {
    "mode": {},
    "optimism": {}
}

@lru_cache(None)
def fetch_coin_list():
    """Fetches the list of coins from CoinGecko API only once."""
    url = "https://api.coingecko.com/api/v3/coins/list"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        errors.append((f"Failed to fetch coin list: {e}"))
        return None


def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    return [field for field in REQUIRED_FIELDS if kwargs.get(field) is None]


def remove_irrelevant_fields(
    kwargs: Dict[str, Any], required_fields: Tuple
) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key in required_fields}


def run_query(query, graphql_endpoint, variables=None) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    payload = {"query": query, "variables": variables or {}}
    response = requests.post(graphql_endpoint, json=payload, headers=headers)
    if response.status_code != 200:
        return {
            "error": f"GraphQL query failed with status code {response.status_code}"
        }
    result = response.json()
    if "errors" in result:
        return {"error": f"GraphQL Errors: {result['errors']}"}
    return result["data"]


def get_total_apr(pool) -> float:
    apr_items = pool.get("dynamicData", {}).get("aprItems", [])
    return sum(
        item["apr"] for item in apr_items if item["type"] not in EXCLUDED_APR_TYPES
    )


@lru_cache(None)
def create_web3_connection(chain_name: str):
    chain_url = CHAIN_URLS.get(chain_name)
    if not chain_url:
        return None
    web3 = Web3(Web3.HTTPProvider(chain_url))
    return web3 if web3.is_connected() else None


@lru_cache(None)
def fetch_token_name_from_contract(chain_name, token_address):
    ERC20_ABI = [
        {
            "constant": True,
            "inputs": [],
            "name": "name",
            "outputs": [{"name": "", "type": "string"}],
            "payable": False,
            "stateMutability": "view",
            "type": "function",
        }
    ]
    web3 = create_web3_connection(chain_name)
    if not web3:
        return None
    contract = web3.eth.contract(
        address=Web3.to_checksum_address(token_address), abi=ERC20_ABI
    )
    try:
        return contract.functions.name().call()
    except:
        return None


def get_balancer_pools(
    chains, graphql_endpoint
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    chain_list_str = ", ".join(chain.upper() for chain in chains)
    graphql_query = f"""
    {{
      poolGetPools(where: {{chainIn: [{chain_list_str}]}} first: 100) {{
        id
        address
        chain
        type
        poolTokens {{
          address
          symbol
        }}
        dynamicData {{
          totalLiquidity
          aprItems {{
            type
            apr
          }}
        }}
      }}
    }}
    """
    data = run_query(graphql_query, graphql_endpoint)
    if "error" in data:
        return data
    return data.get("poolGetPools", [])


def get_filtered_pools(pools, current_positions):
    # Filter by type and token count
    qualifying_pools = []
    for pool in pools:
        mapped_type = SUPPORTED_POOL_TYPES.get(pool.get("type"))
        chain = pool["chain"].lower()
        whitelisted_tokens = WHITELISTED_ASSETS.get(chain, [])
        if (
            mapped_type
            and len(pool.get("poolTokens", [])) == 2
            and Web3.to_checksum_address(pool.get("address")) not in current_positions
            and chain in WHITELISTED_ASSETS
            and (
                not whitelisted_tokens
                or (
                    pool["poolTokens"][0]["address"] in whitelisted_tokens
                    and pool["poolTokens"][1]["address"] in whitelisted_tokens
                )
            )
        ):
            pool["type"] = mapped_type
            pool["apr"] = get_total_apr(pool)
            pool["tvl"] = pool.get("dynamicData", {}).get("totalLiquidity", 0)
            qualifying_pools.append(pool)

    
    tvl_list = [float(p.get("tvl", 0)) for p in qualifying_pools]
    apr_list = [float(p.get("apr", 0)) for p in qualifying_pools]

    tvl_threshold = np.percentile(tvl_list, TVL_PERCENTILE)
    apr_threshold = np.percentile(apr_list, APR_PERCENTILE)
    max_tvl = max(tvl_list) if tvl_list else 1
    max_apr = max(apr_list) if apr_list else 1

    # Score and filter
    scored_pools = []
    for p in qualifying_pools:
        tvl = float(p["tvl"])
        apr = float(p["apr"])
        if tvl >= tvl_threshold and apr >= apr_threshold:
            score = TVL_WEIGHT * (tvl / max_tvl) + APR_WEIGHT * (apr / max_apr)
            p["score"] = score
            scored_pools.append(p)

    if not scored_pools:
        return []

    score_threshold = np.percentile(
        [p["score"] for p in scored_pools], SCORE_PERCENTILE
    )
    filtered_scored_pools = [p for p in scored_pools if p["score"] >= score_threshold]
    filtered_scored_pools.sort(key=lambda x: x["score"], reverse=True)

    return filtered_scored_pools[:20]

def get_token_id_from_symbol_cached(symbol, token_name, coin_list):
    # Try to find a coin matching symbol first.
    candidates = [
        coin for coin in coin_list if coin["symbol"].lower() == symbol.lower()
    ]
    if not candidates:
        return None

    # If single candidate, return it
    if len(candidates) == 1:
        return candidates[0]["id"]

    # If multiple candidates, match by name if possible
    normalized_token_name = token_name.replace(" ", "").lower()
    for coin in candidates:
        coin_name = coin["name"].replace(" ", "").lower()
        if coin_name == normalized_token_name or coin_name == symbol.lower():
            return coin["id"]
    return None


def get_token_id_from_symbol(token_address, symbol, coin_list, chain_name):
    token_name = fetch_token_name_from_contract(chain_name, token_address)
    if not token_name:
        matching_coins = [
            coin for coin in coin_list if coin["symbol"].lower() == symbol.lower()
        ]
        return matching_coins[0]["id"] if len(matching_coins) == 1 else None

    return get_token_id_from_symbol_cached(symbol, token_name, coin_list)


def calculate_il_impact(P0, P1):
    # Impermanent Loss impact calculation
    return 2 * np.sqrt(P1 / P0) / (1 + P1 / P0) - 1


def calculate_il_risk_score(
    token_0, token_1, coingecko_api_key: str, time_period: int = 90
) -> float:
    is_pro = is_pro_api_key(coingecko_api_key)
    if is_pro:
        cg = CoinGeckoAPI(api_key=coingecko_api_key)
    else:
        cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=time_period)).timestamp())
    try:
        prices_1 = cg.get_coin_market_chart_range_by_id(
            id=token_0,
            vs_currency="usd",
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )
        prices_2 = cg.get_coin_market_chart_range_by_id(
            id=token_1,
            vs_currency="usd",
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )
    except Exception as e:
        errors.append(f"Error fetching price data: {e}")
        return None
    
    prices_1_data = np.array([x[1] for x in prices_1["prices"]])
    prices_2_data = np.array([x[1] for x in prices_2["prices"]])
    min_length = min(len(prices_1_data), len(prices_2_data))
    prices_1_data = prices_1_data[:min_length]
    prices_2_data = prices_2_data[:min_length]

    if min_length < 2:
        return None

    price_correlation = np.corrcoef(prices_1_data, prices_2_data)[0, 1]
    volatility_1 = np.std(prices_1_data)
    volatility_2 = np.std(prices_2_data)
    volatility_multiplier = np.sqrt(volatility_1 * volatility_2)
    P0 = prices_1_data[0] / prices_2_data[0]
    P1 = prices_1_data[-1] / prices_2_data[-1]
    il_impact = calculate_il_impact(P0, P1)

    return float(il_impact * abs(price_correlation) * volatility_multiplier)


def create_graphql_client(api_url="https://api-v3.balancer.fi") -> Client:
    transport = RequestsHTTPTransport(url=api_url, verify=True, retries=3)
    return Client(transport=transport, fetch_schema_from_transport=False)


def create_pool_snapshots_query(
    pool_id: str, chain: str, range: str = "NINETY_DAYS"
) -> gql:
    return gql(
        f"""
    query GetLiquidityMetrics {{
      poolGetSnapshots(
        id: "{pool_id}",
        range: {range},
        chain: {chain}
      ) {{
        totalLiquidity
        volume24h
        timestamp
      }}
    }}
    """
    )


def fetch_liquidity_metrics(
    pool_id: str,
    chain: str,
    client: Optional[Client] = None,
    price_impact: float = 0.01,
) -> Optional[Dict[str, Any]]:
    if client is None:
        client = create_graphql_client()
    try:
        query = create_pool_snapshots_query(pool_id, chain)
        response = client.execute(query)
        pool_snapshots = response["poolGetSnapshots"]
        if not pool_snapshots:
            return None

        avg_tvl = statistics.mean(float(s["totalLiquidity"]) for s in pool_snapshots)
        avg_volume = statistics.mean(
            float(s.get("volume24h", 0)) for s in pool_snapshots
        )

        depth_score = (
            (np.log1p(avg_tvl) * np.log1p(avg_volume)) / (price_impact * 100)
            if avg_tvl and avg_volume
            else 0
        )
        liquidity_risk_multiplier = (
            max(0, 1 - (1 / depth_score)) if depth_score != 0 else 0
        )
        max_position_size = 50 * (avg_tvl * liquidity_risk_multiplier) / 100

        return {
            "Average TVL": avg_tvl,
            "Average Daily Volume": avg_volume,
            "Depth Score": depth_score,
            "Liquidity Risk Multiplier": liquidity_risk_multiplier,
            "Maximum Position Size": max_position_size,
            "Meets Depth Score Threshold": depth_score > 50,
        }

    except Exception:
        return None


def analyze_pool_liquidity(
    pool_id: str,
    chain: str,
    client: Optional[Client] = None,
    price_impact: float = 0.01,
):
    metrics = fetch_liquidity_metrics(pool_id, chain, client, price_impact)
    if metrics is None:
        errors.append("Could not retrieve depth score and maximum position size.")
        return float("nan"), float("nan")
    return metrics["Depth Score"], metrics["Maximum Position Size"]


def get_balancer_pool_sharpe_ratio(pool_id, chain, timerange="ONE_YEAR"):
    query = """
    {
        poolGetSnapshots(
            chain: %s
            id: "%s"
            range: %s
        ) {
            timestamp
            sharePrice
            fees24h
            totalLiquidity
        }
    }
    """ % (
        chain,
        pool_id,
        timerange,
    )
    try:
        response = requests.post("https://api-v3.balancer.fi/", json={"query": query})
        data = response.json()["data"]["poolGetSnapshots"]
        if not data:
            return None

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)

        df["sharePrice"] = pd.to_numeric(df["sharePrice"])
        price_returns = df["sharePrice"].pct_change()

        df["fees24h"] = pd.to_numeric(df["fees24h"])
        df["totalLiquidity"] = pd.to_numeric(df["totalLiquidity"])
        fee_returns = df["fees24h"] / df["totalLiquidity"]
        total_returns = (price_returns + fee_returns).dropna()
        total_returns = total_returns.replace([np.inf, -np.inf], np.nan)
        sharpe_ratio = pf.timeseries.sharpe_ratio(total_returns)
        return sharpe_ratio
    except Exception as e:
        errors.append(f"Error calculating Sharpe ratio: {e}")
        return None


def format_pool_data(pool) -> Dict[str, Any]:
    dex_type = BALANCER
    return {
        "dex_type": dex_type,
        "chain": pool["chain"].lower(),
        "apr": pool["apr"] * 100,
        "pool_address": pool["address"],
        "pool_id": pool["id"],
        "pool_type": pool["type"],
        "token0": pool["poolTokens"][0]["address"],
        "token1": pool["poolTokens"][1]["address"],
        "token0_symbol": pool["poolTokens"][0]["symbol"],
        "token1_symbol": pool["poolTokens"][1]["symbol"],
        "il_risk_score": pool["il_risk_score"],
        "sharpe_ratio": pool["sharpe_ratio"],
        "depth_score": pool["depth_score"],
        "max_position_size": pool["max_position_size"],
        "type": pool["trading_type"]
    }


def get_opportunities(
    chains, graphql_endpoint, current_positions, coingecko_api_key, coin_list
):
    pools = get_balancer_pools(chains, graphql_endpoint)
    if isinstance(pools, dict) and "error" in pools:
        return pools
    filtered_pools = get_filtered_pools(pools, current_positions)
    if not filtered_pools:
        return {"error": "No suitable pools found"}
    
    token_id_cache = {}
    for pool in filtered_pools:
        pool["chain"] = pool["chain"].lower()

        # Token 0
        t0_sym = pool["poolTokens"][0]["symbol"].lower()
        if t0_sym not in token_id_cache:
            token_0_id = get_token_id_from_symbol(
                pool["poolTokens"][0]["address"], t0_sym, coin_list, pool["chain"]
            )
            if token_0_id:
                token_id_cache[t0_sym] = token_0_id
        else:
            token_0_id = token_id_cache[t0_sym]

        # Token 1
        t1_sym = pool["poolTokens"][1]["symbol"].lower()
        if t1_sym not in token_id_cache:
            token_1_id = get_token_id_from_symbol(
                pool["poolTokens"][1]["address"], t1_sym, coin_list, pool["chain"]
            )
            if token_1_id:
                token_id_cache[t1_sym] = token_1_id
        else:
            token_1_id = token_id_cache[t1_sym]

        if token_0_id and token_1_id:
            pool["il_risk_score"] = calculate_il_risk_score(
                token_0_id, token_1_id, coingecko_api_key
            )
        else:
            pool["il_risk_score"] = None

        pool["sharpe_ratio"] = get_balancer_pool_sharpe_ratio(
            pool["id"], pool["chain"].upper()
        )
        pool["depth_score"], pool["max_position_size"] = analyze_pool_liquidity(
            pool["id"], pool["chain"].upper()
        )

        pool["trading_type"] = LP
    return [format_pool_data(pool) for pool in filtered_pools]


def calculate_metrics(
    position: Dict[str, Any], coingecko_api_key: str, coin_list: List[Any], **kwargs
) -> Optional[Dict[str, Any]]:
    token_0_id = get_token_id_from_symbol(
        position["token0"], position["token0_symbol"], coin_list, position["chain"]
    )
    token_1_id = get_token_id_from_symbol(
        position["token1"], position["token1_symbol"], coin_list, position["chain"]
    )
    il_risk_score = (
        calculate_il_risk_score(token_0_id, token_1_id, coingecko_api_key)
        if token_0_id and token_1_id
        else None
    )
    sharpe_ratio = get_balancer_pool_sharpe_ratio(
        position["pool_id"], position["chain"].upper()
    )
    depth_score, max_position_size = analyze_pool_liquidity(
        position["pool_id"], position["chain"].upper()
    )
    return {
        "il_risk_score": il_risk_score,
        "sharpe_ratio": sharpe_ratio,
        "depth_score": depth_score,
        "max_position_size": max_position_size,
    }

def is_pro_api_key(coingecko_api_key: str) -> bool:
    """
    Check if the provided CoinGecko API key is a pro key.
    """
    # Try using the key as a pro API key
    cg_pro = CoinGeckoAPI(api_key=coingecko_api_key)
    try:
        response = cg_pro.get_coin_market_chart_range_by_id(
            id="bitcoin",
            vs_currency="usd",
            from_timestamp=0,
            to_timestamp=0
        )
        if response:
            return True
    except Exception:
        return False

    return False

def run(*_args, **kwargs) -> Dict[str, Union[bool, str, List[str]]]:
    missing = check_missing_fields(kwargs)
    if missing:
        errors.append(f"Required kwargs {missing} were not provided.")
        return {"error": errors}

    required_fields = list(REQUIRED_FIELDS)
    get_metrics = kwargs.get("get_metrics", False)
    if get_metrics:
        required_fields.append("position")

    kwargs = remove_irrelevant_fields(kwargs, required_fields)

    coin_list = fetch_coin_list()
    if coin_list is None:
        errors.append("Failed to fetch coin list.")
        return {"error": errors}

    kwargs.update({"coin_list": coin_list})

    if get_metrics:
        metrics = calculate_metrics(**kwargs)
        if metrics is None:
            errors.append("Failed to calculate metrics.")
        return {"error": errors} if errors else metrics
    else:
        result = get_opportunities(**kwargs)
        if isinstance(result, dict) and "error" in result:
            errors.append(result["error"])
        if not result:
            errors.append("No suitable aggregators found")
        return {"error": errors} if errors else {"result": result}
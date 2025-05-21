import warnings

warnings.filterwarnings("ignore")  # Suppress all warnings

import json
import time
import statistics
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyfolio as pf
import requests
from pycoingecko import CoinGeckoAPI
from web3 import Web3

# Constants and mappings
UNISWAP = "UniswapV3"
REQUIRED_FIELDS = (
    "chains",
    "graphql_endpoints",
    "current_positions",
    "coingecko_api_key",
)
FEE_RATE_DIVISOR = 1000000
DAYS_IN_YEAR = 365
PERCENT_CONVERSION = 100
TVL_WEIGHT = 0.7
APR_WEIGHT = 0.3
TVL_PERCENTILE = 50
APR_PERCENTILE = 50
SCORE_PERCENTILE = 80
PRICE_IMPACT = 0.01  # 1% standard price impact
MAX_POSITION_BASE = 50  # Base for maximum position calculation
CHAIN_URLS = {
    "mode": "https://1rpc.io/mode",
    "optimism": "https://mainnet.optimism.io",
    "base": "https://1rpc.io/base",
}

LP = "lp"
errors = []
WHITELISTED_ASSETS = {
    "base": [],
    "optimism": [
        "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",  # USDC
        "0xCB8FA9a76b8e203D8C3797bF438d8FB81Ea3326A",  # alUSD
        "0x01bFF41798a0BcF287b996046Ca68b395DbC1071",  # USDT0
        "0x94b008aA00579c1307B0EF2c499aD98a8ce58e58",  # USDT
        "0x9dAbAE7274D28A45F0B65Bf8ED201A5731492ca0",  # msUSD
        "0x7F5c764cBc14f9669B88837ca1490cCa17c31607",  # USDC.e
        "0xbfD291DA8A403DAAF7e5E9DC1ec0aCEaCd4848B9",  # USX
        "0x8aE125E8653821E851F12A49F7765db9a9ce7384",  # DOLA
        "0xc40F949F8a4e094D1b49a23ea9241D289B7b2819",  # LUSD
        "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1",  # DAI
        "0x087C440F251Ff6Cfe62B86DdE1bE558B95b4bb9b",  # BOLD
        "0x2E3D870790dC77A83DD1d18184Acc7439A53f475",  # FRAX
        "0x2218a117083f5B482B0bB821d27056Ba9c04b1D3",  # sDAI
        "0x73cb180bf0521828d8849bc8CF2B920918e23032",  # USD+
        "0x1217BfE6c773EEC6cc4A38b5Dc45B92292B6E189",  # oUSDT
        "0x4F604735c1cF31399C6E711D5962b2B3E0225AD3"   # USDGLO
    ]
}

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


def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    return [field for field in REQUIRED_FIELDS if kwargs.get(field) is None]


def remove_irrelevant_fields(
    kwargs: Dict[str, Any], required_fields: Tuple
) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key in required_fields}


@lru_cache(None)
def fetch_coin_list():
    """Fetches the list of coins from CoinGecko API only once."""
    url = "https://api.coingecko.com/api/v3/coins/list"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        errors.append(f"Failed to fetch coin list: {e}")
        return None


@lru_cache(None)
def create_web3_connection(chain_name: str):
    chain_url = CHAIN_URLS.get(chain_name)
    if not chain_url:
        return None
    web3 = Web3(Web3.HTTPProvider(chain_url))
    return web3 if web3.is_connected() else None


@lru_cache(None)
def fetch_token_name_from_contract(chain_name, token_address):
    web3 = create_web3_connection(chain_name)
    if not web3:
        return None
    contract = web3.eth.contract(
        address=Web3.to_checksum_address(token_address), abi=ERC20_ABI
    )
    try:
        return contract.functions.name().call()
    except Exception:
        return None


def get_token_id_from_symbol(token_address, symbol, coin_list, chain_name):
    matching_coins = [c for c in coin_list if c["symbol"].lower() == symbol.lower()]
    if not matching_coins:
        return None
    if len(matching_coins) == 1:
        return matching_coins[0]["id"]

    token_name = fetch_token_name_from_contract(chain_name, token_address)
    if not token_name:
        return None

    for coin in matching_coins:
        # Match after removing spaces and lowering case
        if (
            coin["name"].replace(" ", "").lower() == token_name.replace(" ", "").lower()
            or coin["name"].lower() == symbol.lower()
        ):
            return coin["id"]
    return None


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
    return result.get("data", {})


def calculate_apr(daily_volume: float, tvl: float, fee_rate: float) -> float:
    """Calculate APR: (Daily Volume / TVL) × Fee Rate × 365 × 100"""
    return (
        0
        if tvl == 0
        else (daily_volume / tvl) * fee_rate * DAYS_IN_YEAR * PERCENT_CONVERSION
    )


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


def get_filtered_pools(pools, current_positions) -> List[Dict[str, Any]]:
    qualifying_pools = []
    for pool in pools:
        fee_rate = float(pool["feeTier"]) / FEE_RATE_DIVISOR
        tvl = float(pool["totalValueLockedUSD"])
        daily_volume = float(pool["volumeUSD"])
        apr = calculate_apr(daily_volume, tvl, fee_rate)
        pool["apr"] = apr
        pool["tvl"] = tvl
        
        chain = pool["chain"].lower()
        whitelisted_tokens = WHITELISTED_ASSETS.get(chain, [])
        
        if (Web3.to_checksum_address(pool["id"]) not in current_positions
            and chain in WHITELISTED_ASSETS
            and (
                not whitelisted_tokens
                or (
                    pool["token0"]["id"] in whitelisted_tokens
                    and pool["token1"]["id"] in whitelisted_tokens
                )
            )
        ):
            qualifying_pools.append(pool)

    if not qualifying_pools:
        errors.append("No suitable pools found.")
        return []

    if len(qualifying_pools) <= 5:
        return qualifying_pools

    tvl_list = [p["tvl"] for p in qualifying_pools]
    apr_list = [p["apr"] for p in qualifying_pools]

    tvl_threshold = np.percentile(tvl_list, TVL_PERCENTILE)
    apr_threshold = np.percentile(apr_list, APR_PERCENTILE)

    max_tvl = max(tvl_list)
    max_apr = max(apr_list)

    scored_pools = []
    for pool in qualifying_pools:
        tvl = pool["tvl"]
        apr = pool["apr"]
        if tvl < tvl_threshold or apr < apr_threshold:
            continue
        score = TVL_WEIGHT * (tvl / max_tvl) + APR_WEIGHT * (apr / max_apr)
        pool["score"] = score
        scored_pools.append(pool)

    if not scored_pools:
        return []

    score_threshold = np.percentile(
        [p["score"] for p in scored_pools], SCORE_PERCENTILE
    )
    filtered_scored_pools = [p for p in scored_pools if p["score"] >= score_threshold]
    filtered_scored_pools.sort(key=lambda x: x["score"], reverse=True)

    return (
        filtered_scored_pools[:20]
        if len(filtered_scored_pools) > 10
        else filtered_scored_pools
    )


def fetch_graphql_data(
    chains, graphql_endpoints
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    graphql_query = """
    {
        pools(
            first: 100,
            orderBy: totalValueLockedUSD,
            orderDirection: desc,
            subgraphError: allow
        ) {
            id
            feeTier
            liquidity
            volumeUSD
            totalValueLockedUSD
            token0 {
                id
                symbol
                decimals
                derivedETH
            }
            token1 {
                id
                symbol
                decimals
                derivedETH
            }
        }
        bundles(where: {id: "1"}) {
            ethPriceUSD
        }
    }
    """
    all_pools = []
    for chain in chains:
        graphql_endpoint = graphql_endpoints.get(chain)
        if not graphql_endpoint:
            continue
        data = run_query(graphql_query, graphql_endpoint)
        if "error" in data:
            errors.append(f"Error in fetching pools data: {data['error']}")
            continue
        pools = data.get("pools", [])
        for p in pools:
            p["chain"] = chain
        all_pools.extend(pools)
    return all_pools


def get_uniswap_pool_sharpe_ratio(
    pool_address, graphql_endpoint, days_back=365
) -> float:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    start_timestamp = int(start_date.timestamp())

    query = f"""
    {{
      poolDayDatas(
        where: {{
          pool: "{pool_address.lower()}"
          date_gt: {start_timestamp}
        }}
        orderBy: date
        orderDirection: asc
      ) {{
        date
        tvlUSD
        feesUSD
      }}
    }}
    """
    response = requests.post(graphql_endpoint, json={"query": query})
    data = response.json().get("data", {}).get("poolDayDatas", [])
    if not data:
        return float("nan")

    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(df["date"].astype(int), unit="s")
    df["tvlUSD"] = pd.to_numeric(df["tvlUSD"])
    df["feesUSD"] = pd.to_numeric(df["feesUSD"])
    df["total_value"] = df["tvlUSD"] + df["feesUSD"]
    returns = (
        df.set_index("date")["total_value"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    return float(pf.timeseries.sharpe_ratio(returns))


def calculate_il_impact(P0, P1):
    # Impermanent Loss impact calculation
    return 2 * np.sqrt(P1 / P0) / (1 + P1 / P0) - 1


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


def fetch_pool_data(pool_id: str, SUBGRAPH_URL: str) -> Optional[Dict[str, Any]]:
    query = {
        "query": f"""
        {{
          pool(id: "{pool_id.lower()}") {{
            id
            token0 {{
              id
              symbol
              decimals
            }}
            token1 {{
              id
              symbol
              decimals
            }}
            liquidity
            totalValueLockedUSD
            totalValueLockedToken0
            totalValueLockedToken1
          }}
        }}
        """
    }
    try:
        response = requests.post(
            SUBGRAPH_URL, json=query, headers={"Content-Type": "application/json"}
        )
        response_json = response.json()
        if response.status_code == 200 and "data" in response_json:
            return response_json["data"].get("pool")
    except Exception as e:
        errors.append(f"Error fetching pool data: {e}")
    return None


def calculate_metrics_liquidity_risk(pool_data: Dict[str, Any]) -> Tuple[float, float]:
    try:
        tvl = float(pool_data.get("totalValueLockedUSD", 0))
        tvl_token0 = float(pool_data.get("totalValueLockedToken0", 0))
        tvl_token1 = float(pool_data.get("totalValueLockedToken1", 0))

        depth_score = (
            (np.log1p(tvl_token0) * np.log1p(tvl_token1)) / (PRICE_IMPACT * 100)
            if tvl_token0 > 0 and tvl_token1 > 0
            else 0
        )
        liquidity_risk_multiplier = (
            max(0, 1 - (1 / depth_score)) if depth_score > 0 else 0
        )
        max_position_size = MAX_POSITION_BASE * (tvl * liquidity_risk_multiplier) / 100
        return depth_score, max_position_size
    except Exception as e:
        errors.append(f"Error calculating metrics: {e}")
        return float("nan"), float("nan")


def assess_pool_liquidity(pool_id: str, SUBGRAPH_URL: str) -> Tuple[float, float]:
    pool_data = fetch_pool_data(pool_id, SUBGRAPH_URL)
    if pool_data is None:
        return float("nan"), float("nan")
    return calculate_metrics_liquidity_risk(pool_data)


def format_pool_data(pool) -> Dict[str, Any]:
    return {
        "dex_type": UNISWAP,
        "chain": pool["chain"].lower(),
        "apr": pool["apr"],
        "pool_address": pool["id"],
        "token0": pool["token0"]["id"],
        "token1": pool["token1"]["id"],
        "token0_symbol": pool["token0"]["symbol"],
        "token1_symbol": pool["token1"]["symbol"],
        "il_risk_score": pool["il_risk_score"],
        "sharpe_ratio": pool["sharpe_ratio"],
        "depth_score": pool["depth_score"],
        "max_position_size": pool["max_position_size"],
        "type": pool["type"]
    }


def get_opportunities(
    chains, graphql_endpoints, current_positions, coingecko_api_key, coin_list
) -> List[Dict[str, Any]]:
    pools = fetch_graphql_data(chains, graphql_endpoints)
    if isinstance(pools, dict) and "error" in pools:
        return pools

    filtered_pools = get_filtered_pools(pools, current_positions)
    if not filtered_pools:
        return {"error": "No suitable pools found"}

    token_id_cache = {}
    for pool in filtered_pools:
        pool_chain = pool["chain"].lower()
        token_0_symbol = pool["token0"]["symbol"].lower()
        token_1_symbol = pool["token1"]["symbol"].lower()

        # Token 0
        if token_0_symbol not in token_id_cache:
            token_0_id = get_token_id_from_symbol(
                pool["token0"]["id"], token_0_symbol, coin_list, pool_chain
            )
            if token_0_id:
                token_id_cache[token_0_symbol] = token_0_id
        else:
            token_0_id = token_id_cache[token_0_symbol]

        # Token 1
        if token_1_symbol not in token_id_cache:
            token_1_id = get_token_id_from_symbol(
                pool["token1"]["id"], token_1_symbol, coin_list, pool_chain
            )
            if token_1_id:
                token_id_cache[token_1_symbol] = token_1_id
        else:
            token_1_id = token_id_cache[token_1_symbol]

        if token_0_id and token_1_id:
            pool["il_risk_score"] = calculate_il_risk_score(
                token_0_id, token_1_id, coingecko_api_key
            )
        else:
            pool["il_risk_score"] = None
        graphql_endpoint = graphql_endpoints.get(pool_chain)
        pool["sharpe_ratio"] = get_uniswap_pool_sharpe_ratio(
            pool["id"], graphql_endpoint
        )
        depth_score, max_position_size = assess_pool_liquidity(
            pool["id"], graphql_endpoint
        )
        pool["depth_score"] = depth_score
        pool["max_position_size"] = max_position_size
        pool["type"] = LP

    return [format_pool_data(pool) for pool in filtered_pools]


def calculate_metrics(
    position: Dict[str, Any],
    coingecko_api_key: str,
    coin_list: List[Any],
    graphql_endpoints,
    **kwargs,
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
    graphql_endpoint = graphql_endpoints.get(position["chain"])
    sharpe_ratio = get_uniswap_pool_sharpe_ratio(
        position["pool_address"], graphql_endpoint
    )
    depth_score, max_position_size = assess_pool_liquidity(
        position["pool_address"], graphql_endpoint
    )

    return {
        "il_risk_score": il_risk_score,
        "sharpe_ratio": sharpe_ratio,
        "depth_score": depth_score,
        "max_position_size": max_position_size,
    }


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

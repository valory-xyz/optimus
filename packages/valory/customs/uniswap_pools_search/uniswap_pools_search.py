import warnings

from packages.valory.connections.x402.clients.requests import x402_requests

warnings.filterwarnings("ignore")  # Suppress all warnings

import json
import logging
import time
import threading
import statistics
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyfolio as pf
import requests
from aea.helpers.logging import setup_logger
from pycoingecko import CoinGeckoAPI
from web3 import Web3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_logger = setup_logger(__name__)

# Constants and mappings
UNISWAP = "UniswapV3"
REQUIRED_FIELDS = (
    "chains",
    "graphql_endpoints",
    "current_positions",
    "whitelisted_assets",
)
FEE_RATE_DIVISOR = 1000000
DAYS_IN_YEAR = 365
PERCENT_CONVERSION = 100
TVL_WEIGHT = 0.3
APR_WEIGHT = 0.7
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

# Thread-local storage for errors
_thread_local = threading.local()

def get_errors():
    """Get thread-local error list."""
    if not hasattr(_thread_local, 'errors'):
        _thread_local.errors = []
    return _thread_local.errors

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
    missing = [field for field in REQUIRED_FIELDS if kwargs.get(field) is None]
    if missing:
        logger.warning(f"Missing required fields: {missing}")
    return missing


def remove_irrelevant_fields(
    kwargs: Dict[str, Any], required_fields: Tuple
) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key in required_fields}

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

def get_coin_id_from_symbol(
        coin_id_mapping, symbol, chain_name
    ) -> Optional[str]:
        """Retrieve the CoinGecko token ID using the token's address, symbol, and chain name."""
        # Check if coin_list is valid
        symbol = symbol.lower()
        if symbol in coin_id_mapping.get(chain_name, {}):
            return coin_id_mapping[chain_name][symbol]

        return None


def run_query(query, graphql_endpoint, variables=None) -> Dict[str, Any]:
    logger.info(f"Running GraphQL query to endpoint: {graphql_endpoint}")
    headers = {"Content-Type": "application/json"}
    payload = {"query": query, "variables": variables or {}}
    response = requests.post(graphql_endpoint, json=payload, headers=headers)
    if response.status_code != 200:
        logger.error(f"GraphQL query failed with status code {response.status_code}")
        return {
            "error": f"GraphQL query failed with status code {response.status_code}"
        }
    result = response.json()
    if "errors" in result:
        logger.error(f"GraphQL Errors: {result['errors']}")
        return {"error": f"GraphQL Errors: {result['errors']}"}
    logger.info("GraphQL query executed successfully")
    return result.get("data", {})


def calculate_apr(daily_volume: float, tvl: float, fee_rate: float) -> float:
    """Calculate APR: (Daily Volume / TVL) × Fee Rate × 365 × 100"""
    return (
        0
        if tvl == 0
        else (daily_volume / tvl) * fee_rate * DAYS_IN_YEAR * PERCENT_CONVERSION
    )


def standardize_metrics(pools, apr_weight=0.7, tvl_weight=0.3):
    """
    Standardize APR and TVL using Z-score normalization and calculate composite scores.
    
    Args:
        pools: List of pool dictionaries
        apr_weight: Weight for APR in composite score (0-1)
        tvl_weight: Weight for TVL in composite score (0-1)
    
    Returns:
        List of pools with added standardized metrics and composite scores
    """
    if not pools:
        return pools
    
    # Extract APR and TVL values with proper type conversion
    aprs = []
    tvls = []
    
    for pool in pools:
        try:
            apr_value = float(pool.get('apr', 0))
            tvl_value = float(pool.get('tvl', 0))
            aprs.append(apr_value)
            tvls.append(tvl_value)
        except (ValueError, TypeError):
            # Use 0 for invalid values
            aprs.append(0.0)
            tvls.append(0.0)
    
    # Calculate means and standard deviations
    apr_mean = np.mean(aprs) if aprs else 0
    apr_std = np.std(aprs) if aprs else 1
    tvl_mean = np.mean(tvls) if tvls else 0
    tvl_std = np.std(tvls) if tvls else 1
    
    # Avoid division by zero
    if apr_std == 0:
        apr_std = 1
    if tvl_std == 0:
        tvl_std = 1
    
    # Standardize metrics and calculate composite scores
    for i, pool in enumerate(pools):
        apr = aprs[i]
        tvl = tvls[i]
        
        # Z-score normalization
        pool['apr_standardized'] = (apr - apr_mean) / apr_std
        pool['tvl_standardized'] = (tvl - tvl_mean) / tvl_std
        
        # Composite score with configurable weights
        pool['composite_score'] = (
            pool['apr_standardized'] * apr_weight + 
            pool['tvl_standardized'] * tvl_weight
        )
    
    return pools


def apply_composite_pre_filter(pools, top_n=10, apr_weight=0.7, tvl_weight=0.3, 
                              min_tvl_threshold=1000, use_composite_filter=True):
    """
    Apply composite pre-filtering to select top pools based on standardized APR and TVL.
    
    Args:
        pools: List of pool dictionaries
        top_n: Number of top pools to select
        apr_weight: Weight for APR in composite score
        tvl_weight: Weight for TVL in composite score
        min_tvl_threshold: Minimum TVL threshold for inclusion
        use_composite_filter: Whether to use composite filtering
    
    Returns:
        List of filtered and ranked pools
    """
    if not pools or not use_composite_filter:
        return pools[:top_n] if pools else []
    
    # Filter by minimum TVL threshold with proper type conversion
    tvl_filtered = []
    for pool in pools:
        try:
            tvl_value = float(pool.get('tvl', 0))
            if tvl_value >= float(min_tvl_threshold):
                tvl_filtered.append(pool)
        except (ValueError, TypeError):
            # Skip pools with invalid TVL values
            continue
    
    if not tvl_filtered:
        return []
    
    # Apply standardization and composite scoring
    standardized_pools = standardize_metrics(tvl_filtered, apr_weight, tvl_weight)
    
    # Sort by composite score (descending)
    standardized_pools.sort(key=lambda x: x.get('composite_score', float('-inf')), reverse=True)
    
    # Select top N pools
    final_selection = standardized_pools[:top_n]
    
    return final_selection

def get_filtered_pools_for_uniswap(pools, current_positions, whitelisted_assets, **kwargs) -> List[Dict[str, Any]]:
    logger.info(f"Filtering Uniswap pools - Total pools: {len(pools)}")
    logger.info(f"Current positions to exclude: {current_positions}")
    
    # Extract composite filtering parameters
    top_n = kwargs.get('top_n', 10)
    apr_weight = kwargs.get('apr_weight', 0.7)
    tvl_weight = kwargs.get('tvl_weight', 0.3)
    min_tvl_threshold = kwargs.get('min_tvl_threshold', 1000)
    
    logger.info(f"Filtering parameters: top_n={top_n}, apr_weight={apr_weight}, tvl_weight={tvl_weight}, min_tvl_threshold={min_tvl_threshold}")

    qualifying_pools = []
    for pool in pools:
        fee_rate = float(pool["feeTier"]) / FEE_RATE_DIVISOR
        tvl = float(pool["totalValueLockedUSD"])
        daily_volume = float(pool["volumeUSD"])
        apr = calculate_apr(daily_volume, tvl, fee_rate)
        pool["apr"] = apr
        pool["tvl"] = tvl
        
        chain = pool["chain"].lower()
        whitelisted_tokens = list(whitelisted_assets.get(chain, {}).keys())
        
        if (Web3.to_checksum_address(pool["id"]) not in current_positions
            and chain in whitelisted_assets
            and (
                not whitelisted_tokens
                or (
                    pool["token0"]["id"] in whitelisted_tokens
                    and pool["token1"]["id"] in whitelisted_tokens
                )
            )
        ):
            qualifying_pools.append(pool)
            logger.info(f"Added qualifying pool: {pool['id']} with APR: {apr:.2f}%, TVL: ${tvl:,.0f}")

    logger.info(f"After initial filtering: {len(qualifying_pools)} qualifying pools")

    if not qualifying_pools:
        logger.warning("No suitable pools found after initial filtering")
        get_errors().append("No suitable pools found.")
        return []

    # Apply composite pre-filtering
    filtered_pools = apply_composite_pre_filter(
        qualifying_pools,
        top_n=top_n,
        apr_weight=apr_weight,
        tvl_weight=tvl_weight,
        min_tvl_threshold=min_tvl_threshold,
    )

    logger.info(f"After composite filtering: {len(filtered_pools)} pools selected")
    return filtered_pools


def fetch_graphql_data(
    chains, graphql_endpoints
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    logger.info(f"Fetching GraphQL data for chains: {chains}")
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
        logger.info(f"Fetching pools for chain: {chain}")
        graphql_endpoint = graphql_endpoints.get(chain)
        if not graphql_endpoint:
            logger.warning(f"No GraphQL endpoint found for chain: {chain}")
            continue
        data = run_query(graphql_query, graphql_endpoint)
        if "error" in data:
            logger.error(f"Error fetching pools data for {chain}: {data['error']}")
            get_errors().append(f"Error in fetching pools data: {data['error']}")
            continue
        pools = data.get("pools", [])
        logger.info(f"Fetched {len(pools)} pools for chain {chain}")
        for p in pools:
            p["chain"] = chain
        all_pools.extend(pools)
    
    logger.info(f"Total pools fetched across all chains: {len(all_pools)}")
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
    token_0, token_1, coingecko_api_key: str, x402_signer: Optional[str] = None, x402_proxy: Optional[str] = None, time_period: int = 90
) -> float:
    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=time_period)).timestamp())
    try:
        cg = CoinGeckoAPI()
        if x402_signer is not None and x402_proxy is not None:
            logger.info("Using x402 signer for CoinGecko API requests")
            cg.session = x402_requests(account=x402_signer)
            cg.api_base_url = x402_proxy.rstrip("/") + "/api/v3/"
        else:
            if not coingecko_api_key:
                return None
            is_pro = is_pro_api_key(coingecko_api_key)
            if is_pro:
                cg = CoinGeckoAPI(api_key=coingecko_api_key)
            else:
                cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
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
        get_errors().append(f"Error fetching price data: {e}")
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
        get_errors().append(f"Error fetching pool data: {e}")
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
        get_errors().append(f"Error calculating metrics: {e}")
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


def get_opportunities_for_uniswap(
    chains, graphql_endpoints, current_positions, coingecko_api_key, whitelisted_assets, coin_id_mapping, x402_signer=None, x402_proxy=None, **kwargs
) -> List[Dict[str, Any]]:
    logger.info(f"Getting Uniswap opportunities for chains: {chains}")
    logger.info(f"Current positions to exclude: {current_positions}")
    
    pools = fetch_graphql_data(chains, graphql_endpoints)
    if isinstance(pools, dict) and "error" in pools:
        logger.error(f"Error fetching GraphQL data: {pools}")
        return pools

    filtered_pools = get_filtered_pools_for_uniswap(pools, current_positions, whitelisted_assets, **kwargs)
    if not filtered_pools:
        logger.warning("No suitable pools found after filtering")
        return {"error": "No suitable pools found"}

    logger.info(f"Processing {len(filtered_pools)} filtered pools for metrics calculation")
    token_id_cache = {}
    for i, pool in enumerate(filtered_pools):
        logger.info(f"Processing pool {i+1}/{len(filtered_pools)}: {pool['id']}")
        pool_chain = pool["chain"].lower()
        token_0_symbol = pool["token0"]["symbol"].lower()
        token_1_symbol = pool["token1"]["symbol"].lower()

        # Token 0
        if token_0_symbol not in token_id_cache:
            token_0_id = get_coin_id_from_symbol(
                coin_id_mapping, token_0_symbol, pool_chain
            )
            if token_0_id:
                token_id_cache[token_0_symbol] = token_0_id
                logger.info(f"Token0 {token_0_symbol} mapped to CoinGecko ID: {token_0_id}")
        else:
            token_0_id = token_id_cache[token_0_symbol]

        # Token 1
        if token_1_symbol not in token_id_cache:
            token_1_id = get_coin_id_from_symbol(
                coin_id_mapping, token_1_symbol, pool_chain
            )
            if token_1_id:
                token_id_cache[token_1_symbol] = token_1_id
                logger.info(f"Token1 {token_1_symbol} mapped to CoinGecko ID: {token_1_id}")
        else:
            token_1_id = token_id_cache[token_1_symbol]

        if token_0_id and token_1_id:
            logger.info(f"Calculating IL risk score for {token_0_symbol}/{token_1_symbol}")
            pool["il_risk_score"] = calculate_il_risk_score(
                token_0_id, token_1_id, coingecko_api_key, x402_signer, x402_proxy
            )
        else:
            logger.warning(f"Could not find CoinGecko IDs for {token_0_symbol}/{token_1_symbol}")
            pool["il_risk_score"] = None
            
        logger.info(f"Calculating Sharpe ratio for pool {pool['id']}")
        graphql_endpoint = graphql_endpoints.get(pool_chain)
        pool["sharpe_ratio"] = get_uniswap_pool_sharpe_ratio(
            pool["id"], graphql_endpoint
        )
        
        logger.info(f"Calculating liquidity metrics for pool {pool['id']}")
        depth_score, max_position_size = assess_pool_liquidity(
            pool["id"], graphql_endpoint
        )
        pool["depth_score"] = depth_score
        pool["max_position_size"] = max_position_size
        pool["type"] = LP

    formatted_results = [format_pool_data(pool) for pool in filtered_pools]
    logger.info(f"Returning {len(formatted_results)} formatted Uniswap opportunities")
    return formatted_results


def calculate_metrics(
    position: Dict[str, Any],
    coingecko_api_key: str,
    graphql_endpoints,
    coin_id_mapping,
    x402_signer,
    x402_proxy,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    token_0_id = get_coin_id_from_symbol(
        coin_id_mapping, position["token0_symbol"], position["chain"]
    )
    token_1_id = get_coin_id_from_symbol(
        coin_id_mapping, position["token1_symbol"], position["chain"]
    )
    il_risk_score = (
        calculate_il_risk_score(token_0_id, token_1_id, coingecko_api_key, x402_signer, x402_proxy)
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
    logger.info("Starting Uniswap pools search strategy execution")
    logger.info(f"Received kwargs: {list(kwargs.keys())}")
    
    missing = check_missing_fields(kwargs)
    if missing:
        logger.error(f"Required kwargs {missing} were not provided")
        get_errors().append(f"Required kwargs {missing} were not provided.")
        return {"error": get_errors()}

    required_fields = list(REQUIRED_FIELDS)
    get_metrics = kwargs.get("get_metrics", False)
    logger.info(f"Get metrics mode: {get_metrics}")
    
    if get_metrics:
        required_fields.append("position")

    if get_metrics:
        logger.info("Calculating metrics for existing position")
        metrics = calculate_metrics(**kwargs)
        if metrics is None:
            logger.error("Failed to calculate metrics")
            get_errors().append("Failed to calculate metrics.")
        return {"error": get_errors()} if get_errors() else metrics
    else:
        logger.info("Finding best Uniswap opportunities")
        result = get_opportunities_for_uniswap(**kwargs)
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Error in get_opportunities_for_uniswap: {result['error']}")
            get_errors().append(result["error"])
        if not result:
            logger.warning("No suitable pools found")
            get_errors().append("No suitable aggregators found")
        
        logger.info(f"Successfully found opportunities: {result}")
        return {"result": result, "error": get_errors()}

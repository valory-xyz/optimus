import warnings

warnings.filterwarnings("ignore")

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

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

# Constants
REQUIRED_FIELDS = ("chains", "lending_asset", "current_positions", "coingecko_api_key")
STURDY = "Sturdy"
TVL_WEIGHT = 0.3
APR_WEIGHT = 0.7
TVL_PERCENTILE = 50
APR_PERCENTILE = 50
SCORE_PERCENTILE = 80
PERCENT_CONVERSION = 100
PRICE_IMPACT = 0.01
FETCH_AGGREGATOR_ENDPOINT = (
    "https://us-central1-stu-dashboard-a0ba2.cloudfunctions.net/v2Aggregators"
)
LENDING = "lending"
# Map known token symbols to CoinGecko IDs
coingecko_name_to_id = {
    "weth": "weth",
    "stone": "stakestone-ether",
    "ezeth": "renzo-restaked-eth",
    "mode": "mode",
}

# Thread-safe global caches with locks
_coin_list_cache = None
_aggregators_cache = None
_historical_data_cache = None
_coin_list_lock = threading.Lock()
_aggregators_lock = threading.Lock()
_historical_data_lock = threading.Lock()

# Thread-local storage for errors
_thread_local = threading.local()

# Request throttling
_last_request_time = {}
_request_lock = threading.Lock()

def get_errors():
    """Get thread-local error list."""
    if not hasattr(_thread_local, 'errors'):
        _thread_local.errors = []
    return _thread_local.errors

def throttled_request(url, min_interval=0.1):
    """Make a throttled HTTP request to prevent rate limiting."""
    with _request_lock:
        now = time.time()
        if url in _last_request_time:
            elapsed = now - _last_request_time[url]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        _last_request_time[url] = time.time()
    
    return requests.get(url)

def get_coin_list():
    global _coin_list_cache
    logger.info("Fetching coin list from CoinGecko")
    
    with _coin_list_lock:
        if _coin_list_cache is None:
            url = "https://api.coingecko.com/api/v3/coins/list"
            try:
                response = throttled_request(url, 0.2)
                response.raise_for_status()
                _coin_list_cache = response.json()
                logger.info(f"Successfully fetched {len(_coin_list_cache)} coins from CoinGecko")
            except requests.RequestException as e:
                logger.error(f"Failed to fetch coin list: {e}")
                get_errors().append(f"Failed to fetch coin list: {e}")
                _coin_list_cache = []
        return _coin_list_cache


def fetch_token_id(symbol):
    logger.info(f"Fetching token ID for symbol: {symbol}")
    symbol = symbol.lower()
    # First check known mappings
    if symbol in coingecko_name_to_id:
        logger.info(f"Found token ID in known mappings: {symbol} -> {coingecko_name_to_id[symbol]}")
        return coingecko_name_to_id[symbol]

    coin_list = get_coin_list()
    for coin in coin_list:
        if coin["symbol"].lower() == symbol:
            logger.info(f"Found token ID in coin list: {symbol} -> {coin['id']}")
            return coin["id"]

    logger.warning(f"Failed to fetch id for coin with symbol: {symbol}")
    get_errors().append(f"Failed to fetch id for coin with symbol: {symbol}")
    return None


def fetch_historical_data(limit: int = 720):
    global _historical_data_cache
    logger.info(f"Fetching historical data with limit: {limit}")
    
    with _historical_data_lock:
        if _historical_data_cache is not None:
            logger.info("Using cached historical data")
            return _historical_data_cache

        current_time_ms = int(datetime.now().timestamp() * 1000)
        one_month_ago_ms = current_time_ms - (30 * 24 * 60 * 60 * 1000)
        url = f"https://us-central1-stu-dashboard-a0ba2.cloudfunctions.net/getV2AggregatorHistoricalData?last_time={one_month_ago_ms}&limit={limit}"
        logger.info(f"Fetching historical data from: {url}")
        
        response = throttled_request(url, 0.2)
        if response.status_code != 200:
            logger.error(f"Failed to fetch historical data. Status code: {response.status_code}")
            get_errors().append("Failed to fetch historical data from STURDY API.")
            return None
        
        _historical_data_cache = response.json()
        logger.info(f"Successfully fetched historical data with {len(_historical_data_cache)} entries")
        return _historical_data_cache


def calculate_daily_returns(base_apy, reward_apy=0):
    annual_return = base_apy + reward_apy
    return (1 + annual_return) ** (1 / 365) - 1


def calculate_sharpe_ratio(returns, risk_free_rate=0.0003):
    """
    Calculate the Sharpe Ratio without pyfolio.
    Sharpe = (mean(returns) - risk_free_rate) / std(returns)
    """
    if len(returns) < 2:
        return np.nan  # Not enough data

    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std(ddof=1)


def get_sharpe_ratio_for_address(historical_data, address: str) -> float:
    logger.info(f"Calculating Sharpe ratio for address: {address}")
    records = []
    for _, entry in enumerate(historical_data):
        timestamp = entry["timestamp"]
        mapping = {}
        for ent in entry["doc"]:
            if len(ent.split("_")) < 2:
                continue
            addr = ent.split("_")[1]
            mapping[addr] = ent
        if address not in mapping:
            continue
        address_key = mapping[address]
        if address_key in entry["doc"]:
            data = entry["doc"][address_key]
            base_apy = data.get("baseAPY", 0)
            rewards_apy = data.get("rewardsAPY", 0)
            records.append(
                {
                    "timestamp": timestamp,
                    "base_apy": base_apy,
                    "rewards_apy": rewards_apy,
                }
            )

    if not records:
        logger.warning(f"No historical records found for address: {address}")
        return float("nan")

    logger.info(f"Found {len(records)} historical records for address: {address}")
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    daily_df = df.groupby("date").median().reset_index()

    daily_df["daily_return"] = daily_df.apply(
        lambda row: calculate_daily_returns(row["base_apy"], row["rewards_apy"]), axis=1
    )
    sharpe_ratio = calculate_sharpe_ratio(daily_df["daily_return"])
    logger.info(f"Calculated Sharpe ratio for {address}: {sharpe_ratio}")
    return sharpe_ratio


def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    missing = [field for field in REQUIRED_FIELDS if kwargs.get(field) is None]
    if missing:
        logger.warning(f"Missing required fields: {missing}")
    return missing


def remove_irrelevant_fields(
    kwargs: Dict[str, Any], required_fields: Tuple
) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key in required_fields}


def fetch_aggregators() -> List[Dict[str, Any]]:
    global _aggregators_cache
    logger.info("Fetching aggregators from STURDY API")
    
    with _aggregators_lock:
        if _aggregators_cache is not None:
            logger.info("Using cached aggregators data")
            return _aggregators_cache

        try:
            logger.info(f"Making request to: {FETCH_AGGREGATOR_ENDPOINT}")
            response = throttled_request(FETCH_AGGREGATOR_ENDPOINT, 0.2)
            response.raise_for_status()
            result = response.json()
            if "errors" in result:
                logger.error(f"REST API returned errors: {result['errors']}")
                get_errors().append(f"REST API Errors: {result['errors']}")
                _aggregators_cache = []
            else:
                _aggregators_cache = result
                logger.info(f"Successfully fetched {len(_aggregators_cache)} aggregators")
        except requests.RequestException as e:
            logger.error(f"REST API request failed: {e}")
            get_errors().append(f"REST API request failed: {e}")
            _aggregators_cache = []
        return _aggregators_cache


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
    
    # Extract APR and TVL values
    aprs = [pool.get('total_apr', 0) for pool in pools]
    tvls = [pool.get('tvl', 0) for pool in pools]
    
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
    for pool in pools:
        apr = pool.get('total_apr', 0)
        tvl = pool.get('tvl', 0)
        
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
        logger.info("Skipping composite pre-filter")
        return pools[:top_n] if pools else []
    
    logger.info(f"Starting composite pre-filter with {len(pools)} pools")
    logger.info(f"Parameters: top_n={top_n}, apr_weight={apr_weight}, tvl_weight={tvl_weight}, min_tvl_threshold={min_tvl_threshold}")
    
    # Filter by minimum TVL threshold with proper type conversion
    tvl_filtered = []
    for pool in pools:
        try:
            tvl_value = float(pool.get('tvl', 0))
            if tvl_value >= float(min_tvl_threshold):
                tvl_filtered.append(pool)
        except (ValueError, TypeError):
            # Skip pools with invalid TVL values
            logger.warning(f"Skipping pool with invalid TVL: {pool.get('address', 'unknown')}")
            continue
    logger.info(f"After TVL filter (>= {min_tvl_threshold}): {len(tvl_filtered)} pools")
    
    if not tvl_filtered:
        logger.warning("No pools meet minimum TVL threshold")
        return []
    
    # Apply standardization and composite scoring
    standardized_pools = standardize_metrics(tvl_filtered, apr_weight, tvl_weight)
    
    # Sort by composite score (descending)
    standardized_pools.sort(key=lambda x: x.get('composite_score', float('-inf')), reverse=True)
    
    # Select top N pools
    final_selection = standardized_pools[:top_n]
    logger.info(f"Final selection: {len(final_selection)} pools")
    
    # Log top pools for debugging
    for i, pool in enumerate(final_selection[:5]):
        apr = pool.get('total_apr', 0)
        tvl = pool.get('tvl', 0)
        composite_score = pool.get('composite_score', 'N/A')
        pool_address = pool.get('address', 'N/A')
        
        logger.info(f"Pool #{i+1}: {pool_address} - APR: {apr:.2f}%, TVL: ${tvl:,.0f}, Composite Score: {composite_score:.3f}")
    
    logger.info("Applied composite pre-filter")
    return final_selection


def filter_aggregators(
    chains, aggregators, lending_asset, current_positions, **kwargs
) -> List[Dict[str, Any]]:
    logger.info(f"Filtering aggregators for chains: {chains}, lending_asset: {lending_asset}")
    logger.info(f"Total aggregators to filter: {len(aggregators)}")
    logger.info(f"Current positions to exclude: {current_positions}")
    
    # Extract composite filtering parameters
    top_n = kwargs.get('top_n', 10)
    apr_weight = kwargs.get('apr_weight', 0.7)
    tvl_weight = kwargs.get('tvl_weight', 0.3)
    min_tvl_threshold = kwargs.get('min_tvl_threshold', 1000)    
    filtered_aggregators = []

    # Filter by chain, asset, and exclude current_positions
    for aggregator in aggregators:
        if (
            aggregator.get("chainName") in chains
            and Web3.to_checksum_address(aggregator.get("address"))
            not in current_positions
        ):
            if aggregator.get("asset", {}).get("address") == lending_asset:
                total_apr = aggregator.get("apy", {}).get("total", 0)
                tvl = aggregator.get("tvl", 0)
                aggregator["total_apr"] = total_apr
                aggregator["tvl"] = tvl
                filtered_aggregators.append(aggregator)
                logger.info(f"Added aggregator: {aggregator.get('address')} with TVL: {tvl}, APR: {total_apr}")

    logger.info(f"After initial filtering: {len(filtered_aggregators)} aggregators")

    if not filtered_aggregators:
        logger.warning("No suitable aggregator found after initial filtering")
        get_errors().append("No suitable aggregator found.")
        return []

    # Apply composite pre-filtering
    filtered_aggregators = apply_composite_pre_filter(
        filtered_aggregators, 
        top_n=top_n,
        apr_weight=apr_weight,
        tvl_weight=tvl_weight,
        min_tvl_threshold=min_tvl_threshold,
    )
    
    return filtered_aggregators


def calculate_il_risk_score_for_lending(
    asset_token_1: str,
    asset_token_2: str,
    coingecko_api_key: str,
    time_period: int = 90,
) -> float:
    logger.info(f"Calculating IL risk score for tokens: {asset_token_1}, {asset_token_2}")
    if not asset_token_1 or not asset_token_2:
        logger.error("Tokens are required for IL risk score calculation")
        get_errors().append(
            "Tokens are required. Cannot calculate IL risk score without asset tokens"
        )
        return None

    is_pro = is_pro_api_key(coingecko_api_key)
    logger.info(f"Using {'pro' if is_pro else 'demo'} CoinGecko API")
    
    if is_pro:
        cg = CoinGeckoAPI(api_key=coingecko_api_key)
    else:
        cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=time_period)).timestamp())

    try:
        logger.info(f"Fetching price data for {asset_token_1}")
        prices_1 = cg.get_coin_market_chart_range_by_id(
            id=asset_token_1,
            vs_currency="usd",
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )
        logger.info(f"Fetching price data for {asset_token_2}")
        prices_2 = cg.get_coin_market_chart_range_by_id(
            id=asset_token_2,
            vs_currency="usd",
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )
    except Exception as e:
        logger.error(f"Error fetching price data: {e}")
        get_errors().append(f"Error fetching price data: {e}")
        return None
    
    prices_1_data = np.array([x[1] for x in prices_1["prices"]])
    prices_2_data = np.array([x[1] for x in prices_2["prices"]])
    logger.info(f"Price data points: {len(prices_1_data)} for {asset_token_1}, {len(prices_2_data)} for {asset_token_2}")

    min_length = min(len(prices_1_data), len(prices_2_data))
    if min_length == 0:
        logger.warning("No price data available for IL calculation")
        return None

    prices_1_data = prices_1_data[:min_length]
    prices_2_data = prices_2_data[:min_length]

    price_correlation = np.corrcoef(prices_1_data, prices_2_data)[0, 1]
    volatility_1 = np.std(prices_1_data)
    volatility_2 = np.std(prices_2_data)
    volatility_multiplier = np.sqrt(volatility_1 * volatility_2)

    P0 = prices_1_data[0] / prices_2_data[0]
    P1 = prices_1_data[-1] / prices_2_data[-1]
    il_impact = 2 * np.sqrt(P1 / P0) / (1 + P1 / P0) - 1
    il_risk_score = il_impact * abs(price_correlation) * volatility_multiplier

    logger.info(f"IL risk score calculated: {il_risk_score}")
    return float(il_risk_score)


def calculate_il_risk_score_for_silos(token0_symbol, silos, coingecko_api_key):
    logger.info(f"Calculating IL risk score for silos with token0: {token0_symbol}")
    logger.info(f"Number of silos: {len(silos)}")
    
    il_risk_scores = []
    token_id_cache = {}

    def get_token_id(symbol):
        symbol = symbol.lower()
        if symbol in token_id_cache:
            return token_id_cache[symbol]
        token_id = coingecko_name_to_id.get(symbol) or fetch_token_id(symbol)
        token_id_cache[symbol] = token_id
        return token_id

    token_0_id = get_token_id(token0_symbol)
    if not token_0_id:
        logger.warning(f"Could not find token ID for {token0_symbol}")
        return None

    for silo in silos:
        token_1_symbol = silo["collateral"].lower()
        logger.info(f"Processing silo with collateral: {token_1_symbol}")
        token_1_id = get_token_id(token_1_symbol)

        if token_1_id:
            il_risk_score = calculate_il_risk_score_for_lending(
                token_0_id, token_1_id, coingecko_api_key
            )
            if not il_risk_score:
                logger.warning(f"Could not calculate IL risk score for silo: {token_1_symbol}")
                return None
            
            il_risk_scores.append(il_risk_score)
            logger.info(f"IL risk score for {token_1_symbol}: {il_risk_score}")
        else:
            logger.error(f"Failed to fetch token IDs for silo: {silo['collateral']}")
            get_errors().append(f"Failed to fetch token IDs for silo: {silo['collateral']}")

    if not il_risk_scores:
        logger.warning("No IL risk scores calculated")
        return None

    avg_il_risk = sum(il_risk_scores) / len(il_risk_scores)
    logger.info(f"Average IL risk score: {avg_il_risk}")
    return avg_il_risk


def analyze_vault_liquidity(aggregator):
    logger.info(f"Analyzing vault liquidity for aggregator: {aggregator.get('address')}")
    tvl = float(aggregator.get("tvl", 0))
    total_assets = float(aggregator.get("totalAssets", 0))
    logger.info(f"Initial TVL: {tvl}, Total Assets: {total_assets}")

    # If missing, try to fetch again from cached aggregators
    if not tvl or not total_assets:
        logger.info("TVL or total assets missing, fetching from aggregators")
        aggregators = fetch_aggregators()
        for item in aggregators:
            if item["address"] == aggregator.get("address") or item[
                "address"
            ] == aggregator.get("pool_address"):
                tvl = float(item.get("tvl", 0))
                total_assets = float(item.get("totalAssets", 0))
                logger.info(f"Updated TVL: {tvl}, Total Assets: {total_assets}")
                break

    if not tvl or not total_assets:
        logger.error("Could not retrieve TVL and total assets for depth score calculation")
        get_errors().append("Could not retrieve depth score and maximum position size.")
        return float("nan"), float("nan")

    depth_score = (
        (np.log1p(tvl) * np.log1p(total_assets)) / (PRICE_IMPACT * 1000)
        if tvl and total_assets
        else 0
    )
    liquidity_risk_multiplier = max(0, 1 - (1 / depth_score)) if depth_score > 0 else 0
    max_position_size = 50 * (tvl * liquidity_risk_multiplier) / 100

    logger.info(f"Calculated depth score: {depth_score}, max position size: {max_position_size}")
    return depth_score, max_position_size


def format_aggregator(aggregator) -> Dict[str, Any]:
    logger.info(f"Formatting aggregator: {aggregator.get('address')}")
    return {
        "chain": aggregator["chainName"],
        "pool_address": aggregator["address"],
        "dex_type": STURDY,
        "token0_symbol": aggregator["asset"]["symbol"],
        "token0": aggregator["asset"]["address"],
        "apr": aggregator["total_apr"] * PERCENT_CONVERSION,
        "whitelistedSilos": aggregator["whitelistedSilos"],
        "il_risk_score": aggregator["il_risk_score"],
        "sharpe_ratio": aggregator["sharpe_ratio"],
        "depth_score": aggregator["depth_score"],
        "max_position_size": aggregator["max_position_size"],
        "type": aggregator["type"]
    }


def get_best_opportunities(
    chains, lending_asset, current_positions, coingecko_api_key, **kwargs
) -> List[Dict[str, Any]]:
    logger.info(f"Getting best opportunities for chains: {chains}, lending_asset: {lending_asset}")
    
    data = fetch_aggregators()
    if not data:
        logger.error("Failed to fetch aggregators")
        get_errors().append("Failed to fetch aggregators.")
        return {"error": get_errors()}

    filtered_aggregators = filter_aggregators(
        chains, data, lending_asset, current_positions, **kwargs
    )
    if not filtered_aggregators:
        logger.warning("No suitable aggregators found after filtering")
        get_errors().append("No suitable aggregators found.")
        return {"error": get_errors()}

    historical_data = fetch_historical_data()
    if historical_data is None:
        logger.error("Failed to fetch historical data")
        return {"error": get_errors()}

    logger.info(f"Processing {len(filtered_aggregators)} aggregators for metrics calculation")
    for i, aggregator in enumerate(filtered_aggregators):
        logger.info(f"Processing aggregator {i+1}/{len(filtered_aggregators)}: {aggregator.get('address')}")
        
        silos = aggregator.get("whitelistedSilos", [])
        aggregator["il_risk_score"] = calculate_il_risk_score_for_silos(
            aggregator["asset"]["symbol"], silos, coingecko_api_key
        )
        aggregator["sharpe_ratio"] = get_sharpe_ratio_for_address(
            historical_data, aggregator["address"]
        )
        depth_score, max_position_size = analyze_vault_liquidity(aggregator)
        aggregator["depth_score"] = depth_score
        aggregator["max_position_size"] = max_position_size
        aggregator["type"] = LENDING

    formatted_results = [
        format_aggregator(aggregator) for aggregator in filtered_aggregators
    ]

    logger.info(f"Returning {len(formatted_results)} formatted opportunities")
    return formatted_results


def calculate_metrics(
    position: Dict[str, Any], coingecko_api_key: str, **kwargs
) -> Optional[Dict[str, Any]]:
    logger.info(f"Calculating metrics for position: {position.get('pool_address')}")
    
    il_risk_score = calculate_il_risk_score_for_silos(
        position.get("token0_symbol"),
        position.get("whitelistedSilos", []),
        coingecko_api_key,
    )
    historical_data = fetch_historical_data()
    if historical_data is None:
        logger.error("Failed to fetch historical data for metrics calculation")
        return {"error": get_errors()}

    sharpe_ratio = get_sharpe_ratio_for_address(
        historical_data, position["pool_address"]
    )
    depth_score, max_position_size = analyze_vault_liquidity(position)
    
    metrics = {
        "il_risk_score": il_risk_score,
        "sharpe_ratio": sharpe_ratio,
        "max_position_size": max_position_size,
        "depth_score": depth_score,
    }
    logger.info(f"Calculated metrics: {metrics}")
    return metrics


def is_pro_api_key(coingecko_api_key: str) -> bool:
    """
    Check if the provided CoinGecko API key is a pro key.
    """
    logger.info("Checking if CoinGecko API key is pro")
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
            logger.info("CoinGecko API key is pro")
            return True
    except Exception:
        logger.info("CoinGecko API key is not pro, using demo")
        return False

    return False

def run(*_args, **kwargs) -> Any:
    logger.info("Starting asset lending strategy execution")
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

    kwargs = remove_irrelevant_fields(kwargs, required_fields)

    if get_metrics:
        logger.info("Calculating metrics for existing position")
        metrics = calculate_metrics(**kwargs)
        if metrics is None:
            logger.error("Failed to calculate metrics")
            get_errors().append("Failed to calculate metrics.")
        return {"error": get_errors()} if get_errors() else metrics
    else:
        logger.info("Finding best opportunities")
        result = get_best_opportunities(**kwargs)
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Error in get_best_opportunities: {result['error']}")
            get_errors().append(result["error"])
        if not result:
            logger.warning("No suitable aggregators found")
            get_errors().append("No suitable aggregators found")
        
        
        logger.info(f"Successfully found opportunities: {result}")
        return {"result": result, "error": get_errors()}
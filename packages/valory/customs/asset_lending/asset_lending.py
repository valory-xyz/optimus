import warnings
warnings.filterwarnings("ignore")

import requests
from typing import Dict, Any, List, Optional
import numpy as np
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import pandas as pd
import pyfolio as pf
import json

from aea.helpers.logging import setup_logger

# Configure _logger
_logger = setup_logger(__name__)

# Constants
REQUIRED_FIELDS = ("chains", "apr_threshold", "lending_asset", "current_pool", "coingecko_api_key")
STURDY = 'Sturdy'
TVL_WEIGHT = 0.6
APR_WEIGHT = 0.4
TVL_PERCENTILE = 50
APR_PERCENTILE = 50
SCORE_PERCENTILE = 80
PERCENT_CONVERSION = 100
PRICE_IMPACT = 0.01
FETCH_AGGREGATOR_ENDPOINT = "https://us-central1-stu-dashboard-a0ba2.cloudfunctions.net/v2Aggregators"

# Map known token symbols to CoinGecko IDs
coingecko_name_to_id = {
    "weth": "weth",
    "stone": "stakestone-ether",
    "ezeth": "renzo-restaked-eth",
    "mode": "mode",
}

# Global caches
_coin_list_cache = None
_aggregators_cache = None
_historical_data_cache = None


def get_coin_list():
    global _coin_list_cache
    if _coin_list_cache is None:
        url = "https://api.coingecko.com/api/v3/coins/list"
        try:
            response = requests.get(url)
            response.raise_for_status()
            _coin_list_cache = response.json()
        except requests.RequestException as e:
            _logger.error(f"Failed to fetch coin list: {e}")
            _coin_list_cache = []
    return _coin_list_cache


def fetch_token_id(symbol):
    symbol = symbol.lower()
    # First check known mappings
    if symbol in coingecko_name_to_id:
        return coingecko_name_to_id[symbol]

    coin_list = get_coin_list()
    for coin in coin_list:
        if coin['symbol'].lower() == symbol:
            return coin['id']

    _logger.error(f"Failed to fetch id for coin with symbol: {symbol}")
    return None


def fetch_historical_data(limit: int = 720):
    global _historical_data_cache
    if _historical_data_cache is not None:
        return _historical_data_cache

    current_time_ms = int(datetime.now().timestamp() * 1000)
    one_month_ago_ms = current_time_ms - (30 * 24 * 60 * 60 * 1000)
    url = f"https://us-central1-stu-dashboard-a0ba2.cloudfunctions.net/getV2AggregatorHistoricalData?last_time={one_month_ago_ms}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch historical data from STURDY API.")
    _historical_data_cache = response.json()
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
    records = []
    for _, entry in enumerate(historical_data):
        timestamp = entry['timestamp']
        mapping = {}
        for ent in entry['doc']:
            if len(ent.split("_")) < 2:
                continue
            addr = ent.split("_")[1]
            mapping[addr] = ent
        if address not in mapping:
            continue
        address_key = mapping[address]
        if address_key in entry['doc']:
            data = entry['doc'][address_key]
            base_apy = data.get('baseAPY', 0)
            rewards_apy = data.get('rewardsAPY', 0)
            records.append({
                'timestamp': timestamp,
                'base_apy': base_apy,
                'rewards_apy': rewards_apy
            })

    if not records:
        return float('nan')

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    daily_df = df.groupby('date').median().reset_index()  
    
    daily_df['daily_return'] = daily_df.apply(lambda row: calculate_daily_returns(row['base_apy'], row['rewards_apy']), axis=1)
    return calculate_sharpe_ratio(daily_df['daily_return'])


def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    return [field for field in REQUIRED_FIELDS if kwargs.get(field) is None]


def remove_irrelevant_fields(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key in REQUIRED_FIELDS}


def fetch_aggregators() -> List[Dict[str, Any]]:
    global _aggregators_cache
    if _aggregators_cache is not None:
        return _aggregators_cache

    try:
        response = requests.get(FETCH_AGGREGATOR_ENDPOINT)
        response.raise_for_status()
        result = response.json()
        if 'errors' in result:
            _logger.error(f"REST API Errors: {result['errors']}")
            _aggregators_cache = []
        else:
            _aggregators_cache = result
    except requests.RequestException as e:
        _logger.error(f"REST API request failed: {e}")
        _aggregators_cache = []
    return _aggregators_cache


def filter_aggregators(chains, apr_threshold, aggregators, lending_asset, current_pool) -> List[Dict[str, Any]]:
    filtered_aggregators = []
    tvl_list = []
    apr_list = []

    # Filter by chain, asset, and exclude current_pool
    for aggregator in aggregators:
        if aggregator.get("chainName") in chains and aggregator.get('address') != current_pool:
            if aggregator.get("asset", {}).get("address") == lending_asset:
                total_apr = aggregator.get('apy', {}).get('total', 0)
                tvl = aggregator.get('tvl', 0)
                tvl_list.append(tvl)
                apr_list.append(total_apr)
                aggregator["total_apr"] = total_apr
                aggregator["tvl"] = tvl
                filtered_aggregators.append(aggregator)

    if not filtered_aggregators:
        _logger.error("No suitable aggregator found.")
        return []

    # If very few aggregators, return them directly
    if len(filtered_aggregators) <= 5:
        return filtered_aggregators

    tvl_threshold = np.percentile(tvl_list, TVL_PERCENTILE)
    apr_threshold_val = np.percentile(apr_list, APR_PERCENTILE)

    scored_aggregators = []
    max_tvl = max(tvl_list)
    max_apr = max(apr_list)

    for aggregator in filtered_aggregators:
        tvl = aggregator["tvl"]
        total_apr = aggregator["total_apr"]

        if tvl < tvl_threshold or total_apr < apr_threshold_val:
            continue

        score = TVL_WEIGHT * (tvl / max_tvl) + APR_WEIGHT * (total_apr / max_apr)
        aggregator["score"] = score
        scored_aggregators.append(aggregator)

    if not scored_aggregators:
        _logger.error("No suitable aggregator found after scoring.")
        return []

    score_threshold = np.percentile([agg["score"] for agg in scored_aggregators], SCORE_PERCENTILE)
    filtered_scored_aggregators = [agg for agg in scored_aggregators if agg["score"] >= score_threshold]

    filtered_scored_aggregators.sort(key=lambda x: x["score"], reverse=True)

    if not filtered_scored_aggregators:
        _logger.error("No suitable aggregator found after score threshold.")
        return []

    # Limit to top 10 scored pools if more than 10
    if len(filtered_scored_aggregators) > 10:
        return filtered_scored_aggregators[:10]
    else:
        return filtered_scored_aggregators


def calculate_il_risk_score_for_lending(asset_token_1: str, asset_token_2: str, coingecko_api_key: str, time_period: int = 90) -> float:
    if not asset_token_1 or not asset_token_2:
        _logger.error("Tokens are required. Cannot calculate IL risk score without asset tokens")
        return float('nan')

    cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=time_period)).timestamp())

    try:
        prices_1 = cg.get_coin_market_chart_range_by_id(id=asset_token_1, vs_currency='usd',
                                                        from_timestamp=from_timestamp, to_timestamp=to_timestamp)
        prices_2 = cg.get_coin_market_chart_range_by_id(id=asset_token_2, vs_currency='usd',
                                                        from_timestamp=from_timestamp, to_timestamp=to_timestamp)
    except Exception as e:
        _logger.error(f"Error fetching price data: {e}")
        return float('nan')

    prices_1_data = np.array([x[1] for x in prices_1['prices']])
    prices_2_data = np.array([x[1] for x in prices_2['prices']])

    min_length = min(len(prices_1_data), len(prices_2_data))
    if min_length == 0:
        return float('nan')

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

    return float(il_risk_score)


def calculate_il_risk_score_for_silos(token0_symbol, silos, coingecko_api_key):
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
        return float('nan')

    for silo in silos:
        token_1_symbol = silo['collateral'].lower()
        token_1_id = get_token_id(token_1_symbol)

        if token_1_id:
            il_risk_score = calculate_il_risk_score_for_lending(token_0_id, token_1_id, coingecko_api_key)
            il_risk_scores.append(il_risk_score)
        else:
            _logger.error(f"Failed to fetch token IDs for silo: {silo['collateral']}")

    if not il_risk_scores:
        return float('nan')

    return sum(il_risk_scores) / len(il_risk_scores)


def analyze_vault_liquidity(aggregator):
    tvl = float(aggregator.get('tvl', 0))
    total_assets = float(aggregator.get('totalAssets', 0))

    # If missing, try to fetch again from cached aggregators
    if not tvl or not total_assets:
        aggregators = fetch_aggregators()
        for item in aggregators:
            if item['address'] == aggregator.get('address') or item['address'] == aggregator.get('pool_address'):
                tvl = float(item.get('tvl', 0))
                total_assets = float(item.get('totalAssets', 0))
                break

    if not tvl or not total_assets:
        return float('nan'), float('nan')

    depth_score = (np.log1p(tvl) * np.log1p(total_assets)) / (PRICE_IMPACT * 1000) if tvl and total_assets else 0
    liquidity_risk_multiplier = max(0, 1 - (1 / depth_score)) if depth_score > 0 else 0
    max_position_size = 50 * (tvl * liquidity_risk_multiplier) / 100

    return depth_score, max_position_size


def format_aggregator(aggregator) -> Dict[str, Any]:
    return {
        "chain": aggregator['chainName'],
        "pool_address": aggregator['address'],
        "dex_type": STURDY,
        "token0_symbol": aggregator['asset']['symbol'],
        "token0": aggregator['asset']['address'],
        "apr": aggregator['total_apr'] * PERCENT_CONVERSION,
        "whitelistedSilos": aggregator['whitelistedSilos'],
        "il_risk_score": aggregator['il_risk_score'],
        "sharpe_ratio": aggregator['sharpe_ratio'],
        "depth_score": aggregator['depth_score'],
        "max_position_size": aggregator['max_position_size']
    }


def get_best_opportunities(chains, apr_threshold, lending_asset, current_pool, coingecko_api_key) -> List[Dict[str, Any]]:
    data = fetch_aggregators()
    if not data:
        return []

    filtered_aggregators = filter_aggregators(chains, apr_threshold, data, lending_asset, current_pool)
    if not filtered_aggregators:
        return []

    historical_data = fetch_historical_data()

    for aggregator in filtered_aggregators:
        silos = aggregator.get('whitelistedSilos', [])
        aggregator['il_risk_score'] = calculate_il_risk_score_for_silos(aggregator['asset']['symbol'], silos, coingecko_api_key)
        aggregator['sharpe_ratio'] = get_sharpe_ratio_for_address(historical_data, aggregator['address'])
        depth_score, max_position_size = analyze_vault_liquidity(aggregator)
        aggregator["depth_score"] = depth_score
        aggregator["max_position_size"] = max_position_size

    formatted_results = [format_aggregator(aggregator) for aggregator in filtered_aggregators]

    return formatted_results


def calculate_metrics(current_pool: Dict[str, Any], coingecko_api_key: str, **kwargs) -> Optional[Dict[str, Any]]:
    il_risk_score = calculate_il_risk_score_for_silos(current_pool.get("token0_symbol"), current_pool.get('whitelistedSilos', []), coingecko_api_key)
    historical_data = fetch_historical_data()
    sharpe_ratio = get_sharpe_ratio_for_address(historical_data, current_pool['pool_address'])
    depth_score, max_position_size = analyze_vault_liquidity(current_pool)
    return {
        "il_risk_score": il_risk_score,
        "sharpe_ratio": sharpe_ratio,
        "max_position_size": max_position_size,
        "depth_score": depth_score
    }


def run(*_args, **kwargs) -> Any:
    missing = check_missing_fields(kwargs)
    if missing:
        return {"error": f"Required kwargs {missing} were not provided."}

    get_metrics = kwargs.get('get_metrics', False)
    kwargs = remove_irrelevant_fields(kwargs)

    if get_metrics:
        return calculate_metrics(**kwargs)
    else:
        result = get_best_opportunities(**kwargs)
        if not result:
            return {"error": "No suitable aggregators found"}
        return result
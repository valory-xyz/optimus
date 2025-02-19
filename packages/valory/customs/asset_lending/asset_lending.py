import warnings

warnings.filterwarnings("ignore")

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyfolio as pf
import requests
from aea.helpers.logging import setup_logger
from pycoingecko import CoinGeckoAPI
from web3 import Web3

# Configure _logger
_logger = setup_logger(__name__)

# Constants
REQUIRED_FIELDS = ("chains", "lending_asset", "current_positions", "coingecko_api_key")
STURDY = "Sturdy"
TVL_WEIGHT = 0.6
APR_WEIGHT = 0.4
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

# Global caches
_coin_list_cache = None
_aggregators_cache = None
_historical_data_cache = None

# Error list
errors = []

def get_coin_list():
    global _coin_list_cache
    if _coin_list_cache is None:
        url = "https://api.coingecko.com/api/v3/coins/list"
        try:
            response = requests.get(url)
            response.raise_for_status()
            _coin_list_cache = response.json()
        except requests.RequestException as e:
            errors.append(f"Failed to fetch coin list: {e}")
            _coin_list_cache = []
    return _coin_list_cache


def fetch_token_id(symbol):
    symbol = symbol.lower()
    # First check known mappings
    if symbol in coingecko_name_to_id:
        return coingecko_name_to_id[symbol]

    coin_list = get_coin_list()
    for coin in coin_list:
        if coin["symbol"].lower() == symbol:
            return coin["id"]

    errors.append(f"Failed to fetch id for coin with symbol: {symbol}")
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
        errors.append("Failed to fetch historical data from STURDY API.")
        return None
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
        return float("nan")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    daily_df = df.groupby("date").median().reset_index()

    daily_df["daily_return"] = daily_df.apply(
        lambda row: calculate_daily_returns(row["base_apy"], row["rewards_apy"]), axis=1
    )
    return calculate_sharpe_ratio(daily_df["daily_return"])


def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    return [field for field in REQUIRED_FIELDS if kwargs.get(field) is None]


def remove_irrelevant_fields(
    kwargs: Dict[str, Any], required_fields: Tuple
) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key in required_fields}


def fetch_aggregators() -> List[Dict[str, Any]]:
    global _aggregators_cache
    if _aggregators_cache is not None:
        return _aggregators_cache

    try:
        response = requests.get(FETCH_AGGREGATOR_ENDPOINT)
        response.raise_for_status()
        result = response.json()
        if "errors" in result:
            errors.append(f"REST API Errors: {result['errors']}")
            _aggregators_cache = []
        else:
            _aggregators_cache = result
    except requests.RequestException as e:
        errors.append(f"REST API request failed: {e}")
        _aggregators_cache = []
    return _aggregators_cache


def filter_aggregators(
    chains, aggregators, lending_asset, current_positions
) -> List[Dict[str, Any]]:
    filtered_aggregators = []
    tvl_list = []
    apr_list = []

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
                tvl_list.append(tvl)
                apr_list.append(total_apr)
                aggregator["total_apr"] = total_apr
                aggregator["tvl"] = tvl
                filtered_aggregators.append(aggregator)

    if not filtered_aggregators:
        errors.append("No suitable aggregator found.")
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
        errors.append("No suitable aggregator found after scoring.")
        return []

    score_threshold = np.percentile(
        [agg["score"] for agg in scored_aggregators], SCORE_PERCENTILE
    )
    filtered_scored_aggregators = [
        agg for agg in scored_aggregators if agg["score"] >= score_threshold
    ]

    filtered_scored_aggregators.sort(key=lambda x: x["score"], reverse=True)

    if not filtered_scored_aggregators:
        errors.append("No suitable aggregator found after score threshold.")
        return []

    # Limit to top 10 scored pools if more than 10
    if len(filtered_scored_aggregators) > 10:
        return filtered_scored_aggregators[:10]
    else:
        return filtered_scored_aggregators


def calculate_il_risk_score_for_lending(
    asset_token_1: str,
    asset_token_2: str,
    coingecko_api_key: str,
    time_period: int = 90,
) -> float:
    if not asset_token_1 or not asset_token_2:
        errors.append(
            "Tokens are required. Cannot calculate IL risk score without asset tokens"
        )
        return None

    is_pro = is_pro_api_key(coingecko_api_key)
    if is_pro:
        cg = CoinGeckoAPI(api_key=coingecko_api_key)
    else:
        cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=time_period)).timestamp())

    try:
        prices_1 = cg.get_coin_market_chart_range_by_id(
            id=asset_token_1,
            vs_currency="usd",
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )
        prices_2 = cg.get_coin_market_chart_range_by_id(
            id=asset_token_2,
            vs_currency="usd",
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )
    except Exception as e:
        errors.append(f"Error fetching price data: Incorrect Coingecko API Key")
        return None
    
    prices_1_data = np.array([x[1] for x in prices_1["prices"]])
    prices_2_data = np.array([x[1] for x in prices_2["prices"]])

    min_length = min(len(prices_1_data), len(prices_2_data))
    if min_length == 0:
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
        return None

    for silo in silos:
        token_1_symbol = silo["collateral"].lower()
        token_1_id = get_token_id(token_1_symbol)

        if token_1_id:
            il_risk_score = calculate_il_risk_score_for_lending(
                token_0_id, token_1_id, coingecko_api_key
            )
            if not il_risk_score:
                return None
            
            il_risk_scores.append(il_risk_score)
        else:
            errors.append(f"Failed to fetch token IDs for silo: {silo['collateral']}")

    if not il_risk_scores:
        return None

    return sum(il_risk_scores) / len(il_risk_scores)


def analyze_vault_liquidity(aggregator):
    tvl = float(aggregator.get("tvl", 0))
    total_assets = float(aggregator.get("totalAssets", 0))

    # If missing, try to fetch again from cached aggregators
    if not tvl or not total_assets:
        aggregators = fetch_aggregators()
        for item in aggregators:
            if item["address"] == aggregator.get("address") or item[
                "address"
            ] == aggregator.get("pool_address"):
                tvl = float(item.get("tvl", 0))
                total_assets = float(item.get("totalAssets", 0))
                break

    if not tvl or not total_assets:
        errors.append("Could not retrieve depth score and maximum position size.")
        return float("nan"), float("nan")

    depth_score = (
        (np.log1p(tvl) * np.log1p(total_assets)) / (PRICE_IMPACT * 1000)
        if tvl and total_assets
        else 0
    )
    liquidity_risk_multiplier = max(0, 1 - (1 / depth_score)) if depth_score > 0 else 0
    max_position_size = 50 * (tvl * liquidity_risk_multiplier) / 100

    return depth_score, max_position_size


def format_aggregator(aggregator) -> Dict[str, Any]:
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
    chains, lending_asset, current_positions, coingecko_api_key
) -> List[Dict[str, Any]]:
    data = fetch_aggregators()
    if not data:
        errors.append("Failed to fetch aggregators.")
        return {"error": errors}

    filtered_aggregators = filter_aggregators(
        chains, data, lending_asset, current_positions
    )
    if not filtered_aggregators:
        errors.append("No suitable aggregators found.")
        return {"error": errors}

    historical_data = fetch_historical_data()
    if historical_data is None:
        return {"error": errors}

    for aggregator in filtered_aggregators:
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

    return formatted_results


def calculate_metrics(
    position: Dict[str, Any], coingecko_api_key: str, **kwargs
) -> Optional[Dict[str, Any]]:
    il_risk_score = calculate_il_risk_score_for_silos(
        position.get("token0_symbol"),
        position.get("whitelistedSilos", []),
        coingecko_api_key,
    )
    historical_data = fetch_historical_data()
    if historical_data is None:
        return {"error": errors}

    sharpe_ratio = get_sharpe_ratio_for_address(
        historical_data, position["pool_address"]
    )
    depth_score, max_position_size = analyze_vault_liquidity(position)
    return {
        "il_risk_score": il_risk_score,
        "sharpe_ratio": sharpe_ratio,
        "max_position_size": max_position_size,
        "depth_score": depth_score,
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

def run(*_args, **kwargs) -> Any:
    missing = check_missing_fields(kwargs)
    if missing:
        errors.append(f"Required kwargs {missing} were not provided.")
        return {"error": errors}

    required_fields = list(REQUIRED_FIELDS)
    get_metrics = kwargs.get("get_metrics", False)
    if get_metrics:
        required_fields.append("position")

    kwargs = remove_irrelevant_fields(kwargs, required_fields)

    if get_metrics:
        metrics = calculate_metrics(**kwargs)
        if metrics is None:
            errors.append("Failed to calculate metrics.")
        return {"error": errors} if errors else metrics
    else:
        result = get_best_opportunities(**kwargs)
        if isinstance(result, dict) and "error" in result:
            errors.append(result["error"])
        if not result:
            errors.append("No suitable aggregators found")
        return {"error": errors} if errors else {"result": result}
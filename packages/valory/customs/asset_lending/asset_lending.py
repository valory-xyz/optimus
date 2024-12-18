import requests
from typing import Dict, Any, List, Optional
import numpy as np
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
REQUIRED_FIELDS = ("chains", "apr_threshold", "lending_asset", "current_pool", "coingecko_api_key")
STURDY = 'Sturdy'
TVL_WEIGHT = 0.6  # Weight for TVL
APR_WEIGHT = 0.4  # Weight for APR
TVL_PERCENTILE = 50
APR_PERCENTILE = 50
SCORE_PERCENTILE = 80
PERCENT_CONVERSION = 100
FETCH_AGGREGATOR_ENDPOINT = "https://us-central1-stu-dashboard-a0ba2.cloudfunctions.net/v2Aggregators"
coingecko_name_to_id = {
    "weth": "weth",
    "stone": "stakestone-ether",
    "ezeth": "renzo-restaked-eth",
    "mode": "mode",
}

def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    """Check for missing fields and return them, if any."""
    missing = [field for field in REQUIRED_FIELDS if kwargs.get(field) is None]
    return missing

def remove_irrelevant_fields(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the irrelevant fields from the given kwargs."""
    return {key: value for key, value in kwargs.items() if key in REQUIRED_FIELDS}

def filter_aggregators(chains, apr_threshold, aggregators, lending_asset, current_pool) -> List[Dict[str, Any]]:
    filtered_aggregators = []
    tvl_list = []
    apr_list = []

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
        logging.error("No suitable aggregator found.")
        return []

    if len(filtered_aggregators) <= 5:
        return filtered_aggregators

    tvl_threshold = np.percentile(tvl_list, TVL_PERCENTILE)
    apr_threshold = np.percentile(apr_list, APR_PERCENTILE)

    scored_aggregators = []
    max_tvl = max(tvl_list)
    max_apr = max(apr_list)

    for aggregator in filtered_aggregators:
        tvl = aggregator["tvl"]
        total_apr = aggregator["total_apr"]

        if tvl < tvl_threshold or total_apr < apr_threshold:
            continue
        score = TVL_WEIGHT * (tvl / max_tvl) + APR_WEIGHT * (total_apr / max_apr)
        aggregator["score"] = score
        scored_aggregators.append(aggregator)

    score_threshold = np.percentile([agg["score"] for agg in scored_aggregators], SCORE_PERCENTILE)
    filtered_scored_aggregators = [agg for agg in scored_aggregators if agg["score"] >= score_threshold]

    filtered_scored_aggregators.sort(key=lambda x: x["score"], reverse=True)

    if not filtered_scored_aggregators:
        logging.error("No suitable aggregator found after scoring.")
        return []

    if len(filtered_scored_aggregators) > 10:
        # Take only the top 10 scored pools
        top_pools = filtered_scored_aggregators[:10]
    else:
        top_pools = filtered_scored_aggregators

    return top_pools

def fetch_aggregators() -> List[Dict[str, Any]]:
    try:
        response = requests.get(FETCH_AGGREGATOR_ENDPOINT)
        response.raise_for_status()
        result = response.json()
        if 'errors' in result:
            logging.error(f"REST API Errors: {result['errors']}")
            return []
        return result
    except requests.RequestException as e:
        logging.error(f"REST API request failed: {e}")
        return []

def calculate_il_risk_score_for_lending(asset_token_1: str, asset_token_2: str, coingecko_api_key: str) -> float:
    """Calculate the IL Risk Score for a lending asset using two tokens."""
    if not (asset_token_1 or asset_token_1):
        logging.error("Tokens are required. Cannot calculate IL risk score without asset tokens")
        return  float('nan')
    
    cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
    
    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
    
    try:
        prices_1 = cg.get_coin_market_chart_range_by_id(id=asset_token_1, vs_currency='usd', from_timestamp=from_timestamp, to_timestamp=to_timestamp)
        prices_2 = cg.get_coin_market_chart_range_by_id(id=asset_token_2, vs_currency='usd', from_timestamp=from_timestamp, to_timestamp=to_timestamp)
    except Exception as e:
        logging.error(f"Error fetching price data: {e}")
        return float('nan')
    
    prices_1_data = np.array([x[1] for x in prices_1['prices']])
    prices_2_data = np.array([x[1] for x in prices_2['prices']])

    min_length = min(len(prices_1_data), len(prices_2_data))
    prices_1_data = prices_1_data[:min_length]
    prices_2_data = prices_2_data[:min_length]
 
    price_correlation = np.corrcoef(prices_1_data, prices_2_data)[0, 1]

    volatility_1 = np.std(prices_1_data)
    volatility_2 = np.std(prices_2_data)
    volatility_multiplier = np.sqrt(volatility_1 * volatility_2)

    P0 = prices_1_data[0] / prices_2_data[0]
    P1 = prices_1_data[-1] / prices_2_data[-1]
    il_impact = 1 - np.sqrt(P1 / P0) * (2 / (1 + P1 / P0))

    il_risk_score = il_impact * price_correlation * volatility_multiplier

    return float(il_risk_score)

def fetch_token_id(symbol):
    """Fetches the list of coins from the CoinGecko API."""
    url = "https://api.coingecko.com/api/v3/coins/list"
    try:
        response = requests.get(url)
        response.raise_for_status()
        coin_list = response.json()
    except requests.RequestException as e:
        logging.error(f"Failed to fetch coin list: {e}")
        return None

    for coin in coin_list:
        if coin['symbol'].lower() == symbol.lower():
            return coin['id']
        
    logging.error(f"Failed to fetch id for coin with symbol: {symbol}")
    return None

def get_best_opportunities(chains, apr_threshold, lending_asset, current_pool, coingecko_api_key) -> List[Dict[str, Any]]:
    data = fetch_aggregators()
    if not data:
        return []
    
    filtered_aggregators = filter_aggregators(chains, apr_threshold, data, lending_asset, current_pool)
    print(filtered_aggregators)

    if not filtered_aggregators:
        return []
    
    for aggregator in filtered_aggregators:
        silos = aggregator.get('whitelistedSilos', [])
        aggregator['il_risk_score'] = calculate_il_risk_score_for_silos(aggregator['asset']['symbol'], silos, coingecko_api_key)

    formatted_results = [format_aggregator(aggregator) for aggregator in filtered_aggregators]
    print(formatted_results)
    return formatted_results

def format_aggregator(aggregator) -> Dict[str, Any]:
    """Format a single aggregator into the desired structure."""
    return {
        "chain": aggregator['chainName'],
        "pool_address": aggregator['address'],
        "dex_type": STURDY,
        "token0_symbol": aggregator['asset']['symbol'],
        "token0": aggregator['asset']['address'],
        "apr": aggregator['total_apr'] * PERCENT_CONVERSION,
        "il_risk_score": aggregator['il_risk_score'],
        "whitelistedSilos": aggregator['whitelistedSilos']
    }

def calculate_il_risk_score_for_silos(token0_symbol, silos, coingecko_api_key):
    """Calculate the IL Risk Score for multiple silos."""
    il_risk_scores = []
    token_id_cache = {}
    
    # For sturdy, the token 0 becomes the lending asset, and since the aggregator invests in multiple silos we take an average of il risk score with all the silos
    # token 1 becomes the collateral token for each whitelisted silo
    token_0_symbol = token0_symbol.lower()
    if token_0_symbol in token_id_cache:
        token_0_id = token_id_cache[token_0_symbol]
    else:
        token_0_id = coingecko_name_to_id.get(token_0_symbol)
        if not token_0_id:
            token_0_id = fetch_token_id(token_0_symbol)

    for silo in silos:
        token_1_symbol = silo['collateral'].lower()
        if token_1_symbol in token_id_cache:
            token_1_id = token_id_cache[token_1_symbol]
        else:
            token_1_id = coingecko_name_to_id.get(token_1_symbol)
            if not token_1_id:
                token_1_id = fetch_token_id(token_1_symbol)

        if token_0_id and token_1_id:
            token_id_cache[token_0_symbol] = token_0_id
            token_id_cache[token_1_symbol] = token_1_id

            il_risk_score = calculate_il_risk_score_for_lending(token_0_id, token_1_id, coingecko_api_key)
            # Normalize the IL risk score to be between 0 and 1
            normalized_il_risk_score = min(max(il_risk_score, 0), 1)
            il_risk_scores.append(normalized_il_risk_score)
        else:
            logging.error(f"Failed to fetch token IDs for silo: {silo['collateral']}")

    if not il_risk_scores:
        return float('nan')

    # Calculate the average IL risk score
    return sum(il_risk_scores) / len(il_risk_scores)

def calculate_metrics(current_pool: Dict[str, Any], coingecko_api_key: str, **kwargs) -> Optional[Dict[str, Any]]:
    il_risk_score = calculate_il_risk_score_for_silos(current_pool.get("token0_symbol"), current_pool.get('whitelistedSilos',[]), coingecko_api_key)
    return {
        "il_risk_score": il_risk_score
    }

def run(*_args, **kwargs) -> List[Dict[str, Any]]:
    """Run the strategy."""
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
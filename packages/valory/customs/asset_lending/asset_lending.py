import requests
from typing import Dict, Union, Any, List
import numpy as np
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta

# Constants
REQUIRED_FIELDS = ("chains", "apr_threshold", "endpoint", "lending_asset", "current_pool", "coingecko-api-key")
STURDY = 'Sturdy'
TVL_WEIGHT = 0.7  # Weight for TVL
APR_WEIGHT = 0.3  # Weight for APR
TVL_PERCENTILE = 50
APR_PERCENTILE = 50
SCORE_PERCENTILE = 80
PERCENT_CONVERSION = 100

def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    """Check for missing fields and return them, if any."""
    missing = []
    for field in REQUIRED_FIELDS:
        if kwargs.get(field, None) is None:
            missing.append(field)
    return missing

def remove_irrelevant_fields(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the irrelevant fields from the given kwargs."""
    result = {key: value for key, value in kwargs.items() if key in REQUIRED_FIELDS}
    return result

def filter_aggregators(chains, apr_threshold, aggregators, lending_asset, current_pool) -> Dict[str, Any]:
    filtered_aggregators = []
    tvl_list = []
    apr_list = []

    for aggregator in aggregators:
        if aggregator.get("chainName") in chains:
            if aggregator.get('address') != current_pool:
                if aggregator.get("asset", {}).get("address") == lending_asset:
                    total_apr = aggregator.get('apy', {}).get('total', 0) * PERCENT_CONVERSION
                    tvl = aggregator.get('tvl', 0)
                    
                    tvl_list.append(tvl)
                    apr_list.append(total_apr)
                    aggregator["total_apr"] = total_apr
                    aggregator["tvl"] = tvl
                    filtered_aggregators.append(aggregator)

    if not filtered_aggregators:
        return {"error": "No suitable aggregator found."}

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

    if filtered_scored_aggregators is None:
        return {"error": "No suitable aggregator found."}

    return filtered_scored_aggregators

def fetch_aggregators(endpoint) -> List[Dict[str, Any]]:
    response = requests.get(endpoint)
    if response.status_code != 200:
        return {"error": f"REST API request failed with status code {response.status_code}"}
    
    result = response.json()
    
    if 'errors' in result:
        return {"error": f"REST API Errors: {result['errors']}"}
    
    return result


def calculate_il_risk_score_for_lending(asset_token: str, coingecko_api_key: str) -> float:
    """Calculate the IL Risk Score for a lending asset using USDC as the default second token."""
    cg = CoinGeckoAPI(api_key=coingecko_api_key)
    
    token_1 = asset_token
    token_2 = 'usd-coin'  # USDC token ID in CoinGecko
    
    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
    
    # Fetch historical price data for the token pair
    prices_1 = cg.get_coin_market_chart_range_by_id(id=token_1, vs_currency='usd', from_timestamp=from_timestamp, to_timestamp=to_timestamp)
    prices_2 = cg.get_coin_market_chart_range_by_id(id=token_2, vs_currency='usd', from_timestamp=from_timestamp, to_timestamp=to_timestamp)
    
    # Extract price data
    prices_1_data = np.array([x[1] for x in prices_1['prices']])
    prices_2_data = np.array([x[1] for x in prices_2['prices']])
    
    # Price Correlation Calculation
    price_correlation = np.corrcoef(prices_1_data, prices_2_data)[0, 1]
    
    # Volatility Calculation
    volatility_1 = np.std(prices_1_data)
    volatility_2 = np.std(prices_2_data)
    volatility_multiplier = np.sqrt(volatility_1 * volatility_2)
    
    # Calculate IL Impact
    P0 = prices_1_data[0] / prices_2_data[0]
    P1 = prices_1_data[-1] / prices_2_data[-1]
    il_impact = 2 * np.sqrt(P1 / P0) / (1 + P1 / P0) - 1
    
    # Calculate IL Risk Score
    il_risk_score = il_impact * price_correlation * volatility_multiplier
    
    return il_risk_score

def get_coingecko_id_from_address(asset_address: str, coingecko_api_key: str) -> str:
    """Fetch the CoinGecko ID for a given asset address."""
    cg = CoinGeckoAPI(api_key=coingecko_api_key)
    try:
        # Fetch coin data using the contract address
        coin_data = cg.get_coin_info_from_contract_address_by_id(id='ethereum', contract_address=asset_address)
        return coin_data['id']
    except Exception as e:
        print(f"Error fetching CoinGecko ID: {e}")
        return None
    
def get_best_opportunities(chains, apr_threshold, endpoint, lending_asset, current_pool, coingecko_api_key) -> Dict[str, Any]:
    data = fetch_aggregators(endpoint)
    if "error" in data:
        return data
    
    aggregators = data
    filtered_aggregators = filter_aggregators(chains, apr_threshold, aggregators, lending_asset, current_pool)
    if "error" in filtered_aggregators:
        return filtered_aggregators
    
    lending_asset_id = get_coingecko_id_from_address(lending_asset, coingecko_api_key)
    
    # Calculate IL Risk Score for each aggregator
    for aggregator in filtered_aggregators:
        aggregator['il_risk_score'] = calculate_il_risk_score_for_lending(lending_asset_id, coingecko_api_key)

    formatted_results = [format_aggregator(aggregator) for aggregator in filtered_aggregators]
    return formatted_results

def format_aggregator(aggregator) -> Dict[str, Any]:
    """Format a single aggregator into the desired structure."""
    return {
        "chain": aggregator['chainName'],
        "pool_address": aggregator['address'],
        "dex_type": STURDY,
        "token0_symbol": aggregator['asset']['symbol'],
        "token0": aggregator['asset']['address'],
        "apr": aggregator['total_apr'],
        "il_risk_score": aggregator['il_risk_score']
    }

def run(*_args, **kwargs) -> Dict[str, Union[bool, str]]:
    """Run the strategy."""
    missing = check_missing_fields(kwargs)
    if len(missing) > 0:
        return {"error": f"Required kwargs {missing} were not provided."}

    kwargs = remove_irrelevant_fields(kwargs)
    result = get_best_opportunities(**kwargs)
    return result
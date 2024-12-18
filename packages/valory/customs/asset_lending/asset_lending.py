import requests
from typing import Dict, Any, List
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
    "WETH": "weth",
    "STONE": "stakestone-ether",
    "EZETH": "renzo-restaked-eth",
    "MODE": "mode",
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
    cg = CoinGeckoAPI(api_key=coingecko_api_key)
    
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
    if not filtered_aggregators:
        return []
    
    for aggregator in filtered_aggregators:
        token_0_id = coingecko_name_to_id.get(aggregator['asset']['symbol'].upper())
        if not token_0_id:
            token_0_id = fetch_token_id(aggregator['asset']['symbol'].lower())
        
        token_1_id = coingecko_name_to_id.get(aggregator['whitelistedSilos'][0]['collateral'].upper())
        if not token_1_id:
            token_1_id = fetch_token_id(aggregator['whitelistedSilos'][0]['collateral'].lower())

        if token_0_id and token_1_id:
            aggregator['il_risk_score'] = calculate_il_risk_score_for_lending(token_0_id, token_1_id, coingecko_api_key)
        else:
            logging.error(f"Failed to fetch token IDs for aggregator: {aggregator['name']}")
            aggregator['il_risk_score'] = float('nan')

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
        "apr": aggregator['total_apr'] * PERCENT_CONVERSION,
        "il_risk_score": aggregator['il_risk_score']
    }

# # New Liquidity Analytics functions  

def analyze_vault_liquidity(vault_data):
    """
    Analyze liquidity risk and key metrics for a given vault strategy.
    
    Parameters:
    vault_data (dict): Comprehensive vault strategy data
    
    Returns:
    dict: Detailed liquidity risk analysis
    """
    # Extract key data points
    tvl = vault_data.get('tvl', 0)
    total_assets = vault_data.get('totalAssets', 0)
    apy_total = vault_data.get('apy', {}).get('total', 0)
    asset_price = float(vault_data.get('asset', {}).get('price', 0))
    
    # Constant for price impact (standardized at 1%)
    PRICE_IMPACT = 0.01
    
    # Calculate Depth Score (Sturdy Protocol variant)
    # Formula: (TVL × Total Assets) / (Price Impact × 100)
    depth_score = (tvl * total_assets) / (PRICE_IMPACT * 100)
    
    # Liquidity Risk Multiplier
    # Formula: max(0, 1 - (1/depth_score))
    liquidity_risk_multiplier = max(0, 1 - (1 / depth_score)) if depth_score > 0 else 0
    
    # Maximum Position Size Calculation
    # Formula: 50 × (TVL × Liquidity Risk Multiplier) / 100
    max_position_size = 50 * (tvl * liquidity_risk_multiplier) / 100
    
    # Risk Assessment
    risk_assessment = {
        'depth_score': depth_score,
        'liquidity_risk_multiplier': liquidity_risk_multiplier,
        'max_position_size': max_position_size,
        'is_safe': depth_score > 50,
        'additional_metrics': {
            'tvl': tvl,
            'total_assets': total_assets,
            'total_apy': apy_total,
            'asset_price': asset_price,
            'chain': vault_data.get('chainName'),
            'vault_name': vault_data.get('name')
        }
    }
    
    return risk_assessment

# this function need to call for liquidity analytics
def process_vault_strategy(vault_data):
    """
    Process and print liquidity risk analysis for a vault strategy.
    
    Parameters:
    vault_data (dict): Comprehensive vault strategy data
    """
    analysis = analyze_vault_liquidity(vault_data)
    
    print("Vault Liquidity Risk Analysis")
    print("-" * 30)
    print(f"Vault: {analysis['additional_metrics']['vault_name']}")
    print(f"Chain: {analysis['additional_metrics']['chain']}")
    print(f"Depth Score: {analysis['depth_score']:.2f}")
    print(f"Liquidity Risk Multiplier: {analysis['liquidity_risk_multiplier']:.4f}")
    print(f"Maximum Position Size: ${analysis['max_position_size']:.2f}")
    print(f"Investment Safety: {'Safe' if analysis['is_safe'] else 'Risky'}")
    print("\nAdditional Metrics:")
    for key, value in analysis['additional_metrics'].items():
        print(f"{key.replace('_', ' ').title()}: {value}")

#example of vault_data - example_vault = {
        # "chainName": "ethereum",
        # "address": "0x7077ef67fe49ffb1260b893f2cd8475eeb72bbbb",
        # "totalAssets": 238681809123,
        # "baseAPY": 0.20493693509525,
        # "totalDebt": 2.2921391997888378e+23,
        # "name": "USDC AeraVault Strategy",
        # "tvl": 238669.968118449,
        # "apy": {
            # "total": 0.566377372091836,
            # "base": 0.20493693509525
        # },
        # "asset": {
            # "symbol": "USDC",
            # "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            # "price": "0.99995039",
            # "decimals": 6
        # }
    # } 


def run(*_args, **kwargs) -> Dict[str, Union[bool, str]]:
    """Run the strategy."""
    missing = check_missing_fields(kwargs)
    if missing:
        return {"error": f"Required kwargs {missing} were not provided."}

    kwargs = remove_irrelevant_fields(kwargs)
    result = get_best_opportunities(**kwargs)

    if not result:
        return {"error": "No suitable aggregators found"}
    
    return result
import requests
from typing import (
    Dict,
    Union,
    Any,
    List
)

REQUIRED_FIELDS = ("chains", "apr_threshold", "endpoint", "lending_asset", "current_pool")
STURDY = 'Sturdy'


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

def get_best_aggregator(chains, apr_threshold, aggregators, lending_asset, current_pool) -> Dict[str, Any]:
    best_aggregator = None
    highest_total_apr = 0

    for aggregator in aggregators:
        if aggregator.get("chainName") in chains:
            if aggregator.get('address') != current_pool:
                if aggregator.get("asset", {}).get("address") == lending_asset:
                    total_apr = aggregator.get('apy', {}).get('total', 0) * 100
                    if total_apr > apr_threshold and total_apr > highest_total_apr:
                        highest_total_apr = total_apr
                        best_aggregator = aggregator

    if best_aggregator is None:
        return {"error": "No suitable aggregator found."}

    return best_aggregator

def fetch_aggregators(endpoint) -> List[Dict[str, Any]]:
    response = requests.get(endpoint)
    if response.status_code != 200:
        return {"error": f"REST API request failed with status code {response.status_code}"}
    
    result = response.json()
    
    if 'errors' in result:
        return {"error": f"REST API Errors: {result['errors']}"}
    
    return result

def get_best_opportunity(chains, apr_threshold, endpoint, lending_asset, current_pool) -> Dict[str, Any]:
    data = fetch_aggregators(endpoint)
    if "error" in data:
        return data
    
    aggregators = data
    best_aggregator = get_best_aggregator(chains, apr_threshold, aggregators, lending_asset, current_pool)
    if "error" in best_aggregator:
        return best_aggregator

    final_result = {
        "chain": "mode",
        "pool_address": best_aggregator['address'],
        "dex_type": STURDY,
        "token0_symbol": best_aggregator['asset']['symbol'],
        "token0": best_aggregator['asset']['address'],
        "apr": best_aggregator['apy']['total'] * 100
    }
    return final_result

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
    if len(missing) > 0:
        return {"error": f"Required kwargs {missing} were not provided."}

    kwargs = remove_irrelevant_fields(kwargs)
    result = get_best_opportunity(**kwargs)
    return result
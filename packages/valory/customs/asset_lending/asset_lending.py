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

def run(*_args, **kwargs) -> Dict[str, Union[bool, str]]:
    """Run the strategy."""
    missing = check_missing_fields(kwargs)
    if len(missing) > 0:
        return {"error": f"Required kwargs {missing} were not provided."}

    kwargs = remove_irrelevant_fields(kwargs)
    result = get_best_opportunity(**kwargs)
    return result
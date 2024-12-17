import requests
import pandas as pd
import pyfolio as pf
import numpy as np
from datetime import datetime
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

def fetch_historical_data(limit: int = 4000):
    """
    Fetch historical data for WETH strategy.
    
    Args:
        limit (int): The number of data points to fetch.
    
    Returns:
        list: List of historical data entries.
    """
    # Calculate the timestamp for one year ago
    current_time_ms = int(datetime.now().timestamp() * 1000)
    one_year_ago_ms = current_time_ms - (365 * 24 * 60 * 60 * 1000)
    
    url = f"https://us-central1-stu-dashboard-a0ba2.cloudfunctions.net/getV2AggregatorHistoricalData?last_time={one_year_ago_ms}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch historical data from STURDY API.")
    return response.json()

def calculate_daily_returns(base_apy, reward_apy=0):
    """
    Convert annualized APY to daily returns.

    Args:
        base_apy (float): Base APY as a decimal (e.g., 0.01 for 1%).
        reward_apy (float): Rewards APY as a decimal.

    Returns:
        float: Daily return as a decimal.
    """
    annual_return = base_apy + reward_apy
    daily_return = (1 + annual_return) ** (1 / 365) - 1
    return daily_return

def calculate_sharpe_ratio(daily_returns):
    """
    Calculate Sharpe ratio using Pyfolio.

    Args:
        daily_returns (pd.Series): Series of daily returns.

    Returns:
        float: Sharpe ratio.
    """
    return pf.timeseries.sharpe_ratio(daily_returns)

def get_sharpe_ratio_for_address(address: str) -> float:
    """
    Calculate the Sharpe ratio for a given aggregator address.

    Args:
        address (str): The aggregator address.

    Returns:
        float: Sharpe ratio for the given address.
    """
    # Fetch historical data
    historical_data = fetch_historical_data()
    
    records = []
    for entry in historical_data:
        timestamp = entry['timestamp']
        if address in entry['doc']:
            data = entry['doc'][address]
            base_apy = data.get('baseAPY', 0)
            rewards_apy = data.get('rewardsAPY', 0)
            records.append({
                'timestamp': timestamp,
                'base_apy': base_apy,
                'rewards_apy': rewards_apy
            })
    
    # Convert records to DataFrame
    df = pd.DataFrame(records)
    # Calculate daily returns
    df['daily_return'] = df.apply(
        lambda row: calculate_daily_returns(row['base_apy'], row['rewards_apy']), axis=1
    )
    
    # Calculate Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(df['daily_return'])
    return sharpe_ratio

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
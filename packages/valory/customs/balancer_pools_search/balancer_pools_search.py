import requests
import pyfolio as pf
import pandas as pd
from typing import (
    Dict,
    Union,
    Any,
    List
)
# Supported pool types and their mappings
SUPPORTED_POOL_TYPES = {
    "WEIGHTED": "Weighted",
    "COMPOSABLE_STABLE": "ComposableStable",
    "LIQUIDITY_BOOTSTRAPING": "LiquidityBootstrapping",
    "META_STABLE": "MetaStable",
    "STABLE": "Stable",
    "INVESTMENT": "Investment"
}

REQUIRED_FIELDS = ("chains", "apr_threshold", "graphql_endpoint", "current_pool")
BALANCER = "balancerPool"

def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    """Check for missing fields and return them, if any."""
    missing = []
    for field in REQUIRED_FIELDS:
        if kwargs.get(field, None) is None:
            missing.append(field)
    return missing

def remove_irrelevant_fields(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the irrelevant fields from the given kwargs."""
    return {key: value for key, value in kwargs.items() if key in REQUIRED_FIELDS}

def run_query(query, graphql_endpoint, variables=None) -> Dict[str, Any]:
    headers = {
        'Content-Type': 'application/json',
    }
    payload = {
        'query': query,
        'variables': variables or {}
    }
    
    response = requests.post(graphql_endpoint, json=payload, headers=headers)
    if response.status_code != 200:
        return {"error": f"GraphQL query failed with status code {response.status_code}"}
    
    result = response.json()
    
    if 'errors' in result:
        return {"error": f"GraphQL Errors: {result['errors']}"}
    
    return result['data']

def get_best_pool(chains, apr_threshold, graphql_endpoint, current_pool) -> Dict[str, Any]:
    # Ensure all chain names are uppercase
    chain_names = [chain.upper() for chain in chains]
    chain_list_str = ', '.join(chain_names)
    
    # Build the GraphQL query with the specified chain names
    graphql_query = f"""
    {{
      poolGetPools(where: {{chainIn: [{chain_list_str}]}}) {{
        id
        address
        chain
        type
        poolTokens {{
          address
          symbol
        }}
        dynamicData {{
          aprItems {{
            type
            apr
          }}
        }}
      }}
    }}
    """

    # Execute the GraphQL query
    data = run_query(graphql_query, graphql_endpoint)
    if "error" in data:
        return data
    
    # Extract pools from the response
    pools = data.get("poolGetPools", [])
    # Filter pools by supported types and those with exactly two tokens
    filtered_pools = []
    for pool in pools:
        pool_type = pool.get('type')
        pool_address = pool.get('address')
        mapped_type = SUPPORTED_POOL_TYPES.get(pool_type)
        if mapped_type and len(pool.get('poolTokens', [])) == 2 and pool_address != current_pool:
            pool['type'] = mapped_type  # Update pool type to the mapped type
            filtered_pools.append(pool)

    best_pool = None
    highest_apr = 0
    for pool in filtered_pools:
        total_apr = get_total_apr(pool)
        if total_apr > (apr_threshold / 100) and total_apr > highest_apr:
            highest_apr = total_apr
            best_pool = pool

    if best_pool is None:
        return {"error": "No suitable pool found."}

    total_apr = get_total_apr(best_pool)
    
    dex_type = BALANCER
    chain = best_pool['chain'].lower()
    apr = total_apr * 100
    pool_address = best_pool['address']
    pool_id = best_pool['id']
    pool_type = best_pool['type']
    
    pool_tokens = best_pool['poolTokens']
    token0 = pool_tokens[0].get('address')
    token1 = pool_tokens[1].get('address')
    token0_symbol = pool_tokens[0].get('symbol')
    token1_symbol = pool_tokens[1].get('symbol')
    
    pool_token_dict = {
        "token0": token0,
        "token1": token1,
        "token0_symbol": token0_symbol,
        "token1_symbol": token1_symbol,
    }
    
    # Check for missing token information
    if any(v is None for v in pool_token_dict.values()):
        return {"error": "Missing token information in the pool."}
    
    result = {
        "dex_type": dex_type,
        "chain": chain,
        "apr": apr,
        "pool_address": pool_address,
        "pool_id": pool_id,
        "pool_type": pool_type,
        **pool_token_dict,
    }

    return result

def get_pool_sharpe_ratio(pool_id, chain, timerange='ONE_YEAR'):
    """
    Calculate Sharpe ratio for a Balancer pool.
    Parameters:
    pool_id (str): Balancer pool ID
    chain (str): Blockchain network (e.g., 'OPTIMISM', 'ETHEREUM')
    timerange (str): Time range for analysis ('ONE_YEAR', 'ONE_MONTH', etc.)
    """
    query = """
    {
        poolGetSnapshots(
            chain: %s
            id: "%s"
            range: %s
        ) {
            timestamp
            sharePrice
            fees24h
            totalLiquidity
        }
    }
    """ % (chain, pool_id, timerange)
    try:
        # Get data from Balancer API
        response = requests.post(
            "https://api-v3.balancer.fi/",
            json={'query': query}
        )
        data = response.json()['data']['poolGetSnapshots']
        print(data)
        # Prepare DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        # Calculate returns
        df['sharePrice'] = pd.to_numeric(df['sharePrice'])
        price_returns = df['sharePrice'].pct_change()
        # Add fee returns
        df['fees24h'] = pd.to_numeric(df['fees24h'])
        df['totalLiquidity'] = pd.to_numeric(df['totalLiquidity'])
        fee_returns = df['fees24h'] / df['totalLiquidity']
        # Total returns (price change + fees)
        total_returns = price_returns + fee_returns
        returns = total_returns.dropna()
        # Calculate Sharpe ratio using pyfolio
        sharpe_ratio = pf.timeseries.sharpe_ratio(returns)
        return sharpe_ratio
    except Exception as e:
        print(f"Error calculating Sharpe ratio: {str(e)}")
        return None


def get_total_apr(pool) -> float:
    apr_items = pool.get('dynamicData', {}).get('aprItems', [])
    filtered_apr_items = [
        item for item in apr_items
        if item['type'] not in {"IB_YIELD", "MERKL", "SWAP_FEE", "SWAP_FEE_7D", "SWAP_FEE_30D"}
    ]
    return sum(item['apr'] for item in filtered_apr_items)
    
def run(*_args, **kwargs) -> Dict[str, Union[bool, str]]:
    """Run the strategy."""
    missing = check_missing_fields(kwargs)
    if len(missing) > 0:
        return {"error": f"Required kwargs {missing} were not provided."}

    kwargs = remove_irrelevant_fields(kwargs)
    return get_best_pool(**kwargs)
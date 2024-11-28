import requests
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

REQUIRED_FIELDS = ("chains", "apr_threshold", "graphql_endpoint")
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

def run_query(query, graphql_endpoint, variables=None):
    headers = {
        'Content-Type': 'application/json',
    }
    payload = {
        'query': query,
        'variables': variables or {}
    }
    
    response = requests.post(graphql_endpoint, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    
    if 'errors' in result:
        print("GraphQL Errors:", result['errors'])
        return None
    return result['data']

def get_best_pool(chains, apr_threshold, graphql_endpoint):
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
    if data is None:
        return None
    
    # Extract pools from the response
    pools = data.get("poolGetPools", [])
    # Filter pools by supported types and those with exactly two tokens
    filtered_pools = []
    for pool in pools:
        pool_type = pool.get('type')
        mapped_type = SUPPORTED_POOL_TYPES.get(pool_type)
        if mapped_type and len(pool.get('poolTokens', [])) == 2:
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
        return None

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
        return None
    
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

def get_total_apr(pool):
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
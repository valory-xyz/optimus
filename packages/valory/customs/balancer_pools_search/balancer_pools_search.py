import requests
from typing import (
    Dict,
    Union,
    Any,
    List,
    Optional
)
import statistics
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

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

def get_total_apr(pool) -> float:
    apr_items = pool.get('dynamicData', {}).get('aprItems', [])
    filtered_apr_items = [
        item for item in apr_items
        if item['type'] not in {"IB_YIELD", "MERKL", "SWAP_FEE", "SWAP_FEE_7D", "SWAP_FEE_30D"}
    ]
    return sum(item['apr'] for item in filtered_apr_items)



# New Liquidity Analytics functions 

def create_graphql_client(api_url='https://api-v3.balancer.fi') -> Client:
    """
    Create a GraphQL client for Balancer API
    
    :param api_url: GraphQL API endpoint
    :return: A configured GraphQL client ready for executing queries
    """
    transport = RequestsHTTPTransport(url=api_url, verify=True, retries=3)
    return Client(transport=transport, fetch_schema_from_transport=False)

def create_pool_snapshots_query(pool_id: str, range: str = 'NINETY_DAYS', chain: str = 'MAINNET') -> gql:
    """
    Create GraphQL query for fetching pool snapshots
    
    :param pool_id: Balancer pool ID
    :param range: Time range for snapshots
    :param chain: Blockchain network
    :return: A GraphQL query object for retrieving pool snapshots
    """
    return gql(f'''
    query GetLiquidityMetrics {{
      poolGetSnapshots(
        id: "{pool_id}",
        range: {range},
        chain: {chain}
      ) {{
        totalLiquidity
        volume24h
        timestamp
      }}
    }}
    ''')

def fetch_liquidity_metrics(
    pool_id: str, 
    client: Optional[Client] = None, 
    price_impact: float = 0.01
) -> Optional[Dict[str, Any]]:
    """
    Fetch and analyze liquidity metrics for a specific pool
    
    :param pool_id: Balancer pool ID
    :param client: Optional GraphQL client (will create one if not provided)
    :param price_impact: Standardized price impact (default 1%)
    :return: A dictionary containing calculated liquidity metrics, or None if retrieval fails
             Returned dictionary includes:
             - 'Average TVL': Total Value Locked average
             - 'Average Daily Volume': Average 24h trading volume
             - 'Depth Score': Liquidity depth calculation
             - 'Liquidity Risk Multiplier': Risk assessment factor
             - 'Maximum Position Size': Recommended max investment
             - 'Meets Depth Score Threshold': Boolean indicating liquidity quality
    """
    # Use provided client or create a new one
    if client is None:
        client = create_graphql_client()
    
    try:
        # Create and execute query
        query = create_pool_snapshots_query(pool_id)
        response = client.execute(query)
        pool_snapshots = response['poolGetSnapshots']
        
        # Validate snapshots
        if not pool_snapshots:
            raise ValueError("No pool snapshots retrieved")
        
        # Calculate average metrics
        avg_tvl = statistics.mean(float(snapshot['totalLiquidity']) for snapshot in pool_snapshots)
        avg_volume = statistics.mean(float(snapshot.get('volume24h', 0)) for snapshot in pool_snapshots)
        
        # Depth Score Calculation
        depth_score = (avg_tvl * avg_volume) / (price_impact * 100)
        
        # Liquidity Risk Multiplier
        liquidity_risk_multiplier = max(0, 1 - (1 / depth_score)) if depth_score != 0 else 0
        
        # Maximum Position Size
        max_position_size = 50 * (avg_tvl * liquidity_risk_multiplier) / 100
        
        # Prepare results
        metrics = {
            'Average TVL': avg_tvl,
            'Average Daily Volume': avg_volume,
            'Depth Score': depth_score,
            'Liquidity Risk Multiplier': liquidity_risk_multiplier,
            'Maximum Position Size': max_position_size,
            'Meets Depth Score Threshold': depth_score > 50
        }
        
        return metrics
    
    except Exception as e:
        print(f"An error occurred while analyzing pool metrics: {e}")
        return None


# this function need to call for liquidity analytics

def analyze_pool_liquidity(
    pool_id: str, 
    client: Optional[Client] = None, 
    price_impact: float = 0.01
) -> Optional[Dict[str, Any]]:
    """
    Comprehensive analysis of pool liquidity with risk assessment
    
    :param pool_id: Balancer pool ID
    :param client: Optional GraphQL client
    :param price_impact: Standardized price impact
    :return: Detailed analysis metrics dictionary if successful, None otherwise
             Returns the same dictionary as fetch_liquidity_metrics()
             When called, also prints a detailed console report of liquidity metrics
             and risk assessment
    """
    # Fetch and calculate metrics
    metrics = fetch_liquidity_metrics(pool_id, client, price_impact)
    
    if metrics is None:
        print("Could not retrieve pool metrics.")
        return None
    
    # Print detailed report
    print("Balancer Pool Liquidity Analysis Report")
    print("-" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Risk Assessment
    risk_assessment = []
    if metrics['Depth Score'] > 50:
        risk_assessment.append("✓ Depth Score meets threshold")
    else:
        risk_assessment.append("✗ Depth Score below recommended threshold")
    
    if metrics['Liquidity Risk Multiplier'] > 0.5:
        risk_assessment.append("✓ Low Liquidity Risk")
    else:
        risk_assessment.append("⚠ Moderate to High Liquidity Risk")
    
    print("\nRisk Assessment:")
    for assessment in risk_assessment:
        print(assessment)
    
    return metrics


def run(*_args, **kwargs) -> Dict[str, Union[bool, str]]:
    """Run the strategy."""
    missing = check_missing_fields(kwargs)
    if len(missing) > 0:
        return {"error": f"Required kwargs {missing} were not provided."}

    kwargs = remove_irrelevant_fields(kwargs)
    return get_best_pool(**kwargs)
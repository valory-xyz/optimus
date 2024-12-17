from typing import Dict, Union, Any, List
import requests
import logging
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

# Constants
UNISWAP = "UniswapV3"
REQUIRED_FIELDS = ("chains", "apr_threshold", "graphql_endpoints", "current_pool")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    """Check for missing fields and return them, if any."""
    logger.debug("Checking for missing fields in kwargs.")
    missing = []
    for field in REQUIRED_FIELDS:
        if kwargs.get(field, None) is None:
            missing.append(field)
    logger.info(f"Missing fields: {missing}")
    return missing

def remove_irrelevant_fields(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the irrelevant fields from the given kwargs."""
    logger.debug("Removing irrelevant fields from kwargs.")
    return {key: value for key, value in kwargs.items() if key in REQUIRED_FIELDS}

def run_query(query: str, graphql_endpoint: str, variables: Dict = None) -> Dict[str, Any]:
    """Execute a GraphQL query and return the results."""
    logger.info(f"Running GraphQL query on endpoint: {graphql_endpoint}")
    transport = RequestsHTTPTransport(url=graphql_endpoint)
    client = Client(transport=transport, fetch_schema_from_transport=True)
    
    try:
        query = gql(query)
        result = client.execute(query, variable_values=variables)
        logger.info("GraphQL query executed successfully.")
        return result
    except Exception as e:
        logger.error(f"GraphQL query failed: {str(e)}")
        return {"error": f"GraphQL query failed: {str(e)}"}

def calculate_apr(daily_volume: float, tvl: float, fee_rate: float) -> float:
    """Calculate APR using the formula: (Daily Volume / TVL) × Fee Rate × 365 × 100"""
    logger.debug(f"Calculating APR with daily_volume: {daily_volume}, tvl: {tvl}, fee_rate: {fee_rate}")
    if tvl == 0:
        logger.warning("TVL is zero, returning APR as 0.")
        return 0
    apr = (daily_volume / tvl) * fee_rate * 365 * 100
    logger.info(f"Calculated APR: {apr}")
    return apr

def get_pools_for_chain(chain: str, graphql_endpoint: str, current_pool: str, apr_threshold: float) -> List[Dict[str, Any]]:
    """Get all qualifying pools for a specific chain."""
    graphql_query = """
    {
        pools(
            first: 1000,
            orderBy: totalValueLockedUSD,
            orderDirection: desc,
            subgraphError: allow
        ) {
            id
            feeTier
            liquidity
            volumeUSD
            totalValueLockedUSD
            token0 {
                id
                symbol
                decimals
                derivedETH
            }
            token1 {
                id
                symbol
                decimals
                derivedETH
            }
        }
        bundles(where: {id: "1"}) {
            ethPriceUSD
        }
    }
    """
    logger.info(f"Fetching pools for chain: {chain}")
    # ... existing code ...
    data = run_query(graphql_query, graphql_endpoint)
    if "error" in data:
        logger.error("Error in fetching pools data.")
        return []
    
    qualifying_pools = []
    pools = data.get("pools", [])
    
    for pool in pools:
        if pool['id'] == current_pool:
            continue
            
        # Calculate pool metrics
        fee_rate = float(pool['feeTier']) / 1000000  # Convert basis points to decimal
        tvl = float(pool['totalValueLockedUSD'])
        daily_volume = float(pool['volumeUSD'])
        
        # Calculate APR
        apr = calculate_apr(daily_volume, tvl, fee_rate)
        
        if apr > apr_threshold and pool['id'] != current_pool:
            logger.debug(f"Pool {pool['id']} qualifies with APR: {apr}")
            qualifying_pools.append({
                "chain": chain.lower(),
                "apr": apr,
                "pool_address": pool['id'],
                "pool_id": pool['id'],
                "token0": pool['token0']['id'],
                "token1": pool['token1']['id'],
                "token0_symbol": pool['token0']['symbol'],
                "token1_symbol": pool['token1']['symbol'],
                "tvl": tvl,
                "daily_volume": daily_volume
            })
    
    logger.info(f"Found {len(qualifying_pools)} qualifying pools for chain: {chain}")
    return qualifying_pools

def get_uniswap_pool_sharpe_ratio(pool_id: str, graphql_endpoint: str) -> float:
    """Calculate Sharpe ratio for a Uniswap pool."""
    query = """
    {
        pool(id: "%s") {
            liquidity
            swaps(first: 5, orderBy: timestamp, orderDirection: desc) {
                timestamp
                amountUSD
            }
        }
    }
    """ % pool_id

    try:
        # Get data from Uniswap subgraph
        response = requests.post(
            graphql_endpoint,
            json={'query': query}
        )
        pool_data = response.json()['data']['pool']
        swaps_data = pool_data['swaps']
        liquidity = float(pool_data['liquidity'])
        
        # Prepare DataFrame
        df = pd.DataFrame(swaps_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        # Calculate returns
        df['amountUSD'] = pd.to_numeric(df['amountUSD'])
        price_returns = df['amountUSD'].pct_change()
        # Liquidity returns
        liquidity_returns = df['amountUSD'] / liquidity
        # Total returns (price change + liquidity)
        total_returns = price_returns + liquidity_returns
        returns = total_returns.dropna()
        # Calculate Sharpe ratio using pyfolio
        sharpe_ratio = pf.timeseries.sharpe_ratio(returns)
        return sharpe_ratio
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {str(e)}")
        return None
    
def get_best_pool(chains: List[str], apr_threshold: float, graphql_endpoints: Dict[str, str], current_pool: str) -> Dict[str, Any]:
    """Find the best Uniswap pool across all specified chains."""
    logger.info("Finding the best pool across all chains.")
    all_qualifying_pools = []
    
    # Collect qualifying pools from all chains
    for chain in chains:
        if chain not in graphql_endpoints:
            logger.warning(f"GraphQL endpoint not found for chain: {chain}")
            continue
            
        chain_pools = get_pools_for_chain(
            chain=chain,
            graphql_endpoint=graphql_endpoints[chain],
            current_pool=current_pool,
            apr_threshold=apr_threshold
        )
        all_qualifying_pools.extend(chain_pools)
    
    if not all_qualifying_pools:
        logger.error("No suitable pools found across any chain.")
        return {"error": "No suitable pools found across any chain."}
    
    # Find the pool with highest APR
    best_pool = max(all_qualifying_pools, key=lambda x: x['apr'])
    logger.info(f"Best pool found: {best_pool['pool_id']} with APR: {best_pool['apr']}")
    
    # Format the result
    result = {
        "dex_type": UNISWAP,
        "chain": best_pool['chain'],
        "apr": best_pool['apr'],
        "pool_address": best_pool['pool_address'],
        "pool_id": best_pool['pool_id'],
        "token0": best_pool['token0'],
        "token1": best_pool['token1'],
        "token0_symbol": best_pool['token0_symbol'],
        "token1_symbol": best_pool['token1_symbol'],
    }
    
    return result

def run(*_args, **kwargs) -> Dict[str, Union[bool, str]]:
    """Run the strategy."""
    logger.info("Running the strategy.")
    # Validate required fields
    missing = check_missing_fields(kwargs)
    if len(missing) > 0:
        logger.error(f"Required kwargs {missing} were not provided.")
        return {"error": f"Required kwargs {missing} were not provided."}
    
    # Remove irrelevant fields and execute
    kwargs = remove_irrelevant_fields(kwargs)
    return get_best_pool(**kwargs)
from typing import Dict, Union, Any, List, Optional
import requests
import logging
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import time
import json

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


# requirement to run the function successful
# API details
# API_KEY = ""  # Replace with your API key
# SUBGRAPH_URL = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

# # Specific pool ID to analyze
# POOL_ID = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"

# New Liquidity Analytics functions  

def fetch_pool_data(pool_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch pool data for a specific pool ID from The Graph.

    Args:
        pool_id (str): The unique identifier of the pool to fetch data for.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing pool data, or None if retrieval fails.
    """
    query = {
        "query": f"""
        {{
          pool(id: "{pool_id.lower()}") {{
            id
            token0 {{
              id
              symbol
              decimals
            }}
            token1 {{
              id
              symbol
              decimals
            }}
            liquidity
            totalValueLockedUSD
            totalValueLockedToken0
            totalValueLockedToken1
          }}
        }}
        """
    }
    try:
        print(f"Fetching data for pool ID: {pool_id}")
        response = requests.post(
            SUBGRAPH_URL,
            json=query,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

        print(f"Response Status Code: {response.status_code}")

        # Print full response for debugging
        response_json = response.json()
        print("Full Response:")
        print(json.dumps(response_json, indent=2))

        if response.status_code == 200:
            # Check for GraphQL errors
            if "errors" in response_json:
                print("GraphQL Errors:")
                print(json.dumps(response_json["errors"], indent=2))
                return None

            # Check for valid pool data
            data = response_json.get("data", {})
            pool = data.get("pool")

            if pool is None:
                print("No pool data found for the given ID")
                return None

            return pool
        else:
            print(f"HTTP Error: {response.status_code}")
            print(f"Response Text: {response.text}")
            return None

    except requests.RequestException as e:
        print(f"Request Exception: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

def fetch_24_hour_volume(pool_id: str) -> List[Dict[str, Union[int, float, str]]]:
    """
    Fetch 24-hour volume data for a specific pool from The Graph.

    Args:
        pool_id (str): The unique identifier of the pool to fetch volume data for.

    Returns:
        List[Dict[str, Union[int, float, str]]]: A list of dictionaries containing volume data,
        or an empty list if retrieval fails.
    """
    timestamp_24h_ago = int(time.time()) - 86400
    query = {
        "query": f"""
        {{
          poolDayDatas(first: 1, orderBy: date, orderDirection: desc, where: {{
            pool: "{pool_id.lower()}",
            date_gt: {timestamp_24h_ago}
          }}) {{
            date
            liquidity
            volumeUSD
            volumeToken0
            volumeToken1
          }}
        }}
        """
    }
    try:
        response = requests.post(
            SUBGRAPH_URL,
            json=query,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

        if response.status_code == 200:
            data = response.json()
            pool_day_datas = data.get("data", {}).get("poolDayDatas", [])

            if not pool_day_datas:
                print("No volume data found for the pool")
                return []

            return pool_day_datas
        else:
            print(f"Error fetching 24-hour volume: {response.status_code}")
            print(f"Response Text: {response.text}")
            return []

    except Exception as e:
        print(f"Exception in volume fetch: {e}")
        return []

def calculate_metrics(
    pool_data: Dict[str, Any], 
    volume_data: List[Dict[str, Union[int, float, str]]]
) -> Optional[Dict[str, Union[str, float]]]:
    """
    Calculate liquidity and risk metrics for a pool based on pool and volume data.

    Args:
        pool_data (Dict[str, Any]): Comprehensive data about the pool.
        volume_data (List[Dict[str, Union[int, float, str]]]): 24-hour volume data.

    Returns:
        Optional[Dict[str, Union[str, float]]]: A dictionary of calculated metrics,
        or None if calculation fails.
    """
    try:
        # Default values to handle potential missing data
        liquidity = float(pool_data.get("liquidity", 0))
        tvl = float(pool_data.get("totalValueLockedUSD", 0))

        # Use total value locked for tokens instead of reserves
        tvl_token0 = float(pool_data.get("totalValueLockedToken0", 0))
        tvl_token1 = float(pool_data.get("totalValueLockedToken1", 0))

        volume_usd = float(volume_data[0]["volumeUSD"]) if volume_data else 0
        token0 = pool_data.get("token0", {}).get("symbol", "Token0")
        token1 = pool_data.get("token1", {}).get("symbol", "Token1")

        # Depth Score Calculation (using TVL of tokens)
        depth_score = (
            (tvl_token0 * tvl_token1) / (PRICE_IMPACT * 100)
            if tvl_token0 > 0 and tvl_token1 > 0
            else 0
        )

        # Liquidity Risk Multiplier
        liquidity_risk_multiplier = (
            max(0, 1 - (1 / depth_score)) if depth_score > 0 else 0
        )

        # Maximum Position Size
        max_position_size = MAX_POSITION_BASE * (tvl * liquidity_risk_multiplier) / 100

        # Print Detailed Metrics
        print(f"\n===== Pool Analysis: {token0}-{token1} =====")
        print(f"Total Value Locked: ${tvl:,.2f}")
        print(f"Total Value Locked Token0: {tvl_token0:,.4f} {token0}")
        print(f"Total Value Locked Token1: {tvl_token1:,.4f} {token1}")
        print(f"24h Volume: ${volume_usd:,.2f}")
        print(f"\nDepth Score: {depth_score:,.4f}")
        print(f"Liquidity Risk Multiplier: {liquidity_risk_multiplier:.4f}")
        print(f"Maximum Position Size: ${max_position_size:,.2f}")

        return {
            "token_pair": f"{token0}-{token1}",
            "tvl": tvl,
            "tvl_token0": tvl_token0,
            "tvl_token1": tvl_token1,
            "volume_24h": volume_usd,
            "depth_score": depth_score,
            "liquidity_risk_multiplier": liquidity_risk_multiplier,
            "max_position_size": max_position_size,
        }

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

# this function need to call for liquidity analytics

def assess_pool_liquidity(pool_id: str) -> Optional[Dict[str, Union[str, float]]]:
    """
    Comprehensively assess the liquidity of a specific pool.

    Args:
        pool_id (str): The unique identifier of the pool to assess.

    Returns:
        Optional[Dict[str, Union[str, float]]]: A dictionary of pool liquidity metrics,
        or None if assessment fails.
    """
    # Fetch pool data
    pool_data = fetch_pool_data(pool_id)

    # Add explicit check for None
    if pool_data is None:
        print(f"Could not retrieve data for pool {pool_id}")
        return None

    try:
        # Fetch volume data
        volumes = fetch_24_hour_volume(pool_id)

        # Calculate and return metrics
        return calculate_metrics(pool_data, volumes)

    except Exception as e:
        print(f"Error processing pool data: {e}")
        return None



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
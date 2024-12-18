import requests
from typing import Dict, Union, Any, List
from pycoingecko import CoinGeckoAPI
import numpy as np
from datetime import datetime, timedelta
import logging
from web3 import Web3

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
UNISWAP = "UniswapV3"
REQUIRED_FIELDS = ("chains", "apr_threshold", "graphql_endpoints", "current_pool", "coingecko_api_key")
FEE_RATE_DIVISOR = 1000000
DAYS_IN_YEAR = 365
PERCENT_CONVERSION = 100
TVL_WEIGHT = 0.7
APR_WEIGHT = 0.3
TVL_PERCENTILE = 50
APR_PERCENTILE = 50
SCORE_PERCENTILE = 80
CHAIN_URLS = {
    "mode": "https://1rpc.io/mode",
    "optimism": "https://mainnet.optimism.io",
    "base": "https://1rpc.io/base"
}

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

def fetch_coin_list():
    """Fetches the list of coins from the CoinGecko API."""
    url = "https://api.coingecko.com/api/v3/coins/list"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Failed to fetch coin list: {e}")
        return None

def get_token_id_from_symbol(token_address, symbol, coin_list, chain_name):
    matching_coins = [coin for coin in coin_list if coin['symbol'].lower() == symbol.lower()]

    if not matching_coins:
        logging.error(f"No entries found for symbol: {symbol}")
        return None

    # If there's only one matching coin, return its ID
    if len(matching_coins) == 1:
        return matching_coins[0]['id']

    # If multiple entries exist, fetch the token name from the contract
    token_name = fetch_token_name_from_contract(chain_name, token_address)
    
    if not token_name:
        logging.error(f"Failed to fetch token name for address: {token_address} on chain: {chain_name}")
        return None

    for coin in matching_coins:
        if coin['name'].replace(" ", "") == token_name.replace(" ", "") or coin['name'].lower() == symbol.lower():
            return coin['id']
        
    logging.error(f"Failed to fetch id for coin with symbol: {symbol} and name: {token_name}")
    return None

def fetch_token_name_from_contract(chain_name, token_address):
    ERC20_ABI = [
        {
            "constant": True,
            "inputs": [],
            "name": "name",
            "outputs": [{"name": "", "type": "string"}],
            "payable": False,
            "stateMutability": "view",
            "type": "function",
        }
    ]

    # Get the appropriate URL for the chain
    chain_url = CHAIN_URLS.get(chain_name)
    if not chain_url:
        logging.error(f"Unsupported chain: {chain_name}")
        return None

    # Connect to the blockchain
    web3 = Web3(Web3.HTTPProvider(chain_url))
    if not web3.is_connected():
        logging.error(f"Failed to connect to the {chain_name} blockchain.")
        return None

    # Create a contract instance
    contract = web3.eth.contract(address=Web3.to_checksum_address(token_address), abi=ERC20_ABI)

    try:
        # Call the 'name' function of the contract
        token_name = contract.functions.name().call()
        return token_name
    except Exception as e:
        logging.error(f"Error fetching token name from contract: {e}")
        return None

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

def get_filtered_pools(pools, current_pool) -> List[Dict[str, Any]]:
    # Filter pools by those with exactly two tokens
    qualifying_pools = []
    tvl_list = []
    apr_list = []

    for pool in pools:
        # Calculate pool metrics
        fee_rate = float(pool['feeTier']) / FEE_RATE_DIVISOR  # Convert basis points to decimal
        tvl = float(pool['totalValueLockedUSD'])
        daily_volume = float(pool['volumeUSD'])
        
        # Calculate APR
        apr = calculate_apr(daily_volume, tvl, fee_rate)
        pool["apr"] = apr
        pool["tvl"] = tvl
        if pool['id'] != current_pool:
            qualifying_pools.append(pool)


    if not qualifying_pools:
        logging.error("No suitable pools found.")
        return []
    
    if len(qualifying_pools) <= 5:
        return qualifying_pools
    
    tvl_list = [float(pool.get("tvl", 0)) for pool in qualifying_pools]
    apr_list = [float(pool.get("apr", 0)) for pool in qualifying_pools]

    tvl_threshold = np.percentile(tvl_list, TVL_PERCENTILE)
    apr_threshold = np.percentile(apr_list, APR_PERCENTILE)

    # Prioritize pools using combined TVL and APR score
    scored_pools = []
    max_tvl = max(tvl_list)
    max_apr = max(apr_list)

    for pool in qualifying_pools:
        tvl = float(pool.get("tvl", 0))
        apr = float(pool.get("apr", 0))
                
        if tvl < tvl_threshold or apr < apr_threshold:
            continue

        score = TVL_WEIGHT * (tvl / max_tvl) + APR_WEIGHT * (apr / max_apr)
        pool["score"] = score
        scored_pools.append(pool)

    # Apply score threshold
    score_threshold = np.percentile([pool["score"] for pool in scored_pools], SCORE_PERCENTILE)
    filtered_scored_pools = [pool for pool in scored_pools if pool["score"] >= score_threshold]

    filtered_scored_pools.sort(key=lambda x: x["score"], reverse=True)

    if len(filtered_scored_pools) > 10:
        # Take only the top 10 scored pools
        top_pools = filtered_scored_pools[:10]
    else:
        top_pools = filtered_scored_pools

    return top_pools

def fetch_graphql_data(chains, graphql_endpoints, current_pool, apr_threshold) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Fetch data from GraphQL endpoint for the specified chains."""

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
    
    # Calculate IL Risk Score for each pool
    for pool in filtered_pools:
        pool['chain'] = pool['chain'].lower()
        token_0_id = get_token_id_from_symbol(pool['token0']['address'], pool['token0']['symbol'].lower(), coin_list, pool['chain'])
        token_1_id = get_token_id_from_symbol(pool['token1']['address'], pool['token1']['symbol'].lower(), coin_list, pool['chain'])

        if token_0_id and token_1_id:
            pool['il_risk_score'] = calculate_il_risk_score(token_0_id, token_1_id, coingecko_api_key)
        else:
            pool['il_risk_score'] = float('nan')
            
    formatted_pools = [format_pool_data(pool) for pool in filtered_pools]

    return formatted_pools


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
    missing = check_missing_fields(kwargs)
    if len(missing) > 0:
        return {"error": f"Required kwargs {missing} were not provided."}

    kwargs = remove_irrelevant_fields(kwargs)
    coin_list = fetch_coin_list()
    kwargs.update({"coin_list": coin_list})
    return get_best_pools(**kwargs)
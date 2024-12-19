import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import requests
from typing import Dict, Union, Any, List, Optional
from pycoingecko import CoinGeckoAPI
import numpy as np
from datetime import datetime, timedelta
import logging
from web3 import Web3
import pyfolio as pf
import pandas as pd
# Configure logging
logging.basicConfig(level=logging.INFO)
import time
import json

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
        return None

def get_token_id_from_symbol(token_address, symbol, coin_list, chain_name):
    matching_coins = [coin for coin in coin_list if coin['symbol'].lower() == symbol.lower()]

    if not matching_coins:
        return None

    # If there's only one matching coin, return its ID
    if len(matching_coins) == 1:
        return matching_coins[0]['id']

    # If multiple entries exist, fetch the token name from the contract
    token_name = fetch_token_name_from_contract(chain_name, token_address)

    if not token_name:
        return None

    for coin in matching_coins:
        if coin['name'].replace(" ", "") == token_name.replace(" ", "") or coin['name'].lower() == symbol.lower():
            return coin['id']
        
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
        return None

    # Connect to the blockchain
    web3 = Web3(Web3.HTTPProvider(chain_url))
    if not web3.is_connected():
        return None

    # Create a contract instance
    contract = web3.eth.contract(address=Web3.to_checksum_address(token_address), abi=ERC20_ABI)

    try:
        # Call the 'name' function of the contract
        token_name = contract.functions.name().call()
        return token_name
    except Exception as e:
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
                first: 100,
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
        data = run_query(graphql_query, graphql_endpoint)
        if "error" in data:
            logging.error("Error in fetching pools data.")
            return []
        
        pools = data.get("pools", [])
        for pool in pools:
            pool['chain'] = chain

        return pools

    all_pools = [] 
    for chain in chains:
        graphql_endpoint = graphql_endpoints.get(chain)
        if graphql_endpoint:
            data = get_pools_for_chain(chain, graphql_endpoint, current_pool, apr_threshold)
            if data:
                all_pools.extend(data)

    return all_pools

def get_uniswap_pool_sharpe_ratio(pool_address, graphql_endpoint, days_back=365):
    """
    Calculate Sharpe ratio for a Uniswap pool
    
    Parameters:
    pool_address (str): The Uniswap pool address
    days_back (int): Number of days of historical data
    
    Returns:
    float: Sharpe ratio
    """
    # Calculate start timestamp
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    start_timestamp = int(start_date.timestamp())
    
    # Query Uniswap subgraph
    query = """
    {
      poolDayDatas(
        where: {
          pool: "%s"
          date_gt: %d
        }
        orderBy: date
        orderDirection: asc
      ) {
        date
        tvlUSD
        feesUSD
      }
    }
    """ % (pool_address.lower(), start_timestamp)
    
    # Fetch data
    response = requests.post(graphql_endpoint, json={'query': query})
    pool_data = response.json()['data']['poolDayDatas']
    
    # Process data
    df = pd.DataFrame(pool_data)
    df['date'] = pd.to_datetime(df['date'].astype(int), unit='s')
    df['tvlUSD'] = pd.to_numeric(df['tvlUSD'])
    df['feesUSD'] = pd.to_numeric(df['feesUSD'])
    
    # Calculate returns
    df['total_value'] = df['tvlUSD'] + df['feesUSD']
    returns = df.set_index('date')['total_value'].pct_change().dropna()
    
    # Calculate Sharpe ratio using pyfolio
    sharpe = pf.timeseries.sharpe_ratio(returns)
    
    return float(sharpe)
    
def calculate_apr(daily_volume: float, tvl: float, fee_rate: float) -> float:
    """Calculate APR using the formula: (Daily Volume / TVL) × Fee Rate × 365 × 100"""
    if tvl == 0:
        return 0
    return (daily_volume / tvl) * fee_rate * DAYS_IN_YEAR * PERCENT_CONVERSION

def format_pool_data(pool) -> Dict[str, Any]:
    """Format pool data into the desired structure."""
    dex_type = UNISWAP
    chain = pool['chain'].lower()
    apr = pool['apr']
    pool_address = pool['id']
    
    token0 = pool['token0']['id']
    token1 = pool['token1']['id']
    token0_symbol = pool['token0']['symbol']
    token1_symbol = pool['token1']['symbol']
    
    return {
        "dex_type": dex_type,
        "chain": chain,
        "apr": apr,
        "pool_address": pool_address,
        "token0": token0,
        "token1": token1,
        "token0_symbol": token0_symbol,
        "token1_symbol": token1_symbol,
        "il_risk_score": pool['il_risk_score'],
        "sharpe_ratio": pool['sharpe_ratio'],
        "depth_score": pool['depth_score'],
        "max_position_size": pool['max_position_size']
    }

def calculate_il_risk_score(token_0, token_1, coingecko_api_key: str) -> float:
    """Calculate the IL Risk Score for a given pool."""
    cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
    
    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
    
    try:
        prices_1 = cg.get_coin_market_chart_range_by_id(id=token_0, vs_currency='usd', from_timestamp=from_timestamp, to_timestamp=to_timestamp)
        prices_2 = cg.get_coin_market_chart_range_by_id(id=token_1, vs_currency='usd', from_timestamp=from_timestamp, to_timestamp=to_timestamp)
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

def get_best_pools(chains, apr_threshold, graphql_endpoints, current_pool, coingecko_api_key, coin_list) -> List[Dict[str, Any]]:
    pools = fetch_graphql_data(chains, graphql_endpoints, current_pool, apr_threshold)
    if isinstance(pools, dict) and "error" in pools:
        return pools
    
    filtered_pools = get_filtered_pools(pools, current_pool)
    if not filtered_pools:
        return {"error": "No suitable pools found"}
    
    token_id_cache = {}
    # Calculate IL Risk Score for each pool
    for pool in filtered_pools:
        pool['chain'] = pool['chain'].lower()
        # Check if token0 ID is already cached
        token_0_symbol = pool['token0']['symbol'].lower()
        if token_0_symbol in token_id_cache:
            token_0_id = token_id_cache[token_0_symbol]
        else:
            token_0_id = get_token_id_from_symbol(pool['token0']['id'], token_0_symbol, coin_list, pool['chain'])
            if token_0_id:
                token_id_cache[token_0_symbol] = token_0_id

        # Check if token1 ID is already cached
        token_1_symbol = pool['token1']['symbol'].lower()
        if token_1_symbol in token_id_cache:
            token_1_id = token_id_cache[token_1_symbol]
        else:
            token_1_id = get_token_id_from_symbol(pool['token1']['id'], token_1_symbol, coin_list, pool['chain'])
            if token_1_id:
                token_id_cache[token_1_symbol] = token_1_id

        if token_0_id and token_1_id:
            pool['il_risk_score'] = calculate_il_risk_score(token_0_id, token_1_id, coingecko_api_key)
        else:
            pool['il_risk_score'] = float('nan')
        
        # Calculate Sharpe Ratio
        graphql_endpoint = graphql_endpoints.get(pool["chain"])
        pool['sharpe_ratio'] = get_uniswap_pool_sharpe_ratio(pool['id'], graphql_endpoint)   
        
        (pool['depth_score'],pool['max_position_size']) = assess_pool_liquidity(pool['id'], graphql_endpoint)

    formatted_pools = [format_pool_data(pool) for pool in filtered_pools]

    return formatted_pools


def calculate_metrics(current_pool: Dict[str, Any], coingecko_api_key: str, coin_list: List[Any], graphql_endpoints, **kwargs) -> Optional[Dict[str, Any]]:
    token_0_id = get_token_id_from_symbol(current_pool['token0'], current_pool['token0_symbol'], coin_list, current_pool['chain'])
    token_1_id = get_token_id_from_symbol(current_pool['token1'], current_pool['token1_symbol'], coin_list, current_pool['chain'])

    if token_0_id and token_1_id:
        il_risk_score = calculate_il_risk_score(token_0_id, token_1_id, coingecko_api_key)
    else:
        il_risk_score = float('nan')

    graphql_endpoint = graphql_endpoints.get(current_pool["chain"])
    sharpe_ratio = get_uniswap_pool_sharpe_ratio(current_pool['id'], graphql_endpoint)
    (depth_score,max_position_size) = assess_pool_liquidity(current_pool['id'], graphql_endpoint)
    return {
        "il_risk_score": il_risk_score,
        "sharpe_ratio": sharpe_ratio,
        "depth_score":depth_score,
        "max_position_size":max_position_size
    }


# Constants
PRICE_IMPACT = 0.01  # 1% standard price impact
MAX_POSITION_BASE = 50  # Base for maximum position calculation


def fetch_pool_data(pool_id: str, SUBGRAPH_URL: str) -> Optional[Dict[str, Any]]:
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
        logging.info(f"Fetching data for pool ID: {pool_id}")
        response = requests.post(
            SUBGRAPH_URL,
            json=query,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

        # logging.info full response for debugging
        response_json = response.json()

        if response.status_code == 200:
            # Check for GraphQL errors
            if "errors" in response_json:
                logging.error("GraphQL Errors:")
                logging.error(json.dumps(response_json["errors"], indent=2))
                return None

            # Check for valid pool data
            data = response_json.get("data", {})
            pool = data.get("pool")

            if pool is None:
                logging.info("No pool data found for the given ID")
                return None

            return pool
        else:
            logging.info(f"HTTP Error: {response.status_code}")
            logging.info(f"Response Text: {response.text}")
            return None

    except requests.RequestException as e:
        logging.error(f"Request Exception: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected Error: {e}")
        return None

def fetch_24_hour_volume(pool_id: str,SUBGRAPH_URL:str) -> List[Dict[str, Union[int, float, str]]]:
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
                logging.info("No volume data found for the pool")
                return []

            return pool_day_datas
        else:
            logging.info(f"Error fetching 24-hour volume: {response.status_code}")
            logging.info(f"Response Text: {response.text}")
            return []

    except Exception as e:
        logging.error(f"Exception in volume fetch: {e}")
        return []

def calculate_metrics_liquidity_risk(
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
        tvl = float(pool_data.get("totalValueLockedUSD", 0))

        # Use total value locked for tokens instead of reserves
        tvl_token0 = float(pool_data.get("totalValueLockedToken0", 0))
        tvl_token1 = float(pool_data.get("totalValueLockedToken1", 0))

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

        return {
            "depth_score": depth_score,
            "max_position_size": max_position_size,
        }

    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        return float('nan'), float('nan')

# this function need to call for liquidity analytics 

def assess_pool_liquidity(pool_id: str,SUBGRAPH_URL: str) -> Optional[Dict[str, Union[str, float]]]:
    """
    Comprehensively assess the liquidity of a specific pool.

    Args:
        pool_id (str): The unique identifier of the pool to assess.

    Returns:
        Optional[Dict[str, Union[str, float]]]: A dictionary of pool liquidity metrics,
        or None if assessment fails.
    """
    # Fetch pool data
    pool_data = fetch_pool_data(pool_id, SUBGRAPH_URL)

    # Add explicit check for None
    if pool_data is None:
        logging.info(f"Could not retrieve data for pool {pool_id}")
        return float('nan'), float('nan')

    try:
        # Fetch volume data
        volumes = fetch_24_hour_volume(pool_id, SUBGRAPH_URL)

        # Calculate and return metrics
        return calculate_metrics_liquidity_risk(pool_data, volumes)

    except Exception as e:
        logging.error(f"Error processing pool data: {e}")
        return float('nan'), float('nan')



def run(*_args, **kwargs) -> Dict[str, Union[bool, str]]:
    """Run the strategy."""
    missing = check_missing_fields(kwargs)
    if len(missing) > 0:
        return {"error": f"Required kwargs {missing} were not provided."}

    get_metrics = kwargs.get('get_metrics', False)
    kwargs = remove_irrelevant_fields(kwargs)
    coin_list = fetch_coin_list()
    kwargs.update({"coin_list": coin_list})

    if get_metrics:        
        return calculate_metrics(**kwargs)
    else:
        result = get_best_pools(**kwargs)
        if not result:
            return {"error": "No suitable aggregators found"}
        return result

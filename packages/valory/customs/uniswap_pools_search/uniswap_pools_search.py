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
    il_risk_score = pool['il_risk_score']
    
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
        "il_risk_score": il_risk_score
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

def run(*_args, **kwargs) -> Dict[str, Union[bool, str]]:
    """Run the strategy."""
    missing = check_missing_fields(kwargs)
    if len(missing) > 0:
        return {"error": f"Required kwargs {missing} were not provided."}

    kwargs = remove_irrelevant_fields(kwargs)
    coin_list = fetch_coin_list()
    kwargs.update({"coin_list": coin_list})
    return get_best_pools(**kwargs)
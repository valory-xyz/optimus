import requests
from typing import (
    Dict,
    Union,
    Any,
    List,
    Optional
)
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import numpy as np
import logging
from web3 import Web3

# Configure logging
logging.basicConfig(level=logging.INFO)

# Supported pool types and their mappings
SUPPORTED_POOL_TYPES = {
    "WEIGHTED": "Weighted",
    "COMPOSABLE_STABLE": "ComposableStable",
    "LIQUIDITY_BOOTSTRAPING": "LiquidityBootstrapping",
    "META_STABLE": "MetaStable",
    "STABLE": "Stable",
    "INVESTMENT": "Investment"
}

REQUIRED_FIELDS = ("chains", "apr_threshold", "graphql_endpoint", "current_pool", "coingecko_api_key")
BALANCER = "balancerPool"
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

def get_balancer_pools(chains, graphql_endpoint) -> Dict[str, Any]:
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
          totalLiquidity
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
    return pools

def get_filtered_pools(pools, current_pool):
    # Filter pools by supported types and those with exactly two tokens
    qualifying_pools = []
    for pool in pools:
        pool_type = pool.get('type')
        pool_address = pool.get('address')
        mapped_type = SUPPORTED_POOL_TYPES.get(pool_type)
        if mapped_type and len(pool.get('poolTokens', [])) == 2 and pool_address != current_pool:
            pool['type'] = mapped_type  # Update pool type to the mapped type
            pool['apr'] = get_total_apr(pool)
            pool['tvl'] = pool.get('dynamicData', {}).get('totalLiquidity')
            qualifying_pools.append(pool)

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

def get_total_apr(pool) -> float:
    apr_items = pool.get('dynamicData', {}).get('aprItems', [])
    filtered_apr_items = [
        item for item in apr_items
        if item['type'] not in {"IB_YIELD", "MERKL", "SWAP_FEE", "SWAP_FEE_7D", "SWAP_FEE_30D"}
    ]
    return sum(item['apr'] for item in filtered_apr_items)
    
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

def format_pool_data(pool) -> Dict[str, Any]:
    """Format pool data into the desired structure."""

    dex_type = BALANCER
    chain = pool['chain'].lower()
    apr = pool['apr'] * 100
    pool_address = pool['address']
    pool_id = pool['id']
    pool_type = pool['type']
    
    pool_tokens = pool['poolTokens']
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
    
    return {
        "dex_type": dex_type,
        "chain": chain,
        "apr": apr,
        "pool_address": pool_address,
        "pool_id": pool_id,
        "pool_type": pool_type,
        **pool_token_dict,
    }

def get_opportunities(chains, apr_threshold, graphql_endpoint, current_pool, coingecko_api_key, coin_list):
    pools = get_balancer_pools(chains, graphql_endpoint)
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
        token_0_symbol = pool['poolTokens'][0]["symbol"].lower()
        if token_0_symbol in token_id_cache:
            token_0_id = token_id_cache[token_0_symbol]
        else:
            token_0_id = get_token_id_from_symbol(pool['poolTokens'][0]["address"], token_0_symbol, coin_list, pool['chain'])
            if token_0_id:
                token_id_cache[token_0_symbol] = token_0_id

        # Check if token1 ID is already cached
        token_1_symbol = pool['poolTokens'][1]["symbol"].lower()
        if token_1_symbol in token_id_cache:
            token_1_id = token_id_cache[token_1_symbol]
        else:
            token_1_id = get_token_id_from_symbol(pool['poolTokens'][1]["address"], token_1_symbol, coin_list, pool['chain'])
            if token_1_id:
                token_id_cache[token_1_symbol] = token_1_id

        if token_0_id and token_1_id:
            pool['il_risk_score'] = calculate_il_risk_score(token_0_id, token_1_id, coingecko_api_key)
        else:
            pool['il_risk_score'] = float('nan')

    formatted_pools = [format_pool_data(pool) for pool in filtered_pools]
    return formatted_pools

def calculate_metrics(current_pool: Dict[str, Any], coingecko_api_key: str, coin_list: List[Any], **kwargs) -> Optional[Dict[str, Any]]:
    token_0_id = get_token_id_from_symbol(current_pool['token0'], current_pool['token0_symbol'], coin_list, current_pool['chain'])
    token_1_id = get_token_id_from_symbol(current_pool['token1'], current_pool['token1_symbol'], coin_list, current_pool['chain'])

    if token_0_id and token_1_id:
        il_risk_score = calculate_il_risk_score(token_0_id, token_1_id, coingecko_api_key)
    else:
        il_risk_score = float('nan')
        
    return {
        "il_risk_score": il_risk_score
    }

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
        result = get_opportunities(**kwargs)
        if not result:
            return {"error": "No suitable aggregators found"}
        
        return result
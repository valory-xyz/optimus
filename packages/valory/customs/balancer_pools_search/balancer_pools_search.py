import warnings

warnings.filterwarnings("ignore")  # Suppress all warnings

import requests
from typing import Dict, Union, Any, List, Optional,Tuple
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
import pandas as pd
import pyfolio as pf
import statistics
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from functools import lru_cache
import logging
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Constants and mappings
SUPPORTED_POOL_TYPES = {
    "WEIGHTED": "Weighted",
    "COMPOSABLE_STABLE": "ComposableStable",
    "LIQUIDITY_BOOTSTRAPING": "LiquidityBootstrapping",
    "META_STABLE": "MetaStable",
    "STABLE": "Stable",
    "INVESTMENT": "Investment"
}
REQUIRED_FIELDS = ("chains", "graphql_endpoint", "current_positions", "coingecko_api_key")
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

EXCLUDED_APR_TYPES = {"IB_YIELD", "MERKL", "SWAP_FEE", "SWAP_FEE_7D", "SWAP_FEE_30D"}
LP = "lp"
errors = []

@lru_cache(None)
def fetch_coin_list():
    """Fetches the list of coins from CoinGecko API only once."""
    url = "https://api.coingecko.com/api/v3/coins/list"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        errors.append((f"Failed to fetch coin list: {e}"))
        return None

def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    return [field for field in REQUIRED_FIELDS if kwargs.get(field) is None]

def remove_irrelevant_fields(kwargs: Dict[str, Any], required_fields: Tuple) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key in required_fields}

def run_query(query, graphql_endpoint, variables=None) -> Dict[str, Any]:
    headers = {'Content-Type': 'application/json'}
    payload = {'query': query, 'variables': variables or {}}
    response = requests.post(graphql_endpoint, json=payload, headers=headers)
    if response.status_code != 200:
        return {"error": f"GraphQL query failed with status code {response.status_code}"}
    result = response.json()
    if 'errors' in result:
        return {"error": f"GraphQL Errors: {result['errors']}"}
    return result['data']

def get_total_apr(pool) -> float:
    apr_items = pool.get('dynamicData', {}).get('aprItems', [])
    return sum(item['apr'] for item in apr_items if item['type'] not in EXCLUDED_APR_TYPES)

@lru_cache(None)
def create_web3_connection(chain_name: str):
    chain_url = CHAIN_URLS.get(chain_name)
    if not chain_url:
        return None
    web3 = Web3(Web3.HTTPProvider(chain_url))
    return web3 if web3.is_connected() else None

@lru_cache(None)
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
    web3 = create_web3_connection(chain_name)
    if not web3:
        return None
    contract = web3.eth.contract(address=Web3.to_checksum_address(token_address), abi=ERC20_ABI)
    try:
        return contract.functions.name().call()
    except:
        return None

def get_balancer_pools(chains, graphql_endpoint) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    chain_list_str = ', '.join(chain.upper() for chain in chains)
    graphql_query = f"""
    {{
      poolGetPools(where: {{chainIn: [{chain_list_str}]}} first: 100) {{
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
    data = run_query(graphql_query, graphql_endpoint)
    if "error" in data:
        return data
    return data.get("poolGetPools", [])

def get_filtered_pools(pools, current_positions):
    # Filter by type and exclude current positions
    qualifying_pools = []
    for pool in pools:
        mapped_type = SUPPORTED_POOL_TYPES.get(pool.get('type'))
        if mapped_type and pool.get('address') not in current_positions:
            pool['type'] = mapped_type
            pool['apr'] = get_total_apr(pool)
            pool['tvl'] = pool.get('dynamicData', {}).get('totalLiquidity', 0)
            qualifying_pools.append(pool)

    if len(qualifying_pools) <= 5:
        return qualifying_pools

    tvl_list = [float(p.get("tvl", 0)) for p in qualifying_pools]
    apr_list = [float(p.get("apr", 0)) for p in qualifying_pools]

    tvl_threshold = np.percentile(tvl_list, TVL_PERCENTILE)
    apr_threshold = np.percentile(apr_list, APR_PERCENTILE)
    max_tvl = max(tvl_list) if tvl_list else 1
    max_apr = max(apr_list) if apr_list else 1

    # Score and filter
    scored_pools = []
    for p in qualifying_pools:
        tvl = float(p["tvl"])
        apr = float(p["apr"])
        if tvl >= tvl_threshold and apr >= apr_threshold:
            score = TVL_WEIGHT * (tvl / max_tvl) + APR_WEIGHT * (apr / max_apr)
            p["score"] = score
            scored_pools.append(p)

    if not scored_pools:
        return []

    score_threshold = np.percentile([p["score"] for p in scored_pools], SCORE_PERCENTILE)
    filtered_scored_pools = [p for p in scored_pools if p["score"] >= score_threshold]
    filtered_scored_pools.sort(key=lambda x: x["score"], reverse=True)

    return filtered_scored_pools[:10]


def get_token_id_from_symbol_cached(symbol, token_name, coin_list):
    # Try to find a coin matching symbol first.
    candidates = [coin for coin in coin_list if coin['symbol'].lower() == symbol.lower()]
    if not candidates:
        return None

    # If single candidate, return it
    if len(candidates) == 1:
        return candidates[0]['id']

    # If multiple candidates, match by name if possible
    normalized_token_name = token_name.replace(" ", "").lower()
    for coin in candidates:
        coin_name = coin['name'].replace(" ", "").lower()
        if coin_name == normalized_token_name or coin_name == symbol.lower():
            return coin['id']
    return None

def get_token_id_from_symbol(token_address, symbol, coin_list, chain_name):
    token_name = fetch_token_name_from_contract(chain_name, token_address)
    if not token_name:
        matching_coins = [coin for coin in coin_list if coin['symbol'].lower() == symbol.lower()]
        return matching_coins[0]['id'] if len(matching_coins) == 1 else None

    return get_token_id_from_symbol_cached(symbol, token_name, coin_list)

def calculate_il_impact(P0, P1):
    # Impermanent Loss impact calculation
    return 2 * np.sqrt(P1 / P0) / (1 + P1 / P0) - 1


def calculate_il_risk_score(
    token_0, token_1, coingecko_api_key: str, time_period: int = 90
) -> float:
    is_pro = is_pro_api_key(coingecko_api_key)
    if is_pro:
        cg = CoinGeckoAPI(api_key=coingecko_api_key)
    else:
        cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=time_period)).timestamp())
    
    try:
        prices_1 = cg.get_coin_market_chart_range_by_id(
            id=token_0,
            vs_currency="usd",
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )
        prices_2 = cg.get_coin_market_chart_range_by_id(
            id=token_1,
            vs_currency="usd",
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )
    except Exception as e:
        errors.append(f"Error fetching price data: Incorrect Coingecko API Key")
        return None
    
    prices_1_data = np.array([x[1] for x in prices_1["prices"]])
    prices_2_data = np.array([x[1] for x in prices_2["prices"]])
    min_length = min(len(prices_1_data), len(prices_2_data))
    prices_1_data = prices_1_data[:min_length]
    prices_2_data = prices_2_data[:min_length]

    if min_length < 2:
        return None

    price_correlation = np.corrcoef(prices_1_data, prices_2_data)[0, 1]
    volatility_1 = np.std(prices_1_data)
    volatility_2 = np.std(prices_2_data)
    volatility_multiplier = np.sqrt(volatility_1 * volatility_2)
    P0 = prices_1_data[0] / prices_2_data[0]
    P1 = prices_1_data[-1] / prices_2_data[-1]
    il_impact = calculate_il_impact(P0, P1)

    return float(il_impact * abs(price_correlation) * volatility_multiplier)


def create_graphql_client(api_url='https://api-v3.balancer.fi') -> Client:
    transport = RequestsHTTPTransport(url=api_url, verify=True, retries=3)
    return Client(transport=transport, fetch_schema_from_transport=False)

def create_pool_snapshots_query(pool_id: str, chain: str, range: str = 'NINETY_DAYS') -> gql:
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

def fetch_liquidity_metrics(pool_id: str, chain: str, client: Optional[Client] = None, price_impact: float = 0.01) -> Optional[Dict[str, Any]]:
    if client is None:
        client = create_graphql_client()
    try:
        query = create_pool_snapshots_query(pool_id, chain)
        response = client.execute(query)
        pool_snapshots = response['poolGetSnapshots']
        if not pool_snapshots:
            return None

        avg_tvl = statistics.mean(float(s['totalLiquidity']) for s in pool_snapshots)
        avg_volume = statistics.mean(float(s.get('volume24h', 0)) for s in pool_snapshots)

        depth_score = (np.log1p(avg_tvl) * np.log1p(avg_volume)) / (price_impact * 100) if avg_tvl and avg_volume else 0
        liquidity_risk_multiplier = max(0, 1 - (1 / depth_score)) if depth_score != 0 else 0
        max_position_size = 50 * (avg_tvl * liquidity_risk_multiplier) / 100

        return {
            'Average TVL': avg_tvl,
            'Average Daily Volume': avg_volume,
            'Depth Score': depth_score,
            'Liquidity Risk Multiplier': liquidity_risk_multiplier,
            'Maximum Position Size': max_position_size,
            'Meets Depth Score Threshold': depth_score > 50
        }

    except Exception:
        return None

def analyze_pool_liquidity(pool_id: str, chain: str, client: Optional[Client] = None, price_impact: float = 0.01):
    metrics = fetch_liquidity_metrics(pool_id, chain, client, price_impact)
    if metrics is None:
        errors.append("Could not retrieve depth score and maximum position size.")
        return float("nan"), float("nan")
    return metrics["Depth Score"], metrics["Maximum Position Size"]

def get_balancer_pool_sharpe_ratio(pool_id, chain, timerange='ONE_YEAR'):
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
        response = requests.post("https://api-v3.balancer.fi/", json={'query': query})
        data = response.json()
        
        # Add error checking for response structure
        if not data or 'data' not in data or 'poolGetSnapshots' not in data['data']:
            logger.error("Invalid response structure from API")
            return None
            
        snapshots = data['data']['poolGetSnapshots']
        if not snapshots:
            return None
        
        # Convert to DataFrame with proper error handling
        df = pd.DataFrame(snapshots)
        
        # Ensure timestamp conversion is safe
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s', errors='coerce')
        df.set_index('timestamp', inplace=True)
        
        # Convert numeric columns with error handling
        df['sharePrice'] = pd.to_numeric(df['sharePrice'], errors='coerce')
        df['fees24h'] = pd.to_numeric(df['fees24h'], errors='coerce')
        df['totalLiquidity'] = pd.to_numeric(df['totalLiquidity'], errors='coerce')
        
        # Calculate returns with proper error handling
        price_returns = df['sharePrice'].pct_change()
        fee_returns = df.apply(lambda x: float(x['fees24h']) / float(x['totalLiquidity']) 
                             if x['totalLiquidity'] and float(x['totalLiquidity']) != 0 
                             else np.nan, axis=1)
        
        # Combine returns and handle infinities/NaNs
        total_returns = (price_returns + fee_returns).dropna()
        total_returns = total_returns.replace([np.inf, -np.inf], np.nan)
        
        # Check if we have enough valid data
        if len(total_returns) < 2:
            logger.warning("Insufficient valid data points for Sharpe ratio calculation")
            return None
        
        # Calculate Sharpe ratio with annualization
        returns_mean = total_returns.mean() * 365  # Annualize mean
        returns_std = total_returns.std() * np.sqrt(365)  # Annualize volatility
        
        if returns_std == 0:
            logger.warning("Zero standard deviation in returns")
            return None
            
        sharpe_ratio = returns_mean / returns_std
        
        # Validate final result
        if not np.isfinite(sharpe_ratio):
            logger.warning("Non-finite Sharpe ratio calculated")
            return None
            
        return float(sharpe_ratio)
        
    except Exception as e:
        errors.append(f"Error calculating Sharpe ratio: {e}")
        return None

def format_pool_data(pool) -> Dict[str, Any]:
    dex_type = BALANCER
    formatted_data = {
        "dex_type": dex_type,
        "chain": pool['chain'].lower(),
        "apr": pool['apr'] * 100,
        "pool_address": pool['address'],
        "pool_id": pool['id'],
        "pool_type": pool['type'],
        "il_risk_score": pool['il_risk_score'],
        "sharpe_ratio": pool['sharpe_ratio'],
        "depth_score": pool['depth_score'],
        "max_position_size": pool['max_position_size']
    }
    
    # Add dynamic number of tokens
    for i, token in enumerate(pool['poolTokens']):
        formatted_data[f"token{i}"] = token['address']
        formatted_data[f"token{i}_symbol"] = token['symbol']
    
    return formatted_data

def get_opportunities(chains, graphql_endpoint, current_positions, coingecko_api_key, coin_list):
    pools = get_balancer_pools(chains, graphql_endpoint)
    if isinstance(pools, dict) and "error" in pools:
        return pools
    
    filtered_pools = get_filtered_pools(pools, current_positions)
    if not filtered_pools:
        return {"error": "No suitable pools found"}
    
    token_id_cache = {}
    for pool in filtered_pools:
        pool['chain'] = pool['chain'].lower()
        token_ids = []
        
        # Get token IDs for all tokens in the pool
        for token in pool['poolTokens']:
            symbol = token["symbol"].lower()
            if symbol not in token_id_cache:
                token_id = get_token_id_from_symbol(token["address"], symbol, coin_list, pool['chain'])
                if token_id:
                    token_id_cache[symbol] = token_id
            token_ids.append(token_id_cache.get(symbol))
        
        # Calculate metrics if we have all token IDs
        if all(token_ids):
            pool['il_risk_score'] = calculate_il_risk_score_multi(token_ids, coingecko_api_key)
        else:
            pool['il_risk_score'] = float('nan')
        
        pool['sharpe_ratio'] = get_balancer_pool_sharpe_ratio(pool['id'], pool['chain'].upper())
        pool['depth_score'], pool['max_position_size'] = analyze_pool_liquidity(pool['id'], pool['chain'].upper())
    
    return [format_pool_data(pool) for pool in filtered_pools]

def calculate_metrics(position: Dict[str, Any], coingecko_api_key: str, coin_list: List[Any], **kwargs) -> Optional[Dict[str, Any]]:
    # Get all token IDs from the position
    token_ids = []
    i = 0
    while True:
        token_key = f'token{i}'
        symbol_key = f'token{i}_symbol'
        if token_key not in position or symbol_key not in position:
            break
        token_id = get_token_id_from_symbol(position[token_key], position[symbol_key], coin_list, position['chain'])
        if token_id:
            token_ids.append(token_id)
        i += 1
    
    il_risk_score = calculate_il_risk_score_multi(token_ids, coingecko_api_key) if token_ids else float('nan')
    sharpe_ratio = get_balancer_pool_sharpe_ratio(position['pool_id'], position['chain'].upper())
    depth_score, max_position_size = analyze_pool_liquidity(position['pool_id'], position['chain'].upper())
    
    return {
        "il_risk_score": il_risk_score,
        "sharpe_ratio": sharpe_ratio,
        "depth_score": depth_score,
        "max_position_size": max_position_size,
    }

def is_pro_api_key(coingecko_api_key: str) -> bool:
    """
    Check if the provided CoinGecko API key is a pro key.
    """
    # Try using the key as a pro API key
    cg_pro = CoinGeckoAPI(api_key=coingecko_api_key)
    try:
        response = cg_pro.get_coin_market_chart_range_by_id(
            id="bitcoin",
            vs_currency="usd",
            from_timestamp=0,
            to_timestamp=0
        )
        if response:
            return True
    except Exception:
        return False

    return False

def run(*_args, **kwargs) -> Dict[str, Union[bool, str, List[str]]]:
    missing = check_missing_fields(kwargs)
    if missing:
        errors.append(f"Required kwargs {missing} were not provided.")
        return {"error": errors}

    required_fields = list(REQUIRED_FIELDS)
    get_metrics = kwargs.get('get_metrics', False)
    if get_metrics:
        required_fields.append('position')

    kwargs = remove_irrelevant_fields(kwargs, required_fields)
    
    coin_list = fetch_coin_list()
    if coin_list is None:
        errors.append("Failed to fetch coin list.")
        return {"error": errors}

    kwargs.update({"coin_list": coin_list})

    if get_metrics:
        metrics = calculate_metrics(**kwargs)
        if metrics is None:
            errors.append("Failed to calculate metrics.")
        return {"error": errors} if errors else metrics
    else:
        result = get_opportunities(**kwargs)
        if isinstance(result, dict) and "error" in result:
            errors.append(result["error"])
        if not result:
            errors.append("No suitable aggregators found")
        return {"error": errors} if errors else {"result": result}
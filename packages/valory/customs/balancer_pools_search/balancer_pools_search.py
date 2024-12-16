import requests
from typing import Dict, Union, Any, List
import numpy as np
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta

# Constants
UNISWAP = "UniswapV3"
REQUIRED_FIELDS = ("chains", "apr_threshold", "graphql_endpoint", "current_pool", "coingecko_api_key")
FEE_RATE_DIVISOR = 1_000_000  # Convert basis points to decimal
DAYS_IN_YEAR = 365
PERCENT_CONVERSION = 100
TVL_WEIGHT = 0.7  # Weight for TVL
APR_WEIGHT = 0.3  # Weight for APR
TVL_PERCENTILE = 50
APR_PERCENTILE = 50
SCORE_PERCENTILE = 80

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

def get_filtered_pools(pools, current_pool) -> List[Dict[str, Any]]:
    # Filter pools by those with exactly two tokens
    filtered_pools = []
    tvl_list = []
    apr_list = []

    for pool in pools:
        pool_address = pool.get('address')
        if len(pool.get('token0', [])) == 2 and pool_address != current_pool:
            tvl = float(pool.get("totalValueLockedUSD", 0))
            daily_volume = float(pool.get("volumeUSD", 0))
            fee_rate = float(pool.get("feeTier", 0)) / FEE_RATE_DIVISOR
            apr = calculate_apr(daily_volume, tvl, fee_rate)
            
            if tvl == 0 or apr == 0:
                continue

            tvl_list.append(tvl)
            apr_list.append(apr)
            
            pool["tvl"] = tvl
            pool["apr"] = apr * PERCENT_CONVERSION
            filtered_pools.append(pool)

    if filtered_pools:
        tvl_list = [float(pool.get("tvl", 0)) for pool in filtered_pools]
        apr_list = [float(pool.get("apr", 0)) for pool in filtered_pools]
        
        tvl_threshold = np.percentile(tvl_list, TVL_PERCENTILE)
        apr_threshold = np.percentile(apr_list, APR_PERCENTILE)

        # Prioritize pools using combined TVL and APR score
        scored_pools = []
        max_tvl = max(tvl_list)
        max_apr = max(apr_list)

        for pool in filtered_pools:
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

        top_pools = filtered_scored_pools
    else:
        top_pools = []

    return top_pools

def calculate_apr(daily_volume: float, tvl: float, fee_rate: float) -> float:
    """Calculate APR using the formula: (Daily Volume / TVL) × Fee Rate × 365 × 100"""
    if tvl == 0:
        return 0
    return (daily_volume / tvl) * fee_rate * DAYS_IN_YEAR * PERCENT_CONVERSION

def calculate_il_risk_score(pool, coingecko_api_key: str) -> float:
    """Calculate the IL Risk Score for a given pool."""
    cg = CoinGeckoAPI(api_key=coingecko_api_key)
    
    token_1 = pool['token0']['id']
    token_2 = pool['token1']['id']
    
    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
    
    # Fetch historical price data for the token pair
    prices_1 = cg.get_coin_market_chart_range_by_id(id=token_1, vs_currency='usd', from_timestamp=from_timestamp, to_timestamp=to_timestamp)
    prices_2 = cg.get_coin_market_chart_range_by_id(id=token_2, vs_currency='usd', from_timestamp=from_timestamp, to_timestamp=to_timestamp)
    
    # Extract price data
    prices_1_data = np.array([x[1] for x in prices_1['prices']])
    prices_2_data = np.array([x[1] for x in prices_2['prices']])
    
    # Price Correlation Calculation
    price_correlation = np.corrcoef(prices_1_data, prices_2_data)[0, 1]
    
    # Volatility Calculation
    volatility_1 = np.std(prices_1_data)
    volatility_2 = np.std(prices_2_data)
    volatility_multiplier = np.sqrt(volatility_1 * volatility_2)
    
    # Calculate IL Impact
    P0 = prices_1_data[0] / prices_2_data[0]
    P1 = prices_1_data[-1] / prices_2_data[-1]
    il_impact = 2 * np.sqrt(P1 / P0) / (1 + P1 / P0) - 1
    
    # Calculate IL Risk Score
    il_risk_score = il_impact * price_correlation * volatility_multiplier
    
    return il_risk_score

def fetch_graphql_data(chains, graphql_endpoint) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Fetch data from GraphQL endpoint for the specified chains."""
    # Ensure all chain names are uppercase
    chain_names = [chain.upper() for chain in chains]
    chain_list_str = ', '.join(chain_names)
    
    # Build the GraphQL query with the specified chain names
    graphql_query = f"""
    {{
      pools(where: {{chainIn: [{chain_list_str}]}}) {{
        id
        address
        chain
        feeTier
        liquidity
        volumeUSD
        totalValueLockedUSD
        token0 {{
          id
          symbol
        }}
        token1 {{
          id
          symbol
        }}
      }}
    }}
    """

    # Execute the GraphQL query
    data = run_query(graphql_query, graphql_endpoint)
    if "error" in data:
        return data
    
    # Extract pools from the response
    return data.get("pools", [])

def format_pool_data(pool) -> Dict[str, Any]:
    """Format pool data into the desired structure."""
    dex_type = UNISWAP
    chain = pool['chain'].lower()
    apr = pool['apr']
    pool_address = pool['address']
    pool_id = pool['id']
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
        "pool_id": pool_id,
        "token0": token0,
        "token1": token1,
        "token0_symbol": token0_symbol,
        "token1_symbol": token1_symbol,
        "il_risk_score": il_risk_score
    }

def get_best_pools(chains, apr_threshold, graphql_endpoint, current_pool, coingecko_api_key) -> List[Dict[str, Any]]:
    pools = fetch_graphql_data(chains, graphql_endpoint)
    if isinstance(pools, dict) and "error" in pools:
        return pools
    
    top_pools = get_filtered_pools(pools, current_pool)

    if not top_pools:
        return {"error": "No suitable pools found"}
    
    # Calculate IL Risk Score for each pool
    for pool in top_pools:
        pool['il_risk_score'] = calculate_il_risk_score(pool, coingecko_api_key)

    formatted_pools = [format_pool_data(pool) for pool in top_pools]

    return formatted_pools

def run(*_args, **kwargs) -> Dict[str, Union[bool, str]]:
    """Run the strategy."""
    missing = check_missing_fields(kwargs)
    if len(missing) > 0:
        return {"error": f"Required kwargs {missing} were not provided."}

    kwargs = remove_irrelevant_fields(kwargs)
    return get_best_pools(**kwargs)
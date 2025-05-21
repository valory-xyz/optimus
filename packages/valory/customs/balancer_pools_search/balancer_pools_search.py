import warnings

warnings.filterwarnings("ignore")  # Suppress all warnings

import statistics
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import numpy as np
import pandas as pd
import pyfolio as pf
import requests
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from pycoingecko import CoinGeckoAPI
from web3 import Web3
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and mappings
SUPPORTED_POOL_TYPES = {
    "WEIGHTED": "Weighted",
    "COMPOSABLE_STABLE": "ComposableStable",
    "LIQUIDITY_BOOTSTRAPING": "LiquidityBootstrapping",
    "META_STABLE": "MetaStable",
    "STABLE": "Stable",
    "INVESTMENT": "Investment",
}
REQUIRED_FIELDS = (
    "chains",
    "graphql_endpoint",
    "current_positions",
    "coingecko_api_key",
)
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
    "base": "https://1rpc.io/base",
}

EXCLUDED_APR_TYPES = {"IB_YIELD", "MERKL", "SWAP_FEE", "SWAP_FEE_7D", "SWAP_FEE_30D"}
LP = "lp"
errors = []
WHITELISTED_ASSETS = {
    "mode": [],
    "optimism": [
        "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",  # USDC
        "0xCB8FA9a76b8e203D8C3797bF438d8FB81Ea3326A",  # alUSD
        "0x01bFF41798a0BcF287b996046Ca68b395DbC1071",  # USDT0
        "0x94b008aA00579c1307B0EF2c499aD98a8ce58e58",  # USDT
        "0x9dAbAE7274D28A45F0B65Bf8ED201A5731492ca0",  # msUSD
        "0x7F5c764cBc14f9669B88837ca1490cCa17c31607",  # USDC.e
        "0xbfD291DA8A403DAAF7e5E9DC1ec0aCEaCd4848B9",  # USX
        "0x8aE125E8653821E851F12A49F7765db9a9ce7384",  # DOLA
        "0xc40F949F8a4e094D1b49a23ea9241D289B7b2819",  # LUSD
        "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1",  # DAI
        "0x087C440F251Ff6Cfe62B86DdE1bE558B95b4bb9b",  # BOLD
        "0x2E3D870790dC77A83DD1d18184Acc7439A53f475",  # FRAX
        "0x2218a117083f5B482B0bB821d27056Ba9c04b1D3",  # sDAI
        "0x73cb180bf0521828d8849bc8CF2B920918e23032",  # USD+
        "0x1217BfE6c773EEC6cc4A38b5Dc45B92292B6E189",  # oUSDT
        "0x4F604735c1cF31399C6E711D5962b2B3E0225AD3"   # USDGLO
    ],
    "base": []
}

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

def remove_irrelevant_fields(
    kwargs: Dict[str, Any], required_fields: Tuple
) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key in required_fields}

def run_query(query, graphql_endpoint, variables=None) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    payload = {"query": query, "variables": variables or {}}
    response = requests.post(graphql_endpoint, json=payload, headers=headers)
    if response.status_code != 200:
        return {
            "error": f"GraphQL query failed with status code {response.status_code}"
        }
    result = response.json()
    if "errors" in result:
        return {"error": f"GraphQL Errors: {result['errors']}"}
    return result["data"]

def get_total_apr(pool) -> float:
    apr_items = pool.get("dynamicData", {}).get("aprItems", [])
    return sum(
        item["apr"] for item in apr_items if item["type"] not in EXCLUDED_APR_TYPES
    )

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
    contract = web3.eth.contract(
        address=Web3.to_checksum_address(token_address), abi=ERC20_ABI
    )
    try:
        return contract.functions.name().call()
    except:
        return None

def get_balancer_pools(
    chains, graphql_endpoint
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    chain_list_str = ", ".join(chain.upper() for chain in chains)
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
    # Filter by type and token count - removing the 2-token restriction
    qualifying_pools = []
    for pool in pools:
        token_count = len(pool.get("poolTokens", []))
        # Accept pools with 3-8 tokens (we're removing the 2-token restriction)
        mapped_type = SUPPORTED_POOL_TYPES.get(pool.get("type"))
        chain = pool["chain"].lower()
        whitelisted_tokens = WHITELISTED_ASSETS.get(chain, [])
        if (
            mapped_type
            and 2 <= token_count 
            and Web3.to_checksum_address(pool.get("address")) not in current_positions
            and chain in WHITELISTED_ASSETS
            and (
                not whitelisted_tokens
                or (
                    pool["poolTokens"][0]["address"] in whitelisted_tokens
                    and pool["poolTokens"][1]["address"] in whitelisted_tokens
                )
            )
        ):
            pool["type"] = mapped_type
            pool["apr"] = get_total_apr(pool)
            pool["tvl"] = pool.get("dynamicData", {}).get("totalLiquidity", 0)
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

    score_threshold = np.percentile(
        [p["score"] for p in scored_pools], SCORE_PERCENTILE
    )
    filtered_scored_pools = [p for p in scored_pools if p["score"] >= score_threshold]
    filtered_scored_pools.sort(key=lambda x: x["score"], reverse=True)

    return filtered_scored_pools[:10]

def get_token_id_from_symbol_cached(symbol, token_name, coin_list):
    # Try to find a coin matching symbol first.
    candidates = [
        coin for coin in coin_list if coin["symbol"].lower() == symbol.lower()
    ]
    if not candidates:
        return None

    # If single candidate, return it
    if len(candidates) == 1:
        return candidates[0]["id"]

    # If multiple candidates, match by name if possible
    normalized_token_name = token_name.replace(" ", "").lower()
    for coin in candidates:
        coin_name = coin["name"].replace(" ", "").lower()
        if coin_name == normalized_token_name or coin_name == symbol.lower():
            return coin["id"]
    return None

def get_token_id_from_symbol(token_address, symbol, coin_list, chain_name):
    token_name = fetch_token_name_from_contract(chain_name, token_address)
    if not token_name:
        matching_coins = [
            coin for coin in coin_list if coin["symbol"].lower() == symbol.lower()
        ]
        return matching_coins[0]["id"] if len(matching_coins) == 1 else None

    return get_token_id_from_symbol_cached(symbol, token_name, coin_list)

def calculate_il_impact_multi(initial_prices, final_prices, weights=None):
    """
    Calculate impermanent loss impact for multiple tokens.
    
    Args:
        initial_prices: List of initial token prices
        final_prices: List of final token prices
        weights: List of token weights in the pool (defaults to equal weights)
        
    Returns:
        float: Impermanent loss impact
    """
    if not initial_prices or not final_prices:
        return 0
        
    n = len(initial_prices)
    if n != len(final_prices):
        return 0
        
    # Default to equal weights if not provided
    if not weights:
        weights = [1/n] * n
    elif len(weights) != n:
        return 0
        
    # Normalize weights to sum to 1
    total_weight = sum(weights)
    weights = [w/total_weight for w in weights]
    
    # Calculate price ratios
    price_ratios = [final_prices[i]/initial_prices[i] for i in range(n)]
    
    # Calculate hodl value
    hodl_value = sum(weights[i] * price_ratios[i] for i in range(n))
    
    # Calculate pool value
    geometric_mean = 1
    for i in range(n):
        geometric_mean *= price_ratios[i] ** weights[i]
    
    # Calculate impermanent loss
    il = geometric_mean / hodl_value - 1
    
    return il

def calculate_il_risk_score_multi(token_ids, coingecko_api_key: str, time_period: int = 90) -> float:
    """Calculate IL risk score for multiple tokens."""
    is_pro = is_pro_api_key(coingecko_api_key)
    if is_pro:
        cg = CoinGeckoAPI(api_key=coingecko_api_key)
    else:
        cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
    
    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=time_period)).timestamp())
    
    # Filter out None token_ids
    valid_token_ids = [tid for tid in token_ids if tid]
    if len(valid_token_ids) < 2:
        return None
    
    try:
        # Get price data for all tokens
        prices_data = []
        for token_id in valid_token_ids:
            try:
                prices = cg.get_coin_market_chart_range_by_id(
                    id=token_id,
                    vs_currency="usd",
                    from_timestamp=from_timestamp,
                    to_timestamp=to_timestamp,
                )
                prices_data.append([x[1] for x in prices["prices"]])
                time.sleep(1)  # Rate limiting
            except Exception as e:
                errors.append(f"Error fetching price data for {token_id}: {str(e)}")
                return None
        
        # Find minimum length to align all price series
        min_length = min(len(prices) for prices in prices_data)
        if min_length < 2:
            return None
            
        # Truncate all price series to the same length
        aligned_prices = [prices[:min_length] for prices in prices_data]
        
        # Calculate correlation matrix
        price_matrix = np.array(aligned_prices)
        correlation_matrix = np.corrcoef(price_matrix)
        
        # Calculate average correlation (excluding self-correlations)
        n = len(valid_token_ids)
        avg_correlation = 0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                avg_correlation += abs(correlation_matrix[i, j])
                count += 1
        
        avg_correlation = avg_correlation / count if count > 0 else 0
        
        # Calculate volatility for each token
        volatilities = [np.std(prices) for prices in aligned_prices]
        avg_volatility = np.mean(volatilities)
        
        # Calculate IL impact
        initial_prices = [prices[0] for prices in aligned_prices]
        final_prices = [prices[-1] for prices in aligned_prices]
        il_impact = calculate_il_impact_multi(initial_prices, final_prices)
        
        return float(il_impact * avg_correlation * avg_volatility)
        
    except Exception as e:
        errors.append(f"Error calculating IL risk score: {str(e)}")
        return None

def create_graphql_client(api_url="https://api-v3.balancer.fi") -> Client:
    transport = RequestsHTTPTransport(url=api_url, verify=True, retries=3)
    return Client(transport=transport, fetch_schema_from_transport=False)

def create_pool_snapshots_query(
    pool_id: str, chain: str, range: str = "NINETY_DAYS"
) -> gql:
    return gql(
        f"""
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
    """
    )

def fetch_liquidity_metrics(
    pool_id: str,
    chain: str,
    client: Optional[Client] = None,
    price_impact: float = 0.01,
) -> Optional[Dict[str, Any]]:
    if client is None:
        client = create_graphql_client()
    try:
        query = create_pool_snapshots_query(pool_id, chain)
        response = client.execute(query)
        pool_snapshots = response["poolGetSnapshots"]
        if not pool_snapshots:
            return None

        avg_tvl = statistics.mean(float(s["totalLiquidity"]) for s in pool_snapshots)
        avg_volume = statistics.mean(
            float(s.get("volume24h", 0)) for s in pool_snapshots
        )

        depth_score = (
            (np.log1p(avg_tvl) * np.log1p(avg_volume)) / (price_impact * 100)
            if avg_tvl and avg_volume
            else 0
        )
        liquidity_risk_multiplier = (
            max(0, 1 - (1 / depth_score)) if depth_score != 0 else 0
        )
        max_position_size = 50 * (avg_tvl * liquidity_risk_multiplier) / 100

        return {
            "Average TVL": avg_tvl,
            "Average Daily Volume": avg_volume,
            "Depth Score": depth_score,
            "Liquidity Risk Multiplier": liquidity_risk_multiplier,
            "Maximum Position Size": max_position_size,
            "Meets Depth Score Threshold": depth_score > 50,
        }

    except Exception:
        return None

def analyze_pool_liquidity(
    pool_id: str,
    chain: str,
    client: Optional[Client] = None,
    price_impact: float = 0.01,
):
    try:
        metrics = fetch_liquidity_metrics(pool_id, chain, client, price_impact)
        if metrics is None:
            errors.append(f"Could not retrieve depth score for pool {pool_id}")
            return float("nan"), float("nan")
        return metrics["Depth Score"], metrics["Maximum Position Size"]
    except Exception as e:
        errors.append(f"Error analyzing pool liquidity for {pool_id}: {str(e)}")
        return float("nan"), float("nan")

def fetch_liquidity_metrics(
    pool_id: str,
    chain: str,
    client: Optional[Client] = None,
    price_impact: float = 0.01,
) -> Optional[Dict[str, Any]]:
    if client is None:
        client = create_graphql_client()
    try:
        query = create_pool_snapshots_query(pool_id, chain)
        response = client.execute(query)
        pool_snapshots = response.get("poolGetSnapshots", [])
        if not pool_snapshots:
            return None

        # Filter out potentially corrupted data
        filtered_snapshots = []
        for snapshot in pool_snapshots:
            try:
                liquidity_value = float(snapshot.get("totalLiquidity", 0))
                volume_value = float(snapshot.get("volume24h", 0))
                
                # Skip extreme values that could cause numerical issues
                if liquidity_value > 1e16 or volume_value > 1e16:
                    continue
                    
                filtered_snapshots.append(snapshot)
            except (ValueError, TypeError):
                continue
                
        if not filtered_snapshots:
            return None
                
        # Calculate metrics on filtered data
        try:
            avg_tvl = statistics.mean(float(s.get("totalLiquidity", 0)) for s in filtered_snapshots)
            avg_volume = statistics.mean(float(s.get("volume24h", 0)) for s in filtered_snapshots)
        except statistics.StatisticsError:
            return None

        # Handle potential division by zero or very small values
        if price_impact <= 0.001:  # Avoid division by extremely small values
            price_impact = 0.001
            
        # Calculate depth score with safeguards
        try:
            depth_score = (
                (np.log1p(max(0, avg_tvl)) * np.log1p(max(0, avg_volume))) / (price_impact * 100)
                if avg_tvl > 0 and avg_volume > 0
                else 0
            )
            
            # Cap depth score at reasonable values
            depth_score = min(depth_score, 1e6)
            
            liquidity_risk_multiplier = (
                max(0, 1 - (1 / max(1, depth_score)))
                if depth_score > 0
                else 0
            )
            
            max_position_size = min(
                50 * (avg_tvl * liquidity_risk_multiplier) / 100,
                avg_tvl * 0.1  # Cap at 10% of TVL as a safety measure
            )
            
            # Cap max position size to reasonable values
            max_position_size = min(max_position_size, 1e7)
            
        except (ZeroDivisionError, OverflowError, ValueError) as e:
            errors.append(f"Error calculating metrics for pool {pool_id}: {str(e)}")
            return None

        return {
            "Average TVL": avg_tvl,
            "Average Daily Volume": avg_volume,
            "Depth Score": depth_score,
            "Liquidity Risk Multiplier": liquidity_risk_multiplier,
            "Maximum Position Size": max_position_size,
            "Meets Depth Score Threshold": depth_score > 50,
        }

    except Exception as e:
        errors.append(f"Error fetching liquidity metrics for pool {pool_id}: {str(e)}")
        return None

def get_balancer_pool_sharpe_ratio(pool_id, chain, timerange="ONE_YEAR"):
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
    """ % (
        chain,
        pool_id,
        timerange,
    )
    try:
        response = requests.post("https://api-v3.balancer.fi/", json={"query": query})
        data = response.json().get("data", {}).get("poolGetSnapshots", [])
        if not data:
            return None

        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Handle extremely large values that could cause overflow
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("timestamp", inplace=True)

            # Convert to numeric with error handling
            df["sharePrice"] = pd.to_numeric(df["sharePrice"], errors='coerce')
            df["fees24h"] = pd.to_numeric(df["fees24h"], errors='coerce')
            df["totalLiquidity"] = pd.to_numeric(df["totalLiquidity"], errors='coerce')
            
            # Filter out extreme values that could cause numerical issues
            for col in ["sharePrice", "fees24h", "totalLiquidity"]:
                if col in df.columns:
                    # Replace extreme values with NaN
                    df[col] = df[col].mask(df[col] > 1e16, np.nan)
                    df[col] = df[col].mask(df[col] < -1e16, np.nan)
            
            # Calculate returns
            price_returns = df["sharePrice"].pct_change()
            
            # Calculate fee returns, handling potential division by zero
            fee_returns = pd.Series(0, index=df.index)  # Default to zero
            mask = (df["totalLiquidity"] > 0) & df["fees24h"].notnull()
            fee_returns[mask] = df.loc[mask, "fees24h"] / df.loc[mask, "totalLiquidity"]
            
            # Combine returns
            total_returns = (price_returns + fee_returns).dropna()
            
            # Replace infinity and extreme values
            total_returns = total_returns.replace([np.inf, -np.inf], np.nan)
            total_returns = total_returns.mask(total_returns > 1.0, np.nan)  # Cap at 100% daily return
            total_returns = total_returns.mask(total_returns < -1.0, np.nan)  # Cap at -100% daily return
            
            # Remove NaN values
            total_returns = total_returns.dropna()
            
            # Check if we have enough data
            if len(total_returns) < 5:  # Need at least a few data points
                return None
                
            # Calculate Sharpe ratio with error handling
            sharpe_ratio = pf.timeseries.sharpe_ratio(total_returns)
            
            # Cap the Sharpe ratio to reasonable bounds
            if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                return None
                
            # Cap at reasonable values
            sharpe_ratio = max(min(sharpe_ratio, 10), -10)
            
            return sharpe_ratio
            
        except Exception as e:
            errors.append(f"Error processing Sharpe ratio data: {str(e)}")
            return None
            
    except Exception as e:
        errors.append(f"Error calculating Sharpe ratio: {str(e)}")
        return None

def get_underlying_token_symbol(symbol: str) -> str:
    """Map wrapped/synthetic tokens to their underlying asset symbols."""
    symbol = symbol.lower()
    
    # Mapping of synthetic/wrapped tokens to their underlying assets
    token_mappings = {
        'csusdc': 'usdc',
        'csusdl': 'usd',  # Assuming this is a USD-pegged token
        'waethlidowsteth': 'steth',
        'rsteth': 'steth',
        'wsteth': 'steth',
        'weth': 'ethereum',
        'ausdc': 'usdc',
        'ausdt': 'usdt',
        'adai': 'dai',
        'cdai': 'dai',
        'cusdc': 'usdc',
        'dai': 'dai'
    }
    
    return token_mappings.get(symbol, symbol)

def normalize_token_symbol(symbol: str) -> str:
    """Normalize token symbols for better matching."""
    # Remove common prefixes/suffixes
    prefixes = ['cs', 'wa', 'w', 'a', 'c', 'v', 'x']
    for prefix in prefixes:
        if symbol.lower().startswith(prefix):
            return symbol[len(prefix):]
    return symbol

def get_pool_token_prices(token_symbols: List[str], coingecko_api: CoinGeckoAPI) -> Dict[str, float]:
    """Enhanced token price fetching with support for synthetic tokens."""
    prices = {}
    
    try:
        # Add initial delay
        time.sleep(3)
        
        for original_symbol in token_symbols:
            try:
                symbol = original_symbol.lower()
                underlying_symbol = get_underlying_token_symbol(symbol)
                normalized_symbol = normalize_token_symbol(symbol)
                
                # List of possible IDs to try
                possible_ids = [
                    symbol,
                    underlying_symbol,
                    normalized_symbol,
                    f"{underlying_symbol}-token",
                    f"{normalized_symbol}-token"
                ]
                
                price_found = False
                
                # Try each possible ID
                for token_id in possible_ids:
                    if price_found:
                        break
                        
                    time.sleep(2)  # Rate limiting
                    try:
                        response = coingecko_api.get_price(
                            ids=token_id,
                            vs_currencies='usd',
                            timeout=30
                        )
                        if response and token_id in response:
                            prices[original_symbol] = response[token_id]['usd']
                            price_found = True
                            break
                    except Exception:
                        continue
                
                # If still no price, try search
                if not price_found:
                    time.sleep(2)
                    try:
                        search_result = coingecko_api.search(underlying_symbol)
                        if search_result and 'coins' in search_result and search_result['coins']:
                            coin_id = search_result['coins'][0]['id']
                            time.sleep(2)
                            price_data = coingecko_api.get_price(ids=coin_id, vs_currencies='usd')
                            if price_data and coin_id in price_data:
                                prices[original_symbol] = price_data[coin_id]['usd']
                                price_found = True
                    except Exception:
                        pass
                
                # Special handling for USD-pegged tokens
                if not price_found and any(x in symbol.lower() for x in ['usd', 'dai', 'usdt', 'usdc']):
                    prices[original_symbol] = 1.0
                    price_found = True
                
                # Log warning if still no price found
                if not price_found:
                    prices[original_symbol] = 0.0
                    
            except Exception as e:
                errors.append(f"Error fetching price for {original_symbol}: {str(e)}")
                prices[original_symbol] = 0.0
                
        return prices
    except Exception as e:
        errors.append(f"Error in price fetching: {str(e)}")
        return {symbol: 0.0 for symbol in token_symbols}

def get_token_investments_multi(diff_investment: float, token_prices: Dict[str, float]) -> List[float]:
    """
    Calculate how many tokens should be invested for multiple tokens based on USD investment amount.
    Only invests in the first two tokens (indices 0 and 1) with equal distribution.
    Sets all other token amounts to zero.
    
    Args:
        diff_investment: Total USD amount to invest in the pool
        token_prices: Dictionary mapping token symbols to their USD prices
        
    Returns:
        List[float]: List of token amounts to invest. Returns empty list if investment
                     is not possible.
    """
    # Validate inputs
    if diff_investment <= 0:
        return []
        
    if not token_prices:
        return []
    
    # Extract the first two tokens' prices
    token_symbols = list(token_prices.keys())
    
    if len(token_symbols) < 2:
        return []
    
    # Get first two tokens' symbols
    first_token_symbol = token_symbols[0]
    second_token_symbol = token_symbols[1]
    
    # Get prices
    first_token_price = token_prices[first_token_symbol]
    second_token_price = token_prices[second_token_symbol]
    
    # If both prices are invalid, we can't proceed
    if first_token_price <= 0 and second_token_price <= 0:
        return []
        
    # Calculate per-token investment for valid tokens only
    valid_token_count = 0
    per_token_investment = 0
    
    if first_token_price > 0:
        valid_token_count += 1
    
    if second_token_price > 0:
        valid_token_count += 1
        
    if valid_token_count > 0:
        per_token_investment = diff_investment / valid_token_count
    else:
        return []
    
    # Initialize all amounts to 0
    token_amounts = [0.0] * len(token_prices)
    
    # Calculate token amounts for first two tokens
    if first_token_price > 0:
        token_amounts[0] = round(per_token_investment / first_token_price, 8)
    
    if second_token_price > 0:
        token_amounts[1] = round(per_token_investment / second_token_price, 8)
    
    return token_amounts

def calculate_single_pool_investment(apr: float, tvl: float) -> float:
    """
    Calculate investment amount for a single pool based on APR thresholds.
    Enforces a maximum investment amount of $1000.
    
    Args:
        apr: Annual Percentage Rate
        tvl: Total Value Locked
        
    Returns:
        float: Calculated investment amount, capped at $1000
    """
    MIN_APR_THRESHOLD = 0.02  # 2% minimum APR
    MAX_TVL_ALLOCATION = 0.20  # 20% maximum TVL allocation
    
    if apr < MIN_APR_THRESHOLD:
        return 0.0
        
    # Calculate investment based on APR premium
    apr_premium = apr / MIN_APR_THRESHOLD
    base_investment = tvl * (apr_premium - 1) * 0.1  # 10% of the APR premium
    
    # Cap at maximum TVL allocation
    max_investment = min(tvl * MAX_TVL_ALLOCATION, 1000.0)
    investment = min(base_investment, max_investment)
    
    # Apply minimum investment threshold
    return investment if investment >= 100 else 0.0

def calculate_differential_investment(apr_current: float, apr_base: float, tvl: float, is_single_pool: bool = False) -> float:
    """
    Calculate the differential investment amount based on APR differences or single pool metrics.
    Enforces a maximum investment amount of $1000.
    When apr_base is zero, uses a fixed reduction (25%) of the current APR for calculation.
    
    Args:
        apr_current: APR of the current pool
        apr_base: APR of the second-best pool (or 0 for single pool)
        tvl: Total Value Locked in the current pool
        is_single_pool: Flag indicating if this is a single pool calculation
        
    Returns:
        float: Calculated investment amount, capped at $1000
    """
    try:
        # Handle invalid inputs
        if tvl <= 0:
            return 0.0
            
        # Handle single pool case
        if is_single_pool:
            return min(calculate_single_pool_investment(apr_current, tvl), 1000.0)
            
        # Handle case where current APR is too low
        if apr_current <= 0.01:
            return 0.0
            
        # Handle zero base APR case by using a fixed 25% reduction of current APR
        if apr_base <= 0:
            # Use 75% of current APR as the base APR (25% reduction)
            apr_base = apr_current * 0.75
            
        # Calculate ratio with the possibly adjusted base APR
        ratio = apr_current / apr_base
        if ratio <= 1:
            return 0.0
            
        # Calculate differential investment
        diff_investment = (ratio - 1) * tvl
        
        # Apply minimum investment threshold and maximum cap
        if diff_investment >= 100:
            return min(diff_investment, 1000.0)
        else:
            return 0.0
        
    except ZeroDivisionError:
        # If we somehow still get a zero division, use a fallback calculation
        if apr_current > 0.02:  # Only invest if APR is above 2%
            diff_investment = apr_current * tvl * 0.05  # Invest 5% of TVL multiplied by APR
            return min(diff_investment, 1000.0) if diff_investment >= 100 else 0.0
        return 0.0

def filter_valid_investment_pools(formatted_pools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filters pools to include only those where:
    1. max_investment_usd > 0
    2. BOTH token0 and token1 have non-zero investment amounts
    3. Excludes pools where either token0 or token1 is zero
    
    Args:
        formatted_pools: List of formatted pool data
        
    Returns:
        List[Dict[str, Any]]: Filtered list of pools
    """
    valid_pools = []
    
    for pool in formatted_pools:
        # Check if max_investment_usd is positive
        if pool.get("max_investment_usd", 0) <= 0:
            continue
            
        # Check if max_investment_amounts exists and has at least 2 elements
        investment_amounts = pool.get("max_investment_amounts", [])
        if len(investment_amounts) < 2:
            continue
            
        # Check if BOTH token0 and token1 have NON-ZERO investment amounts
        # This is the key change - requiring both to be positive
        if investment_amounts[0] > 0 and investment_amounts[1] > 0:
            valid_pools.append(pool)
    
    return valid_pools

def format_pool_data(pools: List[Dict[str, Any]], coingecko_api_key: str) -> List[Dict[str, Any]]:
    """Enhanced pool data formatter with improved investment calculations for multi-token pools."""
    # Initialize CoinGecko API
    cg = CoinGeckoAPI(api_key=coingecko_api_key) if is_pro_api_key(coingecko_api_key) else CoinGeckoAPI(demo_api_key=coingecko_api_key)
    
    # Determine if we're dealing with a single pool
    is_single_pool = len(pools) == 1
    
    # Sort pools by APR if multiple pools
    sorted_pools = sorted(pools, key=lambda x: float(x.get('apr', 0)), reverse=True) if not is_single_pool else pools
    
    formatted_pools = []
    max_apr = float(sorted_pools[0].get('apr', 0)) if sorted_pools else 0
    
    # Pre-fetch all token prices
    all_token_symbols = []
    for pool in sorted_pools:
        for token in pool["poolTokens"]:
            all_token_symbols.append(token["symbol"])
    
    time.sleep(5)  # Rate limiting
    all_prices = get_pool_token_prices(list(set(all_token_symbols)), cg)
    
    for i, pool in enumerate(sorted_pools):
        # Get TVL from dynamicData
        tvl = float(pool.get('dynamicData', {}).get('totalLiquidity', 0))
        
        # Prepare base data with dynamic tokens
        base_data = {
            "dex_type": BALANCER,
            "chain": pool["chain"].lower(),
            "apr": pool["apr"] * 100,
            "pool_address": pool["address"],
            "pool_id": pool["id"],
            "pool_type": pool["type"],
            "token_count": len(pool["poolTokens"]),
            "il_risk_score": pool.get("il_risk_score"),
            "sharpe_ratio": pool.get("sharpe_ratio"),
            "depth_score": pool.get("depth_score"),
            "max_position_size": pool.get("max_position_size"),
            "type": pool.get("trading_type", LP),
            "tvl": tvl
        }
        
        # Add all tokens dynamically
        for j, token in enumerate(pool["poolTokens"]):
            base_data[f"token{j}"] = token["address"]
            base_data[f"token{j}_symbol"] = token["symbol"]
        
        # Calculate differential investment
        current_apr = float(pool.get('apr', 0))
        if current_apr <= 0.01:  # Skip very low APR pools
            base_data["max_investment_amounts"] = []
            base_data["max_investment_usd"] = 0.0
            formatted_pools.append(base_data)
            continue
        
        # Get base APR for comparison (or 0 for single pool)
        next_best_apr = 0
        if not is_single_pool and i < len(sorted_pools) - 1:
            next_best_apr = float(sorted_pools[i + 1].get('apr', 0))
        
        # Calculate investment amount
        diff_investment = calculate_differential_investment(
            current_apr,
            next_best_apr,
            tvl,
            is_single_pool
        )
        
        # Get token prices from pre-fetched prices
        token_prices = {}
        for token in pool["poolTokens"]:
            token_prices[token["symbol"]] = all_prices.get(token["symbol"], 0)
        
        if diff_investment > 0 and any(price > 0 for price in token_prices.values()):
            token_amounts = get_token_investments_multi(diff_investment, token_prices)
            base_data["max_investment_amounts"] = token_amounts
            base_data["max_investment_usd"] = diff_investment
        else:
            base_data["max_investment_amounts"] = []
            base_data["max_investment_usd"] = 0.0
            
        formatted_pools.append(base_data)
        
    return formatted_pools

def get_opportunities(chains, graphql_endpoint, current_positions, coingecko_api_key, coin_list):
    """Get and format pool opportunities with investment calculations."""
    # Get initial pools
    pools = get_balancer_pools(chains, graphql_endpoint)
    if isinstance(pools, dict) and "error" in pools:
        return pools

    # Filter pools
    filtered_pools = get_filtered_pools(pools, current_positions)
    if not filtered_pools:
        return {"error": "No suitable pools found"}

    # Process basic metrics for each pool
    for pool in filtered_pools:
        pool["chain"] = pool["chain"].lower()
        pool["trading_type"] = LP
        
        # Calculate metrics
        pool["sharpe_ratio"] = get_balancer_pool_sharpe_ratio(
            pool["id"], pool["chain"].upper()
        )
        pool["depth_score"], pool["max_position_size"] = analyze_pool_liquidity(
            pool["id"], pool["chain"].upper()
        )
        
        # Calculate IL risk score for all tokens in the pool
        token_ids = []
        for token in pool["poolTokens"]:
            token_id = get_token_id_from_symbol(
                token["address"],
                token["symbol"],
                coin_list,
                pool["chain"]
            )
            token_ids.append(token_id)
        
        # Only calculate IL risk if we have at least 2 valid token IDs
        valid_token_ids = [tid for tid in token_ids if tid]
        if len(valid_token_ids) >= 2:
            pool["il_risk_score"] = calculate_il_risk_score_multi(
                valid_token_ids, coingecko_api_key
            )
        else:
            pool["il_risk_score"] = None

    # Format pools with investment calculations
    formatted_pools = format_pool_data(filtered_pools, coingecko_api_key)
    
    # Filter pools to only include those with valid investments in both token0 and token1
    valid_investment_pools = filter_valid_investment_pools(formatted_pools)
    
    return valid_investment_pools

def calculate_metrics(
    position: Dict[str, Any], coingecko_api_key: str, coin_list: List[Any], **kwargs
) -> Optional[Dict[str, Any]]:
    # Dynamic handling of tokens
    token_ids = []
    token_count = position.get("token_count", 0)
    
    # If token_count is provided, use it to find all tokens
    if token_count > 0:
        for i in range(token_count):
            token_address_key = f"token{i}"
            token_symbol_key = f"token{i}_symbol"
            
            if token_address_key in position and token_symbol_key in position:
                token_id = get_token_id_from_symbol(
                    position[token_address_key], 
                    position[token_symbol_key], 
                    coin_list,
                    position["chain"]
                )
                token_ids.append(token_id)
    # Fallback to the old token0/token1 format
    elif "token0" in position and "token1" in position:
        token0_id = get_token_id_from_symbol(
            position["token0"], position["token0_symbol"], coin_list, position["chain"]
        )
        token1_id = get_token_id_from_symbol(
            position["token1"], position["token1_symbol"], coin_list, position["chain"]
        )
        token_ids = [token0_id, token1_id]
    
    # Calculate IL risk score if we have at least 2 valid token IDs
    valid_token_ids = [tid for tid in token_ids if tid]
    il_risk_score = None
    if len(valid_token_ids) >= 2:
        il_risk_score = calculate_il_risk_score_multi(valid_token_ids, coingecko_api_key)
    
    sharpe_ratio = get_balancer_pool_sharpe_ratio(
        position["pool_id"], position["chain"].upper()
    )
    depth_score, max_position_size = analyze_pool_liquidity(
        position["pool_id"], position["chain"].upper()
    )
    
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
    get_metrics = kwargs.get("get_metrics", False)
    if get_metrics:
        required_fields.append("position")

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
            errors.append("No suitable pools with valid investments found")
        return {"error": errors} if errors else {"result": result}
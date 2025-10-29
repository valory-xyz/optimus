import warnings

from packages.valory.connections.x402.clients.requests import x402_requests

warnings.filterwarnings("ignore")  # Suppress all warnings

import statistics
import time
import threading
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
    "whitelisted_assets",
    "coin_id_mapping",
    "x402_signer",
    "x402_proxy",
)
BALANCER = "balancerPool"
FEE_RATE_DIVISOR = 1000000
DAYS_IN_YEAR = 365
PERCENT_CONVERSION = 100
TVL_WEIGHT = 0.3
APR_WEIGHT = 0.7
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

# Thread-local storage for errors
_thread_local = threading.local()

def get_errors():
    """Get thread-local error list."""
    if not hasattr(_thread_local, 'errors'):
        _thread_local.errors = []
    return _thread_local.errors


def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    return [field for field in REQUIRED_FIELDS if kwargs.get(field) is None]

def remove_irrelevant_fields(
    kwargs: Dict[str, Any], required_fields: Tuple
) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key in required_fields}

def run_query(query, graphql_endpoint, variables=None) -> Dict[str, Any]:
    logger.info(f"Executing GraphQL query to endpoint: {graphql_endpoint}")
    headers = {"Content-Type": "application/json"}
    payload = {"query": query, "variables": variables or {}}
    response = requests.post(graphql_endpoint, json=payload, headers=headers)
    if response.status_code != 200:
        logger.error(f"GraphQL query failed with status code {response.status_code}")
        return {
            "error": f"GraphQL query failed with status code {response.status_code}"
        }
    result = response.json()
    if "errors" in result:
        logger.error(f"GraphQL Errors: {result['errors']}")
        return {"error": f"GraphQL Errors: {result['errors']}"}
    logger.info("GraphQL query executed successfully")
    return result["data"]

def get_total_apr(pool) -> float:
    apr_items = pool.get("dynamicData", {}).get("aprItems", [])
    return sum(
        item["apr"] for item in apr_items if item["type"] not in EXCLUDED_APR_TYPES
    )


def standardize_metrics(pools, apr_weight=0.7, tvl_weight=0.3):
    """
    Standardize APR and TVL using Z-score normalization and calculate composite scores.
    
    Args:
        pools: List of pool dictionaries
        apr_weight: Weight for APR in composite score (0-1)
        tvl_weight: Weight for TVL in composite score (0-1)
    
    Returns:
        List of pools with added standardized metrics and composite scores
    """
    if not pools:
        return pools
    
    # Extract APR and TVL values with proper type conversion
    aprs = []
    tvls = []
    
    for pool in pools:
        try:
            apr_value = float(pool.get('apr', 0))
            tvl_value = float(pool.get('tvl', 0))
            aprs.append(apr_value)
            tvls.append(tvl_value)
        except (ValueError, TypeError):
            # Use 0 for invalid values
            aprs.append(0.0)
            tvls.append(0.0)
    
    # Calculate means and standard deviations
    apr_mean = np.mean(aprs) if aprs else 0
    apr_std = np.std(aprs) if aprs else 1
    tvl_mean = np.mean(tvls) if tvls else 0
    tvl_std = np.std(tvls) if tvls else 1
    
    # Avoid division by zero
    if apr_std == 0:
        apr_std = 1
    if tvl_std == 0:
        tvl_std = 1
    
    # Standardize metrics and calculate composite scores
    for i, pool in enumerate(pools):
        apr = aprs[i]
        tvl = tvls[i]
        
        # Z-score normalization
        pool['apr_standardized'] = (apr - apr_mean) / apr_std
        pool['tvl_standardized'] = (tvl - tvl_mean) / tvl_std
        
        # Composite score with configurable weights
        pool['composite_score'] = (
            pool['apr_standardized'] * apr_weight + 
            pool['tvl_standardized'] * tvl_weight
        )
    
    return pools


def apply_composite_pre_filter(pools, top_n=10, apr_weight=0.7, tvl_weight=0.3, 
                              min_tvl_threshold=1000, use_composite_filter=True):
    """
    Apply composite pre-filtering to select top pools based on standardized APR and TVL.
    
    Args:
        pools: List of pool dictionaries
        top_n: Number of top pools to select
        apr_weight: Weight for APR in composite score
        tvl_weight: Weight for TVL in composite score
        min_tvl_threshold: Minimum TVL threshold for inclusion
        use_composite_filter: Whether to use composite filtering
    
    Returns:
        List of filtered and ranked pools
    """
    if not pools or not use_composite_filter:
        return pools[:top_n] if pools else []
    
    # Filter by minimum TVL threshold with proper type conversion
    tvl_filtered = []
    for pool in pools:
        try:
            tvl_value = float(pool.get('tvl', 0))
            if tvl_value >= float(min_tvl_threshold):
                tvl_filtered.append(pool)
        except (ValueError, TypeError):
            # Skip pools with invalid TVL values
            continue
    
    if not tvl_filtered:
        return []
    
    # Apply standardization and composite scoring
    standardized_pools = standardize_metrics(tvl_filtered, apr_weight, tvl_weight)
    
    # Sort by composite score (descending)
    standardized_pools.sort(key=lambda x: x.get('composite_score', float('-inf')), reverse=True)
    
    # Select top N pools
    final_selection = standardized_pools[:top_n]
    
    return final_selection

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
    logger.info(f"Fetching Balancer pools for chains: {chains}")
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
        logger.error(f"Error fetching Balancer pools: {data['error']}")
        return data
    pools = data.get("poolGetPools", [])
    logger.info(f"Successfully fetched {len(pools)} Balancer pools")
    return pools

def get_filtered_pools_for_balancer(pools, current_positions, whitelisted_assets, **kwargs):
    # Extract composite filtering parameters
    top_n = kwargs.get('top_n', 10)
    apr_weight = kwargs.get('apr_weight', 0.7)
    tvl_weight = kwargs.get('tvl_weight', 0.3)
    min_tvl_threshold = kwargs.get('min_tvl_threshold', 1000)

    # Filter by type and token count - removing the 2-token restriction
    qualifying_pools = []
    for pool in pools:
        token_count = len(pool.get("poolTokens", []))
        # Accept pools with 3-8 tokens (we're removing the 2-token restriction)
        mapped_type = SUPPORTED_POOL_TYPES.get(pool.get("type"))
        chain = pool["chain"].lower()
        whitelisted_tokens = list(whitelisted_assets.get(chain, {}).keys())
        if (
            mapped_type
            and 2 <= token_count 
            and Web3.to_checksum_address(pool.get("address")) not in current_positions
            and chain in whitelisted_assets
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

    logger.info(f"Found {len(qualifying_pools)} qualifying pools after initial filtering")

    # Apply composite pre-filtering
    filtered_pools = apply_composite_pre_filter(
        qualifying_pools,
        top_n=top_n,
        apr_weight=apr_weight,
        tvl_weight=tvl_weight,
        min_tvl_threshold=min_tvl_threshold,
    )

    logger.info(f"Final filtered pools: {len(filtered_pools)} (top {top_n})")
    return filtered_pools

def get_coin_id_from_symbol(
       coin_id_mapping, symbol, chain_name
    ) -> Optional[str]:
        """Retrieve the CoinGecko token ID using the token's address, symbol, and chain name."""
        # Check if coin_list is valid
        symbol = symbol.lower()
        if symbol in coin_id_mapping.get(chain_name, {}):
            return coin_id_mapping[chain_name][symbol]

        return None

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

def calculate_il_risk_score_multi(token_ids, x402_signer: Optional[str] = None, x402_proxy: Optional[str] = None, time_period: int = 90) -> float:
    """Calculate IL risk score for multiple tokens."""
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
                logger.info(f"[COINGECKO] API Call - Fetching price data for token: {token_id}")
                logger.info(f"[COINGECKO] Request params - from: {from_timestamp}, to: {to_timestamp}, vs_currency: usd")
                
                if x402_signer is not None and x402_proxy is not None:
                    logger.info(f"[COINGECKO] Configuring X402 proxy: {x402_proxy}")
                    cg = CoinGeckoAPI()
                    cg.session = x402_requests(account=x402_signer)
                    cg.api_base_url = x402_proxy.rstrip("/") + "/api/v3/"
                else:
                    return None
                
                prices = cg.get_coin_market_chart_range_by_id(
                    id=token_id,
                    vs_currency="usd",
                    from_timestamp=from_timestamp,
                    to_timestamp=to_timestamp,
                )
                
                logger.info(f"[COINGECKO] API Success - Response type: {type(prices)}, keys: {list(prices.keys()) if isinstance(prices, dict) else 'N/A'}")
                
                prices_list = [x[1] for x in prices["prices"]]
                prices_data.append(prices_list)
                
                logger.info(f"[COINGECKO] Parsed {len(prices_list)} price points for {token_id} - First: ${prices_list[0]:.4f}, Last: ${prices_list[-1]:.4f}")
                time.sleep(1)  # Rate limiting
            except Exception as e:
                error_msg = f"Error fetching price data for {token_id}: {str(e)}"
                logger.error(f"[COINGECKO] API Error - {error_msg}")
                get_errors().append(error_msg)
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
        get_errors().append(f"Error calculating IL risk score: {str(e)}")
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
    logger.info(f"Fetching liquidity metrics for pool {pool_id} on chain {chain}")
    if client is None:
        client = create_graphql_client()
    try:
        query = create_pool_snapshots_query(pool_id, chain)
        logger.info(f"Executing pool snapshots query for pool {pool_id}")
        response = client.execute(query)
        pool_snapshots = response["poolGetSnapshots"]
        if not pool_snapshots:
            logger.warning(f"No pool snapshots found for pool {pool_id}")
            return None

        logger.info(f"Found {len(pool_snapshots)} snapshots for pool {pool_id}")
        avg_tvl = statistics.mean(float(s["totalLiquidity"]) for s in pool_snapshots)
        avg_volume = statistics.mean(
            float(s.get("volume24h", 0)) for s in pool_snapshots
        )
        
        logger.info(f"Pool {pool_id} - Average TVL: {avg_tvl:.2f}, Average Volume: {avg_volume:.2f}")

        depth_score = (
            (np.log1p(avg_tvl) * np.log1p(avg_volume)) / (price_impact * 100)
            if avg_tvl and avg_volume
            else 0
        )
        liquidity_risk_multiplier = (
            max(0, 1 - (1 / depth_score)) if depth_score != 0 else 0
        )
        max_position_size = 50 * (avg_tvl * liquidity_risk_multiplier) / 100

        logger.info(f"Pool {pool_id} - Depth Score: {depth_score:.2f}, Max Position Size: {max_position_size:.2f}")

        return {
            "Average TVL": avg_tvl,
            "Average Daily Volume": avg_volume,
            "Depth Score": depth_score,
            "Liquidity Risk Multiplier": liquidity_risk_multiplier,
            "Maximum Position Size": max_position_size,
            "Meets Depth Score Threshold": depth_score > 50,
        }

    except Exception as e:
        logger.error(f"Error fetching liquidity metrics for pool {pool_id}: {str(e)}")
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
            get_errors().append(f"Could not retrieve depth score for pool {pool_id}")
            return float("nan"), float("nan")
        return metrics["Depth Score"], metrics["Maximum Position Size"]
    except Exception as e:
        get_errors().append(f"Error analyzing pool liquidity for {pool_id}: {str(e)}")
        return float("nan"), float("nan")

def fetch_liquidity_metrics(
    pool_id: str,
    chain: str,
    client: Optional[Client] = None,
    price_impact: float = 0.01,
) -> Optional[Dict[str, Any]]:
    logger.info(f"Fetching enhanced liquidity metrics for pool {pool_id} on chain {chain}")
    if client is None:
        client = create_graphql_client()
    try:
        query = create_pool_snapshots_query(pool_id, chain)
        logger.info(f"Executing enhanced pool snapshots query for pool {pool_id}")
        response = client.execute(query)
        pool_snapshots = response.get("poolGetSnapshots", [])
        if not pool_snapshots:
            logger.warning(f"No pool snapshots found for enhanced metrics calculation for pool {pool_id}")
            return None

        logger.info(f"Found {len(pool_snapshots)} snapshots for enhanced processing of pool {pool_id}")

        # Filter out potentially corrupted data
        filtered_snapshots = []
        for snapshot in pool_snapshots:
            try:
                liquidity_value = float(snapshot.get("totalLiquidity", 0))
                volume_value = float(snapshot.get("volume24h", 0))
                
                # Skip extreme values that could cause numerical issues
                if liquidity_value > 1e16 or volume_value > 1e16:
                    logger.warning(f"Skipping extreme value snapshot for pool {pool_id}: TVL={liquidity_value}, Volume={volume_value}")
                    continue
                    
                filtered_snapshots.append(snapshot)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid snapshot for pool {pool_id}: {str(e)}")
                continue
                
        if not filtered_snapshots:
            logger.error(f"No valid snapshots remaining after filtering for pool {pool_id}")
            return None
        
        logger.info(f"Pool {pool_id} - {len(filtered_snapshots)} valid snapshots after filtering")
                
        # Calculate metrics on filtered data
        try:
            avg_tvl = statistics.mean(float(s.get("totalLiquidity", 0)) for s in filtered_snapshots)
            avg_volume = statistics.mean(float(s.get("volume24h", 0)) for s in filtered_snapshots)
            logger.info(f"Pool {pool_id} - Enhanced metrics: Average TVL: {avg_tvl:.2f}, Average Volume: {avg_volume:.2f}")
        except statistics.StatisticsError as e:
            logger.error(f"Statistics error calculating averages for pool {pool_id}: {str(e)}")
            return None

        # Handle potential division by zero or very small values
        if price_impact <= 0.001:  # Avoid division by extremely small values
            logger.warning(f"Adjusting price impact from {price_impact} to 0.001 for pool {pool_id}")
            price_impact = 0.001
            
        # Calculate depth score with safeguards
        try:
            depth_score = (
                (np.log1p(max(0, avg_tvl)) * np.log1p(max(0, avg_volume))) / (price_impact * 100)
                if avg_tvl > 0 and avg_volume > 0
                else 0
            )
            
            logger.info(f"Pool {pool_id} - Raw depth score calculation: {depth_score}")
            
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
            
            logger.info(f"Pool {pool_id} - Final enhanced metrics: Depth Score: {depth_score:.2f}, Risk Multiplier: {liquidity_risk_multiplier:.4f}, Max Position: {max_position_size:.2f}")
            
        except (ZeroDivisionError, OverflowError, ValueError) as e:
            logger.error(f"Error calculating enhanced metrics for pool {pool_id}: {str(e)}")
            get_errors().append(f"Error calculating metrics for pool {pool_id}: {str(e)}")
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
        logger.error(f"Error fetching enhanced liquidity metrics for pool {pool_id}: {str(e)}")
        get_errors().append(f"Error fetching liquidity metrics for pool {pool_id}: {str(e)}")
        return None

def get_balancer_pool_sharpe_ratio(pool_id, chain, timerange="ONE_YEAR"):
    logger.info(f"Calculating Sharpe ratio for pool {pool_id} on chain {chain} with timerange {timerange}")
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
        logger.info(f"Executing Sharpe ratio query for pool {pool_id}")
        response = requests.post("https://api-v3.balancer.fi/", json={"query": query})
        data = response.json().get("data", {}).get("poolGetSnapshots", [])
        if not data:
            logger.warning(f"No snapshot data found for Sharpe ratio calculation for pool {pool_id}")
            return None

        logger.info(f"Found {len(data)} snapshots for Sharpe ratio calculation for pool {pool_id}")

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
            
            logger.info(f"Pool {pool_id} - Data conversion completed, processing {len(df)} rows")
            
            # Filter out extreme values that could cause numerical issues
            for col in ["sharePrice", "fees24h", "totalLiquidity"]:
                if col in df.columns:
                    # Replace extreme values with NaN
                    extreme_count = ((df[col] > 1e16) | (df[col] < -1e16)).sum()
                    if extreme_count > 0:
                        logger.warning(f"Pool {pool_id} - Filtering {extreme_count} extreme values in {col}")
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
            
            logger.info(f"Pool {pool_id} - Combined returns calculated, {len(total_returns)} valid returns")
            
            # Replace infinity and extreme values
            inf_count = (total_returns == np.inf).sum() + (total_returns == -np.inf).sum()
            if inf_count > 0:
                logger.warning(f"Pool {pool_id} - Replacing {inf_count} infinite values")
            total_returns = total_returns.replace([np.inf, -np.inf], np.nan)
            total_returns = total_returns.mask(total_returns > 1.0, np.nan)  # Cap at 100% daily return
            total_returns = total_returns.mask(total_returns < -1.0, np.nan)  # Cap at -100% daily return
            
            # Remove NaN values
            total_returns = total_returns.dropna()
            
            logger.info(f"Pool {pool_id} - Final returns data: {len(total_returns)} valid returns after filtering")
            
            # Check if we have enough data
            if len(total_returns) < 5:  # Need at least a few data points
                logger.warning(f"Pool {pool_id} - Insufficient data for Sharpe ratio: {len(total_returns)} returns")
                return None
                
            # Calculate Sharpe ratio with error handling
            logger.info(f"Pool {pool_id} - Computing Sharpe ratio with {len(total_returns)} returns")
            sharpe_ratio = pf.timeseries.sharpe_ratio(total_returns)
            
            logger.info(f"Pool {pool_id} - Raw Sharpe ratio: {sharpe_ratio}")
            
            # Cap the Sharpe ratio to reasonable bounds
            if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                logger.warning(f"Pool {pool_id} - Invalid Sharpe ratio (NaN or Inf): {sharpe_ratio}")
                return None
                
            # Cap at reasonable values
            sharpe_ratio = max(min(sharpe_ratio, 10), -10)
            
            logger.info(f"Pool {pool_id} - Final Sharpe ratio: {sharpe_ratio}")
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error processing Sharpe ratio data for pool {pool_id}: {str(e)}")
            get_errors().append(f"Error processing Sharpe ratio data: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio for pool {pool_id}: {str(e)}")
        get_errors().append(f"Error calculating Sharpe ratio: {str(e)}")
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

def get_pool_token_prices(token_symbols: List[str], x402_signer: Optional[str] = None, x402_proxy: Optional[str] = None) -> Dict[str, float]:
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
                        if x402_signer is not None and x402_proxy is not None:
                            coingecko_api = CoinGeckoAPI()
                            coingecko_api.session = x402_requests(account=x402_signer)
                            coingecko_api.api_base_url = x402_proxy.rstrip("/") + "/api/v3/"
                        else:
                            return None
                        
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

def format_pool_data(pools: List[Dict[str, Any]], x402_signer: Optional[str] = None, x402_proxy: Optional[str] = None) -> List[Dict[str, Any]]:
    """Enhanced pool data formatter with improved investment calculations for multi-token pools."""
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
    all_prices = get_pool_token_prices(list(set(all_token_symbols)), x402_signer, x402_proxy)
    
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

def get_opportunities_for_balancer(chains, graphql_endpoint, current_positions, whitelisted_assets, coin_id_mapping, x402_signer, x402_proxy, **kwargs):
    """Get and format pool opportunities with investment calculations."""
    logger.info(f"Starting opportunity search for chains: {chains}")
    logger.info(f"Current positions to exclude: {len(current_positions)}")
    
    # Get initial pools
    pools = get_balancer_pools(chains, graphql_endpoint)
    if isinstance(pools, dict) and "error" in pools:
        logger.error(f"Failed to get pools: {pools['error']}")
        return pools

    # Filter pools
    filtered_pools = get_filtered_pools_for_balancer(pools, current_positions, whitelisted_assets, **kwargs)
    if not filtered_pools:
        logger.warning("No suitable pools found after filtering")
        return {"error": "No suitable pools found"}

    logger.info(f"Processing metrics for {len(filtered_pools)} filtered pools")
    
    # Process basic metrics for each pool
    for i, pool in enumerate(filtered_pools):
        logger.info(f"Processing pool {i+1}/{len(filtered_pools)}: {pool['id']}")
        pool["chain"] = pool["chain"].lower()
        pool["trading_type"] = LP
        
        # Calculate metrics
        logger.info(f"Calculating Sharpe ratio for pool {pool['id']}")
        pool["sharpe_ratio"] = get_balancer_pool_sharpe_ratio(
            pool["id"], pool["chain"].upper()
        )
        
        logger.info(f"Analyzing liquidity for pool {pool['id']}")
        pool["depth_score"], pool["max_position_size"] = analyze_pool_liquidity(
            pool["id"], pool["chain"].upper()
        )
        
        # Calculate IL risk score for all tokens in the pool
        logger.info(f"Calculating IL risk score for pool {pool['id']}")
        token_ids = []
        for token in pool["poolTokens"]:
            token_id = get_coin_id_from_symbol(
                coin_id_mapping,
                token["symbol"],
                pool["chain"]
            )
            token_ids.append(token_id)
        
        # Only calculate IL risk if we have at least 2 valid token IDs
        valid_token_ids = [tid for tid in token_ids if tid]
        if len(valid_token_ids) >= 2:
            pool["il_risk_score"] = calculate_il_risk_score_multi(
                valid_token_ids, x402_signer, x402_proxy
            )
            logger.info(f"IL risk score calculated: {pool['il_risk_score']}")
        else:
            pool["il_risk_score"] = None
            logger.warning(f"Insufficient valid token IDs for IL calculation: {len(valid_token_ids)}")

    # Format pools with investment calculations
    logger.info("Formatting pool data with investment calculations")
    formatted_pools = format_pool_data(filtered_pools, x402_signer, x402_proxy)
    
    # Filter pools to only include those with valid investments in both token0 and token1
    logger.info("Filtering pools for valid investments")
    valid_investment_pools = filter_valid_investment_pools(formatted_pools)
    
    logger.info(f"Final result: {len(valid_investment_pools)} pools with valid investments")
    return valid_investment_pools

def calculate_metrics(
    position: Dict[str, Any], coin_id_mapping: dict, x402_signer: Optional[str] = None, x402_proxy: Optional[str] = None, **kwargs
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
                token_id = get_coin_id_from_symbol(
                    coin_id_mapping,
                    position[token_symbol_key], 
                    position["chain"]
                )
                token_ids.append(token_id)
    # Fallback to the old token0/token1 format
    elif "token0" in position and "token1" in position:
        token0_id = get_coin_id_from_symbol(
            coin_id_mapping, position["token0_symbol"], position["chain"]
        )
        token1_id = get_coin_id_from_symbol(
            coin_id_mapping, position["token1_symbol"], position["chain"]
        )
        token_ids = [token0_id, token1_id]
    
    # Calculate IL risk score if we have at least 2 valid token IDs
    valid_token_ids = [tid for tid in token_ids if tid]
    il_risk_score = None
    if len(valid_token_ids) >= 2:
        il_risk_score = calculate_il_risk_score_multi(valid_token_ids, x402_signer, x402_proxy)
    
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
    logger.info("Starting Balancer pools search execution")
    logger.info(f"Input parameters: {list(kwargs.keys())}")
    
    missing = check_missing_fields(kwargs)
    if missing:
        logger.error(f"Missing required fields: {missing}")
        get_errors().append(f"Required kwargs {missing} were not provided.")
        return {"error": get_errors()}

    required_fields = list(REQUIRED_FIELDS)
    get_metrics = kwargs.get("get_metrics", False)
    logger.info(f"Mode: {'metrics calculation' if get_metrics else 'opportunity search'}")
    
    if get_metrics:
        required_fields.append("position")


    if get_metrics:
        logger.info("Calculating metrics for position")
        metrics = calculate_metrics(**kwargs)
        if metrics is None:
            logger.error("Failed to calculate metrics")
            get_errors().append("Failed to calculate metrics.")
        else:
            logger.info("Metrics calculation completed successfully")
        return {"error": get_errors()} if get_errors() else metrics
    else:
        logger.info("Searching for investment opportunities")
        result = get_opportunities_for_balancer(**kwargs)
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Error in opportunity search: {result['error']}")
            get_errors().append(result["error"])
        elif not result:
            logger.warning("No suitable pools with valid investments found")
            get_errors().append("No suitable pools with valid investments found")
        else:
            logger.info(f"Opportunity search completed successfully with results: {result}")
        
        if get_errors():
            logger.error(f"Execution completed with errors: {get_errors()}")
        else:
            logger.info("Execution completed successfully")
            
        return {"result": result, "error": get_errors()}

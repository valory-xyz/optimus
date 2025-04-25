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
import requests
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from pycoingecko import CoinGeckoAPI
import logging
import random
from collections import deque
import os
import xlsxwriter

# Set up more detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and mappings
REQUIRED_FIELDS = (
    "graphql_endpoint",
    "current_positions",
    "coingecko_api_key",
)
VELODROME = "Velodrome"
PERCENT_CONVERSION = 100
TVL_WEIGHT = 0.7
APR_WEIGHT = 0.3
TVL_PERCENTILE = 50
APR_PERCENTILE = 50
SCORE_PERCENTILE = 80

# Centralized cache storage
CACHE = {
    "snapshots": {},  # Pool snapshots
    "metrics": {},    # Liquidity metrics
    "sharpe": {}      # Sharpe ratios
}

# Rate limiter for CoinGecko API
class RateLimiter:
    def __init__(self, calls_per_minute=30, burst=10):
        self.calls_per_minute = calls_per_minute
        self.burst = burst
        self.call_history = deque(maxlen=calls_per_minute)
        self.last_call_time = 0
        
    def wait_if_needed(self, force_delay=False):
        """Wait if rate limit is approaching, with exponential backoff on force_delay"""
        current_time = time.time()
        
        # Clean up old calls from history
        while self.call_history and current_time - self.call_history[0] > 60:
            self.call_history.popleft()
        
        # If forced delay or approaching limit, wait
        if force_delay or len(self.call_history) >= self.calls_per_minute - self.burst:
            # Calculate dynamic delay with some randomness to avoid synchronized calls
            delay_time = max(1, 60 / self.calls_per_minute) 
            if force_delay:
                # Exponential backoff for forced delays (typically after an error)
                delay_time = min(30, delay_time * (2 ** len(self.call_history) / self.calls_per_minute))
            
            # Add small randomness to avoid synchronized calls
            delay_time += random.uniform(0, 0.5)
            
            logger.debug(f"Rate limiting: waiting {delay_time:.2f} seconds")
            time.sleep(delay_time)
            current_time = time.time()
        
        # Record this call
        self.call_history.append(current_time)
        self.last_call_time = current_time

# Initialize global rate limiter
coingecko_rate_limiter = RateLimiter()

# Cache for token data to reduce redundant API calls
token_id_cache = {}

errors = []

def is_pro_api_key(coingecko_api_key: str) -> bool:
    """
    Check if the provided CoinGecko API key is a pro key.
    """
    # Try using the key as a pro API key with a direct request
    url = "https://pro-api.coingecko.com/api/v3/ping"
    headers = {"x-cg-pro-api-key": coingecko_api_key}
    
    try:
        coingecko_rate_limiter.wait_if_needed()
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    
    # Try as demo key if pro check failed
    url = "https://api.coingecko.com/api/v3/ping"
    headers = {"x-cg-demo-api-key": coingecko_api_key}
    
    try:
        coingecko_rate_limiter.wait_if_needed()
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return False  # It's a valid demo key, not a pro key
    except Exception:
        pass
    
    # If we can't determine for sure, assume it's a demo key
    return False

@lru_cache(None)
def fetch_coin_list(api_key):
    """Fetches the list of coins from CoinGecko API only once."""
    is_pro = is_pro_api_key(api_key)
    
    if is_pro:
        # Pro API endpoint with authentication
        url = "https://pro-api.coingecko.com/api/v3/coins/list"
        headers = {"x-cg-pro-api-key": api_key}
    else:
        # Demo API endpoint
        url = "https://api.coingecko.com/api/v3/coins/list"
        headers = {"x-cg-demo-api-key": api_key}
    
    try:
        coingecko_rate_limiter.wait_if_needed()
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        logger.info(f"Successfully fetched coin list: {len(response.json())} coins")
        return response.json()
    except requests.RequestException as e:
        error_msg = f"Failed to fetch coin list: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
        return None

def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    """Check if any required fields are missing from kwargs."""
    return [field for field in REQUIRED_FIELDS if kwargs.get(field) is None]

def remove_irrelevant_fields(
    kwargs: Dict[str, Any], required_fields: Tuple
) -> Dict[str, Any]:
    """Remove fields that are not in required_fields."""
    return {key: value for key, value in kwargs.items() if key in required_fields}

def run_query(query, graphql_endpoint, variables=None, max_retries=3) -> Dict[str, Any]:
    """Run a GraphQL query and return the result with retries on failure."""
    headers = {"Content-Type": "application/json"}
    payload = {"query": query, "variables": variables or {}}
    
    for retry in range(max_retries):
        try:
            response = requests.post(graphql_endpoint, json=payload, headers=headers)
            if response.status_code != 200:
                error_msg = f"GraphQL query failed with status code {response.status_code}"
                logger.warning(f"{error_msg}, retry {retry+1}/{max_retries}")
                if retry == max_retries - 1:
                    return {"error": error_msg}
                time.sleep(1 * (retry + 1))  # Exponential backoff
                continue
            
            result = response.json()
            if "errors" in result:
                error_msg = f"GraphQL Errors: {result['errors']}"
                logger.warning(f"{error_msg}, retry {retry+1}/{max_retries}")
                if retry == max_retries - 1:
                    return {"error": error_msg}
                time.sleep(1 * (retry + 1))
                continue
                
            return result["data"]
            
        except Exception as e:
            error_msg = f"GraphQL query exception: {str(e)}"
            logger.warning(f"{error_msg}, retry {retry+1}/{max_retries}")
            if retry == max_retries - 1:
                return {"error": error_msg}
            time.sleep(1 * (retry + 1))
    
    return {"error": "Maximum retries reached for GraphQL query"}

def get_velodrome_pools(graphql_endpoint) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Get pools from Velodrome."""
    graphql_query = """
    {
      liquidityPools(first: 20, orderBy: totalValueLockedUSD, orderDirection: desc) {
        id
        fees {
          feeType
          feePercentage
        }
        inputTokens {
          id
          symbol
        }
        isSingleSided
        totalValueLockedUSD
        inputTokenBalances
        inputTokenWeights
        cumulativeVolumeUSD
      }
    }
    """
    
    logger.info("Fetching Velodrome pools")
    data = run_query(graphql_query, graphql_endpoint)
    if isinstance(data, dict) and "error" in data:
        logger.error(f"Error fetching pools: {data['error']}")
        return data
    
    pools = data.get("liquidityPools", [])
    logger.info(f"Successfully fetched {len(pools)} pools")
    return pools

def fetch_daily_snapshots(pool_id, graphql_endpoint, days=30):
    """Fetch daily snapshots for a given pool."""
    query = """
    query DailySnapshots($poolId: ID!, $days: Int!) {
      liquidityPoolDailySnapshots(
        first: $days, 
        orderBy: timestamp, 
        orderDirection: asc, 
        where: { pool: $poolId }
      ) {
        timestamp
        totalValueLockedUSD
        dailyVolumeUSD
        outputTokenPriceUSD
        dailyTotalRevenueUSD
      }
    }
    """
    
    variables = {
        "poolId": pool_id,
        "days": days
    }
    
    logger.info(f"Fetching daily snapshots for pool {pool_id}")
    try:
        data = run_query(query, graphql_endpoint, variables)
        if isinstance(data, dict) and "error" in data:
            logger.error(f"Error fetching daily snapshots: {data['error']}")
            return []
        
        snapshots = data.get("liquidityPoolDailySnapshots", [])
        logger.info(f"Successfully fetched {len(snapshots)} daily snapshots")
        return snapshots
    except Exception as e:
        error_msg = f"Error fetching daily snapshots for pool {pool_id}: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        return []

def calculate_apr(snapshots):
    """Calculate APR from daily snapshots."""
    if not snapshots or len(snapshots) < 7:  # Need at least a week of data
        logger.warning("Not enough snapshots to calculate APR")
        return 0.0
    
    try:
        # Extract daily revenue
        daily_revenues = [float(snapshot.get("dailyTotalRevenueUSD", 0)) for snapshot in snapshots]
        
        # Calculate average daily revenue
        avg_daily_revenue = statistics.mean(daily_revenues)
        
        # Get the latest TVL
        latest_tvl = float(snapshots[-1].get("totalValueLockedUSD", 0))
        
        if latest_tvl <= 0:
            logger.warning("Zero or negative TVL, cannot calculate APR")
            return 0.0
        
        # Calculate APR: (Average Daily Revenue * 365 / TVL) * 100
        apr = (avg_daily_revenue * 365 / latest_tvl) * 100
        
        # Cap at reasonable values
        apr = min(apr, 1000.0)  # Cap at 1000% APR
        
        logger.info(f"Calculated APR: {apr:.2f}%")
        return apr
        
    except Exception as e:
        error_msg = f"Error calculating APR: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        return 0.0

def get_token_id_from_symbol(token_address, symbol, coin_list):
    """Get token ID from symbol."""
    # Check if we already processed this token
    cache_key = f"{token_address}"
    if cache_key in token_id_cache:
        return token_id_cache[cache_key]
    
    # Try to find by symbol
    matching_coins = [
        coin for coin in coin_list if coin["symbol"].lower() == symbol.lower()
    ]
    token_id = matching_coins[0]["id"] if len(matching_coins) == 1 else None
    
    # If multiple matches, try to find by name
    if len(matching_coins) > 1:
        # Look for exact matches or common patterns
        for coin in matching_coins:
            if coin["id"].lower() == symbol.lower() or coin["id"].endswith(symbol.lower()):
                token_id = coin["id"]
                break
    
    # Cache the result
    token_id_cache[cache_key] = token_id
    return token_id

def calculate_price_correlation(price_data_1, price_data_2):
    """
    Calculate correlation between price movement of two tokens using Pearson correlation.
    """
    if len(price_data_1) != len(price_data_2) or len(price_data_1) < 2:
        logger.debug("Insufficient price data for correlation calculation")
        return 0
        
    # Convert to numpy arrays for efficient calculation
    x = np.array(price_data_1)
    y = np.array(price_data_2)
    
    # Calculate correlation coefficient using numpy's corrcoef
    try:
        correlation_matrix = np.corrcoef(x, y)
        correlation = correlation_matrix[0, 1]
        
        # Handle NaN or infinity results
        if np.isnan(correlation) or np.isinf(correlation):
            logger.debug("Correlation calculation resulted in NaN or infinity")
            return 0
            
        return correlation
    except Exception as e:
        logger.debug(f"Error calculating price correlation: {str(e)}")
        return 0

def calculate_volatility_multiplier(price_data, baseline_volatility=0.01):
    """
    Calculate a volatility multiplier based on the standard deviation
    of token prices compared to a baseline.
    """
    if len(price_data) < 2:
        logger.debug("Insufficient price data for volatility calculation")
        return 1.0
        
    try:
        # Calculate standard deviation
        volatility = np.std(price_data)
        
        # Compare to baseline and create a multiplier
        # Higher volatility = higher multiplier
        multiplier = volatility / baseline_volatility
        
        # Cap at reasonable values
        multiplier = max(min(multiplier, 10.0), 0.1)
        
        return multiplier
    except Exception as e:
        logger.debug(f"Error calculating volatility multiplier: {str(e)}")
        return 1.0

def calculate_il_impact_standard(initial_price_ratio, final_price_ratio):
    """
    Calculate impermanent loss impact using the standard formula for a pair of tokens.
    
    Formula: IL Impact = 2√(P1/P0) / (1 + P1/P0) - 1
    """
    if initial_price_ratio <= 0 or final_price_ratio <= 0:
        logger.debug("Invalid price ratios for IL calculation")
        return 0
    
    price_ratio = final_price_ratio / initial_price_ratio
    if price_ratio <= 0:
        logger.debug("Negative or zero price ratio")
        return 0
        
    try:
        # IL Impact = 2√(P1/P0) / (1 + P1/P0) - 1
        # This formula gives the IL impact as a negative value (representing loss)
        il_impact = (2 * np.sqrt(price_ratio)) / (1 + price_ratio) - 1
        return il_impact
    except Exception as e:
        logger.debug(f"Error calculating IL impact: {str(e)}")
        return 0

def batch_fetch_price_data(token_ids, coingecko_api, from_timestamp, to_timestamp, api_key):
    """
    Fetch price data for multiple tokens in batches to optimize API calls.
    """
    price_data = {}
    
    # Process tokens in batches of 5 to reduce API calls
    batch_size = 5
    for i in range(0, len(token_ids), batch_size):
        batch = token_ids[i:i+batch_size]
        logger.info(f"Fetching price data for batch of {len(batch)} tokens")
        
        for token_id in batch:
            try:
                # Wait for rate limit if needed
                coingecko_rate_limiter.wait_if_needed()
                
                # Set API params
                params = {
                    "id": token_id,
                    "vs_currency": "usd",
                    "from_timestamp": from_timestamp,
                    "to_timestamp": to_timestamp,
                }
                
                # Use CoinGecko API with the appropriate authentication
                is_pro = is_pro_api_key(api_key)
                if is_pro:
                    prices = coingecko_api.get_coin_market_chart_range_by_id(
                        **params, 
                        x_cg_pro_api_key=api_key
                    )
                else:
                    prices = coingecko_api.get_coin_market_chart_range_by_id(
                        **params,
                        x_cg_demo_api_key=api_key
                    )
                
                # Extract price data
                token_prices = [x[1] for x in prices["prices"]]
                price_data[token_id] = token_prices
                
                logger.debug(f"Fetched {len(token_prices)} price points for {token_id}")
                
            except Exception as e:
                error_msg = f"Error fetching price data for {token_id}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Force delay on error to avoid rate limit issues
                coingecko_rate_limiter.wait_if_needed(force_delay=True)
        
        # Add a small delay between batches to be nice to the API
        time.sleep(1)
    
    return price_data

def calculate_il_risk_score(token_ids, coingecko_api_key: str, time_period: int = 90) -> float:
    """
    Calculate IL risk score for multiple tokens based on:
    IL Risk Score = IL Impact × Price Correlation × Volatility Multiplier
    """
    # Set up CoinGecko client with appropriate API key type
    is_pro = is_pro_api_key(coingecko_api_key)
    if is_pro:
        cg = CoinGeckoAPI(api_key=coingecko_api_key)
    else:
        cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
    
    # Set up time range
    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=time_period)).timestamp())
    
    # Filter out None token_ids
    valid_token_ids = [tid for tid in token_ids if tid]
    if len(valid_token_ids) < 2:
        logger.warning(f"Not enough valid token IDs for IL calculation: {len(valid_token_ids)}")
        return None
    
    try:
        # Use batch fetching for price data
        price_data_map = batch_fetch_price_data(
            valid_token_ids, cg, from_timestamp, to_timestamp, coingecko_api_key
        )
        
        # Make sure we have data for all tokens
        if len(price_data_map) != len(valid_token_ids):
            missing = set(valid_token_ids) - set(price_data_map.keys())
            logger.warning(f"Missing price data for tokens: {missing}")
            return None
        
        # Create aligned price data arrays
        prices_data = [price_data_map[tid] for tid in valid_token_ids]
        
        # Find minimum length to align all price series
        min_length = min(len(prices) for prices in prices_data)
        if min_length < 2:
            logger.warning(f"Not enough price data points: {min_length}")
            return None
            
        # Truncate all price series to the same length
        aligned_prices = [prices[:min_length] for prices in prices_data]
        
        # For two-token pools, calculate specific correlation and volatility
        if len(aligned_prices) == 2:
            price_correlation = calculate_price_correlation(aligned_prices[0], aligned_prices[1])
            logger.debug(f"Price correlation: {price_correlation}")
            
            # Calculate volatility multiplier for each token
            volatility_mul_1 = calculate_volatility_multiplier(aligned_prices[0])
            volatility_mul_2 = calculate_volatility_multiplier(aligned_prices[1])
            
            # Average the multipliers
            volatility_multiplier = (volatility_mul_1 + volatility_mul_2) / 2
            logger.debug(f"Volatility multiplier: {volatility_multiplier}")
            
            # Calculate IL impact using the standard formula for two tokens
            initial_prices = [aligned_prices[0][0], aligned_prices[1][0]]
            final_prices = [aligned_prices[0][-1], aligned_prices[1][-1]]
            il_impact = calculate_il_impact_standard(
                initial_prices[0] / initial_prices[1], 
                final_prices[0] / final_prices[1]
            )
            logger.debug(f"IL impact: {il_impact}")
            
            # IL Risk Score = IL Impact × Price Correlation × Volatility Multiplier
            il_risk_score = float(il_impact * abs(price_correlation) * volatility_multiplier)
            logger.info(f"Calculated IL risk score: {il_risk_score}")
            return il_risk_score
        
        # For multi-token pools (more than 2)
        else:
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
            logger.debug(f"Average correlation: {avg_correlation}")
            
            # Calculate volatility for each token and average them
            volatility_multipliers = [calculate_volatility_multiplier(prices) for prices in aligned_prices]
            avg_volatility_multiplier = sum(volatility_multipliers) / len(volatility_multipliers)
            logger.debug(f"Average volatility multiplier: {avg_volatility_multiplier}")
            
            # For now, return a simplified score for multi-token pools
            il_risk_score = float(avg_correlation * avg_volatility_multiplier)
            logger.info(f"Calculated multi-token IL risk score: {il_risk_score}")
            return il_risk_score
        
    except Exception as e:
        error_msg = f"Error calculating IL risk score: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        return None

def get_risk_free_rate():
    """
    Get the current risk-free rate based on average stablecoin lending rates.
    """
    # This is a placeholder that would ideally fetch real-time data
    # from major lending protocols like Aave, Compound, etc.
    
    # Typical stablecoin lending rates range from 1-5% APY
    # We use a weighted average across major platforms
    rates = {
        "aave_usdc": 0.029,
        "compound_usdc": 0.031,
        "aave_usdt": 0.028,
        "compound_usdt": 0.030,
        "aave_dai": 0.027,
        "compound_dai": 0.029
    }
    
    # Calculate weighted average
    return sum(rates.values()) / len(rates)

def calculate_sharpe_ratio(snapshots, risk_free_rate=None):
    """
    Calculate Sharpe ratio based on daily revenue data.
    
    Sharpe Ratio = (Rp - Rf) / σp
    
    Where:
    Rp = Average Daily Return
    Rf = Risk Free Rate (Daily)
    σp = Standard Deviation of Daily Returns
    """
    if not snapshots or len(snapshots) < 7:  # Need at least a week of data
        logger.warning("Not enough snapshots to calculate Sharpe ratio")
        return None
    
    try:
        # If risk-free rate not provided, get the default one
        if risk_free_rate is None:
            risk_free_rate = get_risk_free_rate()
        
        # Convert annual risk-free rate to daily
        daily_risk_free_rate = risk_free_rate / 365
        
        # Extract relevant data and calculate daily returns
        daily_returns = []
        prev_tvl = None
        
        for snapshot in snapshots:
            tvl = float(snapshot.get("totalValueLockedUSD", 0))
            revenue = float(snapshot.get("dailyTotalRevenueUSD", 0))
            
            if prev_tvl and prev_tvl > 0:
                # Calculate daily return rate including revenue
                # (Current TVL + Revenue - Previous TVL) / Previous TVL
                daily_return = (tvl + revenue - prev_tvl) / prev_tvl
                daily_returns.append(daily_return)
            
            prev_tvl = tvl
        
        if not daily_returns:
            logger.warning("No valid returns data for Sharpe ratio calculation")
            return None
        
        # Calculate mean and standard deviation of daily returns
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        if std_return == 0:
            logger.warning("Zero standard deviation in returns, cannot calculate Sharpe ratio")
            return None
        
        # Calculate daily Sharpe ratio
        daily_sharpe = (mean_return - daily_risk_free_rate) / std_return
        
        # Annualize Sharpe ratio (multiply by sqrt of trading days)
        annualized_sharpe = daily_sharpe * np.sqrt(365)
        
        # Cap at reasonable values
        sharpe_ratio = max(min(annualized_sharpe, 10), -10)
        
        logger.info(f"Calculated Sharpe ratio: {sharpe_ratio}")
        return sharpe_ratio
        
    except Exception as e:
        error_msg = f"Error calculating Sharpe ratio: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        return None


# Add these functions to calculate depth score in the Velodrome script

def calculate_depth_scores_multi_bp(avg_tvl, avg_volume):
    """
    Calculate depth scores for multiple basis point levels.
    
    Depth Score = (TVL × Average Daily Volume) / (Price Impact × 100)
    
    Args:
        avg_tvl: Average total value locked
        avg_volume: Average daily volume
        
    Returns:
        dict: Depth scores for different basis point levels
    """
    # Basis point levels to calculate (in bp units)
    bp_levels = [10, 50, 100, 200, 300, 400, 500]
    
    depth_scores = {}
    for bp in bp_levels:
        # Convert bp to percentage as decimal (e.g., 10bp = 0.001, 50bp = 0.005)
        # Note: 100bp = 1%, so bp/10000 converts to decimal percentage
        price_impact = bp / 10000
        
        # Calculate depth score using the formula
        if avg_tvl > 0 and avg_volume > 0 and price_impact > 0:
            depth_score = (avg_tvl * avg_volume) / (price_impact * 100)
            # Cap at reasonable values
            depth_score = min(depth_score, 1e12)
        else:
            depth_score = 0
            
        # Store in the result dictionary
        depth_scores[f"depth_score_{bp}bp"] = depth_score
        
    return depth_scores

def fetch_liquidity_metrics(
    pool_id: str,
    snapshots: List[Dict],
    price_impact: float = 0.01,  # Price impact in decimal format (0.01 = 1%)
) -> Optional[Dict[str, Any]]:
    """
    Calculate liquidity metrics for a pool based on snapshots.
    
    Args:
        pool_id: Pool identifier
        snapshots: List of pool snapshots
        price_impact: Price impact in decimal format (e.g., 0.01 = 1%)
        
    Returns:
        dict: Liquidity metrics including depth scores
    """
    cache_key = f"{pool_id}:liquidity_metrics:{price_impact}"
    
    # Check if we have a cached result
    if cache_key in CACHE["metrics"]:
        return CACHE["metrics"][cache_key]
        
    try:
        # Filter out potentially corrupted data
        filtered_snapshots = []
        for snapshot in snapshots:
            try:
                liquidity_value = float(snapshot.get("totalValueLockedUSD", 0))
                volume_value = float(snapshot.get("dailyVolumeUSD", 0))
                
                # Skip extreme values that could cause numerical issues
                if liquidity_value > 1e12 or volume_value > 1e12:
                    continue
                    
                filtered_snapshots.append(snapshot)
            except (ValueError, TypeError):
                continue
                
        if not filtered_snapshots:
            logger.warning(f"No valid snapshots after filtering for pool {pool_id}")
            return None
                
        # Calculate metrics on filtered data
        try:
            avg_tvl = statistics.mean(float(s.get("totalValueLockedUSD", 0)) for s in filtered_snapshots)
            avg_volume = statistics.mean(float(s.get("dailyVolumeUSD", 0)) for s in filtered_snapshots)
            logger.debug(f"Average TVL: {avg_tvl}, Average volume: {avg_volume}")
        except statistics.StatisticsError as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return None

        # Calculate depth scores for different basis point levels
        depth_scores = calculate_depth_scores_multi_bp(avg_tvl, avg_volume)
        
        # Handle potential division by zero or very small values for main depth score
        # Note: price_impact is in decimal format (e.g., 0.01 = 1%)
        if price_impact <= 0.001:  # Avoid division by extremely small values (0.1%)
            price_impact = 0.001
            
        # Calculate depth score using the formula: (TVL × Average Daily Volume) / (Price Impact × 100)
        try:
            # Depth Score = (TVL × Average Daily Volume) / (Price Impact × 100)
            depth_score = (
                (avg_tvl * avg_volume) / (price_impact * 100)
                if avg_tvl > 0 and avg_volume > 0
                else 0
            )
            
            # Cap depth score at reasonable values to prevent numerical issues
            depth_score = min(depth_score, 1e12)
            
            # Liquidity Risk Multiplier = 1 - (1 / Depth Score)
            liquidity_risk_multiplier = (
                max(0, 1 - (1 / max(1, depth_score)))
                if depth_score > 0
                else 0
            )
            
            # Maximum Position Size = 50 * (TVL * Liquidity Risk Multiplier)/100
            max_position_size = min(
                50 * (avg_tvl * liquidity_risk_multiplier) / 100,
                avg_tvl * 0.1  # Cap at 10% of TVL as a safety measure
            )
            
            # Cap max position size to reasonable values
            max_position_size = min(max_position_size, 1e7)
            
        except (ZeroDivisionError, OverflowError, ValueError) as e:
            error_msg = f"Error calculating metrics for pool {pool_id}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            return None

        result = {
            "Average TVL": avg_tvl,
            "Average Daily Volume": avg_volume,
            "Depth Score": depth_score,
            "Liquidity Risk Multiplier": liquidity_risk_multiplier,
            "Maximum Position Size": max_position_size,
            "Meets Depth Score Threshold": depth_score > 50,
        }
        
        # Add depth scores for different basis point levels
        result.update(depth_scores)
        
        # Cache the result
        CACHE["metrics"][cache_key] = result
        return result

    except Exception as e:
        error_msg = f"Error calculating liquidity metrics for pool {pool_id}: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        return None

def analyze_pool_liquidity(
    pool_id: str,
    snapshots: List[Dict],
    price_impact: float = 0.01,  # Price impact in decimal format (0.01 = 1%)
):
    """
    Analyze pool liquidity and return depth scores and maximum position size.
    
    Args:
        pool_id: Pool identifier
        snapshots: List of pool snapshots
        price_impact: Price impact in decimal format (e.g., 0.01 = 1%)
        
    Returns:
        tuple: (depth_score, max_position_size, depth_scores_multi_bp)
    """
    try:
        logger.info(f"Analyzing liquidity for pool {pool_id}")
        metrics = fetch_liquidity_metrics(pool_id, snapshots, price_impact)
        if metrics is None:
            error_msg = f"Could not retrieve depth score for pool {pool_id}"
            logger.error(error_msg)
            errors.append(error_msg)
            return float("nan"), float("nan"), {}
        
        # Extract the main depth score and max position size
        depth_score = metrics["Depth Score"]
        max_position_size = metrics["Maximum Position Size"]
        
        # Extract depth scores for different basis point levels
        depth_scores_multi_bp = {k: v for k, v in metrics.items() if k.startswith("depth_score_")}
        
        logger.debug(f"Depth score: {depth_score}, Max position: {max_position_size}")
        logger.debug(f"Multi-BP depth scores: {depth_scores_multi_bp}")
        
        return depth_score, max_position_size, depth_scores_multi_bp
    except Exception as e:
        error_msg = f"Error analyzing pool liquidity for {pool_id}: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        return float("nan"), float("nan"), {}

# Function to calculate APR-based max position size (copied from Balancer script)
def calculate_max_position_size_apr(current_apr, next_best_apr, current_tvl):
    """
    Calculate maximum position size based on APR dilution compared to next best opportunity.
    
    Formula: a = ((APRi / APRb) - 1) * TVLi
    
    Where:
    - a = Maximum allocation amount
    - APRi = APR of current pool (formatted as 1.xx where 6% = 1.06)
    - APRb = APR of next best pool (formatted as 1.xx where 6% = 1.06)
    - TVLi = TVL of current pool
    """
    # Convert APRs to the correct format for formula (decimal to decimal+1 format)
    current_apr_formatted = 1 + current_apr
    next_best_apr_formatted = 1 + next_best_apr
    
    # Avoid division by zero or invalid calculations
    if next_best_apr_formatted <= 1 or current_apr_formatted <= next_best_apr_formatted:
        logger.debug(f"Invalid APR comparison: current={current_apr}, next_best={next_best_apr}")
        return 0
        
    try:
        # Calculate maximum allocation using the formula: a = ((APRi / APRb) - 1) * TVLi
        max_allocation = (current_apr_formatted / next_best_apr_formatted - 1) * current_tvl
        
        # Ensure non-negative allocation and cap at reasonable values (50% of pool TVL)
        result = max(0, min(max_allocation, current_tvl * 0.5))
        
        logger.debug(f"Calculated APR-based max position: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating APR-based max position size: {str(e)}")
        return 0

def get_filtered_pools(pools, current_positions):
    """Filter pools based on criteria and exclude current positions."""
    qualifying_pools = []
    
    for pool in pools:
        pool_id = pool.get("id")
        
        # Skip if this is a current position
        if pool_id in current_positions:
            continue
        
        # Get token count
        token_count = len(pool.get("inputTokens", []))
        
        # For this adaptation, we'll focus on 2-token pools
        if token_count == 2:
            # Add basic metrics
            pool["token_count"] = token_count
            pool["tvl"] = float(pool.get("totalValueLockedUSD", 0))
            qualifying_pools.append(pool)
    
    logger.info(f"Identified {len(qualifying_pools)} qualifying pools after filtering")
    
    # If very few pools, return them all
    if len(qualifying_pools) <= 5:
        return qualifying_pools

    # Get TVL distribution for scoring
    tvl_list = [float(p.get("totalValueLockedUSD", 0)) for p in qualifying_pools]
    tvl_threshold = np.percentile(tvl_list, TVL_PERCENTILE)
    max_tvl = max(tvl_list) if tvl_list else 1
    
    # We'll calculate APR in a separate step
    return [p for p in qualifying_pools if p["tvl"] >= tvl_threshold]

def get_opportunities(graphql_endpoint, current_positions, coingecko_api_key, coin_list):
    """Get and format pool opportunities with all required metrics."""
    start_time = time.time()
    logger.info("Starting opportunity discovery")
    
    # Get pools
    pools = get_velodrome_pools(graphql_endpoint)
    if isinstance(pools, dict) and "error" in pools:
        error_msg = f"Error in pool discovery: {pools['error']}"
        logger.error(error_msg)
        return pools

    # Filter pools
    filtered_pools = get_filtered_pools(pools, current_positions)
    if not filtered_pools:
        logger.warning("No suitable pools found after filtering")
        return {"error": "No suitable pools found"}

    # Process metrics for each pool
    logger.info(f"Processing metrics for {len(filtered_pools)} filtered pools")
    for i, pool in enumerate(filtered_pools):
        pool_id = pool["id"]
        logger.info(f"Processing pool {i+1}/{len(filtered_pools)}: {pool_id}")
        
        # Fetch daily snapshots
        snapshots = fetch_daily_snapshots(pool_id, graphql_endpoint)
        
        # Calculate APR
        pool["total_apr"] = calculate_apr(snapshots)
        pool["organic_apr"] = pool["total_apr"]  # For simplicity, all APR is organic
        
        # Calculate Sharpe ratio
        pool["sharpe_ratio"] = calculate_sharpe_ratio(snapshots)
        
        # Calculate IL risk score
        token_ids = []
        for token in pool["inputTokens"]:
            token_id = get_token_id_from_symbol(
                token["id"],
                token["symbol"],
                coin_list
            )
            token_ids.append(token_id)
        
        # Only calculate IL risk if we have at least 2 valid token IDs
        valid_token_ids = [tid for tid in token_ids if tid]
        if len(valid_token_ids) >= 2:
            pool["il_risk_score"] = calculate_il_risk_score(
                valid_token_ids, coingecko_api_key
            )
        else:
            logger.warning(f"Not enough valid token IDs for IL calculation for pool {pool_id}")
            pool["il_risk_score"] = None
            
        # NEW: Calculate depth score
        pool["depth_score"], pool["max_position_size"], pool["depth_scores_multi_bp"] = analyze_pool_liquidity(
            pool_id, snapshots
        )
        logger.info(f"Calculated depth score: {pool.get('depth_score')} and max position size: {pool.get('max_position_size')}")
    
    # After calculating APR, apply the APR threshold
    apr_list = [float(p.get("total_apr", 0)) for p in filtered_pools]
    
    # Rest of the function remains the same...
    
    # NEW: Calculate APR-based max position size for all pools
    if len(filtered_pools) > 1:
        sorted_pools = sorted(filtered_pools, key=lambda x: float(x.get('total_apr', 0)), reverse=True)
        logger.info("Calculating APR-based max position sizes using next best opportunities")
        
        # For each pool, calculate max position size based on next best opportunity
        for i, pool in enumerate(sorted_pools):
            current_apr = float(pool.get('total_apr', 0))
            current_tvl = float(pool["totalValueLockedUSD"])
            
            # For the highest APR pool, use the second best as reference
            if i < len(sorted_pools) - 1:
                next_best_apr = float(sorted_pools[i+1].get('total_apr', 0))
            else:
                # For the last pool, use a reasonable fallback (e.g., 70% of its own APR)
                next_best_apr = current_apr * 0.7
            
            # Calculate APR-based max position size
            apr_based_max_position = calculate_max_position_size_apr(
                current_apr, 
                next_best_apr, 
                current_tvl
            )
            
            # Store the result
            pool["max_position_size_apr"] = apr_based_max_position
            
            # Update the max_position_size to be the smaller of the two values
            if apr_based_max_position > 0 and not np.isnan(pool.get("max_position_size", float('nan'))):
                pool["max_position_size"] = min(pool["max_position_size"], apr_based_max_position)
            elif apr_based_max_position > 0:
                pool["max_position_size"] = apr_based_max_position
    
    # Format pools with all required metrics
    formatted_pools = format_pool_data(filtered_pools)
    
    execution_time = time.time() - start_time
    logger.info(f"Opportunity discovery completed in {execution_time:.2f} seconds")
    logger.info(f"Found {len(formatted_pools)} valid opportunities")
    
    return formatted_pools

# 3. Update format_pool_data function to include depth score in the output
def format_pool_data(pools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format pool data for output according to required schema."""
    formatted_pools = []
    
    for pool in pools:
        # Skip pools with more than two tokens
        if pool.get("token_count", 0) != 2:
            continue
            
        # Prepare base data including all required fields
        formatted_pool = {
            "dex_type": VELODROME,
            "pool_address": pool["id"],
            "pool_id": pool["id"],
            "has_incentives": False,  # We'd need to adapt this for Velodrome's incentives
            "total_apr": pool.get("total_apr", 0),
            "organic_apr": pool.get("organic_apr", 0),
            "incentive_apr": 0,  # Placeholder, would need to adapt for Velodrome
            "tvl": float(pool.get("totalValueLockedUSD", 0)),
            "is_lp": True,  # All pools are LP opportunities
            "sharpe_ratio": pool.get("sharpe_ratio"),
            "il_risk_score": pool.get("il_risk_score"),
            "token_count": pool.get("token_count", 0),
            "volume": float(pool.get("cumulativeVolumeUSD", 0)),
            # NEW: Add depth score and max position size
            "chain":"optimism",
            "depth_score": pool.get("depth_score"),
            "max_position_size": pool.get("max_position_size"),
            "max_position_size_apr": pool.get("max_position_size_apr", 0)
        }
        
        # NEW: Add depth scores for different basis point levels
        if "depth_scores_multi_bp" in pool:
            formatted_pool.update(pool["depth_scores_multi_bp"])
        
        # Add tokens (should be exactly 2 tokens)
        tokens = pool.get("inputTokens", [])
        if len(tokens) >= 1:
            formatted_pool["token0"] = tokens[0]["id"]
            formatted_pool["token0_symbol"] = tokens[0]["symbol"]
        
        if len(tokens) >= 2:
            formatted_pool["token1"] = tokens[1]["id"]
            formatted_pool["token1_symbol"] = tokens[1]["symbol"]
            
        formatted_pools.append(formatted_pool)
        
    logger.info(f"Formatted {len(formatted_pools)} pools for output")
    return formatted_pools

def run(**kwargs) -> Dict[str, Union[bool, str, List[Dict[str, Any]]]]:
    """Main function to run the Velodrome pool analysis.
    
    Args:
        **kwargs: Arbitrary keyword arguments
            graphql_endpoint: GraphQL endpoint for Velodrome API
            current_positions: List of current position IDs to exclude
            coingecko_api_key: API key for CoinGecko
            export_excel: Boolean flag to export results to Excel
            excel_filename: Filename for Excel export
            
    Returns:
        Dict containing either error messages or result data
    """
    # Clear previous errors
    errors.clear()
    
    # Clear caches for a fresh run
    for cache_type in CACHE:
        CACHE[cache_type].clear()
    token_id_cache.clear()
    
    start_time = time.time()
    logger.info("Starting Velodrome pool analysis")
    
    # Check for missing required fields
    missing = check_missing_fields(kwargs)
    if missing:
        error_msg = f"Required kwargs {missing} were not provided."
        logger.error(error_msg)
        errors.append(error_msg)
        return {"error": errors}

    # Fetch coin list with API key
    coin_list = fetch_coin_list(kwargs.get("coingecko_api_key"))
    if coin_list is None:
        error_msg = "Failed to fetch coin list."
        logger.error(error_msg)
        errors.append(error_msg)
        return {"error": errors}

    # Get opportunities
    result = get_opportunities(
        kwargs["graphql_endpoint"],
        kwargs.get("current_positions", []),
        kwargs["coingecko_api_key"],
        coin_list
    )
    
    if isinstance(result, dict) and "error" in result:
        errors.append(result["error"])
        logger.error(f"Error in opportunity discovery: {result['error']}")
        return {"error": errors}
    elif not result:
        error_msg = "No suitable pools found"
        logger.warning(error_msg)
        errors.append(error_msg)
        return {"error": errors}
    
    execution_time = time.time() - start_time
    logger.info(f"Full execution completed in {execution_time:.2f} seconds")
    
    return {"result": result}

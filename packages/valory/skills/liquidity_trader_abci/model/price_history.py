#!/usr/bin/env python3
"""Price history utilities for liquidity pools."""

import logging
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any, Union

# Set up logging
logger = logging.getLogger(__name__)

# Optional CoinGecko API integration if available
try:
    from pycoingecko import CoinGeckoAPI
    COINGECKO_AVAILABLE = True
except ImportError:
    logger.warning("pycoingecko not installed. Using mock data only.")
    COINGECKO_AVAILABLE = False

# LRU cache for API calls
# Use a relatively large cache size to handle multiple token pairs
API_CACHE_SIZE = 128
DEFAULT_DAYS = 30
PRICE_VOLATILITY_THRESHOLD = 0.02  # 2% threshold for stablecoin detection


class CoinGeckoNotAvailable(Exception):
    """Exception raised when CoinGecko API is not available."""
    pass


@lru_cache(maxsize=API_CACHE_SIZE)
def get_coin_id_from_address(chain: str, address: str) -> Optional[str]:
    """
    Get CoinGecko coin ID from token address.
    
    Args:
        chain: The blockchain name (e.g., "optimism")
        address: Token contract address
        
    Returns:
        CoinGecko coin ID or None if not found
    """
    if not COINGECKO_AVAILABLE:
        raise CoinGeckoNotAvailable("CoinGecko API not available")
    
    # Convert chain name to CoinGecko platform ID
    platform_map = {
        "ethereum": "ethereum",
        "optimism": "optimistic-ethereum",
        "arbitrum": "arbitrum-one",
        "polygon": "polygon-pos",
        "base": "base",
    }
    
    platform = platform_map.get(chain.lower())
    if not platform:
        logger.warning(f"Unsupported chain: {chain}")
        return None
    
    cg = CoinGeckoAPI()
    
    try:
        # Rate limiting to avoid CoinGecko API restrictions
        time.sleep(0.6)
        
        # Try to find coin by contract address
        # This endpoint returns a list of coins matching the contract address
        contract_info = cg.get_coin_info_from_contract_address_by_id(
            id=platform,
            contract_address=address
        )
        
        return contract_info.get("id")
    except Exception as e:
        logger.warning(f"Error getting coin ID for {chain}/{address}: {str(e)}")
        return None


@lru_cache(maxsize=API_CACHE_SIZE)
def get_historical_market_data(coin_id: str, days: int = DEFAULT_DAYS) -> Dict[str, Any]:
    """
    Get historical market data for a coin from CoinGecko.
    
    Args:
        coin_id: CoinGecko coin ID
        days: Number of days of history
        
    Returns:
        Dictionary with price history data
    """
    if not COINGECKO_AVAILABLE:
        raise CoinGeckoNotAvailable("CoinGecko API not available")
    
    cg = CoinGeckoAPI()
    
    try:
        # Rate limiting to avoid CoinGecko API restrictions
        time.sleep(0.6)
        
        # Get market chart data
        # This includes prices, market caps, and volumes
        chart_data = cg.get_coin_market_chart_by_id(
            id=coin_id,
            vs_currency="usd",
            days=days
        )
        
        # Extract price data (timestamps and prices)
        prices_data = chart_data.get("prices", [])
        
        # Format data as separate lists of timestamps and prices
        timestamps = [entry[0]/1000 for entry in prices_data]  # Convert ms to seconds
        prices = [entry[1] for entry in prices_data]
        
        return {
            "coin_id": coin_id,
            "timestamps": timestamps,
            "prices": prices,
            "days": days,
            "last_updated": time.time()
        }
    except Exception as e:
        logger.warning(f"Error getting price history for {coin_id}: {str(e)}")
        return {
            "coin_id": coin_id,
            "timestamps": [],
            "prices": [],
            "days": days,
            "error": str(e)
        }


@lru_cache(maxsize=API_CACHE_SIZE)
def get_pool_token_history(
    chain: str, 
    token0_address: str, 
    token1_address: str, 
    days: int = DEFAULT_DAYS,
    use_mock_data: bool = False
) -> Dict[str, Any]:
    """
    Get historical price data for both tokens in a pool.
    
    Args:
        chain: The blockchain name (e.g., "optimism")
        token0_address: Token0 contract address
        token1_address: Token1 contract address
        days: Number of days of history
        use_mock_data: Whether to use mock data instead of API calls
        
    Returns:
        Dictionary with price history for both tokens and ratio prices
    """
    if use_mock_data or not COINGECKO_AVAILABLE:
        logger.info("Using mock price data")
        # Import mock data generator from velodrome model
        from .velodrome import generate_mock_price_data
        return generate_mock_price_data(days=days)
    
    try:
        # Get coin IDs for both tokens
        token0_id = get_coin_id_from_address(chain, token0_address)
        token1_id = get_coin_id_from_address(chain, token1_address)
        
        if not token0_id or not token1_id:
            logger.warning(f"Could not find coin IDs for tokens: {token0_address}, {token1_address}")
            # Fall back to mock data
            from .velodrome import generate_mock_price_data
            return generate_mock_price_data(days=days)
        
        # Get price history for both tokens
        token0_data = get_historical_market_data(token0_id, days)
        token1_data = get_historical_market_data(token1_id, days)
        
        # Check if we have valid price data
        if not token0_data.get("prices") or not token1_data.get("prices"):
            logger.warning("Missing price data for one or both tokens")
            # Fall back to mock data
            from .velodrome import generate_mock_price_data
            return generate_mock_price_data(days=days)
        
        # Combine data and calculate price ratios
        token0_prices = token0_data.get("prices", [])
        token1_prices = token1_data.get("prices", [])
        token0_timestamps = token0_data.get("timestamps", [])
        token1_timestamps = token1_data.get("timestamps", [])
        
        # Ensure price lists are the same length by taking the shorter one
        min_length = min(len(token0_prices), len(token1_prices))
        token0_prices = token0_prices[:min_length]
        token1_prices = token1_prices[:min_length]
        token0_timestamps = token0_timestamps[:min_length]
        token1_timestamps = token1_timestamps[:min_length]
        
        # Calculate ratio prices (token0/token1)
        ratio_prices = []
        for i in range(min_length):
            if token1_prices[i] > 0:
                ratio_prices.append(token0_prices[i] / token1_prices[i])
            else:
                ratio_prices.append(token0_prices[i])  # Avoid division by zero
        
        # Get current price (latest)
        current_price = ratio_prices[-1] if ratio_prices else 1.0
        
        # Detect if this is a stablecoin pool
        is_stablecoin = check_is_stablecoin_pool(token0_prices, token1_prices)
        
        return {
            "token0": {
                "address": token0_address,
                "coin_id": token0_id,
                "prices": token0_prices,
                "timestamps": token0_timestamps
            },
            "token1": {
                "address": token1_address,
                "coin_id": token1_id,
                "prices": token1_prices,
                "timestamps": token1_timestamps
            },
            "ratio_prices": ratio_prices,
            "current_price": current_price,
            "days": days,
            "is_stablecoin": is_stablecoin
        }
    except Exception as e:
        logger.error(f"Error getting pool token history: {str(e)}")
        # Fall back to mock data
        from .velodrome import generate_mock_price_data
        return generate_mock_price_data(days=days)


def check_is_stablecoin_pool(
    token0_prices: List[float], 
    token1_prices: List[float],
    threshold: float = PRICE_VOLATILITY_THRESHOLD
) -> bool:
    """
    Check if a pool is a stablecoin pool based on price volatility.
    
    Args:
        token0_prices: List of token0 prices
        token1_prices: List of token1 prices
        threshold: Volatility threshold to classify as stablecoin
        
    Returns:
        True if pool is a stablecoin pool, False otherwise
    """
    # Check if either list is empty
    if not token0_prices or not token1_prices:
        return False
    
    # Helper function to calculate price volatility
    def calculate_volatility(prices):
        if not prices:
            return 1.0
            
        min_price = min(prices)
        max_price = max(prices)
        
        if min_price <= 0:
            return 1.0
            
        # Calculate max percent change
        return (max_price / min_price) - 1.0
    
    # Calculate volatility for both tokens
    token0_volatility = calculate_volatility(token0_prices)
    token1_volatility = calculate_volatility(token1_prices)
    
    # A pool is considered a stablecoin pool if both tokens have low volatility
    is_token0_stable = token0_volatility < threshold
    is_token1_stable = token1_volatility < threshold
    
    return is_token0_stable and is_token1_stable 
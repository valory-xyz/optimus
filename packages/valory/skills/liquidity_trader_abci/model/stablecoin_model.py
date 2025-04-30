#!/usr/bin/env python3
"""Stablecoin model for concentrated liquidity pools."""

import logging
import numpy as np
import scipy.optimize as optimize
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

from .utils import (
    price_to_tick, 
    tick_to_price, 
    MIN_TICK, 
    MAX_TICK, 
    OPTIMIZATION_CACHE_SIZE
)

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_STABLECOIN_TICK_WIDTHS = {
    "min": 30,     # Narrow range for very stable pairs
    "low": 60,     # Default low risk range
    "medium": 120, # Medium risk/reward
    "high": 240,   # High risk/reward
    "max": 400     # Very wide range for more volatile periods
}


@lru_cache(maxsize=OPTIMIZATION_CACHE_SIZE)
def optimize_stablecoin_liquidity(
    price_data: Tuple[float, ...],
    current_price: float,
    min_tick_spacing: int,
    fee_tier: float,
    risk_level: str = "low"
) -> Dict[str, Any]:
    """
    Optimize liquidity placement for stablecoin pools.
    
    Args:
        price_data: Historical price data as tuple (for LRU caching)
        current_price: Current price ratio between token0/token1
        min_tick_spacing: Minimum tick spacing required by the pool
        fee_tier: Fee tier of the pool (e.g., 0.0005, 0.003)
        risk_level: Risk level profile (min, low, medium, high, max)
        
    Returns:
        Dictionary with optimized parameters
    """
    # Convert tuple back to list for processing
    prices = list(price_data)
    
    if not prices:
        logger.warning("No price data provided")
        return {
            "current_price": current_price,
            "tick_lower": -1000,  # Default wide range
            "tick_upper": 1000,
            "success": False,
            "error": "No price data"
        }
    
    # Get tick width based on risk level
    tick_width = DEFAULT_STABLECOIN_TICK_WIDTHS.get(
        risk_level, DEFAULT_STABLECOIN_TICK_WIDTHS["low"]
    )
    
    # Adjust based on fee tier
    fee_multiplier = {
        0.0001: 0.7,  # Lowest fee tier (narrowest range)
        0.0005: 0.85, # Low fee tier
        0.003: 1.0,   # Medium fee tier (standard)
        0.01: 1.2     # High fee tier (wider range)
    }.get(fee_tier, 1.0)
    
    # Apply fee multiplier to tick width
    tick_width = int(tick_width * fee_multiplier)
    
    # Ensure tick width is a multiple of min_tick_spacing
    if min_tick_spacing > 0:
        tick_width = (tick_width // min_tick_spacing) * min_tick_spacing
        # Ensure at least one tick spacing
        tick_width = max(tick_width, min_tick_spacing)
    
    # Calculate price range statistics
    min_price = min(prices)
    max_price = max(prices)
    median_price = np.median(prices)
    
    # Use log mean for better balance
    if min_price > 0:
        log_mean = np.exp(np.mean(np.log(prices)))
    else:
        log_mean = median_price
    
    # Center tick based on statistical measures and current price
    if current_price > 0:
        # Weighted average of log mean, median, and current price
        # Favor current price more for stability
        center_price = (log_mean * 0.2) + (median_price * 0.3) + (current_price * 0.5)
    else:
        center_price = median_price if median_price > 0 else max(1.0, max_price)
    
    # Calculate optimal lower and upper ticks
    # For stablecoins, we use a symmetric approach centered around the middle
    center_tick = price_to_tick(center_price, token0_decimals=18, token1_decimals=18)
    tick_lower = center_tick - (tick_width // 2)
    tick_upper = center_tick + (tick_width // 2)
    
    # Round to required tick spacing
    if min_tick_spacing > 0:
        tick_lower = (tick_lower // min_tick_spacing) * min_tick_spacing
        tick_upper = (tick_upper // min_tick_spacing) * min_tick_spacing
    
    # Ensure minimum range size
    if tick_upper - tick_lower < min_tick_spacing:
        tick_upper = tick_lower + min_tick_spacing
    
    return {
        "current_price": current_price,
        "center_price": center_price,
        "center_tick": center_tick,
        "tick_width": tick_width,
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "price_stats": {
            "min": min_price,
            "max": max_price,
            "median": median_price,
            "log_mean": log_mean
        },
        "success": True
    }


def expected_liquidity_fees(
    price_data: List[float], 
    lower_price: float, 
    upper_price: float,
    fee_tier: float
) -> float:
    """
    Calculate expected liquidity fees from historical price movements.
    
    Args:
        price_data: Historical price data
        lower_price: Lower price bound
        upper_price: Upper price bound
        fee_tier: Pool fee tier (e.g., 0.0005, 0.003)
        
    Returns:
        Expected fee APR as a decimal
    """
    if not price_data or len(price_data) < 2:
        return 0.0
    
    volume_proxy = 0.0
    in_range_time = 0
    
    # Estimate trading volume based on price movements
    for i in range(1, len(price_data)):
        price_prev = price_data[i-1]
        price_curr = price_data[i]
        
        # Check if price is in range
        if lower_price <= price_curr <= upper_price:
            in_range_time += 1
            
            # Add absolute price change as proxy for trading volume
            volume_proxy += abs(price_curr - price_prev) / price_prev
    
    # If price was never in range
    if in_range_time == 0:
        return 0.0
    
    # Annualize the estimate (assuming daily data)
    avg_daily_volume = volume_proxy / len(price_data)
    annual_volume = avg_daily_volume * 365
    
    # Calculate expected share of fees
    # This is a simplification - real fee calculation would depend on
    # the concentration of liquidity in the selected range
    range_width = (upper_price / lower_price) - 1.0
    if range_width <= 0:
        return 0.0
    
    # Inverse relationship between range size and fee share
    # Narrower ranges capture higher proportion of fees when in range
    fee_capture_ratio = max(0.1, min(1.0, 0.2 / range_width))
    
    # Time in range factor
    time_in_range = in_range_time / (len(price_data) - 1)
    
    # Expected fee APR
    fee_apr = annual_volume * fee_tier * fee_capture_ratio * time_in_range
    
    return fee_apr


def find_optimal_tick_range(
    price_data: List[float],
    current_price: float,
    fee_tier: float = 0.003,
    min_tick_spacing: int = 60,
    risk_level: str = "medium",
    token0_decimals: int = 18,
    token1_decimals: int = 18
) -> Dict[str, Any]:
    """
    Find optimal tick range for a concentrated liquidity position.
    
    This function balances risk (time in range) vs. reward (fees earned)
    based on historical price movements.
    
    Args:
        price_data: Historical price data
        current_price: Current price
        fee_tier: Pool fee tier
        min_tick_spacing: Minimum tick spacing for the pool
        risk_level: Risk preference (min, low, medium, high, max)
        token0_decimals: Decimals for token0
        token1_decimals: Decimals for token1
        
    Returns:
        Dictionary with optimized parameters
    """
    # Convert price data to immutable tuple for cache key
    price_data_tuple = tuple(price_data)
    
    # Call cached optimization function
    result = optimize_stablecoin_liquidity(
        price_data_tuple,
        current_price,
        min_tick_spacing,
        fee_tier,
        risk_level
    )
    
    # Convert ticks back to prices for reporting
    lower_price = tick_to_price(result["tick_lower"], token0_decimals=token0_decimals, token1_decimals=token1_decimals)
    upper_price = tick_to_price(result["tick_upper"], token0_decimals=token0_decimals, token1_decimals=token1_decimals)
    
    # Calculate expected fees
    expected_fees = expected_liquidity_fees(
        price_data,
        lower_price,
        upper_price,
        fee_tier
    )
    
    # Calculate time in range
    in_range_count = sum(1 for p in price_data if lower_price <= p <= upper_price)
    time_in_range = in_range_count / len(price_data) if price_data else 0
    
    # Add additional metrics to result
    result.update({
        "lower_price": lower_price,
        "upper_price": upper_price,
        "expected_fee_apr": expected_fees,
        "time_in_range": time_in_range
    })
    
    return result 
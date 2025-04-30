#!/usr/bin/env python3
"""
Velodrome Concentrated Liquidity model for the liquidity trader ABCI skill.

This module provides utilities for calculating optimal tick ranges for Velodrome
concentrated liquidity pools based on historical price data and economic modeling.
"""

import logging
import random
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
from scipy import optimize

from .stablecoin_model import optimize_stablecoin_liquidity, find_optimal_tick_range
from .price_history import get_pool_token_history, check_is_stablecoin_pool
from .utils import (
    price_to_tick, 
    tick_to_price, 
    round_to_tick_spacing,
    sqrtPriceX96_to_price,
    price_to_sqrtPriceX96,
    MIN_TICK, 
    MAX_TICK, 
    BASE_TOKEN_DECIMALS,
    QUOTE_TOKEN_DECIMALS,
    Q96,
    DEFAULT_DAYS,
    STABLECOIN_THRESHOLD
)


# Set up logging
logger = logging.getLogger(__name__)


# Token address maps for stablecoin identification
STABLECOIN_ADDRESSES = {
    "optimism": [
        "0x7f5c764cbc14f9669b88837ca1490cca17c31607",  # USDC
        "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58",  # USDT
        "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1",  # DAI
        "0xc40f949f8a4e094d1b49a23ea9241d289b7b2819",  # LUSD
    ]
}


def generate_mock_price_data(days: int = 30, is_stablecoin: bool = False, volatility: float = 0.1) -> Dict[str, Any]:
    """
    Generate synthetic price data for testing.
    
    Args:
        days: Number of days of history to generate
        is_stablecoin: Whether to generate stablecoin-like price movement
        volatility: The volatility parameter for price generation
        
    Returns:
        Dict with synthetic price data
    """
    start_date = (datetime.now() - timedelta(days=days)).timestamp()
    dates = [start_date + i * 86400 for i in range(days)]
    
    if is_stablecoin:
        # Stablecoin pairs hover around 1.0 with tiny movements
        base_price = 1.0
        price_variation = volatility  # Very small for stablecoins
    else:
        # Regular tokens have more meaningful price variation
        base_price = random.uniform(0.0001, 1000)
        price_variation = volatility * base_price  # Proportional to base price
    
    # Generate a random walk for the price
    np.random.seed(42)  # For reproducibility
    changes = np.random.normal(0, price_variation, days)
    prices = [base_price]
    
    for i in range(1, days):
        # Ensure prices remain positive and don't drift too far
        new_price = prices[-1] + changes[i]
        if new_price <= 0:
            new_price = prices[-1] * 0.95  # Prevent negative prices
        prices.append(new_price)
    
    # For stablecoins, ensure they stay close to 1.0
    if is_stablecoin:
        prices = [max(0.95, min(1.05, p)) for p in prices]
    
    ratio_prices = prices
    current_price = ratio_prices[-1]
    
    return {
        "token0": {
            "prices": [1.0] * days,
            "dates": dates
        },
        "token1": {
            "prices": prices,
            "dates": dates
        },
        "ratio_prices": ratio_prices,
        "current_price": current_price,
        "days": days,
        "is_stablecoin": is_stablecoin
    }


def is_stablecoin_pool(price_data: Dict[str, Any]) -> bool:
    """
    Determine if a pool is a stablecoin pool based on price variance.
    
    Args:
        price_data: Historical price data dictionary
        
    Returns:
        Boolean indicating if this is a stablecoin pool
    """
    if "is_stablecoin" in price_data:
        return price_data["is_stablecoin"]
        
    ratio_prices = price_data.get("ratio_prices", [])
    if not ratio_prices:
        return False
        
    # Calculate price variation
    min_price = min(ratio_prices)
    max_price = max(ratio_prices)
    
    if min_price <= 0:
        return False
        
    price_range_pct = (max_price / min_price - 1) * 100
    
    # If price range is less than threshold, consider it a stablecoin pool
    return price_range_pct < STABLECOIN_THRESHOLD * 100


def optimize_pool_parameters(
    price_data: Dict[str, Any],
    tick_spacing: int,
    token0_decimals: int = BASE_TOKEN_DECIMALS,
    token1_decimals: int = QUOTE_TOKEN_DECIMALS
) -> Dict[str, Any]:
    """
    Optimize liquidity pool parameters based on historical prices.
    
    This function implements a multi-band optimization strategy for 
    concentrated liquidity positions, adaptively handling both 
    stablecoin and non-stablecoin pools.
    
    Args:
        price_data: Historical price data
        tick_spacing: The tick spacing for the pool
        token0_decimals: Number of decimals for token0
        token1_decimals: Number of decimals for token1
        
    Returns:
        Dict with optimized parameters including ticks and economics
    """
    ratio_prices = price_data.get("ratio_prices", [])
    current_price = price_data.get("current_price")
    
    if not ratio_prices or current_price is None:
        logger.error("Missing price data for optimization")
        return {
            "tick_lower": MIN_TICK,
            "tick_upper": MAX_TICK,
            "economic_score": 0.0,
            "bands": {}
        }
    
    # Detect if this is a stablecoin pool
    stablecoin_pool = is_stablecoin_pool(price_data)
    
    # Different strategies for stablecoins vs regular tokens
    if stablecoin_pool:
        logger.info("Detected stablecoin pool - using specialized narrow range strategy")
        return optimize_stablecoin_pool(ratio_prices, current_price, tick_spacing, token0_decimals, token1_decimals)
    else:
        logger.info("Regular token pool - using adaptive multi-band strategy")
        return optimize_regular_pool(ratio_prices, current_price, tick_spacing, token0_decimals, token1_decimals)


def optimize_stablecoin_pool(
    ratio_prices: List[float],
    current_price: float,
    tick_spacing: int,
    token0_decimals: int,
    token1_decimals: int
) -> Dict[str, Any]:
    """
    Optimize parameters specifically for stablecoin pools.
    
    Args:
        ratio_prices: List of historical price ratios
        current_price: Current price ratio
        tick_spacing: The tick spacing for the pool
        token0_decimals: Number of decimals for token0
        token1_decimals: Number of decimals for token1
        
    Returns:
        Dict with optimized parameters for stablecoin pools
    """
    # Use the LRU-cached implementation from stablecoin_model
    # First convert list to tuple for caching
    price_data_tuple = tuple(ratio_prices)
    
    # Call the stablecoin optimization function
    result = optimize_stablecoin_liquidity(
        price_data=price_data_tuple,
        current_price=current_price,
        min_tick_spacing=tick_spacing,
        fee_tier=0.003,  # Default fee tier
        risk_level="low"  # Conservative for stablecoins
    )
    
    # Extract relevant fields to match expected result format
    return {
        "tick_lower": result.get("tick_lower", MIN_TICK),
        "tick_upper": result.get("tick_upper", MAX_TICK),
        "economic_score": 0.9,  # High score for stablecoin optimization
        "price_range_percentage": ((result.get("tick_upper", MAX_TICK) - 
                                   result.get("tick_lower", MIN_TICK)) / 100),
        "stablecoin_model_result": result,
        "bands": {
            "inner": {
                "tick_lower": result.get("tick_lower", MIN_TICK),
                "tick_upper": result.get("tick_upper", MAX_TICK),
                "price_lower": tick_to_price(result.get("tick_lower", MIN_TICK), 
                                            token0_decimals, token1_decimals),
                "price_upper": tick_to_price(result.get("tick_upper", MAX_TICK), 
                                            token0_decimals, token1_decimals),
                "allocation": 1.0
            }
        }
    }


def optimize_regular_pool(
    ratio_prices: List[float],
    current_price: float,
    tick_spacing: int,
    token0_decimals: int,
    token1_decimals: int
) -> Dict[str, Any]:
    """
    Optimize parameters for regular (non-stablecoin) pools.
    
    Args:
        ratio_prices: List of historical price ratios
        current_price: Current price ratio
        tick_spacing: The tick spacing for the pool
        token0_decimals: Number of decimals for token0
        token1_decimals: Number of decimals for token1
        
    Returns:
        Dict with optimized parameters for regular pools
    """
    # Analyze price data
    prices = np.array(ratio_prices)
    min_price = np.min(prices)
    max_price = np.max(prices)
    mean_price = np.mean(prices)
    std_price = np.std(prices)
    
    # Calculate range bounds based on price volatility
    range_width_factor = 2.0  # How many std deviations to include
    
    # Three-band strategy with different ranges
    inner_width = std_price * 0.5  # Narrower inner band
    middle_width = std_price * 1.0  # Middle band
    outer_width = std_price * range_width_factor  # Wider outer band
    
    # Calculate band prices centered on current price
    inner_lower = max(0.0001, current_price - inner_width)
    inner_upper = current_price + inner_width
    
    middle_lower = max(0.0001, current_price - middle_width)
    middle_upper = current_price + middle_width
    
    outer_lower = max(0.0001, current_price - outer_width)
    outer_upper = current_price + outer_width
    
    # Allocations for regular token pools favor middle band
    inner_allocation = 0.2   # 20% allocation to inner band
    middle_allocation = 0.5  # 50% allocation to middle band
    outer_allocation = 0.3   # 30% allocation to outer band
    
    # Convert prices to ticks
    inner_tick_lower = price_to_tick(inner_lower, token0_decimals, token1_decimals)
    inner_tick_upper = price_to_tick(inner_upper, token0_decimals, token1_decimals)
    
    middle_tick_lower = price_to_tick(middle_lower, token0_decimals, token1_decimals)
    middle_tick_upper = price_to_tick(middle_upper, token0_decimals, token1_decimals)
    
    outer_tick_lower = price_to_tick(outer_lower, token0_decimals, token1_decimals)
    outer_tick_upper = price_to_tick(outer_upper, token0_decimals, token1_decimals)
    
    # Round to tick spacing
    inner_tick_lower = round_to_tick_spacing(inner_tick_lower, tick_spacing)
    inner_tick_upper = round_to_tick_spacing(inner_tick_upper, tick_spacing)
    
    middle_tick_lower = round_to_tick_spacing(middle_tick_lower, tick_spacing)
    middle_tick_upper = round_to_tick_spacing(middle_tick_upper, tick_spacing)
    
    outer_tick_lower = round_to_tick_spacing(outer_tick_lower, tick_spacing)
    outer_tick_upper = round_to_tick_spacing(outer_tick_upper, tick_spacing)
    
    # Ensure ticks are within boundaries
    inner_tick_lower = max(MIN_TICK, inner_tick_lower)
    inner_tick_upper = min(MAX_TICK, inner_tick_upper)
    
    middle_tick_lower = max(MIN_TICK, middle_tick_lower)
    middle_tick_upper = min(MAX_TICK, middle_tick_upper)
    
    outer_tick_lower = max(MIN_TICK, outer_tick_lower)
    outer_tick_upper = min(MAX_TICK, outer_tick_upper)
    
    # Calculate price range percentage for reporting
    price_range_pct = (outer_upper / outer_lower - 1) * 100
    
    # Calculate economic score based on how well the range covers historical prices
    price_coverage = sum(1 for p in prices if outer_lower <= p <= outer_upper) / len(prices)
    economic_score = price_coverage * 0.8  # 80% weight on price coverage
    
    return {
        "tick_lower": outer_tick_lower,
        "tick_upper": outer_tick_upper,
        "economic_score": economic_score,
        "price_range_percentage": price_range_pct,
        "bands": {
            "inner": {
                "tick_lower": inner_tick_lower,
                "tick_upper": inner_tick_upper,
                "price_lower": tick_to_price(inner_tick_lower, token0_decimals, token1_decimals),
                "price_upper": tick_to_price(inner_tick_upper, token0_decimals, token1_decimals),
                "allocation": inner_allocation
            },
            "middle": {
                "tick_lower": middle_tick_lower,
                "tick_upper": middle_tick_upper,
                "price_lower": tick_to_price(middle_tick_lower, token0_decimals, token1_decimals),
                "price_upper": tick_to_price(middle_tick_upper, token0_decimals, token1_decimals),
                "allocation": middle_allocation
            },
            "outer": {
                "tick_lower": outer_tick_lower,
                "tick_upper": outer_tick_upper,
                "price_lower": tick_to_price(outer_tick_lower, token0_decimals, token1_decimals),
                "price_upper": tick_to_price(outer_tick_upper, token0_decimals, token1_decimals),
                "allocation": outer_allocation
            }
        }
    }


@lru_cache(maxsize=32)
def calculate_tick_range(
    chain: str,
    token0_address: str,
    token1_address: str,
    tick_spacing: int = 10,
    days: int = DEFAULT_DAYS,
    token0_decimals: int = BASE_TOKEN_DECIMALS,
    token1_decimals: int = QUOTE_TOKEN_DECIMALS,
    verbose: bool = False,
    use_mock_data: bool = False,
) -> Dict[str, Any]:
    """
    Calculate the optimal tick range for a Velodrome concentrated liquidity pool.
    
    This is the main entry point for the model.
    
    Args:
        chain: The blockchain name (e.g., "optimism")
        token0_address: The address of token0
        token1_address: The address of token1
        tick_spacing: The tick spacing for the pool
        days: Number of days of price history to consider
        token0_decimals: Number of decimals for token0
        token1_decimals: Number of decimals for token1
        verbose: Whether to print verbose output
        use_mock_data: Use generated data instead of API calls

    Returns:
        Dict with optimized tick range and band information
    """
    start_time = time.time()
    
    if verbose:
        logger.setLevel(logging.INFO)
        logger.info(f"Calculating optimal tick range for {chain}/{token0_address}/{token1_address}")
    
    # Get historical price data
    price_data = get_pool_token_history(
        chain=chain,
        token0_address=token0_address,
        token1_address=token1_address,
        days=days,
        use_mock_data=use_mock_data
    )
    
    # Run optimization
    result = optimize_pool_parameters(
        price_data=price_data,
        tick_spacing=tick_spacing,
        token0_decimals=token0_decimals,
        token1_decimals=token1_decimals
    )
    
    elapsed = time.time() - start_time
    
    if verbose:
        logger.info(f"Calculation completed in {elapsed:.6f} seconds")
        logger.info(f"Tick range: {result['tick_lower']} to {result['tick_upper']}")
        
        # Log band information
        for band_name, band in result.get("bands", {}).items():
            logger.info(f"{band_name.capitalize()} band:")
            logger.info(f"  Ticks: {band['tick_lower']} to {band['tick_upper']}")
            logger.info(f"  Prices: {band['price_lower']:.6f} to {band['price_upper']:.6f}")
            logger.info(f"  Allocation: {band['allocation'] * 100:.1f}%")
    
    return result


def format_tick_range_result(result: Dict[str, Any], verbose: bool = False) -> str:
    """
    Format the tick range result for human-readable output.
    
    Args:
        result: The result dictionary from calculate_tick_range
        verbose: Whether to include detailed information
        
    Returns:
        Formatted string for display
    """
    output = []
    
    # Basic information
    output.append("VELODROME CONCENTRATED LIQUIDITY TICK RANGE")
    output.append("=" * 50)
    output.append(f"Tick Range: {result['tick_lower']} to {result['tick_upper']}")
    
    if "price_range_percentage" in result:
        output.append(f"Price Range: {result['price_range_percentage']:.2f}%")
    
    if "economic_score" in result:
        output.append(f"Economic Score: {result['economic_score']:.2f}")
    
    if "is_stablecoin" in result:
        output.append(f"Stablecoin Pool: {'Yes' if result['is_stablecoin'] else 'No'}")
    
    # Add band information if available and verbose output is requested
    if verbose and "bands" in result:
        output.append("\nBAND INFORMATION")
        output.append("-" * 50)
        
        for band_name, band in result["bands"].items():
            output.append(f"\n{band_name.capitalize()} Band:")
            output.append(f"  Ticks: {band['tick_lower']} to {band['tick_upper']}")
            
            if "price_lower" in band and "price_upper" in band:
                output.append(f"  Prices: {band['price_lower']:.8f} to {band['price_upper']:.8f}")
            
            if "allocation" in band:
                output.append(f"  Allocation: {band['allocation'] * 100:.1f}%")
    
    # Add stablecoin model result if available and verbose output is requested
    if verbose and "stablecoin_model_result" in result:
        output.append("\nSTABLECOIN MODEL DETAILS")
        output.append("-" * 50)
        
        model_result = result["stablecoin_model_result"]
        if "price_stats" in model_result:
            stats = model_result["price_stats"]
            output.append(f"  Min Price: {stats.get('min', 'N/A'):.8f}")
            output.append(f"  Max Price: {stats.get('max', 'N/A'):.8f}")
            output.append(f"  Median Price: {stats.get('median', 'N/A'):.8f}")
            output.append(f"  Log Mean Price: {stats.get('log_mean', 'N/A'):.8f}")
        
        if "center_price" in model_result:
            output.append(f"  Center Price: {model_result['center_price']:.8f}")
        
        if "center_tick" in model_result:
            output.append(f"  Center Tick: {model_result['center_tick']}")
        
        if "tick_width" in model_result:
            output.append(f"  Tick Width: {model_result['tick_width']}")
    
    return "\n".join(output)


def main():
    """Command line entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate optimal tick range for Velodrome pools")
    parser.add_argument("--chain", type=str, default="optimism", help="Chain name")
    parser.add_argument("--token0", type=str, required=True, help="Token0 address")
    parser.add_argument("--token1", type=str, required=True, help="Token1 address")
    parser.add_argument("--tick-spacing", type=int, default=10, help="Tick spacing")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Days of history")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Run calculation
    result = calculate_tick_range(
        chain=args.chain,
        token0_address=args.token0,
        token1_address=args.token1,
        tick_spacing=args.tick_spacing,
        days=args.days,
        verbose=args.verbose,
        use_mock_data=args.mock
    )
    
    # Print result in a nice format
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main() 
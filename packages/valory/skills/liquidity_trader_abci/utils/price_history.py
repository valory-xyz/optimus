#!/usr/bin/env python3
"""
Bridge module for price history utilities.

This module provides compatibility functions to use the model/price_history.py 
implementation from the existing code in pools/velodrome.py.
"""

import logging
from typing import Dict, List, Any, Optional, Union

from ..model.price_history import (
    get_pool_token_history as model_get_pool_token_history,
    check_is_stablecoin_pool as model_check_is_stablecoin_pool,
    PRICE_VOLATILITY_THRESHOLD
)

# Set up logging
logger = logging.getLogger(__name__)


def get_pool_token_history(
    chain: str, 
    token0_address: str, 
    token1_address: str, 
    days: int = 30,
    use_mock_data: bool = False
) -> Dict[str, Any]:
    """
    Bridge function to get historical price data for a token pair.
    
    Args:
        chain: The blockchain name (e.g., "optimism")
        token0_address: The address of token0
        token1_address: The address of token1
        days: Number of days of history to fetch
        use_mock_data: If True, generate mock data instead of API call
        
    Returns:
        Dict with historical price data and metadata
    """
    try:
        return model_get_pool_token_history(
            chain=chain,
            token0_address=token0_address,
            token1_address=token1_address,
            days=days,
            use_mock_data=use_mock_data
        )
    except Exception as e:
        logger.error(f"Error in get_pool_token_history: {str(e)}")
        # Return sensible default in case of error
        return {
            "token0": {
                "prices": [],
                "dates": []
            },
            "token1": {
                "prices": [],
                "dates": []
            },
            "ratio_prices": [],
            "current_price": 1.0,
            "days": days,
            "is_stablecoin": False,
            "error": str(e)
        }


def get_stablecoin_pair_history(
    chain: str, 
    token0_address: str, 
    token1_address: str, 
    days: int = 30
) -> Dict[str, Any]:
    """
    Get price history for stablecoin pairs with special handling.
    
    Args:
        chain: The blockchain name (e.g., "optimism")
        token0_address: The address of token0
        token1_address: The address of token1
        days: Number of days of history to fetch
        
    Returns:
        Dict with historical price data and metadata
    """
    # Call the main function but set is_stablecoin to True
    result = get_pool_token_history(
        chain=chain,
        token0_address=token0_address,
        token1_address=token1_address,
        days=days,
        use_mock_data=False
    )
    
    # Check if we have price data
    if not result.get("ratio_prices"):
        # Try fallback to mock data
        result = get_pool_token_history(
            chain=chain,
            token0_address=token0_address,
            token1_address=token1_address,
            days=days,
            use_mock_data=True
        )
        result["is_stablecoin"] = True
    
    return result


def check_is_stablecoin_pool(
    token0_prices: List[float], 
    token1_prices: List[float],
    threshold: float = PRICE_VOLATILITY_THRESHOLD
) -> bool:
    """
    Bridge function to check if a pool is a stablecoin pool.
    
    Args:
        token0_prices: Historical prices for token0
        token1_prices: Historical prices for token1
        threshold: Volatility threshold for stablecoin detection
        
    Returns:
        Boolean indicating if this is a stablecoin pool
    """
    if not token0_prices or not token1_prices:
        return False
    
    try:
        return model_check_is_stablecoin_pool(
            token0_prices=token0_prices,
            token1_prices=token1_prices,
            threshold=threshold
        )
    except Exception as e:
        logger.error(f"Error in check_is_stablecoin_pool: {str(e)}")
        # Default to False in case of error
        return False 
#!/usr/bin/env python3
"""Common utility functions for liquidity models."""

import math
from typing import Dict, Any, List, Optional, Tuple, Union

# Constants
MIN_TICK = -887272
MAX_TICK = 887272
BASE_TOKEN_DECIMALS = 18
QUOTE_TOKEN_DECIMALS = 6
Q96 = 2**96
DEFAULT_DAYS = 30
STABLECOIN_THRESHOLD = 0.02  # 2% price variance threshold for stablecoin detection
OPTIMIZATION_CACHE_SIZE = 32  # Default LRU cache size


def sqrtPriceX96_to_price(sqrt_price_x96: int, token0_decimals: int, token1_decimals: int) -> float:
    """Convert sqrtPriceX96 to human-readable price."""
    price = (sqrt_price_x96 / Q96) ** 2
    return price * (10 ** (token0_decimals - token1_decimals))


def price_to_sqrtPriceX96(price: float, token0_decimals: int, token1_decimals: int) -> int:
    """Convert a human-readable price to sqrtPriceX96."""
    adjusted_price = price * (10 ** (token1_decimals - token0_decimals))
    sqrt_price = math.sqrt(adjusted_price)
    return int(sqrt_price * Q96)


def price_to_tick(price: float, token0_decimals: int, token1_decimals: int) -> int:
    """Convert a price to its corresponding tick."""
    adjusted_price = price * (10 ** (token1_decimals - token0_decimals))
    return math.floor(math.log(adjusted_price) / math.log(1.0001))


def tick_to_price(tick: int, token0_decimals: int, token1_decimals: int) -> float:
    """Convert a tick to its corresponding price."""
    price = (1.0001 ** tick)
    return price * (10 ** (token0_decimals - token1_decimals))


def round_to_tick_spacing(tick: int, tick_spacing: int) -> int:
    """Round a tick to the nearest multiple of tick spacing."""
    return math.floor(tick / tick_spacing) * tick_spacing 
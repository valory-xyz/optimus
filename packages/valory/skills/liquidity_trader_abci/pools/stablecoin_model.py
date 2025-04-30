"""This module implements the stablecoin model for optimal band calculation."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

# Import the model implementation
from ..model.velodrome_bridge import (
    calculate_ema,
    calculate_std_dev,
    optimize_stablecoin_bands,
    calculate_tick_range_from_bands_wrapper as calculate_tick_range_from_bands
)


def calculate_zscores(prices: List[float], ema: np.ndarray, std_dev: np.ndarray) -> np.ndarray:
    """
    Calculate Z-scores from price data.
    
    Args:
        prices: Vector of price data
        ema: Vector of EMA values
        std_dev: Vector of standard deviation values
    
    Returns:
        Vector of Z-score values
    """
    prices_array = np.array(prices)
    # Calculate absolute z-scores (distance from EMA in terms of standard deviations)
    z_scores = np.abs(prices_array - ema) / np.maximum(std_dev, 1e-6)  # Avoid division by zero
    return z_scores 
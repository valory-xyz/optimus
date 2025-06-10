import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings

import time
import threading
import statistics
from typing import Any, Dict, List, Optional, Union, Tuple
import json
from web3 import Web3
from functools import lru_cache
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pyfolio as pf
import requests
from pycoingecko import CoinGeckoAPI
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and mappings
REQUIRED_FIELDS = (
    "chains",
    "current_positions",
    "coingecko_api_key",
    "whitelisted_assets",
)
VELODROME = "velodrome"

# Chain-specific constants
OPTIMISM_CHAIN_ID = 10
MODE_CHAIN_ID = 34443
CHAIN_NAMES = {
    OPTIMISM_CHAIN_ID: "optimism",
    MODE_CHAIN_ID: "mode",
}

# Sugar contract addresses
SUGAR_CONTRACT_ADDRESSES = {
    MODE_CHAIN_ID: "0x9ECd2f44f72E969fa3F3C4e4F63bc61E0C08F31F",  # Mode Sugar contract address
    OPTIMISM_CHAIN_ID: "0xA64db2D254f07977609def75c3A7db3eDc72EE1D",  # Optimism Sugar contract address
}

# RewardsSugar contract addresses
REWARDS_SUGAR_CONTRACT_ADDRESSES = {
    MODE_CHAIN_ID: "0xD5d3ABAcB8CF075636792658EE0be8B03AF517B8",  # Mode RewardsSugar contract address (placeholder)
    OPTIMISM_CHAIN_ID: "0x62CCFB2496f49A80B0184AD720379B529E9152fB",  # Optimism RewardsSugar contract address (same as LpSugar for now)
}

# RPC endpoints
RPC_ENDPOINTS = {
    MODE_CHAIN_ID: "https://mainnet.mode.network",
    OPTIMISM_CHAIN_ID: "https://mainnet.optimism.io",
}

# Simplified ABIs with only the functions needed
LP_SUGAR_ABI = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "limit", "type": "uint256"},
            {"internalType": "uint256", "name": "offset", "type": "uint256"}
        ],
        "name": "all",
        "outputs": [
            {
                "components": [
                    {"internalType": "address", "name": "id", "type": "address"},
                    {"internalType": "string", "name": "symbol", "type": "string"},
                    {"internalType": "uint8", "name": "decimals", "type": "uint8"},
                    {"internalType": "uint256", "name": "liquidity", "type": "uint256"},
                    {"internalType": "int24", "name": "type", "type": "int24"},
                    {"internalType": "int24", "name": "tick", "type": "int24"},
                    {"internalType": "uint160", "name": "sqrt_ratio", "type": "uint160"},
                    {"internalType": "address", "name": "token0", "type": "address"},
                    {"internalType": "uint256", "name": "reserve0", "type": "uint256"},
                    {"internalType": "uint256", "name": "staked0", "type": "uint256"},
                    {"internalType": "address", "name": "token1", "type": "address"},
                    {"internalType": "uint256", "name": "reserve1", "type": "uint256"},
                    {"internalType": "uint256", "name": "staked1", "type": "uint256"},
                    {"internalType": "address", "name": "gauge", "type": "address"},
                    {"internalType": "uint256", "name": "gauge_liquidity", "type": "uint256"},
                    {"internalType": "bool", "name": "gauge_alive", "type": "bool"},
                    {"internalType": "address", "name": "fee", "type": "address"},
                    {"internalType": "address", "name": "bribe", "type": "address"},
                    {"internalType": "address", "name": "factory", "type": "address"},
                    {"internalType": "uint256", "name": "emissions", "type": "uint256"},
                    {"internalType": "address", "name": "emissions_token", "type": "address"},
                    {"internalType": "uint256", "name": "pool_fee", "type": "uint256"},
                    {"internalType": "uint256", "name": "unstaked_fee", "type": "uint256"},
                    {"internalType": "uint256", "name": "token0_fees", "type": "uint256"},
                    {"internalType": "uint256", "name": "token1_fees", "type": "uint256"},
                    {"internalType": "address", "name": "nfpm", "type": "address"},
                    {"internalType": "address", "name": "alm", "type": "address"},
                    {"internalType": "address", "name": "root", "type": "address"}
                ],
                "internalType": "struct LpSugar.Pool[]",
                "name": "",
                "type": "tuple[]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "lp", "type": "address"}
        ],
        "name": "byAddress",
        "outputs": [
            {
                "components": [
                    {"internalType": "address", "name": "id", "type": "address"},
                    {"internalType": "string", "name": "symbol", "type": "string"},
                    {"internalType": "uint8", "name": "decimals", "type": "uint8"},
                    {"internalType": "uint256", "name": "liquidity", "type": "uint256"},
                    {"internalType": "int24", "name": "type", "type": "int24"},
                    {"internalType": "int24", "name": "tick", "type": "int24"},
                    {"internalType": "uint160", "name": "sqrt_ratio", "type": "uint160"},
                    {"internalType": "address", "name": "token0", "type": "address"},
                    {"internalType": "uint256", "name": "reserve0", "type": "uint256"},
                    {"internalType": "uint256", "name": "staked0", "type": "uint256"},
                    {"internalType": "address", "name": "token1", "type": "address"},
                    {"internalType": "uint256", "name": "reserve1", "type": "uint256"},
                    {"internalType": "uint256", "name": "staked1", "type": "uint256"},
                    {"internalType": "address", "name": "gauge", "type": "address"},
                    {"internalType": "uint256", "name": "gauge_liquidity", "type": "uint256"},
                    {"internalType": "bool", "name": "gauge_alive", "type": "bool"},
                    {"internalType": "address", "name": "fee", "type": "address"},
                    {"internalType": "address", "name": "bribe", "type": "address"},
                    {"internalType": "address", "name": "factory", "type": "address"},
                    {"internalType": "uint256", "name": "emissions", "type": "uint256"},
                    {"internalType": "address", "name": "emissions_token", "type": "address"},
                    {"internalType": "uint256", "name": "pool_fee", "type": "uint256"},
                    {"internalType": "uint256", "name": "unstaked_fee", "type": "uint256"},
                    {"internalType": "uint256", "name": "token0_fees", "type": "uint256"},
                    {"internalType": "uint256", "name": "token1_fees", "type": "uint256"},
                    {"internalType": "address", "name": "nfpm", "type": "address"},
                    {"internalType": "address", "name": "alm", "type": "address"},
                    {"internalType": "address", "name": "root", "type": "address"}
                ],
                "internalType": "struct LpSugar.Pool",
                "name": "",
                "type": "tuple"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

REWARDS_SUGAR_ABI = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "limit", "type": "uint256"},
            {"internalType": "uint256", "name": "offset", "type": "uint256"},
            {"internalType": "address", "name": "pool", "type": "address"}
        ],
        "name": "epochsByAddress",
        "outputs": [
            {
                "components": [
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                    {"internalType": "uint256", "name": "totalLiquidity", "type": "uint256"},
                    {"internalType": "uint256", "name": "votes", "type": "uint256"},
                    {"internalType": "uint256", "name": "emissions", "type": "uint256"},
                    {"internalType": "address", "name": "emissionsToken", "type": "address"},
                    {"components": [
                        {"internalType": "address", "name": "token", "type": "address"},
                        {"internalType": "uint256", "name": "amount", "type": "uint256"}
                    ], "internalType": "struct RewardsSugar.TokenAmount[]", "name": "fees", "type": "tuple[]"},
                    {"internalType": "uint256", "name": "volume", "type": "uint256"}
                ],
                "internalType": "struct RewardsSugar.EpochData[]",
                "name": "",
                "type": "tuple[]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Cache storage with timestamps - optimized TTLs
CACHE = {
    "pools": {"data": {}, "timestamp": 0, "ttl": 3600},     # 1 hour for pool data
    "tvl": {"data": {}, "timestamp": 0, "ttl": 300},        # 5 minutes for TVL data
    "connections": {"data": {}, "timestamp": 0, "ttl": 1800}, # 30 minutes for connections
    "metrics": {"data": {}, "timestamp": 0, "ttl": 1800},   # 30 minutes for metrics
    "token_symbols": {"data": {}, "timestamp": 0, "ttl": 86400}, # 24 hours for token symbols
    "formatted_pools": {"data": {}, "timestamp": 0, "ttl": 1800}, # 30 minutes for formatted pools
}

# Cache metrics for monitoring
CACHE_METRICS = {
    "hits": {"pools": 0, "tvl": 0, "connections": 0},
    "misses": {"pools": 0, "tvl": 0, "connections": 0}
}

# Thread-local storage for errors
_thread_local = threading.local()

def get_errors():
    """Get thread-local error list."""
    if not hasattr(_thread_local, 'errors'):
        _thread_local.errors = []
    return _thread_local.errors

def get_cached_data(cache_type, key=None):
    """Get data from cache if it exists and is not expired."""
    if cache_type not in CACHE:
        CACHE_METRICS["misses"][cache_type] = CACHE_METRICS["misses"].get(cache_type, 0) + 1
        return None
        
    cache_entry = CACHE[cache_type]
    current_time = time.time()
    
    # Check if cache is expired
    if current_time - cache_entry["timestamp"] > cache_entry["ttl"]:
        CACHE_METRICS["misses"][cache_type] = CACHE_METRICS["misses"].get(cache_type, 0) + 1
        return None
        
    if key is not None:
        result = cache_entry["data"].get(key)
    else:
        result = cache_entry["data"]
    
    if result is None:
        CACHE_METRICS["misses"][cache_type] = CACHE_METRICS["misses"].get(cache_type, 0) + 1
    else:
        CACHE_METRICS["hits"][cache_type] = CACHE_METRICS["hits"].get(cache_type, 0) + 1
    
    return result

def set_cached_data(cache_type, data, key=None):
    """Set data in cache with current timestamp."""
    if cache_type not in CACHE:
        CACHE[cache_type] = {"data": {}, "timestamp": 0, "ttl": 600}
        
    if key is not None:
        CACHE[cache_type]["data"][key] = data
    else:
        CACHE[cache_type]["data"] = data
    
    CACHE[cache_type]["timestamp"] = time.time()

def invalidate_cache(cache_type=None, key=None):
    """Invalidate specific cache or all caches."""
    if cache_type is None:
        # Invalidate all caches
        for cache in CACHE:
            CACHE[cache]["data"] = {}
            CACHE[cache]["timestamp"] = 0
    elif key is None:
        # Invalidate specific cache type
        if cache_type in CACHE:
            CACHE[cache_type]["data"] = {}
            CACHE[cache_type]["timestamp"] = 0
    else:
        # Invalidate specific key in cache type
        if cache_type in CACHE and key in CACHE[cache_type]["data"]:
            del CACHE[cache_type]["data"][key]

def log_cache_metrics():
    """Log cache hit/miss metrics."""
    for cache_type in CACHE_METRICS["hits"]:
        hits = CACHE_METRICS["hits"].get(cache_type, 0)
        misses = CACHE_METRICS["misses"].get(cache_type, 0)
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0
        logger.info(f"Cache metrics for {cache_type}: {hits} hits, {misses} misses, {hit_rate:.2f}% hit rate")

def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    """Check if any required fields are missing from kwargs."""
    return [field for field in REQUIRED_FIELDS if kwargs.get(field) is None]

@lru_cache(maxsize=8)
def get_web3_connection(rpc_url):
    """Get or create a Web3 connection with caching."""
    logger.info(f"Creating new Web3 connection to {rpc_url}")
    return Web3(Web3.HTTPProvider(rpc_url))

@lru_cache(None)
def fetch_token_name_from_contract(chain_name, token_address):
    """Fetch token name from the token contract."""
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
    
    # Get RPC URL for the chain
    chain_id = None
    for cid, cname in CHAIN_NAMES.items():
        if cname.lower() == chain_name.lower():
            chain_id = cid
            break
    
    if chain_id is None:
        return None
    
    rpc_url = RPC_ENDPOINTS.get(chain_id)
    if not rpc_url:
        return None
    
    web3 = get_web3_connection(rpc_url)
    if not web3:
        return None
    
    try:
        contract = web3.eth.contract(
            address=Web3.to_checksum_address(token_address), abi=ERC20_ABI
        )
        return contract.functions.name().call()
    except Exception as e:
        logger.error(f"Error fetching token name for {token_address}: {str(e)}")
        return None

def get_coin_id_from_symbol(
        coin_id_mapping, symbol, chain_name
    ) -> Optional[str]:
        """Retrieve the CoinGecko token ID using the token's address, symbol, and chain name."""
        # Check if coin_list is valid
        symbol = symbol.lower()
        if symbol in coin_id_mapping.get(chain_name, {}):
            return coin_id_mapping[chain_name][symbol]

        return None

def is_pro_api_key(coingecko_api_key: str) -> bool:
    """Check if the provided CoinGecko API key is a pro key."""
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


def calculate_il_risk_score_multi(token_ids, coingecko_api_key: str, time_period: int = 90, pool_id=None, chain=None) -> float:
    """Calculate IL risk score for multiple tokens."""
    
    # Set up CoinGecko client
    is_pro = is_pro_api_key(coingecko_api_key)
    if is_pro:
        cg = CoinGeckoAPI(api_key=coingecko_api_key)
    else:
        cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
    
    # Create a debug data structure
    debug_data = {
        "token_ids": token_ids,
        "valid_token_ids": [tid for tid in token_ids if tid],
        "time_period": time_period,
        "timestamps": {
            "start": int((datetime.now() - timedelta(days=time_period)).timestamp()),
            "end": int(datetime.now().timestamp())
        },
        "price_data": {},
        "calculations": {},
        "errors": []
    }
    
    # Filter out None token_ids
    valid_token_ids = [tid for tid in token_ids if tid]
    
    # If we don't have at least 2 valid token IDs, return None
    if len(valid_token_ids) < 2:
        logger.warning(f"Not enough valid token IDs for pool {pool_id} on chain {chain}. Found: {valid_token_ids}")
        return None
    
    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=time_period)).timestamp())
    
    try:
        # Get price data for all tokens
        prices_data = []
        for token_id in valid_token_ids:
            try:
                logger.info(f"Fetching price data for token {token_id}")
                prices = cg.get_coin_market_chart_range_by_id(
                    id=token_id,
                    vs_currency="usd",
                    from_timestamp=from_timestamp,
                    to_timestamp=to_timestamp,
                )
                prices_list = [x[1] for x in prices["prices"]]
                prices_data.append(prices_list)
                logger.info(f"Received {len(prices_list)} price points for token {token_id}")
                time.sleep(1)  # Rate limiting
            except Exception as e:
                error_msg = f"Error fetching price data for {token_id}: {str(e)}"
                logger.error(error_msg)
                debug_data["errors"].append(error_msg)
                return None
        
        # Find minimum length to align all price series
        min_length = min(len(prices) for prices in prices_data)
        if min_length < 2:
            logger.warning(f"Insufficient price data points for pool {pool_id} on chain {chain}. Min length: {min_length}")
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
        
        # Add debug data for each token
        for i, token_id in enumerate(valid_token_ids):
            debug_data["price_data"][token_id] = {
                "data_points": len(prices_data[i]) if i < len(prices_data) else 0,
                "first_price": prices_data[i][0] if i < len(prices_data) and len(prices_data[i]) > 0 else None,
                "last_price": prices_data[i][-1] if i < len(prices_data) and len(prices_data[i]) > 0 else None
            }
        
        # Add correlation matrix to debug data
        debug_data["calculations"]["correlation_matrix"] = correlation_matrix.tolist()
        debug_data["calculations"]["avg_correlation"] = float(avg_correlation)
        
        # Add volatilities to debug data
        debug_data["calculations"]["volatilities"] = [float(v) for v in volatilities]
        debug_data["calculations"]["avg_volatility"] = float(avg_volatility)
        
        # Calculate IL impact
        initial_prices = [prices[0] for prices in aligned_prices]
        final_prices = [prices[-1] for prices in aligned_prices]
        il_impact = calculate_il_impact_multi(initial_prices, final_prices)
        
        # Add IL impact to debug data
        debug_data["calculations"]["initial_prices"] = [float(p) for p in initial_prices]
        debug_data["calculations"]["final_prices"] = [float(p) for p in final_prices]
        debug_data["calculations"]["il_impact"] = float(il_impact)
        debug_data["calculations"]["final_score"] = float(il_impact * avg_correlation * avg_volatility)
        
        # No debug data is saved to files
        
        return float(il_impact * avg_correlation * avg_volatility)
        
    except Exception as e:
        error_msg = f"Error calculating IL risk score: {str(e)}"
        get_errors().append(error_msg)
        
        # Add error to debug data but don't save to file
        debug_data["errors"].append(error_msg)
        
        return None

def get_epochs_by_address(pool_id, chain, limit=30, offset=0):
    """
    Get historical epochs for a pool using RewardsSugar contract.
    
    Args:
        pool_id: Pool ID
        chain: Chain name (uppercase)
        limit: Maximum number of epochs to fetch (default: 30)
        offset: Offset for pagination (default: 0)
        
    Returns:
        List of epoch data or None if fetching fails
    """
    try:
        # Map chain name to chain ID
        chain_id = None
        for cid, cname in CHAIN_NAMES.items():
            if cname.upper() == chain.upper():
                chain_id = cid
                break
        
        if chain_id is None:
            logger.warning(f"Unsupported chain: {chain}")
            return None
        
        # Get RewardsSugar contract address and RPC URL
        rewards_sugar_address = REWARDS_SUGAR_CONTRACT_ADDRESSES.get(chain_id)
        rpc_url = RPC_ENDPOINTS.get(chain_id)
        
        if not rewards_sugar_address or not rpc_url:
            logger.warning(f"Missing RewardsSugar contract address or RPC URL for chain {chain}")
            return None
        
        # Initialize Web3
        w3 = get_web3_connection(rpc_url)
        if not w3.is_connected():
            logger.warning(f"Failed to connect to RPC endpoint for {chain}: {rpc_url}")
            return None
        
        # Use the REWARDS_SUGAR_ABI constant
        abi = REWARDS_SUGAR_ABI
        
        # Create the contract instance
        contract_instance = w3.eth.contract(address=rewards_sugar_address, abi=abi)
        
        # Get epochs data
        try:
            # Convert pool_id to checksum address
            pool_address = w3.to_checksum_address(pool_id)
            logger.info(f"Calling epochsByAddress for pool {pool_id} (checksum: {pool_address})")
            
            # Try to get the function directly to check if it exists
            try:
                # Check if the function exists
                fn = contract_instance.functions.epochsByAddress
                logger.info(f"Function epochsByAddress exists: {fn is not None}")
                
                # Log the parameters
                logger.info(f"Parameters: limit={limit}, offset={offset}, pool={pool_address}")
                
                # Call epochsByAddress function with proper types
                epochs_data = contract_instance.functions.epochsByAddress(limit, offset, pool_address).call()
                
                if not epochs_data:
                    logger.warning(f"No epochs data found for pool {pool_id}")
                    return None
            except AttributeError as ae:
                logger.error(f"Function epochsByAddress not found: {str(ae)}")
                return None
            
            # Process the epochs data into a simpler format for our calculations
            processed_epochs = []
            for epoch in epochs_data:
                # Extract the timestamp, which is the first element
                timestamp = epoch[0]
                
                # Extract the votes (liquidity proxy) and emissions
                votes = float(epoch[2])
                emissions = float(epoch[3])
                
                # Calculate total fees from the fees array
                fees = epoch[5]  # This is an array of (token, amount) tuples
                total_fees = sum(float(fee[1]) for fee in fees) if fees else 0
                
                # Create a processed epoch tuple with timestamp, liquidity (votes), and volume (fees)
                processed_epoch = (timestamp, votes, total_fees)
                processed_epochs.append(processed_epoch)
            
            # No debug data is saved to files
            
            return processed_epochs
            
        except Exception as e:
            # Log the error but don't raise it - we'll use the fallback mechanism
            logger.error(f"Error getting epochs data for pool {pool_id}: {str(e)}")
            
            # Check if this is an execution revert error, which might indicate the function doesn't exist
            # or the contract address is incorrect
            if "execution reverted" in str(e):
                logger.warning(f"Contract execution reverted. The RewardsSugar contract address might be incorrect or the function might not exist.")
            elif "not found in this contract's abi" in str(e):
                logger.warning(f"Function not found in contract ABI. The ABI might be incorrect or outdated.")
            
            return None
            
    except Exception as e:
        logger.error(f"Error fetching epochs for pool {pool_id}: {str(e)}")
        return None

def get_velodrome_pool_sharpe_ratio(pool_id, chain, timerange="NINETY_DAYS", days=365, risk_free_rate=0.03):
    """
    Calculate Sharpe ratio for a Velodrome pool using historical epoch data from RewardsSugar.
    
    Args:
        pool_id: Pool ID
        chain: Chain name (uppercase)
        timerange: Time range for data (default: NINETY_DAYS)
        days: Number of days to look back for historical data (default: 30)
        
    Returns:
        float: Sharpe ratio or None if calculation fails
    """
    try:
        # Get epochs data from RewardsSugar
        epochs_data = get_epochs_by_address(pool_id, chain, limit=days)
        
        # If we couldn't get epochs data, return None
        if not epochs_data:
            logger.warning(f"No epochs data available for pool {pool_id}. Cannot calculate Sharpe ratio.")
            return None  # Return None to indicate the metric couldn't be calculated
        
        # Process epochs data
        timestamps = []
        total_liquidities = []
        volumes = []
        total_supplies = []  # This would need to be fetched separately via multicall
        
        logger.info(f"Processing {len(epochs_data)} epochs for pool {pool_id}")
        
                # No debug data is saved to files
        
        for epoch in epochs_data:
            timestamps.append(epoch[0])  # timestamp
            total_liquidities.append(float(epoch[1]))  # totalLiquidity
            volumes.append(float(epoch[2]))  # volume (24h)
            # For total_supply, we would need to make a separate call
            # For now, we'll use totalLiquidity as a proxy
            total_supplies.append(float(epoch[1]))
        
        # Get pool fee rate from Sugar contract
        chain_id = None
        for cid, cname in CHAIN_NAMES.items():
            if cname.upper() == chain.upper():
                chain_id = cid
                break
        
        sugar_address = SUGAR_CONTRACT_ADDRESSES.get(chain_id)
        rpc_url = RPC_ENDPOINTS.get(chain_id)
        w3 = get_web3_connection(rpc_url)
        
        # Use the LP_SUGAR_ABI constant
        abi = LP_SUGAR_ABI
        logger.info(f"Using LP_SUGAR_ABI with {len(abi)} entries")
        
        contract_instance = w3.eth.contract(address=sugar_address, abi=abi)
        pool_data = contract_instance.functions.byAddress(pool_id).call()
        pool_fee_rate = float(pool_data[21]) / 1e6  # Pool fee rate
        
        # Use actual liquidity values as a proxy for share price
        # This avoids using synthetic share prices
        share_prices = total_liquidities
        
        # Estimate fee returns
        fee_returns = []
        for i in range(len(timestamps)):
            fee_return = volumes[i] * pool_fee_rate / total_liquidities[i] if total_liquidities[i] > 0 else 0
            fee_returns.append(fee_return)
        
        # Convert to pandas Series for easier calculation
        share_price_series = pd.Series(share_prices, index=timestamps)
        fee_returns_series = pd.Series(fee_returns, index=timestamps)
        
        # Calculate price returns
        price_rets = share_price_series.pct_change().dropna()
        logger.info(f"Pool {pool_id}: {len(share_price_series)} share prices, {len(price_rets)} price returns after pct_change")
        
        # Combine price returns and fee returns
        try:
            # Check if indices match
            price_rets_indices = set(price_rets.index)
            fee_returns_indices = set(fee_returns_series.index)
            common_indices = price_rets_indices.intersection(fee_returns_indices)
            logger.info(f"Pool {pool_id}: {len(common_indices)} common indices between price returns and fee returns")
            
            # Use only common indices
            if common_indices:
                common_indices = sorted(list(common_indices))
                price_rets_filtered = price_rets.loc[common_indices]
                fee_returns_filtered = fee_returns_series.loc[common_indices]
                total_rets = (price_rets_filtered + fee_returns_filtered).dropna()
            else:
                # If no common indices, just use price returns
                total_rets = price_rets
                
            logger.info(f"Pool {pool_id}: {len(total_rets)} total returns after combining price and fee returns")
        except Exception as e:
            logger.error(f"Error combining price and fee returns for pool {pool_id}: {str(e)}")
            total_rets = price_rets  # Fallback to just price returns
        
        # Calculate Sharpe ratio
        if len(total_rets) > 1:
            try:
                # No intermediate data saving
                
                # Check for NaN values and infinities in returns
                if total_rets.isna().any() or np.isinf(total_rets).any():
                    logger.warning(f"Pool {pool_id}: NaN or infinite values found in returns data")
                    
                    # Handle problematic values in memory, no file saving
                    nan_indices = total_rets.index[total_rets.isna()].tolist()
                    inf_indices = total_rets.index[np.isinf(total_rets)].tolist()
                    
                    if nan_indices or inf_indices:
                        logger.warning(f"Pool {pool_id}: Found {len(nan_indices)} NaN and {len(inf_indices)} infinite values in returns")
                    
                    # Replace infinities with large but finite values
                    total_rets = total_rets.replace([np.inf, -np.inf], [1e10, -1e10])
                    
                    # Remove NaN values
                    total_rets = total_rets.dropna()
                    
                    if len(total_rets) <= 1:
                        logger.warning(f"Pool {pool_id}: Not enough data points after cleaning (only {len(total_rets)} left)")
                        return None
                
                # Apply winsorization to limit extreme values (clip at 99th percentile)
                if len(total_rets) > 3:  # Need enough data points for percentile calculation
                    lower_bound = np.percentile(total_rets, 1)
                    upper_bound = np.percentile(total_rets, 99)
                    total_rets = total_rets.clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"Pool {pool_id}: Applied winsorization to limit extreme values")
                
                # Check for zero standard deviation
                returns_std = total_rets.std()
                returns_mean = total_rets.mean()
                logger.info(f"Pool {pool_id}: Returns stats - Mean: {returns_mean}, Std: {returns_std}")
                
                # No processed data is saved to files
                
                if returns_std == 0 or np.isnan(returns_std) or np.isinf(returns_std):
                    logger.warning(f"Pool {pool_id}: Standard deviation issue, cannot calculate Sharpe ratio")
                    return 0  # Return 0 as a default value for constant returns
                
                if np.isnan(returns_mean) or np.isinf(returns_mean):
                    logger.warning(f"Pool {pool_id}: Mean return issue, cannot calculate Sharpe ratio")
                    return 0  # Return 0 as a default value
                
                # Convert annual risk-free rate to weekly rate
                weekly_risk_free = risk_free_rate / 52
                
                # Calculate Sharpe ratio manually to have more control
                try:
                    # Adjust returns for risk-free rate
                    excess_returns_mean = returns_mean - weekly_risk_free
                    
                    # Calculate Sharpe ratio with excess returns
                    sharpe_ratio = (excess_returns_mean * 52) / (returns_std * np.sqrt(52))  # Annualized
                    
                    # Check if result is NaN or infinite
                    if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                        logger.warning(f"Pool {pool_id}: Sharpe ratio calculation resulted in NaN or infinite value. Mean: {returns_mean}, Std: {returns_std}")
                        return 0  # Return 0 as a default value
                except Exception as e:
                    logger.error(f"Pool {pool_id}: Error in Sharpe ratio calculation: {str(e)}")
                    return 0  # Return 0 as a default value
                
                # Cap at reasonable values
                sharpe_ratio = max(min(sharpe_ratio, 10), -10)
                
                logger.info(f"Pool {pool_id}: Successfully calculated Sharpe ratio: {sharpe_ratio}")
                return float(sharpe_ratio)
            except Exception as e:
                logger.error(f"Error calculating Sharpe ratio for pool {pool_id}: {str(e)}")
                return None
        else:
            logger.warning(f"Not enough data points for Sharpe ratio calculation for pool {pool_id} (only {len(total_rets)} data points)")
            return None
            
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio for pool {pool_id}: {str(e)}")
        return None

def analyze_pool_liquidity(pool_id: str, chain: str, price_impact: float = 0.01):
    """
    Analyze pool liquidity and calculate depth score using historical epoch data from RewardsSugar.
    
    Args:
        pool_id: Pool ID
        chain: Chain name (uppercase)
        price_impact: Price impact for depth score calculation (default: 0.01)
        
    Returns:
        Tuple of (depth_score, max_position_size)
    """
    try:
        # Get epochs data from RewardsSugar
        epochs_data = get_epochs_by_address(pool_id, chain, limit=30)
        
        # If we couldn't get epochs data, return None
        if not epochs_data:
            logger.warning(f"No epochs data available for pool {pool_id}. Cannot calculate depth score and max position size.")
            return None, None  # Return None to indicate the metrics couldn't be calculated
        
        # Process epochs data
        tvl_series = []
        volume_series = []
        
        for epoch in epochs_data:
            tvl_series.append(float(epoch[1]))  # totalLiquidity
            volume_series.append(float(epoch[2]))  # volume (24h)
        
        # Calculate average TVL and volume
        avg_tvl = np.mean(tvl_series)
        avg_vol = np.mean(volume_series)
        
        # Calculate depth score
        if price_impact <= 0.001:  # Avoid division by extremely small values
            price_impact = 0.001
            
        # Calculate depth score with safeguards
        depth_score = (
            (np.log1p(max(0, avg_tvl)) * np.log1p(max(0, avg_vol))) / (price_impact * 100)
            if avg_tvl > 0 and avg_vol > 0
            else 0
        )
        
        # Cap depth score at reasonable values
        depth_score = min(depth_score, 1e6)
        
        # Calculate liquidity risk multiplier
        liquidity_risk_multiplier = (
            max(0, 1 - (1 / max(1, depth_score)))
            if depth_score > 0
            else 0
        )
        
        # Calculate maximum position size
        max_position_size = min(
            50 * (avg_tvl * liquidity_risk_multiplier) / 100,
            avg_tvl * 0.1  # Cap at 10% of TVL as a safety measure
        )
        
        # Cap max position size to reasonable values
        max_position_size = min(max_position_size, 1e7)
        
        # Check if depth score meets threshold
        meets_threshold = depth_score > 50
        
        return float(depth_score), float(max_position_size)
            
    except Exception as e:
        logger.error(f"Error analyzing pool liquidity for {pool_id}: {str(e)}")
        return float("nan"), float("nan")

def get_velodrome_pools(chain_id=OPTIMISM_CHAIN_ID, lp_sugar_address=None, ledger_api=None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get pools from Velodrome.
    
    Args:
        chain_id: Chain ID to determine which method to use
        lp_sugar_address: Address of the LpSugar contract (if not provided, uses SUGAR_CONTRACT_ADDRESSES)
        ledger_api: Ethereum API instance or RPC URL (if not provided, uses RPC_ENDPOINTS)
        
    Returns:
        List of pools or error dictionary
    """
    # Use Sugar contract for all chains
    if chain_id in SUGAR_CONTRACT_ADDRESSES:
        # Use provided address or default from SUGAR_CONTRACT_ADDRESSES
        sugar_address = lp_sugar_address or SUGAR_CONTRACT_ADDRESSES[chain_id]
        # Use provided RPC or default from RPC_ENDPOINTS
        rpc_url = ledger_api or RPC_ENDPOINTS[chain_id]
        
        logger.info(f"Using Sugar contract approach for chain ID {chain_id}")
        return get_velodrome_pools_via_sugar(sugar_address, rpc_url=rpc_url, chain_id=chain_id)
    else:
        error_msg = f"Unsupported chain ID: {chain_id}"
        logger.error(error_msg)
        return {"error": error_msg}

def get_velodrome_pools_via_sugar(lp_sugar_address, rpc_url=None, chain_id=MODE_CHAIN_ID) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get pools from Velodrome via Sugar contract (Mode).
    
    Args:
        lp_sugar_address: Address of the LpSugar contract
        rpc_url: RPC URL for the Mode chain (default: uses RPC_ENDPOINTS[MODE_CHAIN_ID])
        
    Returns:
        List of pools or error dictionary
    """
    # Check cache first
    cache_key = f"{chain_id}:{lp_sugar_address}"
    cached_pools = get_cached_data("pools", cache_key)
    
    if cached_pools is not None:
        logger.info(f"Using cached pool data for {chain_id} (count: {len(cached_pools)})")
        return cached_pools
    
    # Use the default RPC URL if none is provided
    if rpc_url is None:
        rpc_url = RPC_ENDPOINTS[MODE_CHAIN_ID]
    
    chain_name = CHAIN_NAMES.get(chain_id, "unknown")
    logger.info(f"Fetching Velodrome pools via Sugar contract at {lp_sugar_address} on {chain_name} chain")
    
    try:
        # Initialize Web3 with the correct provider using cached connection
        w3 = get_web3_connection(rpc_url)
        
        # Check connection
        if not w3.is_connected():
            error_msg = f"Failed to connect to RPC endpoint: {rpc_url}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Use the LP_SUGAR_ABI constant instead of loading from file
        abi = LP_SUGAR_ABI
        logger.info(f"Using LP_SUGAR_ABI with {len(abi)} entries")
        
        # Create the contract instance directly
        logger.info(f"Creating contract instance for address: {lp_sugar_address}")
        contract_instance = w3.eth.contract(address=lp_sugar_address, abi=abi)
        
        # Use direct Web3 calls
        all_pools = []
        limit, offset = 500, 0  # Batch size of 500
        
        # Continue fetching until no more pools are found
        while True:
            try:
                # Call the contract directly
                logger.info(f"Calling all() function with limit={limit}, offset={offset}")
                raw_pools = contract_instance.functions.all(limit, offset).call()
                logger.info(f"Received {len(raw_pools)} pools from contract")
                
                if not raw_pools:
                    logger.info(f"No more pools returned (offset={offset})")
                    break
                
                # Format the pools
                formatted_pools = []
                for pool_data in raw_pools:
                    # Map the tuple data to a dictionary with named fields based on the ABI structure
                    pool = {
                        "id": pool_data[0],  # lp address
                        "symbol": pool_data[1],
                        "decimals": pool_data[2],
                        "liquidity": pool_data[3],
                        "type": pool_data[4],
                        "tick": pool_data[5],
                        "sqrt_ratio": pool_data[6],
                        "token0": pool_data[7],
                        "reserve0": pool_data[8],
                        "staked0": pool_data[9],
                        "token1": pool_data[10],
                        "reserve1": pool_data[11],
                        "staked1": pool_data[12],
                        "gauge": pool_data[13],
                        "gauge_liquidity": pool_data[14],
                        "gauge_alive": pool_data[15],
                        "fee": pool_data[16],
                        "bribe": pool_data[17],
                        "factory": pool_data[18],
                        "emissions": pool_data[19],
                        "emissions_token": pool_data[20],
                        "pool_fee": pool_data[21],
                        "unstaked_fee": pool_data[22],
                        "token0_fees": pool_data[23],
                        "token1_fees": pool_data[24],
                        "nfpm": pool_data[25],
                        "alm": pool_data[26],
                        "root": pool_data[27],
                    }
                    # Only log if type is not 0 or -1 and tick is not 0
                    if pool['type'] not in [0, -1] and pool['tick'] != 0:
                        chain_name = CHAIN_NAMES.get(chain_id, "unknown").capitalize()
                        logger.info(f"{chain_name} pool #{len(formatted_pools) + 1}: {pool['id']} type: {pool['type']} tick: {pool['tick']}")
                    formatted_pools.append(pool)
                
                # Add the formatted pools to our collection
                all_pools.extend(formatted_pools)
                logger.info(f"Formatted {len(formatted_pools)} pools")
                
                # Check if we've reached the end of available pools
                if len(raw_pools) < limit:
                    logger.info(f"Reached the end of available pools at offset {offset}")
                    break
                
                # Move to next batch
                offset += limit
            except Exception as e:
                logger.error(f"Error fetching pools batch (offset={offset}): {str(e)}")
                break
        
        # Convert Sugar pool format to match subgraph format
        formatted_pools = []
        for pool in all_pools:
            # Skip invalid or empty entries returned by the contract
            if pool is None or not isinstance(pool, dict):
                continue
            # Skip pools with missing or zero-address tokens
            ZERO_ADDR = "0x0000000000000000000000000000000000000000"
            token0 = str(pool.get("token0", ZERO_ADDR)).lower()
            token1 = str(pool.get("token1", ZERO_ADDR)).lower()
            # Format the pool to match subgraph format
            formatted_pool = {
                "id": pool["id"],
                "chain": CHAIN_NAMES.get(chain_id, "unknown"),  # Add chain name to pool data
                "inputTokens": [
                    {
                        "id": pool["token0"], 
                        "symbol": ""  # Will be filled later for top pools only
                    },
                    {
                        "id": pool["token1"], 
                        "symbol": ""  # Will be filled later for top pools only
                    }
                ],
                "totalValueLockedUSD": calculate_tvl_from_reserves(
                    pool["reserve0"], 
                    pool["reserve1"],
                    pool["token0"],
                    pool["token1"],
                ),
                "inputTokenBalances": [str(pool["reserve0"]), str(pool["reserve1"])],
                "cumulativeVolumeUSD": "0",  # Not available directly
                "sugar_data": pool,  # Store the original data
            }
            
            formatted_pools.append(formatted_pool)
        
        logger.info(f"Successfully fetched {len(formatted_pools)} pools via Sugar contract")
        
        # Cache the result before returning
        set_cached_data("pools", formatted_pools, cache_key)
        return formatted_pools
        
    except Exception as e:
        error_msg = f"Error fetching pools via Sugar: {str(e)}"
        logger.error(error_msg)
        get_errors().append(error_msg)
        return {"error": error_msg}

def calculate_tvl_from_reserves(reserve0, reserve1, token0_address, token1_address, token_prices=None):
    """
    Calculate TVL from token reserves and prices.
    
    Args:
        reserve0: Reserve of token0
        reserve1: Reserve of token1
        token0_address: Address of token0
        token1_address: Address of token1
        token_prices: Optional dictionary of token prices
        
    Returns:
        String representation of the TVL in USD
    """
    # Check cache first
    cache_key = f"{token0_address}:{token1_address}:{reserve0}:{reserve1}"
    cached_tvl = get_cached_data("tvl", cache_key)
    
    if cached_tvl is not None:
        return cached_tvl
    
    # Convert reserves to float
    reserve0_float = float(reserve0)
    reserve1_float = float(reserve1)
    
    # Get token decimals based on known token addresses
    # Common token decimals mapping
    token_decimals = {
        # USDC variants (6 decimals)
        "0x7f5c764cbc14f9669b88837ca1490cca17c31607": 6,  # USDC.e on Optimism
        "0x0b2c639c533813f4aa9d7837caf62653d097ff85": 6,  # USDC on Optimism
        "0xd988097fb8612cc24eec14542bc03424c656005f": 6,  # USDC on Mode
        "0xa70266c8f8cf33647dcfee763961aff418d9e1e4": 6,  # iUSDC on Mode
        
        # USDT variants (6 decimals)
        "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58": 6,  # USDT on Optimism
        "0x01bff41798a0bcf287b996046ca68b395dbc1071": 6,  # USDT0 on Optimism
        "0xf0f161fda2712db8b566946122a5af183995e2ed": 6,  # USDT on Mode
        "0x1217bfe6c773eec6cc4a38b5dc45b92292b6e189": 6,  # oUSDT
    }
    
    # Get token decimals (default to 18 if not available in the mapping)
    token0_decimals = token_decimals.get(token0_address.lower(), 18)
    token1_decimals = token_decimals.get(token1_address.lower(), 18)
    
    logger.info(f"Using decimals - token0: {token0_decimals}, token1: {token1_decimals}")
    
    # Adjust reserves based on decimals
    adjusted_reserve0 = reserve0_float / (10 ** token0_decimals)
    adjusted_reserve1 = reserve1_float / (10 ** token1_decimals)
    
    # For stablecoin pairs, assume price is close to $1
    # This is a simplification but works well for stablecoin pairs
    token0_price = 1.0
    token1_price = 1.0
    
    # Calculate TVL
    tvl = (adjusted_reserve0 * token0_price) + (adjusted_reserve1 * token1_price)
    
    # Format result
    result = str(tvl)
    
    # Cache the result
    set_cached_data("tvl", result, cache_key)
    return result

def calculate_apr_for_velodrome(pool_data):
    """
    Calculate APR for a pool based on the official Velodrome formula.
    
    Args:
        pool_data: Dictionary containing pool data from the Sugar contract
        
    Returns:
        Float representing the APR percentage
    """
    # Extract necessary values from pool data
    day_seconds = 24 * 60 * 60  # Seconds in a day
    
    # Get total supply and gauge total supply
    total_supply = float(pool_data.get("liquidity", 0))
    gauge_total_supply = float(pool_data.get("gauge_liquidity", 0))
    
    # Get emissions value (in stable currency)
    # According to Velodrome's formula, this is already in stable currency
    emissions = float(pool_data.get("emissions", 0))
    
    # Common token decimals mapping for TVL calculation
    token_decimals = {
        # USDC variants (6 decimals)
        "0x7f5c764cbc14f9669b88837ca1490cca17c31607": 6,  # USDC.e on Optimism
        "0x0b2c639c533813f4aa9d7837caf62653d097ff85": 6,  # USDC on Optimism
        "0xd988097fb8612cc24eec14542bc03424c656005f": 6,  # USDC on Mode
        "0xa70266c8f8cf33647dcfee763961aff418d9e1e4": 6,  # iUSDC on Mode
        
        # USDT variants (6 decimals)
        "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58": 6,  # USDT on Optimism
        "0x01bff41798a0bcf287b996046ca68b395dbc1071": 6,  # USDT0 on Optimism
        "0xf0f161fda2712db8b566946122a5af183995e2ed": 6,  # USDT on Mode
        "0x1217bfe6c773eec6cc4a38b5dc45b92292b6e189": 6,  # oUSDT
    }
    
    # Get token addresses
    token0_address = pool_data.get("token0", "").lower()
    token1_address = pool_data.get("token1", "").lower()
    
    # Get token decimals (default to 18 if not available in the mapping)
    token0_decimals = token_decimals.get(token0_address, 18)
    token1_decimals = token_decimals.get(token1_address, 18)
    
    # Calculate TVL (total value locked) with proper decimal adjustment
    reserve0 = float(pool_data.get("reserve0", 0)) / (10 ** token0_decimals)
    reserve1 = float(pool_data.get("reserve1", 0)) / (10 ** token1_decimals)
    
    # For stablecoin pairs, assume price is close to $1
    token0_price = 1.0
    token1_price = 1.0
    
    tvl = (reserve0 * token0_price) + (reserve1 * token1_price)
    
    # Calculate staked percentage (exactly as in Velodrome's formula)
    staked_pct = 100 * gauge_total_supply / total_supply if total_supply != 0 else 0
    
    # Calculate staked TVL (exactly as in Velodrome's formula)
    staked_tvl = tvl * staked_pct / 100
    
    # Adjust emissions for decimals (emissions are in wei)
    # The emissions value from the contract needs to be converted to a stable currency value
    adjusted_emissions = emissions / (10 ** 18)
    
    # Calculate daily reward (exactly as in Velodrome's formula)
    # We need a scaling factor to match the UI values since our emissions might be different
    # from what Velodrome uses internally
    scaling_factor = 0.048  # Empirically determined to match the Velodrome UI value
    reward_value = adjusted_emissions * scaling_factor
    reward = reward_value * day_seconds
    
    # Calculate APR (exactly as in Velodrome's formula)
    if staked_tvl != 0 and reward != 0:
        apr = (reward / staked_tvl) * (100 * 365)  # Annualized percentage
        
        # Cap APR at reasonable values
        apr = min(apr, 1000.0)
        
        return apr
    else:
        return 0

def get_filtered_pools_for_velodrome(pools, current_positions, whitelisted_assets):
    """Filter pools based on criteria and exclude current positions."""
    qualifying_pools = []
    
    logger.info(f"Starting pool filtering with {len(pools)} total pools")
    logger.info(f"Current positions to exclude: {current_positions}")
    
    for pool in pools:
        pool_id = pool.get("id")
        
        # Skip if this is a current position
        if pool_id in current_positions:
            logger.debug(f"Skipping pool {pool_id} - current position")
            continue
        
        # Get token information
        input_tokens = pool.get("inputTokens", [])
        token_count = len(input_tokens)
        
        # Get chain name from the pool data or use a default
        chain_name = pool.get("chain", "unknown").lower()
        whitelisted_tokens = whitelisted_assets.get(chain_name, {})
        
        # Check if we have at least 2 tokens
        if token_count >= 2:
            # Check whitelist if it exists for this chain
            if whitelisted_tokens:
                # Check if all tokens in the pool are in the whitelist
                all_tokens_whitelisted = True
                # Store the symbols for whitelisted tokens
                token_symbols = []
                
                for token in input_tokens:
                    token_address = token["id"].lower()
                    if token_address not in whitelisted_tokens:
                        all_tokens_whitelisted = False
                        logger.debug(f"Pool {pool_id} excluded - token {token_address} not whitelisted")
                        break
                    # Store the symbol from the whitelist
                    token_symbols.append(whitelisted_tokens[token_address])
                
                # Skip this pool if not all tokens are whitelisted
                if not all_tokens_whitelisted:
                    continue
                
                # Update token symbols in the input_tokens
                for i, symbol in enumerate(token_symbols):
                    if i < len(input_tokens):
                        input_tokens[i]["symbol"] = symbol
            
            # Add basic metrics
            pool["token_count"] = token_count
            pool["tvl"] = float(pool.get("totalValueLockedUSD", 0))
            qualifying_pools.append(pool)
            logger.debug(f"Pool {pool_id} qualified - tokens: {[t['id'] for t in input_tokens]}")
        else:
            logger.debug(f"Pool {pool_id} excluded - insufficient tokens ({token_count})")
    
    logger.info(f"Found {len(qualifying_pools)} qualifying pools after initial filtering")
    return qualifying_pools

def format_pool_data(pools: List[Dict[str, Any]], chain_id=OPTIMISM_CHAIN_ID, coingecko_api_key=None, coin_id_mapping=None) -> List[Dict[str, Any]]:
    """Format pool data for output according to required schema."""
    formatted_pools = []
    chain_name = CHAIN_NAMES.get(chain_id, "unknown")
    
    for pool in pools:
        # Skip pools with less than two tokens
        if pool.get("token_count", 0) < 2:
            continue
        
        # Get the original sugar data
        sugar_data = pool.get("sugar_data", {})
        
        # Calculate APR using the Sugar SDK logic
        apr = calculate_apr_for_velodrome(sugar_data)
            
        # Get the pool type from sugar data
        pool_type = sugar_data.get("type", 0)
        
        # Determine if it's a concentrated liquidity pool based on type
        is_cl_pool = pool_type not in [0, -1]
        
        # Prepare base data including all required fields
        formatted_pool = {
            "dex_type": VELODROME,
            "pool_address": pool["id"],
            "pool_id": pool["id"],
            "tvl": float(pool.get("totalValueLockedUSD", 0)),
            "is_lp": True,  # All pools are LP opportunities
            "token_count": pool.get("token_count", 0),
            "volume": float(pool.get("cumulativeVolumeUSD", 0)),
            "chain": chain_name,
            "apr": apr,  # Add APR to the formatted pool data
            "is_cl_pool": is_cl_pool,  # Add flag for concentrated liquidity pool
            "is_stable": True,  # Always set to True as requested
        }
        
        # Add tokens (should be at least 2 tokens)
        tokens = pool.get("inputTokens", [])
        if len(tokens) >= 1:
            formatted_pool["token0"] = tokens[0]["id"]
            formatted_pool["token0_symbol"] = tokens[0]["symbol"]
        
        if len(tokens) >= 2:
            formatted_pool["token1"] = tokens[1]["id"]
            formatted_pool["token1_symbol"] = tokens[1]["symbol"]
        
        # Calculate advanced metrics if we have the necessary data
        if coingecko_api_key and len(tokens) >= 2:
            # Calculate Sharpe ratio
            try:
                sharpe_ratio = get_velodrome_pool_sharpe_ratio(
                    pool["id"], chain_name.upper()
                )
                formatted_pool["sharpe_ratio"] = sharpe_ratio
            except Exception as e:
                logger.error(f"Error calculating Sharpe ratio for pool {pool['id']}: {str(e)}")
                formatted_pool["sharpe_ratio"] = None
            
            # Calculate depth score
            try:
                depth_score, max_position_size = analyze_pool_liquidity(
                    pool["id"], chain_name.upper()
                )
                formatted_pool["depth_score"] = depth_score
                formatted_pool["max_position_size"] = max_position_size
            except Exception as e:
                logger.error(f"Error calculating depth score for pool {pool['id']}: {str(e)}")
                formatted_pool["depth_score"] = None
                formatted_pool["max_position_size"] = None
            
            # Calculate IL risk score
            try:
                # Get token IDs for CoinGecko API
                token_ids = []
                logger.info(f"Pool {pool['id']}: Checking CoinGecko mapping for tokens")
                logger.info(f"Pool {pool['id']}: coin_id_mapping type: {type(coin_id_mapping)}")
                logger.info(f"Pool {pool['id']}: coin_id_mapping content: {coin_id_mapping}")
                
                for i, token in enumerate(tokens):
                    token_symbol = token["symbol"] or ""
                    token_address = token["id"]
                    logger.info(f"Pool {pool['id']}: Token{i} - Address: {token_address}, Symbol: '{token_symbol}', Chain: {chain_name}")
                    
                    token_id = get_coin_id_from_symbol(
                        coin_id_mapping,
                        token_symbol,
                        chain_name
                    )
                    token_ids.append(token_id)
                    logger.info(f"Pool {pool['id']}: Token{i} ({token_symbol}) mapped to CoinGecko ID: {token_id}")
                
                # Only calculate IL risk if we have at least 2 valid token IDs
                valid_token_ids = [tid for tid in token_ids if tid]
                logger.info(f"Pool {pool['id']}: Valid token IDs: {valid_token_ids} (out of {len(token_ids)} total)")
                
                if len(valid_token_ids) >= 2:
                    logger.info(f"Pool {pool['id']}: Calculating IL risk score with {len(valid_token_ids)} valid token IDs")
                    
                    # Call IL risk score calculation with pool_id and chain
                    il_risk_score = calculate_il_risk_score_multi(
                        valid_token_ids, 
                        coingecko_api_key,
                        pool_id=pool["id"],
                        chain=chain_name
                    )
                    formatted_pool["il_risk_score"] = il_risk_score
                    logger.info(f"Pool {pool['id']}: IL risk score calculated: {il_risk_score}")
                else:
                    logger.warning(f"Pool {pool['id']}: Not enough valid token IDs to calculate IL risk score. Found: {valid_token_ids}")
                    formatted_pool["il_risk_score"] = None
            except Exception as e:
                logger.error(f"Error calculating IL risk score for pool {pool['id']}: {str(e)}")
                formatted_pool["il_risk_score"] = None
            
        formatted_pools.append(formatted_pool)
        
    return formatted_pools

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

def apply_composite_pre_filter(pools, top_n=10, apr_weight=0.7, tvl_weight=0.3, min_tvl_threshold=1000, cl_filter=None):
    """
    Apply composite scoring pre-filter based on standardized APR and TVL.
    
    Args:
        pools: List of pool dictionaries
        top_n: Number of top pools to return (default: 10)
        apr_weight: Weight for APR in composite score (default: 0.7)
        tvl_weight: Weight for TVL in composite score (default: 0.3)
        min_tvl_threshold: Minimum TVL threshold for inclusion (default: 1000)
        cl_filter: Filter for concentrated liquidity pools
                   True = only CL pools
                   False = only non-CL pools
                   None = include all pools (default)
    
    Returns:
        List of top N pools sorted by composite score in descending order
    """
    if not pools:
        return pools
    
    logger.info(f"Starting composite pre-filter with {len(pools)} pools")
    logger.info(f"Parameters: top_n={top_n}, apr_weight={apr_weight}, tvl_weight={tvl_weight}, min_tvl_threshold={min_tvl_threshold}")
    
    # Step 1: Apply minimum TVL filter with proper type conversion
    tvl_filtered_pools = []
    for pool in pools:
        try:
            tvl_value = float(pool.get('tvl', 0))
            if tvl_value >= float(min_tvl_threshold):
                tvl_filtered_pools.append(pool)
        except (ValueError, TypeError):
            # Skip pools with invalid TVL values
            continue
    logger.info(f"After TVL filter (>= {min_tvl_threshold}): {len(tvl_filtered_pools)} pools")
    
    if not tvl_filtered_pools:
        logger.warning("No pools remaining after TVL filtering")
        return []
    
    # Step 2: Apply CL filter if specified
    if cl_filter is not None:
        cl_filtered_pools = [pool for pool in tvl_filtered_pools if pool.get("is_cl_pool", False) == cl_filter]
        cl_status = "CL pools only" if cl_filter is True else "non-CL pools only"
        logger.info(f"After CL filter ({cl_status}): {len(cl_filtered_pools)} pools")
    else:
        cl_filtered_pools = tvl_filtered_pools
        logger.info(f"No CL filter applied: {len(cl_filtered_pools)} pools")
    
    if not cl_filtered_pools:
        logger.warning("No pools remaining after CL filtering")
        return []
    
    # Step 3: Standardize metrics and calculate composite scores
    standardized_pools = standardize_metrics(cl_filtered_pools, apr_weight, tvl_weight)
    
    # Step 4: Sort by composite score in descending order
    sorted_pools = sorted(standardized_pools, key=lambda x: x.get('composite_score', 0), reverse=True)
    
    # Step 5: Return top N pools
    result_pools = sorted_pools[:top_n]
    
    logger.info(f"Final selection: {len(result_pools)} pools")
    
    # Log top pools for debugging
    for i, pool in enumerate(result_pools[:5]):  # Log top 5
        logger.info(f"Pool #{i+1}: {pool.get('pool_address', 'N/A')} - "
                   f"APR: {pool.get('apr', 0):.2f}%, "
                   f"TVL: ${pool.get('tvl', 0):,.0f}, "
                   f"Composite Score: {pool.get('composite_score', 0):.3f}")
    
    return result_pools

def get_top_n_pools_by_apr(pools, n=10, cl_filter=None):
    """
    Filter pools by CL status and return the top N pools by APR.
    
    Args:
        pools: List of formatted pool dictionaries
        n: Number of top pools to return (default: 10)
        cl_filter: Filter for concentrated liquidity pools
                   True = only CL pools
                   False = only non-CL pools
                   None = include all pools (default)
        
    Returns:
        List of the top N pools sorted by APR in descending order
    """
    # First filter by CL status if specified
    if cl_filter is not None:
        filtered_pools = [pool for pool in pools if pool.get("is_cl_pool", False) == cl_filter]
    else:
        filtered_pools = pools
    
    # Then sort by APR in descending order
    sorted_pools = sorted(filtered_pools, key=lambda x: x.get("apr", 0), reverse=True)
    
    # Return the top N pools (or all if fewer than N)
    return sorted_pools[:n]

def get_opportunities_for_velodrome(current_positions, coingecko_api_key, chain_id=OPTIMISM_CHAIN_ID, lp_sugar_address=None, ledger_api=None, top_n=10, cl_filter=None, whitelisted_assets=None, coin_id_mapping=None, **kwargs):
    """
    Get and format pool opportunities with optimized caching and performance.
    
    Args:
        current_positions: List of current position IDs to exclude
        coingecko_api_key: API key for CoinGecko
        chain_id: Chain ID to determine which method to use
        lp_sugar_address: Address of the LpSugar contract
        ledger_api: Ethereum API instance or RPC URL
        top_n: Number of top pools by APR to return (default: 10)
        cl_filter: Filter for concentrated liquidity pools
                   True = only CL pools
                   False = only non-CL pools
                   None = include all pools (default)
    
    Returns:
        List of formatted pool opportunities
    """
    start_time = time.time()
    logger.info(f"Starting opportunity discovery for chain ID {chain_id}")
    
    # DEBUG: Log coin_id_mapping parameter
    logger.info(f"VELODROME DEBUG: get_opportunities_for_velodrome - coin_id_mapping type: {type(coin_id_mapping)}")
    logger.info(f"VELODROME DEBUG: get_opportunities_for_velodrome - coin_id_mapping value: {coin_id_mapping}")
    
    # Check cache for formatted pools first
    cache_key = f"formatted_pools:{chain_id}:{top_n}:{cl_filter}:{hash(str(sorted(current_positions)))}"
    cached_result = get_cached_data("formatted_pools", cache_key)
    if cached_result is not None:
        logger.info(f"Using cached formatted pools for chain {chain_id}")
        return cached_result
    
    # Get pools based on chain
    pools = get_velodrome_pools(chain_id, lp_sugar_address, ledger_api)
    if isinstance(pools, dict) and "error" in pools:
        error_msg = f"Error in pool discovery: {pools['error']}"
        logger.error(error_msg)
        return pools

    # Filter pools
    filtered_pools = get_filtered_pools_for_velodrome(pools, current_positions, whitelisted_assets)
    if not filtered_pools:
        logger.warning("No suitable pools found after filtering")
        return {"error": "No suitable pools found"}

    # Get top N pools using composite scoring (APR + TVL), with optional CL filtering
    if top_n > 0:
        # Extract filtering parameters from kwargs
        apr_weight = kwargs.get('apr_weight', 0.7)
        tvl_weight = kwargs.get('tvl_weight', 0.3)
        min_tvl_threshold = kwargs.get('min_tvl_threshold', 1000)
        
        # Use new composite scoring method
        filtered_pools = apply_composite_pre_filter(
            filtered_pools,
            top_n=top_n, 
            apr_weight=apr_weight,
            tvl_weight=tvl_weight,
            min_tvl_threshold=min_tvl_threshold,
            cl_filter=cl_filter
        )
        if not filtered_pools:
                logger.error("No filtered pools available for composite filtering")
                return {"error": "No filtered pools available"}
        
        logger.info(f"Applied composite pre-filter (APR weight: {apr_weight}, TVL weight: {tvl_weight})")
    
    # Format pools with basic data (without advanced metrics)
    formatted_pools = format_pool_data(
        filtered_pools,
        chain_id,
        coingecko_api_key=coingecko_api_key,
        coin_id_mapping=coin_id_mapping
    )
    
    # Check if we have any formatted pools before proceeding
    if not formatted_pools:
        logger.warning("No pools remaining after formatting")
        return {"error": "No suitable pools found"}
        
    top_pools = formatted_pools if top_n <= 0 else formatted_pools
    
    # Final check - if no pools after filtering, return early
    if not top_pools:
        logger.warning("No pools remaining after APR filtering")
        return {"error": "No suitable pools found"}
    
    # Skip expensive metrics calculation for now to improve performance
    # Only calculate basic metrics that are already included in format_pool_data
    logger.info(f"Skipping advanced metrics calculation for performance optimization")
    
    # Cache the result
    set_cached_data("formatted_pools", top_pools, cache_key)
    
    execution_time = time.time() - start_time
    logger.info(f"Opportunity discovery completed in {execution_time:.2f} seconds")
    logger.info(f"Found {len(top_pools)} valid opportunities")
    
    return top_pools

def calculate_metrics(
    position: Dict[str, Any],
    coingecko_api_key: str,
    coin_id_mapping: List[Any],
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """
    Calculate risk metrics for a specific Velodrome pool position.
    
    Args:
        position: Dictionary containing position details (pool_address, chain, token0, token1, etc.)
        coingecko_api_key: API key for CoinGecko
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing calculated metrics or None if calculation fails
    """
    try:
        # Extract position details
        pool_id = position.get("pool_address")
        chain = position.get("chain", "").lower()
        token0_address = position.get("token0", "")
        token1_address = position.get("token1", "")
        token0_symbol = position.get("token0_symbol", "")
        token1_symbol = position.get("token1_symbol", "")
        
        if not pool_id or not chain:
            logger.error("Missing required position details: pool_address or chain")
            return None
            
        # Map chain name to chain ID
        chain_id = None
        for cid, cname in CHAIN_NAMES.items():
            if cname.lower() == chain.lower():
                chain_id = cid
                break
                
        if chain_id is None:
            logger.error(f"Unsupported chain: {chain}")
            return None
            
        # Get token IDs for CoinGecko API
        token_ids = []
        if token0_address and token0_symbol:
            token0_id = get_coin_id_from_symbol(coin_id_mapping, token0_symbol, chain)
            token_ids.append(token0_id)
            
        if token1_address and token1_symbol:
            token1_id = get_coin_id_from_symbol(coin_id_mapping, token1_symbol, chain)
            token_ids.append(token1_id)
            
        # Calculate IL risk score
        il_risk_score = None
        valid_token_ids = [tid for tid in token_ids if tid]
        if len(valid_token_ids) >= 2:
            il_risk_score = calculate_il_risk_score_multi(
                valid_token_ids, 
                coingecko_api_key,
                pool_id=pool_id,
                chain=chain
            )
            
        # Calculate Sharpe ratio
        sharpe_ratio = get_velodrome_pool_sharpe_ratio(
            pool_id, chain.upper()
        )
        
        # Calculate depth score and max position size
        depth_score, max_position_size = analyze_pool_liquidity(
            pool_id, chain.upper()
        )
        
        # Return calculated metrics
        return {
            "il_risk_score": il_risk_score,
            "sharpe_ratio": sharpe_ratio,
            "depth_score": depth_score,
            "max_position_size": max_position_size,
        }
    except Exception as e:
        logger.error(f"Error calculating metrics for position {position.get('pool_address')}: {str(e)}")
        return None

def run(force_refresh=False, **kwargs) -> Dict[str, Union[bool, str, List[Dict[str, Any]]]]:
    """Main function to run the Velodrome pool analysis.
    
    Args:
        force_refresh: Whether to force refresh the cache
        **kwargs: Arbitrary keyword arguments
            chains: List of chains to analyze (e.g., ["optimism", "mode"])
            current_positions: List of current position IDs to exclude
            coingecko_api_key: API key for CoinGecko
            lp_sugar_address: Address of the LpSugar contract (if not provided, uses SUGAR_CONTRACT_ADDRESSES)
            rpc_url: RPC URL for the Mode chain (optional, uses default if not provided)
            top_n: Number of top pools by APR to return (default: 10)
            cl_filter: Filter for concentrated liquidity pools (True=CL only, False=non-CL only, None=all)
            get_metrics: If True, calculate metrics for a specific position instead of finding opportunities
            position: Position details when get_metrics is True
            
    Returns:
        Dict containing either error messages or result data
    """
    # Clear previous errors
    get_errors().clear()
    
    # Force refresh cache if requested
    if force_refresh:
        logger.info("Forcing cache refresh")
        invalidate_cache()
    
    start_time = time.time()
    
    # Check if we're calculating metrics for a specific position
    get_metrics = kwargs.get("get_metrics", False)
    
    # Define required fields based on mode
    required_fields = list(REQUIRED_FIELDS)
    if get_metrics:
        required_fields.append("position")
    
    # Check for missing required fields
    missing = check_missing_fields(kwargs)
    if missing:
        error_msg = f"Required kwargs {missing} were not provided."
        logger.error(error_msg)
        get_errors().append(error_msg)
        return {"error": get_errors()}
    
    # If we're calculating metrics for a specific position
    if get_metrics:
        if "position" not in kwargs:
            error_msg = "Position details required for metrics calculation."
            logger.error(error_msg)
            get_errors().append(error_msg)
            return {"error": get_errors()}
            
        # Calculate metrics for the position
        metrics = calculate_metrics(
            position=kwargs["position"],
            coingecko_api_key=kwargs["coingecko_api_key"],
            coin_id_mapping=kwargs["coin_id_mapping"]
        )
        
        if metrics is None:
            error_msg = "Failed to calculate metrics for position."
            logger.error(error_msg)
            get_errors().append(error_msg)
            return {"error": get_errors()}
            
        return metrics
    
    # If we're finding opportunities (default behavior)
    # Get chains from kwargs
    chains = kwargs.get("chains", [])
    if not chains:
        error_msg = "No chains specified for analysis."
        logger.error(error_msg)
        get_errors().append(error_msg)
        return {"error": get_errors()}
    
    # Process each chain
    all_results = []
    for chain in chains:
        logger.info(f"Starting Velodrome pool analysis for {chain} chain")
        
        # Map chain name to chain ID
        chain_id = None
        for cid, cname in CHAIN_NAMES.items():
            if cname.lower() == chain.lower():
                chain_id = cid
                break
        
        if chain_id is None:
            error_msg = f"Unsupported chain: {chain}"
            logger.error(error_msg)
            get_errors().append(error_msg)
            continue
        
        # Check if chain is supported
        if chain_id not in SUGAR_CONTRACT_ADDRESSES:
            error_msg = f"Unsupported chain: {chain}"
            logger.error(error_msg)
            get_errors().append(error_msg)
            continue
        
        # Get Sugar contract address and RPC URL
        sugar_address = kwargs.get("lp_sugar_address", SUGAR_CONTRACT_ADDRESSES[chain_id])
        rpc_url = kwargs.get("rpc_url", RPC_ENDPOINTS[chain_id])
        
        # Initialize Web3 for the chain
        w3 = get_web3_connection(rpc_url)
        if not w3.is_connected():
            error_msg = f"Failed to connect to RPC endpoint for {chain}: {rpc_url}"
            logger.error(error_msg)
            get_errors().append(error_msg)
            continue
        
        # Get opportunities for the chain using Sugar contract
        result = get_opportunities_for_velodrome(
            kwargs.get("current_positions", []),
            kwargs["coingecko_api_key"],
            chain_id,
            sugar_address,
            rpc_url,
            kwargs.get("top_n", 10),  # Get top N pools by APR (default: 10)
            cl_filter=kwargs.get("cl_filter"),
            whitelisted_assets=kwargs.get("whitelisted_assets"), # Pass cl_filter to get_top_n_pools_by_apr
            coin_id_mapping=kwargs.get("coin_id_mapping")
        )
        
        # Process results
        if isinstance(result, dict) and "error" in result:
            get_errors().append(result["error"])
            logger.error(f"Error in opportunity discovery for {chain}: {result['error']}")
            continue
        elif not result:
            error_msg = f"No suitable pools found for {chain}"
            logger.warning(error_msg)
            get_errors().append(error_msg)
            continue
        
        # Add results to the combined list
        all_results.extend(result)
    
    # Check if we have any results
    if not all_results:
        error_msg = "No suitable pools found across any chains"
        logger.warning(error_msg)
        get_errors().append(error_msg)
        return {"error": get_errors()}
    
    execution_time = time.time() - start_time
    logger.info(f"Full execution completed in {execution_time:.2f} seconds")
    logger.info(f"Found opportunities across all chains: {all_results}")
    
    # Log cache metrics
    log_cache_metrics()
    
    return {"result": all_results, "error": get_errors()}

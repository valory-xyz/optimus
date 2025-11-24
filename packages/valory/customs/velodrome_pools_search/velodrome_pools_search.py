import warnings

from packages.valory.connections.x402.clients.requests import x402_requests
warnings.filterwarnings("ignore")  # Suppress all warnings

import time
import threading
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from collections import defaultdict
from web3 import Web3
from functools import lru_cache
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pycoingecko import CoinGeckoAPI
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and mappings
REQUIRED_FIELDS = (
    "chains",
    "current_positions",
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

# Configurable filter thresholds
DEFAULT_MIN_TVL_THRESHOLD = 10000.0  
DEFAULT_MIN_VOLUME_THRESHOLD = 0.0 
DEFAULT_MAX_ALLOCATION_PERCENTAGE = 0.02  
MIN_TICK = -887272
MAX_TICK = 887272
DEFAULT_DAYS = 30

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

VELODROME_CL_POOL_ABI = [
    {
        "inputs": [],
        "name": "tickSpacing",
        "outputs": [{"internalType": "int24", "name": "", "type": "int24"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "token0",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "token1",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "slot0",
        "outputs": [
            {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
            {"internalType": "int24", "name": "tick", "type": "int24"},
            {"internalType": "uint16", "name": "observationIndex", "type": "uint16"},
            {"internalType": "uint16", "name": "observationCardinality", "type": "uint16"},
            {"internalType": "uint16", "name": "observationCardinalityNext", "type": "uint16"},
            {"internalType": "bool", "name": "unlocked", "type": "bool"}
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

def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    """Check if any required fields are missing from kwargs."""
    return [field for field in REQUIRED_FIELDS if kwargs.get(field) is None]

@lru_cache(maxsize=8)
def get_web3_connection(rpc_url):
    """Get or create a Web3 connection with caching."""
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


def calculate_velodrome_il_risk_score_multi(token_ids, coingecko_api_key: str, time_period: int = 90, pool_id=None, chain=None, x402_signer=None, x402_proxy=None) -> float:
    """Calculate IL risk score for multiple tokens."""

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
                cg = CoinGeckoAPI()
                if x402_signer is not None and x402_proxy is not None:
                    logger.info("Using x402 signer for CoinGecko API requests")
                    cg.session = x402_requests(account=x402_signer)
                    cg.api_base_url = x402_proxy.rstrip("/") + "/api/v3/"
                else:
                    if not coingecko_api_key:
                        return None
                    is_pro = is_pro_api_key(coingecko_api_key)
                    if is_pro:
                        cg = CoinGeckoAPI(api_key=coingecko_api_key)
                    else:
                        cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)

                prices = cg.get_coin_market_chart_range_by_id(
                    id=token_id,
                    vs_currency="usd",
                    from_timestamp=from_timestamp,
                    to_timestamp=to_timestamp,
                )
                prices_list = [x[1] for x in prices["prices"]]
                prices_data.append(prices_list)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                error_msg = f"Error fetching price data for {token_id}: {str(e)}"
                logger.error(f"[COINGECKO] API Error - {error_msg}")
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
            # Try to get the function directly to check if it exists
            try:
                # Check if the function exists
                fn = contract_instance.functions.epochsByAddress                
                
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
                
                # Extract the votes (liquidity proxy)
                votes = float(epoch[2])
                
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
                
                # No debug data is saved to files
        
        for epoch in epochs_data:
            timestamps.append(epoch[0])  # timestamp
            total_liquidities.append(float(epoch[1]))  # totalLiquidity (votes)
            volumes.append(float(epoch[2]))  # volume (fees)
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
        # Combine price returns and fee returns
        try:
            # Check if indices match
            price_rets_indices = set(price_rets.index)
            fee_returns_indices = set(fee_returns_series.index)
            common_indices = price_rets_indices.intersection(fee_returns_indices)
            
            # Use only common indices
            if common_indices:
                common_indices = sorted(list(common_indices))
                price_rets_filtered = price_rets.loc[common_indices]
                fee_returns_filtered = fee_returns_series.loc[common_indices]
                total_rets = (price_rets_filtered + fee_returns_filtered).dropna()
            else:
                # If no common indices, just use price returns
                total_rets = price_rets
                
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
                
                # Check for zero standard deviation
                returns_std = total_rets.std()
                returns_mean = total_rets.mean()
                
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

def analyze_velodrome_pool_liquidity(pool_id: str, chain: str, price_impact: float = 0.01, max_allocation_percentage: float = DEFAULT_MAX_ALLOCATION_PERCENTAGE):
    """
    Analyze pool liquidity and calculate depth score using historical epoch data from RewardsSugar.
    
    Args:
        pool_id: Pool ID
        chain: Chain name (uppercase)
        price_impact: Price impact for depth score calculation (default: 0.01)
        max_allocation_percentage: Maximum allocation as percentage of pool TVL (default: 0.02 for 2%)
        
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
            tvl_series.append(float(epoch[1]))  # totalLiquidity (votes)
            volume_series.append(float(epoch[2]))  # volume (fees)
        
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
        

        
        # Apply hard allocation cap based on max_allocation_percentage
        hard_max_allocation = avg_tvl * max_allocation_percentage
        max_position_size = hard_max_allocation
        logger.info(f"Pool {pool_id}: Applied {max_allocation_percentage*100}% allocation cap: {hard_max_allocation:.2f}")
        
        # Cap max position size to reasonable values
        max_position_size = min(max_position_size, 1e7)
        
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
        return cached_pools
    
    # Use the default RPC URL if none is provided
    if rpc_url is None:
        rpc_url = RPC_ENDPOINTS[MODE_CHAIN_ID]
    
    chain_name = CHAIN_NAMES.get(chain_id, "unknown")
    
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
        
        # Create the contract instance directly
        contract_instance = w3.eth.contract(address=lp_sugar_address, abi=abi)
        
        # Use direct Web3 calls
        all_pools = []
        limit, offset = 500, 0  # Batch size of 500
        
        # Continue fetching until no more pools are found
        while True:
            try:
                # Call the contract directly
                raw_pools = contract_instance.functions.all(limit, offset).call()
                
                if not raw_pools:
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
                    if pool['tick'] == 0 and pool['type'] in [0,-1]:
                        pool['is_stable'] = True if pool['type'] == 0 else False
                    else:
                        pool['is_stable'] = None
                    formatted_pools.append(pool)
                
                # Add the formatted pools to our collection
                all_pools.extend(formatted_pools)
                
                # Check if we've reached the end of available pools
                if len(raw_pools) < limit:
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

def calculate_position_details_for_velodrome(pool_data, coingecko_api_key: None,x402_signer=None,x402_proxy=None):
    """Calculate APR and other position data for a pool based on the official Velodrome formula"""
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
        advertised_apr = (reward / staked_tvl) * (100 * 365)  # Annualized percentage
        # For CL pools, calculate tick width data
        pool_type = pool_data.get("type", 0)
        is_cl_pool = pool_type not in [0, -1]
       
        if is_cl_pool:
            # Get chain ID from pool data or context            
            try:
                tick_bands = calculate_tick_lower_and_upper_velodrome(
                    chain="optimism",  # Should be passed as parameter
                    pool_address=pool_data.get("id"),
                    is_stable=(pool_type == 0),
                    coingecko_api_key=coingecko_api_key,
                    x402_signer=x402_signer,
                    x402_proxy=x402_proxy
                )
                
                if tick_bands:
                    effective_width = 0
                    for tick_band in tick_bands:
                        if tick_band.get('band_type') == "inner":
                            effective_width = tick_band.get("effective_width")
                    
                    # Calculate adjusted APR
                    if effective_width > 0:
                        apr = advertised_apr/effective_width
                    else:
                        apr = advertised_apr
                    
                    return {
                        "apr": apr,
                        "advertised_apr": advertised_apr,
                        "tick_bands": tick_bands,
                    }
            except Exception as e:
                logger.error(f"Error calculating tick data for CL pool: {str(e)}")
        else:
            return {"apr": advertised_apr}  # Return basic APR for non-CL pools or if calculation fails
    else:
        return {"apr": 0}
    
def calculate_ema(prices: List[float], period: int) -> np.ndarray:
    """Calculate Exponential Moving Average (EMA) from a list of price data points."""
    prices_array = np.array(prices)
    ema = np.zeros_like(prices_array)

    # Initialize with first price
    ema[0] = prices_array[0]
    # Calculate EMA
    alpha = 2 / (period + 1)
    for i in range(1, len(prices_array)):
        ema[i] = prices_array[i] * alpha + ema[i - 1] * (1 - alpha)

    return ema

def calculate_std_dev(prices: List[float], ema: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling standard deviation of price deviations from EMA over a window."""
    prices_array = np.array(prices)
    length = len(prices_array)
    std_dev = np.zeros(length)

    # Calculate rolling standard deviation
    for i in range(window - 1, length):
        window_prices = prices_array[i - window + 1 : i + 1]
        window_ema = ema[i - window + 1 : i + 1]
        deviations = window_prices - window_ema
        std_dev[i] = np.std(deviations)

    # Fill initial values
    for i in range(window - 1):
        if i > 0:
            window_prices = prices_array[: i + 1]
            window_ema = ema[: i + 1]
            deviations = window_prices - window_ema
            std_dev[i] = np.std(deviations)
        else:
            std_dev[i] = 0.001 * prices_array[i]  # Small default value

    return std_dev

def evaluate_band_configuration(
    prices: np.ndarray,
    ema: np.ndarray,
    std_dev: np.ndarray,
    band_multipliers: List[float],
    z_scores: List[float],
    band_allocations: List[float],
    min_width_pct: float,
) -> Dict[str, float]:
    """Evaluate and simulate performance of a specific liquidity band configuration."""
    # Calculate band regions using Z-scores
    # Note: R uses 1-indexed arrays, but Python uses 0-indexed arrays
    # So band_multipliers[0] in Python corresponds to band_multipliers[1] in R
    band1_mask = z_scores <= band_multipliers[0]
    band2_mask = (z_scores > band_multipliers[0]) & (
        z_scores <= band_multipliers[1]
    )
    band3_mask = (z_scores > band_multipliers[1]) & (
        z_scores <= band_multipliers[2]
    )

    # Calculate price coverage by band
    band1_count = np.sum(band1_mask)
    band2_count = np.sum(band2_mask)
    band3_count = np.sum(band3_mask)
    total_count = len(z_scores)

    # Calculate percentage in bounds (all bands combined)
    percent_in_bounds = (
        (band1_count + band2_count + band3_count) / total_count * 100
    )

    # Calculate average weighted width based on allocations and band multipliers
    avg_width_pct = np.mean(
        std_dev / ema * 100
    )  # Base width as percentage of price

    # Calculate weighted width based on band allocation and multipliers
    # This matches the R implementation exactly, accounting for 0-indexing in Python
    b1_width = 2 * band_multipliers[0] * avg_width_pct * band_allocations[0]
    b2_width = (
        2
        * (band_multipliers[1] - band_multipliers[0])
        * avg_width_pct
        * band_allocations[1]
    )
    b3_width = (
        2
        * (band_multipliers[2] - band_multipliers[1])
        * avg_width_pct
        * band_allocations[2]
    )

    # Ensure minimum width
    b1_width = max(b1_width, min_width_pct * band_allocations[0])
    b2_width = max(b2_width, min_width_pct * band_allocations[1])
    b3_width = max(b3_width, min_width_pct * band_allocations[2])

    # Total weighted width
    avg_weighted_width = b1_width + b2_width + b3_width

    # Calculate Z-score weighted coverage
    band1_coverage = band1_count / total_count
    band2_coverage = band2_count / total_count
    band3_coverage = band3_count / total_count

    weighted_coverage = (
        band1_coverage * band_allocations[0]
        + band2_coverage * band_allocations[1]
        + band3_coverage * band_allocations[2]
    )

    # Calculate Z-score economic score
    zscore_economic_score = weighted_coverage * (1 / avg_weighted_width) * 100

    # Return results
    return {
        "percent_in_bounds": percent_in_bounds,
        "avg_weighted_width": avg_weighted_width,
        "zscore_economic_score": zscore_economic_score,
        "band_coverage": [band1_coverage, band2_coverage, band3_coverage],
    }

def run_monte_carlo_level(
    prices: np.ndarray,
    ema: np.ndarray,
    std_dev: np.ndarray,
    z_scores: np.ndarray,
    min_multiplier: float,
    max_multiplier: float,
    num_simulations: int,
    min_width_pct: float,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run Monte Carlo simulations to optimize a single liquidity band level."""
    # Prepare arrays to hold simulation results
    sim_nums = np.arange(num_simulations)
    m1_values = np.zeros(num_simulations)
    m2_values = np.zeros(num_simulations)
    m3_values = np.zeros(num_simulations)
    a1_values = np.zeros(num_simulations)
    a2_values = np.zeros(num_simulations)
    a3_values = np.zeros(num_simulations)
    percent_in_bounds_values = np.zeros(num_simulations)
    avg_weighted_width_values = np.zeros(num_simulations)
    zscore_economic_score_values = np.zeros(num_simulations)

    # Run simulations
    for sim_num in range(num_simulations):
        if verbose and sim_num % 20 == 0:
            logger.info(
                f"Level {min_multiplier:.4f}-{max_multiplier:.4f}: "
                f"Simulation {sim_num+1}/{num_simulations}"
            )

        # Generate random inner band multiplier within range
        m1 = np.random.uniform(min_multiplier, max_multiplier)

        # Apply proportional scaling for middle and outer bands
        m2_min = m1 * 1.5
        m2_max = m1 * 2.5
        m2 = np.random.uniform(m2_min, m2_max)

        m3_min = m2 * 1.3
        m3_max = m2 * 2.0
        m3 = np.random.uniform(m3_min, m3_max)

        # Random allocation with minimum 0.0001 (0.01%) per band
        # Higher probability of selecting high inner band allocation
        if (
            np.random.random() < 0.7
        ):  # 70% chance of selecting from high allocation range
            a1 = np.random.uniform(
                0.95, 0.998
            )  # Focus on high inner band allocations
        else:
            a1 = np.random.uniform(0.5, 0.95)  # Also test lower allocations

        # Distribute remaining allocation
        remaining = 1.0 - a1
        a2_proportion = np.random.uniform(
            0.6, 0.8
        )  # Middle gets 60-80% of remainder
        a2 = max(0.0001, remaining * a2_proportion)
        a3 = 1.0 - a1 - a2

        # Ensure minimum allocation
        if a3 < 0.0001:
            a3 = 0.0001
            a2 = 1.0 - a1 - a3

        # Double-check allocation sum (floating point errors)
        total = a1 + a2 + a3
        if abs(total - 1.0) > 1e-10:
            a1 = a1 / total  # Normalize to ensure exact sum of 1.0
            a2 = a2 / total
            a3 = a3 / total

        # Combine parameters
        band_multipliers = np.array([m1, m2, m3])
        band_allocations = np.array([a1, a2, a3])

        # Evaluate band configuration
        result = evaluate_band_configuration(
            prices,
            ema,
            std_dev,
            band_multipliers,
            z_scores,
            band_allocations,
            min_width_pct,
        )

        # Store results
        m1_values[sim_num] = m1
        m2_values[sim_num] = m2
        m3_values[sim_num] = m3
        a1_values[sim_num] = a1
        a2_values[sim_num] = a2
        a3_values[sim_num] = a3
        percent_in_bounds_values[sim_num] = result["percent_in_bounds"]
        avg_weighted_width_values[sim_num] = result["avg_weighted_width"]
        zscore_economic_score_values[sim_num] = result["zscore_economic_score"]

    # Find the best configuration (highest Z-score economic score)
    best_idx = np.argmax(zscore_economic_score_values)

    # Extract the top configuration
    top_config = {
        "band_multipliers": np.array(
            [m1_values[best_idx], m2_values[best_idx], m3_values[best_idx]]
        ),
        "band_allocations": np.array(
            [a1_values[best_idx], a2_values[best_idx], a3_values[best_idx]]
        ),
        "zscore_economic_score": zscore_economic_score_values[best_idx],
        "percent_in_bounds": percent_in_bounds_values[best_idx],
        "avg_weighted_width": avg_weighted_width_values[best_idx],
    }

    if verbose:
        logger.info(
            f"Best configuration for level {min_multiplier:.4f}-{max_multiplier:.4f}:"
        )
        logger.info(
            f"  Inner Band: {top_config['band_multipliers'][0]:.4f}σ "
            f"({top_config['band_allocations'][0]*100:.1f}% allocation)"
        )
        logger.info(
            f"  Middle Band: {top_config['band_multipliers'][1]:.4f}σ "
            f"({top_config['band_allocations'][1]*100:.1f}% allocation)"
        )
        logger.info(
            f"  Outer Band: {top_config['band_multipliers'][2]:.4f}σ "
            f"({top_config['band_allocations'][2]*100:.1f}% allocation)"
        )
        logger.info(
            f"  Performance: {top_config['percent_in_bounds']:.2f}% in bounds, "
            f"{top_config['avg_weighted_width']:.4f}% weighted width"
        )
        logger.info(
            f"  Z-Score Economic Score: {top_config['zscore_economic_score']:.4f}"
        )

    # Return the best configuration and all results
    return {
        "best_config": top_config,
        "all_results": {
            "sim_nums": sim_nums,
            "m1_values": m1_values,
            "m2_values": m2_values,
            "m3_values": m3_values,
            "a1_values": a1_values,
            "a2_values": a2_values,
            "a3_values": a3_values,
            "percent_in_bounds_values": percent_in_bounds_values,
            "avg_weighted_width_values": avg_weighted_width_values,
            "zscore_economic_score_values": zscore_economic_score_values,
        },
    }

def optimize_stablecoin_bands(
    prices: List[float],
    min_width_pct: float = 0.0001,  # Updated from 0.01 to 0.0001
    ema_period: int = 18,  # Updated from 14 to 18
    std_dev_window: int = 14,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Optimize liquidity band allocation for stablecoin pools based on historical price data."""
    # Calculate EMA and standard deviation
    prices_array = np.array(prices)
    ema = calculate_ema(prices_array, ema_period)
    std_dev = calculate_std_dev(prices_array, ema, std_dev_window)

    # Calculate Z-scores
    z_scores = np.abs(prices_array - ema) / np.maximum(
        std_dev, 1e-6
    )  # Avoid division by zero

    # Define the three recursion levels similar to the R model
    recursion_levels = [
        {
            "name": "Level 1 (Initial)",
            "min_multiplier": 0.1,
            "max_multiplier": 1.5,
            "num_simulations": 250,
            "trigger_threshold": 0.95,  # >95% in inner band
            "trigger_multiplier": 0.15,  # if multiplier < 0.15
        },
        {
            "name": "Level 2 (Narrow)",
            "min_multiplier": 0.01,
            "max_multiplier": 0.2,
            "num_simulations": 300,
            "trigger_threshold": 0.95,  # >95% in inner band
            "trigger_multiplier": 0.02,  # if multiplier < 0.02
        },
        {
            "name": "Level 3 (Ultra-Narrow)",
            "min_multiplier": 0.001,
            "max_multiplier": 0.03,
            "num_simulations": 350,
            "trigger_threshold": None,  # Final level, no trigger
            "trigger_multiplier": None,
        },
    ]

    # Initialize results storage
    results_by_level = []
    best_overall = None
    current_level = 0

    # Run through recursion levels
    while current_level < len(recursion_levels):
        level_config = recursion_levels[current_level]

        if verbose:
            logger.info(
                f"Running {level_config['name']} optimization "
                f"({level_config['min_multiplier']}σ to {level_config['max_multiplier']}σ)"
            )

        # Run Monte Carlo optimization for this level
        level_results = run_monte_carlo_level(
            prices_array,
            ema,
            std_dev,
            z_scores,
            level_config["min_multiplier"],
            level_config["max_multiplier"],
            level_config["num_simulations"],
            min_width_pct,
            verbose,
        )

        # Store this level's results
        results_by_level.append(level_results)

        # Update best overall if needed
        if (
            best_overall is None
            or level_results["best_config"]["zscore_economic_score"]
            > best_overall["best_config"]["zscore_economic_score"]
        ):
            best_overall = {
                "best_config": level_results["best_config"],
                "level": current_level,
                "level_name": level_config["name"],
            }

        # Check if we should move to the next recursion level
        if (
            level_config["trigger_threshold"] is not None
            and level_config["trigger_multiplier"] is not None
        ):
            best_config = level_results["best_config"]

            # Check if inner band allocation exceeds threshold and multiplier is below trigger
            if (
                best_config["band_allocations"][0]
                > level_config["trigger_threshold"]
                and best_config["band_multipliers"][0]
                < level_config["trigger_multiplier"]
            ):
                if verbose:
                    logger.info(
                        ">> Triggering next recursion level <<"
                    )
                    logger.info(
                        f"Inner band allocation: {best_config['band_allocations'][0]*100:.1f}% "
                        f"(threshold: {level_config['trigger_threshold']*100:.1f}%)"
                    )
                    logger.info(
                        f"Inner band multiplier: {best_config['band_multipliers'][0]:.4f}σ "
                        f"(trigger: {level_config['trigger_multiplier']:.4f}σ)"
                    )
                current_level += 1
            else:
                if verbose:
                    logger.info(
                        "Optimal configuration found, no further recursion needed"
                    )
                break  # Exit the recursion if trigger conditions not met
        else:
            # This is the final level, so we're done
            break

    # After all recursion levels, return the best overall result
    if verbose:
        logger.info("BEST OVERALL CONFIGURATION:")
        logger.info(f"From {best_overall['level_name']}")
        logger.info(
            f"Band Multipliers: {best_overall['best_config']['band_multipliers'][0]:.4f}, "
            f"{best_overall['best_config']['band_multipliers'][1]:.4f}, "
            f"{best_overall['best_config']['band_multipliers'][2]:.4f}"
        )
        logger.info(
            f"Band Allocations: {best_overall['best_config']['band_allocations'][0]*100:.1f}%, "
            f"{best_overall['best_config']['band_allocations'][1]*100:.1f}%, "
            f"{best_overall['best_config']['band_allocations'][2]*100:.1f}%"
        )
        logger.info(
            f"Z-Score Economic Score: {best_overall['best_config']['zscore_economic_score']:.4f}"
        )

    # Return the best configuration
    return {
        "band_multipliers": best_overall["best_config"]["band_multipliers"],
        "band_allocations": best_overall["best_config"]["band_allocations"],
        "zscore_economic_score": best_overall["best_config"][
            "zscore_economic_score"
        ],
        "percent_in_bounds": best_overall["best_config"]["percent_in_bounds"],
        "avg_weighted_width": best_overall["best_config"]["avg_weighted_width"],
        "from_level": best_overall["level"],
        "from_level_name": best_overall["level_name"],
    }

def calculate_tick_range_from_bands_wrapper(
    band_multipliers: List[float],
    standard_deviation: float,
    ema: float,
    tick_spacing: int,
    price_to_tick_function: Callable,
    min_tick: int = MIN_TICK,
    max_tick: int = MAX_TICK,
) -> Dict[str, Any]:
    """Convert band multipliers to tick ranges for Velodrome liquidity provision."""
    # Convert band multipliers to price ranges using the formula:
    # Upper bound = EMA + (sigma*multiplier)
    # Lower bound = EMA - (sigma*multiplier)

    # Calculate band price ranges
    band1_lower = ema - (band_multipliers[0] * standard_deviation)
    band1_upper = ema + (band_multipliers[0] * standard_deviation)
    logger.info(
        f"Band 1 price range: lower={band1_lower}, upper={band1_upper}"
    )

    band2_lower = ema - (band_multipliers[1] * standard_deviation)
    band2_upper = ema + (band_multipliers[1] * standard_deviation)
    logger.info(
        f"Band 2 price range: lower={band2_lower}, upper={band2_upper}"
    )

    band3_lower = ema - (band_multipliers[2] * standard_deviation)
    band3_upper = ema + (band_multipliers[2] * standard_deviation)
    logger.info(
        f"Band 3 price range: lower={band3_lower}, upper={band3_upper}"
    )

    # Convert to ticks and round to tick spacing
    def round_to_spacing(tick):
        return int(tick // tick_spacing) * tick_spacing

    # Convert prices to ticks
    band1_tick_lower = round_to_spacing(price_to_tick_function(band1_lower))
    band1_tick_upper = round_to_spacing(price_to_tick_function(band1_upper))
    logger.info(
        f"Band 1 ticks: lower={band1_tick_lower}, upper={band1_tick_upper}"
    )

    band2_tick_lower = round_to_spacing(price_to_tick_function(band2_lower))
    band2_tick_upper = round_to_spacing(price_to_tick_function(band2_upper))
    logger.info(
        f"Band 2 ticks: lower={band2_tick_lower}, upper={band2_tick_upper}"
    )

    band3_tick_lower = round_to_spacing(price_to_tick_function(band3_lower))
    band3_tick_upper = round_to_spacing(price_to_tick_function(band3_upper))
    logger.info(
        f"Band 3 ticks: lower={band3_tick_lower}, upper={band3_tick_upper}"
    )

    # Ensure ticks are within allowed range
    band1_tick_lower = max(min_tick, min(max_tick, band1_tick_lower))
    band1_tick_upper = max(min_tick, min(max_tick, band1_tick_upper))
    logger.info(
        f"Band 1 ticks adjusted: lower={band1_tick_lower}, upper={band1_tick_upper}"
    )

    band2_tick_lower = max(min_tick, min(max_tick, band2_tick_lower))
    band2_tick_upper = max(min_tick, min(max_tick, band2_tick_upper))
    logger.info(
        f"Band 2 ticks adjusted: lower={band2_tick_lower}, upper={band2_tick_upper}"
    )

    band3_tick_lower = max(min_tick, min(max_tick, band3_tick_lower))
    band3_tick_upper = max(min_tick, min(max_tick, band3_tick_upper))
    logger.info(
        f"Band 3 ticks adjusted: lower={band3_tick_lower}, upper={band3_tick_upper}"
    )

    # Build result dictionary
    return {
        "band1": {
            "tick_lower": band1_tick_lower,
            "tick_upper": band1_tick_upper,
        },
        "band2": {
            "tick_lower": band2_tick_lower,
            "tick_upper": band2_tick_upper,
        },
        "band3": {
            "tick_lower": band3_tick_lower,
            "tick_upper": band3_tick_upper,
        },
        "inner_ticks": (band1_tick_lower, band1_tick_upper),
        "middle_ticks": (band2_tick_lower, band2_tick_upper),
        "outer_ticks": (band3_tick_lower, band3_tick_upper),
    }

def get_tick_spacing_velodrome(pool_address: str, chain_id: int) -> Optional[int]:
    """Get velodrome pool tick spacing using VelodromeCLPoolContract"""
    try:
        rpc_url = RPC_ENDPOINTS.get(chain_id)
        if not rpc_url:
            logger.error(f"No RPC URL found for chain {chain_id}")
            return None
            
        web3 = get_web3_connection(rpc_url)
        if not web3.is_connected():
            logger.error(f"Failed to connect to RPC endpoint: {rpc_url}")
            return None

        # Create contract instance using VelodromeCLPool ABI
        contract = web3.eth.contract(
            address=Web3.to_checksum_address(pool_address),
            abi=VELODROME_CL_POOL_ABI
        )

        # Call tickSpacing function
        tick_spacing = contract.functions.tickSpacing().call()

        if not tick_spacing:
            logger.error(f"Could not fetch tick spacing for velodrome pool {pool_address}")
            return None

        logger.info(f"Tick spacing for velodrome pool {pool_address}: {tick_spacing}")
        return tick_spacing
        
    except Exception as e:
        logger.error(f"Error fetching tick spacing for pool {pool_address}: {str(e)}")
        return None

def get_pool_tokens(pool_address: str, chain_id: int) -> Optional[Tuple[str, str]]:
    """
    STEP 2: Get the token addresses from a Velodrome pool using VelodromeCLPoolContract
    Exact implementation from velodrome.py adapted for standalone use
    """
    try:
        rpc_url = RPC_ENDPOINTS.get(chain_id)
        if not rpc_url:
            logger.error(f"No RPC URL found for chain {chain_id}")
            return None
            
        web3 = get_web3_connection(rpc_url)
        if not web3.is_connected():
            logger.error(f"Failed to connect to RPC endpoint: {rpc_url}")
            return None

        # Create contract instance using VelodromeCLPool ABI
        contract = web3.eth.contract(
            address=Web3.to_checksum_address(pool_address),
            abi=VELODROME_CL_POOL_ABI
        )

        # Call token0 and token1 functions
        token0 = contract.functions.token0().call()
        token1 = contract.functions.token1().call()

        if not token0 or not token1:
            logger.error(f"Could not get token addresses for pool {pool_address}")
            return None

        logger.info(f"Retrieved tokens for pool {pool_address}: {token0}, {token1}")
        return token0, token1

    except Exception as e:
        logger.error(f"Error getting pool tokens: {str(e)}")
        return None

def get_current_pool_price(pool_address: str, chain_id: int) -> Optional[float]:
    """Get the current price from a Velodrome concentrated liquidity pool"""
    try:
        rpc_url = RPC_ENDPOINTS.get(chain_id)
        if not rpc_url:
            logger.error(f"No RPC URL found for chain {chain_id}")
            return None
            
        web3 = get_web3_connection(rpc_url)
        if not web3.is_connected():
            logger.error(f"Failed to connect to RPC endpoint: {rpc_url}")
            return None

        # Create contract instance using VelodromeCLPool ABI
        contract = web3.eth.contract(
            address=Web3.to_checksum_address(pool_address),
            abi=VELODROME_CL_POOL_ABI
        )

        # Use the slot0 function to get pool state including sqrt_price_x96
        slot0_data = contract.functions.slot0().call()

        if slot0_data is None:
            logger.error(f"Could not get slot0 data for pool {pool_address}")
            return None

        # Extract sqrt_price_x96 from slot0 result (first element)
        sqrt_price_x96 = slot0_data[0]
        
        if sqrt_price_x96 == 0:
            logger.error(f"Invalid sqrt_price_x96 for pool {pool_address}")
            return None

        # Convert sqrt_price_x96 to price
        # The formula is: price = (sqrt_price_x96 / 2^96)^2
        price = (sqrt_price_x96 / (2**96)) ** 2
        logger.info(f"Current pool price: {price}")
        return price
        
    except Exception as e:
        logger.error(f"Error getting current pool price: {str(e)}")
        return None

def get_coin_id_from_address(
    chain: str, 
    address: str, 
    platform: str,
    coingecko_api_key: Optional[str] = None,
    x402_signer=None,
    x402_proxy=None
) -> Optional[str]:
    """Retrieve CoinGecko coin ID"""
    try:
        # Check stablecoin mappings first
        stablecoin_mappings = {
            "0x0b2c639c533813f4aa9d7837caf62653d097ff85": "usd-coin",
            "0xcb8fa9a76b8e203d8c3797bf438d8fb81ea3326a": "alchemix-usd",
            "0x01bff41798a0bcf287b996046ca68b395dbc1071": "usdt0",
            "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58": "bridged-usdt",
            "0x9dabae7274d28a45f0b65bf8ed201a5731492ca0": None,
            "0x7f5c764cbc14f9669b88837ca1490cca17c31607": "bridged-usd-coin-optimism",
            "0xbfd291da8a403daaf7e5e9dc1ec0aceacd4848b9": "token-dforce-usd",
            "0x8ae125e8653821e851f12a49f7765db9a9ce7384": "dola-usd",
            "0xc40f949f8a4e094d1b49a23ea9241d289b7b2819": "liquity-usd",
            "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1": "makerdao-optimism-bridged-dai-optimism",
            "0x087c440f251ff6cfe62b86dde1be558b95b4bb9b": "liquity-bold",
            "0x2e3d870790dc77a83dd1d18184acc7439a53f475": "frax",
            "0x2218a117083f5b482b0bb821d27056ba9c04b1d3": "savings-dai",
            "0x73cb180bf0521828d8849bc8cf2b920918e23032": "overnight-fi-usd-optimism",
            "0x1217bfe6c773eec6cc4a38b5dc45b92292b6e189": "openusdt",
            "0x4f604735c1cf31399c6e711d5962b2b3e0225ad3": "glo-dollar",
            "0xd988097fb8612cc24eec14542bc03424c656005f": "mode-bridged-usdc-mode",
            "0x3f51c6c5927b88cdec4b61e2787f9bd0f5249138": None,
            "0xf0f161fda2712db8b566946122a5af183995e2ed": "mode-bridged-usdt-mode",
            "0x1217bfe6c773eec6cc4a38b5dc45b92292b6e189": "openusdt",
        }

        if address.lower() in stablecoin_mappings:
            coin_id = stablecoin_mappings[address.lower()]
            logger.info(f"Using stablecoin mapping for {address}: {coin_id}")
            return coin_id

        # Rate limiting
        time.sleep(1)

        # Use x402 requests if available
        if x402_signer is not None and x402_proxy is not None:
            logger.info("Using x402 signer for CoinGecko API requests")
            try:
                
                # Create CoinGecko instance with x402
                cg = CoinGeckoAPI()
                cg.session = x402_requests(account=x402_signer)
                cg.api_base_url = x402_proxy.rstrip("/") + "/api/v3/"
                
                # Make the request using x402
                endpoint = f"coins/{platform}/contract/{address}"
                response = cg.session.get(f"{cg.api_base_url}{endpoint}")
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("id")
                else:
                    logger.warning(f"Failed to get coin ID via x402 for {chain}/{address}: {response.status_code}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error using x402 for coin ID lookup: {str(e)}")
                return None
        else:
            # Fallback to regular API
            if not coingecko_api_key:
                logger.warning("No CoinGecko API key provided and no x402 available")
                return None
                
            try:
                # Determine if it's a pro key
                is_pro = is_pro_api_key(coingecko_api_key)
                if is_pro:
                    cg = CoinGeckoAPI(api_key=coingecko_api_key)
                else:
                    cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
                
                # Make the request
                response = cg.get_coin_info_from_contract_address_by_id(platform, address)
                return response.get("id") if response else None
                
            except Exception as e:
                logger.error(f"Error getting coin ID via regular API: {str(e)}")
                return None

    except Exception as e:
        logger.warning(f"Error getting coin ID for {chain}/{address}: {str(e)}")
        return None

def get_historical_market_data(
    coin_id: str, 
    days: int,
    coingecko_api_key: Optional[str] = None,
    x402_signer=None,
    x402_proxy=None
) -> Optional[Dict[str, Any]]:
    """Get historical market data using x402 requests when available"""
    try:
        # Rate limiting
        time.sleep(2)

        # Use x402 requests if available
        if x402_signer is not None and x402_proxy is not None:
            logger.info("Using x402 signer for CoinGecko API requests")
            try:                
                # Create CoinGecko instance with x402
                cg = CoinGeckoAPI()
                cg.session = x402_requests(account=x402_signer)
                cg.api_base_url = x402_proxy.rstrip("/") + "/api/v3/"
                
                # Make the request using x402
                response = cg.get_coin_market_chart_by_id(
                    id=coin_id,
                    vs_currency="usd",
                    days=days
                )
                
                if response:
                    prices_data = response.get("prices", [])
                    timestamps = [entry[0] / 1000 for entry in prices_data]  # ms to seconds
                    prices = [entry[1] for entry in prices_data]
                    return {
                        "coin_id": coin_id,
                        "timestamps": timestamps,
                        "prices": prices,
                        "days": days,
                        "last_updated": time.time(),
                    }
                else:
                    logger.warning(f"Failed to fetch market data via x402 for {coin_id}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error using x402 for market data: {str(e)}")
                return None
        else:
            # Fallback to regular API
            if not coingecko_api_key:
                logger.warning("No CoinGecko API key provided and no x402 available")
                return None
                
            try:
                # Determine if it's a pro key
                is_pro = is_pro_api_key(coingecko_api_key)
                if is_pro:
                    cg = CoinGeckoAPI(api_key=coingecko_api_key)
                else:
                    cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
                
                # Make the request
                response = cg.get_coin_market_chart_by_id(
                    id=coin_id,
                    vs_currency="usd", 
                    days=days
                )
                
                if response:
                    prices_data = response.get("prices", [])
                    timestamps = [entry[0] / 1000 for entry in prices_data]  # ms to seconds
                    prices = [entry[1] for entry in prices_data]
                    return {
                        "coin_id": coin_id,
                        "timestamps": timestamps,
                        "prices": prices,
                        "days": days,
                        "last_updated": time.time(),
                    }
                else:
                    logger.warning(f"Failed to fetch market data for {coin_id}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error getting market data via regular API: {str(e)}")
                return None

    except Exception as e:
        logger.warning(f"Error getting market data for {coin_id}: {str(e)}")
        return None

def get_pool_token_history(
    chain: str,
    token0_address: str,
    token1_address: str,
    days: int = 30,
    coingecko_api_key: Optional[str] = None,
    x402_signer=None,
    x402_proxy=None
) -> Optional[Dict[str, Any]]:
    """Fetch historical price data for a token pair"""
    logger.info(f"Fetching historical price data for tokens: {token0_address} and {token1_address}")

    try:
        # Convert chain name to CoinGecko platform ID
        platform_map = {
            "ethereum": "ethereum",
            "optimism": "optimistic-ethereum",
            "arbitrum": "arbitrum-one",
            "polygon": "polygon-pos",
            "base": "base",
            "mode": "mode",
        }

        platform = platform_map.get(chain.lower())
        if not platform:
            logger.warning(f"Unsupported chain: {chain}")
            return None

        # Get coin IDs for both tokens
        token0_id = get_coin_id_from_address(
            chain, token0_address, platform, coingecko_api_key, x402_signer, x402_proxy
        )
        token1_id = get_coin_id_from_address(
            chain, token1_address, platform, coingecko_api_key, x402_signer, x402_proxy
        )

        if not token0_id or not token1_id:
            logger.warning(f"Could not find coin IDs for tokens: {token0_address}, {token1_address}")
            return None

        # Get price history for both tokens
        token0_data = get_historical_market_data(
            token0_id, days, coingecko_api_key, x402_signer, x402_proxy
        )
        token1_data = get_historical_market_data(
            token1_id, days, coingecko_api_key, x402_signer, x402_proxy
        )

        # Check if we have valid price data
        if (
            not token0_data
            or not token1_data
            or not token0_data.get("prices")
            or not token1_data.get("prices")
        ):
            logger.warning("Missing price data for one or both tokens")
            return None

        # Combine data and calculate price ratios
        token0_prices = token0_data.get("prices", [])
        token1_prices = token1_data.get("prices", [])

        # Ensure price lists are the same length by taking the shorter one
        min_length = min(len(token0_prices), len(token1_prices))
        token0_prices = token0_prices[:min_length]
        token1_prices = token1_prices[:min_length]

        # Calculate ratio prices (token1/token0) - exact formula from velodrome.py
        ratio_prices = []
        for i in range(min_length):
            if token0_prices[i] and token0_prices[i] > 0:
                ratio_prices.append(token1_prices[i] / token0_prices[i])

        # Get current price (latest)
        current_price = ratio_prices[-1] if ratio_prices else 1.0

        return {
            "ratio_prices": ratio_prices,
            "current_price": current_price,
            "days": days,
        }
        
    except Exception as e:
        logger.error(f"Error getting pool token history: {str(e)}")
        return None

def calculate_tick_lower_and_upper_velodrome(
    chain: str, 
    pool_address: str, 
    is_stable: bool,
    coingecko_api_key=None, 
    x402_signer=None, 
    x402_proxy=None,
) -> Optional[List[Dict[str, Any]]]:
    """Calculate optimal tick ranges for a Velodrome Concentrated position based on pool features."""
    logger.info(
        f"Calculating tick ranges using stablecoin model for pool {pool_address}"
    )

       # Map chain name to chain ID
    chain_id = None
    for cid, cname in CHAIN_NAMES.items():
        if cname.lower() == chain.lower():
            chain_id = cid
            break
    
    if chain_id is None:
        logger.error(f"Unsupported chain: {chain}")
        return None

    try:
        # 1. Fetch tick spacing from velodrome cl pool
        tick_spacing = get_tick_spacing_velodrome(pool_address, chain_id)
        if not tick_spacing:
            logger.error(f"Failed to get tick spacing for pool {pool_address}")
            return None

        # 2. Get the token addresses from the pool
        token0, token1 = get_pool_tokens(pool_address, chain_id)
        if not token0 or not token1:
            logger.error(f"Failed to get tokens for pool {pool_address}")
            return None

        # 3. Get current price
        current_price = get_current_pool_price(pool_address, chain_id)
        if current_price is None:
            logger.error(f"Failed to get current price for pool {pool_address}")
            return None

        # 4. Get historical price data for both tokens and calculate price ratio history
        logger.info(f"Fetching historical price data for tokens: {token0} and {token1}")
        try:
            pool_data = get_pool_token_history(
                chain=chain, 
                token0_address=token0, 
                token1_address=token1,
                coingecko_api_key=coingecko_api_key,
                x402_signer=x402_signer,
                x402_proxy=x402_proxy
            )

            # Check if we have valid pool data
            if pool_data is None:
                logger.error(f"Could not get pool token history for {token0} and {token1}. Aborting operation.")
                return None

            # Check if we have price data
            ratio_prices = pool_data.get("ratio_prices", [])

            if not ratio_prices:
                logger.error(f"Could not get price ratio history for pool {pool_address}. Aborting operation.")
                return None
        except Exception as e:
            logger.error(f"Error fetching historical price data: {str(e)}")
            return None

        # 4. Use stablecoin model to optimize bands
        # For stablecoin pools, we want narrow ranges
        # For volatile pools, we might adjust parameters
        model_params = {
            "ema_period": 18,  # Updated from 10 to 18
            "std_dev_window": 100,  # Default from the model
            "verbose": True,
        }

        # For stablecoin pools, we can use more aggressive settings
        if is_stable:
            model_params["min_width_pct"] = 0.0001  # Updated from 0.00001 to 0.0001

        # Run the optimization
        result = optimize_stablecoin_bands(prices=ratio_prices, **model_params)
        logger.info(f"Result from models: {result}")

        if not result:
            logger.error("Error in stablecoin model calculation")
            return None

        # 7. Calculate standard deviation for current window
        ema = calculate_ema(ratio_prices[-100:], model_params["ema_period"])
        std_dev = calculate_std_dev(
            ratio_prices[-100:], ema, model_params["std_dev_window"]
        )
        current_std_dev = std_dev[-1]  # Use the most recent standard deviation

        logger.info(f"EMA: {ema} Current_Std_Dev: {current_std_dev}")

        # 8. Calculate tick range using model band multipliers
        band_multipliers = result["band_multipliers"]

        # Get the most recent EMA value (human price space, ~1.0 for stables)
        current_ema = ema[-1]

        # Derive current tick from the pool's raw price (already decimals-adjusted by the AMM)
        # This anchors band ticks around the actual pool price instead of near zero
        tick_current = int(np.log(current_price) / np.log(1.0001))

        # Define a converter that maps a human price to a tick relative to tick_current
        def price_to_tick(price: float) -> int:
            ratio = price / current_ema if current_ema > 0 else 1.0
            delta = np.log(ratio) / np.log(1.0001)
            return int(np.rint(delta) + tick_current)

        # Calculate tick range using the exact formula: Upper bound = EMA + (sigma*multiplier)
        tick_range_results = calculate_tick_range_from_bands_wrapper(
            band_multipliers=band_multipliers,
            standard_deviation=current_std_dev,
            ema=current_ema,  # Use EMA as the center in human price space
            tick_spacing=tick_spacing,
            price_to_tick_function=price_to_tick,
            min_tick=MIN_TICK,
            max_tick=MAX_TICK,
        )

        # Prepare positions data for all three bands
        positions = []
        band_allocations = result["band_allocations"]
        effective_width = 0
        for i, _band_name in enumerate(["inner", "middle", "outer"]):
            band_data = tick_range_results[f"band{i+1}"]
            tick_lower = band_data["tick_lower"]
            tick_upper = band_data["tick_upper"]
            effective_width = abs(tick_upper - tick_lower)


            positions.append(
                {
                    "tick_lower": tick_lower,
                    "tick_upper": tick_upper,
                    "allocation": band_allocations[i],
                    "effective_width": effective_width,
                    "band_type": _band_name
                }
            )

        for p in positions:
            if p["tick_lower"] == p["tick_upper"]:
                logger.info(
                    f"Adjusting position with equal ticks: tick_lower={p['tick_lower']}, tick_upper={p['tick_upper']}. "
                    f"Setting tick_upper to {p['tick_lower'] + tick_spacing}."
                )
                p["tick_upper"] = p["tick_lower"] + tick_spacing
                logger.info(
                    f"Adjusted position: tick_lower={p['tick_lower']}, tick_upper={p['tick_upper']}."
                )

        tick_to_band = defaultdict(
            lambda: {"tick_lower": None, "tick_upper": None, "allocation": 0.0}
        )

        for p in positions:
            key = (p["tick_lower"], p["tick_upper"])
            if tick_to_band[key]["tick_lower"] is None:
                tick_to_band[key]["tick_lower"] = p["tick_lower"]
                tick_to_band[key]["tick_upper"] = p["tick_upper"]
            tick_to_band[key]["allocation"] += p["allocation"]

        collapsed_positions = [
            {
                "tick_lower": v["tick_lower"],
                "tick_upper": v["tick_upper"],
                "allocation": v["allocation"],
            }
            for v in tick_to_band.values()
        ]

        logger.info(
            f"Collapsed positions before normalization: {collapsed_positions}"
        )
        total_alloc = sum(p["allocation"] for p in collapsed_positions)
        logger.info(
            f"Total allocation before normalization: {total_alloc}"
        )

        if total_alloc > 0:
            for p in collapsed_positions:
                p["allocation"] /= total_alloc
                logger.info(
                    f"Normalized allocation for position with ticks ({p['tick_lower']}, {p['tick_upper']}): {p['allocation']:.1%}"
                )

        positions = collapsed_positions
        logger.info(
            f"Final positions after normalization: {positions}"
        )
        logger.info("Band details (inner, middle, outer):")

        for i, position in enumerate(positions):
            band_name = ["inner", "middle", "outer"][i] if i < 3 else f"band_{i}"
            logger.info(
                f"  {band_name.upper()}: ticks=({position['tick_lower']}, {position['tick_upper']}), "
                f"allocation={position['allocation']:.1%}"
            )

        logger.info(
            f"Model band multipliers: {band_multipliers[0]:.4f}, "
            f"{band_multipliers[1]:.4f}, {band_multipliers[2]:.4f}"
        )

        # Extract percent_in_bounds from the optimization result
        percent_in_bounds = result.get("percent_in_bounds", 1.0)
        logger.info(f"Percent in bounds: {percent_in_bounds:.2f}%")

        # Add percent_in_bounds to each position
        for position in positions:
            position["percent_in_bounds"] = percent_in_bounds
            # Also store EMA and std_dev metadata for caching
            position["ema"] = ema.tolist() if hasattr(ema, "tolist") else list(ema)
            position["std_dev"] = (
                std_dev.tolist() if hasattr(std_dev, "tolist") else list(std_dev)
            )
            position["current_ema"] = float(current_ema)
            position["current_std_dev"] = float(current_std_dev)
            position["band_multipliers"] = (
                band_multipliers.tolist()
                if hasattr(band_multipliers, "tolist")
                else list(band_multipliers)
            )

        return positions

    except Exception as e:
        logger.error(
            f"Error in stablecoin model calculation: {str(e)}"
        )
        return None

def get_filtered_pools_for_velodrome(pools, current_positions, whitelisted_assets):
    """Filter pools based on criteria and exclude current positions."""
    qualifying_pools = []
    
    logger.info(f"Starting pool filtering with {len(pools)} total pools")
    
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

def format_velodrome_pool_data(pools: List[Dict[str, Any]], chain_id=OPTIMISM_CHAIN_ID, coingecko_api_key=None, coin_id_mapping=None, x402_signer=None, x402_proxy=None, **kwargs) -> List[Dict[str, Any]]:
    """Format pool data for output according to required schema."""
    formatted_pools = []
    chain_name = CHAIN_NAMES.get(chain_id, "unknown")
    
    # Extract max_allocation_percentage from kwargs
    max_allocation_percentage = kwargs.get('max_allocation_percentage', DEFAULT_MAX_ALLOCATION_PERCENTAGE)
    
    for pool in pools:
        # Skip pools with less than two tokens
        if pool.get("token_count", 0) < 2:
            continue
        
        # Get the original sugar data
        sugar_data = pool.get("sugar_data", {})
        
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
            "is_cl_pool": is_cl_pool,  # Add flag for concentrated liquidity pool
            "is_stable": pool.get("is_stable")
        }

        position_data = calculate_position_details_for_velodrome(sugar_data, coingecko_api_key, x402_signer, x402_proxy)  
        formatted_pool.update(position_data)

        # Add tokens (should be at least 2 tokens)
        tokens = pool.get("inputTokens", [])
        if len(tokens) >= 1:
            formatted_pool["token0"] = tokens[0]["id"]
            formatted_pool["token0_symbol"] = tokens[0]["symbol"]
        
        if len(tokens) >= 2:
            formatted_pool["token1"] = tokens[1]["id"]
            formatted_pool["token1_symbol"] = tokens[1]["symbol"]
        
        # Calculate advanced metrics if we have the necessary data
        if coingecko_api_key or (x402_signer and x402_proxy) and len(tokens) >= 2:
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
                depth_score, max_position_size = analyze_velodrome_pool_liquidity(
                    pool["id"], chain_name.upper(), max_allocation_percentage=max_allocation_percentage
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
                
                for i, token in enumerate(tokens):
                    token_symbol = token["symbol"] or ""
                    token_address = token["id"]
                    
                    token_id = get_coin_id_from_symbol(
                        coin_id_mapping,
                        token_symbol,
                        chain_name
                    )
                    token_ids.append(token_id)
                
                # Only calculate IL risk if we have at least 2 valid token IDs
                valid_token_ids = [tid for tid in token_ids if tid]
                
                if len(valid_token_ids) >= 2:
                    
                    # Call IL risk score calculation with pool_id and chain
                    il_risk_score = calculate_velodrome_il_risk_score_multi(
                        valid_token_ids, 
                        coingecko_api_key,
                        pool_id=pool["id"],
                        chain=chain_name,
                        x402_signer=x402_signer,
                        x402_proxy=x402_proxy
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

def apply_composite_pre_filter(pools, top_n=10, apr_weight=0.7, tvl_weight=0.3, min_tvl_threshold=1000, min_volume_threshold=DEFAULT_MIN_VOLUME_THRESHOLD, cl_filter=None):
    """
    Apply composite scoring pre-filter based on standardized APR and TVL.
    
    Args:
        pools: List of pool dictionaries
        top_n: Number of top pools to return (default: 10)
        apr_weight: Weight for APR in composite score (default: 0.7)
        tvl_weight: Weight for TVL in composite score (default: 0.3)
        min_tvl_threshold: Minimum TVL threshold for inclusion (default: 1000)
        min_volume_threshold: Minimum volume threshold for inclusion (default: DEFAULT_MIN_VOLUME_THRESHOLD)
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
    logger.info(f"Parameters: top_n={top_n}, apr_weight={apr_weight}, tvl_weight={tvl_weight}, min_tvl_threshold={min_tvl_threshold}, min_volume_threshold={min_volume_threshold}")
    
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
    
    # Step 2: Apply volume filter if threshold is provided
    if min_volume_threshold is not None and min_volume_threshold > 0:
        volume_filtered_pools = []
        for pool in tvl_filtered_pools:
            try:
                volume_value = float(pool.get('cumulativeVolumeUSD', 0))
                if volume_value >= float(min_volume_threshold):
                    volume_filtered_pools.append(pool)
            except (ValueError, TypeError):
                # Skip pools with invalid volume values
                continue
        logger.info(f"After volume filter (>= {min_volume_threshold}): {len(volume_filtered_pools)} pools")
        
        if not volume_filtered_pools:
            logger.warning("No pools remaining after volume filtering")
            return []
        
        tvl_filtered_pools = volume_filtered_pools
    
    # Step 3: Apply CL filter if specified
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
    
    # Step 4: Standardize metrics and calculate composite scores
    standardized_pools = standardize_metrics(cl_filtered_pools, apr_weight, tvl_weight)
    
    # Step 5: Sort by composite score in descending order
    sorted_pools = sorted(standardized_pools, key=lambda x: x.get('composite_score', 0), reverse=True)
    
    # Step 6: Return top N pools
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

def get_opportunities_for_velodrome(current_positions, coingecko_api_key, chain_id=OPTIMISM_CHAIN_ID, lp_sugar_address=None, ledger_api=None, top_n=10, whitelisted_assets=None, coin_id_mapping=None, x402_signer=None, x402_proxy=None, **kwargs):
    """
    Get and format pool opportunities with optimized caching and performance.
    
    Args:
        current_positions: List of current position IDs to exclude
        coingecko_api_key: API key for CoinGecko
        chain_id: Chain ID to determine which method to use
        lp_sugar_address: Address of the LpSugar contract
        ledger_api: Ethereum API instance or RPC URL
        top_n: Number of top pools by APR to return (default: 10)
        whitelisted_assets: List of whitelisted assets
        coin_id_mapping: Coin ID mapping
        x402_signer: Optional signer for X402
        x402_proxy: Optional proxy for X402
        **kwargs: Additional arguments
    
    Returns:
        List of formatted pool opportunities
    """
    start_time = time.time()    
    
    # Check cache for formatted pools first
    cache_key = f"formatted_pools:{chain_id}:{top_n}:{hash(str(sorted(current_positions)))}"
    cached_result = get_cached_data("formatted_pools", cache_key)
    if cached_result is not None:
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

    # Extract filtering parameters from kwargs
    apr_weight = kwargs.get('apr_weight', 0.7)
    tvl_weight = kwargs.get('tvl_weight', 0.3)
    min_tvl_threshold = kwargs.get('min_tvl_threshold', DEFAULT_MIN_TVL_THRESHOLD)
    min_volume_threshold = kwargs.get('min_volume_threshold', DEFAULT_MIN_VOLUME_THRESHOLD)
    max_allocation_percentage = kwargs.get('max_allocation_percentage', DEFAULT_MAX_ALLOCATION_PERCENTAGE)
    
    # Get top N pools using composite scoring (APR + TVL), with optional CL filtering
    if top_n > 0:
        # Use new composite scoring method
        filtered_pools = apply_composite_pre_filter(
            filtered_pools,
            top_n=top_n, 
            apr_weight=apr_weight,
            tvl_weight=tvl_weight,
            min_tvl_threshold=min_tvl_threshold,
            min_volume_threshold=min_volume_threshold,
        )
        if not filtered_pools:
                logger.error("No filtered pools available for composite filtering")
                return {"error": "No filtered pools available"}
            
    # Format pools with basic data (without advanced metrics)
    
    formatted_pools = format_velodrome_pool_data(
        filtered_pools,
        chain_id,
        coingecko_api_key,
        coin_id_mapping,
        x402_signer=x402_signer,
        x402_proxy=x402_proxy,
        max_allocation_percentage=max_allocation_percentage,
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
    x402_signer: Optional[Any] = None,
    x402_proxy: Optional[Any] = None,
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
            il_risk_score = calculate_velodrome_il_risk_score_multi(
                valid_token_ids, 
                coingecko_api_key,
                pool_id=pool_id,
                chain=chain,
                x402_signer=x402_signer,
                x402_proxy=x402_proxy
            )
            
        # Calculate Sharpe ratio
        sharpe_ratio = get_velodrome_pool_sharpe_ratio(
            pool_id, chain.upper()
        )
        
        # Calculate depth score and max position size
        depth_score, max_position_size = analyze_velodrome_pool_liquidity(
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
            coin_id_mapping=kwargs["coin_id_mapping"],
            x402_signer=kwargs.get("x402_signer"),
            x402_proxy=kwargs.get("x402_proxy"),
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
        # Extract explicitly passed parameters to avoid duplication with **kwargs
        explicit_params = {
            "whitelisted_assets": kwargs.get("whitelisted_assets"),
            "coin_id_mapping": kwargs.get("coin_id_mapping"),
            "x402_signer": kwargs.get("x402_signer"),
            "x402_proxy": kwargs.get("x402_proxy"),
        }
        
        # Create a copy of kwargs without the explicitly passed parameters
        remaining_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ["whitelisted_assets", "coin_id_mapping", "x402_signer", "x402_proxy", 
                                       "current_positions", "coingecko_api_key", "top_n", "lp_sugar_address", "rpc_url"]}
        
        # Pass explicit parameters separately and remaining kwargs for filter parameters
        result = get_opportunities_for_velodrome(
            kwargs.get("current_positions", []),
            kwargs["coingecko_api_key"],
            chain_id,
            sugar_address,
            rpc_url,
            kwargs.get("top_n", 10),  # Get top N pools by APR (default: 10)
            **explicit_params,
            **remaining_kwargs  # Pass remaining kwargs for filter parameters (min_tvl_threshold, min_volume_threshold, max_allocation_percentage)
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

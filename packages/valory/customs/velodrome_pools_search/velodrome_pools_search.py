import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings

import time
from typing import Any, Dict, List, Optional, Union
import json
import logging
from pathlib import Path
from web3 import Web3
from functools import lru_cache

# Set up more detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and mappings
REQUIRED_FIELDS = (
    "chains",
    "current_positions",
    "coingecko_api_key",
)
VELODROME = "Velodrome"

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

# RPC endpoints
RPC_ENDPOINTS = {
    MODE_CHAIN_ID: "https://mainnet.mode.network",
    OPTIMISM_CHAIN_ID: "https://mainnet.optimism.io",
}

# Cache storage with timestamps
CACHE = {
    "pools": {"data": {}, "timestamp": 0, "ttl": 7200},  # 2 hours for pool existence
    "tvl": {"data": {}, "timestamp": 0, "ttl": 600},     # 10 minutes for TVL data
    "connections": {"data": {}, "timestamp": 0, "ttl": 1800}  # 30 minutes for connections
}

# Cache metrics for monitoring
CACHE_METRICS = {
    "hits": {"pools": 0, "tvl": 0, "connections": 0},
    "misses": {"pools": 0, "tvl": 0, "connections": 0}
}

errors = []

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
        
        # Load the ABI directly
        abi_path = Path("packages/valory/customs/velodrome_pools_search/velodrome_lp_sugar.json")
        logger.info(f"Loading ABI from: {abi_path}")
        
        with open(abi_path, "r") as f:
            abi_json = json.load(f)
        
        # Extract the ABI
        abi = abi_json["abi"]
        logger.info(f"ABI loaded with {len(abi)} entries")
        
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
                "inputTokens": [
                    {"id": pool["token0"], "symbol": ""},  # Symbol will be filled later
                    {"id": pool["token1"], "symbol": ""},  # Symbol will be filled later
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
        errors.append(error_msg)
        return {"error": error_msg}

def calculate_tvl_from_reserves(reserve0, reserve1, token0_address, token1_address, token_prices=None):
    """
    Calculate TVL from token reserves and prices.
    
    This is a placeholder function. In a real implementation, you would:
    1. Get token prices from an oracle or price API
    2. Calculate TVL as reserve0 * price0 + reserve1 * price1
    
    For now, we'll return a placeholder value.
    """
    # Check cache first
    cache_key = f"{token0_address}:{token1_address}:{reserve0}:{reserve1}"
    cached_tvl = get_cached_data("tvl", cache_key)
    
    if cached_tvl is not None:
        return cached_tvl
    
    # Placeholder TVL calculation based on the reserves
    placeholder_tvl = float(reserve0) + float(reserve1)
    result = str(placeholder_tvl)
    
    # Cache the result
    set_cached_data("tvl", result, cache_key)
    return result

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
    return qualifying_pools

def format_pool_data(pools: List[Dict[str, Any]], chain_id=OPTIMISM_CHAIN_ID) -> List[Dict[str, Any]]:
    """Format pool data for output according to required schema."""
    formatted_pools = []
    chain_name = CHAIN_NAMES.get(chain_id, "unknown")
    
    for pool in pools:
        # Skip pools with more than two tokens
        if pool.get("token_count", 0) != 2:
            continue
            
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
        }
        
        # Add tokens (should be exactly 2 tokens)
        tokens = pool.get("inputTokens", [])
        if len(tokens) >= 1:
            formatted_pool["token0"] = tokens[0]["id"]
            formatted_pool["token0_symbol"] = tokens[0]["symbol"]
        
        if len(tokens) >= 2:
            formatted_pool["token1"] = tokens[1]["id"]
            formatted_pool["token1_symbol"] = tokens[1]["symbol"]
            
        formatted_pools.append(formatted_pool)
        
    return formatted_pools

def get_opportunities(current_positions, coingecko_api_key, chain_id=OPTIMISM_CHAIN_ID, lp_sugar_address=None, ledger_api=None):
    """Get and format pool opportunities."""
    start_time = time.time()
    logger.info(f"Starting opportunity discovery for chain ID {chain_id}")
    
    # Get pools based on chain
    pools = get_velodrome_pools(chain_id, lp_sugar_address, ledger_api)
    if isinstance(pools, dict) and "error" in pools:
        error_msg = f"Error in pool discovery: {pools['error']}"
        logger.error(error_msg)
        return pools

    # Filter pools
    filtered_pools = get_filtered_pools(pools, current_positions)
    if not filtered_pools:
        logger.warning("No suitable pools found after filtering")
        return {"error": "No suitable pools found"}

    # Format pools
    formatted_pools = format_pool_data(filtered_pools, chain_id)
    
    execution_time = time.time() - start_time
    logger.info(f"Opportunity discovery completed in {execution_time:.2f} seconds")
    logger.info(f"Found {len(formatted_pools)} valid opportunities")
    
    return formatted_pools

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
            
    Returns:
        Dict containing either error messages or result data
    """
    # Clear previous errors
    errors.clear()
    
    # Force refresh cache if requested
    if force_refresh:
        logger.info("Forcing cache refresh")
        invalidate_cache()
    
    start_time = time.time()
    
    # Check for missing required fields
    missing = check_missing_fields(kwargs)
    if missing:
        error_msg = f"Required kwargs {missing} were not provided."
        logger.error(error_msg)
        errors.append(error_msg)
        return {"error": errors}
    
    # Get chains from kwargs
    chains = kwargs.get("chains", [])
    if not chains:
        error_msg = "No chains specified for analysis."
        logger.error(error_msg)
        errors.append(error_msg)
        return {"error": errors}
    
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
            errors.append(error_msg)
            continue
        
        # Check if chain is supported
        if chain_id not in SUGAR_CONTRACT_ADDRESSES:
            error_msg = f"Unsupported chain: {chain}"
            logger.error(error_msg)
            errors.append(error_msg)
            continue
        
        # Get Sugar contract address and RPC URL
        sugar_address = kwargs.get("lp_sugar_address", SUGAR_CONTRACT_ADDRESSES[chain_id])
        rpc_url = kwargs.get("rpc_url", RPC_ENDPOINTS[chain_id])
        
        # Initialize Web3 for the chain
        w3 = get_web3_connection(rpc_url)
        if not w3.is_connected():
            error_msg = f"Failed to connect to RPC endpoint for {chain}: {rpc_url}"
            logger.error(error_msg)
            errors.append(error_msg)
            continue
        
        # Get opportunities for the chain using Sugar contract
        result = get_opportunities(
            kwargs.get("current_positions", []),
            kwargs["coingecko_api_key"],
            chain_id,
            sugar_address,
            rpc_url
        )
        
        # Process results
        if isinstance(result, dict) and "error" in result:
            errors.append(result["error"])
            logger.error(f"Error in opportunity discovery for {chain}: {result['error']}")
            continue
        elif not result:
            error_msg = f"No suitable pools found for {chain}"
            logger.warning(error_msg)
            errors.append(error_msg)
            continue
        
        # Add results to the combined list
        all_results.extend(result)
    
    # Check if we have any results
    if not all_results:
        error_msg = "No suitable pools found across any chains"
        logger.warning(error_msg)
        errors.append(error_msg)
        return {"error": errors}
    
    execution_time = time.time() - start_time
    logger.info(f"Full execution completed in {execution_time:.2f} seconds")
    logger.info(f"Found {len(all_results)} valid opportunities across all chains")
    
    # Log cache metrics
    log_cache_metrics()
    
    return {"result": all_results}

# Example runner function for testing
def run_example():
    """
    Example function to demonstrate how to use the run function.
    
    This function shows how to call the run function with the necessary parameters
    to analyze Velodrome pools on both Optimism and Mode chains.
    
    Returns:
        The result of the run function
    """
    # Set your CoinGecko API key here
    coingecko_api_key = "CG-mf5xZnGELpSXeSqmHDLY2nNU"
    
    # First run - will populate the cache
    logger.info("First run - will populate the cache")
    start_time = time.time()
    result1 = run(
        chains=["optimism", "mode"],  # Run for both chains
        current_positions=[],
        coingecko_api_key=coingecko_api_key,
    )
    first_run_time = time.time() - start_time
    logger.info(f"First run completed in {first_run_time:.2f} seconds")
    
    # Second run - should use the cache
    logger.info("Second run - should use the cache")
    start_time = time.time()
    result2 = run(
        chains=["optimism", "mode"],  # Run for both chains
        current_positions=[],
        coingecko_api_key=coingecko_api_key,
    )
    second_run_time = time.time() - start_time
    logger.info(f"Second run completed in {second_run_time:.2f} seconds")
    
    # Calculate improvement
    if second_run_time > 0:
        speedup = first_run_time / second_run_time
        logger.info(f"Speed improvement: {speedup:.1f}x faster")
        logger.info(f"Time saved: {first_run_time - second_run_time:.2f} seconds ({(1 - second_run_time/first_run_time) * 100:.1f}%)")
    
    # Third run - force refresh the cache
    logger.info("Third run - force refresh the cache")
    start_time = time.time()
    result3 = run(
        force_refresh=True,
        chains=["optimism", "mode"],  # Run for both chains
        current_positions=[],
        coingecko_api_key=coingecko_api_key,
    )
    third_run_time = time.time() - start_time
    logger.info(f"Third run (forced refresh) completed in {third_run_time:.2f} seconds")
    
    # Process the result
    if "error" in result3:
        print(f"Error: {result3['error']}")
        return result3
    
    # Print some basic information about the results
    pools = result3["result"]
    print(f"Found {len(pools)} pools across all chains")
    
    # Group pools by chain
    pools_by_chain = {}
    for pool in pools:
        chain = pool.get("chain", "unknown")
        if chain not in pools_by_chain:
            pools_by_chain[chain] = []
        pools_by_chain[chain].append(pool)
    
    # Print summary by chain
    for chain, chain_pools in pools_by_chain.items():
        print(f"\n{chain.upper()} Chain: {len(chain_pools)} pools")
    
    return result3

# If this file is run directly, execute the example
if __name__ == "__main__":
    run_example()

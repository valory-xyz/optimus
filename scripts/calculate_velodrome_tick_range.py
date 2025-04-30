#!/usr/bin/env python3
"""
CLI script for calculating optimal tick ranges for Velodrome pools.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add the project root to the Python path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the model
from packages.valory.skills.liquidity_trader_abci.model.velodrome import (
    calculate_tick_range,
    format_tick_range_result
)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to show debug logs
    
    Returns:
        Configured logger instance
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Create and configure our script's logger
    logger = logging.getLogger("tick_range_calculator")
    logger.setLevel(log_level)
    
    return logger


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Calculate optimal tick ranges for Velodrome pools"
    )
    
    # Required arguments
    parser.add_argument(
        "--chain", 
        required=True,
        help="Chain name (e.g., optimism, base)"
    )
    parser.add_argument(
        "--token0", 
        required=True,
        help="Token0 contract address"
    )
    parser.add_argument(
        "--token1", 
        required=True,
        help="Token1 contract address"
    )
    
    # Optional arguments
    parser.add_argument(
        "--tick-spacing",
        type=int,
        default=1,
        help="Pool tick spacing (default: 1)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of days of historical data (default: 180)"
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--output-file",
        help="Write output to file instead of stdout"
    )
    parser.add_argument(
        "--use-mock-data",
        action="store_true",
        help="Use mock data instead of API calls"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat calculation N times to test caching (default: 1)"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info(f"Starting Velodrome tick range calculator")
    
    try:
        # Run the tick range calculation
        start_time = time.time()
        logger.info(f"Calculating tick range for {args.token0}/{args.token1} on {args.chain}")
        
        # Initial calculation
        result = calculate_tick_range(
            chain=args.chain,
            token0_address=args.token0,
            token1_address=args.token1,
            tick_spacing=args.tick_spacing,
            days=args.days,
            verbose=args.verbose,
            use_mock_data=args.use_mock_data
        )
        
        # Run multiple times if requested (to test caching)
        if args.repeat > 1:
            logger.info(f"Repeating calculation {args.repeat-1} more times to test caching")
            for i in range(1, args.repeat):
                repeat_start = time.time()
                repeat_result = calculate_tick_range(
                    chain=args.chain,
                    token0_address=args.token0,
                    token1_address=args.token1,
                    tick_spacing=args.tick_spacing,
                    days=args.days,
                    verbose=False,
                    use_mock_data=args.use_mock_data
                )
                repeat_time = time.time() - repeat_start
                logger.info(f"Repeat {i} completed in {repeat_time:.6f} seconds")
        
        # Format the output
        if args.output == "json":
            # Add timestamp to JSON output
            result["timestamp"] = datetime.now().isoformat()
            output = json.dumps(result, indent=2)
        else:
            output = format_tick_range_result(result, args.verbose)
        
        # Write output
        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(output)
            logger.info(f"Results written to {args.output_file}")
        else:
            print("\n" + output + "\n")
        
        total_time = time.time() - start_time
        logger.info(f"Calculation completed in {total_time:.3f} seconds")
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
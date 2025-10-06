"""Test constants and addresses."""

from pathlib import Path

# File paths
PACKAGES_DIR = Path(__file__).parent.parent.parent.parent.parent.parent
TEST_DATA_DIR = Path(__file__).parent / "data"

# Test Safe address (will be deployed)
TEST_SAFE_ADDRESS = "0x1234567890123456789012345678901234567890"

# Token addresses (deployed by Hardhat)
USDC_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
USDT_ADDRESS = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"

# Balancer addresses (Optimism mainnet)
BALANCER_VAULT_ADDRESS = "0xBA12222222228d8Ba445958a75a0704d566BF2C8"

# Test pool (USDC-USDT Weighted Pool)
POOL_ADDRESS = "0x7b50775383d3d6f0215a8f290f2c9e2eebbeceb2"
POOL_ID = "0x7b50775383d3d6f0215a8f290f2c9e2eebbeceb20000000000000000000000fe"

# Initial token amounts for Safe
INITIAL_USDC_AMOUNT = 10 * 10**6  # 10 USDC (6 decimals)
INITIAL_USDT_AMOUNT = 10 * 10**6  # 10 USDT (6 decimals)

# Mock API endpoints
MOCK_API_PORT = 3000
MOCK_BALANCER_SUBGRAPH = f"http://127.0.0.1:{MOCK_API_PORT}/balancer/graphql"
MOCK_COINGECKO_API = f"http://127.0.0.1:{MOCK_API_PORT}/coingecko"

# Hardhat
HARDHAT_PORT = 8545
HARDHAT_ADDRESS = "http://127.0.0.1"
HARDHAT_RPC = f"{HARDHAT_ADDRESS}:{HARDHAT_PORT}"

# Agent storage
TEST_STORAGE_PATH = "/tmp/test_optimus_agent"

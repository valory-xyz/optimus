# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""Test helper utilities for protocol integration tests."""

import json
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from packages.valory.skills.liquidity_trader_abci.behaviours.base import DexType


class TestDataGenerator:
    """Generate test data for different protocols."""

    @staticmethod
    def generate_balancer_pool_data(
        pool_id: Optional[str] = None,
        pool_type: str = "Weighted",
        num_tokens: int = 2,
        weights: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Generate test data for Balancer pools."""
        if pool_id is None:
            pool_id = "0x1234567890123456789012345678901234567890123456789012345678901234"
        
        if weights is None:
            # Equal weights for all tokens
            weight_per_token = 1000000000000000000 // num_tokens
            weights = [weight_per_token] * num_tokens
        
        tokens = [f"0xToken{i:02d}" for i in range(num_tokens)]
        
        return {
            "pool_id": pool_id,
            "pool_address": "0xPoolAddress",
            "pool_type": pool_type,
            "tokens": tokens,
            "weights": weights,
            "total_supply": 1000000000000000000000,  # 1000 BPT
            "swap_fees": 2500000000000000,  # 0.25%
            "daily_volume": 10000000000000000000000,  # 10,000 tokens
            "pool_tvl": 5000000000000000000000,  # 5,000 tokens
            "apr": 12.5,
            "dex_type": DexType.BALANCER.value,
        }

    @staticmethod
    def generate_uniswap_v3_pool_data(
        fee: int = 3000,
        tick_spacing: int = 60,
        current_tick: int = -276310,
    ) -> Dict[str, Any]:
        """Generate test data for Uniswap V3 pools."""
        return {
            "pool_address": "0xPoolAddress",
            "token0": "0xTokenA",
            "token1": "0xTokenB",
            "fee": fee,
            "tick_spacing": tick_spacing,
            "current_tick": current_tick,
            "sqrt_price_x96": 79228162514264337593543950336,
            "liquidity": 1000000000000000000000,
            "daily_volume": 5000000000000000000000,  # 5,000 tokens
            "pool_tvl": 10000000000000000000000,  # 10,000 tokens
            "apr": 15.2,
            "dex_type": DexType.UNISWAP_V3.value,
        }

    @staticmethod
    def generate_velodrome_pool_data(
        is_stable: bool = False,
        is_cl_pool: bool = False,
    ) -> Dict[str, Any]:
        """Generate test data for Velodrome pools."""
        return {
            "pool_address": "0xPoolAddress",
            "token0": "0xTokenA",
            "token1": "0xTokenB",
            "is_stable": is_stable,
            "is_cl_pool": is_cl_pool,
            "gauge_address": "0xGaugeAddress",
            "voter_address": "0xVoterAddress",
            "reserve0": 1000000000000000000000,
            "reserve1": 2000000000000000000000,
            "total_supply": 1000000000000000000000,
            "daily_volume": 3000000000000000000000,  # 3,000 tokens
            "pool_tvl": 3000000000000000000000,  # 3,000 tokens
            "apr": 18.7,
            "dex_type": DexType.VELODROME.value,
        }

    @staticmethod
    def generate_position_data(
        protocol: str,
        pool_data: Dict[str, Any],
        user_balance: int = 100000000000000000000,  # 100 tokens
    ) -> Dict[str, Any]:
        """Generate test data for user positions."""
        base_data = {
            "pool_address": pool_data["pool_address"],
            "dex_type": protocol,
            "user_balance": user_balance,
            "timestamp": 1234567890,
        }

        if protocol == DexType.BALANCER.value:
            return {
                **base_data,
                "pool_id": pool_data["pool_id"],
                "user_bpt_balance": user_balance,
                "total_supply": pool_data["total_supply"],
                "tokens": pool_data["tokens"],
            }
        elif protocol == DexType.UNISWAP_V3.value:
            return {
                **base_data,
                "token_id": 12345,
                "token0": pool_data["token0"],
                "token1": pool_data["token1"],
                "fee": pool_data["fee"],
                "tick_lower": -276320,
                "tick_upper": -276300,
                "liquidity": user_balance,
                "tokens_owed0": 0,
                "tokens_owed1": 0,
            }
        elif protocol == DexType.VELODROME.value:
            return {
                **base_data,
                "token_id": 12345 if pool_data.get("is_cl_pool", False) else None,
                "is_cl_pool": pool_data.get("is_cl_pool", False),
                "gauge_address": pool_data["gauge_address"],
                "staked_amount": user_balance,
                "lp_balance": user_balance,
            }
        else:
            return base_data

    @staticmethod
    def generate_user_assets(
        token_addresses: List[str],
        amounts: Optional[List[int]] = None,
    ) -> Dict[str, int]:
        """Generate user asset balances."""
        if amounts is None:
            amounts = [1000000000000000000000] * len(token_addresses)  # 1000 tokens each
        
        return dict(zip(token_addresses, amounts))

    @staticmethod
    def generate_price_data(
        token_pairs: List[tuple],
        prices: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Generate token price data."""
        if prices is None:
            prices = [1.0] * len(token_pairs)
        
        price_data = {}
        for (token0, token1), price in zip(token_pairs, prices):
            pair_key = f"{token0.lower()}_{token1.lower()}"
            price_data[pair_key] = price
        
        return price_data


class TestAssertions:
    """Custom assertions for protocol tests."""

    @staticmethod
    def assert_transaction_structure(tx_data: Dict[str, Any]) -> None:
        """Assert that transaction data has the correct structure."""
        assert isinstance(tx_data, dict), "Transaction data must be a dictionary"
        assert "tx_hash" in tx_data, "Transaction must contain tx_hash"
        assert tx_data["tx_hash"] is not None, "Transaction hash must not be None"
        
        if isinstance(tx_data["tx_hash"], bytes):
            assert len(tx_data["tx_hash"]) > 0, "Transaction hash must not be empty"
        elif isinstance(tx_data["tx_hash"], str):
            assert len(tx_data["tx_hash"]) > 0, "Transaction hash must not be empty"
            if tx_data["tx_hash"].startswith("0x"):
                assert len(tx_data["tx_hash"]) == 66, "Hex transaction hash must be 66 characters"

    @staticmethod
    def assert_pool_data_structure(pool_data: Dict[str, Any], protocol: str) -> None:
        """Assert that pool data has the correct structure for the protocol."""
        assert isinstance(pool_data, dict), "Pool data must be a dictionary"
        assert "pool_address" in pool_data, "Pool data must contain pool_address"
        assert "dex_type" in pool_data, "Pool data must contain dex_type"
        assert pool_data["dex_type"] == protocol, f"Pool dex_type must be {protocol}"
        
        if protocol == DexType.BALANCER.value:
            assert "pool_id" in pool_data, "Balancer pool must contain pool_id"
            assert "tokens" in pool_data, "Balancer pool must contain tokens"
            assert "weights" in pool_data, "Balancer pool must contain weights"
        elif protocol == DexType.UNISWAP_V3.value:
            assert "token0" in pool_data, "Uniswap V3 pool must contain token0"
            assert "token1" in pool_data, "Uniswap V3 pool must contain token1"
            assert "fee" in pool_data, "Uniswap V3 pool must contain fee"
        elif protocol == DexType.VELODROME.value:
            assert "token0" in pool_data, "Velodrome pool must contain token0"
            assert "token1" in pool_data, "Velodrome pool must contain token1"
            assert "is_stable" in pool_data, "Velodrome pool must contain is_stable"

    @staticmethod
    def assert_position_data_structure(position_data: Dict[str, Any], protocol: str) -> None:
        """Assert that position data has the correct structure for the protocol."""
        assert isinstance(position_data, dict), "Position data must be a dictionary"
        assert "pool_address" in position_data, "Position must contain pool_address"
        assert "dex_type" in position_data, "Position must contain dex_type"
        assert position_data["dex_type"] == protocol, f"Position dex_type must be {protocol}"
        
        if protocol == DexType.BALANCER.value:
            assert "pool_id" in position_data, "Balancer position must contain pool_id"
            assert "user_bpt_balance" in position_data, "Balancer position must contain user_bpt_balance"
        elif protocol == DexType.UNISWAP_V3.value:
            assert "token_id" in position_data, "Uniswap V3 position must contain token_id"
            assert "liquidity" in position_data, "Uniswap V3 position must contain liquidity"
        elif protocol == DexType.VELODROME.value:
            assert "lp_balance" in position_data, "Velodrome position must contain lp_balance"

    @staticmethod
    def assert_yield_calculation_accuracy(
        calculated_yield: Union[float, Decimal],
        expected_yield: Union[float, Decimal],
        tolerance: float = 0.01,
    ) -> None:
        """Assert that yield calculation is accurate within tolerance."""
        calculated = float(calculated_yield)
        expected = float(expected_yield)
        difference = abs(calculated - expected)
        assert difference <= tolerance, f"Yield calculation error: {difference} > {tolerance}"

    @staticmethod
    def assert_apr_calculation_accuracy(
        calculated_apr: Union[float, Decimal],
        expected_apr: Union[float, Decimal],
        tolerance: float = 0.1,
    ) -> None:
        """Assert that APR calculation is accurate within tolerance."""
        calculated = float(calculated_apr)
        expected = float(expected_apr)
        difference = abs(calculated - expected)
        assert difference <= tolerance, f"APR calculation error: {difference} > {tolerance}"


class MockResponseBuilder:
    """Builder for mock HTTP responses."""

    @staticmethod
    def build_success_response(data: Dict[str, Any], status_code: int = 200) -> Dict[str, Any]:
        """Build a successful HTTP response."""
        return {
            "status_code": status_code,
            "body": data,
        }

    @staticmethod
    def build_error_response(
        error_message: str, status_code: int = 400
    ) -> Dict[str, Any]:
        """Build an error HTTP response."""
        return {
            "status_code": status_code,
            "body": {"error": error_message},
        }

    @staticmethod
    def build_graphql_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a GraphQL response."""
        return {
            "status_code": 200,
            "body": {"data": data},
        }

    @staticmethod
    def build_balancer_api_response(pools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a Balancer API response."""
        return {
            "status_code": 200,
            "body": {
                "pools": pools,
                "total": len(pools),
            },
        }

    @staticmethod
    def build_uniswap_graphql_response(pools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a Uniswap GraphQL response."""
        return {
            "status_code": 200,
            "body": {
                "data": {
                    "pools": pools,
                },
            },
        }

    @staticmethod
    def build_velodrome_graphql_response(pools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a Velodrome GraphQL response."""
        return {
            "status_code": 200,
            "body": {
                "data": {
                    "pairs": pools,
                },
            },
        }

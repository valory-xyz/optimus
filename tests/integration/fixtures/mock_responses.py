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

"""Mock responses for integration tests."""

import json
from typing import Any, Dict, List

from tests.integration.protocols.base.test_helpers import MockResponseBuilder


class MockResponseFixtures:
    """Collection of mock response fixtures for different protocols."""

    @staticmethod
    def get_balancer_pools_response(pools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get mock response for Balancer pools API."""
        return MockResponseBuilder.build_balancer_api_response(pools)

    @staticmethod
    def get_uniswap_pools_response(pools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get mock response for Uniswap GraphQL API."""
        return MockResponseBuilder.build_uniswap_graphql_response(pools)

    @staticmethod
    def get_velodrome_pools_response(pools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get mock response for Velodrome GraphQL API."""
        return MockResponseBuilder.build_velodrome_graphql_response(pools)

    @staticmethod
    def get_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """Get a generic success response."""
        return MockResponseBuilder.build_success_response(data)

    @staticmethod
    def get_error_response(error_message: str, status_code: int = 400) -> Dict[str, Any]:
        """Get a generic error response."""
        return MockResponseBuilder.build_error_response(error_message, status_code)

    @staticmethod
    def get_contract_call_response(data: Any) -> Dict[str, Any]:
        """Get a mock contract call response."""
        return {"data": data}

    @staticmethod
    def get_balance_response(balance: int) -> Dict[str, Any]:
        """Get a mock balance response."""
        return {"balance": balance}

    @staticmethod
    def get_decimals_response(decimals: int) -> Dict[str, Any]:
        """Get a mock decimals response."""
        return {"decimals": decimals}

    @staticmethod
    def get_total_supply_response(total_supply: int) -> Dict[str, Any]:
        """Get a mock total supply response."""
        return {"data": total_supply}

    @staticmethod
    def get_pool_tokens_response(tokens: List[str]) -> Dict[str, Any]:
        """Get a mock pool tokens response."""
        return {"tokens": tokens}

    @staticmethod
    def get_slot0_response(
        sqrt_price_x96: int = 79228162514264337593543950336,
        tick: int = -276310,
        unlocked: bool = True,
    ) -> Dict[str, Any]:
        """Get a mock slot0 response for Uniswap V3."""
        return {
            "slot0": {
                "sqrt_price_x96": sqrt_price_x96,
                "tick": tick,
                "observation_index": 0,
                "observation_cardinality": 1,
                "observation_cardinality_next": 1,
                "fee_protocol": 0,
                "unlocked": unlocked,
            }
        }

    @staticmethod
    def get_position_response(
        liquidity: int = 1000000000000000000,
        token0: str = "0xTokenA",
        token1: str = "0xTokenB",
        fee: int = 3000,
        tick_lower: int = -276320,
        tick_upper: int = -276300,
        tokens_owed0: int = 0,
        tokens_owed1: int = 0,
    ) -> Dict[str, Any]:
        """Get a mock position response for Uniswap V3."""
        return {
            "liquidity": liquidity,
            "token0": token0,
            "token1": token1,
            "fee": fee,
            "tickLower": tick_lower,
            "tickUpper": tick_upper,
            "tokensOwed0": tokens_owed0,
            "tokensOwed1": tokens_owed1,
        }

    @staticmethod
    def get_gauge_balance_response(balance: int) -> Dict[str, Any]:
        """Get a mock gauge balance response."""
        return {"balance": balance}

    @staticmethod
    def get_earned_response(earned: int) -> Dict[str, Any]:
        """Get a mock earned rewards response."""
        return {"earned": earned}

    @staticmethod
    def get_reserves_response(reserve0: int, reserve1: int) -> Dict[str, Any]:
        """Get a mock reserves response."""
        return {"data": [reserve0, reserve1]}

    @staticmethod
    def get_transaction_hash_response(tx_hash: str) -> Dict[str, Any]:
        """Get a mock transaction hash response."""
        return {"tx_hash": tx_hash}

    @staticmethod
    def get_approval_response(approved: bool = True) -> Dict[str, Any]:
        """Get a mock approval response."""
        return {"approved": approved}

    @staticmethod
    def get_multisend_response(tx_hash: str) -> Dict[str, Any]:
        """Get a mock multisend response."""
        return {"tx_hash": tx_hash}


# Predefined response sets for common test scenarios
class CommonResponses:
    """Common response sets for testing."""

    # Balancer responses
    BALANCER_VAULT_RESPONSES = {
        "getPoolTokens": MockResponseFixtures.get_pool_tokens_response(
            ["0xTokenA", "0xTokenB"]
        ),
        "getPool": {"pool": "0xPoolAddress", "specialization": "Weighted"},
    }

    BALANCER_POOL_RESPONSES = {
        "balanceOf": MockResponseFixtures.get_balance_response(100000000000000000000),
        "totalSupply": MockResponseFixtures.get_total_supply_response(1000000000000000000000),
        "getPoolId": {"pool_id": "0x1234567890123456789012345678901234567890123456789012345678901234"},
        "getVault": {"vault": "0xVaultAddress"},
        "name": {"name": "Test Pool"},
    }

    # Uniswap V3 responses
    UNISWAP_V3_POOL_RESPONSES = {
        "token0": {"token0": "0xTokenA"},
        "token1": {"token1": "0xTokenB"},
        "fee": {"fee": 3000},
        "tickSpacing": {"tickSpacing": 60},
        "slot0": MockResponseFixtures.get_slot0_response(),
    }

    UNISWAP_V3_POSITION_MANAGER_RESPONSES = {
        "positions": MockResponseFixtures.get_position_response(),
        "balanceOf": MockResponseFixtures.get_balance_response(1),
        "tokenOfOwnerByIndex": {"tokenId": 12345},
    }

    # Velodrome responses
    VELODROME_POOL_RESPONSES = {
        "balanceOf": MockResponseFixtures.get_balance_response(100000000000000000000),
        "totalSupply": MockResponseFixtures.get_total_supply_response(1000000000000000000000),
        "reserve0": {"reserve0": 1000000000000000000000},
        "reserve1": {"reserve1": 2000000000000000000000},
        "token0": {"token0": "0xTokenA"},
        "token1": {"token1": "0xTokenB"},
        "stable": {"stable": False},
    }

    VELODROME_GAUGE_RESPONSES = {
        "balanceOf": MockResponseFixtures.get_gauge_balance_response(100000000000000000000),
        "totalSupply": MockResponseFixtures.get_total_supply_response(1000000000000000000000),
        "earned": MockResponseFixtures.get_earned_response(50000000000000000000),
        "rewardRate": {"rewardRate": 1000000000000000000},
    }

    VELODROME_VOTER_RESPONSES = {
        "gauges": {"gauge": "0xGaugeAddress"},
        "isGauge": {"isGauge": True},
        "votes": {"votes": 100000000000000000000},
    }

    # ERC20 responses
    ERC20_RESPONSES = {
        "balanceOf": MockResponseFixtures.get_balance_response(100000000000000000000),
        "totalSupply": MockResponseFixtures.get_total_supply_response(1000000000000000000000),
        "decimals": MockResponseFixtures.get_decimals_response(18),
        "name": {"name": "Test Token"},
        "symbol": {"symbol": "TEST"},
        "allowance": {"allowance": 0},
    }

    # HTTP API responses
    HTTP_API_RESPONSES = {
        "balancer_pools": MockResponseFixtures.get_balancer_pools_response([
            {
                "id": "0x1234567890123456789012345678901234567890123456789012345678901234",
                "address": "0xPoolAddress",
                "poolType": "Weighted",
                "tokens": [
                    {"address": "0xTokenA", "weight": "0.5"},
                    {"address": "0xTokenB", "weight": "0.5"},
                ],
                "totalLiquidity": "5000000000000000000000",
                "swapFee": "0.0025",
            }
        ]),
        "uniswap_pools": MockResponseFixtures.get_uniswap_pools_response([
            {
                "id": "0xPoolAddress",
                "token0": {"id": "0xTokenA"},
                "token1": {"id": "0xTokenB"},
                "feeTier": "3000",
                "tick": "-276310",
                "sqrtPrice": "79228162514264337593543950336",
                "liquidity": "1000000000000000000000",
                "totalValueLockedUSD": "10000000",
            }
        ]),
        "velodrome_pools": MockResponseFixtures.get_velodrome_pools_response([
            {
                "id": "0xPoolAddress",
                "token0": {"id": "0xTokenA"},
                "token1": {"id": "0xTokenB"},
                "stable": False,
                "reserve0": "1000000000000000000000",
                "reserve1": "2000000000000000000000",
                "totalSupply": "1000000000000000000000",
                "gauge": {"id": "0xGaugeAddress"},
            }
        ]),
    }

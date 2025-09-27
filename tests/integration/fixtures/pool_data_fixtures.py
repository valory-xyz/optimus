# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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

"""Pool data fixtures for integration tests."""

import pytest

from tests.integration.protocols.base.test_helpers import TestDataGenerator


@pytest.fixture
def balancer_weighted_pool_data():
    """Fixture for Balancer weighted pool test data."""
    return TestDataGenerator.generate_balancer_pool_data(
        pool_id="0x1234567890123456789012345678901234567890123456789012345678901234",
        pool_type="Weighted",
        num_tokens=2,
        weights=[500000000000000000, 500000000000000000],  # 50/50 weights
    )


@pytest.fixture
def balancer_stable_pool_data():
    """Fixture for Balancer stable pool test data."""
    return TestDataGenerator.generate_balancer_pool_data(
        pool_id="0x2345678901234567890123456789012345678901234567890123456789012345",
        pool_type="ComposableStable",
        num_tokens=3,
        weights=[333333333333333333, 333333333333333333, 333333333333333334],  # Equal weights
    )


@pytest.fixture
def balancer_3token_pool_data():
    """Fixture for Balancer 3-token pool test data."""
    return TestDataGenerator.generate_balancer_pool_data(
        pool_id="0x3456789012345678901234567890123456789012345678901234567890123456",
        pool_type="Weighted",
        num_tokens=3,
        weights=[400000000000000000, 300000000000000000, 300000000000000000],  # 40/30/30 weights
    )


@pytest.fixture
def uniswap_v3_pool_data():
    """Fixture for Uniswap V3 pool test data."""
    return TestDataGenerator.generate_uniswap_v3_pool_data(
        fee=3000,
        tick_spacing=60,
        current_tick=-276310,
    )


@pytest.fixture
def uniswap_v3_high_fee_pool_data():
    """Fixture for Uniswap V3 high fee pool test data."""
    return TestDataGenerator.generate_uniswap_v3_pool_data(
        fee=10000,
        tick_spacing=200,
        current_tick=-276310,
    )


@pytest.fixture
def uniswap_v3_low_fee_pool_data():
    """Fixture for Uniswap V3 low fee pool test data."""
    return TestDataGenerator.generate_uniswap_v3_pool_data(
        fee=500,
        tick_spacing=10,
        current_tick=-276310,
    )


@pytest.fixture
def velodrome_stable_pool_data():
    """Fixture for Velodrome stable pool test data."""
    return TestDataGenerator.generate_velodrome_pool_data(
        is_stable=True,
        is_cl_pool=False,
    )


@pytest.fixture
def velodrome_volatile_pool_data():
    """Fixture for Velodrome volatile pool test data."""
    return TestDataGenerator.generate_velodrome_pool_data(
        is_stable=False,
        is_cl_pool=False,
    )


@pytest.fixture
def velodrome_cl_pool_data():
    """Fixture for Velodrome concentrated liquidity pool test data."""
    return TestDataGenerator.generate_velodrome_pool_data(
        is_stable=False,
        is_cl_pool=True,
    )


@pytest.fixture
def test_token_addresses():
    """Fixture for test token addresses."""
    return {
        "token_a": "0x3456789012345678901234567890123456789012",
        "token_b": "0x4567890123456789012345678901234567890123",
        "token_c": "0x5678901234567890123456789012345678901234",
        "weth": "0x4200000000000000000000000000000000000006",
        "usdc": "0x7F5c764cBc14f9669B88837ca1490cCa17c31607",
        "usdt": "0x94b008aA00579c1307B0EF2c499aD98a8ce58e58",
    }


@pytest.fixture
def test_user_addresses():
    """Fixture for test user addresses."""
    return {
        "user": "0x1234567890123456789012345678901234567890",
        "safe": "0x2345678901234567890123456789012345678901",
        "pool": "0x3456789012345678901234567890123456789012",
        "vault": "0x4567890123456789012345678901234567890123",
        "gauge": "0x5678901234567890123456789012345678901234",
        "voter": "0x6789012345678901234567890123456789012345",
    }


@pytest.fixture
def test_amounts():
    """Fixture for test amounts."""
    return {
        "small": 1000000000000000000,      # 1 token
        "medium": 100000000000000000000,   # 100 tokens
        "large": 1000000000000000000000,   # 1000 tokens
        "very_large": 10000000000000000000000,  # 10,000 tokens
    }


@pytest.fixture
def balancer_position_data(balancer_weighted_pool_data):
    """Fixture for Balancer position test data."""
    return TestDataGenerator.generate_position_data(
        protocol="balancer",
        pool_data=balancer_weighted_pool_data,
        user_balance=100000000000000000000,  # 100 BPT
    )


@pytest.fixture
def uniswap_v3_position_data(uniswap_v3_pool_data):
    """Fixture for Uniswap V3 position test data."""
    return TestDataGenerator.generate_position_data(
        protocol="uniswap_v3",
        pool_data=uniswap_v3_pool_data,
        user_balance=100000000000000000000,  # 100 tokens
    )


@pytest.fixture
def velodrome_position_data(velodrome_volatile_pool_data):
    """Fixture for Velodrome position test data."""
    return TestDataGenerator.generate_position_data(
        protocol="velodrome",
        pool_data=velodrome_volatile_pool_data,
        user_balance=100000000000000000000,  # 100 LP tokens
    )


@pytest.fixture
def user_assets_2tokens(test_token_addresses, test_amounts):
    """Fixture for user assets with 2 tokens."""
    return TestDataGenerator.generate_user_assets(
        token_addresses=[test_token_addresses["token_a"], test_token_addresses["token_b"]],
        amounts=[test_amounts["large"], test_amounts["large"]],
    )


@pytest.fixture
def user_assets_3tokens(test_token_addresses, test_amounts):
    """Fixture for user assets with 3 tokens."""
    return TestDataGenerator.generate_user_assets(
        token_addresses=[
            test_token_addresses["token_a"],
            test_token_addresses["token_b"],
            test_token_addresses["token_c"],
        ],
        amounts=[test_amounts["large"], test_amounts["large"], test_amounts["large"]],
    )


@pytest.fixture
def token_price_data(test_token_addresses):
    """Fixture for token price data."""
    return TestDataGenerator.generate_price_data(
        token_pairs=[
            (test_token_addresses["token_a"], test_token_addresses["token_b"]),
            (test_token_addresses["token_a"], test_token_addresses["weth"]),
            (test_token_addresses["token_b"], test_token_addresses["weth"]),
        ],
        prices=[1.5, 0.8, 1.2],
    )


@pytest.fixture
def mock_http_responses():
    """Fixture for mock HTTP responses."""
    return {
        "balancer_api": {
            "status_code": 200,
            "body": {
                "pools": [
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
                ],
            },
        },
        "uniswap_graphql": {
            "status_code": 200,
            "body": {
                "data": {
                    "pools": [
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
                    ],
                },
            },
        },
        "velodrome_graphql": {
            "status_code": 200,
            "body": {
                "data": {
                    "pairs": [
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
                    ],
                },
            },
        },
    }

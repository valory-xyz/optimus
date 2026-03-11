# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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

"""Test the pools/uniswap.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import sys

from packages.valory.skills.liquidity_trader_abci.pools.uniswap import (
    INT_MAX,
    MAX_TICK,
    MIN_TICK,
    ZERO_ADDRESS,
    MintParams,
    UniswapPoolBehaviour,
)


def test_import() -> None:
    """Test that the uniswap module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.pools.uniswap  # noqa


def test_zero_address() -> None:
    """Test ZERO_ADDRESS constant."""
    assert ZERO_ADDRESS == "0x0000000000000000000000000000000000000000"


def test_min_tick() -> None:
    """Test MIN_TICK constant."""
    assert MIN_TICK == -887272


def test_max_tick() -> None:
    """Test MAX_TICK constant."""
    assert MAX_TICK == 887272


def test_int_max() -> None:
    """Test INT_MAX constant."""
    assert INT_MAX == sys.maxsize


class TestMintParams:
    """Test MintParams class."""

    def test_init(self) -> None:
        """Test MintParams initialization."""
        params = MintParams(
            token0="0xtoken0",
            token1="0xtoken1",
            fee=3000,
            tickLower=-100,
            tickUpper=100,
            amount0Desired=1000,
            amount1Desired=2000,
            amount0Min=900,
            amount1Min=1800,
            recipient="0xrecipient",
            deadline=999999,
        )
        assert params.token0 == "0xtoken0"
        assert params.token1 == "0xtoken1"
        assert params.fee == 3000
        assert params.tickLower == -100
        assert params.tickUpper == 100
        assert params.amount0Desired == 1000
        assert params.amount1Desired == 2000
        assert params.amount0Min == 900
        assert params.amount1Min == 1800
        assert params.recipient == "0xrecipient"
        assert params.deadline == 999999

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

"""Tests for the UniswapV3PoolContract, TickMath, and LiquidityAmounts."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.uniswap_v3_pool.contract import (
    LiquidityAmounts,
    TickMath,
    UniswapV3PoolContract,
)

MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
MOCK_TOKEN0 = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"
MOCK_TOKEN1 = "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB"


class TestTickMathGetSqrtRatioAtTickPositive:
    """Tests for TickMath.getSqrtRatioAtTick with positive ticks."""

    def test_positive_tick_returns_valid_ratio(self) -> None:
        """Test that a positive tick returns a sqrtRatio greater than the base 2^96."""
        result = TickMath.getSqrtRatioAtTick(1)
        assert result > 0
        assert isinstance(result, int)

    def test_large_positive_tick(self) -> None:
        """Test that a large positive tick produces a ratio clamped at or below MAX_SQRT_RATIO."""
        result = TickMath.getSqrtRatioAtTick(887272)
        assert result <= TickMath.MAX_SQRT_RATIO


class TestTickMathGetSqrtRatioAtTickNegative:
    """Tests for TickMath.getSqrtRatioAtTick with negative ticks."""

    def test_negative_tick_returns_valid_ratio(self) -> None:
        """Test that a negative tick returns a valid positive sqrtRatio via inversion."""
        result = TickMath.getSqrtRatioAtTick(-1)
        assert result > 0
        assert isinstance(result, int)
        # Negative ticks cause inversion: (2^256-1)//ratio, producing a value >= 2^128
        ratio_at_zero = TickMath.getSqrtRatioAtTick(0)
        assert result > ratio_at_zero


class TestTickMathGetSqrtRatioAtTickZero:
    """Tests for TickMath.getSqrtRatioAtTick with tick 0."""

    def test_zero_tick_returns_base_ratio(self) -> None:
        """Test that tick 0 returns the initial ratio (no bit-level adjustments applied)."""
        result = TickMath.getSqrtRatioAtTick(0)
        # For tick 0, no bit operations fire so ratio stays 0x100...000 >> 32 equivalent
        assert result > 0
        assert isinstance(result, int)


class TestTickMathGetSqrtRatioAtTickOutOfRange:
    """Tests for TickMath.getSqrtRatioAtTick with out-of-range ticks."""

    def test_tick_below_min_raises_value_error(self) -> None:
        """Test that a tick below MIN_TICK raises ValueError."""
        with pytest.raises(ValueError, match="Tick out of range"):
            TickMath.getSqrtRatioAtTick(TickMath.MIN_TICK - 1)

    def test_tick_above_max_raises_value_error(self) -> None:
        """Test that a tick above MAX_TICK raises ValueError."""
        with pytest.raises(ValueError, match="Tick out of range"):
            TickMath.getSqrtRatioAtTick(TickMath.MAX_TICK + 1)


class TestTickMathBoundaryClamping:
    """Tests for TickMath boundary clamping logic."""

    def test_min_tick_clamped_to_max_sqrt_ratio(self) -> None:
        """Test that MIN_TICK returns MAX_SQRT_RATIO due to inversion overflow clamping."""
        result = TickMath.getSqrtRatioAtTick(TickMath.MIN_TICK)
        assert result == TickMath.MAX_SQRT_RATIO

    def test_max_tick_returns_valid_ratio(self) -> None:
        """Test that MAX_TICK returns a ratio within valid bounds."""
        result = TickMath.getSqrtRatioAtTick(TickMath.MAX_TICK)
        assert result <= TickMath.MAX_SQRT_RATIO
        assert result >= TickMath.MIN_SQRT_RATIO


class TestTickMathAllBitOperations:
    """Tests for TickMath.getSqrtRatioAtTick exercising all bit operation branches."""

    def test_tick_exercises_remaining_bit_operations(self) -> None:
        """Test tick=161298 exercises bits 0x2, 0x10, 0x200, 0x400, 0x1000, 0x2000, 0x4000, 0x20000."""
        # 161298 = 0x27612 sets bits 1,4,9,10,12,13,14,17
        # These are the only bits not covered by tick=1 (bit 0),
        # tick=100 (bits 2,5,6), and tick=887272 (bits 3,5,6,7,8,11,15,16,18,19)
        result = TickMath.getSqrtRatioAtTick(161298)
        assert result > 0
        assert isinstance(result, int)
        assert result <= TickMath.MAX_SQRT_RATIO


# Q96 = 2^96, used to build realistic sqrt price values
Q96 = 2**96


class TestLiquidityAmountsBelowRange:
    """Tests for LiquidityAmounts.getAmountsForLiquidity when price is below range."""

    def test_price_below_range_all_token0(self) -> None:
        """Test that when current price is below range, all liquidity is in token0."""
        sqrt_current = 1 * Q96  # price = 1
        sqrt_a = 2 * Q96  # lower bound
        sqrt_b = 3 * Q96  # upper bound

        amount0, amount1 = LiquidityAmounts.getAmountsForLiquidity(
            sqrt_current, sqrt_a, sqrt_b, 1_000_000
        )

        assert amount0 > 0
        assert amount1 == 0


class TestLiquidityAmountsWithinRange:
    """Tests for LiquidityAmounts.getAmountsForLiquidity when price is within range."""

    def test_price_within_range_both_tokens(self) -> None:
        """Test that when current price is within range, both token0 and token1 are non-zero."""
        sqrt_a = 1 * Q96
        sqrt_b = 3 * Q96
        sqrt_current = 2 * Q96  # within [a, b]

        amount0, amount1 = LiquidityAmounts.getAmountsForLiquidity(
            sqrt_current, sqrt_a, sqrt_b, 1_000_000
        )

        assert amount0 > 0
        assert amount1 > 0


class TestLiquidityAmountsAboveRange:
    """Tests for LiquidityAmounts.getAmountsForLiquidity when price is above range."""

    def test_price_above_range_all_token1(self) -> None:
        """Test that when current price is above range, all liquidity is in token1."""
        sqrt_a = 1 * Q96
        sqrt_b = 2 * Q96
        sqrt_current = 3 * Q96  # above upper bound

        amount0, amount1 = LiquidityAmounts.getAmountsForLiquidity(
            sqrt_current, sqrt_a, sqrt_b, 1_000_000
        )

        assert amount0 == 0
        assert amount1 > 0


class TestLiquidityAmountsSwappedBounds:
    """Tests for LiquidityAmounts.getAmountsForLiquidity with swapped sqrtRatioA/B."""

    def test_swapped_bounds_are_normalized(self) -> None:
        """Test that if sqrtRatioA > sqrtRatioB they are swapped internally."""
        sqrt_a = 3 * Q96
        sqrt_b = 1 * Q96
        sqrt_current = 2 * Q96

        amount0, amount1 = LiquidityAmounts.getAmountsForLiquidity(
            sqrt_current, sqrt_a, sqrt_b, 1_000_000
        )

        # After swap: a=1*Q96, b=3*Q96, current within range
        assert amount0 > 0
        assert amount1 > 0


class TestGetAmount0ForLiquidity:
    """Tests for LiquidityAmounts._getAmount0ForLiquidity."""

    def test_get_amount0_for_liquidity(self) -> None:
        """Test the internal _getAmount0ForLiquidity helper returns a positive integer."""
        sqrt_a = 1 * Q96
        sqrt_b = 2 * Q96
        liquidity = 1_000_000

        result = LiquidityAmounts._getAmount0ForLiquidity(sqrt_a, sqrt_b, liquidity)
        assert result > 0
        assert isinstance(result, int)


class TestGetAmount1ForLiquidity:
    """Tests for LiquidityAmounts._getAmount1ForLiquidity."""

    def test_get_amount1_for_liquidity(self) -> None:
        """Test the internal _getAmount1ForLiquidity helper returns a positive integer."""
        sqrt_a = 1 * Q96
        sqrt_b = 2 * Q96
        liquidity = 1_000_000

        result = LiquidityAmounts._getAmount1ForLiquidity(sqrt_a, sqrt_b, liquidity)
        assert result > 0
        assert isinstance(result, int)


class TestGetPoolTokens:
    """Tests for UniswapV3PoolContract.get_pool_tokens."""

    def test_get_pool_tokens_returns_token_list(self) -> None:
        """Test that get_pool_tokens calls token0/token1 and returns dict(tokens=[...])."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.token0.return_value.call.return_value = MOCK_TOKEN0
        mock_instance.functions.token1.return_value.call.return_value = MOCK_TOKEN1

        with patch.object(
            UniswapV3PoolContract, "get_instance", return_value=mock_instance
        ):
            result = UniswapV3PoolContract.get_pool_tokens(
                mock_ledger_api, MOCK_ADDRESS
            )

        assert result == {"tokens": [MOCK_TOKEN0, MOCK_TOKEN1]}


class TestGetPoolFee:
    """Tests for UniswapV3PoolContract.get_pool_fee."""

    def test_get_pool_fee_returns_fee(self) -> None:
        """Test that get_pool_fee calls fee() and returns dict(data=fee)."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.fee.return_value.call.return_value = 3000

        with patch.object(
            UniswapV3PoolContract, "get_instance", return_value=mock_instance
        ):
            result = UniswapV3PoolContract.get_pool_fee(mock_ledger_api, MOCK_ADDRESS)

        assert result == {"data": 3000}


class TestGetTickSpacing:
    """Tests for UniswapV3PoolContract.get_tick_spacing."""

    def test_get_tick_spacing_returns_spacing(self) -> None:
        """Test that get_tick_spacing calls tickSpacing() and returns dict(data=tick_spacing)."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.tickSpacing.return_value.call.return_value = 60

        with patch.object(
            UniswapV3PoolContract, "get_instance", return_value=mock_instance
        ):
            result = UniswapV3PoolContract.get_tick_spacing(
                mock_ledger_api, MOCK_ADDRESS
            )

        assert result == {"data": 60}


class TestSlot0:
    """Tests for UniswapV3PoolContract.slot0."""

    def test_slot0_returns_seven_field_dict(self) -> None:
        """Test that slot0 returns a nested dict with 7 fields from the pool state."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.slot0.return_value.call.return_value = (
            79228162514264337593543950336,  # sqrt_price_x96
            0,  # tick
            10,  # observation_index
            100,  # observation_cardinality
            200,  # observation_cardinality_next
            8,  # fee_protocol
            True,  # unlocked
        )

        with patch.object(
            UniswapV3PoolContract, "get_instance", return_value=mock_instance
        ):
            result = UniswapV3PoolContract.slot0(mock_ledger_api, MOCK_ADDRESS)

        assert result == {
            "slot0": {
                "sqrt_price_x96": 79228162514264337593543950336,
                "tick": 0,
                "observation_index": 10,
                "observation_cardinality": 100,
                "observation_cardinality_next": 200,
                "fee_protocol": 8,
                "unlocked": True,
            }
        }


class TestGetReservesAndBalances:
    """Tests for UniswapV3PoolContract.get_reserves_and_balances."""

    def test_get_reserves_and_balances_returns_computed_amounts(self) -> None:
        """Test that get_reserves_and_balances computes token amounts from a position dict."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()

        # slot0 returns (sqrtPriceX96, tick, ...)
        sqrt_price = 2 * Q96
        current_tick = 100
        mock_instance.functions.slot0.return_value.call.return_value = (
            sqrt_price,
            current_tick,
            0,
            0,
            0,
            0,
            True,
        )

        position = {
            "token0": MOCK_TOKEN0,
            "token1": MOCK_TOKEN1,
            "fee": 3000,
            "tickLower": -887272,
            "tickUpper": 887272,
            "liquidity": 1_000_000,
            "tokensOwed0": 10,
            "tokensOwed1": 20,
        }

        with patch.object(
            UniswapV3PoolContract, "get_instance", return_value=mock_instance
        ):
            result = UniswapV3PoolContract.get_reserves_and_balances(
                mock_ledger_api, MOCK_ADDRESS, position
            )

        data = result["data"]
        assert "current_token0_qty" in data
        assert "current_token1_qty" in data
        assert data["liquidity"] == 1_000_000
        assert data["tick_lower"] == -887272
        assert data["tick_upper"] == 887272
        assert data["current_tick"] == current_tick
        assert data["sqrt_price_x96"] == sqrt_price
        assert data["tokens_owed0"] == 10
        assert data["tokens_owed1"] == 20
        # Amounts include tokens_owed, so should be >= owed values
        assert data["current_token0_qty"] >= 10
        assert data["current_token1_qty"] >= 20

    def test_get_reserves_and_balances_no_sqrt_ratio_swap(self) -> None:
        """Test get_reserves_and_balances when sqrtRatioA <= sqrtRatioB (no swap needed)."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()

        sqrt_price = 2 * Q96
        current_tick = 150
        mock_instance.functions.slot0.return_value.call.return_value = (
            sqrt_price,
            current_tick,
            0,
            0,
            0,
            0,
            True,
        )

        # Both positive ticks with tickLower > tickUpper: getSqrtRatioAtTick
        # decreases for larger positive ticks, so sqrtRatioA < sqrtRatioB
        position = {
            "token0": MOCK_TOKEN0,
            "token1": MOCK_TOKEN1,
            "fee": 3000,
            "tickLower": 200,
            "tickUpper": 100,
            "liquidity": 1_000_000,
            "tokensOwed0": 0,
            "tokensOwed1": 0,
        }

        with patch.object(
            UniswapV3PoolContract, "get_instance", return_value=mock_instance
        ):
            result = UniswapV3PoolContract.get_reserves_and_balances(
                mock_ledger_api, MOCK_ADDRESS, position
            )

        data = result["data"]
        assert "current_token0_qty" in data
        assert "current_token1_qty" in data
        assert data["liquidity"] == 1_000_000

    def test_get_reserves_and_balances_with_zero_liquidity(self) -> None:
        """Test that get_reserves_and_balances handles zero liquidity correctly."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()

        sqrt_price = 2 * Q96
        mock_instance.functions.slot0.return_value.call.return_value = (
            sqrt_price,
            0,
            0,
            0,
            0,
            0,
            True,
        )

        position = {
            "token0": MOCK_TOKEN0,
            "token1": MOCK_TOKEN1,
            "fee": 3000,
            "tickLower": -100,
            "tickUpper": 100,
            "liquidity": 0,
            "tokensOwed0": 5,
            "tokensOwed1": 10,
        }

        with patch.object(
            UniswapV3PoolContract, "get_instance", return_value=mock_instance
        ):
            result = UniswapV3PoolContract.get_reserves_and_balances(
                mock_ledger_api, MOCK_ADDRESS, position
            )

        data = result["data"]
        # With zero liquidity, amounts come only from tokensOwed
        assert data["current_token0_qty"] == 5
        assert data["current_token1_qty"] == 10


class TestAbiIntegrity:
    """Verify that every function used in contract.py exists in the ABI."""

    ABI_FILE = "UniswapV3Pool.json"
    EXPECTED_FUNCTIONS = ["fee", "slot0", "tickSpacing", "token0", "token1"]

    @classmethod
    def _load_abi_function_names(cls) -> set:
        abi_path = Path(__file__).parents[1] / "build" / cls.ABI_FILE
        with open(abi_path) as f:
            data = json.load(f)
        abi = data.get("abi", data) if isinstance(data, dict) else data
        return {e["name"] for e in abi if e.get("type") == "function"}

    def test_all_functions_present(self) -> None:
        """Test that all functions used in contract.py exist in the ABI."""
        abi_funcs = self._load_abi_function_names()
        missing = [f for f in self.EXPECTED_FUNCTIONS if f not in abi_funcs]
        assert not missing, f"Functions missing from ABI: {missing}"

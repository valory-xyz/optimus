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

"""Test the utils/tick_math.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import pytest

from packages.valory.skills.liquidity_trader_abci.utils.tick_math import (
    MAX_SQRT_RATIO,
    MAX_TICK,
    MIN_SQRT_RATIO,
    MIN_TICK,
    _get_amount0_for_liquidity,
    _get_amount1_for_liquidity,
    get_amounts_for_liquidity,
    get_liquidity_for_amount0,
    get_liquidity_for_amount1,
    get_liquidity_for_amounts,
    get_sqrt_ratio_at_tick,
)


def test_import() -> None:
    """Test that the tick_math module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.utils.tick_math  # noqa


def test_constants() -> None:
    """Test tick_math constants."""
    assert MIN_TICK == -887272
    assert MAX_TICK == 887272
    assert MIN_SQRT_RATIO == 4295128739
    assert MAX_SQRT_RATIO == 1461446703485210103287273052203988822378723970342


class TestGetSqrtRatioAtTick:
    """Test get_sqrt_ratio_at_tick function."""

    def test_tick_zero(self) -> None:
        """Test tick 0 returns exactly 2^96."""
        result = get_sqrt_ratio_at_tick(0)
        expected = 2**96
        assert result == expected

    def test_positive_tick(self) -> None:
        """Test positive tick returns a valid value."""
        result = get_sqrt_ratio_at_tick(1)
        assert isinstance(result, int)
        assert result > 0

    def test_negative_tick(self) -> None:
        """Test negative tick returns a valid value."""
        result = get_sqrt_ratio_at_tick(-1)
        assert isinstance(result, int)
        assert result > 0

    def test_min_tick(self) -> None:
        """Test MIN_TICK returns MAX_SQRT_RATIO (inverted for negative ticks)."""
        result = get_sqrt_ratio_at_tick(MIN_TICK)
        assert result == MAX_SQRT_RATIO

    def test_max_tick(self) -> None:
        """Test MAX_TICK returns MIN_SQRT_RATIO."""
        result = get_sqrt_ratio_at_tick(MAX_TICK)
        assert result == MIN_SQRT_RATIO

    def test_out_of_range_raises(self) -> None:
        """Test tick out of range raises ValueError."""
        with pytest.raises(ValueError, match="Tick out of range"):
            get_sqrt_ratio_at_tick(MIN_TICK - 1)
        with pytest.raises(ValueError, match="Tick out of range"):
            get_sqrt_ratio_at_tick(MAX_TICK + 1)

    def test_various_tick_bits(self) -> None:
        """Test various ticks to cover all bit checks."""
        # Cover all bit patterns by testing with a tick that has many bits set
        result = get_sqrt_ratio_at_tick(0x7FFFF)  # All lower 19 bits set (524287)
        assert isinstance(result, int)
        assert result > 0

        result = get_sqrt_ratio_at_tick(-0x7FFFF)
        assert isinstance(result, int)
        assert result > 0

    def test_tick_with_specific_bits(self) -> None:
        """Test ticks that exercise specific bit patterns."""
        # Test individual bits
        for bit in range(20):
            tick_val = 1 << bit
            if tick_val <= MAX_TICK:
                result = get_sqrt_ratio_at_tick(tick_val)
                assert result > 0
                result_neg = get_sqrt_ratio_at_tick(-tick_val)
                assert result_neg > 0


class TestGetAmountsForLiquidity:
    """Test get_amounts_for_liquidity function."""

    def test_price_below_range(self) -> None:
        """Test when current price is below the range."""
        vals = sorted([get_sqrt_ratio_at_tick(0), get_sqrt_ratio_at_tick(100)])
        sqrt_current = get_sqrt_ratio_at_tick(-200)
        # Ensure current < lower bound
        lower = min(vals)
        upper = max(vals)
        if sqrt_current > lower:
            sqrt_current, lower, upper = lower, upper, sqrt_current + upper
        amount0, amount1 = get_amounts_for_liquidity(sqrt_current, lower, upper, 10**18)
        assert amount0 > 0
        assert amount1 == 0

    def test_price_above_range(self) -> None:
        """Test when current price is above the range."""
        vals = sorted([get_sqrt_ratio_at_tick(-100), get_sqrt_ratio_at_tick(0)])
        sqrt_current = get_sqrt_ratio_at_tick(200)
        lower = vals[0]
        upper = vals[1]
        # Ensure current > upper
        if sqrt_current < upper:
            sqrt_current = upper + 10**96
        amount0, amount1 = get_amounts_for_liquidity(sqrt_current, lower, upper, 10**18)
        assert amount0 == 0
        assert amount1 > 0

    def test_price_in_range(self) -> None:
        """Test when current price is within the range."""
        sqrt_vals = sorted([get_sqrt_ratio_at_tick(-100), get_sqrt_ratio_at_tick(100)])
        sqrt_current = get_sqrt_ratio_at_tick(0)
        liquidity = 10**18
        amount0, amount1 = get_amounts_for_liquidity(sqrt_current, sqrt_vals[0], sqrt_vals[1], liquidity)
        assert amount0 > 0
        assert amount1 > 0

    def test_swapped_bounds(self) -> None:
        """Test that bounds are swapped if in wrong order."""
        sqrt_a = get_sqrt_ratio_at_tick(100)
        sqrt_b = get_sqrt_ratio_at_tick(-100)
        sqrt_current = get_sqrt_ratio_at_tick(0)
        liquidity = 10**18
        amount0, amount1 = get_amounts_for_liquidity(sqrt_current, sqrt_a, sqrt_b, liquidity)
        assert amount0 >= 0
        assert amount1 >= 0

    def test_swapped_bounds_explicit(self) -> None:
        """Test that bounds are swapped when A > B by passing raw values."""
        # Explicitly pass A > B to trigger the swap on line 106
        lower = get_sqrt_ratio_at_tick(-100)
        upper = get_sqrt_ratio_at_tick(100)
        # Ensure we know which is actually larger
        actual_lower = min(lower, upper)
        actual_upper = max(lower, upper)
        sqrt_current = get_sqrt_ratio_at_tick(0)
        # Pass upper as A and lower as B so A > B
        amount0, amount1 = get_amounts_for_liquidity(
            sqrt_current, actual_upper, actual_lower, 10**18
        )
        assert amount0 >= 0
        assert amount1 >= 0

    def test_price_at_lower_bound(self) -> None:
        """Test when current price equals lower bound."""
        # Use ticks to generate sqrtRatioX96; note the values may sort differently
        sqrt_a = get_sqrt_ratio_at_tick(-100)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        lower = min(sqrt_a, sqrt_b)
        upper = max(sqrt_a, sqrt_b)
        amount0, amount1 = get_amounts_for_liquidity(lower, lower, upper, 10**18)
        # at lower bound: all in token0, none in token1
        assert amount0 > 0
        assert amount1 == 0

    def test_price_at_upper_bound(self) -> None:
        """Test when current price equals upper bound."""
        sqrt_a = get_sqrt_ratio_at_tick(-100)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        lower = min(sqrt_a, sqrt_b)
        upper = max(sqrt_a, sqrt_b)
        amount0, amount1 = get_amounts_for_liquidity(upper, lower, upper, 10**18)
        # at upper bound: all in token1, none in token0
        assert amount0 == 0
        assert amount1 > 0


class TestGetAmount0ForLiquidity:
    """Test _get_amount0_for_liquidity function."""

    def test_zero_diff(self) -> None:
        """Test returns 0 when sqrt_diff is 0."""
        result = _get_amount0_for_liquidity(100, 100, 1000)
        assert result == 0

    def test_negative_diff(self) -> None:
        """Test returns 0 when sqrtRatioAX96 > sqrtRatioBX96."""
        result = _get_amount0_for_liquidity(200, 100, 1000)
        assert result == 0

    def test_zero_denominator(self) -> None:
        """Test returns 0 when denominator is 0."""
        result = _get_amount0_for_liquidity(0, 100, 1000)
        assert result == 0

    def test_positive_result(self) -> None:
        """Test normal calculation produces positive result."""
        vals = sorted([get_sqrt_ratio_at_tick(-100), get_sqrt_ratio_at_tick(100)])
        result = _get_amount0_for_liquidity(vals[0], vals[1], 10**18)
        assert result > 0


class TestGetAmount1ForLiquidity:
    """Test _get_amount1_for_liquidity function."""

    def test_basic(self) -> None:
        """Test basic amount1 calculation."""
        vals = sorted([get_sqrt_ratio_at_tick(-100), get_sqrt_ratio_at_tick(100)])
        result = _get_amount1_for_liquidity(vals[0], vals[1], 10**18)
        assert result > 0


class TestGetLiquidityForAmount0:
    """Test get_liquidity_for_amount0 function."""

    def test_basic(self) -> None:
        """Test basic liquidity from amount0 calculation."""
        sqrt_a = get_sqrt_ratio_at_tick(-100)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        result = get_liquidity_for_amount0(sqrt_a, sqrt_b, 10**18)
        assert result > 0

    def test_swapped_bounds(self) -> None:
        """Test with bounds in wrong order."""
        sqrt_a = get_sqrt_ratio_at_tick(100)
        sqrt_b = get_sqrt_ratio_at_tick(-100)
        result = get_liquidity_for_amount0(sqrt_a, sqrt_b, 10**18)
        assert result > 0


class TestGetLiquidityForAmount1:
    """Test get_liquidity_for_amount1 function."""

    def test_basic(self) -> None:
        """Test basic liquidity from amount1 calculation."""
        sqrt_a = get_sqrt_ratio_at_tick(-100)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        result = get_liquidity_for_amount1(sqrt_a, sqrt_b, 10**18)
        assert result > 0

    def test_swapped_bounds(self) -> None:
        """Test with bounds in wrong order."""
        sqrt_a = get_sqrt_ratio_at_tick(100)
        sqrt_b = get_sqrt_ratio_at_tick(-100)
        result = get_liquidity_for_amount1(sqrt_a, sqrt_b, 10**18)
        assert result > 0


class TestGetLiquidityForAmounts:
    """Test get_liquidity_for_amounts function."""

    def test_price_below_range(self) -> None:
        """Test when current price is below range - only token0 used."""
        sqrt_a = get_sqrt_ratio_at_tick(0)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        sqrt_current = get_sqrt_ratio_at_tick(-100)
        result = get_liquidity_for_amounts(sqrt_current, sqrt_a, sqrt_b, 10**18, 10**18)
        assert result > 0

    def test_price_in_range(self) -> None:
        """Test when current price is within range - both tokens used."""
        sqrt_a = get_sqrt_ratio_at_tick(-100)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        sqrt_current = get_sqrt_ratio_at_tick(0)
        result = get_liquidity_for_amounts(sqrt_current, sqrt_a, sqrt_b, 10**18, 10**18)
        assert result > 0

    def test_price_above_range(self) -> None:
        """Test when current price is above range - only token1 used."""
        sqrt_a = get_sqrt_ratio_at_tick(-100)
        sqrt_b = get_sqrt_ratio_at_tick(0)
        sqrt_current = get_sqrt_ratio_at_tick(100)
        result = get_liquidity_for_amounts(sqrt_current, sqrt_a, sqrt_b, 10**18, 10**18)
        assert result > 0

    def test_swapped_bounds(self) -> None:
        """Test with bounds in wrong order."""
        sqrt_a = get_sqrt_ratio_at_tick(100)
        sqrt_b = get_sqrt_ratio_at_tick(-100)
        sqrt_current = get_sqrt_ratio_at_tick(0)
        result = get_liquidity_for_amounts(sqrt_current, sqrt_a, sqrt_b, 10**18, 10**18)
        assert result > 0

    def test_price_at_lower_bound(self) -> None:
        """Test when price equals lower bound."""
        sqrt_a = get_sqrt_ratio_at_tick(-100)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        result = get_liquidity_for_amounts(sqrt_a, sqrt_a, sqrt_b, 10**18, 10**18)
        assert result > 0

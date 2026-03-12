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

"""Test the utils/balancer_math.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import pytest

from packages.valory.skills.liquidity_trader_abci.utils.balancer_math import (
    BalancerProportionalMath,
)


def test_import() -> None:
    """Test that the balancer_math module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.utils.balancer_math  # noqa


class TestBalancerProportionalMathExit:
    """Test BalancerProportionalMath.query_proportional_exit."""

    def test_basic_exit(self) -> None:
        """Test basic proportional exit calculation."""
        amounts = BalancerProportionalMath.query_proportional_exit(
            token_balances=[1000, 2000],
            total_bpt_supply=100,
            bpt_amount_in=10,
        )
        assert amounts == [100, 200]

    def test_exit_full_supply(self) -> None:
        """Test exit with full BPT supply."""
        amounts = BalancerProportionalMath.query_proportional_exit(
            token_balances=[1000, 2000, 3000],
            total_bpt_supply=100,
            bpt_amount_in=100,
        )
        assert amounts == [1000, 2000, 3000]

    def test_exit_string_inputs(self) -> None:
        """Test exit with string inputs."""
        amounts = BalancerProportionalMath.query_proportional_exit(
            token_balances=["1000", "2000"],
            total_bpt_supply="100",
            bpt_amount_in="50",
        )
        assert amounts == [500, 1000]

    def test_exit_zero_total_supply_raises(self) -> None:
        """Test exit with zero total supply raises."""
        with pytest.raises(ValueError, match="Total BPT supply must be positive"):
            BalancerProportionalMath.query_proportional_exit(
                token_balances=[1000],
                total_bpt_supply=0,
                bpt_amount_in=10,
            )

    def test_exit_zero_bpt_amount_raises(self) -> None:
        """Test exit with zero BPT amount raises."""
        with pytest.raises(ValueError, match="BPT amount must be positive"):
            BalancerProportionalMath.query_proportional_exit(
                token_balances=[1000],
                total_bpt_supply=100,
                bpt_amount_in=0,
            )

    def test_exit_more_than_supply_raises(self) -> None:
        """Test exit with more than total supply raises."""
        with pytest.raises(ValueError, match="Cannot burn more BPT"):
            BalancerProportionalMath.query_proportional_exit(
                token_balances=[1000],
                total_bpt_supply=100,
                bpt_amount_in=200,
            )


class TestBalancerProportionalMathJoin:
    """Test BalancerProportionalMath.query_proportional_join."""

    def test_basic_join(self) -> None:
        """Test basic proportional join calculation."""
        bpt_out = BalancerProportionalMath.query_proportional_join(
            token_balances=[1000, 2000],
            total_bpt_supply=100,
            amounts_in=[100, 200],
        )
        assert bpt_out == 10

    def test_join_limiting_factor(self) -> None:
        """Test join uses minimum ratio (limiting factor)."""
        bpt_out = BalancerProportionalMath.query_proportional_join(
            token_balances=[1000, 2000],
            total_bpt_supply=100,
            amounts_in=[100, 100],  # second token is limiting
        )
        assert bpt_out == 5  # 100/2000 * 100

    def test_join_with_zero_amount(self) -> None:
        """Test join with one zero amount."""
        bpt_out = BalancerProportionalMath.query_proportional_join(
            token_balances=[1000, 2000],
            total_bpt_supply=100,
            amounts_in=[100, 0],
        )
        assert bpt_out == 10

    def test_join_mismatched_lengths_raises(self) -> None:
        """Test join with mismatched lengths raises."""
        with pytest.raises(ValueError, match="must match number of tokens"):
            BalancerProportionalMath.query_proportional_join(
                token_balances=[1000, 2000],
                total_bpt_supply=100,
                amounts_in=[100],
            )

    def test_join_zero_total_supply_raises(self) -> None:
        """Test join with zero total supply raises."""
        with pytest.raises(ValueError, match="Total BPT supply must be positive"):
            BalancerProportionalMath.query_proportional_join(
                token_balances=[1000],
                total_bpt_supply=0,
                amounts_in=[100],
            )

    def test_join_all_zero_amounts_raises(self) -> None:
        """Test join with all zero amounts raises."""
        with pytest.raises(
            ValueError, match="At least one token amount must be positive"
        ):
            BalancerProportionalMath.query_proportional_join(
                token_balances=[1000, 2000],
                total_bpt_supply=100,
                amounts_in=[0, 0],
            )

    def test_join_negative_amount_with_zero_balance_raises(self) -> None:
        """Test join with negative amount and zero balance raises."""
        with pytest.raises(ValueError, match="Invalid amount for token"):
            BalancerProportionalMath.query_proportional_join(
                token_balances=[0],
                total_bpt_supply=100,
                amounts_in=[100],
            )

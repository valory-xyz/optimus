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

"""Yield calculation tests for Balancer protocol."""

import pytest
from decimal import Decimal

from packages.valory.skills.liquidity_trader_abci.pools.balancer import BalancerPoolBehaviour

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions, TestDataGenerator
from tests.integration.fixtures.pool_data_fixtures import (
    balancer_weighted_pool_data,
    balancer_stable_pool_data,
)


class TestBalancerYieldCalculations(ProtocolIntegrationTestBase):
    """Test accurate yield and APR calculations for Balancer."""

    def test_balancer_apr_calculation_accuracy(self, balancer_weighted_pool_data):
        """Test accurate APR calculation for Balancer pools."""
        # Test data with known values
        pool_data = balancer_weighted_pool_data.copy()
        pool_data.update({
            "total_supply": 1000000000000000000000,  # 1000 BPT
            "swap_fees": 2500000000000000,  # 0.25%
            "daily_volume": 10000000000000000000000,  # 10,000 tokens
            "pool_tvl": 5000000000000000000000  # 5,000 tokens
        })
        
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Calculate expected APR
        daily_fees = pool_data["daily_volume"] * pool_data["swap_fees"] / 1e18
        annual_fees = daily_fees * 365
        expected_apr = (annual_fees / pool_data["pool_tvl"]) * 100
        
        # Test APR calculation
        calculated_apr = behaviour._calculate_balancer_apr(pool_data)
        
        # Verify calculation accuracy (within 0.01% tolerance)
        TestAssertions.assert_apr_calculation_accuracy(calculated_apr, expected_apr, 0.01)

    def test_balancer_yield_compounding_calculation(self):
        """Test yield compounding calculations."""
        # Initial investment
        initial_investment = 1000000000000000000000  # 1000 tokens
        daily_yield_rate = 0.001  # 0.1% daily
        
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Calculate compounded yield over 30 days
        compounded_value = behaviour._calculate_compounded_yield(
            initial_investment, daily_yield_rate, 30
        )
        
        # Expected value: 1000 * (1.001)^30
        expected_value = initial_investment * (1 + daily_yield_rate) ** 30
        
        # Verify calculation accuracy
        TestAssertions.assert_yield_calculation_accuracy(compounded_value, expected_value, 0.001)

    def test_balancer_impermanent_loss_calculation(self):
        """Test impermanent loss calculation."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test scenarios
        test_cases = [
            {"initial_ratio": 1.0, "final_ratio": 1.0, "expected_il": 0.0},      # No price change
            {"initial_ratio": 1.0, "final_ratio": 2.0, "expected_il": 5.72},     # 100% price change
            {"initial_ratio": 1.0, "final_ratio": 0.5, "expected_il": 5.72},     # -50% price change
            {"initial_ratio": 1.0, "final_ratio": 1.5, "expected_il": 1.25},     # 50% price change
            {"initial_ratio": 1.0, "final_ratio": 1.1, "expected_il": 0.11},     # 10% price change
        ]
        
        for case in test_cases:
            initial_ratio = case["initial_ratio"]
            final_ratio = case["final_ratio"]
            expected_il = case["expected_il"]
            
            calculated_il = behaviour._calculate_impermanent_loss(initial_ratio, final_ratio)
            
            # Allow 0.1% tolerance
            TestAssertions.assert_yield_calculation_accuracy(calculated_il, expected_il, 0.1)

    def test_balancer_fee_accumulation_tracking(self):
        """Test fee accumulation tracking over time."""
        # Pool state over time
        initial_state = {
            "total_supply": 1000000000000000000000,
            "pool_tvl": 5000000000000000000000,
            "cumulative_fees": 0
        }
        
        # Simulate 7 days of trading
        daily_volume = 1000000000000000000000  # 1000 tokens daily
        swap_fee_rate = 2500000000000000  # 0.25%
        
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        current_state = initial_state.copy()
        for day in range(7):
            daily_fees = daily_volume * swap_fee_rate / 1e18
            current_state = behaviour._update_fee_accumulation(
                current_state, daily_fees
            )
        
        # Verify fee accumulation
        expected_total_fees = daily_volume * swap_fee_rate * 7 / 1e18
        TestAssertions.assert_yield_calculation_accuracy(
            current_state["cumulative_fees"], expected_total_fees, 0.001
        )

    def test_balancer_pool_share_calculation(self):
        """Test pool share calculation accuracy."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test data
        user_bpt_balance = 100000000000000000000   # 100 BPT
        total_supply = 1000000000000000000000      # 1000 BPT
        pool_tvl = 5000000000000000000000          # 5000 tokens
        
        # Calculate user's share
        user_share = behaviour._calculate_pool_share(
            user_bpt_balance, total_supply, pool_tvl
        )
        
        # Expected share: (100/1000) * 5000 = 500 tokens
        expected_share = (user_bpt_balance / total_supply) * pool_tvl
        
        TestAssertions.assert_yield_calculation_accuracy(user_share, expected_share, 0.001)

    def test_balancer_apr_with_different_pool_types(self):
        """Test APR calculation for different pool types."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test weighted pool
        weighted_pool_data = TestDataGenerator.generate_balancer_pool_data(
            pool_type="Weighted"
        )
        weighted_pool_data.update({
            "swap_fees": 2500000000000000,  # 0.25%
            "daily_volume": 10000000000000000000000,
            "pool_tvl": 5000000000000000000000
        })
        
        weighted_apr = behaviour._calculate_balancer_apr(weighted_pool_data)
        assert weighted_apr > 0
        
        # Test stable pool
        stable_pool_data = TestDataGenerator.generate_balancer_pool_data(
            pool_type="ComposableStable"
        )
        stable_pool_data.update({
            "swap_fees": 1000000000000000,  # 0.1%
            "daily_volume": 5000000000000000000000,
            "pool_tvl": 10000000000000000000000
        })
        
        stable_apr = behaviour._calculate_balancer_apr(stable_pool_data)
        assert stable_apr > 0
        
        # Stable pools typically have lower fees but higher volume
        # APR should be calculated correctly for both types

    def test_balancer_yield_with_compounding_frequency(self):
        """Test yield calculation with different compounding frequencies."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test data
        initial_investment = 1000000000000000000000  # 1000 tokens
        annual_rate = 0.12  # 12% annual
        time_periods = [30, 90, 365]  # 30 days, 90 days, 1 year
        
        for days in time_periods:
            # Daily compounding
            daily_rate = annual_rate / 365
            compounded_daily = behaviour._calculate_compounded_yield(
                initial_investment, daily_rate, days
            )
            
            # Expected: 1000 * (1 + 0.12/365)^days
            expected_daily = initial_investment * (1 + daily_rate) ** days
            
            TestAssertions.assert_yield_calculation_accuracy(
                compounded_daily, expected_daily, 0.001
            )

    def test_balancer_fee_distribution_calculation(self):
        """Test fee distribution calculation among pool participants."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test data
        total_fees = 100000000000000000000  # 100 tokens
        user_bpt_balance = 100000000000000000000   # 100 BPT
        total_supply = 1000000000000000000000      # 1000 BPT
        
        # Calculate user's share of fees
        user_fee_share = behaviour._calculate_fee_distribution(
            total_fees, user_bpt_balance, total_supply
        )
        
        # Expected: (100/1000) * 100 = 10 tokens
        expected_share = (user_bpt_balance / total_supply) * total_fees
        
        TestAssertions.assert_yield_calculation_accuracy(user_fee_share, expected_share, 0.001)

    def test_balancer_apr_with_impermanent_loss_adjustment(self):
        """Test APR calculation adjusted for impermanent loss."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test data
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        pool_data.update({
            "swap_fees": 2500000000000000,  # 0.25%
            "daily_volume": 10000000000000000000000,
            "pool_tvl": 5000000000000000000000
        })
        
        # Calculate base APR
        base_apr = behaviour._calculate_balancer_apr(pool_data)
        
        # Calculate IL-adjusted APR with 5% impermanent loss
        impermanent_loss = 5.0  # 5%
        adjusted_apr = behaviour._calculate_il_adjusted_apr(base_apr, impermanent_loss)
        
        # Adjusted APR should be lower than base APR
        assert adjusted_apr < base_apr
        
        # The difference should account for the impermanent loss
        expected_adjustment = base_apr * (1 - impermanent_loss / 100)
        TestAssertions.assert_apr_calculation_accuracy(adjusted_apr, expected_adjustment, 0.1)

    def test_balancer_yield_with_rebalancing_costs(self):
        """Test yield calculation including rebalancing costs."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test data
        base_yield = 12.0  # 12% base yield
        rebalancing_frequency = 30  # days
        rebalancing_cost = 0.5  # 0.5% cost per rebalancing
        
        # Calculate yield with rebalancing costs
        net_yield = behaviour._calculate_net_yield_with_rebalancing_costs(
            base_yield, rebalancing_frequency, rebalancing_cost
        )
        
        # Expected: base_yield - (rebalancing_cost * 365 / rebalancing_frequency)
        expected_net_yield = base_yield - (rebalancing_cost * 365 / rebalancing_frequency)
        
        TestAssertions.assert_apr_calculation_accuracy(net_yield, expected_net_yield, 0.1)

    def test_balancer_compound_interest_calculation(self):
        """Test compound interest calculation for different scenarios."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test scenarios
        test_cases = [
            {
                "principal": 1000000000000000000000,  # 1000 tokens
                "rate": 0.12,  # 12% annual
                "time": 365,   # 1 year
                "compounding_frequency": 365,  # Daily
            },
            {
                "principal": 1000000000000000000000,  # 1000 tokens
                "rate": 0.12,  # 12% annual
                "time": 180,   # 6 months
                "compounding_frequency": 365,  # Daily
            },
            {
                "principal": 1000000000000000000000,  # 1000 tokens
                "rate": 0.06,  # 6% annual
                "time": 365,   # 1 year
                "compounding_frequency": 365,  # Daily
            },
        ]
        
        for case in test_cases:
            principal = case["principal"]
            rate = case["rate"]
            time = case["time"]
            frequency = case["compounding_frequency"]
            
            # Calculate compound interest
            compound_amount = behaviour._calculate_compound_interest(
                principal, rate, time, frequency
            )
            
            # Expected: principal * (1 + rate/frequency)^(frequency * time/365)
            expected_amount = principal * (1 + rate / frequency) ** (frequency * time / 365)
            
            TestAssertions.assert_yield_calculation_accuracy(
                compound_amount, expected_amount, 0.001
            )

    def test_balancer_apr_comparison_across_pools(self):
        """Test APR comparison across different pools."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Create multiple pools with different characteristics
        pools = [
            {
                "pool_id": "pool1",
                "swap_fees": 2500000000000000,  # 0.25%
                "daily_volume": 10000000000000000000000,  # 10,000 tokens
                "pool_tvl": 5000000000000000000000,  # 5,000 tokens
                "pool_type": "Weighted"
            },
            {
                "pool_id": "pool2",
                "swap_fees": 1000000000000000,  # 0.1%
                "daily_volume": 20000000000000000000000,  # 20,000 tokens
                "pool_tvl": 10000000000000000000000,  # 10,000 tokens
                "pool_type": "Stable"
            },
            {
                "pool_id": "pool3",
                "swap_fees": 5000000000000000,  # 0.5%
                "daily_volume": 5000000000000000000000,  # 5,000 tokens
                "pool_tvl": 2000000000000000000000,  # 2,000 tokens
                "pool_type": "Weighted"
            },
        ]
        
        # Calculate APR for each pool
        aprs = []
        for pool in pools:
            apr = behaviour._calculate_balancer_apr(pool)
            aprs.append((pool["pool_id"], apr))
        
        # Sort by APR
        aprs.sort(key=lambda x: x[1], reverse=True)
        
        # Verify APR calculation and ranking
        assert len(aprs) == 3
        assert aprs[0][1] > aprs[1][1] > aprs[2][1]  # Should be in descending order
        
        # Pool 3 should have the highest APR (highest fee rate and lowest TVL)
        assert aprs[0][0] == "pool3"

    def test_balancer_yield_with_slippage_impact(self):
        """Test yield calculation including slippage impact."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test data
        base_yield = 15.0  # 15% base yield
        trade_size = 1000000000000000000000  # 1000 tokens
        pool_tvl = 10000000000000000000000   # 10,000 tokens
        slippage_rate = 0.001  # 0.1% slippage
        
        # Calculate slippage impact
        slippage_impact = behaviour._calculate_slippage_impact(
            trade_size, pool_tvl, slippage_rate
        )
        
        # Calculate yield adjusted for slippage
        adjusted_yield = behaviour._calculate_slippage_adjusted_yield(
            base_yield, slippage_impact
        )
        
        # Adjusted yield should be lower than base yield
        assert adjusted_yield < base_yield
        
        # The difference should account for the slippage impact
        expected_adjustment = base_yield * (1 - slippage_impact)
        TestAssertions.assert_apr_calculation_accuracy(adjusted_yield, expected_adjustment, 0.1)

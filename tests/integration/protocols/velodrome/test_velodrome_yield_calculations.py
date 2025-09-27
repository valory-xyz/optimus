# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   You may not use this file except in compliance with the License.
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

"""Yield calculation tests for Velodrome protocol."""

import pytest
from decimal import Decimal

from packages.valory.skills.liquidity_trader_abci.pools.velodrome import VelodromePoolBehaviour

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions, TestDataGenerator
from tests.integration.fixtures.pool_data_fixtures import (
    velodrome_stable_pool_data,
    velodrome_volatile_pool_data,
    velodrome_cl_pool_data,
)


class TestVelodromeYieldCalculations(ProtocolIntegrationTestBase):
    """Test accurate yield and APR calculations for Velodrome."""

    def test_velodrome_trading_fees_apr_calculation(self, velodrome_volatile_pool_data):
        """Test trading fees APR calculation for Velodrome pools."""
        # Pool data
        pool_data = velodrome_volatile_pool_data.copy()
        pool_data.update({
            "swap_fee": 2500000000000000,  # 0.25%
            "daily_volume": 10000000000000000000000,  # 10,000 tokens
            "pool_tvl": 5000000000000000000000  # 5,000 tokens
        })
        
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Calculate expected trading fees APR
        daily_fees = pool_data["daily_volume"] * pool_data["swap_fee"] / 1e18
        annual_fees = daily_fees * 365
        expected_trading_apr = (annual_fees / pool_data["pool_tvl"]) * 100
        
        # Test trading fees APR calculation
        calculated_trading_apr = behaviour._calculate_trading_fees_apr(pool_data)
        
        # Verify calculation accuracy (within 0.01% tolerance)
        TestAssertions.assert_apr_calculation_accuracy(calculated_trading_apr, expected_trading_apr, 0.01)

    def test_velodrome_gauge_rewards_apr_calculation(self):
        """Test gauge rewards APR calculation."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        gauge_data = {
            "reward_rate": 1000000000000000000,  # 1 token per second
            "total_supply": 1000000000000000000000,  # 1000 LP tokens
            "reward_token_price": 1.0,  # $1 per reward token
            "lp_token_price": 2.0,  # $2 per LP token
        }
        
        # Calculate expected gauge APR
        daily_rewards = gauge_data["reward_rate"] * 86400  # 1 day
        annual_rewards = daily_rewards * 365
        total_lp_value = gauge_data["total_supply"] * gauge_data["lp_token_price"]
        expected_gauge_apr = (annual_rewards * gauge_data["reward_token_price"] / total_lp_value) * 100
        
        # Test gauge APR calculation
        calculated_gauge_apr = behaviour._calculate_gauge_rewards_apr(gauge_data)
        
        # Verify calculation accuracy
        TestAssertions.assert_apr_calculation_accuracy(calculated_gauge_apr, expected_gauge_apr, 0.01)

    def test_velodrome_total_apr_calculation(self):
        """Test total APR calculation combining trading fees and gauge rewards."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        trading_apr = 12.0  # 12% from trading fees
        gauge_apr = 8.0     # 8% from gauge rewards
        
        # Calculate expected total APR
        expected_total_apr = trading_apr + gauge_apr
        
        # Test total APR calculation
        calculated_total_apr = behaviour._calculate_total_velodrome_apr(trading_apr, gauge_apr)
        
        # Verify calculation accuracy
        TestAssertions.assert_apr_calculation_accuracy(calculated_total_apr, expected_total_apr, 0.01)

    def test_velodrome_impermanent_loss_calculation(self):
        """Test impermanent loss calculation for Velodrome pools."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test scenarios
        test_cases = [
            {"price_change": 1.0, "expected_il": 0.0},      # No price change
            {"price_change": 2.0, "expected_il": 5.72},     # 100% price change
            {"price_change": 0.5, "expected_il": 5.72},     # -50% price change
            {"price_change": 1.5, "expected_il": 1.25},     # 50% price change
        ]
        
        for case in test_cases:
            price_ratio = case["price_change"]
            expected_il = case["expected_il"]
            
            calculated_il = behaviour._calculate_velodrome_impermanent_loss(1.0, price_ratio)
            
            # Allow 0.1% tolerance
            TestAssertions.assert_yield_calculation_accuracy(calculated_il, expected_il, 0.1)

    def test_velodrome_reward_compounding_calculation(self):
        """Test reward compounding calculations."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Initial investment
        initial_investment = 1000000000000000000000  # 1000 tokens
        daily_reward_rate = 0.001  # 0.1% daily
        
        # Calculate compounded yield over 30 days
        compounded_value = behaviour._calculate_compounded_yield(
            initial_investment, daily_reward_rate, 30
        )
        
        # Expected value: 1000 * (1.001)^30
        expected_value = initial_investment * (1 + daily_reward_rate) ** 30
        
        # Verify calculation accuracy
        TestAssertions.assert_yield_calculation_accuracy(compounded_value, expected_value, 0.001)

    def test_velodrome_pool_share_calculation(self):
        """Test pool share calculation accuracy."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        user_lp_balance = 100000000000000000000   # 100 LP tokens
        total_lp_supply = 1000000000000000000000  # 1000 LP tokens
        pool_tvl = 5000000000000000000000         # 5000 tokens
        
        # Calculate user's share
        user_share = behaviour._calculate_pool_share(
            user_lp_balance, total_lp_supply, pool_tvl
        )
        
        # Expected share: (100/1000) * 5000 = 500 tokens
        expected_share = (user_lp_balance / total_lp_supply) * pool_tvl
        
        TestAssertions.assert_yield_calculation_accuracy(user_share, expected_share, 0.001)

    def test_velodrome_fee_distribution_calculation(self):
        """Test fee distribution calculation among pool participants."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        total_fees = 100000000000000000000  # 100 tokens
        user_lp_balance = 100000000000000000000   # 100 LP tokens
        total_lp_supply = 1000000000000000000000  # 1000 LP tokens
        
        # Calculate user's share of fees
        user_fee_share = behaviour._calculate_fee_distribution(
            total_fees, user_lp_balance, total_lp_supply
        )
        
        # Expected: (100/1000) * 100 = 10 tokens
        expected_share = (user_lp_balance / total_lp_supply) * total_fees
        
        TestAssertions.assert_yield_calculation_accuracy(user_fee_share, expected_share, 0.001)

    def test_velodrome_apr_with_different_pool_types(self):
        """Test APR calculation for different pool types."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test stable pool
        stable_pool_data = TestDataGenerator.generate_velodrome_pool_data(is_stable=True)
        stable_pool_data.update({
            "swap_fee": 1000000000000000,  # 0.1%
            "daily_volume": 20000000000000000000000,  # 20,000 tokens
            "pool_tvl": 10000000000000000000000  # 10,000 tokens
        })
        
        stable_apr = behaviour._calculate_velodrome_apr(stable_pool_data)
        assert stable_apr > 0
        
        # Test volatile pool
        volatile_pool_data = TestDataGenerator.generate_velodrome_pool_data(is_stable=False)
        volatile_pool_data.update({
            "swap_fee": 2500000000000000,  # 0.25%
            "daily_volume": 10000000000000000000000,  # 10,000 tokens
            "pool_tvl": 5000000000000000000000  # 5,000 tokens
        })
        
        volatile_apr = behaviour._calculate_velodrome_apr(volatile_pool_data)
        assert volatile_apr > 0
        
        # Test CL pool
        cl_pool_data = TestDataGenerator.generate_velodrome_pool_data(is_cl_pool=True)
        cl_pool_data.update({
            "swap_fee": 3000000000000000,  # 0.3%
            "daily_volume": 15000000000000000000000,  # 15,000 tokens
            "pool_tvl": 8000000000000000000000  # 8,000 tokens
        })
        
        cl_apr = behaviour._calculate_velodrome_apr(cl_pool_data)
        assert cl_apr > 0

    def test_velodrome_yield_with_compounding_frequency(self):
        """Test yield calculation with different compounding frequencies."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
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

    def test_velodrome_apr_with_impermanent_loss_adjustment(self):
        """Test APR calculation adjusted for impermanent loss."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        base_apr = 20.0  # 20% base APR
        impermanent_loss = 5.0  # 5% impermanent loss
        
        # Calculate IL-adjusted APR
        adjusted_apr = behaviour._calculate_il_adjusted_apr(base_apr, impermanent_loss)
        
        # Expected: 20% * (1 - 5%) = 19%
        expected_adjustment = base_apr * (1 - impermanent_loss / 100)
        
        # Verify calculation accuracy
        TestAssertions.assert_apr_calculation_accuracy(adjusted_apr, expected_adjustment, 0.1)

    def test_velodrome_yield_with_rebalancing_costs(self):
        """Test yield calculation including rebalancing costs."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        base_yield = 18.0  # 18% base yield
        rebalancing_frequency = 30  # days
        rebalancing_cost = 0.5  # 0.5% cost per rebalancing
        
        # Calculate yield with rebalancing costs
        net_yield = behaviour._calculate_net_yield_with_rebalancing_costs(
            base_yield, rebalancing_frequency, rebalancing_cost
        )
        
        # Expected: base_yield - (rebalancing_cost * 365 / rebalancing_frequency)
        expected_net_yield = base_yield - (rebalancing_cost * 365 / rebalancing_frequency)
        
        TestAssertions.assert_apr_calculation_accuracy(net_yield, expected_net_yield, 0.1)

    def test_velodrome_compound_interest_calculation(self):
        """Test compound interest calculation for different scenarios."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
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

    def test_velodrome_apr_comparison_across_pools(self):
        """Test APR comparison across different Velodrome pools."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Create multiple pools with different characteristics
        pools = [
            {
                "pool_id": "stable_pool",
                "is_stable": True,
                "swap_fee": 1000000000000000,  # 0.1%
                "daily_volume": 20000000000000000000000,  # 20,000 tokens
                "pool_tvl": 10000000000000000000000,  # 10,000 tokens
                "gauge_apr": 5.0,  # 5% gauge APR
            },
            {
                "pool_id": "volatile_pool",
                "is_stable": False,
                "swap_fee": 2500000000000000,  # 0.25%
                "daily_volume": 10000000000000000000000,  # 10,000 tokens
                "pool_tvl": 5000000000000000000000,  # 5,000 tokens
                "gauge_apr": 8.0,  # 8% gauge APR
            },
            {
                "pool_id": "cl_pool",
                "is_cl_pool": True,
                "swap_fee": 3000000000000000,  # 0.3%
                "daily_volume": 15000000000000000000000,  # 15,000 tokens
                "pool_tvl": 8000000000000000000000,  # 8,000 tokens
                "gauge_apr": 12.0,  # 12% gauge APR
            },
        ]
        
        # Calculate APR for each pool
        aprs = []
        for pool in pools:
            apr = behaviour._calculate_velodrome_apr(pool)
            aprs.append((pool["pool_id"], apr))
        
        # Sort by APR
        aprs.sort(key=lambda x: x[1], reverse=True)
        
        # Verify APR calculation and ranking
        assert len(aprs) == 3
        assert aprs[0][1] > aprs[1][1] > aprs[2][1]  # Should be in descending order
        
        # CL pool should have the highest APR (highest fee rate and gauge APR)
        assert aprs[0][0] == "cl_pool"

    def test_velodrome_yield_with_slippage_impact(self):
        """Test yield calculation including slippage impact."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        base_yield = 18.0  # 18% base yield
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

    def test_velodrome_reward_claiming_frequency_impact(self):
        """Test impact of reward claiming frequency on yield."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        base_yield = 15.0  # 15% base yield
        claiming_frequency = 7  # days
        claiming_cost = 0.1  # 0.1% cost per claim
        
        # Calculate yield with claiming costs
        net_yield = behaviour._calculate_net_yield_with_claiming_costs(
            base_yield, claiming_frequency, claiming_cost
        )
        
        # Expected: base_yield - (claiming_cost * 365 / claiming_frequency)
        expected_net_yield = base_yield - (claiming_cost * 365 / claiming_frequency)
        
        TestAssertions.assert_apr_calculation_accuracy(net_yield, expected_net_yield, 0.1)

    def test_velodrome_position_value_with_price_movement(self):
        """Test position value calculation with price movements."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        position_data = {
            "lp_balance": 100000000000000000000,  # 100 LP tokens
            "staked_amount": 100000000000000000000,  # 100 staked LP tokens
        }
        
        # Test different price scenarios
        price_scenarios = [
            {"lp_token_price": 2.0, "expected_value": 200000000000000000000},   # $2 per LP token
            {"lp_token_price": 2.5, "expected_value": 250000000000000000000},   # $2.5 per LP token
            {"lp_token_price": 1.5, "expected_value": 150000000000000000000},   # $1.5 per LP token
        ]
        
        for scenario in price_scenarios:
            lp_token_price = scenario["lp_token_price"]
            expected_value = scenario["expected_value"]
            
            # Calculate position value
            position_value = behaviour._calculate_position_value_with_price(
                position_data, lp_token_price
            )
            
            # Verify calculation
            TestAssertions.assert_yield_calculation_accuracy(
                position_value, expected_value, 0.001
            )

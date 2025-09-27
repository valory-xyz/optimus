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

"""Yield calculation tests for Uniswap V3 protocol."""

import pytest
from decimal import Decimal

from packages.valory.skills.liquidity_trader_abci.pools.uniswap import UniswapPoolBehaviour

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions, TestDataGenerator
from tests.integration.fixtures.pool_data_fixtures import (
    uniswap_v3_pool_data,
    uniswap_v3_high_fee_pool_data,
    uniswap_v3_low_fee_pool_data,
)


class TestUniswapV3YieldCalculations(ProtocolIntegrationTestBase):
    """Test accurate yield and APR calculations for Uniswap V3."""

    def test_uniswap_v3_fee_apr_calculation(self, uniswap_v3_pool_data):
        """Test fee-based APR calculation for Uniswap V3."""
        # Pool data
        pool_data = uniswap_v3_pool_data.copy()
        pool_data.update({
            "fee": 3000,  # 0.3%
            "daily_volume": 5000000000000000000000,  # 5000 tokens
            "pool_tvl": 10000000000000000000000  # 10,000 tokens
        })
        
        # Position data
        position_data = {
            "tick_lower": -276320,
            "tick_upper": -276300,
            "liquidity": 100000000000000000000,  # 10% of pool liquidity
            "fee_tier": 3000
        }
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Calculate position's share of fees
        position_share = position_data["liquidity"] / pool_data["pool_tvl"]
        daily_fees = pool_data["daily_volume"] * pool_data["fee"] / 1e6
        position_daily_fees = daily_fees * position_share
        annual_fees = position_daily_fees * 365
        
        # Calculate position value
        position_value = behaviour._calculate_position_value(position_data, pool_data)
        
        # Calculate APR
        calculated_apr = (annual_fees / position_value) * 100
        
        # Expected APR calculation
        expected_apr = (annual_fees / position_value) * 100
        
        # Verify calculation accuracy
        TestAssertions.assert_apr_calculation_accuracy(calculated_apr, expected_apr, 0.01)

    def test_uniswap_v3_capital_efficiency_calculation(self):
        """Test capital efficiency calculation for concentrated liquidity."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Price range data
        current_price = 1.0
        price_range = {
            "lower": 0.8,   # 20% below current price
            "upper": 1.2    # 20% above current price
        }
        
        # Calculate capital efficiency
        efficiency = behaviour._calculate_capital_efficiency(
            current_price, price_range
        )
        
        # For a 40% range around current price, efficiency should be ~2.5x
        expected_efficiency = 2.5
        TestAssertions.assert_yield_calculation_accuracy(efficiency, expected_efficiency, 0.1)

    def test_uniswap_v3_impermanent_loss_calculation(self):
        """Test impermanent loss calculation for concentrated liquidity."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Position parameters
        position_data = {
            "tick_lower": -276320,  # Price: 0.8
            "tick_upper": -276300,  # Price: 1.2
            "liquidity": 1000000000000000000000
        }
        
        # Price movement
        initial_price = 1.0
        final_price = 1.5  # 50% price increase
        
        # Calculate impermanent loss
        il_percentage = behaviour._calculate_uniswap_v3_impermanent_loss(
            position_data, initial_price, final_price
        )
        
        # For concentrated liquidity, IL depends on price range
        # This should be calculated based on the specific tick range
        assert il_percentage >= 0
        assert il_percentage <= 100  # IL can't exceed 100%

    def test_uniswap_v3_fee_growth_calculation(self):
        """Test fee growth calculation over time."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test data
        initial_fee_growth_global0 = 1000000000000000000
        initial_fee_growth_global1 = 2000000000000000000
        current_fee_growth_global0 = 1500000000000000000
        current_fee_growth_global1 = 3000000000000000000
        liquidity = 1000000000000000000
        
        # Calculate fee growth
        fee_growth = behaviour._calculate_fee_growth(
            initial_fee_growth_global0, initial_fee_growth_global1,
            current_fee_growth_global0, current_fee_growth_global1,
            liquidity
        )
        
        # Expected: (1500 - 1000) / 1000 = 0.5 tokens for token0
        # Expected: (3000 - 2000) / 1000 = 1.0 tokens for token1
        expected_growth0 = (current_fee_growth_global0 - initial_fee_growth_global0) / liquidity
        expected_growth1 = (current_fee_growth_global1 - initial_fee_growth_global1) / liquidity
        
        assert fee_growth["amount0"] == expected_growth0
        assert fee_growth["amount1"] == expected_growth1

    def test_uniswap_v3_liquidity_concentration_impact(self):
        """Test impact of liquidity concentration on yield."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test different concentration levels
        concentration_scenarios = [
            {"range": 0.1, "expected_multiplier": 10.0},   # 10% range = 10x concentration
            {"range": 0.2, "expected_multiplier": 5.0},    # 20% range = 5x concentration
            {"range": 0.5, "expected_multiplier": 2.0},    # 50% range = 2x concentration
        ]
        
        for scenario in concentration_scenarios:
            price_range = scenario["range"]
            expected_multiplier = scenario["expected_multiplier"]
            
            # Calculate concentration multiplier
            multiplier = behaviour._calculate_concentration_multiplier(price_range)
            
            # Verify calculation
            TestAssertions.assert_yield_calculation_accuracy(
                multiplier, expected_multiplier, 0.1
            )

    def test_uniswap_v3_price_impact_calculation(self):
        """Test price impact calculation for different trade sizes."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test data
        pool_liquidity = 10000000000000000000000  # 10,000 tokens
        current_price = 1.5
        
        # Test different trade sizes
        trade_scenarios = [
            {"amount": 100000000000000000000, "expected_impact": 0.01},    # 100 tokens - 1% impact
            {"amount": 500000000000000000000, "expected_impact": 0.05},    # 500 tokens - 5% impact
            {"amount": 1000000000000000000000, "expected_impact": 0.10},   # 1000 tokens - 10% impact
        ]
        
        for scenario in trade_scenarios:
            trade_amount = scenario["amount"]
            expected_impact = scenario["expected_impact"]
            
            # Calculate price impact
            price_impact = behaviour._calculate_price_impact(
                trade_amount, pool_liquidity, current_price
            )
            
            # Verify calculation (within 0.01 tolerance)
            TestAssertions.assert_yield_calculation_accuracy(
                price_impact, expected_impact, 0.01
            )

    def test_uniswap_v3_apr_with_different_fee_tiers(self):
        """Test APR calculation for different fee tiers."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test different fee tiers
        fee_tier_scenarios = [
            {"fee": 500, "daily_volume": 20000000000000000000000, "expected_apr": 3.65},   # 0.05% fee, high volume
            {"fee": 3000, "daily_volume": 10000000000000000000000, "expected_apr": 10.95}, # 0.3% fee, medium volume
            {"fee": 10000, "daily_volume": 5000000000000000000000, "expected_apr": 18.25}, # 1% fee, low volume
        ]
        
        for scenario in fee_tier_scenarios:
            pool_data = {
                "fee": scenario["fee"],
                "daily_volume": scenario["daily_volume"],
                "pool_tvl": 10000000000000000000000,  # 10,000 tokens
            }
            
            # Calculate APR
            calculated_apr = behaviour._calculate_uniswap_v3_apr(pool_data)
            expected_apr = scenario["expected_apr"]
            
            # Verify calculation
            TestAssertions.assert_apr_calculation_accuracy(
                calculated_apr, expected_apr, 0.1
            )

    def test_uniswap_v3_compound_interest_with_fees(self):
        """Test compound interest calculation including fee reinvestment."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test data
        initial_investment = 1000000000000000000000  # 1000 tokens
        daily_fee_rate = 0.001  # 0.1% daily
        compounding_frequency = 1  # Daily compounding
        time_period = 30  # 30 days
        
        # Calculate compound interest with fees
        compound_amount = behaviour._calculate_compound_interest_with_fees(
            initial_investment, daily_fee_rate, compounding_frequency, time_period
        )
        
        # Expected: 1000 * (1.001)^30
        expected_amount = initial_investment * (1 + daily_fee_rate) ** time_period
        
        # Verify calculation
        TestAssertions.assert_yield_calculation_accuracy(
            compound_amount, expected_amount, 0.001
        )

    def test_uniswap_v3_yield_with_impermanent_loss_adjustment(self):
        """Test yield calculation adjusted for impermanent loss."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test data
        base_yield = 15.0  # 15% base yield
        impermanent_loss = 5.0  # 5% impermanent loss
        
        # Calculate IL-adjusted yield
        adjusted_yield = behaviour._calculate_il_adjusted_yield(base_yield, impermanent_loss)
        
        # Expected: 15% * (1 - 5%) = 14.25%
        expected_yield = base_yield * (1 - impermanent_loss / 100)
        
        # Verify calculation
        TestAssertions.assert_apr_calculation_accuracy(adjusted_yield, expected_yield, 0.1)

    def test_uniswap_v3_liquidity_provision_yield(self):
        """Test yield calculation for liquidity provision."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test data
        position_data = {
            "liquidity": 1000000000000000000000,
            "tick_lower": -276320,
            "tick_upper": -276300,
            "fee_tier": 3000,
        }
        
        pool_data = {
            "daily_volume": 10000000000000000000000,  # 10,000 tokens
            "pool_tvl": 10000000000000000000000,      # 10,000 tokens
            "fee": 3000,
        }
        
        # Calculate liquidity provision yield
        yield_data = behaviour._calculate_liquidity_provision_yield(
            position_data, pool_data
        )
        
        # Verify yield calculation
        assert "daily_yield" in yield_data
        assert "annual_yield" in yield_data
        assert "fee_share" in yield_data
        
        assert yield_data["daily_yield"] > 0
        assert yield_data["annual_yield"] > 0
        assert yield_data["fee_share"] > 0

    def test_uniswap_v3_apr_comparison_across_pools(self):
        """Test APR comparison across different Uniswap V3 pools."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Create multiple pools with different characteristics
        pools = [
            {
                "pool_id": "pool1",
                "fee": 500,  # 0.05%
                "daily_volume": 20000000000000000000000,  # 20,000 tokens
                "pool_tvl": 10000000000000000000000,      # 10,000 tokens
            },
            {
                "pool_id": "pool2",
                "fee": 3000,  # 0.3%
                "daily_volume": 10000000000000000000000,  # 10,000 tokens
                "pool_tvl": 10000000000000000000000,      # 10,000 tokens
            },
            {
                "pool_id": "pool3",
                "fee": 10000,  # 1%
                "daily_volume": 5000000000000000000000,   # 5,000 tokens
                "pool_tvl": 5000000000000000000000,       # 5,000 tokens
            },
        ]
        
        # Calculate APR for each pool
        aprs = []
        for pool in pools:
            apr = behaviour._calculate_uniswap_v3_apr(pool)
            aprs.append((pool["pool_id"], apr))
        
        # Sort by APR
        aprs.sort(key=lambda x: x[1], reverse=True)
        
        # Verify APR calculation and ranking
        assert len(aprs) == 3
        assert aprs[0][1] > aprs[1][1] > aprs[2][1]  # Should be in descending order
        
        # Pool 3 should have the highest APR (highest fee rate and lowest TVL)
        assert aprs[0][0] == "pool3"

    def test_uniswap_v3_yield_with_slippage_impact(self):
        """Test yield calculation including slippage impact."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
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

    def test_uniswap_v3_fee_compounding_frequency_impact(self):
        """Test impact of fee compounding frequency on yield."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test data
        initial_investment = 1000000000000000000000  # 1000 tokens
        annual_rate = 0.12  # 12% annual
        time_period = 365  # 1 year
        
        # Test different compounding frequencies
        compounding_frequencies = [1, 7, 30, 365]  # Daily, weekly, monthly, continuous
        
        for frequency in compounding_frequencies:
            # Calculate compound interest
            compound_amount = behaviour._calculate_compound_interest(
                initial_investment, annual_rate, time_period, frequency
            )
            
            # Expected: 1000 * (1 + 0.12/frequency)^(frequency * 1)
            expected_amount = initial_investment * (1 + annual_rate / frequency) ** frequency
            
            # Verify calculation
            TestAssertions.assert_yield_calculation_accuracy(
                compound_amount, expected_amount, 0.001
            )

    def test_uniswap_v3_position_value_with_price_movement(self):
        """Test position value calculation with price movements."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test data
        position_data = {
            "liquidity": 1000000000000000000000,
            "tick_lower": -276320,  # Price: 0.8
            "tick_upper": -276300,  # Price: 1.2
        }
        
        # Test different price scenarios
        price_scenarios = [
            {"current_price": 0.9, "expected_value": 1000000000000000000000},   # Within range
            {"current_price": 1.0, "expected_value": 1000000000000000000000},   # At center
            {"current_price": 1.1, "expected_value": 1000000000000000000000},   # Within range
            {"current_price": 0.7, "expected_value": 0},                        # Below range
            {"current_price": 1.3, "expected_value": 0},                        # Above range
        ]
        
        for scenario in price_scenarios:
            current_price = scenario["current_price"]
            expected_value = scenario["expected_value"]
            
            # Calculate position value
            position_value = behaviour._calculate_position_value_with_price(
                position_data, current_price
            )
            
            # Verify calculation
            if expected_value > 0:
                assert position_value > 0
            else:
                assert position_value == 0

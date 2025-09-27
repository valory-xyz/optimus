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

"""Unit integration tests for Velodrome protocol components."""

import pytest
from unittest.mock import MagicMock, patch

from packages.valory.skills.liquidity_trader_abci.behaviours.base import DexType
from packages.valory.skills.liquidity_trader_abci.pools.velodrome import VelodromePoolBehaviour

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions, TestDataGenerator


class TestVelodromeComponents(ProtocolIntegrationTestBase):
    """Test individual Velodrome protocol components in isolation."""

    def test_velodrome_pool_type_detection(self):
        """Test pool type detection logic in isolation."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test stable pool detection
        stable_pool_data = TestDataGenerator.generate_velodrome_pool_data(is_stable=True)
        is_stable = behaviour._detect_pool_type(stable_pool_data)
        assert is_stable == True
        
        # Test volatile pool detection
        volatile_pool_data = TestDataGenerator.generate_velodrome_pool_data(is_stable=False)
        is_stable = behaviour._detect_pool_type(volatile_pool_data)
        assert is_stable == False
        
        # Test CL pool detection
        cl_pool_data = TestDataGenerator.generate_velodrome_pool_data(is_cl_pool=True)
        is_cl_pool = behaviour._detect_cl_pool_type(cl_pool_data)
        assert is_cl_pool == True

    def test_velodrome_gauge_staking_logic(self):
        """Test gauge staking logic in isolation."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        lp_token_balance = 100000000000000000000  # 100 LP tokens
        gauge_address = "0xGaugeAddress"
        
        # Test staking calculation
        staking_amount = behaviour._calculate_optimal_staking_amount(
            lp_token_balance, gauge_address
        )
        
        # Should stake all available LP tokens
        assert staking_amount == lp_token_balance

    def test_velodrome_reward_calculation(self):
        """Test reward calculation logic in isolation."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        staked_amount = 100000000000000000000  # 100 LP tokens
        reward_rate = 1000000000000000000      # 1 token per second
        time_period = 86400                    # 1 day
        
        # Calculate expected rewards
        expected_rewards = (reward_rate * time_period * staked_amount) / 1000000000000000000000
        
        # Test reward calculation
        calculated_rewards = behaviour._calculate_velodrome_rewards(
            staked_amount, reward_rate, time_period
        )
        
        assert calculated_rewards == expected_rewards

    def test_velodrome_apr_calculation_components(self):
        """Test APR calculation components in isolation."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        pool_data = TestDataGenerator.generate_velodrome_pool_data()
        pool_data.update({
            "daily_volume": 10000000000000000000000,  # 10,000 tokens
            "pool_tvl": 5000000000000000000000,       # 5,000 tokens
            "swap_fee": 2500000000000000,             # 0.25%
        })
        
        # Calculate trading fees APR
        daily_fees = pool_data["daily_volume"] * pool_data["swap_fee"] / 1e18
        annual_fees = daily_fees * 365
        trading_apr = (annual_fees / pool_data["pool_tvl"]) * 100
        
        # Test calculation
        calculated_trading_apr = behaviour._calculate_trading_fees_apr(pool_data)
        
        assert calculated_trading_apr == trading_apr

    def test_velodrome_gauge_rewards_apr(self):
        """Test gauge rewards APR calculation."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        gauge_data = {
            "reward_rate": 1000000000000000000,  # 1 token per second
            "total_supply": 1000000000000000000000,  # 1000 LP tokens
            "reward_token_price": 1.0,  # $1 per reward token
            "lp_token_price": 2.0,  # $2 per LP token
        }
        
        # Calculate gauge APR
        daily_rewards = gauge_data["reward_rate"] * 86400  # 1 day
        annual_rewards = daily_rewards * 365
        total_lp_value = gauge_data["total_supply"] * gauge_data["lp_token_price"]
        gauge_apr = (annual_rewards * gauge_data["reward_token_price"] / total_lp_value) * 100
        
        # Test calculation
        calculated_gauge_apr = behaviour._calculate_gauge_rewards_apr(gauge_data)
        
        assert calculated_gauge_apr == gauge_apr

    def test_velodrome_total_apr_calculation(self):
        """Test total APR calculation combining trading fees and gauge rewards."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        trading_apr = 12.0  # 12% from trading fees
        gauge_apr = 8.0     # 8% from gauge rewards
        
        # Calculate total APR
        total_apr = trading_apr + gauge_apr
        
        # Test calculation
        calculated_total_apr = behaviour._calculate_total_velodrome_apr(
            trading_apr, gauge_apr
        )
        
        assert calculated_total_apr == total_apr

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
            
            calculated_il = behaviour._calculate_velodrome_impermanent_loss(
                1.0, price_ratio
            )
            
            # Allow 0.1% tolerance
            assert abs(calculated_il - expected_il) < 0.1

    def test_velodrome_liquidity_provision_calculation(self):
        """Test liquidity provision calculation logic."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        user_assets = {
            "token0": 1000000000000000000000,  # 1000 tokens
            "token1": 2000000000000000000000,  # 2000 tokens
        }
        
        pool_data = TestDataGenerator.generate_velodrome_pool_data()
        pool_data.update({
            "reserve0": 10000000000000000000000,  # 10,000 tokens
            "reserve1": 20000000000000000000000,  # 20,000 tokens
        })
        
        # Calculate optimal amounts
        optimal_amounts = behaviour._calculate_optimal_liquidity_amounts(
            user_assets, pool_data
        )
        
        # Verify calculations
        assert len(optimal_amounts) == 2
        assert optimal_amounts["token0"] > 0
        assert optimal_amounts["token1"] > 0
        assert optimal_amounts["token0"] <= user_assets["token0"]
        assert optimal_amounts["token1"] <= user_assets["token1"]

    def test_velodrome_gauge_voting_logic(self):
        """Test gauge voting logic in isolation."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        gauge_address = "0xGaugeAddress"
        vote_weight = 100000000000000000000  # 100 tokens
        
        # Test voting calculation
        voting_power = behaviour._calculate_voting_power(vote_weight)
        
        # Voting power should be proportional to vote weight
        assert voting_power == vote_weight

    def test_velodrome_reward_claiming_logic(self):
        """Test reward claiming logic in isolation."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        earned_rewards = {
            "token0": 100000000000000000000,  # 100 tokens
            "token1": 200000000000000000000,  # 200 tokens
        }
        
        # Test reward claiming calculation
        claimable_rewards = behaviour._calculate_claimable_rewards(earned_rewards)
        
        # Should be able to claim all earned rewards
        assert claimable_rewards["token0"] == earned_rewards["token0"]
        assert claimable_rewards["token1"] == earned_rewards["token1"]

    def test_velodrome_pool_share_calculation(self):
        """Test pool share calculation logic."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        user_lp_balance = 100000000000000000000   # 100 LP tokens
        total_lp_supply = 1000000000000000000000  # 1000 LP tokens
        pool_tvl = 5000000000000000000000         # 5000 tokens
        
        # Calculate user's share
        user_share = behaviour._calculate_pool_share(
            user_lp_balance, total_lp_supply, pool_tvl
        )
        
        # Expected: (100/1000) * 5000 = 500 tokens
        expected_share = (user_lp_balance / total_lp_supply) * pool_tvl
        
        assert user_share == expected_share

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
        
        assert user_fee_share == expected_share

    def test_velodrome_compound_interest_calculation(self):
        """Test compound interest calculation for Velodrome rewards."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        initial_investment = 1000000000000000000000  # 1000 tokens
        daily_reward_rate = 0.001  # 0.1% daily
        time_period = 30  # 30 days
        
        # Calculate compound interest
        compound_amount = behaviour._calculate_compound_interest(
            initial_investment, daily_reward_rate, time_period
        )
        
        # Expected: 1000 * (1.001)^30
        expected_amount = initial_investment * (1 + daily_reward_rate) ** time_period
        
        # Verify calculation
        assert compound_amount == expected_amount

    def test_velodrome_price_impact_calculation(self):
        """Test price impact calculation for Velodrome swaps."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        swap_amount = 1000000000000000000  # 1000 tokens
        pool_liquidity = 10000000000000000000000  # 10,000 tokens
        current_price = 1.5
        
        # Calculate price impact
        price_impact = behaviour._calculate_price_impact(
            swap_amount, pool_liquidity, current_price
        )
        
        assert price_impact >= 0
        assert price_impact <= 1.0  # Should not exceed 100%

    def test_velodrome_slippage_calculation(self):
        """Test slippage calculation for Velodrome operations."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test data
        expected_amount = 1000000000000000000000  # 1000 tokens
        actual_amount = 950000000000000000000     # 950 tokens
        
        # Calculate slippage
        slippage = behaviour._calculate_slippage(expected_amount, actual_amount)
        
        # Expected: (1000 - 950) / 1000 = 5%
        expected_slippage = (expected_amount - actual_amount) / expected_amount
        
        assert slippage == expected_slippage

    def test_velodrome_optimal_amount_calculation(self):
        """Test optimal amount calculation for different pool types."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test stable pool
        stable_pool_data = TestDataGenerator.generate_velodrome_pool_data(is_stable=True)
        user_assets = TestDataGenerator.generate_user_assets(
            token_addresses=stable_pool_data["tokens"]
        )
        
        optimal_amounts = behaviour._calculate_optimal_amounts(stable_pool_data, user_assets)
        
        # Verify calculations
        assert len(optimal_amounts) == len(stable_pool_data["tokens"])
        for token, amount in optimal_amounts.items():
            assert amount > 0
            assert amount <= user_assets[token]
        
        # Test volatile pool
        volatile_pool_data = TestDataGenerator.generate_velodrome_pool_data(is_stable=False)
        optimal_amounts = behaviour._calculate_optimal_amounts(volatile_pool_data, user_assets)
        
        # Verify calculations
        assert len(optimal_amounts) == len(volatile_pool_data["tokens"])
        for token, amount in optimal_amounts.items():
            assert amount > 0
            assert amount <= user_assets[token]

    def test_velodrome_position_management_logic(self):
        """Test position management logic."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test position lifecycle
        position_data = {
            "pool_address": "0xPoolAddress",
            "lp_balance": 100000000000000000000,
            "staked_amount": 100000000000000000000,
            "gauge_address": "0xGaugeAddress",
            "dex_type": DexType.VELODROME.value
        }
        
        # Test position creation
        created_position = behaviour._create_position(position_data)
        assert created_position["pool_address"] == position_data["pool_address"]
        
        # Test position update
        updated_position = behaviour._update_position(
            created_position, {"lp_balance": 200000000000000000000}
        )
        assert updated_position["lp_balance"] == 200000000000000000000
        
        # Test position removal
        removed_position = behaviour._remove_position(updated_position)
        assert removed_position is None

    def test_velodrome_gauge_staking_parameter_validation(self):
        """Test gauge staking parameter validation."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Valid parameters
        valid_params = {
            "gauge_address": "0xGaugeAddress",
            "amount": 100000000000000000000,
            "recipient": "0xRecipient",
        }
        
        # Validate parameters
        is_valid = behaviour._validate_staking_params(valid_params)
        assert is_valid
        
        # Invalid parameters - missing required field
        invalid_params = valid_params.copy()
        del invalid_params["gauge_address"]
        
        is_valid = behaviour._validate_staking_params(invalid_params)
        assert not is_valid

    def test_velodrome_reward_claiming_parameter_validation(self):
        """Test reward claiming parameter validation."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Valid parameters
        valid_params = {
            "gauge_address": "0xGaugeAddress",
            "recipient": "0xRecipient",
        }
        
        # Validate parameters
        is_valid = behaviour._validate_claiming_params(valid_params)
        assert is_valid
        
        # Invalid parameters - missing required field
        invalid_params = valid_params.copy()
        del invalid_params["gauge_address"]
        
        is_valid = behaviour._validate_claiming_params(invalid_params)
        assert not is_valid

    def test_velodrome_pool_type_specific_logic(self):
        """Test pool type specific logic."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test stable pool logic
        stable_pool_data = TestDataGenerator.generate_velodrome_pool_data(is_stable=True)
        pool_type = behaviour._get_pool_type_specific_logic(stable_pool_data)
        assert pool_type == "stable"
        
        # Test volatile pool logic
        volatile_pool_data = TestDataGenerator.generate_velodrome_pool_data(is_stable=False)
        pool_type = behaviour._get_pool_type_specific_logic(volatile_pool_data)
        assert pool_type == "volatile"
        
        # Test CL pool logic
        cl_pool_data = TestDataGenerator.generate_velodrome_pool_data(is_cl_pool=True)
        pool_type = behaviour._get_pool_type_specific_logic(cl_pool_data)
        assert pool_type == "concentrated_liquidity"

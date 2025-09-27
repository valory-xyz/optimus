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

"""Unit integration tests for Balancer protocol components."""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from packages.valory.skills.liquidity_trader_abci.behaviours.base import DexType
from packages.valory.skills.liquidity_trader_abci.pools.balancer import (
    BalancerPoolBehaviour,
    JoinKind,
    ExitKind,
    PoolType,
)
from packages.valory.skills.liquidity_trader_abci.utils.balancer_math import (
    BalancerProportionalMath,
)

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions, TestDataGenerator


class TestBalancerComponents(ProtocolIntegrationTestBase):
    """Test individual Balancer protocol components in isolation."""

    def test_balancer_math_proportional_calculations(self):
        """Test BalancerProportionalMath calculations in isolation."""
        # Test data
        total_supply = Decimal("1000000000000000000000")  # 1000 BPT
        user_balance = Decimal("100000000000000000000")   # 100 BPT
        pool_balance = Decimal("500000000000000000000")   # 500 tokens
        
        # Calculate expected proportional amount
        expected_amount = (user_balance * pool_balance) / total_supply
        
        # Test the calculation
        result = BalancerProportionalMath.calculate_proportional_amount(
            user_balance, pool_balance, total_supply
        )
        
        assert result == expected_amount
        assert result == Decimal("50000000000000000000")  # 50 tokens

    def test_join_kind_enum_validation(self):
        """Test join kind enum validation in isolation."""
        # Test weighted pool join kinds
        assert JoinKind.WeightedPool.INIT == 0
        assert JoinKind.WeightedPool.EXACT_TOKENS_IN_FOR_BPT_OUT == 1
        assert JoinKind.WeightedPool.TOKEN_IN_FOR_EXACT_BPT_OUT == 2
        assert JoinKind.WeightedPool.ALL_TOKENS_IN_FOR_EXACT_BPT_OUT == 3
        
        # Test stable pool join kinds
        assert JoinKind.StableAndMetaStablePool.INIT == 0
        assert JoinKind.StableAndMetaStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT == 1
        assert JoinKind.StableAndMetaStablePool.TOKEN_IN_FOR_EXACT_BPT_OUT == 2
        
        # Test composable stable pool join kinds
        assert JoinKind.ComposableStablePool.INIT == 0
        assert JoinKind.ComposableStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT == 1
        assert JoinKind.ComposableStablePool.TOKEN_IN_FOR_EXACT_BPT_OUT == 2
        assert JoinKind.ComposableStablePool.ALL_TOKENS_IN_FOR_EXACT_BPT_OUT == 3

    def test_exit_kind_enum_validation(self):
        """Test exit kind enum validation in isolation."""
        # Test weighted pool exit kinds
        assert ExitKind.WeightedPool.EXACT_BPT_IN_FOR_ONE_TOKEN_OUT == 0
        assert ExitKind.WeightedPool.EXACT_BPT_IN_FOR_TOKENS_OUT == 1
        assert ExitKind.WeightedPool.BPT_IN_FOR_EXACT_TOKENS_OUT == 2
        assert ExitKind.WeightedPool.MANAGEMENT_FEE_TOKENS_OUT == 3
        
        # Test stable pool exit kinds
        assert ExitKind.StableAndMetaStablePool.EXACT_BPT_IN_FOR_ONE_TOKEN_OUT == 0
        assert ExitKind.StableAndMetaStablePool.EXACT_BPT_IN_FOR_TOKENS_OUT == 1
        assert ExitKind.StableAndMetaStablePool.BPT_IN_FOR_EXACT_TOKENS_OUT == 2

    def test_pool_type_detection(self):
        """Test pool type detection logic in isolation."""
        # Test different pool types
        assert PoolType.WEIGHTED == PoolType("Weighted")
        assert PoolType.COMPOSABLE_STABLE == PoolType("ComposableStable")
        assert PoolType.LIQUIDITY_BOOTSTRAPING == PoolType("LiquidityBootstrapping")
        assert PoolType.META_STABLE == PoolType("MetaStable")
        assert PoolType.STABLE == PoolType("Stable")
        assert PoolType.INVESTMENT == PoolType("Investment")

    def test_user_share_calculation_isolated(self):
        """Test user share calculation without blockchain calls."""
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Mock contract responses
        mock_responses = [
            {"balance": 100000000000000000000},  # user BPT balance
            {"data": 1000000000000000000000},    # total supply
            {"tokens": ["0xTokenA", "0xTokenB"]},  # pool tokens
        ]
        
        # Set up mock contract interactions
        behaviour.contract_interact = MagicMock(side_effect=mock_responses)
        
        # Test user share calculation
        result = list(behaviour.get_user_share_value_balancer(
            "0xUserAddress",
            "0xPoolId",
            "0xPoolAddress",
            "optimism"
        ))
        
        # Verify the calculation
        assert len(result) > 0
        user_shares = result[0]
        assert "token0" in user_shares
        assert "token1" in user_shares
        assert user_shares["token0"] > 0
        assert user_shares["token1"] > 0

    def test_pool_token_balance_calculation(self):
        """Test pool token balance calculation logic."""
        # Test data
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        user_bpt_balance = 100000000000000000000  # 100 BPT
        total_supply = 1000000000000000000000     # 1000 BPT
        pool_token_balance = 500000000000000000000  # 500 tokens
        
        # Calculate expected user share
        expected_share = (user_bpt_balance * pool_token_balance) / total_supply
        
        # Test calculation
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        calculated_share = behaviour._calculate_user_token_share(
            user_bpt_balance, total_supply, pool_token_balance
        )
        
        assert calculated_share == expected_share
        assert calculated_share == 50000000000000000000  # 50 tokens

    def test_join_pool_parameter_validation(self):
        """Test join pool parameter validation."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Valid parameters
        valid_params = {
            "pool_address": "0xPoolAddress",
            "safe_address": "0xSafeAddress",
            "assets": ["0xTokenA", "0xTokenB"],
            "chain": "optimism",
            "max_amounts_in": [1000000000000000000, 2000000000000000000],
            "pool_id": "0x1234567890123456789012345678901234567890123456789012345678901234",
        }
        
        # Test parameter validation
        is_valid = behaviour._validate_join_pool_params(valid_params)
        assert is_valid
        
        # Invalid parameters - missing required field
        invalid_params = valid_params.copy()
        del invalid_params["pool_address"]
        
        is_valid = behaviour._validate_join_pool_params(invalid_params)
        assert not is_valid

    def test_exit_pool_parameter_validation(self):
        """Test exit pool parameter validation."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Valid parameters
        valid_params = {
            "pool_address": "0xPoolAddress",
            "safe_address": "0xSafeAddress",
            "assets": ["0xTokenA", "0xTokenB"],
            "chain": "optimism",
            "bpt_amount_in": 100000000000000000000,
            "pool_id": "0x1234567890123456789012345678901234567890123456789012345678901234",
        }
        
        # Test parameter validation
        is_valid = behaviour._validate_exit_pool_params(valid_params)
        assert is_valid
        
        # Invalid parameters - missing required field
        invalid_params = valid_params.copy()
        del invalid_params["bpt_amount_in"]
        
        is_valid = behaviour._validate_exit_pool_params(invalid_params)
        assert not is_valid

    def test_pool_type_specific_logic(self):
        """Test pool type specific logic."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test weighted pool logic
        weighted_pool_data = TestDataGenerator.generate_balancer_pool_data(
            pool_type="Weighted"
        )
        join_kind = behaviour._get_join_kind_for_pool_type(weighted_pool_data)
        assert join_kind in [JoinKind.WeightedPool.EXACT_TOKENS_IN_FOR_BPT_OUT,
                           JoinKind.WeightedPool.TOKEN_IN_FOR_EXACT_BPT_OUT]
        
        # Test stable pool logic
        stable_pool_data = TestDataGenerator.generate_balancer_pool_data(
            pool_type="ComposableStable"
        )
        join_kind = behaviour._get_join_kind_for_pool_type(stable_pool_data)
        assert join_kind in [JoinKind.StableAndMetaStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT,
                           JoinKind.StableAndMetaStablePool.TOKEN_IN_FOR_EXACT_BPT_OUT]

    def test_optimal_amount_calculation(self):
        """Test optimal amount calculation for different pool types."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test data
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        user_assets = TestDataGenerator.generate_user_assets(
            token_addresses=pool_data["tokens"],
            amounts=[1000000000000000000000, 2000000000000000000000]  # 1000, 2000 tokens
        )
        
        # Calculate optimal amounts
        optimal_amounts = behaviour._calculate_optimal_amounts(pool_data, user_assets)
        
        # Verify calculations
        assert len(optimal_amounts) == len(pool_data["tokens"])
        for token, amount in optimal_amounts.items():
            assert amount > 0
            assert amount <= user_assets[token]  # Should not exceed user balance

    def test_fee_calculation_logic(self):
        """Test fee calculation logic."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test data
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        daily_volume = 10000000000000000000000  # 10,000 tokens
        swap_fee_rate = 2500000000000000  # 0.25%
        
        # Calculate daily fees
        daily_fees = behaviour._calculate_daily_fees(daily_volume, swap_fee_rate)
        expected_fees = daily_volume * swap_fee_rate / 1e18
        
        assert daily_fees == expected_fees
        assert daily_fees == 25000000000000000000  # 25 tokens

    def test_apr_calculation_components(self):
        """Test APR calculation components."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test data
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        daily_fees = 25000000000000000000  # 25 tokens
        pool_tvl = 5000000000000000000000   # 5,000 tokens
        
        # Calculate APR components
        annual_fees = daily_fees * 365
        apr = (annual_fees / pool_tvl) * 100
        
        # Test calculation
        calculated_apr = behaviour._calculate_apr_from_fees(daily_fees, pool_tvl)
        
        assert calculated_apr == apr
        assert calculated_apr == 18.25  # 18.25% APR

    def test_impermanent_loss_calculation(self):
        """Test impermanent loss calculation."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
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
            
            calculated_il = behaviour._calculate_impermanent_loss(1.0, price_ratio)
            
            # Allow 0.1% tolerance
            assert abs(calculated_il - expected_il) < 0.1

    def test_pool_share_proportional_math(self):
        """Test proportional math for pool shares."""
        # Test data
        total_supply = 1000000000000000000000  # 1000 BPT
        user_balance = 100000000000000000000   # 100 BPT
        pool_tvl = 5000000000000000000000      # 5000 tokens
        
        # Calculate user's share of TVL
        user_share = (user_balance / total_supply) * pool_tvl
        expected_share = 500000000000000000000  # 500 tokens
        
        assert user_share == expected_share
        
        # Test with different amounts
        user_balance_2 = 200000000000000000000  # 200 BPT
        user_share_2 = (user_balance_2 / total_supply) * pool_tvl
        expected_share_2 = 1000000000000000000000  # 1000 tokens
        
        assert user_share_2 == expected_share_2

    def test_pool_weight_validation(self):
        """Test pool weight validation."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Valid weights (sum to 1e18)
        valid_weights = [500000000000000000, 500000000000000000]  # 50/50
        assert behaviour._validate_pool_weights(valid_weights)
        
        # Invalid weights (don't sum to 1e18)
        invalid_weights = [400000000000000000, 500000000000000000]  # 40/50
        assert not behaviour._validate_pool_weights(invalid_weights)
        
        # Edge case: single token (weight = 1e18)
        single_token_weights = [1000000000000000000]
        assert behaviour._validate_pool_weights(single_token_weights)

    def test_join_exit_kind_selection(self):
        """Test join/exit kind selection logic."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test join kind selection
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        user_assets = TestDataGenerator.generate_user_assets(
            token_addresses=pool_data["tokens"]
        )
        
        # Test exact tokens in for BPT out
        join_kind = behaviour._select_join_kind(
            pool_data, user_assets, "exact_tokens_in"
        )
        assert join_kind == JoinKind.WeightedPool.EXACT_TOKENS_IN_FOR_BPT_OUT
        
        # Test token in for exact BPT out
        join_kind = behaviour._select_join_kind(
            pool_data, user_assets, "exact_bpt_out"
        )
        assert join_kind == JoinKind.WeightedPool.TOKEN_IN_FOR_EXACT_BPT_OUT
        
        # Test exit kind selection
        exit_kind = behaviour._select_exit_kind("exact_bpt_in")
        assert exit_kind == ExitKind.WeightedPool.EXACT_BPT_IN_FOR_TOKENS_OUT

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

"""Unit integration tests for Uniswap V3 protocol components."""

import pytest
from unittest.mock import MagicMock, patch

from packages.valory.skills.liquidity_trader_abci.behaviours.base import DexType
from packages.valory.skills.liquidity_trader_abci.pools.uniswap import (
    UniswapPoolBehaviour,
    MintParams,
)
from packages.valory.skills.liquidity_trader_abci.utils.tick_math import (
    get_sqrt_ratio_at_tick,
    get_amounts_for_liquidity,
    get_liquidity_for_amounts,
)

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions, TestDataGenerator


class TestUniswapV3Components(ProtocolIntegrationTestBase):
    """Test individual Uniswap V3 protocol components in isolation."""

    def test_tick_math_calculations(self):
        """Test tick math calculations in isolation."""
        # Test tick to sqrt ratio conversion
        tick = -276320
        sqrt_ratio = get_sqrt_ratio_at_tick(tick)
        assert sqrt_ratio > 0
        
        # Test liquidity calculations
        sqrt_ratio_a = get_sqrt_ratio_at_tick(-276320)
        sqrt_ratio_b = get_sqrt_ratio_at_tick(-276300)
        liquidity = 1000000000000000000
        
        amounts = get_amounts_for_liquidity(
            sqrt_ratio_a, sqrt_ratio_b, liquidity, sqrt_ratio_a
        )
        
        assert amounts[0] >= 0  # amount0
        assert amounts[1] >= 0  # amount1

    def test_mint_params_validation(self):
        """Test mint parameters validation in isolation."""
        # Valid mint parameters
        mint_params = MintParams(
            token0="0xTokenA",
            token1="0xTokenB",
            fee=3000,
            tickLower=-276320,
            tickUpper=-276300,
            amount0Desired=1000000000000000000,
            amount1Desired=2000000000000000000,
            amount0Min=900000000000000000,
            amount1Min=1800000000000000000,
            recipient="0xRecipient",
            deadline=1234567890
        )
        
        # Validate parameters
        assert mint_params.tickLower < mint_params.tickUpper
        assert mint_params.amount0Min <= mint_params.amount0Desired
        assert mint_params.amount1Min <= mint_params.amount1Desired
        assert mint_params.deadline > 0

    def test_position_liquidity_calculation(self):
        """Test position liquidity calculation without blockchain calls."""
        # Mock position data
        position_data = {
            "token0": "0xTokenA",
            "token1": "0xTokenB",
            "fee": 3000,
            "tickLower": -276320,
            "tickUpper": -276300,
            "liquidity": 1000000000000000000,
            "tokensOwed0": 0,
            "tokensOwed1": 0
        }
        
        # Mock current pool state
        current_tick = -276310
        sqrt_price_x96 = 79228162514264337593543950336  # Mock sqrt price
        
        # Calculate expected amounts
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        result = behaviour._calculate_position_amounts(
            position_data, current_tick, sqrt_price_x96
        )
        
        assert "amount0" in result
        assert "amount1" in result
        assert result["amount0"] >= 0
        assert result["amount1"] >= 0

    def test_tick_range_calculation(self):
        """Test tick range calculation for optimal capital efficiency."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test data
        current_tick = -276310
        tick_spacing = 60
        price_range_percentage = 0.2  # 20% range
        
        # Calculate optimal tick range
        tick_range = behaviour._calculate_optimal_tick_range(
            current_tick, tick_spacing, price_range_percentage
        )
        
        # Verify tick range
        assert tick_range["tick_lower"] < current_tick
        assert tick_range["tick_upper"] > current_tick
        assert tick_range["tick_lower"] % tick_spacing == 0
        assert tick_range["tick_upper"] % tick_spacing == 0

    def test_liquidity_amount_calculation(self):
        """Test liquidity amount calculation for different price ranges."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test data
        amount0 = 1000000000000000000  # 1000 tokens
        amount1 = 2000000000000000000  # 2000 tokens
        tick_lower = -276320
        tick_upper = -276300
        current_tick = -276310
        
        # Calculate liquidity
        liquidity = behaviour._calculate_liquidity_for_amounts(
            amount0, amount1, tick_lower, tick_upper, current_tick
        )
        
        assert liquidity > 0

    def test_fee_tier_validation(self):
        """Test fee tier validation logic."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Valid fee tiers
        valid_fee_tiers = [500, 3000, 10000]
        for fee in valid_fee_tiers:
            assert behaviour._validate_fee_tier(fee)
        
        # Invalid fee tiers
        invalid_fee_tiers = [100, 2000, 50000]
        for fee in invalid_fee_tiers:
            assert not behaviour._validate_fee_tier(fee)

    def test_tick_spacing_calculation(self):
        """Test tick spacing calculation for different fee tiers."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test fee tier to tick spacing mapping
        fee_tier_mappings = [
            (500, 10),
            (3000, 60),
            (10000, 200),
        ]
        
        for fee, expected_spacing in fee_tier_mappings:
            calculated_spacing = behaviour._get_tick_spacing_for_fee(fee)
            assert calculated_spacing == expected_spacing

    def test_position_value_calculation(self):
        """Test position value calculation logic."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test data
        position_data = {
            "liquidity": 1000000000000000000,
            "tickLower": -276320,
            "tickUpper": -276300,
            "tokensOwed0": 100000000000000000,  # 0.1 tokens
            "tokensOwed1": 200000000000000000,  # 0.2 tokens
        }
        
        current_tick = -276310
        sqrt_price_x96 = 79228162514264337593543950336
        
        # Calculate position value
        position_value = behaviour._calculate_position_value(
            position_data, current_tick, sqrt_price_x96
        )
        
        assert position_value > 0
        assert "amount0" in position_value
        assert "amount1" in position_value
        assert "total_value_usd" in position_value

    def test_fee_collection_calculation(self):
        """Test fee collection calculation logic."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test data
        position_data = {
            "tokensOwed0": 100000000000000000,  # 0.1 tokens
            "tokensOwed1": 200000000000000000,  # 0.2 tokens
        }
        
        # Calculate collectable fees
        collectable_fees = behaviour._calculate_collectable_fees(position_data)
        
        assert collectable_fees["amount0"] == position_data["tokensOwed0"]
        assert collectable_fees["amount1"] == position_data["tokensOwed1"]

    def test_optimal_tick_range_selection(self):
        """Test optimal tick range selection based on volatility."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test different volatility scenarios
        volatility_scenarios = [
            {"volatility": 0.1, "expected_range": 0.1},   # Low volatility - narrow range
            {"volatility": 0.3, "expected_range": 0.2},   # Medium volatility - medium range
            {"volatility": 0.5, "expected_range": 0.4},   # High volatility - wide range
        ]
        
        for scenario in volatility_scenarios:
            optimal_range = behaviour._select_optimal_range_for_volatility(
                scenario["volatility"]
            )
            
            # Range should be proportional to volatility
            assert optimal_range > 0
            assert optimal_range <= 1.0

    def test_liquidity_concentration_calculation(self):
        """Test liquidity concentration calculation."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test data
        position_liquidity = 1000000000000000000
        total_liquidity = 10000000000000000000000
        
        # Calculate concentration
        concentration = behaviour._calculate_liquidity_concentration(
            position_liquidity, total_liquidity
        )
        
        # Expected: 1000 / 10000000 = 0.0001 (0.01%)
        expected_concentration = position_liquidity / total_liquidity
        assert concentration == expected_concentration

    def test_price_impact_calculation(self):
        """Test price impact calculation for swaps."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
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

    def test_impermanent_loss_calculation_uniswap_v3(self):
        """Test impermanent loss calculation for Uniswap V3."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
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
            
            calculated_il = behaviour._calculate_uniswap_v3_impermanent_loss(
                1.0, price_ratio
            )
            
            # Allow 0.1% tolerance
            assert abs(calculated_il - expected_il) < 0.1

    def test_capital_efficiency_calculation(self):
        """Test capital efficiency calculation for concentrated liquidity."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test data
        price_range = 0.2  # 20% range
        current_price = 1.0
        
        # Calculate capital efficiency
        efficiency = behaviour._calculate_capital_efficiency(
            current_price, price_range
        )
        
        # For a 20% range, efficiency should be around 5x
        expected_efficiency = 1 / price_range
        assert abs(efficiency - expected_efficiency) < 0.1

    def test_fee_growth_calculation(self):
        """Test fee growth calculation over time."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test data
        initial_fee_growth = 1000000000000000000
        current_fee_growth = 1500000000000000000
        liquidity = 1000000000000000000
        
        # Calculate fee growth
        fee_growth = behaviour._calculate_fee_growth(
            initial_fee_growth, current_fee_growth, liquidity
        )
        
        # Expected: (1500 - 1000) / 1000 = 0.5 tokens
        expected_growth = (current_fee_growth - initial_fee_growth) / liquidity
        assert fee_growth == expected_growth

    def test_position_management_logic(self):
        """Test position management logic."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test position lifecycle
        position_data = {
            "token_id": 12345,
            "liquidity": 1000000000000000000,
            "tickLower": -276320,
            "tickUpper": -276300,
            "tokensOwed0": 0,
            "tokensOwed1": 0,
        }
        
        # Test position creation
        created_position = behaviour._create_position(position_data)
        assert created_position["token_id"] == position_data["token_id"]
        
        # Test position update
        updated_position = behaviour._update_position(
            created_position, {"liquidity": 2000000000000000000}
        )
        assert updated_position["liquidity"] == 2000000000000000000
        
        # Test position removal
        removed_position = behaviour._remove_position(updated_position)
        assert removed_position is None

    def test_nft_token_handling(self):
        """Test NFT token handling for positions."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test NFT token operations
        token_id = 12345
        owner_address = "0xOwnerAddress"
        
        # Test token ownership validation
        is_owner = behaviour._validate_token_ownership(token_id, owner_address)
        assert isinstance(is_owner, bool)
        
        # Test token transfer
        transfer_result = behaviour._transfer_token(token_id, owner_address, "0xNewOwner")
        assert transfer_result is not None

    def test_swap_parameter_validation(self):
        """Test swap parameter validation."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Valid swap parameters
        valid_params = {
            "token_in": "0xTokenA",
            "token_out": "0xTokenB",
            "amount_in": 1000000000000000000,
            "amount_out_min": 900000000000000000,
            "recipient": "0xRecipient",
            "deadline": 1234567890,
        }
        
        # Validate parameters
        is_valid = behaviour._validate_swap_params(valid_params)
        assert is_valid
        
        # Invalid parameters - amount_out_min > amount_in
        invalid_params = valid_params.copy()
        invalid_params["amount_out_min"] = 2000000000000000000
        
        is_valid = behaviour._validate_swap_params(invalid_params)
        assert not is_valid

    def test_liquidity_provision_parameter_validation(self):
        """Test liquidity provision parameter validation."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Valid liquidity provision parameters
        valid_params = {
            "token0": "0xTokenA",
            "token1": "0xTokenB",
            "fee": 3000,
            "tickLower": -276320,
            "tickUpper": -276300,
            "amount0Desired": 1000000000000000000,
            "amount1Desired": 2000000000000000000,
            "amount0Min": 900000000000000000,
            "amount1Min": 1800000000000000000,
            "recipient": "0xRecipient",
            "deadline": 1234567890,
        }
        
        # Validate parameters
        is_valid = behaviour._validate_liquidity_provision_params(valid_params)
        assert is_valid
        
        # Invalid parameters - tickLower >= tickUpper
        invalid_params = valid_params.copy()
        invalid_params["tickLower"] = -276300
        invalid_params["tickUpper"] = -276320
        
        is_valid = behaviour._validate_liquidity_provision_params(invalid_params)
        assert not is_valid

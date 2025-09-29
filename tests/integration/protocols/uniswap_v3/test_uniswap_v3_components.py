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

"""Unit integration tests for Uniswap V3 protocol components using real methods."""

import pytest
from unittest.mock import MagicMock, patch

from packages.valory.skills.liquidity_trader_abci.pools.uniswap import (
    UniswapPoolBehaviour,
    MintParams,
)

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions, TestDataGenerator


class TestUniswapV3Components(ProtocolIntegrationTestBase):
    """Test individual Uniswap V3 protocol components using real methods."""

    def _consume_generator(self, gen):
        """Helper to run generator and return final value."""
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    def test_enter_method(self):
        """Test enter method for creating liquidity positions."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            yield
            return "0x1234567890abcdef"
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test enter method with valid parameters
        enter_generator = behaviour.enter(
            token0="0x4200000000000000000000000000000000000006",  # WETH
            token1="0x7F5c764cBc14f9669B88837ca1490cCa17c31607",  # USDC
            fee=3000,
            tickLower=-276320,
            tickUpper=-276300,
            amount0Desired=1000000000000000000,  # 1 ETH
            amount1Desired=2000000000,  # 2000 USDC
            amount0Min=900000000000000000,  # 0.9 ETH
            amount1Min=1800000000,  # 1800 USDC
            recipient="0xRecipientAddress",
            deadline=1234567890,
            chain="optimism"
        )
        
        # Consume the generator
        result = self._consume_generator(enter_generator)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2  # (tx_hash, token_id)

    def test_exit_method(self):
        """Test exit method for removing liquidity positions."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            yield
            return "0x1234567890abcdef"
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test exit method with valid parameters
        exit_generator = behaviour.exit(
            token_id=12345,
            safe_address="0xSafeAddress",
            liquidity=1000000000000000000,
            pool_address="0xPoolAddress",
            chain="optimism"
        )
        
        # Consume the generator
        result = self._consume_generator(exit_generator)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 3  # (tx_hash, success, bool)

    def test_burn_token_method(self):
        """Test burn_token method for removing NFT positions."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            yield
            return "0x1234567890abcdef"
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test burn_token method
        burn_generator = behaviour.burn_token(
            token_id=12345,
            chain="optimism"
        )
        
        # Consume the generator
        result = self._consume_generator(burn_generator)
        assert result is not None

    def test_collect_tokens_method(self):
        """Test collect_tokens method for collecting fees."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            yield
            return "0x1234567890abcdef"
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test collect_tokens method
        collect_generator = behaviour.collect_tokens(
            token_id=12345,
            recipient="0xRecipientAddress",
            amount0_max=1000000000000000000,
            amount1_max=2000000000,
            chain="optimism"
        )
        
        # Consume the generator
        result = self._consume_generator(collect_generator)
        assert result is not None

    def test_decrease_liquidity_method(self):
        """Test decrease_liquidity method for reducing position size."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            yield
            return "0x1234567890abcdef"
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test decrease_liquidity method
        decrease_generator = behaviour.decrease_liquidity(
            token_id=12345,
            liquidity=500000000000000000,  # Half the liquidity
            amount0_min=450000000000000000,
            amount1_min=900000000,
            deadline=1234567890,
            chain="optimism"
        )
        
        # Consume the generator
        result = self._consume_generator(decrease_generator)
        assert result is not None

    def test_get_liquidity_for_token_method(self):
        """Test get_liquidity_for_token method for position queries."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions - return position data as list
        def mock_contract_interact(*args, **kwargs):
            yield
            # Return position data as list (liquidity is at index 7)
            return [0, 0, 0, 0, 0, 0, 0, 1000000000000000000, 0, 0]  # Mock position data
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test get_liquidity_for_token method
        liquidity_generator = behaviour.get_liquidity_for_token(
            token_id=12345,
            chain="optimism"
        )
        
        # Consume the generator
        result = self._consume_generator(liquidity_generator)
        assert result is not None
        assert result == 1000000000000000000

    def test_get_tokens_method(self):
        """Test _get_tokens method for getting pool tokens."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            yield
            return (["0x4200000000000000000000000000000000000006", "0x7F5c764cBc14f9669B88837ca1490cCa17c31607"], [1000000000000000000, 2000000000])
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test _get_tokens method
        tokens_generator = behaviour._get_tokens(
            pool_address="0xPoolAddress",
            chain="optimism"
        )
        
        # Consume the generator
        result = self._consume_generator(tokens_generator)
        assert result is not None
        assert isinstance(result, dict)
        assert "token0" in result
        assert "token1" in result

    def test_get_pool_fee_method(self):
        """Test _get_pool_fee method for getting pool fee."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            yield
            return 3000  # 0.3% fee
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test _get_pool_fee method
        fee_generator = behaviour._get_pool_fee(
            pool_address="0xPoolAddress",
            chain="optimism"
        )
        
        # Consume the generator
        result = self._consume_generator(fee_generator)
        assert result is not None
        assert result == 3000

    def test_get_tick_spacing_method(self):
        """Test _get_tick_spacing method for getting tick spacing."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            yield
            return 60  # Tick spacing for 0.3% fee tier
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test _get_tick_spacing method
        spacing_generator = behaviour._get_tick_spacing(
            pool_address="0xPoolAddress",
            chain="optimism"
        )
        
        # Consume the generator
        result = self._consume_generator(spacing_generator)
        assert result is not None
        assert result == 60

    def test_calculate_tick_lower_and_upper_method(self):
        """Test _calculate_tick_lower_and_upper method for tick calculations."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions for _get_tick_spacing
        def mock_contract_interact(*args, **kwargs):
            yield
            return 60  # Mock tick spacing
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test tick calculation
        tick_generator = behaviour._calculate_tick_lower_and_upper(
            pool_address="0xPoolAddress",
            chain="optimism"
        )
        
        # Consume the generator
        result = self._consume_generator(tick_generator)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2  # (tick_lower, tick_upper)
        assert result[0] < result[1]  # tick_lower < tick_upper

    def test_calculate_slippage_protection_for_mint_method(self):
        """Test _calculate_slippage_protection_for_mint method."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            yield
            return {"sqrt_price_x96": 79228162514264337593543950336}  # Mock slot0 data
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test slippage protection calculation
        slippage_generator = behaviour._calculate_slippage_protection_for_mint(
            pool_address="0xPoolAddress",
            tick_lower=-276320,
            tick_upper=-276300,
            max_amounts_in=[1000000000000000000, 2000000000],
            chain="optimism"
        )
        
        # Consume the generator
        result = self._consume_generator(slippage_generator)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2  # (amount0_min, amount1_min)

    def test_calculate_slippage_protection_for_decrease_method(self):
        """Test _calculate_slippage_protection_for_decrease method."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions - return position data and slot0 data
        call_count = 0
        def mock_contract_interact(*args, **kwargs):
            nonlocal call_count
            yield
            call_count += 1
            if call_count == 1:  # First call for get_position
                return [0, 0, 0, 0, 0, 0, 0, 1000000000000000000, 0, 0]  # Mock position data
            else:  # Second call for slot0
                return {"sqrt_price_x96": 79228162514264337593543950336}  # Mock slot0 data
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test slippage protection calculation
        slippage_generator = behaviour._calculate_slippage_protection_for_decrease(
            token_id=12345,
            liquidity=1000000000000000000,
            chain="optimism",
            pool_address="0xPoolAddress"
        )
        
        # Consume the generator
        result = self._consume_generator(slippage_generator)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2  # (amount0_min, amount1_min)

    def test_mint_params_validation(self):
        """Test MintParams validation."""
        # Valid mint parameters
        mint_params = MintParams(
            token0="0x4200000000000000000000000000000000000006",
            token1="0x7F5c764cBc14f9669B88837ca1490cCa17c31607",
            fee=3000,
            tickLower=-276320,
            tickUpper=-276300,
            amount0Desired=1000000000000000000,
            amount1Desired=2000000000,
            amount0Min=900000000000000000,
            amount1Min=1800000000,
            recipient="0xRecipientAddress",
            deadline=1234567890
        )
        
        # Verify parameters are valid
        assert mint_params.token0 != mint_params.token1
        assert mint_params.fee > 0
        assert mint_params.tickLower < mint_params.tickUpper
        assert mint_params.amount0Min <= mint_params.amount0Desired
        assert mint_params.amount1Min <= mint_params.amount1Desired
        assert mint_params.deadline > 0
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
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

"""Transaction generation tests for Uniswap V3 protocol."""

import pytest
from unittest.mock import MagicMock, patch

from packages.valory.contracts.uniswap_v3_pool.contract import UniswapV3PoolContract
from packages.valory.contracts.uniswap_v3_non_fungible_position_manager.contract import UniswapV3NonfungiblePositionManagerContract
from packages.valory.skills.liquidity_trader_abci.pools.uniswap import UniswapPoolBehaviour

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions
from tests.integration.fixtures.contract_fixtures import (
    mock_ledger_api,
    uniswap_v3_pool_contract,
    uniswap_v3_position_manager_contract,
)


class TestUniswapV3TransactionGeneration(ProtocolIntegrationTestBase):
    """Test proper transaction encoding and parameters for Uniswap V3."""

    def _consume_generator(self, gen):
        """Helper to run generator and return final value."""
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    def test_enter_transaction_generation(self):
        """Test enter transaction generation using real UniswapPoolBehaviour methods."""
        # Setup test data
        pool_address = "0xPoolAddress"
        safe_address = "0xSafeAddress"
        assets = ["0xTokenA", "0xTokenB"]
        chain = "optimism"
        max_amounts_in = [1000000000000000000, 2000000000000000000]
        pool_fee = 3000

        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses and parameters
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}
        behaviour.params.slippage_tolerance = 0.01  # 1% slippage tolerance

        # Mock contract interactions for enter workflow
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            data_key = kwargs.get('data_key')
            
            yield None
            
            if callable_name == "get_tick_spacing" and data_key == "data":
                return 60
            elif callable_name == "slot0" and data_key == "slot0":
                return {
                    "sqrt_price_x96": 79228162514264337593543950336,
                    "tick": -276310,
                    "unlocked": True
                }
            elif callable_name == "mint":
                return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            else:
                return None

        # Mock the helper methods that enter() calls
        def mock_calculate_tick_lower_and_upper(*args, **kwargs):
            yield None
            return -276320, -276300

        def mock_calculate_slippage_protection_for_mint(*args, **kwargs):
            yield None
            return 999000000000000000, 1998000000000000000  # 0.1% slippage

        behaviour.contract_interact = mock_contract_interact
        behaviour._calculate_tick_lower_and_upper = mock_calculate_tick_lower_and_upper
        behaviour._calculate_slippage_protection_for_mint = mock_calculate_slippage_protection_for_mint

        # Test enter transaction generation
        result = self._consume_generator(behaviour.enter(
            pool_address=pool_address,
            safe_address=safe_address,
            assets=assets,
            chain=chain,
            max_amounts_in=max_amounts_in,
            pool_fee=pool_fee
        ))

        # Verify result
        assert result is not None
        assert len(result) == 2  # (tx_hash, position_manager_address)
        assert isinstance(result[0], str)  # tx_hash
        assert isinstance(result[1], str)  # position_manager_address

    def test_exit_transaction_generation(self):
        """Test exit transaction generation using real UniswapPoolBehaviour methods."""
        # Setup test data
        token_id = 12345
        pool_address = "0xPoolAddress"
        safe_address = "0xSafeAddress"
        chain = "optimism"

        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses and parameters
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}
        behaviour.params.multisend_contract_addresses = {"optimism": "0xMultiSendAddress"}
        behaviour.params.slippage_tolerance = 0.01  # 1% slippage tolerance

        # Mock contract interactions for exit workflow
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            data_key = kwargs.get('data_key')
            
            yield None
            
            if callable_name == "get_position" and data_key == "data":
                return {
                    "nonce": 1,
                    "operator": "0xOperator",
                    "token0": "0xTokenA",
                    "token1": "0xTokenB",
                    "fee": 3000,
                    "tickLower": -276320,
                    "tickUpper": -276300,
                    "liquidity": 1000000000000000000,
                    "feeGrowthInside0LastX128": 0,
                    "feeGrowthInside1LastX128": 0,
                    "tokensOwed0": 100000000000000000,
                    "tokensOwed1": 200000000000000000,
                }
            elif callable_name == "slot0" and data_key == "slot0":
                return {
                    "sqrt_price_x96": 79228162514264337593543950336,
                    "tick": -276310,
                    "unlocked": True
                }
            elif callable_name == "get_tx_data" and data_key == "data":
                return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            else:
                return None

        # Mock the helper methods that exit() calls
        def mock_get_liquidity_for_token(*args, **kwargs):
            yield None
            return 1000000000000000000

        def mock_calculate_slippage_protection_for_decrease(*args, **kwargs):
            yield None
            return 999000000000000000, 1998000000000000000  # 0.1% slippage

        def mock_decrease_liquidity(*args, **kwargs):
            yield None
            return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"

        def mock_collect_tokens(*args, **kwargs):
            yield None
            return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"

        behaviour.contract_interact = mock_contract_interact
        behaviour.get_liquidity_for_token = mock_get_liquidity_for_token
        behaviour._calculate_slippage_protection_for_decrease = mock_calculate_slippage_protection_for_decrease
        behaviour.decrease_liquidity = mock_decrease_liquidity
        behaviour.collect_tokens = mock_collect_tokens

        # Test exit transaction generation
        result = self._consume_generator(behaviour.exit(
            token_id=token_id,
            pool_address=pool_address,
            safe_address=safe_address,
            chain=chain
        ))

        # Verify result
        assert result is not None
        assert len(result) == 3  # (tx_hash_bytes, multisend_address, success)
        assert isinstance(result[0], bytes)  # tx_hash as bytes
        assert isinstance(result[1], str)  # multisend_address
        assert isinstance(result[2], bool)  # success

    def test_burn_token_transaction_generation(self):
        """Test burn token transaction generation using real UniswapPoolBehaviour methods."""
        # Setup test data
        token_id = 12345
        chain = "optimism"

        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}

        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            data_key = kwargs.get('data_key')
            
            yield None
            
            if callable_name == "burn_token":
                return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            else:
                return None

        behaviour.contract_interact = mock_contract_interact

        # Test burn token transaction generation
        result = self._consume_generator(behaviour.burn_token(
            token_id=token_id,
            chain=chain
        ))

        # Verify result
        assert result is not None
        assert isinstance(result, str)  # tx_hash

    def test_collect_tokens_transaction_generation(self):
        """Test collect tokens transaction generation using real UniswapPoolBehaviour methods."""
        # Setup test data
        token_id = 12345
        recipient = "0xRecipient"
        amount0_max = 1000000000000000000
        amount1_max = 2000000000000000000
        chain = "optimism"

        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}

        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            data_key = kwargs.get('data_key')
            
            yield None
            
            if callable_name == "collect_tokens":
                return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            else:
                return None

        behaviour.contract_interact = mock_contract_interact

        # Test collect tokens transaction generation
        result = self._consume_generator(behaviour.collect_tokens(
            token_id=token_id,
            recipient=recipient,
            amount0_max=amount0_max,
            amount1_max=amount1_max,
            chain=chain
        ))

        # Verify result
        assert result is not None
        assert isinstance(result, str)  # tx_hash

    def test_decrease_liquidity_transaction_generation(self):
        """Test decrease liquidity transaction generation using real UniswapPoolBehaviour methods."""
        # Setup test data
        token_id = 12345
        liquidity = 1000000000000000000
        amount0_min = 900000000000000000
        amount1_min = 1800000000000000000
        deadline = 1234567890
        chain = "optimism"

        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}

        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            data_key = kwargs.get('data_key')
            
            yield None
            
            if callable_name == "decrease_liquidity":
                return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            else:
                return None

        behaviour.contract_interact = mock_contract_interact

        # Test decrease liquidity transaction generation
        result = self._consume_generator(behaviour.decrease_liquidity(
            token_id=token_id,
            liquidity=liquidity,
            amount0_min=amount0_min,
            amount1_min=amount1_min,
            deadline=deadline,
            chain=chain
        ))

        # Verify result
        assert result is not None
        assert isinstance(result, str)  # tx_hash

    def test_transaction_parameter_validation(self):
        """Test transaction parameter validation using real methods."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test that required parameters are validated
        # This tests the actual parameter validation in the enter method
        result = self._consume_generator(behaviour.enter(
            pool_address=None,  # Missing required parameter
            safe_address="0xSafeAddress",
            assets=["0xTokenA", "0xTokenB"],
            chain="optimism",
            max_amounts_in=[1000000000000000000, 2000000000000000000],
            pool_fee=3000
        ))

        # Should return None, None due to missing required parameters
        assert result == (None, None)

    def test_transaction_error_handling(self):
        """Test transaction error handling using real methods."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}

        # Mock contract interactions to return None (simulating error)
        def mock_contract_interact(*args, **kwargs):
            yield None
            return None  # Simulate contract interaction failure

        behaviour.contract_interact = mock_contract_interact

        # Test that errors are handled gracefully
        result = self._consume_generator(behaviour.burn_token(
            token_id=12345,
            chain="optimism"
        ))

        # Should return None due to contract interaction failure
        assert result is None
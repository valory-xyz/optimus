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

"""End-to-end integration tests for Uniswap V3 protocol workflows."""

import pytest
from unittest.mock import MagicMock, patch

from packages.valory.skills.liquidity_trader_abci.pools.uniswap import UniswapPoolBehaviour

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions, TestDataGenerator


class TestUniswapV3E2EWorkflows(ProtocolIntegrationTestBase):
    """Test complete Uniswap V3 protocol workflows."""

    def _consume_generator(self, gen):
        """Helper to run generator and return final value."""
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    def test_uniswap_v3_enter_workflow(self):
        """Test complete workflow for entering a Uniswap V3 position."""
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
        call_count = {"slot0": 0, "mint": 0}
        
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            data_key = kwargs.get('data_key')
            
            print(f"DEBUG: contract_interact called with callable={callable_name}, data_key={data_key}")
            
            # Simulate the contract_interact method behavior
            # It yields first, then returns the data directly
            yield None
            
            if callable_name == "get_pool_fee" and data_key == "data":
                return 3000
            elif callable_name == "get_tick_spacing" and data_key == "data":
                return 60
            elif callable_name == "slot0" and data_key == "slot0":
                call_count["slot0"] += 1
                return {
                    "sqrt_price_x96": 79228162514264337593543950336,
                    "tick": -276310,
                    "unlocked": True
                }
            elif callable_name == "mint":
                call_count["mint"] += 1
                print(f"DEBUG: mint called, returning tx_hash")
                return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            else:
                print(f"DEBUG: No match for callable={callable_name}, data_key={data_key}")
                return None
        
        # Mock the helper methods that enter() calls
        def mock_get_pool_fee(*args, **kwargs):
            print("DEBUG: _get_pool_fee called")
            yield None
            return 3000
        
        def mock_calculate_tick_lower_and_upper(*args, **kwargs):
            print("DEBUG: _calculate_tick_lower_and_upper called")
            yield None
            return -276320, -276300
        
        def mock_calculate_slippage_protection_for_mint(*args, **kwargs):
            print("DEBUG: _calculate_slippage_protection_for_mint called")
            yield None
            return 999000000000000000, 1998000000000000000  # 0.1% slippage
        
        behaviour.contract_interact = mock_contract_interact
        behaviour._get_pool_fee = mock_get_pool_fee
        behaviour._calculate_tick_lower_and_upper = mock_calculate_tick_lower_and_upper
        behaviour._calculate_slippage_protection_for_mint = mock_calculate_slippage_protection_for_mint
        
        # Test enter workflow
        generator = behaviour.enter(
            pool_address=pool_address,
            safe_address=safe_address,
            assets=assets,
            chain=chain,
            max_amounts_in=max_amounts_in,
            pool_fee=pool_fee
        )
        
        print(f"DEBUG: Generator type: {type(generator)}")
        
        result = self._consume_generator(generator)
        
        print(f"DEBUG: Final result: {result}")
        print(f"DEBUG: Result type: {type(result)}")
        print(f"DEBUG: Call counts: {call_count}")
        
        # Verify result
        assert result is not None
        assert len(result) == 2  # (tx_hash, position_manager_address)
        assert isinstance(result[0], str)  # tx_hash
        assert isinstance(result[1], str)  # position_manager_address

    def test_uniswap_v3_exit_workflow(self):
        """Test complete workflow for exiting a Uniswap V3 position."""
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
            
            if callable_name == "get_position" and data_key == "data":
                yield None
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
                yield None
                return {
                    "sqrt_price_x96": 79228162514264337593543950336,
                    "tick": -276310,
                    "unlocked": True
                }
            elif callable_name == "get_tx_data" and data_key == "data":
                yield None
                return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            else:
                yield None
                return None
        
        # Mock the helper methods that exit() calls
        def mock_get_liquidity_for_token(*args, **kwargs):
            print("DEBUG: get_liquidity_for_token called")
            yield None
            return 1000000000000000000
        
        def mock_calculate_slippage_protection_for_decrease(*args, **kwargs):
            print("DEBUG: _calculate_slippage_protection_for_decrease called")
            yield None
            return 999000000000000000, 1998000000000000000  # 0.1% slippage
        
        def mock_decrease_liquidity(*args, **kwargs):
            print("DEBUG: decrease_liquidity called")
            yield None
            return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        
        def mock_collect_tokens(*args, **kwargs):
            print("DEBUG: collect_tokens called")
            yield None
            return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        
        behaviour.contract_interact = mock_contract_interact
        behaviour.get_liquidity_for_token = mock_get_liquidity_for_token
        behaviour._calculate_slippage_protection_for_decrease = mock_calculate_slippage_protection_for_decrease
        behaviour.decrease_liquidity = mock_decrease_liquidity
        behaviour.collect_tokens = mock_collect_tokens
        
        # Test exit workflow
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

    def test_uniswap_v3_burn_token_workflow(self):
        """Test workflow for burning a Uniswap V3 position."""
        # Setup test data
        token_id = 12345
        chain = "optimism"
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses and parameters
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}
        behaviour.params.slippage_tolerance = 0.01  # 1% slippage tolerance
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            
            if callable_name == "burn_token":
                yield None
                return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            else:
                yield None
                return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test burn token workflow
        result = self._consume_generator(behaviour.burn_token(
            token_id=token_id,
            chain=chain
        ))
        
        # Verify result
        assert result is not None
        assert isinstance(result, str)  # tx_hash

    def test_uniswap_v3_collect_tokens_workflow(self):
        """Test workflow for collecting tokens from a Uniswap V3 position."""
        # Setup test data
        token_id = 12345
        recipient = "0xRecipient"
        amount0_max = 1000000000000000000
        amount1_max = 2000000000000000000
        chain = "optimism"
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses and parameters
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}
        behaviour.params.slippage_tolerance = 0.01  # 1% slippage tolerance
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            
            if callable_name == "collect_tokens":
                yield None
                return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            else:
                yield None
                return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test collect tokens workflow
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

    def test_uniswap_v3_decrease_liquidity_workflow(self):
        """Test workflow for decreasing liquidity in a Uniswap V3 position."""
        # Setup test data
        token_id = 12345
        liquidity = 1000000000000000000
        amount0_min = 900000000000000000
        amount1_min = 1800000000000000000
        deadline = 1234567890
        chain = "optimism"
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses and parameters
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}
        behaviour.params.slippage_tolerance = 0.01  # 1% slippage tolerance
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            
            if callable_name == "decrease_liquidity":
                yield None
                return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            else:
                yield None
                return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test decrease liquidity workflow
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

    def test_uniswap_v3_get_liquidity_for_token_workflow(self):
        """Test workflow for getting liquidity for a specific token."""
        # Setup test data
        token_id = 12345
        chain = "optimism"
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses and parameters
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}
        behaviour.params.slippage_tolerance = 0.01  # 1% slippage tolerance
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            
            if callable_name == "get_position" and kwargs.get('data_key') == "data":
                yield None
                return [
                    1,  # nonce
                    "0xOperator",  # operator
                    "0xTokenA",  # token0
                    "0xTokenB",  # token1
                    3000,  # fee
                    -276320,  # tickLower
                    -276300,  # tickUpper
                    1000000000000000000,  # liquidity (index 7)
                    0,  # feeGrowthInside0LastX128
                    0,  # feeGrowthInside1LastX128
                    100000000000000000,  # tokensOwed0
                    200000000000000000,  # tokensOwed1
                ]
            else:
                yield None
                return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test get liquidity for token workflow
        result = self._consume_generator(behaviour.get_liquidity_for_token(
            token_id=token_id,
            chain=chain
        ))
        
        # Verify result
        assert result is not None
        assert isinstance(result, int)  # liquidity amount

    def test_uniswap_v3_get_tokens_workflow(self):
        """Test workflow for getting pool tokens."""
        # Setup test data
        pool_address = "0xPoolAddress"
        chain = "optimism"
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses and parameters
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}
        behaviour.params.slippage_tolerance = 0.01  # 1% slippage tolerance
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            data_key = kwargs.get('data_key')
            
            if callable_name == "get_pool_tokens" and data_key == "tokens":
                yield None
                return ["0xTokenA", "0xTokenB"]
            else:
                yield None
                return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test get tokens workflow
        result = self._consume_generator(behaviour._get_tokens(
            pool_address=pool_address,
            chain=chain
        ))
        
        # Verify result
        assert result is not None
        assert isinstance(result, dict)
        assert "token0" in result
        assert "token1" in result
        assert result["token0"] == "0xTokenA"
        assert result["token1"] == "0xTokenB"

    def test_uniswap_v3_get_pool_fee_workflow(self):
        """Test workflow for getting pool fee."""
        # Setup test data
        pool_address = "0xPoolAddress"
        chain = "optimism"
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses and parameters
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}
        behaviour.params.slippage_tolerance = 0.01  # 1% slippage tolerance
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            data_key = kwargs.get('data_key')
            
            if callable_name == "get_pool_fee" and data_key == "data":
                yield None
                return 3000
            else:
                yield None
                return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test get pool fee workflow
        result = self._consume_generator(behaviour._get_pool_fee(
            pool_address=pool_address,
            chain=chain
        ))
        
        # Verify result
        assert result is not None
        assert isinstance(result, int)
        assert result == 3000

    def test_uniswap_v3_get_tick_spacing_workflow(self):
        """Test workflow for getting tick spacing."""
        # Setup test data
        pool_address = "0xPoolAddress"
        chain = "optimism"
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses and parameters
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}
        behaviour.params.slippage_tolerance = 0.01  # 1% slippage tolerance
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            data_key = kwargs.get('data_key')
            
            if callable_name == "get_tick_spacing" and data_key == "data":
                yield None
                return 60
            else:
                yield None
                return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test get tick spacing workflow
        result = self._consume_generator(behaviour._get_tick_spacing(
            pool_address=pool_address,
            chain=chain
        ))
        
        # Verify result
        assert result is not None
        assert isinstance(result, int)
        assert result == 60

    def test_uniswap_v3_calculate_tick_lower_and_upper_workflow(self):
        """Test workflow for calculating tick lower and upper."""
        # Setup test data
        pool_address = "0xPoolAddress"
        chain = "optimism"
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses and parameters
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}
        behaviour.params.slippage_tolerance = 0.01  # 1% slippage tolerance
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            data_key = kwargs.get('data_key')
            
            if callable_name == "get_tick_spacing" and data_key == "data":
                yield None
                return 60
            else:
                yield None
                return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test calculate tick lower and upper workflow
        result = self._consume_generator(behaviour._calculate_tick_lower_and_upper(
            pool_address=pool_address,
            chain=chain
        ))
        
        # Verify result
        assert result is not None
        assert len(result) == 2  # (tick_lower, tick_upper)
        assert isinstance(result[0], int)  # tick_lower
        assert isinstance(result[1], int)  # tick_upper
        assert result[0] < result[1]  # tick_lower < tick_upper

    def test_uniswap_v3_calculate_slippage_protection_for_mint_workflow(self):
        """Test workflow for calculating slippage protection for mint."""
        # Setup test data
        pool_address = "0xPoolAddress"
        tick_lower = -276320
        tick_upper = -276300
        max_amounts_in = [1000000000000000000, 2000000000000000000]
        chain = "optimism"
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses and parameters
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}
        behaviour.params.slippage_tolerance = 0.01  # 1% slippage tolerance
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            data_key = kwargs.get('data_key')
            
            if callable_name == "slot0" and data_key == "slot0":
                yield None
                return {
                    "sqrt_price_x96": 79228162514264337593543950336,
                    "tick": -276310,
                    "unlocked": True
                }
            else:
                yield None
                return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test calculate slippage protection for mint workflow
        result = self._consume_generator(behaviour._calculate_slippage_protection_for_mint(
            pool_address=pool_address,
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            max_amounts_in=max_amounts_in,
            chain=chain
        ))
        
        # Verify result
        assert result is not None
        assert len(result) == 2  # (amount0_min, amount1_min)
        assert isinstance(result[0], int)  # amount0_min
        assert isinstance(result[1], int)  # amount1_min

    def test_uniswap_v3_calculate_slippage_protection_for_decrease_workflow(self):
        """Test workflow for calculating slippage protection for decrease."""
        # Setup test data
        token_id = 12345
        liquidity = 1000000000000000000
        chain = "optimism"
        pool_address = "0xPoolAddress"
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Add required contract addresses and parameters
        behaviour.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPositionManagerAddress"}
        behaviour.params.slippage_tolerance = 0.01  # 1% slippage tolerance
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            callable_name = kwargs.get('contract_callable')
            data_key = kwargs.get('data_key')
            
            if callable_name == "get_position" and data_key == "data":
                yield None
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
                yield None
                return {
                    "sqrt_price_x96": 79228162514264337593543950336,
                    "tick": -276310,
                    "unlocked": True
                }
            else:
                yield None
                return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test calculate slippage protection for decrease workflow
        result = self._consume_generator(behaviour._calculate_slippage_protection_for_decrease(
            token_id=token_id,
            liquidity=liquidity,
            chain=chain,
            pool_address=pool_address
        ))
        
        # Verify result
        assert result is not None
        assert len(result) == 2  # (amount0_min, amount1_min)
        assert isinstance(result[0], int)  # amount0_min
        assert isinstance(result[1], int)  # amount1_min
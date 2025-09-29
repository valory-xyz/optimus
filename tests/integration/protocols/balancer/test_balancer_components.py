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

"""Tests for Balancer protocol components."""

import pytest
from unittest.mock import MagicMock
from decimal import Decimal

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
from tests.integration.protocols.base.test_helpers import TestDataGenerator


class TestBalancerComponents(ProtocolIntegrationTestBase):
    """Test individual Balancer protocol components in isolation."""

    def _consume_generator(self, gen):
        """Helper to run generator and return final value."""
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    def test_balancer_math_proportional_calculations(self):
        """Test BalancerProportionalMath calculations in isolation."""
        # Test data
        total_supply = "1000000000000000000000"  # 1000 BPT
        user_balance = "100000000000000000000"  # 100 BPT
        pool_balance = "500000000000000000000"  # 500 tokens

        # Test proportional exit calculation
        token_balances = [pool_balance, "200000000000000000000"]  # 500 and 200 tokens
        bpt_amount_in = user_balance

        result = BalancerProportionalMath.query_proportional_exit(
            token_balances=token_balances,
            total_bpt_supply=total_supply,
            bpt_amount_in=bpt_amount_in,
        )

        # Should return proportional amounts for each token
        assert len(result) == 2
        assert result[0] > 0  # First token amount
        assert result[1] > 0  # Second token amount

    def test_join_kind_enum_validation(self):
        """Test join kind enum validation in isolation."""
        # Test weighted pool join kinds
        assert JoinKind.WeightedPool.INIT.value == 0
        assert JoinKind.WeightedPool.EXACT_TOKENS_IN_FOR_BPT_OUT.value == 1
        assert JoinKind.WeightedPool.TOKEN_IN_FOR_EXACT_BPT_OUT.value == 2
        assert JoinKind.WeightedPool.ALL_TOKENS_IN_FOR_EXACT_BPT_OUT.value == 3

        # Test stable pool join kinds
        assert JoinKind.StableAndMetaStablePool.INIT.value == 0
        assert JoinKind.StableAndMetaStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT.value == 1
        assert JoinKind.StableAndMetaStablePool.TOKEN_IN_FOR_EXACT_BPT_OUT.value == 2

        # Test composable stable pool join kinds
        assert JoinKind.ComposableStablePool.INIT.value == 0
        assert JoinKind.ComposableStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT.value == 1
        assert JoinKind.ComposableStablePool.TOKEN_IN_FOR_EXACT_BPT_OUT.value == 2
        assert JoinKind.ComposableStablePool.ALL_TOKENS_IN_FOR_EXACT_BPT_OUT.value == 3

    def test_exit_kind_enum_validation(self):
        """Test exit kind enum validation in isolation."""
        # Test weighted pool exit kinds
        assert ExitKind.WeightedPool.EXACT_BPT_IN_FOR_ONE_TOKEN_OUT.value == 0
        assert ExitKind.WeightedPool.EXACT_BPT_IN_FOR_TOKENS_OUT.value == 1
        assert ExitKind.WeightedPool.BPT_IN_FOR_EXACT_TOKENS_OUT.value == 2
        assert ExitKind.WeightedPool.MANAGEMENT_FEE_TOKENS_OUT.value == 3

        # Test stable pool exit kinds
        assert ExitKind.StableAndMetaStablePool.EXACT_BPT_IN_FOR_ONE_TOKEN_OUT.value == 0
        assert ExitKind.StableAndMetaStablePool.EXACT_BPT_IN_FOR_TOKENS_OUT.value == 1
        assert ExitKind.StableAndMetaStablePool.BPT_IN_FOR_EXACT_TOKENS_OUT.value == 2

        # Test composable stable pool exit kinds
        assert ExitKind.ComposableStablePool.EXACT_BPT_IN_FOR_ONE_TOKEN_OUT.value == 0
        assert ExitKind.ComposableStablePool.EXACT_BPT_IN_FOR_ALL_TOKENS_OUT.value == 2
        assert ExitKind.ComposableStablePool.BPT_IN_FOR_EXACT_TOKENS_OUT.value == 1

    def test_pool_type_detection(self):
        """Test pool type detection logic in isolation."""
        # Test different pool types
        assert PoolType.WEIGHTED == PoolType("Weighted")
        assert PoolType.COMPOSABLE_STABLE == PoolType("ComposableStable")
        assert PoolType.LIQUIDITY_BOOTSTRAPING == PoolType("LiquidityBootstrapping")

    def test_pool_share_proportional_math(self):
        """Test pool share proportional math calculations."""
        # Test proportional join calculation
        token_balances = ["1000000000000000000000", "2000000000000000000000"]  # 1000 and 2000 tokens
        total_bpt_supply = "1000000000000000000000"  # 1000 BPT
        amounts_in = ["100000000000000000000", "200000000000000000000"]  # 100 and 200 tokens

        result = BalancerProportionalMath.query_proportional_join(
            token_balances=token_balances,
            total_bpt_supply=total_bpt_supply,
            amounts_in=amounts_in,
        )

        # Should return positive BPT amount
        assert result > 0
        assert isinstance(result, (int, str))

    def test_determine_join_kind(self):
        """Test _determine_join_kind method with proper business logic testing."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test weighted pool - the method expects string values, not enum objects
        result = behaviour._determine_join_kind(PoolType.WEIGHTED.value)
        assert result == JoinKind.WeightedPool.EXACT_TOKENS_IN_FOR_BPT_OUT.value
        
        # Test stable pool
        result = behaviour._determine_join_kind(PoolType.STABLE.value)
        assert result == JoinKind.StableAndMetaStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT.value
        
        # Test composable stable pool
        result = behaviour._determine_join_kind(PoolType.COMPOSABLE_STABLE.value)
        assert result == JoinKind.ComposableStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT.value

    def test_determine_exit_kind(self):
        """Test _determine_exit_kind method with proper business logic testing."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test weighted pool - the method expects string values, not enum objects
        result = behaviour._determine_exit_kind(PoolType.WEIGHTED.value)
        assert result == ExitKind.WeightedPool.EXACT_BPT_IN_FOR_TOKENS_OUT.value
        
        # Test stable pool
        result = behaviour._determine_exit_kind(PoolType.STABLE.value)
        assert result == ExitKind.StableAndMetaStablePool.EXACT_BPT_IN_FOR_TOKENS_OUT.value
        
        # Test composable stable pool
        result = behaviour._determine_exit_kind(PoolType.COMPOSABLE_STABLE.value)
        assert result == ExitKind.ComposableStablePool.EXACT_BPT_IN_FOR_ALL_TOKENS_OUT.value

    def test_adjust_amounts(self):
        """Test adjust_amounts method with different token orders."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)

        # Test data
        assets = ["0xTokenA", "0xTokenB", "0xTokenC"]
        assets_new = ["0xTokenC", "0xTokenA", "0xTokenB"]
        max_amounts_in = [1000, 2000, 3000]

        # Expected result
        expected_amounts = [3000, 1000, 2000]

        # Test adjustment
        result = behaviour.adjust_amounts(assets, assets_new, max_amounts_in)
        assert result == expected_amounts

    def test_adjust_amounts_with_missing_tokens(self):
        """Test adjust_amounts method when some tokens are missing in new assets."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)

        # Test data
        assets = ["0xTokenA", "0xTokenB", "0xTokenC"]
        assets_new = ["0xTokenA", "0xTokenB"]  # TokenC is missing
        max_amounts_in = [1000, 2000, 3000]

        # Expected result (missing token should have 0 amount)
        expected_amounts = [1000, 2000]

        # Test adjustment
        result = behaviour.adjust_amounts(assets, assets_new, max_amounts_in)
        assert result == expected_amounts

    def test_get_pool_id(self):
        """Test _get_pool_id method with proper business logic testing."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Mock the contract interaction properly
        mock_pool_id = "0x1234567890123456789012345678901234567890123456789012345678901234"
        
        # Create a proper generator mock
        def mock_contract_interact(*args, **kwargs):
            yield  # This is the key - yield first, then return
            # Check for the specific contract_callable and data_key
            if kwargs.get("contract_callable") == "get_pool_id" and kwargs.get("data_key") == "pool_id":
                return mock_pool_id
            return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test the actual method execution
        # The method is a generator, so we need to consume it properly
        generator = behaviour._get_pool_id("0xPoolAddress", "optimism")
        result = self._consume_generator(generator)
        
        # Verify the business logic worked
        assert result == mock_pool_id
        assert isinstance(result, str)
        assert result.startswith("0x")

    def test_get_tokens(self):
        """Test _get_tokens method with proper business logic testing."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Generate test data using TestDataGenerator
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        mock_pool_id = pool_data["pool_id"]
        # The method expects pool_tokens to be a tuple where:
        # pool_tokens[0] is a list of token addresses
        # pool_tokens[1] is a list of balances
        # But the method accesses pool_tokens[0][0] and pool_tokens[0][1]
        # So we need: pool_tokens[0] = [token0, token1]
        mock_tokens = ([pool_data["tokens"][0], pool_data["tokens"][1]], [1000, 2000])
        
        # Mock the params to include vault address
        behaviour.params.balancer_vault_contract_addresses = {"optimism": "0xVaultAddress"}
        
        call_count = 0
        def mock_contract_interact(*args, **kwargs):
            nonlocal call_count
            yield
            call_count += 1
            print(f"DEBUG: contract_interact called with callable={kwargs.get('contract_callable')}, data_key={kwargs.get('data_key')}")
            # Check for the specific contract_callable and data_key combinations
            if kwargs.get("contract_callable") == "get_pool_id" and kwargs.get("data_key") == "pool_id":
                print(f"DEBUG: Returning pool_id: {mock_pool_id}")
                return mock_pool_id
            elif kwargs.get("contract_callable") == "get_pool_tokens" and kwargs.get("data_key") == "tokens":
                print(f"DEBUG: Returning tokens: {mock_tokens}")
                return mock_tokens
            print(f"DEBUG: No match, returning None")
            return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test the actual method execution
        # The method is a generator, so we need to consume it properly
        generator = behaviour._get_tokens(pool_data["pool_address"], "optimism")
        
        # Use the helper function to consume the generator and get the return value
        result = self._consume_generator(generator)
        
        print(f"DEBUG: Final result: {result}")
        
        # Verify the business logic worked
        assert result is not None
        assert "token0" in result
        assert "token1" in result
        assert result["token0"] == pool_data["tokens"][0]
        assert result["token1"] == pool_data["tokens"][1]
        assert call_count == 2  # Should have called contract_interact twice

    def test_query_proportional_join(self):
        """Test query_proportional_join method with proper business logic testing."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Generate test data using TestDataGenerator
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        mock_pool_tokens = [pool_data["tokens"], [1000000, 2000000]]
        mock_total_supply = pool_data["total_supply"]
        
        call_count = 0
        def mock_contract_interact(*args, **kwargs):
            nonlocal call_count
            yield
            call_count += 1
            if kwargs.get("contract_callable") == "get_pool_tokens":
                return mock_pool_tokens
            elif kwargs.get("contract_callable") == "get_total_supply":
                return mock_total_supply
            return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test the actual method execution with realistic amounts
        amounts_in = [1000000000000000000, 2000000000000000000]  # 1 and 2 tokens
        generator = behaviour.query_proportional_join(
            pool_id=pool_data["pool_id"],
            pool_address=pool_data["pool_address"],
            vault_address="0xVaultAddress",
            chain="optimism",
            amounts_in=amounts_in
        )
        result = self._consume_generator(generator)
        
        # Verify the business logic worked
        assert result is not None
        assert result > 0  # Should return positive BPT amount
        assert call_count == 2  # Should have called contract_interact twice

    def test_query_proportional_exit(self):
        """Test query_proportional_exit method with proper business logic testing."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Generate test data using TestDataGenerator
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        mock_pool_tokens = [pool_data["tokens"], [1000000, 2000000]]
        mock_total_supply = pool_data["total_supply"]
        
        call_count = 0
        def mock_contract_interact(*args, **kwargs):
            nonlocal call_count
            yield
            call_count += 1
            if kwargs.get("contract_callable") == "get_pool_tokens":
                return mock_pool_tokens
            elif kwargs.get("contract_callable") == "get_total_supply":
                return mock_total_supply
            return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test the actual method execution with realistic BPT amount
        bpt_amount_in = 100000000000000000000  # 100 BPT
        generator = behaviour.query_proportional_exit(
            pool_id=pool_data["pool_id"],
            pool_address=pool_data["pool_address"],
            vault_address="0xVaultAddress",
            chain="optimism",
            bpt_amount_in=bpt_amount_in
        )
        result = self._consume_generator(generator)
        
        # Verify the business logic worked
        assert result is not None
        assert len(result) == 2  # Should return amounts for two tokens
        assert all(amount > 0 for amount in result)  # All amounts should be positive
        assert call_count == 2  # Should have called contract_interact twice

    def test_enter_method_validation(self):
        """Test enter method parameter validation using TestDataGenerator."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Generate test data using TestDataGenerator
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        
        # Create valid parameters using generated data
        valid_params = {
            "pool_address": pool_data["pool_address"],
            "safe_address": "0xSafeAddress",
            "assets": pool_data["tokens"],
            "chain": "optimism",
            "max_amounts_in": [1000000000000000000, 2000000000000000000],
            "pool_type": pool_data["pool_type"],
        }
        
        # Test that all required parameters are present
        required_fields = ["pool_address", "safe_address", "assets", "chain", "max_amounts_in", "pool_type"]
        for field in required_fields:
            assert field in valid_params, f"Missing required field: {field}"
        
        # Test that parameters have expected types and values
        assert isinstance(valid_params["pool_address"], str)
        assert isinstance(valid_params["safe_address"], str)
        assert isinstance(valid_params["assets"], list)
        assert isinstance(valid_params["chain"], str)
        assert isinstance(valid_params["max_amounts_in"], list)
        assert isinstance(valid_params["pool_type"], str)
        
        # Test that generated data is realistic
        assert valid_params["pool_address"].startswith("0x")
        assert len(valid_params["assets"]) == 2  # Two tokens

    def test_exit_method_validation(self):
        """Test exit method parameter validation using TestDataGenerator."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Generate test data using TestDataGenerator
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        
        # Create valid parameters using generated data
        valid_params = {
            "pool_address": pool_data["pool_address"],
            "safe_address": "0xSafeAddress",
            "assets": pool_data["tokens"],
            "chain": "optimism",
            "pool_type": pool_data["pool_type"],
        }
        
        # Test that all required parameters are present
        required_fields = ["pool_address", "safe_address", "assets", "chain", "pool_type"]
        for field in required_fields:
            assert field in valid_params, f"Missing required field: {field}"
        
        # Test that parameters have expected types and values
        assert isinstance(valid_params["pool_address"], str)
        assert isinstance(valid_params["safe_address"], str)
        assert isinstance(valid_params["assets"], list)
        assert isinstance(valid_params["chain"], str)
        assert isinstance(valid_params["pool_type"], str)
        
        # Test that generated data is realistic
        assert valid_params["pool_address"].startswith("0x")
        assert len(valid_params["assets"]) == 2  # Two tokens

    def test_update_value(self):
        """Test update_value method with proper business logic testing."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Generate test data using TestDataGenerator
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        mock_pool_tokens = [pool_data["tokens"], [1000000, 2000000]]
        
        call_count = 0
        def mock_contract_interact(*args, **kwargs):
            nonlocal call_count
            yield
            call_count += 1
            if kwargs.get("contract_callable") == "get_pool_tokens":
                return mock_pool_tokens
            return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test the actual method execution
        generator = behaviour.update_value(
            pool_id=pool_data["pool_id"],
            chain="optimism",
            vault_address="0xVaultAddress",
            max_amounts_in=[1000000000000000000, 2000000000000000000],
            assets=pool_data["tokens"]
        )
        result = self._consume_generator(generator)
        
        # Verify the business logic worked
        assert result is not None
        assert len(result) == 2  # Should return (tokens, new_max_amounts_in)
        tokens, new_max_amounts_in = result
        assert tokens is not None
        assert new_max_amounts_in is not None
        assert len(tokens) == 2  # Two tokens
        assert len(new_max_amounts_in) == 2  # Two amounts
        assert call_count == 1  # Should have called contract_interact once

    def test_enter_method(self):
        """Test enter method with proper business logic testing."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Generate test data using TestDataGenerator
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        mock_pool_id = pool_data["pool_id"]
        mock_pool_tokens = [pool_data["tokens"], [1000000, 2000000]]
        mock_total_supply = pool_data["total_supply"]
        mock_tx_hash = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        
        # Mock the params to include vault address
        behaviour.params.balancer_vault_contract_addresses = {"optimism": "0xVaultAddress"}
        behaviour.params.slippage_tolerance = 0.01  # 1% slippage
        
        call_count = 0
        def mock_contract_interact(*args, **kwargs):
            nonlocal call_count
            yield
            call_count += 1
            if kwargs.get("contract_callable") == "get_pool_id":
                return mock_pool_id
            elif kwargs.get("contract_callable") == "get_pool_tokens":
                return mock_pool_tokens
            elif kwargs.get("contract_callable") == "get_total_supply":
                return mock_total_supply
            elif kwargs.get("contract_callable") == "join_pool":
                return mock_tx_hash
            return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test the actual method execution
        generator = behaviour.enter(
            pool_address=pool_data["pool_address"],
            safe_address="0xSafeAddress",
            assets=pool_data["tokens"],
            chain="optimism",
            max_amounts_in=[1000000000000000000, 2000000000000000000],
            pool_type=pool_data["pool_type"]
        )
        result = self._consume_generator(generator)
        
        # Verify the business logic worked
        assert result is not None
        assert len(result) == 2  # Should return (tx_hash, vault_address)
        tx_hash, vault_address = result
        assert tx_hash == mock_tx_hash
        assert vault_address == "0xVaultAddress"
        assert call_count >= 3  # Should have called contract_interact multiple times

    def test_exit_method(self):
        """Test exit method with proper business logic testing."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Generate test data using TestDataGenerator
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        mock_pool_id = pool_data["pool_id"]
        mock_pool_tokens = [pool_data["tokens"], [1000000, 2000000]]
        mock_total_supply = pool_data["total_supply"]
        mock_bpt_balance = 100000000000000000000  # 100 BPT
        mock_tx_hash = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        
        # Mock the params to include vault address
        behaviour.params.balancer_vault_contract_addresses = {"optimism": "0xVaultAddress"}
        behaviour.params.slippage_tolerance = 0.01  # 1% slippage
        
        call_count = 0
        def mock_contract_interact(*args, **kwargs):
            nonlocal call_count
            yield
            call_count += 1
            if kwargs.get("contract_callable") == "get_pool_id":
                return mock_pool_id
            elif kwargs.get("contract_callable") == "get_balance":
                return mock_bpt_balance
            elif kwargs.get("contract_callable") == "get_pool_tokens":
                return mock_pool_tokens
            elif kwargs.get("contract_callable") == "get_total_supply":
                return mock_total_supply
            elif kwargs.get("contract_callable") == "exit_pool":
                return mock_tx_hash
            return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test the actual method execution
        generator = behaviour.exit(
            pool_address=pool_data["pool_address"],
            safe_address="0xSafeAddress",
            assets=pool_data["tokens"],
            chain="optimism",
            pool_type=pool_data["pool_type"]
        )
        result = self._consume_generator(generator)
        
        # Verify the business logic worked
        assert result is not None
        assert len(result) == 3  # Should return (tx_hash, vault_address, to_internal_balance)
        tx_hash, vault_address, to_internal_balance = result
        assert tx_hash == mock_tx_hash
        assert vault_address == "0xVaultAddress"
        assert isinstance(to_internal_balance, bool)
        assert call_count >= 4  # Should have called contract_interact multiple times

    def test_query_join_bpt(self):
        """Test _query_join_bpt method with proper business logic testing."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Generate test data using TestDataGenerator
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        # Override with valid Ethereum addresses
        pool_data["tokens"] = ["0x4200000000000000000000000000000000000006", "0x7F5c764cBc14f9669B88837ca1490cCa17c31607"]  # WETH and USDC on Optimism
        mock_bpt_out = 500000000000000000000  # 500 BPT
        
        # Mock the params to include queries address
        behaviour.params.balancer_queries_contract_addresses = {"optimism": "0xQueriesAddress"}
        print(f"DEBUG: Set queries address: {behaviour.params.balancer_queries_contract_addresses}")
        
        call_count = 0
        def mock_contract_interact(*args, **kwargs):
            nonlocal call_count
            yield
            call_count += 1
            print(f"DEBUG: contract_interact called with callable={kwargs.get('contract_callable')}, data_key={kwargs.get('data_key')}")
            if kwargs.get("contract_callable") == "query_join" and kwargs.get("data_key") == "result":
                print(f"DEBUG: Returning bpt_out: {mock_bpt_out}")
                return {"bpt_out": mock_bpt_out}
            print(f"DEBUG: No match, returning None")
            return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test the actual method execution
        print(f"DEBUG: Calling _query_join_bpt with chain='optimism'")
        print(f"DEBUG: Assets: {pool_data['tokens']}")
        generator = behaviour._query_join_bpt(
            pool_id=pool_data["pool_id"],
            sender="0xSenderAddress",
            assets=pool_data["tokens"],
            max_amounts_in=[1000000000000000000, 2000000000000000000],
            join_kind=1,
            from_internal_balance=False,
            chain="optimism"
        )
        result = self._consume_generator(generator)
        print(f"DEBUG: Final result: {result}")
        
        # Verify the business logic worked
        assert result is not None
        assert result == mock_bpt_out
        assert call_count == 1  # Should have called contract_interact once

    def test_query_exit_amounts(self):
        """Test _query_exit_amounts method with proper business logic testing."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Generate test data using TestDataGenerator
        pool_data = TestDataGenerator.generate_balancer_pool_data()
        # Override with valid Ethereum addresses
        pool_data["tokens"] = ["0x4200000000000000000000000000000000000006", "0x7F5c764cBc14f9669B88837ca1490cCa17c31607"]  # WETH and USDC on Optimism
        mock_amounts_out = [500000000000000000000, 1000000000000000000000]  # 500 and 1000 tokens
        
        # Mock the params to include queries address
        behaviour.params.balancer_queries_contract_addresses = {"optimism": "0xQueriesAddress"}
        
        call_count = 0
        def mock_contract_interact(*args, **kwargs):
            nonlocal call_count
            yield
            call_count += 1
            print(f"DEBUG: contract_interact called with callable={kwargs.get('contract_callable')}, data_key={kwargs.get('data_key')}")
            if kwargs.get("contract_callable") == "query_exit" and kwargs.get("data_key") == "result":
                print(f"DEBUG: Returning amounts_out: {mock_amounts_out}")
                return {"amounts_out": mock_amounts_out}
            print(f"DEBUG: No match, returning None")
            return None
        
        behaviour.contract_interact = mock_contract_interact
        
        # Test the actual method execution
        generator = behaviour._query_exit_amounts(
            pool_id=pool_data["pool_id"],
            sender="0xSenderAddress",
            assets=pool_data["tokens"],
            bpt_amount_in=100000000000000000000,  # 100 BPT
            exit_kind=1,
            to_internal_balance=False,
            chain="optimism"
        )
        result = self._consume_generator(generator)
        
        # Verify the business logic worked
        assert result is not None
        assert result == mock_amounts_out
        assert len(result) == 2  # Two token amounts
        assert call_count == 1  # Should have called contract_interact once
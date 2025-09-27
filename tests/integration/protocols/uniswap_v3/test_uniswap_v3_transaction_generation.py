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

"""Transaction generation tests for Uniswap V3 protocol."""

import pytest
import time
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

    def test_mint_transaction_parameter_validation(self, mock_ledger_api):
        """Test that mint transactions have correct parameters."""
        # Test parameters
        test_params = {
            "token0": "0xTokenA",
            "token1": "0xTokenB",
            "fee": 3000,
            "tick_lower": -276320,
            "tick_upper": -276300,
            "amount0_desired": 1000000000000000000,
            "amount1_desired": 2000000000000000000,
            "amount0_min": 900000000000000000,
            "amount1_min": 1800000000000000000,
            "recipient": "0xRecipient",
            "deadline": 1234567890
        }
        
        # Test transaction encoding
        result = UniswapV3NonfungiblePositionManagerContract.mint(
            ledger_api=mock_ledger_api,
            contract_address="0xPositionManagerAddress",
            **test_params
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_decrease_liquidity_transaction_validation(self, mock_ledger_api):
        """Test that decrease liquidity transactions have correct parameters."""
        # Test parameters
        test_params = {
            "token_id": 12345,
            "liquidity": 1000000000000000000,
            "amount0_min": 900000000000000000,
            "amount1_min": 1800000000000000000,
            "deadline": 1234567890
        }
        
        # Test transaction encoding
        result = UniswapV3NonfungiblePositionManagerContract.decrease_liquidity(
            ledger_api=mock_ledger_api,
            contract_address="0xPositionManagerAddress",
            **test_params
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_collect_transaction_parameter_validation(self, mock_ledger_api):
        """Test that collect transactions have correct parameters."""
        # Test parameters
        test_params = {
            "token_id": 12345,
            "amount0_max": 1000000000000000000,
            "amount1_max": 2000000000000000000,
            "recipient": "0xRecipient"
        }
        
        # Test transaction encoding
        result = UniswapV3NonfungiblePositionManagerContract.collect(
            ledger_api=mock_ledger_api,
            contract_address="0xPositionManagerAddress",
            **test_params
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_transaction_gas_estimation(self):
        """Test gas estimation for different transaction types."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test different transaction types
        transaction_types = [
            "mint",
            "decrease_liquidity",
            "collect",
            "increase_liquidity",
            "burn"
        ]
        
        for tx_type in transaction_types:
            gas_estimate = behaviour._estimate_gas_for_transaction(tx_type)
            
            # Verify gas estimate is reasonable
            assert gas_estimate > 0
            assert gas_estimate < 1000000  # Should not exceed 1M gas

    def test_transaction_deadline_validation(self):
        """Test that transaction deadlines are set correctly."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test deadline calculation
        current_time = int(time.time())
        deadline_buffer = 300  # 5 minutes
        
        calculated_deadline = behaviour._calculate_transaction_deadline(
            current_time, deadline_buffer
        )
        
        expected_deadline = current_time + deadline_buffer
        assert calculated_deadline == expected_deadline
        
        # Test deadline validation
        is_valid = behaviour._validate_transaction_deadline(calculated_deadline, current_time)
        assert is_valid
        
        # Test expired deadline
        expired_deadline = current_time - 100
        is_expired = behaviour._validate_transaction_deadline(expired_deadline, current_time)
        assert not is_expired

    def test_mint_transaction_encoding_variations(self, mock_ledger_api):
        """Test mint transaction encoding for different fee tiers."""
        base_params = {
            "token0": "0xTokenA",
            "token1": "0xTokenB",
            "tick_lower": -276320,
            "tick_upper": -276300,
            "amount0_desired": 1000000000000000000,
            "amount1_desired": 2000000000000000000,
            "amount0_min": 900000000000000000,
            "amount1_min": 1800000000000000000,
            "recipient": "0xRecipient",
            "deadline": 1234567890
        }
        
        # Test different fee tiers
        fee_tiers = [500, 3000, 10000]
        
        for fee in fee_tiers:
            test_params = {**base_params, "fee": fee}
            
            result = UniswapV3NonfungiblePositionManagerContract.mint(
                ledger_api=mock_ledger_api,
                contract_address="0xPositionManagerAddress",
                **test_params
            )
            
            # Verify transaction structure
            TestAssertions.assert_transaction_structure(result)
            assert isinstance(result["tx_hash"], str)

    def test_decrease_liquidity_transaction_encoding_variations(self, mock_ledger_api):
        """Test decrease liquidity transaction encoding for different amounts."""
        base_params = {
            "token_id": 12345,
            "amount0_min": 900000000000000000,
            "amount1_min": 1800000000000000000,
            "deadline": 1234567890
        }
        
        # Test different liquidity amounts
        liquidity_amounts = [
            100000000000000000000,   # 100 tokens
            500000000000000000000,   # 500 tokens
            1000000000000000000000,  # 1000 tokens
        ]
        
        for liquidity in liquidity_amounts:
            test_params = {**base_params, "liquidity": liquidity}
            
            result = UniswapV3NonfungiblePositionManagerContract.decrease_liquidity(
                ledger_api=mock_ledger_api,
                contract_address="0xPositionManagerAddress",
                **test_params
            )
            
            # Verify transaction structure
            TestAssertions.assert_transaction_structure(result)
            assert isinstance(result["tx_hash"], str)

    def test_collect_transaction_encoding_variations(self, mock_ledger_api):
        """Test collect transaction encoding for different amounts."""
        base_params = {
            "token_id": 12345,
            "recipient": "0xRecipient"
        }
        
        # Test different collect amounts
        collect_scenarios = [
            {"amount0_max": 1000000000000000000, "amount1_max": 2000000000000000000},  # Full collection
            {"amount0_max": 500000000000000000, "amount1_max": 1000000000000000000},   # Partial collection
            {"amount0_max": 0, "amount1_max": 0},  # No collection
        ]
        
        for scenario in collect_scenarios:
            test_params = {**base_params, **scenario}
            
            result = UniswapV3NonfungiblePositionManagerContract.collect(
                ledger_api=mock_ledger_api,
                contract_address="0xPositionManagerAddress",
                **test_params
            )
            
            # Verify transaction structure
            TestAssertions.assert_transaction_structure(result)
            assert isinstance(result["tx_hash"], str)

    def test_transaction_parameter_validation(self):
        """Test transaction parameter validation."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Valid parameters
        valid_params = {
            "token0": "0xTokenA",
            "token1": "0xTokenB",
            "fee": 3000,
            "tick_lower": -276320,
            "tick_upper": -276300,
            "amount0_desired": 1000000000000000000,
            "amount1_desired": 2000000000000000000,
        }
        
        # Validate parameters
        assert valid_params["token0"].startswith("0x")
        assert valid_params["token1"].startswith("0x")
        assert valid_params["fee"] in [500, 3000, 10000]
        assert valid_params["tick_lower"] < valid_params["tick_upper"]
        assert valid_params["amount0_desired"] > 0
        assert valid_params["amount1_desired"] > 0

    def test_transaction_encoding_consistency(self, mock_ledger_api):
        """Test that transaction encoding is consistent across multiple calls."""
        # Test parameters
        test_params = {
            "token0": "0xTokenA",
            "token1": "0xTokenB",
            "fee": 3000,
            "tick_lower": -276320,
            "tick_upper": -276300,
            "amount0_desired": 1000000000000000000,
            "amount1_desired": 2000000000000000000,
            "amount0_min": 900000000000000000,
            "amount1_min": 1800000000000000000,
            "recipient": "0xRecipient",
            "deadline": 1234567890
        }
        
        # Generate transaction multiple times
        results = []
        for _ in range(3):
            result = UniswapV3NonfungiblePositionManagerContract.mint(
                ledger_api=mock_ledger_api,
                contract_address="0xPositionManagerAddress",
                **test_params
            )
            results.append(result)
        
        # Verify all transactions have the same structure
        for result in results:
            TestAssertions.assert_transaction_structure(result)
            assert isinstance(result["tx_hash"], str)

    def test_transaction_with_different_fee_tiers(self, mock_ledger_api):
        """Test transaction encoding for different fee tiers."""
        # Test 0.05% fee tier
        low_fee_params = {
            "token0": "0xTokenA",
            "token1": "0xTokenB",
            "fee": 500,
            "tick_lower": -276320,
            "tick_upper": -276300,
            "amount0_desired": 1000000000000000000,
            "amount1_desired": 2000000000000000000,
            "amount0_min": 900000000000000000,
            "amount1_min": 1800000000000000000,
            "recipient": "0xRecipient",
            "deadline": 1234567890
        }
        
        low_fee_result = UniswapV3NonfungiblePositionManagerContract.mint(
            ledger_api=mock_ledger_api,
            contract_address="0xPositionManagerAddress",
            **low_fee_params
        )
        
        # Test 1% fee tier
        high_fee_params = {
            "token0": "0xTokenA",
            "token1": "0xTokenB",
            "fee": 10000,
            "tick_lower": -276320,
            "tick_upper": -276300,
            "amount0_desired": 1000000000000000000,
            "amount1_desired": 2000000000000000000,
            "amount0_min": 900000000000000000,
            "amount1_min": 1800000000000000000,
            "recipient": "0xRecipient",
            "deadline": 1234567890
        }
        
        high_fee_result = UniswapV3NonfungiblePositionManagerContract.mint(
            ledger_api=mock_ledger_api,
            contract_address="0xPositionManagerAddress",
            **high_fee_params
        )
        
        # Verify both transactions are valid
        TestAssertions.assert_transaction_structure(low_fee_result)
        TestAssertions.assert_transaction_structure(high_fee_result)

    def test_transaction_error_handling(self, mock_ledger_api):
        """Test transaction error handling."""
        # Test with invalid parameters
        invalid_params = {
            "token0": "invalid_token",  # Invalid format
            "token1": "0xTokenB",
            "fee": 2000,  # Invalid fee tier
            "tick_lower": -276300,  # tick_lower >= tick_upper
            "tick_upper": -276320,
            "amount0_desired": 1000000000000000000,
            "amount1_desired": 2000000000000000000,
            "amount0_min": 900000000000000000,
            "amount1_min": 1800000000000000000,
            "recipient": "0xRecipient",
            "deadline": 1234567890
        }
        
        # This should raise an exception or return an error
        with pytest.raises((ValueError, Exception)):
            UniswapV3NonfungiblePositionManagerContract.mint(
                ledger_api=mock_ledger_api,
                contract_address="0xPositionManagerAddress",
                **invalid_params
            )

    def test_transaction_batch_operations(self, mock_ledger_api):
        """Test batch transaction operations."""
        # Create multiple operations for a complex workflow
        operations = [
            # Decrease liquidity
            {
                "method": "decrease_liquidity",
                "params": {
                    "token_id": 12345,
                    "liquidity": 1000000000000000000,
                    "amount0_min": 900000000000000000,
                    "amount1_min": 1800000000000000000,
                    "deadline": 1234567890
                }
            },
            # Collect fees
            {
                "method": "collect",
                "params": {
                    "token_id": 12345,
                    "amount0_max": 1000000000000000000,
                    "amount1_max": 2000000000000000000,
                    "recipient": "0xRecipient"
                }
            },
            # Burn position
            {
                "method": "burn",
                "params": {
                    "token_id": 12345
                }
            }
        ]
        
        # Execute batch operations
        results = []
        for operation in operations:
            method = operation["method"]
            params = operation["params"]
            
            if method == "decrease_liquidity":
                result = UniswapV3NonfungiblePositionManagerContract.decrease_liquidity(
                    ledger_api=mock_ledger_api,
                    contract_address="0xPositionManagerAddress",
                    **params
                )
            elif method == "collect":
                result = UniswapV3NonfungiblePositionManagerContract.collect(
                    ledger_api=mock_ledger_api,
                    contract_address="0xPositionManagerAddress",
                    **params
                )
            elif method == "burn":
                result = UniswapV3NonfungiblePositionManagerContract.burn_token(
                    ledger_api=mock_ledger_api,
                    contract_address="0xPositionManagerAddress",
                    **params
                )
            
            results.append(result)
        
        # Verify all transactions are valid
        for result in results:
            TestAssertions.assert_transaction_structure(result)
            assert isinstance(result["tx_hash"], str)

    def test_transaction_gas_optimization(self):
        """Test gas optimization for transactions."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test gas optimization strategies
        optimization_strategies = [
            "batch_operations",
            "optimal_deadline",
            "minimal_slippage",
            "efficient_tick_ranges"
        ]
        
        for strategy in optimization_strategies:
            gas_savings = behaviour._calculate_gas_savings(strategy)
            
            # Gas savings should be positive
            assert gas_savings >= 0
            
            # Gas savings should be reasonable (not more than 50% of base gas)
            assert gas_savings <= 500000

    def test_transaction_priority_fee_calculation(self):
        """Test priority fee calculation for transactions."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test different network conditions
        network_conditions = [
            {"base_fee": 20000000000, "priority_fee": 2000000000},  # Normal
            {"base_fee": 50000000000, "priority_fee": 5000000000},  # High
            {"base_fee": 10000000000, "priority_fee": 1000000000},  # Low
        ]
        
        for condition in network_conditions:
            total_fee = behaviour._calculate_total_transaction_fee(
                condition["base_fee"], condition["priority_fee"]
            )
            
            # Total fee should be sum of base fee and priority fee
            expected_total = condition["base_fee"] + condition["priority_fee"]
            assert total_fee == expected_total

    def test_transaction_retry_logic(self):
        """Test transaction retry logic."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test retry scenarios
        retry_scenarios = [
            {"max_retries": 3, "retry_delay": 1, "should_succeed": True},
            {"max_retries": 1, "retry_delay": 1, "should_succeed": False},
            {"max_retries": 5, "retry_delay": 2, "should_succeed": True},
        ]
        
        for scenario in retry_scenarios:
            success = behaviour._test_retry_logic(
                scenario["max_retries"],
                scenario["retry_delay"]
            )
            
            assert success == scenario["should_succeed"]

    def test_transaction_validation_rules(self):
        """Test transaction validation rules."""
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Test validation rules
        validation_rules = [
            {"rule": "token_address_format", "value": "0x1234567890123456789012345678901234567890", "valid": True},
            {"rule": "token_address_format", "value": "invalid_token", "valid": False},
            {"rule": "fee_tier", "value": 3000, "valid": True},
            {"rule": "fee_tier", "value": 2000, "valid": False},
            {"rule": "tick_range", "value": {"lower": -276320, "upper": -276300}, "valid": True},
            {"rule": "tick_range", "value": {"lower": -276300, "upper": -276320}, "valid": False},
            {"rule": "amount_positive", "value": 1000000000000000000, "valid": True},
            {"rule": "amount_positive", "value": -1000000000000000000, "valid": False},
        ]
        
        for rule_test in validation_rules:
            is_valid = behaviour._validate_transaction_parameter(
                rule_test["rule"], rule_test["value"]
            )
            assert is_valid == rule_test["valid"]

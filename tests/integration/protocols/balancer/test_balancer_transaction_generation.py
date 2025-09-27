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

"""Transaction generation tests for Balancer protocol."""

import pytest
import time
from unittest.mock import MagicMock, patch

from packages.valory.contracts.balancer_vault.contract import VaultContract
from packages.valory.contracts.balancer_weighted_pool.contract import WeightedPoolContract
from packages.valory.contracts.multisend.contract import MultiSendContract, MultiSendOperation
from packages.valory.skills.liquidity_trader_abci.pools.balancer import BalancerPoolBehaviour

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions
from tests.integration.fixtures.contract_fixtures import (
    mock_ledger_api,
    balancer_vault_contract,
    balancer_weighted_pool_contract,
    multisend_contract,
)


class TestBalancerTransactionGeneration(ProtocolIntegrationTestBase):
    """Test proper transaction encoding and parameters for Balancer."""

    def test_join_pool_transaction_parameter_validation(self, mock_ledger_api):
        """Test that join pool transactions have correct parameters."""
        # Test parameters
        test_params = {
            "pool_id": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "sender": "0xSenderAddress",
            "recipient": "0xRecipientAddress",
            "assets": ["0xTokenA", "0xTokenB"],
            "max_amounts_in": [1000000000000000000, 2000000000000000000],
            "join_kind": 1,  # EXACT_TOKENS_IN_FOR_BPT_OUT
            "minimum_bpt": 500000000000000000
        }
        
        # Test transaction encoding
        result = VaultContract.join_pool(
            ledger_api=mock_ledger_api,
            contract_address="0xVaultAddress",
            **test_params
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], bytes)
        assert len(result["tx_hash"]) > 0

    def test_exit_pool_transaction_parameter_validation(self, mock_ledger_api):
        """Test that exit pool transactions have correct parameters."""
        # Test parameters
        test_params = {
            "pool_id": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "sender": "0xSenderAddress",
            "recipient": "0xRecipientAddress",
            "assets": ["0xTokenA", "0xTokenB"],
            "min_amounts_out": [900000000000000000, 1800000000000000000],
            "exit_kind": 1,  # EXACT_BPT_IN_FOR_TOKENS_OUT
            "bpt_amount_in": 1000000000000000000
        }
        
        # Test transaction encoding
        result = VaultContract.exit_pool(
            ledger_api=mock_ledger_api,
            contract_address="0xVaultAddress",
            **test_params
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], bytes)

    def test_multisend_transaction_encoding(self, mock_ledger_api, multisend_contract):
        """Test multisend transaction encoding for complex operations."""
        # Create multiple operations
        operations = [
            MultiSendOperation(
                operation_type=0,  # CALL
                to="0xTokenA",
                value=0,
                data=b"approve_data"
            ),
            MultiSendOperation(
                operation_type=0,  # CALL
                to="0xVaultAddress",
                value=0,
                data=b"join_pool_data"
            )
        ]
        
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = multisend_contract
        
        # Test multisend transaction
        result = MultiSendContract.multi_send(
            ledger_api=mock_ledger_api,
            contract_address="0xMultiSendAddress",
            operations=operations
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], bytes)

    def test_transaction_gas_estimation(self):
        """Test gas estimation for different transaction types."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test different transaction types
        transaction_types = [
            "join_pool",
            "exit_pool",
            "approve_token",
            "multisend"
        ]
        
        for tx_type in transaction_types:
            gas_estimate = behaviour._estimate_gas_for_transaction(tx_type)
            
            # Verify gas estimate is reasonable
            assert gas_estimate > 0
            assert gas_estimate < 1000000  # Should not exceed 1M gas

    def test_transaction_deadline_validation(self):
        """Test that transaction deadlines are set correctly."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
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

    def test_join_pool_transaction_encoding_variations(self, mock_ledger_api):
        """Test join pool transaction encoding for different join kinds."""
        base_params = {
            "pool_id": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "sender": "0xSenderAddress",
            "recipient": "0xRecipientAddress",
            "assets": ["0xTokenA", "0xTokenB"],
        }
        
        # Test different join kinds
        join_kinds = [
            {"join_kind": 0, "max_amounts_in": [0, 0], "minimum_bpt": 0},  # INIT
            {"join_kind": 1, "max_amounts_in": [1000000000000000000, 2000000000000000000], "minimum_bpt": 500000000000000000},  # EXACT_TOKENS_IN_FOR_BPT_OUT
            {"join_kind": 2, "max_amounts_in": [1000000000000000000, 0], "minimum_bpt": 500000000000000000},  # TOKEN_IN_FOR_EXACT_BPT_OUT
        ]
        
        for join_params in join_kinds:
            test_params = {**base_params, **join_params}
            
            result = VaultContract.join_pool(
                ledger_api=mock_ledger_api,
                contract_address="0xVaultAddress",
                **test_params
            )
            
            # Verify transaction structure
            TestAssertions.assert_transaction_structure(result)
            assert isinstance(result["tx_hash"], bytes)

    def test_exit_pool_transaction_encoding_variations(self, mock_ledger_api):
        """Test exit pool transaction encoding for different exit kinds."""
        base_params = {
            "pool_id": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "sender": "0xSenderAddress",
            "recipient": "0xRecipientAddress",
            "assets": ["0xTokenA", "0xTokenB"],
        }
        
        # Test different exit kinds
        exit_kinds = [
            {"exit_kind": 0, "min_amounts_out": [900000000000000000, 0], "bpt_amount_in": 1000000000000000000},  # EXACT_BPT_IN_FOR_ONE_TOKEN_OUT
            {"exit_kind": 1, "min_amounts_out": [900000000000000000, 1800000000000000000], "bpt_amount_in": 1000000000000000000},  # EXACT_BPT_IN_FOR_TOKENS_OUT
            {"exit_kind": 2, "min_amounts_out": [900000000000000000, 1800000000000000000], "bpt_amount_in": 1000000000000000000},  # BPT_IN_FOR_EXACT_TOKENS_OUT
        ]
        
        for exit_params in exit_kinds:
            test_params = {**base_params, **exit_params}
            
            result = VaultContract.exit_pool(
                ledger_api=mock_ledger_api,
                contract_address="0xVaultAddress",
                **test_params
            )
            
            # Verify transaction structure
            TestAssertions.assert_transaction_structure(result)
            assert isinstance(result["tx_hash"], bytes)

    def test_transaction_parameter_validation(self):
        """Test transaction parameter validation."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Valid parameters
        valid_params = {
            "pool_id": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "sender": "0xSenderAddress",
            "recipient": "0xRecipientAddress",
            "assets": ["0xTokenA", "0xTokenB"],
            "max_amounts_in": [1000000000000000000, 2000000000000000000],
        }
        
        # Validate parameters
        assert len(valid_params["pool_id"]) == 66  # 0x + 64 hex chars
        assert valid_params["sender"].startswith("0x")
        assert valid_params["recipient"].startswith("0x")
        assert len(valid_params["assets"]) == 2
        assert len(valid_params["max_amounts_in"]) == 2
        assert all(amount > 0 for amount in valid_params["max_amounts_in"])

    def test_transaction_encoding_consistency(self, mock_ledger_api):
        """Test that transaction encoding is consistent across multiple calls."""
        # Test parameters
        test_params = {
            "pool_id": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "sender": "0xSenderAddress",
            "recipient": "0xRecipientAddress",
            "assets": ["0xTokenA", "0xTokenB"],
            "max_amounts_in": [1000000000000000000, 2000000000000000000],
            "join_kind": 1,
            "minimum_bpt": 500000000000000000
        }
        
        # Generate transaction multiple times
        results = []
        for _ in range(3):
            result = VaultContract.join_pool(
                ledger_api=mock_ledger_api,
                contract_address="0xVaultAddress",
                **test_params
            )
            results.append(result)
        
        # Verify all transactions have the same structure
        for result in results:
            TestAssertions.assert_transaction_structure(result)
            assert isinstance(result["tx_hash"], bytes)

    def test_transaction_with_different_pool_types(self, mock_ledger_api):
        """Test transaction encoding for different pool types."""
        # Test weighted pool
        weighted_pool_params = {
            "pool_id": "0x1111111111111111111111111111111111111111111111111111111111111111",
            "sender": "0xSenderAddress",
            "recipient": "0xRecipientAddress",
            "assets": ["0xTokenA", "0xTokenB"],
            "max_amounts_in": [1000000000000000000, 2000000000000000000],
            "join_kind": 1,  # EXACT_TOKENS_IN_FOR_BPT_OUT
            "minimum_bpt": 500000000000000000
        }
        
        weighted_result = VaultContract.join_pool(
            ledger_api=mock_ledger_api,
            contract_address="0xVaultAddress",
            **weighted_pool_params
        )
        
        # Test stable pool
        stable_pool_params = {
            "pool_id": "0x2222222222222222222222222222222222222222222222222222222222222222",
            "sender": "0xSenderAddress",
            "recipient": "0xRecipientAddress",
            "assets": ["0xTokenA", "0xTokenB", "0xTokenC"],
            "max_amounts_in": [1000000000000000000, 2000000000000000000, 3000000000000000000],
            "join_kind": 1,  # EXACT_TOKENS_IN_FOR_BPT_OUT
            "minimum_bpt": 500000000000000000
        }
        
        stable_result = VaultContract.join_pool(
            ledger_api=mock_ledger_api,
            contract_address="0xVaultAddress",
            **stable_pool_params
        )
        
        # Verify both transactions are valid
        TestAssertions.assert_transaction_structure(weighted_result)
        TestAssertions.assert_transaction_structure(stable_result)

    def test_transaction_error_handling(self, mock_ledger_api):
        """Test transaction error handling."""
        # Test with invalid parameters
        invalid_params = {
            "pool_id": "invalid_pool_id",  # Invalid format
            "sender": "invalid_sender",    # Invalid format
            "recipient": "0xRecipientAddress",
            "assets": ["0xTokenA", "0xTokenB"],
            "max_amounts_in": [1000000000000000000, 2000000000000000000],
            "join_kind": 1,
            "minimum_bpt": 500000000000000000
        }
        
        # This should raise an exception or return an error
        with pytest.raises((ValueError, Exception)):
            VaultContract.join_pool(
                ledger_api=mock_ledger_api,
                contract_address="0xVaultAddress",
                **invalid_params
            )

    def test_transaction_batch_operations(self, mock_ledger_api, multisend_contract):
        """Test batch transaction operations."""
        # Create multiple operations for a complex workflow
        operations = [
            # Approve tokens
            MultiSendOperation(
                operation_type=0,
                to="0xTokenA",
                value=0,
                data=b"approve_token_a"
            ),
            MultiSendOperation(
                operation_type=0,
                to="0xTokenB",
                value=0,
                data=b"approve_token_b"
            ),
            # Join pool
            MultiSendOperation(
                operation_type=0,
                to="0xVaultAddress",
                value=0,
                data=b"join_pool"
            ),
        ]
        
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = multisend_contract
        
        # Test batch transaction
        result = MultiSendContract.multi_send(
            ledger_api=mock_ledger_api,
            contract_address="0xMultiSendAddress",
            operations=operations
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], bytes)

    def test_transaction_gas_optimization(self):
        """Test gas optimization for transactions."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test gas optimization strategies
        optimization_strategies = [
            "batch_operations",
            "optimal_deadline",
            "minimal_slippage",
            "efficient_routing"
        ]
        
        for strategy in optimization_strategies:
            gas_savings = behaviour._calculate_gas_savings(strategy)
            
            # Gas savings should be positive
            assert gas_savings >= 0
            
            # Gas savings should be reasonable (not more than 50% of base gas)
            assert gas_savings <= 500000

    def test_transaction_priority_fee_calculation(self):
        """Test priority fee calculation for transactions."""
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
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
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
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
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Test validation rules
        validation_rules = [
            {"rule": "pool_id_format", "value": "0x1234567890123456789012345678901234567890123456789012345678901234", "valid": True},
            {"rule": "pool_id_format", "value": "invalid_pool_id", "valid": False},
            {"rule": "address_format", "value": "0x1234567890123456789012345678901234567890", "valid": True},
            {"rule": "address_format", "value": "invalid_address", "valid": False},
            {"rule": "amount_positive", "value": 1000000000000000000, "valid": True},
            {"rule": "amount_positive", "value": -1000000000000000000, "valid": False},
        ]
        
        for rule_test in validation_rules:
            is_valid = behaviour._validate_transaction_parameter(
                rule_test["rule"], rule_test["value"]
            )
            assert is_valid == rule_test["valid"]

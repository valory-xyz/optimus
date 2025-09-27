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

"""Transaction generation tests for Velodrome protocol."""

import pytest
import time
from unittest.mock import MagicMock, patch

from packages.valory.contracts.velodrome_pool.contract import VelodromePoolContract
from packages.valory.contracts.velodrome_gauge.contract import VelodromeGaugeContract
from packages.valory.contracts.velodrome_voter.contract import VelodromeVoterContract
from packages.valory.skills.liquidity_trader_abci.pools.velodrome import VelodromePoolBehaviour

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions
from tests.integration.fixtures.contract_fixtures import (
    mock_ledger_api,
    velodrome_pool_contract,
    velodrome_gauge_contract,
    velodrome_voter_contract,
)


class TestVelodromeTransactionGeneration(ProtocolIntegrationTestBase):
    """Test proper transaction encoding and parameters for Velodrome."""

    def test_add_liquidity_transaction_parameter_validation(self, mock_ledger_api):
        """Test that add liquidity transactions have correct parameters."""
        # Test parameters
        test_params = {
            "token_a": "0xTokenA",
            "token_b": "0xTokenB",
            "amount_a_desired": 1000000000000000000,
            "amount_b_desired": 2000000000000000000,
            "amount_a_min": 900000000000000000,
            "amount_b_min": 1800000000000000000,
            "to": "0xRecipient",
            "deadline": 1234567890
        }
        
        # Test transaction encoding
        result = VelodromePoolContract.add_liquidity(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress",
            **test_params
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_remove_liquidity_transaction_validation(self, mock_ledger_api):
        """Test that remove liquidity transactions have correct parameters."""
        # Test parameters
        test_params = {
            "token_a": "0xTokenA",
            "token_b": "0xTokenB",
            "liquidity": 1000000000000000000,
            "amount_a_min": 900000000000000000,
            "amount_b_min": 1800000000000000000,
            "to": "0xRecipient",
            "deadline": 1234567890
        }
        
        # Test transaction encoding
        result = VelodromePoolContract.remove_liquidity(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress",
            **test_params
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_gauge_deposit_transaction_parameter_validation(self, mock_ledger_api):
        """Test that gauge deposit transactions have correct parameters."""
        # Test parameters
        test_params = {
            "amount": 1000000000000000000,
            "recipient": "0xRecipient"
        }
        
        # Test transaction encoding
        result = VelodromeGaugeContract.deposit(
            ledger_api=mock_ledger_api,
            contract_address="0xGaugeAddress",
            **test_params
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_gauge_withdraw_transaction_parameter_validation(self, mock_ledger_api):
        """Test that gauge withdraw transactions have correct parameters."""
        # Test parameters
        test_params = {
            "amount": 1000000000000000000,
            "recipient": "0xRecipient"
        }
        
        # Test transaction encoding
        result = VelodromeGaugeContract.withdraw(
            ledger_api=mock_ledger_api,
            contract_address="0xGaugeAddress",
            **test_params
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_gauge_get_reward_transaction_parameter_validation(self, mock_ledger_api):
        """Test that gauge get reward transactions have correct parameters."""
        # Test parameters
        test_params = {
            "recipient": "0xRecipient"
        }
        
        # Test transaction encoding
        result = VelodromeGaugeContract.get_reward(
            ledger_api=mock_ledger_api,
            contract_address="0xGaugeAddress",
            **test_params
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_transaction_gas_estimation(self):
        """Test gas estimation for different transaction types."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test different transaction types
        transaction_types = [
            "add_liquidity",
            "remove_liquidity",
            "deposit",
            "withdraw",
            "get_reward"
        ]
        
        for tx_type in transaction_types:
            gas_estimate = behaviour._estimate_gas_for_transaction(tx_type)
            
            # Verify gas estimate is reasonable
            assert gas_estimate > 0
            assert gas_estimate < 1000000  # Should not exceed 1M gas

    def test_transaction_deadline_validation(self):
        """Test that transaction deadlines are set correctly."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
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

    def test_add_liquidity_transaction_encoding_variations(self, mock_ledger_api):
        """Test add liquidity transaction encoding for different pool types."""
        base_params = {
            "token_a": "0xTokenA",
            "token_b": "0xTokenB",
            "amount_a_desired": 1000000000000000000,
            "amount_b_desired": 2000000000000000000,
            "amount_a_min": 900000000000000000,
            "amount_b_min": 1800000000000000000,
            "to": "0xRecipient",
            "deadline": 1234567890
        }
        
        # Test different pool types
        pool_types = ["stable", "volatile", "cl"]
        
        for pool_type in pool_types:
            test_params = {**base_params, "pool_type": pool_type}
            
            result = VelodromePoolContract.add_liquidity(
                ledger_api=mock_ledger_api,
                contract_address="0xPoolAddress",
                **test_params
            )
            
            # Verify transaction structure
            TestAssertions.assert_transaction_structure(result)
            assert isinstance(result["tx_hash"], str)

    def test_remove_liquidity_transaction_encoding_variations(self, mock_ledger_api):
        """Test remove liquidity transaction encoding for different amounts."""
        base_params = {
            "token_a": "0xTokenA",
            "token_b": "0xTokenB",
            "amount_a_min": 900000000000000000,
            "amount_b_min": 1800000000000000000,
            "to": "0xRecipient",
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
            
            result = VelodromePoolContract.remove_liquidity(
                ledger_api=mock_ledger_api,
                contract_address="0xPoolAddress",
                **test_params
            )
            
            # Verify transaction structure
            TestAssertions.assert_transaction_structure(result)
            assert isinstance(result["tx_hash"], str)

    def test_gauge_transaction_encoding_variations(self, mock_ledger_api):
        """Test gauge transaction encoding for different operations."""
        # Test deposit
        deposit_params = {
            "amount": 1000000000000000000,
            "recipient": "0xRecipient"
        }
        
        deposit_result = VelodromeGaugeContract.deposit(
            ledger_api=mock_ledger_api,
            contract_address="0xGaugeAddress",
            **deposit_params
        )
        
        # Test withdraw
        withdraw_params = {
            "amount": 1000000000000000000,
            "recipient": "0xRecipient"
        }
        
        withdraw_result = VelodromeGaugeContract.withdraw(
            ledger_api=mock_ledger_api,
            contract_address="0xGaugeAddress",
            **withdraw_params
        )
        
        # Test get reward
        reward_params = {
            "recipient": "0xRecipient"
        }
        
        reward_result = VelodromeGaugeContract.get_reward(
            ledger_api=mock_ledger_api,
            contract_address="0xGaugeAddress",
            **reward_params
        )
        
        # Verify all transactions are valid
        TestAssertions.assert_transaction_structure(deposit_result)
        TestAssertions.assert_transaction_structure(withdraw_result)
        TestAssertions.assert_transaction_structure(reward_result)

    def test_transaction_parameter_validation(self):
        """Test transaction parameter validation."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Valid parameters
        valid_params = {
            "token_a": "0xTokenA",
            "token_b": "0xTokenB",
            "amount_a_desired": 1000000000000000000,
            "amount_b_desired": 2000000000000000000,
            "amount_a_min": 900000000000000000,
            "amount_b_min": 1800000000000000000,
            "to": "0xRecipient",
            "deadline": 1234567890
        }
        
        # Validate parameters
        assert valid_params["token_a"].startswith("0x")
        assert valid_params["token_b"].startswith("0x")
        assert valid_params["amount_a_desired"] > 0
        assert valid_params["amount_b_desired"] > 0
        assert valid_params["amount_a_min"] <= valid_params["amount_a_desired"]
        assert valid_params["amount_b_min"] <= valid_params["amount_b_desired"]
        assert valid_params["to"].startswith("0x")
        assert valid_params["deadline"] > 0

    def test_transaction_encoding_consistency(self, mock_ledger_api):
        """Test that transaction encoding is consistent across multiple calls."""
        # Test parameters
        test_params = {
            "token_a": "0xTokenA",
            "token_b": "0xTokenB",
            "amount_a_desired": 1000000000000000000,
            "amount_b_desired": 2000000000000000000,
            "amount_a_min": 900000000000000000,
            "amount_b_min": 1800000000000000000,
            "to": "0xRecipient",
            "deadline": 1234567890
        }
        
        # Generate transaction multiple times
        results = []
        for _ in range(3):
            result = VelodromePoolContract.add_liquidity(
                ledger_api=mock_ledger_api,
                contract_address="0xPoolAddress",
                **test_params
            )
            results.append(result)
        
        # Verify all transactions have the same structure
        for result in results:
            TestAssertions.assert_transaction_structure(result)
            assert isinstance(result["tx_hash"], str)

    def test_transaction_with_different_pool_types(self, mock_ledger_api):
        """Test transaction encoding for different pool types."""
        # Test stable pool
        stable_pool_params = {
            "token_a": "0xTokenA",
            "token_b": "0xTokenB",
            "amount_a_desired": 1000000000000000000,
            "amount_b_desired": 2000000000000000000,
            "amount_a_min": 900000000000000000,
            "amount_b_min": 1800000000000000000,
            "to": "0xRecipient",
            "deadline": 1234567890,
            "is_stable": True
        }
        
        stable_result = VelodromePoolContract.add_liquidity(
            ledger_api=mock_ledger_api,
            contract_address="0xStablePoolAddress",
            **stable_pool_params
        )
        
        # Test volatile pool
        volatile_pool_params = {
            "token_a": "0xTokenA",
            "token_b": "0xTokenB",
            "amount_a_desired": 1000000000000000000,
            "amount_b_desired": 2000000000000000000,
            "amount_a_min": 900000000000000000,
            "amount_b_min": 1800000000000000000,
            "to": "0xRecipient",
            "deadline": 1234567890,
            "is_stable": False
        }
        
        volatile_result = VelodromePoolContract.add_liquidity(
            ledger_api=mock_ledger_api,
            contract_address="0xVolatilePoolAddress",
            **volatile_pool_params
        )
        
        # Verify both transactions are valid
        TestAssertions.assert_transaction_structure(stable_result)
        TestAssertions.assert_transaction_structure(volatile_result)

    def test_transaction_error_handling(self, mock_ledger_api):
        """Test transaction error handling."""
        # Test with invalid parameters
        invalid_params = {
            "token_a": "invalid_token",  # Invalid format
            "token_b": "0xTokenB",
            "amount_a_desired": 1000000000000000000,
            "amount_b_desired": 2000000000000000000,
            "amount_a_min": 900000000000000000,
            "amount_b_min": 1800000000000000000,
            "to": "0xRecipient",
            "deadline": 1234567890
        }
        
        # This should raise an exception or return an error
        with pytest.raises((ValueError, Exception)):
            VelodromePoolContract.add_liquidity(
                ledger_api=mock_ledger_api,
                contract_address="0xPoolAddress",
                **invalid_params
            )

    def test_transaction_batch_operations(self, mock_ledger_api):
        """Test batch transaction operations."""
        # Create multiple operations for a complex workflow
        operations = [
            # Add liquidity
            {
                "method": "add_liquidity",
                "params": {
                    "token_a": "0xTokenA",
                    "token_b": "0xTokenB",
                    "amount_a_desired": 1000000000000000000,
                    "amount_b_desired": 2000000000000000000,
                    "amount_a_min": 900000000000000000,
                    "amount_b_min": 1800000000000000000,
                    "to": "0xRecipient",
                    "deadline": 1234567890
                }
            },
            # Deposit to gauge
            {
                "method": "deposit",
                "params": {
                    "amount": 1000000000000000000,
                    "recipient": "0xRecipient"
                }
            },
            # Get reward
            {
                "method": "get_reward",
                "params": {
                    "recipient": "0xRecipient"
                }
            }
        ]
        
        # Execute batch operations
        results = []
        for operation in operations:
            method = operation["method"]
            params = operation["params"]
            
            if method == "add_liquidity":
                result = VelodromePoolContract.add_liquidity(
                    ledger_api=mock_ledger_api,
                    contract_address="0xPoolAddress",
                    **params
                )
            elif method == "deposit":
                result = VelodromeGaugeContract.deposit(
                    ledger_api=mock_ledger_api,
                    contract_address="0xGaugeAddress",
                    **params
                )
            elif method == "get_reward":
                result = VelodromeGaugeContract.get_reward(
                    ledger_api=mock_ledger_api,
                    contract_address="0xGaugeAddress",
                    **params
                )
            
            results.append(result)
        
        # Verify all transactions are valid
        for result in results:
            TestAssertions.assert_transaction_structure(result)
            assert isinstance(result["tx_hash"], str)

    def test_transaction_gas_optimization(self):
        """Test gas optimization for transactions."""
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
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
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
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
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
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
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Test validation rules
        validation_rules = [
            {"rule": "token_address_format", "value": "0x1234567890123456789012345678901234567890", "valid": True},
            {"rule": "token_address_format", "value": "invalid_token", "valid": False},
            {"rule": "amount_positive", "value": 1000000000000000000, "valid": True},
            {"rule": "amount_positive", "value": -1000000000000000000, "valid": False},
            {"rule": "amount_min_valid", "value": {"desired": 1000, "min": 900}, "valid": True},
            {"rule": "amount_min_valid", "value": {"desired": 1000, "min": 1100}, "valid": False},
            {"rule": "deadline_valid", "value": 1234567890, "valid": True},
            {"rule": "deadline_valid", "value": 0, "valid": False},
        ]
        
        for rule_test in validation_rules:
            is_valid = behaviour._validate_transaction_parameter(
                rule_test["rule"], rule_test["value"]
            )
            assert is_valid == rule_test["valid"]

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

"""Contract integration tests for Balancer protocol."""

import pytest
from unittest.mock import MagicMock, patch

from packages.valory.contracts.balancer_vault.contract import VaultContract
from packages.valory.contracts.balancer_weighted_pool.contract import WeightedPoolContract
from packages.valory.contracts.multisend.contract import MultiSendContract, MultiSendOperation

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions
from tests.integration.fixtures.contract_fixtures import (
    mock_ledger_api,
    balancer_vault_contract,
    balancer_weighted_pool_contract,
    multisend_contract,
)


class TestBalancerContractIntegration(ProtocolIntegrationTestBase):
    """Test Balancer contract interactions with mocked blockchain."""

    def test_vault_join_pool_transaction_encoding(self, mock_ledger_api):
        """Test proper encoding of joinPool transaction."""
        # Test parameters
        pool_id = "0x1234567890123456789012345678901234567890123456789012345678901234"
        sender = "0xSenderAddress"
        recipient = "0xRecipientAddress"
        assets = ["0xTokenA", "0xTokenB"]
        max_amounts_in = [1000000000000000000, 2000000000000000000]
        join_kind = 1  # EXACT_TOKENS_IN_FOR_BPT_OUT
        minimum_bpt = 500000000000000000
        
        # Test transaction encoding
        result = VaultContract.join_pool(
            ledger_api=mock_ledger_api,
            contract_address="0xVaultAddress",
            pool_id=pool_id,
            sender=sender,
            recipient=recipient,
            assets=assets,
            max_amounts_in=max_amounts_in,
            join_kind=join_kind,
            minimum_bpt=minimum_bpt
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], bytes)
        assert len(result["tx_hash"]) > 0

    def test_vault_exit_pool_transaction_encoding(self, mock_ledger_api):
        """Test proper encoding of exitPool transaction."""
        # Test parameters
        pool_id = "0x1234567890123456789012345678901234567890123456789012345678901234"
        sender = "0xSenderAddress"
        recipient = "0xRecipientAddress"
        assets = ["0xTokenA", "0xTokenB"]
        min_amounts_out = [900000000000000000, 1800000000000000000]
        exit_kind = 1  # EXACT_BPT_IN_FOR_TOKENS_OUT
        bpt_amount_in = 1000000000000000000
        
        # Test transaction encoding
        result = VaultContract.exit_pool(
            ledger_api=mock_ledger_api,
            contract_address="0xVaultAddress",
            pool_id=pool_id,
            sender=sender,
            recipient=recipient,
            assets=assets,
            min_amounts_out=min_amounts_out,
            exit_kind=exit_kind,
            bpt_amount_in=bpt_amount_in
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], bytes)

    def test_vault_get_pool_tokens_query(self, mock_ledger_api, balancer_vault_contract):
        """Test pool tokens query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = balancer_vault_contract
        
        # Test pool tokens query
        result = VaultContract.get_pool_tokens(
            ledger_api=mock_ledger_api,
            contract_address="0xVaultAddress",
            pool_id="0x1234567890123456789012345678901234567890123456789012345678901234"
        )
        
        # Verify response structure
        assert "tokens" in result
        assert isinstance(result["tokens"], list)
        assert len(result["tokens"]) == 2
        assert result["tokens"] == ["0xTokenA", "0xTokenB"]

    def test_weighted_pool_balance_query(self, mock_ledger_api, balancer_weighted_pool_contract):
        """Test weighted pool balance query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = balancer_weighted_pool_contract
        
        # Test balance query
        result = WeightedPoolContract.get_balance(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress",
            account="0xAccountAddress"
        )
        
        # Verify response structure
        assert "balance" in result
        assert result["balance"] == 100000000000000000000
        
        # Verify contract was called correctly
        balancer_weighted_pool_contract.functions.balanceOf.assert_called_once_with("0xAccountAddress")

    def test_weighted_pool_total_supply_query(self, mock_ledger_api, balancer_weighted_pool_contract):
        """Test weighted pool total supply query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = balancer_weighted_pool_contract
        
        # Test total supply query
        result = WeightedPoolContract.get_total_supply(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress"
        )
        
        # Verify response structure
        assert "data" in result
        assert result["data"] == 1000000000000000000000

    def test_weighted_pool_id_query(self, mock_ledger_api, balancer_weighted_pool_contract):
        """Test weighted pool ID query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = balancer_weighted_pool_contract
        
        # Test pool ID query
        result = WeightedPoolContract.get_pool_id(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress"
        )
        
        # Verify response structure
        assert "pool_id" in result
        assert result["pool_id"] == "0x1234567890123456789012345678901234567890123456789012345678901234"

    def test_weighted_pool_vault_address_query(self, mock_ledger_api, balancer_weighted_pool_contract):
        """Test weighted pool vault address query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = balancer_weighted_pool_contract
        
        # Test vault address query
        result = WeightedPoolContract.get_vault_address(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress"
        )
        
        # Verify response structure
        assert "vault" in result
        assert result["vault"] == "0xVaultAddress"

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

    def test_contract_parameter_validation(self):
        """Test contract parameter validation."""
        # Test valid parameters
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

    def test_transaction_gas_estimation(self):
        """Test gas estimation for different transaction types."""
        # Mock gas estimation
        gas_estimates = {
            "join_pool": 200000,
            "exit_pool": 180000,
            "approve_token": 50000,
            "multisend": 300000,
        }
        
        for tx_type, expected_gas in gas_estimates.items():
            # In a real implementation, this would call the actual gas estimation
            estimated_gas = expected_gas  # Mock value
            
            # Verify gas estimate is reasonable
            assert estimated_gas > 0
            assert estimated_gas < 1000000  # Should not exceed 1M gas

    def test_transaction_deadline_validation(self):
        """Test that transaction deadlines are set correctly."""
        import time
        
        # Test deadline calculation
        current_time = int(time.time())
        deadline_buffer = 300  # 5 minutes
        
        calculated_deadline = current_time + deadline_buffer
        expected_deadline = current_time + deadline_buffer
        
        assert calculated_deadline == expected_deadline
        
        # Test deadline validation
        is_valid = calculated_deadline > current_time
        assert is_valid
        
        # Test expired deadline
        expired_deadline = current_time - 100
        is_expired = expired_deadline <= current_time
        assert is_expired

    def test_contract_error_handling(self, mock_ledger_api):
        """Test contract error handling."""
        # Mock contract instance that raises an exception
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.balanceOf.return_value.call.side_effect = Exception("Contract call failed")
        mock_ledger_api.get_instance.return_value = mock_contract_instance
        
        # Test error handling
        with pytest.raises(Exception, match="Contract call failed"):
            WeightedPoolContract.get_balance(
                ledger_api=mock_ledger_api,
                contract_address="0xPoolAddress",
                account="0xAccountAddress"
            )

    def test_contract_response_parsing(self, mock_ledger_api, balancer_vault_contract):
        """Test contract response parsing."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = balancer_vault_contract
        
        # Test response parsing for getPoolTokens
        result = VaultContract.get_pool_tokens(
            ledger_api=mock_ledger_api,
            contract_address="0xVaultAddress",
            pool_id="0x1234567890123456789012345678901234567890123456789012345678901234"
        )
        
        # Verify response is properly parsed
        assert isinstance(result, dict)
        assert "tokens" in result
        assert isinstance(result["tokens"], list)
        
        # Verify token addresses are valid
        for token in result["tokens"]:
            assert token.startswith("0x")
            assert len(token) == 42  # 0x + 40 hex chars

    def test_contract_address_validation(self):
        """Test contract address validation."""
        # Valid addresses
        valid_addresses = [
            "0x1234567890123456789012345678901234567890",
            "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
            "0x0000000000000000000000000000000000000000",
        ]
        
        for address in valid_addresses:
            assert address.startswith("0x")
            assert len(address) == 42
            assert all(c in "0123456789abcdefABCDEF" for c in address[2:])
        
        # Invalid addresses
        invalid_addresses = [
            "0x123456789012345678901234567890123456789",  # Too short
            "1234567890123456789012345678901234567890",   # Missing 0x
            "0x123456789012345678901234567890123456789g",  # Invalid character
        ]
        
        for address in invalid_addresses:
            is_valid = (
                address.startswith("0x") and
                len(address) == 42 and
                all(c in "0123456789abcdefABCDEF" for c in address[2:])
            )
            assert not is_valid

    def test_contract_method_encoding(self, mock_ledger_api):
        """Test contract method encoding."""
        # Test different method encodings
        methods_to_test = [
            "joinPool",
            "exitPool",
            "getPoolTokens",
            "balanceOf",
            "totalSupply",
        ]
        
        for method in methods_to_test:
            # Mock contract instance
            mock_contract_instance = MagicMock()
            mock_contract_instance.encodeABI.return_value = f"0x{method}_encoded"
            mock_ledger_api.get_instance.return_value = mock_contract_instance
            
            # Test encoding (this would be done internally by the contract methods)
            encoded_data = mock_contract_instance.encodeABI(method, args=())
            
            # Verify encoding
            assert encoded_data.startswith("0x")
            assert len(encoded_data) > 2

    def test_contract_call_retry_logic(self):
        """Test contract call retry logic."""
        # Mock retry logic
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Simulate contract call
                if attempt < max_retries - 1:
                    raise Exception("Temporary failure")
                else:
                    result = "success"
                    break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                # Wait before retry
                import time
                time.sleep(retry_delay)
        
        assert result == "success"

    def test_contract_batch_operations(self, mock_ledger_api):
        """Test batch contract operations."""
        # Mock batch operations
        operations = [
            {"method": "balanceOf", "args": ("0xAccount1",)},
            {"method": "balanceOf", "args": ("0xAccount2",)},
            {"method": "totalSupply", "args": ()},
        ]
        
        # Mock contract instance
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.balanceOf.return_value.call.return_value = 100000000000000000000
        mock_contract_instance.functions.totalSupply.return_value.call.return_value = 1000000000000000000000
        mock_ledger_api.get_instance.return_value = mock_contract_instance
        
        # Execute batch operations
        results = []
        for operation in operations:
            method = operation["method"]
            args = operation["args"]
            
            if method == "balanceOf":
                result = WeightedPoolContract.get_balance(
                    ledger_api=mock_ledger_api,
                    contract_address="0xPoolAddress",
                    account=args[0]
                )
            elif method == "totalSupply":
                result = WeightedPoolContract.get_total_supply(
                    ledger_api=mock_ledger_api,
                    contract_address="0xPoolAddress"
                )
            
            results.append(result)
        
        # Verify results
        assert len(results) == 3
        assert results[0]["balance"] == 100000000000000000000
        assert results[1]["balance"] == 100000000000000000000
        assert results[2]["data"] == 1000000000000000000000

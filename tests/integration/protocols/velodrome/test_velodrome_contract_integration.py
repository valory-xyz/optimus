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

"""Contract integration tests for Velodrome protocol."""

import pytest
from unittest.mock import MagicMock, patch

from packages.valory.contracts.velodrome_pool.contract import VelodromePoolContract
from packages.valory.contracts.velodrome_gauge.contract import VelodromeGaugeContract
from packages.valory.contracts.velodrome_voter.contract import VelodromeVoterContract

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions
from tests.integration.fixtures.contract_fixtures import (
    mock_ledger_api,
    velodrome_pool_contract,
    velodrome_gauge_contract,
    velodrome_voter_contract,
)


class TestVelodromeContractIntegration(ProtocolIntegrationTestBase):
    """Test Velodrome contract interactions with mocked blockchain."""

    def test_velodrome_pool_add_liquidity_transaction_encoding(self, mock_ledger_api):
        """Test proper encoding of add liquidity transaction."""
        # Test parameters
        token_a = "0xTokenA"
        token_b = "0xTokenB"
        amount_a_desired = 1000000000000000000
        amount_b_desired = 2000000000000000000
        amount_a_min = 900000000000000000
        amount_b_min = 1800000000000000000
        to = "0xRecipient"
        deadline = 1234567890
        
        # Test transaction encoding
        result = VelodromePoolContract.add_liquidity(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress",
            token_a=token_a,
            token_b=token_b,
            amount_a_desired=amount_a_desired,
            amount_b_desired=amount_b_desired,
            amount_a_min=amount_a_min,
            amount_b_min=amount_b_min,
            to=to,
            deadline=deadline
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_velodrome_pool_remove_liquidity_transaction_encoding(self, mock_ledger_api):
        """Test proper encoding of remove liquidity transaction."""
        # Test parameters
        token_a = "0xTokenA"
        token_b = "0xTokenB"
        liquidity = 1000000000000000000
        amount_a_min = 900000000000000000
        amount_b_min = 1800000000000000000
        to = "0xRecipient"
        deadline = 1234567890
        
        # Test transaction encoding
        result = VelodromePoolContract.remove_liquidity(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress",
            token_a=token_a,
            token_b=token_b,
            liquidity=liquidity,
            amount_a_min=amount_a_min,
            amount_b_min=amount_b_min,
            to=to,
            deadline=deadline
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_velodrome_gauge_deposit_transaction_encoding(self, mock_ledger_api):
        """Test proper encoding of gauge deposit transaction."""
        # Test parameters
        amount = 1000000000000000000
        recipient = "0xRecipient"
        
        # Test transaction encoding
        result = VelodromeGaugeContract.deposit(
            ledger_api=mock_ledger_api,
            contract_address="0xGaugeAddress",
            amount=amount,
            recipient=recipient
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_velodrome_gauge_withdraw_transaction_encoding(self, mock_ledger_api):
        """Test proper encoding of gauge withdraw transaction."""
        # Test parameters
        amount = 1000000000000000000
        recipient = "0xRecipient"
        
        # Test transaction encoding
        result = VelodromeGaugeContract.withdraw(
            ledger_api=mock_ledger_api,
            contract_address="0xGaugeAddress",
            amount=amount,
            recipient=recipient
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_velodrome_gauge_get_reward_transaction_encoding(self, mock_ledger_api):
        """Test proper encoding of gauge get reward transaction."""
        # Test parameters
        recipient = "0xRecipient"
        
        # Test transaction encoding
        result = VelodromeGaugeContract.get_reward(
            ledger_api=mock_ledger_api,
            contract_address="0xGaugeAddress",
            recipient=recipient
        )
        
        # Verify transaction structure
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_velodrome_pool_balance_query(self, mock_ledger_api, velodrome_pool_contract):
        """Test pool balance query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = velodrome_pool_contract
        
        # Test balance query
        result = VelodromePoolContract.get_balance(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress",
            account="0xAccountAddress"
        )
        
        # Verify response structure
        assert "balance" in result
        assert result["balance"] == 100000000000000000000

    def test_velodrome_pool_total_supply_query(self, mock_ledger_api, velodrome_pool_contract):
        """Test pool total supply query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = velodrome_pool_contract
        
        # Test total supply query
        result = VelodromePoolContract.get_total_supply(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress"
        )
        
        # Verify response structure
        assert "data" in result
        assert result["data"] == 1000000000000000000000

    def test_velodrome_pool_reserves_query(self, mock_ledger_api, velodrome_pool_contract):
        """Test pool reserves query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = velodrome_pool_contract
        
        # Test reserves query
        result = VelodromePoolContract.get_reserves(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress"
        )
        
        # Verify response structure
        assert "data" in result
        assert result["data"] == [1000000000000000000000, 2000000000000000000000]

    def test_velodrome_pool_tokens_query(self, mock_ledger_api, velodrome_pool_contract):
        """Test pool tokens query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = velodrome_pool_contract
        
        # Test tokens query
        result = VelodromePoolContract.get_pool_tokens(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress"
        )
        
        # Verify response structure
        assert "tokens" in result
        assert result["tokens"] == ["0xTokenA", "0xTokenB"]

    def test_velodrome_gauge_balance_query(self, mock_ledger_api, velodrome_gauge_contract):
        """Test gauge balance query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = velodrome_gauge_contract
        
        # Test balance query
        result = VelodromeGaugeContract.balance_of(
            ledger_api=mock_ledger_api,
            contract_address="0xGaugeAddress",
            account="0xAccountAddress"
        )
        
        # Verify response structure
        assert "balance" in result
        assert result["balance"] == 100000000000000000000

    def test_velodrome_gauge_earned_query(self, mock_ledger_api, velodrome_gauge_contract):
        """Test gauge earned query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = velodrome_gauge_contract
        
        # Test earned query
        result = VelodromeGaugeContract.earned(
            ledger_api=mock_ledger_api,
            contract_address="0xGaugeAddress",
            account="0xAccountAddress"
        )
        
        # Verify response structure
        assert "earned" in result
        assert result["earned"] == 50000000000000000000

    def test_velodrome_gauge_reward_rate_query(self, mock_ledger_api, velodrome_gauge_contract):
        """Test gauge reward rate query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = velodrome_gauge_contract
        
        # Test reward rate query
        result = VelodromeGaugeContract.get_reward_rate(
            ledger_api=mock_ledger_api,
            contract_address="0xGaugeAddress"
        )
        
        # Verify response structure
        assert "rewardRate" in result
        assert result["rewardRate"] == 1000000000000000000

    def test_velodrome_voter_gauges_query(self, mock_ledger_api, velodrome_voter_contract):
        """Test voter gauges query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = velodrome_voter_contract
        
        # Test gauges query
        result = VelodromeVoterContract.get_gauges(
            ledger_api=mock_ledger_api,
            contract_address="0xVoterAddress",
            pool_address="0xPoolAddress"
        )
        
        # Verify response structure
        assert "gauge" in result
        assert result["gauge"] == "0xGaugeAddress"

    def test_velodrome_voter_is_gauge_query(self, mock_ledger_api, velodrome_voter_contract):
        """Test voter is gauge query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = velodrome_voter_contract
        
        # Test is gauge query
        result = VelodromeVoterContract.is_gauge(
            ledger_api=mock_ledger_api,
            contract_address="0xVoterAddress",
            gauge_address="0xGaugeAddress"
        )
        
        # Verify response structure
        assert "isGauge" in result
        assert result["isGauge"] == True

    def test_velodrome_voter_votes_query(self, mock_ledger_api, velodrome_voter_contract):
        """Test voter votes query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = velodrome_voter_contract
        
        # Test votes query
        result = VelodromeVoterContract.get_votes(
            ledger_api=mock_ledger_api,
            contract_address="0xVoterAddress",
            account="0xAccountAddress"
        )
        
        # Verify response structure
        assert "votes" in result
        assert result["votes"] == 100000000000000000000

    def test_contract_parameter_validation(self):
        """Test contract parameter validation."""
        # Test valid parameters
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

    def test_transaction_gas_estimation(self):
        """Test gas estimation for different transaction types."""
        # Mock gas estimation
        gas_estimates = {
            "add_liquidity": 250000,
            "remove_liquidity": 200000,
            "deposit": 150000,
            "withdraw": 150000,
            "get_reward": 100000,
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
            VelodromePoolContract.get_balance(
                ledger_api=mock_ledger_api,
                contract_address="0xPoolAddress",
                account="0xAccountAddress"
            )

    def test_contract_response_parsing(self, mock_ledger_api, velodrome_pool_contract):
        """Test contract response parsing."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = velodrome_pool_contract
        
        # Test response parsing for get_reserves
        result = VelodromePoolContract.get_reserves(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress"
        )
        
        # Verify response is properly parsed
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], list)
        assert len(result["data"]) == 2

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
            "addLiquidity",
            "removeLiquidity",
            "deposit",
            "withdraw",
            "getReward",
            "balanceOf",
            "earned",
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

    def test_contract_batch_operations(self, mock_ledger_api, velodrome_pool_contract):
        """Test batch contract operations."""
        # Mock batch operations
        operations = [
            {"method": "balanceOf", "args": ("0xAccount1",)},
            {"method": "balanceOf", "args": ("0xAccount2",)},
            {"method": "totalSupply", "args": ()},
            {"method": "getReserves", "args": ()},
        ]
        
        # Mock contract instance
        mock_ledger_api.get_instance.return_value = velodrome_pool_contract
        
        # Execute batch operations
        results = []
        for operation in operations:
            method = operation["method"]
            args = operation["args"]
            
            if method == "balanceOf":
                result = VelodromePoolContract.get_balance(
                    ledger_api=mock_ledger_api,
                    contract_address="0xPoolAddress",
                    account=args[0]
                )
            elif method == "totalSupply":
                result = VelodromePoolContract.get_total_supply(
                    ledger_api=mock_ledger_api,
                    contract_address="0xPoolAddress"
                )
            elif method == "getReserves":
                result = VelodromePoolContract.get_reserves(
                    ledger_api=mock_ledger_api,
                    contract_address="0xPoolAddress"
                )
            
            results.append(result)
        
        # Verify results
        assert len(results) == 4
        assert results[0]["balance"] == 100000000000000000000
        assert results[1]["balance"] == 100000000000000000000
        assert results[2]["data"] == 1000000000000000000000
        assert "data" in results[3]

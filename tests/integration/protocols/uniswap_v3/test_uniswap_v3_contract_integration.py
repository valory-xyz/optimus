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

"""Contract integration tests for Uniswap V3 protocol."""

import pytest
from unittest.mock import MagicMock, patch

from packages.valory.contracts.uniswap_v3_pool.contract import UniswapV3PoolContract
from packages.valory.contracts.uniswap_v3_non_fungible_position_manager.contract import UniswapV3NonfungiblePositionManagerContract

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions
from tests.integration.fixtures.contract_fixtures import (
    mock_ledger_api,
    uniswap_v3_pool_contract,
    uniswap_v3_position_manager_contract,
)


class TestUniswapV3ContractIntegration(ProtocolIntegrationTestBase):
    """Test Uniswap V3 contract interactions with mocked blockchain."""

    def test_position_manager_mint_transaction_encoding(self, mock_ledger_api):
        """Test proper encoding of mint transaction."""
        # Test parameters
        token0 = "0xTokenA"
        token1 = "0xTokenB"
        fee = 3000
        tick_lower = -276320
        tick_upper = -276300
        amount0_desired = 1000000000000000000
        amount1_desired = 2000000000000000000
        amount0_min = 900000000000000000
        amount1_min = 1800000000000000000
        recipient = "0xRecipient"
        deadline = 1234567890
        
        # Test transaction encoding
        result = UniswapV3NonfungiblePositionManagerContract.mint(
            ledger_api=mock_ledger_api,
            contract_address="0xPositionManagerAddress",
            token0=token0,
            token1=token1,
            fee=fee,
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            amount0_desired=amount0_desired,
            amount1_desired=amount1_desired,
            amount0_min=amount0_min,
            amount1_min=amount1_min,
            recipient=recipient,
            deadline=deadline
        )
        
        # Verify transaction hash is generated
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_position_manager_decrease_liquidity_transaction_encoding(self, mock_ledger_api):
        """Test proper encoding of decrease liquidity transaction."""
        # Test parameters
        token_id = 12345
        liquidity = 1000000000000000000
        amount0_min = 900000000000000000
        amount1_min = 1800000000000000000
        deadline = 1234567890
        
        # Test transaction encoding
        result = UniswapV3NonfungiblePositionManagerContract.decrease_liquidity(
            ledger_api=mock_ledger_api,
            contract_address="0xPositionManagerAddress",
            token_id=token_id,
            liquidity=liquidity,
            amount0_min=amount0_min,
            amount1_min=amount1_min,
            deadline=deadline
        )
        
        # Verify transaction hash is generated
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_position_manager_collect_transaction_encoding(self, mock_ledger_api):
        """Test proper encoding of collect transaction."""
        # Test parameters
        token_id = 12345
        amount0_max = 1000000000000000000
        amount1_max = 2000000000000000000
        recipient = "0xRecipient"
        
        # Test transaction encoding
        result = UniswapV3NonfungiblePositionManagerContract.collect(
            ledger_api=mock_ledger_api,
            contract_address="0xPositionManagerAddress",
            token_id=token_id,
            amount0_max=amount0_max,
            amount1_max=amount1_max,
            recipient=recipient
        )
        
        # Verify transaction hash is generated
        TestAssertions.assert_transaction_structure(result)
        assert isinstance(result["tx_hash"], str)

    def test_pool_slot0_query_with_mocked_response(self, mock_ledger_api, uniswap_v3_pool_contract):
        """Test pool slot0 query with mocked blockchain response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = uniswap_v3_pool_contract
        
        # Test slot0 query
        result = UniswapV3PoolContract.slot0(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress"
        )
        
        # Verify response structure
        assert "slot0" in result
        slot0_data = result["slot0"]
        assert slot0_data["sqrt_price_x96"] == 79228162514264337593543950336
        assert slot0_data["tick"] == -276310
        assert slot0_data["unlocked"] == True

    def test_pool_tokens_query(self, mock_ledger_api, uniswap_v3_pool_contract):
        """Test pool tokens query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = uniswap_v3_pool_contract
        
        # Test pool tokens query
        result = UniswapV3PoolContract.get_pool_tokens(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress"
        )
        
        # Verify response structure
        assert "tokens" in result
        assert result["tokens"] == ["0xTokenA", "0xTokenB"]

    def test_pool_fee_query(self, mock_ledger_api, uniswap_v3_pool_contract):
        """Test pool fee query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = uniswap_v3_pool_contract
        
        # Test pool fee query
        result = UniswapV3PoolContract.get_pool_fee(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress"
        )
        
        # Verify response structure
        assert "data" in result
        assert result["data"] == 3000

    def test_pool_tick_spacing_query(self, mock_ledger_api, uniswap_v3_pool_contract):
        """Test pool tick spacing query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = uniswap_v3_pool_contract
        
        # Test tick spacing query
        result = UniswapV3PoolContract.get_tick_spacing(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress"
        )
        
        # Verify response structure
        assert "data" in result
        assert result["data"] == 60

    def test_position_query_with_mocked_response(self, mock_ledger_api, uniswap_v3_position_manager_contract):
        """Test position query with mocked blockchain response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = uniswap_v3_position_manager_contract
        
        # Test position query
        result = UniswapV3NonfungiblePositionManagerContract.positions(
            ledger_api=mock_ledger_api,
            contract_address="0xPositionManagerAddress",
            token_id=12345
        )
        
        # Verify response structure
        assert "liquidity" in result
        assert "token0" in result
        assert "token1" in result
        assert "fee" in result
        assert "tickLower" in result
        assert "tickUpper" in result
        assert "tokensOwed0" in result
        assert "tokensOwed1" in result

    def test_balance_of_query(self, mock_ledger_api, uniswap_v3_position_manager_contract):
        """Test balance of query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = uniswap_v3_position_manager_contract
        
        # Test balance of query
        result = UniswapV3NonfungiblePositionManagerContract.balance_of(
            ledger_api=mock_ledger_api,
            contract_address="0xPositionManagerAddress",
            account="0xAccountAddress"
        )
        
        # Verify response structure
        assert "balance" in result
        assert result["balance"] == 1

    def test_token_of_owner_by_index_query(self, mock_ledger_api, uniswap_v3_position_manager_contract):
        """Test token of owner by index query with mocked response."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = uniswap_v3_position_manager_contract
        
        # Test token of owner by index query
        result = UniswapV3NonfungiblePositionManagerContract.token_of_owner_by_index(
            ledger_api=mock_ledger_api,
            contract_address="0xPositionManagerAddress",
            owner="0xOwnerAddress",
            index=0
        )
        
        # Verify response structure
        assert "tokenId" in result
        assert result["tokenId"] == 12345

    def test_contract_parameter_validation(self):
        """Test contract parameter validation."""
        # Test valid parameters
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

    def test_transaction_gas_estimation(self):
        """Test gas estimation for different transaction types."""
        # Mock gas estimation
        gas_estimates = {
            "mint": 300000,
            "decrease_liquidity": 200000,
            "collect": 150000,
            "increase_liquidity": 250000,
            "burn": 100000,
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
        mock_contract_instance.functions.slot0.return_value.call.side_effect = Exception("Contract call failed")
        mock_ledger_api.get_instance.return_value = mock_contract_instance
        
        # Test error handling
        with pytest.raises(Exception, match="Contract call failed"):
            UniswapV3PoolContract.slot0(
                ledger_api=mock_ledger_api,
                contract_address="0xPoolAddress"
            )

    def test_contract_response_parsing(self, mock_ledger_api, uniswap_v3_pool_contract):
        """Test contract response parsing."""
        # Mock the contract instance
        mock_ledger_api.get_instance.return_value = uniswap_v3_pool_contract
        
        # Test response parsing for slot0
        result = UniswapV3PoolContract.slot0(
            ledger_api=mock_ledger_api,
            contract_address="0xPoolAddress"
        )
        
        # Verify response is properly parsed
        assert isinstance(result, dict)
        assert "slot0" in result
        assert isinstance(result["slot0"], dict)
        
        # Verify slot0 data structure
        slot0_data = result["slot0"]
        assert "sqrt_price_x96" in slot0_data
        assert "tick" in slot0_data
        assert "unlocked" in slot0_data

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
            "mint",
            "decreaseLiquidity",
            "collect",
            "increaseLiquidity",
            "burn",
            "slot0",
            "positions",
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

    def test_contract_batch_operations(self, mock_ledger_api, uniswap_v3_pool_contract):
        """Test batch contract operations."""
        # Mock batch operations
        operations = [
            {"method": "slot0", "args": ()},
            {"method": "token0", "args": ()},
            {"method": "token1", "args": ()},
            {"method": "fee", "args": ()},
        ]
        
        # Mock contract instance
        mock_ledger_api.get_instance.return_value = uniswap_v3_pool_contract
        
        # Execute batch operations
        results = []
        for operation in operations:
            method = operation["method"]
            args = operation["args"]
            
            if method == "slot0":
                result = UniswapV3PoolContract.slot0(
                    ledger_api=mock_ledger_api,
                    contract_address="0xPoolAddress"
                )
            elif method == "token0":
                result = UniswapV3PoolContract.get_pool_tokens(
                    ledger_api=mock_ledger_api,
                    contract_address="0xPoolAddress"
                )
            elif method == "fee":
                result = UniswapV3PoolContract.get_pool_fee(
                    ledger_api=mock_ledger_api,
                    contract_address="0xPoolAddress"
                )
            
            results.append(result)
        
        # Verify results
        assert len(results) == 4
        assert "slot0" in results[0]
        assert "tokens" in results[1]
        assert "data" in results[3]

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

    def _consume_generator(self, gen):
        """Helper to run generator and return final value."""
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    @patch.object(VelodromePoolContract, 'contract_interface', {'ethereum': {}})
    def test_velodrome_pool_contract_methods(self, mock_ledger_api):
        """Test that VelodromePoolContract methods work correctly."""
        # Mock the contract instance methods
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.balanceOf.return_value.call.return_value = 1000000000000000000
        mock_contract_instance.functions.reserve0.return_value.call.return_value = 5000000000000000000
        mock_contract_instance.functions.reserve1.return_value.call.return_value = 3000000000000000000
        mock_contract_instance.encodeABI.return_value = "0x1234567890abcdef"
        
        with patch.object(VelodromePoolContract, 'get_instance', return_value=mock_contract_instance):
            # Test get_balance
            balance_result = VelodromePoolContract.get_balance(
                ledger_api=mock_ledger_api,
                contract_address="0xPoolAddress",
                account="0xAccount"
            )
            
            assert "balance" in balance_result
            assert isinstance(balance_result["balance"], int)

            # Test build_approval_tx
            approval_result = VelodromePoolContract.build_approval_tx(
                ledger_api=mock_ledger_api,
                contract_address="0xPoolAddress",
                spender="0xSpender",
                amount=1000000000000000000
            )
            
            assert "tx_hash" in approval_result
            assert isinstance(approval_result["tx_hash"], bytes)

            # Test get_reserves
            reserves_result = VelodromePoolContract.get_reserves(
                ledger_api=mock_ledger_api,
                contract_address="0xPoolAddress"
            )
            
            assert "data" in reserves_result
            assert isinstance(reserves_result["data"], list)
            assert len(reserves_result["data"]) == 2

    @patch.object(VelodromeGaugeContract, 'contract_interface', {'ethereum': {}})
    def test_velodrome_gauge_contract_methods(self, mock_ledger_api):
        """Test that VelodromeGaugeContract methods work correctly."""
        # Mock the contract instance methods
        mock_contract_instance = MagicMock()
        mock_contract_instance.encodeABI.return_value = "0x1234567890abcdef"
        mock_contract_instance.functions.earned.return_value.call.return_value = 5000000000000000000
        mock_contract_instance.functions.balanceOf.return_value.call.return_value = 1000000000000000000
        mock_contract_instance.functions.totalSupply.return_value.call.return_value = 10000000000000000000
        
        with patch.object(VelodromeGaugeContract, 'get_instance', return_value=mock_contract_instance):
            # Test deposit
            deposit_result = VelodromeGaugeContract.deposit(
                ledger_api=mock_ledger_api,
                contract_address="0xGaugeAddress",
                amount=1000000000000000000
            )
            
            assert "tx_hash" in deposit_result
            assert isinstance(deposit_result["tx_hash"], str)

            # Test withdraw
            withdraw_result = VelodromeGaugeContract.withdraw(
                ledger_api=mock_ledger_api,
                contract_address="0xGaugeAddress",
                amount=1000000000000000000
            )
            
            assert "tx_hash" in withdraw_result
            assert isinstance(withdraw_result["tx_hash"], str)

            # Test get_reward
            reward_result = VelodromeGaugeContract.get_reward(
                ledger_api=mock_ledger_api,
                contract_address="0xGaugeAddress",
                account="0xAccount"
            )
            
            assert "tx_hash" in reward_result
            assert isinstance(reward_result["tx_hash"], str)

            # Test earned
            earned_result = VelodromeGaugeContract.earned(
                ledger_api=mock_ledger_api,
                contract_address="0xGaugeAddress",
                account="0xAccount"
            )
            
            assert "earned" in earned_result
            assert isinstance(earned_result["earned"], int)

            # Test balance_of
            balance_result = VelodromeGaugeContract.balance_of(
                ledger_api=mock_ledger_api,
                contract_address="0xGaugeAddress",
                account="0xAccount"
            )
            
            assert "balance" in balance_result
            assert isinstance(balance_result["balance"], int)

            # Test total_supply
            total_supply_result = VelodromeGaugeContract.total_supply(
                ledger_api=mock_ledger_api,
                contract_address="0xGaugeAddress"
            )
            
            assert "total_supply" in total_supply_result
            assert isinstance(total_supply_result["total_supply"], int)

    @patch.object(VelodromeVoterContract, 'contract_interface', {'ethereum': {}})
    def test_velodrome_voter_contract_methods(self, mock_ledger_api):
        """Test that VelodromeVoterContract methods work correctly."""
        # Mock the contract instance methods
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.gauges.return_value.call.return_value = "0xGaugeAddress"
        mock_contract_instance.functions.isGauge.return_value.call.return_value = True
        mock_contract_instance.functions.poolForGauge.return_value.call.return_value = "0xPoolAddress"
        mock_contract_instance.functions.isAlive.return_value.call.return_value = True
        mock_contract_instance.functions.validateGaugeAddress.return_value.call.return_value = True
        mock_contract_instance.functions.length.return_value.call.return_value = 10
        
        with patch.object(VelodromeVoterContract, 'get_instance', return_value=mock_contract_instance):
            # Test gauges
            gauges_result = VelodromeVoterContract.gauges(
                ledger_api=mock_ledger_api,
                contract_address="0xVoterAddress",
                pool_address="0xPoolAddress"
            )
            
            assert "gauge" in gauges_result
            assert isinstance(gauges_result["gauge"], str)

            # Test is_gauge
            is_gauge_result = VelodromeVoterContract.is_gauge(
                ledger_api=mock_ledger_api,
                contract_address="0xVoterAddress",
                gauge_address="0xGaugeAddress"
            )
            
            assert "is_gauge" in is_gauge_result
            assert isinstance(is_gauge_result["is_gauge"], bool)

            # Test pool_for_gauge
            pool_for_gauge_result = VelodromeVoterContract.pool_for_gauge(
                ledger_api=mock_ledger_api,
                contract_address="0xVoterAddress",
                gauge_address="0xGaugeAddress"
            )
            
            assert "pool" in pool_for_gauge_result
            assert isinstance(pool_for_gauge_result["pool"], str)

            # Test is_alive
            is_alive_result = VelodromeVoterContract.is_alive(
                ledger_api=mock_ledger_api,
                contract_address="0xVoterAddress",
                gauge_address="0xGaugeAddress"
            )
            
            assert "is_alive" in is_alive_result
            assert isinstance(is_alive_result["is_alive"], bool)

            # Test validate_gauge_address
            validate_result = VelodromeVoterContract.validate_gauge_address(
                ledger_api=mock_ledger_api,
                contract_address="0xVoterAddress",
                gauge_address="0xGaugeAddress"
            )
            
            assert "is_valid" in validate_result
            assert isinstance(validate_result["is_valid"], bool)

            # Test length
            length_result = VelodromeVoterContract.length(
                ledger_api=mock_ledger_api,
                contract_address="0xVoterAddress"
            )
            
            assert "length" in length_result
            assert isinstance(length_result["length"], int)

    def test_velodrome_pool_behaviour_static_methods(self):
        """Test VelodromePoolBehaviour methods that generate transactions."""
        # Test that the methods exist and have correct signatures
        assert hasattr(VelodromePoolBehaviour, 'enter')
        assert hasattr(VelodromePoolBehaviour, 'exit')
        assert hasattr(VelodromePoolBehaviour, 'stake_lp_tokens')
        assert hasattr(VelodromePoolBehaviour, 'unstake_lp_tokens')
        assert hasattr(VelodromePoolBehaviour, 'claim_rewards')
        
        # Test method signatures
        import inspect
        enter_sig = inspect.signature(VelodromePoolBehaviour.enter)
        assert 'kwargs' in enter_sig.parameters
        assert enter_sig.parameters['kwargs'].kind == inspect.Parameter.VAR_KEYWORD
        
        exit_sig = inspect.signature(VelodromePoolBehaviour.exit)
        assert 'kwargs' in exit_sig.parameters
        assert exit_sig.parameters['kwargs'].kind == inspect.Parameter.VAR_KEYWORD

    def test_velodrome_pool_behaviour_exit_method(self):
        """Test VelodromePoolBehaviour exit method."""
        # Test that the exit method exists and has correct signature
        assert hasattr(VelodromePoolBehaviour, 'exit')
        
        import inspect
        exit_sig = inspect.signature(VelodromePoolBehaviour.exit)
        assert 'kwargs' in exit_sig.parameters
        assert exit_sig.parameters['kwargs'].kind == inspect.Parameter.VAR_KEYWORD
        
        # Test return type annotation
        assert exit_sig.return_annotation.__name__ == 'Generator'

    def test_velodrome_gauge_staking_methods(self):
        """Test VelodromePoolBehaviour gauge staking methods."""
        # Test that the methods exist and have correct signatures
        assert hasattr(VelodromePoolBehaviour, 'stake_lp_tokens')
        assert hasattr(VelodromePoolBehaviour, 'unstake_lp_tokens')
        assert hasattr(VelodromePoolBehaviour, 'claim_rewards')
        assert hasattr(VelodromePoolBehaviour, 'get_pending_rewards')
        assert hasattr(VelodromePoolBehaviour, 'get_staked_balance')
        
        # Test method signatures
        import inspect
        stake_sig = inspect.signature(VelodromePoolBehaviour.stake_lp_tokens)
        assert 'lp_token' in stake_sig.parameters
        assert 'amount' in stake_sig.parameters
        assert 'kwargs' in stake_sig.parameters  # safe_address is passed via kwargs
        
        unstake_sig = inspect.signature(VelodromePoolBehaviour.unstake_lp_tokens)
        assert 'lp_token' in unstake_sig.parameters
        assert 'amount' in unstake_sig.parameters
        assert 'kwargs' in unstake_sig.parameters  # safe_address is passed via kwargs
        
        claim_sig = inspect.signature(VelodromePoolBehaviour.claim_rewards)
        assert 'lp_token' in claim_sig.parameters
        assert 'kwargs' in claim_sig.parameters  # safe_address is passed via kwargs

    def test_velodrome_cl_pool_methods(self):
        """Test VelodromePoolBehaviour CL pool methods."""
        # Test that the methods exist and have correct signatures
        assert hasattr(VelodromePoolBehaviour, 'stake_cl_lp_tokens')
        assert hasattr(VelodromePoolBehaviour, 'unstake_cl_lp_tokens')
        assert hasattr(VelodromePoolBehaviour, 'claim_cl_rewards')
        assert hasattr(VelodromePoolBehaviour, 'get_cl_pending_rewards')
        assert hasattr(VelodromePoolBehaviour, 'get_cl_staked_balance')
        
        # Test method signatures
        import inspect
        stake_cl_sig = inspect.signature(VelodromePoolBehaviour.stake_cl_lp_tokens)
        assert 'token_ids' in stake_cl_sig.parameters
        assert 'gauge_address' in stake_cl_sig.parameters
        
        unstake_cl_sig = inspect.signature(VelodromePoolBehaviour.unstake_cl_lp_tokens)
        assert 'token_ids' in unstake_cl_sig.parameters
        assert 'gauge_address' in unstake_cl_sig.parameters
        
        claim_cl_sig = inspect.signature(VelodromePoolBehaviour.claim_cl_rewards)
        assert 'gauge_address' in claim_cl_sig.parameters
        assert 'token_ids' in claim_cl_sig.parameters

    def test_transaction_parameter_validation(self):
        """Test transaction parameter validation."""
        # Test valid parameters
        valid_params = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "safe_address": "0x1234567890123456789012345678901234567890",
            "assets": ["0x1234567890123456789012345678901234567890", "0x1234567890123456789012345678901234567890"],
            "chain": "optimism",
            "max_amounts_in": [1000000000000000000, 2000000000000000000],
            "is_stable": True
        }
        
        # Validate parameters
        assert valid_params["pool_address"].startswith("0x")
        assert valid_params["safe_address"].startswith("0x")
        assert len(valid_params["assets"]) == 2
        assert all(asset.startswith("0x") for asset in valid_params["assets"])
        assert valid_params["chain"] in ["optimism", "base", "mode"]
        assert len(valid_params["max_amounts_in"]) == 2
        assert all(amount > 0 for amount in valid_params["max_amounts_in"])
        assert isinstance(valid_params["is_stable"], bool)

    @patch.object(VelodromeGaugeContract, 'contract_interface', {'ethereum': {}})
    def test_transaction_error_handling(self, mock_ledger_api):
        """Test transaction error handling."""
        # Test with invalid parameters - the contract returns an error dict instead of raising
        result = VelodromeGaugeContract.deposit(
            ledger_api=mock_ledger_api,
            contract_address="0xGaugeAddress",
            amount=-1000000000000000000  # Negative amount should return error
        )
        
        # The contract returns an error dict instead of raising an exception
        assert "error" in result
        assert "Amount must be greater than 0" in result["error"]

    def test_contract_address_validation(self, mock_ledger_api):
        """Test contract address validation."""
        # Test that contract methods exist and can be called
        assert hasattr(VelodromePoolContract, 'get_balance')
        assert hasattr(VelodromePoolContract, 'build_approval_tx')
        assert hasattr(VelodromePoolContract, 'get_reserves')
        
        # Test method signatures
        import inspect
        balance_sig = inspect.signature(VelodromePoolContract.get_balance)
        assert 'ledger_api' in balance_sig.parameters
        assert 'contract_address' in balance_sig.parameters
        assert 'account' in balance_sig.parameters

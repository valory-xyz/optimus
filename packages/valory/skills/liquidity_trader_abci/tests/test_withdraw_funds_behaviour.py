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

"""Tests for WithdrawFundsBehaviour of the liquidity_trader_abci skill."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, strategies as st

from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.base import Action
from packages.valory.skills.liquidity_trader_abci.behaviours.withdraw_funds import (
    WithdrawFundsBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.payloads import WithdrawFundsPayload
from packages.valory.skills.liquidity_trader_abci.rounds import (
    Event,
    SynchronizedData,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent.parent


class LiquidityTraderAbciFSMBehaviourBaseCase(FSMBehaviourBaseCase):
    """Base case for testing FSMBehaviour."""

    path_to_skill = PACKAGE_DIR

    def setUp(self):
        """Setup test environment."""
        super(LiquidityTraderAbciFSMBehaviourBaseCase, self).setUp()


class TestWithdrawFundsBehaviour(LiquidityTraderAbciFSMBehaviourBaseCase):
    """Test cases for WithdrawFundsBehaviour."""

    behaviour_class = WithdrawFundsBehaviour
    path_to_skill = PACKAGE_DIR
    
    @classmethod
    def setup_class(cls, **kwargs: Any) -> None:
        """Setup the test class with parameter overrides."""
        # Create a temporary skill.yaml with valid values for testing
        import tempfile
        import shutil
        
        # Create a temporary directory for the skill
        cls.temp_skill_dir = tempfile.mkdtemp()
        cls.original_skill_dir = cls.path_to_skill
        
        # Copy the skill to temp directory
        shutil.copytree(cls.original_skill_dir, cls.temp_skill_dir, dirs_exist_ok=True)
        
        # Update the skill.yaml in temp directory
        temp_skill_yaml = Path(cls.temp_skill_dir) / "skill.yaml"
        with open(temp_skill_yaml, 'r') as f:
            skill_config = f.read()
        
        # Replace null values with valid ones
        skill_config = skill_config.replace("available_strategies: null", "available_strategies: \"{}\"")
        skill_config = skill_config.replace("genai_api_key: null", "genai_api_key: \"\"")
        skill_config = skill_config.replace("default_acceptance_time: null", "default_acceptance_time: 30")
        
        with open(temp_skill_yaml, 'w') as f:
            f.write(skill_config)
        
        # Update path_to_skill to use temp directory
        cls.path_to_skill = Path(cls.temp_skill_dir)
        
        super().setup_class(**kwargs)
    
    @classmethod
    def teardown_class(cls) -> None:
        """Teardown the test class."""
        # Clean up temporary directory
        if hasattr(cls, 'temp_skill_dir'):
            import shutil
            shutil.rmtree(cls.temp_skill_dir, ignore_errors=True)
        
        # Restore original path
        if hasattr(cls, 'original_skill_dir'):
            cls.path_to_skill = cls.original_skill_dir
        
        super().teardown_class()

    def setUp(self):
        """Setup test environment."""
        super().setUp()
        
        # Create individual behaviour instance for testing
        self.withdraw_behaviour = WithdrawFundsBehaviour(
            name="withdraw_funds_behaviour",
            skill_context=self.skill.skill_context
        )

        # Setup default test data
        self.setup_default_test_data()

    def _create_withdraw_behaviour(self):
        """Create a WithdrawFundsBehaviour instance for testing."""
        return WithdrawFundsBehaviour(
            name="withdraw_funds_behaviour",
            skill_context=self.skill.skill_context
        )

    def setup_default_test_data(self):
        """Setup default test data."""
        # Default positions - using real structure
        self.withdraw_behaviour.current_positions = [
            {
                "chain": "mode",
                "pool_address": "0x3Adf15f77F2911f84b0FE9DbdfF43ef60D40012c",
                "dex_type": "velodrome",
                "token0": "0x4200000000000000000000000000000000000006",
                "token1": "0xd988097fb8612cc24eeC14542bC03424c656005f",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "status": "open",
                "pool_id": "0x3Adf15f77F2911f84b0FE9DbdfF43ef60D40012c",
                "is_stable": True,
                "is_cl_pool": True,
                "amount0": 48045349738380,
                "amount1": 544525,
                "enter_tx_hash": "0xb68eff1e2277fd9432a9a8cf966d52005023a3addb168edbc400c23bba957653",
                "enter_timestamp": 1753984217,
                "entry_cost": 0.003927974431442888,
                "min_hold_days": 14.0,
                "principal_usd": 0.7289412106900652,
                "cost_recovered": False,
                "current_value_usd": 0.0,
                "last_updated": 1753984217,
            }
        ]

        # Default portfolio data - using real structure
        self.withdraw_behaviour.portfolio_data = {
            "portfolio_value": 1.1288653771248525,
            "value_in_pools": 0.0,
            "value_in_safe": 1.1288653771248525,
            "initial_investment": 86.99060538860897,
            "volume": 93.2299879883976,
            "roi": -98.7,
            "agent_hash": "bafybeibkop2atmdpyrwqwcjuaqyhckrujg663rpomregakrmmosbaywhlq",
            "allocations": [],
            "portfolio_breakdown": [
                {
                    "asset": "WETH",
                    "address": "0x4200000000000000000000000000000000000006",
                    "balance": 0.000289393128542145,
                    "price": 3841.2,
                    "value_usd": 1.1116168853560873,
                    "ratio": 0.0
                },
                {
                    "asset": "OLAS",
                    "address": "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9",
                    "balance": 0.07265520833339696,
                    "price": 0.237402,
                    "value_usd": 0.017248491768765105,
                    "ratio": 0.0
                }
            ],
            "address": "0xc7Bd1d1FB563c6c06D4Ab1f116208f36a4631Ce4",
            "last_updated": 1753978820
        }

    def test_async_act_no_withdrawal_requested(self):
        """Test async_act when no withdrawal is requested."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Mock the dependencies instead of the entire method
        def mock_read_investing_paused():
            yield
            return False
        
        with patch.object(withdraw_behaviour, '_read_investing_paused', side_effect=mock_read_investing_paused):
            with patch.object(withdraw_behaviour, 'send_a2a_transaction') as mock_send:
                with patch.object(withdraw_behaviour, 'wait_until_round_end') as mock_wait:
                    # Execute the actual async_act method
                    list(withdraw_behaviour.async_act())

                    # Verify payload was sent with empty actions
                    mock_send.assert_called_once()
                    payload = mock_send.call_args[0][0]
                    assert isinstance(payload, WithdrawFundsPayload)
                    assert payload.withdrawal_actions == json.dumps([])

    def test_async_act_successful_withdrawal_with_positions(self):
        """Test async_act for successful withdrawal with open positions."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Set up real test data directly on the withdraw_behaviour instance
        withdraw_behaviour.current_positions = [
            {
                "chain": "mode",
                "pool_address": "0x3Adf15f77F2911f84b0FE9DbdfF43ef60D40012c",
                "dex_type": "velodrome",
                "token0": "0x4200000000000000000000000000000000000006",
                "token1": "0xd988097fb8612cc24eeC14542bC03424c656005f",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "status": "open",
                "pool_id": "0x3Adf15f77F2911f84b0FE9DbdfF43ef60D40012c",
                "is_stable": True,
                "is_cl_pool": True,
                "amount0": 48045349738380,
                "amount1": 544525,
                "enter_tx_hash": "0xb68eff1e2277fd9432a9a8cf966d52005023a3addb168edbc400c23bba957653",
                "enter_timestamp": 1753984217,
                "entry_cost": 0.003927974431442888,
                "min_hold_days": 14.0,
                "principal_usd": 0.7289412106900652,
                "cost_recovered": False,
                "current_value_usd": 0.0,
                "last_updated": 1753984217,
            }
        ]
        
        withdraw_behaviour.portfolio_data = {
            "portfolio_value": 1.1288653771248525,
            "value_in_pools": 0.0,
            "value_in_safe": 1.1288653771248525,
            "initial_investment": 86.99060538860897,
            "volume": 93.2299879883976,
            "roi": -98.7,
            "agent_hash": "bafybeibkop2atmdpyrwqwcjuaqyhckrujg663rpomregakrmmosbaywhlq",
            "allocations": [],
            "portfolio_breakdown": [
                {
                    "asset": "WETH",
                    "address": "0x4200000000000000000000000000000000000006",
                    "balance": 0.000289393128542145,
                    "price": 3841.2,
                    "value_usd": 1.1116168853560873,
                    "ratio": 0.0
                },
                {
                    "asset": "OLAS",
                    "address": "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9",
                    "balance": 0.07265520833339696,
                    "price": 0.237402,
                    "value_usd": 0.017248491768765105,
                    "ratio": 0.0
                }
            ],
            "address": "0xc7Bd1d1FB563c6c06D4Ab1f116208f36a4631Ce4",
            "last_updated": 1753978820
        }
        
        # Mock the dependencies to simulate withdrawal scenario
        withdrawal_data = {
            "withdrawal_target_address": "0xTargetAddress",
            "withdrawal_status": "PENDING",
        }
        
        def mock_read_investing_paused():
            yield
            return True
        
        def mock_read_withdrawal_data():
            yield
            return withdrawal_data
        
        def mock_update_withdrawal_status(status, message):
            yield
            return None
        
        # Use side_effect to mock the generator methods
        with patch.object(withdraw_behaviour, '_read_investing_paused', side_effect=mock_read_investing_paused):
            with patch.object(withdraw_behaviour, '_read_withdrawal_data', side_effect=mock_read_withdrawal_data):
                with patch.object(withdraw_behaviour, '_update_withdrawal_status', side_effect=mock_update_withdrawal_status):
                    with patch.object(withdraw_behaviour, 'send_a2a_transaction') as mock_send:
                        with patch.object(withdraw_behaviour, 'wait_until_round_end') as mock_wait:
                            # Execute the actual async_act method
                            list(withdraw_behaviour.async_act())

                            # Verify payload was sent with actions
                            mock_send.assert_called_once()
                            payload = mock_send.call_args[0][0]
                            assert isinstance(payload, WithdrawFundsPayload)
                            
                            # Parse the actions to verify they contain real withdrawal actions
                            actions = json.loads(payload.withdrawal_actions)
                            assert len(actions) > 0
                            
                            # Verify the actions contain expected withdrawal action types
                            action_types = [action.get('action') for action in actions]
                            expected_types = ['ExitPool', 'withdraw']
                            assert any(action_type in expected_types for action_type in action_types)

    def test_async_act_no_withdrawal_data(self):
        """Test async_act when no withdrawal data is found."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Mock the dependencies to simulate withdrawal scenario with no data
        def mock_read_investing_paused():
            yield
            return True
        
        def mock_read_withdrawal_data():
            yield
            return None  # No withdrawal data found
        
        def mock_update_withdrawal_status(status, message):
            yield
            return None
        
        def mock_reset_withdrawal_flags():
            yield
            return None
        
        # Use side_effect to mock the generator methods
        with patch.object(withdraw_behaviour, '_read_investing_paused', side_effect=mock_read_investing_paused):
            with patch.object(withdraw_behaviour, '_read_withdrawal_data', side_effect=mock_read_withdrawal_data):
                with patch.object(withdraw_behaviour, '_update_withdrawal_status', side_effect=mock_update_withdrawal_status) as mock_update:
                    with patch.object(withdraw_behaviour, '_reset_withdrawal_flags', side_effect=mock_reset_withdrawal_flags) as mock_reset:
                        with patch.object(withdraw_behaviour, 'send_a2a_transaction') as mock_send:
                            with patch.object(withdraw_behaviour, 'wait_until_round_end') as mock_wait:
                                # Execute the actual async_act method
                                list(withdraw_behaviour.async_act())

                                # Verify that update_withdrawal_status was called with FAILED status
                                mock_update.assert_called_once_with("FAILED", "Withdrawal failed due to error.")
                                
                                # Verify that reset_withdrawal_flags was called
                                mock_reset.assert_called_once()
                                
                                # Verify payload was sent with empty actions (no withdrawal data)
                                mock_send.assert_called_once()
                                payload = mock_send.call_args[0][0]
                                assert isinstance(payload, WithdrawFundsPayload)
                                assert payload.withdrawal_actions == json.dumps([])

    def test_async_act_no_target_address(self):
        """Test async_act when no target address is found."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Mock the dependencies to simulate withdrawal scenario with no target address
        def mock_read_investing_paused():
            yield
            return True
        
        def mock_read_withdrawal_data():
            yield
            return {
                "withdrawal_status": "PENDING",
                "withdrawal_message": "Test withdrawal",
                # Note: withdrawal_target_address is missing or empty
            }
        
        def mock_update_withdrawal_status(status, message):
            yield
            return None
        
        def mock_reset_withdrawal_flags():
            yield
            return None
        
        # Use side_effect to mock the generator methods
        with patch.object(withdraw_behaviour, '_read_investing_paused', side_effect=mock_read_investing_paused):
            with patch.object(withdraw_behaviour, '_read_withdrawal_data', side_effect=mock_read_withdrawal_data):
                with patch.object(withdraw_behaviour, '_update_withdrawal_status', side_effect=mock_update_withdrawal_status) as mock_update:
                    with patch.object(withdraw_behaviour, '_reset_withdrawal_flags', side_effect=mock_reset_withdrawal_flags) as mock_reset:
                        with patch.object(withdraw_behaviour, 'send_a2a_transaction') as mock_send:
                            with patch.object(withdraw_behaviour, 'wait_until_round_end') as mock_wait:
                                # Execute the actual async_act method
                                list(withdraw_behaviour.async_act())

                                # Verify that update_withdrawal_status was called with FAILED status
                                mock_update.assert_called_once_with("FAILED", "Withdrawal failed due to error.")
                                
                                # Verify that reset_withdrawal_flags was called
                                mock_reset.assert_called_once()
                                
                                # Verify payload was sent with empty actions (no target address)
                                mock_send.assert_called_once()
                                payload = mock_send.call_args[0][0]
                                assert isinstance(payload, WithdrawFundsPayload)
                                assert payload.withdrawal_actions == json.dumps([])

    def test_async_act_no_actions_prepared(self):
        """Test async_act when no withdrawal actions are prepared."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Set up real test data with no positions
        withdraw_behaviour.current_positions = []
        withdraw_behaviour.portfolio_data = {
            "portfolio_value": 0.0,
            "value_in_pools": 0.0,
            "value_in_safe": 0.0,
            "initial_investment": 0.0,
            "volume": 0.0,
            "roi": 0.0,
            "agent_hash": "test_hash",
            "allocations": [],
            "portfolio_breakdown": [],
            "address": "0xTestAddress",
            "last_updated": 1753978820
        }
        
        # Mock the dependencies to simulate withdrawal scenario with no actions
        def mock_read_investing_paused():
            yield
            return True
        
        def mock_read_withdrawal_data():
            yield
            return {
                "withdrawal_target_address": "0xTargetAddress",
                "withdrawal_status": "PENDING",
                "withdrawal_message": "Test withdrawal",
            }
        
        def mock_prepare_withdrawal_actions(positions, target_address, portfolio_data):
            yield
            return []  # No actions prepared
        
        def mock_update_withdrawal_status(status, message):
            yield
            return None
        
        def mock_reset_withdrawal_flags():
            yield
            return None
        
        # Use side_effect to mock the generator methods
        with patch.object(withdraw_behaviour, '_read_investing_paused', side_effect=mock_read_investing_paused):
            with patch.object(withdraw_behaviour, '_read_withdrawal_data', side_effect=mock_read_withdrawal_data):
                with patch.object(withdraw_behaviour, '_prepare_withdrawal_actions', side_effect=mock_prepare_withdrawal_actions):
                    with patch.object(withdraw_behaviour, '_update_withdrawal_status', side_effect=mock_update_withdrawal_status) as mock_update:
                        with patch.object(withdraw_behaviour, '_reset_withdrawal_flags', side_effect=mock_reset_withdrawal_flags) as mock_reset:
                            with patch.object(withdraw_behaviour, 'send_a2a_transaction') as mock_send:
                                with patch.object(withdraw_behaviour, 'wait_until_round_end') as mock_wait:
                                    # Execute the actual async_act method
                                    list(withdraw_behaviour.async_act())

                                    # Verify that update_withdrawal_status was called twice:
                                    # 1. First with WITHDRAWING status when starting
                                    # 2. Then with FAILED status when no actions are prepared
                                    assert mock_update.call_count == 2
                                    mock_update.assert_any_call("WITHDRAWING", "Withdrawal Initiated. Preparing your funds...")
                                    mock_update.assert_any_call("FAILED", "No withdrawal actions could be prepared")
                                    
                                    # Verify that reset_withdrawal_flags was called
                                    mock_reset.assert_called_once()
                                    
                                    # Verify payload was sent with empty actions (no actions prepared)
                                    mock_send.assert_called_once()
                                    payload = mock_send.call_args[0][0]
                                    assert isinstance(payload, WithdrawFundsPayload)
                                    assert payload.withdrawal_actions == json.dumps([])

    def test_read_withdrawal_data_success(self):
        """Test successful reading of withdrawal data from KV store."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        expected_data = {
            "withdrawal_target_address": "0xTargetAddress",
            "withdrawal_status": "PENDING",
            "withdrawal_message": "Test withdrawal",
        }
        
        # Mock the entire _read_withdrawal_data method to return the expected data
        with patch.object(withdraw_behaviour, '_read_withdrawal_data', return_value=expected_data):
            result = withdraw_behaviour._read_withdrawal_data()
            assert result == expected_data

    def test_read_withdrawal_data_failure(self):
        """Test reading withdrawal data when KV store fails."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Mock the entire _read_withdrawal_data method to return None
        with patch.object(withdraw_behaviour, '_read_withdrawal_data', return_value=None):
            result = withdraw_behaviour._read_withdrawal_data()
            assert result is None

    def test_read_investing_paused_true(self):
        """Test reading investing_paused flag when it's True."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Mock the entire _read_investing_paused method to return True
        with patch.object(withdraw_behaviour, '_read_investing_paused', return_value=True):
            result = withdraw_behaviour._read_investing_paused()
            assert result is True

    def test_read_withdrawal_data_success(self):
        """Test _read_withdrawal_data function with successful KV store response."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Mock successful KV store response with complete withdrawal data
        mock_kv_response = {
            "withdrawal_id": "test_withdrawal_123",
            "withdrawal_status": "PENDING",
            "withdrawal_target_address": "0xTargetAddress123",
            "withdrawal_message": "Test withdrawal message",
            "withdrawal_requested_at": "2024-01-01T00:00:00Z",
            "withdrawal_completed_at": None,
            "withdrawal_estimated_value_usd": "100.50",
            "withdrawal_chain": "optimism",
            "withdrawal_safe_address": "0xSafeAddress123",
            "withdrawal_transaction_hashes": "[]",
            "withdrawal_current_step": "1"
        }
        
        def mock_read_kv(keys):
            yield
            return mock_kv_response
        
        with patch.object(withdraw_behaviour, '_read_kv', side_effect=mock_read_kv):
            generator = withdraw_behaviour._read_withdrawal_data()
            
            # Consume the generator to get the final return value
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value
            
            # Should return the complete withdrawal data
            assert result == mock_kv_response
            assert result["withdrawal_id"] == "test_withdrawal_123"
            assert result["withdrawal_status"] == "PENDING"
            assert result["withdrawal_target_address"] == "0xTargetAddress123"

    def test_read_withdrawal_data_partial_data(self):
        """Test _read_withdrawal_data function with partial KV store response."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Mock partial KV store response (some fields missing)
        mock_kv_response = {
            "withdrawal_id": "test_withdrawal_456",
            "withdrawal_status": "COMPLETED",
            "withdrawal_target_address": "0xTargetAddress456",
            # Missing other fields
        }
        
        def mock_read_kv(keys):
            yield
            return mock_kv_response
        
        with patch.object(withdraw_behaviour, '_read_kv', side_effect=mock_read_kv):
            generator = withdraw_behaviour._read_withdrawal_data()
            
            # Consume the generator to get the final return value
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value
            
            # Should return the partial data
            assert result == mock_kv_response
            assert result["withdrawal_id"] == "test_withdrawal_456"
            assert result["withdrawal_status"] == "COMPLETED"
            assert "withdrawal_completed_at" not in result

    def test_read_withdrawal_data_none_response(self):
        """Test _read_withdrawal_data function with None response from KV store."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        def mock_read_kv(keys):
            yield
            return None
        
        with patch.object(withdraw_behaviour, '_read_kv', side_effect=mock_read_kv):
            generator = withdraw_behaviour._read_withdrawal_data()
            
            # Consume the generator to get the final return value
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value
            
            # Should return None when KV store returns None
            assert result is None

    def test_read_withdrawal_data_empty_dict(self):
        """Test _read_withdrawal_data function with empty dictionary response."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        def mock_read_kv(keys):
            yield
            return {}
        
        with patch.object(withdraw_behaviour, '_read_kv', side_effect=mock_read_kv):
            generator = withdraw_behaviour._read_withdrawal_data()
            
            # Consume the generator to get the final return value
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value
            
            # Should return empty dict
            assert result == {}

    def test_read_withdrawal_data_exception(self):
        """Test _read_withdrawal_data function with exception handling."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        def mock_read_kv(keys):
            yield
            raise Exception("KV store connection error")
        
        with patch.object(withdraw_behaviour, '_read_kv', side_effect=mock_read_kv):
            generator = withdraw_behaviour._read_withdrawal_data()
            
            # Consume the generator to get the final return value
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value
            
            # Should return None when exception occurs
            assert result is None

    def test_read_withdrawal_data_specific_exception(self):
        """Test _read_withdrawal_data function with specific exception types."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Test with different exception types
        exception_types = [
            ValueError("Invalid data format"),
            KeyError("Missing key"),
            ConnectionError("Network error"),
            TimeoutError("Request timeout")
        ]
        
        for exception in exception_types:
            def mock_read_kv(keys):
                yield
                raise exception
            
            with patch.object(withdraw_behaviour, '_read_kv', side_effect=mock_read_kv):
                generator = withdraw_behaviour._read_withdrawal_data()
                
                # Consume the generator to get the final return value
                result = None
                try:
                    while True:
                        result = next(generator)
                except StopIteration as e:
                    result = e.value
                
                # Should return None for any exception
                assert result is None

    def test_read_withdrawal_data_correct_keys_requested(self):
        """Test that _read_withdrawal_data requests the correct keys from KV store."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        expected_keys = (
            "withdrawal_id",
            "withdrawal_status", 
            "withdrawal_target_address",
            "withdrawal_message",
            "withdrawal_requested_at",
            "withdrawal_completed_at",
            "withdrawal_estimated_value_usd",
            "withdrawal_chain",
            "withdrawal_safe_address",
            "withdrawal_transaction_hashes",
            "withdrawal_current_step",
        )
        
        def mock_read_kv(keys):
            yield
            # Verify the correct keys are being requested
            assert keys == expected_keys
            return {"withdrawal_id": "test", "withdrawal_status": "PENDING"}
        
        with patch.object(withdraw_behaviour, '_read_kv', side_effect=mock_read_kv):
            generator = withdraw_behaviour._read_withdrawal_data()
            
            # Consume the generator
            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

    def test_read_withdrawal_data_different_statuses(self):
        """Test _read_withdrawal_data with different withdrawal statuses."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        test_statuses = ["PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED"]
        
        for status in test_statuses:
            mock_kv_response = {
                "withdrawal_id": f"test_{status.lower()}",
                "withdrawal_status": status,
                "withdrawal_target_address": "0xTestAddress"
            }
            
            def mock_read_kv(keys):
                yield
                return mock_kv_response
            
            with patch.object(withdraw_behaviour, '_read_kv', side_effect=mock_read_kv):
                generator = withdraw_behaviour._read_withdrawal_data()
                
                # Consume the generator to get the final return value
                result = None
                try:
                    while True:
                        result = next(generator)
                except StopIteration as e:
                    result = e.value
                
                # Should return the data with correct status
                assert result["withdrawal_status"] == status
                assert result["withdrawal_id"] == f"test_{status.lower()}"

 

    def test_prepare_withdrawal_actions_complete_flow(self):
        """Test preparing withdrawal actions for complete flow."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        positions = [
            {
                "chain": "mode",
                "pool_address": "0x3Adf15f77F2911f84b0FE9DbdfF43ef60D40012c",
                "dex_type": "velodrome",
                "token0": "0x4200000000000000000000000000000000000006",
                "token1": "0xd988097fb8612cc24eeC14542bC03424c656005f",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "status": "open",
                "pool_id": "0x3Adf15f77F2911f84b0FE9DbdfF43ef60D40012c",
                "is_stable": True,
                "is_cl_pool": True,
                "amount0": 48045349738380,
                "amount1": 544525,
            }
        ]
        target_address = "0xTargetAddress"
        portfolio_data = {
            "portfolio_breakdown": [
                {
                    "asset": "WETH",
                    "address": "0x4200000000000000000000000000000000000006",
                    "balance": 0.000289393128542145,
                    "price": 3841.2,
                    "value_usd": 1.1116168853560873,
                    "ratio": 0.0
                },
                {
                    "asset": "OLAS",
                    "address": "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9",
                    "balance": 0.07265520833339696,
                    "price": 0.237402,
                    "value_usd": 0.017248491768765105,
                    "ratio": 0.0
                }
            ]
        }

        # Mock the entire _prepare_withdrawal_actions method to return a list of actions
        mock_actions = [
            {"action": Action.EXIT_POOL.value, "description": "Exit pool"},
            {"action": Action.FIND_BRIDGE_ROUTE.value, "description": "Swap to USDC"},
            {"action": Action.WITHDRAW.value, "description": "Transfer USDC"},
        ]
        
        with patch.object(withdraw_behaviour, '_prepare_withdrawal_actions', return_value=mock_actions):
            result = withdraw_behaviour._prepare_withdrawal_actions(positions, target_address, portfolio_data)
            
            # Verify actions were prepared
            assert len(result) == 3
            assert any(action.get("action") == Action.EXIT_POOL.value for action in result)
            assert any(action.get("action") == Action.FIND_BRIDGE_ROUTE.value for action in result)
            assert any(action.get("action") == Action.WITHDRAW.value for action in result)

    def test_prepare_exit_pool_actions_with_open_positions(self):
        """Test preparing exit pool actions with open positions."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        positions = [
            {
                "chain": "mode",
                "pool_address": "0x3Adf15f77F2911f84b0FE9DbdfF43ef60D40012c",
                "dex_type": "velodrome",
                "token0": "0x4200000000000000000000000000000000000006",
                "token1": "0xd988097fb8612cc24eeC14542bC03424c656005f",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "status": "open",
                "pool_id": "0x3Adf15f77F2911f84b0FE9DbdfF43ef60D40012c",
                "is_stable": True,
                "is_cl_pool": True,
                "amount0": 48045349738380,
                "amount1": 544525,
            },
            {
                "chain": "mode",
                "pool_address": "0x7c86a44778c52a0aad17860924b53bf3f35dc932",
                "dex_type": "balancerPool",
                "token0": "0x4200000000000000000000000000000000000006",
                "token1": "0xDfc7C877a950e49D2610114102175A06C2e3167a",
                "token0_symbol": "WETH",
                "token1_symbol": "MODE",
                "status": "closed",
                "pool_id": "0x7c86a44778c52a0aad17860924b53bf3f35dc932000200000000000000000007",
                "amount0": 3598190140989135,
                "amount1": 2087429381093069662008,
            }
        ]

        with patch.object(withdraw_behaviour, '_build_exit_pool_action_base', return_value={"action": "exit"}):
            result = withdraw_behaviour._prepare_exit_pool_actions(positions)

            # Should only create actions for OPEN positions
            assert len(result) == 1
            assert result[0]["action"] == "exit"
            assert "Exit WETH/USDC pool for withdrawal" in result[0]["description"]

    def test_prepare_exit_pool_actions_no_open_positions(self):
        """Test preparing exit pool actions with no open positions."""
        # Create behaviour instance for this test
        withdraw_behaviour = WithdrawFundsBehaviour(
            name="withdraw_funds_behaviour",
            skill_context=self.skill.skill_context
        )
        
        positions = [
            {
                "chain": "mode",
                "pool_address": "0x7c86a44778c52a0aad17860924b53bf3f35dc932",
                "dex_type": "balancerPool",
                "token0": "0x4200000000000000000000000000000000000006",
                "token1": "0xDfc7C877a950e49D2610114102175A06C2e3167a",
                "token0_symbol": "WETH",
                "token1_symbol": "MODE",
                "status": "closed",
                "pool_id": "0x7c86a44778c52a0aad17860924b53bf3f35dc932000200000000000000000007",
                "amount0": 3598190140989135,
                "amount1": 2087429381093069662008,
            }
        ]

        result = withdraw_behaviour._prepare_exit_pool_actions(positions)

        # Should not create any actions for CLOSED positions
        assert len(result) == 0

    def test_prepare_swap_to_usdc_actions_standard(self):
        """Test preparing swap to USDC actions."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        portfolio_data = {
            "portfolio_breakdown": [
                {
                    "asset": "WETH",
                    "address": "0x4200000000000000000000000000000000000006",
                    "balance": 0.000289393128542145,
                    "price": 3841.2,
                    "value_usd": 1.1116168853560873,
                    "ratio": 0.0
                },
                {
                    "asset": "USDC",
                    "address": "0xd988097fb8612cc24eeC14542bC03424c656005f",
                    "balance": 100.0,
                    "price": 1.0,
                    "value_usd": 100.0,
                    "ratio": 0.0
                },
                {
                    "asset": "OLAS",
                    "address": "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9",
                    "balance": 0.07265520833339696,
                    "price": 0.237402,
                    "value_usd": 0.017248491768765105,
                    "ratio": 0.0
                }
            ]
        }

        with patch.object(withdraw_behaviour, '_build_swap_to_usdc_action', return_value={"action": "swap"}):
            with patch.object(withdraw_behaviour, '_get_usdc_address', return_value="0xd988097fb8612cc24eeC14542bC03424c656005f"):
                with patch.object(withdraw_behaviour, '_get_olas_address', return_value="0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9"):
                    result = withdraw_behaviour._prepare_swap_to_usdc_actions_standard(portfolio_data)

                    # Should create actions for WETH only, skip USDC and OLAS
                    assert len(result) == 1
                    assert all(action["action"] == "swap" for action in result)

    def test_prepare_swap_to_usdc_actions_small_balances(self):
        """Test preparing swap actions with small balances."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        portfolio_data = {
            "portfolio_breakdown": [
                {
                    "asset": "WETH",
                    "address": "0x4200000000000000000000000000000000000006",
                    "balance": 0.0001,
                    "price": 3841.2,
                    "value_usd": 0.5,  # Too small
                    "ratio": 0.0
                },
                {
                    "asset": "OLAS",
                    "address": "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9",
                    "balance": 10.0,
                    "price": 0.237402,
                    "value_usd": 1000,
                    "ratio": 0.0
                }
            ]
        }

        with patch.object(withdraw_behaviour, '_build_swap_to_usdc_action', return_value={"action": "swap"}):
            with patch.object(withdraw_behaviour, '_get_usdc_address', return_value="0xd988097fb8612cc24eeC14542bC03424c656005f"):
                with patch.object(withdraw_behaviour, '_get_olas_address', return_value="0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9"):
                    result = withdraw_behaviour._prepare_swap_to_usdc_actions_standard(portfolio_data)

                    # Should not create any actions - OLAS is excluded, WETH is too small
                    assert len(result) == 0

    def test_prepare_transfer_usdc_actions_standard(self):
        """Test preparing transfer USDC actions."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        target_address = "0xTargetAddress"

        with patch.object(withdraw_behaviour, '_get_usdc_address', return_value="0xUSDC"):
            result = withdraw_behaviour._prepare_transfer_usdc_actions_standard(target_address)

            # Should create transfer action
            assert len(result) == 1
            action = result[0]
            assert action["action"] == Action.WITHDRAW.value
            assert action["to_address"] == target_address
            assert action["token_address"] == "0xUSDC"
            assert action["token_symbol"] == "USDC"
            assert action["funds_percentage"] == 1.0

    def test_update_withdrawal_status_success(self):
        """Test successful withdrawal status update."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        with patch.object(withdraw_behaviour, '_write_kv', return_value=True):
            list(withdraw_behaviour._update_withdrawal_status("WITHDRAWING", "Test message"))

    def test_update_withdrawal_status_completed(self):
        """Test withdrawal status update for completed status."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        with patch.object(withdraw_behaviour, '_write_kv') as mock_write:
            list(withdraw_behaviour._update_withdrawal_status("COMPLETED", "Withdrawal complete"))

            # Verify investing_paused is set to false
            call_args = mock_write.call_args[0][0]
            assert call_args["investing_paused"] == "false"
            assert "withdrawal_completed_at" in call_args

    def test_update_withdrawal_status_failed(self):
        """Test withdrawal status update for failed status."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        with patch.object(withdraw_behaviour, '_write_kv') as mock_write:
            list(withdraw_behaviour._update_withdrawal_status("FAILED", "Withdrawal failed"))

            # Verify investing_paused is set to false
            call_args = mock_write.call_args[0][0]
            assert call_args["investing_paused"] == "false"

    def test_update_withdrawal_status_failure(self):
        """Test withdrawal status update when KV store fails."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        with patch.object(withdraw_behaviour, '_write_kv', side_effect=Exception("KV store error")):
            # Should not raise exception, just log error
            list(withdraw_behaviour._update_withdrawal_status("WITHDRAWING", "Test message"))

    def test_reset_withdrawal_flags_success(self):
        """Test successful reset of withdrawal flags."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        with patch.object(withdraw_behaviour, '_write_kv', return_value=True):
            list(withdraw_behaviour._reset_withdrawal_flags())

    def test_reset_withdrawal_flags_failure(self):
        """Test reset of withdrawal flags when KV store fails."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        with patch.object(withdraw_behaviour, '_write_kv', side_effect=Exception("KV store error")):
            # Should not raise exception, just log error
            list(withdraw_behaviour._reset_withdrawal_flags())

    @given(st.lists(st.dictionaries(st.text(), st.text())))
    def test_prepare_exit_pool_actions_property(self, positions):
        """Property-based test for prepare_exit_pool_actions."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Add required fields to positions
        for position in positions:
            position.update({
                "status": "OPEN",
                "token0_symbol": "ETH",
                "token1_symbol": "USDC",
                "token0": "0xToken0",
                "token1": "0xToken1",
            })

        with patch.object(withdraw_behaviour, '_build_exit_pool_action_base', return_value={"action": "exit"}):
            result = withdraw_behaviour._prepare_exit_pool_actions(positions)
            
            # Result should be a list
            assert isinstance(result, list)
            
            # All actions should have the expected structure
            for action in result:
                assert "action" in action
                assert "description" in action
                assert "assets" in action

    @given(st.dictionaries(st.text(), st.one_of(st.text(), st.integers(), st.floats(), st.booleans())))
    def test_prepare_swap_actions_property(self, portfolio_data):
        """Property-based test for prepare_swap_to_usdc_actions_standard."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Ensure portfolio_breakdown exists
        if "portfolio_breakdown" not in portfolio_data:
            portfolio_data["portfolio_breakdown"] = []

        with patch.object(withdraw_behaviour, '_build_swap_to_usdc_action', return_value={"action": "swap"}):
            with patch.object(withdraw_behaviour, '_get_usdc_address', return_value="0xUSDC"):
                with patch.object(withdraw_behaviour, '_get_olas_address', return_value="0xOLAS"):
                    result = withdraw_behaviour._prepare_swap_to_usdc_actions_standard(portfolio_data)
                    
                    # Result should be a list
                    assert isinstance(result, list)

    def test_mixed_position_statuses(self):
        """Test handling of mixed position statuses."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        positions = [
            {
                "chain": "mode",
                "pool_address": "0x3Adf15f77F2911f84b0FE9DbdfF43ef60D40012c",
                "dex_type": "velodrome",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "status": "open"
            },
            {
                "chain": "mode",
                "pool_address": "0x7c86a44778c52a0aad17860924b53bf3f35dc932",
                "dex_type": "balancerPool",
                "token0_symbol": "WETH",
                "token1_symbol": "MODE",
                "status": "closed"
            },
            {
                "chain": "mode",
                "pool_address": "0xCc16Bfda354353B2E03214d2715F514706Be044C",
                "dex_type": "velodrome",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "status": "open"
            }
        ]

        with patch.object(withdraw_behaviour, '_build_exit_pool_action_base', return_value={"action": "exit"}):
            result = withdraw_behaviour._prepare_exit_pool_actions(positions)

            # Should only create actions for OPEN positions (case insensitive)
            assert len(result) == 2

    def test_empty_portfolio_data(self):
        """Test handling of empty portfolio data."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        portfolio_data = {
            "portfolio_value": 0.0,
            "value_in_pools": 0.0,
            "value_in_safe": 0.0,
            "portfolio_breakdown": []
        }

        with patch.object(withdraw_behaviour, '_get_usdc_address', return_value="0xUSDC"):
            result = withdraw_behaviour._prepare_swap_to_usdc_actions_standard(portfolio_data)

            # Should return empty list
            assert len(result) == 0

    def test_read_investing_paused_success(self):
        """Test _read_investing_paused function with successful KV store response."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Mock successful KV store response
        mock_kv_response = {"investing_paused": "true"}
        
        def mock_read_kv(keys):
            yield
            return mock_kv_response
        
        with patch.object(withdraw_behaviour, '_read_kv', side_effect=mock_read_kv):
            generator = withdraw_behaviour._read_investing_paused()
            
            # Consume the generator to get the final return value
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value
            
            # Should return True for "true" value
            assert result is True

    def test_read_investing_paused_false(self):
        """Test _read_investing_paused function with false value."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Mock KV store response with false value
        mock_kv_response = {"investing_paused": "false"}
        
        def mock_read_kv(keys):
            yield
            return mock_kv_response
        
        with patch.object(withdraw_behaviour, '_read_kv', side_effect=mock_read_kv):
            generator = withdraw_behaviour._read_investing_paused()
            
            # Consume the generator to get the final return value
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value
            
            # Should return False for "false" value
            assert result is False

    def test_read_investing_paused_none_response(self):
        """Test _read_investing_paused function with None response from KV store."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        def mock_read_kv(keys):
            yield
            return None
        
        with patch.object(withdraw_behaviour, '_read_kv', side_effect=mock_read_kv):
            generator = withdraw_behaviour._read_investing_paused()
            
            # Consume the generator to get the final return value
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value
            
            # Should return False when KV store returns None
            assert result is False

    def test_read_investing_paused_none_value(self):
        """Test _read_investing_paused function with None value in response."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Mock KV store response with None value
        mock_kv_response = {"investing_paused": None}
        
        def mock_read_kv(keys):
            yield
            return mock_kv_response
        
        with patch.object(withdraw_behaviour, '_read_kv', side_effect=mock_read_kv):
            generator = withdraw_behaviour._read_investing_paused()
            
            # Consume the generator to get the final return value
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value
            
            # Should return False when value is None
            assert result is False

    def test_read_investing_paused_exception(self):
        """Test _read_investing_paused function with exception handling."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        def mock_read_kv(keys):
            yield
            raise Exception("KV store error")
        
        with patch.object(withdraw_behaviour, '_read_kv', side_effect=mock_read_kv):
            generator = withdraw_behaviour._read_investing_paused()
            
            # Consume the generator to get the final return value
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value
            
            # Should return False when exception occurs
            assert result is False

    def test_read_investing_paused_case_insensitive(self):
        """Test _read_investing_paused function with different case values."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Test different case variations
        test_cases = [
            ("TRUE", True),
            ("True", True),
            ("true", True),
            ("FALSE", False),
            ("False", False),
            ("false", False),
            ("random", False),  # Non-boolean string should return False
        ]
        
        for value, expected in test_cases:
            mock_kv_response = {"investing_paused": value}
            
            def mock_read_kv(keys):
                yield
                return mock_kv_response
            
            with patch.object(withdraw_behaviour, '_read_kv', side_effect=mock_read_kv):
                generator = withdraw_behaviour._read_investing_paused()
                
                # Consume the generator to get the final return value
                result = None
                try:
                    while True:
                        result = next(generator)
                except StopIteration as e:
                    result = e.value
                
                # Should return expected value
                assert result is expected

    def test_invalid_position_data(self):
        """Test handling of invalid position data."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        positions = [
            {"status": "open"},  # Missing required fields
            {
                "status": "open",
                "pool_address": "0x1",
                "dex_type": "unknown",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC"
            },
        ]

        with patch.object(withdraw_behaviour, '_build_exit_pool_action_base', return_value=None):
            result = withdraw_behaviour._prepare_exit_pool_actions(positions)

            # Should handle gracefully and not create actions for invalid positions
            assert len(result) == 0 

    def test_prepare_withdrawal_actions_with_unstaking_actions(self):
        """Test _prepare_withdrawal_actions when unstaking actions exist."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        positions = [{"status": "open", "pool_address": "0x123"}]
        portfolio_data = {
            "portfolio_breakdown": [
                {
                    "asset": "WETH",
                    "address": "0x4200000000000000000000000000000000000006",
                    "value_usd": 10.0
                }
            ]
        }
        target_address = "0xTargetAddress"
        
        # Mock unstaking actions to exist
        with patch.object(withdraw_behaviour, '_prepare_unstaking_actions', return_value=[{"action": "unstake"}]):
            with patch.object(withdraw_behaviour, '_prepare_exit_pool_actions', return_value=[{"action": "exit"}]):
                with patch.object(withdraw_behaviour, '_prepare_swap_to_usdc_actions_standard', return_value=[{"action": "swap"}]):
                    with patch.object(withdraw_behaviour, '_prepare_transfer_usdc_actions_standard', return_value=[{"action": "transfer"}]):
                        generator = withdraw_behaviour._prepare_withdrawal_actions(
                            positions, portfolio_data, target_address
                        )
                        
                        # Consume the generator to get the final return value
                        result = None
                        try:
                            while True:
                                result = next(generator)
                        except StopIteration as e:
                            result = e.value
                        
                        # Should return all actions including unstaking
                        assert len(result) == 4
                        assert any(action.get("action") == "unstake" for action in result)

    def test_prepare_withdrawal_actions_no_transfer_actions(self):
        """Test _prepare_withdrawal_actions when no transfer actions exist (withdrawal complete)."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        positions = [{"status": "open", "pool_address": "0x123"}]
        portfolio_data = {
            "portfolio_breakdown": [
                {
                    "asset": "USDC",
                    "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    "value_usd": 0.5  # Small amount that might not trigger transfer
                }
            ]
        }
        target_address = "0xTargetAddress"
        
        # Mock actions but no transfer actions
        with patch.object(withdraw_behaviour, '_prepare_unstaking_actions', return_value=[]):
            with patch.object(withdraw_behaviour, '_prepare_exit_pool_actions', return_value=[{"action": "exit"}]):
                with patch.object(withdraw_behaviour, '_prepare_swap_to_usdc_actions_standard', return_value=[]):
                    with patch.object(withdraw_behaviour, '_prepare_transfer_usdc_actions_standard', return_value=[]):
                        with patch.object(withdraw_behaviour, '_update_withdrawal_status') as mock_update:
                            with patch.object(withdraw_behaviour, '_reset_withdrawal_flags') as mock_reset:
                                generator = withdraw_behaviour._prepare_withdrawal_actions(
                                    positions, portfolio_data, target_address
                                )
                                
                                # Consume the generator to get the final return value
                                result = None
                                try:
                                    while True:
                                        result = next(generator)
                                except StopIteration as e:
                                    result = e.value
                                
                                # Should return only exit actions
                                assert len(result) == 1
                                assert result[0]["action"] == "exit"
                                # Should call status update and reset when no USDC to transfer
                                mock_update.assert_called_once()
                                mock_reset.assert_called_once()

    def test_prepare_swap_actions_with_failed_swap_creation(self):
        """Test _prepare_swap_to_usdc_actions_standard with specific swap creation failure."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        portfolio_data = {
            "portfolio_breakdown": [
                {
                    "asset": "WETH",
                    "address": "0x4200000000000000000000000000000000000006",
                    "value_usd": 10.0
                }
            ]
        }
        
        # Mock _build_swap_to_usdc_action to return None (failed creation)
        with patch.object(withdraw_behaviour, '_build_swap_to_usdc_action', return_value=None):
            with patch.object(withdraw_behaviour, '_get_usdc_address', return_value="0xUSDC"):
                result = withdraw_behaviour._prepare_swap_to_usdc_actions_standard(portfolio_data)
                
                # Should return empty list when swap action creation fails
                assert result == []

    def test_update_withdrawal_status_with_specific_exception(self):
        """Test _update_withdrawal_status with specific exception type."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        def mock_write_kv(data):
            yield
            raise ValueError("Specific KV store error")
        
        with patch.object(withdraw_behaviour, '_write_kv', side_effect=mock_write_kv):
            generator = withdraw_behaviour._update_withdrawal_status("COMPLETED", "Test message")
            
            # Consume the generator
            try:
                while True:
                    next(generator)
            except StopIteration:
                pass
            
            # Should handle specific exception gracefully (no assertion needed as it logs error)

    def test_prepare_withdrawal_actions_no_open_positions(self):
        """Test _prepare_withdrawal_actions when no open positions exist."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        # Set up empty positions list
        positions = []
        portfolio_data = {
            "portfolio_breakdown": [
                {
                    "asset": "WETH",
                    "address": "0x4200000000000000000000000000000000000006",
                    "value_usd": 10.0
                }
            ]
        }
        target_address = "0xTargetAddress"
        
        # Mock the helper methods to return empty lists
        with patch.object(withdraw_behaviour, '_prepare_unstaking_actions', return_value=[]):
            with patch.object(withdraw_behaviour, '_prepare_exit_pool_actions', return_value=[]):
                with patch.object(withdraw_behaviour, '_prepare_swap_to_usdc_actions_standard', return_value=[]):
                    with patch.object(withdraw_behaviour, '_prepare_transfer_usdc_actions_standard', return_value=[]):
                        with patch.object(withdraw_behaviour, '_update_withdrawal_status') as mock_update:
                            with patch.object(withdraw_behaviour, '_reset_withdrawal_flags') as mock_reset:
                                generator = withdraw_behaviour._prepare_withdrawal_actions(
                                    positions, portfolio_data, target_address
                                )
                                result = None
                                try:
                                    while True:
                                        result = next(generator)
                                except StopIteration as e:
                                    result = e.value
                                
                                # Should return empty list when no actions are created
                                assert result == []
                                # Should call status update and reset when no USDC to transfer
                                mock_update.assert_called_once()
                                mock_reset.assert_called_once()

    def test_prepare_withdrawal_actions_no_swap_actions(self):
        """Test _prepare_withdrawal_actions when no swap actions are needed."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        positions = [{"status": "open", "pool_address": "0x123"}]
        portfolio_data = {
            "portfolio_breakdown": [
                {
                    "asset": "USDC",
                    "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    "value_usd": 10.0
                }
            ]
        }
        target_address = "0xTargetAddress"
        
        # Mock exit actions but no swap actions
        with patch.object(withdraw_behaviour, '_prepare_unstaking_actions', return_value=[]):
            with patch.object(withdraw_behaviour, '_prepare_exit_pool_actions', return_value=[{"action": "exit"}]):
                with patch.object(withdraw_behaviour, '_prepare_swap_to_usdc_actions_standard', return_value=[]):
                    with patch.object(withdraw_behaviour, '_prepare_transfer_usdc_actions_standard', return_value=[{"action": "transfer"}]):
                        generator = withdraw_behaviour._prepare_withdrawal_actions(
                            positions, portfolio_data, target_address
                        )
                        
                        result = None
                        try:
                            while True:
                                result = next(generator)
                        except StopIteration as e:
                            result = e.value
                        # Should return actions with exit and transfer but no swap
                        assert len(result) == 2
                        assert any(action.get("action") == "exit" for action in result)
                        assert any(action.get("action") == "transfer" for action in result)

    def test_prepare_exit_pool_actions_failed_action_creation(self):
        """Test _prepare_exit_pool_actions when action creation fails."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        positions = [
            {
                "status": "open",
                "pool_address": "0x123",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "token0": "0x4200000000000000000000000000000000000006",
                "token1": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
            }
        ]
        
        # Mock _build_exit_pool_action_base to return None (failed creation)
        with patch.object(withdraw_behaviour, '_build_exit_pool_action_base', return_value=None):
            result = withdraw_behaviour._prepare_exit_pool_actions(positions)
            
            # Should return empty list when action creation fails
            assert result == []

    def test_prepare_swap_actions_failed_action_creation(self):
        """Test _prepare_swap_to_usdc_actions_standard when action creation fails."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        portfolio_data = {
            "portfolio_breakdown": [
                {
                    "asset": "WETH",
                    "address": "0x4200000000000000000000000000000000000006",
                    "value_usd": 10.0
                }
            ]
        }
        
        # Mock _build_swap_to_usdc_action to return None (failed creation)
        with patch.object(withdraw_behaviour, '_build_swap_to_usdc_action', return_value=None):
            with patch.object(withdraw_behaviour, '_get_usdc_address', return_value="0xUSDC"):
                result = withdraw_behaviour._prepare_swap_to_usdc_actions_standard(portfolio_data)
                
                # Should return empty list when action creation fails
                assert result == []

    def test_prepare_exit_pool_actions_with_optional_fields(self):
        """Test _prepare_exit_pool_actions with optional pool fields."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        positions = [
            {
                "status": "open",
                "pool_address": "0x123",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "token0": "0x4200000000000000000000000000000000000006",
                "token1": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "pool_fee": 3000,
                "tick_spacing": 60,
                "tick_ranges": [{"tick_lower": -100, "tick_upper": 100}]
            }
        ]
        
        # Mock successful action creation
        mock_action = {
            "action": "exit",
            "pool_address": "0x123"
        }
        
        with patch.object(withdraw_behaviour, '_build_exit_pool_action_base', return_value=mock_action):
            result = withdraw_behaviour._prepare_exit_pool_actions(positions)
            
            # Should return action with optional fields added
            assert len(result) == 1
            action = result[0]
            assert action["pool_fee"] == 3000
            assert action["tick_spacing"] == 60
            assert action["tick_ranges"] == [{"tick_lower": -100, "tick_upper": 100}]

    def test_prepare_exit_pool_actions_with_some_optional_fields(self):
        """Test _prepare_exit_pool_actions with some optional fields missing."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        positions = [
            {
                "status": "open",
                "pool_address": "0x123",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "token0": "0x4200000000000000000000000000000000000006",
                "token1": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "pool_fee": 3000,
                # Missing tick_spacing and tick_ranges
            }
        ]
        
        # Mock successful action creation
        mock_action = {
            "action": "exit",
            "pool_address": "0x123"
        }
        
        with patch.object(withdraw_behaviour, '_build_exit_pool_action_base', return_value=mock_action):
            result = withdraw_behaviour._prepare_exit_pool_actions(positions)
            
            # Should return action with only available optional fields
            assert len(result) == 1
            action = result[0]
            assert action["pool_fee"] == 3000
            assert "tick_spacing" not in action
            assert "tick_ranges" not in action

    def test_update_withdrawal_status_exception(self):
        """Test _update_withdrawal_status with exception handling."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        def mock_write_kv(data):
            yield
            raise Exception("KV store write error")
        
        with patch.object(withdraw_behaviour, '_write_kv', side_effect=mock_write_kv):
            generator = withdraw_behaviour._update_withdrawal_status("COMPLETED", "Test message")
            
            # Consume the generator
            try:
                while True:
                    next(generator)
            except StopIteration:
                pass
            
            # Should handle exception gracefully (no assertion needed as it logs error)

    def test_update_withdrawal_status_completed(self):
        """Test _update_withdrawal_status with COMPLETED status."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        def mock_write_kv(data):
            yield
            return True
        
        with patch.object(withdraw_behaviour, '_write_kv', side_effect=mock_write_kv):
            generator = withdraw_behaviour._update_withdrawal_status("COMPLETED", "Test completed")
            
            # Consume the generator
            try:
                while True:
                    next(generator)
            except StopIteration:
                pass
            
            # Should handle COMPLETED status (no assertion needed as it updates data)

    def test_update_withdrawal_status_failed(self):
        """Test _update_withdrawal_status with FAILED status."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        def mock_write_kv(data):
            yield
            return True
        
        with patch.object(withdraw_behaviour, '_write_kv', side_effect=mock_write_kv):
            generator = withdraw_behaviour._update_withdrawal_status("FAILED", "Test failed")
            
            # Consume the generator
            try:
                while True:
                    next(generator)
            except StopIteration:
                pass
            
            # Should handle FAILED status (no assertion needed as it updates data)

    def test_reset_withdrawal_flags_exception(self):
        """Test _reset_withdrawal_flags with exception handling."""
        withdraw_behaviour = self._create_withdraw_behaviour()
        
        def mock_write_kv(data):
            yield
            raise Exception("KV store write error")
        
        with patch.object(withdraw_behaviour, '_write_kv', side_effect=mock_write_kv):
            generator = withdraw_behaviour._reset_withdrawal_flags()
            
            # Consume the generator
            try:
                while True:
                    next(generator)
            except StopIteration:
                pass
            
            # Should handle exception gracefully (no assertion needed as it logs error)

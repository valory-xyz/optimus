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

"""This module contains the tests for the CheckStakingKPIMetBehaviour."""

import datetime
import logging
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from hypothesis import given, strategies as st

from packages.valory.contracts.gnosis_safe.contract import SafeOperation
from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met import (
    CheckStakingKPIMetBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.payloads import CheckStakingKPIMetPayload
from packages.valory.skills.liquidity_trader_abci.states.base import (
    StakingState,
    SynchronizedData,
)


class TestCheckStakingKPIMetBehaviour(FSMBehaviourBaseCase):
    """Test CheckStakingKPIMetBehaviour."""

    path_to_skill = Path(__file__).parent.parent.parent

    @pytest.fixture
    def check_staking_kpi_behaviour(self):
        """Create a CheckStakingKPIMetBehaviour instance for testing."""
        return CheckStakingKPIMetBehaviour(
            name="check_staking_kpi_behaviour",
            skill_context=self.skill.skill_context,
        )

    def _consume_generator(self, generator: Generator) -> Any:
        """Consume a generator and return the final value."""
        result = None
        try:
            while True:
                result = next(generator)
        except StopIteration as e:
            result = e.value
        return result

    def setup_method(self, **kwargs: Any) -> None:
        """Set up the test method with mocked dependencies."""
        # Mock the store path validation before calling super().setup()
        with patch(
            "packages.valory.skills.liquidity_trader_abci.models.Params.get_store_path",
            return_value=Path("/tmp/mock_store"),
        ):
            super().setup(**kwargs)
        self.check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        self.setup_default_test_data()

    def teardown_method(self, **kwargs: Any) -> None:
        """Teardown the test method."""
        super().teardown(**kwargs)

    def _create_check_staking_kpi_behaviour(self):
        """Create a CheckStakingKPIMetBehaviour instance for testing."""
        return CheckStakingKPIMetBehaviour(
            name="check_staking_kpi_behaviour",
            skill_context=self.skill.skill_context,
        )

    def setup_default_test_data(self):
        """Setup default test data."""
        # Set default parameters
        self.check_staking_kpi_behaviour.params.staking_chain = "optimism"
        self.check_staking_kpi_behaviour.params.safe_contract_addresses = {
            "optimism": "0xSafeAddress123456789012345678901234567890123456"
        }
        self.check_staking_kpi_behaviour.params.staking_threshold_period = 5

    @pytest.mark.parametrize("is_staking_kpi_met,period_count,period_number_at_last_cp,threshold_period,min_num_of_safe_tx_required,multisig_nonces_since_last_cp,expected_vanity_tx,expected_is_staking_kpi_met,test_description", [
        # Test 1: KPI already met
        (
            True,
            10,
            5,
            5,
            5,
            10,
            None,
            True,
            "KPI already met"
        ),
        # Test 2: KPI check error (None)
        (
            None,
            10,
            5,
            5,
            5,
            10,
            None,
            None,
            "KPI check error"
        ),
        # Test 3: Period threshold not exceeded
        (
            False,
            10,
            8,
            5,
            5,
            10,
            None,
            False,
            "period threshold not exceeded"
        ),
        # Test 4: Period threshold exceeded, no transactions left to meet KPI
        (
            False,
            10,
            3,
            5,
            5,
            10,
            None,
            False,
            "period threshold exceeded, no tx left"
        ),
        # Test 5: Period threshold exceeded, transactions left to meet KPI, successful vanity tx
        (
            False,
            10,
            3,
            5,
            10,
            5,
            "0xVanityTxHash1234567890123456789012345678901234567890123456789012345678901234",
            False,
            "period threshold exceeded, tx left, successful vanity tx"
        ),
        # Test 6: Period threshold exceeded, transactions left to meet KPI, vanity tx preparation fails
        (
            False,
            10,
            3,
            5,
            10,
            5,
            None,
            False,
            "period threshold exceeded, tx left, vanity tx fails"
        ),
        # Test 7: Multisig nonces since last cp is None
        (
            False,
            10,
            3,
            5,
            10,
            None,
            None,
            False,
            "multisig nonces since last cp is None"
        ),
        # Test 8: Min num of safe tx required is None
        (
            False,
            10,
            3,
            5,
            None,
            5,
            None,
            False,
            "min num of safe tx required is None"
        ),
    ])
    def test_async_act_variations(self, is_staking_kpi_met, period_count, period_number_at_last_cp, threshold_period, min_num_of_safe_tx_required, multisig_nonces_since_last_cp, expected_vanity_tx, expected_is_staking_kpi_met, test_description):
        """Test async_act method with various scenarios."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        # Set up parameters - unfreeze params first
        check_staking_kpi_behaviour.params.__dict__["_frozen"] = False
        check_staking_kpi_behaviour.params.staking_chain = "optimism"
        check_staking_kpi_behaviour.params.safe_contract_addresses = {
            "optimism": "0xSafeAddress123456789012345678901234567890123456"
        }
        check_staking_kpi_behaviour.params.staking_threshold_period = threshold_period
        check_staking_kpi_behaviour.params.__dict__["_frozen"] = True
        
        # Mock synchronized data
        mock_synchronized_data = MagicMock()
        mock_synchronized_data.period_count = period_count
        mock_synchronized_data.period_number_at_last_cp = period_number_at_last_cp
        mock_synchronized_data.min_num_of_safe_tx_required = min_num_of_safe_tx_required
        
        # Mock dependencies
        def mock_is_staking_kpi_met():
            yield
            return is_staking_kpi_met
        
        def mock_get_multisig_nonces_since_last_cp(chain, multisig):
            yield
            return multisig_nonces_since_last_cp
        
        def mock_prepare_vanity_tx(chain):
            yield
            return expected_vanity_tx
        
        with patch.object(check_staking_kpi_behaviour, '_is_staking_kpi_met', side_effect=mock_is_staking_kpi_met):
            with patch.object(check_staking_kpi_behaviour, '_get_multisig_nonces_since_last_cp', side_effect=mock_get_multisig_nonces_since_last_cp):
                with patch.object(check_staking_kpi_behaviour, '_prepare_vanity_tx', side_effect=mock_prepare_vanity_tx):
                    with patch.object(check_staking_kpi_behaviour, 'send_a2a_transaction') as mock_send:
                        with patch.object(check_staking_kpi_behaviour, 'wait_until_round_end') as mock_wait:
                            with patch.object(type(check_staking_kpi_behaviour), 'synchronized_data', new_callable=PropertyMock, return_value=mock_synchronized_data):
                                # Execute the actual async_act method
                                list(check_staking_kpi_behaviour.async_act())

                                # Verify payload was sent with correct values
                                mock_send.assert_called_once()
                                payload = mock_send.call_args[0][0]
                                assert isinstance(payload, CheckStakingKPIMetPayload)
                                assert payload.tx_hash == expected_vanity_tx
                                assert payload.is_staking_kpi_met == expected_is_staking_kpi_met
                                assert payload.safe_contract_address == "0xSafeAddress123456789012345678901234567890123456"
                                assert payload.chain_id == "optimism"

    def test_prepare_vanity_tx_success(self):
        """Test _prepare_vanity_tx with successful contract interaction."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        mock_safe_tx_hash = "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
        expected_final_hash = "0xFinalHash1234567890123456789012345678901234567890123456789012345678901234"
        
        def mock_contract_interact(**kwargs):
            yield
            return mock_safe_tx_hash
        
        with patch.object(check_staking_kpi_behaviour, 'contract_interact', side_effect=mock_contract_interact):
            with patch('packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.hash_payload_to_hex') as mock_hash_payload:
                mock_hash_payload.return_value = expected_final_hash
                
                generator = check_staking_kpi_behaviour._prepare_vanity_tx("optimism")
                result = self._consume_generator(generator)
                
                # Should return the final hash
                assert result == expected_final_hash
                
                # Verify hash_payload_to_hex was called with correct parameters
                mock_hash_payload.assert_called_once_with(
                    safe_tx_hash="SafeTxHash1234567890123456789012345678901234567890123456789012345678901234",
                    ether_value=0,
                    safe_tx_gas=0,
                    operation=SafeOperation.CALL.value,
                    to_address="0x0000000000000000000000000000000000000000",
                    data=b"0x"
                )

    def test_prepare_vanity_tx_contract_failure(self):
        """Test _prepare_vanity_tx when contract interaction fails."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        def mock_contract_interact(**kwargs):
            yield
            raise Exception("Contract interaction failed")
        
        with patch.object(check_staking_kpi_behaviour, 'contract_interact', side_effect=mock_contract_interact):
            generator = check_staking_kpi_behaviour._prepare_vanity_tx("optimism")
            result = self._consume_generator(generator)
            
            # Should return None when contract interaction fails
            assert result is None

    def test_prepare_vanity_tx_none_safe_tx_hash(self):
        """Test _prepare_vanity_tx when safe_tx_hash is None."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        def mock_contract_interact(**kwargs):
            yield
            return None
        
        with patch.object(check_staking_kpi_behaviour, 'contract_interact', side_effect=mock_contract_interact):
            generator = check_staking_kpi_behaviour._prepare_vanity_tx("optimism")
            result = self._consume_generator(generator)
            
            # Should return None when safe_tx_hash is None
            assert result is None

    def test_prepare_vanity_tx_hash_payload_exception(self):
        """Test _prepare_vanity_tx when hash_payload_to_hex raises an exception."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        mock_safe_tx_hash = "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
        
        def mock_contract_interact(**kwargs):
            yield
            return mock_safe_tx_hash
        
        with patch.object(check_staking_kpi_behaviour, 'contract_interact', side_effect=mock_contract_interact):
            with patch('packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.hash_payload_to_hex') as mock_hash_payload:
                mock_hash_payload.side_effect = Exception("Hash payload conversion failed")
                
                generator = check_staking_kpi_behaviour._prepare_vanity_tx("optimism")
                result = self._consume_generator(generator)
                
                # Should return None when hash_payload_to_hex fails
                assert result is None

    def test_prepare_vanity_tx_empty_hash(self):
        """Test _prepare_vanity_tx with empty hash."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        mock_safe_tx_hash = ""
        
        # Mock contract_interact to return the empty hash
        def mock_contract_interact(**kwargs):
            yield
            return mock_safe_tx_hash
        
        with patch.object(check_staking_kpi_behaviour, 'contract_interact', side_effect=mock_contract_interact):
            # The function should return None when hash is empty
            generator = check_staking_kpi_behaviour._prepare_vanity_tx("optimism")
            result = self._consume_generator(generator)
            
            # Should return None for empty hash
            assert result is None

    def test_prepare_vanity_tx_single_character_hash(self):
        """Test _prepare_vanity_tx with single character hash."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        mock_safe_tx_hash = "0x1"  # Single character after prefix
        
        # Mock contract_interact to return the safe_tx_hash
        def mock_contract_interact(**kwargs):
            yield
            return mock_safe_tx_hash
        
        with patch.object(check_staking_kpi_behaviour, 'contract_interact', side_effect=mock_contract_interact):
            with patch('packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.hash_payload_to_hex') as mock_hash_payload:
                mock_hash_payload.return_value = "0xFinalHash"
                
                generator = check_staking_kpi_behaviour._prepare_vanity_tx("optimism")
                result = self._consume_generator(generator)
                
                # Should return the final hash
                assert result == "0xFinalHash"
                
                # Verify hash_payload_to_hex was called with "1" (prefix removed)
                mock_hash_payload.assert_called_once_with(
                    safe_tx_hash="1",
                    ether_value=0,
                    safe_tx_gas=0,
                    operation=SafeOperation.CALL.value,
                    to_address="0x0000000000000000000000000000000000000000",
                    data=b"0x"
                )

    @pytest.mark.parametrize("staking_chain,safe_addresses,expected_safe_address,test_description", [
        # Test 1: Valid addresses
        (
            "optimism",
            {"optimism": "0xSafeAddress123456789012345678901234567890123456"},
            "0xSafeAddress123456789012345678901234567890123456",
            "valid addresses"
        ),
        # Test 2: Missing safe address
        (
            "optimism",
            {},
            None,
            "missing safe address"
        ),
        # Test 3: Different chain
        (
            "mode",
            {"optimism": "0xSafeAddress123456789012345678901234567890123456"},
            None,
            "different chain"
        ),
    ])
    def test_address_handling(self, staking_chain, safe_addresses, expected_safe_address, test_description):
        """Test address handling in payload creation."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        # Set parameters - unfreeze params first
        check_staking_kpi_behaviour.params.__dict__["_frozen"] = False
        check_staking_kpi_behaviour.params.staking_chain = staking_chain
        check_staking_kpi_behaviour.params.safe_contract_addresses = safe_addresses
        check_staking_kpi_behaviour.params.__dict__["_frozen"] = True
        
        # Mock dependencies to avoid actual calls
        def mock_is_staking_kpi_met():
            yield
            return True  # KPI already met
        
        with patch.object(check_staking_kpi_behaviour, '_is_staking_kpi_met', side_effect=mock_is_staking_kpi_met):
            with patch.object(check_staking_kpi_behaviour, 'send_a2a_transaction') as mock_send:
                with patch.object(check_staking_kpi_behaviour, 'wait_until_round_end') as mock_wait:
                    # Execute the actual async_act method
                    list(check_staking_kpi_behaviour.async_act())

                    # Verify payload was sent with correct addresses
                    mock_send.assert_called_once()
                    payload = mock_send.call_args[0][0]
                    assert isinstance(payload, CheckStakingKPIMetPayload)
                    assert payload.safe_contract_address == expected_safe_address
                    assert payload.chain_id == staking_chain

    def test_async_act_benchmark_tool_usage(self):
        """Test that async_act uses benchmark tool correctly."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        # Set up parameters - unfreeze params first
        check_staking_kpi_behaviour.params.__dict__["_frozen"] = False
        check_staking_kpi_behaviour.params.staking_chain = "optimism"
        check_staking_kpi_behaviour.params.safe_contract_addresses = {
            "optimism": "0xSafeAddress123456789012345678901234567890123456"
        }
        check_staking_kpi_behaviour.params.__dict__["_frozen"] = True
        
        # Mock dependencies
        with patch.object(check_staking_kpi_behaviour, '_is_staking_kpi_met') as mock_is_staking_kpi_met:
            mock_is_staking_kpi_met.return_value = iter([True])  # KPI already met
            
            with patch.object(check_staking_kpi_behaviour, 'send_a2a_transaction') as mock_send:
                with patch.object(check_staking_kpi_behaviour, 'wait_until_round_end') as mock_wait:
                    # Mock benchmark tool
                    mock_benchmark = MagicMock()
                    mock_measure = MagicMock()
                    mock_local = MagicMock()
                    mock_consensus = MagicMock()
                    
                    mock_benchmark.measure.return_value = mock_measure
                    mock_measure.local.return_value = mock_local
                    mock_measure.consensus.return_value = mock_consensus
                    
                    check_staking_kpi_behaviour.context.benchmark_tool = mock_benchmark
                    
                    # Execute the actual async_act method
                    list(check_staking_kpi_behaviour.async_act())

                    # Verify benchmark tool was used
                    mock_benchmark.measure.assert_called_with(check_staking_kpi_behaviour.behaviour_id)
                    mock_measure.local.assert_called()
                    mock_measure.consensus.assert_called()

    def test_async_act_auto_round_id(self):
        """Test that async_act uses auto_round_id correctly."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        # Set up parameters - unfreeze params first
        check_staking_kpi_behaviour.params.__dict__["_frozen"] = False
        check_staking_kpi_behaviour.params.staking_chain = "optimism"
        check_staking_kpi_behaviour.params.safe_contract_addresses = {
            "optimism": "0xSafeAddress123456789012345678901234567890123456"
        }
        check_staking_kpi_behaviour.params.__dict__["_frozen"] = True
        
        # Mock dependencies
        def mock_is_staking_kpi_met():
            yield
            return True  # KPI already met
        
        # Mock matching_round.auto_round_id
        mock_auto_round_id = MagicMock()
        mock_auto_round_id.return_value = "test_round_id"
        check_staking_kpi_behaviour.matching_round.auto_round_id = mock_auto_round_id
        
        with patch.object(check_staking_kpi_behaviour, '_is_staking_kpi_met', side_effect=mock_is_staking_kpi_met):
            with patch.object(check_staking_kpi_behaviour, 'send_a2a_transaction') as mock_send:
                with patch.object(check_staking_kpi_behaviour, 'wait_until_round_end') as mock_wait:
                    # Execute the actual async_act method
                    list(check_staking_kpi_behaviour.async_act())

                    # Verify auto_round_id was called
                    mock_auto_round_id.assert_called_once()
                    
                    # Verify payload was sent with correct tx_submitter
                    mock_send.assert_called_once()
                    payload = mock_send.call_args[0][0]
                    assert payload.tx_submitter == "test_round_id"

    def test_async_act_set_done(self):
        """Test that async_act calls set_done at the end."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        # Set up parameters - unfreeze params first
        check_staking_kpi_behaviour.params.__dict__["_frozen"] = False
        check_staking_kpi_behaviour.params.staking_chain = "optimism"
        check_staking_kpi_behaviour.params.safe_contract_addresses = {
            "optimism": "0xSafeAddress123456789012345678901234567890123456"
        }
        check_staking_kpi_behaviour.params.__dict__["_frozen"] = True
        
        # Mock dependencies
        def mock_is_staking_kpi_met():
            yield
            return True  # KPI already met
        
        with patch.object(check_staking_kpi_behaviour, '_is_staking_kpi_met', side_effect=mock_is_staking_kpi_met):
            with patch.object(check_staking_kpi_behaviour, 'send_a2a_transaction') as mock_send:
                with patch.object(check_staking_kpi_behaviour, 'wait_until_round_end') as mock_wait:
                    with patch.object(check_staking_kpi_behaviour, 'set_done') as mock_set_done:
                        # Execute the actual async_act method
                        list(check_staking_kpi_behaviour.async_act())

                        # Verify set_done was called
                        mock_set_done.assert_called_once()

    def test_prepare_vanity_tx_contract_interaction_called(self):
        """Test that _prepare_vanity_tx calls contract_interact."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        mock_safe_tx_hash = "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
        
        # Mock contract_interact to return the safe_tx_hash
        def mock_contract_interact(**kwargs):
            yield
            return mock_safe_tx_hash
        
        with patch.object(check_staking_kpi_behaviour, 'contract_interact', side_effect=mock_contract_interact) as mock_contract_interact_patch:
            with patch('packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.hash_payload_to_hex') as mock_hash_payload:
                mock_hash_payload.return_value = "0xFinalHash"
                
                generator = check_staking_kpi_behaviour._prepare_vanity_tx("optimism")
                result = self._consume_generator(generator)
                
                # Verify contract_interact was called once
                mock_contract_interact_patch.assert_called_once()
                
                # Verify hash_payload_to_hex was called with the processed hash
                mock_hash_payload.assert_called_once_with(
                    safe_tx_hash="SafeTxHash1234567890123456789012345678901234567890123456789012345678901234",
                    ether_value=0,
                    safe_tx_gas=0,
                    operation=SafeOperation.CALL.value,
                    to_address="0x0000000000000000000000000000000000000000",
                    data=b"0x"
                )

    @given(st.integers(min_value=0, max_value=100))
    def test_prepare_vanity_tx_property_chain_names(self, chain_id):
        """Property-based test for _prepare_vanity_tx with different chain IDs."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        mock_safe_tx_hash = "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
        
        # Mock contract_interact to return the safe_tx_hash
        def mock_contract_interact(**kwargs):
            yield
            return mock_safe_tx_hash
        
        with patch.object(check_staking_kpi_behaviour, 'contract_interact', side_effect=mock_contract_interact):
            with patch('packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.hash_payload_to_hex') as mock_hash_payload:
                mock_hash_payload.return_value = "0xFinalHash"
                
                generator = check_staking_kpi_behaviour._prepare_vanity_tx(f"chain_{chain_id}")
                result = self._consume_generator(generator)
                
                # Should return the final hash
                assert result == "0xFinalHash"

    def test_async_act_exception_handling(self):
        """Test that async_act handles exceptions correctly."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        # Set up parameters - unfreeze params first
        check_staking_kpi_behaviour.params.__dict__["_frozen"] = False
        check_staking_kpi_behaviour.params.staking_chain = "optimism"
        check_staking_kpi_behaviour.params.safe_contract_addresses = {
            "optimism": "0xSafeAddress123456789012345678901234567890123456"
        }
        check_staking_kpi_behaviour.params.__dict__["_frozen"] = True
        
        # Mock dependencies to raise exception
        with patch.object(check_staking_kpi_behaviour, '_is_staking_kpi_met') as mock_is_staking_kpi_met:
            mock_is_staking_kpi_met.side_effect = Exception("KPI check error")
            
            # The exception should be raised when calling async_act
            with pytest.raises(Exception, match="KPI check error"):
                list(check_staking_kpi_behaviour.async_act())

    def test_prepare_vanity_tx_logging(self):
        """Test that _prepare_vanity_tx logs correctly."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        mock_safe_tx_hash = "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
        
        # Mock contract_interact to return the safe_tx_hash
        def mock_contract_interact(**kwargs):
            yield
            return mock_safe_tx_hash
        
        with patch.object(check_staking_kpi_behaviour, 'contract_interact', side_effect=mock_contract_interact):
            with patch('packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.hash_payload_to_hex') as mock_hash_payload:
                mock_hash_payload.return_value = "0xFinalHash"
                
                # Mock logger
                mock_logger = MagicMock()
                check_staking_kpi_behaviour.context.logger = mock_logger
                
                generator = check_staking_kpi_behaviour._prepare_vanity_tx("optimism")
                result = self._consume_generator(generator)
                
                # Verify logging calls
                mock_logger.info.assert_any_call("Preparing vanity transaction for chain: optimism")
                mock_logger.debug.assert_any_call("Safe address for chain optimism: 0xSafeAddress123456789012345678901234567890123456")
                mock_logger.debug.assert_any_call("Transaction data: b'0x'")
                mock_logger.debug.assert_any_call("Safe transaction hash: 0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234")
                mock_logger.info.assert_any_call("Vanity transaction hash: 0xFinalHash")
                
                # Should return the final hash
                assert result == "0xFinalHash"

    def test_prepare_vanity_tx_logging_errors(self):
        """Test that _prepare_vanity_tx logs errors correctly."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        # Mock logger
        mock_logger = MagicMock()
        check_staking_kpi_behaviour.context.logger = mock_logger
        
        with patch.object(check_staking_kpi_behaviour, 'contract_interact') as mock_contract_interact:
            mock_contract_interact.side_effect = Exception("Contract interaction failed")
            
            generator = check_staking_kpi_behaviour._prepare_vanity_tx("optimism")
            result = self._consume_generator(generator)
            
            # Verify error logging
            mock_logger.error.assert_called_with("Exception during contract interaction: Contract interaction failed")
            
            # Should return None
            assert result is None

    def test_prepare_vanity_tx_logging_none_hash(self):
        """Test that _prepare_vanity_tx logs when safe_tx_hash is None."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        # Mock logger
        mock_logger = MagicMock()
        check_staking_kpi_behaviour.context.logger = mock_logger
        
        with patch.object(check_staking_kpi_behaviour, 'contract_interact') as mock_contract_interact:
            mock_contract_interact.return_value = iter([None])
            
            generator = check_staking_kpi_behaviour._prepare_vanity_tx("optimism")
            result = self._consume_generator(generator)
            
            # Verify error logging
            mock_logger.error.assert_called_with("Error preparing vanity tx: safe_tx_hash is None")
            
            # Should return None
            assert result is None

    def test_prepare_vanity_tx_logging_hash_payload_error(self):
        """Test that _prepare_vanity_tx logs when hash_payload_to_hex fails."""
        check_staking_kpi_behaviour = self._create_check_staking_kpi_behaviour()
        
        mock_safe_tx_hash = "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
        
        # Mock logger
        mock_logger = MagicMock()
        check_staking_kpi_behaviour.context.logger = mock_logger
        
        # Mock contract_interact to return the safe_tx_hash
        def mock_contract_interact(**kwargs):
            yield
            return mock_safe_tx_hash
        
        with patch.object(check_staking_kpi_behaviour, 'contract_interact', side_effect=mock_contract_interact):
            with patch('packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.hash_payload_to_hex') as mock_hash_payload:
                mock_hash_payload.side_effect = Exception("Hash payload conversion failed")
                
                generator = check_staking_kpi_behaviour._prepare_vanity_tx("optimism")
                result = self._consume_generator(generator)
                
                # Verify error logging
                mock_logger.error.assert_called_with("Exception during hash payload conversion: Hash payload conversion failed")
                
                # Should return None
                assert result is None 
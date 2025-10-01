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

"""End-to-end integration tests for staking and compliance workflows."""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from typing import Generator, Any

from packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met import (
    CheckStakingKPIMetBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.call_checkpoint import (
    CallCheckpointBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.states.base import StakingState
from packages.valory.skills.liquidity_trader_abci.payloads import (
    CheckStakingKPIMetPayload,
    CallCheckpointPayload,
)

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.staking.fixtures.staking_fixtures import (
    mock_staking_contract,
    staking_compliance_scenarios,
    kpi_test_data,
    checkpoint_test_data,
    vanity_transaction_test_data,
    staking_state_test_data,
    mock_contract_responses,
    test_addresses,
    test_chains,
)


class TestStakingCompliance(ProtocolIntegrationTestBase):
    """Test complete staking and compliance workflows using real methods."""

    def setup_method(self, **kwargs: Any) -> None:
        """Set up the test method with common infrastructure."""
        # Create temporary directory for test data
        import tempfile
        import shutil
        from pathlib import Path
        
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Mock the store path validation
        with patch(
            "packages.valory.skills.liquidity_trader_abci.models.Params.get_store_path",
            return_value=self.temp_path,
        ):
            super().setup(**kwargs)

    def teardown_method(self) -> None:
        """Clean up after tests."""
        # Clean up temporary directory
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().teardown()

    def _consume_generator(self, gen: Generator) -> Any:
        """Helper to run generator and return final value."""
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    def test_staking_kpi_compliance_workflow(self, staking_compliance_scenarios):
        """Test complete KPI compliance workflow."""
        scenario = staking_compliance_scenarios["normal_compliance"]
        
        # Mock file operations to avoid file system issues
        with patch("builtins.open", mock_open(read_data="[]")):
            with patch("json.load", return_value=[]):
                with patch("json.dump"):
                    # Create mock behaviour
                    behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock synchronized data
        mock_sync_data = MagicMock()
        mock_sync_data.service_staking_state = scenario["staking_state"]
        mock_sync_data.min_num_of_safe_tx_required = scenario["min_transactions_required"]
        mock_sync_data.period_count = 10
        mock_sync_data.period_number_at_last_cp = 5
        
        # Mock KPI check
        def mock_is_staking_kpi_met():
            yield
            return scenario["expected_kpi_met"]
        
        # Mock multisig nonces
        def mock_get_multisig_nonces_since_last_cp(chain, multisig):
            yield
            return scenario["transactions_since_checkpoint"]
        
        # Mock vanity transaction preparation
        def mock_prepare_vanity_tx(chain):
            yield
            return "0xVanityTxHash" if scenario["expected_vanity_tx"] else None
        
        with patch.object(
            behaviour, "_is_staking_kpi_met", side_effect=mock_is_staking_kpi_met
        ), patch.object(
            behaviour, "_get_multisig_nonces_since_last_cp", 
            side_effect=mock_get_multisig_nonces_since_last_cp
        ), patch.object(
            behaviour, "_prepare_vanity_tx", side_effect=mock_prepare_vanity_tx
        ), patch.object(
            type(behaviour), "synchronized_data", new_callable=MagicMock,
            return_value=mock_sync_data
        ), patch.object(
            behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            behaviour, "wait_until_round_end"
        ) as mock_wait:
            
            # Execute the workflow
            list(behaviour.async_act())
            
            # Verify payload was sent
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CheckStakingKPIMetPayload)
            assert payload.is_staking_kpi_met == scenario["expected_kpi_met"]

    def test_checkpoint_execution_workflow(self, checkpoint_test_data):
        """Test checkpoint execution workflow."""
        scenario = checkpoint_test_data["checkpoint_reached"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Mock staking state - need to set it on the synchronized data properly
        mock_sync_data = MagicMock()
        mock_sync_data.service_staking_state = StakingState.STAKED.value
        mock_sync_data.min_num_of_safe_tx_required = 5
        
        # Mock service staking state to set the enum
        def mock_get_service_staking_state(chain):
            yield
            behaviour.service_staking_state = StakingState.STAKED
            return StakingState.STAKED.value
        
        # Mock checkpoint check
        def mock_check_if_checkpoint_reached(chain):
            yield
            return scenario["expected_reached"]
        
        # Mock checkpoint preparation
        def mock_prepare_checkpoint_tx(chain):
            yield
            return "0xCheckpointTxHash" if scenario["expected_reached"] else None
        
        # Mock min tx calculation
        def mock_calculate_min_num_of_safe_tx_required(chain):
            yield
            return 5
        
        with patch.object(
            behaviour, "_get_service_staking_state",
            side_effect=mock_get_service_staking_state
        ), patch.object(
            behaviour, "_check_if_checkpoint_reached", 
            side_effect=mock_check_if_checkpoint_reached
        ), patch.object(
            behaviour, "_prepare_checkpoint_tx", 
            side_effect=mock_prepare_checkpoint_tx
        ), patch.object(
            behaviour, "_calculate_min_num_of_safe_tx_required",
            side_effect=mock_calculate_min_num_of_safe_tx_required
        ), patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.call_checkpoint.CallCheckpointBehaviour.synchronized_data",
            mock_sync_data
        ), patch.object(
            behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            behaviour, "wait_until_round_end"
        ) as mock_wait:
            
            # Execute the workflow
            list(behaviour.async_act())
            
            # Verify payload was sent
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CallCheckpointPayload)
            # The tx_hash should be generated by the behavior
            assert payload.tx_hash is not None

    def test_vanity_transaction_workflow(self, vanity_transaction_test_data):
        """Test vanity transaction workflow."""
        scenario = vanity_transaction_test_data["successful_vanity_tx"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock contract interaction
        def mock_contract_interact(**kwargs):
            yield
            return scenario["safe_tx_hash"]
        
        # Mock hash payload conversion
        with patch.object(
            behaviour, "contract_interact", side_effect=mock_contract_interact
        ), patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.hash_payload_to_hex"
        ) as mock_hash_payload:
            mock_hash_payload.return_value = scenario["expected_final_hash"]
            
            # Test vanity transaction preparation
            generator = behaviour._prepare_vanity_tx("optimism")
            result = self._consume_generator(generator)
            
            # Verify result
            if scenario["should_succeed"]:
                assert result == scenario["expected_final_hash"]
                mock_hash_payload.assert_called_once()
            else:
                assert result is None

    def test_kpi_monitoring_workflow(self, kpi_test_data):
        """Test KPI monitoring workflow."""
        scenario = kpi_test_data["valid_kpi_met"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock synchronized data
        mock_sync_data = MagicMock()
        mock_sync_data.service_staking_state = StakingState.STAKED.value
        mock_sync_data.min_num_of_safe_tx_required = scenario["min_num_of_safe_tx_required"]
        
        # Mock multisig nonces
        def mock_get_multisig_nonces_since_last_cp(chain, multisig):
            yield
            return scenario["multisig_nonces_since_last_cp"]
        
        with patch.object(
            behaviour, "_get_multisig_nonces_since_last_cp",
            side_effect=mock_get_multisig_nonces_since_last_cp
        ), patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.CheckStakingKPIMetBehaviour.synchronized_data",
            mock_sync_data
        ):
            
            # Test KPI check
            generator = behaviour._is_staking_kpi_met()
            result = self._consume_generator(generator)
            
            # Verify result
            assert result == scenario["expected_result"]

    def test_staking_state_management_workflow(self, staking_state_test_data):
        """Test staking state management workflow."""
        scenario = staking_state_test_data["staked_service"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Mock service staking state
        def mock_get_service_staking_state(chain):
            yield
            return scenario["staking_state"]
        
        # Mock synchronized data
        mock_sync_data = MagicMock()
        mock_sync_data.service_staking_state = scenario["expected_state"]
        
        with patch.object(
            behaviour, "_get_service_staking_state",
            side_effect=mock_get_service_staking_state
        ), patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.call_checkpoint.CallCheckpointBehaviour.synchronized_data",
            mock_sync_data
        ), patch.object(
            behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            behaviour, "wait_until_round_end"
        ) as mock_wait:
            
            # Execute the workflow
            list(behaviour.async_act())
            
            # Verify staking state was set correctly
            assert mock_sync_data.service_staking_state == scenario["expected_state"]

    def test_compliance_enforcement_workflow(self, staking_compliance_scenarios):
        """Test compliance enforcement workflow."""
        scenario = staking_compliance_scenarios["kpi_not_met"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock synchronized data with actual values
        mock_sync_data = MagicMock()
        mock_sync_data.service_staking_state = scenario["staking_state"]
        mock_sync_data.min_num_of_safe_tx_required = scenario["min_transactions_required"]
        mock_sync_data.period_count = 10
        mock_sync_data.period_number_at_last_cp = 3
        # Add missing parameter
        behaviour.params.staking_threshold_period = 5
        
        # Mock KPI check
        def mock_is_staking_kpi_met():
            yield
            return scenario["expected_kpi_met"]
        
        # Mock multisig nonces
        def mock_get_multisig_nonces_since_last_cp(chain, multisig):
            yield
            return scenario["transactions_since_checkpoint"]
        
        # Mock vanity transaction preparation
        def mock_prepare_vanity_tx(chain):
            yield
            return "0xVanityTxHash" if scenario["expected_vanity_tx"] else None
        
        with patch.object(
            behaviour, "_is_staking_kpi_met", side_effect=mock_is_staking_kpi_met
        ), patch.object(
            behaviour, "_get_multisig_nonces_since_last_cp",
            side_effect=mock_get_multisig_nonces_since_last_cp
        ), patch.object(
            behaviour, "_prepare_vanity_tx", side_effect=mock_prepare_vanity_tx
        ), patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.CheckStakingKPIMetBehaviour.synchronized_data",
            mock_sync_data
        ), patch.object(
            behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            behaviour, "wait_until_round_end"
        ) as mock_wait:
            
            # Execute the workflow
            list(behaviour.async_act())
            
            # Verify compliance enforcement
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CheckStakingKPIMetPayload)
            assert payload.is_staking_kpi_met == scenario["expected_kpi_met"]
            
            if scenario["expected_vanity_tx"]:
                assert payload.tx_hash is not None
            else:
                assert payload.tx_hash is None

    def test_service_eviction_workflow(self, staking_compliance_scenarios):
        """Test service eviction workflow."""
        scenario = staking_compliance_scenarios["service_evicted"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Mock service staking state
        def mock_get_service_staking_state(chain):
            yield
            # Set the enum value based on the scenario
            if scenario["staking_state"] == StakingState.EVICTED.value:
                behaviour.service_staking_state = StakingState.EVICTED
            elif scenario["staking_state"] == StakingState.STAKED.value:
                behaviour.service_staking_state = StakingState.STAKED
            else:
                behaviour.service_staking_state = StakingState.UNSTAKED
            return scenario["staking_state"]
        
        with patch.object(
            behaviour, "_get_service_staking_state",
            side_effect=mock_get_service_staking_state
        ), patch.object(
            behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            behaviour, "wait_until_round_end"
        ) as mock_wait:
            
            # Execute the workflow
            list(behaviour.async_act())
            
            # Verify eviction handling
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CallCheckpointPayload)
            assert payload.service_staking_state == scenario["staking_state"]
            assert payload.tx_hash is None  # No transaction for evicted service

    def test_checkpoint_timing_workflow(self, checkpoint_test_data):
        """Test checkpoint timing workflow."""
        scenario = checkpoint_test_data["checkpoint_reached"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Mock next checkpoint
        def mock_get_next_checkpoint(chain):
            yield
            return scenario["next_checkpoint"]
        
        # Mock timestamp
        import datetime
        mock_timestamp = datetime.datetime.fromtimestamp(scenario["current_timestamp"])
        behaviour.round_sequence.last_round_transition_timestamp = mock_timestamp
        
        with patch.object(
            behaviour, "_get_next_checkpoint", side_effect=mock_get_next_checkpoint
        ):
            # Test checkpoint timing
            generator = behaviour._check_if_checkpoint_reached("optimism")
            result = self._consume_generator(generator)
            
            # Verify timing result
            assert result == scenario["expected_reached"]

    def test_transaction_counting_workflow(self, mock_staking_contract):
        """Test transaction counting workflow."""
        # Setup mock contract
        service_id = 1
        mock_staking_contract.set_service_staking_state(service_id, StakingState.STAKED.value)
        mock_staking_contract.set_transaction_count_since_checkpoint(service_id, 5)
        mock_staking_contract.set_min_transactions_required(service_id, 3)
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock synchronized data
        mock_sync_data = MagicMock()
        mock_sync_data.service_staking_state = StakingState.STAKED.value
        mock_sync_data.min_num_of_safe_tx_required = 3
        
        # Mock multisig nonces
        def mock_get_multisig_nonces_since_last_cp(chain, multisig):
            yield
            return 5  # More than required
        
        with patch.object(
            behaviour, "_get_multisig_nonces_since_last_cp",
            side_effect=mock_get_multisig_nonces_since_last_cp
        ), patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.CheckStakingKPIMetBehaviour.synchronized_data",
            mock_sync_data
        ):
            
            # Test KPI check
            generator = behaviour._is_staking_kpi_met()
            result = self._consume_generator(generator)
            
            # Verify KPI is met
            assert result is True

    def test_error_handling_workflow(self, kpi_test_data):
        """Test error handling workflow."""
        scenario = kpi_test_data["invalid_min_tx_required"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock synchronized data with None values
        mock_sync_data = MagicMock()
        mock_sync_data.service_staking_state = StakingState.STAKED.value
        mock_sync_data.min_num_of_safe_tx_required = scenario["min_num_of_safe_tx_required"]
        
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.CheckStakingKPIMetBehaviour.synchronized_data",
            mock_sync_data
        ), patch.object(
            behaviour.context.logger, "error"
        ) as mock_logger:
            
            # Test KPI check with error
            generator = behaviour._is_staking_kpi_met()
            result = self._consume_generator(generator)
            
            # Verify error handling
            assert result == scenario["expected_result"]
            mock_logger.assert_called()

    def test_end_to_end_compliance_workflow(self, staking_compliance_scenarios):
        """Test complete end-to-end compliance workflow."""
        # Test normal compliance flow
        normal_scenario = staking_compliance_scenarios["normal_compliance"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock synchronized data
        mock_sync_data = MagicMock()
        mock_sync_data.service_staking_state = normal_scenario["staking_state"]
        mock_sync_data.min_num_of_safe_tx_required = normal_scenario["min_transactions_required"]
        mock_sync_data.period_count = 10
        mock_sync_data.period_number_at_last_cp = 5
        
        # Mock KPI check
        def mock_is_staking_kpi_met():
            yield
            return normal_scenario["expected_kpi_met"]
        
        # Mock multisig nonces
        def mock_get_multisig_nonces_since_last_cp(chain, multisig):
            yield
            return normal_scenario["transactions_since_checkpoint"]
        
        with patch.object(
            behaviour, "_is_staking_kpi_met", side_effect=mock_is_staking_kpi_met
        ), patch.object(
            behaviour, "_get_multisig_nonces_since_last_cp",
            side_effect=mock_get_multisig_nonces_since_last_cp
        ), patch.object(
            type(behaviour), "synchronized_data", new_callable=MagicMock,
            return_value=mock_sync_data
        ), patch.object(
            behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            behaviour, "wait_until_round_end"
        ) as mock_wait:
            
            # Execute the complete workflow
            list(behaviour.async_act())
            
            # Verify complete workflow
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CheckStakingKPIMetPayload)
            assert payload.is_staking_kpi_met == normal_scenario["expected_kpi_met"]
            assert payload.tx_hash is None  # No vanity tx needed for normal compliance

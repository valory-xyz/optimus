# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
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

"""End-to-end integration tests for staking compliance workflows."""

import pytest
import datetime
from unittest.mock import MagicMock, patch
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
    checkpoint_test_data,
    vanity_transaction_test_data,
    kpi_test_data,
    test_addresses,
    test_chains,
)


class TestStakingIntegration(ProtocolIntegrationTestBase):
    """Test end-to-end staking compliance workflows."""

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

    def test_complete_staking_compliance_workflow(self, staking_compliance_scenarios):
        """Test complete staking compliance workflow."""
        # Test normal compliance flow
        scenario = staking_compliance_scenarios["normal_compliance"]
        
        # Create mock behaviours
        kpi_behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        checkpoint_behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Mock synchronized data for KPI check
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
        
        # Test KPI compliance workflow
        with patch.object(
            kpi_behaviour, "_is_staking_kpi_met", side_effect=mock_is_staking_kpi_met
        ), patch.object(
            kpi_behaviour, "_get_multisig_nonces_since_last_cp",
            side_effect=mock_get_multisig_nonces_since_last_cp
        ), patch.object(
            type(kpi_behaviour), "synchronized_data", new_callable=MagicMock,
            return_value=mock_sync_data
        ), patch.object(
            kpi_behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            kpi_behaviour, "wait_until_round_end"
        ) as mock_wait:
            
            # Execute KPI compliance workflow
            list(kpi_behaviour.async_act())
            
            # Verify KPI compliance
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CheckStakingKPIMetPayload)
            assert payload.is_staking_kpi_met == scenario["expected_kpi_met"]

    def test_complete_checkpoint_workflow(self, checkpoint_test_data):
        """Test complete checkpoint workflow."""
        scenario = checkpoint_test_data["checkpoint_reached"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Mock service staking state
        def mock_get_service_staking_state(chain):
            yield
            behaviour.service_staking_state = StakingState.STAKED
        
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
        
        # Test checkpoint workflow
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
        ), patch.object(
            behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            behaviour, "wait_until_round_end"
        ) as mock_wait:
            
            # Execute checkpoint workflow
            list(behaviour.async_act())
            
            # Verify checkpoint workflow
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CallCheckpointPayload)
            if scenario["expected_reached"]:
                assert payload.tx_hash is not None
            else:
                assert payload.tx_hash is None

    def test_complete_vanity_transaction_workflow(self, vanity_transaction_test_data):
        """Test complete vanity transaction workflow."""
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
            
            # Test vanity transaction workflow
            generator = behaviour._prepare_vanity_tx("optimism")
            result = self._consume_generator(generator)
            
            # Verify vanity transaction workflow
            assert result == scenario["expected_final_hash"]
            mock_hash_payload.assert_called_once()

    def test_complete_kpi_monitoring_workflow(self, kpi_test_data):
        """Test complete KPI monitoring workflow."""
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
        
        # Test KPI monitoring workflow
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.CheckStakingKPIMetBehaviour.synchronized_data",
            mock_sync_data
        ), patch.object(
            behaviour, "_get_multisig_nonces_since_last_cp",
            side_effect=mock_get_multisig_nonces_since_last_cp
        ):
            
            # Execute KPI monitoring workflow
            generator = behaviour._is_staking_kpi_met()
            result = self._consume_generator(generator)
            
            # Verify KPI monitoring workflow
            assert result == scenario["expected_result"]

    def test_complete_compliance_enforcement_workflow(self, staking_compliance_scenarios):
        """Test complete compliance enforcement workflow."""
        # Test KPI not met scenario
        scenario = staking_compliance_scenarios["kpi_not_met"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock synchronized data
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
        
        # Test compliance enforcement workflow
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
            
            # Execute compliance enforcement workflow
            list(behaviour.async_act())
            
            # Verify compliance enforcement workflow
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CheckStakingKPIMetPayload)
            assert payload.is_staking_kpi_met == scenario["expected_kpi_met"]
            
            if scenario["expected_vanity_tx"]:
                assert payload.tx_hash == "0xVanityTxHash"
            else:
                assert payload.tx_hash is None

    def test_complete_service_eviction_workflow(self, staking_compliance_scenarios):
        """Test complete service eviction workflow."""
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
        
        # Test service eviction workflow
        with patch.object(
            behaviour, "_get_service_staking_state",
            side_effect=mock_get_service_staking_state
        ), patch.object(
            behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            behaviour, "wait_until_round_end"
        ) as mock_wait:
            
            # Execute service eviction workflow
            list(behaviour.async_act())
            
            # Verify service eviction workflow
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CallCheckpointPayload)
            assert payload.service_staking_state == scenario["staking_state"]
            assert payload.tx_hash is None  # No transaction for evicted service

    def test_complete_error_handling_workflow(self, kpi_test_data):
        """Test complete error handling workflow."""
        scenario = kpi_test_data["invalid_min_tx_required"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock synchronized data with None values
        mock_sync_data = MagicMock()
        mock_sync_data.service_staking_state = StakingState.STAKED.value
        mock_sync_data.min_num_of_safe_tx_required = scenario["min_num_of_safe_tx_required"]
        
        # Test error handling workflow
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.CheckStakingKPIMetBehaviour.synchronized_data",
            mock_sync_data
        ), patch.object(
            behaviour.context.logger, "error"
        ) as mock_logger:
            
            # Execute error handling workflow
            generator = behaviour._is_staking_kpi_met()
            result = self._consume_generator(generator)
            
            # Verify error handling workflow
            assert result == scenario["expected_result"]
            mock_logger.assert_called()

    def test_complete_chain_handling_workflow(self, test_chains):
        """Test complete chain handling workflow."""
        # Create mock behaviours
        kpi_behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        checkpoint_behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        for chain_name, chain_config in test_chains.items():
            # Set up parameters for both behaviours
            for behaviour in [kpi_behaviour, checkpoint_behaviour]:
                behaviour.params.__dict__["_frozen"] = False
                behaviour.params.staking_chain = chain_name
                behaviour.params.safe_contract_addresses = {
                    chain_name: chain_config["safe_address"]
                }
                if hasattr(behaviour.params, 'staking_token_contract_address'):
                    behaviour.params.staking_token_contract_address = chain_config["staking_token_address"]
                behaviour.params.__dict__["_frozen"] = True
            
            # Test KPI compliance workflow
            mock_sync_data = MagicMock()
            mock_sync_data.service_staking_state = StakingState.STAKED.value
            mock_sync_data.min_num_of_safe_tx_required = 5
            
            def mock_get_multisig_nonces_since_last_cp(chain, multisig):
                yield
                return 8  # More than required
            
            with patch(
                "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.CheckStakingKPIMetBehaviour.synchronized_data",
                mock_sync_data
            ), patch.object(
                kpi_behaviour, "_get_multisig_nonces_since_last_cp",
                side_effect=mock_get_multisig_nonces_since_last_cp
            ):
                
                # Test KPI check
                generator = kpi_behaviour._is_staking_kpi_met()
                result = self._consume_generator(generator)
                
                # Verify chain handling
                assert result is True

    def test_complete_timing_workflow(self, checkpoint_test_data):
        """Test complete timing workflow."""
        scenario = checkpoint_test_data["checkpoint_reached"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Mock next checkpoint
        def mock_get_next_checkpoint(chain):
            yield
            return scenario["next_checkpoint"]
        
        # Mock timestamp
        mock_timestamp = datetime.datetime.fromtimestamp(scenario["current_timestamp"])
        behaviour.round_sequence.last_round_transition_timestamp = mock_timestamp
        
        # Test timing workflow
        with patch.object(
            behaviour, "_get_next_checkpoint", side_effect=mock_get_next_checkpoint
        ):
            # Execute timing workflow
            generator = behaviour._check_if_checkpoint_reached("optimism")
            result = self._consume_generator(generator)
            
            # Verify timing workflow
            assert result == scenario["expected_reached"]

    def test_complete_transaction_counting_workflow(self, mock_staking_contract):
        """Test complete transaction counting workflow."""
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
        
        # Test transaction counting workflow
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.CheckStakingKPIMetBehaviour.synchronized_data",
            mock_sync_data
        ), patch.object(
            behaviour, "_get_multisig_nonces_since_last_cp",
            side_effect=mock_get_multisig_nonces_since_last_cp
        ):
            
            # Execute transaction counting workflow
            generator = behaviour._is_staking_kpi_met()
            result = self._consume_generator(generator)
            
            # Verify transaction counting workflow
            assert result is True

    def test_complete_end_to_end_workflow(self, staking_compliance_scenarios):
        """Test complete end-to-end workflow."""
        # Test normal compliance flow
        scenario = staking_compliance_scenarios["normal_compliance"]
        
        # Create mock behaviours
        kpi_behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        checkpoint_behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Mock synchronized data for KPI check
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
        
        # Test complete end-to-end workflow
        with patch.object(
            kpi_behaviour, "_is_staking_kpi_met", side_effect=mock_is_staking_kpi_met
        ), patch.object(
            kpi_behaviour, "_get_multisig_nonces_since_last_cp",
            side_effect=mock_get_multisig_nonces_since_last_cp
        ), patch.object(
            type(kpi_behaviour), "synchronized_data", new_callable=MagicMock,
            return_value=mock_sync_data
        ), patch.object(
            kpi_behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            kpi_behaviour, "wait_until_round_end"
        ) as mock_wait:
            
            # Execute complete end-to-end workflow
            list(kpi_behaviour.async_act())
            
            # Verify complete end-to-end workflow
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CheckStakingKPIMetPayload)
            assert payload.is_staking_kpi_met == scenario["expected_kpi_met"]
            assert payload.tx_hash is None  # No vanity tx needed for normal compliance

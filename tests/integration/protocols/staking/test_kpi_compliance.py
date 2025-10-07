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

"""Tests for KPI compliance monitoring and enforcement."""

import pytest
from unittest.mock import MagicMock, patch
from typing import Generator, Any

from packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met import (
    CheckStakingKPIMetBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.states.base import StakingState
from packages.valory.skills.liquidity_trader_abci.payloads import CheckStakingKPIMetPayload

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.staking.fixtures.staking_fixtures import (
    kpi_test_data,
    staking_compliance_scenarios,
    mock_contract_responses,
    test_addresses,
    test_chains,
)


class TestKPICompliance(ProtocolIntegrationTestBase):
    """Test KPI compliance monitoring and enforcement."""

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

    def test_kpi_requirement_calculation(self, kpi_test_data):
        """Test KPI requirement calculation."""
        for scenario_name, scenario in kpi_test_data.items():
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
            
            with patch(
                "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.CheckStakingKPIMetBehaviour.synchronized_data",
                mock_sync_data
            ), patch.object(
                behaviour, "_get_multisig_nonces_since_last_cp",
                side_effect=mock_get_multisig_nonces_since_last_cp
            ):
                
                # Test KPI check
                generator = behaviour._is_staking_kpi_met()
                result = self._consume_generator(generator)
                
                # Verify KPI calculation
                assert result == scenario["expected_result"], f"Failed for scenario: {scenario_name}"

    def test_transaction_counting_accuracy(self, mock_contract_responses):
        """Test transaction counting accuracy."""
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock synchronized data
        mock_sync_data = MagicMock()
        mock_sync_data.service_staking_state = StakingState.STAKED.value
        mock_sync_data.min_num_of_safe_tx_required = 5
        
        # Mock multisig nonces
        def mock_get_multisig_nonces_since_last_cp(chain, multisig):
            yield
            return 8  # More than required
        
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.CheckStakingKPIMetBehaviour.synchronized_data",
            mock_sync_data
        ), patch.object(
            behaviour, "_get_multisig_nonces_since_last_cp",
            side_effect=mock_get_multisig_nonces_since_last_cp
        ):
            
            # Test KPI check
            generator = behaviour._is_staking_kpi_met()
            result = self._consume_generator(generator)
            
            # Verify transaction counting
            assert result is True

    def test_kpi_threshold_evaluation(self, kpi_test_data):
        """Test KPI threshold evaluation."""
        # Test cases for threshold evaluation
        threshold_cases = [
            {"min_required": 5, "actual_count": 8, "expected": True, "description": "Above threshold"},
            {"min_required": 10, "actual_count": 3, "expected": False, "description": "Below threshold"},
            {"min_required": 5, "actual_count": 5, "expected": True, "description": "At threshold"},
        ]
        
        for case in threshold_cases:
            # Create mock behaviour
            behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
            
            # Mock synchronized data
            mock_sync_data = MagicMock()
            mock_sync_data.service_staking_state = StakingState.STAKED.value
            mock_sync_data.min_num_of_safe_tx_required = case["min_required"]
            
            # Mock multisig nonces
            def mock_get_multisig_nonces_since_last_cp(chain, multisig):
                yield
                return case["actual_count"]
            
            with patch(
                "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.CheckStakingKPIMetBehaviour.synchronized_data",
                mock_sync_data
            ), patch.object(
                behaviour, "_get_multisig_nonces_since_last_cp",
                side_effect=mock_get_multisig_nonces_since_last_cp
            ):
                
                # Test KPI check
                generator = behaviour._is_staking_kpi_met()
                result = self._consume_generator(generator)
                
                # Verify threshold evaluation
                assert result == case["expected"], f"Failed for case: {case['description']}"

    def test_compliance_enforcement_triggers(self, staking_compliance_scenarios):
        """Test compliance enforcement triggers."""
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

    def test_service_eviction_scenarios(self, staking_compliance_scenarios):
        """Test service eviction scenarios."""
        scenario = staking_compliance_scenarios["service_evicted"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock synchronized data
        mock_sync_data = MagicMock()
        mock_sync_data.service_staking_state = scenario["staking_state"]
        mock_sync_data.min_num_of_safe_tx_required = scenario["min_transactions_required"]
        
        # Mock KPI check
        def mock_is_staking_kpi_met():
            yield
            return scenario["expected_kpi_met"]
        
        # Mock multisig nonces
        def mock_get_multisig_nonces_since_last_cp(chain, multisig):
            yield
            return scenario["transactions_since_checkpoint"]
        
        with patch.object(
            behaviour, "_is_staking_kpi_met", side_effect=mock_is_staking_kpi_met
        ), patch.object(
            behaviour, "_get_multisig_nonces_since_last_cp",
            side_effect=mock_get_multisig_nonces_since_last_cp
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
            
            # Verify eviction handling
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CheckStakingKPIMetPayload)
            assert payload.is_staking_kpi_met == scenario["expected_kpi_met"]
            assert payload.tx_hash is None  # No vanity tx for evicted service

    def test_kpi_monitoring_error_handling(self, kpi_test_data):
        """Test KPI monitoring error handling."""
        # Test invalid min_tx_required scenario
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

    def test_kpi_monitoring_nonces_error_handling(self, kpi_test_data):
        """Test KPI monitoring nonces error handling."""
        # Test invalid nonces scenario
        scenario = kpi_test_data["invalid_nonces"]
        
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
        
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.CheckStakingKPIMetBehaviour.synchronized_data",
            mock_sync_data
        ), patch.object(
            behaviour, "_get_multisig_nonces_since_last_cp",
            side_effect=mock_get_multisig_nonces_since_last_cp
        ):
            
            # Test KPI check with error
            generator = behaviour._is_staking_kpi_met()
            result = self._consume_generator(generator)
            
            # Verify error handling
            assert result == scenario["expected_result"]

    def test_kpi_monitoring_unstaked_service(self, staking_compliance_scenarios):
        """Test KPI monitoring for unstaked service."""
        scenario = staking_compliance_scenarios["service_unstaked"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock synchronized data
        mock_sync_data = MagicMock()
        mock_sync_data.service_staking_state = scenario["staking_state"]
        mock_sync_data.min_num_of_safe_tx_required = scenario["min_transactions_required"]
        
        with patch.object(
            type(behaviour), "synchronized_data", new_callable=MagicMock,
            return_value=mock_sync_data
        ):
            
            # Test KPI check for unstaked service
            generator = behaviour._is_staking_kpi_met()
            result = self._consume_generator(generator)
            
            # Verify unstaked service handling
            assert result == scenario["expected_kpi_met"]

    def test_kpi_monitoring_chain_handling(self, test_chains):
        """Test KPI monitoring chain handling."""
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        for chain_name, chain_config in test_chains.items():
            # Set up parameters
            behaviour.params.__dict__["_frozen"] = False
            behaviour.params.staking_chain = chain_name
            behaviour.params.safe_contract_addresses = {
                chain_name: chain_config["safe_address"]
            }
            behaviour.params.__dict__["_frozen"] = True
            
            # Mock synchronized data
            mock_sync_data = MagicMock()
            mock_sync_data.service_staking_state = StakingState.STAKED.value
            mock_sync_data.min_num_of_safe_tx_required = 5
            
            # Mock multisig nonces
            def mock_get_multisig_nonces_since_last_cp(chain, multisig):
                yield
                return 8  # More than required
            
            with patch(
                "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.CheckStakingKPIMetBehaviour.synchronized_data",
                mock_sync_data
            ), patch.object(
                behaviour, "_get_multisig_nonces_since_last_cp",
                side_effect=mock_get_multisig_nonces_since_last_cp
            ):
                
                # Test KPI check
                generator = behaviour._is_staking_kpi_met()
                result = self._consume_generator(generator)
                
                # Verify chain handling
                assert result is True

    def test_kpi_monitoring_period_threshold(self, staking_compliance_scenarios):
        """Test KPI monitoring period threshold."""
        scenario = staking_compliance_scenarios["threshold_exceeded"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock synchronized data
        mock_sync_data = MagicMock()
        mock_sync_data.service_staking_state = scenario["staking_state"]
        mock_sync_data.min_num_of_safe_tx_required = scenario["min_transactions_required"]
        mock_sync_data.period_count = scenario["period_count"]
        mock_sync_data.period_number_at_last_cp = scenario["period_number_at_last_cp"]
        # Add missing parameter
        behaviour.params.staking_threshold_period = scenario["staking_threshold_period"]
        
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
            
            # Verify period threshold handling
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CheckStakingKPIMetPayload)
            assert payload.is_staking_kpi_met == scenario["expected_kpi_met"]
            
            if scenario["expected_vanity_tx"]:
                assert payload.tx_hash == "0xVanityTxHash"
            else:
                assert payload.tx_hash is None

    def test_kpi_monitoring_comprehensive_workflow(self, staking_compliance_scenarios):
        """Test comprehensive KPI monitoring workflow."""
        # Test normal compliance scenario
        scenario = staking_compliance_scenarios["normal_compliance"]
        
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
        
        with patch.object(
            behaviour, "_is_staking_kpi_met", side_effect=mock_is_staking_kpi_met
        ), patch.object(
            behaviour, "_get_multisig_nonces_since_last_cp",
            side_effect=mock_get_multisig_nonces_since_last_cp
        ), patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.CheckStakingKPIMetBehaviour.synchronized_data",
            mock_sync_data
        ), patch.object(
            behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            behaviour, "wait_until_round_end"
        ) as mock_wait:
            
            # Execute the comprehensive workflow
            list(behaviour.async_act())
            
            # Verify comprehensive workflow
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CheckStakingKPIMetPayload)
            assert payload.is_staking_kpi_met == scenario["expected_kpi_met"]
            assert payload.tx_hash is None  # No vanity tx needed for normal compliance

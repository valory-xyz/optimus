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

"""Tests for checkpoint mechanisms and timing."""

import pytest
import datetime
from unittest.mock import MagicMock, patch
from typing import Generator, Any

from packages.valory.skills.liquidity_trader_abci.behaviours.call_checkpoint import (
    CallCheckpointBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.states.base import StakingState
from packages.valory.skills.liquidity_trader_abci.payloads import CallCheckpointPayload

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.staking.fixtures.staking_fixtures import (
    checkpoint_test_data,
    mock_contract_responses,
    test_addresses,
    test_chains,
)


class TestCheckpointMechanisms(ProtocolIntegrationTestBase):
    """Test checkpoint mechanisms and timing."""

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

    def test_checkpoint_timing_accuracy(self, checkpoint_test_data):
        """Test checkpoint timing accuracy."""
        for scenario_name, scenario in checkpoint_test_data.items():
            # Create mock behaviour
            behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
            
            # Mock next checkpoint
            def mock_get_next_checkpoint(chain):
                yield
                return scenario["next_checkpoint"]
            
            # Mock timestamp
            mock_timestamp = datetime.datetime.fromtimestamp(scenario["current_timestamp"])
            behaviour.round_sequence.last_round_transition_timestamp = mock_timestamp
            
            with patch.object(
                behaviour, "_get_next_checkpoint", side_effect=mock_get_next_checkpoint
            ):
                # Test checkpoint timing
                generator = behaviour._check_if_checkpoint_reached("optimism")
                result = self._consume_generator(generator)
                
                # Verify timing result
                assert result == scenario["expected_reached"], f"Failed for scenario: {scenario_name}"

    def test_checkpoint_transaction_encoding(self, mock_contract_responses):
        """Test checkpoint transaction encoding."""
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Mock contract interaction
        def mock_contract_interact(**kwargs):
            yield
            return mock_contract_responses["build_checkpoint_tx"]
        
        # Mock safe tx preparation
        def mock_prepare_safe_tx(chain, data):
            yield
            return "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
        
        with patch.object(
            behaviour, "contract_interact", side_effect=mock_contract_interact
        ), patch.object(
            behaviour, "_prepare_safe_tx", side_effect=mock_prepare_safe_tx
        ), patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.call_checkpoint.hash_payload_to_hex"
        ) as mock_hash_payload:
            mock_hash_payload.return_value = "0xFinalCheckpointHash"
            
            # Test checkpoint preparation
            generator = behaviour._prepare_checkpoint_tx("optimism")
            result = self._consume_generator(generator)
            
            # Verify transaction encoding
            assert result == "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"

    def test_checkpoint_state_validation(self, test_addresses):
        """Test checkpoint state validation."""
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Test different staking states
        test_states = [
            (StakingState.STAKED, True),
            (StakingState.UNSTAKED, False),
            (StakingState.EVICTED, False),
        ]
        
        for staking_state, should_prepare_tx in test_states:
            # Mock service staking state
            def mock_get_service_staking_state(chain):
                yield
                behaviour.service_staking_state = staking_state
            
            # Mock checkpoint reached
            def mock_check_if_checkpoint_reached(chain):
                yield
                return True  # Checkpoint reached
            
            # Mock checkpoint preparation
            def mock_prepare_checkpoint_tx(chain):
                yield
                return "0xCheckpointTxHash" if should_prepare_tx else None
            
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
            ), patch.object(
                behaviour, "send_a2a_transaction"
            ) as mock_send, patch.object(
                behaviour, "wait_until_round_end"
            ) as mock_wait:
                
                # Execute the workflow
                list(behaviour.async_act())
                
                # Verify state validation
                mock_send.assert_called_once()
                payload = mock_send.call_args[0][0]
                assert isinstance(payload, CallCheckpointPayload)
                assert payload.service_staking_state == staking_state.value
                
                if should_prepare_tx:
                    assert payload.tx_hash is not None
                else:
                    assert payload.tx_hash is None

    def test_checkpoint_failure_recovery(self, mock_contract_responses):
        """Test checkpoint failure recovery."""
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Mock service staking state
        def mock_get_service_staking_state(chain):
            yield
            behaviour.service_staking_state = StakingState.STAKED
        
        # Mock checkpoint reached
        def mock_check_if_checkpoint_reached(chain):
            yield
            return True
        
        # Mock checkpoint preparation failure
        def mock_prepare_checkpoint_tx(chain):
            yield
            return None  # Preparation failed
        
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
        ), patch.object(
            behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            behaviour, "wait_until_round_end"
        ) as mock_wait:
            
            # Execute the workflow
            list(behaviour.async_act())
            
            # Verify failure recovery
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CallCheckpointPayload)
            assert payload.tx_hash is None  # No transaction due to failure

    def test_checkpoint_timing_edge_cases(self):
        """Test checkpoint timing edge cases."""
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Test edge cases
        edge_cases = [
            {"next_checkpoint": None, "expected": False, "description": "No next checkpoint"},
            {"next_checkpoint": 0, "expected": True, "description": "Zero checkpoint"},
            {"next_checkpoint": -1, "expected": True, "description": "Negative checkpoint"},
        ]
        
        for case in edge_cases:
            # Mock next checkpoint
            def mock_get_next_checkpoint(chain):
                yield
                return case["next_checkpoint"]
            
            # Mock timestamp
            behaviour.round_sequence._last_round_transition_timestamp = datetime.datetime.fromtimestamp(1700000000)
            
            with patch.object(
                behaviour, "_get_next_checkpoint", side_effect=mock_get_next_checkpoint
            ):
                # Test checkpoint timing
                generator = behaviour._check_if_checkpoint_reached("optimism")
                result = self._consume_generator(generator)
                
                # Verify edge case handling
                assert result == case["expected"], f"Failed for case: {case['description']}"

    def test_checkpoint_chain_handling(self, test_chains):
        """Test checkpoint chain handling."""
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        for chain_name, chain_config in test_chains.items():
            # Set up parameters
            behaviour.params.__dict__["_frozen"] = False
            behaviour.params.staking_chain = chain_name
            behaviour.params.safe_contract_addresses = {
                chain_name: chain_config["safe_address"]
            }
            behaviour.params.staking_token_contract_address = chain_config["staking_token_address"]
            behaviour.params.__dict__["_frozen"] = True
            
            # Mock service staking state
            def mock_get_service_staking_state(chain):
                yield
                behaviour.service_staking_state = StakingState.STAKED
            
            # Mock checkpoint reached
            def mock_check_if_checkpoint_reached(chain):
                yield
                return True
            
            # Mock checkpoint preparation
            def mock_prepare_checkpoint_tx(chain):
                yield
                return "0xCheckpointTxHash"
            
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
            ), patch.object(
                behaviour, "send_a2a_transaction"
            ) as mock_send, patch.object(
                behaviour, "wait_until_round_end"
            ) as mock_wait:
                
                # Execute the workflow
                list(behaviour.async_act())
                
                # Verify chain handling
                mock_send.assert_called_once()
                payload = mock_send.call_args[0][0]
                assert isinstance(payload, CallCheckpointPayload)
                assert payload.chain_id == chain_name
                assert payload.safe_contract_address == chain_config["safe_address"]

    def test_checkpoint_contract_interaction(self, mock_contract_responses):
        """Test checkpoint contract interaction."""
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Mock contract interaction for checkpoint
        def mock_contract_interact_checkpoint(**kwargs):
            yield
            return mock_contract_responses["build_checkpoint_tx"]
        
        # Mock contract interaction for safe tx
        def mock_contract_interact_safe(**kwargs):
            yield
            return "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
        
        with patch.object(
            behaviour, "contract_interact", side_effect=mock_contract_interact_checkpoint
        ) as mock_contract_interact, patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.call_checkpoint.hash_payload_to_hex"
        ) as mock_hash_payload:
            mock_hash_payload.return_value = "0xFinalCheckpointHash"
            
            # Test checkpoint preparation
            generator = behaviour._prepare_checkpoint_tx("optimism")
            result = self._consume_generator(generator)
            
            # Verify contract interaction (called twice: once for checkpoint, once for safe tx hash)
            assert mock_contract_interact.call_count == 2
            assert result == "0xFinalCheckpointHash"

    def test_checkpoint_timing_boundary_conditions(self):
        """Test checkpoint timing boundary conditions."""
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Test boundary conditions
        boundary_cases = [
            {"next_checkpoint": 1700000000, "current_timestamp": 1700000000, "expected": True, "description": "Exact match"},
            {"next_checkpoint": 1700000001, "current_timestamp": 1700000000, "expected": False, "description": "1 second before"},
            {"next_checkpoint": 1699999999, "current_timestamp": 1700000000, "expected": True, "description": "1 second after"},
        ]
        
        for case in boundary_cases:
            # Mock next checkpoint
            def mock_get_next_checkpoint(chain):
                yield
                return case["next_checkpoint"]
            
            # Mock timestamp
            mock_timestamp = datetime.datetime.fromtimestamp(case["current_timestamp"])
            behaviour.round_sequence.last_round_transition_timestamp = mock_timestamp
            
            with patch.object(
                behaviour, "_get_next_checkpoint", side_effect=mock_get_next_checkpoint
            ):
                # Test checkpoint timing
                generator = behaviour._check_if_checkpoint_reached("optimism")
                result = self._consume_generator(generator)
                
                # Verify boundary condition handling
                assert result == case["expected"], f"Failed for boundary case: {case['description']}"

    def test_checkpoint_error_handling(self):
        """Test checkpoint error handling."""
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CallCheckpointBehaviour)
        
        # Mock service staking state
        def mock_get_service_staking_state(chain):
            yield
            behaviour.service_staking_state = StakingState.STAKED
        
        # Mock checkpoint reached
        def mock_check_if_checkpoint_reached(chain):
            yield
            return True
        
        # Mock checkpoint preparation with exception
        def mock_prepare_checkpoint_tx(chain):
            yield
            raise Exception("Checkpoint preparation failed")
        
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
        ), patch.object(
            behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            behaviour, "wait_until_round_end"
        ) as mock_wait:
            
            # Execute the workflow - should raise exception
            with pytest.raises(Exception, match="Checkpoint preparation failed"):
                list(behaviour.async_act())
            
            # Verify error handling - no transaction should be sent due to exception
            mock_send.assert_not_called()

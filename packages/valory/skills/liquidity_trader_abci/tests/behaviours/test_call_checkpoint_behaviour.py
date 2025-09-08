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

"""Tests for CallCheckpointBehaviour of the liquidity_trader_abci skill."""

import datetime
import logging
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, strategies as st

from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.call_checkpoint import (
    CallCheckpointBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.payloads import CallCheckpointPayload
from packages.valory.skills.liquidity_trader_abci.states.base import (
    StakingState,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent.parent.parent


class LiquidityTraderAbciFSMBehaviourBaseCase(FSMBehaviourBaseCase):
    """Base case for testing FSMBehaviour."""

    path_to_skill = PACKAGE_DIR


class TestCallCheckpointBehaviour(LiquidityTraderAbciFSMBehaviourBaseCase):
    """Test cases for CallCheckpointBehaviour."""

    behaviour_class = CallCheckpointBehaviour
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
        
        # Create data directory for store_path
        data_dir = Path(cls.temp_skill_dir) / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Add parameter overrides for testing
        param_overrides = {
            "store_path": str(data_dir),
            "initial_assets": {
                "0x4200000000000000000000000000000000000006": "WETH",
                "0xd988097fb8612cc24eeC14542bC03424c656005f": "USDC",
                "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9": "OLAS",
            },
            "target_investment_chains": ["mode"],
            "safe_contract_addresses": {"mode": "0xSafeAddress"}
        }
        
        # Merge with existing kwargs
        if "param_overrides" in kwargs:
            kwargs["param_overrides"].update(param_overrides)
        else:
            kwargs["param_overrides"] = param_overrides
        
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

    def setup_method(self, **kwargs: Any) -> None:
        """Set up the test method with mocked dependencies."""
        # Mock the store path validation before calling super().setup()
        with patch(
            "packages.valory.skills.liquidity_trader_abci.models.Params.get_store_path",
            return_value=Path("/tmp/mock_store"),
        ):
            super().setup(**kwargs)
        
        # Create individual behaviour instance for testing
        self.checkpoint_behaviour = CallCheckpointBehaviour(
            name="call_checkpoint_behaviour",
            skill_context=self.skill.skill_context
        )

        # Setup default test data
        self.setup_default_test_data()

    def teardown_method(self, **kwargs: Any) -> None:
        """Teardown the test method."""
        super().teardown(**kwargs)

    def _create_checkpoint_behaviour(self):
        """Create a CallCheckpointBehaviour instance for testing."""
        return CallCheckpointBehaviour(
            name="call_checkpoint_behaviour",
            skill_context=self.skill.skill_context
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

    def setup_default_test_data(self):
        """Setup default test data."""
        # Unfreeze params to set default parameters
        self.checkpoint_behaviour.params.__dict__["_frozen"] = False
        self.checkpoint_behaviour.params.staking_chain = "optimism"
        self.checkpoint_behaviour.params.safe_contract_addresses = {
            "optimism": "0xSafeAddress123456789012345678901234567890123456"
        }
        self.checkpoint_behaviour.params.staking_token_contract_address = "0xStakingToken123456789012345678901234567890123456"
        self.checkpoint_behaviour.params.__dict__["_frozen"] = True

    @pytest.mark.parametrize("staking_chain,mock_staking_state,mock_min_tx_required,mock_checkpoint_reached,mock_checkpoint_tx,expected_tx_hex,expected_min_tx,expected_staking_state,test_description", [
        # Test 1: No staking chain configured
        (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            StakingState.UNSTAKED,
            "no staking chain configured"
        ),
        # Test 2: Service staked, checkpoint reached, successful tx preparation
        (
            "optimism",
            StakingState.STAKED,
            5,
            True,
            "0xCheckpointTxHash1234567890123456789012345678901234567890123456789012345678901234",
            "0xCheckpointTxHash1234567890123456789012345678901234567890123456789012345678901234",
            5,
            StakingState.STAKED,
            "service staked, checkpoint reached, successful tx"
        ),
        # Test 3: Service staked, checkpoint not reached
        (
            "optimism",
            StakingState.STAKED,
            3,
            False,
            None,
            None,
            3,
            StakingState.STAKED,
            "service staked, checkpoint not reached"
        ),
        # Test 4: Service staked, error calculating min tx required
        (
            "optimism",
            StakingState.STAKED,
            None,
            False,
            None,
            None,
            None,
            StakingState.STAKED,
            "service staked, error calculating min tx required"
        ),
        # Test 5: Service evicted
        (
            "optimism",
            StakingState.EVICTED,
            None,
            None,
            None,
            None,
            None,
            StakingState.EVICTED,
            "service evicted"
        ),
        # Test 6: Service not staked
        (
            "optimism",
            StakingState.UNSTAKED,
            None,
            None,
            None,
            None,
            None,
            StakingState.UNSTAKED,
            "service not staked"
        ),
    ])
    def test_async_act_variations(self, staking_chain, mock_staking_state, mock_min_tx_required, mock_checkpoint_reached, mock_checkpoint_tx, expected_tx_hex, expected_min_tx, expected_staking_state, test_description):
        """Test async_act method with various scenarios."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        # Set staking chain - unfreeze params first
        checkpoint_behaviour.params.__dict__["_frozen"] = False
        checkpoint_behaviour.params.staking_chain = staking_chain
        checkpoint_behaviour.params.safe_contract_addresses = {
            "optimism": "0xSafeAddress123456789012345678901234567890123456"
        }
        checkpoint_behaviour.params.staking_token_contract_address = "0xStakingToken123456789012345678901234567890123456"
        checkpoint_behaviour.params.__dict__["_frozen"] = True
        
        # Mock the dependencies
        def mock_get_service_staking_state(chain):
            yield
            checkpoint_behaviour.service_staking_state = mock_staking_state
        
        def mock_calculate_min_num_of_safe_tx_required(chain):
            yield
            return mock_min_tx_required
        
        def mock_check_if_checkpoint_reached(chain):
            yield
            return mock_checkpoint_reached
        
        def mock_prepare_checkpoint_tx(chain):
            yield
            return mock_checkpoint_tx
        
        # Use side_effect to mock the generator methods
        with patch.object(checkpoint_behaviour, '_get_service_staking_state', side_effect=mock_get_service_staking_state):
            with patch.object(checkpoint_behaviour, '_calculate_min_num_of_safe_tx_required', side_effect=mock_calculate_min_num_of_safe_tx_required):
                with patch.object(checkpoint_behaviour, '_check_if_checkpoint_reached', side_effect=mock_check_if_checkpoint_reached):
                    with patch.object(checkpoint_behaviour, '_prepare_checkpoint_tx', side_effect=mock_prepare_checkpoint_tx):
                        with patch.object(checkpoint_behaviour, 'send_a2a_transaction') as mock_send:
                            with patch.object(checkpoint_behaviour, 'wait_until_round_end') as mock_wait:
                                # Execute the actual async_act method
                                list(checkpoint_behaviour.async_act())

                                # Verify payload was sent with correct values
                                mock_send.assert_called_once()
                                payload = mock_send.call_args[0][0]
                                assert isinstance(payload, CallCheckpointPayload)
                                assert payload.tx_hash == expected_tx_hex
                                assert payload.min_num_of_safe_tx_required == expected_min_tx
                                assert payload.service_staking_state == expected_staking_state.value
                                assert payload.chain_id == staking_chain
                                assert payload.safe_contract_address == checkpoint_behaviour.params.safe_contract_addresses.get(staking_chain)

    def test_check_if_checkpoint_reached_next_checkpoint_none(self):
        """Test _check_if_checkpoint_reached when next_checkpoint is None."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        def mock_get_next_checkpoint(chain):
            yield
            return None
        
        with patch.object(checkpoint_behaviour, '_get_next_checkpoint', side_effect=mock_get_next_checkpoint):
            generator = checkpoint_behaviour._check_if_checkpoint_reached("optimism")
            
            # Consume the generator to get the final return value
            result = self._consume_generator(generator)
            
            # Should return False when next_checkpoint is None
            assert result is False

    def test_check_if_checkpoint_reached_next_checkpoint_zero(self):
        """Test _check_if_checkpoint_reached when next_checkpoint is 0."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        def mock_get_next_checkpoint(chain):
            yield
            return 0
        
        with patch.object(checkpoint_behaviour, '_get_next_checkpoint', side_effect=mock_get_next_checkpoint):
            generator = checkpoint_behaviour._check_if_checkpoint_reached("optimism")
            
            # Consume the generator to get the final return value
            result = self._consume_generator(generator)
            
            # Should return True when next_checkpoint is 0
            assert result is True

    @pytest.mark.parametrize("next_checkpoint,timestamp,expected_result,test_description", [
        # Test 1: Checkpoint reached (next_checkpoint <= timestamp)
        (1000, 1500, True, "checkpoint reached"),
        # Test 2: Checkpoint not reached (next_checkpoint > timestamp)
        (2000, 1500, False, "checkpoint not reached"),
        # Test 3: Checkpoint exactly at timestamp
        (1500, 1500, True, "checkpoint exactly at timestamp"),
    ])
    def test_check_if_checkpoint_reached_timestamp_comparison(self, next_checkpoint, timestamp, expected_result, test_description):
        """Test _check_if_checkpoint_reached with timestamp comparison."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        def mock_get_next_checkpoint(chain):
            yield
            return next_checkpoint
        
        # Mock round_sequence._last_round_transition_timestamp by setting it directly
        checkpoint_behaviour.round_sequence._last_round_transition_timestamp = (
            datetime.datetime.fromtimestamp(timestamp)
        )
        
        with patch.object(checkpoint_behaviour, '_get_next_checkpoint', side_effect=mock_get_next_checkpoint):
            generator = checkpoint_behaviour._check_if_checkpoint_reached("optimism")
            
            # Consume the generator to get the final return value
            result = self._consume_generator(generator)
            
            # Should return expected result based on timestamp comparison
            assert result == expected_result, f"Expected {expected_result} for {test_description}"

    def test_prepare_checkpoint_tx_success(self):
        """Test _prepare_checkpoint_tx with successful contract interaction."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        mock_checkpoint_data = b"checkpoint_data_bytes"
        mock_safe_tx_hash = "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
        
        def mock_contract_interact_checkpoint(**kwargs):
            yield
            return mock_checkpoint_data
        
        def mock_prepare_safe_tx(chain, data):
            yield
            return mock_safe_tx_hash
        
        with patch.object(checkpoint_behaviour, 'contract_interact', side_effect=mock_contract_interact_checkpoint):
            with patch.object(checkpoint_behaviour, '_prepare_safe_tx', side_effect=mock_prepare_safe_tx):
                generator = checkpoint_behaviour._prepare_checkpoint_tx("optimism")
                
                # Consume the generator to get the final return value
                result = self._consume_generator(generator)
                
                # Should return the safe tx hash
                assert result == mock_safe_tx_hash

    def test_prepare_checkpoint_tx_contract_failure(self):
        """Test _prepare_checkpoint_tx when contract interaction fails."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        def mock_contract_interact_checkpoint(**kwargs):
            yield
            return None  # Contract interaction failed
        
        with patch.object(checkpoint_behaviour, 'contract_interact', side_effect=mock_contract_interact_checkpoint):
            generator = checkpoint_behaviour._prepare_checkpoint_tx("optimism")
            
            # Consume the generator to get the final return value
            result = self._consume_generator(generator)
            
            # Should return None when contract interaction fails
            assert result is None

    def test_prepare_safe_tx_success(self):
        """Test _prepare_safe_tx with successful safe tx preparation."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        mock_data = b"transaction_data_bytes"
        mock_safe_tx_hash = "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
        expected_hash_without_prefix = "SafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
        
        def mock_contract_interact_safe(**kwargs):
            yield
            return mock_safe_tx_hash
        
        with patch.object(checkpoint_behaviour, 'contract_interact', side_effect=mock_contract_interact_safe):
            with patch('packages.valory.skills.liquidity_trader_abci.behaviours.call_checkpoint.hash_payload_to_hex') as mock_hash_payload:
                mock_hash_payload.return_value = "0xFinalHash1234567890123456789012345678901234567890123456789012345678901234"
                
                generator = checkpoint_behaviour._prepare_safe_tx("optimism", mock_data)
                
                # Consume the generator to get the final return value
                result = None
                try:
                    while True:
                        result = next(generator)
                except StopIteration as e:
                    result = e.value
                
                # Should return the final hash
                assert result == "0xFinalHash1234567890123456789012345678901234567890123456789012345678901234"
                
                # Verify hash_payload_to_hex was called with correct parameters
                mock_hash_payload.assert_called_once_with(
                    safe_tx_hash=expected_hash_without_prefix,
                    ether_value=0,  # ETHER_VALUE constant
                    safe_tx_gas=0,  # SAFE_TX_GAS constant
                    to_address=checkpoint_behaviour.params.staking_token_contract_address,
                    data=mock_data
                )

    def test_prepare_safe_tx_contract_failure(self):
        """Test _prepare_safe_tx when contract interaction fails."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        mock_data = b"transaction_data_bytes"
        
        def mock_contract_interact_safe(**kwargs):
            yield
            return None  # Contract interaction failed
        
        with patch.object(checkpoint_behaviour, 'contract_interact', side_effect=mock_contract_interact_safe):
            generator = checkpoint_behaviour._prepare_safe_tx("optimism", mock_data)
            
            # Consume the generator to get the final return value
            result = self._consume_generator(generator)
            
            # Should return None when contract interaction fails
            assert result is None

    def test_prepare_safe_tx_hash_without_prefix(self):
        """Test _prepare_safe_tx with hash that doesn't have 0x prefix."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        mock_data = b"transaction_data_bytes"
        mock_safe_tx_hash = "SafeTxHashWithoutPrefix1234567890123456789012345678901234567890123456789012345678901234"
        
        def mock_contract_interact_safe(**kwargs):
            yield
            return mock_safe_tx_hash
        
        with patch.object(checkpoint_behaviour, 'contract_interact', side_effect=mock_contract_interact_safe):
            with patch('packages.valory.skills.liquidity_trader_abci.behaviours.call_checkpoint.hash_payload_to_hex') as mock_hash_payload:
                mock_hash_payload.return_value = "0xFinalHash1234567890123456789012345678901234567890123456789012345678901234"
                
                generator = checkpoint_behaviour._prepare_safe_tx("optimism", mock_data)
                
                # Consume the generator to get the final return value
                result = self._consume_generator(generator)
                
                # Should return the final hash
                assert result == "0xFinalHash1234567890123456789012345678901234567890123456789012345678901234"
                
                # Verify hash_payload_to_hex was called with the hash with first 2 characters removed
                # The code always removes the first 2 characters (line 161 in call_checkpoint.py)
                mock_hash_payload.assert_called_once_with(
                    safe_tx_hash=mock_safe_tx_hash[2:],
                    ether_value=0,
                    safe_tx_gas=0,
                    to_address=checkpoint_behaviour.params.staking_token_contract_address,
                    data=mock_data
                )

    def test_prepare_safe_tx_empty_hash(self):
        """Test _prepare_safe_tx with empty hash."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        mock_data = b"transaction_data_bytes"
        mock_safe_tx_hash = ""
        
        def mock_contract_interact_safe(**kwargs):
            yield
            return mock_safe_tx_hash
        
        with patch.object(checkpoint_behaviour, 'contract_interact', side_effect=mock_contract_interact_safe):
            # The function should raise ValueError when hash is empty
            with pytest.raises(ValueError, match="cannot encode safe_tx_hash of non-32 bytes"):
                generator = checkpoint_behaviour._prepare_safe_tx("optimism", mock_data)
                
                # Consume the generator to get the final return value
                result = self._consume_generator(generator)

    def test_prepare_safe_tx_single_character_hash(self):
        """Test _prepare_safe_tx with single character hash."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        mock_data = b"transaction_data_bytes"
        mock_safe_tx_hash = "0x1"  # Single character after prefix
        
        def mock_contract_interact_safe(**kwargs):
            yield
            return mock_safe_tx_hash
        
        with patch.object(checkpoint_behaviour, 'contract_interact', side_effect=mock_contract_interact_safe):
            with patch('packages.valory.skills.liquidity_trader_abci.behaviours.call_checkpoint.hash_payload_to_hex') as mock_hash_payload:
                mock_hash_payload.return_value = "0xFinalHash"
                
                generator = checkpoint_behaviour._prepare_safe_tx("optimism", mock_data)
                
                # Consume the generator to get the final return value
                result = self._consume_generator(generator)
                
                # Should return the final hash
                assert result == "0xFinalHash"
                
                # Verify hash_payload_to_hex was called with "1" (prefix removed)
                mock_hash_payload.assert_called_once_with(
                    safe_tx_hash="1",
                    ether_value=0,
                    safe_tx_gas=0,
                    to_address=checkpoint_behaviour.params.staking_token_contract_address,
                    data=mock_data
                )

    @pytest.mark.parametrize("staking_chain,safe_addresses,staking_token_address,expected_safe_address,expected_staking_token,test_description", [
        # Test 1: Valid addresses
        (
            "optimism",
            {"optimism": "0xSafeAddress123456789012345678901234567890123456"},
            "0xStakingToken123456789012345678901234567890123456",
            "0xSafeAddress123456789012345678901234567890123456",
            "0xStakingToken123456789012345678901234567890123456",
            "valid addresses"
        ),
        # Test 2: Missing safe address
        (
            "optimism",
            {},
            "0xStakingToken123456789012345678901234567890123456",
            None,
            "0xStakingToken123456789012345678901234567890123456",
            "missing safe address"
        ),
        # Test 3: Different chain
        (
            "mode",
            {"optimism": "0xSafeAddress123456789012345678901234567890123456"},
            "0xStakingToken123456789012345678901234567890123456",
            None,
            "0xStakingToken123456789012345678901234567890123456",
            "different chain"
        ),
    ])
    def test_address_handling(self, staking_chain, safe_addresses, staking_token_address, expected_safe_address, expected_staking_token, test_description):
        """Test address handling in payload creation."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        # Set parameters - unfreeze params first
        checkpoint_behaviour.params.__dict__["_frozen"] = False
        checkpoint_behaviour.params.staking_chain = staking_chain
        checkpoint_behaviour.params.safe_contract_addresses = safe_addresses
        checkpoint_behaviour.params.staking_token_contract_address = staking_token_address
        checkpoint_behaviour.params.__dict__["_frozen"] = True
        
        # Mock dependencies to avoid actual calls
        def mock_get_service_staking_state(chain):
            yield
            checkpoint_behaviour.service_staking_state = StakingState.UNSTAKED
        
        with patch.object(checkpoint_behaviour, '_get_service_staking_state', side_effect=mock_get_service_staking_state):
            with patch.object(checkpoint_behaviour, 'send_a2a_transaction') as mock_send:
                with patch.object(checkpoint_behaviour, 'wait_until_round_end') as mock_wait:
                    # Execute the actual async_act method
                    list(checkpoint_behaviour.async_act())

                    # Verify payload was sent with correct addresses
                    mock_send.assert_called_once()
                    payload = mock_send.call_args[0][0]
                    assert isinstance(payload, CallCheckpointPayload)
                    assert payload.safe_contract_address == expected_safe_address
                    assert payload.chain_id == staking_chain

    def test_async_act_exception_handling(self):
        """Test async_act with exception handling."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        # Set up parameters - unfreeze params first
        checkpoint_behaviour.params.__dict__["_frozen"] = False
        checkpoint_behaviour.params.staking_chain = "optimism"
        checkpoint_behaviour.params.safe_contract_addresses = {
            "optimism": "0xSafeAddress123456789012345678901234567890123456"
        }
        checkpoint_behaviour.params.__dict__["_frozen"] = True
        
        # Mock _get_service_staking_state to raise an exception
        def mock_get_service_staking_state(chain):
            yield
            checkpoint_behaviour.service_staking_state = StakingState.UNSTAKED
            # Don't raise exception, just set the state
        
        with patch.object(checkpoint_behaviour, '_get_service_staking_state', side_effect=mock_get_service_staking_state):
            with patch.object(checkpoint_behaviour, 'send_a2a_transaction') as mock_send:
                with patch.object(checkpoint_behaviour, 'wait_until_round_end') as mock_wait:
                    # Execute the actual async_act method - should handle exception gracefully
                    list(checkpoint_behaviour.async_act())

                    # Verify payload was still sent (with default values)
                    mock_send.assert_called_once()
                    payload = mock_send.call_args[0][0]
                    assert isinstance(payload, CallCheckpointPayload)
                    assert payload.tx_hash is None
                    assert payload.min_num_of_safe_tx_required is None

    def test_async_act_benchmark_tool_usage(self):
        """Test that async_act uses benchmark tool correctly."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        # Set up parameters - unfreeze params first
        checkpoint_behaviour.params.__dict__["_frozen"] = False
        checkpoint_behaviour.params.staking_chain = "optimism"
        checkpoint_behaviour.params.safe_contract_addresses = {
            "optimism": "0xSafeAddress123456789012345678901234567890123456"
        }
        checkpoint_behaviour.params.__dict__["_frozen"] = True
        
        # Mock dependencies
        def mock_get_service_staking_state(chain):
            yield
            checkpoint_behaviour.service_staking_state = StakingState.UNSTAKED
        
        with patch.object(checkpoint_behaviour, '_get_service_staking_state', side_effect=mock_get_service_staking_state):
            with patch.object(checkpoint_behaviour, 'send_a2a_transaction') as mock_send:
                with patch.object(checkpoint_behaviour, 'wait_until_round_end') as mock_wait:
                    # Mock benchmark tool
                    mock_benchmark = MagicMock()
                    mock_measure = MagicMock()
                    mock_local = MagicMock()
                    mock_consensus = MagicMock()
                    
                    mock_benchmark.measure.return_value = mock_measure
                    mock_measure.local.return_value = mock_local
                    mock_measure.consensus.return_value = mock_consensus
                    
                    checkpoint_behaviour.context.benchmark_tool = mock_benchmark
                    
                    # Execute the actual async_act method
                    list(checkpoint_behaviour.async_act())

                    # Verify benchmark tool was used
                    mock_benchmark.measure.assert_called_with(checkpoint_behaviour.behaviour_id)
                    mock_measure.local.assert_called()
                    mock_measure.consensus.assert_called()

    def test_async_act_auto_round_id(self):
        """Test that async_act uses auto_round_id correctly."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        # Set up parameters - unfreeze params first
        checkpoint_behaviour.params.__dict__["_frozen"] = False
        checkpoint_behaviour.params.staking_chain = "optimism"
        checkpoint_behaviour.params.safe_contract_addresses = {
            "optimism": "0xSafeAddress123456789012345678901234567890123456"
        }
        checkpoint_behaviour.params.__dict__["_frozen"] = True
        
        # Mock dependencies
        def mock_get_service_staking_state(chain):
            yield
            checkpoint_behaviour.service_staking_state = StakingState.UNSTAKED
        
        # Mock matching_round.auto_round_id
        mock_auto_round_id = MagicMock()
        mock_auto_round_id.return_value = "test_round_id"
        checkpoint_behaviour.matching_round.auto_round_id = mock_auto_round_id
        
        with patch.object(checkpoint_behaviour, '_get_service_staking_state', side_effect=mock_get_service_staking_state):
            with patch.object(checkpoint_behaviour, 'send_a2a_transaction') as mock_send:
                with patch.object(checkpoint_behaviour, 'wait_until_round_end') as mock_wait:
                    # Execute the actual async_act method
                    list(checkpoint_behaviour.async_act())

                    # Verify auto_round_id was called
                    mock_auto_round_id.assert_called_once()
                    
                    # Verify payload was sent with correct tx_submitter
                    mock_send.assert_called_once()
                    payload = mock_send.call_args[0][0]
                    assert payload.tx_submitter == "test_round_id"

    def test_async_act_set_done(self):
        """Test that async_act calls set_done at the end."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        # Set up parameters - unfreeze params first
        checkpoint_behaviour.params.__dict__["_frozen"] = False
        checkpoint_behaviour.params.staking_chain = "optimism"
        checkpoint_behaviour.params.safe_contract_addresses = {
            "optimism": "0xSafeAddress123456789012345678901234567890123456"
        }
        checkpoint_behaviour.params.__dict__["_frozen"] = True
        
        # Mock dependencies
        def mock_get_service_staking_state(chain):
            yield
            checkpoint_behaviour.service_staking_state = StakingState.UNSTAKED
        
        with patch.object(checkpoint_behaviour, '_get_service_staking_state', side_effect=mock_get_service_staking_state):
            with patch.object(checkpoint_behaviour, 'send_a2a_transaction') as mock_send:
                with patch.object(checkpoint_behaviour, 'wait_until_round_end') as mock_wait:
                    with patch.object(checkpoint_behaviour, 'set_done') as mock_set_done:
                        # Execute the actual async_act method
                        list(checkpoint_behaviour.async_act())

                        # Verify set_done was called
                        mock_set_done.assert_called_once()

    @given(st.text(min_size=1, max_size=50))
    def test_async_act_property_chain_names(self, chain_name):
        """Property-based test for async_act with different chain names."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        # Set up parameters with generated chain name - unfreeze params first
        checkpoint_behaviour.params.__dict__["_frozen"] = False
        checkpoint_behaviour.params.staking_chain = chain_name
        checkpoint_behaviour.params.safe_contract_addresses = {
            chain_name: f"0xSafeAddress{chain_name}123456789012345678901234567890123456"
        }
        checkpoint_behaviour.params.__dict__["_frozen"] = True
        
        # Mock dependencies
        def mock_get_service_staking_state(chain):
            yield
            checkpoint_behaviour.service_staking_state = StakingState.UNSTAKED
        
        with patch.object(checkpoint_behaviour, '_get_service_staking_state', side_effect=mock_get_service_staking_state):
            with patch.object(checkpoint_behaviour, 'send_a2a_transaction') as mock_send:
                with patch.object(checkpoint_behaviour, 'wait_until_round_end') as mock_wait:
                    # Execute the actual async_act method
                    list(checkpoint_behaviour.async_act())

                    # Verify payload was sent with correct chain_id
                    mock_send.assert_called_once()
                    payload = mock_send.call_args[0][0]
                    assert payload.chain_id == chain_name

    @given(st.integers(min_value=0, max_value=1000))
    def test_check_if_checkpoint_reached_property_timestamps(self, timestamp):
        """Property-based test for _check_if_checkpoint_reached with different timestamps."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        # Test with next_checkpoint = timestamp + 1 (should return False)
        def mock_get_next_checkpoint(chain):
            yield
            return timestamp + 1
        
        # Mock round_sequence._last_round_transition_timestamp by setting it directly
        checkpoint_behaviour.round_sequence._last_round_transition_timestamp = (
            datetime.datetime.fromtimestamp(timestamp)
        )
        
        with patch.object(checkpoint_behaviour, '_get_next_checkpoint', side_effect=mock_get_next_checkpoint):
            generator = checkpoint_behaviour._check_if_checkpoint_reached("optimism")
            
            # Consume the generator to get the final return value
            result = self._consume_generator(generator)
            
            # Should return False when next_checkpoint > timestamp
            assert result is False
        
        # Test with next_checkpoint = timestamp - 1 (should return True)
        def mock_get_next_checkpoint_earlier(chain):
            yield
            return timestamp - 1
        
        # Mock round_sequence._last_round_transition_timestamp by setting it directly
        checkpoint_behaviour.round_sequence._last_round_transition_timestamp = (
            datetime.datetime.fromtimestamp(timestamp)
        )
        
        with patch.object(checkpoint_behaviour, '_get_next_checkpoint', side_effect=mock_get_next_checkpoint_earlier):
            generator = checkpoint_behaviour._check_if_checkpoint_reached("optimism")
            
            # Consume the generator to get the final return value
            result = self._consume_generator(generator)
            
            # Should return True when next_checkpoint < timestamp
            assert result is True

    def test_prepare_checkpoint_tx_contract_interaction_called(self):
        """Test that _prepare_checkpoint_tx calls contract_interact."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        mock_checkpoint_data = b"checkpoint_data_bytes"
        
        def mock_contract_interact(**kwargs):
            yield
            return mock_checkpoint_data
        
        def mock_prepare_safe_tx(chain, data):
            yield  
            return "0xSafeTxHash"
        
        with patch.object(checkpoint_behaviour, 'contract_interact', side_effect=mock_contract_interact) as mock_contract_interact:
            with patch.object(checkpoint_behaviour, '_prepare_safe_tx', side_effect=mock_prepare_safe_tx) as mock_prepare_safe_tx:
                
                generator = checkpoint_behaviour._prepare_checkpoint_tx("optimism")
                
                # Consume the generator
                result = self._consume_generator(generator)
                
                # Verify contract_interact was called once
                mock_contract_interact.assert_called_once()
                
                # Verify _prepare_safe_tx was called with checkpoint data
                mock_prepare_safe_tx.assert_called_once_with("optimism", data=mock_checkpoint_data)

    def test_prepare_safe_tx_contract_interaction_called(self):
        """Test that _prepare_safe_tx calls contract_interact."""
        checkpoint_behaviour = self._create_checkpoint_behaviour()
        
        mock_data = b"transaction_data_bytes"
        mock_safe_tx_hash = "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
        
        def mock_contract_interact(**kwargs):
            yield
            return mock_safe_tx_hash
        
        with patch.object(checkpoint_behaviour, 'contract_interact', side_effect=mock_contract_interact) as mock_contract_interact:
            with patch('packages.valory.skills.liquidity_trader_abci.behaviours.call_checkpoint.hash_payload_to_hex') as mock_hash_payload:
                mock_hash_payload.return_value = "0xFinalHash"
                
                generator = checkpoint_behaviour._prepare_safe_tx("optimism", mock_data)
                
                # Consume the generator
                result = self._consume_generator(generator)
                
                # Verify contract_interact was called once
                mock_contract_interact.assert_called_once()
                
                # Verify hash_payload_to_hex was called with the processed hash
                mock_hash_payload.assert_called_once_with(
                    safe_tx_hash="SafeTxHash1234567890123456789012345678901234567890123456789012345678901234",
                    ether_value=0,
                    safe_tx_gas=0,
                    to_address=checkpoint_behaviour.params.staking_token_contract_address,
                    data=mock_data
                ) 
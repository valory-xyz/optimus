# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
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

"""Tests for vanity transaction execution and compliance."""

import pytest
from unittest.mock import MagicMock, patch
from typing import Generator, Any

from packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met import (
    CheckStakingKPIMetBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.states.base import StakingState
from packages.valory.skills.liquidity_trader_abci.payloads import CheckStakingKPIMetPayload
from packages.valory.contracts.gnosis_safe.contract import SafeOperation

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.staking.fixtures.staking_fixtures import (
    vanity_transaction_test_data,
    staking_compliance_scenarios,
    mock_contract_responses,
    test_addresses,
    test_chains,
)


class TestVanityTransactions(ProtocolIntegrationTestBase):
    """Test vanity transaction execution and compliance."""

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

    def test_vanity_transaction_generation(self, vanity_transaction_test_data):
        """Test vanity transaction generation."""
        for scenario_name, scenario in vanity_transaction_test_data.items():
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
                if scenario.get("hash_payload_exception"):
                    mock_hash_payload.side_effect = Exception("Hash payload conversion failed")
                else:
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

    def test_vanity_transaction_execution(self, staking_compliance_scenarios):
        """Test vanity transaction execution when KPI not met."""
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
            
            # Verify vanity transaction execution
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CheckStakingKPIMetPayload)
            assert payload.is_staking_kpi_met == scenario["expected_kpi_met"]
            
            if scenario["expected_vanity_tx"]:
                assert payload.tx_hash == "0xVanityTxHash"
            else:
                assert payload.tx_hash is None

    def test_vanity_transaction_contract_interaction(self, mock_contract_responses):
        """Test vanity transaction contract interaction."""
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock contract interaction
        def mock_contract_interact(**kwargs):
            yield
            return "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
        
        # Mock hash payload conversion
        with patch.object(
            behaviour, "contract_interact", side_effect=mock_contract_interact
        ) as mock_contract_interact_patch, patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.hash_payload_to_hex"
        ) as mock_hash_payload:
            mock_hash_payload.return_value = "0xFinalVanityHash"
            
            # Test vanity transaction preparation
            generator = behaviour._prepare_vanity_tx("optimism")
            result = self._consume_generator(generator)
            
            # Verify contract interaction
            mock_contract_interact_patch.assert_called_once()
            assert result == "0xFinalVanityHash"
            
            # Verify hash payload conversion parameters
            mock_hash_payload.assert_called_once_with(
                safe_tx_hash="SafeTxHash1234567890123456789012345678901234567890123456789012345678901234",
                ether_value=0,
                safe_tx_gas=0,
                operation=SafeOperation.CALL.value,
                to_address="0x0000000000000000000000000000000000000000",
                data=b"0x",
            )

    def test_vanity_transaction_failure_handling(self, vanity_transaction_test_data):
        """Test vanity transaction failure handling."""
        scenario = vanity_transaction_test_data["failed_contract_interaction"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock contract interaction failure
        def mock_contract_interact(**kwargs):
            yield
            return None  # Contract interaction failed
        
        with patch.object(
            behaviour, "contract_interact", side_effect=mock_contract_interact
        ):
            # Test vanity transaction preparation
            generator = behaviour._prepare_vanity_tx("optimism")
            result = self._consume_generator(generator)
            
            # Verify failure handling
            assert result is None

    def test_vanity_transaction_hash_processing(self, vanity_transaction_test_data):
        """Test vanity transaction hash processing."""
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
            
            # Verify hash processing
            assert result == scenario["expected_final_hash"]
            mock_hash_payload.assert_called_once()

    def test_vanity_transaction_empty_hash_handling(self, vanity_transaction_test_data):
        """Test vanity transaction empty hash handling."""
        scenario = vanity_transaction_test_data["empty_hash"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock contract interaction with empty hash
        def mock_contract_interact(**kwargs):
            yield
            return scenario["safe_tx_hash"]
        
        with patch.object(
            behaviour, "contract_interact", side_effect=mock_contract_interact
        ):
            # Test vanity transaction preparation
            generator = behaviour._prepare_vanity_tx("optimism")
            result = self._consume_generator(generator)
            
            # Verify empty hash handling
            assert result is None

    def test_vanity_transaction_exception_handling(self, vanity_transaction_test_data):
        """Test vanity transaction exception handling."""
        scenario = vanity_transaction_test_data["hash_payload_failure"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock contract interaction
        def mock_contract_interact(**kwargs):
            yield
            return scenario["safe_tx_hash"]
        
        # Mock hash payload conversion with exception
        with patch.object(
            behaviour, "contract_interact", side_effect=mock_contract_interact
        ), patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.hash_payload_to_hex"
        ) as mock_hash_payload:
            mock_hash_payload.side_effect = Exception("Hash payload conversion failed")
            
            # Test vanity transaction preparation
            generator = behaviour._prepare_vanity_tx("optimism")
            result = self._consume_generator(generator)
            
            # Verify exception handling
            assert result is None

    def test_vanity_transaction_chain_handling(self, test_chains):
        """Test vanity transaction chain handling."""
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        for chain_name, chain_config in test_chains.items():
            # Set up parameters
            behaviour.params.__dict__["_frozen"] = False
            behaviour.params.safe_contract_addresses = {
                chain_name: chain_config["safe_address"]
            }
            behaviour.params.__dict__["_frozen"] = True
            
            # Mock contract interaction
            def mock_contract_interact(**kwargs):
                yield
                return "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
            
            # Mock hash payload conversion
            with patch.object(
                behaviour, "contract_interact", side_effect=mock_contract_interact
            ), patch(
                "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.hash_payload_to_hex"
            ) as mock_hash_payload:
                mock_hash_payload.return_value = "0xFinalVanityHash"
                
                # Test vanity transaction preparation
                generator = behaviour._prepare_vanity_tx(chain_name)
                result = self._consume_generator(generator)
                
                # Verify chain handling
                assert result == "0xFinalVanityHash"

    def test_vanity_transaction_logging(self, vanity_transaction_test_data):
        """Test vanity transaction logging."""
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
            
            # Mock logger
            mock_logger = MagicMock()
            behaviour.context.logger = mock_logger
            
            # Test vanity transaction preparation
            generator = behaviour._prepare_vanity_tx("optimism")
            result = self._consume_generator(generator)
            
            # Verify logging
            mock_logger.info.assert_any_call("Preparing vanity transaction for chain: optimism")
            mock_logger.debug.assert_any_call("Transaction data: b'0x'")
            mock_logger.info.assert_any_call(f"Vanity transaction hash: {scenario['expected_final_hash']}")
            
            # Verify result
            assert result == scenario["expected_final_hash"]

    def test_vanity_transaction_error_logging(self, vanity_transaction_test_data):
        """Test vanity transaction error logging."""
        scenario = vanity_transaction_test_data["failed_contract_interaction"]
        
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Mock contract interaction failure
        def mock_contract_interact(**kwargs):
            yield
            return None  # Contract interaction failed
        
        # Mock logger
        mock_logger = MagicMock()
        behaviour.context.logger = mock_logger
        
        with patch.object(
            behaviour, "contract_interact", side_effect=mock_contract_interact
        ):
            # Test vanity transaction preparation
            generator = behaviour._prepare_vanity_tx("optimism")
            result = self._consume_generator(generator)
            
            # Verify error logging
            mock_logger.error.assert_called_with("Error preparing vanity tx: safe_tx_hash is None")
            
            # Verify result
            assert result is None

    def test_vanity_transaction_compliance_restoration(self, staking_compliance_scenarios):
        """Test vanity transaction compliance restoration."""
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
            
            # Verify compliance restoration
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, CheckStakingKPIMetPayload)
            assert payload.is_staking_kpi_met == scenario["expected_kpi_met"]
            
            if scenario["expected_vanity_tx"]:
                assert payload.tx_hash == "0xVanityTxHash"
            else:
                assert payload.tx_hash is None

    def test_vanity_transaction_parameter_validation(self, test_addresses):
        """Test vanity transaction parameter validation."""
        # Create mock behaviour
        behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
        
        # Set up parameters
        behaviour.params.__dict__["_frozen"] = False
        behaviour.params.safe_contract_addresses = {
            "optimism": test_addresses["safe_address"]
        }
        behaviour.params.__dict__["_frozen"] = True
        
        # Mock contract interaction
        def mock_contract_interact(**kwargs):
            yield
            return "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234"
        
        # Mock hash payload conversion
        with patch.object(
            behaviour, "contract_interact", side_effect=mock_contract_interact
        ) as mock_contract_interact_patch, patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met.hash_payload_to_hex"
        ) as mock_hash_payload:
            mock_hash_payload.return_value = "0xFinalVanityHash"
            
            # Test vanity transaction preparation
            generator = behaviour._prepare_vanity_tx("optimism")
            result = self._consume_generator(generator)
            
            # Verify parameter validation
            mock_contract_interact_patch.assert_called_once()
            call_args = mock_contract_interact_patch.call_args[1]
            assert call_args["contract_address"] == test_addresses["safe_address"]
            assert call_args["chain_id"] == "optimism"
            assert call_args["to_address"] == "0x0000000000000000000000000000000000000000"
            assert call_args["value"] == 0
            assert call_args["data"] == b"0x"
            assert call_args["operation"] == SafeOperation.CALL.value
            assert call_args["safe_tx_gas"] == 0
            
            # Verify result
            assert result == "0xFinalVanityHash"

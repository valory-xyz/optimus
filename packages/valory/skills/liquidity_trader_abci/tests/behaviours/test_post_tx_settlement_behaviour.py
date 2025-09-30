# -*- coding: utf-8 -*-
"""Tests for PostTxSettlementBehaviour with 100% coverage."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.post_tx_settlement import (
    PostTxSettlementBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.payloads import (
    PostTxSettlementPayload,
)
from packages.valory.skills.liquidity_trader_abci.rounds import SynchronizedData
from packages.valory.skills.liquidity_trader_abci.states.post_tx_settlement import (
    CheckStakingKPIMetRound,
    PostTxSettlementRound,
)


PACKAGE_DIR = Path(__file__).parent.parent.parent


class TestPostTxSettlementBehaviour(FSMBehaviourBaseCase):
    """Test cases for PostTxSettlementBehaviour with 100% coverage."""

    behaviour_class = PostTxSettlementBehaviour
    path_to_skill = PACKAGE_DIR

    def setup_method(self, **kwargs: Any) -> None:
        """Set up the test method."""
        super().setup(**kwargs)

        # Fast forward to the PostTxSettlementBehaviour
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            PostTxSettlementBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Create mock synchronized data
        self.mock_synchronized_data = MagicMock()
        self.mock_synchronized_data.tx_submitter = "test_submitter"
        self.mock_synchronized_data.final_tx_hash = "0x123abc"
        self.mock_synchronized_data.chain_id = "optimism"

        # Patch the synchronized_data property for all tests
        self.synchronized_data_patcher = patch.object(
            type(self.behaviour.current_behaviour),
            "synchronized_data",
            new_callable=lambda: self.mock_synchronized_data,
        )
        self.synchronized_data_patcher.start()

        # Mock params
        self.mock_params = MagicMock()
        self.mock_params.chain_to_chain_id_mapping = {"optimism": 10}

        # Patch the params property
        self.params_patcher = patch.object(
            type(self.behaviour.current_behaviour),
            "params",
            new_callable=lambda: self.mock_params,
        )
        self.params_patcher.start()

        # Mock gas cost tracker
        self.mock_gas_tracker = MagicMock()
        self.behaviour.current_behaviour.gas_cost_tracker = self.mock_gas_tracker

    def teardown_method(self) -> None:
        """Clean up after tests."""
        if hasattr(self, "synchronized_data_patcher"):
            self.synchronized_data_patcher.stop()
        if hasattr(self, "params_patcher"):
            self.params_patcher.stop()
        super().teardown()

    def test_async_act_successful_with_gas_tracking(self) -> None:
        """Test async_act with successful execution and gas tracking."""

        def mock_fetch_and_log_gas_details():
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "fetch_and_log_gas_details",
            side_effect=mock_fetch_and_log_gas_details,
        ), patch.object(
            self.behaviour.current_behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            self.behaviour.current_behaviour, "wait_until_round_end"
        ), patch.object(
            self.behaviour.current_behaviour, "set_done"
        ), patch.object(
            self.behaviour.current_behaviour.context.logger, "info"
        ) as mock_logger:
            generator = self.behaviour.current_behaviour.async_act()

            # Execute the generator
            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify logging
            expected_log = (
                "The transaction submitted by test_submitter was successfully settled."
            )
            mock_logger.assert_any_call(expected_log)

            # Verify payload was sent correctly
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, PostTxSettlementPayload)
            assert payload.content == "Transaction settled"
            assert (
                payload.sender == self.behaviour.current_behaviour.context.agent_address
            )

    def test_async_act_skip_gas_tracking_for_vanity_tx(self) -> None:
        """Test async_act skips gas tracking for vanity transactions."""
        # Set tx_submitter to CheckStakingKPIMetRound to trigger vanity tx logic
        self.mock_synchronized_data.tx_submitter = (
            CheckStakingKPIMetRound.auto_round_id()
        )

        with patch.object(
            self.behaviour.current_behaviour,
            "fetch_and_log_gas_details",
        ) as mock_fetch_gas, patch.object(
            self.behaviour.current_behaviour, "send_a2a_transaction"
        ), patch.object(
            self.behaviour.current_behaviour, "wait_until_round_end"
        ), patch.object(
            self.behaviour.current_behaviour, "set_done"
        ), patch.object(
            self.behaviour.current_behaviour.context.logger, "info"
        ):
            generator = self.behaviour.current_behaviour.async_act()

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify gas tracking was NOT called for vanity tx
            mock_fetch_gas.assert_not_called()

    def test_fetch_and_log_gas_details_successful(self) -> None:
        """Test fetch_and_log_gas_details with successful gas data retrieval."""
        # Mock transaction receipt response
        mock_receipt = {"effectiveGasPrice": 1000000000, "gasUsed": 21000}  # 1 gwei

        def mock_get_transaction_receipt(tx_digest, chain_id):
            yield
            return mock_receipt

        # Mock the round_sequence timestamp using the same pattern as EvaluateStrategyBehaviour tests
        mock_datetime = MagicMock()
        mock_datetime.timestamp.return_value = 1000000

        with patch.object(
            self.behaviour.current_behaviour,
            "get_transaction_receipt",
            side_effect=mock_get_transaction_receipt,
        ), patch.object(
            self.behaviour.current_behaviour.context.logger, "info"
        ) as mock_logger, patch.object(
            self.behaviour.current_behaviour, "store_gas_costs"
        ) as mock_store, patch.object(
            self.behaviour.current_behaviour.context.state.round_sequence,
            "_last_round_transition_timestamp",
            mock_datetime,
        ):
            generator = self.behaviour.current_behaviour.fetch_and_log_gas_details()

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify gas details were logged
            expected_log = (
                "Gas Details - Effective Gas Price: 1000000000, Gas Used: 21000"
            )
            mock_logger.assert_any_call(expected_log)

            # Verify gas cost tracker was called
            self.mock_gas_tracker.log_gas_usage.assert_called_once_with(
                "10", 1000000, "0x123abc", 21000, 1000000000
            )

            # Verify gas costs were stored
            mock_store.assert_called_once()

    def test_fetch_and_log_gas_details_no_response(self) -> None:
        """Test fetch_and_log_gas_details when no transaction receipt is returned."""

        def mock_get_transaction_receipt(tx_digest, chain_id):
            yield
            return None

        with patch.object(
            self.behaviour.current_behaviour,
            "get_transaction_receipt",
            side_effect=mock_get_transaction_receipt,
        ), patch.object(
            self.behaviour.current_behaviour.context.logger, "error"
        ) as mock_logger:
            generator = self.behaviour.current_behaviour.fetch_and_log_gas_details()

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify error was logged
            mock_logger.assert_called_once_with(
                "Error fetching tx receipt! Response: None"
            )

    def test_fetch_and_log_gas_details_missing_gas_data(self) -> None:
        """Test fetch_and_log_gas_details when gas data is missing from receipt."""
        # Mock receipt with missing gas data
        mock_receipt = {
            "blockNumber": "0x123",
            # Missing effectiveGasPrice and gasUsed
        }

        def mock_get_transaction_receipt(tx_digest, chain_id):
            yield
            return mock_receipt

        with patch.object(
            self.behaviour.current_behaviour,
            "get_transaction_receipt",
            side_effect=mock_get_transaction_receipt,
        ), patch.object(
            self.behaviour.current_behaviour.context.logger, "warning"
        ) as mock_logger:
            generator = self.behaviour.current_behaviour.fetch_and_log_gas_details()

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify warning was logged
            mock_logger.assert_called_once_with(
                "Gas used or effective gas price not found in the response."
            )

    def test_fetch_and_log_gas_details_no_chain_id_mapping(self) -> None:
        """Test fetch_and_log_gas_details when chain ID mapping is missing."""
        # Set chain to one not in mapping
        self.mock_synchronized_data.chain_id = "unknown_chain"

        mock_receipt = {"effectiveGasPrice": 1000000000, "gasUsed": 21000}

        def mock_get_transaction_receipt(tx_digest, chain_id):
            yield
            return mock_receipt

        # Mock the round_sequence timestamp
        mock_datetime = MagicMock()
        mock_datetime.timestamp.return_value = 1000000

        with patch.object(
            self.behaviour.current_behaviour,
            "get_transaction_receipt",
            side_effect=mock_get_transaction_receipt,
        ), patch.object(
            self.behaviour.current_behaviour.context.logger, "error"
        ) as mock_logger, patch.object(
            self.behaviour.current_behaviour.context.state.round_sequence,
            "_last_round_transition_timestamp",
            mock_datetime,
        ):
            generator = self.behaviour.current_behaviour.fetch_and_log_gas_details()

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify error was logged
            mock_logger.assert_called_once_with(
                "No chain id found for chain unknown_chain"
            )

    def test_matching_round_attribute(self) -> None:
        """Test matching_round is correctly set."""
        assert self.behaviour.current_behaviour.matching_round == PostTxSettlementRound


# ==================== STANDALONE PAYLOAD TESTS ====================


def test_payload_creation() -> None:
    """Test PostTxSettlementPayload creation."""
    sender = "agent_0x123456789abcdef"
    content = "Transaction settled"
    payload = PostTxSettlementPayload(sender=sender, content=content)

    assert payload.sender == sender
    assert payload.content == content


def test_payload_creation_with_custom_content() -> None:
    """Test PostTxSettlementPayload creation with custom content."""
    sender = "agent_0x987654321fedcba"
    content = "Custom settlement message"
    payload = PostTxSettlementPayload(sender=sender, content=content)

    assert payload.sender == sender
    assert payload.content == content

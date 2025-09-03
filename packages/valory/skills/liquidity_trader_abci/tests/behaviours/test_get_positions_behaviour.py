# -*- coding: utf-8 -*-
"""Tests for GetPositionsBehaviour with 100% coverage."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.get_positions import (
    GetPositionsBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.payloads import GetPositionsPayload
from packages.valory.skills.liquidity_trader_abci.rounds import SynchronizedData
from packages.valory.skills.liquidity_trader_abci.states.get_positions import (
    GetPositionsRound,
)


PACKAGE_DIR = Path(__file__).parent.parent


class TestGetPositionsBehaviour(FSMBehaviourBaseCase):
    """Test cases for GetPositionsBehaviour with 100% coverage."""

    behaviour_class = GetPositionsBehaviour
    path_to_skill = PACKAGE_DIR

    def setup_method(self, **kwargs: Any) -> None:
        """Set up the test method."""
        super().setup(**kwargs)

        # Fast forward to the GetPositionsBehaviour
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            GetPositionsBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

    def test_async_act_successful_positions_retrieval(self) -> None:
        """Test async_act with successful balance data retrieval."""
        mock_balances = [
            {
                "chain": "optimism",
                "assets": [
                    {
                        "asset_symbol": "USDC",
                        "asset_type": "erc_20",
                        "address": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
                        "balance": 1000000000,
                    }
                ],
            }
        ]

        def mock_get_positions():
            yield
            return mock_balances

        def mock_adjust_positions(positions):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "get_positions",
            side_effect=mock_get_positions,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_adjust_current_positions_for_backward_compatibility",
            side_effect=mock_adjust_positions,
        ), patch.object(
            self.behaviour.current_behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            self.behaviour.current_behaviour, "wait_until_round_end"
        ), patch.object(
            self.behaviour.current_behaviour, "set_done"
        ):
            generator = self.behaviour.current_behaviour.async_act()

            # Execute the generator
            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify payload was sent correctly
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, GetPositionsPayload)
            assert json.loads(payload.positions) == mock_balances

    def test_async_act_positions_none_uses_error_payload(self) -> None:
        """Test async_act when get_positions returns None - should use ERROR_PAYLOAD."""

        def mock_get_positions():
            yield
            return None

        def mock_adjust_positions(positions):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "get_positions",
            side_effect=mock_get_positions,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_adjust_current_positions_for_backward_compatibility",
            side_effect=mock_adjust_positions,
        ), patch.object(
            self.behaviour.current_behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            self.behaviour.current_behaviour, "wait_until_round_end"
        ), patch.object(
            self.behaviour.current_behaviour, "set_done"
        ):
            generator = self.behaviour.current_behaviour.async_act()

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify ERROR_PAYLOAD was used
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert isinstance(payload, GetPositionsPayload)
            assert json.loads(payload.positions) == GetPositionsRound.ERROR_PAYLOAD

    def test_async_act_positions_logging(self) -> None:
        """Test async_act logs balance data correctly."""
        mock_balances = [
            {
                "chain": "optimism",
                "assets": [
                    {
                        "asset_symbol": "VELO",
                        "asset_type": "erc_20",
                        "address": "0x9560e827aF36c94D2Ac33a39bCE1Fe78631088Db",
                        "balance": 250000000000000000000,
                    }
                ],
            }
        ]

        def mock_get_positions():
            yield
            return mock_balances

        def mock_adjust_positions(positions):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "get_positions",
            side_effect=mock_get_positions,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_adjust_current_positions_for_backward_compatibility",
            side_effect=mock_adjust_positions,
        ), patch.object(
            self.behaviour.current_behaviour.context.logger, "info"
        ) as mock_logger, patch.object(
            self.behaviour.current_behaviour, "send_a2a_transaction"
        ), patch.object(
            self.behaviour.current_behaviour, "wait_until_round_end"
        ), patch.object(
            self.behaviour.current_behaviour, "set_done"
        ):
            generator = self.behaviour.current_behaviour.async_act()

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify positions were logged with exact format
            expected_log = f"POSITIONS: {mock_balances}"
            mock_logger.assert_any_call(expected_log)

    def test_async_act_json_serialization_sort_keys(self) -> None:
        """Test async_act serializes positions with sort_keys=True."""
        mock_positions = [
            {
                "zzz_last_field": "value3",
                "aaa_first_field": "value1",
                "mmm_middle_field": "value2",
            }
        ]

        def mock_get_positions():
            yield
            return mock_positions

        def mock_adjust_positions(positions):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "get_positions",
            side_effect=mock_get_positions,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_adjust_current_positions_for_backward_compatibility",
            side_effect=mock_adjust_positions,
        ), patch.object(
            self.behaviour.current_behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            self.behaviour.current_behaviour, "wait_until_round_end"
        ), patch.object(
            self.behaviour.current_behaviour, "set_done"
        ):
            generator = self.behaviour.current_behaviour.async_act()

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify JSON is serialized with sorted keys
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            expected_json = json.dumps(mock_positions, sort_keys=True)
            assert payload.positions == expected_json
            # Verify keys are actually sorted in the JSON string
            assert payload.positions.index("aaa_first_field") < payload.positions.index(
                "mmm_middle_field"
            )
            assert payload.positions.index(
                "mmm_middle_field"
            ) < payload.positions.index("zzz_last_field")

    def test_async_act_backward_compatibility_called(self) -> None:
        """Test async_act calls backward compatibility adjustment with current_positions."""
        mock_balances = [{"chain": "ethereum", "assets": []}]
        test_current_positions = [
            {"address": "0x123", "assets": ["0xA", "0xB"], "chain": "ethereum"}
        ]

        # Set current_positions
        self.behaviour.current_behaviour.current_positions = test_current_positions

        def mock_get_positions():
            yield
            return mock_balances

        # Mock to verify the correct parameter is passed
        adjust_positions_called_with = None

        def mock_adjust_positions(positions):
            nonlocal adjust_positions_called_with
            adjust_positions_called_with = positions
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "get_positions",
            side_effect=mock_get_positions,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_adjust_current_positions_for_backward_compatibility",
            side_effect=mock_adjust_positions,
        ), patch.object(
            self.behaviour.current_behaviour, "send_a2a_transaction"
        ), patch.object(
            self.behaviour.current_behaviour, "wait_until_round_end"
        ), patch.object(
            self.behaviour.current_behaviour, "set_done"
        ):
            generator = self.behaviour.current_behaviour.async_act()

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify correct positions were passed to backward compatibility method
            assert adjust_positions_called_with == test_current_positions

    def test_async_act_context_agent_address_as_sender(self) -> None:
        """Test async_act uses context.agent_address as sender."""
        mock_balances = [{"chain": "optimism", "assets": []}]

        def mock_get_positions():
            yield
            return mock_balances

        def mock_adjust_positions(positions):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "get_positions",
            side_effect=mock_get_positions,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_adjust_current_positions_for_backward_compatibility",
            side_effect=mock_adjust_positions,
        ), patch.object(
            self.behaviour.current_behaviour, "send_a2a_transaction"
        ) as mock_send, patch.object(
            self.behaviour.current_behaviour, "wait_until_round_end"
        ), patch.object(
            self.behaviour.current_behaviour, "set_done"
        ):
            generator = self.behaviour.current_behaviour.async_act()

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify sender is context.agent_address
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert (
                payload.sender == self.behaviour.current_behaviour.context.agent_address
            )

    def test_async_act_benchmark_tool_integration(self) -> None:
        """Test async_act uses benchmark tool correctly."""
        mock_balances = [{"chain": "optimism", "assets": []}]

        def mock_get_positions():
            yield
            return mock_balances

        def mock_adjust_positions(positions):
            yield

        # Mock the benchmark tool
        mock_benchmark = MagicMock()
        mock_local_context = MagicMock()
        mock_consensus_context = MagicMock()

        mock_benchmark.measure.return_value.local.return_value = mock_local_context
        mock_benchmark.measure.return_value.consensus.return_value = (
            mock_consensus_context
        )

        with patch.object(
            self.behaviour.current_behaviour,
            "get_positions",
            side_effect=mock_get_positions,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_adjust_current_positions_for_backward_compatibility",
            side_effect=mock_adjust_positions,
        ), patch.object(
            self.behaviour.current_behaviour.context, "benchmark_tool", mock_benchmark
        ), patch.object(
            self.behaviour.current_behaviour, "send_a2a_transaction"
        ), patch.object(
            self.behaviour.current_behaviour, "wait_until_round_end"
        ), patch.object(
            self.behaviour.current_behaviour, "set_done"
        ):
            generator = self.behaviour.current_behaviour.async_act()

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify benchmark tool was called correctly
            mock_benchmark.measure.assert_called_with(
                self.behaviour.current_behaviour.behaviour_id
            )
            mock_benchmark.measure.return_value.local.assert_called_once()
            mock_benchmark.measure.return_value.consensus.assert_called_once()

            # Verify context managers were entered
            mock_local_context.__enter__.assert_called_once()
            mock_consensus_context.__enter__.assert_called_once()

    def test_async_act_set_done_called(self) -> None:
        """Test async_act calls set_done at the end."""
        mock_balances = [{"chain": "optimism", "assets": []}]

        def mock_get_positions():
            yield
            return mock_balances

        def mock_adjust_positions(positions):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "get_positions",
            side_effect=mock_get_positions,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_adjust_current_positions_for_backward_compatibility",
            side_effect=mock_adjust_positions,
        ), patch.object(
            self.behaviour.current_behaviour, "send_a2a_transaction"
        ), patch.object(
            self.behaviour.current_behaviour, "wait_until_round_end"
        ), patch.object(
            self.behaviour.current_behaviour, "set_done"
        ) as mock_set_done:
            generator = self.behaviour.current_behaviour.async_act()

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify set_done was called
            mock_set_done.assert_called_once()

    def test_matching_round_attribute(self) -> None:
        """Test matching_round is correctly set."""
        assert self.behaviour.current_behaviour.matching_round == GetPositionsRound

    def test_current_positions_class_attribute(self) -> None:
        """Test current_positions class attribute is None."""
        assert GetPositionsBehaviour.current_positions is None


# ==================== STANDALONE PAYLOAD TESTS ====================


def test_payload_creation_with_realistic_balance_data() -> None:
    """Test GetPositionsPayload creation with realistic balance data."""
    balance_data = [
        {
            "chain": "optimism",
            "assets": [
                {
                    "asset_symbol": "USDC",
                    "asset_type": "erc_20",
                    "address": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
                    "balance": 1000000000,
                },
                {
                    "asset_symbol": "ETH",
                    "asset_type": "native",
                    "address": "0x0000000000000000000000000000000000000000",
                    "balance": 500000000000000000,
                },
            ],
        }
    ]

    sender = "agent_0x123456789abcdef"
    serialized_positions = json.dumps(balance_data, sort_keys=True)
    payload = GetPositionsPayload(sender=sender, positions=serialized_positions)

    assert payload.sender == sender
    assert payload.positions == serialized_positions
    assert json.loads(payload.positions) == balance_data


def test_payload_creation_with_error_payload() -> None:
    """Test GetPositionsPayload creation with ERROR_PAYLOAD."""
    sender = "agent_0x123456789abcdef"
    error_payload = GetPositionsRound.ERROR_PAYLOAD
    serialized_error = json.dumps(error_payload, sort_keys=True)
    payload = GetPositionsPayload(sender=sender, positions=serialized_error)

    assert payload.sender == sender
    assert payload.positions == serialized_error
    assert json.loads(payload.positions) == error_payload

# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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

"""Test the states/decision_making.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import json
from unittest.mock import MagicMock, PropertyMock, patch

from packages.valory.skills.abstract_round_abci.base import (
    CollectSameUntilThresholdRound,
)
from packages.valory.skills.liquidity_trader_abci.payloads import DecisionMakingPayload
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)
from packages.valory.skills.liquidity_trader_abci.states.decision_making import (
    DecisionMakingRound,
)


def test_import() -> None:
    """Test that the decision_making module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.states.decision_making  # noqa


class TestDecisionMakingRound:
    """Test DecisionMakingRound class."""

    def test_payload_class(self) -> None:
        """Test payload_class attribute."""
        assert DecisionMakingRound.payload_class is DecisionMakingPayload

    def test_done_event(self) -> None:
        """Test done_event attribute."""
        assert DecisionMakingRound.done_event == Event.DONE

    def test_settle_event(self) -> None:
        """Test settle_event attribute."""
        assert DecisionMakingRound.settle_event == Event.SETTLE

    def test_update_event(self) -> None:
        """Test update_event attribute."""
        assert DecisionMakingRound.update_event == Event.UPDATE

    def test_error_event(self) -> None:
        """Test error_event attribute."""
        assert DecisionMakingRound.error_event == Event.ERROR

    def test_end_block_threshold_reached_done(self) -> None:
        """Test end_block returns DONE event when threshold reached."""
        round_obj = object.__new__(DecisionMakingRound)
        payload = json.dumps({"event": "done", "updates": {"chain_id": "1"}})

        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.actions = []
        mock_synced.last_executed_action_index = None

        type(round_obj).threshold_reached = PropertyMock(return_value=True)
        type(round_obj).most_voted_payload = PropertyMock(return_value=payload)
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)
        mock_synced.update.return_value = mock_synced

        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.DONE

    def test_end_block_threshold_reached_settle(self) -> None:
        """Test end_block returns SETTLE event."""
        round_obj = object.__new__(DecisionMakingRound)
        payload = json.dumps({"event": "settle", "updates": {"chain_id": "1"}})

        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.actions = []
        mock_synced.last_executed_action_index = None

        type(round_obj).threshold_reached = PropertyMock(return_value=True)
        type(round_obj).most_voted_payload = PropertyMock(return_value=payload)
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)
        mock_synced.update.return_value = mock_synced

        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.SETTLE

    def test_end_block_with_new_action(self) -> None:
        """Test end_block inserts new action into actions list."""
        round_obj = object.__new__(DecisionMakingRound)
        new_action = {"action": "enter_pool"}
        payload = json.dumps({
            "event": "done",
            "updates": {"new_action": new_action, "chain_id": "1"},
        })

        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.actions = [{"action": "existing"}]
        mock_synced.last_executed_action_index = 0

        type(round_obj).threshold_reached = PropertyMock(return_value=True)
        type(round_obj).most_voted_payload = PropertyMock(return_value=payload)
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)
        mock_synced.update.return_value = mock_synced

        result = round_obj.end_block()
        assert result is not None

    def test_end_block_with_new_action_no_index(self) -> None:
        """Test end_block inserts new action at index 0 when last index is None."""
        round_obj = object.__new__(DecisionMakingRound)
        new_action = {"action": "enter_pool"}
        payload = json.dumps({
            "event": "update",
            "updates": {"new_action": new_action},
        })

        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.actions = []
        mock_synced.last_executed_action_index = None

        type(round_obj).threshold_reached = PropertyMock(return_value=True)
        type(round_obj).most_voted_payload = PropertyMock(return_value=payload)
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)
        mock_synced.update.return_value = mock_synced

        result = round_obj.end_block()
        assert result is not None

    def test_end_block_with_positions_serialization(self) -> None:
        """Test end_block serializes positions if they are not a string."""
        round_obj = object.__new__(DecisionMakingRound)
        payload = json.dumps({
            "event": "done",
            "updates": {"positions": [{"pool": "test"}], "chain_id": "1"},
        })

        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.actions = []
        mock_synced.last_executed_action_index = None

        type(round_obj).threshold_reached = PropertyMock(return_value=True)
        type(round_obj).most_voted_payload = PropertyMock(return_value=payload)
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)
        mock_synced.update.return_value = mock_synced

        result = round_obj.end_block()
        assert result is not None

    def test_end_block_no_majority(self) -> None:
        """Test end_block returns NO_MAJORITY when majority not possible."""
        round_obj = object.__new__(DecisionMakingRound)

        type(round_obj).threshold_reached = PropertyMock(return_value=False)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.nb_participants = 4
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)
        type(round_obj).collection = PropertyMock(return_value={})

        with patch.object(
            CollectSameUntilThresholdRound,
            "is_majority_possible",
            return_value=False,
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.NO_MAJORITY)

    def test_end_block_waiting_for_threshold(self) -> None:
        """Test end_block returns None when waiting for threshold."""
        round_obj = object.__new__(DecisionMakingRound)

        type(round_obj).threshold_reached = PropertyMock(return_value=False)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.nb_participants = 4
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)
        type(round_obj).collection = PropertyMock(return_value={})

        with patch.object(
            CollectSameUntilThresholdRound,
            "is_majority_possible",
            return_value=True,
        ):
            result = round_obj.end_block()
        assert result is None

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

"""Test the states/evaluate_strategy.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from unittest.mock import MagicMock, patch

from packages.valory.skills.abstract_round_abci.base import (
    CollectSameUntilThresholdRound,
)
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)
from packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy import (
    EvaluateStrategyRound,
)


def test_import() -> None:
    """Test that the evaluate_strategy module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy  # noqa


class TestEvaluateStrategyRound:
    """Test EvaluateStrategyRound class."""

    def test_end_block_none_from_super(self) -> None:
        """Test end_block returns None when super returns None."""
        round_obj = object.__new__(EvaluateStrategyRound)
        with patch.object(
            CollectSameUntilThresholdRound, "end_block", return_value=None
        ):
            result = round_obj.end_block()
        assert result is None

    def test_end_block_non_done_event(self) -> None:
        """Test end_block returns super result when event is not DONE."""
        round_obj = object.__new__(EvaluateStrategyRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.actions = []
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.NO_MAJORITY),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.NO_MAJORITY)

    def test_end_block_done_with_empty_actions(self) -> None:
        """Test end_block returns WAIT when actions are empty."""
        round_obj = object.__new__(EvaluateStrategyRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.actions = []
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.WAIT)

    def test_end_block_done_with_actions(self) -> None:
        """Test end_block returns DONE when actions are present."""
        round_obj = object.__new__(EvaluateStrategyRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.actions = [{"action": "enter"}]
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.DONE)

    def test_end_block_withdrawal_initiated(self) -> None:
        """Test end_block detects withdrawal initiation from actions data."""
        round_obj = object.__new__(EvaluateStrategyRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.actions = {"event": Event.WITHDRAWAL_INITIATED.value}
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.WITHDRAWAL_INITIATED)

    def test_end_block_actions_dict_without_withdrawal(self) -> None:
        """Test end_block with actions as dict but not withdrawal event."""
        round_obj = object.__new__(EvaluateStrategyRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.actions = {"event": "some_other_event"}
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        # dict is truthy but event != WITHDRAWAL_INITIATED, falls through to DONE
        assert result == (mock_synced, Event.DONE)

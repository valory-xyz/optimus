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

"""Test the states/fetch_strategies.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import json
from unittest.mock import MagicMock, PropertyMock, patch

from packages.valory.skills.abstract_round_abci.base import (
    CollectSameUntilThresholdRound,
)
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)
from packages.valory.skills.liquidity_trader_abci.states.fetch_strategies import (
    FetchStrategiesRound,
)


def test_import() -> None:
    """Test that the fetch_strategies module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.states.fetch_strategies  # noqa


class TestFetchStrategiesRound:
    """Test FetchStrategiesRound class."""

    def test_end_block_withdrawal_initiated(self) -> None:
        """Test end_block returns WITHDRAWAL_INITIATED."""
        round_obj = object.__new__(FetchStrategiesRound)
        payload = json.dumps({"event": Event.WITHDRAWAL_INITIATED.value})

        mock_synced = MagicMock(spec=SynchronizedData)

        type(round_obj).threshold_reached = PropertyMock(return_value=True)
        type(round_obj).most_voted_payload = PropertyMock(return_value=payload)
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)

        result = round_obj.end_block()
        assert result == (mock_synced, Event.WITHDRAWAL_INITIATED)

    def test_end_block_settle(self) -> None:
        """Test end_block returns SETTLE with updates."""
        round_obj = object.__new__(FetchStrategiesRound)
        payload = json.dumps(
            {
                "event": Event.SETTLE.value,
                "updates": {"chain_id": "1"},
            }
        )

        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.update.return_value = mock_synced

        type(round_obj).threshold_reached = PropertyMock(return_value=True)
        type(round_obj).most_voted_payload = PropertyMock(return_value=payload)
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)

        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.SETTLE

    def test_end_block_done_with_protocols(self) -> None:
        """Test end_block returns DONE with selected protocols."""
        round_obj = object.__new__(FetchStrategiesRound)
        payload = json.dumps(
            {
                "selected_protocols": ["balancerPool"],
                "trading_type": "pool_manager",
            }
        )

        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.update.return_value = mock_synced

        type(round_obj).threshold_reached = PropertyMock(return_value=True)
        type(round_obj).most_voted_payload = PropertyMock(return_value=payload)
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)

        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.DONE

    def test_end_block_wait_empty_protocols(self) -> None:
        """Test end_block returns WAIT when no protocols selected."""
        round_obj = object.__new__(FetchStrategiesRound)
        payload = json.dumps(
            {
                "selected_protocols": [],
                "trading_type": "",
            }
        )

        mock_synced = MagicMock(spec=SynchronizedData)

        type(round_obj).threshold_reached = PropertyMock(return_value=True)
        type(round_obj).most_voted_payload = PropertyMock(return_value=payload)
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)

        result = round_obj.end_block()
        assert result == (mock_synced, Event.WAIT)

    def test_end_block_wait_no_trading_type(self) -> None:
        """Test end_block returns WAIT when trading_type is empty."""
        round_obj = object.__new__(FetchStrategiesRound)
        payload = json.dumps(
            {
                "selected_protocols": ["balancerPool"],
                "trading_type": "",
            }
        )

        mock_synced = MagicMock(spec=SynchronizedData)

        type(round_obj).threshold_reached = PropertyMock(return_value=True)
        type(round_obj).most_voted_payload = PropertyMock(return_value=payload)
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)

        result = round_obj.end_block()
        assert result == (mock_synced, Event.WAIT)

    def test_end_block_no_majority(self) -> None:
        """Test end_block returns NO_MAJORITY."""
        round_obj = object.__new__(FetchStrategiesRound)

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

    def test_end_block_waiting(self) -> None:
        """Test end_block returns None while waiting."""
        round_obj = object.__new__(FetchStrategiesRound)

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

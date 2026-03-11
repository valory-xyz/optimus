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

"""Test the states/withdraw_funds.py module of the liquidity_trader_abci skill."""

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
from packages.valory.skills.liquidity_trader_abci.states.withdraw_funds import (
    WithdrawFundsRound,
)


def test_import() -> None:
    """Test that the withdraw_funds module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.states.withdraw_funds  # noqa


class TestWithdrawFundsRound:
    """Test WithdrawFundsRound class."""

    def test_done_event(self) -> None:
        """Test done_event attribute."""
        assert WithdrawFundsRound.done_event == Event.DONE

    def test_none_event(self) -> None:
        """Test none_event attribute."""
        assert WithdrawFundsRound.none_event == Event.NONE

    def test_no_majority_event(self) -> None:
        """Test no_majority_event attribute."""
        assert WithdrawFundsRound.no_majority_event == Event.NO_MAJORITY

    def test_end_block_threshold_reached(self) -> None:
        """Test end_block returns DONE when threshold reached."""
        round_obj = object.__new__(WithdrawFundsRound)
        withdrawal_actions = [{"action": "withdraw_from_pool"}]

        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.update.return_value = mock_synced

        type(round_obj).threshold_reached = PropertyMock(return_value=True)
        type(round_obj).most_voted_payload = PropertyMock(
            return_value=json.dumps(withdrawal_actions)
        )
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)

        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.DONE

    def test_end_block_no_majority(self) -> None:
        """Test end_block returns NO_MAJORITY when majority not possible."""
        round_obj = object.__new__(WithdrawFundsRound)

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
        """Test end_block returns None while waiting for threshold."""
        round_obj = object.__new__(WithdrawFundsRound)

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

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

"""Test the states/call_checkpoint.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from unittest.mock import MagicMock, patch

from packages.valory.skills.abstract_round_abci.base import (
    CollectSameUntilThresholdRound,
)
from packages.valory.skills.liquidity_trader_abci.payloads import CallCheckpointPayload
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    StakingState,
    SynchronizedData,
)
from packages.valory.skills.liquidity_trader_abci.states.call_checkpoint import (
    CallCheckpointRound,
)


def test_import() -> None:
    """Test that the call_checkpoint module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.states.call_checkpoint  # noqa


class TestCallCheckpointRound:
    """Test CallCheckpointRound class."""

    def test_payload_class(self) -> None:
        """Test payload_class attribute."""
        assert CallCheckpointRound.payload_class is CallCheckpointPayload

    def test_done_event(self) -> None:
        """Test done_event attribute."""
        assert CallCheckpointRound.done_event == Event.DONE

    def test_end_block_none_from_super(self) -> None:
        """Test end_block returns None when super returns None."""
        round_obj = object.__new__(CallCheckpointRound)
        with patch.object(
            CollectSameUntilThresholdRound, "end_block", return_value=None
        ):
            result = round_obj.end_block()
        assert result is None

    def test_end_block_non_done_event(self) -> None:
        """Test end_block returns super result when event is not DONE."""
        round_obj = object.__new__(CallCheckpointRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.NO_MAJORITY),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.NO_MAJORITY)

    def test_end_block_withdrawal_initiated(self) -> None:
        """Test end_block returns WITHDRAWAL_INITIATED event."""
        round_obj = object.__new__(CallCheckpointRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.WITHDRAWAL_INITIATED),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.WITHDRAWAL_INITIATED)

    def test_end_block_staked_no_tx_hash(self) -> None:
        """Test end_block with STAKED and no tx_hash returns NEXT_CHECKPOINT_NOT_REACHED_YET."""
        round_obj = object.__new__(CallCheckpointRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.service_staking_state = StakingState.STAKED.value
        mock_synced.most_voted_tx_hash = None
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.NEXT_CHECKPOINT_NOT_REACHED_YET)

    def test_end_block_staked_with_tx_hash(self) -> None:
        """Test end_block with STAKED and tx_hash returns SETTLE."""
        round_obj = object.__new__(CallCheckpointRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.service_staking_state = StakingState.STAKED.value
        mock_synced.most_voted_tx_hash = "0xhash"
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.SETTLE)

    def test_end_block_unstaked(self) -> None:
        """Test end_block with UNSTAKED returns SERVICE_NOT_STAKED."""
        round_obj = object.__new__(CallCheckpointRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.service_staking_state = StakingState.UNSTAKED.value
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.SERVICE_NOT_STAKED)

    def test_end_block_evicted(self) -> None:
        """Test end_block with EVICTED returns SERVICE_EVICTED."""
        round_obj = object.__new__(CallCheckpointRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.service_staking_state = StakingState.EVICTED.value
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.SERVICE_EVICTED)

    def test_end_block_unknown_staking_state(self) -> None:
        """Test end_block with unknown staking state returns original result."""
        round_obj = object.__new__(CallCheckpointRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.service_staking_state = 99  # Unknown state
        mock_synced.most_voted_tx_hash = None
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        # Falls through all conditions, returns original res
        assert result == (mock_synced, Event.DONE)

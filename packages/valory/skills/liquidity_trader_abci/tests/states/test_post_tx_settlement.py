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

"""Test the states/post_tx_settlement.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from unittest.mock import MagicMock, PropertyMock, patch

from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)
from packages.valory.skills.liquidity_trader_abci.states.call_checkpoint import (
    CallCheckpointRound,
)
from packages.valory.skills.liquidity_trader_abci.states.check_staking_kpi_met import (
    CheckStakingKPIMetRound,
)
from packages.valory.skills.liquidity_trader_abci.states.decision_making import (
    DecisionMakingRound,
)
from packages.valory.skills.liquidity_trader_abci.states.fetch_strategies import (
    FetchStrategiesRound,
)
from packages.valory.skills.liquidity_trader_abci.states.post_tx_settlement import (
    PostTxSettlementRound,
)
from packages.valory.skills.liquidity_trader_abci.states.withdraw_funds import (
    WithdrawFundsRound,
)


def test_import() -> None:
    """Test that the post_tx_settlement module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.states.post_tx_settlement  # noqa


class TestPostTxSettlementRound:
    """Test PostTxSettlementRound class."""

    def test_done_event(self) -> None:
        """Test done_event attribute."""
        assert PostTxSettlementRound.done_event == Event.DONE

    def _make_round_with_submitter(self, submitter: str) -> PostTxSettlementRound:
        """Create a PostTxSettlementRound with a specific tx_submitter."""
        round_obj = object.__new__(PostTxSettlementRound)
        db = AbciAppDB(setup_data={"tx_submitter": [submitter]})
        synced = SynchronizedData(db=db)
        type(round_obj).threshold_reached = PropertyMock(return_value=True)
        type(round_obj).synchronized_data = PropertyMock(return_value=synced)
        return round_obj

    def test_end_block_checkpoint_executed(self) -> None:
        """Test end_block returns CHECKPOINT_TX_EXECUTED for CallCheckpointRound."""
        round_obj = self._make_round_with_submitter(
            CallCheckpointRound.auto_round_id()
        )
        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.CHECKPOINT_TX_EXECUTED

    def test_end_block_vanity_tx_executed(self) -> None:
        """Test end_block returns VANITY_TX_EXECUTED for CheckStakingKPIMetRound."""
        round_obj = self._make_round_with_submitter(
            CheckStakingKPIMetRound.auto_round_id()
        )
        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.VANITY_TX_EXECUTED

    def test_end_block_action_executed(self) -> None:
        """Test end_block returns ACTION_EXECUTED for DecisionMakingRound."""
        round_obj = self._make_round_with_submitter(
            DecisionMakingRound.auto_round_id()
        )
        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.ACTION_EXECUTED

    def test_end_block_transfer_completed(self) -> None:
        """Test end_block returns TRANSFER_COMPLETED for FetchStrategiesRound."""
        round_obj = self._make_round_with_submitter(
            FetchStrategiesRound.auto_round_id()
        )
        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.TRANSFER_COMPLETED

    def test_end_block_withdrawal_completed(self) -> None:
        """Test end_block returns WITHDRAWAL_COMPLETED for WithdrawFundsRound."""
        round_obj = self._make_round_with_submitter(
            WithdrawFundsRound.auto_round_id()
        )
        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.WITHDRAWAL_COMPLETED

    def test_end_block_unrecognized_submitter(self) -> None:
        """Test end_block returns UNRECOGNIZED for unknown submitter."""
        round_obj = self._make_round_with_submitter("unknown_round_id")
        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.UNRECOGNIZED

    def test_end_block_not_threshold_reached(self) -> None:
        """Test end_block returns None when threshold not reached."""
        round_obj = object.__new__(PostTxSettlementRound)
        type(round_obj).threshold_reached = PropertyMock(return_value=False)
        result = round_obj.end_block()
        assert result is None

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

from types import SimpleNamespace
from unittest.mock import PropertyMock

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
from packages.valory.skills.mech_interact_abci.states.purchase_subscription import (
    MechPurchaseSubscriptionRound,
)
from packages.valory.skills.mech_interact_abci.states.request import MechRequestRound


def test_import() -> None:
    """Test that the post_tx_settlement module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.states.post_tx_settlement  # noqa


class TestPostTxSettlementRound:
    """Test PostTxSettlementRound class."""

    def _make_round_with_submitter(self, submitter: str) -> PostTxSettlementRound:
        """Create a PostTxSettlementRound with a specific tx_submitter."""
        round_obj = object.__new__(PostTxSettlementRound)
        db = AbciAppDB(setup_data={"tx_submitter": [submitter]})
        synced = SynchronizedData(db=db)
        type(round_obj).threshold_reached = PropertyMock(return_value=True)  # type: ignore[method-assign]
        type(round_obj).synchronized_data = PropertyMock(return_value=synced)  # type: ignore[method-assign]
        # ``end_block`` logs at WARNING on the UNRECOGNIZED branch via
        # self.context.logger; stub a no-op logger so the call is safe.
        round_obj.context = SimpleNamespace(  # type: ignore[attr-defined]
            logger=SimpleNamespace(
                warning=lambda *a, **k: None,
                info=lambda *a, **k: None,
                error=lambda *a, **k: None,
                debug=lambda *a, **k: None,
            )
        )
        return round_obj

    def test_end_block_checkpoint_executed(self) -> None:
        """Test end_block returns CHECKPOINT_TX_EXECUTED for CallCheckpointRound."""
        round_obj = self._make_round_with_submitter(CallCheckpointRound.auto_round_id())
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

    def test_end_block_mech_request_tx_executed(self) -> None:
        """New-regime activity tx (MechRequestRound) maps to MECH_REQUEST_TX_EXECUTED."""
        round_obj = self._make_round_with_submitter(MechRequestRound.auto_round_id())
        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.MECH_REQUEST_TX_EXECUTED

    def test_end_block_mech_subscription_tx_executed(self) -> None:
        """Subscription purchase round maps back to the staking loop, not UNRECOGNIZED."""
        round_obj = self._make_round_with_submitter(
            MechPurchaseSubscriptionRound.auto_round_id()
        )
        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.MECH_REQUEST_TX_EXECUTED

    def test_end_block_offchain_mech_deposit_settled(self) -> None:
        """Off-chain deposit retry sentinel maps to OFFCHAIN_MECH_DEPOSIT_SETTLED.

        Without this dispatch the multiplexer would not recognise the
        ``OFFCHAIN_DEPOSIT_TX_SUBMITTER`` sentinel and fall through to
        ``Event.UNRECOGNIZED``, routing the settled deposit into
        ``FailedMultiplexerRound`` and dropping the retry. The dedicated
        event is what lets the composition route back into
        ``MechRequestRound`` for ``_retry_pending`` to fire.
        """
        from packages.valory.skills.mech_interact_abci.states.request import (
            OFFCHAIN_DEPOSIT_TX_SUBMITTER,
        )

        round_obj = self._make_round_with_submitter(OFFCHAIN_DEPOSIT_TX_SUBMITTER)
        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.OFFCHAIN_MECH_DEPOSIT_SETTLED

    def test_end_block_action_executed(self) -> None:
        """Test end_block returns ACTION_EXECUTED for DecisionMakingRound."""
        round_obj = self._make_round_with_submitter(DecisionMakingRound.auto_round_id())
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
        round_obj = self._make_round_with_submitter(WithdrawFundsRound.auto_round_id())
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

    def test_end_block_returns_none_when_still_collecting(self) -> None:
        """end_block returns None when threshold not reached and majority still possible."""
        round_obj = object.__new__(PostTxSettlementRound)
        db = AbciAppDB(
            setup_data={"tx_submitter": [DecisionMakingRound.auto_round_id()]}
        )
        synced = SynchronizedData(db=db)
        type(round_obj).threshold_reached = PropertyMock(return_value=False)  # type: ignore[method-assign]
        type(round_obj).synchronized_data = PropertyMock(return_value=synced)  # type: ignore[method-assign]
        type(round_obj).collection = PropertyMock(return_value={})  # type: ignore[method-assign]
        round_obj.is_majority_possible = lambda collection, n: True  # type: ignore[method-assign]
        assert round_obj.end_block() is None

    def test_end_block_returns_no_majority_when_majority_impossible(self) -> None:
        """end_block returns NO_MAJORITY when majority is no longer possible."""
        round_obj = object.__new__(PostTxSettlementRound)
        db = AbciAppDB(
            setup_data={"tx_submitter": [DecisionMakingRound.auto_round_id()]}
        )
        synced = SynchronizedData(db=db)
        type(round_obj).threshold_reached = PropertyMock(return_value=False)  # type: ignore[method-assign]
        type(round_obj).synchronized_data = PropertyMock(return_value=synced)  # type: ignore[method-assign]
        type(round_obj).collection = PropertyMock(return_value={})  # type: ignore[method-assign]
        round_obj.is_majority_possible = lambda collection, n: False  # type: ignore[method-assign]
        result = round_obj.end_block()
        assert result is not None
        _, event = result
        assert event == Event.NO_MAJORITY

    def test_no_withdrawal_initiated_class_attribute(self) -> None:
        """PostTxSettlement does not expose a withdrawal_initiated transition anchor."""  # noqa: D403
        assert "withdrawal_initiated" not in vars(PostTxSettlementRound)

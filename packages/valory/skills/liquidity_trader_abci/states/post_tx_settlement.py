# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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

"""This module contains the PostTxSettlementRound of LiquidityTraderAbciApp."""

from typing import Dict, Optional, Tuple

from packages.valory.skills.abstract_round_abci.base import (
    BaseSynchronizedData,
    COLLECTION_KEY_ATTRIBUTE,
    CollectSameUntilThresholdRound,
    NO_MAJORITY_EVENT_ATTRIBUTE,
    SELECTION_KEY_ATTRIBUTE,
    get_name,
)
from packages.valory.skills.liquidity_trader_abci.payloads import (
    PostTxSettlementPayload,
)
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
from packages.valory.skills.liquidity_trader_abci.states.withdraw_funds import (
    WithdrawFundsRound,
)
from packages.valory.skills.mech_interact_abci.states.purchase_subscription import (
    MechPurchaseSubscriptionRound,
)
from packages.valory.skills.mech_interact_abci.states.request import (
    OFFCHAIN_DEPOSIT_TX_SUBMITTER,
    MechRequestRound,
)


class PostTxSettlementRound(CollectSameUntilThresholdRound):
    """A round that will be called after tx settlement is done."""

    payload_class = PostTxSettlementPayload
    synchronized_data_class = SynchronizedData
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_post_tx_settlement)
    selection_key = (get_name(SynchronizedData.chain_id),)
    extended_requirements: Tuple[str, ...] = (
        NO_MAJORITY_EVENT_ATTRIBUTE,
        COLLECTION_KEY_ATTRIBUTE,
        SELECTION_KEY_ATTRIBUTE,
    )

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            submitter_to_event: Dict[str, Event] = {
                CallCheckpointRound.auto_round_id(): Event.CHECKPOINT_TX_EXECUTED,
                CheckStakingKPIMetRound.auto_round_id(): Event.VANITY_TX_EXECUTED,
                # New regime: the activity tx is submitted by mech_interact_abci's
                # MechRequestRound, so it must map back to the staking loop or the
                # multiplexer returns UNRECOGNIZED and the loop dies.
                MechRequestRound.auto_round_id(): Event.MECH_REQUEST_TX_EXECUTED,
                # Settled off-chain auto-deposit: re-enter MechRequestRound
                # for ``_retry_pending``, not forward to MechResponseRound.
                OFFCHAIN_DEPOSIT_TX_SUBMITTER: Event.OFFCHAIN_MECH_DEPOSIT_SETTLED,
                # Defensive: if the priority mech is ever a subscription-type mech,
                # the subscription purchase settles via MechPurchaseSubscriptionRound.
                # Unreachable with the current USDC mech config, but mapped so the
                # staking loop doesn't fall out to FailedMultiplexerRound.
                MechPurchaseSubscriptionRound.auto_round_id(): Event.MECH_REQUEST_TX_EXECUTED,
                DecisionMakingRound.auto_round_id(): Event.ACTION_EXECUTED,
                FetchStrategiesRound.auto_round_id(): Event.TRANSFER_COMPLETED,
                WithdrawFundsRound.auto_round_id(): Event.WITHDRAWAL_COMPLETED,
            }

            synced_data = SynchronizedData(self.synchronized_data.db)
            event = submitter_to_event.get(synced_data.tx_submitter, Event.UNRECOGNIZED)

            # An unrecognized submitter sends the round to
            # FailedMultiplexerRound and silently drops the settled tx —
            # including the off-chain deposit, where the funds have moved
            # but the retry never fires. Log the offending value at WARNING
            # so the cause is visible in agent logs instead of only in raw
            # Tendermint state.
            if event == Event.UNRECOGNIZED:
                self.context.logger.warning(
                    f"PostTxSettlementRound: unrecognized tx_submitter "
                    f"{synced_data.tx_submitter!r}; routing to "
                    f"FailedMultiplexerRound. Settled tx is dropped."
                )

            if event == Event.CHECKPOINT_TX_EXECUTED:
                synced_data = synced_data.update(  # type: ignore[assignment]
                    synchronized_data_class=SynchronizedData,
                    period_number_at_last_cp=synced_data.period_count,
                )

            return synced_data, event

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self.synchronized_data, Event.NO_MAJORITY
        return None

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

"""This module contains the CheckMechPreDepositRound of LiquidityTraderAbciApp."""

from typing import Optional, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    get_name,
)
from packages.valory.skills.liquidity_trader_abci.payloads import (
    CheckMechPreDepositPayload,
)
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
    peek_withdrawal_event,
)


class CheckMechPreDepositRound(CollectSameUntilThresholdRound):
    """Top the marketplace pre-deposit up when it is running low.

    Paid API calls are debited from the Safe's pre-deposit held by the
    marketplace balance tracker, not from the Safe's own token balance, and
    nothing else in this agent moves funds between the two. Without this the
    Safe can hold the payment token indefinitely while every paid call is
    refused for want of a deposit.
    """

    payload_class = CheckMechPreDepositPayload
    synchronized_data_class = SynchronizedData
    done_event: Event = Event.DONE
    no_majority_event: Event = Event.NO_MAJORITY
    none_event: Event = Event.NONE
    collection_key = get_name(SynchronizedData.participant_to_mech_pre_deposit)
    selection_key = (
        get_name(SynchronizedData.tx_submitter),
        get_name(SynchronizedData.most_voted_tx_hash),
        get_name(SynchronizedData.safe_contract_address),
        get_name(SynchronizedData.chain_id),
    )

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block.

        :return: the synchronized data and the event, or ``None``.
        """
        if peek_withdrawal_event(self) == Event.WITHDRAWAL_INITIATED.value:
            return self.synchronized_data, Event.WITHDRAWAL_INITIATED

        res = super().end_block()
        if res is None:
            return None

        synced_data, event = cast(Tuple[SynchronizedData, Event], res)

        # No hash means the pre-deposit already covers the target, so there is
        # nothing to settle and the period carries on.
        if event == self.done_event and synced_data.most_voted_tx_hash is not None:
            return synced_data, Event.SETTLE

        return res

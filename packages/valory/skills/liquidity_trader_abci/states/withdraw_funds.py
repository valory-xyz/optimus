# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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

"""This module contains the WithdrawFundsRound of LiquidityTraderAbciApp."""

from typing import Dict, Optional, Set, Tuple

from packages.valory.skills.abstract_round_abci.base import (
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    get_name,
)
from packages.valory.skills.liquidity_trader_abci.payloads import WithdrawFundsPayload
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)


class WithdrawFundsRound(CollectSameUntilThresholdRound):
    """A round that handles withdrawal operations."""

    payload_class = WithdrawFundsPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    none_event: Event = Event.NONE
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_withdraw_funds)
    selection_key = (get_name(SynchronizedData.withdrawal_actions),)

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            # Get the withdrawal actions from the majority payload
            majority_payload = self.majority_payload
            withdrawal_actions = majority_payload.withdrawal_actions
            
            # Create synchronized data with withdrawal actions
            synced_data = SynchronizedData(self.synchronized_data.db)
            synced_data.db.set("withdrawal_actions", withdrawal_actions)
            
            return synced_data, Event.DONE
        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return synced_data, Event.NO_MAJORITY
        return None 
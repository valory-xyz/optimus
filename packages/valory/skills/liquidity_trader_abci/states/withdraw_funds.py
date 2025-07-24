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

import json
from typing import Optional, Tuple, cast

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
    withdrawal_completed_event = Event.WITHDRAWAL_COMPLETED
    collection_key = get_name(SynchronizedData.participant_to_withdraw_funds)
    selection_key = (get_name(SynchronizedData.actions),)  # Use standard actions field

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            # Parse the payload as JSON - it contains the withdrawal actions list directly
            withdrawal_actions = json.loads(self.most_voted_payload)
            synchronized_data = cast(SynchronizedData, self.synchronized_data)

            # Store withdrawal actions in the standard actions field for normal flow processing
            synchronized_data = synchronized_data.update(
                synchronized_data_class=SynchronizedData,
                actions=json.dumps(withdrawal_actions),  # Use standard actions field
                last_action="WITHDRAWAL_INITIATED",
                last_executed_action_index=None,  # Reset action index for new actions
            )

            return synchronized_data, Event.DONE
        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self.synchronized_data, Event.NO_MAJORITY
        return None

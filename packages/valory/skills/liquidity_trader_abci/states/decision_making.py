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

"""This module contains the DecisionMakingRound of LiquidityTraderAbciApp."""

import json
from typing import Optional, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    get_name,
)
from packages.valory.skills.liquidity_trader_abci.payloads import DecisionMakingPayload
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)


class DecisionMakingRound(CollectSameUntilThresholdRound):
    """DecisionMakingRound"""

    payload_class = DecisionMakingPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    settle_event = Event.SETTLE
    update_event = Event.UPDATE
    error_event = Event.ERROR
    none_event: Event = Event.NONE
    no_majority_event = Event.NO_MAJORITY
    withdrawal_initiated: Event = Event.WITHDRAWAL_INITIATED
    collection_key = get_name(SynchronizedData.participant_to_decision_making)
    selection_key = (get_name(SynchronizedData.chain_id),)

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            # We reference all the events here to prevent the check-abciapp-specs tool from complaining
            payload = json.loads(self.most_voted_payload)
            event = Event(payload["event"])
            synchronized_data = cast(SynchronizedData, self.synchronized_data)

            # Ensure positions is always serialized
            positions = payload.get("updates", {}).get("positions", None)
            if positions and not isinstance(positions, str):
                payload["updates"]["positions"] = json.dumps(
                    positions, sort_keys=True, ensure_ascii=True
                )

            new_action = payload.get("updates", {}).get("new_action", {})
            updated_actions = self.synchronized_data.actions
            if new_action:
                if self.synchronized_data.last_executed_action_index is None:
                    index = 0
                else:
                    index = self.synchronized_data.last_executed_action_index + 1
                updated_actions.insert(index, new_action)

            serialized_actions = json.dumps(updated_actions, ensure_ascii=True)
            synchronized_data = synchronized_data.update(
                synchronized_data_class=SynchronizedData,
                **payload.get("updates", {}),
                actions=serialized_actions
            )
            return synchronized_data, event

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self.synchronized_data, Event.NO_MAJORITY
        return None

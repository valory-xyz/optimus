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

"""This module contains the FetchStrategiesRound of LiquidityTraderAbciApp."""

import json
from typing import Optional, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    get_name,
)
from packages.valory.skills.liquidity_trader_abci.payloads import FetchStrategiesPayload
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)


class FetchStrategiesRound(CollectSameUntilThresholdRound):
    """FetchStrategiesRound"""

    payload_class = FetchStrategiesPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    none_event: Event = Event.NONE
    settle_event = Event.SETTLE
    collection_key = get_name(SynchronizedData.participant_to_strategies_round)
    selection_key = (get_name(SynchronizedData.chain_id),)

    ERROR_PAYLOAD = {}

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            # We reference all the events here to prevent the check-abciapp-specs tool from complaining
            payload = json.loads(self.most_voted_payload)
            synchronized_data = cast(SynchronizedData, self.synchronized_data)

            # Check if this is a withdrawal initiation event
            if payload.get("event") == Event.WITHDRAWAL_INITIATED.value:
                return synchronized_data, Event.WITHDRAWAL_INITIATED

            # Check if this is an ETH transfer settlement event
            if payload.get("event") == Event.SETTLE.value:
                updates = payload.get("updates", {})
                synchronized_data = synchronized_data.update(
                    synchronized_data_class=SynchronizedData, **updates
                )
                return synchronized_data, Event.SETTLE

            # Original logic for normal strategy selection
            selected_protocols = payload.get("selected_protocols", [])
            trading_type = payload.get("trading_type", "")

            if not selected_protocols or not trading_type:
                return synchronized_data, Event.WAIT
            else:
                serialized_selected_protocols = json.dumps(selected_protocols)
                synchronized_data = synchronized_data.update(
                    synchronized_data_class=SynchronizedData,
                    selected_protocols=serialized_selected_protocols,
                    trading_type=trading_type,
                )
                return synchronized_data, Event.DONE

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self.synchronized_data, Event.NO_MAJORITY
        return None

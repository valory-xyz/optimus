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

"""This module contains the state for preparing a transaction for the next swap in the queue of instructions."""

from enum import Enum
from typing import Optional, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    get_name,
)
from packages.valory.skills.strategy_evaluator_abci.payloads import SendSwapProxyPayload
from packages.valory.skills.strategy_evaluator_abci.states.base import (
    Event,
    SynchronizedData,
)


class ProxySwapQueueRound(CollectSameUntilThresholdRound):
    """A round in which one agent utilizes the proxy server to perform the next swap transaction in priority."""

    payload_class = SendSwapProxyPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.PROXY_SWAPPED
    none_event = Event.PROXY_SWAP_FAILED
    no_majority_event = Event.NO_MAJORITY
    selection_key = get_name(SynchronizedData.tx_id)
    collection_key = get_name(SynchronizedData.participant_to_tx_preparation)

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        res = super().end_block()
        if res is None:
            return None

        synced_data, event = cast(Tuple[SynchronizedData, Enum], res)
        if event == self.done_event and synced_data.tx_id == "":
            return synced_data, Event.SWAPS_QUEUE_EMPTY
        return synced_data, event

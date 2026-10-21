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

"""This module contains the AgentTopUpRound of LiquidityTraderAbciApp."""

from typing import Optional, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    BaseSynchronizedData,
    COLLECTION_KEY_ATTRIBUTE,
    CollectSameUntilThresholdRound,
    DONE_EVENT_ATTRIBUTE,
    NO_MAJORITY_EVENT_ATTRIBUTE,
    SELECTION_KEY_ATTRIBUTE,
    get_name,
)
from packages.valory.skills.liquidity_trader_abci.payloads import AgentTopUpPayload
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)


class AgentTopUpRound(CollectSameUntilThresholdRound):
    """A round that moves native token from the Agent Safe to the Agent Signer."""

    payload_class = AgentTopUpPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_agent_topup)
    selection_key = (
        get_name(SynchronizedData.tx_submitter),
        get_name(SynchronizedData.most_voted_tx_hash),
        get_name(SynchronizedData.safe_contract_address),
        get_name(SynchronizedData.chain_id),
    )
    # The payload always carries the submitter, so the none event cannot fire.
    extended_requirements: Tuple[str, ...] = (
        DONE_EVENT_ATTRIBUTE,
        NO_MAJORITY_EVENT_ATTRIBUTE,
        COLLECTION_KEY_ATTRIBUTE,
        SELECTION_KEY_ATTRIBUTE,
    )

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        res = super().end_block()
        if res is None:
            return None

        synced_data, event = cast(Tuple[SynchronizedData, Event], res)
        if event == Event.DONE and synced_data.most_voted_tx_hash is not None:
            return synced_data, Event.SETTLE
        return synced_data, event

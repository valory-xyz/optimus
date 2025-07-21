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

"""This module contains the EvaluateStrategyRound of LiquidityTraderAbciApp."""

from typing import Optional, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    get_name,
)
from packages.valory.skills.liquidity_trader_abci.payloads import (
    EvaluateStrategyPayload,
)
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)


class EvaluateStrategyRound(CollectSameUntilThresholdRound):
    """EvaluateStrategyRound"""

    payload_class = EvaluateStrategyPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    none_event: Event = Event.NONE
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_actions_round)
    selection_key = get_name(SynchronizedData.actions)

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        res = super().end_block()
        if res is None:
            return None

        synced_data, event = cast(Tuple[SynchronizedData, Event], res)

        # Check if this is a withdrawal initiation event by examining the payload content
        if synced_data.actions:
            try:
                actions_data = synced_data.actions
                if (
                    isinstance(actions_data, dict)
                    and actions_data.get("event") == Event.WITHDRAWAL_INITIATED.value
                ):
                    return synced_data, Event.WITHDRAWAL_INITIATED
            except AttributeError:
                pass

        if event != Event.DONE:
            return res

        if not synced_data.actions:
            return synced_data, Event.WAIT

        return synced_data, Event.DONE

    # Event.NO_MAJORITY, Event.WAIT, Event.ROUND_TIMEOUT

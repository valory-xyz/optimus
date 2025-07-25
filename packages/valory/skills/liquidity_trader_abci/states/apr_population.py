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

"""This module contains the APRPopulationRound of LiquidityTraderAbciApp."""

from packages.valory.skills.abstract_round_abci.base import (
    CollectSameUntilThresholdRound,
    get_name,
)
from packages.valory.skills.liquidity_trader_abci.payloads import APRPopulationPayload
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)


class APRPopulationRound(CollectSameUntilThresholdRound):
    """APRPopulationRound"""

    payload_class = APRPopulationPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    none_event: Event = Event.NONE
    withdrawal_initiated: Event = Event.WITHDRAWAL_INITIATED
    collection_key = get_name(SynchronizedData.participant_to_context_round)
    selection_key = get_name(SynchronizedData.context)

    ERROR_PAYLOAD = {}

    # Event.ROUND_TIMEOUT

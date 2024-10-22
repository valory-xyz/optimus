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

"""This module contains the swap(s) instructions' preparation state of the strategy evaluator abci app."""

from packages.valory.skills.abstract_round_abci.base import (
    CollectSameUntilThresholdRound,
    get_name,
)
from packages.valory.skills.strategy_evaluator_abci.payloads import (
    TransactionHashPayload,
)
from packages.valory.skills.strategy_evaluator_abci.states.base import (
    Event,
    IPFSRound,
    SynchronizedData,
)


class PrepareSwapRound(IPFSRound):
    """A round in which the agents prepare swap(s) instructions."""

    done_event = Event.INSTRUCTIONS_PREPARED
    incomplete_event = Event.INCOMPLETE_INSTRUCTIONS_PREPARED
    no_hash_event = Event.NO_INSTRUCTIONS
    none_event = Event.ERROR_PREPARING_INSTRUCTIONS
    selection_key = (
        get_name(SynchronizedData.instructions_hash),
        get_name(SynchronizedData.incomplete_instructions),
    )
    collection_key = get_name(SynchronizedData.participant_to_instructions)


class PrepareEvmSwapRound(CollectSameUntilThresholdRound):
    """A round in which agents compute the transaction hash."""

    payload_class = TransactionHashPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.TRANSACTION_PREPARED
    none_event = Event.NO_INSTRUCTIONS
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_signature)
    selection_key = get_name(SynchronizedData.most_voted_tx_hash)

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

"""This package contains the rounds of MarketDataFetcherAbciApp."""

from enum import Enum
from typing import Dict, FrozenSet, Set, Type

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AppState,
    BaseSynchronizedData,
    BaseTxPayload,
    CollectSameUntilThresholdRound,
    CollectionRound,
    DegenerateRound,
    DeserializedCollection,
    EventToTimeout,
    get_name,
)
from packages.valory.skills.market_data_fetcher_abci.payloads import (
    MarketDataPayload,
    TransformedMarketDataPayload,
)


class Event(Enum):
    """MarketDataFetcherAbciApp Events"""

    DONE = "done"
    NONE = "none"
    NO_MAJORITY = "no_majority"
    ROUND_TIMEOUT = "round_timeout"


class SynchronizedData(BaseSynchronizedData):
    """
    Class to represent the synchronized data.

    This data is replicated by the tendermint application.
    """

    def _get_deserialized(self, key: str) -> DeserializedCollection:
        """Strictly get a collection and return it deserialized."""
        serialized = self.db.get_strict(key)
        return CollectionRound.deserialize_collection(serialized)

    @property
    def data_hash(self) -> str:
        """Get the hash of the tokens' data."""
        return str(self.db.get_strict("data_hash"))

    @property
    def transformed_data_hash(self) -> str:
        """Get the hash of the tokens' data."""
        return str(self.db.get_strict("transformed_data_hash"))

    @property
    def participant_to_fetching(self) -> DeserializedCollection:
        """Get the participants to market fetching."""
        return self._get_deserialized("participant_to_fetching")

    @property
    def participant_to_transforming(self) -> DeserializedCollection:
        """Get the participants to market data transformation."""
        return self._get_deserialized("participant_to_transforming")

    @property
    def selected_strategy(self) -> str:
        """Get the selected strategy."""
        return self.db.get_strict("selected_strategy")


class FetchMarketDataRound(CollectSameUntilThresholdRound):
    """FetchMarketDataRound"""

    payload_class: Type[BaseTxPayload] = MarketDataPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    none_event = Event.NONE
    no_majority_event = Event.NO_MAJORITY
    selection_key = get_name(SynchronizedData.data_hash)
    collection_key = get_name(SynchronizedData.participant_to_fetching)


class TransformMarketDataRound(FetchMarketDataRound):
    """Round to transform the fetched signals."""

    payload_class = TransformedMarketDataPayload
    selection_key = get_name(SynchronizedData.transformed_data_hash)
    collection_key = get_name(SynchronizedData.participant_to_transforming)


class FinishedMarketFetchRound(DegenerateRound):
    """FinishedMarketFetchRound"""


class FailedMarketFetchRound(DegenerateRound):
    """FailedMarketFetchRound"""


class MarketDataFetcherAbciApp(AbciApp[Event]):
    """MarketDataFetcherAbciApp

    Initial round: FetchMarketDataRound

    Initial states: {FetchMarketDataRound}

    Transition states:
        0. FetchMarketDataRound
            - done: 1.
            - none: 3.
            - no majority: 0.
            - round timeout: 0.
        1. TransformMarketDataRound
            - done: 2.
            - none: 3.
            - no majority: 1.
            - round timeout: 1.
        2. FinishedMarketFetchRound
        3. FailedMarketFetchRound

    Final states: {FailedMarketFetchRound, FinishedMarketFetchRound}

    Timeouts:
        round timeout: 30.0
    """

    initial_round_cls: AppState = FetchMarketDataRound
    initial_states: Set[AppState] = {FetchMarketDataRound}
    transition_function: AbciAppTransitionFunction = {
        FetchMarketDataRound: {
            Event.DONE: TransformMarketDataRound,
            Event.NONE: FailedMarketFetchRound,
            Event.NO_MAJORITY: FetchMarketDataRound,
            Event.ROUND_TIMEOUT: FetchMarketDataRound,
        },
        TransformMarketDataRound: {
            Event.DONE: FinishedMarketFetchRound,
            Event.NONE: FailedMarketFetchRound,
            Event.NO_MAJORITY: TransformMarketDataRound,
            Event.ROUND_TIMEOUT: TransformMarketDataRound,
        },
        FinishedMarketFetchRound: {},
        FailedMarketFetchRound: {},
    }
    final_states: Set[AppState] = {FinishedMarketFetchRound, FailedMarketFetchRound}
    event_to_timeout: EventToTimeout = {
        Event.ROUND_TIMEOUT: 30.0,
    }
    cross_period_persisted_keys: FrozenSet[str] = frozenset()
    db_pre_conditions: Dict[AppState, Set[str]] = {
        FetchMarketDataRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedMarketFetchRound: {get_name(SynchronizedData.data_hash)},
        FailedMarketFetchRound: set(),
    }

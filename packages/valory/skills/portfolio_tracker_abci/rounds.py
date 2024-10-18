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

"""This package contains the rounds of PortfolioTrackerAbciApp."""

from enum import Enum
from typing import Dict, Optional, Set, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AppState,
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    CollectionRound,
    DegenerateRound,
    DeserializedCollection,
    EventToTimeout,
    get_name,
)
from packages.valory.skills.portfolio_tracker_abci.payloads import (
    PortfolioTrackerPayload,
)


class Event(Enum):
    """PortfolioTrackerAbciApp Events"""

    DONE = "done"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    FAILED = "failed"
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
    def portfolio_hash(self) -> Optional[str]:
        """Get the hash of the portfolio's data."""
        return self.db.get_strict("portfolio_hash")

    @property
    def is_balance_sufficient(self) -> Optional[bool]:
        """Get whether the balance is sufficient."""
        return self.db.get("is_balance_sufficient", None)

    @property
    def participant_to_portfolio(self) -> DeserializedCollection:
        """Get the participants to portfolio tracking."""
        return self._get_deserialized("participant_to_portfolio")


class PortfolioTrackerRound(CollectSameUntilThresholdRound):
    """PortfolioTrackerRound"""

    payload_class = PortfolioTrackerPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    none_event = Event.FAILED
    no_majority_event = Event.NO_MAJORITY
    selection_key = (
        get_name(SynchronizedData.portfolio_hash),
        get_name(SynchronizedData.is_balance_sufficient),
    )
    collection_key = get_name(SynchronizedData.participant_to_portfolio)

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        res = super().end_block()
        if res is None:
            return None

        synced_data, event = cast(Tuple[SynchronizedData, Enum], res)
        if event == self.done_event and not synced_data.is_balance_sufficient:
            return synced_data, Event.INSUFFICIENT_BALANCE
        return synced_data, event


class FinishedPortfolioTrackerRound(DegenerateRound):
    """This class represents that the portfolio tracking has finished."""


class FailedPortfolioTrackerRound(DegenerateRound):
    """This class represents that the portfolio tracking has failed."""


class PortfolioTrackerAbciApp(AbciApp[Event]):
    """PortfolioTrackerAbciApp

    Initial round: PortfolioTrackerRound

    Initial states: {PortfolioTrackerRound}

    Transition states:
        0. PortfolioTrackerRound
            - done: 1.
            - failed: 2.
            - insufficient balance: 0.
            - no majority: 0.
            - round timeout: 0.
        1. FinishedPortfolioTrackerRound
        2. FailedPortfolioTrackerRound

    Final states: {FailedPortfolioTrackerRound, FinishedPortfolioTrackerRound}

    Timeouts:
        round timeout: 30.0
    """

    initial_round_cls: AppState = PortfolioTrackerRound
    initial_states: Set[AppState] = {PortfolioTrackerRound}
    transition_function: AbciAppTransitionFunction = {
        PortfolioTrackerRound: {
            Event.DONE: FinishedPortfolioTrackerRound,
            Event.FAILED: FailedPortfolioTrackerRound,
            Event.INSUFFICIENT_BALANCE: PortfolioTrackerRound,
            Event.NO_MAJORITY: PortfolioTrackerRound,
            Event.ROUND_TIMEOUT: PortfolioTrackerRound,
        },
        FinishedPortfolioTrackerRound: {},
        FailedPortfolioTrackerRound: {},
    }
    final_states: Set[AppState] = {
        FinishedPortfolioTrackerRound,
        FailedPortfolioTrackerRound,
    }
    event_to_timeout: EventToTimeout = {
        Event.ROUND_TIMEOUT: 30.0,
    }
    db_pre_conditions: Dict[AppState, Set[str]] = {
        PortfolioTrackerRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedPortfolioTrackerRound: {
            get_name(SynchronizedData.portfolio_hash),
            get_name(SynchronizedData.is_balance_sufficient),
            get_name(SynchronizedData.participant_to_portfolio),
        },
        FailedPortfolioTrackerRound: set(),
    }

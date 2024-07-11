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

"""This package contains the rounds of LiquidityTraderAbciApp."""

import json
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, cast

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
from packages.valory.skills.liquidity_trader_abci.payloads import (
    DecisionMakingPayload,
    EvaluateStrategyPayload,
    GetPositionsPayload,
)


class Event(Enum):
    """LiquidityTraderAbciApp Events"""

    ERROR = "error"
    ROUND_TIMEOUT = "round_timeout"
    NO_MAJORITY = "no_majority"
    DONE = "done"
    WAIT = "wait"
    SETTLE = "settle"


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
    def most_voted_tx_hash(self) -> Optional[float]:
        """Get the token most_voted_tx_hash."""
        return self.db.get("most_voted_tx_hash", None)

    @property
    def participant_to_tx_round(self) -> DeserializedCollection:
        """Get the participants to the tx round."""
        return self._get_deserialized("participant_to_tx_round")

    @property
    def tx_submitter(self) -> str:
        """Get the round that submitted a tx to transaction_settlement_abci."""
        return str(self.db.get_strict("tx_submitter"))

    @property
    def participant_to_positions_round(self) -> DeserializedCollection:
        """Get the participants to the positions round."""
        return self._get_deserialized("participant_to_positions_round")

    @property
    def positions(self) -> List[Dict[str, Any]]:
        """Get the positions."""
        serialized = self.db.get("positions", "[]")
        if serialized is None:
            serialized = "[]"
        positions = json.loads(serialized)
        return positions

    @property
    def current_pool(self) -> Dict[str, Any]:
        """Get the current pool"""
        serialized = self.db.get("current_pool", "{}")
        if serialized is None:
            serialized = "{}"
        positions = json.loads(serialized)
        return positions

    @property
    def participant_to_actions_round(self) -> DeserializedCollection:
        """Get the participants to actions rounds"""
        return self._get_deserialized("participant_to_actions_round")

    @property
    def actions(self) -> Optional[List[Dict[str, Any]]]:
        """Get the actions"""
        serialized = self.db.get("actions", "[]")
        if serialized is None:
            serialized = "[]"
        actions = json.loads(serialized)
        return actions

    @property
    def transaction_history(self) -> Optional[List[Dict[str, Any]]]:
        """Get the transactions"""
        serialized = self.db.get("transactions", "[]")
        if serialized is None:
            serialized = "[]"
        transactions = json.loads(serialized)
        return transactions

    @property
    def most_voted_tx_hash(self) -> Optional[float]:
        """Get the token most_voted_tx_hash."""
        return self.db.get("most_voted_tx_hash", None)


class GetPositionsRound(CollectSameUntilThresholdRound):
    """GetPositionsRound"""

    payload_class = GetPositionsPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_positions_round)
    selection_key = (
        get_name(SynchronizedData.positions),
        get_name(SynchronizedData.current_pool),
    )

    ERROR_PAYLOAD = {}


class EvaluateStrategyRound(CollectSameUntilThresholdRound):
    """EvaluateStrategyRound"""

    payload_class = EvaluateStrategyPayload
    synchronized_data_class = SynchronizedData
    selection_key = get_name(SynchronizedData.actions)
    done_event = Event.DONE
    collection_key = get_name(SynchronizedData.participant_to_actions_round)
    selection_key = get_name(SynchronizedData.actions)


class DecisionMakingRound(CollectSameUntilThresholdRound):
    """DecisionMakingRound"""

    payload_class = DecisionMakingPayload
    synchronized_data_class = SynchronizedData

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            # We reference all the events here to prevent the check-abciapp-specs tool from complaining
            payload = json.loads(self.most_voted_payload)
            event = Event(payload["event"])
            synchronized_data = cast(SynchronizedData, self.synchronized_data)

            synchronized_data = synchronized_data.update(
                synchronized_data_class=SynchronizedData, **payload.get("updates", {})
            )
            return synchronized_data, event

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self.synchronized_data, Event.NO_MAJORITY
        return None


class FinishedEvaluateStrategyRound(DegenerateRound):
    """FinishedEvaluateStrategyRound"""


class FinishedDecisionMakingRound(DegenerateRound):
    """FinishedDecisionMakingRound"""


class FinishedTxPreparationRound(DegenerateRound):
    """FinishedTxPreparationRound"""


class LiquidityTraderAbciApp(AbciApp[Event]):
    """LiquidityTraderAbciApp"""

    initial_round_cls: AppState = GetPositionsRound
    initial_states: Set[AppState] = {GetPositionsRound, DecisionMakingRound}
    transition_function: AbciAppTransitionFunction = {
        GetPositionsRound: {
            Event.DONE: EvaluateStrategyRound,
            Event.NO_MAJORITY: GetPositionsRound,
            Event.ROUND_TIMEOUT: GetPositionsRound,
        },
        EvaluateStrategyRound: {
            Event.DONE: DecisionMakingRound,
            Event.NO_MAJORITY: EvaluateStrategyRound,
            Event.ROUND_TIMEOUT: EvaluateStrategyRound,
            Event.WAIT: FinishedEvaluateStrategyRound,
        },
        DecisionMakingRound: {
            Event.DONE: FinishedDecisionMakingRound,
            Event.ERROR: FinishedDecisionMakingRound,
            Event.NO_MAJORITY: DecisionMakingRound,
            Event.ROUND_TIMEOUT: DecisionMakingRound,
            Event.SETTLE: FinishedTxPreparationRound,
        },
        FinishedEvaluateStrategyRound: {},
        FinishedTxPreparationRound: {},
        FinishedDecisionMakingRound: {},
    }
    final_states: Set[AppState] = {
        FinishedEvaluateStrategyRound,
        FinishedDecisionMakingRound,
        FinishedTxPreparationRound,
    }
    event_to_timeout: EventToTimeout = {}
    cross_period_persisted_keys: FrozenSet[str] = frozenset()
    db_pre_conditions: Dict[AppState, Set[str]] = {
        GetPositionsRound: set(),
        DecisionMakingRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedEvaluateStrategyRound: set(),
        FinishedDecisionMakingRound: set(),
        FinishedTxPreparationRound: {get_name(SynchronizedData.most_voted_tx_hash)},
    }

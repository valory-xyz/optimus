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

from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, cast
import json 

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AbstractRound,
    AppState,
    BaseSynchronizedData,
    CollectionRound,
    CollectSameUntilThresholdRound,
    DegenerateRound,
    DeserializedCollection,
    EventToTimeout,
    get_name,
)

from packages.valory.skills.liquidity_trader_abci.payloads import (
    ClaimOPPayload,
    DecisionMakingPayload,
    EvaluateStrategyPayload,
    GetPositionsPayload,
    PrepareExitPoolTxPayload,
    PrepareSwapTxPayload,
    TxPreparationPayload,
)


class Event(Enum):
    """LiquidityTraderAbciApp Events"""

    ERROR = "error"
    SWAP = "swap"
    ROUND_TIMEOUT = "round_timeout"
    NO_MAJORITY = "no_majority"
    EXIT = "exit"
    CLAIM = "claim"
    DONE = "done"
    WAIT = "wait"


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
    def positions(self) -> Optional[List[Dict[str, any]]]:
        """Get the positions."""
        serialized = self.db.get("positions", "[]")
        if serialized is None:
            serialized = "[]"
        positions = json.loads(serialized)
        return positions
    
    @property
    def participant_to_actions_round(self) -> DeserializedCollection:
        """Get the participants to actions rounds"""
        return self._get_deserialized("participant_to_actions_round")
    
    @property
    def actions(self) -> Optional[List[Dict[str, any]]]:
        """Get the actions"""
        serialized = self.db.get("actions", "[]")
        if serialized is None:
            serialized = "[]"
        actions = json.loads(serialized)
        return actions
    
    @property
    def transaction_history(self) -> Optional[List[Dict[str,any]]]:
        """Get the transactions"""
        serialized = self.db.get("transactions", "[]")
        if serialized is None:
            serialized = "[]"
        transactions = json.loads(serialized)
        return transactions

class ClaimOPRound(AbstractRound):
    """ClaimOPRound"""

    payload_class = ClaimOPPayload
    payload_attribute = ""  # TODO: update
    synchronized_data_class = SynchronizedData

    # TODO: replace AbstractRound with one of CollectDifferentUntilAllRound,
    # CollectSameUntilAllRound, CollectSameUntilThresholdRound,
    # CollectDifferentUntilThresholdRound, OnlyKeeperSendsRound, VotingRound,
    # from packages/valory/skills/abstract_round_abci/base.py
    # or implement the methods

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        raise NotImplementedError

    def check_payload(self, payload: ClaimOPPayload) -> None:
        """Check payload."""
        raise NotImplementedError

    def process_payload(self, payload: ClaimOPPayload) -> None:
        """Process payload."""
        raise NotImplementedError


class DecisionMakingRound(AbstractRound):
    """DecisionMakingRound"""

    payload_class = DecisionMakingPayload
    payload_attribute = ""  # TODO: update
    synchronized_data_class = SynchronizedData

    # TODO: replace AbstractRound with one of CollectDifferentUntilAllRound,
    # CollectSameUntilAllRound, CollectSameUntilThresholdRound,
    # CollectDifferentUntilThresholdRound, OnlyKeeperSendsRound, VotingRound,
    # from packages/valory/skills/abstract_round_abci/base.py
    # or implement the methods

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        raise NotImplementedError

    def check_payload(self, payload: DecisionMakingPayload) -> None:
        """Check payload."""
        raise NotImplementedError

    def process_payload(self, payload: DecisionMakingPayload) -> None:
        """Process payload."""
        raise NotImplementedError


class EvaluateStrategyRound(CollectSameUntilThresholdRound):
    """EvaluateStrategyRound"""

    payload_class = EvaluateStrategyPayload
    synchronized_data_class = SynchronizedData
    selection_key = (get_name(SynchronizedData.actions))
    done_event = Event.DONE
    collection_key = get_name(SynchronizedData.participant_to_actions_round)
    selection_key = get_name(SynchronizedData.actions)

    def end_block(self) -> Optional[Tuple[SynchronizedData, Enum]]:
        """Process the end of the block."""
        res = super().end_block()
        if res is None:
            return None

        synced_data, event = cast(Tuple[SynchronizedData, Enum], res)

        if event == Event.DONE and synced_data.actions is None:
            return synced_data, Event.WAIT

        return synced_data, event

class GetPositionsRound(CollectSameUntilThresholdRound):
    """GetPositionsRound"""

    payload_class = GetPositionsPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_positions_round)
    selection_key = (get_name(SynchronizedData.positions))

    ERROR_PAYLOAD = "error"

class PrepareExitPoolTxRound(AbstractRound):
    """PrepareExitPoolTxRound"""

    payload_class = PrepareExitPoolTxPayload
    payload_attribute = ""  # TODO: update
    synchronized_data_class = SynchronizedData

    # TODO: replace AbstractRound with one of CollectDifferentUntilAllRound,
    # CollectSameUntilAllRound, CollectSameUntilThresholdRound,
    # CollectDifferentUntilThresholdRound, OnlyKeeperSendsRound, VotingRound,
    # from packages/valory/skills/abstract_round_abci/base.py
    # or implement the methods

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        raise NotImplementedError

    def check_payload(self, payload: PrepareExitPoolTxPayload) -> None:
        """Check payload."""
        raise NotImplementedError

    def process_payload(self, payload: PrepareExitPoolTxPayload) -> None:
        """Process payload."""
        raise NotImplementedError


class PrepareSwapTxRound(AbstractRound):
    """PrepareSwapTxRound"""

    payload_class = PrepareSwapTxPayload
    payload_attribute = ""  # TODO: update
    synchronized_data_class = SynchronizedData

    # TODO: replace AbstractRound with one of CollectDifferentUntilAllRound,
    # CollectSameUntilAllRound, CollectSameUntilThresholdRound,
    # CollectDifferentUntilThresholdRound, OnlyKeeperSendsRound, VotingRound,
    # from packages/valory/skills/abstract_round_abci/base.py
    # or implement the methods

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        raise NotImplementedError

    def check_payload(self, payload: PrepareSwapTxPayload) -> None:
        """Check payload."""
        raise NotImplementedError

    def process_payload(self, payload: PrepareSwapTxPayload) -> None:
        """Process payload."""
        raise NotImplementedError


class TxPreparationRound(AbstractRound):
    """TxPreparationRound"""

    payload_class = TxPreparationPayload
    payload_attribute = ""  # TODO: update
    synchronized_data_class = SynchronizedData

    # TODO: replace AbstractRound with one of CollectDifferentUntilAllRound,
    # CollectSameUntilAllRound, CollectSameUntilThresholdRound,
    # CollectDifferentUntilThresholdRound, OnlyKeeperSendsRound, VotingRound,
    # from packages/valory/skills/abstract_round_abci/base.py
    # or implement the methods

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        raise NotImplementedError

    def check_payload(self, payload: TxPreparationPayload) -> None:
        """Check payload."""
        raise NotImplementedError

    def process_payload(self, payload: TxPreparationPayload) -> None:
        """Process payload."""
        raise NotImplementedError


class FinishedDecisionMakingRound(DegenerateRound):
    """FinishedDecisionMakingRound"""


class FinishedEvaluateStrategyRound(DegenerateRound):
    """FinishedEvaluateStrategyRound"""


class FinishedTxPreparationRound(DegenerateRound):
    """FinishedTxPreparationRound"""


class LiquidityTraderAbciApp(AbciApp[Event]):
    """LiquidityTraderAbciApp"""

    initial_round_cls: AppState = GetPositionsRound
    initial_states: Set[AppState] = {GetPositionsRound, DecisionMakingRound}
    transition_function: AbciAppTransitionFunction = {
        ClaimOPRound: {
            Event.DONE: TxPreparationRound,
            Event.NO_MAJORITY: ClaimOPRound,
            Event.ROUND_TIMEOUT: ClaimOPRound
        },
        GetPositionsRound: {
            Event.DONE: EvaluateStrategyRound,
            Event.NO_MAJORITY: GetPositionsRound,
            Event.ROUND_TIMEOUT: GetPositionsRound
        },
        EvaluateStrategyRound: {
            Event.DONE: DecisionMakingRound,
            Event.NO_MAJORITY: EvaluateStrategyRound,
            Event.ROUND_TIMEOUT: EvaluateStrategyRound,
            Event.WAIT: FinishedEvaluateStrategyRound
        },
        DecisionMakingRound: {
            Event.DONE: FinishedDecisionMakingRound,
            Event.ERROR: FinishedDecisionMakingRound,
            Event.NO_MAJORITY: DecisionMakingRound,
            Event.ROUND_TIMEOUT: DecisionMakingRound,
            Event.EXIT: PrepareExitPoolTxRound,
            Event.SWAP: PrepareSwapTxRound,
            Event.CLAIM: ClaimOPRound
        },
        PrepareExitPoolTxRound: {
            Event.DONE: TxPreparationRound,
            Event.NO_MAJORITY: PrepareExitPoolTxRound,
            Event.ROUND_TIMEOUT: PrepareExitPoolTxRound
        },
        PrepareSwapTxRound: {
            Event.DONE: TxPreparationRound,
            Event.NO_MAJORITY: PrepareSwapTxRound,
            Event.ROUND_TIMEOUT: PrepareSwapTxRound
        },
        TxPreparationRound: {
            Event.DONE: FinishedTxPreparationRound,
            Event.NO_MAJORITY: TxPreparationRound,
            Event.ROUND_TIMEOUT: TxPreparationRound
        },
        FinishedEvaluateStrategyRound: {},
        FinishedTxPreparationRound: {},
        FinishedDecisionMakingRound: {}
    }
    final_states: Set[AppState] = {FinishedEvaluateStrategyRound, FinishedDecisionMakingRound, FinishedTxPreparationRound}
    event_to_timeout: EventToTimeout = {}
    cross_period_persisted_keys: FrozenSet[str] = frozenset()
    db_pre_conditions: Dict[AppState, Set[str]] = {
        GetPositionsRound: set(),
        DecisionMakingRound: set()
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedEvaluateStrategyRound: set(),
    	FinishedDecisionMakingRound: set(),
    	FinishedTxPreparationRound: {get_name(SynchronizedData.most_voted_tx_hash)},
    }

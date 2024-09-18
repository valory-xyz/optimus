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
    CallCheckpointPayload,
    CheckStakingKPIMetPayload,
    DecisionMakingPayload,
    EvaluateStrategyPayload,
    GetPositionsPayload,
)


class StakingState(Enum):
    """Staking state enumeration for the staking."""

    UNSTAKED = 0
    STAKED = 1
    EVICTED = 2


class Event(Enum):
    """LiquidityTraderAbciApp Events"""

    ACTION_EXECUTED = "execute_next_action"
    CHECKPOINT_TX_EXECUTED = "checkpoint_tx_executed"
    DONE = "done"
    ERROR = "error"
    NEXT_CHECKPOINT_NOT_REACHED_YET = "next_checkpoint_not_reached_yet"
    NO_MAJORITY = "no_majority"
    ROUND_TIMEOUT = "round_timeout"
    SERVICE_EVICTED = "service_evicted"
    SERVICE_NOT_STAKED = "service_not_staked"
    SETTLE = "settle"
    UNRECOGNIZED = "unrecognized"
    UPDATE = "update"
    VANITY_TX_EXECUTED = "vanity_tx_executed"
    WAIT = "wait"
    STAKING_KPI_NOT_MET = "staking_kpi_not_met"
    STAKING_KPI_MET = "staking_kpi_met"
    WAIT_FOR_PERIODS_TO_PASS = "wait_for_periods_to_pass"


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
    def last_executed_action_index(self) -> Optional[int]:
        """Get the last executed action index"""
        return cast(int, self.db.get("last_executed_action_index", None))

    @property
    def final_tx_hash(self) -> str:
        """Get the verified tx hash."""
        return cast(str, self.db.get_strict("final_tx_hash"))

    @property
    def last_reward_claimed_timestamp(self) -> Optional[int]:
        """Get the last reward claimed timestamp."""
        return cast(int, self.db.get("last_reward_claimed_timestamp", None))

    @property
    def service_staking_state(self) -> Optional[int]:
        """Get the min number of safe tx required."""
        return cast(int, self.db.get("service_staking_state"))

    @property
    def min_num_of_safe_tx_required(self) -> Optional[int]:
        """Get the min number of safe tx required."""
        return cast(int, self.db.get("min_num_of_safe_tx_required"))

    @property
    def participant_to_checkpoint(self) -> DeserializedCollection:
        """Get the participants to the checkpoint round."""
        return self._get_deserialized("participant_to_checkpoint")

    @property
    def participant_to_staking_kpi(self) -> DeserializedCollection:
        """Get the participants to the CheckStakingKPIMet round."""
        return self._get_deserialized("participant_to_staking_kpi")

    @property
    def is_staking_kpi_met(self) -> Optional[bool]:
        """Get kpi met for the day."""
        return cast(int, self.db.get("is_staking_kpi_met", False))

    @property
    def chain_id(self) -> Optional[str]:
        """Get the chain id."""
        return cast(str, self.db.get("chain_id", None))
    
    @property
    def period_number_at_last_cp(self) -> Optional[int]:
        """Get the period number at last cp."""
        return cast(int, self.db.get("period_number_at_last_cp", 0))
    

class CallCheckpointRound(CollectSameUntilThresholdRound):
    """A round for the checkpoint call preparation."""

    payload_class = CallCheckpointPayload
    done_event: Enum = Event.DONE
    no_majority_event: Enum = Event.NO_MAJORITY
    selection_key = (
        get_name(SynchronizedData.tx_submitter),
        get_name(SynchronizedData.service_staking_state),
        get_name(SynchronizedData.min_num_of_safe_tx_required),
        get_name(SynchronizedData.is_staking_kpi_met),
        get_name(SynchronizedData.most_voted_tx_hash),
        get_name(SynchronizedData.safe_contract_address),
        get_name(SynchronizedData.chain_id),
    )
    collection_key = get_name(SynchronizedData.participant_to_checkpoint)
    synchronized_data_class = SynchronizedData

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        res = super().end_block()
        if res is None:
            return None

        synced_data, event = cast(Tuple[SynchronizedData, Enum], res)

        if event != Event.DONE:
            return res

        if (
            synced_data.service_staking_state == StakingState.STAKED.value
            and synced_data.most_voted_tx_hash is not None
        ):
            return synced_data, Event.SETTLE

        if synced_data.service_staking_state == StakingState.UNSTAKED.value:
            return synced_data, Event.SERVICE_NOT_STAKED

        if synced_data.service_staking_state == StakingState.EVICTED.value:
            return synced_data, Event.SERVICE_EVICTED

        if synced_data.is_staking_kpi_met is False:
            return synced_data, Event.STAKING_KPI_NOT_MET

        if synced_data.is_staking_kpi_met is True:
            return synced_data, Event.STAKING_KPI_MET

        return res


class CheckStakingKPIMetRound(CollectSameUntilThresholdRound):
    """CheckStakingKPIMetRound"""

    payload_class = CheckStakingKPIMetPayload
    synchronized_data_class = SynchronizedData
    done_event: Enum = Event.DONE
    no_majority_event: Enum = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_staking_kpi)
    selection_key = (
        get_name(SynchronizedData.tx_submitter),
        get_name(SynchronizedData.is_staking_kpi_met),
        get_name(SynchronizedData.most_voted_tx_hash),
        get_name(SynchronizedData.safe_contract_address),
        get_name(SynchronizedData.chain_id),
    )

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        res = super().end_block()
        if res is None:
            return None

        synced_data, event = cast(Tuple[SynchronizedData, Enum], res)

        if event != Event.DONE:
            return res

        if synced_data.most_voted_tx_hash is not None:
            return synced_data, Event.SETTLE

        if synced_data.is_staking_kpi_met is True:
            return synced_data, Event.STAKING_KPI_MET

        if synced_data.is_staking_kpi_met is False:
            return synced_data, Event.WAIT_FOR_PERIODS_TO_PASS
        
        return res

class GetPositionsRound(CollectSameUntilThresholdRound):
    """GetPositionsRound"""

    payload_class = GetPositionsPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_positions_round)
    selection_key = get_name(SynchronizedData.positions)

    ERROR_PAYLOAD = {}

    # Event.ROUND_TIMEOUT


class EvaluateStrategyRound(CollectSameUntilThresholdRound):
    """EvaluateStrategyRound"""

    payload_class = EvaluateStrategyPayload
    synchronized_data_class = SynchronizedData
    selection_key = get_name(SynchronizedData.actions)
    done_event = Event.DONE
    collection_key = get_name(SynchronizedData.participant_to_actions_round)
    selection_key = get_name(SynchronizedData.actions)

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        res = super().end_block()
        if res is None:
            return None

        synced_data, event = cast(Tuple[SynchronizedData, Enum], res)
        if event != Event.DONE:
            return res

        if not synced_data.actions:
            return synced_data, Event.WAIT

        return synced_data, Event.DONE

    # Event.NO_MAJORITY, Event.WAIT, Event.ROUND_TIMEOUT


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

            # Ensure positions is always serialized
            positions = payload.get("updates", {}).get("positions", None)
            if positions and not isinstance(positions, str):
                payload["updates"]["positions"] = json.dumps(positions, sort_keys=True)

            bridge_and_swap_actions = payload.get("bridge_and_swap_actions", {})
            updated_actions = self.synchronized_data.actions
            if bridge_and_swap_actions and bridge_and_swap_actions.get("actions"):
                if not self.synchronized_data.last_executed_action_index:
                    index = 1
                else:
                    index = self.synchronized_data.last_executed_action_index + 2
                updated_actions[index:index] = bridge_and_swap_actions.get("actions")

            serialized_actions = json.dumps(updated_actions)
            synchronized_data = synchronized_data.update(
                synchronized_data_class=SynchronizedData,
                **payload.get("updates", {}),
                actions=serialized_actions
            )
            return synchronized_data, event

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self.synchronized_data, Event.NO_MAJORITY
        return None


class PostTxSettlementRound(CollectSameUntilThresholdRound):
    """A round that will be called after tx settlement is done."""

    payload_class: Any = object()
    synchronized_data_class = SynchronizedData

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """
        The end block.

        This is a special type of round. No consensus is necessary here.
        There is no need to send a tx through, nor to check for a majority.
        We simply use this round to check which round submitted the tx,
        and move to the next state in accordance with that.

        :return: the synchronized data and the event, otherwise `None` if the round is still running.
        """
        submitter_to_event: Dict[str, Event] = {
            CallCheckpointRound.auto_round_id(): Event.CHECKPOINT_TX_EXECUTED,
            CheckStakingKPIMetRound.auto_round_id(): Event.VANITY_TX_EXECUTED,
            DecisionMakingRound.auto_round_id(): Event.ACTION_EXECUTED,
        }
                    
        synced_data = SynchronizedData(self.synchronized_data.db)
        event = submitter_to_event.get(synced_data.tx_submitter, Event.UNRECOGNIZED)

        if event == Event.CHECKPOINT_TX_EXECUTED:
            synced_data = synced_data.update(
                synchronized_data_class=SynchronizedData,
                period_number_at_last_cp=synced_data.period_count
            )

        return synced_data, event


class FinishedCallCheckpointRound(DegenerateRound):
    """FinishedCallCheckpointRound"""


class FinishedCheckStakingKPIMetRound(DegenerateRound):
    """FinishedCheckStakingKPIMetRound"""


class FinishedEvaluateStrategyRound(DegenerateRound):
    """FinishedEvaluateStrategyRound"""


class FinishedDecisionMakingRound(DegenerateRound):
    """FinishedDecisionMakingRound"""


class FinishedTxPreparationRound(DegenerateRound):
    """FinishedTxPreparationRound"""


class FailedMultiplexerRound(DegenerateRound):
    """FailedMultiplexerRound"""


class LiquidityTraderAbciApp(AbciApp[Event]):
    """LiquidityTraderAbciApp"""

    initial_round_cls: AppState = CallCheckpointRound
    initial_states: Set[AppState] = {
        CallCheckpointRound,
        CheckStakingKPIMetRound,
        GetPositionsRound,
        DecisionMakingRound,
        PostTxSettlementRound,
    }
    transition_function: AbciAppTransitionFunction = {
        CallCheckpointRound: {
            Event.DONE: CheckStakingKPIMetRound,
            Event.SETTLE: FinishedCallCheckpointRound,
            Event.SERVICE_NOT_STAKED: GetPositionsRound,
            Event.SERVICE_EVICTED: GetPositionsRound,
            Event.ROUND_TIMEOUT: CallCheckpointRound,
            Event.NO_MAJORITY: CallCheckpointRound,
            Event.STAKING_KPI_NOT_MET: CheckStakingKPIMetRound,
            Event.STAKING_KPI_MET: GetPositionsRound,
        },
        CheckStakingKPIMetRound: {
            Event.STAKING_KPI_MET: GetPositionsRound,
            Event.SETTLE: FinishedCheckStakingKPIMetRound,
            Event.ROUND_TIMEOUT: CheckStakingKPIMetRound,
            Event.NO_MAJORITY: CheckStakingKPIMetRound,
            Event.WAIT_FOR_PERIODS_TO_PASS: GetPositionsRound,
        },
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
            Event.UPDATE: DecisionMakingRound,
        },
        PostTxSettlementRound: {
            Event.ACTION_EXECUTED: DecisionMakingRound,
            Event.CHECKPOINT_TX_EXECUTED: CallCheckpointRound,
            Event.VANITY_TX_EXECUTED: CheckStakingKPIMetRound,
            Event.ROUND_TIMEOUT: PostTxSettlementRound,
            Event.UNRECOGNIZED: FailedMultiplexerRound,
        },
        FinishedEvaluateStrategyRound: {},
        FinishedTxPreparationRound: {},
        FinishedDecisionMakingRound: {},
        FinishedCallCheckpointRound: {},
        FinishedCheckStakingKPIMetRound: {},
        FailedMultiplexerRound: {},
    }
    final_states: Set[AppState] = {
        FinishedEvaluateStrategyRound,
        FinishedDecisionMakingRound,
        FinishedTxPreparationRound,
        FinishedCallCheckpointRound,
        FinishedCheckStakingKPIMetRound,
        FailedMultiplexerRound,
    }
    event_to_timeout: Dict[Event, float] = {
        Event.ROUND_TIMEOUT: 30.0,
    }
    cross_period_persisted_keys: FrozenSet[str] = frozenset(
        {
            get_name(SynchronizedData.last_reward_claimed_timestamp),
            get_name(SynchronizedData.min_num_of_safe_tx_required),
            get_name(SynchronizedData.is_staking_kpi_met),
            get_name(SynchronizedData.period_number_at_last_cp),
        }
    )
    db_pre_conditions: Dict[AppState, Set[str]] = {
        CallCheckpointRound: set(),
        CheckStakingKPIMetRound: set(),
        GetPositionsRound: set(),
        DecisionMakingRound: set(),
        PostTxSettlementRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedCallCheckpointRound: {get_name(SynchronizedData.most_voted_tx_hash)},
        FinishedCheckStakingKPIMetRound: {
            get_name(SynchronizedData.most_voted_tx_hash)
        },
        FailedMultiplexerRound: set(),
        FinishedEvaluateStrategyRound: set(),
        FinishedDecisionMakingRound: set(),
        FinishedTxPreparationRound: {get_name(SynchronizedData.most_voted_tx_hash)},
    }

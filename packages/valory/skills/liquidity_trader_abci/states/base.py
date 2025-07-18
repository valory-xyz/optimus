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

"""Base classes for the LiquidityTraderAbciApp."""

import json
from enum import Enum
from typing import Any, Dict, List, Optional, cast

from packages.valory.skills.abstract_round_abci.base import (
    BaseSynchronizedData,
    CollectionRound,
    DeserializedCollection,
)


class StakingState(Enum):
    """Staking state enumeration for the staking."""

    UNSTAKED = 0
    STAKED = 1
    EVICTED = 2


class Event(Enum):
    """Event enumeration for the LiquidityTraderAbciApp."""

    DONE = "done"
    ROUND_TIMEOUT = "round_timeout"
    NO_MAJORITY = "no_majority"
    NONE = "none"
    WAIT = "wait"
    SETTLE = "settle"
    UPDATE = "update"
    ERROR = "error"
    ACTION_EXECUTED = "action_executed"
    CHECKPOINT_TX_EXECUTED = "checkpoint_tx_executed"
    VANITY_TX_EXECUTED = "vanity_tx_executed"
    TRANSFER_COMPLETED = "transfer_completed"
    WITHDRAWAL_COMPLETED = "withdrawal_completed"
    WITHDRAWAL_INITIATED = "withdrawal_initiated"
    UNRECOGNIZED = "unrecognized"
    NEXT_CHECKPOINT_NOT_REACHED_YET = "next_checkpoint_not_reached_yet"
    SERVICE_NOT_STAKED = "service_not_staked"
    SERVICE_EVICTED = "service_evicted"
    STAKING_KPI_MET = "staking_kpi_met"
    STAKING_KPI_NOT_MET = "staking_kpi_not_met"


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
    def participant_to_strategies_round(self) -> DeserializedCollection:
        """Get the participants to the strategies round."""
        return self._get_deserialized("participant_to_strategies_round")

    @property
    def positions(self) -> List[Dict[str, Any]]:
        """Get the positions."""
        serialized = self.db.get("positions", "[]")
        if serialized is None:
            serialized = "[]"
        positions = json.loads(serialized)
        return positions

    @property
    def participant_to_context_round(self) -> DeserializedCollection:
        """Get the participants to actions rounds"""
        return self._get_deserialized("participant_to_context_round")

    @property
    def context(self) -> Optional[List[Dict[str, Any]]]:
        """Get the actions"""
        serialized = self.db.get("context", "[]")
        if serialized is None:
            serialized = "[]"
        context = json.loads(serialized)
        return context

    @property
    def selected_protocols(self) -> List[Dict[str, Any]]:
        """Get the selected protocols."""
        serialized = self.db.get("selected_protocols", "[]")
        if serialized is None:
            serialized = "[]"
        selected_protocols = json.loads(serialized)
        return selected_protocols

    @property
    def trading_type(self) -> List[Dict[str, Any]]:
        """Get the trading_type"""
        trading_type = self.db.get("trading_type", "")
        return trading_type

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
    def participant_to_decision_making(self) -> DeserializedCollection:
        """Get the participants to the DecisionMaking round."""
        return self._get_deserialized("participant_to_decision_making")

    @property
    def participant_to_post_tx_settlement(self) -> DeserializedCollection:
        """Get the participants to the PostTxSettlement round."""
        return self._get_deserialized("participant_to_post_tx_settlement")

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

    @property
    def last_executed_route_index(self) -> Optional[int]:
        """Get the last executed route index."""
        return cast(int, self.db.get("last_executed_route_index", None))

    @property
    def last_executed_step_index(self) -> Optional[int]:
        """Get the last executed step index."""
        return cast(int, self.db.get("last_executed_step_index", None))

    @property
    def routes_retry_attempt(self) -> Optional[int]:
        """Get the routes retry attempt index."""
        return cast(int, self.db.get("routes_retry_attempt", 0))

    @property
    def routes(self) -> Optional[List[Dict[str, Any]]]:
        """Get the routes"""
        serialized = self.db.get("routes", "[]")
        if serialized is None:
            serialized = "[]"
        routes = json.loads(serialized)
        return routes

    @property
    def fee_details(self) -> Optional[Dict[str, Any]]:
        """Get fee details related to route"""
        return self.db.get("fee_details", {})

    @property
    def max_allowed_steps_in_a_route(self) -> Optional[int]:
        """Get the max allowed steps in a route index."""
        return cast(int, self.db.get("max_allowed_steps_in_a_route", None))

    @property
    def last_action(self) -> Optional[str]:
        """Get the last action."""
        return cast(str, self.db.get("last_action", None))

    @property
    def participant_to_withdraw_funds(self) -> DeserializedCollection:
        """Get the participants to the WithdrawFunds round."""
        return self._get_deserialized("participant_to_withdraw_funds")

    @property
    def withdrawal_actions(self) -> Optional[List[Dict[str, Any]]]:
        """Get the withdrawal actions."""
        serialized = self.db.get("withdrawal_actions", "[]")
        if serialized is None:
            serialized = "[]"
        withdrawal_actions = json.loads(serialized)
        return withdrawal_actions

    @property
    def investing_paused(self) -> bool:
        """Get whether investing is paused due to withdrawal."""
        return cast(bool, self.db.get("investing_paused", False))

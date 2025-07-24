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

"""This module contains the LiquidityTraderAbciApp."""

from typing import Dict, FrozenSet, Set

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AppState,
    get_name,
)
from packages.valory.skills.liquidity_trader_abci.states.apr_population import (
    APRPopulationRound,
)
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)
from packages.valory.skills.liquidity_trader_abci.states.call_checkpoint import (
    CallCheckpointRound,
)
from packages.valory.skills.liquidity_trader_abci.states.check_staking_kpi_met import (
    CheckStakingKPIMetRound,
)
from packages.valory.skills.liquidity_trader_abci.states.decision_making import (
    DecisionMakingRound,
)
from packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy import (
    EvaluateStrategyRound,
)
from packages.valory.skills.liquidity_trader_abci.states.fetch_strategies import (
    FetchStrategiesRound,
)
from packages.valory.skills.liquidity_trader_abci.states.final_rounds import (
    FailedMultiplexerRound,
    FinishedCallCheckpointRound,
    FinishedCheckStakingKPIMetRound,
    FinishedDecisionMakingRound,
    FinishedEvaluateStrategyRound,
    FinishedTxPreparationRound,
)
from packages.valory.skills.liquidity_trader_abci.states.get_positions import (
    GetPositionsRound,
)
from packages.valory.skills.liquidity_trader_abci.states.post_tx_settlement import (
    PostTxSettlementRound,
)
from packages.valory.skills.liquidity_trader_abci.states.withdraw_funds import (
    WithdrawFundsRound,
)


class LiquidityTraderAbciApp(AbciApp[Event]):
    """LiquidityTraderAbciApp"""

    initial_round_cls: AppState = FetchStrategiesRound
    initial_states: Set[AppState] = {
        FetchStrategiesRound,
        CallCheckpointRound,
        CheckStakingKPIMetRound,
        GetPositionsRound,
        DecisionMakingRound,
        PostTxSettlementRound,
        WithdrawFundsRound,
    }
    transition_function: AbciAppTransitionFunction = {
        APRPopulationRound: {
            Event.DONE: EvaluateStrategyRound,
            Event.NO_MAJORITY: APRPopulationRound,
            Event.ROUND_TIMEOUT: APRPopulationRound,
            Event.NONE: APRPopulationRound,
            Event.WITHDRAWAL_INITIATED: WithdrawFundsRound,
        },
        CallCheckpointRound: {
            Event.DONE: CheckStakingKPIMetRound,
            Event.NEXT_CHECKPOINT_NOT_REACHED_YET: CheckStakingKPIMetRound,
            Event.SETTLE: FinishedCallCheckpointRound,
            Event.SERVICE_NOT_STAKED: GetPositionsRound,
            Event.SERVICE_EVICTED: GetPositionsRound,
            Event.ROUND_TIMEOUT: CallCheckpointRound,
            Event.NO_MAJORITY: CallCheckpointRound,
            Event.NONE: CallCheckpointRound,
            Event.WITHDRAWAL_INITIATED: WithdrawFundsRound,
        },
        CheckStakingKPIMetRound: {
            Event.DONE: GetPositionsRound,
            Event.STAKING_KPI_MET: GetPositionsRound,
            Event.SETTLE: FinishedCheckStakingKPIMetRound,
            Event.ROUND_TIMEOUT: CheckStakingKPIMetRound,
            Event.NO_MAJORITY: CheckStakingKPIMetRound,
            Event.STAKING_KPI_NOT_MET: GetPositionsRound,
            Event.ERROR: GetPositionsRound,
            Event.NONE: CheckStakingKPIMetRound,
            Event.WITHDRAWAL_INITIATED: WithdrawFundsRound,
        },
        GetPositionsRound: {
            Event.DONE: APRPopulationRound,
            Event.NO_MAJORITY: GetPositionsRound,
            Event.ROUND_TIMEOUT: GetPositionsRound,
            Event.NONE: GetPositionsRound,
            Event.WITHDRAWAL_INITIATED: WithdrawFundsRound,
        },
        EvaluateStrategyRound: {
            Event.DONE: DecisionMakingRound,
            Event.NO_MAJORITY: EvaluateStrategyRound,
            Event.ROUND_TIMEOUT: EvaluateStrategyRound,
            Event.WAIT: FinishedEvaluateStrategyRound,
            Event.NONE: EvaluateStrategyRound,
            Event.WITHDRAWAL_INITIATED: WithdrawFundsRound,
        },
        DecisionMakingRound: {
            Event.DONE: FinishedDecisionMakingRound,
            Event.ERROR: FinishedDecisionMakingRound,
            Event.NO_MAJORITY: DecisionMakingRound,
            Event.ROUND_TIMEOUT: DecisionMakingRound,
            Event.SETTLE: FinishedTxPreparationRound,
            Event.UPDATE: DecisionMakingRound,
            Event.NONE: DecisionMakingRound,
            Event.WITHDRAWAL_INITIATED: WithdrawFundsRound,
        },
        PostTxSettlementRound: {
            Event.ACTION_EXECUTED: DecisionMakingRound,
            Event.CHECKPOINT_TX_EXECUTED: CallCheckpointRound,
            Event.VANITY_TX_EXECUTED: CheckStakingKPIMetRound,
            Event.TRANSFER_COMPLETED: FetchStrategiesRound,
            Event.WITHDRAWAL_COMPLETED: FetchStrategiesRound,
            Event.ROUND_TIMEOUT: PostTxSettlementRound,
            Event.UNRECOGNIZED: FailedMultiplexerRound,
            Event.DONE: PostTxSettlementRound,
            Event.NONE: PostTxSettlementRound,
            Event.NO_MAJORITY: PostTxSettlementRound,
            Event.WITHDRAWAL_INITIATED: WithdrawFundsRound,
        },
        FetchStrategiesRound: {
            Event.DONE: CallCheckpointRound,
            Event.WAIT: FetchStrategiesRound,
            Event.NO_MAJORITY: FetchStrategiesRound,
            Event.ROUND_TIMEOUT: FetchStrategiesRound,
            Event.SETTLE: FinishedTxPreparationRound,
            Event.NONE: FetchStrategiesRound,
            Event.WITHDRAWAL_INITIATED: WithdrawFundsRound,
        },
        WithdrawFundsRound: {
            Event.DONE: DecisionMakingRound,
            Event.NO_MAJORITY: WithdrawFundsRound,
            Event.ROUND_TIMEOUT: WithdrawFundsRound,
            Event.NONE: WithdrawFundsRound,
            Event.WITHDRAWAL_COMPLETED: FetchStrategiesRound,
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
            get_name(SynchronizedData.selected_protocols),
        }
    )
    db_pre_conditions: Dict[AppState, Set[str]] = {
        FetchStrategiesRound: set(),
        CallCheckpointRound: set(),
        CheckStakingKPIMetRound: set(),
        GetPositionsRound: set(),
        DecisionMakingRound: set(),
        PostTxSettlementRound: set(),
        WithdrawFundsRound: set(),
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

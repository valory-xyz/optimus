# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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
    FinishedWithMechRequestRound,
    FinishedWithMechResponsePollRound,
    FinishedWithOffchainMechDepositSettledRound,
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
    """LiquidityTraderAbciApp

    Initial round: FetchStrategiesRound

    Initial states: {CallCheckpointRound, CheckStakingKPIMetRound, DecisionMakingRound, FetchStrategiesRound, GetPositionsRound, PostTxSettlementRound, WithdrawFundsRound}

    Transition states:
        0. CallCheckpointRound
            - done: 1.
            - next checkpoint not reached yet: 1.
            - settle: 11.
            - service not staked: 2.
            - service evicted: 2.
            - round timeout: 0.
            - no majority: 0.
            - none: 0.
            - withdrawal initiated: 7.
        1. CheckStakingKPIMetRound
            - done: 2.
            - staking kpi met: 2.
            - settle: 12.
            - mech request needed: 13.
            - round timeout: 1.
            - no majority: 1.
            - staking kpi not met: 2.
            - error: 2.
            - none: 1.
            - withdrawal initiated: 7.
        2. GetPositionsRound
            - done: 3.
            - no majority: 2.
            - round timeout: 2.
            - none: 2.
            - withdrawal initiated: 7.
        3. EvaluateStrategyRound
            - done: 4.
            - no majority: 3.
            - round timeout: 3.
            - wait: 8.
            - none: 3.
            - withdrawal initiated: 7.
        4. DecisionMakingRound
            - done: 10.
            - error: 10.
            - no majority: 4.
            - round timeout: 4.
            - settle: 9.
            - update: 4.
            - withdrawal initiated: 7.
        5. PostTxSettlementRound
            - action executed: 4.
            - checkpoint tx executed: 0.
            - vanity tx executed: 1.
            - mech request tx executed: 14.
            - offchain mech deposit settled: 15.
            - transfer completed: 6.
            - withdrawal completed: 6.
            - round timeout: 5.
            - unrecognized: 16.
            - no majority: 5.
        6. FetchStrategiesRound
            - done: 0.
            - wait: 6.
            - no majority: 6.
            - round timeout: 6.
            - settle: 9.
            - withdrawal initiated: 7.
        7. WithdrawFundsRound
            - done: 4.
            - no majority: 7.
            - round timeout: 7.
        8. FinishedEvaluateStrategyRound
        9. FinishedTxPreparationRound
        10. FinishedDecisionMakingRound
        11. FinishedCallCheckpointRound
        12. FinishedCheckStakingKPIMetRound
        13. FinishedWithMechRequestRound
        14. FinishedWithMechResponsePollRound
        15. FinishedWithOffchainMechDepositSettledRound
        16. FailedMultiplexerRound

    Final states: {FailedMultiplexerRound, FinishedCallCheckpointRound, FinishedCheckStakingKPIMetRound, FinishedDecisionMakingRound, FinishedEvaluateStrategyRound, FinishedTxPreparationRound, FinishedWithMechRequestRound, FinishedWithMechResponsePollRound, FinishedWithOffchainMechDepositSettledRound}

    Timeouts:
        round timeout: 30.0
    """

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
            Event.MECH_REQUEST_NEEDED: FinishedWithMechRequestRound,
            Event.ROUND_TIMEOUT: CheckStakingKPIMetRound,
            Event.NO_MAJORITY: CheckStakingKPIMetRound,
            Event.STAKING_KPI_NOT_MET: GetPositionsRound,
            Event.ERROR: GetPositionsRound,
            Event.NONE: CheckStakingKPIMetRound,
            Event.WITHDRAWAL_INITIATED: WithdrawFundsRound,
        },
        GetPositionsRound: {
            Event.DONE: EvaluateStrategyRound,
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
            Event.WITHDRAWAL_INITIATED: WithdrawFundsRound,
        },
        PostTxSettlementRound: {
            Event.ACTION_EXECUTED: DecisionMakingRound,
            Event.CHECKPOINT_TX_EXECUTED: CallCheckpointRound,
            Event.VANITY_TX_EXECUTED: CheckStakingKPIMetRound,
            Event.MECH_REQUEST_TX_EXECUTED: FinishedWithMechResponsePollRound,
            Event.OFFCHAIN_MECH_DEPOSIT_SETTLED: FinishedWithOffchainMechDepositSettledRound,
            Event.TRANSFER_COMPLETED: FetchStrategiesRound,
            Event.WITHDRAWAL_COMPLETED: FetchStrategiesRound,
            Event.ROUND_TIMEOUT: PostTxSettlementRound,
            Event.UNRECOGNIZED: FailedMultiplexerRound,
            Event.NO_MAJORITY: PostTxSettlementRound,
        },
        FetchStrategiesRound: {
            Event.DONE: CallCheckpointRound,
            Event.WAIT: FetchStrategiesRound,
            Event.NO_MAJORITY: FetchStrategiesRound,
            Event.ROUND_TIMEOUT: FetchStrategiesRound,
            Event.SETTLE: FinishedTxPreparationRound,
            Event.WITHDRAWAL_INITIATED: WithdrawFundsRound,
        },
        WithdrawFundsRound: {
            Event.DONE: DecisionMakingRound,
            Event.NO_MAJORITY: WithdrawFundsRound,
            Event.ROUND_TIMEOUT: WithdrawFundsRound,
        },
        FinishedEvaluateStrategyRound: {},
        FinishedTxPreparationRound: {},
        FinishedDecisionMakingRound: {},
        FinishedCallCheckpointRound: {},
        FinishedCheckStakingKPIMetRound: {},
        FinishedWithMechRequestRound: {},
        FinishedWithMechResponsePollRound: {},
        FinishedWithOffchainMechDepositSettledRound: {},
        FailedMultiplexerRound: {},
    }
    final_states: Set[AppState] = {
        FinishedEvaluateStrategyRound,
        FinishedDecisionMakingRound,
        FinishedTxPreparationRound,
        FinishedCallCheckpointRound,
        FinishedCheckStakingKPIMetRound,
        FinishedWithMechRequestRound,
        FinishedWithMechResponsePollRound,
        FinishedWithOffchainMechDepositSettledRound,
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
            # Keep the /healthcheck activity-target signal stable across period
            # transitions until CheckStakingKPIMetRound recomputes it.
            get_name(SynchronizedData.is_activity_target_met),
            get_name(SynchronizedData.activity_target),
            get_name(SynchronizedData.activity_completed),
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
        FinishedWithMechRequestRound: {get_name(SynchronizedData.mech_requests)},
        FinishedWithMechResponsePollRound: set(),
        FinishedWithOffchainMechDepositSettledRound: set(),
        FailedMultiplexerRound: set(),
        FinishedEvaluateStrategyRound: set(),
        FinishedDecisionMakingRound: set(),
        FinishedTxPreparationRound: {get_name(SynchronizedData.most_voted_tx_hash)},
    }

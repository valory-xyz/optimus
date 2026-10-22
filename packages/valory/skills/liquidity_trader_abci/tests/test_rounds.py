# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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

"""Test the rounds.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import json

from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.liquidity_trader_abci.rounds import (
    AgentTopUpRound,
    CallCheckpointRound,
    CheckStakingKPIMetRound,
    DecisionMakingRound,
    EvaluateStrategyRound,
    Event,
    FetchStrategiesRound,
    FinishedAgentTopUpRound,
    FinishedStandbyRound,
    GetPositionsRound,
    LiquidityTraderAbciApp,
    PostTxSettlementRound,
    WithdrawFundsRound,
)
from packages.valory.skills.liquidity_trader_abci.states.base import SynchronizedData


def test_import() -> None:
    """Test that the rounds module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.rounds  # noqa


class TestWithdrawalTransitionMap:
    """Verify which rounds expose a WITHDRAWAL_INITIATED transition."""

    def test_withdrawal_initiated_wired_from_cycle_rounds(self) -> None:
        """Every cycle round except PostTxSettlement routes WITHDRAWAL_INITIATED to WithdrawFunds."""
        expected = {
            FetchStrategiesRound,
            CallCheckpointRound,
            CheckStakingKPIMetRound,
            GetPositionsRound,
            EvaluateStrategyRound,
            DecisionMakingRound,
        }
        wired = {
            cls
            for cls, transitions in LiquidityTraderAbciApp.transition_function.items()
            if Event.WITHDRAWAL_INITIATED in transitions
        }
        assert wired == expected

    def test_post_tx_settlement_does_not_route_withdrawal(self) -> None:
        """A freshly settled tx must not be preempted by withdrawal."""
        assert (
            Event.WITHDRAWAL_INITIATED
            not in LiquidityTraderAbciApp.transition_function[PostTxSettlementRound]
        )

    def test_withdrawal_initiated_targets_withdraw_funds_round(self) -> None:
        """All WITHDRAWAL_INITIATED transitions go to the same target."""
        targets = {
            transitions[Event.WITHDRAWAL_INITIATED]
            for transitions in LiquidityTraderAbciApp.transition_function.values()
            if Event.WITHDRAWAL_INITIATED in transitions
        }
        assert targets == {WithdrawFundsRound}


class TestCoreCycleTransitions:
    """Pin the load-bearing transitions in the GetPositions -> EvaluateStrategy -> DecisionMaking cycle."""

    def test_the_kpi_check_hands_over_to_get_positions(self) -> None:
        """Every exit from the staking check continues into the trading cycle."""
        transitions = LiquidityTraderAbciApp.transition_function[
            CheckStakingKPIMetRound
        ]
        assert {
            transitions[event]
            for event in (
                Event.DONE,
                Event.STAKING_KPI_MET,
                Event.STAKING_KPI_NOT_MET,
                Event.ERROR,
            )
        } == {GetPositionsRound}

    def test_get_positions_done_targets_evaluate_strategy(self) -> None:
        """Test that GetPositionsRound DONE goes directly to EvaluateStrategyRound."""
        assert (
            LiquidityTraderAbciApp.transition_function[GetPositionsRound][Event.DONE]
            is EvaluateStrategyRound
        )

    def test_evaluate_strategy_done_targets_decision_making(self) -> None:
        """Test that EvaluateStrategyRound DONE continues into DecisionMakingRound."""
        assert (
            LiquidityTraderAbciApp.transition_function[EvaluateStrategyRound][
                Event.DONE
            ]
            is DecisionMakingRound
        )


class TestStandbyTransitions:
    """A met goal and staking side end the period without trading."""

    def test_standby_leaves_through_its_own_final_round(self) -> None:
        """STANDBY from the staking check lands in FinishedStandbyRound."""
        assert (
            LiquidityTraderAbciApp.transition_function[CheckStakingKPIMetRound][
                Event.STANDBY
            ]
            is FinishedStandbyRound
        )

    def test_standby_round_is_final_with_no_post_conditions(self) -> None:
        """The composed app needs nothing from the db to reset and pause."""
        assert FinishedStandbyRound in LiquidityTraderAbciApp.final_states
        assert LiquidityTraderAbciApp.transition_function[FinishedStandbyRound] == {}
        assert LiquidityTraderAbciApp.db_post_conditions[FinishedStandbyRound] == set()


class TestAgentTopUpTransitions:
    """The Agent Signer top-up runs at period start, ahead of the strategies."""

    def test_period_starts_with_the_top_up(self) -> None:
        """The app starts in AgentTopUpRound."""
        assert LiquidityTraderAbciApp.initial_round_cls is AgentTopUpRound
        assert AgentTopUpRound in LiquidityTraderAbciApp.initial_states
        assert LiquidityTraderAbciApp.db_pre_conditions[AgentTopUpRound] == set()

    def test_nothing_to_move_continues_to_fetch_strategies(self) -> None:
        """DONE without a tx goes straight to FetchStrategiesRound."""
        transitions = LiquidityTraderAbciApp.transition_function[AgentTopUpRound]
        assert transitions[Event.DONE] is FetchStrategiesRound

    def test_a_move_leaves_through_its_own_final_round(self) -> None:
        """SETTLE hands the tx to settlement through FinishedAgentTopUpRound."""
        transitions = LiquidityTraderAbciApp.transition_function[AgentTopUpRound]
        assert transitions[Event.SETTLE] is FinishedAgentTopUpRound
        assert FinishedAgentTopUpRound in LiquidityTraderAbciApp.final_states
        assert LiquidityTraderAbciApp.db_post_conditions[FinishedAgentTopUpRound] == {
            "most_voted_tx_hash"
        }

    def test_a_settled_top_up_continues_to_fetch_strategies(self) -> None:
        """TOPUP_DONE after settlement goes to FetchStrategiesRound."""
        transitions = LiquidityTraderAbciApp.transition_function[PostTxSettlementRound]
        assert transitions[Event.TOPUP_DONE] is FetchStrategiesRound

    def test_withdrawal_is_left_to_fetch_strategies(self) -> None:
        """The top-up runs in withdrawal mode too and does not route WITHDRAWAL_INITIATED."""
        assert (
            Event.WITHDRAWAL_INITIATED
            not in LiquidityTraderAbciApp.transition_function[AgentTopUpRound]
        )

    def test_withdrawal_actions_reach_the_next_period(self) -> None:
        """The actions WithdrawFundsRound stored are still there when the top-up runs."""
        persisted = LiquidityTraderAbciApp.cross_period_persisted_keys
        db = AbciAppDB(setup_data={}, cross_period_persisted_keys=persisted)
        db.update(**{key: None for key in db.cross_period_persisted_keys})
        actions = [{"action": "exit_pool"}, {"action": "withdraw"}]
        db.update(withdrawal_actions=json.dumps(actions))

        db.create()

        assert SynchronizedData(db=db).withdrawal_actions == actions

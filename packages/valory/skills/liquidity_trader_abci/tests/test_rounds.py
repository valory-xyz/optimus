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

from packages.valory.skills.liquidity_trader_abci.rounds import (
    APRPopulationRound,
    CallCheckpointRound,
    CheckStakingKPIMetRound,
    DecisionMakingRound,
    EvaluateStrategyRound,
    Event,
    FetchStrategiesRound,
    GetPositionsRound,
    LiquidityTraderAbciApp,
    PostTxSettlementRound,
    WithdrawFundsRound,
)


def test_import() -> None:
    """Test that the rounds module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.rounds  # noqa


class TestWithdrawalTransitionMap:
    """Verify which rounds expose a WITHDRAWAL_INITIATED transition."""

    def test_withdrawal_initiated_wired_from_seven_rounds(self) -> None:
        """Every cycle round except PostTxSettlement routes WITHDRAWAL_INITIATED to WithdrawFunds."""
        expected = {
            FetchStrategiesRound,
            CallCheckpointRound,
            CheckStakingKPIMetRound,
            GetPositionsRound,
            APRPopulationRound,
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

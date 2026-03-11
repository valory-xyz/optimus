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

from packages.valory.skills.abstract_round_abci.base import AbciApp, get_name
from packages.valory.skills.liquidity_trader_abci.rounds import LiquidityTraderAbciApp
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


def test_import() -> None:
    """Test that the rounds module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.rounds  # noqa


class TestLiquidityTraderAbciApp:
    """Test LiquidityTraderAbciApp class."""

    def test_is_abci_app(self) -> None:
        """Test that it inherits from AbciApp."""
        assert issubclass(LiquidityTraderAbciApp, AbciApp)

    def test_initial_round_cls(self) -> None:
        """Test initial_round_cls is FetchStrategiesRound."""
        assert LiquidityTraderAbciApp.initial_round_cls is FetchStrategiesRound

    def test_initial_states(self) -> None:
        """Test initial_states contains expected rounds."""
        expected = {
            FetchStrategiesRound,
            CallCheckpointRound,
            CheckStakingKPIMetRound,
            GetPositionsRound,
            DecisionMakingRound,
            PostTxSettlementRound,
            WithdrawFundsRound,
        }
        assert LiquidityTraderAbciApp.initial_states == expected

    def test_final_states(self) -> None:
        """Test final_states contains expected rounds."""
        expected = {
            FinishedEvaluateStrategyRound,
            FinishedDecisionMakingRound,
            FinishedTxPreparationRound,
            FinishedCallCheckpointRound,
            FinishedCheckStakingKPIMetRound,
            FailedMultiplexerRound,
        }
        assert LiquidityTraderAbciApp.final_states == expected

    def test_transition_function_has_all_rounds(self) -> None:
        """Test transition function has entries for all required rounds."""
        expected_rounds = [
            APRPopulationRound,
            CallCheckpointRound,
            CheckStakingKPIMetRound,
            GetPositionsRound,
            EvaluateStrategyRound,
            DecisionMakingRound,
            PostTxSettlementRound,
            FetchStrategiesRound,
            WithdrawFundsRound,
        ]
        for round_cls in expected_rounds:
            assert round_cls in LiquidityTraderAbciApp.transition_function

    def test_event_to_timeout(self) -> None:
        """Test event_to_timeout configuration."""
        assert Event.ROUND_TIMEOUT in LiquidityTraderAbciApp.event_to_timeout
        assert LiquidityTraderAbciApp.event_to_timeout[Event.ROUND_TIMEOUT] == 30.0

    def test_cross_period_persisted_keys(self) -> None:
        """Test cross_period_persisted_keys contains expected keys."""
        expected_keys = {
            get_name(SynchronizedData.last_reward_claimed_timestamp),
            get_name(SynchronizedData.min_num_of_safe_tx_required),
            get_name(SynchronizedData.is_staking_kpi_met),
            get_name(SynchronizedData.period_number_at_last_cp),
            get_name(SynchronizedData.selected_protocols),
        }
        assert LiquidityTraderAbciApp.cross_period_persisted_keys == expected_keys

    def test_db_pre_conditions(self) -> None:
        """Test db_pre_conditions has entries for initial states."""
        for state in LiquidityTraderAbciApp.initial_states:
            assert state in LiquidityTraderAbciApp.db_pre_conditions

    def test_db_post_conditions(self) -> None:
        """Test db_post_conditions has entries for final states."""
        for state in LiquidityTraderAbciApp.final_states:
            assert state in LiquidityTraderAbciApp.db_post_conditions

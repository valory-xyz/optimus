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

"""This package contains round behaviours of OptimusAbciApp."""

import packages.valory.skills.liquidity_trader_abci.rounds as LiquidityTraderAbci
import packages.valory.skills.market_data_fetcher_abci.rounds as MarketDataFetcherAbci
import packages.valory.skills.portfolio_tracker_abci.rounds as PortfolioTrackerAbci
import packages.valory.skills.registration_abci.rounds as RegistrationAbci
import packages.valory.skills.reset_pause_abci.rounds as ResetAndPauseAbci
import packages.valory.skills.strategy_evaluator_abci.rounds as StrategyEvaluatorAbci
import packages.valory.skills.trader_decision_maker_abci.rounds as TraderDecisionMakerAbci
import packages.valory.skills.transaction_settlement_abci.rounds as TransactionSettlementAbci
from packages.valory.skills.abstract_round_abci.abci_app_chain import (
    AbciAppTransitionMapping,
    chain,
)
from packages.valory.skills.abstract_round_abci.base import BackgroundAppConfig
from packages.valory.skills.termination_abci.rounds import (
    BackgroundRound,
    Event,
    TerminationAbciApp,
)


abci_app_transition_mapping: AbciAppTransitionMapping = {
    RegistrationAbci.FinishedRegistrationRound: LiquidityTraderAbci.DecideAgentStartingRound,
    LiquidityTraderAbci.FinishedCallCheckpointRound: TransactionSettlementAbci.RandomnessTransactionSubmissionRound,
    LiquidityTraderAbci.FinishedCheckStakingKPIMetRound: TransactionSettlementAbci.RandomnessTransactionSubmissionRound,
    LiquidityTraderAbci.SwitchAgentStartingRound: TraderDecisionMakerAbci.RandomnessRound,
    TraderDecisionMakerAbci.FinishedTraderDecisionMakerRound: MarketDataFetcherAbci.FetchMarketDataRound,
    TraderDecisionMakerAbci.FailedTraderDecisionMakerRound: TraderDecisionMakerAbci.RandomnessRound,
    MarketDataFetcherAbci.FinishedMarketFetchRound: PortfolioTrackerAbci.PortfolioTrackerRound,
    MarketDataFetcherAbci.FailedMarketFetchRound: TraderDecisionMakerAbci.RandomnessRound,
    PortfolioTrackerAbci.FinishedPortfolioTrackerRound: StrategyEvaluatorAbci.StrategyExecRound,
    PortfolioTrackerAbci.FailedPortfolioTrackerRound: TraderDecisionMakerAbci.RandomnessRound,
    StrategyEvaluatorAbci.SwapTxPreparedRound: TransactionSettlementAbci.RandomnessTransactionSubmissionRound,
    StrategyEvaluatorAbci.NoMoreSwapsRound: ResetAndPauseAbci.ResetAndPauseRound,
    StrategyEvaluatorAbci.StrategyExecutionFailedRound: TraderDecisionMakerAbci.RandomnessRound,
    StrategyEvaluatorAbci.InstructionPreparationFailedRound: TraderDecisionMakerAbci.RandomnessRound,
    StrategyEvaluatorAbci.HodlRound: ResetAndPauseAbci.ResetAndPauseRound,
    StrategyEvaluatorAbci.BacktestingNegativeRound: TraderDecisionMakerAbci.RandomnessRound,
    StrategyEvaluatorAbci.BacktestingFailedRound: TraderDecisionMakerAbci.RandomnessRound,
    LiquidityTraderAbci.FinishedDecisionMakingRound: ResetAndPauseAbci.ResetAndPauseRound,
    LiquidityTraderAbci.FinishedEvaluateStrategyRound: ResetAndPauseAbci.ResetAndPauseRound,
    LiquidityTraderAbci.FinishedTxPreparationRound: TransactionSettlementAbci.RandomnessTransactionSubmissionRound,
    LiquidityTraderAbci.FailedMultiplexerRound: ResetAndPauseAbci.ResetAndPauseRound,
    TransactionSettlementAbci.FinishedTransactionSubmissionRound: LiquidityTraderAbci.DecideAgentEndingRound,
    LiquidityTraderAbci.SwitchAgentEndingRound: TraderDecisionMakerAbci.RandomnessRound,
    TransactionSettlementAbci.FailedRound: ResetAndPauseAbci.ResetAndPauseRound,
    ResetAndPauseAbci.FinishedResetAndPauseRound: LiquidityTraderAbci.DecideAgentStartingRound,
    ResetAndPauseAbci.FinishedResetAndPauseErrorRound: RegistrationAbci.RegistrationRound,
}

termination_config = BackgroundAppConfig(
    round_cls=BackgroundRound,
    start_event=Event.TERMINATE,
    abci_app=TerminationAbciApp,
)

OptimusAbciApp = chain(
    (
        RegistrationAbci.AgentRegistrationAbciApp,
        TraderDecisionMakerAbci.TraderDecisionMakerAbciApp,
        MarketDataFetcherAbci.MarketDataFetcherAbciApp,
        PortfolioTrackerAbci.PortfolioTrackerAbciApp,
        StrategyEvaluatorAbci.StrategyEvaluatorAbciApp,
        LiquidityTraderAbci.LiquidityTraderAbciApp,
        TransactionSettlementAbci.TransactionSubmissionAbciApp,
        ResetAndPauseAbci.ResetPauseAbciApp,
    ),
    abci_app_transition_mapping,
).add_background_app(termination_config)

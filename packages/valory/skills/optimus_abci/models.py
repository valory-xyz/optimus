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

"""This module contains the shared state for the abci skill of OptimusAbciApp."""

from typing import Any, Dict, Type, Union, cast

from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.tests.data.dummy_abci.models import (
    RandomnessApi as BaseRandomnessApi,
)
from packages.valory.skills.liquidity_trader_abci.models import Coingecko
from packages.valory.skills.liquidity_trader_abci.models import (
    Params as LiquidityTraderParams,
)
from packages.valory.skills.liquidity_trader_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.liquidity_trader_abci.rounds import (
    Event as LiquidityTraderEvent,
)
from packages.valory.skills.market_data_fetcher_abci.models import (
    Params as MarketDataFetcherParams,
)
from packages.valory.skills.market_data_fetcher_abci.rounds import (
    Event as MarketDataFetcherEvent,
)
from packages.valory.skills.optimus_abci.composition import OptimusAbciApp
from packages.valory.skills.portfolio_tracker_abci.models import GetBalance
from packages.valory.skills.portfolio_tracker_abci.models import (
    Params as PortfolioTrackerParams,
)
from packages.valory.skills.portfolio_tracker_abci.models import TokenAccounts
from packages.valory.skills.reset_pause_abci.rounds import Event as ResetPauseEvent
from packages.valory.skills.strategy_evaluator_abci.models import (
    StrategyEvaluatorParams as StrategyEvaluatorParams,
)
from packages.valory.skills.strategy_evaluator_abci.models import (
    SwapInstructionsSpecs,
    SwapQuotesSpecs,
    TxSettlementProxy,
)
from packages.valory.skills.strategy_evaluator_abci.rounds import (
    Event as StrategyEvaluatorEvent,
)
from packages.valory.skills.termination_abci.models import TerminationParams
from packages.valory.skills.trader_decision_maker_abci.models import (
    Params as TraderDecisionMakerParams,
)
from packages.valory.skills.trader_decision_maker_abci.rounds import (
    Event as DecisionMakingEvent,
)
from packages.valory.skills.transaction_settlement_abci.models import TransactionParams
from packages.valory.skills.transaction_settlement_abci.rounds import (
    Event as TransactionSettlementEvent,
)


EventType = Union[
    Type[LiquidityTraderEvent],
    Type[TransactionSettlementEvent],
    Type[ResetPauseEvent],
    Type[MarketDataFetcherEvent],
    Type[DecisionMakingEvent],
    Type[StrategyEvaluatorEvent],
]
EventToTimeoutMappingType = Dict[
    Union[
        LiquidityTraderEvent,
        TransactionSettlementEvent,
        ResetPauseEvent,
        MarketDataFetcherEvent,
        DecisionMakingEvent,
        StrategyEvaluatorEvent,
    ],
    float,
]

Coingecko = Coingecko
Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool
SwapQuotesSpecs = SwapQuotesSpecs
SwapInstructionsSpecs = SwapInstructionsSpecs
TxSettlementProxy = TxSettlementProxy
GetBalance = GetBalance
TokenAccounts = TokenAccounts

RandomnessApi = BaseRandomnessApi

MARGIN = 5
MULTIPLIER = 40


class Params(  # pylint: disable=too-many-ancestors
    TerminationParams,
    LiquidityTraderParams,
    TraderDecisionMakerParams,
    MarketDataFetcherParams,
    StrategyEvaluatorParams,
    PortfolioTrackerParams,
    TransactionParams,
):
    """A model to represent params for multiple abci apps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init"""
        self.service_endpoint_base = self._ensure("service_endpoint_base", kwargs, str)
        super().__init__(*args, **kwargs)


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = OptimusAbciApp  # type: ignore

    @property
    def params(self) -> Params:
        """Get the parameters."""
        return cast(Params, self.context.params)

    def setup(self) -> None:
        """Set up."""
        super().setup()

        events = (
            LiquidityTraderEvent,
            TransactionSettlementEvent,
            ResetPauseEvent,
            MarketDataFetcherEvent,
            DecisionMakingEvent,
            StrategyEvaluatorEvent,
        )
        round_timeout = self.params.round_timeout_seconds
        round_timeout_overrides = {
            cast(EventType, event).ROUND_TIMEOUT: round_timeout for event in events
        }
        reset_pause_timeout = self.params.reset_pause_duration + MARGIN
        event_to_timeout_overrides: EventToTimeoutMappingType = {
            **round_timeout_overrides,
            TransactionSettlementEvent.RESET_TIMEOUT: round_timeout,
            TransactionSettlementEvent.VALIDATE_TIMEOUT: self.params.validate_timeout,
            LiquidityTraderEvent.ROUND_TIMEOUT: self.params.round_timeout_seconds
            * MULTIPLIER,
            ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT: reset_pause_timeout,
            StrategyEvaluatorEvent.PROXY_SWAP_TIMEOUT: self.params.proxy_round_timeout_seconds,
        }

        for event, override in event_to_timeout_overrides.items():
            OptimusAbciApp.event_to_timeout[event] = override

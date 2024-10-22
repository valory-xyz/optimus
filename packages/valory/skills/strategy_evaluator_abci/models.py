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

"""This module contains the models for the skill."""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from aea.skills.base import SkillContext

from packages.valory.skills.abstract_round_abci.base import AbciApp
from packages.valory.skills.abstract_round_abci.models import ApiSpecs, BaseParams
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.strategy_evaluator_abci.rounds import (
    StrategyEvaluatorAbciApp,
)


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool


AMOUNT_PARAM = "amount"
SLIPPAGE_PARAM = "slippageBps"


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls: Type[AbciApp] = StrategyEvaluatorAbciApp

    def __init__(self, *args: Any, skill_context: SkillContext, **kwargs: Any) -> None:
        """Initialize the state."""
        super().__init__(*args, skill_context=skill_context, **kwargs)
        # utilized if using the proxy server
        self.orders: Optional[List[Dict[str, str]]] = None
        # utilized if using the Solana tx settlement
        self.instructions: Optional[List[Dict[str, Any]]] = None

    def setup(self) -> None:
        """Set up the model."""
        super().setup()
        if (
            self.context.params.use_proxy_server
            and self.synchronized_data.max_participants != 1
        ):
            raise ValueError("Cannot use proxy server with a multi-agent service!")

        swap_apis: Tuple[ApiSpecs, ApiSpecs] = (
            self.context.swap_quotes,
            self.context.tx_settlement_proxy,
        )
        required_swap_params = (AMOUNT_PARAM, SLIPPAGE_PARAM)
        for swap_api in swap_apis:
            for swap_param in required_swap_params:
                if swap_param not in swap_api.parameters:
                    exc = f"Api with id {swap_api.api_id!r} missing required parameter: {swap_param}!"
                    raise ValueError(exc)

        amounts = (api.parameters[AMOUNT_PARAM] for api in swap_apis)
        expected_swap_tx_cost = self.context.params.expected_swap_tx_cost
        if any(expected_swap_tx_cost > amount for amount in amounts):
            exc = "The expected cost of the swap transaction cannot be greater than the swap amount!"
            raise ValueError(exc)


def _raise_incorrect_config(key: str, values: Any) -> None:
    """Raise a `ValueError` for incorrect configuration of a nested_list workaround."""
    raise ValueError(
        f"The given configuration for {key!r} is incorrectly formatted: {values}!"
        "The value is expected to be a list of lists that can be represented as a dictionary."
    )


def nested_list_todict_workaround(
    kwargs: Dict,
    key: str,
) -> Dict:
    """Get a nested list from the kwargs and convert it to a dictionary."""
    values = list(kwargs.get(key, []))
    if len(values) == 0:
        raise ValueError(f"No {key!r} specified in agent's configurations: {kwargs}!")
    if any(not issubclass(type(nested_values), Iterable) for nested_values in values):
        _raise_incorrect_config(key, values)
    if any(len(nested_values) % 2 == 1 for nested_values in values):
        _raise_incorrect_config(key, values)
    return {value[0]: value[1] for value in values}


class StrategyEvaluatorParams(BaseParams):
    """Strategy evaluator's parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters' object."""
        self.strategies_kwargs: Dict[str, List[Any]] = nested_list_todict_workaround(
            kwargs, "strategies_kwargs"
        )
        self.use_proxy_server: bool = self._ensure("use_proxy_server", kwargs, bool)
        self.proxy_round_timeout_seconds: float = self._ensure(
            "proxy_round_timeout_seconds", kwargs, float
        )
        self.expected_swap_tx_cost: int = self._ensure(
            "expected_swap_tx_cost", kwargs, int
        )
        self.ipfs_fetch_retries: int = self._ensure("ipfs_fetch_retries", kwargs, int)
        self.sharpe_threshold: float = self._ensure("sharpe_threshold", kwargs, float)
        self.use_solana = self._ensure("use_solana", kwargs, bool)
        self.base_tokens = self._ensure("base_tokens", kwargs, Dict[str, str])
        self.native_currencies = self._ensure(
            "native_currencies", kwargs, Dict[str, str]
        )
        self.trade_size_in_base_token = self._ensure(
            "trade_size_in_base_token", kwargs, float
        )
        super().__init__(*args, **kwargs)


class SwapQuotesSpecs(ApiSpecs):
    """A model that wraps ApiSpecs for the Jupiter quotes specifications."""


class SwapInstructionsSpecs(ApiSpecs):
    """A model that wraps ApiSpecs for the Jupiter instructions specifications."""


class TxSettlementProxy(ApiSpecs):
    """A model that wraps ApiSpecs for the Solana transaction settlement proxy server."""

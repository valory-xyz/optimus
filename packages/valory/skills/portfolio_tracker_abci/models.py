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

"""This module contains the models for the Portfolio Tracker."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Union

from packages.valory.skills.abstract_round_abci.models import ApiSpecs, BaseParams
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.portfolio_tracker_abci.rounds import PortfolioTrackerAbciApp


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool


class GetBalance(ApiSpecs):
    """A model that wraps ApiSpecs for the Solana balance check."""


class TokenAccounts(ApiSpecs):
    """A model that wraps ApiSpecs for the Solana tokens' balance check."""


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = PortfolioTrackerAbciApp


class Params(BaseParams):
    """Parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""
        self.agent_balance_threshold: int = self._ensure(
            "agent_balance_threshold", kwargs, int
        )
        self.multisig_balance_threshold: int = self._ensure(
            "multisig_balance_threshold", kwargs, int
        )
        self.squad_vault: str = self._ensure("squad_vault", kwargs, str)
        self.tracked_tokens: List[str] = self._ensure(
            "tracked_tokens", kwargs, List[str]
        )
        self.refill_action_timeout: int = self._ensure(
            "refill_action_timeout", kwargs, int
        )
        self.rpc_polling_interval: int = self._ensure(
            "rpc_polling_interval", kwargs, int
        )
        # We depend on the same keys across all the models, so we can just use the same keys.
        if not getattr(self, "ledger_ids", None):
            self.ledger_ids = self._ensure("ledger_ids", kwargs, List[str])
        if not getattr(self, "exchange_ids", None):
            self.exchange_ids = self._ensure(
                "exchange_ids", kwargs, Dict[str, List[str]]
            )
        super().__init__(*args, **kwargs)


@dataclass
class RPCPayload:
    """An RPC request's payload."""

    method: str
    params: list
    id: int = 1
    jsonrpc: str = "2.0"

    def __getitem__(self, attr: str) -> Union[int, str, list]:
        """Implemented so we can easily unpack using `**`."""
        return getattr(self, attr)

    def keys(self) -> Iterable[str]:
        """Implemented so we can easily unpack using `**`."""
        return asdict(self).keys()

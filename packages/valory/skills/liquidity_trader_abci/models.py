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

"""This module contains the shared state for the abci skill of LiquidityTraderAbciApp."""

import json
from typing import Any

from packages.valory.skills.abstract_round_abci.models import BaseParams
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.liquidity_trader_abci.rounds import LiquidityTraderAbciApp


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = LiquidityTraderAbciApp


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool


class Params(BaseParams):
    """Parameters"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.allowed_lp_pool_addresses = json.loads(
            self._ensure("allowed_lp_pool_addresses", kwargs, str)
        )
        self.allowed_assets = json.loads(self._ensure("allowed_assets", kwargs, str))
        self.safe_contract_addresses = json.loads(
            self._ensure("safe_contract_addresses", kwargs, str)
        )
        self.pool_data_api_url = self._ensure("pool_data_api_url", kwargs, str)
        self.allowed_chains = json.loads(self._ensure("allowed_chains", kwargs, str))
        self.gas_reserve = json.loads(self._ensure("gas_reserve", kwargs, str))
        self.apr_threshold = self._ensure("apr_threshold", kwargs, int)
        self.round_threshold = self._ensure("round_threshold", kwargs, int)
        self.min_balance_multiplier = self._ensure("min_balance_multiplier", kwargs, int)
        super().__init__(*args, **kwargs)

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
import os
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from aea.skills.base import Model, SkillContext

from packages.valory.skills.abstract_round_abci.models import BaseParams
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.abstract_round_abci.models import TypeCheckMixin
from packages.valory.skills.liquidity_trader_abci.rounds import LiquidityTraderAbciApp


MINUTE_UNIX = 60


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = LiquidityTraderAbciApp

    def __init__(self, *args: Any, skill_context: SkillContext, **kwargs: Any) -> None:
        """Initialize the state."""
        super().__init__(*args, skill_context=skill_context, **kwargs)
        self.in_flight_req: bool = False
        self.strategy_to_filehash: Dict[str, str] = {}
        self.strategies_executables: Dict[str, Tuple[str, str]] = {}
        self.trading_type: str = ""
        self.selected_protocols: List[str] = []
        self.request_count: int = 0
        self.request_queue = []
        self.req_to_callback: Dict[str, Tuple[Callable, Dict[str, Any]]] = {}
        self.agent_reasoning: str = ""
        self._token_price_cache = {}
        self._token_price_cache_ttl = 600

    def setup(self) -> None:
        """Set up the model."""
        super().setup()
        params = self.context.params
        self.strategy_to_filehash = {
            value: key for key, value in params.file_hash_to_strategies.items()
        }
        strategy_exec = self.strategy_to_filehash.keys()
        # Extract all strategy values from the available_strategies dictionary
        available_strategies_list = [
            strategy
            for strategies in params.available_strategies.values()
            for strategy in strategies
        ]

        # Iterate over each strategy in the flattened list of available strategies
        for selected_strategy in available_strategies_list:
            if selected_strategy not in strategy_exec:
                raise ValueError(
                    f"The selected trading strategy {selected_strategy} "
                    f"is not in the strategies' executables {strategy_exec}."
                )


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool


class CoingeckoRateLimiter:
    """Keeps track of the rate limiting for Coingecko."""

    def __init__(self, limit: int, credits_: int) -> None:
        """Initialize the Coingecko rate limiter."""
        self._limit = self._remaining_limit = limit
        self._credits = self._remaining_credits = credits_
        self._last_request_time = time()

    @property
    def limit(self) -> int:
        """Get the limit per minute."""
        return self._limit

    @property
    def credits(self) -> int:
        """Get the requests' cap per month."""
        return self._credits

    @property
    def remaining_limit(self) -> int:
        """Get the remaining limit per minute."""
        return self._remaining_limit

    @property
    def remaining_credits(self) -> int:
        """Get the remaining requests' cap per month."""
        return self._remaining_credits

    @property
    def last_request_time(self) -> float:
        """Get the timestamp of the last request."""
        return self._last_request_time

    @property
    def rate_limited(self) -> bool:
        """Check whether we are rate limited."""
        return self.remaining_limit == 0

    @property
    def no_credits(self) -> bool:
        """Check whether all the credits have been spent."""
        return self.remaining_credits == 0

    @property
    def cannot_request(self) -> bool:
        """Check whether we cannot perform a request."""
        return self.rate_limited or self.no_credits

    @property
    def credits_reset_timestamp(self) -> int:
        """Get the UNIX timestamp in which the Coingecko credits reset."""
        current_date = datetime.now()
        first_day_of_next_month = datetime(current_date.year, current_date.month + 1, 1)
        return int(first_day_of_next_month.timestamp())

    @property
    def can_reset_credits(self) -> bool:
        """Check whether the Coingecko credits can be reset."""
        return self.last_request_time >= self.credits_reset_timestamp

    def _update_limits(self) -> None:
        """Update the remaining limits and the credits if necessary."""
        time_passed = time() - self.last_request_time
        limit_increase = int(time_passed / MINUTE_UNIX) * self.limit
        self._remaining_limit = min(self.limit, self.remaining_limit + limit_increase)
        if self.can_reset_credits:
            self._remaining_credits = self.credits

    def _burn_credit(self) -> None:
        """Use one credit."""
        self._remaining_limit -= 1
        self._remaining_credits -= 1
        self._last_request_time = time()

    def check_and_burn(self) -> bool:
        """Check whether we can perform a new request, and if yes, update the remaining limit and credits."""
        self._update_limits()
        if self.cannot_request:
            return False
        self._burn_credit()
        return True


class Coingecko(Model, TypeCheckMixin):
    """Coingecko configuration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Coingecko object."""
        self.token_price_endpoint: str = self._ensure(
            "token_price_endpoint", kwargs, str
        )
        self.coin_price_endpoint: str = self._ensure("coin_price_endpoint", kwargs, str)
        self.api_key: Optional[str] = self._ensure("api_key", kwargs, Optional[str])
        self.rate_limited_code: int = self._ensure("rate_limited_code", kwargs, int)
        self.historical_price_endpoint: str = self._ensure(
            "historical_price_endpoint", kwargs, str
        )
        self.chain_to_platform_id_mapping: Dict[str, str] = json.loads(
            self._ensure("chain_to_platform_id_mapping", kwargs, str)
        )
        limit: int = self._ensure("requests_per_minute", kwargs, int)
        credits_: int = self._ensure("credits", kwargs, int)
        self.rate_limiter = CoingeckoRateLimiter(limit, credits_)
        super().__init__(*args, **kwargs)

    def rate_limited_status_callback(self) -> None:
        """Callback when a rate-limited status is returned from the API."""
        self.context.logger.error(
            "Unexpected rate-limited status code was received from the Coingecko API! "
            "Setting the limit to 0 on the local rate limiter to partially address the issue. "
            "Please check whether the `Coingecko` overrides are set corresponding to the API's rules."
        )
        self.rate_limiter._remaining_limit = 0
        self.rate_limiter._last_request_time = time()


class Params(BaseParams):
    """Parameters"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init"""
        self.initial_assets = json.loads(self._ensure("initial_assets", kwargs, str))
        self.safe_contract_addresses = json.loads(
            self._ensure("safe_contract_addresses", kwargs, str)
        )
        self.merkl_fetch_campaigns_args = json.loads(
            self._ensure("merkl_fetch_campaigns_args", kwargs, str)
        )
        self.allowed_chains: List[str] = self._ensure(
            "allowed_chains", kwargs, List[str]
        )
        self.gas_reserve = json.loads(self._ensure("gas_reserve", kwargs, str))
        self.apr_threshold = self._ensure("apr_threshold", kwargs, int)
        self.round_threshold = self._ensure("round_threshold", kwargs, int)
        self.min_balance_multiplier = self._ensure(
            "min_balance_multiplier", kwargs, int
        )
        self.multisend_contract_addresses = json.loads(
            self._ensure("multisend_contract_addresses", kwargs, str)
        )
        self.lifi_advance_routes_url = self._ensure(
            "lifi_advance_routes_url", kwargs, str
        )
        self.lifi_fetch_step_transaction_url = self._ensure(
            "lifi_fetch_step_transaction_url", kwargs, str
        )
        self.lifi_check_status_url = self._ensure("lifi_check_status_url", kwargs, str)
        self.slippage_for_swap = self._ensure("slippage_for_swap", kwargs, float)
        self.allowed_dexs: List[str] = self._ensure("allowed_dexs", kwargs, List[str])
        self.balancer_vault_contract_addresses = json.loads(
            self._ensure("balancer_vault_contract_addresses", kwargs, str)
        )
        self.uniswap_position_manager_contract_addresses = json.loads(
            self._ensure("uniswap_position_manager_contract_addresses", kwargs, str)
        )
        self.chain_to_chain_key_mapping = json.loads(
            self._ensure("chain_to_chain_key_mapping", kwargs, str)
        )
        self.max_num_of_retries = self._ensure("max_num_of_retries", kwargs, int)
        self.waiting_period_for_status_check = self._ensure(
            "waiting_period_for_status_check", kwargs, int
        )
        self.reward_claiming_time_period = self._ensure(
            "reward_claiming_time_period", kwargs, int
        )
        self.merkl_distributor_contract_addresses = json.loads(
            self._ensure("merkl_distributor_contract_addresses", kwargs, str)
        )
        self.intermediate_tokens = json.loads(
            self._ensure("intermediate_tokens", kwargs, str)
        )
        self.lifi_fetch_tools_url = self._ensure("lifi_fetch_tools_url", kwargs, str)
        self.merkl_user_rewards_url = self._ensure(
            "merkl_user_rewards_url", kwargs, str
        )
        self.tenderly_bundle_simulation_url = self._ensure(
            "tenderly_bundle_simulation_url", kwargs, str
        )
        self.tenderly_access_key = self._ensure("tenderly_access_key", kwargs, str)
        self.tenderly_account_slug = self._ensure("tenderly_account_slug", kwargs, str)
        self.tenderly_project_slug = self._ensure("tenderly_project_slug", kwargs, str)
        self.chain_to_chain_id_mapping = json.loads(
            self._ensure("chain_to_chain_id_mapping", kwargs, str)
        )
        self.staking_token_contract_address = self._ensure(
            "staking_token_contract_address", kwargs, str
        )
        self.staking_activity_checker_contract_address = self._ensure(
            "staking_activity_checker_contract_address", kwargs, str
        )
        self.staking_threshold_period = self._ensure(
            "staking_threshold_period", kwargs, int
        )
        self.store_path: Path = self.get_store_path(kwargs)
        self.assets_info_filename: str = self._ensure(
            "assets_info_filename", kwargs, str
        )
        self.pool_info_filename: str = self._ensure("pool_info_filename", kwargs, str)
        self.portfolio_info_filename: str = self._ensure(
            "portfolio_info_filename", kwargs, str
        )
        self.gas_cost_info_filename: str = self._ensure(
            "gas_cost_info_filename", kwargs, str
        )
        self.whitelisted_assets_filename: str = self._ensure(
            "whitelisted_assets_filename", kwargs, str
        )
        self.funding_events_filename: str = self._ensure(
            "funding_events_filename", kwargs, str
        )
        self.min_investment_amount = self._ensure("min_investment_amount", kwargs, int)
        self.max_fee_percentage = self._ensure("max_fee_percentage", kwargs, float)
        self.max_gas_percentage = self._ensure("max_gas_percentage", kwargs, float)
        self.balancer_graphql_endpoints = json.loads(
            self._ensure("balancer_graphql_endpoints", kwargs, str)
        )
        self.target_investment_chains: List[str] = self._ensure(
            "target_investment_chains", kwargs, List[str]
        )
        self.staking_chain: Optional[str] = self._ensure(
            "staking_chain", kwargs, Optional[str]
        )
        self.file_hash_to_strategies = json.loads(
            self._ensure("file_hash_to_strategies", kwargs, str)
        )
        self.strategies_kwargs = json.loads(
            self._ensure("strategies_kwargs", kwargs, str)
        )
        self.available_protocols = self._ensure(
            "available_protocols", kwargs, List[str]
        )
        self.selected_hyper_strategy = self._ensure(
            "selected_hyper_strategy", kwargs, str
        )
        self.dex_type_to_strategy = json.loads(
            self._ensure("dex_type_to_strategy", kwargs, str)
        )
        self.default_acceptance_time = self._ensure(
            "default_acceptance_time", kwargs, int
        )
        self.max_pools = self._ensure("max_pools", kwargs, int)
        self.profit_threshold = self._ensure("profit_threshold", kwargs, int)
        self.loss_threshold = self._ensure("loss_threshold", kwargs, int)
        self.pnl_check_interval = self._ensure("pnl_check_interval", kwargs, int)
        self.available_strategies = json.loads(
            self._ensure("available_strategies", kwargs, str)
        )
        self.cleanup_freq = self._ensure("cleanup_freq", kwargs, int)
        self.genai_api_key = self._ensure("genai_api_key", kwargs, str)
        self.velodrome_router_contract_addresses = json.loads(
            self._ensure("velodrome_router_contract_addresses", kwargs, str)
        )
        self.velodrome_non_fungible_position_manager_contract_addresses = json.loads(
            self._ensure(
                "velodrome_non_fungible_position_manager_contract_addresses",
                kwargs,
                str,
            )
        )
        self.velo_token_contract_addresses = json.loads(
            self._ensure("velo_token_contract_addresses", kwargs, str)
        )
        self.xvelo_token_contract_addresses = json.loads(
            self._ensure("xvelo_token_contract_addresses", kwargs, str)
        )
        self.voter_contract_addresses = json.loads(
            self._ensure("voter_contract_addresses", kwargs, str)
        )
        self.gauge_factory_v1_contract_addresses = json.loads(
            self._ensure("gauge_factory_v1_contract_addresses", kwargs, str)
        )
        self.gauge_factory_v2_contract_addresses = json.loads(
            self._ensure("gauge_factory_v2_contract_addresses", kwargs, str)
        )
        self.service_registry_contract_addresses = json.loads(
            self._ensure("service_registry_contract_addresses", kwargs, str)
        )
        self.staking_subgraph_endpoints = json.loads(
            self._ensure("staking_subgraph_endpoints", kwargs, str)
        )
        super().__init__(*args, **kwargs)

    def get_store_path(self, kwargs: Dict) -> Path:
        """Get the path of the store."""
        path = self._ensure("store_path", kwargs, str)
        # check if path exists, and we can write to it
        if (
            not os.path.isdir(path)
            or not os.access(path, os.W_OK)
            or not os.access(path, os.R_OK)
        ):
            raise ValueError(
                f"Policy store path {path!r} is not a directory or is not writable."
            )
        return Path(path)

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

"""This module contains the 'withdrawal_abci' skill."""

import json
from typing import Any, Dict, Optional

from aea.skills.base import Model
from packages.valory.skills.abstract_round_abci.models import BaseParams


class Params(BaseParams):
    """Model for withdrawal parameters."""

    def __init__(self, **kwargs: Any) -> None:
        
        # Initialize all parameters from skill.yaml
        self.safe_contract_addresses = json.loads(self._ensure("safe_contract_addresses", kwargs, str))
        self.multisend_contract_addresses = json.loads(self._ensure("multisend_contract_addresses", kwargs, str))
        self.balancer_vault_contract_addresses = json.loads(self._ensure("balancer_vault_contract_addresses", kwargs, str))
        self.uniswap_position_manager_contract_addresses = json.loads(self._ensure("uniswap_position_manager_contract_addresses", kwargs, str))
        self.velodrome_non_fungible_position_manager_contract_addresses = json.loads(self._ensure("velodrome_non_fungible_position_manager_contract_addresses", kwargs, str))
        self.target_investment_chains = self._ensure("target_investment_chains", kwargs, list)
        self.initial_assets = json.loads(self._ensure("initial_assets", kwargs, str))
        self.slippage_for_swap = self._ensure("slippage_for_swap", kwargs, float)
        self.chain_to_chain_id_mapping = json.loads(self._ensure("chain_to_chain_id_mapping", kwargs, str))
        self.tenderly_access_key = self._ensure("tenderly_access_key", kwargs, str)
        self.tenderly_account_slug = self._ensure("tenderly_account_slug", kwargs, str)
        self.tenderly_project_slug = self._ensure("tenderly_project_slug", kwargs, str)
        self.store_path = self._ensure("store_path", kwargs, str)
        self.portfolio_info_filename = self._ensure("portfolio_info_filename", kwargs, str)
        self.waiting_period_for_status_check = self._ensure("waiting_period_for_status_check", kwargs, int)
        self.assets_info_filename = self._ensure("assets_info_filename", kwargs, str)
        self.pool_info_filename = self._ensure("pool_info_filename", kwargs, str)
        self.gas_cost_info_filename = self._ensure("gas_cost_info_filename", kwargs, str)
        self.whitelisted_assets_filename = self._ensure("whitelisted_assets_filename", kwargs, str)
        self.funding_events_filename = self._ensure("funding_events_filename", kwargs, str)
        self.merkl_fetch_campaigns_args = json.loads(self._ensure("merkl_fetch_campaigns_args", kwargs, str))
        self.allowed_chains = self._ensure("allowed_chains", kwargs, list)
        self.gas_reserve = json.loads(self._ensure("gas_reserve", kwargs, str))
        self.apr_threshold = self._ensure("apr_threshold", kwargs, int)
        self.round_threshold = self._ensure("round_threshold", kwargs, int)
        self.min_balance_multiplier = self._ensure("min_balance_multiplier", kwargs, int)
        self.lifi_advance_routes_url = self._ensure("lifi_advance_routes_url", kwargs, str)
        self.lifi_fetch_step_transaction_url = self._ensure("lifi_fetch_step_transaction_url", kwargs, str)
        self.lifi_check_status_url = self._ensure("lifi_check_status_url", kwargs, str)
        self.allowed_dexs = self._ensure("allowed_dexs", kwargs, list)
        self.chain_to_chain_key_mapping = json.loads(self._ensure("chain_to_chain_key_mapping", kwargs, str))
        self.max_num_of_retries = self._ensure("max_num_of_retries", kwargs, int)
        self.reward_claiming_time_period = self._ensure("reward_claiming_time_period", kwargs, int)
        self.merkl_distributor_contract_addresses = json.loads(self._ensure("merkl_distributor_contract_addresses", kwargs, str))
        self.intermediate_tokens = json.loads(self._ensure("intermediate_tokens", kwargs, str))
        self.lifi_fetch_tools_url = self._ensure("lifi_fetch_tools_url", kwargs, str)
        self.merkl_user_rewards_url = self._ensure("merkl_user_rewards_url", kwargs, str)
        self.tenderly_bundle_simulation_url = self._ensure("tenderly_bundle_simulation_url", kwargs, str)
        self.staking_token_contract_address = self._ensure("staking_token_contract_address", kwargs, str)
        self.staking_activity_checker_contract_address = self._ensure("staking_activity_checker_contract_address", kwargs, str)
        self.staking_threshold_period = self._ensure("staking_threshold_period", kwargs, int)
        self.min_investment_amount = self._ensure("min_investment_amount", kwargs, int)
        self.max_fee_percentage = self._ensure("max_fee_percentage", kwargs, float)
        self.max_gas_percentage = self._ensure("max_gas_percentage", kwargs, float)
        self.balancer_graphql_endpoints = json.loads(self._ensure("balancer_graphql_endpoints", kwargs, str))
        self.staking_chain = self._ensure("staking_chain", kwargs, str)
        self.file_hash_to_strategies = json.loads(self._ensure("file_hash_to_strategies", kwargs, str))
        self.strategies_kwargs = json.loads(self._ensure("strategies_kwargs", kwargs, str))
        self.available_protocols = self._ensure("available_protocols", kwargs, list)
        self.selected_hyper_strategy = self._ensure("selected_hyper_strategy", kwargs, str)
        self.dex_type_to_strategy = json.loads(self._ensure("dex_type_to_strategy", kwargs, str))
        self.default_acceptance_time = self._ensure("default_acceptance_time", kwargs, int)
        self.max_pools = self._ensure("max_pools", kwargs, int)
        self.profit_threshold = self._ensure("profit_threshold", kwargs, int)
        self.loss_threshold = self._ensure("loss_threshold", kwargs, int)
        self.pnl_check_interval = self._ensure("pnl_check_interval", kwargs, int)
        self.cleanup_freq = self._ensure("cleanup_freq", kwargs, int)
        self.genai_api_key = self._ensure("genai_api_key", kwargs, str)
        self.velodrome_router_contract_addresses = json.loads(self._ensure("velodrome_router_contract_addresses", kwargs, str))
        self.service_registry_contract_addresses = json.loads(self._ensure("service_registry_contract_addresses", kwargs, str))
        
        super().__init__(**kwargs)

class WithdrawalSharedState(Model):
    """Model for withdrawal shared state."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the withdrawal shared state."""
        super().__init__(**kwargs)
        self.withdrawal_id: Optional[str] = None
        self.withdrawal_status: Optional[str] = None
        self.withdrawal_message: Optional[str] = None
        self.withdrawal_target_address: Optional[str] = None
        self.withdrawal_chain: Optional[str] = None
        self.withdrawal_safe_address: Optional[str] = None
        self.withdrawal_tx_link: Optional[str] = None
        self.portfolio_data: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Set a value in the shared state."""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the shared state."""
        return getattr(self, key, default) 
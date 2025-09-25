# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2025 Valory AG
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

"""Integration tests for the valory/optimus agent."""

from pathlib import Path
from typing import Tuple

import pytest
from aea.configurations.data_types import PublicId
from aea_test_autonomy.base_test_classes.agents import (
    BaseTestEnd2EndExecution,
    RoundChecks,
)
from aea_test_autonomy.fixture_helpers import abci_host  # noqa: F401
from aea_test_autonomy.fixture_helpers import abci_port  # noqa: F401
from aea_test_autonomy.fixture_helpers import flask_tendermint  # noqa: F401
from aea_test_autonomy.fixture_helpers import hardhat_addr  # noqa: F401
from aea_test_autonomy.fixture_helpers import hardhat_port  # noqa: F401
from aea_test_autonomy.fixture_helpers import ipfs_daemon  # noqa: F401
from aea_test_autonomy.fixture_helpers import ipfs_domain  # noqa: F401
from aea_test_autonomy.fixture_helpers import key_pairs  # noqa: F401
from aea_test_autonomy.fixture_helpers import tendermint  # noqa: F401
from aea_test_autonomy.fixture_helpers import tendermint_port  # noqa: F401

from packages.valory.agents.optimus.tests.helpers.docker import (
    DEFAULT_JSON_SERVER_ADDR as _DEFAULT_JSON_SERVER_ADDR,
)
from packages.valory.agents.optimus.tests.helpers.docker import (
    DEFAULT_JSON_SERVER_PORT as _DEFAULT_JSON_SERVER_PORT,
)
from packages.valory.agents.optimus.tests.helpers.fixtures import (  # noqa: F401
    UseHardHatOptimusBaseTest,
    UseMockAPIDockerImageBaseTest,
    UseMockLiFiDockerImageBaseTest,
    UseMockTenderlyDockerImageBaseTest,
    UseMockGraphQLDockerImageBaseTest,
)
from packages.valory.skills.registration_abci.rounds import RegistrationStartupRound
from packages.valory.skills.reset_pause_abci.rounds import ResetAndPauseRound

HAPPY_PATH: Tuple[RoundChecks, ...] = (
    RoundChecks(RegistrationStartupRound.auto_round_id(), n_periods=1),
    RoundChecks(ResetAndPauseRound.auto_round_id(), n_periods=2),
)

# strict check log messages of the happy path
STRICT_CHECK_STRINGS = (
    "Period end",
)
PACKAGES_DIR = Path(__file__).parent.parent.parent.parent.parent


# Mock API URLs
MOCK_COINGECKO_URL = f"{_DEFAULT_JSON_SERVER_ADDR}:{_DEFAULT_JSON_SERVER_PORT}/coingecko"
MOCK_BALANCER_URL = f"{_DEFAULT_JSON_SERVER_ADDR}:{_DEFAULT_JSON_SERVER_PORT}/balancer"
MOCK_TENDERLY_URL = f"{_DEFAULT_JSON_SERVER_ADDR}:{_DEFAULT_JSON_SERVER_PORT}/tenderly"
MOCK_LIFI_URL = f"http://127.0.0.1:5000"
MOCK_GRAPHQL_URL = f"http://127.0.0.1:4000"
MOCK_MERKL_URL = f"{_DEFAULT_JSON_SERVER_ADDR}:{_DEFAULT_JSON_SERVER_PORT}/merkl"
MOCK_SAFE_API_URL = f"{_DEFAULT_JSON_SERVER_ADDR}:{_DEFAULT_JSON_SERVER_PORT}/safe"


@pytest.mark.usefixtures("ipfs_daemon")
class BaseTestEnd2EndOptimusNormalExecution(BaseTestEnd2EndExecution):
    """Base class for the optimus service e2e tests."""

    agent_package = "valory/optimus:0.1.0"
    skill_package = "valory/optimus_abci:0.1.0"
    wait_to_finish = 300
    strict_check_strings = STRICT_CHECK_STRINGS
    happy_path = HAPPY_PATH
    package_registry_src_rel = PACKAGES_DIR

    __models_prefix = f"vendor.valory.skills.{PublicId.from_str(skill_package).name}.models"
    __param_args_prefix = f"{__models_prefix}.params.args"
    __coingecko_args_prefix = f"{__models_prefix}.coingecko.args"

    # Set param overrides for all external APIs
    extra_configs = [
        # CoinGecko API endpoints
        {
            "dotted_path": f"{__coingecko_args_prefix}.token_price_endpoint",
            "value": f"{MOCK_COINGECKO_URL}/simple/token_price/{{asset_platform_id}}?contract_addresses={{token_address}}&vs_currencies=usd",
        },
        {
            "dotted_path": f"{__coingecko_args_prefix}.coin_price_endpoint",
            "value": f"{MOCK_COINGECKO_URL}/simple/price?ids={{coin_id}}&vs_currencies=usd",
        },
        {
            "dotted_path": f"{__coingecko_args_prefix}.historical_price_endpoint",
            "value": f"{MOCK_COINGECKO_URL}/coins/{{coin_id}}/history?date={{date}}",
        },
        # LiFi API endpoints
        {
            "dotted_path": f"{__param_args_prefix}.lifi_advance_routes_url",
            "value": f"{MOCK_LIFI_URL}/v1/advanced/routes",
        },
        {
            "dotted_path": f"{__param_args_prefix}.lifi_fetch_step_transaction_url",
            "value": f"{MOCK_LIFI_URL}/v1/advanced/stepTransaction",
        },
        {
            "dotted_path": f"{__param_args_prefix}.lifi_check_status_url",
            "value": f"{MOCK_LIFI_URL}/v1/status",
        },
        {
            "dotted_path": f"{__param_args_prefix}.lifi_fetch_tools_url",
            "value": f"{MOCK_LIFI_URL}/v1/tools",
        },
        # Tenderly API endpoints
        {
            "dotted_path": f"{__param_args_prefix}.tenderly_bundle_simulation_url",
            "value": f"http://127.0.0.1:6000/api/v1/account/{{tenderly_account_slug}}/project/{{tenderly_project_slug}}/simulate-bundle",
        },
        {
            "dotted_path": f"{__param_args_prefix}.tenderly_access_key",
            "value": "test_access_key",
        },
        {
            "dotted_path": f"{__param_args_prefix}.tenderly_account_slug",
            "value": "test_account",
        },
        {
            "dotted_path": f"{__param_args_prefix}.tenderly_project_slug",
            "value": "test_project",
        },
        # Subgraph endpoints
        {
            "dotted_path": f"{__param_args_prefix}.balancer_graphql_endpoints",
            "value": '{"optimism":"' + MOCK_GRAPHQL_URL + '/subgraphs/balancer"}',
        },
        # Merkl API endpoints
        {
            "dotted_path": f"{__param_args_prefix}.merkl_fetch_campaigns_args",
            "value": '{"url":"' + MOCK_MERKL_URL + '/campaigns","creator":"","live":"false"}',
        },
        {
            "dotted_path": f"{__param_args_prefix}.merkl_user_rewards_url",
            "value": f"{MOCK_MERKL_URL}/userRewards",
        },
        # Safe API endpoints
        {
            "dotted_path": f"{__param_args_prefix}.safe_api_base_url",
            "value": f"{MOCK_SAFE_API_URL}/api/v2/safes",
        },
        # Chain configuration
        {
            "dotted_path": f"{__param_args_prefix}.allowed_chains",
            "value": ["optimism"],
        },
        {
            "dotted_path": f"{__param_args_prefix}.target_investment_chains",
            "value": ["optimism"],
        },
        # Strategy configuration for testing
        {
            "dotted_path": f"{__param_args_prefix}.available_strategies",
            "value": '{"optimism":["balancer_pools_search"]}',
        },
        {
            "dotted_path": f"{__param_args_prefix}.selected_hyper_strategy",
            "value": "max_apr_selection",
        },
    ]

    # Set the http server port config
    http_server_port_config = {
        "dotted_path": "vendor.valory.connections.http_server.config.port",
        "value": 8000,
    }

    def _BaseTestEnd2End__set_extra_configs(self) -> None:
        """Set the current agent's extra config overrides that are skill specific."""
        for config in self.extra_configs:
            self.set_config(**config)

        self.set_config(**self.http_server_port_config)
        self.http_server_port_config["value"] += 1  # avoid collisions in multi-agent setups


@pytest.mark.e2e
@pytest.mark.parametrize("nb_nodes", (1,))
class TestEnd2EndOptimusSingleAgent(
    BaseTestEnd2EndOptimusNormalExecution,
    UseMockAPIDockerImageBaseTest,
    UseHardHatOptimusBaseTest,
):
    """Test the optimus with only one agent."""

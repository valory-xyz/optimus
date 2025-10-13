"""E2E test for Optimus agent - Complete flow."""

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
from aea_test_autonomy.fixture_helpers import ipfs_daemon  # noqa: F401
from aea_test_autonomy.fixture_helpers import ipfs_domain  # noqa: F401
from aea_test_autonomy.fixture_helpers import tendermint  # noqa: F401
from aea_test_autonomy.fixture_helpers import tendermint_port  # noqa: F401

from packages.valory.agents.optimus.tests.helpers.constants import (
    PACKAGES_DIR,
    TEST_STORAGE_PATH,
    HARDHAT_RPC,
    TEST_SAFE_ADDRESS,
)
from packages.valory.agents.optimus.tests.helpers.fixtures import (
    UseOptimismHardhatTest,
    UseMockAPIServerTest,
)
from packages.valory.skills.liquidity_trader_abci.rounds import (
    FetchStrategiesRound,
    GetPositionsRound,
    APRPopulationRound,
    EvaluateStrategyRound,
    DecisionMakingRound,
    PostTxSettlementRound,
)


# Complete E2E Flow
HAPPY_PATH: Tuple[RoundChecks, ...] = (
    RoundChecks(FetchStrategiesRound.auto_round_id(), n_periods=1),
    RoundChecks(GetPositionsRound.auto_round_id(), n_periods=1),
    RoundChecks(APRPopulationRound.auto_round_id(), n_periods=1),
    RoundChecks(EvaluateStrategyRound.auto_round_id(), n_periods=1),
    RoundChecks(DecisionMakingRound.auto_round_id(), n_periods=1),
    RoundChecks(PostTxSettlementRound.auto_round_id(), n_periods=1),
)

STRICT_CHECK_STRINGS = (
    "Starting AEA",
    "Reading values from kv store",
    "Fetching balances for optimism safe",
    "Opportunities found using balancer_pools_search strategy",
    "Preparing Safe transaction",
    "Transaction executed successfully",
)


@pytest.mark.usefixtures("ipfs_daemon")
class BaseTestEnd2EndOptimusExecution(BaseTestEnd2EndExecution):
    """Base test class for Optimus E2E tests."""
    
    agent_package = "valory/optimus:0.1.0"
    skill_package = "valory/liquidity_trader_abci:0.1.0"
    wait_to_finish = 300  # 5 minutes for multiple rounds
    strict_check_strings = STRICT_CHECK_STRINGS
    happy_path = HAPPY_PATH
    package_registry_src_rel = PACKAGES_DIR
    
    __params_prefix = f"vendor.valory.skills.{PublicId.from_str(skill_package).name}.models.params.args"
    
    extra_configs = [
        # Storage path - use /tmp/ like working trader-tests
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.store_path",
            "value": "/tmp/",
        },
        # KV store path - configure KV store to use /tmp/
        {
            "dotted_path": "vendor.dvilela.connections.kv_store.config.store_path",
            "value": "/tmp/",
        },
        # Connect to Hardhat
        {
            "dotted_path": "vendor.valory.connections.ledger.config.ledger_apis.optimism.address",
            "value": HARDHAT_RPC,
        },
        # Safe address - use the exact path from aea-config.yaml
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.safe_contract_addresses",
            "value": '{"optimism": "' + TEST_SAFE_ADDRESS + '"}',
        },
        # Mock Balancer subgraph
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.balancer_graphql_endpoints",
            "value": '{"optimism": "http://127.0.0.1:3000/balancer/graphql"}',
        },
        # Balancer vault address
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.balancer_vault_contract_addresses",
            "value": '{"optimism": "0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9"}',
        },
        # MultiSend address
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.multisend_contract_addresses",
            "value": '{"optimism": "0x40A2aCCbd92BCA938b02010E17A5b8929b49130D"}',
        },
        # Configure for optimism chain and add staking subgraph mock
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.staking_chain",
            "value": "optimism",
        },
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.staking_subgraph_endpoints",
            "value": '{"optimism": "http://127.0.0.1:3000/subgraphs/staking"}',
        },
        # Add Safe API base URL for mock
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.safe_api_base_url",
            "value": "http://127.0.0.1:3000/safe-transaction-optimism.safe.global/api/v1/safes",
        },
        # Add CoinGecko API endpoints for mock
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.coingecko.args.token_price_endpoint",
            "value": "http://127.0.0.1:3000/coingecko/simple/token_price/{asset_platform_id}?contract_addresses={token_address}&vs_currencies=usd",
        },
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.coingecko.args.coin_price_endpoint",
            "value": "http://127.0.0.1:3000/coingecko/simple/price?ids={coin_id}&vs_currencies=usd",
        },
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.coingecko.args.historical_price_endpoint",
            "value": "http://127.0.0.1:3000/coingecko/coins/{coin_id}/history?date={date}",
        }
    ]
    
    http_server_port_config = {
        "dotted_path": "vendor.valory.connections.http_server.config.port",
        "value": 8000,
    }
    
    def _BaseTestEnd2End__set_extra_configs(self) -> None:
        """Set extra configs and pre-populate KV store with contract checks."""
        for config in self.extra_configs:
            self.set_config(**config)
        self.set_config(**self.http_server_port_config)
        
        # Pre-populate KV store with contract check results to prevent infinite loops
        self._pre_populate_contract_checks()
    
    def _pre_populate_contract_checks(self) -> None:
        """Pre-populate KV store with contract check results."""
        # Common contract check results - mark most addresses as EOAs to allow transfers
        contract_checks = {
            # Test safe address - EOA
            "contract_check_optimism_0x1234567890123456789012345678901234567890": '{"is_eoa": true}',
            # Agent address - EOA  
            "contract_check_optimism_0xffcf8fdee72ac11b5c542428b35eef5769c409f0": '{"is_eoa": true}',
            # Token contracts - not EOAs
            "contract_check_optimism_0x7f5c764cbc14f9669b88837ca1490cca17c31607": '{"is_eoa": false}',  # USDC
            "contract_check_optimism_0x94b008aa00579c1307b0ef2c499ad98a8ce58e58": '{"is_eoa": false}',  # USDT
            "contract_check_optimism_0x4200000000000000000000000000000000000006": '{"is_eoa": false}',  # WETH
            # Pool contracts - not EOAs
            "contract_check_optimism_0x7b50775383d3d6f0215a8f290f2c9e2eebbeceb2": '{"is_eoa": false}',  # Balancer pool
            "contract_check_optimism_0xdc64a140aa3e981100a9beca4e685f962f0cf6c9": '{"is_eoa": false}',  # Balancer vault
        }
        
        # Add contract checks as KV store configurations
        for key, value in contract_checks.items():
            try:
                self.set_config(
                    dotted_path=f"vendor.dvilela.connections.kv_store.config.initial_data.{key}",
                    value=value,
                    type_="str"
                )
            except Exception as e:
                # If the config path doesn't work, try a different approach
                print(f"Failed to set contract check {key}: {e}")
                continue


@pytest.mark.e2e
@pytest.mark.parametrize("nb_nodes", (1,))
class TestEnd2EndOptimusSingleAgent(
    BaseTestEnd2EndOptimusExecution,
    UseOptimismHardhatTest,
    UseMockAPIServerTest,
):
    """Test Optimus agent - Complete E2E Flow."""
    pass


# 1. Investment Cap Reached - agent has reached $950+ investment threshold
INVESTMENT_CAP_PATH: Tuple[RoundChecks, ...] = (
    RoundChecks(FetchStrategiesRound.auto_round_id(), n_periods=1),
    RoundChecks(GetPositionsRound.auto_round_id(), n_periods=1),
    RoundChecks(APRPopulationRound.auto_round_id(), n_periods=1),
    RoundChecks(EvaluateStrategyRound.auto_round_id(), n_periods=1),
    # No PostTxSettlementRound as investment is capped
)

INVESTMENT_CAP_CHECK_STRINGS = (
    "Starting AEA",
    "Reading values from kv store",
    "Opportunities found using balancer_pools_search strategy",
    "Investment threshold reached, limiting actions",
    "No actions to prepare",
)


@pytest.mark.e2e
@pytest.mark.parametrize("nb_nodes", (1,))
class TestEnd2EndOptimusInvestmentCap(
    BaseTestEnd2EndOptimusExecution,
    UseOptimismHardhatTest,
    UseMockAPIServerTest,
):
    """Test Optimus agent when investment cap is reached."""
    
    strict_check_strings = INVESTMENT_CAP_CHECK_STRINGS
    happy_path = INVESTMENT_CAP_PATH
    wait_to_finish = 240
    
    extra_configs = [
        # Storage path - use /tmp/ like working trader-tests
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.store_path",
            "value": "/tmp/",
        },
        # KV store path - configure KV store to use /tmp/
        {
            "dotted_path": "vendor.dvilela.connections.kv_store.config.store_path",
            "value": "/tmp/",
        },
        # Connect to Hardhat
        {
            "dotted_path": "vendor.valory.connections.ledger.config.ledger_apis.optimism.address",
            "value": HARDHAT_RPC,
        },
        # Safe address - use the exact path from aea-config.yaml
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.safe_contract_addresses",
            "value": '{"optimism": "' + TEST_SAFE_ADDRESS + '"}',
        },
        # Mock Balancer subgraph
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.balancer_graphql_endpoints",
            "value": '{"optimism": "http://127.0.0.1:3000/balancer/graphql"}',
        },
        # Balancer vault address
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.balancer_vault_contract_addresses",
            "value": '{"optimism": "0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9"}',
        },
        # MultiSend address
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.multisend_contract_addresses",
            "value": '{"optimism": "0x40A2aCCbd92BCA938b02010E17A5b8929b49130D"}',
        },
        # Configure for optimism chain and add staking subgraph mock
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.staking_chain",
            "value": "optimism",
        },
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.staking_subgraph_endpoints",
            "value": '{"optimism": "http://127.0.0.1:3000/subgraphs/staking"}',
        },
        # Add Safe API base URL for mock
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.safe_api_base_url",
            "value": "http://127.0.0.1:3000/safe-transaction-optimism.safe.global/api",
        },
        # Add CoinGecko API endpoints for mock
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.coingecko.args.token_price_endpoint",
            "value": "http://127.0.0.1:3000/coingecko/simple/token_price/{asset_platform_id}?contract_addresses={token_address}&vs_currencies=usd",
        },
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.coingecko.args.coin_price_endpoint",
            "value": "http://127.0.0.1:3000/coingecko/simple/price?ids={coin_id}&vs_currencies=usd",
        },
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.coingecko.args.historical_price_endpoint",
            "value": "http://127.0.0.1:3000/coingecko/coins/{coin_id}/history?date={date}",
        }
    ]


# 2. TiP Blocking Exit - positions blocked by Time in Position requirements
TIP_BLOCKING_PATH: Tuple[RoundChecks, ...] = (
    RoundChecks(FetchStrategiesRound.auto_round_id(), n_periods=1),
    RoundChecks(GetPositionsRound.auto_round_id(), n_periods=1),
    RoundChecks(APRPopulationRound.auto_round_id(), n_periods=1),
    RoundChecks(EvaluateStrategyRound.auto_round_id(), n_periods=1),
)

TIP_BLOCKING_CHECK_STRINGS = (
    "Starting AEA",
    "Reading values from kv store",
    "All positions blocked by TiP conditions",
    "TiP blocking exit",
    "No actions to prepare",
)


@pytest.mark.e2e
@pytest.mark.parametrize("nb_nodes", (1,))
class TestEnd2EndOptimusTiPBlocking(
    BaseTestEnd2EndOptimusExecution,
    UseOptimismHardhatTest,
    UseMockAPIServerTest,
):
    """Test Optimus agent when positions are blocked by TiP requirements."""
    
    strict_check_strings = TIP_BLOCKING_CHECK_STRINGS
    happy_path = TIP_BLOCKING_PATH
    wait_to_finish = 240
    
    extra_configs = [
        # Storage path - use /tmp/ like working trader-tests
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.store_path",
            "value": "/tmp/",
        },
        # KV store path - configure KV store to use /tmp/
        {
            "dotted_path": "vendor.dvilela.connections.kv_store.config.store_path",
            "value": "/tmp/",
        },
        # Connect to Hardhat
        {
            "dotted_path": "vendor.valory.connections.ledger.config.ledger_apis.optimism.address",
            "value": HARDHAT_RPC,
        },
        # Safe address - use the exact path from aea-config.yaml
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.safe_contract_addresses",
            "value": '{"optimism": "' + TEST_SAFE_ADDRESS + '"}',
        },
        # Mock Balancer subgraph
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.balancer_graphql_endpoints",
            "value": '{"optimism": "http://127.0.0.1:3000/balancer/graphql"}',
        },
        # Balancer vault address
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.balancer_vault_contract_addresses",
            "value": '{"optimism": "0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9"}',
        },
        # MultiSend address
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.multisend_contract_addresses",
            "value": '{"optimism": "0x40A2aCCbd92BCA938b02010E17A5b8929b49130D"}',
        },
        # Configure for optimism chain and add staking subgraph mock
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.staking_chain",
            "value": "optimism",
        },
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.staking_subgraph_endpoints",
            "value": '{"optimism": "http://127.0.0.1:3000/subgraphs/staking"}',
        },
        # Add Safe API base URL for mock
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.safe_api_base_url",
            "value": "http://127.0.0.1:3000/safe-transaction-optimism.safe.global/api",
        },
        # Add CoinGecko API endpoints for mock
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.coingecko.args.token_price_endpoint",
            "value": "http://127.0.0.1:3000/coingecko/simple/token_price/{asset_platform_id}?contract_addresses={token_address}&vs_currencies=usd",
        },
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.coingecko.args.coin_price_endpoint",
            "value": "http://127.0.0.1:3000/coingecko/simple/price?ids={coin_id}&vs_currencies=usd",
        },
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.coingecko.args.historical_price_endpoint",
            "value": "http://127.0.0.1:3000/coingecko/coins/{coin_id}/history?date={date}",
        }
    ]


# 3. Withdrawal Mode - agent is in withdrawal mode and exits all positions
WITHDRAWAL_MODE_PATH: Tuple[RoundChecks, ...] = (
    RoundChecks(FetchStrategiesRound.auto_round_id(), n_periods=1),
    RoundChecks(GetPositionsRound.auto_round_id(), n_periods=1),
    RoundChecks(APRPopulationRound.auto_round_id(), n_periods=1),
    RoundChecks(EvaluateStrategyRound.auto_round_id(), n_periods=1),
)

WITHDRAWAL_MODE_CHECK_STRINGS = (
    "Starting AEA",
    "Reading values from kv store",
    "Investing paused due to withdrawal request",
)


@pytest.mark.e2e
@pytest.mark.parametrize("nb_nodes", (1,))
class TestEnd2EndOptimusWithdrawalMode(
    BaseTestEnd2EndOptimusExecution,
    UseOptimismHardhatTest,
    UseMockAPIServerTest,
):
    """Test Optimus agent when in withdrawal mode."""
    
    strict_check_strings = WITHDRAWAL_MODE_CHECK_STRINGS
    happy_path = WITHDRAWAL_MODE_PATH
    wait_to_finish = 240
    
    extra_configs = [
        # Storage path - use /tmp/ like working trader-tests
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.store_path",
            "value": "/tmp/",
        },
        # KV store path - configure KV store to use /tmp/
        {
            "dotted_path": "vendor.dvilela.connections.kv_store.config.store_path",
            "value": "/tmp/",
        },
        # Connect to Hardhat
        {
            "dotted_path": "vendor.valory.connections.ledger.config.ledger_apis.optimism.address",
            "value": HARDHAT_RPC,
        },
        # Safe address - use the exact path from aea-config.yaml
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.safe_contract_addresses",
            "value": '{"optimism": "' + TEST_SAFE_ADDRESS + '"}',
        },
        # Mock Balancer subgraph
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.balancer_graphql_endpoints",
            "value": '{"optimism": "http://127.0.0.1:3000/balancer/graphql"}',
        },
        # Balancer vault address
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.balancer_vault_contract_addresses",
            "value": '{"optimism": "0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9"}',
        },
        # MultiSend address
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.multisend_contract_addresses",
            "value": '{"optimism": "0x40A2aCCbd92BCA938b02010E17A5b8929b49130D"}',
        },
        # Configure for optimism chain and add staking subgraph mock
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.staking_chain",
            "value": "optimism",
        },
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.params.args.staking_subgraph_endpoints",
            "value": '{"optimism": "http://127.0.0.1:3000/subgraphs/staking"}',
        },
        # Add Safe API base URL for mock
        {
            "dotted_path": f"vendor.valory.skills.optimus_abci.models.params.args.safe_api_base_url",
            "value": "http://127.0.0.1:3000/safe-transaction-optimism.safe.global/api",
        },
        # Add CoinGecko API endpoints for mock
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.coingecko.args.token_price_endpoint",
            "value": "http://127.0.0.1:3000/coingecko/simple/token_price/{asset_platform_id}?contract_addresses={token_address}&vs_currencies=usd",
        },
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.coingecko.args.coin_price_endpoint",
            "value": "http://127.0.0.1:3000/coingecko/simple/price?ids={coin_id}&vs_currencies=usd",
        },
        {
            "dotted_path": "vendor.valory.skills.optimus_abci.models.coingecko.args.historical_price_endpoint",
            "value": "http://127.0.0.1:3000/coingecko/coins/{coin_id}/history?date={date}",
        },
        # Enable withdrawal mode by setting investing_paused flag
        {
            "dotted_path": "vendor.dvilela.connections.kv_store.config.initial_data.investing_paused",
            "value": "true",
        },
        # Set withdrawal status to INITIATED
        {
            "dotted_path": "vendor.dvilela.connections.kv_store.config.initial_data.withdrawal_status",
            "value": "INITIATED",
        }
    ]

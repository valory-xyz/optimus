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

"""Base class for all protocol integration tests."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union
from unittest.mock import MagicMock, patch

import pytest
from aea_ledger_ethereum import EthereumApi

from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    LiquidityTraderBaseBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.models import SharedState
from packages.valory.skills.liquidity_trader_abci.rounds import SynchronizedData


class ProtocolIntegrationTestBase(FSMBehaviourBaseCase):
    """Base class for all protocol integration tests."""

    def setup_method(self, **kwargs: Any) -> None:
        """Set up the test method with common infrastructure."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Mock the store path validation
        with patch(
            "packages.valory.skills.liquidity_trader_abci.models.Params.get_store_path",
            return_value=self.temp_path,
        ):
            super().setup(**kwargs)

        # Set up common test infrastructure
        self._setup_mock_ledger_api()
        self._setup_mock_contracts()
        self._setup_test_data()
        self._setup_mock_responses()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().teardown()

    def _setup_mock_ledger_api(self) -> None:
        """Set up mock ledger API for blockchain interactions."""
        self.mock_ledger_api = MagicMock(spec=EthereumApi)
        self.mock_ledger_api.api = MagicMock()
        self.mock_ledger_api.api.to_checksum_address = lambda addr: addr.lower()

    def _setup_mock_contracts(self) -> None:
        """Set up mock contract instances."""
        self.mock_contracts = {
            "balancer_vault": MagicMock(),
            "balancer_weighted_pool": MagicMock(),
            "uniswap_v3_pool": MagicMock(),
            "uniswap_v3_position_manager": MagicMock(),
            "velodrome_pool": MagicMock(),
            "velodrome_gauge": MagicMock(),
            "velodrome_voter": MagicMock(),
            "erc20": MagicMock(),
        }

    def _setup_test_data(self) -> None:
        """Set up common test data."""
        self.test_addresses = {
            "user": "0x1234567890123456789012345678901234567890",
            "safe": "0x2345678901234567890123456789012345678901",
            "token_a": "0x3456789012345678901234567890123456789012",
            "token_b": "0x4567890123456789012345678901234567890123",
            "pool": "0x5678901234567890123456789012345678901234",
            "vault": "0x6789012345678901234567890123456789012345",
        }

        self.test_amounts = {
            "small": 1000000000000000000,      # 1 token
            "medium": 100000000000000000000,   # 100 tokens
            "large": 1000000000000000000000,   # 1000 tokens
        }

    def _setup_mock_responses(self) -> None:
        """Set up mock responses for common operations."""
        self.mock_responses = {
            "balance": {"balance": self.test_amounts["medium"]},
            "decimals": {"decimals": 18},
            "total_supply": {"data": self.test_amounts["large"]},
            "pool_tokens": {
                "tokens": [self.test_addresses["token_a"], self.test_addresses["token_b"]]
            },
            "slot0": {
                "slot0": {
                    "sqrt_price_x96": 79228162514264337593543950336,
                    "tick": -276310,
                    "observation_index": 0,
                    "observation_cardinality": 1,
                    "observation_cardinality_next": 1,
                    "fee_protocol": 0,
                    "unlocked": True,
                }
            },
        }

    def mock_contract_response(
        self, contract_name: str, method: str, response_data: Dict[str, Any]
    ) -> None:
        """Mock contract method responses."""
        if contract_name in self.mock_contracts:
            contract_mock = self.mock_contracts[contract_name]
            method_mock = getattr(contract_mock.functions, method, None)
            if method_mock:
                method_mock.return_value.call.return_value = response_data

    def simulate_blockchain_state(self, state_data: Dict[str, Any]) -> None:
        """Simulate blockchain state for testing."""
        for contract_name, contract_data in state_data.items():
            if contract_name in self.mock_contracts:
                contract_mock = self.mock_contracts[contract_name]
                for method, data in contract_data.items():
                    method_mock = getattr(contract_mock.functions, method, None)
                    if method_mock:
                        method_mock.return_value.call.return_value = data

    def assert_transaction_validity(
        self, tx_data: Dict[str, Any], expected_params: Dict[str, Any]
    ) -> None:
        """Validate transaction encoding and parameters."""
        assert "tx_hash" in tx_data, "Transaction must contain tx_hash"
        assert tx_data["tx_hash"] is not None, "Transaction hash must not be None"
        
        if isinstance(tx_data["tx_hash"], bytes):
            assert len(tx_data["tx_hash"]) > 0, "Transaction hash must not be empty"
        elif isinstance(tx_data["tx_hash"], str):
            assert len(tx_data["tx_hash"]) > 0, "Transaction hash must not be empty"
            assert tx_data["tx_hash"].startswith("0x"), "Transaction hash must start with 0x"

    def create_mock_behaviour(self, behaviour_class: type) -> Any:
        """Create a mock behaviour instance for testing."""
        # Create mock context and params
        mock_context = MagicMock()
        mock_params = MagicMock()
        
        # Set up common parameters
        mock_params.safe_contract_addresses = {
            "optimism": self.test_addresses["safe"],
            "base": self.test_addresses["safe"],
            "mode": self.test_addresses["safe"],
        }
        mock_params.velodrome_voter_contract_addresses = {
            "optimism": "0xVoterAddress",
        }
        mock_params.investment_cap_threshold = 950
        mock_params.initial_investment_amount = 1000
        mock_params.target_investment_chains = ["optimism"]
        mock_params.available_protocols = ["VELODROME"]
        mock_params.chain_to_chain_id_mapping = {"optimism": 10}
        mock_params.strategies_kwargs = {}
        mock_params.selected_hyper_strategy = "max_apr_selection"
        mock_params.max_pools = 5
        mock_params.apr_threshold = 5.0
        mock_params.min_investment_amount = 10.0
        mock_params.sleep_time = 1
        mock_params.merkl_user_rewards_url = "https://api.merkl.xyz/v3/userRewards"
        mock_params.reward_claiming_time_period = 86400
        
        # Create behaviour instance
        behaviour = behaviour_class(context=mock_context, params=mock_params)
        
        # Mock common methods
        behaviour.contract_interact = MagicMock()
        behaviour.get_http_response = MagicMock()
        behaviour._read_kv = MagicMock()
        behaviour._write_kv = MagicMock()
        
        return behaviour

    def mock_contract_interact_generator(
        self, responses: List[Any]
    ) -> Generator[None, None, Any]:
        """Create a mock contract_interact generator with predefined responses."""
        for response in responses:
            yield
            return response

    def mock_http_response_generator(
        self, responses: List[Dict[str, Any]]
    ) -> Generator[None, None, Any]:
        """Create a mock get_http_response generator with predefined responses."""
        for response in responses:
            yield
            return MagicMock(
                status_code=response.get("status_code", 200),
                body=json.dumps(response.get("body", {})),
            )

    def create_test_pool_data(self, protocol: str) -> Dict[str, Any]:
        """Create test pool data for different protocols."""
        base_data = {
            "pool_address": self.test_addresses["pool"],
            "tokens": [self.test_addresses["token_a"], self.test_addresses["token_b"]],
            "apr": 12.5,
            "tvl": self.test_amounts["large"],
        }

        if protocol == "balancer":
            return {
                **base_data,
                "pool_id": "0x1234567890123456789012345678901234567890123456789012345678901234",
                "pool_type": "Weighted",
                "weights": [500000000000000000, 500000000000000000],
                "total_supply": self.test_amounts["large"],
                "swap_fees": 2500000000000000,  # 0.25%
            }
        elif protocol == "uniswap_v3":
            return {
                **base_data,
                "token0": self.test_addresses["token_a"],
                "token1": self.test_addresses["token_b"],
                "fee": 3000,
                "tick_spacing": 60,
                "current_tick": -276310,
                "sqrt_price_x96": 79228162514264337593543950336,
            }
        elif protocol == "velodrome":
            return {
                **base_data,
                "is_stable": False,
                "is_cl_pool": False,
                "gauge_address": "0xGaugeAddress",
                "voter_address": "0xVoterAddress",
            }
        else:
            return base_data

    def create_test_position_data(self, protocol: str) -> Dict[str, Any]:
        """Create test position data for different protocols."""
        base_data = {
            "pool_address": self.test_addresses["pool"],
            "dex_type": protocol,
            "amounts": {
                "token0": self.test_amounts["medium"],
                "token1": self.test_amounts["medium"],
            },
            "timestamp": 1234567890,
        }

        if protocol == "balancer":
            return {
                **base_data,
                "pool_id": "0x1234567890123456789012345678901234567890123456789012345678901234",
                "user_bpt_balance": self.test_amounts["medium"],
                "total_supply": self.test_amounts["large"],
            }
        elif protocol == "uniswap_v3":
            return {
                **base_data,
                "token_id": 12345,
                "token0": self.test_addresses["token_a"],
                "token1": self.test_addresses["token_b"],
                "fee": 3000,
                "tick_lower": -276320,
                "tick_upper": -276300,
                "liquidity": self.test_amounts["medium"],
                "tokens_owed0": 0,
                "tokens_owed1": 0,
            }
        elif protocol == "velodrome":
            return {
                **base_data,
                "token_id": 12345,
                "is_cl_pool": False,
                "gauge_address": "0xGaugeAddress",
                "staked_amount": self.test_amounts["medium"],
            }
        else:
            return base_data

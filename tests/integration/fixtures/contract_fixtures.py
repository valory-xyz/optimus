# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
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

"""Contract fixtures for integration tests."""

import pytest
from unittest.mock import MagicMock

from tests.integration.protocols.base.mock_contracts import MockContractFactory


@pytest.fixture
def mock_ledger_api():
    """Fixture for mock ledger API."""
    return MockContractFactory.create_ledger_api_mock()


@pytest.fixture
def balancer_vault_contract():
    """Fixture for Balancer Vault contract."""
    return MockContractFactory.create_balancer_vault_mock()


@pytest.fixture
def balancer_weighted_pool_contract():
    """Fixture for Balancer Weighted Pool contract."""
    return MockContractFactory.create_balancer_weighted_pool_mock()


@pytest.fixture
def uniswap_v3_pool_contract():
    """Fixture for Uniswap V3 Pool contract."""
    return MockContractFactory.create_uniswap_v3_pool_mock()


@pytest.fixture
def uniswap_v3_position_manager_contract():
    """Fixture for Uniswap V3 Position Manager contract."""
    return MockContractFactory.create_uniswap_v3_position_manager_mock()


@pytest.fixture
def velodrome_pool_contract():
    """Fixture for Velodrome Pool contract."""
    return MockContractFactory.create_velodrome_pool_mock()


@pytest.fixture
def velodrome_gauge_contract():
    """Fixture for Velodrome Gauge contract."""
    return MockContractFactory.create_velodrome_gauge_mock()


@pytest.fixture
def velodrome_voter_contract():
    """Fixture for Velodrome Voter contract."""
    return MockContractFactory.create_velodrome_voter_mock()


@pytest.fixture
def erc20_contract():
    """Fixture for ERC20 contract."""
    return MockContractFactory.create_erc20_mock()


@pytest.fixture
def multisend_contract():
    """Fixture for MultiSend contract."""
    return MockContractFactory.create_multisend_mock()


@pytest.fixture
def all_contract_mocks():
    """Fixture for all contract mocks."""
    return MockContractFactory.create_all_contract_mocks()


@pytest.fixture
def mock_contract_factory():
    """Fixture for the mock contract factory."""
    return MockContractFactory

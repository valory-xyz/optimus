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

"""Pytest configuration for integration tests."""

import pytest
import tempfile
from pathlib import Path
from typing import Generator

from tests.integration.protocols.base.mock_contracts import MockContractFactory


@pytest.fixture(scope="session")
def temp_test_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    yield temp_path
    # Cleanup is handled by the test framework


@pytest.fixture(scope="session")
def mock_contract_factory() -> MockContractFactory:
    """Provide mock contract factory for all tests."""
    return MockContractFactory


@pytest.fixture(scope="session")
def all_contract_mocks():
    """Provide all contract mocks for integration tests."""
    return MockContractFactory.create_all_contract_mocks()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment for each test."""
    # This fixture runs before each test
    # Add any global test setup here
    pass


@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Clean up test environment after each test."""
    # This fixture runs after each test
    # Add any global test cleanup here
    yield
    # Cleanup code here


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "contract: mark test as contract integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "yield: mark test as yield calculation test"
    )
    config.addinivalue_line(
        "markers", "transaction: mark test as transaction generation test"
    )
    config.addinivalue_line(
        "markers", "balancer: mark test as Balancer protocol test"
    )
    config.addinivalue_line(
        "markers", "uniswap_v3: mark test as Uniswap V3 protocol test"
    )
    config.addinivalue_line(
        "markers", "velodrome: mark test as Velodrome protocol test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add protocol markers based on test file path
        if "balancer" in str(item.fspath):
            item.add_marker(pytest.mark.balancer)
        elif "uniswap_v3" in str(item.fspath):
            item.add_marker(pytest.mark.uniswap_v3)
        elif "velodrome" in str(item.fspath):
            item.add_marker(pytest.mark.velodrome)
        
        # Add test type markers based on test file name
        if "components" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "contract_integration" in str(item.fspath):
            item.add_marker(pytest.mark.contract)
        elif "e2e_workflows" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "yield_calculations" in str(item.fspath):
            item.add_marker(pytest.mark.yield)
        elif "transaction_generation" in str(item.fspath):
            item.add_marker(pytest.mark.transaction)
        
        # All tests in integration directory are integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

"""Pytest fixtures for E2E tests."""

import logging
from typing import Generator

import docker
import pytest
from aea_test_autonomy.docker.base import launch_image

from packages.valory.agents.optimus.tests.helpers.docker import (
    OptimismHardhatImage,
    MockAPIServerImage,
)
from packages.valory.agents.optimus.tests.helpers.constants import (
    HARDHAT_ADDRESS,
    HARDHAT_PORT,
)


@pytest.fixture(scope="session")
def ipfs_daemon() -> Generator:
    """Mock IPFS daemon fixture for testing."""
    logging.info("Starting mock IPFS daemon for testing...")
    yield
    logging.info("Stopping mock IPFS daemon...")


@pytest.mark.integration
class UseOptimismHardhatTest:
    """Fixture to start Hardhat with Optimism setup."""
    
    NETWORK_ADDRESS = HARDHAT_ADDRESS
    NETWORK_PORT = HARDHAT_PORT
    
    @classmethod
    @pytest.fixture(autouse=True)
    def _start_hardhat(
        cls,
        timeout: int = 5,
        max_attempts: int = 60,
    ) -> Generator:
        """Start Hardhat with deployed contracts."""
        client = docker.from_env()
        logging.info(f"Launching Hardhat on {cls.NETWORK_ADDRESS}:{cls.NETWORK_PORT}")
        
        image = OptimismHardhatImage(
            client,
            addr=cls.NETWORK_ADDRESS,
            port=cls.NETWORK_PORT,
        )
        
        yield from launch_image(image, timeout=timeout, max_attempts=max_attempts)


@pytest.mark.integration
class UseMockAPIServerTest:
    """Fixture to start mock API server."""
    
    @classmethod
    @pytest.fixture(autouse=True)
    def _start_mock_api(
        cls,
        timeout: int = 5,
        max_attempts: int = 30,
    ) -> Generator:
        """Start mock API server using existing JSON server."""
        logging.info("Mock API Server ready (using existing JSON server)")
        yield
        logging.info("Mock API Server stopped")

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
# pylint: disable=import-error

"""The Optimus fixtures."""
import logging
from typing import Generator, List, Tuple

import docker
import pytest
from aea_test_autonomy.docker.base import launch_image

from packages.valory.agents.optimus.tests.helpers.constants import ACCOUNTS
from packages.valory.agents.optimus.tests.helpers.docker import (
    DEFAULT_HARDHAT_ADDR,
    DEFAULT_HARDHAT_PORT,
    DEFAULT_JSON_SERVER_ADDR,
    DEFAULT_JSON_SERVER_PORT,
    OptimusNetworkDockerImage,
    MockAPIDockerImage,
)
from packages.valory.agents.optimus.tests.helpers.mock_servers import (
    DEFAULT_GRAPHQL_ADDR,
    DEFAULT_GRAPHQL_PORT,
    DEFAULT_LIFI_ADDR,
    DEFAULT_LIFI_PORT,
    DEFAULT_TENDERLY_ADDR,
    DEFAULT_TENDERLY_PORT,
    MockGraphQLDockerImage,
    MockLiFiDockerImage,
    MockTenderlyDockerImage,
)


@pytest.mark.integration
class UseHardHatOptimusBaseTest:  # pylint: disable=too-few-public-methods
    """Inherit from this class to use HardHat local net with basic contracts deployed."""

    key_pairs: List[Tuple[str, str]] = ACCOUNTS
    NETWORK_ADDRESS = DEFAULT_HARDHAT_ADDR
    NETWORK_PORT = DEFAULT_HARDHAT_PORT

    @classmethod
    @pytest.fixture(autouse=True)
    def _start_network(
        cls,
        timeout: int = 3,
        max_attempts: int = 200,
    ) -> Generator:
        """Start a HardHat instance."""
        client = docker.from_env()
        logging.info(
            f"Launching the Optimus network on {cls.NETWORK_ADDRESS}:{cls.NETWORK_PORT}"
        )
        image = OptimusNetworkDockerImage(
            client,
            addr=cls.NETWORK_ADDRESS,
            port=cls.NETWORK_PORT,
        )
        yield from launch_image(image, timeout=timeout, max_attempts=max_attempts)


@pytest.mark.integration
class UseMockAPIDockerImageBaseTest:  # pylint: disable=too-few-public-methods
    """Inherit from this class to use a mock API server."""

    MOCK_API_ADDRESS = DEFAULT_JSON_SERVER_ADDR
    MOCK_API_PORT = DEFAULT_JSON_SERVER_PORT

    @classmethod
    @pytest.fixture(autouse=True)
    def _start_mock_api(
        cls,
        timeout: int = 3,
        max_attempts: int = 200,
    ) -> Generator:
        """Start a Mock API instance."""
        client = docker.from_env()
        logging.info(f"Launching the Mock API on {cls.MOCK_API_ADDRESS}:{cls.MOCK_API_PORT}")
        image = MockAPIDockerImage(
            client,
            addr=cls.MOCK_API_ADDRESS,
            port=cls.MOCK_API_PORT,
        )
        yield from launch_image(image, timeout=timeout, max_attempts=max_attempts)


@pytest.mark.integration
class UseMockLiFiDockerImageBaseTest:  # pylint: disable=too-few-public-methods
    """Inherit from this class to use a mock LiFi API server."""

    MOCK_LIFI_ADDRESS = DEFAULT_LIFI_ADDR
    MOCK_LIFI_PORT = DEFAULT_LIFI_PORT

    @classmethod
    @pytest.fixture(autouse=True)
    def _start_mock_lifi(
        cls,
        timeout: int = 3,
        max_attempts: int = 200,
    ) -> Generator:
        """Start a Mock LiFi API instance."""
        client = docker.from_env()
        logging.info(f"Launching the Mock LiFi API on {cls.MOCK_LIFI_ADDRESS}:{cls.MOCK_LIFI_PORT}")
        image = MockLiFiDockerImage(
            client,
            addr=cls.MOCK_LIFI_ADDRESS,
            port=cls.MOCK_LIFI_PORT,
        )
        yield from launch_image(image, timeout=timeout, max_attempts=max_attempts)


@pytest.mark.integration
class UseMockTenderlyDockerImageBaseTest:  # pylint: disable=too-few-public-methods
    """Inherit from this class to use a mock Tenderly API server."""

    MOCK_TENDERLY_ADDRESS = DEFAULT_TENDERLY_ADDR
    MOCK_TENDERLY_PORT = DEFAULT_TENDERLY_PORT

    @classmethod
    @pytest.fixture(autouse=True)
    def _start_mock_tenderly(
        cls,
        timeout: int = 3,
        max_attempts: int = 200,
    ) -> Generator:
        """Start a Mock Tenderly API instance."""
        client = docker.from_env()
        logging.info(f"Launching the Mock Tenderly API on {cls.MOCK_TENDERLY_ADDRESS}:{cls.MOCK_TENDERLY_PORT}")
        image = MockTenderlyDockerImage(
            client,
            addr=cls.MOCK_TENDERLY_ADDRESS,
            port=cls.MOCK_TENDERLY_PORT,
        )
        yield from launch_image(image, timeout=timeout, max_attempts=max_attempts)


@pytest.mark.integration
class UseMockGraphQLDockerImageBaseTest:  # pylint: disable=too-few-public-methods
    """Inherit from this class to use a mock GraphQL server for subgraphs."""

    MOCK_GRAPHQL_ADDRESS = DEFAULT_GRAPHQL_ADDR
    MOCK_GRAPHQL_PORT = DEFAULT_GRAPHQL_PORT

    @classmethod
    @pytest.fixture(autouse=True)
    def _start_mock_graphql(
        cls,
        timeout: int = 3,
        max_attempts: int = 200,
    ) -> Generator:
        """Start a Mock GraphQL instance."""
        client = docker.from_env()
        logging.info(f"Launching the Mock GraphQL server on {cls.MOCK_GRAPHQL_ADDRESS}:{cls.MOCK_GRAPHQL_PORT}")
        image = MockGraphQLDockerImage(
            client,
            addr=cls.MOCK_GRAPHQL_ADDRESS,
            port=cls.MOCK_GRAPHQL_PORT,
        )
        yield from launch_image(image, timeout=timeout, max_attempts=max_attempts)

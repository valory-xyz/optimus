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

"""Mock servers for complex API interactions."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import docker
import requests
from aea.exceptions import enforce
from aea_test_autonomy.docker.base import DockerImage
from docker.models.containers import Container

from packages.valory.agents.optimus import PACKAGE_DIR


DEFAULT_GRAPHQL_ADDR = "http://127.0.0.1"
DEFAULT_GRAPHQL_PORT = 4000
DEFAULT_GRAPHQL_DATA_DIR = (
    PACKAGE_DIR / "tests" / "helpers" / "data" / "subgraph_responses.json"
)


class MockGraphQLDockerImage(DockerImage):
    """Spawn a GraphQL mock server for subgraph endpoints."""

    def __init__(
        self,
        client: docker.DockerClient,
        addr: str = DEFAULT_GRAPHQL_ADDR,
        port: int = DEFAULT_GRAPHQL_PORT,
        data_file: Path = DEFAULT_GRAPHQL_DATA_DIR,
    ):
        """Initialize."""
        super().__init__(client)
        self.addr = addr
        self.port = port
        self.data_file = data_file

    def create_many(self, nb_containers: int) -> List[Container]:
        """Instantiate the image in many containers, parametrized."""
        raise NotImplementedError()

    @property
    def image(self) -> str:
        """Get the image."""
        return "graphql/graphql-playground:latest"

    def create(self) -> Container:
        """Create the container."""
        # Use a simple HTTP server that can handle GraphQL-like requests
        ports = {"4000/tcp": ("0.0.0.0", self.port)}  # nosec
        
        # Create a simple GraphQL mock using Node.js
        container = self._client.containers.run(
            "node:16-alpine",
            command=[
                "sh", "-c", 
                """
                npm install -g json-server &&
                echo '{"data": {"pools": []}}' > /tmp/graphql.json &&
                json-server --watch /tmp/graphql.json --port 4000 --host 0.0.0.0
                """
            ],
            detach=True,
            ports=ports,
            extra_hosts={"host.docker.internal": "host-gateway"},
        )
        return container

    def wait(self, max_attempts: int = 15, sleep_rate: float = 1.0) -> bool:
        """
        Wait until the image is running.
        :param max_attempts: max number of attempts.
        :param sleep_rate: the amount of time to sleep between different requests.
        :return: True if the wait was successful, False otherwise.
        """
        for i in range(max_attempts):
            try:
                response = requests.get(f"{self.addr}:{self.port}", timeout=5)
                enforce(response.status_code == 200, "")
                return True
            except Exception as e:  # pylint: disable=broad-except
                logging.error("Exception: %s: %s", type(e).__name__, str(e))
                logging.info(
                    "Attempt %s failed. Retrying in %s seconds...", i, sleep_rate
                )
                time.sleep(sleep_rate)
        return False


DEFAULT_LIFI_ADDR = "http://127.0.0.1"
DEFAULT_LIFI_PORT = 5000


class MockLiFiDockerImage(DockerImage):
    """Spawn a dedicated LiFi API mock server."""

    def __init__(
        self,
        client: docker.DockerClient,
        addr: str = DEFAULT_LIFI_ADDR,
        port: int = DEFAULT_LIFI_PORT,
    ):
        """Initialize."""
        super().__init__(client)
        self.addr = addr
        self.port = port

    def create_many(self, nb_containers: int) -> List[Container]:
        """Instantiate the image in many containers, parametrized."""
        raise NotImplementedError()

    @property
    def image(self) -> str:
        """Get the image."""
        return "node:16-alpine"

    def create(self) -> Container:
        """Create the container."""
        ports = {"5000/tcp": ("0.0.0.0", self.port)}  # nosec
        
        # Create a mock LiFi server with proper endpoints
        lifi_mock_script = """
const express = require('express');
const app = express();
app.use(express.json());

// Mock LiFi routes endpoint
app.post('/v1/advanced/routes', (req, res) => {
  res.json({
    routes: [{
      id: 'route-1',
      fromChainId: 10,
      toChainId: 10,
      fromAmountUSD: '1000.0',
      toAmountUSD: '995.0',
      steps: [{
        tool: 'velodrome',
        action: req.body,
        estimate: {
          gasCosts: [{ amount: '21000', amountUSD: '0.05' }],
          feeCosts: [{ amount: '1000000', amountUSD: '1.0' }]
        }
      }]
    }]
  });
});

// Mock LiFi step transaction endpoint
app.post('/v1/advanced/stepTransaction', (req, res) => {
  res.json({
    transactionRequest: {
      to: '0xa062aE8A9c5e11aaA026fc2670B0D65cCc8B2858',
      data: '0x1234567890abcdef',
      value: '0x0'
    },
    estimate: {
      fromAmount: '400000000000000000',
      toAmount: '995000000',
      gasCosts: [{ amountUSD: '0.05' }],
      feeCosts: [{ amountUSD: '1.0' }]
    }
  });
});

// Mock LiFi status endpoint
app.get('/v1/status', (req, res) => {
  res.json({
    status: 'DONE',
    substatus: 'COMPLETED',
    sending: { amountUSD: '1000.0' },
    receiving: { amountUSD: '995.0' }
  });
});

// Mock LiFi tools endpoint
app.get('/v1/tools', (req, res) => {
  res.json([
    { name: 'velodrome', logoURI: 'https://example.com/velodrome.png' },
    { name: 'balancer', logoURI: 'https://example.com/balancer.png' }
  ]);
});

app.listen(5000, '0.0.0.0', () => console.log('LiFi mock server running on port 5000'));
"""
        
        container = self._client.containers.run(
            "node:16-alpine",
            command=[
                "sh", "-c", 
                f"""
                npm install express &&
                echo '{lifi_mock_script}' > /tmp/server.js &&
                node /tmp/server.js
                """
            ],
            detach=True,
            ports=ports,
            extra_hosts={"host.docker.internal": "host-gateway"},
        )
        return container

    def wait(self, max_attempts: int = 15, sleep_rate: float = 1.0) -> bool:
        """
        Wait until the image is running.
        :param max_attempts: max number of attempts.
        :param sleep_rate: the amount of time to sleep between different requests.
        :return: True if the wait was successful, False otherwise.
        """
        for i in range(max_attempts):
            try:
                response = requests.get(f"{self.addr}:{self.port}/v1/tools", timeout=5)
                enforce(response.status_code == 200, "")
                return True
            except Exception as e:  # pylint: disable=broad-except
                logging.error("Exception: %s: %s", type(e).__name__, str(e))
                logging.info(
                    "Attempt %s failed. Retrying in %s seconds...", i, sleep_rate
                )
                time.sleep(sleep_rate)
        return False


DEFAULT_TENDERLY_ADDR = "http://127.0.0.1"
DEFAULT_TENDERLY_PORT = 6000


class MockTenderlyDockerImage(DockerImage):
    """Spawn a dedicated Tenderly API mock server."""

    def __init__(
        self,
        client: docker.DockerClient,
        addr: str = DEFAULT_TENDERLY_ADDR,
        port: int = DEFAULT_TENDERLY_PORT,
    ):
        """Initialize."""
        super().__init__(client)
        self.addr = addr
        self.port = port

    def create_many(self, nb_containers: int) -> List[Container]:
        """Instantiate the image in many containers, parametrized."""
        raise NotImplementedError()

    @property
    def image(self) -> str:
        """Get the image."""
        return "node:16-alpine"

    def create(self) -> Container:
        """Create the container."""
        ports = {"6000/tcp": ("0.0.0.0", self.port)}  # nosec
        
        # Create a mock Tenderly server
        tenderly_mock_script = """
const express = require('express');
const app = express();
app.use(express.json());

// Mock Tenderly simulation endpoint
app.post('/api/v1/account/:account/project/:project/simulate-bundle', (req, res) => {
  res.json({
    simulation_results: [{
      simulation: {
        id: 'sim-123',
        status: true,
        gas_used: 150000,
        block_number: 12345678,
        transaction: {
          hash: '0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890',
          status: 1,
          gas_used: 150000
        }
      }
    }]
  });
});

app.listen(6000, '0.0.0.0', () => console.log('Tenderly mock server running on port 6000'));
"""
        
        container = self._client.containers.run(
            "node:16-alpine",
            command=[
                "sh", "-c", 
                f"""
                npm install express &&
                echo '{tenderly_mock_script}' > /tmp/server.js &&
                node /tmp/server.js
                """
            ],
            detach=True,
            ports=ports,
            extra_hosts={"host.docker.internal": "host-gateway"},
        )
        return container

    def wait(self, max_attempts: int = 15, sleep_rate: float = 1.0) -> bool:
        """
        Wait until the image is running.
        :param max_attempts: max number of attempts.
        :param sleep_rate: the amount of time to sleep between different requests.
        :return: True if the wait was successful, False otherwise.
        """
        for i in range(max_attempts):
            try:
                # Test with a simple GET request to the root
                response = requests.get(f"{self.addr}:{self.port}", timeout=5)
                # Tenderly mock might return 404 for root, but server is running
                if response.status_code in [200, 404]:
                    return True
            except Exception as e:  # pylint: disable=broad-except
                logging.error("Exception: %s: %s", type(e).__name__, str(e))
                logging.info(
                    "Attempt %s failed. Retrying in %s seconds...", i, sleep_rate
                )
                time.sleep(sleep_rate)
        return False

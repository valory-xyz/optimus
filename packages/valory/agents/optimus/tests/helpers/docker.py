"""Docker images for E2E tests."""

import logging
import time
from pathlib import Path
from typing import List

import docker
import requests
from aea.exceptions import enforce
from aea_test_autonomy.docker.base import DockerImage
from docker.models.containers import Container

from packages.valory.agents.optimus.tests.helpers.constants import (
    HARDHAT_PORT,
    HARDHAT_ADDRESS,
    USDC_ADDRESS,
)

# JSON Server constants
DEFAULT_JSON_SERVER_ADDR = "http://127.0.0.1"
DEFAULT_JSON_SERVER_PORT = 3000
DEFAULT_JSON_DATA_DIR = (
    Path(__file__).parent / "data" / "json_server" / "data.json"
)


class OptimismHardhatImage(DockerImage):
    """Hardhat with Optimism setup and deployed contracts."""
    
    def __init__(
        self,
        client: docker.DockerClient,
        addr: str = HARDHAT_ADDRESS,
        port: int = HARDHAT_PORT,
    ):
        super().__init__(client)
        self.addr = addr
        self.port = port
    
    def create_many(self, nb_containers: int) -> List[Container]:
        raise NotImplementedError()
    
    @property
    def image(self) -> str:
        return "valory/hardhat-optimism:latest"
    
    def create(self) -> Container:
        """Create and start Hardhat container."""
        ports = {f"{self.port}/tcp": ("0.0.0.0", self.port)}
        
        container = self._client.containers.run(
            self.image,
            detach=True,
            ports=ports,
            extra_hosts={"host.docker.internal": "host-gateway"},
        )
        return container
    
    def wait(self, max_attempts: int = 30, sleep_rate: float = 1.0) -> bool:
        """Wait for Hardhat to be ready and contracts deployed."""
        for i in range(max_attempts):
            try:
                # Check if node is responsive
                response = requests.post(
                    f"{self.addr}:{self.port}",
                    json={
                        "jsonrpc": "2.0",
                        "method": "eth_blockNumber",
                        "params": [],
                        "id": 1
                    },
                    timeout=5
                )
                enforce(response.status_code == 200, "Hardhat not ready")
                
                # Check if USDC is deployed
                response = requests.post(
                    f"{self.addr}:{self.port}",
                    json={
                        "jsonrpc": "2.0",
                        "method": "eth_getCode",
                        "params": [USDC_ADDRESS, "latest"],
                        "id": 2
                    },
                    timeout=5
                )
                code = response.json().get("result", "0x")
                enforce(code != "0x", "USDC not deployed")
                
                logging.info("Hardhat ready with contracts deployed")
                return True
                
            except Exception as e:
                logging.info(f"Attempt {i+1}/{max_attempts} failed: {e}")
                time.sleep(sleep_rate)
        
        return False


class MockAPIServerImage(DockerImage):
    """JSON server for mocking APIs (Balancer, CoinGecko, Safe API, etc.)."""
    
    def __init__(
        self,
        client: docker.DockerClient,
        addr: str = DEFAULT_JSON_SERVER_ADDR,
        port: int = DEFAULT_JSON_SERVER_PORT,
        json_data: Path = DEFAULT_JSON_DATA_DIR,
    ):
        """Initialize the JSON server image."""
        super().__init__(client)
        self.addr = addr
        self.port = port
        self.json_data = json_data
    
    def create_many(self, nb_containers: int) -> List[Container]:
        raise NotImplementedError()
    
    @property
    def image(self) -> str:
        return "ajoelpod/mock-json-server:latest"
    
    def create(self) -> Container:
        """Create the JSON server container."""
        data = "/usr/src/app/data.json"
        volumes = {
            str(self.json_data): {
                "bind": data,
                "mode": "rw",
            },
        }
        ports = {"8000/tcp": ("0.0.0.0", self.port)}
        container = self._client.containers.run(
            self.image,
            detach=True,
            ports=ports,
            volumes=volumes,
            extra_hosts={"host.docker.internal": "host-gateway"},
        )
        return container
    
    def wait(self, max_attempts: int = 30, sleep_rate: float = 1.0) -> bool:
        """Wait until the JSON server is running."""
        for i in range(max_attempts):
            try:
                response = requests.get(f"{self.addr}:{self.port}")
                enforce(response.status_code == 200, "JSON server not ready")
                logging.info("JSON server ready")
                return True
            except Exception as e:
                logging.info(f"Attempt {i+1}/{max_attempts} failed: {e}")
                time.sleep(sleep_rate)
        return False

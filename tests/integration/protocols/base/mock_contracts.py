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

"""Factory for creating mock contract instances."""

from typing import Any, Dict, List
from unittest.mock import MagicMock

from aea_ledger_ethereum import EthereumApi


class MockContractFactory:
    """Factory for creating mock contract instances."""

    @staticmethod
    def create_ledger_api_mock() -> MagicMock:
        """Create mock ledger API."""
        mock_ledger_api = MagicMock(spec=EthereumApi)
        mock_ledger_api.api = MagicMock()
        mock_ledger_api.api.to_checksum_address = lambda addr: addr.lower()
        mock_ledger_api.identifier = "ethereum"
        
        # Add get_instance method that returns a mock contract instance
        def mock_get_instance(contract_address=None):
            return MockContractFactory.create_contract_instance_mock()
        
        mock_ledger_api.get_instance = mock_get_instance
        return mock_ledger_api

    @staticmethod
    def create_contract_instance_mock() -> MagicMock:
        """Create a generic mock contract instance."""
        mock_instance = MagicMock()
        mock_instance.functions = MagicMock()
        return mock_instance

    @staticmethod
    def create_balancer_vault_mock() -> MagicMock:
        """Create mock Balancer Vault contract."""
        mock_instance = MockContractFactory.create_contract_instance_mock()
        
        # Mock common methods
        mock_instance.functions.getPoolTokens.return_value.call.return_value = (
            ["0xTokenA", "0xTokenB"],  # tokens
            [1000000000000000000, 2000000000000000000],  # balances
            12345  # lastChangeBlock
        )
        
        mock_instance.functions.getPool.return_value.call.return_value = (
            "0xPoolAddress",  # pool
            "Weighted"  # specialization
        )
        
        return mock_instance

    @staticmethod
    def create_balancer_weighted_pool_mock() -> MagicMock:
        """Create mock Balancer Weighted Pool contract."""
        mock_instance = MockContractFactory.create_contract_instance_mock()
        
        # Mock common methods
        mock_instance.functions.balanceOf.return_value.call.return_value = 100000000000000000000
        mock_instance.functions.totalSupply.return_value.call.return_value = 1000000000000000000000
        mock_instance.functions.getPoolId.return_value.call.return_value = bytes.fromhex(
            "1234567890123456789012345678901234567890123456789012345678901234"
        )
        mock_instance.functions.getVault.return_value.call.return_value = "0xVaultAddress"
        mock_instance.functions.name.return_value.call.return_value = "Test Pool"
        
        return mock_instance

    @staticmethod
    def create_uniswap_v3_pool_mock() -> MagicMock:
        """Create mock Uniswap V3 Pool contract."""
        mock_instance = MockContractFactory.create_contract_instance_mock()
        
        # Mock common methods
        mock_instance.functions.token0.return_value.call.return_value = "0xTokenA"
        mock_instance.functions.token1.return_value.call.return_value = "0xTokenB"
        mock_instance.functions.fee.return_value.call.return_value = 3000
        mock_instance.functions.tickSpacing.return_value.call.return_value = 60
        
        # Mock slot0 response
        mock_instance.functions.slot0.return_value.call.return_value = (
            79228162514264337593543950336,  # sqrtPriceX96
            -276310,                        # tick
            0,                              # observationIndex
            1,                              # observationCardinality
            1,                              # observationCardinalityNext
            0,                              # feeProtocol
            True                            # unlocked
        )
        
        return mock_instance

    @staticmethod
    def create_uniswap_v3_position_manager_mock() -> MagicMock:
        """Create mock Uniswap V3 Position Manager contract."""
        mock_instance = MockContractFactory.create_contract_instance_mock()
        
        # Mock common methods
        mock_instance.functions.positions.return_value.call.return_value = (
            1000000000000000000,  # liquidity
            "0xTokenA",           # token0
            "0xTokenB",           # token1
            3000,                 # fee
            -276320,              # tickLower
            -276300,              # tickUpper
            0,                    # tokensOwed0
            0,                    # tokensOwed1
        )
        
        mock_instance.functions.balanceOf.return_value.call.return_value = 1
        mock_instance.functions.tokenOfOwnerByIndex.return_value.call.return_value = 12345
        
        return mock_instance

    @staticmethod
    def create_velodrome_pool_mock() -> MagicMock:
        """Create mock Velodrome Pool contract."""
        mock_instance = MockContractFactory.create_contract_instance_mock()
        
        # Mock common methods
        mock_instance.functions.balanceOf.return_value.call.return_value = 100000000000000000000
        mock_instance.functions.totalSupply.return_value.call.return_value = 1000000000000000000000
        mock_instance.functions.reserve0.return_value.call.return_value = 1000000000000000000000
        mock_instance.functions.reserve1.return_value.call.return_value = 2000000000000000000000
        mock_instance.functions.token0.return_value.call.return_value = "0xTokenA"
        mock_instance.functions.token1.return_value.call.return_value = "0xTokenB"
        mock_instance.functions.stable.return_value.call.return_value = False
        
        return mock_instance

    @staticmethod
    def create_velodrome_gauge_mock() -> MagicMock:
        """Create mock Velodrome Gauge contract."""
        mock_instance = MockContractFactory.create_contract_instance_mock()
        
        # Mock common methods
        mock_instance.functions.balanceOf.return_value.call.return_value = 100000000000000000000
        mock_instance.functions.totalSupply.return_value.call.return_value = 1000000000000000000000
        mock_instance.functions.earned.return_value.call.return_value = 50000000000000000000
        mock_instance.functions.rewardRate.return_value.call.return_value = 1000000000000000000
        
        return mock_instance

    @staticmethod
    def create_velodrome_voter_mock() -> MagicMock:
        """Create mock Velodrome Voter contract."""
        mock_instance = MockContractFactory.create_contract_instance_mock()
        
        # Mock common methods
        mock_instance.functions.gauges.return_value.call.return_value = "0xGaugeAddress"
        mock_instance.functions.isGauge.return_value.call.return_value = True
        mock_instance.functions.votes.return_value.call.return_value = 100000000000000000000
        
        return mock_instance

    @staticmethod
    def create_erc20_mock() -> MagicMock:
        """Create mock ERC20 contract."""
        mock_instance = MockContractFactory.create_contract_instance_mock()
        
        # Mock common methods
        mock_instance.functions.balanceOf.return_value.call.return_value = 100000000000000000000
        mock_instance.functions.totalSupply.return_value.call.return_value = 1000000000000000000000
        mock_instance.functions.decimals.return_value.call.return_value = 18
        mock_instance.functions.name.return_value.call.return_value = "Test Token"
        mock_instance.functions.symbol.return_value.call.return_value = "TEST"
        mock_instance.functions.allowance.return_value.call.return_value = 0
        
        return mock_instance

    @staticmethod
    def create_multisend_mock() -> MagicMock:
        """Create mock MultiSend contract."""
        mock_instance = MockContractFactory.create_contract_instance_mock()
        
        # Mock common methods
        mock_instance.functions.multiSend.return_value.encodeABI.return_value = b"multisend_data"
        
        return mock_instance

    @classmethod
    def create_all_contract_mocks(cls) -> Dict[str, MagicMock]:
        """Create all contract mocks."""
        return {
            "ledger_api": cls.create_ledger_api_mock(),
            "balancer_vault": cls.create_balancer_vault_mock(),
            "balancer_weighted_pool": cls.create_balancer_weighted_pool_mock(),
            "uniswap_v3_pool": cls.create_uniswap_v3_pool_mock(),
            "uniswap_v3_position_manager": cls.create_uniswap_v3_position_manager_mock(),
            "velodrome_pool": cls.create_velodrome_pool_mock(),
            "velodrome_gauge": cls.create_velodrome_gauge_mock(),
            "velodrome_voter": cls.create_velodrome_voter_mock(),
            "erc20": cls.create_erc20_mock(),
            "multisend": cls.create_multisend_mock(),
        }

    @staticmethod
    def setup_contract_interact_mock(
        behaviour: Any, contract_mocks: Dict[str, MagicMock]
    ) -> None:
        """Set up contract_interact mock for a behaviour."""
        def mock_contract_interact(*args, **kwargs):
            contract_callable = kwargs.get("contract_callable", "")
            contract_address = kwargs.get("contract_address", "")
            
            # Determine which contract mock to use based on callable or address
            if "balancer" in contract_callable.lower() or "vault" in contract_address.lower():
                if "getPoolTokens" in contract_callable:
                    return {"tokens": ["0xTokenA", "0xTokenB"]}
                elif "balanceOf" in contract_callable:
                    return {"balance": 100000000000000000000}
                elif "totalSupply" in contract_callable:
                    return {"data": 1000000000000000000000}
            elif "uniswap" in contract_callable.lower() or "pool" in contract_address.lower():
                if "slot0" in contract_callable:
                    return {
                        "slot0": {
                            "sqrt_price_x96": 79228162514264337593543950336,
                            "tick": -276310,
                            "unlocked": True,
                        }
                    }
                elif "positions" in contract_callable:
                    return {
                        "liquidity": 1000000000000000000,
                        "token0": "0xTokenA",
                        "token1": "0xTokenB",
                        "fee": 3000,
                        "tickLower": -276320,
                        "tickUpper": -276300,
                        "tokensOwed0": 0,
                        "tokensOwed1": 0,
                    }
            elif "velodrome" in contract_callable.lower() or "gauge" in contract_address.lower():
                if "balanceOf" in contract_callable:
                    return {"balance": 100000000000000000000}
                elif "earned" in contract_callable:
                    return {"earned": 50000000000000000000}
            
            # Default response
            return {"data": "0x123456"}
        
        behaviour.contract_interact = mock_contract_interact

# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2022-2023 Valory AG
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

"""Wrapper for Velodrome LpSugar contract interface."""

from typing import Dict, List, Any, Optional

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi

PUBLIC_ID = PublicId.from_str("valory/velodrome_lp_sugar:0.1.0")


class VelodromeLpSugarContract(Contract):
    """Velodrome LpSugar contract wrapper."""

    contract_id = PUBLIC_ID
    
    @classmethod
    def get_instance(cls, ledger_api: EthereumApi, contract_address: str):
        """
        Get the instance of the contract.
        
        Args:
            ledger_api: Ethereum API
            contract_address: Contract address
            
        Returns:
            Contract instance
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"Getting contract instance with address: {contract_address}")
            # Get the contract instance from the parent class
            instance = super().get_instance(ledger_api, contract_address)
            logger.info(f"Contract instance type: {type(instance)}")
            return instance
        except Exception as e:
            logger.error(f"Error in get_instance: {str(e)}")
            # Re-raise the exception to be caught by the calling method
            raise

    @classmethod
    def all(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        limit: int,
        offset: int,
    ) -> JSONLike:
        """
        Get all pools with pagination.
        
        Args:
            ledger_api: Ethereum API
            contract_address: LpSugar contract address
            limit: Maximum number of pools to return
            offset: Offset for pagination
            
        Returns:
            List of pool data
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Getting contract instance for address: {contract_address}")
        try:
            contract_instance = cls.get_instance(ledger_api, contract_address)
            logger.info(f"Contract instance obtained: {contract_instance}")
            
            logger.info(f"Calling all() function with limit={limit}, offset={offset}")
            result = contract_instance.functions.all(limit, offset).call()
            logger.info(f"Raw result from contract: {result}")
            
            formatted_result = cls._format_pools_data(result)
            logger.info(f"Formatted result: {formatted_result}")
            
            return dict(pools=formatted_result)
        except Exception as e:
            logger.error(f"Error in all() method: {str(e)}")
            # Return an empty result instead of raising an exception
            return dict(pools=[])
    
    @classmethod
    def by_address(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        pool_address: str,
    ) -> JSONLike:
        """
        Get pool data by address.
        
        Args:
            ledger_api: Ethereum API
            contract_address: LpSugar contract address
            pool_address: Address of the pool to query
            
        Returns:
            Pool data
        """
        contract_instance = cls.get_instance(ledger_api, contract_address)
        result = contract_instance.functions.byAddress(pool_address).call()
        return dict(pool=cls._format_pool_data(result))
    
    @classmethod
    def _format_pool_data(cls, pool_data: tuple) -> Dict[str, Any]:
        """
        Format pool data from contract response.
        
        The Lp struct in Sugar contracts contains fields according to the ABI:
        - lp, symbol, decimals, liquidity, type, tick, sqrt_ratio
        - token0, reserve0, staked0, token1, reserve1, staked1
        - gauge, gauge_liquidity, gauge_alive, fee, bribe, factory
        - emissions, emissions_token, pool_fee, unstaked_fee
        - token0_fees, token1_fees, nfpm, alm, root
        
        Args:
            pool_data: Raw pool data from contract
            
        Returns:
            Formatted pool data
        """
        # Map the tuple data to a dictionary with named fields based on the ABI structure
        return {
            "id": pool_data[0],  # lp address
            "symbol": pool_data[1],
            "decimals": pool_data[2],
            "liquidity": pool_data[3],
            "type": pool_data[4],
            "tick": pool_data[5],
            "sqrt_ratio": pool_data[6],
            "token0": pool_data[7],
            "reserve0": pool_data[8],
            "staked0": pool_data[9],
            "token1": pool_data[10],
            "reserve1": pool_data[11],
            "staked1": pool_data[12],
            "gauge": pool_data[13],
            "gauge_liquidity": pool_data[14],
            "gauge_alive": pool_data[15],
            "fee": pool_data[16],
            "bribe": pool_data[17],
            "factory": pool_data[18],
            "emissions": pool_data[19],
            "emissions_token": pool_data[20],
            "pool_fee": pool_data[21],
            "unstaked_fee": pool_data[22],
            "token0_fees": pool_data[23],
            "token1_fees": pool_data[24],
            "nfpm": pool_data[25],
            "alm": pool_data[26],
            "root": pool_data[27],
        }
    
    @classmethod
    def _format_pools_data(cls, pools_data: List[tuple]) -> List[Dict[str, Any]]:
        """
        Format multiple pools data from contract response.
        
        Args:
            pools_data: List of raw pool data from contract
            
        Returns:
            List of formatted pool data
        """
        return [cls._format_pool_data(pool) for pool in pools_data]

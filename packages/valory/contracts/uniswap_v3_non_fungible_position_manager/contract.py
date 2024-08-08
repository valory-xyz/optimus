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

"""This class contains a wrapper for UniswapV3 Pool contract interface."""

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi
PUBLIC_ID = PublicId.from_str("valory/uniswap_v3_non_fungible_position_manager:0.1.0")

class UniswapV3NonfungiblePositionManagerContract(Contract):
    """The Uniswap V3 NonfungiblePositionManager contract."""

    contract_id = PUBLIC_ID
    
    @classmethod
    def add_liquidity(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token0: str,
        token1: str,
        fee: int,
        tick_lower: int,
        tick_upper: int,
        amount0_desired: int,
        amount1_desired: int,
        amount0_min: int,
        amount1_min: int,
        recipient: str,
        deadline: int
    ) -> JSONLike:
        """Prepare mint position transaction"""
        mint_params = (
            token0,
            token1,
            fee,
            tick_lower,
            tick_upper,
            amount0_desired,
            amount1_desired,
            amount0_min,
            amount1_min,
            recipient,
            deadline
        )

        contract_instance = cls.get_instance(ledger_api, contract_address)
        tx_hash = contract_instance.encodeABI(
            "mint",
            args=(mint_params,)
        )

        return dict(tx_hash=tx_hash)
    
    @classmethod
    def burn_position(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
    ) -> JSONLike:
        """Prepare burn position transaction"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        tx_hash = contract_instance.encodeABI(
            "burn",
            args=(token_id)
        )
        return dict(tx_hash=tx_hash)
    
    @classmethod
    def collect_tokens(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
        recipient: str,
        amount0_max: int,
        amount1_max: int
    ) -> JSONLike:
        """Prepare collect transaction"""
        collect_params = (
            token_id,
            recipient,
            amount0_max,
            amount1_max
        )

        contract_instance = cls.get_instance(ledger_api, contract_address)
        tx_hash = contract_instance.encodeABI(
            "collect",
            args=(collect_params)
        )

        return dict(tx_hash=tx_hash)
    
    @classmethod
    def get_pool_tokens(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """get the pool tokens"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        token0 = contract_instance.functions.token0().call()
        token1 = contract_instance.functions.token1().call()
        return dict(tokens=[token0,token1])

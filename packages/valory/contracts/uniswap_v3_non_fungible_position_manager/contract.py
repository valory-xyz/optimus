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
    def mint(
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
        deadline: int,
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
            deadline,
        )

        contract_instance = cls.get_instance(ledger_api, contract_address)
        tx_hash = contract_instance.encodeABI("mint", args=(mint_params,))

        return dict(tx_hash=tx_hash)

    @classmethod
    def decrease_liquidity(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
        liquidity: int,
        amount0_min: int,
        amount1_min: int,
        deadline: int,
    ) -> JSONLike:
        """Prepare decrease liquidity transaction"""
        decrease_liquidity_params = (
            token_id,
            liquidity,
            amount0_min,
            amount1_min,
            deadline,
        )

        contract_instance = cls.get_instance(ledger_api, contract_address)
        tx_hash = contract_instance.encodeABI(
            "decreaseLiquidity", args=(decrease_liquidity_params,)
        )

        return dict(tx_hash=tx_hash)

    @classmethod
    def burn_token(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
    ) -> JSONLike:
        """Prepare burn position transaction"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        tx_hash = contract_instance.encodeABI("burn", args=(token_id,))
        return dict(tx_hash=tx_hash)

    @classmethod
    def collect_tokens(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
        recipient: str,
        amount0_max: int,
        amount1_max: int,
    ) -> JSONLike:
        """Prepare collect transaction"""
        collect_params = (token_id, recipient, amount0_max, amount1_max)

        contract_instance = cls.get_instance(ledger_api, contract_address)
        tx_hash = contract_instance.encodeABI("collect", args=(collect_params,))

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
        return dict(tokens=[token0, token1])
    
    @classmethod
    def balanceOf(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        owner: str,
    ) -> JSONLike:
        """Get the number of NFT positions owned by an address"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        balance = contract_instance.functions.balanceOf(owner).call()
        return dict(balance=balance)
    
    @classmethod
    def ownerOf(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
    ) -> JSONLike:
        """Get the owner of a specific NFT position"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        owner = contract_instance.functions.ownerOf(token_id).call()
        return dict(owner=owner)
    
    @classmethod
    def get_position(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
    ) -> JSONLike:
        """get the position info"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        position = contract_instance.functions.positions(token_id).call()
        if not position:
            return dict(data={})
        return dict(data={
            "nonce": position[0],
            "operator": position[1],
            "token0": position[2],
            "token1": position[3],
            "fee": position[4],
            "tickLower": position[5],
            "tickUpper": position[6],
            "liquidity": position[7],
            "feeGrowthInside0LastX128": position[8],
            "feeGrowthInside1LastX128": position[9],
            "tokensOwed0": position[10],
            "tokensOwed1": position[11],
        })
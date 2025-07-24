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

"""Wrapper for Velodrome CL Pool Manager contract interface."""

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi

PUBLIC_ID = PublicId.from_str("valory/velodrome_non_fungible_position_manager:0.1.0")


class VelodromeNonFungiblePositionManagerContract(Contract):
    """Velodrome CL Pool Manager contract wrapper."""

    contract_id = PUBLIC_ID

    @classmethod
    def _encode_call(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        method_name: str,
        args: tuple,
    ) -> JSONLike:
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI(method_name, args=args)
        return dict(tx_hash=data)

    # ------------------- Methods -------------------

    @classmethod
    def mint(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token0: str,
        token1: str,
        tick_spacing: int,
        tick_lower: int,
        tick_upper: int,
        amount0_desired: int,
        amount1_desired: int,
        amount0_min: int,
        amount1_min: int,
        recipient: str,
        deadline: int,
        sqrt_price_x96: int,
    ) -> JSONLike:
        """Prepare encoded tx for mint."""
        params = (
            token0,
            token1,
            tick_spacing,
            tick_lower,
            tick_upper,
            amount0_desired,
            amount1_desired,
            amount0_min,
            amount1_min,
            recipient,
            deadline,
            sqrt_price_x96,
        )
        return cls._encode_call(
            ledger_api,
            contract_address,
            "mint",
            (params,),
        )

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
        """Prepare encoded tx for decreaseLiquidity."""
        params = (
            token_id,
            liquidity,
            amount0_min,
            amount1_min,
            deadline,
        )
        return cls._encode_call(
            ledger_api,
            contract_address,
            "decreaseLiquidity",
            (params,),
        )

    @classmethod
    def burn(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
    ) -> JSONLike:
        """Prepare encoded tx for burn."""
        return cls._encode_call(
            ledger_api,
            contract_address,
            "burn",
            (token_id,),
        )

    @classmethod
    def collect(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
        recipient: str,
        amount0_max: int,
        amount1_max: int,
    ) -> JSONLike:
        """Prepare encoded tx for collect."""
        params = (
            token_id,
            recipient,
            amount0_max,
            amount1_max,
        )
        return cls._encode_call(
            ledger_api,
            contract_address,
            "collect",
            (params,),
        )

    @classmethod
    def get_pool_tokens(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
    ) -> JSONLike:
        """Get tokens information from a position."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        result = contract_instance.functions.positions(token_id).call()
        return dict(
            token0=result[2],
            token1=result[3],
            tick_spacing=result[4],
            tick_lower=result[5],
            tick_upper=result[6],
            liquidity=result[7],
        )

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
            "tickSpacing": position[4],
            "tickLower": position[5],
            "tickUpper": position[6],
            "liquidity": position[7],
            "feeGrowthInside0LastX128": position[8],
            "feeGrowthInside1LastX128": position[9],
            "tokensOwed0": position[10],
            "tokensOwed1": position[11],
        })
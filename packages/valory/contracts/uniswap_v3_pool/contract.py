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

"""This class contains a wrapper for Pool contract interface."""

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi


PUBLIC_ID = PublicId.from_str("valory/uniswap_v3_pool:0.1.0")


class UniswapV3PoolContract(Contract):
    """The Uniswap V3 Pool contract."""

    contract_id = PUBLIC_ID

    @classmethod
    def get_pool_tokens(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """get the pool tokens."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        token0 = contract_instance.functions.token0().call()
        token1 = contract_instance.functions.token1().call()
        return dict(tokens=[token0, token1])

    @classmethod
    def get_pool_fee(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """get the fee."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        fee = contract_instance.functions.fee().call()
        return dict(data=fee)

    @classmethod
    def get_tick_spacing(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """get the tick spacing."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        tick_spacing = contract_instance.functions.tickSpacing().call()
        return dict(data=tick_spacing)

    @classmethod
    def get_reserves_and_balances(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        your_address: str,
    ) -> JSONLike:
        """Get reserves, total supply, and your balance."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        reserves = contract_instance.functions.getReserves().call()
        total_supply = contract_instance.functions.totalSupply().call()
        your_lp_balance = contract_instance.functions.balanceOf(your_address).call()

        pool_share = your_lp_balance / total_supply
        current_token0_qty = reserves[0] * pool_share
        current_token1_qty = reserves[1] * pool_share

        return dict(
            data={
                "reserve0": reserves[0],
                "reserve1": reserves[1],
                "total_supply": total_supply,
                "lp_balance": your_lp_balance,
                "current_token0_qty": current_token0_qty,
                "current_token1_qty": current_token1_qty,
            }
        )

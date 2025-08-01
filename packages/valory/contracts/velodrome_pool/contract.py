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

"""This class contains a wrapper for Velodrome Pool contract interface."""

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi, LedgerApi


PUBLIC_ID = PublicId.from_str("valory/velodrome_pool:0.1.0")


class VelodromePoolContract(Contract):
    """The Velodrome Pool contract."""

    contract_id = PUBLIC_ID

    @classmethod
    def get_balance(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        account: str,
    ) -> JSONLike:
        """get the balance of the given account."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.functions.balanceOf(account).call()
        return dict(balance=data)
    
    @classmethod
    def build_approval_tx(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        spender: str,
        amount: int,
    ) -> JSONLike:
        """Build an ERC20 approval."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        checksumed_spender = ledger_api.api.to_checksum_address(spender)
        data = contract_instance.encodeABI("approve", args=(checksumed_spender, amount))
        return {"tx_hash": bytes.fromhex(data[2:])}
    
    @classmethod
    def get_reserves(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get the current reserves of token0 and token1."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        reserve0 = contract_instance.functions.reserve0().call()
        reserve1 = contract_instance.functions.reserve1().call()
        return dict(data=[reserve0, reserve1])

    @classmethod
    def gauge(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get the gauge address for this pool."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        gauge_address = contract_instance.functions.gauge().call()
        return dict(data=gauge_address)

    @classmethod
    def get_total_supply(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get the total supply of LP tokens."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        total_supply = contract_instance.functions.totalSupply().call()
        return dict(data=total_supply)

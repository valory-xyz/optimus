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

"""This class contains a wrapper for Sturdy's YearnV3Vault contract interface."""
import logging

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi
from eth_abi import encode


PUBLIC_ID = PublicId.from_str("valory/sturdy_yearn_v3_vault:0.1.0")
_logger = logging.getLogger(
    f"aea.packages.{PUBLIC_ID.author}.contracts.{PUBLIC_ID.name}.contract"
)


class YearnV3VaultContract(Contract):
    """The Vault contract."""

    contract_id = PUBLIC_ID

    @classmethod
    def deposit(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        assets: int,
        receiver: str,
    ) -> JSONLike:
        """Prepare a deposit transaction."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI("deposit", args=(assets, receiver))
        return {"tx_hash": bytes.fromhex(data[2:])}

    @classmethod
    def withdraw(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        assets: int,
        receiver: str,
        owner: str,
    ) -> JSONLike:
        """Prepare a withdraw transaction."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI("withdraw", args=(assets, receiver, owner))
        return {"tx_hash": bytes.fromhex(data[2:])}

    @classmethod
    def max_withdraw(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        owner: str,
    ) -> JSONLike:
        """Get the maximum amount that can be withdrawn by the owner."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        max_withdraw_amount = contract_instance.functions.maxWithdraw(owner).call()
        return {"amount": max_withdraw_amount}

    @classmethod
    def balance_of(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        owner: str,
    ) -> JSONLike:
        """Get the balance of a user in the vault."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        balance = contract_instance.functions.balanceOf(owner).call()
        return {"amount": balance}

    @classmethod
    def name(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get the name of the aggregator"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        name = contract_instance.functions.name().call()
        return {"name": name}
    
    @classmethod
    def total_supply(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get the total supply of the vault."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        total_supply = contract_instance.functions.totalSupply().call()
        return {"total_supply": total_supply}

    @classmethod
    def total_assets(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get the total assets of the vault."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        total_assets = contract_instance.functions.totalAssets().call()
        return {"total_assets": total_assets}

    @classmethod
    def decimals(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get the number of decimals used by the vault."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        decimals = contract_instance.functions.decimals().call()
        return {"decimals": decimals}
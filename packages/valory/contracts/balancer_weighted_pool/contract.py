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

"""This class contains a wrapper for Weighted Pool contract interface."""

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi, LedgerApi


PUBLIC_ID = PublicId.from_str("valory/balancer_weighted_pool:0.1.0")


class WeightedPoolContract(Contract):
    """The Weighted Stable Pool contract."""

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
    def get_pool_id(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """get pool id"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.functions.getPoolId().call()
        return dict(pool_id="0x" + data.hex())

    @classmethod
    def get_vault_address(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """get vault address"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.functions.getVault().call()
        return dict(vault=data)

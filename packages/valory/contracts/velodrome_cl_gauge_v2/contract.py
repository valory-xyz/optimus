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

"""This module contains the class to interact with the Velodrome CL Gauge V2 contract."""

from typing import Any, Dict

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea.crypto.base import LedgerApi


class VelodromeCLGaugeV2Contract(Contract):
    """The Velodrome CL Gauge V2 contract."""

    contract_id = PublicId.from_str("valory/velodrome_cl_gauge_v2:0.1.0")

    @classmethod
    def deposit(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        token_id: int,
        **kwargs: Any,
    ) -> JSONLike:
        """
        Deposit NFT position to the gauge.

        :param ledger_api: the ledger API object
        :param contract_address: the contract address
        :param token_id: the NFT token ID to deposit
        :param kwargs: additional keyword arguments
        :return: the raw transaction
        """
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI(
            fn_name="deposit",
            args=[token_id],
        )
        return {"data": data}

    @classmethod
    def withdraw(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        token_id: int,
        **kwargs: Any,
    ) -> JSONLike:
        """
        Withdraw NFT position from the gauge.

        :param ledger_api: the ledger API object
        :param contract_address: the contract address
        :param token_id: the NFT token ID to withdraw
        :param kwargs: additional keyword arguments
        :return: the raw transaction
        """
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI(
            fn_name="withdraw",
            args=[token_id],
        )
        return {"data": data}

    @classmethod
    def get_reward(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        account: str,
        token: str,
        **kwargs: Any,
    ) -> JSONLike:
        """
        Get reward for an account and token.

        :param ledger_api: the ledger API object
        :param contract_address: the contract address
        :param account: the account address
        :param token: the token address
        :param kwargs: additional keyword arguments
        :return: the reward amount
        """
        contract_instance = cls.get_instance(ledger_api, contract_address)
        result = contract_instance.functions.getReward(account, token).call()
        return {"data": result}

    @classmethod
    def balance_of(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        account: str,
        **kwargs: Any,
    ) -> JSONLike:
        """
        Get the balance of staked NFTs for an account.

        :param ledger_api: the ledger API object
        :param contract_address: the contract address
        :param account: the account address
        :param kwargs: additional keyword arguments
        :return: the balance
        """
        contract_instance = cls.get_instance(ledger_api, contract_address)
        result = contract_instance.functions.balanceOf(account).call()
        return {"data": result}

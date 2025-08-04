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

"""This class contains a wrapper for Velodrome CL Gauge contract interface."""

import logging
from typing import Any

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi, LedgerApi


PUBLIC_ID = PublicId.from_str("valory/velodrome_cl_gauge:0.1.0")

_logger = logging.getLogger(__name__)


class VelodromeCLGaugeContract(Contract):
    """The Velodrome CL Gauge contract."""

    contract_id = PUBLIC_ID

    @classmethod
    def _encode_call(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        method_name: str,
        args: tuple,
    ) -> JSONLike:
        """Helper to build transaction bytes."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI(method_name, args=args)
        _logger.debug(f"Encoded {method_name} call with args: {args}")
        return dict(tx_hash=data)

    @classmethod
    def deposit(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        amount: int,
        recipient: str,
    ) -> JSONLike:
        """Prepare encoded tx for depositing tokens to CL gauge with recipient."""
        _logger.debug(f"Preparing CL deposit transaction for amount: {amount}, recipient: {recipient}")
        
        if amount <= 0:
            error_msg = "Amount must be greater than 0"
            _logger.error(error_msg)
            return dict(error=error_msg)
        
        checksumed_recipient = ledger_api.api.to_checksum_address(recipient)
        
        return cls._encode_call(
            ledger_api,
            contract_address,
            "deposit",
            (amount, checksumed_recipient),
        )

    @classmethod
    def withdraw(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
    ) -> JSONLike:
        """Prepare encoded tx for withdrawing tokens from CL gauge using token ID."""
        _logger.debug(f"Preparing CL withdraw transaction for token ID: {token_id}")
        
        if token_id < 0:
            error_msg = "Token ID must be non-negative"
            _logger.error(error_msg)
            return dict(error=error_msg)
        
        return cls._encode_call(
            ledger_api,
            contract_address,
            "withdraw",
            (token_id,),
        )

    @classmethod
    def get_reward(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        account: str,
    ) -> JSONLike:
        """Prepare encoded tx for claiming rewards from CL gauge."""
        _logger.debug(f"Preparing CL getReward transaction for account: {account}")
        
        checksumed_account = ledger_api.api.to_checksum_address(account)
        
        return cls._encode_call(
            ledger_api,
            contract_address,
            "getReward",
            (checksumed_account,),
        )

    @classmethod
    def earned(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        account: str,
    ) -> JSONLike:
        """Get the amount of rewards earned by an account in CL gauge."""
        _logger.debug(f"Getting CL earned rewards for account: {account}")
        
        checksumed_account = ledger_api.api.to_checksum_address(account)
        contract_instance = cls.get_instance(ledger_api, contract_address)
        earned_amount = contract_instance.functions.earned(checksumed_account).call()
        _logger.debug(f"CL earned amount for {account}: {earned_amount}")
        return dict(earned=earned_amount)

    @classmethod
    def balance_of(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        account: str,
    ) -> JSONLike:
        """Get the staked balance of an account in the CL gauge."""
        _logger.debug(f"Getting CL staked balance for account: {account}")
        
        checksumed_account = ledger_api.api.to_checksum_address(account)
        contract_instance = cls.get_instance(ledger_api, contract_address)
        balance = contract_instance.functions.balanceOf(checksumed_account).call()
        _logger.debug(f"CL staked balance for {account}: {balance}")
        return dict(balance=balance)

    @classmethod
    def total_supply(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get the total supply of staked tokens in CL gauge."""
        _logger.debug("Getting CL total supply of staked tokens")
        
        contract_instance = cls.get_instance(ledger_api, contract_address)
        total_supply = contract_instance.functions.totalSupply().call()
        _logger.debug(f"CL total supply: {total_supply}")
        return dict(total_supply=total_supply)

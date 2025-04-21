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

"""Wrapper for Velodrome Concentrated Liquidity Pool contract interface."""

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi

PUBLIC_ID = PublicId.from_str("valory/velodrome_cl_pool:0.1.0")


class VelodromeCLPoolContract(Contract):
    """Velodrome CL Pool contract wrapper."""

    contract_id = PUBLIC_ID

    # Helper to build tx bytes
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
        return {"tx_bytes": bytes.fromhex(data[2:])}

    # ----------------------- Methods -----------------------
    @classmethod
    def mint(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        recipient: str,
        tick_lower: int,
        tick_upper: int,
        amount: int,
        data: bytes,
    ) -> JSONLike:
        """Prepare encoded tx for mint."""
        return cls._encode_call(
            ledger_api,
            contract_address,
            "mint",
            (
                recipient,
                tick_lower,
                tick_upper,
                amount,
                data,
            ),
        )

    @classmethod
    def burn(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        tick_lower: int,
        tick_upper: int,
        amount: int,
    ) -> JSONLike:
        """Prepare encoded tx for burn."""
        return cls._encode_call(
            ledger_api,
            contract_address,
            "burn",
            (
                tick_lower,
                tick_upper,
                amount,
            ),
        )

    @classmethod
    def collect(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        recipient: str,
        tick_lower: int,
        tick_upper: int,
        amount0_requested: int,
        amount1_requested: int,
    ) -> JSONLike:
        """Prepare encoded tx for collect."""
        return cls._encode_call(
            ledger_api,
            contract_address,
            "collect",
            (
                recipient,
                tick_lower,
                tick_upper,
                amount0_requested,
                amount1_requested,
            ),
        )

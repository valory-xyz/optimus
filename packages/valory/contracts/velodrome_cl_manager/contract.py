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

PUBLIC_ID = PublicId.from_str("valory/velodrome_cl_manager:0.1.0")


class VelodromeCLPoolManagerContract(Contract):
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
        return {"tx_bytes": bytes.fromhex(data[2:])}

    # ------------------- Methods -------------------
    @classmethod
    def create_position(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        pool: str,
        tick_lower: int,
        tick_upper: int,
        amount0_desired: int,
        amount1_desired: int,
        amount0_min: int,
        amount1_min: int,
    ) -> JSONLike:
        """Prepare encoded tx for createPosition."""
        return cls._encode_call(
            ledger_api,
            contract_address,
            "createPosition",
            (
                pool,
                tick_lower,
                tick_upper,
                amount0_desired,
                amount1_desired,
                amount0_min,
                amount1_min,
            ),
        )

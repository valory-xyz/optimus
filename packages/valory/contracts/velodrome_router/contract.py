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

"""This module contains a wrapper for Velodrome Router contract interface."""

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi

PUBLIC_ID = PublicId.from_str("valory/velodrome_router:0.1.0")


class VelodromeRouterContract(Contract):
    """Velodrome Router contract wrapper."""

    contract_id = PUBLIC_ID

    # ------------------------------- helper ------------------------------
    @classmethod
    def _encode_call(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        method_name: str,
        args: tuple,
    ) -> JSONLike:
        """Return the ABI encoded tx bytes for a contract method call."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI(method_name, args=args)
        return dict(tx_hash=data)

    # ---------------------------- main methods --------------------------
    @classmethod
    def add_liquidity(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_a: str,
        token_b: str,
        stable: bool,
        amount_a_desired: int,
        amount_b_desired: int,
        amount_a_min: int,
        amount_b_min: int,
        to: str,
        deadline: int,
    ) -> JSONLike:
        """Prepare encoded data for addLiquidity."""
        return cls._encode_call(
            ledger_api,
            contract_address,
            "addLiquidity",
            (
                token_a,
                token_b,
                stable,
                amount_a_desired,
                amount_b_desired,
                amount_a_min,
                amount_b_min,
                to,
                deadline,
            ),
        )

    @classmethod
    def remove_liquidity(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_a: str,
        token_b: str,
        stable: bool,
        liquidity: int,
        amount_a_min: int,
        amount_b_min: int,
        to: str,
        deadline: int,
    ) -> JSONLike:
        """Prepare encoded data for removeLiquidity."""
        return cls._encode_call(
            ledger_api,
            contract_address,
            "removeLiquidity",
            (
                token_a,
                token_b,
                stable,
                liquidity,
                amount_a_min,
                amount_b_min,
                to,
                deadline,
            ),
        )

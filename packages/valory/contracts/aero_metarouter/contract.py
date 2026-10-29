# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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

"""This module contains a wrapper for the Aero Metarouter contract interface."""

from enum import IntEnum
from typing import List

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi
from eth_abi import encode

PUBLIC_ID = PublicId.from_str("valory/aero_metarouter:0.1.0")

# Metarouter command IDs (metadex-public V3/src/metarouter/libraries/Commands.sol)
ADD_LIQUIDITY_COMMAND = 0x22
REMOVE_LIQUIDITY_COMMAND = 0x23

# IMetarouter.BalanceSpend: (SpendMode mode, uint256 value)
BALANCE_SPEND_TYPE = "(uint8,uint256)"
# IMetarouter.AddLiquidityParams
ADD_LIQUIDITY_PARAMS_TYPE = (
    f"(address,address,address,bool,{BALANCE_SPEND_TYPE},{BALANCE_SPEND_TYPE},"
    "bool,bool,uint256,uint256,address,uint256)"
)
# IMetarouter.RemoveLiquidityParams
REMOVE_LIQUIDITY_PARAMS_TYPE = (
    f"(address,{BALANCE_SPEND_TYPE},bool,uint256,uint256,address)"
)


class SpendMode(IntEnum):
    """IMetarouter.SpendMode."""

    AMOUNT = 0
    PIPS = 1


class AeroMetarouterContract(Contract):
    """The Aero Metarouter contract."""

    contract_id = PUBLIC_ID

    @classmethod
    def get_execute_tx(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        commands: bytes,
        inputs: List[bytes],
        deadline: int,
    ) -> JSONLike:
        """Prepare encoded tx for running a batch of Metarouter commands."""
        if len(commands) != len(inputs):
            return dict(error="Each Metarouter command needs exactly one input")

        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encode_abi(
            "execute", args=(commands, list(inputs), deadline)
        )
        return dict(tx_hash=bytes.fromhex(data[2:]))

    @classmethod
    def get_add_liquidity_tx(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        factory: str,
        token_a: str,
        token_b: str,
        amount_a: int,
        amount_b: int,
        amount_a_min: int,
        amount_b_min: int,
        recipient: str,
        deadline: int,
    ) -> JSONLike:
        """Prepare encoded tx adding liquidity to an existing v2 pool, pulling both tokens from the caller."""
        to_checksum = ledger_api.api.to_checksum_address
        params = (
            to_checksum(factory),
            to_checksum(token_a),
            to_checksum(token_b),
            False,  # createPool: never seed a new pool
            (SpendMode.AMOUNT, amount_a),
            (SpendMode.AMOUNT, amount_b),
            True,  # payerAIsUser: pull tokenA from the Safe
            True,  # payerBIsUser: pull tokenB from the Safe
            amount_a_min,
            amount_b_min,
            to_checksum(recipient),
            0,  # liquidityMin: the per-token minimums bound the deposit
        )
        command_input = encode([ADD_LIQUIDITY_PARAMS_TYPE], [params])
        return cls.get_execute_tx(
            ledger_api,
            contract_address,
            bytes([ADD_LIQUIDITY_COMMAND]),
            [command_input],
            deadline,
        )

    @classmethod
    def get_remove_liquidity_tx(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        pool: str,
        liquidity: int,
        amount0_min: int,
        amount1_min: int,
        recipient: str,
        deadline: int,
    ) -> JSONLike:
        """Prepare encoded tx burning LP tokens pulled from the caller."""
        to_checksum = ledger_api.api.to_checksum_address
        params = (
            to_checksum(pool),
            (SpendMode.AMOUNT, liquidity),
            True,  # payerIsUser: pull the LP tokens from the Safe
            amount0_min,
            amount1_min,
            to_checksum(recipient),
        )
        command_input = encode([REMOVE_LIQUIDITY_PARAMS_TYPE], [params])
        return cls.get_execute_tx(
            ledger_api,
            contract_address,
            bytes([REMOVE_LIQUIDITY_COMMAND]),
            [command_input],
            deadline,
        )

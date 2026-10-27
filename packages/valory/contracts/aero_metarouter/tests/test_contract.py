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

"""Tests for the AeroMetarouterContract contract."""

import json
from pathlib import Path
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

from eth_abi import decode
from web3 import Web3

from packages.valory.contracts.aero_metarouter.contract import (
    ADD_LIQUIDITY_COMMAND,
    ADD_LIQUIDITY_PARAMS_TYPE,
    AeroMetarouterContract,
    REMOVE_LIQUIDITY_COMMAND,
    REMOVE_LIQUIDITY_PARAMS_TYPE,
    SpendMode,
)

METAROUTER = "0x0fEf9e88A7b76BCDBF451780C6774FE32965Ee98"
FACTORY = "0x8E6a9FCd1d638CC760Ffba397605Fe3355dE76D2"
TOKEN_A = "0x4200000000000000000000000000000000000006"
TOKEN_B = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
POOL = "0x800eb395b577b234c5065c8E0Ccb0F3a1401b2C2"
SAFE = "0x1111111111111111111111111111111111111111"
ABI_PATH = Path(__file__).parents[1] / "build" / "contract_interface.json"


def _real_instance() -> Tuple[MagicMock, Any]:
    """Return a ledger api whose contract instance encodes with the real ABI."""
    w3 = Web3()
    abi = json.loads(ABI_PATH.read_text())["abi"]
    ledger_api = MagicMock()
    ledger_api.api = w3
    return ledger_api, w3.eth.contract(address=METAROUTER, abi=abi)


def _decode_execute(tx_hash: bytes) -> tuple:
    """Decode the arguments of an execute(bytes,bytes[],uint256) call."""
    assert tx_hash[:4] == Web3.keccak(text="execute(bytes,bytes[],uint256)")[:4]
    return decode(["bytes", "bytes[]", "uint256"], tx_hash[4:])


class TestGetExecuteTx:
    """Tests for the get_execute_tx method."""

    def test_rejects_mismatched_commands_and_inputs(self) -> None:
        """Test that a command without its input is refused before encoding."""
        result = AeroMetarouterContract.get_execute_tx(
            MagicMock(), METAROUTER, bytes([0x22, 0x23]), [b"x"], 1
        )
        assert "error" in result

    def test_encodes_execute(self) -> None:
        """Test that get_execute_tx encodes the commands, inputs and deadline."""
        ledger_api, instance = _real_instance()
        with patch.object(
            AeroMetarouterContract, "get_instance", return_value=instance
        ):
            result = AeroMetarouterContract.get_execute_tx(
                ledger_api, METAROUTER, bytes([0x22]), [b"\x01\x02"], 99
            )

        assert _decode_execute(result["tx_hash"]) == (b"\x22", (b"\x01\x02",), 99)


class TestGetAddLiquidityTx:
    """Tests for the get_add_liquidity_tx method."""

    def test_encodes_add_liquidity_params(self) -> None:
        """Test that the ADD_LIQUIDITY input round-trips as AddLiquidityParams."""
        ledger_api, instance = _real_instance()
        with patch.object(
            AeroMetarouterContract, "get_instance", return_value=instance
        ):
            result = AeroMetarouterContract.get_add_liquidity_tx(
                ledger_api,
                METAROUTER,
                factory=FACTORY.lower(),
                token_a=TOKEN_A,
                token_b=TOKEN_B.lower(),
                amount_a=10,
                amount_b=20,
                amount_a_min=9,
                amount_b_min=19,
                recipient=SAFE,
                deadline=1234,
            )

        commands, inputs, deadline = _decode_execute(result["tx_hash"])
        assert commands == bytes([ADD_LIQUIDITY_COMMAND])
        assert deadline == 1234
        (params,) = decode([ADD_LIQUIDITY_PARAMS_TYPE], inputs[0])
        assert params == (
            FACTORY.lower(),
            TOKEN_A.lower(),
            TOKEN_B.lower(),
            False,
            (SpendMode.AMOUNT, 10),
            (SpendMode.AMOUNT, 20),
            True,
            True,
            9,
            19,
            SAFE.lower(),
            0,
        )


class TestGetRemoveLiquidityTx:
    """Tests for the get_remove_liquidity_tx method."""

    def test_encodes_remove_liquidity_params(self) -> None:
        """Test that the REMOVE_LIQUIDITY input round-trips as RemoveLiquidityParams."""
        ledger_api, instance = _real_instance()
        with patch.object(
            AeroMetarouterContract, "get_instance", return_value=instance
        ):
            result = AeroMetarouterContract.get_remove_liquidity_tx(
                ledger_api,
                METAROUTER,
                pool=POOL,
                liquidity=500,
                amount0_min=4,
                amount1_min=5,
                recipient=SAFE,
                deadline=77,
            )

        commands, inputs, deadline = _decode_execute(result["tx_hash"])
        assert commands == bytes([REMOVE_LIQUIDITY_COMMAND])
        assert deadline == 77
        (params,) = decode([REMOVE_LIQUIDITY_PARAMS_TYPE], inputs[0])
        assert params == (
            POOL.lower(),
            (SpendMode.AMOUNT, 500),
            True,
            4,
            5,
            SAFE.lower(),
        )

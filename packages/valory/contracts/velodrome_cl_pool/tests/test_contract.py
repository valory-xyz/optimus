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

"""Tests for the VelodromeCLPoolContract."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.velodrome_cl_pool.contract import (
    VelodromeCLPoolContract,
)

MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
MOCK_RECIPIENT = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"


class TestEncodeCall:
    """Tests for _encode_call."""

    def test_encode_call_returns_tx_hash_dict(self) -> None:
        """Test that _encode_call returns a dict with tx_hash key containing raw ABI data."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xdeadbeef"

        with patch.object(
            VelodromeCLPoolContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeCLPoolContract._encode_call(
                mock_ledger_api, MOCK_ADDRESS, "someMethod", (1, 2)
            )

        assert result == {"tx_hash": "0xdeadbeef"}
        mock_instance.encode_abi.assert_called_once_with("someMethod", args=(1, 2))


class TestMint:
    """Tests for mint."""

    def test_mint_delegates_to_encode_call(self) -> None:
        """Test that mint delegates to _encode_call with correct args."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xm1nt"

        with patch.object(
            VelodromeCLPoolContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeCLPoolContract.mint(
                mock_ledger_api,
                MOCK_ADDRESS,
                recipient=MOCK_RECIPIENT,
                tick_lower=-100,
                tick_upper=100,
                amount=5000,
                data=b"\x00",
            )

        assert result == {"tx_hash": "0xm1nt"}
        mock_instance.encode_abi.assert_called_once_with(
            "mint",
            args=(MOCK_RECIPIENT, -100, 100, 5000, b"\x00"),
        )


class TestBurn:
    """Tests for burn."""

    def test_burn_delegates_to_encode_call(self) -> None:
        """Test that burn delegates to _encode_call with correct args."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xburn"

        with patch.object(
            VelodromeCLPoolContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeCLPoolContract.burn(
                mock_ledger_api,
                MOCK_ADDRESS,
                tick_lower=-200,
                tick_upper=200,
                amount=3000,
            )

        assert result == {"tx_hash": "0xburn"}
        mock_instance.encode_abi.assert_called_once_with(
            "burn",
            args=(-200, 200, 3000),
        )


class TestCollect:
    """Tests for collect."""

    def test_collect_delegates_to_encode_call(self) -> None:
        """Test that collect delegates to _encode_call with correct args."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xc011ect"

        with patch.object(
            VelodromeCLPoolContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeCLPoolContract.collect(
                mock_ledger_api,
                MOCK_ADDRESS,
                recipient=MOCK_RECIPIENT,
                tick_lower=-50,
                tick_upper=50,
                amount0_requested=1000,
                amount1_requested=2000,
            )

        assert result == {"tx_hash": "0xc011ect"}
        mock_instance.encode_abi.assert_called_once_with(
            "collect",
            args=(MOCK_RECIPIENT, -50, 50, 1000, 2000),
        )


class TestSlot0:
    """Tests for slot0."""

    def test_slot0_returns_nested_dict_with_six_fields(self) -> None:
        """Test that slot0 returns a nested dict with 6 fields from the pool state."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.slot0.return_value.call.return_value = (
            79228162514264337593543950336,  # sqrt_price_x96
            0,  # tick
            10,  # observation_index
            100,  # observation_cardinality
            200,  # observation_cardinality_next
            True,  # unlocked
        )

        with patch.object(
            VelodromeCLPoolContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeCLPoolContract.slot0(mock_ledger_api, MOCK_ADDRESS)

        assert result == {
            "slot0": {
                "sqrt_price_x96": 79228162514264337593543950336,
                "tick": 0,
                "observation_index": 10,
                "observation_cardinality": 100,
                "observation_cardinality_next": 200,
                "unlocked": True,
            }
        }


class TestGetTickSpacing:
    """Tests for get_tick_spacing."""

    def test_get_tick_spacing_returns_data_dict(self) -> None:
        """Test that get_tick_spacing returns dict(data=tick_spacing)."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.tickSpacing.return_value.call.return_value = 60

        with patch.object(
            VelodromeCLPoolContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeCLPoolContract.get_tick_spacing(
                mock_ledger_api, MOCK_ADDRESS
            )

        assert result == {"data": 60}


class TestGetPoolTokens:
    """Tests for get_pool_tokens."""

    def test_get_pool_tokens_returns_list_of_two_tokens(self) -> None:
        """Test that get_pool_tokens calls token0() and token1() and returns dict(tokens=[...])."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        token0_addr = "0xToken0Address000000000000000000000000000"
        token1_addr = "0xToken1Address000000000000000000000000000"
        mock_instance.functions.token0.return_value.call.return_value = token0_addr
        mock_instance.functions.token1.return_value.call.return_value = token1_addr

        with patch.object(
            VelodromeCLPoolContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeCLPoolContract.get_pool_tokens(
                mock_ledger_api, MOCK_ADDRESS
            )

        assert result == {"tokens": [token0_addr, token1_addr]}


class TestAbiIntegrity:
    """Verify that every function used in contract.py exists in the ABI."""

    ABI_FILE = "contract_interface.json"
    EXPECTED_FUNCTIONS = [
        "burn",
        "collect",
        "mint",
        "slot0",
        "tickSpacing",
        "token0",
        "token1",
    ]

    @classmethod
    def _load_abi_function_names(cls) -> set:
        abi_path = Path(__file__).parents[1] / "build" / cls.ABI_FILE
        with open(abi_path) as f:
            data = json.load(f)
        abi = data.get("abi", data) if isinstance(data, dict) else data
        return {e["name"] for e in abi if e.get("type") == "function"}

    def test_all_functions_present(self) -> None:
        """Test that all functions used in contract.py exist in the ABI."""
        abi_funcs = self._load_abi_function_names()
        missing = [f for f in self.EXPECTED_FUNCTIONS if f not in abi_funcs]
        assert not missing, f"Functions missing from ABI: {missing}"

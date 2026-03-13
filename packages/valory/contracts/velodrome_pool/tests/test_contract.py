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

"""Tests for the VelodromePoolContract contract."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.velodrome_pool.contract import VelodromePoolContract


MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
MOCK_ACCOUNT = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
MOCK_SPENDER = "0x00000000000000000000000000000000DeaDBeef"


class TestGetBalance:
    """Tests for the get_balance method."""

    def test_get_balance_returns_correct_balance(self) -> None:
        """Test that get_balance returns a dict with the balance key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.balanceOf.return_value.call.return_value = 1000

        with patch.object(
            VelodromePoolContract, "get_instance", return_value=mock_contract_instance
        ):
            result = VelodromePoolContract.get_balance(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT
            )

        assert result == {"balance": 1000}
        mock_contract_instance.functions.balanceOf.assert_called_once_with(MOCK_ACCOUNT)


class TestBuildApprovalTx:
    """Tests for the build_approval_tx method."""

    def test_build_approval_tx_returns_tx_hash(self) -> None:
        """Test that build_approval_tx returns a dict with a tx_hash key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_SPENDER
        mock_contract_instance = MagicMock()
        encoded_data = "0xabcdef1234"
        mock_contract_instance.encode_abi.return_value = encoded_data

        with patch.object(
            VelodromePoolContract, "get_instance", return_value=mock_contract_instance
        ):
            result = VelodromePoolContract.build_approval_tx(
                mock_ledger_api, MOCK_ADDRESS, MOCK_SPENDER, 1000
            )

        assert result == {"tx_hash": bytes.fromhex("abcdef1234")}

    def test_build_approval_tx_checksums_spender(self) -> None:
        """Test that build_approval_tx checksums the spender address.

        :return: None
        """
        mock_ledger_api = MagicMock()
        checksummed = "0x00000000000000000000000000000000DeaDBeef"
        mock_ledger_api.api.to_checksum_address.return_value = checksummed
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = "0xaa"

        with patch.object(
            VelodromePoolContract, "get_instance", return_value=mock_contract_instance
        ):
            VelodromePoolContract.build_approval_tx(
                mock_ledger_api, MOCK_ADDRESS, "0xdeadbeef", 500
            )

        mock_ledger_api.api.to_checksum_address.assert_called_once_with("0xdeadbeef")
        mock_contract_instance.encode_abi.assert_called_once_with(
            "approve", args=(checksummed, 500)
        )


class TestGetReserves:
    """Tests for the get_reserves method."""

    def test_get_reserves_returns_both_reserves(self) -> None:
        """Test that get_reserves returns a dict with both reserve values.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.reserve0.return_value.call.return_value = 5000
        mock_contract_instance.functions.reserve1.return_value.call.return_value = 10000

        with patch.object(
            VelodromePoolContract, "get_instance", return_value=mock_contract_instance
        ):
            result = VelodromePoolContract.get_reserves(mock_ledger_api, MOCK_ADDRESS)

        assert result == {"data": [5000, 10000]}


class TestAbiIntegrity:
    """Verify that every function used in contract.py exists in the ABI."""

    ABI_FILE = "VelodromePool.json"
    EXPECTED_FUNCTIONS = ["approve", "balanceOf", "reserve0", "reserve1"]

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

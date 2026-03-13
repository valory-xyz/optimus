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

"""Tests for the VaultContract (Balancer Vault)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.balancer_vault.contract import VaultContract

MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
MOCK_SENDER = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"
MOCK_RECIPIENT = "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB"
MOCK_POOL_ID = "0x" + "ab" * 32


class TestGetPoolTokens:
    """Tests for get_pool_tokens."""

    def test_get_pool_tokens_converts_pool_id_and_returns_tokens(self) -> None:
        """Test that get_pool_tokens converts pool_id hex to bytes and calls getPoolTokens."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        expected_data = (
            ["0xToken0", "0xToken1"],
            [1000, 2000],
            99999,
        )
        mock_instance.functions.getPoolTokens.return_value.call.return_value = (
            expected_data
        )

        with patch.object(VaultContract, "get_instance", return_value=mock_instance):
            result = VaultContract.get_pool_tokens(
                mock_ledger_api,
                MOCK_ADDRESS,
                pool_id=MOCK_POOL_ID,
            )

        assert result == {"tokens": expected_data}
        pool_id_bytes = bytes.fromhex(MOCK_POOL_ID[2:])
        mock_instance.functions.getPoolTokens.assert_called_once_with(pool_id_bytes)


class TestJoinPool:
    """Tests for join_pool."""

    def test_join_pool_returns_tx_hash_bytes(self) -> None:
        """Test that join_pool encodes user_data, calls encode_abi, and returns tx_hash as bytes."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        # encode_abi returns a hex string starting with "0x"
        mock_instance.encode_abi.return_value = "0xabcdef1234"

        with patch.object(VaultContract, "get_instance", return_value=mock_instance):
            result = VaultContract.join_pool(
                mock_ledger_api,
                MOCK_ADDRESS,
                pool_id=MOCK_POOL_ID,
                sender=MOCK_SENDER,
                recipient=MOCK_RECIPIENT,
                assets=["0xToken0", "0xToken1"],
                max_amounts_in=[1000, 2000],
                join_kind=1,
                minimum_bpt=100,
                from_internal_balance=False,
            )

        assert result == {"tx_hash": bytes.fromhex("abcdef1234")}
        mock_instance.encode_abi.assert_called_once()
        call_args = mock_instance.encode_abi.call_args
        assert call_args[0][0] == "joinPool"


class TestExitPool:
    """Tests for exit_pool."""

    def test_exit_pool_returns_tx_hash_bytes(self) -> None:
        """Test that exit_pool encodes user_data, calls encode_abi, and returns tx_hash as bytes."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xfeedface"

        with patch.object(VaultContract, "get_instance", return_value=mock_instance):
            result = VaultContract.exit_pool(
                mock_ledger_api,
                MOCK_ADDRESS,
                pool_id=MOCK_POOL_ID,
                sender=MOCK_SENDER,
                recipient=MOCK_RECIPIENT,
                assets=["0xToken0", "0xToken1"],
                min_amounts_out=[100, 200],
                exit_kind=1,
                bpt_amount_in=500,
                to_internal_balance=False,
            )

        assert result == {"tx_hash": bytes.fromhex("feedface")}
        mock_instance.encode_abi.assert_called_once()
        call_args = mock_instance.encode_abi.call_args
        assert call_args[0][0] == "exitPool"


class TestSimulateTx:
    """Tests for simulate_tx."""

    def test_simulate_tx_success_returns_data_true(self) -> None:
        """Test that simulate_tx returns dict(data=True) when eth.call succeeds."""
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.eth.call.return_value = b"\x00"
        mock_ledger_api.api.to_checksum_address.side_effect = lambda x: x

        with patch.object(VaultContract, "get_instance"):
            result = VaultContract.simulate_tx(
                mock_ledger_api,
                MOCK_ADDRESS,
                sender_address=MOCK_SENDER,
                data="0xdeadbeef",
                gas_limit=100000,
            )

        assert result == {"data": True}
        mock_ledger_api.api.eth.call.assert_called_once()

    def test_simulate_tx_failure_returns_data_false(self) -> None:
        """Test that simulate_tx returns dict(data=False) when eth.call raises an exception."""
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.eth.call.side_effect = Exception("revert")
        mock_ledger_api.api.to_checksum_address.side_effect = lambda x: x

        with patch.object(VaultContract, "get_instance"):
            result = VaultContract.simulate_tx(
                mock_ledger_api,
                MOCK_ADDRESS,
                sender_address=MOCK_SENDER,
                data="0xdeadbeef",
                gas_limit=100000,
            )

        assert result == {"data": False}


class TestAbiIntegrity:
    """Verify that every function used in contract.py exists in the ABI."""

    ABI_FILE = "Vault.json"
    EXPECTED_FUNCTIONS = ["exitPool", "getPoolTokens", "joinPool"]

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

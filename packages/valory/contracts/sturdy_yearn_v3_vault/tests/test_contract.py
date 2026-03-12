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

"""Tests for the YearnV3VaultContract (Sturdy)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.sturdy_yearn_v3_vault.contract import (
    YearnV3VaultContract,
)

MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
MOCK_RECEIVER = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"
MOCK_OWNER = "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB"


class TestDeposit:
    """Tests for deposit."""

    def test_deposit_returns_tx_hash_bytes(self) -> None:
        """Test that deposit encodes a deposit call and returns tx_hash as bytes."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xdeadbeef"

        with patch.object(
            YearnV3VaultContract, "get_instance", return_value=mock_instance
        ):
            result = YearnV3VaultContract.deposit(
                mock_ledger_api,
                MOCK_ADDRESS,
                assets=1000,
                receiver=MOCK_RECEIVER,
            )

        assert result == {"tx_hash": bytes.fromhex("deadbeef")}
        mock_instance.encode_abi.assert_called_once_with(
            "deposit", args=(1000, MOCK_RECEIVER)
        )


class TestWithdraw:
    """Tests for withdraw."""

    def test_withdraw_returns_tx_hash_bytes(self) -> None:
        """Test that withdraw encodes a withdraw call and returns tx_hash as bytes."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xfeedface"

        with patch.object(
            YearnV3VaultContract, "get_instance", return_value=mock_instance
        ):
            result = YearnV3VaultContract.withdraw(
                mock_ledger_api,
                MOCK_ADDRESS,
                assets=500,
                receiver=MOCK_RECEIVER,
                owner=MOCK_OWNER,
            )

        assert result == {"tx_hash": bytes.fromhex("feedface")}
        mock_instance.encode_abi.assert_called_once_with(
            "withdraw", args=(500, MOCK_RECEIVER, MOCK_OWNER)
        )


class TestRedeem:
    """Tests for redeem."""

    def test_redeem_returns_tx_hash_bytes(self) -> None:
        """Test that redeem encodes a redeem call and returns tx_hash as bytes."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xcafebabe"

        with patch.object(
            YearnV3VaultContract, "get_instance", return_value=mock_instance
        ):
            result = YearnV3VaultContract.redeem(
                mock_ledger_api,
                MOCK_ADDRESS,
                shares=300,
                receiver=MOCK_RECEIVER,
                owner=MOCK_OWNER,
            )

        assert result == {"tx_hash": bytes.fromhex("cafebabe")}
        mock_instance.encode_abi.assert_called_once_with(
            "redeem", args=(300, MOCK_RECEIVER, MOCK_OWNER)
        )


class TestMaxRedeem:
    """Tests for max_redeem."""

    def test_max_redeem_returns_amount(self) -> None:
        """Test that max_redeem calls maxRedeem and returns the maximum redeemable amount."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.maxRedeem.return_value.call.return_value = 99999

        with patch.object(
            YearnV3VaultContract, "get_instance", return_value=mock_instance
        ):
            result = YearnV3VaultContract.max_redeem(
                mock_ledger_api, MOCK_ADDRESS, owner=MOCK_OWNER
            )

        assert result == {"amount": 99999}
        mock_instance.functions.maxRedeem.assert_called_once_with(MOCK_OWNER)


class TestMaxWithdraw:
    """Tests for max_withdraw."""

    def test_max_withdraw_returns_amount(self) -> None:
        """Test that max_withdraw calls maxWithdraw and returns the maximum withdrawable amount."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.maxWithdraw.return_value.call.return_value = 88888

        with patch.object(
            YearnV3VaultContract, "get_instance", return_value=mock_instance
        ):
            result = YearnV3VaultContract.max_withdraw(
                mock_ledger_api, MOCK_ADDRESS, owner=MOCK_OWNER
            )

        assert result == {"amount": 88888}
        mock_instance.functions.maxWithdraw.assert_called_once_with(MOCK_OWNER)


class TestBalanceOf:
    """Tests for balance_of."""

    def test_balance_of_returns_amount(self) -> None:
        """Test that balance_of calls balanceOf and returns the balance."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.balanceOf.return_value.call.return_value = 5000

        with patch.object(
            YearnV3VaultContract, "get_instance", return_value=mock_instance
        ):
            result = YearnV3VaultContract.balance_of(
                mock_ledger_api, MOCK_ADDRESS, owner=MOCK_OWNER
            )

        assert result == {"amount": 5000}
        mock_instance.functions.balanceOf.assert_called_once_with(MOCK_OWNER)


class TestName:
    """Tests for name."""

    def test_name_returns_vault_name(self) -> None:
        """Test that name calls name() and returns the vault name string."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.name.return_value.call.return_value = "Sturdy Vault"

        with patch.object(
            YearnV3VaultContract, "get_instance", return_value=mock_instance
        ):
            result = YearnV3VaultContract.name(mock_ledger_api, MOCK_ADDRESS)

        assert result == {"name": "Sturdy Vault"}


class TestTotalSupply:
    """Tests for total_supply."""

    def test_total_supply_returns_supply(self) -> None:
        """Test that total_supply calls totalSupply() and returns the total supply."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.totalSupply.return_value.call.return_value = 1000000

        with patch.object(
            YearnV3VaultContract, "get_instance", return_value=mock_instance
        ):
            result = YearnV3VaultContract.total_supply(mock_ledger_api, MOCK_ADDRESS)

        assert result == {"total_supply": 1000000}


class TestTotalAssets:
    """Tests for total_assets."""

    def test_total_assets_returns_assets(self) -> None:
        """Test that total_assets calls totalAssets() and returns the total assets."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.totalAssets.return_value.call.return_value = 2000000

        with patch.object(
            YearnV3VaultContract, "get_instance", return_value=mock_instance
        ):
            result = YearnV3VaultContract.total_assets(mock_ledger_api, MOCK_ADDRESS)

        assert result == {"total_assets": 2000000}


class TestDecimals:
    """Tests for decimals."""

    def test_decimals_returns_decimal_count(self) -> None:
        """Test that decimals calls decimals() and returns the number of decimals."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.decimals.return_value.call.return_value = 18

        with patch.object(
            YearnV3VaultContract, "get_instance", return_value=mock_instance
        ):
            result = YearnV3VaultContract.decimals(mock_ledger_api, MOCK_ADDRESS)

        assert result == {"decimals": 18}


class TestAbiIntegrity:
    """Verify that every function used in contract.py exists in the ABI."""

    ABI_FILE = "YearnV3Vault.json"
    EXPECTED_FUNCTIONS = [
        "balanceOf",
        "decimals",
        "deposit",
        "maxRedeem",
        "maxWithdraw",
        "name",
        "redeem",
        "totalAssets",
        "totalSupply",
        "withdraw",
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

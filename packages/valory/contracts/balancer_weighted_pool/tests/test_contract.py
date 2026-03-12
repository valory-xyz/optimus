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

"""Tests for the WeightedPoolContract contract."""

from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.balancer_weighted_pool.contract import (
    WeightedPoolContract,
)


MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
MOCK_ACCOUNT = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"


class TestGetBalance:
    """Tests for the get_balance method."""

    def test_get_balance_returns_correct_balance(self) -> None:
        """Test that get_balance returns a dict with the balance key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.balanceOf.return_value.call.return_value = 5000

        with patch.object(
            WeightedPoolContract, "get_instance", return_value=mock_contract_instance
        ):
            result = WeightedPoolContract.get_balance(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT
            )

        assert result == {"balance": 5000}
        mock_contract_instance.functions.balanceOf.assert_called_once_with(MOCK_ACCOUNT)


class TestGetName:
    """Tests for the get_name method."""

    def test_get_name_returns_pool_name(self) -> None:
        """Test that get_name returns a dict with the name key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.name.return_value.call.return_value = (
            "Balancer 80BAL-20WETH"
        )

        with patch.object(
            WeightedPoolContract, "get_instance", return_value=mock_contract_instance
        ):
            result = WeightedPoolContract.get_name(mock_ledger_api, MOCK_ADDRESS)

        assert result == {"name": "Balancer 80BAL-20WETH"}


class TestGetPoolId:
    """Tests for the get_pool_id method."""

    def test_get_pool_id_returns_hex_prefixed_id(self) -> None:
        """Test that get_pool_id returns a dict with the pool_id prefixed with 0x.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        pool_id_bytes = bytes.fromhex("abcdef1234567890" * 4)
        mock_contract_instance.functions.getPoolId.return_value.call.return_value = (
            pool_id_bytes
        )

        with patch.object(
            WeightedPoolContract, "get_instance", return_value=mock_contract_instance
        ):
            result = WeightedPoolContract.get_pool_id(mock_ledger_api, MOCK_ADDRESS)

        assert result == {"pool_id": "0x" + pool_id_bytes.hex()}


class TestGetVaultAddress:
    """Tests for the get_vault_address method."""

    def test_get_vault_address_returns_vault(self) -> None:
        """Test that get_vault_address returns a dict with the vault key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        vault_address = "0xBA12222222228d8Ba445958a75a0704d566BF2C8"
        mock_contract_instance.functions.getVault.return_value.call.return_value = (
            vault_address
        )

        with patch.object(
            WeightedPoolContract, "get_instance", return_value=mock_contract_instance
        ):
            result = WeightedPoolContract.get_vault_address(
                mock_ledger_api, MOCK_ADDRESS
            )

        assert result == {"vault": vault_address}


class TestGetTotalSupply:
    """Tests for the get_total_supply method."""

    def test_get_total_supply_returns_supply(self) -> None:
        """Test that get_total_supply returns a dict with the data key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.totalSupply.return_value.call.return_value = (
            10**18
        )

        with patch.object(
            WeightedPoolContract, "get_instance", return_value=mock_contract_instance
        ):
            result = WeightedPoolContract.get_total_supply(
                mock_ledger_api, MOCK_ADDRESS
            )

        assert result == {"data": 10**18}

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

"""Tests for the VelodromeCLGaugeContract contract."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.velodrome_cl_gauge.contract import (
    VelodromeCLGaugeContract,
)


MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
MOCK_ACCOUNT = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
MOCK_CHECKSUMMED_ACCOUNT = "0xAbCdEfAbCdEfAbCdEfAbCdEfAbCdEfAbCdEfAbCd"
MOCK_TOKEN_ID = 42


class TestEncodeCall:
    """Tests for the _encode_call method."""

    def test_encode_call_returns_tx_hash(self) -> None:
        """Test that _encode_call returns a dict with the tx_hash key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        encoded_data = b"0xabcdef"
        mock_contract_instance.encode_abi.return_value = encoded_data

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract._encode_call(
                mock_ledger_api, MOCK_ADDRESS, "deposit", (MOCK_TOKEN_ID,)
            )

        assert result == {"tx_hash": encoded_data}

    def test_encode_call_calls_encode_abi_with_correct_args(self) -> None:
        """Test that _encode_call calls encode_abi with the correct method name and args.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = b"0x00"

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeCLGaugeContract._encode_call(
                mock_ledger_api, MOCK_ADDRESS, "withdraw", (99,)
            )

        mock_contract_instance.encode_abi.assert_called_once_with(
            "withdraw", args=(99,)
        )


class TestDeposit:
    """Tests for the deposit method."""

    def test_deposit_returns_tx_hash(self) -> None:
        """Test that deposit returns a dict with the tx_hash key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        encoded_data = b"0xdeposit_data"
        mock_contract_instance.encode_abi.return_value = encoded_data

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.deposit(
                mock_ledger_api, MOCK_ADDRESS, MOCK_TOKEN_ID
            )

        assert result == {"tx_hash": encoded_data}

    def test_deposit_calls_encode_abi_with_correct_args(self) -> None:
        """Test that deposit calls encode_abi with 'deposit' method name and token_id tuple.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = b"0x00"

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeCLGaugeContract.deposit(
                mock_ledger_api, MOCK_ADDRESS, MOCK_TOKEN_ID
            )

        mock_contract_instance.encode_abi.assert_called_once_with(
            "deposit", args=(MOCK_TOKEN_ID,)
        )


class TestWithdraw:
    """Tests for the withdraw method."""

    def test_withdraw_returns_tx_hash(self) -> None:
        """Test that withdraw returns a dict with the tx_hash key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        encoded_data = b"0xwithdraw_data"
        mock_contract_instance.encode_abi.return_value = encoded_data

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.withdraw(
                mock_ledger_api, MOCK_ADDRESS, MOCK_TOKEN_ID
            )

        assert result == {"tx_hash": encoded_data}

    def test_withdraw_calls_encode_abi_with_correct_args(self) -> None:
        """Test that withdraw calls encode_abi with 'withdraw' method name and token_id tuple.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = b"0x00"

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeCLGaugeContract.withdraw(
                mock_ledger_api, MOCK_ADDRESS, MOCK_TOKEN_ID
            )

        mock_contract_instance.encode_abi.assert_called_once_with(
            "withdraw", args=(MOCK_TOKEN_ID,)
        )


class TestGetReward:
    """Tests for the get_reward method."""

    def test_get_reward_returns_tx_hash(self) -> None:
        """Test that get_reward returns a dict with the tx_hash key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        encoded_data = b"0xreward_data"
        mock_contract_instance.encode_abi.return_value = encoded_data

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.get_reward(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT
            )

        assert result == {"tx_hash": encoded_data}

    def test_get_reward_checksums_account(self) -> None:
        """Test that get_reward checksums the account address before encoding.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = b"0x00"

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeCLGaugeContract.get_reward(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT
            )

        mock_ledger_api.api.to_checksum_address.assert_called_once_with(MOCK_ACCOUNT)

    def test_get_reward_calls_encode_abi_with_checksummed_account(self) -> None:
        """Test that get_reward passes the checksummed account to encode_abi.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = b"0x00"

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeCLGaugeContract.get_reward(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT
            )

        mock_contract_instance.encode_abi.assert_called_once_with(
            "getReward", args=(MOCK_CHECKSUMMED_ACCOUNT,)
        )


class TestGetRewardForTokenId:
    """Tests for the get_reward_for_token_id method."""

    def test_get_reward_for_token_id_returns_tx_hash(self) -> None:
        """Test that get_reward_for_token_id returns a dict with the tx_hash key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        encoded_data = b"0xreward_token_data"
        mock_contract_instance.encode_abi.return_value = encoded_data

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.get_reward_for_token_id(
                mock_ledger_api, MOCK_ADDRESS, MOCK_TOKEN_ID
            )

        assert result == {"tx_hash": encoded_data}

    def test_get_reward_for_token_id_calls_encode_abi_with_correct_args(self) -> None:
        """Test that get_reward_for_token_id calls encode_abi with 'getReward' and token_id.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = b"0x00"

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeCLGaugeContract.get_reward_for_token_id(
                mock_ledger_api, MOCK_ADDRESS, MOCK_TOKEN_ID
            )

        mock_contract_instance.encode_abi.assert_called_once_with(
            "getReward", args=(MOCK_TOKEN_ID,)
        )


class TestEarned:
    """Tests for the earned method."""

    def test_earned_returns_earned_amount(self) -> None:
        """Test that earned returns a dict with the earned key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.earned.return_value.call.return_value = 5000

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.earned(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT, MOCK_TOKEN_ID
            )

        assert result == {"earned": 5000}

    def test_earned_checksums_account(self) -> None:
        """Test that earned checksums the account address before querying.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.earned.return_value.call.return_value = 0

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeCLGaugeContract.earned(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT, MOCK_TOKEN_ID
            )

        mock_ledger_api.api.to_checksum_address.assert_called_once_with(MOCK_ACCOUNT)

    def test_earned_calls_contract_function_with_correct_args(self) -> None:
        """Test that earned calls the contract function with checksummed account and token_id.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.earned.return_value.call.return_value = 0

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeCLGaugeContract.earned(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT, MOCK_TOKEN_ID
            )

        mock_contract_instance.functions.earned.assert_called_once_with(
            MOCK_CHECKSUMMED_ACCOUNT, MOCK_TOKEN_ID
        )


class TestBalanceOf:
    """Tests for the balance_of method."""

    def test_balance_of_returns_balance(self) -> None:
        """Test that balance_of returns a dict with the balance key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.balanceOf.return_value.call.return_value = 2000

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.balance_of(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT
            )

        assert result == {"balance": 2000}

    def test_balance_of_checksums_account(self) -> None:
        """Test that balance_of checksums the account address before querying.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.balanceOf.return_value.call.return_value = 0

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeCLGaugeContract.balance_of(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT
            )

        mock_ledger_api.api.to_checksum_address.assert_called_once_with(MOCK_ACCOUNT)

    def test_balance_of_calls_contract_function_with_checksummed_account(self) -> None:
        """Test that balance_of calls the contract function with the checksummed account.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.balanceOf.return_value.call.return_value = 0

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeCLGaugeContract.balance_of(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT
            )

        mock_contract_instance.functions.balanceOf.assert_called_once_with(
            MOCK_CHECKSUMMED_ACCOUNT
        )


class TestStakedContains:
    """Tests for the staked_contains method."""

    def test_staked_contains_true(self) -> None:
        """Returns a dict with is_staked=True when the gauge has the token."""
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.stakedContains.return_value.call.return_value = (
            True
        )

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.staked_contains(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT, MOCK_TOKEN_ID
            )

        assert result == {"is_staked": True}

    def test_staked_contains_false(self) -> None:
        """Returns is_staked=False when the token is not in the gauge's stake set."""
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.stakedContains.return_value.call.return_value = (
            False
        )

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.staked_contains(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT, MOCK_TOKEN_ID
            )

        assert result == {"is_staked": False}

    def test_staked_contains_checksums_account(self) -> None:
        """Account is checksummed before the on-chain call."""
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.stakedContains.return_value.call.return_value = (
            False
        )

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeCLGaugeContract.staked_contains(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT, MOCK_TOKEN_ID
            )

        mock_ledger_api.api.to_checksum_address.assert_called_once_with(MOCK_ACCOUNT)
        mock_contract_instance.functions.stakedContains.assert_called_once_with(
            MOCK_CHECKSUMMED_ACCOUNT, MOCK_TOKEN_ID
        )


class TestTotalSupply:
    """Tests for the total_supply method."""

    def test_total_supply_returns_total(self) -> None:
        """Test that total_supply returns a dict with the total_supply key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.totalSupply.return_value.call.return_value = (
            1000000
        )

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.total_supply(
                mock_ledger_api, MOCK_ADDRESS
            )

        assert result == {"total_supply": 1000000}


class TestAbiIntegrity:
    """Verify that every function used in contract.py exists in the ABI."""

    ABI_FILE = "contract_interface.json"
    EXPECTED_FUNCTIONS = [
        "balanceOf",
        "deposit",
        "earned",
        "getReward",
        "stakedContains",
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

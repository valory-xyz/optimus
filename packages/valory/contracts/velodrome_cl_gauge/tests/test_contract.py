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

    def test_encode_call_calls_get_instance_with_correct_args(self) -> None:
        """Test that _encode_call calls get_instance with the correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = b"0x00"

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ) as mock_get_instance:
            VelodromeCLGaugeContract._encode_call(
                mock_ledger_api, MOCK_ADDRESS, "deposit", (MOCK_TOKEN_ID,)
            )

        mock_get_instance.assert_called_once_with(mock_ledger_api, MOCK_ADDRESS)


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

    def test_deposit_calls_get_instance(self) -> None:
        """Test that deposit calls get_instance with correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = b"0x00"

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ) as mock_get_instance:
            VelodromeCLGaugeContract.deposit(
                mock_ledger_api, MOCK_ADDRESS, MOCK_TOKEN_ID
            )

        mock_get_instance.assert_called_once_with(mock_ledger_api, MOCK_ADDRESS)


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

    def test_withdraw_calls_get_instance(self) -> None:
        """Test that withdraw calls get_instance with correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = b"0x00"

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ) as mock_get_instance:
            VelodromeCLGaugeContract.withdraw(
                mock_ledger_api, MOCK_ADDRESS, MOCK_TOKEN_ID
            )

        mock_get_instance.assert_called_once_with(mock_ledger_api, MOCK_ADDRESS)


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

    def test_get_reward_calls_get_instance(self) -> None:
        """Test that get_reward calls get_instance with correct arguments.

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
        ) as mock_get_instance:
            VelodromeCLGaugeContract.get_reward(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT
            )

        mock_get_instance.assert_called_once_with(mock_ledger_api, MOCK_ADDRESS)


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

    def test_get_reward_for_token_id_calls_get_instance(self) -> None:
        """Test that get_reward_for_token_id calls get_instance with correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = b"0x00"

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ) as mock_get_instance:
            VelodromeCLGaugeContract.get_reward_for_token_id(
                mock_ledger_api, MOCK_ADDRESS, MOCK_TOKEN_ID
            )

        mock_get_instance.assert_called_once_with(mock_ledger_api, MOCK_ADDRESS)


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

    def test_earned_calls_get_instance(self) -> None:
        """Test that earned calls get_instance with correct arguments.

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
        ) as mock_get_instance:
            VelodromeCLGaugeContract.earned(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT, MOCK_TOKEN_ID
            )

        mock_get_instance.assert_called_once_with(mock_ledger_api, MOCK_ADDRESS)

    def test_earned_with_zero_earned(self) -> None:
        """Test that earned correctly handles a zero earned amount.

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
            result = VelodromeCLGaugeContract.earned(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT, MOCK_TOKEN_ID
            )

        assert result == {"earned": 0}


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

    def test_balance_of_calls_get_instance(self) -> None:
        """Test that balance_of calls get_instance with correct arguments.

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
        ) as mock_get_instance:
            VelodromeCLGaugeContract.balance_of(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT
            )

        mock_get_instance.assert_called_once_with(mock_ledger_api, MOCK_ADDRESS)

    def test_balance_of_with_zero_balance(self) -> None:
        """Test that balance_of correctly handles a zero balance.

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
            result = VelodromeCLGaugeContract.balance_of(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT
            )

        assert result == {"balance": 0}


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

    def test_total_supply_calls_contract_function(self) -> None:
        """Test that total_supply calls the totalSupply contract function.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.totalSupply.return_value.call.return_value = 0

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeCLGaugeContract.total_supply(mock_ledger_api, MOCK_ADDRESS)

        mock_contract_instance.functions.totalSupply.assert_called_once()

    def test_total_supply_calls_get_instance(self) -> None:
        """Test that total_supply calls get_instance with correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.totalSupply.return_value.call.return_value = 0

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ) as mock_get_instance:
            VelodromeCLGaugeContract.total_supply(mock_ledger_api, MOCK_ADDRESS)

        mock_get_instance.assert_called_once_with(mock_ledger_api, MOCK_ADDRESS)

    def test_total_supply_with_zero(self) -> None:
        """Test that total_supply correctly handles a zero total supply.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.totalSupply.return_value.call.return_value = 0

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.total_supply(
                mock_ledger_api, MOCK_ADDRESS
            )

        assert result == {"total_supply": 0}

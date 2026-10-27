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
        """Test that _encode_call returns a dict with the tx_hash key."""
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        encoded_data = "0xabcdef"
        mock_contract_instance.encode_abi.return_value = encoded_data

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract._encode_call(
                mock_ledger_api, MOCK_ADDRESS, "deposit", (MOCK_TOKEN_ID,)
            )

        assert result == {"tx_hash": bytes.fromhex(encoded_data[2:])}

    def test_encode_call_calls_encode_abi_with_correct_args(self) -> None:
        """Test that _encode_call calls encode_abi with the correct method name and args."""
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = "0x00"

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
        """Test that deposit returns a dict with the tx_hash key."""
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        encoded_data = "0xde90517da70a"
        mock_contract_instance.encode_abi.return_value = encoded_data

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.deposit(
                mock_ledger_api, MOCK_ADDRESS, MOCK_TOKEN_ID
            )

        assert result == {"tx_hash": bytes.fromhex(encoded_data[2:])}

    def test_deposit_calls_encode_abi_with_correct_args(self) -> None:
        """Test that deposit calls encode_abi with 'deposit' method name and token_id tuple."""
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = "0x00"

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
        """Test that withdraw returns a dict with the tx_hash key."""
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        encoded_data = "0xa17fd7a7"
        mock_contract_instance.encode_abi.return_value = encoded_data

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.withdraw(
                mock_ledger_api, MOCK_ADDRESS, MOCK_TOKEN_ID
            )

        assert result == {"tx_hash": bytes.fromhex(encoded_data[2:])}

    def test_withdraw_calls_encode_abi_with_correct_args(self) -> None:
        """Test that withdraw calls encode_abi with 'withdraw' method name and token_id tuple."""
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = "0x00"

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
        """Test that get_reward returns a dict with the tx_hash key."""
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        encoded_data = "0x7e7a7da7"
        mock_contract_instance.encode_abi.return_value = encoded_data

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.get_reward(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT
            )

        assert result == {"tx_hash": bytes.fromhex(encoded_data[2:])}

    def test_get_reward_checksums_account(self) -> None:
        """Test that get_reward checksums the account address before encoding."""
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = "0x00"

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
        """Test that get_reward passes the checksummed account to encode_abi."""
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = "0x00"

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
        """Test that get_reward_for_token_id returns a dict with the tx_hash key."""
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        encoded_data = "0x7e7a70de"
        mock_contract_instance.encode_abi.return_value = encoded_data

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.get_reward_for_token_id(
                mock_ledger_api, MOCK_ADDRESS, MOCK_TOKEN_ID
            )

        assert result == {"tx_hash": bytes.fromhex(encoded_data[2:])}

    def test_get_reward_for_token_id_calls_encode_abi_with_correct_args(self) -> None:
        """Test that get_reward_for_token_id calls encode_abi with 'getReward' and token_id."""
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = "0x00"

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
        """Test that earned returns a dict with the earned key."""
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
        """Test that earned checksums the account address before querying."""
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
        """Test that earned calls the contract function with checksummed account and token_id."""
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
        """Test that balance_of returns a dict with the balance key."""
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
        """Test that balance_of checksums the account address before querying."""
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
        """Test that balance_of calls the contract function with the checksummed account."""
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
        """Test that total_supply returns a dict with the total_supply key."""
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


class TestEarnedByAccount:
    """Tests for the earned_by_account method."""

    def test_earned_by_account_calls_single_argument_overload(self) -> None:
        """Test that earned_by_account reads earned(address), not earned(address,uint256)."""
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.get_function_by_signature.return_value.return_value.call.return_value = (
            77
        )

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.earned_by_account(
                mock_ledger_api, MOCK_ADDRESS, MOCK_ACCOUNT
            )

        assert result == {"earned": 77}
        mock_contract_instance.get_function_by_signature.assert_called_once_with(
            "earned(address)"
        )
        mock_contract_instance.get_function_by_signature.return_value.assert_called_once_with(
            MOCK_CHECKSUMMED_ACCOUNT
        )


class TestClaimClEmissions:
    """Tests for the claim_cl_emissions method."""

    def test_claim_cl_emissions_encodes_token_ids(self) -> None:
        """Test that claim_cl_emissions encodes claimEmissions(account, recipient, tokenIds)."""
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.side_effect = lambda a: a.upper()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = "0xc1a2"

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.claim_cl_emissions(
                mock_ledger_api,
                MOCK_ADDRESS,
                MOCK_ACCOUNT,
                MOCK_ADDRESS,
                (MOCK_TOKEN_ID, 43),
            )

        assert result == {"tx_hash": bytes.fromhex("c1a2")}
        mock_contract_instance.encode_abi.assert_called_once_with(
            "claimEmissions",
            args=(MOCK_ACCOUNT.upper(), MOCK_ADDRESS.upper(), [MOCK_TOKEN_ID, 43]),
        )


class TestDepositBlock:
    """Tests for the deposit_block method."""

    def test_deposit_block_returns_block(self) -> None:
        """Test that deposit_block returns depositBlock(tokenId)."""
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.depositBlock.return_value.call.return_value = (
            123
        )

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.deposit_block(
                MagicMock(), MOCK_ADDRESS, MOCK_TOKEN_ID
            )

        assert result == {"deposit_block": 123}
        mock_contract_instance.functions.depositBlock.assert_called_once_with(
            MOCK_TOKEN_ID
        )


class TestGaugeFactory:
    """Tests for the gauge_factory method."""

    def test_gauge_factory_returns_factory(self) -> None:
        """Test that gauge_factory returns gaugeFactory()."""
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.gaugeFactory.return_value.call.return_value = (
            MOCK_ADDRESS
        )

        with patch.object(
            VelodromeCLGaugeContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeCLGaugeContract.gauge_factory(MagicMock(), MOCK_ADDRESS)

        assert result == {"gauge_factory": MOCK_ADDRESS}


class TestAbiIntegrity:
    """Verify that every function used in contract.py exists in the ABI."""

    ABI_FILE = "contract_interface.json"
    EXPECTED_FUNCTIONS = [
        "balanceOf",
        "claimEmissions",
        "deposit",
        "depositBlock",
        "earned",
        "gaugeFactory",
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

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

"""Tests for the DistributorContract contract."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.merkl_distributor.contract import DistributorContract


MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"


class TestClaimRewards:
    """Tests for the claim_rewards method."""

    def test_claim_rewards_returns_tx_hash(self) -> None:
        """Test that claim_rewards returns a dict with a tx_hash key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        encoded_data = "0xaabbccdd"
        mock_contract_instance.encode_abi.return_value = encoded_data

        users = ["0xuser1"]
        tokens = ["0xtoken1"]
        amounts = [1000]
        proofs = [
            ["0xaabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccdd"]
        ]

        with patch.object(
            DistributorContract, "get_instance", return_value=mock_contract_instance
        ):
            result = DistributorContract.claim_rewards(
                mock_ledger_api, MOCK_ADDRESS, users, tokens, amounts, proofs
            )

        assert result == {"tx_hash": bytes.fromhex("aabbccdd")}

    def test_claim_rewards_converts_proofs_to_bytes(self) -> None:
        """Test that claim_rewards converts hex proof strings to bytes.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = "0x00"

        users = ["0xuser1"]
        tokens = ["0xtoken1"]
        amounts = [500]
        proof_hex = "0xaabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccdd"
        proofs = [[proof_hex]]

        expected_proof_bytes = bytes.fromhex(proof_hex[2:])

        with patch.object(
            DistributorContract, "get_instance", return_value=mock_contract_instance
        ):
            DistributorContract.claim_rewards(
                mock_ledger_api, MOCK_ADDRESS, users, tokens, amounts, proofs
            )

        call_args = mock_contract_instance.encode_abi.call_args
        assert call_args[0][0] == "claim"
        proofs_arg = (
            call_args[1]["args"][3] if "args" in call_args[1] else call_args[0][1][3]
        )
        assert proofs_arg == [[expected_proof_bytes]]

    def test_claim_rewards_with_multiple_proofs(self) -> None:
        """Test that claim_rewards correctly handles multiple proof lists.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = "0xff"

        users = ["0xuser1", "0xuser2"]
        tokens = ["0xtoken1", "0xtoken2"]
        amounts = [100, 200]
        proof1 = "0x1111111111111111111111111111111111111111111111111111111111111111"
        proof2 = "0x2222222222222222222222222222222222222222222222222222222222222222"
        proof3 = "0x3333333333333333333333333333333333333333333333333333333333333333"
        proofs = [[proof1, proof2], [proof3]]

        with patch.object(
            DistributorContract, "get_instance", return_value=mock_contract_instance
        ):
            DistributorContract.claim_rewards(
                mock_ledger_api, MOCK_ADDRESS, users, tokens, amounts, proofs
            )

        call_args = mock_contract_instance.encode_abi.call_args
        args_tuple = call_args[1]["args"] if "args" in call_args[1] else call_args[0][1]
        proofs_converted = args_tuple[3]
        assert len(proofs_converted) == 2
        assert len(proofs_converted[0]) == 2
        assert len(proofs_converted[1]) == 1
        assert proofs_converted[0][0] == bytes.fromhex(proof1[2:])
        assert proofs_converted[0][1] == bytes.fromhex(proof2[2:])
        assert proofs_converted[1][0] == bytes.fromhex(proof3[2:])

    def test_claim_rewards_with_empty_proof_lists(self) -> None:
        """Test that claim_rewards handles empty inner proof lists correctly.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.encode_abi.return_value = "0xee"

        users = ["0xuser1"]
        tokens = ["0xtoken1"]
        amounts = [100]
        proofs: list = [[]]

        with patch.object(
            DistributorContract, "get_instance", return_value=mock_contract_instance
        ):
            DistributorContract.claim_rewards(
                mock_ledger_api, MOCK_ADDRESS, users, tokens, amounts, proofs
            )

        call_args = mock_contract_instance.encode_abi.call_args
        args_tuple = call_args[1]["args"] if "args" in call_args[1] else call_args[0][1]
        proofs_converted = args_tuple[3]
        assert proofs_converted == [[]]


class TestAbiIntegrity:
    """Verify that every function used in contract.py exists in the ABI."""

    ABI_FILE = "Distributor.json"
    EXPECTED_FUNCTIONS = ["claim"]

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

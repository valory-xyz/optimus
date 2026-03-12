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

"""Tests for the VelodromeSugarContract contract."""

from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.velodrome_sugar.contract import VelodromeSugarContract


MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
MOCK_ACCOUNT = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
CHECKSUMMED_ACCOUNT = "0xAbCdEfAbCdEfAbCdEfAbCdEfAbCdEfAbCdEfAbCd"

POSITION_FIELDS = [
    "id",
    "lp",
    "liquidity",
    "staked",
    "amount0",
    "amount1",
    "staked0",
    "staked1",
    "unstaked_earned0",
    "unstaked_earned1",
    "emissions_earned",
    "tick_lower",
    "tick_upper",
    "sqrt_ratio_lower",
    "sqrt_ratio_upper",
    "alm",
]


def _make_position_tuple(base_value: int = 0) -> tuple:
    """Create a mock position tuple with 16 fields.

    :param base_value: base value to use for generating field values.
    :return: a tuple representing a single position.
    """
    return tuple(base_value + i for i in range(16))


class TestPositions:
    """Tests for the positions method."""

    def test_positions_returns_single_position(self) -> None:
        """Test that positions returns correctly parsed position data for a single position.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        position_tuple = _make_position_tuple(100)
        mock_contract_instance.functions.positions.return_value.call.return_value = [
            position_tuple
        ]

        with patch.object(
            VelodromeSugarContract, "get_instance", return_value=mock_contract_instance
        ):
            result = VelodromeSugarContract.positions(
                mock_ledger_api, MOCK_ADDRESS, 10, 0, MOCK_ACCOUNT
            )

        assert "positions" in result
        assert len(result["positions"]) == 1
        pos = result["positions"][0]
        for i, field in enumerate(POSITION_FIELDS):
            assert pos[field] == 100 + i

    def test_positions_returns_multiple_positions(self) -> None:
        """Test that positions correctly parses multiple position entries.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        positions_data = [_make_position_tuple(0), _make_position_tuple(100)]
        mock_contract_instance.functions.positions.return_value.call.return_value = (
            positions_data
        )

        with patch.object(
            VelodromeSugarContract, "get_instance", return_value=mock_contract_instance
        ):
            result = VelodromeSugarContract.positions(
                mock_ledger_api, MOCK_ADDRESS, 100, 0, MOCK_ACCOUNT
            )

        assert len(result["positions"]) == 2
        assert result["positions"][0]["id"] == 0
        assert result["positions"][1]["id"] == 100

    def test_positions_returns_empty_list_when_no_positions(self) -> None:
        """Test that positions returns an empty list when there are no positions.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.positions.return_value.call.return_value = []

        with patch.object(
            VelodromeSugarContract, "get_instance", return_value=mock_contract_instance
        ):
            result = VelodromeSugarContract.positions(
                mock_ledger_api, MOCK_ADDRESS, 10, 0, MOCK_ACCOUNT
            )

        assert result == {"positions": []}

    def test_positions_checksums_account_address(self) -> None:
        """Test that positions checksums the account address before use.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.positions.return_value.call.return_value = []

        with patch.object(
            VelodromeSugarContract, "get_instance", return_value=mock_contract_instance
        ):
            VelodromeSugarContract.positions(
                mock_ledger_api, MOCK_ADDRESS, 10, 0, MOCK_ACCOUNT
            )

        mock_ledger_api.api.to_checksum_address.assert_called_once_with(MOCK_ACCOUNT)

    def test_positions_passes_correct_args_to_contract(self) -> None:
        """Test that positions passes limit, offset, and checksummed account to the contract.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = CHECKSUMMED_ACCOUNT
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.positions.return_value.call.return_value = []

        with patch.object(
            VelodromeSugarContract, "get_instance", return_value=mock_contract_instance
        ):
            VelodromeSugarContract.positions(
                mock_ledger_api, MOCK_ADDRESS, 50, 5, MOCK_ACCOUNT
            )

        mock_contract_instance.functions.positions.assert_called_once_with(
            50, 5, CHECKSUMMED_ACCOUNT
        )

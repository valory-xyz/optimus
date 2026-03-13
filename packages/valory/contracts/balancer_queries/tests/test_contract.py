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

"""Tests for the BalancerQueriesContract."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.balancer_queries.contract import (
    BalancerQueriesContract,
)

MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
MOCK_SENDER = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"
MOCK_RECIPIENT = "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB"
MOCK_POOL_ID = "0x" + "ab" * 32


class TestQueryJoin:
    """Tests for query_join."""

    def test_query_join_returns_bpt_out_and_amounts_in(self) -> None:
        """Test that query_join calls queryJoin view function and returns nested result dict."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.queryJoin.return_value.call.return_value = (
            500,
            [100, 200],
        )

        with patch.object(
            BalancerQueriesContract, "get_instance", return_value=mock_instance
        ):
            result = BalancerQueriesContract.query_join(
                mock_ledger_api,
                MOCK_ADDRESS,
                pool_id=MOCK_POOL_ID,
                sender=MOCK_SENDER,
                recipient=MOCK_RECIPIENT,
                assets=["0xToken0", "0xToken1"],
                max_amounts_in=[1000, 2000],
                join_kind=1,
                minimum_bpt=50,
                from_internal_balance=False,
            )

        assert result == {"result": {"bpt_out": 500, "amounts_in": [100, 200]}}
        mock_instance.functions.queryJoin.assert_called_once()
        call_args = mock_instance.functions.queryJoin.call_args
        # First arg should be pool_id bytes
        assert call_args[0][0] == bytes.fromhex(MOCK_POOL_ID[2:])
        assert call_args[0][1] == MOCK_SENDER
        assert call_args[0][2] == MOCK_RECIPIENT


class TestQueryExit:
    """Tests for query_exit."""

    def test_query_exit_returns_bpt_in_and_amounts_out(self) -> None:
        """Test that query_exit calls queryExit view function and returns nested result dict."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.queryExit.return_value.call.return_value = (
            1000,
            [400, 600],
        )

        with patch.object(
            BalancerQueriesContract, "get_instance", return_value=mock_instance
        ):
            result = BalancerQueriesContract.query_exit(
                mock_ledger_api,
                MOCK_ADDRESS,
                pool_id=MOCK_POOL_ID,
                sender=MOCK_SENDER,
                recipient=MOCK_RECIPIENT,
                assets=["0xToken0", "0xToken1"],
                min_amounts_out=[100, 200],
                exit_kind=1,
                bpt_amount_in=1000,
                to_internal_balance=False,
            )

        assert result == {"result": {"bpt_in": 1000, "amounts_out": [400, 600]}}
        mock_instance.functions.queryExit.assert_called_once()
        call_args = mock_instance.functions.queryExit.call_args
        assert call_args[0][0] == bytes.fromhex(MOCK_POOL_ID[2:])
        assert call_args[0][1] == MOCK_SENDER
        assert call_args[0][2] == MOCK_RECIPIENT


class TestAbiIntegrity:
    """Verify that every function used in contract.py exists in the ABI."""

    ABI_FILE = "BalancerQueries.json"
    EXPECTED_FUNCTIONS = ["queryExit", "queryJoin"]

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

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

    def test_query_join_with_internal_balance(self) -> None:
        """Test query_join with from_internal_balance set to True."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.queryJoin.return_value.call.return_value = (
            300,
            [50, 150],
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
                assets=["0xToken0"],
                max_amounts_in=[500],
                join_kind=0,
                minimum_bpt=10,
                from_internal_balance=True,
            )

        assert result == {"result": {"bpt_out": 300, "amounts_in": [50, 150]}}


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

    def test_query_exit_with_internal_balance(self) -> None:
        """Test query_exit with to_internal_balance set to True."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.queryExit.return_value.call.return_value = (
            800,
            [300, 500],
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
                assets=["0xToken0"],
                min_amounts_out=[50],
                exit_kind=0,
                bpt_amount_in=800,
                to_internal_balance=True,
            )

        assert result == {"result": {"bpt_in": 800, "amounts_out": [300, 500]}}

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

"""Tests for the VelodromeRouterContract."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.velodrome_router.contract import (
    VelodromeRouterContract,
)

MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
MOCK_TOKEN_A = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"
MOCK_TOKEN_B = "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB"
MOCK_FACTORY = "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC"


class TestEncodeCall:
    """Tests for _encode_call."""

    def test_encode_call_returns_tx_hash_dict(self) -> None:
        """Test that _encode_call returns a dict with tx_hash key containing the ABI bytes."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xdeadbeef"

        with patch.object(
            VelodromeRouterContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeRouterContract._encode_call(
                mock_ledger_api, MOCK_ADDRESS, "someMethod", (1, 2)
            )

        assert result == {"tx_hash": bytes.fromhex("deadbeef")}
        mock_instance.encode_abi.assert_called_once_with("someMethod", args=(1, 2))


class TestAddLiquidity:
    """Tests for add_liquidity."""

    def test_add_liquidity_delegates_to_encode_call(self) -> None:
        """Test that add_liquidity delegates to _encode_call with correct args."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xadd11ad1"

        with patch.object(
            VelodromeRouterContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeRouterContract.add_liquidity(
                mock_ledger_api,
                MOCK_ADDRESS,
                token_a=MOCK_TOKEN_A,
                token_b=MOCK_TOKEN_B,
                stable=True,
                amount_a_desired=1000,
                amount_b_desired=2000,
                amount_a_min=900,
                amount_b_min=1800,
                to=MOCK_ADDRESS,
                deadline=9999999,
            )

        assert result == {"tx_hash": bytes.fromhex("add11ad1")}
        mock_instance.encode_abi.assert_called_once_with(
            "addLiquidity",
            args=(
                MOCK_TOKEN_A,
                MOCK_TOKEN_B,
                True,
                1000,
                2000,
                900,
                1800,
                MOCK_ADDRESS,
                9999999,
            ),
        )


class TestRemoveLiquidity:
    """Tests for remove_liquidity."""

    def test_remove_liquidity_delegates_to_encode_call(self) -> None:
        """Test that remove_liquidity delegates to _encode_call with correct args."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0x7e110ace"

        with patch.object(
            VelodromeRouterContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeRouterContract.remove_liquidity(
                mock_ledger_api,
                MOCK_ADDRESS,
                token_a=MOCK_TOKEN_A,
                token_b=MOCK_TOKEN_B,
                stable=False,
                liquidity=5000,
                amount_a_min=400,
                amount_b_min=800,
                to=MOCK_ADDRESS,
                deadline=8888888,
            )

        assert result == {"tx_hash": bytes.fromhex("7e110ace")}
        mock_instance.encode_abi.assert_called_once_with(
            "removeLiquidity",
            args=(
                MOCK_TOKEN_A,
                MOCK_TOKEN_B,
                False,
                5000,
                400,
                800,
                MOCK_ADDRESS,
                8888888,
            ),
        )


class TestQuoteAddLiquidity:
    """Tests for quote_add_liquidity."""

    def test_quote_add_liquidity_returns_result_dict(self) -> None:
        """Test that quote_add_liquidity calls quoteAddLiquidity and returns nested result dict."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.quoteAddLiquidity.return_value.call.return_value = (
            100,
            200,
            300,
        )

        with patch.object(
            VelodromeRouterContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeRouterContract.quote_add_liquidity(
                mock_ledger_api,
                MOCK_ADDRESS,
                token_a=MOCK_TOKEN_A,
                token_b=MOCK_TOKEN_B,
                stable=True,
                factory=MOCK_FACTORY,
                amount_a_desired=1000,
                amount_b_desired=2000,
            )

        assert result == {
            "result": {"amount_a": 100, "amount_b": 200, "liquidity": 300}
        }
        mock_instance.functions.quoteAddLiquidity.assert_called_once_with(
            MOCK_TOKEN_A, MOCK_TOKEN_B, True, MOCK_FACTORY, 1000, 2000
        )


class TestQuoteRemoveLiquidity:
    """Tests for quote_remove_liquidity."""

    def test_quote_remove_liquidity_returns_result_dict(self) -> None:
        """Test that quote_remove_liquidity calls quoteRemoveLiquidity and returns nested result dict."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.quoteRemoveLiquidity.return_value.call.return_value = (
            500,
            600,
        )

        with patch.object(
            VelodromeRouterContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeRouterContract.quote_remove_liquidity(
                mock_ledger_api,
                MOCK_ADDRESS,
                token_a=MOCK_TOKEN_A,
                token_b=MOCK_TOKEN_B,
                stable=False,
                factory=MOCK_FACTORY,
                liquidity=1000,
            )

        assert result == {"result": {"amount_a": 500, "amount_b": 600}}
        mock_instance.functions.quoteRemoveLiquidity.assert_called_once_with(
            MOCK_TOKEN_A, MOCK_TOKEN_B, False, MOCK_FACTORY, 1000
        )


class TestQuoteAddLiquidityMode:
    """Tests for quote_add_liquidity_mode (no factory param)."""

    def test_quote_add_liquidity_mode_returns_result_dict(self) -> None:
        """Test that quote_add_liquidity_mode calls quoteAddLiquidity without factory param."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.quoteAddLiquidity.return_value.call.return_value = (
            110,
            220,
            330,
        )

        with patch.object(
            VelodromeRouterContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeRouterContract.quote_add_liquidity_mode(
                mock_ledger_api,
                MOCK_ADDRESS,
                token_a=MOCK_TOKEN_A,
                token_b=MOCK_TOKEN_B,
                stable=True,
                amount_a_desired=500,
                amount_b_desired=600,
            )

        assert result == {
            "result": {"amount_a": 110, "amount_b": 220, "liquidity": 330}
        }
        mock_instance.functions.quoteAddLiquidity.assert_called_once_with(
            MOCK_TOKEN_A, MOCK_TOKEN_B, True, 500, 600
        )


class TestQuoteRemoveLiquidityMode:
    """Tests for quote_remove_liquidity_mode (no factory param)."""

    def test_quote_remove_liquidity_mode_returns_result_dict(self) -> None:
        """Test that quote_remove_liquidity_mode calls quoteRemoveLiquidity without factory param."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.quoteRemoveLiquidity.return_value.call.return_value = (
            700,
            800,
        )

        with patch.object(
            VelodromeRouterContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeRouterContract.quote_remove_liquidity_mode(
                mock_ledger_api,
                MOCK_ADDRESS,
                token_a=MOCK_TOKEN_A,
                token_b=MOCK_TOKEN_B,
                stable=True,
                liquidity=3000,
            )

        assert result == {"result": {"amount_a": 700, "amount_b": 800}}
        mock_instance.functions.quoteRemoveLiquidity.assert_called_once_with(
            MOCK_TOKEN_A, MOCK_TOKEN_B, True, 3000
        )


class TestFactory:
    """Tests for factory."""

    def test_factory_returns_factory_address(self) -> None:
        """Test that factory calls factory() and returns the factory address."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.factory.return_value.call.return_value = MOCK_FACTORY

        with patch.object(
            VelodromeRouterContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeRouterContract.factory(mock_ledger_api, MOCK_ADDRESS)

        assert result == {"factory": MOCK_FACTORY}
        mock_instance.functions.factory.assert_called_once_with()


class TestDefaultFactory:
    """Tests for defaultFactory."""

    def test_default_factory_returns_factory_address(self) -> None:
        """Test that defaultFactory calls defaultFactory() and returns the factory address."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.defaultFactory.return_value.call.return_value = (
            MOCK_FACTORY
        )

        with patch.object(
            VelodromeRouterContract, "get_instance", return_value=mock_instance
        ):
            result = VelodromeRouterContract.defaultFactory(
                mock_ledger_api, MOCK_ADDRESS
            )

        assert result == {"factory": MOCK_FACTORY}
        mock_instance.functions.defaultFactory.assert_called_once_with()


class TestAbiIntegrity:
    """Verify that every function used in contract.py exists in the ABI."""

    ABI_FILE = "contract_interface.json"
    EXPECTED_FUNCTIONS = [
        "addLiquidity",
        "defaultFactory",
        "factory",
        "quoteAddLiquidity",
        "quoteRemoveLiquidity",
        "removeLiquidity",
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

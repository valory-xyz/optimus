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

"""Tests for the UniswapV3NonfungiblePositionManagerContract."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.uniswap_v3_non_fungible_position_manager.contract import (
    UniswapV3NonfungiblePositionManagerContract,
)

MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
MOCK_RECIPIENT = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"
MOCK_OWNER = "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB"
MOCK_TOKEN0 = "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC"
MOCK_TOKEN1 = "0xDdDDddDdDdddDDddDDddDDDDdDdDDdDDdDDDDDDd"


class TestMint:
    """Tests for mint."""

    def test_mint_returns_tx_hash(self) -> None:
        """Test that mint encodes a mint call and returns dict(tx_hash=tx_hash) as raw string."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xm1ntdata"

        with patch.object(
            UniswapV3NonfungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = UniswapV3NonfungiblePositionManagerContract.mint(
                mock_ledger_api,
                MOCK_ADDRESS,
                token0=MOCK_TOKEN0,
                token1=MOCK_TOKEN1,
                fee=3000,
                tick_lower=-60,
                tick_upper=60,
                amount0_desired=1000,
                amount1_desired=2000,
                amount0_min=900,
                amount1_min=1800,
                recipient=MOCK_RECIPIENT,
                deadline=9999999,
            )

        assert result == {"tx_hash": "0xm1ntdata"}
        mock_instance.encode_abi.assert_called_once_with(
            "mint",
            args=(
                (
                    MOCK_TOKEN0,
                    MOCK_TOKEN1,
                    3000,
                    -60,
                    60,
                    1000,
                    2000,
                    900,
                    1800,
                    MOCK_RECIPIENT,
                    9999999,
                ),
            ),
        )


class TestDecreaseLiquidity:
    """Tests for decrease_liquidity."""

    def test_decrease_liquidity_returns_tx_hash(self) -> None:
        """Test that decrease_liquidity encodes a decreaseLiquidity call and returns tx_hash."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xdecr"

        with patch.object(
            UniswapV3NonfungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = UniswapV3NonfungiblePositionManagerContract.decrease_liquidity(
                mock_ledger_api,
                MOCK_ADDRESS,
                token_id=42,
                liquidity=5000,
                amount0_min=100,
                amount1_min=200,
                deadline=8888888,
            )

        assert result == {"tx_hash": "0xdecr"}
        mock_instance.encode_abi.assert_called_once_with(
            "decreaseLiquidity",
            args=((42, 5000, 100, 200, 8888888),),
        )


class TestBurnToken:
    """Tests for burn_token."""

    def test_burn_token_returns_tx_hash(self) -> None:
        """Test that burn_token encodes a burn call and returns tx_hash."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xburnt0ken"

        with patch.object(
            UniswapV3NonfungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = UniswapV3NonfungiblePositionManagerContract.burn_token(
                mock_ledger_api,
                MOCK_ADDRESS,
                token_id=99,
            )

        assert result == {"tx_hash": "0xburnt0ken"}
        mock_instance.encode_abi.assert_called_once_with("burn", args=(99,))


class TestCollectTokens:
    """Tests for collect_tokens."""

    def test_collect_tokens_returns_tx_hash(self) -> None:
        """Test that collect_tokens encodes a collect call and returns tx_hash."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xc011ect"

        with patch.object(
            UniswapV3NonfungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = UniswapV3NonfungiblePositionManagerContract.collect_tokens(
                mock_ledger_api,
                MOCK_ADDRESS,
                token_id=7,
                recipient=MOCK_RECIPIENT,
                amount0_max=10000,
                amount1_max=20000,
            )

        assert result == {"tx_hash": "0xc011ect"}
        mock_instance.encode_abi.assert_called_once_with(
            "collect",
            args=((7, MOCK_RECIPIENT, 10000, 20000),),
        )


class TestGetPoolTokens:
    """Tests for get_pool_tokens."""

    def test_get_pool_tokens_returns_token_list(self) -> None:
        """Test that get_pool_tokens calls token0() and token1() and returns dict(tokens=[...])."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.token0.return_value.call.return_value = MOCK_TOKEN0
        mock_instance.functions.token1.return_value.call.return_value = MOCK_TOKEN1

        with patch.object(
            UniswapV3NonfungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = UniswapV3NonfungiblePositionManagerContract.get_pool_tokens(
                mock_ledger_api, MOCK_ADDRESS
            )

        assert result == {"tokens": [MOCK_TOKEN0, MOCK_TOKEN1]}


class TestBalanceOf:
    """Tests for balanceOf."""

    def test_balance_of_returns_balance(self) -> None:
        """Test that balanceOf calls balanceOf on the contract and returns the balance."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.balanceOf.return_value.call.return_value = 5

        with patch.object(
            UniswapV3NonfungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = UniswapV3NonfungiblePositionManagerContract.balanceOf(
                mock_ledger_api, MOCK_ADDRESS, owner=MOCK_OWNER
            )

        assert result == {"balance": 5}
        mock_instance.functions.balanceOf.assert_called_once_with(MOCK_OWNER)


class TestOwnerOf:
    """Tests for ownerOf."""

    def test_owner_of_returns_owner(self) -> None:
        """Test that ownerOf calls ownerOf on the contract and returns the owner address."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.ownerOf.return_value.call.return_value = MOCK_OWNER

        with patch.object(
            UniswapV3NonfungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = UniswapV3NonfungiblePositionManagerContract.ownerOf(
                mock_ledger_api, MOCK_ADDRESS, token_id=42
            )

        assert result == {"owner": MOCK_OWNER}
        mock_instance.functions.ownerOf.assert_called_once_with(42)


class TestGetPosition:
    """Tests for get_position."""

    def test_get_position_returns_12_field_dict(self) -> None:
        """Test that get_position returns a dict with 12 position fields when position exists."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        position_data = (
            0,  # nonce
            MOCK_OWNER,  # operator
            MOCK_TOKEN0,  # token0
            MOCK_TOKEN1,  # token1
            3000,  # fee
            -60,  # tickLower
            60,  # tickUpper
            1000000,  # liquidity
            100,  # feeGrowthInside0LastX128
            200,  # feeGrowthInside1LastX128
            50,  # tokensOwed0
            75,  # tokensOwed1
        )
        mock_instance.functions.positions.return_value.call.return_value = position_data

        with patch.object(
            UniswapV3NonfungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = UniswapV3NonfungiblePositionManagerContract.get_position(
                mock_ledger_api, MOCK_ADDRESS, token_id=42
            )

        assert result == {
            "data": {
                "nonce": 0,
                "operator": MOCK_OWNER,
                "token0": MOCK_TOKEN0,
                "token1": MOCK_TOKEN1,
                "fee": 3000,
                "tickLower": -60,
                "tickUpper": 60,
                "liquidity": 1000000,
                "feeGrowthInside0LastX128": 100,
                "feeGrowthInside1LastX128": 200,
                "tokensOwed0": 50,
                "tokensOwed1": 75,
            }
        }

    def test_get_position_returns_empty_dict_when_no_position(self) -> None:
        """Test that get_position returns dict(data={}) when position is falsy."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        # Empty list or None are falsy
        mock_instance.functions.positions.return_value.call.return_value = []

        with patch.object(
            UniswapV3NonfungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = UniswapV3NonfungiblePositionManagerContract.get_position(
                mock_ledger_api, MOCK_ADDRESS, token_id=999
            )

        assert result == {"data": {}}


class TestAbiIntegrity:
    """Verify that every function used in contract.py exists in the ABI."""

    ABI_FILE = "NonfungiblePositionManager.json"
    EXPECTED_FUNCTIONS = [
        "balanceOf",
        "burn",
        "collect",
        "decreaseLiquidity",
        "mint",
        "ownerOf",
        "positions",
        "token0",
        "token1",
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

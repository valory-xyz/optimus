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

"""Tests for the VelodromeNonFungiblePositionManagerContract."""

from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.velodrome_non_fungible_position_manager.contract import (
    VelodromeNonFungiblePositionManagerContract,
)

MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
MOCK_RECIPIENT = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"
MOCK_OWNER = "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB"
MOCK_OPERATOR = "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC"
MOCK_TOKEN0 = "0xDdDDddDdDdddDDddDDddDDDDdDdDDdDDdDDDDDDd"
MOCK_TOKEN1 = "0xEeEeEeEeEeEeEeEeEeEeEeEeEeEeEeEeEeEeEeEe"


class TestEncodeCall:
    """Tests for _encode_call."""

    def test_encode_call_returns_tx_hash_dict(self) -> None:
        """Test that _encode_call returns a dict with tx_hash key containing raw ABI data."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xdeadbeef"

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract._encode_call(
                mock_ledger_api, MOCK_ADDRESS, "someMethod", (1, 2)
            )

        assert result == {"tx_hash": "0xdeadbeef"}
        mock_instance.encode_abi.assert_called_once_with(
            "someMethod", args=(1, 2)
        )


class TestMint:
    """Tests for mint."""

    def test_mint_delegates_to_encode_call(self) -> None:
        """Test that mint delegates to _encode_call with correct tuple-wrapped params."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xm1nt"

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.mint(
                mock_ledger_api,
                MOCK_ADDRESS,
                token0=MOCK_TOKEN0,
                token1=MOCK_TOKEN1,
                tick_spacing=60,
                tick_lower=-120,
                tick_upper=120,
                amount0_desired=1000,
                amount1_desired=2000,
                amount0_min=900,
                amount1_min=1800,
                recipient=MOCK_RECIPIENT,
                deadline=9999999,
                sqrt_price_x96=79228162514264337593543950336,
            )

        assert result == {"tx_hash": "0xm1nt"}
        mock_instance.encode_abi.assert_called_once_with(
            "mint",
            args=(
                (
                    MOCK_TOKEN0,
                    MOCK_TOKEN1,
                    60,
                    -120,
                    120,
                    1000,
                    2000,
                    900,
                    1800,
                    MOCK_RECIPIENT,
                    9999999,
                    79228162514264337593543950336,
                ),
            ),
        )


class TestDecreaseLiquidity:
    """Tests for decrease_liquidity."""

    def test_decrease_liquidity_delegates_to_encode_call(self) -> None:
        """Test that decrease_liquidity delegates to _encode_call with correct params tuple."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xdecr"

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.decrease_liquidity(
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


class TestBurn:
    """Tests for burn."""

    def test_burn_delegates_to_encode_call(self) -> None:
        """Test that burn delegates to _encode_call with token_id."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xburn"

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.burn(
                mock_ledger_api,
                MOCK_ADDRESS,
                token_id=99,
            )

        assert result == {"tx_hash": "0xburn"}
        mock_instance.encode_abi.assert_called_once_with(
            "burn", args=(99,)
        )


class TestCollect:
    """Tests for collect."""

    def test_collect_delegates_to_encode_call(self) -> None:
        """Test that collect delegates to _encode_call with correct params tuple."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xc011ect"

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.collect(
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

    def test_get_pool_tokens_returns_position_info(self) -> None:
        """Test that get_pool_tokens calls positions() and returns token/tick/liquidity info."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        position_data = (
            0,  # nonce
            MOCK_OPERATOR,  # operator
            MOCK_TOKEN0,  # token0
            MOCK_TOKEN1,  # token1
            60,  # tick_spacing
            -120,  # tick_lower
            120,  # tick_upper
            1000000,  # liquidity
            100,  # feeGrowthInside0LastX128
            200,  # feeGrowthInside1LastX128
            50,  # tokensOwed0
            75,  # tokensOwed1
        )
        mock_instance.functions.positions.return_value.call.return_value = position_data

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.get_pool_tokens(
                mock_ledger_api, MOCK_ADDRESS, token_id=42
            )

        assert result == {
            "token0": MOCK_TOKEN0,
            "token1": MOCK_TOKEN1,
            "tick_spacing": 60,
            "tick_lower": -120,
            "tick_upper": 120,
            "liquidity": 1000000,
        }


class TestBalanceOf:
    """Tests for balanceOf."""

    def test_balance_of_returns_balance(self) -> None:
        """Test that balanceOf calls balanceOf on the contract and returns the balance."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.balanceOf.return_value.call.return_value = 3

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.balanceOf(
                mock_ledger_api, MOCK_ADDRESS, owner=MOCK_OWNER
            )

        assert result == {"balance": 3}
        mock_instance.functions.balanceOf.assert_called_once_with(MOCK_OWNER)


class TestOwnerOf:
    """Tests for ownerOf."""

    def test_owner_of_returns_owner_address(self) -> None:
        """Test that ownerOf calls ownerOf on the contract and returns the owner address."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.ownerOf.return_value.call.return_value = MOCK_OWNER

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.ownerOf(
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
            MOCK_OPERATOR,  # operator
            MOCK_TOKEN0,  # token0
            MOCK_TOKEN1,  # token1
            60,  # tickSpacing
            -120,  # tickLower
            120,  # tickUpper
            1000000,  # liquidity
            100,  # feeGrowthInside0LastX128
            200,  # feeGrowthInside1LastX128
            50,  # tokensOwed0
            75,  # tokensOwed1
        )
        mock_instance.functions.positions.return_value.call.return_value = position_data

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.get_position(
                mock_ledger_api, MOCK_ADDRESS, token_id=42
            )

        assert result == {
            "data": {
                "nonce": 0,
                "operator": MOCK_OPERATOR,
                "token0": MOCK_TOKEN0,
                "token1": MOCK_TOKEN1,
                "tickSpacing": 60,
                "tickLower": -120,
                "tickUpper": 120,
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
        mock_instance.functions.positions.return_value.call.return_value = []

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.get_position(
                mock_ledger_api, MOCK_ADDRESS, token_id=999
            )

        assert result == {"data": {}}


class TestApprove:
    """Tests for approve."""

    def test_approve_delegates_to_encode_call(self) -> None:
        """Test that approve delegates to _encode_call with (to, token_id) args."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xappr0ve"

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.approve(
                mock_ledger_api,
                MOCK_ADDRESS,
                to=MOCK_OPERATOR,
                token_id=42,
            )

        assert result == {"tx_hash": "0xappr0ve"}
        mock_instance.encode_abi.assert_called_once_with(
            "approve", args=(MOCK_OPERATOR, 42)
        )


class TestSetApprovalForAll:
    """Tests for set_approval_for_all."""

    def test_set_approval_for_all_delegates_to_encode_call(self) -> None:
        """Test that set_approval_for_all encodes setApprovalForAll with correct args."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xsetall"

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.set_approval_for_all(
                mock_ledger_api,
                MOCK_ADDRESS,
                operator=MOCK_OPERATOR,
                approved=True,
            )

        assert result == {"tx_hash": "0xsetall"}
        mock_instance.encode_abi.assert_called_once_with(
            "setApprovalForAll", args=(MOCK_OPERATOR, True)
        )

    def test_set_approval_for_all_revoke(self) -> None:
        """Test set_approval_for_all with approved=False to revoke approval."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.encode_abi.return_value = "0xrevoke"

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.set_approval_for_all(
                mock_ledger_api,
                MOCK_ADDRESS,
                operator=MOCK_OPERATOR,
                approved=False,
            )

        assert result == {"tx_hash": "0xrevoke"}
        mock_instance.encode_abi.assert_called_once_with(
            "setApprovalForAll", args=(MOCK_OPERATOR, False)
        )


class TestIsApprovedForAll:
    """Tests for is_approved_for_all."""

    def test_is_approved_for_all_returns_true(self) -> None:
        """Test that is_approved_for_all returns dict(is_approved=True) when approved."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.isApprovedForAll.return_value.call.return_value = True

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.is_approved_for_all(
                mock_ledger_api,
                MOCK_ADDRESS,
                owner=MOCK_OWNER,
                operator=MOCK_OPERATOR,
            )

        assert result == {"is_approved": True}
        mock_instance.functions.isApprovedForAll.assert_called_once_with(
            MOCK_OWNER, MOCK_OPERATOR
        )

    def test_is_approved_for_all_returns_false(self) -> None:
        """Test that is_approved_for_all returns dict(is_approved=False) when not approved."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.isApprovedForAll.return_value.call.return_value = False

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.is_approved_for_all(
                mock_ledger_api,
                MOCK_ADDRESS,
                owner=MOCK_OWNER,
                operator=MOCK_OPERATOR,
            )

        assert result == {"is_approved": False}


class TestGetApproved:
    """Tests for get_approved."""

    def test_get_approved_returns_approved_address(self) -> None:
        """Test that get_approved calls getApproved and returns the approved address."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.getApproved.return_value.call.return_value = (
            MOCK_OPERATOR
        )

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.get_approved(
                mock_ledger_api, MOCK_ADDRESS, token_id=42
            )

        assert result == {"approved": MOCK_OPERATOR}
        mock_instance.functions.getApproved.assert_called_once_with(42)


class TestTokenOfOwnerByIndex:
    """Tests for token_of_owner_by_index."""

    def test_token_of_owner_by_index_returns_token_id(self) -> None:
        """Test that token_of_owner_by_index calls tokenOfOwnerByIndex and returns token_id."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.tokenOfOwnerByIndex.return_value.call.return_value = 77

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.token_of_owner_by_index(
                mock_ledger_api, MOCK_ADDRESS, owner=MOCK_OWNER, index=0
            )

        assert result == {"token_id": 77}
        mock_instance.functions.tokenOfOwnerByIndex.assert_called_once_with(
            MOCK_OWNER, 0
        )


class TestGetAllTokenIdsForOwner:
    """Tests for get_all_token_ids_for_owner."""

    def test_get_all_token_ids_returns_all_ids(self) -> None:
        """Test that get_all_token_ids_for_owner iterates balance and collects token IDs."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.balanceOf.return_value.call.return_value = 3
        mock_instance.functions.tokenOfOwnerByIndex.return_value.call.side_effect = [
            10,
            20,
            30,
        ]

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.get_all_token_ids_for_owner(
                mock_ledger_api, MOCK_ADDRESS, owner=MOCK_OWNER
            )

        assert result == {"token_ids": [10, 20, 30], "count": 3}

    def test_get_all_token_ids_with_zero_balance(self) -> None:
        """Test that get_all_token_ids_for_owner returns empty list for zero balance."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.balanceOf.return_value.call.return_value = 0

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.get_all_token_ids_for_owner(
                mock_ledger_api, MOCK_ADDRESS, owner=MOCK_OWNER
            )

        assert result == {"token_ids": [], "count": 0}

    def test_get_all_token_ids_skips_failed_lookups(self) -> None:
        """Test that get_all_token_ids_for_owner continues on exception in the loop."""
        mock_ledger_api = MagicMock()
        mock_instance = MagicMock()
        mock_instance.functions.balanceOf.return_value.call.return_value = 3

        # First call succeeds, second raises, third succeeds
        call_mock = MagicMock()
        call_mock.call.side_effect = [10, Exception("reverted"), 30]
        mock_instance.functions.tokenOfOwnerByIndex.return_value = call_mock

        with patch.object(
            VelodromeNonFungiblePositionManagerContract,
            "get_instance",
            return_value=mock_instance,
        ):
            result = VelodromeNonFungiblePositionManagerContract.get_all_token_ids_for_owner(
                mock_ledger_api, MOCK_ADDRESS, owner=MOCK_OWNER
            )

        assert result == {"token_ids": [10, 30], "count": 2}

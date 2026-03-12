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

"""Tests for the VelodromeSlipstreamHelperContract contract."""

from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.velodrome_slipstream_helper.contract import (
    VelodromeSlipstreamHelperContract,
)


MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
MOCK_POSITION_MANAGER = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
MOCK_TOKEN_ID = 42
MOCK_SQRT_PRICE_X96 = 79228162514264337593543950336
MOCK_SQRT_RATIO_A_X96 = 4295128739
MOCK_SQRT_RATIO_B_X96 = 1461446703485210103287273052203988822378723970342
MOCK_LIQUIDITY = 1000000
MOCK_LIQUIDITY_DELTA = 500000
MOCK_TICK = -887272


class TestPrincipal:
    """Tests for the principal method."""

    def test_principal_returns_amounts(self) -> None:
        """Test that principal returns a dict with the amounts key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.principal.return_value.call.return_value = (
            100,
            200,
        )

        with patch.object(
            VelodromeSlipstreamHelperContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeSlipstreamHelperContract.principal(
                mock_ledger_api,
                MOCK_ADDRESS,
                MOCK_POSITION_MANAGER,
                MOCK_TOKEN_ID,
                MOCK_SQRT_PRICE_X96,
            )

        assert result == {"amounts": (100, 200)}

    def test_principal_calls_contract_function_with_correct_args(self) -> None:
        """Test that principal calls the contract function with the correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.principal.return_value.call.return_value = (
            0,
            0,
        )

        with patch.object(
            VelodromeSlipstreamHelperContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeSlipstreamHelperContract.principal(
                mock_ledger_api,
                MOCK_ADDRESS,
                MOCK_POSITION_MANAGER,
                MOCK_TOKEN_ID,
                MOCK_SQRT_PRICE_X96,
            )

        mock_contract_instance.functions.principal.assert_called_once_with(
            MOCK_POSITION_MANAGER, MOCK_TOKEN_ID, MOCK_SQRT_PRICE_X96
        )


class TestGetAmountsForLiquidity:
    """Tests for the get_amounts_for_liquidity method."""

    def test_get_amounts_for_liquidity_returns_amounts(self) -> None:
        """Test that get_amounts_for_liquidity returns a dict with the amounts key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.getAmountsForLiquidity.return_value.call.return_value = (
            300,
            400,
        )

        with patch.object(
            VelodromeSlipstreamHelperContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeSlipstreamHelperContract.get_amounts_for_liquidity(
                mock_ledger_api,
                MOCK_ADDRESS,
                MOCK_SQRT_PRICE_X96,
                MOCK_SQRT_RATIO_A_X96,
                MOCK_SQRT_RATIO_B_X96,
                MOCK_LIQUIDITY,
            )

        assert result == {"amounts": (300, 400)}

    def test_get_amounts_for_liquidity_calls_contract_function_with_correct_args(
        self,
    ) -> None:
        """Test that get_amounts_for_liquidity calls the contract function with the correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.getAmountsForLiquidity.return_value.call.return_value = (
            0,
            0,
        )

        with patch.object(
            VelodromeSlipstreamHelperContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeSlipstreamHelperContract.get_amounts_for_liquidity(
                mock_ledger_api,
                MOCK_ADDRESS,
                MOCK_SQRT_PRICE_X96,
                MOCK_SQRT_RATIO_A_X96,
                MOCK_SQRT_RATIO_B_X96,
                MOCK_LIQUIDITY,
            )

        mock_contract_instance.functions.getAmountsForLiquidity.assert_called_once_with(
            MOCK_SQRT_PRICE_X96,
            MOCK_SQRT_RATIO_A_X96,
            MOCK_SQRT_RATIO_B_X96,
            MOCK_LIQUIDITY,
        )


class TestGetSqrtRatioAtTick:
    """Tests for the get_sqrt_ratio_at_tick method."""

    def test_get_sqrt_ratio_at_tick_returns_sqrt_ratio(self) -> None:
        """Test that get_sqrt_ratio_at_tick returns a dict with the sqrt_ratio key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        expected_ratio = 79228162514264337593543950336
        mock_contract_instance.functions.getSqrtRatioAtTick.return_value.call.return_value = (
            expected_ratio
        )

        with patch.object(
            VelodromeSlipstreamHelperContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeSlipstreamHelperContract.get_sqrt_ratio_at_tick(
                mock_ledger_api, MOCK_ADDRESS, MOCK_TICK
            )

        assert result == {"sqrt_ratio": expected_ratio}

    def test_get_sqrt_ratio_at_tick_calls_contract_function_with_correct_args(
        self,
    ) -> None:
        """Test that get_sqrt_ratio_at_tick calls the contract function with the correct tick.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.getSqrtRatioAtTick.return_value.call.return_value = (
            0
        )

        with patch.object(
            VelodromeSlipstreamHelperContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeSlipstreamHelperContract.get_sqrt_ratio_at_tick(
                mock_ledger_api, MOCK_ADDRESS, MOCK_TICK
            )

        mock_contract_instance.functions.getSqrtRatioAtTick.assert_called_once_with(
            MOCK_TICK
        )


class TestGetAmount0Delta:
    """Tests for the get_amount0_delta method."""

    def test_get_amount0_delta_returns_delta(self) -> None:
        """Test that get_amount0_delta returns a dict with the amount0_delta key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.getAmount0Delta.return_value.call.return_value = (
            12345
        )

        with patch.object(
            VelodromeSlipstreamHelperContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeSlipstreamHelperContract.get_amount0_delta(
                mock_ledger_api,
                MOCK_ADDRESS,
                MOCK_SQRT_RATIO_A_X96,
                MOCK_SQRT_RATIO_B_X96,
                MOCK_LIQUIDITY_DELTA,
            )

        assert result == {"amount0_delta": 12345}

    def test_get_amount0_delta_calls_contract_function_with_correct_args(
        self,
    ) -> None:
        """Test that get_amount0_delta calls the contract function with the correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.getAmount0Delta.return_value.call.return_value = (
            0
        )

        with patch.object(
            VelodromeSlipstreamHelperContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeSlipstreamHelperContract.get_amount0_delta(
                mock_ledger_api,
                MOCK_ADDRESS,
                MOCK_SQRT_RATIO_A_X96,
                MOCK_SQRT_RATIO_B_X96,
                MOCK_LIQUIDITY_DELTA,
            )

        mock_contract_instance.functions.getAmount0Delta.assert_called_once_with(
            MOCK_SQRT_RATIO_A_X96, MOCK_SQRT_RATIO_B_X96, MOCK_LIQUIDITY_DELTA
        )


class TestGetAmount1Delta:
    """Tests for the get_amount1_delta method."""

    def test_get_amount1_delta_returns_delta(self) -> None:
        """Test that get_amount1_delta returns a dict with the amount1_delta key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.getAmount1Delta.return_value.call.return_value = (
            67890
        )

        with patch.object(
            VelodromeSlipstreamHelperContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeSlipstreamHelperContract.get_amount1_delta(
                mock_ledger_api,
                MOCK_ADDRESS,
                MOCK_SQRT_RATIO_A_X96,
                MOCK_SQRT_RATIO_B_X96,
                MOCK_LIQUIDITY_DELTA,
            )

        assert result == {"amount1_delta": 67890}

    def test_get_amount1_delta_calls_contract_function_with_correct_args(
        self,
    ) -> None:
        """Test that get_amount1_delta calls the contract function with the correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.getAmount1Delta.return_value.call.return_value = (
            0
        )

        with patch.object(
            VelodromeSlipstreamHelperContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeSlipstreamHelperContract.get_amount1_delta(
                mock_ledger_api,
                MOCK_ADDRESS,
                MOCK_SQRT_RATIO_A_X96,
                MOCK_SQRT_RATIO_B_X96,
                MOCK_LIQUIDITY_DELTA,
            )

        mock_contract_instance.functions.getAmount1Delta.assert_called_once_with(
            MOCK_SQRT_RATIO_A_X96, MOCK_SQRT_RATIO_B_X96, MOCK_LIQUIDITY_DELTA
        )

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

"""Tests for the VelodromeVoterContract contract."""

from unittest.mock import MagicMock, patch

import pytest

from packages.valory.contracts.velodrome_voter.contract import VelodromeVoterContract


MOCK_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
MOCK_POOL_ADDRESS = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
MOCK_GAUGE_ADDRESS = "0x00000000000000000000000000000000DeaDBeef"
MOCK_CHECKSUMMED_POOL = "0xAbCdEfAbCdEfAbCdEfAbCdEfAbCdEfAbCdEfAbCd"
MOCK_CHECKSUMMED_GAUGE = "0x00000000000000000000000000000000DeaDBeef"


class TestGauges:
    """Tests for the gauges method."""

    def test_gauges_returns_gauge_address(self) -> None:
        """Test that gauges returns a dict with the gauge key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_POOL
        mock_contract_instance = MagicMock()
        expected_gauge = "0x9999999999999999999999999999999999999999"
        mock_contract_instance.functions.gauges.return_value.call.return_value = (
            expected_gauge
        )

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeVoterContract.gauges(
                mock_ledger_api, MOCK_ADDRESS, MOCK_POOL_ADDRESS
            )

        assert result == {"gauge": expected_gauge}

    def test_gauges_checksums_pool_address(self) -> None:
        """Test that gauges checksums the pool address before querying.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_POOL
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.gauges.return_value.call.return_value = (
            MOCK_GAUGE_ADDRESS
        )

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeVoterContract.gauges(
                mock_ledger_api, MOCK_ADDRESS, MOCK_POOL_ADDRESS
            )

        mock_ledger_api.api.to_checksum_address.assert_called_once_with(
            MOCK_POOL_ADDRESS
        )

    def test_gauges_calls_contract_function_with_checksummed_pool(self) -> None:
        """Test that gauges calls the contract function with the checksummed pool address.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_POOL
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.gauges.return_value.call.return_value = (
            MOCK_GAUGE_ADDRESS
        )

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeVoterContract.gauges(
                mock_ledger_api, MOCK_ADDRESS, MOCK_POOL_ADDRESS
            )

        mock_contract_instance.functions.gauges.assert_called_once_with(
            MOCK_CHECKSUMMED_POOL
        )

    def test_gauges_calls_get_instance(self) -> None:
        """Test that gauges calls get_instance with correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_POOL
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.gauges.return_value.call.return_value = (
            MOCK_GAUGE_ADDRESS
        )

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ) as mock_get_instance:
            VelodromeVoterContract.gauges(
                mock_ledger_api, MOCK_ADDRESS, MOCK_POOL_ADDRESS
            )

        mock_get_instance.assert_called_once_with(mock_ledger_api, MOCK_ADDRESS)


class TestIsGauge:
    """Tests for the is_gauge method."""

    def test_is_gauge_returns_true(self) -> None:
        """Test that is_gauge returns is_gauge=True when address is a valid gauge.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_GAUGE
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.isGauge.return_value.call.return_value = True

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeVoterContract.is_gauge(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        assert result == {"is_gauge": True}

    def test_is_gauge_returns_false(self) -> None:
        """Test that is_gauge returns is_gauge=False when address is not a valid gauge.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_GAUGE
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.isGauge.return_value.call.return_value = False

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeVoterContract.is_gauge(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        assert result == {"is_gauge": False}

    def test_is_gauge_checksums_gauge_address(self) -> None:
        """Test that is_gauge checksums the gauge address before querying.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_GAUGE
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.isGauge.return_value.call.return_value = True

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeVoterContract.is_gauge(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        mock_ledger_api.api.to_checksum_address.assert_called_once_with(
            MOCK_GAUGE_ADDRESS
        )

    def test_is_gauge_calls_contract_function_with_checksummed_gauge(self) -> None:
        """Test that is_gauge calls the contract function with the checksummed gauge address.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_GAUGE
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.isGauge.return_value.call.return_value = True

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeVoterContract.is_gauge(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        mock_contract_instance.functions.isGauge.assert_called_once_with(
            MOCK_CHECKSUMMED_GAUGE
        )

    def test_is_gauge_calls_get_instance(self) -> None:
        """Test that is_gauge calls get_instance with correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_GAUGE
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.isGauge.return_value.call.return_value = True

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ) as mock_get_instance:
            VelodromeVoterContract.is_gauge(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        mock_get_instance.assert_called_once_with(mock_ledger_api, MOCK_ADDRESS)


class TestPoolForGauge:
    """Tests for the pool_for_gauge method."""

    def test_pool_for_gauge_returns_pool_address(self) -> None:
        """Test that pool_for_gauge returns a dict with the pool key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_GAUGE
        mock_contract_instance = MagicMock()
        expected_pool = "0x1111111111111111111111111111111111111111"
        mock_contract_instance.functions.poolForGauge.return_value.call.return_value = (
            expected_pool
        )

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeVoterContract.pool_for_gauge(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        assert result == {"pool": expected_pool}

    def test_pool_for_gauge_checksums_gauge_address(self) -> None:
        """Test that pool_for_gauge checksums the gauge address before querying.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_GAUGE
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.poolForGauge.return_value.call.return_value = (
            MOCK_POOL_ADDRESS
        )

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeVoterContract.pool_for_gauge(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        mock_ledger_api.api.to_checksum_address.assert_called_once_with(
            MOCK_GAUGE_ADDRESS
        )

    def test_pool_for_gauge_calls_contract_function_with_checksummed_gauge(
        self,
    ) -> None:
        """Test that pool_for_gauge calls the contract function with the checksummed gauge address.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_GAUGE
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.poolForGauge.return_value.call.return_value = (
            MOCK_POOL_ADDRESS
        )

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeVoterContract.pool_for_gauge(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        mock_contract_instance.functions.poolForGauge.assert_called_once_with(
            MOCK_CHECKSUMMED_GAUGE
        )

    def test_pool_for_gauge_calls_get_instance(self) -> None:
        """Test that pool_for_gauge calls get_instance with correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_GAUGE
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.poolForGauge.return_value.call.return_value = (
            MOCK_POOL_ADDRESS
        )

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ) as mock_get_instance:
            VelodromeVoterContract.pool_for_gauge(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        mock_get_instance.assert_called_once_with(mock_ledger_api, MOCK_ADDRESS)


class TestIsAlive:
    """Tests for the is_alive method."""

    def test_is_alive_returns_true(self) -> None:
        """Test that is_alive returns is_alive=True when gauge is active.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_GAUGE
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.isAlive.return_value.call.return_value = True

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeVoterContract.is_alive(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        assert result == {"is_alive": True}

    def test_is_alive_returns_false(self) -> None:
        """Test that is_alive returns is_alive=False when gauge is not active.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_GAUGE
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.isAlive.return_value.call.return_value = False

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeVoterContract.is_alive(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        assert result == {"is_alive": False}

    def test_is_alive_checksums_gauge_address(self) -> None:
        """Test that is_alive checksums the gauge address before querying.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_GAUGE
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.isAlive.return_value.call.return_value = True

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeVoterContract.is_alive(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        mock_ledger_api.api.to_checksum_address.assert_called_once_with(
            MOCK_GAUGE_ADDRESS
        )

    def test_is_alive_calls_contract_function_with_checksummed_gauge(self) -> None:
        """Test that is_alive calls the contract function with the checksummed gauge address.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_GAUGE
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.isAlive.return_value.call.return_value = True

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeVoterContract.is_alive(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        mock_contract_instance.functions.isAlive.assert_called_once_with(
            MOCK_CHECKSUMMED_GAUGE
        )

    def test_is_alive_calls_get_instance(self) -> None:
        """Test that is_alive calls get_instance with correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_ledger_api.api.to_checksum_address.return_value = MOCK_CHECKSUMMED_GAUGE
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.isAlive.return_value.call.return_value = True

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ) as mock_get_instance:
            VelodromeVoterContract.is_alive(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        mock_get_instance.assert_called_once_with(mock_ledger_api, MOCK_ADDRESS)


class TestValidateGaugeAddress:
    """Tests for the validate_gauge_address method."""

    def test_validate_gauge_address_valid_and_alive(self) -> None:
        """Test that validate_gauge_address returns is_valid=True when gauge is valid and alive.

        :return: None
        """
        mock_ledger_api = MagicMock()

        with patch.object(
            VelodromeVoterContract,
            "is_gauge",
            return_value={"is_gauge": True},
        ), patch.object(
            VelodromeVoterContract,
            "is_alive",
            return_value={"is_alive": True},
        ):
            result = VelodromeVoterContract.validate_gauge_address(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        assert result == {"is_valid": True, "is_gauge": True, "is_alive": True}

    def test_validate_gauge_address_valid_but_not_alive(self) -> None:
        """Test that validate_gauge_address returns is_valid=False when gauge is valid but not alive.

        :return: None
        """
        mock_ledger_api = MagicMock()

        with patch.object(
            VelodromeVoterContract,
            "is_gauge",
            return_value={"is_gauge": True},
        ), patch.object(
            VelodromeVoterContract,
            "is_alive",
            return_value={"is_alive": False},
        ):
            result = VelodromeVoterContract.validate_gauge_address(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        assert result == {"is_valid": False, "is_gauge": True, "is_alive": False}

    def test_validate_gauge_address_not_a_gauge(self) -> None:
        """Test that validate_gauge_address returns error when address is not a gauge.

        :return: None
        """
        mock_ledger_api = MagicMock()

        with patch.object(
            VelodromeVoterContract,
            "is_gauge",
            return_value={"is_gauge": False},
        ):
            result = VelodromeVoterContract.validate_gauge_address(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        assert "error" in result
        assert result["is_valid"] is False

    def test_validate_gauge_address_is_gauge_returns_error(self) -> None:
        """Test that validate_gauge_address propagates error from is_gauge.

        :return: None
        """
        mock_ledger_api = MagicMock()
        error_result = {"error": "Some contract error"}

        with patch.object(
            VelodromeVoterContract,
            "is_gauge",
            return_value=error_result,
        ):
            result = VelodromeVoterContract.validate_gauge_address(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        assert result == error_result

    def test_validate_gauge_address_is_alive_returns_error(self) -> None:
        """Test that validate_gauge_address propagates error from is_alive.

        :return: None
        """
        mock_ledger_api = MagicMock()
        error_result = {"error": "is_alive contract error"}

        with patch.object(
            VelodromeVoterContract,
            "is_gauge",
            return_value={"is_gauge": True},
        ), patch.object(
            VelodromeVoterContract,
            "is_alive",
            return_value=error_result,
        ):
            result = VelodromeVoterContract.validate_gauge_address(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        assert result == error_result

    def test_validate_gauge_address_calls_is_gauge(self) -> None:
        """Test that validate_gauge_address calls is_gauge with correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()

        with patch.object(
            VelodromeVoterContract,
            "is_gauge",
            return_value={"is_gauge": True},
        ) as mock_is_gauge, patch.object(
            VelodromeVoterContract,
            "is_alive",
            return_value={"is_alive": True},
        ):
            VelodromeVoterContract.validate_gauge_address(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        mock_is_gauge.assert_called_once_with(
            mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
        )

    def test_validate_gauge_address_calls_is_alive_when_gauge_valid(self) -> None:
        """Test that validate_gauge_address calls is_alive when is_gauge returns True.

        :return: None
        """
        mock_ledger_api = MagicMock()

        with patch.object(
            VelodromeVoterContract,
            "is_gauge",
            return_value={"is_gauge": True},
        ), patch.object(
            VelodromeVoterContract,
            "is_alive",
            return_value={"is_alive": True},
        ) as mock_is_alive:
            VelodromeVoterContract.validate_gauge_address(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        mock_is_alive.assert_called_once_with(
            mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
        )

    def test_validate_gauge_address_does_not_call_is_alive_when_not_gauge(
        self,
    ) -> None:
        """Test that validate_gauge_address does not call is_alive when is_gauge returns False.

        :return: None
        """
        mock_ledger_api = MagicMock()

        with patch.object(
            VelodromeVoterContract,
            "is_gauge",
            return_value={"is_gauge": False},
        ), patch.object(
            VelodromeVoterContract,
            "is_alive",
        ) as mock_is_alive:
            VelodromeVoterContract.validate_gauge_address(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        mock_is_alive.assert_not_called()

    def test_validate_gauge_address_does_not_call_is_alive_when_is_gauge_errors(
        self,
    ) -> None:
        """Test that validate_gauge_address does not call is_alive when is_gauge returns an error.

        :return: None
        """
        mock_ledger_api = MagicMock()

        with patch.object(
            VelodromeVoterContract,
            "is_gauge",
            return_value={"error": "contract error"},
        ), patch.object(
            VelodromeVoterContract,
            "is_alive",
        ) as mock_is_alive:
            VelodromeVoterContract.validate_gauge_address(
                mock_ledger_api, MOCK_ADDRESS, MOCK_GAUGE_ADDRESS
            )

        mock_is_alive.assert_not_called()


class TestLength:
    """Tests for the length method."""

    def test_length_returns_total_pools(self) -> None:
        """Test that length returns a dict with the length key.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.length.return_value.call.return_value = 42

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeVoterContract.length(mock_ledger_api, MOCK_ADDRESS)

        assert result == {"length": 42}

    def test_length_calls_contract_function(self) -> None:
        """Test that length calls the length contract function.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.length.return_value.call.return_value = 0

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            VelodromeVoterContract.length(mock_ledger_api, MOCK_ADDRESS)

        mock_contract_instance.functions.length.assert_called_once()

    def test_length_calls_get_instance(self) -> None:
        """Test that length calls get_instance with correct arguments.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.length.return_value.call.return_value = 0

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ) as mock_get_instance:
            VelodromeVoterContract.length(mock_ledger_api, MOCK_ADDRESS)

        mock_get_instance.assert_called_once_with(mock_ledger_api, MOCK_ADDRESS)

    def test_length_with_zero_pools(self) -> None:
        """Test that length correctly handles zero pools.

        :return: None
        """
        mock_ledger_api = MagicMock()
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.length.return_value.call.return_value = 0

        with patch.object(
            VelodromeVoterContract,
            "get_instance",
            return_value=mock_contract_instance,
        ):
            result = VelodromeVoterContract.length(mock_ledger_api, MOCK_ADDRESS)

        assert result == {"length": 0}

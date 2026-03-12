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

import json
from pathlib import Path
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


class TestAbiIntegrity:
    """Verify that every function used in contract.py exists in the ABI."""

    ABI_FILE = "contract_interface.json"
    EXPECTED_FUNCTIONS = ["gauges", "isAlive", "isGauge", "length", "poolForGauge"]

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

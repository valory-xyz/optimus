# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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

"""Comprehensive tests for APRPopulationBehaviour with 100% coverage target."""

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.apr_population import (
    APRPopulationBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    APR_UPDATE_INTERVAL,
    PositionStatus,
)
from packages.valory.skills.liquidity_trader_abci.payloads import APRPopulationPayload
from packages.valory.skills.liquidity_trader_abci.rounds import SynchronizedData
from packages.valory.skills.liquidity_trader_abci.states.apr_population import (
    APRPopulationRound,
)


PACKAGE_DIR = Path(__file__).parent.parent.parent


class LiquidityTraderAbciFSMBehaviourBaseCase(FSMBehaviourBaseCase):
    """Base case for testing FSMBehaviour."""

    path_to_skill = PACKAGE_DIR


class TestAPRPopulationBehaviour(LiquidityTraderAbciFSMBehaviourBaseCase):
    """Comprehensive test suite for APRPopulationBehaviour."""

    behaviour_class = APRPopulationBehaviour
    path_to_skill = PACKAGE_DIR

    def setup_method(self, **kwargs: Any) -> None:
        """Set up the test method with mocked dependencies."""
        # Mock the store path validation before calling super().setup()
        with patch(
            "packages.valory.skills.liquidity_trader_abci.models.Params.get_store_path",
            return_value=Path("/tmp/mock_store"),
        ):
            super().setup(**kwargs)

        # Fast forward to the APRPopulationBehaviour
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            APRPopulationBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Ensure we're in the correct round by setting the round sequence properly
        from packages.valory.skills.liquidity_trader_abci.states.apr_population import (
            APRPopulationRound,
        )

        self.behaviour.current_behaviour.context.state.round_sequence._abci_app._current_round = APRPopulationRound(
            synchronized_data, self.behaviour.current_behaviour.context
        )

        # Create mock shared state
        self.mock_shared_state = MagicMock()
        self.mock_shared_state.trading_type = "test_trading"
        self.mock_shared_state.selected_protocols = ["uniswap", "balancer"]

        # Patch the shared_state property for all tests
        self.shared_state_patcher = patch.object(
            type(self.behaviour.current_behaviour),
            "shared_state",
            new_callable=lambda: self.mock_shared_state,
        )
        self.shared_state_patcher.start()

        # Create mock synchronized data
        self.mock_synchronized_data = MagicMock()
        self.mock_synchronized_data.positions = []
        self.mock_synchronized_data.trading_type = "conservative"

        # Patch the synchronized_data property for all tests
        self.synchronized_data_patcher = patch.object(
            type(self.behaviour.current_behaviour),
            "synchronized_data",
            new_callable=lambda: self.mock_synchronized_data,
        )
        self.synchronized_data_patcher.start()

        # Mock all KV store operations to prevent actual database calls
        self.kv_read_patcher = patch.object(
            type(self.behaviour.current_behaviour),
            "_read_kv",
            side_effect=self._mock_read_kv,
        )
        self.kv_read_patcher.start()

        self.kv_write_patcher = patch.object(
            type(self.behaviour.current_behaviour),
            "_write_kv",
            side_effect=self._mock_write_kv,
        )
        self.kv_write_patcher.start()

        # Initialize behavior attributes
        self.behaviour.current_behaviour.portfolio_data = {
            "portfolio_value": 1000.0,
            "initial_investment": 800.0,
            "volume": 500.0,
        }
        self.behaviour.current_behaviour.current_positions = [
            {
                "status": PositionStatus.OPEN.value,
                "timestamp": 1234567890,
                "enter_timestamp": 1234567890,
                "pool_address": "0x123",
            }
        ]
        self.behaviour.current_behaviour._initial_value = None
        self.behaviour.current_behaviour._final_value = None

        # Mock environment variable
        os.environ["AEA_AGENT"] = "test_agent:agent_hash_123"

        # Unfreeze params to allow modifications
        self.behaviour.current_behaviour.context.params.__dict__["_frozen"] = False

        # Set required parameters
        self.behaviour.current_behaviour.context.params.waiting_period_for_status_check = (
            1
        )

    def teardown_method(self) -> None:
        """Tear down the test method."""
        self.shared_state_patcher.stop()
        self.synchronized_data_patcher.stop()
        self.kv_read_patcher.stop()
        self.kv_write_patcher.stop()

    def _mock_read_kv(self, keys):
        """Mock KV store read operations."""
        yield
        if keys == ("last_apr_calculation",):
            return {"last_apr_calculation": "1234567890"}
        elif keys == ("investing_paused",):
            return {"investing_paused": "false"}
        else:
            return {}

    def _mock_write_kv(self, data):
        """Mock KV store write operations."""
        return None

    def _consume_generator(self, generator: Generator) -> None:
        """Consume a generator to completion."""
        try:
            while True:
                next(generator)
        except StopIteration:
            pass

    def _get_generator_result(self, generator: Generator) -> Any:
        """Get the final result from a generator."""
        result = None
        try:
            while True:
                result = next(generator)
        except StopIteration as e:
            # The return value from yield from is in the StopIteration exception
            result = e.value
        return result

    def test_async_act_success(self) -> None:
        """Test successful async_act execution."""

        # Mock all required methods
        def mock_should_calculate_apr():
            yield
            return True

        def mock_get_or_create_agent_type(*args, **kwargs):
            yield
            return {"type_id": "type_123"}

        def mock_get_or_create_agent_registry(*args, **kwargs):
            yield
            return {"agent_id": "agent_123"}

        def mock_get_or_create_attr_def(*args, **kwargs):
            yield
            return {"attr_def_id": "attr_123"}

        def mock_calculate_and_store_apr(*args, **kwargs):
            yield
            return None

        with patch.object(
            self.behaviour.current_behaviour,
            "_should_calculate_apr",
            side_effect=mock_should_calculate_apr,
        ) as mock_should_calc, patch.object(
            self.behaviour.current_behaviour,
            "_get_or_create_agent_type",
            side_effect=mock_get_or_create_agent_type,
        ) as mock_agent_type, patch.object(
            self.behaviour.current_behaviour,
            "_get_or_create_agent_registry",
            side_effect=mock_get_or_create_agent_registry,
        ) as mock_agent_registry, patch.object(
            self.behaviour.current_behaviour,
            "_get_or_create_attr_def",
            side_effect=mock_get_or_create_attr_def,
        ) as mock_attr_def, patch.object(
            self.behaviour.current_behaviour,
            "_calculate_and_store_apr",
            side_effect=mock_calculate_and_store_apr,
        ) as mock_calc_store, patch.object(
            self.behaviour.current_behaviour, "wait_until_round_end"
        ):
            # Create generator for async_act
            generator = self.behaviour.current_behaviour.async_act()

            # Consume the generator
            self._consume_generator(generator)

            # Verify all methods were called
            mock_should_calc.assert_called_once()
            mock_agent_type.assert_called_once()
            mock_agent_registry.assert_called_once()
            mock_attr_def.assert_called_once()
            mock_calc_store.assert_called_once_with("agent_123", "attr_123")

    def test_async_act_should_not_calculate(self) -> None:
        """Test async_act when should not calculate APR."""

        def mock_should_calculate_apr():
            yield
            return False

        with patch.object(
            self.behaviour.current_behaviour,
            "_should_calculate_apr",
            side_effect=mock_should_calculate_apr,
        ) as mock_should_calc, patch.object(
            self.behaviour.current_behaviour, "_get_or_create_agent_type"
        ) as mock_agent_type, patch.object(
            self.behaviour.current_behaviour, "wait_until_round_end"
        ):
            # Create generator for async_act
            generator = self.behaviour.current_behaviour.async_act()

            # Consume the generator
            self._consume_generator(generator)

            # Verify should_calculate was called but other methods were not
            mock_should_calc.assert_called_once()
            mock_agent_type.assert_not_called()

    def test_async_act_exception_handling(self) -> None:
        """Test async_act exception handling."""
        with patch.object(
            self.behaviour.current_behaviour,
            "_should_calculate_apr",
            side_effect=Exception("Test error"),
        ) as mock_should_calc, patch.object(
            self.behaviour.current_behaviour, "wait_until_round_end"
        ):
            # Create generator for async_act
            generator = self.behaviour.current_behaviour.async_act()

            # Consume the generator
            self._consume_generator(generator)

            # Verify exception was handled
            mock_should_calc.assert_called_once()

    def test_should_calculate_apr_no_last_calculation(self) -> None:
        """Test _should_calculate_apr when no last calculation exists."""

        def mock_read_kv(keys):
            yield
            return None

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ) as mock_read_kv_patch:
            # Use yield from to get the return value, just like in the real code
            def test_generator():
                result = (
                    yield from self.behaviour.current_behaviour._should_calculate_apr()
                )
                return result

            generator = test_generator()
            result = self._get_generator_result(generator)

            assert result is True
            mock_read_kv_patch.assert_called_once_with(keys=("last_apr_calculation",))

    def test_should_calculate_apr_invalid_last_calculation(self) -> None:
        """Test _should_calculate_apr with invalid last calculation time."""

        def mock_read_kv(keys):
            yield
            return {"last_apr_calculation": "invalid"}

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ) as mock_read_kv_patch:
            # Use yield from to get the return value, just like in the real code
            def test_generator():
                result = (
                    yield from self.behaviour.current_behaviour._should_calculate_apr()
                )
                return result

            generator = test_generator()
            result = self._get_generator_result(generator)

            assert result is True
            mock_read_kv_patch.assert_called_once_with(keys=("last_apr_calculation",))

    def test_should_calculate_apr_new_position(self) -> None:
        """Test _should_calculate_apr when new position opened since last calculation."""
        current_time = 1234567890
        last_calculation = current_time - 1000  # 1000 seconds ago
        position_time = (
            current_time - 500
        )  # 500 seconds ago (newer than last calculation)

        def mock_read_kv(keys):
            yield
            return {"last_apr_calculation": str(last_calculation)}

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ) as mock_read_kv_patch, patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=current_time,
        ):
            # Update position timestamp
            self.behaviour.current_behaviour.current_positions[0][
                "timestamp"
            ] = position_time

            # Use yield from to get the return value, just like in the real code
            def test_generator():
                result = (
                    yield from self.behaviour.current_behaviour._should_calculate_apr()
                )
                return result

            generator = test_generator()
            result = self._get_generator_result(generator)

            assert result is True

    def test_should_calculate_apr_interval_exceeded(self) -> None:
        """Test _should_calculate_apr when update interval exceeded."""
        current_time = 1234567890
        last_calculation = current_time - APR_UPDATE_INTERVAL - 100  # Exceeded interval

        def mock_read_kv(keys):
            yield
            return {"last_apr_calculation": str(last_calculation)}

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ) as mock_read_kv_patch, patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=current_time,
        ):
            # Use yield from to get the return value, just like in the real code
            def test_generator():
                result = (
                    yield from self.behaviour.current_behaviour._should_calculate_apr()
                )
                return result

            generator = test_generator()
            result = self._get_generator_result(generator)

            assert result is True

    def test_should_calculate_apr_skip_calculation(self) -> None:
        """Test _should_calculate_apr when calculation should be skipped."""
        current_time = 1234567890
        last_calculation = current_time - 100  # Within interval

        def mock_read_kv(keys):
            yield
            return {"last_apr_calculation": str(last_calculation)}

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ) as mock_read_kv_patch, patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=current_time,
        ):
            # Clear current positions to test the skip logic without new position interference
            self.behaviour.current_behaviour.current_positions = []

            # Use yield from to get the return value, just like in the real code
            def test_generator():
                result = (
                    yield from self.behaviour.current_behaviour._should_calculate_apr()
                )
                return result

            generator = test_generator()
            result = self._get_generator_result(generator)

            assert result is False

    def test_create_portfolio_snapshot(self) -> None:
        """Test _create_portfolio_snapshot method."""
        with patch.object(
            self.behaviour.current_behaviour, "_convert_decimals"
        ) as mock_convert:
            mock_convert.side_effect = lambda x: x  # Return as-is for simplicity

            result = self.behaviour.current_behaviour._create_portfolio_snapshot()

            expected = {
                "portfolio": self.behaviour.current_behaviour.portfolio_data,
                "positons": self.behaviour.current_behaviour.current_positions,
            }
            assert result == expected
            assert mock_convert.call_count == 2

    def test_to_decimal_none(self) -> None:
        """Test _to_decimal with None input."""
        result = self.behaviour.current_behaviour._to_decimal(None)
        assert result is None

    def test_to_decimal_already_decimal(self) -> None:
        """Test _to_decimal with Decimal input."""
        value = Decimal("123.45")
        result = self.behaviour.current_behaviour._to_decimal(value)
        assert result == value

    def test_to_decimal_float(self) -> None:
        """Test _to_decimal with float input."""
        value = 123.45
        result = self.behaviour.current_behaviour._to_decimal(value)
        assert result == Decimal("123.45")

    def test_to_decimal_int(self) -> None:
        """Test _to_decimal with int input."""
        value = 123
        result = self.behaviour.current_behaviour._to_decimal(value)
        assert result == Decimal("123")

    def test_to_decimal_string(self) -> None:
        """Test _to_decimal with string input."""
        value = "123.45"
        result = self.behaviour.current_behaviour._to_decimal(value)
        assert result == Decimal("123.45")

    def test_to_decimal_invalid(self) -> None:
        """Test _to_decimal with invalid input."""
        with patch.object(
            self.behaviour.current_behaviour.context.logger, "warning"
        ) as mock_warning:
            result = self.behaviour.current_behaviour._to_decimal("invalid")
            assert result is None
            mock_warning.assert_called_once()

    def test_convert_decimals_dict(self) -> None:
        """Test _convert_decimals with dict input."""
        data = {"key1": Decimal("123.45"), "key2": "string"}
        result = self.behaviour.current_behaviour._convert_decimals(data)
        expected = {"key1": 123.45, "key2": "string"}
        assert result == expected

    def test_convert_decimals_list(self) -> None:
        """Test _convert_decimals with list input."""
        data = [Decimal("123.45"), "string", 123]
        result = self.behaviour.current_behaviour._convert_decimals(data)
        expected = [123.45, "string", 123]
        assert result == expected

    def test_convert_decimals_decimal(self) -> None:
        """Test _convert_decimals with Decimal input."""
        data = Decimal("123.45")
        result = self.behaviour.current_behaviour._convert_decimals(data)
        assert result == 123.45

    def test_convert_decimals_other(self) -> None:
        """Test _convert_decimals with other type input."""
        data = "string"
        result = self.behaviour.current_behaviour._convert_decimals(data)
        assert result == "string"

    def test_get_apr_calculation_metrics_no_values(self) -> None:
        """Test _get_apr_calculation_metrics with no initial/final values."""
        self.behaviour.current_behaviour._initial_value = None
        self.behaviour.current_behaviour._final_value = None

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_first_investment_timestamp",
            return_value=None,
        ):
            result = self.behaviour.current_behaviour._get_apr_calculation_metrics()

            expected = {
                "initial_value": None,
                "final_value": None,
                "f_i_ratio": None,
                "first_investment_timestamp": None,
                "time_ratio": None,
            }
            assert result == expected

    def test_get_apr_calculation_metrics_with_values(self) -> None:
        """Test _get_apr_calculation_metrics with valid values."""
        self.behaviour.current_behaviour._initial_value = 800.0
        self.behaviour.current_behaviour._final_value = 1000.0

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_first_investment_timestamp",
            return_value=1234567890,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=1234567890 + 86400,  # 24 hours later
        ):
            result = self.behaviour.current_behaviour._get_apr_calculation_metrics()

            assert result["initial_value"] == 800.0
            assert result["final_value"] == 1000.0
            assert result["f_i_ratio"] == 0.25  # (1000/800) - 1
            assert result["first_investment_timestamp"] == 1234567890
            assert result["time_ratio"] == 365.0  # 8760 hours in a year / 24 hours
            assert result["volatility_warning"] == "LOW"
            assert result["actual_hours"] == 24.0
            assert result["calculation_hours"] == 24.0

    def test_get_apr_calculation_metrics_short_period(self) -> None:
        """Test _get_apr_calculation_metrics with very short period."""
        self.behaviour.current_behaviour._initial_value = 800.0
        self.behaviour.current_behaviour._final_value = 1000.0

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_first_investment_timestamp",
            return_value=1234567890,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=1234567890 + 30,  # 30 seconds later
        ):
            result = self.behaviour.current_behaviour._get_apr_calculation_metrics()

            assert result["volatility_warning"] == "VERY_HIGH"
            assert result["calculation_hours"] == 0.0167  # MIN_HOURS

    def test_sign_message_success(self) -> None:
        """Test sign_message with successful signature."""
        message = "test message"
        signature = b"\x12\x34\x56\x78"

        def mock_get_signature(*args, **kwargs):
            yield
            return f"0x{signature.hex()}"

        with patch.object(
            self.behaviour.current_behaviour,
            "get_signature",
            side_effect=mock_get_signature,
        ):
            generator = self.behaviour.current_behaviour.sign_message(message)
            result = self._get_generator_result(generator)

            assert result == signature.hex()

    def test_sign_message_no_signature(self) -> None:
        """Test sign_message with no signature."""
        message = "test message"

        def mock_get_signature(*args, **kwargs):
            yield
            return None

        with patch.object(
            self.behaviour.current_behaviour,
            "get_signature",
            side_effect=mock_get_signature,
        ):
            generator = self.behaviour.current_behaviour.sign_message(message)
            result = self._get_generator_result(generator)

            assert result is None

    def test_calculate_actual_apr_no_valid_data(self) -> None:
        """Test calculate_actual_apr with no valid portfolio data."""
        with patch.object(
            self.behaviour.current_behaviour,
            "_has_valid_portfolio_data",
            return_value=False,
        ):
            generator = self.behaviour.current_behaviour.calculate_actual_apr(1000.0)
            result = self._get_generator_result(generator)

            assert result is None

    def test_calculate_actual_apr_no_initial_investment(self) -> None:
        """Test calculate_actual_apr with no initial investment."""
        with patch.object(
            self.behaviour.current_behaviour,
            "_has_valid_portfolio_data",
            return_value=True,
        ), patch.object(
            self.behaviour.current_behaviour,
            "get_stored_initial_investment",
            return_value=None,
        ):
            generator = self.behaviour.current_behaviour.calculate_actual_apr(1000.0)
            result = self._get_generator_result(generator)

            assert result is None

    def test_calculate_actual_apr_success(self) -> None:
        """Test calculate_actual_apr with successful calculation."""
        with patch.object(
            self.behaviour.current_behaviour,
            "_has_valid_portfolio_data",
            return_value=True,
        ), patch.object(
            self.behaviour.current_behaviour,
            "get_stored_initial_investment",
            return_value=800.0,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_get_first_investment_timestamp",
            return_value=1234567890,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=1234567890 + 3600,
        ), patch.object(
            self.behaviour.current_behaviour, "_adjust_apr_for_eth_price"
        ) as mock_adjust:
            generator = self.behaviour.current_behaviour.calculate_actual_apr(1000.0)
            result = self._get_generator_result(generator)

            assert result is not None
            assert "total_actual_apr" in result
            mock_adjust.assert_called_once()

    def test_get_stored_initial_investment_success(self) -> None:
        """Test get_stored_initial_investment with valid data."""
        result = self.behaviour.current_behaviour.get_stored_initial_investment()
        assert result == 800.0

    def test_get_stored_initial_investment_no_portfolio(self) -> None:
        """Test get_stored_initial_investment with no portfolio data."""
        self.behaviour.current_behaviour.portfolio_data = None
        result = self.behaviour.current_behaviour.get_stored_initial_investment()
        assert result is None

    def test_get_stored_initial_investment_no_initial_investment(self) -> None:
        """Test get_stored_initial_investment with no initial investment."""
        self.behaviour.current_behaviour.portfolio_data = {"portfolio_value": 1000.0}
        result = self.behaviour.current_behaviour.get_stored_initial_investment()
        assert result is None

    def test_has_valid_portfolio_data_valid(self) -> None:
        """Test _has_valid_portfolio_data with valid data."""
        result = self.behaviour.current_behaviour._has_valid_portfolio_data()
        assert result is True

    def test_calculate_and_store_apr_no_actual_apr_data_empty_dict(self) -> None:
        """Test _calculate_and_store_apr with empty actual_apr_data dict."""

        def mock_calculate_actual_apr(*args, **kwargs):
            yield
            return {}  # Empty dict should trigger early return

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_actual_apr",
            side_effect=mock_calculate_actual_apr,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_create_portfolio_snapshot",
            return_value={"snapshot": "data"},
        ), patch.object(
            self.behaviour.current_behaviour, "create_agent_attribute"
        ) as mock_create_attr:
            generator = self.behaviour.current_behaviour._calculate_and_store_apr(
                "agent_123", "attr_123"
            )
            self._consume_generator(generator)

            # Should not call create_agent_attribute due to early return
            mock_create_attr.assert_not_called()

    def test_calculate_and_store_apr_no_total_actual_apr_zero(self) -> None:
        """Test _calculate_and_store_apr with zero total_actual_apr."""

        def mock_calculate_actual_apr(*args, **kwargs):
            yield
            return {
                "total_actual_apr": 0,
                "adjusted_apr": 0,
            }  # Zero should trigger early return

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_actual_apr",
            side_effect=mock_calculate_actual_apr,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_create_portfolio_snapshot",
            return_value={"snapshot": "data"},
        ), patch.object(
            self.behaviour.current_behaviour, "create_agent_attribute"
        ) as mock_create_attr:
            generator = self.behaviour.current_behaviour._calculate_and_store_apr(
                "agent_123", "attr_123"
            )
            self._consume_generator(generator)

            # Should not call create_agent_attribute due to early return
            mock_create_attr.assert_not_called()

    def test_get_apr_calculation_metrics_high_volatility(self) -> None:
        """Test _get_apr_calculation_metrics with HIGH volatility warning."""
        # Mock time difference to be between MIN_HOURS and 1 hour (HIGH volatility)
        with patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=1234567890 + 1800,  # 30 minutes later
        ), patch.object(
            self.behaviour.current_behaviour,
            "_get_first_investment_timestamp",
            return_value=1234567890,
        ):
            metrics = self.behaviour.current_behaviour._get_apr_calculation_metrics()

            assert metrics["volatility_warning"] == "HIGH"
            assert metrics["actual_hours"] == 0.5  # 30 minutes = 0.5 hours

    def test_get_apr_calculation_metrics_medium_volatility(self) -> None:
        """Test _get_apr_calculation_metrics with MEDIUM volatility warning."""
        # Mock time difference to be between 1 and 24 hours (MEDIUM volatility)
        with patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=1234567890 + 7200,  # 2 hours later
        ), patch.object(
            self.behaviour.current_behaviour,
            "_get_first_investment_timestamp",
            return_value=1234567890,
        ):
            metrics = self.behaviour.current_behaviour._get_apr_calculation_metrics()

            assert metrics["volatility_warning"] == "MEDIUM"
            assert metrics["actual_hours"] == 2.0  # 2 hours

    def test_calculate_apr_negative_apr(self) -> None:
        """Test _calculate_apr with negative APR scenario."""
        # Set up the final and initial values directly
        self.behaviour.current_behaviour._final_value = (
            600.0  # Final value less than initial (800.0) but positive
        )
        self.behaviour.current_behaviour._initial_value = 800.0

        result = {}
        self.behaviour.current_behaviour._calculate_apr(
            current_timestamp=1234567890 + 3600,  # 1 hour later
            first_investment_timestamp=1234567890,
            result=result,
        )

        # Should handle negative APR case and return a negative value
        assert "total_actual_apr" in result
        assert result["total_actual_apr"] < 0  # Negative APR due to loss

    def test_has_valid_portfolio_data_no_portfolio(self) -> None:
        """Test _has_valid_portfolio_data with no portfolio data."""
        self.behaviour.current_behaviour.portfolio_data = None
        result = self.behaviour.current_behaviour._has_valid_portfolio_data()
        assert result is False

    def test_has_valid_portfolio_data_no_portfolio_value(self) -> None:
        """Test _has_valid_portfolio_data with no portfolio_value."""
        self.behaviour.current_behaviour.portfolio_data = {"other_key": "value"}
        result = self.behaviour.current_behaviour._has_valid_portfolio_data()
        assert result is False

    def test_get_first_investment_timestamp_from_positions(self) -> None:
        """Test _get_first_investment_timestamp from positions."""
        result = self.behaviour.current_behaviour._get_first_investment_timestamp()
        assert result == 1234567890

    def test_get_first_investment_timestamp_from_enter_timestamp(self) -> None:
        """Test _get_first_investment_timestamp from enter_timestamp."""
        self.behaviour.current_behaviour.current_positions[0]["timestamp"] = None
        self.behaviour.current_behaviour.current_positions[0][
            "enter_timestamp"
        ] = 9876543210

        result = self.behaviour.current_behaviour._get_first_investment_timestamp()
        assert result == 9876543210

    def test_get_first_investment_timestamp_fallback(self) -> None:
        """Test _get_first_investment_timestamp fallback to current time."""
        self.behaviour.current_behaviour.current_positions = []

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=1234567890,
        ):
            result = self.behaviour.current_behaviour._get_first_investment_timestamp()
            assert result == 1234567890

    def test_calculate_apr_invalid_final_value(self) -> None:
        """Test _calculate_apr with invalid final value."""
        with patch.object(
            self.behaviour.current_behaviour.context.logger, "warning"
        ) as mock_warning:
            result = self.behaviour.current_behaviour._calculate_apr(
                1234567890, 1234567890, {}
            )
            assert result == 0.0
            mock_warning.assert_called_once()

    def test_calculate_apr_invalid_initial_value(self) -> None:
        """Test _calculate_apr with invalid initial value."""
        self.behaviour.current_behaviour._final_value = 1000.0
        self.behaviour.current_behaviour._initial_value = 0.0

        with patch.object(
            self.behaviour.current_behaviour.context.logger, "error"
        ) as mock_error:
            result = self.behaviour.current_behaviour._calculate_apr(
                1234567890, 1234567890, {}
            )
            assert result == 0.0
            mock_error.assert_called_once()

    def test_calculate_apr_success(self) -> None:
        """Test _calculate_apr with successful calculation."""
        self.behaviour.current_behaviour._final_value = 1000.0
        self.behaviour.current_behaviour._initial_value = 800.0

        result = {}
        self.behaviour.current_behaviour._calculate_apr(
            1234567890 + 3600, 1234567890, result  # 1 hour later
        )

        assert "total_actual_apr" in result
        assert result["total_actual_apr"] > 0

    def test_calculate_apr_very_short_period(self) -> None:
        """Test _calculate_apr with very short period."""
        self.behaviour.current_behaviour._final_value = 1000.0
        self.behaviour.current_behaviour._initial_value = 800.0

        with patch.object(
            self.behaviour.current_behaviour.context.logger, "warning"
        ) as mock_warning:
            result = {}
            self.behaviour.current_behaviour._calculate_apr(
                1234567890 + 30, 1234567890, result  # 30 seconds later
            )

            assert "total_actual_apr" in result
            mock_warning.assert_called_once()

    def test_calculate_apr_short_period(self) -> None:
        """Test _calculate_apr with short period."""
        self.behaviour.current_behaviour._final_value = 1000.0
        self.behaviour.current_behaviour._initial_value = 800.0

        with patch.object(
            self.behaviour.current_behaviour.context.logger, "info"
        ) as mock_info:
            result = {}
            self.behaviour.current_behaviour._calculate_apr(
                1234567890 + 1800, 1234567890, result  # 30 minutes later
            )

            assert "total_actual_apr" in result
            mock_info.assert_called()

    def test_adjust_apr_for_eth_price_no_apr(self) -> None:
        """Test _adjust_apr_for_eth_price with no APR data."""
        result = {}

        def mock_fetch_zero_address_price():
            yield
            return 2000.0

        def mock_fetch_historical_token_price(*args, **kwargs):
            yield
            return 1800.0

        with patch.object(
            self.behaviour.current_behaviour,
            "_fetch_zero_address_price",
            side_effect=mock_fetch_zero_address_price,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_fetch_historical_token_price",
            side_effect=mock_fetch_historical_token_price,
        ):
            generator = self.behaviour.current_behaviour._adjust_apr_for_eth_price(
                result, 1234567890
            )
            next(generator)

            # Should not modify result since no APR data
            assert result == {}

    def test_adjust_apr_for_eth_price_success(self) -> None:
        """Test _adjust_apr_for_eth_price with successful adjustment."""
        result = {"total_actual_apr": 10.0}

        def mock_fetch_zero_address_price():
            yield
            return 2000.0

        def mock_fetch_historical_token_price(*args, **kwargs):
            yield
            return 1800.0

        with patch.object(
            self.behaviour.current_behaviour,
            "_fetch_zero_address_price",
            side_effect=mock_fetch_zero_address_price,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_fetch_historical_token_price",
            side_effect=mock_fetch_historical_token_price,
        ):
            generator = self.behaviour.current_behaviour._adjust_apr_for_eth_price(
                result, 1234567890
            )
            self._consume_generator(generator)

            assert "adjusted_apr" in result
            assert "adjustment_factor" in result
            assert "current_price" in result
            assert "initial_price" in result

    def test_adjust_apr_for_eth_price_zero_start_price(self) -> None:
        """Test _adjust_apr_for_eth_price with zero start price."""
        result = {"total_actual_apr": 10.0}

        def mock_fetch_zero_address_price():
            yield
            return 2000.0

        def mock_fetch_historical_token_price(*args, **kwargs):
            yield
            return 0.0

        with patch.object(
            self.behaviour.current_behaviour,
            "_fetch_zero_address_price",
            side_effect=mock_fetch_zero_address_price,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_fetch_historical_token_price",
            side_effect=mock_fetch_historical_token_price,
        ), patch.object(
            self.behaviour.current_behaviour.context.logger, "warning"
        ) as mock_warning:
            generator = self.behaviour.current_behaviour._adjust_apr_for_eth_price(
                result, 1234567890
            )
            self._consume_generator(generator)

            mock_warning.assert_called_once()

    def test_adjust_apr_for_eth_price_conversion_failure(self) -> None:
        """Test _adjust_apr_for_eth_price with conversion failure."""
        result = {"total_actual_apr": 10.0}

        def mock_fetch_zero_address_price():
            yield
            return "invalid"

        def mock_fetch_historical_token_price(*args, **kwargs):
            yield
            return 1800.0

        with patch.object(
            self.behaviour.current_behaviour,
            "_fetch_zero_address_price",
            side_effect=mock_fetch_zero_address_price,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_fetch_historical_token_price",
            side_effect=mock_fetch_historical_token_price,
        ), patch.object(
            self.behaviour.current_behaviour.context.logger, "warning"
        ) as mock_warning:
            generator = self.behaviour.current_behaviour._adjust_apr_for_eth_price(
                result, 1234567890
            )
            self._consume_generator(generator)

            assert mock_warning.call_count == 2

    def test_read_investing_paused_success_true(self) -> None:
        """Test _read_investing_paused with true value."""

        def mock_read_kv(keys):
            yield
            return {"investing_paused": "true"}

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ):
            generator = self.behaviour.current_behaviour._read_investing_paused()
            result = self._get_generator_result(generator)

            assert result is True

    def test_read_investing_paused_success_false(self) -> None:
        """Test _read_investing_paused with false value."""

        def mock_read_kv(keys):
            yield
            return {"investing_paused": "false"}

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ):
            generator = self.behaviour.current_behaviour._read_investing_paused()
            result = self._get_generator_result(generator)

            assert result is False

    def test_read_investing_paused_no_response(self) -> None:
        """Test _read_investing_paused with no response."""

        def mock_read_kv(keys):
            yield
            return None

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ), patch.object(
            self.behaviour.current_behaviour.context.logger, "warning"
        ) as mock_warning:
            generator = self.behaviour.current_behaviour._read_investing_paused()
            result = self._get_generator_result(generator)

            assert result is False
            mock_warning.assert_called_once()

    def test_read_investing_paused_none_value(self) -> None:
        """Test _read_investing_paused with None value."""

        def mock_read_kv(keys):
            yield
            return {"investing_paused": None}

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ), patch.object(
            self.behaviour.current_behaviour.context.logger, "warning"
        ) as mock_warning:
            generator = self.behaviour.current_behaviour._read_investing_paused()
            result = self._get_generator_result(generator)

            assert result is False
            mock_warning.assert_called_once()

    def test_read_investing_paused_exception(self) -> None:
        """Test _read_investing_paused with exception."""

        def mock_read_kv(keys):
            raise Exception("Test error")

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ), patch.object(
            self.behaviour.current_behaviour.context.logger, "error"
        ) as mock_error:
            generator = self.behaviour.current_behaviour._read_investing_paused()
            result = self._get_generator_result(generator)

            assert result is False
            mock_error.assert_called_once()

    def test_calculate_and_store_apr_no_actual_apr_data(self) -> None:
        """Test _calculate_and_store_apr with no actual APR data."""

        def mock_calculate_actual_apr(*args, **kwargs):
            yield
            return None

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_actual_apr",
            side_effect=mock_calculate_actual_apr,
        ):
            generator = self.behaviour.current_behaviour._calculate_and_store_apr(
                "agent_123", "attr_123"
            )
            next(generator)

            # Should return early without storing

    def test_calculate_and_store_apr_no_total_apr(self) -> None:
        """Test _calculate_and_store_apr with no total APR."""

        def mock_calculate_actual_apr(*args, **kwargs):
            yield
            return {"adjusted_apr": 5.0}

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_actual_apr",
            side_effect=mock_calculate_actual_apr,
        ):
            generator = self.behaviour.current_behaviour._calculate_and_store_apr(
                "agent_123", "attr_123"
            )
            next(generator)

            # Should return early without storing

    def test_calculate_and_store_apr_success(self) -> None:
        """Test _calculate_and_store_apr with successful storage."""

        def mock_calculate_actual_apr(*args, **kwargs):
            yield
            return {"total_actual_apr": 10.0, "adjusted_apr": 8.0}

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_actual_apr",
            side_effect=mock_calculate_actual_apr,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_create_portfolio_snapshot",
            return_value={"snapshot": "data"},
        ), patch.object(
            self.behaviour.current_behaviour,
            "_get_apr_calculation_metrics",
            return_value={"metrics": "data"},
        ), patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=1234567890,
        ), patch.object(
            self.behaviour.current_behaviour, "create_agent_attribute"
        ) as mock_create_attr, patch.object(
            self.behaviour.current_behaviour, "_write_kv"
        ) as mock_write_kv:
            generator = self.behaviour.current_behaviour._calculate_and_store_apr(
                "agent_123", "attr_123"
            )
            self._consume_generator(generator)

            mock_create_attr.assert_called_once()
            mock_write_kv.assert_called_once_with(
                {"last_apr_calculation": "1234567890"}
            )

    def test_calculate_and_store_apr_no_positions(self) -> None:
        """Test _calculate_and_store_apr with no current positions."""
        self.behaviour.current_behaviour.current_positions = []

        def mock_calculate_actual_apr(*args, **kwargs):
            yield
            return {"total_actual_apr": 10.0, "adjusted_apr": 8.0}

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_actual_apr",
            side_effect=mock_calculate_actual_apr,
        ), patch.object(
            self.behaviour.current_behaviour,
            "_create_portfolio_snapshot",
            return_value={"snapshot": "data"},
        ), patch.object(
            self.behaviour.current_behaviour,
            "_get_apr_calculation_metrics",
            return_value={"metrics": "data"},
        ), patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=1234567890,
        ), patch.object(
            self.behaviour.current_behaviour, "create_agent_attribute"
        ) as mock_create_attr, patch.object(
            self.behaviour.current_behaviour, "_write_kv"
        ) as mock_write_kv:
            generator = self.behaviour.current_behaviour._calculate_and_store_apr(
                "agent_123", "attr_123"
            )
            self._consume_generator(generator)

            mock_create_attr.assert_called_once()
            # Should handle None first_investment_timestamp
            call_args = mock_create_attr.call_args[0]
            enhanced_data = call_args[2]
            assert enhanced_data["first_investment_timestamp"] is None

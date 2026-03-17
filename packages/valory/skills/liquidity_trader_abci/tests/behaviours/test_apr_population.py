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

"""Tests for behaviours/apr_population.py."""

# pylint: skip-file

import os
from decimal import Decimal
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.skills.liquidity_trader_abci.behaviours.apr_population import (
    APRPopulationBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    APR_UPDATE_INTERVAL,
    PositionStatus,
)


def _make_behaviour():
    """Create an APRPopulationBehaviour without __init__."""
    obj = object.__new__(APRPopulationBehaviour)
    ctx = MagicMock()
    obj.__dict__["_context"] = ctx
    obj._initial_value = None
    obj._final_value = None
    return obj


def _drive(gen):
    """Drive a generator to completion."""
    val = None
    while True:
        try:
            val = gen.send(val)
        except StopIteration as exc:
            return exc.value


class TestAPRPopulationBehaviour:
    """Tests for APRPopulationBehaviour."""

    def test_async_act_should_not_calculate(self) -> None:
        """Test async_act when should_calculate returns False."""
        obj = _make_behaviour()
        benchmark_mock = MagicMock()
        obj.context.benchmark_tool.measure.return_value = benchmark_mock
        obj.context.agent_address = "0xagent"

        def fake_should_calc():
            yield
            return False

        def fake_send(*args, **kwargs):
            yield

        def fake_wait(*args, **kwargs):
            yield

        obj._should_calculate_apr = fake_should_calc
        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()

        gen = obj.async_act()
        _drive(gen)
        obj.set_done.assert_called_once()

    def test_async_act_should_calculate(self) -> None:
        """Test async_act when should_calculate returns True."""
        obj = _make_behaviour()
        benchmark_mock = MagicMock()
        obj.context.benchmark_tool.measure.return_value = benchmark_mock
        obj.context.agent_address = "0xagent"

        def fake_should_calc():
            yield
            return True

        def fake_get_agent_type(sender):
            yield
            return {"type_id": "t1"}

        def fake_get_agent_reg():
            yield
            return {"agent_id": "a1"}

        def fake_get_attr_def(type_id, agent_id):
            yield
            return {"attr_def_id": "ad1"}

        def fake_calc_and_store(agent_id, attr_def_id):
            yield

        def fake_send(*args, **kwargs):
            yield

        def fake_wait(*args, **kwargs):
            yield

        obj._should_calculate_apr = fake_should_calc
        obj._get_or_create_agent_type = fake_get_agent_type
        obj._get_or_create_agent_registry = fake_get_agent_reg
        obj._get_or_create_attr_def = fake_get_attr_def
        obj._calculate_and_store_apr = fake_calc_and_store
        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()

        gen = obj.async_act()
        _drive(gen)
        obj.set_done.assert_called_once()

    def test_async_act_exception(self) -> None:
        """Test async_act when an exception is raised."""
        obj = _make_behaviour()
        benchmark_mock = MagicMock()
        obj.context.benchmark_tool.measure.return_value = benchmark_mock
        obj.context.agent_address = "0xagent"

        def fake_should_calc():
            raise ValueError("test error")
            yield  # noqa

        def fake_send(*args, **kwargs):
            yield

        def fake_wait(*args, **kwargs):
            yield

        obj._should_calculate_apr = fake_should_calc
        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()

        gen = obj.async_act()
        _drive(gen)
        obj.context.logger.error.assert_called()
        obj.set_done.assert_called_once()


class TestShouldCalculateApr:
    """Tests for _should_calculate_apr."""

    def test_no_last_calc_time(self) -> None:
        """Test returns True when no last calculation time."""
        obj = _make_behaviour()
        obj.current_positions = []

        def fake_read_kv(keys):
            yield
            return None

        obj._read_kv = fake_read_kv
        obj._get_current_timestamp = MagicMock(return_value=1700000000)

        gen = obj._should_calculate_apr()
        result = _drive(gen)
        assert result is True

    def test_last_calc_time_invalid(self) -> None:
        """Test returns True when last calculation time is invalid."""
        obj = _make_behaviour()
        obj.current_positions = []

        def fake_read_kv(keys):
            yield
            return {"last_apr_calculation": "invalid"}

        obj._read_kv = fake_read_kv
        obj._get_current_timestamp = MagicMock(return_value=1700000000)

        gen = obj._should_calculate_apr()
        result = _drive(gen)
        assert result is True

    def test_interval_not_passed(self) -> None:
        """Test returns False when interval not passed."""
        obj = _make_behaviour()
        obj.current_positions = []

        def fake_read_kv(keys):
            yield
            return {"last_apr_calculation": "1700000000"}

        obj._read_kv = fake_read_kv
        # Just 100 seconds later - less than APR_UPDATE_INTERVAL
        obj._get_current_timestamp = MagicMock(return_value=1700000100)

        gen = obj._should_calculate_apr()
        result = _drive(gen)
        assert result is False

    def test_interval_passed(self) -> None:
        """Test returns True when interval has passed."""
        obj = _make_behaviour()
        obj.current_positions = []

        def fake_read_kv(keys):
            yield
            return {"last_apr_calculation": "1700000000"}

        obj._read_kv = fake_read_kv
        obj._get_current_timestamp = MagicMock(
            return_value=1700000000 + APR_UPDATE_INTERVAL + 1
        )

        gen = obj._should_calculate_apr()
        result = _drive(gen)
        assert result is True

    def test_new_position_opened(self) -> None:
        """Test returns True when new position opened since last calc."""
        obj = _make_behaviour()
        obj.current_positions = [
            {
                "status": PositionStatus.OPEN.value,
                "enter_timestamp": 1700000100,
                "timestamp": None,
            }
        ]

        def fake_read_kv(keys):
            yield
            return {"last_apr_calculation": "1700000000"}

        obj._read_kv = fake_read_kv
        obj._get_current_timestamp = MagicMock(return_value=1700000100)

        gen = obj._should_calculate_apr()
        result = _drive(gen)
        assert result is True

    def test_closed_position_ignored(self) -> None:
        """Test closed positions don't trigger early recalc."""
        obj = _make_behaviour()
        obj.current_positions = [
            {
                "status": "closed",
                "enter_timestamp": 1700000100,
            }
        ]

        def fake_read_kv(keys):
            yield
            return {"last_apr_calculation": "1700000000"}

        obj._read_kv = fake_read_kv
        obj._get_current_timestamp = MagicMock(return_value=1700000100)

        gen = obj._should_calculate_apr()
        result = _drive(gen)
        assert result is False

    def test_position_with_timestamp_fallback(self) -> None:
        """Test position using 'timestamp' when enter_timestamp is None."""
        obj = _make_behaviour()
        obj.current_positions = [
            {
                "status": PositionStatus.OPEN.value,
                "enter_timestamp": None,
                "timestamp": 1700000200,
            }
        ]

        def fake_read_kv(keys):
            yield
            return {"last_apr_calculation": "1700000000"}

        obj._read_kv = fake_read_kv
        obj._get_current_timestamp = MagicMock(return_value=1700000200)

        gen = obj._should_calculate_apr()
        result = _drive(gen)
        assert result is True

    def test_position_older_than_last_calc(self) -> None:
        """Test position opened before last calc doesn't trigger early recalc."""
        obj = _make_behaviour()
        obj.current_positions = [
            {
                "status": PositionStatus.OPEN.value,
                "enter_timestamp": 1699999000,
                "timestamp": None,
            }
        ]

        def fake_read_kv(keys):
            yield
            return {"last_apr_calculation": "1700000000"}

        obj._read_kv = fake_read_kv
        # current_time - last_calculation_time < APR_UPDATE_INTERVAL so returns False
        obj._get_current_timestamp = MagicMock(return_value=1700000100)

        gen = obj._should_calculate_apr()
        result = _drive(gen)
        assert result is False

    def test_position_no_timestamp(self) -> None:
        """Test position with no timestamps doesn't trigger early recalc."""
        obj = _make_behaviour()
        obj.current_positions = [
            {
                "status": PositionStatus.OPEN.value,
                "enter_timestamp": None,
                "timestamp": None,
            }
        ]

        def fake_read_kv(keys):
            yield
            return {"last_apr_calculation": "1700000000"}

        obj._read_kv = fake_read_kv
        obj._get_current_timestamp = MagicMock(return_value=1700000100)

        gen = obj._should_calculate_apr()
        result = _drive(gen)
        assert result is False

    def test_no_last_calc_with_positions(self) -> None:
        """Test returns True when no last_calculation_time even with positions."""
        obj = _make_behaviour()
        obj.current_positions = [
            {
                "status": PositionStatus.OPEN.value,
                "enter_timestamp": 1700000100,
            }
        ]

        def fake_read_kv(keys):
            yield
            return {"last_apr_calculation": None}

        obj._read_kv = fake_read_kv
        obj._get_current_timestamp = MagicMock(return_value=1700000100)

        gen = obj._should_calculate_apr()
        result = _drive(gen)
        assert result is True


class TestCalculateAndStoreApr:
    """Tests for _calculate_and_store_apr."""

    def test_no_actual_apr_data(self) -> None:
        """Test early return when calculate_actual_apr returns None."""
        obj = _make_behaviour()
        obj.portfolio_data = {"portfolio_value": 100}

        def fake_calc_actual_apr(pv):
            yield
            return None

        obj.calculate_actual_apr = fake_calc_actual_apr
        obj._create_portfolio_snapshot = MagicMock(return_value={})

        gen = obj._calculate_and_store_apr("a1", "ad1")
        result = _drive(gen)
        assert result is None

    def test_no_total_actual_apr(self) -> None:
        """Test early return when total_actual_apr is missing."""
        obj = _make_behaviour()
        obj.portfolio_data = {"portfolio_value": 100}

        def fake_calc_actual_apr(pv):
            yield
            return {"total_actual_apr": None, "adjusted_apr": 5.0}

        obj.calculate_actual_apr = fake_calc_actual_apr
        obj._create_portfolio_snapshot = MagicMock(return_value={})

        gen = obj._calculate_and_store_apr("a1", "ad1")
        result = _drive(gen)
        assert result is None

    def test_zero_total_actual_apr_is_valid(self) -> None:
        """Test that total_actual_apr=0.0 does NOT trigger early return."""
        obj = _make_behaviour()
        obj.portfolio_data = {"portfolio_value": 100, "volume": 50}
        obj.current_positions = [{"timestamp": 1700000000}]

        shared = MagicMock()
        shared.trading_type = "balanced"
        shared.selected_protocols = ["uniswap"]

        def fake_calc_actual_apr(pv):
            yield
            return {"total_actual_apr": 0.0, "adjusted_apr": 0.0}

        def fake_create_agent_attr(agent_id, attr_def_id, data):
            yield
            return {"id": "attr1"}

        def fake_write_kv(data):
            yield

        obj.calculate_actual_apr = fake_calc_actual_apr
        obj._create_portfolio_snapshot = MagicMock(return_value={"snap": True})
        obj._get_apr_calculation_metrics = MagicMock(return_value={"m": 1})
        obj._get_current_timestamp = MagicMock(return_value=1700000000)
        obj.create_agent_attribute = fake_create_agent_attr
        obj._write_kv = fake_write_kv

        with patch.object(
            type(obj), "shared_state", new_callable=PropertyMock, return_value=shared
        ):
            gen = obj._calculate_and_store_apr("a1", "ad1")
            _drive(gen)
            # Should NOT return early; should log stored APR data
            obj.context.logger.info.assert_called()

    def test_none_total_actual_apr_returns_early(self) -> None:
        """Test that total_actual_apr=None triggers early return."""
        obj = _make_behaviour()
        obj.portfolio_data = {"portfolio_value": 100}

        def fake_calc_actual_apr(pv):
            yield
            return {"total_actual_apr": None, "adjusted_apr": 5.0}

        obj.calculate_actual_apr = fake_calc_actual_apr
        obj._create_portfolio_snapshot = MagicMock(return_value={})

        gen = obj._calculate_and_store_apr("a1", "ad1")
        result = _drive(gen)
        assert result is None

    def test_success(self) -> None:
        """Test full success path."""
        obj = _make_behaviour()
        obj.portfolio_data = {"portfolio_value": 100, "volume": 50}
        obj.current_positions = [{"timestamp": 1700000000}]

        shared = MagicMock()
        shared.trading_type = "balanced"
        shared.selected_protocols = ["uniswap"]

        def fake_calc_actual_apr(pv):
            yield
            return {"total_actual_apr": 12.5, "adjusted_apr": 10.0}

        def fake_create_agent_attr(agent_id, attr_def_id, data):
            yield
            return {"id": "attr1"}

        def fake_write_kv(data):
            yield

        obj.calculate_actual_apr = fake_calc_actual_apr
        obj._create_portfolio_snapshot = MagicMock(return_value={"snap": True})
        obj._get_apr_calculation_metrics = MagicMock(return_value={"m": 1})
        obj._get_current_timestamp = MagicMock(return_value=1700000000)
        obj.create_agent_attribute = fake_create_agent_attr
        obj._write_kv = fake_write_kv

        with patch.object(
            type(obj), "shared_state", new_callable=PropertyMock, return_value=shared
        ):
            gen = obj._calculate_and_store_apr("a1", "ad1")
            _drive(gen)
            obj.context.logger.info.assert_called()


class TestConvertDecimals:
    """Tests for _convert_decimals."""

    def test_dict(self) -> None:
        obj = _make_behaviour()
        result = obj._convert_decimals({"a": Decimal("1.5"), "b": "hello"})
        assert result == {"a": 1.5, "b": "hello"}

    def test_list(self) -> None:
        obj = _make_behaviour()
        result = obj._convert_decimals([Decimal("1.5"), "hello", 42])
        assert result == [1.5, "hello", 42]

    def test_decimal(self) -> None:
        obj = _make_behaviour()
        result = obj._convert_decimals(Decimal("3.14"))
        assert result == 3.14

    def test_plain_value(self) -> None:
        obj = _make_behaviour()
        assert obj._convert_decimals(42) == 42
        assert obj._convert_decimals("hello") == "hello"


class TestToDecimal:
    """Tests for _to_decimal."""

    def test_none(self) -> None:
        obj = _make_behaviour()
        assert obj._to_decimal(None) is None

    def test_decimal(self) -> None:
        obj = _make_behaviour()
        d = Decimal("1.5")
        assert obj._to_decimal(d) is d

    def test_float(self) -> None:
        obj = _make_behaviour()
        result = obj._to_decimal(1.5)
        assert result == Decimal("1.5")

    def test_int(self) -> None:
        obj = _make_behaviour()
        result = obj._to_decimal(100)
        assert result == Decimal("100")

    def test_invalid(self) -> None:
        obj = _make_behaviour()
        result = obj._to_decimal("not_a_number")
        assert result is None


class TestCreatePortfolioSnapshot:
    """Tests for _create_portfolio_snapshot."""

    def test_snapshot(self) -> None:
        obj = _make_behaviour()
        obj.portfolio_data = {"value": Decimal("100")}
        obj.current_positions = [{"amount": Decimal("50")}]
        result = obj._create_portfolio_snapshot()
        assert result["portfolio"] == {"value": 100.0}
        assert result["positons"] == [{"amount": 50.0}]


class TestGetAprCalculationMetrics:
    """Tests for _get_apr_calculation_metrics."""

    def test_no_initial_final(self) -> None:
        """Test with None initial and final values."""
        obj = _make_behaviour()
        obj._initial_value = None
        obj._final_value = None
        obj.current_positions = []
        obj._get_first_investment_timestamp = MagicMock(return_value=None)
        result = obj._get_apr_calculation_metrics()
        assert result["initial_value"] is None
        assert result["final_value"] is None
        assert result["f_i_ratio"] is None

    def test_with_values_and_timestamp(self) -> None:
        """Test with valid initial, final, and timestamp."""
        obj = _make_behaviour()
        obj._initial_value = 100
        obj._final_value = 110
        obj.current_positions = [{"timestamp": 1700000000}]
        obj._get_first_investment_timestamp = MagicMock(return_value=1700000000)
        # current_timestamp needs to be far enough for LOW volatility warning
        obj._get_current_timestamp = MagicMock(return_value=1700000000 + 100000)
        result = obj._get_apr_calculation_metrics()
        assert result["initial_value"] == 100.0
        assert result["final_value"] == 110.0
        assert result["f_i_ratio"] is not None
        assert result["volatility_warning"] == "LOW"

    def test_very_short_period(self) -> None:
        """Test with very short investment period (VERY_HIGH volatility)."""
        obj = _make_behaviour()
        obj._initial_value = 100
        obj._final_value = 110
        obj.current_positions = [{"timestamp": 1700000000}]
        obj._get_first_investment_timestamp = MagicMock(return_value=1700000000)
        # Only 10 seconds later - less than 1 minute
        obj._get_current_timestamp = MagicMock(return_value=1700000010)
        result = obj._get_apr_calculation_metrics()
        assert result["volatility_warning"] == "VERY_HIGH"

    def test_short_period(self) -> None:
        """Test with short period (HIGH volatility)."""
        obj = _make_behaviour()
        obj._initial_value = 100
        obj._final_value = 110
        obj.current_positions = [{"timestamp": 1700000000}]
        obj._get_first_investment_timestamp = MagicMock(return_value=1700000000)
        # 30 minutes later
        obj._get_current_timestamp = MagicMock(return_value=1700000000 + 1800)
        result = obj._get_apr_calculation_metrics()
        assert result["volatility_warning"] == "HIGH"

    def test_medium_period(self) -> None:
        """Test with medium period (MEDIUM volatility)."""
        obj = _make_behaviour()
        obj._initial_value = 100
        obj._final_value = 110
        obj.current_positions = [{"timestamp": 1700000000}]
        obj._get_first_investment_timestamp = MagicMock(return_value=1700000000)
        # 5 hours later
        obj._get_current_timestamp = MagicMock(return_value=1700000000 + 18000)
        result = obj._get_apr_calculation_metrics()
        assert result["volatility_warning"] == "MEDIUM"

    def test_zero_initial(self) -> None:
        """Test with zero initial value (no f_i_ratio)."""
        obj = _make_behaviour()
        obj._initial_value = 0
        obj._final_value = 110
        obj.current_positions = []
        obj._get_first_investment_timestamp = MagicMock(return_value=None)
        result = obj._get_apr_calculation_metrics()
        assert result["f_i_ratio"] is None


class TestSignMessage:
    """Tests for sign_message."""

    def test_sign_success(self) -> None:
        obj = _make_behaviour()

        def fake_get_signature(msg_bytes):
            yield
            return "0xabcdef"

        obj.get_signature = fake_get_signature
        gen = obj.sign_message("hello")
        result = _drive(gen)
        assert result == "abcdef"

    def test_sign_failure(self) -> None:
        obj = _make_behaviour()

        def fake_get_signature(msg_bytes):
            yield
            return None

        obj.get_signature = fake_get_signature
        gen = obj.sign_message("hello")
        result = _drive(gen)
        assert result is None


class TestCalculateActualApr:
    """Tests for calculate_actual_apr."""

    def test_no_valid_portfolio_data(self) -> None:
        obj = _make_behaviour()
        obj.portfolio_data = None
        gen = obj.calculate_actual_apr(100)
        result = _drive(gen)
        assert result is None

    def test_no_initial_investment(self) -> None:
        obj = _make_behaviour()
        obj.portfolio_data = {"portfolio_value": 100, "initial_investment": None}

        def fake_adjust(*a, **kw):
            yield

        obj._adjust_apr_for_eth_price = fake_adjust
        gen = obj.calculate_actual_apr(100)
        result = _drive(gen)
        assert result is None

    def test_success(self) -> None:
        obj = _make_behaviour()
        obj.portfolio_data = {"portfolio_value": 100, "initial_investment": 90}
        obj.current_positions = [{"timestamp": 1700000000}]
        obj._get_current_timestamp = MagicMock(return_value=1700000000 + 100000)

        def fake_adjust(*a, **kw):
            yield

        obj._adjust_apr_for_eth_price = fake_adjust
        gen = obj.calculate_actual_apr(100)
        result = _drive(gen)
        assert result is not None
        assert "total_actual_apr" in result


class TestGetStoredInitialInvestment:
    """Tests for get_stored_initial_investment."""

    def test_no_portfolio_data(self) -> None:
        obj = _make_behaviour()
        obj.portfolio_data = None
        assert obj.get_stored_initial_investment() is None

    def test_no_initial_investment(self) -> None:
        obj = _make_behaviour()
        obj.portfolio_data = {"other": "data"}
        assert obj.get_stored_initial_investment() is None

    def test_has_initial_investment(self) -> None:
        obj = _make_behaviour()
        obj.portfolio_data = {"initial_investment": 100}
        assert obj.get_stored_initial_investment() == 100.0


class TestHasValidPortfolioData:
    """Tests for _has_valid_portfolio_data."""

    def test_no_portfolio_data(self) -> None:
        obj = _make_behaviour()
        obj.portfolio_data = None
        assert obj._has_valid_portfolio_data() is False

    def test_no_portfolio_value(self) -> None:
        obj = _make_behaviour()
        obj.portfolio_data = {"other": "data"}
        assert obj._has_valid_portfolio_data() is False

    def test_valid(self) -> None:
        obj = _make_behaviour()
        obj.portfolio_data = {"portfolio_value": 100}
        assert obj._has_valid_portfolio_data() is True


class TestGetFirstInvestmentTimestamp:
    """Tests for _get_first_investment_timestamp."""

    def test_timestamp_field(self) -> None:
        obj = _make_behaviour()
        obj.current_positions = [{"timestamp": 1700000000}]
        assert obj._get_first_investment_timestamp() == 1700000000

    def test_enter_timestamp_field(self) -> None:
        obj = _make_behaviour()
        obj.current_positions = [{"enter_timestamp": 1700000000}]
        assert obj._get_first_investment_timestamp() == 1700000000

    def test_no_timestamp_fallback(self) -> None:
        obj = _make_behaviour()
        obj.current_positions = [{"no_ts": True}]
        obj._get_current_timestamp = MagicMock(return_value=1700000000.5)
        result = obj._get_first_investment_timestamp()
        assert result == 1700000000


class TestCalculateApr:
    """Tests for _calculate_apr."""

    def test_zero_final_value(self) -> None:
        obj = _make_behaviour()
        obj._final_value = 0
        obj._initial_value = 100
        result = {}
        ret = obj._calculate_apr(1700000000, 1699900000, result)
        assert ret == 0.0

    def test_zero_initial_value(self) -> None:
        obj = _make_behaviour()
        obj._final_value = 100
        obj._initial_value = 0
        result = {}
        ret = obj._calculate_apr(1700000000, 1699900000, result)
        assert ret == 0.0

    def test_negative_apr(self) -> None:
        """Test negative APR calculation (loss scenario)."""
        obj = _make_behaviour()
        obj._final_value = 90
        obj._initial_value = 100
        result = {}
        # Short period to amplify negative APR
        obj._calculate_apr(1700000000, 1700000000 - 3600, result)
        # With negative APR, the code recalculates as (final/initial - 1) * 100
        assert "total_actual_apr" in result

    def test_positive_apr(self) -> None:
        obj = _make_behaviour()
        obj._final_value = 110
        obj._initial_value = 100
        result = {}
        obj._calculate_apr(1700000000, 1700000000 - 86400, result)
        assert result.get("total_actual_apr") is not None
        assert result["total_actual_apr"] > 0

    def test_very_short_investment(self) -> None:
        """Test APR with very short investment period."""
        obj = _make_behaviour()
        obj._final_value = 110
        obj._initial_value = 100
        result = {}
        obj._calculate_apr(1700000000, 1700000000 - 10, result)
        assert "total_actual_apr" in result

    def test_short_investment_under_1h(self) -> None:
        """Test APR with investment period under 1 hour."""
        obj = _make_behaviour()
        obj._final_value = 110
        obj._initial_value = 100
        result = {}
        obj._calculate_apr(1700000000, 1700000000 - 1800, result)
        assert "total_actual_apr" in result

    def test_apr_is_zero(self) -> None:
        """Test when APR would be exactly zero (no change)."""
        obj = _make_behaviour()
        obj._final_value = 100
        obj._initial_value = 100
        result = {}
        obj._calculate_apr(1700000000, 1700000000 - 86400, result)
        # APR is 0, so total_actual_apr should not be set (falsy check fails)
        assert "total_actual_apr" not in result


class TestAdjustAprForEthPrice:
    """Tests for _adjust_apr_for_eth_price."""

    def test_both_prices_none(self) -> None:
        obj = _make_behaviour()

        def fake_fetch_zero(*a, **kw):
            yield
            return None

        def fake_fetch_hist(*a, **kw):
            yield
            return None

        obj._fetch_zero_address_price = fake_fetch_zero
        obj._fetch_historical_token_price = fake_fetch_hist

        result = {"total_actual_apr": 10.0}
        gen = obj._adjust_apr_for_eth_price(result, 1700000000)
        _drive(gen)
        assert "adjusted_apr" not in result

    def test_success(self) -> None:
        obj = _make_behaviour()

        def fake_fetch_zero(*a, **kw):
            yield
            return 2000.0

        def fake_fetch_hist(*a, **kw):
            yield
            return 1800.0

        obj._fetch_zero_address_price = fake_fetch_zero
        obj._fetch_historical_token_price = fake_fetch_hist

        result = {"total_actual_apr": 10.0}
        gen = obj._adjust_apr_for_eth_price(result, 1700000000)
        _drive(gen)
        assert "adjusted_apr" in result
        assert "adjustment_factor" in result

    def test_start_price_zero(self) -> None:
        """Test when start ETH price is zero."""
        obj = _make_behaviour()

        def fake_fetch_zero(*a, **kw):
            yield
            return 2000.0

        def fake_fetch_hist(*a, **kw):
            yield
            return 0.0

        obj._fetch_zero_address_price = fake_fetch_zero
        obj._fetch_historical_token_price = fake_fetch_hist

        result = {"total_actual_apr": 10.0}
        gen = obj._adjust_apr_for_eth_price(result, 1700000000)
        _drive(gen)
        assert "adjusted_apr" not in result

    def test_conversion_failure(self) -> None:
        """Test when one price can't be converted to Decimal."""
        obj = _make_behaviour()

        def fake_fetch_zero(*a, **kw):
            yield
            return "not_a_number"

        def fake_fetch_hist(*a, **kw):
            yield
            return 1800.0

        obj._fetch_zero_address_price = fake_fetch_zero
        obj._fetch_historical_token_price = fake_fetch_hist

        result = {"total_actual_apr": 10.0}
        gen = obj._adjust_apr_for_eth_price(result, 1700000000)
        _drive(gen)
        # conversion fails so adjustment is skipped
        assert "adjusted_apr" not in result

    def test_no_total_apr_in_result(self) -> None:
        """Test when total_actual_apr is not in result."""
        obj = _make_behaviour()

        def fake_fetch_zero(*a, **kw):
            yield
            return 2000.0

        def fake_fetch_hist(*a, **kw):
            yield
            return 1800.0

        obj._fetch_zero_address_price = fake_fetch_zero
        obj._fetch_historical_token_price = fake_fetch_hist

        result = {}
        gen = obj._adjust_apr_for_eth_price(result, 1700000000)
        _drive(gen)
        assert "adjusted_apr" not in result


class TestReadInvestingPaused:
    """Tests for _read_investing_paused."""

    def test_result_none(self) -> None:
        obj = _make_behaviour()

        def fake_read_kv(keys):
            yield
            return None

        obj._read_kv = fake_read_kv
        gen = obj._read_investing_paused()
        result = _drive(gen)
        assert result is False

    def test_value_none(self) -> None:
        obj = _make_behaviour()

        def fake_read_kv(keys):
            yield
            return {"investing_paused": None}

        obj._read_kv = fake_read_kv
        gen = obj._read_investing_paused()
        result = _drive(gen)
        assert result is False

    def test_value_true(self) -> None:
        obj = _make_behaviour()

        def fake_read_kv(keys):
            yield
            return {"investing_paused": "true"}

        obj._read_kv = fake_read_kv
        gen = obj._read_investing_paused()
        result = _drive(gen)
        assert result is True

    def test_value_false(self) -> None:
        obj = _make_behaviour()

        def fake_read_kv(keys):
            yield
            return {"investing_paused": "false"}

        obj._read_kv = fake_read_kv
        gen = obj._read_investing_paused()
        result = _drive(gen)
        assert result is False

    def test_exception(self) -> None:
        obj = _make_behaviour()

        def fake_read_kv(keys):
            raise ValueError("test")
            yield  # noqa

        obj._read_kv = fake_read_kv
        gen = obj._read_investing_paused()
        result = _drive(gen)
        assert result is False

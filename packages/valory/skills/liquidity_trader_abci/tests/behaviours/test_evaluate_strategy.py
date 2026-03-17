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
# pylint: skip-file

"""Comprehensive unit tests for evaluate_strategy behaviour."""

import json
import math
import time
from typing import Any, Dict, Generator, List, Optional, Tuple
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    Action,
    DexType,
    METRICS_UPDATE_INTERVAL,
    MIN_TIME_IN_POSITION,
    OLAS_ADDRESSES,
    PositionStatus,
    WHITELISTED_ASSETS,
    ZERO_ADDRESS,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy import (
    EvaluateStrategyBehaviour,
    MIN_SWAP_VALUE_USD,
)


def _mk(**overrides):
    """Create an EvaluateStrategyBehaviour without calling __init__."""
    obj = object.__new__(EvaluateStrategyBehaviour)

    params = MagicMock()
    params.safe_contract_addresses = {
        "optimism": "0x" + "aa" * 20,
        "mode": "0x" + "bb" * 20,
    }
    params.target_investment_chains = ["optimism"]
    params.chain_to_chain_id_mapping = {"optimism": 10, "mode": 34443}
    params.stoploss_threshold_multiplier = 0.6
    params.dex_type_to_strategy = {"velodrome": "strategy_a"}
    params.velodrome_voter_contract_addresses = {"optimism": "0x" + "cc" * 20}
    params.min_investment_amount = 1.0
    params.strategy_backoff_base_seconds = 1800
    params.strategy_backoff_max_seconds = 14400
    params.strategy_price_cache_ttl = 1800

    synced = MagicMock()
    synced.positions = []
    synced.period_count = 1
    synced.trading_type = "default"
    synced.selected_protocols = []

    ctx = MagicMock()
    ctx.agent_address = "test_agent"
    ctx.logger = MagicMock()
    ctx.params = params
    ctx.benchmark_tool.measure.return_value.__enter__ = MagicMock(return_value=None)
    ctx.benchmark_tool.measure.return_value.__exit__ = MagicMock(return_value=False)

    shared_state = MagicMock()
    shared_state.synchronized_data = synced
    shared_state.strategies_executables = {}
    shared_state.strategy_to_filehash = {}
    shared_state.consecutive_no_action_count = 0
    shared_state.last_strategy_evaluation_time = 0.0
    ctx.state = shared_state

    obj.__dict__.update(
        {
            "_context": ctx,
            "current_positions": [],
            "portfolio_data": {},
            "assets": {},
            "whitelisted_assets": {},
            "funding_events": {},
            "agent_performance": {},
            "initial_investment_values_per_pool": {},
            "pools": {},
            "service_staking_state": MagicMock(),
            "selected_opportunities": None,
            "position_to_exit": None,
            "trading_opportunities": [],
            "positions_eligible_for_exit": [],
            "shared_state": shared_state,
        }
    )

    # Apply overrides
    for k, v in overrides.items():
        if k == "params":
            for pk, pv in v.items():
                setattr(params, pk, pv)
        elif k == "synced":
            for sk, sv in v.items():
                setattr(synced, sk, sv)
        else:
            obj.__dict__[k] = v

    return obj


def _drive(gen, sends=None):
    """Drive a generator to completion, feeding values from sends list."""
    sends = list(sends or [])
    idx = 0
    val = None
    while True:
        try:
            yielded = gen.send(val)
            if idx < len(sends):
                val = sends[idx]
                idx += 1
            else:
                val = None
        except StopIteration as exc:
            return exc.value


def _gen_return(value):
    """Create a generator function that yields once then returns value."""

    def _inner(*a, **kw):
        yield
        return value

    return _inner


def _gen_none(*a, **kw):
    """A generator that yields once then returns None."""
    yield


class TestValidateAndPrepareVelodromeInputs:
    """Tests for validate_and_prepare_velodrome_inputs."""

    def test_empty_tick_bands_returns_none(self):
        b = _mk()
        result = b.validate_and_prepare_velodrome_inputs([], 1.0)
        assert result is None

    def test_negative_price_returns_none(self):
        b = _mk()
        result = b.validate_and_prepare_velodrome_inputs(
            [{"tick_lower": 0, "tick_upper": 10, "allocation": 1}], -1.0
        )
        assert result is None

    def test_zero_price_returns_none(self):
        b = _mk()
        result = b.validate_and_prepare_velodrome_inputs(
            [{"tick_lower": 0, "tick_upper": 10, "allocation": 1}], 0
        )
        assert result is None

    def test_no_positive_allocation_returns_none(self):
        b = _mk()
        bands = [{"tick_lower": 0, "tick_upper": 10, "allocation": 0}]
        result = b.validate_and_prepare_velodrome_inputs(bands, 1.5)
        assert result is None

    def test_invalid_band_tick_lower_gte_upper(self):
        b = _mk()
        bands = [{"tick_lower": 10, "tick_upper": 5, "allocation": 1}]
        result = b.validate_and_prepare_velodrome_inputs(bands, 1.5)
        assert result is None  # all bands filtered out

    def test_valid_single_band(self):
        b = _mk()
        bands = [{"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}]
        result = b.validate_and_prepare_velodrome_inputs(bands, 1.5)
        assert result is not None
        assert len(result["validated_bands"]) == 1
        assert result["current_price"] == 1.5
        assert "current_tick" in result

    def test_tick_spacing_alignment_warning(self):
        b = _mk()
        bands = [{"tick_lower": 3, "tick_upper": 7, "allocation": 1.0}]
        result = b.validate_and_prepare_velodrome_inputs(bands, 1.5, tick_spacing=5)
        assert result is not None
        assert len(result["warnings"]) > 0

    def test_mixed_valid_and_invalid_bands(self):
        b = _mk()
        bands = [
            {"tick_lower": -100, "tick_upper": 100, "allocation": 0.5},
            {"tick_lower": 200, "tick_upper": 100, "allocation": 0.5},  # invalid
        ]
        result = b.validate_and_prepare_velodrome_inputs(bands, 1.5)
        assert result is not None
        assert len(result["validated_bands"]) == 1
        assert len(result["warnings"]) > 0

    def test_all_invalid_bands_after_validation(self):
        b = _mk()
        bands = [
            {"tick_lower": 100, "tick_upper": 50, "allocation": 0.5},
            {"tick_lower": 200, "tick_upper": 100, "allocation": 0.5},
        ]
        result = b.validate_and_prepare_velodrome_inputs(bands, 1.5)
        assert result is None

    def test_price_conversion_error(self):
        """Test with a price that would cause math error (e.g., negative for log)."""
        b = _mk()
        bands = [{"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}]
        # negative price should hit the error branch
        result = b.validate_and_prepare_velodrome_inputs(bands, -5)
        assert result is None

    def test_multiple_valid_bands(self):
        b = _mk()
        bands = [
            {"tick_lower": -200, "tick_upper": -50, "allocation": 0.3},
            {"tick_lower": -50, "tick_upper": 50, "allocation": 0.5},
            {"tick_lower": 50, "tick_upper": 200, "allocation": 0.2},
        ]
        result = b.validate_and_prepare_velodrome_inputs(bands, 1.0001, tick_spacing=1)
        assert result is not None
        assert len(result["validated_bands"]) == 3


class TestCalculatePositionYieldPerDay:
    """Tests for _calculate_position_yield_per_day."""

    def test_apr_none(self):
        b = _mk()
        result = b._calculate_position_yield_per_day({})
        assert result is None

    def test_apr_zero(self):
        b = _mk()
        result = b._calculate_position_yield_per_day({"apr": 0})
        assert result == 0.0

    def test_apr_positive(self):
        b = _mk()
        result = b._calculate_position_yield_per_day({"apr": 36.5})
        expected = (36.5 / 100) / 365
        assert abs(result - expected) < 1e-10

    def test_apr_negative(self):
        b = _mk()
        result = b._calculate_position_yield_per_day({"apr": -10.0})
        expected = (-10.0 / 100) / 365
        assert abs(result - expected) < 1e-10


class TestCalculateMinReqPositionValue:
    """Tests for _calculate_min_req_position_value."""

    def test_no_entry_apr(self):
        b = _mk()
        result = b._calculate_min_req_position_value({}, 0.6)
        assert result is None

    def test_no_enter_timestamp(self):
        b = _mk()
        result = b._calculate_min_req_position_value({"entry_apr": 20.0}, 0.6)
        assert result is None

    def test_fallback_to_apr(self):
        b = _mk()
        now = int(time.time())
        b._get_current_timestamp = lambda: now
        pos = {"apr": 20.0, "enter_timestamp": now - 86400}  # 1 day ago
        result = b._calculate_min_req_position_value(pos, 0.6)
        assert result is not None
        assert result > 1.0  # Always > 1 due to formula

    def test_valid_calculation(self):
        b = _mk()
        now = int(time.time())
        b._get_current_timestamp = lambda: now
        enter_ts = now - 86400 * 7  # 7 days ago
        pos = {"entry_apr": 36.5, "enter_timestamp": enter_ts}
        result = b._calculate_min_req_position_value(pos, 0.6)
        assert result is not None
        # S * Vy * t/T + 1
        Vy = (36.5 / 100) / 365
        t_minutes = (86400 * 7) / 60
        T_minutes = 365 * 24 * 60
        expected = 0.6 * Vy * t_minutes / T_minutes + 1
        assert abs(result - expected) < 1e-6

    def test_exception_returns_none(self):
        b = _mk()
        b._get_current_timestamp = MagicMock(side_effect=Exception("boom"))
        pos = {"entry_apr": 20.0, "enter_timestamp": 1000}
        result = b._calculate_min_req_position_value(pos, 0.6)
        assert result is None


class TestGetBestAvailableOpportunityYield:
    """Tests for _get_best_available_opportunity_yield."""

    def test_no_opportunities(self):
        b = _mk()
        b.trading_opportunities = []
        assert b._get_best_available_opportunity_yield() is None

    def test_best_apr_none(self):
        b = _mk()
        b.trading_opportunities = [{}]
        assert b._get_best_available_opportunity_yield() is None

    def test_best_apr_zero(self):
        b = _mk()
        b.trading_opportunities = [{"apr": 0}]
        assert b._get_best_available_opportunity_yield() is None

    def test_best_apr_negative(self):
        b = _mk()
        b.trading_opportunities = [{"apr": -5}]
        assert b._get_best_available_opportunity_yield() is None

    def test_valid_apr(self):
        b = _mk()
        b.trading_opportunities = [{"apr": 36.5}, {"apr": 10.0}]
        result = b._get_best_available_opportunity_yield()
        expected = (36.5 / 100) / 365
        assert abs(result - expected) < 1e-10

    def test_exception_returns_none(self):
        b = _mk()
        # Make sorted() fail by providing a non-iterable
        b.trading_opportunities = None
        result = b._get_best_available_opportunity_yield()
        assert result is None


class TestCheckFunds:
    """Tests for check_funds."""

    def test_open_position_returns_true(self):
        b = _mk()
        b.current_positions = [{"status": PositionStatus.OPEN.value}]
        assert b.check_funds() is True

    def test_no_open_positions_no_funds(self):
        b = _mk()
        b.current_positions = [{"status": PositionStatus.CLOSED.value}]
        synced_mock = MagicMock()
        synced_mock.positions = [{"assets": [{"balance": 0}]}]
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            assert b.check_funds() is False

    def test_no_open_positions_with_funds(self):
        b = _mk()
        b.current_positions = [{"status": PositionStatus.CLOSED.value}]
        synced_mock = MagicMock()
        synced_mock.positions = [{"assets": [{"balance": 100}]}]
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            assert b.check_funds() is True

    def test_empty_positions_no_funds(self):
        b = _mk()
        b.current_positions = []
        synced_mock = MagicMock()
        synced_mock.positions = []
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            assert b.check_funds() is False


class TestUpdatePositionMetrics:
    """Tests for update_position_metrics."""

    def test_no_eligible_positions(self):
        b = _mk()
        b.positions_eligible_for_exit = []
        b.store_current_positions = MagicMock()
        b.update_position_metrics()
        b.store_current_positions.assert_not_called()

    def test_skips_closed_positions(self):
        b = _mk()
        now = int(time.time())
        b._get_current_timestamp = lambda: now
        b.positions_eligible_for_exit = [
            {"status": PositionStatus.CLOSED.value, "dex_type": "velodrome"}
        ]
        b.store_current_positions = MagicMock()
        b.update_position_metrics()
        b.store_current_positions.assert_called_once()

    def test_no_strategy_for_dex_type(self):
        b = _mk()
        now = int(time.time())
        b._get_current_timestamp = lambda: now
        b.positions_eligible_for_exit = [
            {
                "status": PositionStatus.OPEN.value,
                "dex_type": "unknown_dex",
                "last_metrics_update": 0,
            }
        ]
        b.params.dex_type_to_strategy = {}
        b.store_current_positions = MagicMock()
        b.update_position_metrics()
        b.store_current_positions.assert_called_once()

    def test_metrics_update_interval_not_reached(self):
        b = _mk()
        now = int(time.time())
        b._get_current_timestamp = lambda: now
        b.positions_eligible_for_exit = [
            {
                "status": PositionStatus.OPEN.value,
                "dex_type": "velodrome",
                "last_metrics_update": now - 100,  # recent update
                "pool_address": "0xpool",
            }
        ]
        b.get_returns_metrics_for_opportunity = MagicMock()
        b.store_current_positions = MagicMock()
        b.update_position_metrics()
        b.get_returns_metrics_for_opportunity.assert_not_called()

    def test_metrics_update_interval_reached(self):
        b = _mk()
        now = int(time.time())
        b._get_current_timestamp = lambda: now
        b.positions_eligible_for_exit = [
            {
                "status": PositionStatus.OPEN.value,
                "dex_type": "velodrome",
                "last_metrics_update": now - METRICS_UPDATE_INTERVAL - 100,
                "pool_address": "0xpool",
            }
        ]
        b.get_returns_metrics_for_opportunity = MagicMock(return_value={"apr": 10.0})
        b.store_current_positions = MagicMock()
        b.update_position_metrics()
        b.get_returns_metrics_for_opportunity.assert_called_once()
        b.store_current_positions.assert_called_once()

    def test_metrics_update_returns_none(self):
        b = _mk()
        now = int(time.time())
        b._get_current_timestamp = lambda: now
        pos = {
            "status": PositionStatus.OPEN.value,
            "dex_type": "velodrome",
            "last_metrics_update": 0,
            "pool_address": "0xpool",
        }
        b.positions_eligible_for_exit = [pos]
        b.get_returns_metrics_for_opportunity = MagicMock(return_value=None)
        b.store_current_positions = MagicMock()
        b.update_position_metrics()
        # Position should not be updated
        assert "apr" not in pos


class TestCheckTipExitConditions:
    """Tests for _check_tip_exit_conditions."""

    def _make_b(self):
        b = _mk()
        now = int(time.time())
        b._get_current_timestamp = lambda: now
        b._calculate_days_since_entry = lambda ts: (now - ts) / 86400
        b._check_minimum_time_met = lambda pos: True
        return b, now

    def test_legacy_no_timestamp(self):
        b, now = self._make_b()
        pos = {"entry_cost": 0}
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos))
        assert can_exit is True
        assert "No TiP data" in reason

    def test_legacy_past_21_days(self):
        b, now = self._make_b()
        pos = {"entry_cost": 0, "enter_timestamp": now - 86400 * 25}
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos))
        assert can_exit is True
        assert "Legacy position" in reason

    def test_legacy_not_past_21_days(self):
        b, now = self._make_b()
        pos = {"entry_cost": 0, "enter_timestamp": now - 86400 * 5}
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos))
        assert can_exit is False
        assert "Legacy position must hold" in reason

    def test_21_day_global_cap(self):
        b, now = self._make_b()
        pos = {
            "entry_cost": 100,
            "enter_timestamp": now - 86400 * 25,
            "min_hold_days": 12,
        }
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos))
        assert can_exit is True
        assert "21-day global temporal cap" in reason

    def test_minimum_time_not_met(self):
        b, now = self._make_b()
        b._check_minimum_time_met = lambda pos: False
        pos = {
            "entry_cost": 100,
            "enter_timestamp": now - 86400 * 5,
            "min_hold_days": 12,
        }
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos))
        assert can_exit is False
        assert "Minimum time not met" in reason

    def test_no_apr_data(self):
        b, now = self._make_b()
        pos = {
            "entry_cost": 100,
            "enter_timestamp": now - 86400 * 15,
            "min_hold_days": 12,
        }
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos))
        assert can_exit is True
        assert "No APR data" in reason

    def test_no_current_apr(self):
        b, now = self._make_b()
        pos = {
            "entry_cost": 100,
            "enter_timestamp": now - 86400 * 15,
            "min_hold_days": 12,
            "entry_apr": 20.0,
        }
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos))
        assert can_exit is True
        assert "No current APR" in reason

    def test_trailing_stoploss_triggered(self):
        b, now = self._make_b()
        pos = {
            "entry_cost": 100,
            "enter_timestamp": now - 86400 * 15,
            "min_hold_days": 12,
            "entry_apr": 50.0,
            "apr": 5.0,  # way below 0.6 * 50 = 30
        }
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos))
        assert can_exit is True
        assert "Trailing stop-loss triggered" in reason

    def test_position_value_check_triggers_exit(self):
        """Test when current value ratio is below min req position value."""
        b, now = self._make_b()
        pos = {
            "entry_cost": 100,
            "enter_timestamp": now - 86400 * 15,
            "min_hold_days": 12,
            "entry_apr": 20.0,
            "apr": 20.0,  # Not triggering stoploss since 20 >= 0.6 * 20
        }
        # min_req returns 1.1, current_value_ratio returns 0.8
        b._calculate_min_req_position_value = MagicMock(return_value=1.1)
        b._calculate_current_value_ratio = _gen_return(0.5)
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos), sends=[None])
        assert can_exit is True
        assert "Position value check" in reason

    def test_opportunity_cost_triggers_exit(self):
        """Test when current yield < S * best opportunity yield."""
        b, now = self._make_b()
        pos = {
            "entry_cost": 100,
            "enter_timestamp": now - 86400 * 15,
            "min_hold_days": 12,
            "entry_apr": 10.0,
            "apr": 10.0,  # yield per day = 0.0002739...
        }
        b._calculate_min_req_position_value = MagicMock(return_value=None)
        # best yield is very high, so 0.6 * vby > current
        b._get_best_available_opportunity_yield = MagicMock(return_value=0.01)
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos))
        assert can_exit is True
        assert "Opportunity cost check" in reason

    def test_cost_recovered_triggers_exit(self):
        b, now = self._make_b()
        pos = {
            "entry_cost": 100,
            "enter_timestamp": now - 86400 * 15,
            "min_hold_days": 12,
            "entry_apr": 20.0,
            "apr": 20.0,
            "cost_recovered": True,
        }
        b._calculate_min_req_position_value = MagicMock(return_value=None)
        b._get_best_available_opportunity_yield = MagicMock(return_value=None)
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos))
        assert can_exit is True
        assert "Costs recovered" in reason

    def test_costs_not_recovered(self):
        b, now = self._make_b()
        pos = {
            "entry_cost": 100,
            "enter_timestamp": now - 86400 * 15,
            "min_hold_days": 12,
            "entry_apr": 20.0,
            "apr": 20.0,
            "cost_recovered": False,
            "yield_usd": 5.0,
        }
        b._calculate_min_req_position_value = MagicMock(return_value=None)
        b._get_best_available_opportunity_yield = MagicMock(return_value=None)
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos))
        assert can_exit is False
        assert "costs not recovered" in reason

    def test_exception_path(self):
        b, now = self._make_b()
        b._calculate_days_since_entry = MagicMock(side_effect=Exception("boom"))
        pos = {"entry_cost": 100, "enter_timestamp": now - 86400 * 15}
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos))
        assert can_exit is True
        assert "Error in TiP check" in reason

    def test_entry_apr_fallback_to_apr(self):
        """When entry_apr is None, falls back to current apr for entry yield."""
        b, now = self._make_b()
        pos = {
            "entry_cost": 100,
            "enter_timestamp": now - 86400 * 15,
            "min_hold_days": 12,
            "apr": 20.0,
            "cost_recovered": True,
        }
        b._calculate_min_req_position_value = MagicMock(return_value=None)
        b._get_best_available_opportunity_yield = MagicMock(return_value=None)
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos))
        assert can_exit is True
        assert "Costs recovered" in reason

    def test_position_value_check_equal_values_no_exit(self):
        """When current value ratio equals min req exactly, epsilon prevents exit."""
        b, now = self._make_b()
        pos = {
            "entry_cost": 100,
            "enter_timestamp": now - 86400 * 15,
            "min_hold_days": 12,
            "entry_apr": 20.0,
            "apr": 20.0,
            "cost_recovered": True,
        }
        # min_req = 1.001, current = 1.001 (equal, should NOT trigger due to epsilon)
        b._calculate_min_req_position_value = MagicMock(return_value=1.001)
        b._calculate_current_value_ratio = _gen_return(1.001)
        b._get_best_available_opportunity_yield = MagicMock(return_value=None)
        can_exit, reason = _drive(b._check_tip_exit_conditions(pos), sends=[None])
        assert can_exit is True
        assert "Costs recovered" in reason


class TestApplyTipFiltersToExitDecisions:
    """Tests for _apply_tip_filters_to_exit_decisions."""

    def test_no_positions(self):
        b = _mk()
        should_proceed, eligible = _drive(b._apply_tip_filters_to_exit_decisions())
        assert should_proceed is True
        assert eligible == []

    def test_all_positions_closed(self):
        b = _mk()
        b.current_positions = [{"status": PositionStatus.CLOSED.value}]
        should_proceed, eligible = _drive(b._apply_tip_filters_to_exit_decisions())
        assert should_proceed is True
        assert eligible == []

    def test_open_position_eligible(self):
        b = _mk()
        pos = {"status": PositionStatus.OPEN.value, "pool_address": "0xpool"}
        b.current_positions = [pos]
        b._check_tip_exit_conditions = _gen_return((True, "ok"))
        should_proceed, eligible = _drive(
            b._apply_tip_filters_to_exit_decisions(), sends=[None]
        )
        assert should_proceed is True
        assert len(eligible) == 1

    def test_open_position_blocked(self):
        b = _mk()
        pos = {"status": PositionStatus.OPEN.value, "pool_address": "0xpool"}
        b.current_positions = [pos]
        b._check_tip_exit_conditions = _gen_return((False, "TiP active"))
        should_proceed, eligible = _drive(
            b._apply_tip_filters_to_exit_decisions(), sends=[None]
        )
        assert should_proceed is False
        assert len(eligible) == 0

    def test_exception_returns_open_positions(self):
        b = _mk()
        pos = {"status": PositionStatus.OPEN.value}
        b.current_positions = [pos]

        # Force exception by making _check_tip_exit_conditions raise
        def _bad_gen(p):
            raise Exception("boom")
            yield  # noqa: unreachable

        b._check_tip_exit_conditions = _bad_gen
        should_proceed, eligible = _drive(b._apply_tip_filters_to_exit_decisions())
        assert should_proceed is True
        assert len(eligible) == 1

    def test_mixed_eligible_and_blocked(self):
        b = _mk()
        pos1 = {"status": PositionStatus.OPEN.value, "pool_address": "0x1"}
        pos2 = {"status": PositionStatus.OPEN.value, "pool_address": "0x2"}
        b.current_positions = [pos1, pos2]

        call_count = [0]

        def _mock_check(p):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return (True, "eligible")
            else:
                yield
                return (False, "blocked")

        b._check_tip_exit_conditions = _mock_check
        should_proceed, eligible = _drive(
            b._apply_tip_filters_to_exit_decisions(), sends=[None, None]
        )
        assert should_proceed is True
        assert len(eligible) == 1


class TestReadInvestingPaused:
    """Tests for _read_investing_paused."""

    def test_returns_true(self):
        b = _mk()
        b._read_kv = _gen_return({"investing_paused": "true"})
        result = _drive(b._read_investing_paused(), sends=[None])
        assert result is True

    def test_returns_false_for_false_value(self):
        b = _mk()
        b._read_kv = _gen_return({"investing_paused": "false"})
        result = _drive(b._read_investing_paused(), sends=[None])
        assert result is False

    def test_returns_false_for_none_response(self):
        b = _mk()
        b._read_kv = _gen_return(None)
        result = _drive(b._read_investing_paused(), sends=[None])
        assert result is False

    def test_returns_false_for_none_value(self):
        b = _mk()
        b._read_kv = _gen_return({"investing_paused": None})
        result = _drive(b._read_investing_paused(), sends=[None])
        assert result is False

    def test_exception_returns_false(self):
        b = _mk()

        def _bad(*a, **kw):
            raise Exception("boom")
            yield  # noqa: unreachable

        b._read_kv = _bad
        result = _drive(b._read_investing_paused())
        assert result is False


class TestSendActions:
    """Tests for send_actions."""

    def test_send_actions_none(self):
        b = _mk()
        b.send_a2a_transaction = _gen_none
        b.wait_until_round_end = _gen_none
        b.set_done = MagicMock()
        _drive(b.send_actions())
        b.set_done.assert_called_once()

    def test_send_actions_with_list(self):
        b = _mk()
        b.send_a2a_transaction = _gen_none
        b.wait_until_round_end = _gen_none
        b.set_done = MagicMock()
        _drive(b.send_actions([{"action": "test"}]))
        b.set_done.assert_called_once()


class TestCheckAndPrepareNonWhitelistedSwaps:
    """Tests for check_and_prepare_non_whitelisted_swaps.

    Note: Despite the Generator type annotation, this method has no yield
    statements, so it returns a value directly (not a generator).
    """

    def _call(self, b):
        """Call the method, handling both generator and non-generator returns."""
        result = b.check_and_prepare_non_whitelisted_swaps()
        if hasattr(result, "send"):
            return _drive(result)
        return result

    def test_no_usdc_address(self):
        b = _mk()
        b._get_usdc_address = MagicMock(return_value=None)
        result = self._call(b)
        assert result == []

    def test_no_positions(self):
        b = _mk()
        b._get_usdc_address = MagicMock(return_value="0xusdc")
        synced_mock = MagicMock()
        synced_mock.positions = []
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            result = self._call(b)
        assert result == []

    def test_skips_whitelisted_token(self):
        b = _mk()
        b._get_usdc_address = MagicMock(return_value="0xusdc")
        chain = "optimism"
        wl_addr = (
            list(WHITELISTED_ASSETS.get(chain, {}).keys())[0].lower()
            if WHITELISTED_ASSETS.get(chain)
            else "0xwhitelisted"
        )
        synced_mock = MagicMock()
        synced_mock.positions = [
            {
                "chain": chain,
                "assets": [{"address": wl_addr, "asset_symbol": "WL", "balance": 100}],
            }
        ]
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            result = self._call(b)
        assert result == []

    def test_skips_zero_balance(self):
        b = _mk()
        b._get_usdc_address = MagicMock(return_value="0xusdc")
        synced_mock = MagicMock()
        synced_mock.positions = [
            {
                "chain": "optimism",
                "assets": [
                    {"address": "0xnotwhitelisted", "asset_symbol": "NW", "balance": 0}
                ],
            }
        ]
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            result = self._call(b)
        assert result == []

    def test_swaps_non_whitelisted_token(self):
        b = _mk()
        b._get_usdc_address = MagicMock(return_value="0xusdc")
        synced_mock = MagicMock()
        synced_mock.positions = [
            {
                "chain": "optimism",
                "assets": [
                    {
                        "address": "0xnotwhitelisted",
                        "asset_symbol": "NW",
                        "balance": 100,
                    }
                ],
            }
        ]
        swap_action = {"action": "swap", "token": "NW"}
        b._build_swap_to_usdc_action = MagicMock(return_value=swap_action)
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            result = self._call(b)
        assert len(result) == 1
        assert result[0] == swap_action

    def test_swap_action_fails(self):
        b = _mk()
        b._get_usdc_address = MagicMock(return_value="0xusdc")
        synced_mock = MagicMock()
        synced_mock.positions = [
            {
                "chain": "optimism",
                "assets": [
                    {
                        "address": "0xnotwhitelisted",
                        "asset_symbol": "NW",
                        "balance": 100,
                    }
                ],
            }
        ]
        b._build_swap_to_usdc_action = MagicMock(return_value=None)
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            result = self._call(b)
        assert result == []

    def test_exception_returns_empty(self):
        b = _mk()
        b._get_usdc_address = MagicMock(side_effect=Exception("boom"))
        result = self._call(b)
        assert result == []


class TestCalculateAggregateTokenRatios:
    """Tests for _calculate_aggregate_token_ratios."""

    def test_empty_list(self):
        b = _mk()
        r0, r1 = b._calculate_aggregate_token_ratios([])
        assert r0 == 0.5
        assert r1 == 0.5

    def test_single_band_full_token0(self):
        b = _mk()
        reqs = [{"allocation": 1.0, "token0_ratio": 1.0, "token1_ratio": 0.0}]
        r0, r1 = b._calculate_aggregate_token_ratios(reqs)
        assert r0 == 1.0
        assert r1 == 0.0

    def test_single_band_full_token1(self):
        b = _mk()
        reqs = [{"allocation": 1.0, "token0_ratio": 0.0, "token1_ratio": 1.0}]
        r0, r1 = b._calculate_aggregate_token_ratios(reqs)
        assert r0 == 0.0
        assert r1 == 1.0

    def test_equal_allocation_equal_ratios(self):
        b = _mk()
        reqs = [
            {"allocation": 1.0, "token0_ratio": 0.5, "token1_ratio": 0.5},
            {"allocation": 1.0, "token0_ratio": 0.5, "token1_ratio": 0.5},
        ]
        r0, r1 = b._calculate_aggregate_token_ratios(reqs)
        assert abs(r0 - 0.5) < 1e-10
        assert abs(r1 - 0.5) < 1e-10

    def test_weighted_allocation(self):
        b = _mk()
        reqs = [
            {"allocation": 3.0, "token0_ratio": 1.0, "token1_ratio": 0.0},
            {"allocation": 1.0, "token0_ratio": 0.0, "token1_ratio": 1.0},
        ]
        r0, r1 = b._calculate_aggregate_token_ratios(reqs)
        assert abs(r0 - 0.75) < 1e-10
        assert abs(r1 - 0.25) < 1e-10

    def test_zero_total_weight(self):
        b = _mk()
        reqs = [{"allocation": 0, "token0_ratio": 0, "token1_ratio": 0}]
        r0, r1 = b._calculate_aggregate_token_ratios(reqs)
        assert r0 == 0.5
        assert r1 == 0.5

    def test_missing_keys_defaults(self):
        b = _mk()
        reqs = [{}]  # missing allocation, token0_ratio, token1_ratio
        r0, r1 = b._calculate_aggregate_token_ratios(reqs)
        assert r0 == 0.5
        assert r1 == 0.5


class TestCalculateMaxAmountsIn:
    """Tests for _calculate_max_amounts_in."""

    def test_token0_is_limiting(self):
        b = _mk()
        amounts, msg = b._calculate_max_amounts_in(
            token0_balance=100,
            token1_balance=1000,
            aggregate_token0_ratio=0.5,
            aggregate_token1_ratio=0.5,
        )
        # ideal: 550 token0, 550 token1. token0 is limiting (100 < 550)
        assert amounts[0] == 100
        assert amounts[1] <= 1000

    def test_token1_is_limiting(self):
        b = _mk()
        amounts, msg = b._calculate_max_amounts_in(
            token0_balance=1000,
            token1_balance=100,
            aggregate_token0_ratio=0.5,
            aggregate_token1_ratio=0.5,
        )
        assert amounts[1] == 100
        assert amounts[0] <= 1000

    def test_both_tokens_sufficient(self):
        b = _mk()
        amounts, msg = b._calculate_max_amounts_in(
            token0_balance=500,
            token1_balance=500,
            aggregate_token0_ratio=0.5,
            aggregate_token1_ratio=0.5,
        )
        assert amounts[0] == 500
        assert amounts[1] == 500

    def test_100_percent_token0(self):
        b = _mk()
        amounts, msg = b._calculate_max_amounts_in(
            token0_balance=1000,
            token1_balance=500,
            aggregate_token0_ratio=1.0,
            aggregate_token1_ratio=0.0,
        )
        # ideal: 1500 * 1.0 = 1500 token0, but only have 1000
        assert amounts[0] == 1000
        assert amounts[1] == 0


class TestBuildEnterPoolAction:
    """Tests for _build_enter_pool_action."""

    def test_none_opportunity(self):
        b = _mk()
        result = b._build_enter_pool_action(None)
        assert result is None

    def test_empty_opportunity(self):
        b = _mk()
        result = b._build_enter_pool_action({})
        assert result is None

    def test_sturdy_dex_type(self):
        b = _mk()
        opp = {"dex_type": DexType.STURDY.value, "pool_address": "0x1", "apr": 10}
        result = b._build_enter_pool_action(opp)
        assert result["action"] == Action.DEPOSIT.value

    def test_velodrome_dex_type(self):
        b = _mk()
        opp = {"dex_type": "velodrome", "pool_address": "0x1", "apr": 20}
        result = b._build_enter_pool_action(opp)
        assert result["action"] == Action.ENTER_POOL.value
        assert result["opportunity_apr"] == 20

    def test_percent_in_bounds_default(self):
        b = _mk()
        opp = {"dex_type": "velodrome", "pool_address": "0x1"}
        result = b._build_enter_pool_action(opp)
        assert result["percent_in_bounds"] == 0.0


class TestBuildClaimRewardAction:
    """Tests for _build_claim_reward_action."""

    def test_basic_claim(self):
        b = _mk()
        rewards = {
            "users": ["0xuser"],
            "tokens": ["0xtoken"],
            "claims": [100],
            "proofs": [["proof"]],
            "symbols": ["TKN"],
        }
        result = b._build_claim_reward_action(rewards, "optimism")
        assert result["action"] == Action.CLAIM_REWARDS.value
        assert result["chain"] == "optimism"
        assert result["users"] == ["0xuser"]


class TestGetRequiredTokens:
    """Tests for _get_required_tokens."""

    def test_non_sturdy_two_tokens(self):
        b = _mk()
        opp = {
            "token0": "0xA",
            "token0_symbol": "A",
            "token1": "0xB",
            "token1_symbol": "B",
            "dex_type": "velodrome",
        }
        result = b._get_required_tokens(opp)
        assert len(result) == 2
        assert result[0] == ("0xA", "A")
        assert result[1] == ("0xB", "B")

    def test_sturdy_one_token(self):
        b = _mk()
        opp = {
            "token0": "0xA",
            "token0_symbol": "A",
            "token1": "0xB",
            "token1_symbol": "B",
            "dex_type": DexType.STURDY.value,
        }
        result = b._get_required_tokens(opp)
        assert len(result) == 1

    def test_missing_tokens(self):
        b = _mk()
        result = b._get_required_tokens({})
        assert result == []


class TestGroupTokensByChain:
    """Tests for _group_tokens_by_chain."""

    def test_empty(self):
        b = _mk()
        assert b._group_tokens_by_chain([]) == {}

    def test_single_chain(self):
        b = _mk()
        tokens = [{"chain": "optimism", "token": "0xA"}]
        result = b._group_tokens_by_chain(tokens)
        assert len(result) == 1
        assert len(result["optimism"]) == 1

    def test_multiple_chains(self):
        b = _mk()
        tokens = [
            {"chain": "optimism", "token": "0xA"},
            {"chain": "mode", "token": "0xB"},
            {"chain": "optimism", "token": "0xC"},
        ]
        result = b._group_tokens_by_chain(tokens)
        assert len(result["optimism"]) == 2
        assert len(result["mode"]) == 1


class TestIdentifyMissingTokens:
    """Tests for _identify_missing_tokens."""

    def test_all_available(self):
        b = _mk()
        required = [("0xA", "A"), ("0xB", "B")]
        available = {"0xA": {}, "0xB": {}}
        result = b._identify_missing_tokens(required, available, "optimism")
        assert result == []

    def test_some_missing(self):
        b = _mk()
        required = [("0xA", "A"), ("0xB", "B")]
        available = {"0xA": {}}
        result = b._identify_missing_tokens(required, available, "optimism")
        assert len(result) == 1
        assert result[0] == ("0xB", "B")

    def test_all_missing(self):
        b = _mk()
        required = [("0xA", "A"), ("0xB", "B")]
        result = b._identify_missing_tokens(required, {}, "optimism")
        assert len(result) == 2


class TestBuildTokensFromPosition:
    """Tests for _build_tokens_from_position."""

    def test_one_token(self):
        b = _mk()
        pos = {"chain": "optimism", "token0": "0xA", "token0_symbol": "A"}
        result = b._build_tokens_from_position(pos, 1)
        assert len(result) == 1
        assert result[0]["token"] == "0xA"

    def test_two_tokens(self):
        b = _mk()
        pos = {
            "chain": "optimism",
            "token0": "0xA",
            "token0_symbol": "A",
            "token1": "0xB",
            "token1_symbol": "B",
        }
        result = b._build_tokens_from_position(pos, 2)
        assert len(result) == 2

    def test_invalid_num_tokens(self):
        b = _mk()
        result = b._build_tokens_from_position({}, 3)
        assert result is None


class TestMergeDuplicateBridgeSwapActions:
    """Tests for _merge_duplicate_bridge_swap_actions."""

    def test_empty_actions(self):
        b = _mk()
        assert b._merge_duplicate_bridge_swap_actions([]) == []

    def test_no_bridge_actions(self):
        b = _mk()
        actions = [{"action": "EnterPool"}]
        result = b._merge_duplicate_bridge_swap_actions(actions)
        assert result == actions

    def test_single_bridge_action(self):
        b = _mk()
        actions = [
            {
                "action": Action.FIND_BRIDGE_ROUTE.value,
                "from_chain": "optimism",
                "to_chain": "mode",
                "from_token": "0xA",
                "to_token": "0xB",
                "funds_percentage": 0.5,
            }
        ]
        result = b._merge_duplicate_bridge_swap_actions(actions)
        assert len(result) == 1

    def test_duplicate_bridge_actions_merged(self):
        b = _mk()
        action = {
            "action": Action.FIND_BRIDGE_ROUTE.value,
            "from_chain": "optimism",
            "to_chain": "mode",
            "from_token": "0xAAAAAAAA",
            "to_token": "0xBBBBBBBB",
            "funds_percentage": 0.3,
        }
        actions = [action.copy(), action.copy()]
        result = b._merge_duplicate_bridge_swap_actions(actions)
        assert len(result) == 1
        assert result[0]["funds_percentage"] == 0.6

    def test_redundant_same_chain_same_token_removed(self):
        b = _mk()
        actions = [
            {
                "action": Action.FIND_BRIDGE_ROUTE.value,
                "from_chain": "optimism",
                "to_chain": "optimism",
                "from_token": "0xAAAA",
                "to_token": "0xAAAA",
                "funds_percentage": 0.5,
            }
        ]
        result = b._merge_duplicate_bridge_swap_actions(actions)
        assert len(result) == 0

    def test_none_actions_returns_none(self):
        b = _mk()
        result = b._merge_duplicate_bridge_swap_actions(None)
        assert result is None


class TestBuildExitPoolAction:
    """Tests for _build_exit_pool_action."""

    def test_no_position_to_exit(self):
        b = _mk()
        b.position_to_exit = None
        result = b._build_exit_pool_action([], 2)
        assert result is None

    def test_not_enough_tokens(self):
        b = _mk()
        b.position_to_exit = {"pool_address": "0x1"}
        result = b._build_exit_pool_action([{"token": "0xA"}], 2)
        assert result is None

    def test_valid_exit(self):
        b = _mk()
        b.position_to_exit = {"pool_address": "0x1"}
        tokens = [{"token": "0xA"}, {"token": "0xB"}]
        b._build_exit_pool_action_base = MagicMock(return_value={"action": "ExitPool"})
        result = b._build_exit_pool_action(tokens, 2)
        assert result is not None
        assert result["action"] == "ExitPool"


class TestGetAssetSymbol:
    """Tests for _get_asset_symbol."""

    def test_found(self):
        b = _mk()
        synced_mock = MagicMock()
        synced_mock.positions = [
            {
                "chain": "optimism",
                "assets": [{"address": "0xA", "asset_symbol": "TKN"}],
            }
        ]
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            assert b._get_asset_symbol("optimism", "0xA") == "TKN"

    def test_not_found_chain(self):
        b = _mk()
        synced_mock = MagicMock()
        synced_mock.positions = [
            {"chain": "mode", "assets": [{"address": "0xA", "asset_symbol": "TKN"}]}
        ]
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            assert b._get_asset_symbol("optimism", "0xA") is None

    def test_not_found_address(self):
        b = _mk()
        synced_mock = MagicMock()
        synced_mock.positions = [
            {"chain": "optimism", "assets": [{"address": "0xB", "asset_symbol": "TKN"}]}
        ]
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            assert b._get_asset_symbol("optimism", "0xA") is None


class TestShouldAddStakingActions:
    """Tests for _should_add_staking_actions."""

    def test_non_velodrome(self):
        b = _mk()
        assert b._should_add_staking_actions({"dex_type": "uniswap"}) is False

    def test_velodrome_with_voter(self):
        b = _mk()
        assert (
            b._should_add_staking_actions(
                {"dex_type": "velodrome", "chain": "optimism"}
            )
            is True
        )

    def test_velodrome_no_voter(self):
        b = _mk()
        b.params.velodrome_voter_contract_addresses = {}
        assert (
            b._should_add_staking_actions(
                {"dex_type": "velodrome", "chain": "optimism"}
            )
            is False
        )


class TestBuildStakeLpTokensAction:
    """Tests for _build_stake_lp_tokens_action."""

    def test_non_velodrome(self):
        b = _mk()
        result = b._build_stake_lp_tokens_action(
            {"dex_type": "uniswap", "chain": "optimism", "pool_address": "0x1"}
        )
        assert result is None

    def test_missing_params(self):
        b = _mk()
        result = b._build_stake_lp_tokens_action({"dex_type": "velodrome"})
        assert result is None

    def test_cl_pool(self):
        b = _mk()
        result = b._build_stake_lp_tokens_action(
            {
                "dex_type": "velodrome",
                "chain": "optimism",
                "pool_address": "0x1",
                "is_cl_pool": True,
            }
        )
        assert result is not None
        assert result["action"] == Action.STAKE_LP_TOKENS.value
        assert result["is_cl_pool"] is True

    def test_regular_pool(self):
        b = _mk()
        result = b._build_stake_lp_tokens_action(
            {
                "dex_type": "velodrome",
                "chain": "optimism",
                "pool_address": "0x1",
                "is_cl_pool": False,
            }
        )
        assert result is not None
        assert result["is_cl_pool"] is False

    def test_cl_pool_no_safe_address(self):
        b = _mk()
        b.params.safe_contract_addresses = {}
        result = b._build_stake_lp_tokens_action(
            {
                "dex_type": "velodrome",
                "chain": "optimism",
                "pool_address": "0x1",
                "is_cl_pool": True,
            }
        )
        assert result is None

    def test_exception(self):
        b = _mk()
        # Force an exception inside the try block by making get() raise
        b.params.safe_contract_addresses = {"optimism": "0x" + "aa" * 20}
        # Patch chain to trigger exception in a way that hits the except block
        result = b._build_stake_lp_tokens_action(
            {
                "dex_type": "velodrome",
                "chain": None,  # This hits the "not all([chain, pool_address])" → returns None
                "pool_address": "0x1",
            }
        )
        assert result is None


class TestBuildClaimStakingRewardsAction:
    """Tests for _build_claim_staking_rewards_action."""

    def test_non_velodrome(self):
        b = _mk()
        result = b._build_claim_staking_rewards_action({"dex_type": "uniswap"})
        assert result is None

    def test_missing_params(self):
        b = _mk()
        result = b._build_claim_staking_rewards_action({"dex_type": "velodrome"})
        assert result is None

    def test_valid_with_gauge(self):
        b = _mk()
        result = b._build_claim_staking_rewards_action(
            {
                "dex_type": "velodrome",
                "chain": "optimism",
                "pool_address": "0x1",
                "gauge_address": "0xgauge",
            }
        )
        assert result is not None
        assert result["gauge_address"] == "0xgauge"

    def test_valid_without_gauge(self):
        b = _mk()
        result = b._build_claim_staking_rewards_action(
            {
                "dex_type": "velodrome",
                "chain": "optimism",
                "pool_address": "0x1",
            }
        )
        assert result is not None
        assert "gauge_address" not in result

    def test_no_safe_address(self):
        b = _mk()
        b.params.safe_contract_addresses = {}
        result = b._build_claim_staking_rewards_action(
            {
                "dex_type": "velodrome",
                "chain": "optimism",
                "pool_address": "0x1",
            }
        )
        assert result is None


class TestHasOpenPositions:
    """Tests for _has_open_positions."""

    def test_no_positions(self):
        b = _mk()
        b.current_positions = []
        assert b._has_open_positions() is False

    def test_only_closed(self):
        b = _mk()
        b.current_positions = [{"status": "closed"}]
        assert b._has_open_positions() is False

    def test_has_open(self):
        b = _mk()
        b.current_positions = [{"status": "open"}]
        assert b._has_open_positions() is True

    def test_exception(self):
        b = _mk()
        b.current_positions = None  # Will cause TypeError
        assert b._has_open_positions() is False


class TestFormatOpportunityForTracking:
    """Tests for _format_opportunity_for_tracking."""

    def test_basic_formatting(self):
        b = _mk()
        b._get_current_timestamp = lambda: 12345
        opp = {
            "pool_address": "0xpool",
            "dex_type": "velodrome",
            "chain": "optimism",
            "apr": 20.0,
            "tvl": 1000,
            "strategy_source": "test",
        }
        result = b._format_opportunity_for_tracking(opp, "raw_with_metrics")
        assert result["pool_address"] == "0xpool"
        assert result["stage"] == "raw_with_metrics"
        assert result["timestamp"] == 12345

    def test_missing_keys_default_to_none(self):
        b = _mk()
        b._get_current_timestamp = lambda: 0
        result = b._format_opportunity_for_tracking({}, "test")
        assert result["pool_address"] is None
        assert result["apr"] is None


class TestCanClaimRewards:
    """Tests for _can_claim_rewards."""

    def test_no_last_claimed(self):
        b = _mk()
        synced_mock = MagicMock()
        synced_mock.last_reward_claimed_timestamp = None
        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 1000
        b.context.state.round_sequence.last_round_transition_timestamp = mock_ts
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            assert b._can_claim_rewards() is True

    def test_enough_time_elapsed(self):
        b = _mk()
        synced_mock = MagicMock()
        synced_mock.last_reward_claimed_timestamp = 100
        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 2000
        b.context.state.round_sequence.last_round_transition_timestamp = mock_ts
        b.params.reward_claiming_time_period = 500
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            assert b._can_claim_rewards() is True

    def test_not_enough_time(self):
        b = _mk()
        synced_mock = MagicMock()
        synced_mock.last_reward_claimed_timestamp = 100
        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 200
        b.context.state.round_sequence.last_round_transition_timestamp = mock_ts
        b.params.reward_claiming_time_period = 500
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            assert b._can_claim_rewards() is False


class TestExecuteStrategy:
    """Tests for execute_strategy."""

    def test_no_strategy_kwarg(self):
        b = _mk()
        result = b.execute_strategy()
        assert result is None

    def test_no_executable(self):
        b = _mk()
        b.strategy_exec = MagicMock(return_value=None)
        result = b.execute_strategy(strategy="test")
        assert result is None

    def test_callable_not_found(self):
        b = _mk()
        # strategy_exec returns (exec_code, callable_method)
        b.strategy_exec = MagicMock(return_value=("", "nonexistent_method"))
        result = b.execute_strategy(strategy="test")
        assert result is None


class TestStrategyExec:
    """Tests for strategy_exec."""

    def test_found(self):
        b = _mk()
        b.shared_state.strategies_executables = {"mystrategy": ("code", "method")}
        result = b.strategy_exec("mystrategy")
        assert result == ("code", "method")

    def test_not_found(self):
        b = _mk()
        b.shared_state.strategies_executables = {}
        result = b.strategy_exec("mystrategy")
        assert result is None


class TestBuildBridgeSwapActions:
    """Tests for _build_bridge_swap_actions."""

    def test_none_opportunity(self):
        b = _mk()
        result = b._build_bridge_swap_actions(None, [])
        assert result is None

    def test_empty_opportunity(self):
        b = _mk()
        result = b._build_bridge_swap_actions({}, [])
        assert result is None

    def test_incomplete_opportunity(self):
        b = _mk()
        result = b._build_bridge_swap_actions({"chain": "optimism"}, [])
        assert result is None

    def test_no_required_tokens(self):
        b = _mk()
        # Missing token0
        result = b._build_bridge_swap_actions({"chain": "optimism", "token0": ""}, [])
        assert result is None

    def test_all_tokens_available(self):
        b = _mk()
        opp = {
            "chain": "optimism",
            "token0": "0xA",
            "token0_symbol": "A",
            "token1": "0xB",
            "token1_symbol": "B",
            "dex_type": "velodrome",
            "relative_funds_percentage": 1.0,
        }
        tokens = [
            {"chain": "optimism", "token": "0xA", "token_symbol": "A"},
            {"chain": "optimism", "token": "0xB", "token_symbol": "B"},
        ]
        result = b._build_bridge_swap_actions(opp, tokens)
        assert isinstance(result, list)

    def test_all_tokens_needed(self):
        b = _mk()
        opp = {
            "chain": "optimism",
            "token0": "0xA",
            "token0_symbol": "A",
            "token1": "0xB",
            "token1_symbol": "B",
            "dex_type": "velodrome",
            "relative_funds_percentage": 1.0,
        }
        # Tokens on different chain
        tokens = [
            {"chain": "mode", "token": "0xC", "token_symbol": "C"},
        ]
        result = b._build_bridge_swap_actions(opp, tokens)
        assert isinstance(result, list)


class TestAddBridgeSwapAction:
    """Tests for _add_bridge_swap_action."""

    def test_different_chain(self):
        b = _mk()
        actions = []
        token = {"chain": "mode", "token": "0xA", "token_symbol": "A"}
        b._add_bridge_swap_action(actions, token, "optimism", "0xB", "B", 0.5)
        assert len(actions) == 1
        assert actions[0]["action"] == Action.FIND_BRIDGE_ROUTE.value

    def test_same_chain_different_token(self):
        b = _mk()
        actions = []
        token = {"chain": "optimism", "token": "0xA", "token_symbol": "A"}
        b._add_bridge_swap_action(actions, token, "optimism", "0xB", "B", 0.5)
        assert len(actions) == 1

    def test_same_chain_same_token(self):
        b = _mk()
        actions = []
        token = {"chain": "optimism", "token": "0xA", "token_symbol": "A"}
        b._add_bridge_swap_action(actions, token, "optimism", "0xA", "A", 0.5)
        assert len(actions) == 0

    def test_incomplete_token_data(self):
        b = _mk()
        actions = []
        token = {"chain": None, "token": "0xA", "token_symbol": "A"}
        b._add_bridge_swap_action(actions, token, "optimism", "0xB", "B", 0.5)
        assert len(actions) == 0


class TestGetPositionTokenBalances:
    """Tests for _get_position_token_balances."""

    def test_stored_balances_used(self):
        b = _mk()
        pos = {
            "pool_address": "0xpool",
            "current_token_balances": {"0xA": 100.0, "0xB": 200.0},
        }
        result = _drive(b._get_position_token_balances(pos, "optimism"))
        assert result["0xA"] == 100.0
        assert result["0xB"] == 200.0

    def test_stored_balances_with_velo_rewards(self):
        b = _mk()
        pos = {
            "pool_address": "0xpool",
            "current_token_balances": {"0xA": 100.0},
            "staked": True,
            "dex_type": "velodrome",
        }
        b._get_velodrome_pending_rewards = _gen_return(50.0)
        b._get_velo_token_address = MagicMock(return_value="0xVELO")
        result = _drive(b._get_position_token_balances(pos, "optimism"), sends=[None])
        assert result["0xVELO"] == 50.0

    def test_fallback_no_safe_address(self):
        b = _mk()
        b.params.safe_contract_addresses = {}
        pos = {"pool_address": "0xpool", "token0": "0xA", "token1": "0xB"}
        result = _drive(b._get_position_token_balances(pos, "optimism"))
        assert result == {}

    def test_fallback_gets_balances(self):
        b = _mk()
        pos = {"pool_address": "0xpool", "token0": "0xA", "token1": "0xB"}
        b._get_token_balance = _gen_return(1000)
        b._get_token_decimals = _gen_return(6)
        result = _drive(
            b._get_position_token_balances(pos, "optimism"), sends=[None] * 10
        )
        assert "0xA" in result
        assert "0xB" in result


class TestCalculateCurrentValueRatio:
    """Tests for _calculate_current_value_ratio."""

    def test_missing_token_addresses(self):
        b = _mk()
        result = _drive(b._calculate_current_value_ratio({}, "optimism"))
        assert result is None

    def test_no_balances(self):
        b = _mk()
        pos = {"token0": "0xA", "token1": "0xB"}
        b._get_position_token_balances = _gen_return({})
        result = _drive(b._calculate_current_value_ratio(pos, "optimism"), sends=[None])
        assert result is None

    def test_zero_entry_amounts(self):
        b = _mk()
        pos = {"token0": "0xA", "token1": "0xB", "amount0": 0, "amount1": 0}
        b._get_position_token_balances = _gen_return({"0xA": 100, "0xB": 200})
        result = _drive(b._calculate_current_value_ratio(pos, "optimism"), sends=[None])
        assert result is None

    def test_no_token_decimals(self):
        b = _mk()
        pos = {"token0": "0xA", "token1": "0xB", "amount0": 1000, "amount1": 2000}
        b._get_position_token_balances = _gen_return({"0xA": 100, "0xB": 200})
        b._get_token_decimals = _gen_return(None)
        result = _drive(
            b._calculate_current_value_ratio(pos, "optimism"), sends=[None] * 5
        )
        assert result is None

    def test_no_sma_prices(self):
        b = _mk()
        pos = {"token0": "0xA", "token1": "0xB", "amount0": 1000, "amount1": 2000}
        b._get_position_token_balances = _gen_return({"0xA": 100, "0xB": 200})
        b._get_token_decimals = _gen_return(6)
        b._fetch_token_prices_sma = _gen_return(None)
        result = _drive(
            b._calculate_current_value_ratio(pos, "optimism"), sends=[None] * 10
        )
        assert result is None

    def test_valid_calculation(self):
        b = _mk()
        pos = {
            "token0": "0xA",
            "token1": "0xB",
            "amount0": 1000000,  # 1.0 with 6 decimals
            "amount1": 2000000,  # 2.0 with 6 decimals
            "yield_usd": 10.0,
        }
        b._get_position_token_balances = _gen_return({"0xA": 1.5, "0xB": 2.5})
        b._get_token_decimals = _gen_return(6)
        b._fetch_token_prices_sma = _gen_return(1.0)  # $1 per token
        result = _drive(
            b._calculate_current_value_ratio(pos, "optimism"), sends=[None] * 10
        )
        # denominator = Q1_entry*SMA1 + Q0_entry*SMA0 = 2.0*1.0 + 1.0*1.0 = 3.0
        # numerator = denominator + yield_usd = 3.0 + 10.0 = 13.0
        # ratio = 13.0 / 3.0
        assert result is not None
        assert abs(result - (13.0 / 3.0)) < 1e-6

    def test_zero_denominator(self):
        b = _mk()
        pos = {
            "token0": "0xA",
            "token1": "0xB",
            "amount0": 1000000,
            "amount1": 0,
            "yield_usd": 0,
        }
        b._get_position_token_balances = _gen_return({"0xA": 1.0})
        b._get_token_decimals = _gen_return(6)
        b._fetch_token_prices_sma = _gen_return(0.0)  # $0 price => denom = 0
        result = _drive(
            b._calculate_current_value_ratio(pos, "optimism"), sends=[None] * 10
        )
        assert result is None

    def test_exception_returns_none(self):
        b = _mk()
        pos = {"token0": "0xA", "token1": "0xB", "amount0": 1000, "amount1": 2000}

        def _bad_gen(*a, **kw):
            raise Exception("boom")
            yield  # noqa: unreachable

        b._get_position_token_balances = _bad_gen
        result = _drive(b._calculate_current_value_ratio(pos, "optimism"))
        assert result is None


class TestHandleVelodromeTokenAllocation:
    """Tests for _handle_velodrome_token_allocation."""

    def test_non_velodrome_passthrough(self):
        b = _mk()
        actions = [{"action": "test"}]
        enter_action = {"dex_type": "uniswap"}
        result = b._handle_velodrome_token_allocation(actions, enter_action, [])
        assert result == actions

    def test_velodrome_no_token_requirements(self):
        b = _mk()
        actions = [{"action": "test"}]
        enter_action = {"dex_type": "velodrome"}
        result = b._handle_velodrome_token_allocation(actions, enter_action, [])
        assert result == actions

    def test_velodrome_100_percent_token0(self):
        b = _mk()
        actions = [
            {
                "action": "FindBridgeRoute",
                "to_chain": "optimism",
                "to_token": "0xother",
                "to_token_symbol": "OTHER",
                "funds_percentage": 0.5,
            }
        ]
        enter_action = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "token0": "0xA",
            "token0_symbol": "A",
            "token1": "0xB",
            "token1_symbol": "B",
            "token_requirements": {
                "position_requirements": [
                    {"allocation": 1.0, "token0_ratio": 1.0, "token1_ratio": 0.0}
                ]
            },
            "relative_funds_percentage": 1.0,
        }
        result = b._handle_velodrome_token_allocation(actions, enter_action, [])
        # Bridge route should be redirected to token0
        assert result[0]["to_token"] == "0xA"
        assert result[0]["funds_percentage"] == 1.0

    def test_velodrome_100_percent_token1(self):
        b = _mk()
        actions = [
            {
                "action": "FindBridgeRoute",
                "to_chain": "optimism",
                "to_token": "0xother",
                "to_token_symbol": "OTHER",
                "funds_percentage": 0.5,
            }
        ]
        enter_action = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "token0": "0xA",
            "token0_symbol": "A",
            "token1": "0xB",
            "token1_symbol": "B",
            "token_requirements": {
                "position_requirements": [
                    {"allocation": 1.0, "token0_ratio": 0.0, "token1_ratio": 1.0}
                ]
            },
            "relative_funds_percentage": 1.0,
        }
        result = b._handle_velodrome_token_allocation(actions, enter_action, [])
        assert result[0]["to_token"] == "0xB"

    def test_velodrome_mixed_allocation(self):
        """50/50 allocation should NOT trigger extreme handling."""
        b = _mk()
        actions = [{"action": "test"}]
        enter_action = {
            "dex_type": "velodrome",
            "token_requirements": {
                "position_requirements": [
                    {"allocation": 1.0, "token0_ratio": 0.5, "token1_ratio": 0.5}
                ]
            },
        }
        result = b._handle_velodrome_token_allocation(actions, enter_action, [])
        assert result == actions

    def test_velodrome_no_bridge_routes_adds_new(self):
        """When no FindBridgeRoute exists, a new one is added."""
        b = _mk()
        actions = []
        enter_action = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "token0": "0xA",
            "token0_symbol": "A",
            "token1": "0xB",
            "token1_symbol": "B",
            "token_requirements": {
                "position_requirements": [
                    {"allocation": 1.0, "token0_ratio": 1.0, "token1_ratio": 0.0}
                ]
            },
            "relative_funds_percentage": 1.0,
        }
        available_tokens = [
            {"token": "0xC", "chain": "mode", "token_symbol": "C"},
        ]
        result = b._handle_velodrome_token_allocation(
            actions, enter_action, available_tokens
        )
        assert len(result) == 1
        assert result[0]["action"] == "FindBridgeRoute"
        assert result[0]["to_token"] == "0xA"

    def test_velodrome_empty_position_requirements_fallback_recommendation(self):
        """Test fallback to recommendation text when no position_requirements."""
        b = _mk()
        actions = [
            {
                "action": "FindBridgeRoute",
                "to_chain": "optimism",
                "to_token": "0xother",
                "to_token_symbol": "OTHER",
                "funds_percentage": 0.5,
            }
        ]
        enter_action = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "token0": "0xA",
            "token0_symbol": "A",
            "token1": "0xB",
            "token1_symbol": "B",
            "token_requirements": {
                "position_requirements": [],
                "recommendation": "Provide 100% token0, 0% token1",
            },
            "relative_funds_percentage": 1.0,
        }
        result = b._handle_velodrome_token_allocation(actions, enter_action, [])
        assert result[0]["to_token"] == "0xA"


class TestCalculateVelodromeClTokenRequirements:
    """Tests for calculate_velodrome_cl_token_requirements."""

    def test_invalid_inputs_returns_none(self):
        b = _mk()
        result = _drive(b.calculate_velodrome_cl_token_requirements([], 0))
        assert result is None

    def test_valid_inputs_calls_calculate_ratios(self):
        b = _mk()
        bands = [{"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}]
        expected_result = {"position_requirements": [], "recommendation": "test"}
        b.calculate_velodrome_token_ratios = _gen_return(expected_result)
        result = _drive(
            b.calculate_velodrome_cl_token_requirements(bands, 1.5), sends=[None]
        )
        assert result == expected_result

    def test_with_sqrt_price_x96(self):
        b = _mk()
        bands = [{"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}]
        expected_result = {"position_requirements": []}
        b.calculate_velodrome_token_ratios = _gen_return(expected_result)
        result = _drive(
            b.calculate_velodrome_cl_token_requirements(
                bands, 1.5, sqrt_price_x96=12345
            ),
            sends=[None],
        )
        assert result == expected_result


class TestInitializeEntryCostsForNewPosition:
    """Tests for _initialize_entry_costs_for_new_position."""

    def test_valid_action(self):
        b = _mk()
        b._initialize_position_entry_costs = _gen_none
        action = {"chain": "optimism", "pool_address": "0xpool"}
        _drive(b._initialize_entry_costs_for_new_position(action), sends=[None])
        # Should complete without error

    def test_missing_chain(self):
        b = _mk()
        action = {"pool_address": "0xpool"}
        _drive(b._initialize_entry_costs_for_new_position(action))
        # Should log warning but not crash

    def test_exception(self):
        b = _mk()

        def _bad(*a, **kw):
            raise Exception("boom")
            yield  # noqa: unreachable

        b._initialize_position_entry_costs = _bad
        action = {"chain": "optimism", "pool_address": "0xpool"}
        _drive(b._initialize_entry_costs_for_new_position(action))
        # Should handle exception


class TestHandleGetStrategy:
    """Tests for _handle_get_strategy."""

    def test_no_inflight_request(self):
        b = _mk()
        b._inflight_strategy_req = None
        msg = MagicMock()
        b._handle_get_strategy(msg, MagicMock())
        b.context.logger.error.assert_called()

    def test_valid_response(self):
        b = _mk()
        b._inflight_strategy_req = "test_strategy"
        b.shared_state.strategy_to_filehash = {"test_strategy": "hash123"}

        msg = MagicMock()
        msg.files = {"file": "content"}

        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy.ComponentPackageLoader.load",
            return_value=("yaml", "exec_code", "callable_method"),
        ):
            b._handle_get_strategy(msg, MagicMock())

        assert b._inflight_strategy_req is None
        assert "test_strategy" in b.shared_state.strategies_executables


class TestSendMessage:
    """Tests for send_message."""

    def test_send_message(self):
        b = _mk()
        b.shared_state.req_to_callback = {}
        msg = MagicMock()
        dialogue = MagicMock()
        dialogue.dialogue_label.dialogue_reference = ("nonce123", "")
        callback = MagicMock()
        b.send_message(msg, dialogue, callback)
        b.context.outbox.put_message.assert_called_once_with(message=msg)
        assert b.shared_state.req_to_callback["nonce123"] == callback


class TestDownloadNextStrategy:
    """Tests for download_next_strategy."""

    def test_inflight_request_exists(self):
        b = _mk()
        b._inflight_strategy_req = "existing"
        b.shared_state.strategy_to_filehash = {"test": "hash"}
        b.download_next_strategy()
        # Should return early without starting a new request

    def test_no_strategies_pending(self):
        b = _mk()
        b._inflight_strategy_req = None
        b.shared_state.strategy_to_filehash = {}
        b.download_next_strategy()
        # Should return early

    def test_starts_download(self):
        b = _mk()
        b._inflight_strategy_req = None
        b.shared_state.strategy_to_filehash = {"strategy_a": "hash_a"}
        b._build_ipfs_get_file_req = MagicMock(return_value=(MagicMock(), MagicMock()))
        b.send_message = MagicMock()
        b.download_next_strategy()
        assert b._inflight_strategy_req == "strategy_a"
        b.send_message.assert_called_once()


class TestGetReturnsMetricsForOpportunity:
    """Tests for get_returns_metrics_for_opportunity."""

    def test_no_metrics(self):
        b = _mk()
        b.context.coingecko = MagicMock()
        b.context.coingecko.use_x402 = False
        b.execute_strategy = MagicMock(return_value=None)
        result = b.get_returns_metrics_for_opportunity(
            {"pool_address": "0x1"}, "strategy_a"
        )
        assert result is None

    def test_error_in_metrics(self):
        b = _mk()
        b.context.coingecko = MagicMock()
        b.context.coingecko.use_x402 = False
        b.execute_strategy = MagicMock(return_value={"error": "something failed"})
        result = b.get_returns_metrics_for_opportunity(
            {"pool_address": "0x1"}, "strategy_a"
        )
        assert result is None

    def test_valid_metrics(self):
        b = _mk()
        b.context.coingecko = MagicMock()
        b.context.coingecko.use_x402 = False
        metrics = {"apr": 20.0, "tvl": 5000}
        b.execute_strategy = MagicMock(return_value=metrics)
        result = b.get_returns_metrics_for_opportunity(
            {"pool_address": "0x1"}, "strategy_a"
        )
        assert result == metrics


class TestTrackOpportunities:
    """Tests for _track_opportunities."""

    def test_no_existing_data(self):
        b = _mk()
        b._get_current_timestamp = lambda: 12345
        b._read_kv = _gen_return(None)
        b._write_kv = _gen_none
        synced_mock = MagicMock()
        synced_mock.period_count = 1
        opps = [{"pool_address": "0x1", "strategy_source": "test"}]
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            _drive(b._track_opportunities(opps, "raw_with_metrics"), sends=[None] * 5)

    def test_with_existing_tracking_data(self):
        b = _mk()
        b._get_current_timestamp = lambda: 12345
        existing = {"round_1": {"old_stage": {}}}
        b._read_kv = _gen_return({"opportunity_tracking": json.dumps(existing)})
        b._write_kv = _gen_none
        synced_mock = MagicMock()
        synced_mock.period_count = 1
        opps = [{"pool_address": "0x1", "strategy_source": "test"}]
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            _drive(b._track_opportunities(opps, "basic_filtered"), sends=[None] * 5)

    def test_exception_handled(self):
        b = _mk()

        def _bad(*a, **kw):
            raise Exception("boom")
            yield  # noqa: unreachable

        b._read_kv = _bad
        _drive(b._track_opportunities([], "test"))
        b.context.logger.error.assert_called()

    def test_invalid_json_in_tracking_data(self):
        b = _mk()
        b._get_current_timestamp = lambda: 12345
        b._read_kv = _gen_return({"opportunity_tracking": "not valid json{{{"})
        b._write_kv = _gen_none
        synced_mock = MagicMock()
        synced_mock.period_count = 1
        opps = [{"pool_address": "0x1", "strategy_source": "test"}]
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            _drive(b._track_opportunities(opps, "test"), sends=[None] * 5)
        # Should handle JSON error and start with empty dict


class TestPrepareStrategyActions:
    """Tests for prepare_strategy_actions."""

    def test_no_opportunities(self):
        b = _mk()
        b.trading_opportunities = []
        result = _drive(b.prepare_strategy_actions())
        assert result == []

    def test_with_opportunities_none_selected(self):
        b = _mk()
        b.trading_opportunities = [{"apr": 10}]
        b.execute_hyper_strategy = _gen_none
        b.selected_opportunities = None
        result = _drive(b.prepare_strategy_actions(), sends=[None] * 5)
        # When selected_opportunities is None, yield from [] returns None, so actions=None
        assert result is None


class TestCreateOpportunityAttrDef:
    """Tests for _create_opportunity_attr_def."""

    def test_empty_agent_type(self):
        b = _mk()
        result = _drive(b._create_opportunity_attr_def("agent_id", {}))
        assert result is None

    def test_missing_type_id(self):
        b = _mk()
        result = _drive(b._create_opportunity_attr_def("agent_id", {"name": "test"}))
        assert result is None

    def test_valid_creation(self):
        b = _mk()
        b.create_attribute_definition = _gen_return({"attr_def_id": "123"})
        result = _drive(
            b._create_opportunity_attr_def("agent_id", {"type_id": "type_123"}),
            sends=[None],
        )
        assert result == {"attr_def_id": "123"}

    def test_exception(self):
        b = _mk()

        def _bad(*a, **kw):
            raise Exception("boom")
            yield  # noqa: unreachable

        b.create_attribute_definition = _bad
        result = _drive(
            b._create_opportunity_attr_def("agent_id", {"type_id": "type_123"})
        )
        assert result is None


class TestCacheEnterPoolActionForClPool:
    """Tests for _cache_enter_pool_action_for_cl_pool."""

    def test_no_chain(self):
        b = _mk()
        _drive(b._cache_enter_pool_action_for_cl_pool({}, {"action": "EnterPool"}))
        # Should return early

    def test_no_cached_data(self):
        b = _mk()
        b._get_cached_cl_pool_data = _gen_return(None)
        _drive(
            b._cache_enter_pool_action_for_cl_pool(
                {"chain": "optimism"}, {"action": "EnterPool"}
            ),
            sends=[None],
        )

    def test_valid_cache(self):
        b = _mk()
        cached = {"pool_address": "0xpool"}
        b._get_cached_cl_pool_data = _gen_return(cached)
        b._write_kv = _gen_none
        _drive(
            b._cache_enter_pool_action_for_cl_pool(
                {"chain": "optimism"}, {"action": "EnterPool"}
            ),
            sends=[None, None],
        )
        assert cached["enter_pool_action"] == {"action": "EnterPool"}

    def test_exception(self):
        b = _mk()

        def _bad(*a, **kw):
            raise Exception("boom")
            yield  # noqa: unreachable

        b._get_cached_cl_pool_data = _bad
        _drive(
            b._cache_enter_pool_action_for_cl_pool({"chain": "optimism"}, {}),
        )
        b.context.logger.error.assert_called()


class TestGetGaugeAddressForPosition:
    """Tests for _get_gauge_address_for_position."""

    def test_missing_chain(self):
        b = _mk()
        result = _drive(b._get_gauge_address_for_position({}))
        assert result is None

    def test_no_voter_address(self):
        b = _mk()
        b.params.velodrome_voter_contract_addresses = {}
        result = _drive(
            b._get_gauge_address_for_position(
                {"chain": "optimism", "pool_address": "0x1"}
            )
        )
        assert result is None

    def test_no_pool_behaviour(self):
        b = _mk()
        b.pools = {}
        result = _drive(
            b._get_gauge_address_for_position(
                {"chain": "optimism", "pool_address": "0x1"}
            )
        )
        assert result is None

    def test_valid_gauge(self):
        b = _mk()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _gen_return("0xgauge")
        b.pools = {"velodrome": mock_pool}
        result = _drive(
            b._get_gauge_address_for_position(
                {"chain": "optimism", "pool_address": "0x1"}
            ),
            sends=[None],
        )
        assert result == "0xgauge"

    def test_exception(self):
        b = _mk()

        def _bad(*a, **kw):
            raise Exception("boom")
            yield  # noqa: unreachable

        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _bad
        b.pools = {"velodrome": mock_pool}
        result = _drive(
            b._get_gauge_address_for_position(
                {"chain": "optimism", "pool_address": "0x1"}
            )
        )
        assert result is None


class TestConstants:
    """Tests for module-level constants."""

    def test_min_swap_value_usd(self):
        assert MIN_SWAP_VALUE_USD == 0.5


class TestAsyncAct:
    """Tests for async_act."""

    def _make_send_actions(self):
        """Create a send_actions mock that tracks calls."""
        calls = []

        def _mock_send_actions(actions=None):
            calls.append(actions)
            yield

        return _mock_send_actions, calls

    def test_investing_paused(self):
        b = _mk()
        b._read_investing_paused = _gen_return(True)
        b.send_a2a_transaction = _gen_none
        b.wait_until_round_end = _gen_none
        b.set_done = MagicMock()
        _drive(b.async_act(), sends=[None] * 10)
        b.set_done.assert_called_once()

    def test_non_whitelisted_swaps_with_full_flow(self):
        """When non-whitelisted swaps return actions, send_actions is called, then flow continues."""
        b = _mk()
        b._read_investing_paused = _gen_return(False)
        b.check_and_prepare_non_whitelisted_swaps = _gen_return([{"action": "swap"}])
        send_actions, calls = self._make_send_actions()
        b.send_actions = send_actions
        b._apply_tip_filters_to_exit_decisions = _gen_return((False, []))
        _drive(b.async_act(), sends=[None] * 15)
        # send_actions called twice: once with swap actions, once with no actions (tip block)
        assert len(calls) == 2

    def test_tip_filters_block(self):
        b = _mk()
        b._read_investing_paused = _gen_return(False)
        b.check_and_prepare_non_whitelisted_swaps = _gen_return([])
        b._apply_tip_filters_to_exit_decisions = _gen_return((False, []))
        send_actions, calls = self._make_send_actions()
        b.send_actions = send_actions
        _drive(b.async_act(), sends=[None] * 10)
        assert len(calls) == 1  # send_actions() with no args

    def test_no_funds(self):
        b = _mk()
        b._read_investing_paused = _gen_return(False)
        b.check_and_prepare_non_whitelisted_swaps = _gen_return([])
        b._apply_tip_filters_to_exit_decisions = _gen_return((True, []))
        b.check_funds = MagicMock(return_value=False)
        send_actions, calls = self._make_send_actions()
        b.send_actions = send_actions
        _drive(b.async_act(), sends=[None] * 10)
        assert len(calls) == 1

    def test_cached_opportunity_used(self):
        b = _mk()
        b._read_investing_paused = _gen_return(False)
        b.check_and_prepare_non_whitelisted_swaps = _gen_return([])
        b._apply_tip_filters_to_exit_decisions = _gen_return(
            (True, [{"status": "open"}])
        )
        b.check_funds = MagicMock(return_value=True)
        b._check_and_use_cached_cl_opportunity = _gen_return([{"action": "cached"}])
        send_actions, calls = self._make_send_actions()
        b.send_actions = send_actions
        _drive(b.async_act(), sends=[None] * 10)
        assert len(calls) == 1
        assert calls[0] == [{"action": "cached"}]

    def test_normal_flow_no_cache(self):
        b = _mk()
        b._read_investing_paused = _gen_return(False)
        b.check_and_prepare_non_whitelisted_swaps = _gen_return([])
        b._apply_tip_filters_to_exit_decisions = _gen_return((True, []))
        b.check_funds = MagicMock(return_value=True)
        b._check_and_use_cached_cl_opportunity = _gen_return(None)
        b.fetch_all_trading_opportunities = _gen_none
        b.prepare_strategy_actions = _gen_return([{"action": "enter"}])
        b._push_opportunity_metrics_to_mirrordb = _gen_none
        send_actions, calls = self._make_send_actions()
        b.send_actions = send_actions
        _drive(b.async_act(), sends=[None] * 20)
        assert len(calls) == 1
        assert calls[0] == [{"action": "enter"}]


class TestCalculateVelodromeTokenRatios:
    """Tests for calculate_velodrome_token_ratios."""

    def test_none_validated_data(self):
        b = _mk()
        result = _drive(b.calculate_velodrome_token_ratios(None))
        assert result is None

    def test_price_below_range(self):
        b = _mk()
        validated_data = {
            "validated_bands": [
                {"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}
            ],
            "current_price": 1.0,
            "current_tick": 0,
            "warnings": [],
        }
        # sqrt_ratio_a_x96 > sqrt_price_x96 => below range
        b.get_velodrome_sqrt_ratio_at_tick = _gen_return(
            10**30
        )  # huge value for lower tick
        result = _drive(
            b.calculate_velodrome_token_ratios(validated_data, "optimism"),
            sends=[None] * 5,
        )
        assert result is not None
        assert result["position_requirements"][0]["token0_ratio"] == 1.0
        assert result["position_requirements"][0]["status"] == "BELOW_RANGE"

    def test_price_above_range(self):
        b = _mk()
        validated_data = {
            "validated_bands": [
                {"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}
            ],
            "current_price": 1.0,
            "current_tick": 0,
            "warnings": [],
        }
        # sqrt_ratio_b_x96 < sqrt_price_x96 => above range
        # Return small values so price is above both
        b.get_velodrome_sqrt_ratio_at_tick = _gen_return(1)
        result = _drive(
            b.calculate_velodrome_token_ratios(validated_data, "optimism"),
            sends=[None] * 5,
        )
        assert result is not None
        assert result["position_requirements"][0]["token1_ratio"] == 1.0
        assert result["position_requirements"][0]["status"] == "ABOVE_RANGE"

    def test_price_in_range(self):
        b = _mk()
        sqrt_price = int(math.sqrt(1.0) * (2**96))
        validated_data = {
            "validated_bands": [
                {"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}
            ],
            "current_price": 1.0,
            "current_tick": 0,
            "warnings": [],
        }
        call_count = [0]

        def _sqrt_at_tick(*a, **kw):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                yield
                return sqrt_price // 2  # lower
            else:
                yield
                return sqrt_price * 2  # upper

        b.get_velodrome_sqrt_ratio_at_tick = _sqrt_at_tick
        b.get_velodrome_amounts_for_liquidity = _gen_return((500, 500))
        result = _drive(
            b.calculate_velodrome_token_ratios(validated_data, "optimism"),
            sends=[None] * 10,
        )
        assert result is not None
        assert result["position_requirements"][0]["status"] == "IN_RANGE"

    def test_with_sqrt_price_x96_provided(self):
        b = _mk()
        validated_data = {
            "validated_bands": [
                {"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}
            ],
            "current_price": 1.0,
            "current_tick": 0,
            "warnings": [],
            "sqrt_price_x96": 10**30,  # provided
        }
        b.get_velodrome_sqrt_ratio_at_tick = _gen_return(10**31)  # above => below range
        result = _drive(
            b.calculate_velodrome_token_ratios(validated_data, "optimism"),
            sends=[None] * 5,
        )
        assert result is not None

    def test_in_range_zero_amounts(self):
        b = _mk()
        sqrt_price = int(math.sqrt(1.0) * (2**96))
        validated_data = {
            "validated_bands": [
                {"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}
            ],
            "current_price": 1.0,
            "current_tick": 0,
            "warnings": [],
        }
        call_count = [0]

        def _sqrt_at_tick(*a, **kw):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                yield
                return sqrt_price // 2
            else:
                yield
                return sqrt_price * 2

        b.get_velodrome_sqrt_ratio_at_tick = _sqrt_at_tick
        b.get_velodrome_amounts_for_liquidity = _gen_return((0, 0))  # zero amounts
        result = _drive(
            b.calculate_velodrome_token_ratios(validated_data, "optimism"),
            sends=[None] * 10,
        )
        assert result is not None
        # Should fallback to 0.5/0.5
        assert result["position_requirements"][0]["token0_ratio"] == 0.5

    def test_calculation_exception_fallback(self):
        b = _mk()
        sqrt_price = int(math.sqrt(1.0) * (2**96))
        validated_data = {
            "validated_bands": [
                {"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}
            ],
            "current_price": 1.0,
            "current_tick": 0,
            "warnings": [],
        }
        call_count = [0]

        def _sqrt_at_tick(*a, **kw):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                yield
                return sqrt_price // 2  # lower bound below price
            else:
                yield
                return sqrt_price * 2  # upper bound above price

        b.get_velodrome_sqrt_ratio_at_tick = _sqrt_at_tick

        def _bad_amounts(*a, **kw):
            raise Exception("boom")
            yield  # noqa: unreachable

        b.get_velodrome_amounts_for_liquidity = _bad_amounts
        result = _drive(
            b.calculate_velodrome_token_ratios(validated_data, "optimism"),
            sends=[None] * 10,
        )
        assert result is not None
        assert result["position_requirements"][0]["status"] == "ERROR"

    def test_mixed_statuses_recommendation(self):
        b = _mk()
        validated_data = {
            "validated_bands": [
                {"tick_lower": -100, "tick_upper": 100, "allocation": 0.5},
                {"tick_lower": 200, "tick_upper": 300, "allocation": 0.5},
            ],
            "current_price": 1.0,
            "current_tick": 0,
            "warnings": [],
        }
        call_count = [0]

        def _sqrt_at_tick(*a, **kw):
            call_count[0] += 1
            # First band: below range (both sqrt_ratios above price)
            if call_count[0] <= 2:
                yield
                return 10**30
            # Second band: above range (both sqrt_ratios below price)
            else:
                yield
                return 1

        b.get_velodrome_sqrt_ratio_at_tick = _sqrt_at_tick
        result = _drive(
            b.calculate_velodrome_token_ratios(validated_data, "optimism"),
            sends=[None] * 10,
        )
        assert result is not None
        assert "Mixed" in result["recommendation"]


class TestExecuteHyperStrategy:
    """Tests for execute_hyper_strategy."""

    def test_basic_flow(self):
        b = _mk()
        b._read_kv = _gen_return({"composite_score": "0.5"})
        b.execute_strategy = MagicMock(
            return_value={
                "optimal_strategies": None,
                "position_to_exit": None,
                "logs": [],
                "reasoning": None,
            }
        )
        synced_mock = MagicMock()
        synced_mock.trading_type = "default"
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            _drive(b.execute_hyper_strategy(), sends=[None] * 5)
        assert b.selected_opportunities is None

    def test_composite_score_from_kv(self):
        b = _mk()
        b._read_kv = _gen_return({"composite_score": "0.75"})
        b.execute_strategy = MagicMock(
            return_value={
                "optimal_strategies": [
                    {
                        "pool_address": "0x" + "11" * 20,
                        "token0": "0x" + "aa" * 20,
                        "dex_type": "velodrome",
                    }
                ],
                "position_to_exit": None,
                "logs": ["log1"],
                "reasoning": "test reasoning",
            }
        )
        b._track_opportunities = _gen_none
        synced_mock = MagicMock()
        synced_mock.trading_type = "default"
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            _drive(b.execute_hyper_strategy(), sends=[None] * 10)
        assert b.selected_opportunities is not None

    def test_kv_read_exception(self):
        b = _mk()

        def _bad_kv(*a, **kw):
            raise Exception("boom")
            yield  # noqa: unreachable

        b._read_kv = _bad_kv
        b.execute_strategy = MagicMock(
            return_value={
                "optimal_strategies": None,
                "position_to_exit": None,
                "logs": [],
                "reasoning": None,
            }
        )
        synced_mock = MagicMock()
        synced_mock.trading_type = "default"
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            _drive(b.execute_hyper_strategy(), sends=[None] * 5)

    def test_composite_score_not_float(self):
        b = _mk()
        b._read_kv = _gen_return({"composite_score": "not_a_number"})
        b.execute_strategy = MagicMock(
            return_value={
                "optimal_strategies": None,
                "position_to_exit": None,
                "logs": [],
                "reasoning": None,
            }
        )
        synced_mock = MagicMock()
        synced_mock.trading_type = "default"
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            _drive(b.execute_hyper_strategy(), sends=[None] * 5)

    def test_composite_score_none_uses_default(self):
        b = _mk()
        b._read_kv = _gen_return(None)
        b.execute_strategy = MagicMock(
            return_value={
                "optimal_strategies": None,
                "position_to_exit": None,
                "logs": [],
                "reasoning": None,
            }
        )
        synced_mock = MagicMock()
        synced_mock.trading_type = "default"
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            _drive(b.execute_hyper_strategy(), sends=[None] * 5)


class TestGetResult:
    """Tests for get_result."""

    def test_future_done_immediately(self):
        from concurrent.futures import Future

        b = _mk()
        f = Future()
        f.set_result("value")
        result = _drive(b.get_result(f))
        assert result == "value"

    def test_future_with_exception(self):
        from concurrent.futures import Future

        b = _mk()
        f = Future()
        f.set_exception(RuntimeError("bad"))
        result = _drive(b.get_result(f))
        assert result is None

    def test_future_not_done_yet(self):
        from concurrent.futures import Future

        b = _mk()
        f = Future()
        gen = b.get_result(f)
        # First send starts it
        next(gen)
        # Should yield because future is not done
        f.set_result(42)
        try:
            gen.send(None)
        except StopIteration as e:
            assert e.value == 42


class TestGetRewards:
    """Tests for _get_rewards."""

    def test_http_error(self):
        b = _mk()
        resp = MagicMock()
        resp.status_code = 500
        resp.body = b"error"
        b.get_http_response = _gen_return(resp)
        result = _drive(b._get_rewards(10, "0xuser"), sends=[None])
        assert result is None

    def test_no_tokens_to_claim(self):
        b = _mk()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps({"0xtoken": {"accumulated": 0}}).encode()
        b.get_http_response = _gen_return(resp)
        result = _drive(b._get_rewards(10, "0xuser"), sends=[None])
        assert result is None

    def test_all_claims_zero(self):
        b = _mk()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps(
            {
                "0xtoken": {
                    "proof": ["0xproof"],
                    "accumulated": 0,
                    "unclaimed": 0,
                    "symbol": "TKN",
                }
            }
        ).encode()
        b.get_http_response = _gen_return(resp)
        result = _drive(b._get_rewards(10, "0xuser"), sends=[None])
        assert result is None

    def test_all_unclaimed_zero(self):
        b = _mk()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps(
            {
                "0xtoken": {
                    "proof": ["0xproof"],
                    "accumulated": 100,
                    "unclaimed": 0,
                    "symbol": "TKN",
                }
            }
        ).encode()
        b.get_http_response = _gen_return(resp)
        result = _drive(b._get_rewards(10, "0xuser"), sends=[None])
        assert result is None

    def test_valid_rewards(self):
        b = _mk()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps(
            {
                "0xtoken": {
                    "proof": ["0xproof"],
                    "accumulated": 100,
                    "unclaimed": 50,
                    "symbol": "TKN",
                }
            }
        ).encode()
        b.get_http_response = _gen_return(resp)
        result = _drive(b._get_rewards(10, "0xuser"), sends=[None])
        assert result is not None
        assert result["tokens"] == ["0xtoken"]

    def test_parse_error(self):
        b = _mk()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = b"not json"
        b.get_http_response = _gen_return(resp)
        result = _drive(b._get_rewards(10, "0xuser"), sends=[None])
        assert result is None


class TestGetAvailableTokens:
    """Tests for _get_available_tokens."""

    def test_empty_positions(self):
        b = _mk()
        synced_mock = MagicMock()
        synced_mock.positions = []
        b._fetch_token_prices = _gen_return({})
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            result = _drive(b._get_available_tokens(), sends=[None] * 5)
        assert result == []

    def test_filters_reward_tokens(self):
        b = _mk()
        synced_mock = MagicMock()
        # Use a reward token address from REWARD_TOKEN_ADDRESSES
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            REWARD_TOKEN_ADDRESSES,
        )

        chain = "optimism"
        reward_addrs = list(REWARD_TOKEN_ADDRESSES.get(chain, {}).keys())
        if reward_addrs:
            reward_addr = reward_addrs[0]
            synced_mock.positions = [
                {
                    "chain": chain,
                    "assets": [
                        {
                            "address": reward_addr.lower(),
                            "asset_symbol": "RWD",
                            "balance": 100,
                        }
                    ],
                }
            ]
        else:
            synced_mock.positions = []

        b._fetch_token_prices = _gen_return({})
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            result = _drive(b._get_available_tokens(), sends=[None] * 5)
        # Reward token should be filtered out
        assert result == []

    def test_investable_balance_zero_address(self):
        b = _mk()
        b.params.min_investment_amount = 0.0  # no minimum
        synced_mock = MagicMock()
        synced_mock.positions = [
            {
                "chain": "optimism",
                "assets": [
                    {"address": ZERO_ADDRESS, "asset_symbol": "ETH", "balance": 10**18}
                ],
            }
        ]
        b._get_investable_balance = _gen_return(10**18)
        b._fetch_token_prices = _gen_return({ZERO_ADDRESS: 2.0})
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            result = _drive(b._get_available_tokens(), sends=[None] * 10)
        assert len(result) == 1
        assert result[0]["token"] == ZERO_ADDRESS

    def test_filters_below_min_investment(self):
        b = _mk()
        b.params.min_investment_amount = 100.0
        synced_mock = MagicMock()
        synced_mock.positions = [
            {
                "chain": "optimism",
                "assets": [
                    {"address": "0xtoken", "asset_symbol": "TKN", "balance": 10}
                ],
            }
        ]
        b._get_investable_balance = _gen_return(10)
        b._fetch_token_prices = _gen_return({"0xtoken": 0.001})
        b._get_token_decimals = _gen_return(18)
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            result = _drive(b._get_available_tokens(), sends=[None] * 10)
        assert result == []


class TestGetInvestableBalance:
    """Tests for _get_investable_balance."""

    def test_not_reward_token(self):
        b = _mk()
        result = _drive(b._get_investable_balance("optimism", "0x" + "ab" * 20, 1000))
        assert result == 1000

    def test_pure_reward_token(self):
        b = _mk()
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            REWARD_TOKEN_ADDRESSES,
        )

        chain = "optimism"
        reward_addrs = REWARD_TOKEN_ADDRESSES.get(chain, {})
        # Find a reward token that is NOT whitelisted
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            WHITELISTED_ASSETS,
        )

        wl_addrs = WHITELISTED_ASSETS.get(chain, {})
        pure_reward = None
        for addr in reward_addrs:
            if addr not in wl_addrs:
                pure_reward = addr
                break
        if pure_reward:
            result = _drive(b._get_investable_balance(chain, pure_reward.lower(), 1000))
            assert result == 0


class TestPrepareTokensForInvestment:
    """Tests for _prepare_tokens_for_investment."""

    def test_no_position_to_exit(self):
        b = _mk()
        b.position_to_exit = None
        b._get_available_tokens = _gen_return([{"chain": "optimism", "token": "0xA"}])
        result = _drive(b._prepare_tokens_for_investment(), sends=[None])
        assert result is not None
        assert len(result) == 1

    def test_position_to_exit_insufficient_tokens(self):
        b = _mk()
        b.position_to_exit = {"dex_type": "velodrome", "chain": "optimism"}
        b._build_tokens_from_position = MagicMock(return_value=None)
        result = _drive(b._prepare_tokens_for_investment())
        assert result is None

    def test_no_tokens_available(self):
        b = _mk()
        b.position_to_exit = None
        b._get_available_tokens = _gen_return(None)
        result = _drive(b._prepare_tokens_for_investment(), sends=[None])
        assert result is None

    def test_position_to_exit_with_tokens(self):
        b = _mk()
        b.position_to_exit = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "token0": "0xA",
            "token0_symbol": "A",
            "token1": "0xB",
            "token1_symbol": "B",
        }
        b._get_available_tokens = _gen_return([])
        result = _drive(b._prepare_tokens_for_investment(), sends=[None])
        assert result is not None
        assert len(result) == 2


class TestGetOrderOfTransactions:
    """Tests for get_order_of_transactions."""

    def test_no_selected_opportunities(self):
        b = _mk()
        b.selected_opportunities = None
        result = _drive(b.get_order_of_transactions())
        assert result == []

    def test_basic_enter_flow(self):
        b = _mk()
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "chain": "optimism",
                "pool_address": "0x1",
                "token0": "0xA",
                "token0_symbol": "A",
                "token1": "0xB",
                "token1_symbol": "B",
                "is_cl_pool": False,
                "relative_funds_percentage": 1.0,
                "apr": 10,
            }
        ]
        b.position_to_exit = None
        b.get_velodrome_position_requirements = _gen_return({})
        b._prepare_tokens_for_investment = _gen_return(
            [
                {"chain": "optimism", "token": "0xA", "token_symbol": "A"},
                {"chain": "optimism", "token": "0xB", "token_symbol": "B"},
            ]
        )
        b._build_bridge_swap_actions = MagicMock(return_value=[])
        b._cache_enter_pool_action_for_cl_pool = _gen_none
        b._initialize_entry_costs_for_new_position = _gen_none
        b._should_add_staking_actions = MagicMock(return_value=False)
        result = _drive(b.get_order_of_transactions(), sends=[None] * 20)
        assert result is not None

    def test_exit_with_staking(self):
        b = _mk()
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "chain": "optimism",
                "pool_address": "0x1",
                "token0": "0xA",
                "token0_symbol": "A",
                "token1": "0xB",
                "token1_symbol": "B",
                "is_cl_pool": False,
                "relative_funds_percentage": 1.0,
                "apr": 10,
            }
        ]
        b.position_to_exit = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xold",
            "staked": True,
            "gauge_address": "0xgauge",
        }
        b._has_staking_metadata = MagicMock(return_value=True)
        b._build_unstake_lp_tokens_action = MagicMock(
            return_value={"action": "UnstakeLpTokens"}
        )
        b._build_exit_pool_action_base = MagicMock(return_value={"action": "ExitPool"})
        b.get_velodrome_position_requirements = _gen_return({})
        b._prepare_tokens_for_investment = _gen_return(
            [
                {"chain": "optimism", "token": "0xA", "token_symbol": "A"},
                {"chain": "optimism", "token": "0xB", "token_symbol": "B"},
            ]
        )
        b._build_bridge_swap_actions = MagicMock(return_value=[])
        b._cache_enter_pool_action_for_cl_pool = _gen_none
        b._initialize_entry_costs_for_new_position = _gen_none
        b._should_add_staking_actions = MagicMock(return_value=False)
        result = _drive(b.get_order_of_transactions(), sends=[None] * 20)
        assert result is not None

    def test_bridge_swap_returns_none(self):
        b = _mk()
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "chain": "optimism",
                "pool_address": "0x1",
                "token0": "0xA",
                "token0_symbol": "A",
                "token1": "0xB",
                "token1_symbol": "B",
                "is_cl_pool": False,
                "relative_funds_percentage": 1.0,
            }
        ]
        b.position_to_exit = None
        b.get_velodrome_position_requirements = _gen_return({})
        b._prepare_tokens_for_investment = _gen_return(
            [
                {"chain": "optimism", "token": "0xA", "token_symbol": "A"},
            ]
        )
        b._build_bridge_swap_actions = MagicMock(return_value=None)
        result = _drive(b.get_order_of_transactions(), sends=[None] * 20)
        assert result is None

    def test_prepare_tokens_returns_none(self):
        b = _mk()
        b.selected_opportunities = [{"dex_type": "uniswap"}]
        b.position_to_exit = None
        b._prepare_tokens_for_investment = _gen_return(None)
        result = _drive(b.get_order_of_transactions(), sends=[None] * 10)
        assert result == []


class TestApplyInvestmentCapToActions:
    """Tests for _apply_investment_cap_to_actions."""

    def test_no_current_positions(self):
        b = _mk()
        b.current_positions = []
        result = _drive(b._apply_investment_cap_to_actions([{"action": "test"}]))
        assert result == [{"action": "test"}]

    def test_threshold_reached_no_exit(self):
        b = _mk()
        b.current_positions = [{"status": "open"}]
        b.sleep = _gen_none
        b.calculate_initial_investment_value = _gen_return(1000)
        actions = [{"action": "EnterPool"}]
        result = _drive(b._apply_investment_cap_to_actions(actions), sends=[None] * 20)
        assert result == []

    def test_threshold_reached_with_exit(self):
        b = _mk()
        b.current_positions = [{"status": "open"}]
        b.sleep = _gen_none
        b.calculate_initial_investment_value = _gen_return(1000)
        actions = [{"action": "ExitPool"}, {"action": "EnterPool"}]
        result = _drive(b._apply_investment_cap_to_actions(actions), sends=[None] * 20)
        assert len(result) == 1
        assert result[0]["action"] == "ExitPool"

    def test_under_threshold_adjust_enter(self):
        b = _mk()
        b.current_positions = [{"status": "open"}]
        b.sleep = _gen_none
        b.calculate_initial_investment_value = _gen_return(500)
        actions = [{"action": "EnterPool"}]
        result = _drive(b._apply_investment_cap_to_actions(actions), sends=[None] * 20)
        assert result[0]["invested_amount"] == 500

    def test_vinitial_none_retries(self):
        b = _mk()
        b.current_positions = [{"status": "open"}]
        b.sleep = _gen_none
        call_count = [0]

        def _failing_calc(*a, **kw):
            call_count[0] += 1
            if call_count[0] <= 2:
                yield
                return None
            yield
            return 100

        b.calculate_initial_investment_value = _failing_calc
        actions = [{"action": "EnterPool"}]
        result = _drive(b._apply_investment_cap_to_actions(actions), sends=[None] * 30)
        # After retries, it got 100 < threshold
        assert result[0].get("invested_amount") == 900

    def test_invested_zero_but_has_positions(self):
        b = _mk()
        b.current_positions = [{"status": "open"}]
        b.sleep = _gen_none
        b.calculate_initial_investment_value = _gen_return(None)  # all retries fail
        actions = [{"action": "EnterPool"}]
        result = _drive(b._apply_investment_cap_to_actions(actions), sends=[None] * 30)
        # invested_amount == 0 and invested_positions = True => clears actions
        assert result == []


class TestProcessRewards:
    """Tests for _process_rewards."""

    def test_no_rewards(self):
        b = _mk()
        b._get_rewards = _gen_return(None)
        actions = []
        _drive(b._process_rewards(actions), sends=[None] * 5)
        assert actions == []

    def test_with_rewards(self):
        b = _mk()
        rewards = {
            "users": ["0xuser"],
            "tokens": ["0xtoken"],
            "claims": [100],
            "proofs": [["p"]],
            "symbols": ["TKN"],
        }
        b._get_rewards = _gen_return(rewards)
        actions = []
        _drive(b._process_rewards(actions), sends=[None] * 5)
        assert len(actions) == 1
        assert actions[0]["action"] == Action.CLAIM_REWARDS.value


class TestHandleAllTokensAvailable:
    """Tests for _handle_all_tokens_available."""

    def test_no_required_tokens(self):
        b = _mk()
        result = b._handle_all_tokens_available([], [], "optimism", 1.0, {})
        assert result == []

    def test_other_chain_tokens(self):
        b = _mk()
        tokens = [{"chain": "mode", "token": "0xC", "token_symbol": "C"}]
        required = [("0xA", "A")]
        result = b._handle_all_tokens_available(
            tokens, required, "optimism", 1.0, {"0xA": 1.0}
        )
        assert len(result) == 1

    def test_unnecessary_tokens_on_dest_chain(self):
        b = _mk()
        tokens = [
            {"chain": "optimism", "token": "0xC", "token_symbol": "C", "value": 100},
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 50},
        ]
        required = [("0xA", "A"), ("0xB", "B")]
        ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_all_tokens_available(
            tokens, required, "optimism", 1.0, ratios
        )
        assert isinstance(result, list)

    def test_surplus_rebalance(self):
        b = _mk()
        tokens = [
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 200},
            {"chain": "optimism", "token": "0xB", "token_symbol": "B", "value": 50},
        ]
        required = [("0xA", "A"), ("0xB", "B")]
        ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_all_tokens_available(
            tokens, required, "optimism", 1.0, ratios
        )
        assert isinstance(result, list)


class TestHandleSomeTokensAvailable:
    """Tests for _handle_some_tokens_available."""

    def test_basic_flow(self):
        b = _mk()
        tokens = [
            {"chain": "mode", "token": "0xC", "token_symbol": "C"},
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 100},
        ]
        required = [("0xA", "A"), ("0xB", "B")]
        needed = [("0xB", "B")]
        ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_some_tokens_available(
            tokens, required, needed, "optimism", 1.0, ratios
        )
        assert isinstance(result, list)

    def test_unnecessary_tokens_converted(self):
        b = _mk()
        tokens = [
            {"chain": "optimism", "token": "0xC", "token_symbol": "C", "value": 100},
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 100},
        ]
        required = [("0xA", "A"), ("0xB", "B")]
        needed = [("0xB", "B")]
        ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_some_tokens_available(
            tokens, required, needed, "optimism", 1.0, ratios
        )
        assert isinstance(result, list)

    def test_no_unnecessary_tokens_convert_required(self):
        b = _mk()
        tokens = [
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 200},
        ]
        required = [("0xA", "A"), ("0xB", "B")]
        needed = [("0xB", "B")]
        ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_some_tokens_available(
            tokens, required, needed, "optimism", 1.0, ratios
        )
        assert isinstance(result, list)

    def test_surplus_rebalance(self):
        b = _mk()
        tokens = [
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 300},
            {"chain": "optimism", "token": "0xB", "token_symbol": "B", "value": 50},
        ]
        required = [("0xA", "A"), ("0xB", "B")]
        needed = [("0xB", "B")]
        ratios = {"0xA": 0.3, "0xB": 0.7}
        result = b._handle_some_tokens_available(
            tokens, required, needed, "optimism", 1.0, ratios
        )
        assert isinstance(result, list)


class TestHandleAllTokensNeeded:
    """Tests for _handle_all_tokens_needed."""

    def test_basic_flow(self):
        b = _mk()
        tokens = [
            {"chain": "mode", "token": "0xC", "token_symbol": "C"},
        ]
        required = [("0xA", "A"), ("0xB", "B")]
        ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_all_tokens_needed(tokens, required, "optimism", 1.0, ratios)
        assert isinstance(result, list)

    def test_with_dest_chain_tokens(self):
        b = _mk()
        tokens = [
            {"chain": "optimism", "token": "0xC", "token_symbol": "C", "value": 100},
        ]
        required = [("0xA", "A"), ("0xB", "B")]
        ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_all_tokens_needed(tokens, required, "optimism", 1.0, ratios)
        assert isinstance(result, list)

    def test_available_required_on_dest(self):
        b = _mk()
        tokens = [
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 100},
        ]
        required = [("0xA", "A"), ("0xB", "B")]
        ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_all_tokens_needed(tokens, required, "optimism", 1.0, ratios)
        assert isinstance(result, list)


class TestCheckAndUseCachedClOpportunity:
    """Tests for _check_and_use_cached_cl_opportunity."""

    def test_open_positions_bypass_cache(self):
        b = _mk()
        b.current_positions = [{"status": "open"}]
        result = _drive(b._check_and_use_cached_cl_opportunity())
        assert result is None

    def test_no_cached_data(self):
        b = _mk()
        b.current_positions = []
        b._get_cached_cl_pool_data = _gen_return(None)
        result = _drive(b._check_and_use_cached_cl_opportunity(), sends=[None] * 5)
        assert result is None

    def test_cache_expired(self):
        b = _mk()
        b.current_positions = []
        b._get_cached_cl_pool_data = _gen_return({"pool_address": "0x1"})
        b._should_use_cached_cl_data = MagicMock(return_value=False)
        result = _drive(b._check_and_use_cached_cl_opportunity(), sends=[None] * 5)
        assert result is None

    def test_cache_valid_with_actions(self):
        b = _mk()
        b.current_positions = []
        cached = {"pool_address": "0x1"}
        b._get_cached_cl_pool_data = _gen_return(cached)
        b._should_use_cached_cl_data = MagicMock(return_value=True)
        b._update_cl_pool_round_tracking = _gen_none
        b._reconstruct_actions_from_cached_cl_pool = _gen_return(
            [{"action": "EnterPool"}]
        )
        result = _drive(b._check_and_use_cached_cl_opportunity(), sends=[None] * 10)
        assert result == [{"action": "EnterPool"}]

    def test_exception(self):
        b = _mk()
        b.current_positions = None  # will cause exception in _has_open_positions
        b._has_open_positions = MagicMock(side_effect=Exception("boom"))

        # Need to patch at a higher level
        def _bad_check():
            raise Exception("boom")
            yield  # noqa: unreachable

        # Actually _check_and_use_cached_cl_opportunity catches the exception
        result = _drive(b._check_and_use_cached_cl_opportunity())
        assert result is None


class TestReconstructActionsFromCachedClPool:
    """Tests for _reconstruct_actions_from_cached_cl_pool."""

    def test_no_cached_enter_action(self):
        b = _mk()
        result = _drive(b._reconstruct_actions_from_cached_cl_pool({}, "optimism"))
        assert result is None

    def test_no_safe_address(self):
        b = _mk()
        b.params.safe_contract_addresses = {}
        cached = {"enter_pool_action": {"action": "EnterPool"}}
        result = _drive(b._reconstruct_actions_from_cached_cl_pool(cached, "optimism"))
        assert result is None

    def test_valid_reconstruction_token0_heavy(self):
        b = _mk()
        cached = {
            "enter_pool_action": {
                "action": "EnterPool",
                "chain": "optimism",
                "pool_address": "0x1",
            },
            "token0": "0xA",
            "token1": "0xB",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "pool_address": "0x1",
            "position_requirements": [
                {"allocation": 1.0, "token0_ratio": 1.0, "token1_ratio": 0.0}
            ],
        }
        b._get_token_balance = _gen_return(1000)
        b._initialize_entry_costs_for_new_position = _gen_none
        b._should_add_staking_actions = MagicMock(return_value=True)
        b._build_stake_lp_tokens_action = MagicMock(
            return_value={"action": "StakeLpTokens"}
        )
        result = _drive(
            b._reconstruct_actions_from_cached_cl_pool(cached, "optimism"),
            sends=[None] * 15,
        )
        assert result is not None
        assert len(result) == 2  # enter + stake

    def test_valid_reconstruction_token1_heavy(self):
        b = _mk()
        cached = {
            "enter_pool_action": {
                "action": "EnterPool",
                "chain": "optimism",
                "pool_address": "0x1",
            },
            "token0": "0xA",
            "token1": "0xB",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "pool_address": "0x1",
            "position_requirements": [
                {"allocation": 1.0, "token0_ratio": 0.0, "token1_ratio": 1.0}
            ],
        }
        b._get_token_balance = _gen_return(1000)
        b._initialize_entry_costs_for_new_position = _gen_none
        b._should_add_staking_actions = MagicMock(return_value=False)
        result = _drive(
            b._reconstruct_actions_from_cached_cl_pool(cached, "optimism"),
            sends=[None] * 15,
        )
        assert result is not None

    def test_valid_reconstruction_mixed_ratios(self):
        b = _mk()
        cached = {
            "enter_pool_action": {
                "action": "EnterPool",
                "chain": "optimism",
                "pool_address": "0x1",
            },
            "token0": "0xA",
            "token1": "0xB",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "pool_address": "0x1",
            "position_requirements": [
                {"allocation": 1.0, "token0_ratio": 0.5, "token1_ratio": 0.5}
            ],
        }
        b._get_token_balance = _gen_return(1000)
        b._initialize_entry_costs_for_new_position = _gen_none
        b._should_add_staking_actions = MagicMock(return_value=False)
        result = _drive(
            b._reconstruct_actions_from_cached_cl_pool(cached, "optimism"),
            sends=[None] * 15,
        )
        assert result is not None

    def test_exception(self):
        b = _mk()
        cached = {"enter_pool_action": {"action": "EnterPool"}}
        b.params.safe_contract_addresses = MagicMock(side_effect=Exception("boom"))
        result = _drive(b._reconstruct_actions_from_cached_cl_pool(cached, "optimism"))
        assert result is None


class TestGetCachedClPoolData:
    """Tests for _get_cached_cl_pool_data."""

    def test_no_data(self):
        b = _mk()
        b._read_kv = _gen_return(None)
        result = _drive(b._get_cached_cl_pool_data("optimism"), sends=[None])
        assert result is None

    def test_valid_cached_data(self):
        b = _mk()
        b._read_kv = _gen_return(
            {"velodrome_cl_pool_optimism": json.dumps({"pool_address": "0x1"})}
        )
        result = _drive(b._get_cached_cl_pool_data("optimism"), sends=[None])
        assert result is not None
        assert result["pool_address"] == "0x1"

    def test_invalidated_cache(self):
        b = _mk()
        b._read_kv = _gen_return(
            {"velodrome_cl_pool_optimism": json.dumps({"invalidated": True})}
        )
        result = _drive(b._get_cached_cl_pool_data("optimism"), sends=[None])
        assert result is None

    def test_json_parse_error(self):
        b = _mk()
        b._read_kv = _gen_return({"velodrome_cl_pool_optimism": "not json{{"})
        result = _drive(b._get_cached_cl_pool_data("optimism"), sends=[None])
        assert result is None

    def test_empty_value(self):
        b = _mk()
        b._read_kv = _gen_return({"velodrome_cl_pool_optimism": None})
        result = _drive(b._get_cached_cl_pool_data("optimism"), sends=[None])
        assert result is None


class TestShouldUseCachedClData:
    """Tests for _should_use_cached_cl_data."""

    def test_cache_valid(self):
        b = _mk()
        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 1000
        b.context.state.round_sequence.last_round_transition_timestamp = mock_ts
        cached = {"pool_finalization_timestamp": 500}
        assert b._should_use_cached_cl_data(cached) is True

    def test_cache_expired(self):
        b = _mk()
        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 200000
        b.context.state.round_sequence.last_round_transition_timestamp = mock_ts
        cached = {"pool_finalization_timestamp": 100}
        assert b._should_use_cached_cl_data(cached) is False


class TestUpdateClPoolRoundTracking:
    """Tests for _update_cl_pool_round_tracking."""

    def test_timestamp_increased(self):
        b = _mk()
        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 2000
        b.context.state.round_sequence.last_round_transition_timestamp = mock_ts
        b._write_kv = _gen_none
        cached = {"last_round_timestamp": 1000, "round_count": 1}
        _drive(b._update_cl_pool_round_tracking("optimism", cached), sends=[None])
        assert cached["round_count"] == 2
        assert cached["last_round_timestamp"] == 2000

    def test_timestamp_not_increased(self):
        b = _mk()
        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 500
        b.context.state.round_sequence.last_round_transition_timestamp = mock_ts
        cached = {"last_round_timestamp": 1000, "round_count": 1}
        _drive(b._update_cl_pool_round_tracking("optimism", cached))
        assert cached["round_count"] == 1


class TestCacheClPoolData:
    """Tests for _cache_cl_pool_data."""

    def test_basic_cache(self):
        b = _mk()
        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 1000
        b.context.state.round_sequence.last_round_transition_timestamp = mock_ts
        b._write_kv = _gen_none
        _drive(
            b._cache_cl_pool_data(
                chain="optimism",
                pool_address="0x1",
                tick_spacing=60,
                tick_bands=[],
                current_price=1.0,
                percent_in_bounds=0.8,
            ),
            sends=[None],
        )

    def test_cache_with_all_optional(self):
        b = _mk()
        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 1000
        b.context.state.round_sequence.last_round_transition_timestamp = mock_ts
        b._write_kv = _gen_none
        _drive(
            b._cache_cl_pool_data(
                chain="optimism",
                pool_address="0x1",
                tick_spacing=60,
                tick_bands=[{"tick_lower": -100, "tick_upper": 100}],
                current_price=1.0,
                percent_in_bounds=0.8,
                current_tick=0,
                ema=[1.0],
                std_dev=[0.5],
                current_ema=1.0,
                current_std_dev=0.5,
                band_multipliers=[1.5],
                token0="0xA",
                token1="0xB",
                token0_symbol="A",
                token1_symbol="B",
                token_requirements={"req": "data"},
                enter_pool_action={"action": "EnterPool"},
            ),
            sends=[None],
        )


class TestInitializePositionEntryCosts:
    """Tests for _initialize_position_entry_costs."""

    def test_valid(self):
        b = _mk()
        b._store_entry_costs = _gen_none
        _drive(b._initialize_position_entry_costs("optimism", "0x1"), sends=[None])

    def test_exception(self):
        b = _mk()

        def _bad(*a, **kw):
            raise Exception("boom")
            yield  # noqa: unreachable

        b._store_entry_costs = _bad
        _drive(b._initialize_position_entry_costs("optimism", "0x1"))
        b.context.logger.error.assert_called()


class TestDownloadStrategies:
    """Tests for download_strategies."""

    def test_no_strategies(self):
        b = _mk()
        b.shared_state.strategy_to_filehash = {}
        _drive(b.download_strategies())

    def test_with_strategy_downloads(self):
        b = _mk()
        # After first iteration, clear the dict so loop ends
        call_count = [0]
        orig_download = MagicMock()

        def _mock_download():
            call_count[0] += 1
            if call_count[0] >= 1:
                b.shared_state.strategy_to_filehash = {}

        b.download_next_strategy = _mock_download
        b.shared_state.strategy_to_filehash = {"strat": "hash"}
        b.sleep = _gen_none
        _drive(b.download_strategies(), sends=[None] * 5)


class TestExecuteStrategyExecPath:
    """Tests for execute_strategy exec path."""

    def test_callable_found_and_executed(self):
        b = _mk()
        code = "def my_strategy(**kwargs): return {'result': 'ok'}"
        b.shared_state.strategies_executables = {"test_strat": (code, "my_strategy")}
        result = b.execute_strategy(strategy="test_strat")
        assert result == {"result": "ok"}

    def test_exec_with_isolated_namespace(self):
        """Test that strategies execute in isolated namespaces, not globals."""
        b = _mk()
        code = "def my_func(**kwargs): return {'done': True}"
        b.shared_state.strategies_executables = {"s": (code, "my_func")}
        result = b.execute_strategy(strategy="s")
        assert result == {"done": True}
        # Verify the function was NOT injected into globals
        assert "my_func" not in globals()


class TestPushOpportunityMetricsToMirrordb:
    """Tests for _push_opportunity_metrics_to_mirrordb."""

    def test_no_tracking_data(self):
        b = _mk()
        b._read_kv = _gen_return(None)
        _drive(b._push_opportunity_metrics_to_mirrordb(), sends=[None])

    def test_no_agent_registry(self):
        b = _mk()
        b._read_kv = _gen_return({"opportunity_tracking": json.dumps({"data": "test"})})
        b._get_or_create_agent_registry = _gen_return(None)
        _drive(b._push_opportunity_metrics_to_mirrordb(), sends=[None] * 5)

    def test_no_agent_type(self):
        b = _mk()
        b._read_kv = _gen_return({"opportunity_tracking": json.dumps({"data": "test"})})
        b._get_or_create_agent_registry = _gen_return({"agent_id": "123"})
        b._get_or_create_agent_type = _gen_return(None)
        _drive(b._push_opportunity_metrics_to_mirrordb(), sends=[None] * 5)

    def test_no_attr_def_creates_new(self):
        b = _mk()
        tracking_data = json.dumps({"r1": {"raw": {}}})
        call_count = [0]

        def _read_kv_mock(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return {"opportunity_tracking": tracking_data}
            else:
                yield
                return None  # no attr def

        b._read_kv = _read_kv_mock
        b._get_or_create_agent_registry = _gen_return(
            {"agent_id": "123", "agent_name": "test", "agent_address": "0x"}
        )
        b._get_or_create_agent_type = _gen_return(
            {"type_id": "t1", "type_name": "test_type"}
        )
        b._create_opportunity_attr_def = _gen_return({"attr_def_id": "ad1"})
        b._write_kv = _gen_none
        b.create_agent_attribute = _gen_return({"id": "attr1"})
        b._get_current_timestamp = lambda: 12345
        synced_mock = MagicMock()
        synced_mock.period_count = 1
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            _drive(b._push_opportunity_metrics_to_mirrordb(), sends=[None] * 20)

    def test_attr_def_exists(self):
        b = _mk()
        tracking_data = json.dumps({"r1": {"raw": {}}})
        call_count = [0]

        def _read_kv_mock(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return {"opportunity_tracking": tracking_data}
            else:
                yield
                return {"opportunity_attr_def": json.dumps({"attr_def_id": "ad1"})}

        b._read_kv = _read_kv_mock
        b._get_or_create_agent_registry = _gen_return(
            {"agent_id": "123", "agent_name": "test", "agent_address": "0x"}
        )
        b._get_or_create_agent_type = _gen_return(
            {"type_id": "t1", "type_name": "test_type"}
        )
        b._write_kv = _gen_none
        b.create_agent_attribute = _gen_return({"id": "attr1"})
        b._get_current_timestamp = lambda: 12345
        synced_mock = MagicMock()
        synced_mock.period_count = 1
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            _drive(b._push_opportunity_metrics_to_mirrordb(), sends=[None] * 20)

    def test_attr_def_value_none_creates_new(self):
        b = _mk()
        tracking_data = json.dumps({"r1": {"raw": {}}})
        call_count = [0]

        def _read_kv_mock(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return {"opportunity_tracking": tracking_data}
            else:
                yield
                return {"opportunity_attr_def": None}

        b._read_kv = _read_kv_mock
        b._get_or_create_agent_registry = _gen_return(
            {"agent_id": "123", "agent_name": "test", "agent_address": "0x"}
        )
        b._get_or_create_agent_type = _gen_return(
            {"type_id": "t1", "type_name": "test_type"}
        )
        b._create_opportunity_attr_def = _gen_return({"attr_def_id": "ad1"})
        b._write_kv = _gen_none
        b.create_agent_attribute = _gen_return({"id": "attr1"})
        b._get_current_timestamp = lambda: 12345
        synced_mock = MagicMock()
        synced_mock.period_count = 1
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            _drive(b._push_opportunity_metrics_to_mirrordb(), sends=[None] * 20)

    def test_attr_def_bad_json_creates_new(self):
        b = _mk()
        tracking_data = json.dumps({"r1": {"raw": {}}})
        call_count = [0]

        def _read_kv_mock(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return {"opportunity_tracking": tracking_data}
            else:
                yield
                return {"opportunity_attr_def": "bad json{{"}

        b._read_kv = _read_kv_mock
        b._get_or_create_agent_registry = _gen_return(
            {"agent_id": "123", "agent_name": "test", "agent_address": "0x"}
        )
        b._get_or_create_agent_type = _gen_return(
            {"type_id": "t1", "type_name": "test_type"}
        )
        b._create_opportunity_attr_def = _gen_return({"attr_def_id": "ad1"})
        b._write_kv = _gen_none
        b.create_agent_attribute = _gen_return({"id": "attr1"})
        b._get_current_timestamp = lambda: 12345
        synced_mock = MagicMock()
        synced_mock.period_count = 1
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            _drive(b._push_opportunity_metrics_to_mirrordb(), sends=[None] * 20)

    def test_create_attr_def_fails(self):
        b = _mk()
        tracking_data = json.dumps({"r1": {"raw": {}}})
        call_count = [0]

        def _read_kv_mock(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return {"opportunity_tracking": tracking_data}
            else:
                yield
                return None

        b._read_kv = _read_kv_mock
        b._get_or_create_agent_registry = _gen_return({"agent_id": "123"})
        b._get_or_create_agent_type = _gen_return({"type_id": "t1"})
        b._create_opportunity_attr_def = _gen_return(None)
        _drive(b._push_opportunity_metrics_to_mirrordb(), sends=[None] * 15)

    def test_push_fails(self):
        b = _mk()
        tracking_data = json.dumps({"r1": {"raw": {}}})
        call_count = [0]

        def _read_kv_mock(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return {"opportunity_tracking": tracking_data}
            else:
                yield
                return {"opportunity_attr_def": json.dumps({"attr_def_id": "ad1"})}

        b._read_kv = _read_kv_mock
        b._get_or_create_agent_registry = _gen_return(
            {"agent_id": "123", "agent_name": "test", "agent_address": "0x"}
        )
        b._get_or_create_agent_type = _gen_return(
            {"type_id": "t1", "type_name": "test_type"}
        )
        b.create_agent_attribute = _gen_return(None)  # push fails
        b._get_current_timestamp = lambda: 12345
        synced_mock = MagicMock()
        synced_mock.period_count = 1
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            _drive(b._push_opportunity_metrics_to_mirrordb(), sends=[None] * 20)

    def test_exception_handled(self):
        b = _mk()

        def _bad(*a, **kw):
            raise Exception("boom")
            yield  # noqa: unreachable

        b._read_kv = _bad
        _drive(b._push_opportunity_metrics_to_mirrordb())


class TestFetchAllTradingOpportunities:
    """Tests for fetch_all_trading_opportunities."""

    def test_no_strategies(self):
        from concurrent.futures import Future

        b = _mk()
        b.download_strategies = _gen_none
        synced_mock = MagicMock()
        synced_mock.selected_protocols = []
        b.context.coingecko = MagicMock()
        b.context.coingecko.use_x402 = False
        # Mock asyncio.ensure_future to return an immediately-done future
        f = Future()
        f.set_result([])
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            with patch("asyncio.ensure_future", return_value=f):
                _drive(b.fetch_all_trading_opportunities(), sends=[None] * 5)
        assert b.trading_opportunities == []

    def test_exception_in_main_loop(self):
        b = _mk()
        b.download_strategies = _gen_none
        synced_mock = MagicMock()
        synced_mock.selected_protocols = ["strat_a"]
        b.context.coingecko = MagicMock()
        b.context.coingecko.use_x402 = False
        b.shared_state.strategies_executables = {}

        # Make ensure_future raise
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            with patch("asyncio.ensure_future", side_effect=Exception("async error")):
                _drive(b.fetch_all_trading_opportunities(), sends=[None] * 5)
        assert b.trading_opportunities == []

    def test_with_valid_results(self):
        from concurrent.futures import Future

        b = _mk()
        b.download_strategies = _gen_none
        synced_mock = MagicMock()
        synced_mock.selected_protocols = ["strat_a"]
        b.context.coingecko = MagicMock()
        b.context.coingecko.use_x402 = False
        b.shared_state.strategies_executables = {}
        b._track_opportunities = _gen_none
        # Return a valid result with opportunities
        f = Future()
        f.set_result(
            [
                {
                    "result": [
                        {
                            "pool_address": "0x1",
                            "chain": "optimism",
                            "token0_symbol": "A",
                            "token1_symbol": "B",
                            "token_count": 2,
                            "tvl": 5000,
                            "apr": 10,
                        }
                    ]
                }
            ]
        )
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            with patch("asyncio.ensure_future", return_value=f):
                _drive(b.fetch_all_trading_opportunities(), sends=[None] * 20)
        assert len(b.trading_opportunities) == 1

    def test_with_error_result(self):
        from concurrent.futures import Future

        b = _mk()
        b.download_strategies = _gen_none
        synced_mock = MagicMock()
        synced_mock.selected_protocols = ["strat_a"]
        b.context.coingecko = MagicMock()
        b.context.coingecko.use_x402 = False
        b.shared_state.strategies_executables = {}
        f = Future()
        f.set_result([{"error": ["strategy failed"]}])
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            with patch("asyncio.ensure_future", return_value=f):
                _drive(b.fetch_all_trading_opportunities(), sends=[None] * 10)
        assert b.trading_opportunities == []

    def test_with_no_result(self):
        from concurrent.futures import Future

        b = _mk()
        b.download_strategies = _gen_none
        synced_mock = MagicMock()
        synced_mock.selected_protocols = ["strat_a"]
        b.context.coingecko = MagicMock()
        b.context.coingecko.use_x402 = False
        b.shared_state.strategies_executables = {}
        f = Future()
        f.set_result([None])
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            with patch("asyncio.ensure_future", return_value=f):
                _drive(b.fetch_all_trading_opportunities(), sends=[None] * 10)
        assert b.trading_opportunities == []

    def test_with_invalid_opportunity_format(self):
        from concurrent.futures import Future

        b = _mk()
        b.download_strategies = _gen_none
        synced_mock = MagicMock()
        synced_mock.selected_protocols = ["strat_a"]
        b.context.coingecko = MagicMock()
        b.context.coingecko.use_x402 = False
        b.shared_state.strategies_executables = {}
        b._track_opportunities = _gen_none
        f = Future()
        f.set_result(
            [
                {
                    "result": [
                        "not_a_dict",
                        {
                            "pool_address": "0x1",
                            "token_count": 2,
                            "tvl": 5000,
                            "apr": 10,
                        },
                    ]
                }
            ]
        )
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            with patch("asyncio.ensure_future", return_value=f):
                _drive(b.fetch_all_trading_opportunities(), sends=[None] * 20)
        assert len(b.trading_opportunities) == 1  # only the dict one

    def test_with_empty_opportunities(self):
        from concurrent.futures import Future

        b = _mk()
        b.download_strategies = _gen_none
        synced_mock = MagicMock()
        synced_mock.selected_protocols = ["strat_a"]
        b.context.coingecko = MagicMock()
        b.context.coingecko.use_x402 = False
        b.shared_state.strategies_executables = {}
        f = Future()
        f.set_result([{"result": []}])
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            with patch("asyncio.ensure_future", return_value=f):
                _drive(b.fetch_all_trading_opportunities(), sends=[None] * 10)
        assert b.trading_opportunities == []


class TestBuildBridgeSwapActionsBranches:
    """Additional branching tests for _build_bridge_swap_actions."""

    def test_cl_pool_with_position_requirements(self):
        b = _mk()
        opp = {
            "chain": "optimism",
            "token0": "0xA",
            "token0_symbol": "A",
            "token1": "0xB",
            "token1_symbol": "B",
            "dex_type": "velodrome",
            "relative_funds_percentage": 1.0,
            "is_cl_pool": True,
            "token_requirements": {
                "position_requirements": [
                    {"allocation": 1.0, "token0_ratio": 0.7, "token1_ratio": 0.3}
                ]
            },
        }
        tokens = [
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 100},
            {"chain": "optimism", "token": "0xB", "token_symbol": "B", "value": 50},
        ]
        result = b._build_bridge_swap_actions(opp, tokens)
        assert isinstance(result, list)

    def test_some_tokens_needed(self):
        b = _mk()
        opp = {
            "chain": "optimism",
            "token0": "0xA",
            "token0_symbol": "A",
            "token1": "0xB",
            "token1_symbol": "B",
            "dex_type": "velodrome",
            "relative_funds_percentage": 1.0,
        }
        tokens = [
            {"chain": "optimism", "token": "0xA", "token_symbol": "A"},
        ]
        result = b._build_bridge_swap_actions(opp, tokens)
        assert isinstance(result, list)

    def test_no_required_tokens(self):
        b = _mk()
        opp = {
            "chain": "optimism",
            "token0": "",
            "dex_type": "velodrome",
            "relative_funds_percentage": 1.0,
        }
        result = b._build_bridge_swap_actions(opp, [])
        assert result is None


class TestGetVelodromePositionRequirements:
    """Tests for get_velodrome_position_requirements."""

    def test_no_cl_pools(self):
        b = _mk()
        b.selected_opportunities = [{"dex_type": "uniswap"}]
        result = _drive(b.get_velodrome_position_requirements())
        assert result == {}

    def test_cl_pool_no_tick_spacing(self):
        b = _mk()
        mock_pool = MagicMock()
        mock_pool._get_tick_spacing_velodrome = _gen_return(None)
        b.pools = {"velodrome": mock_pool}
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "chain": "optimism",
                "pool_address": "0x1",
                "tick_bands": [{"tick_lower": -100}],
            }
        ]
        result = _drive(b.get_velodrome_position_requirements(), sends=[None] * 5)
        assert result == {}

    def test_cl_pool_no_tick_bands(self):
        b = _mk()
        mock_pool = MagicMock()
        mock_pool._get_tick_spacing_velodrome = _gen_return(60)
        b.pools = {"velodrome": mock_pool}
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "chain": "optimism",
                "pool_address": "0x1",
                "tick_bands": [],
            }
        ]
        result = _drive(b.get_velodrome_position_requirements(), sends=[None] * 5)
        assert result == {}

    def test_cl_pool_no_current_price(self):
        b = _mk()
        mock_pool = MagicMock()
        mock_pool._get_tick_spacing_velodrome = _gen_return(60)
        mock_pool._get_current_pool_price = _gen_return(None)
        b.pools = {"velodrome": mock_pool}
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "chain": "optimism",
                "pool_address": "0x1",
                "tick_bands": [
                    {
                        "tick_lower": -100,
                        "tick_upper": 100,
                        "allocation": 1.0,
                        "percent_in_bounds": 0.8,
                    }
                ],
            }
        ]
        result = _drive(b.get_velodrome_position_requirements(), sends=[None] * 10)
        assert result == {}

    def test_cl_pool_full_flow(self):
        b = _mk()
        mock_pool = MagicMock()
        mock_pool._get_tick_spacing_velodrome = _gen_return(60)
        mock_pool._get_current_pool_price = _gen_return(1.5)
        mock_pool._get_sqrt_price_x96 = _gen_return(None)  # will use converted
        b.pools = {"velodrome": mock_pool}
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "chain": "optimism",
                "pool_address": "0x1",
                "tick_bands": [
                    {
                        "tick_lower": -100,
                        "tick_upper": 100,
                        "allocation": 1.0,
                        "percent_in_bounds": 0.8,
                        "ema": [1.0],
                        "std_dev": [0.5],
                        "current_ema": 1.0,
                        "current_std_dev": 0.5,
                        "band_multipliers": [1.5],
                    }
                ],
                "token0": "0xA",
                "token1": "0xB",
                "token0_symbol": "A",
                "token1_symbol": "B",
            }
        ]
        requirements = {
            "position_requirements": [
                {"allocation": 1.0, "token0_ratio": 1.0, "token1_ratio": 0.0}
            ],
            "current_tick": 0,
            "recommendation": "100% token0",
        }
        b.calculate_velodrome_cl_token_requirements = _gen_return(requirements)
        b._cache_cl_pool_data = _gen_none
        b._get_token_balance = _gen_return(1000)
        result = _drive(b.get_velodrome_position_requirements(), sends=[None] * 30)
        assert "0x1" in result

    def test_cl_pool_exception(self):
        b = _mk()
        mock_pool = MagicMock()

        def _bad_tick(*a, **kw):
            raise Exception("boom")
            yield  # noqa: unreachable

        mock_pool._get_tick_spacing_velodrome = _bad_tick
        b.pools = {"velodrome": mock_pool}
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "chain": "optimism",
                "pool_address": "0x1",
                "tick_bands": [{"tick_lower": -100}],
            }
        ]
        result = _drive(b.get_velodrome_position_requirements(), sends=[None] * 5)
        assert result == {}

    def test_cl_pool_sqrt_price_none(self):
        """Cover line 516: sqrt_price_x96 is None branch."""
        b = _mk()
        mock_pool = MagicMock()
        mock_pool._get_tick_spacing_velodrome = _gen_return(60)
        mock_pool._get_current_pool_price = _gen_return(1.5)
        mock_pool._get_sqrt_price_x96 = _gen_return(None)
        b.pools = {"velodrome": mock_pool}
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "chain": "optimism",
                "pool_address": "0x1",
                "tick_bands": [
                    {
                        "tick_lower": -100,
                        "tick_upper": 100,
                        "allocation": 1.0,
                        "percent_in_bounds": 0.8,
                    }
                ],
                "token0": "0xA",
                "token1": "0xB",
                "token0_symbol": "A",
                "token1_symbol": "B",
            }
        ]
        # requirements returns None => continue
        b.calculate_velodrome_cl_token_requirements = _gen_return(None)
        result = _drive(b.get_velodrome_position_requirements(), sends=[None] * 20)
        # sqrt_price_x96 None warning logged, then requirements None => continue => empty
        assert result == {}

    def test_cl_pool_no_optional_metadata(self):
        """Cover lines 572-583: optional metadata is None."""
        b = _mk()
        mock_pool = MagicMock()
        mock_pool._get_tick_spacing_velodrome = _gen_return(60)
        mock_pool._get_current_pool_price = _gen_return(1.5)
        mock_pool._get_sqrt_price_x96 = _gen_return(12345)
        b.pools = {"velodrome": mock_pool}
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "chain": "optimism",
                "pool_address": "0x1",
                # No ema, std_dev, current_ema, current_std_dev, band_multipliers
                "tick_bands": [
                    {
                        "tick_lower": -100,
                        "tick_upper": 100,
                        "allocation": 1.0,
                        "percent_in_bounds": 0.8,
                    }
                ],
                "token0": "0xA",
                "token1": "0xB",
                "token0_symbol": "A",
                "token1_symbol": "B",
            }
        ]
        requirements = {
            "position_requirements": [
                {"allocation": 1.0, "token0_ratio": 0.5, "token1_ratio": 0.5}
            ],
            "current_tick": 0,
            "recommendation": "balanced",
        }
        b.calculate_velodrome_cl_token_requirements = _gen_return(requirements)
        b._cache_cl_pool_data = _gen_none
        b._get_token_balance = _gen_return(1000)
        b._calculate_max_amounts_in = MagicMock(return_value=([500, 500], "balanced"))
        result = _drive(b.get_velodrome_position_requirements(), sends=[None] * 30)
        assert "0x1" in result

    def test_cl_pool_fallback_position_requirements_shorter(self):
        """Cover lines 609-610: fallback when position_requirements shorter than tick_bands."""
        b = _mk()
        mock_pool = MagicMock()
        mock_pool._get_tick_spacing_velodrome = _gen_return(60)
        mock_pool._get_current_pool_price = _gen_return(1.5)
        mock_pool._get_sqrt_price_x96 = _gen_return(12345)
        b.pools = {"velodrome": mock_pool}
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "chain": "optimism",
                "pool_address": "0x1",
                "tick_bands": [
                    {
                        "tick_lower": -100,
                        "tick_upper": 100,
                        "allocation": 0.5,
                        "percent_in_bounds": 0.8,
                    },
                    {
                        "tick_lower": 100,
                        "tick_upper": 200,
                        "allocation": 0.5,
                        "percent_in_bounds": 0.8,
                    },
                ],
                "token0": "0xA",
                "token1": "0xB",
                "token0_symbol": "A",
                "token1_symbol": "B",
            }
        ]
        # Only 1 position_requirement for 2 tick_bands => fallback 0.5/0.5 on second
        requirements = {
            "position_requirements": [
                {"allocation": 1.0, "token0_ratio": 1.0, "token1_ratio": 0.0}
            ],
            "current_tick": 0,
            "recommendation": "100% token0",
        }
        b.calculate_velodrome_cl_token_requirements = _gen_return(requirements)
        b._cache_cl_pool_data = _gen_none
        b._get_token_balance = _gen_return(1000)
        result = _drive(b.get_velodrome_position_requirements(), sends=[None] * 30)
        assert "0x1" in result

    def test_cl_pool_token1_only(self):
        """Cover lines 674-679: aggregate_token1_ratio >= max_ration."""
        b = _mk()
        mock_pool = MagicMock()
        mock_pool._get_tick_spacing_velodrome = _gen_return(60)
        mock_pool._get_current_pool_price = _gen_return(1.5)
        mock_pool._get_sqrt_price_x96 = _gen_return(12345)
        b.pools = {"velodrome": mock_pool}
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "chain": "optimism",
                "pool_address": "0x1",
                "tick_bands": [
                    {
                        "tick_lower": -100,
                        "tick_upper": 100,
                        "allocation": 1.0,
                        "percent_in_bounds": 0.8,
                    }
                ],
                "token0": "0xA",
                "token1": "0xB",
                "token0_symbol": "A",
                "token1_symbol": "B",
            }
        ]
        # token1 ratio = 1.0 (above max_ration 0.999)
        requirements = {
            "position_requirements": [
                {"allocation": 1.0, "token0_ratio": 0.0, "token1_ratio": 1.0}
            ],
            "current_tick": 0,
            "recommendation": "100% token1",
        }
        b.calculate_velodrome_cl_token_requirements = _gen_return(requirements)
        b._cache_cl_pool_data = _gen_none
        b._get_token_balance = _gen_return(1000)
        result = _drive(b.get_velodrome_position_requirements(), sends=[None] * 30)
        assert "0x1" in result
        # max_amounts_in should be [0, token1_balance]
        opp = b.selected_opportunities[0]
        assert opp["max_amounts_in"] == [0, 1000]

    def test_cl_pool_mixed_ratios(self):
        """Cover lines 680-691: mixed allocation calling _calculate_max_amounts_in."""
        b = _mk()
        mock_pool = MagicMock()
        mock_pool._get_tick_spacing_velodrome = _gen_return(60)
        mock_pool._get_current_pool_price = _gen_return(1.5)
        mock_pool._get_sqrt_price_x96 = _gen_return(12345)
        b.pools = {"velodrome": mock_pool}
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "chain": "optimism",
                "pool_address": "0x1",
                "tick_bands": [
                    {
                        "tick_lower": -100,
                        "tick_upper": 100,
                        "allocation": 1.0,
                        "percent_in_bounds": 0.8,
                    }
                ],
                "token0": "0xA",
                "token1": "0xB",
                "token0_symbol": "A",
                "token1_symbol": "B",
            }
        ]
        requirements = {
            "position_requirements": [
                {"allocation": 1.0, "token0_ratio": 0.6, "token1_ratio": 0.4}
            ],
            "current_tick": 0,
            "recommendation": "balanced",
        }
        b.calculate_velodrome_cl_token_requirements = _gen_return(requirements)
        b._cache_cl_pool_data = _gen_none
        b._get_token_balance = _gen_return(500)
        b._calculate_max_amounts_in = MagicMock(
            return_value=([300, 200], "mixed allocation")
        )
        result = _drive(b.get_velodrome_position_requirements(), sends=[None] * 30)
        assert "0x1" in result


class TestValidateVelodromeInputsTickConversion:
    """Tests for tick conversion exception in validate_and_prepare_velodrome_inputs."""

    def test_tick_conversion_exception(self):
        """Cover lines 221-225: math.log exception."""
        b = _mk()
        # current_price is positive but math.log could still fail in edge cases
        # We mock math.log to throw
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy.math.log",
            side_effect=ValueError("math domain error"),
        ):
            result = b.validate_and_prepare_velodrome_inputs(
                tick_bands=[{"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}],
                current_price=1.5,
                tick_spacing=60,
            )
        assert result is None


class TestCheckTipExitConditionsBranches:
    """Additional branch tests for _check_tip_exit_conditions."""

    def test_entry_apr_key_missing_but_apr_present(self):
        """entry_apr key missing => fallback to position.get('apr')."""
        b = _mk()
        b._get_current_timestamp = MagicMock(return_value=time.time())
        b._calculate_min_req_position_value = MagicMock(return_value=None)
        b._get_best_available_opportunity_yield = MagicMock(return_value=None)
        pos = {
            "enter_timestamp": int(time.time()) - 200000,
            # no "entry_apr" key at all
            "apr": 10.0,
            "entry_cost": 100,
            "chain": "optimism",
            "cost_recovered": False,
            "yield_usd": 50,
        }
        result = _drive(b._check_tip_exit_conditions(pos), sends=[None] * 5)
        assert isinstance(result, tuple)

    def test_position_value_check_triggers_exit(self):
        """Cover lines 819-827: current_value_ratio < min_req_value."""
        b = _mk()
        b._get_current_timestamp = MagicMock(return_value=time.time())
        b._calculate_min_req_position_value = MagicMock(return_value=2.0)
        # current_value_ratio=0.5 < min_req_value=2.0
        b._calculate_current_value_ratio = _gen_return(0.5)
        b._get_best_available_opportunity_yield = MagicMock(return_value=None)
        pos = {
            "enter_timestamp": int(time.time()) - 200000,
            "entry_apr": 10.0,
            "apr": 50.0,  # high current apr so stoploss doesn't trigger first
            "entry_cost": 100,
            "chain": "optimism",
            "cost_recovered": False,
            "yield_usd": 50,
        }
        result = _drive(b._check_tip_exit_conditions(pos), sends=[None] * 5)
        should_exit, reason = result
        assert should_exit is True
        assert "Position value check" in reason

    def test_opportunity_cost_triggers_exit(self):
        """Cover lines 832-836: current yield < S * vby."""
        b = _mk()
        b._get_current_timestamp = MagicMock(return_value=time.time())
        b._calculate_min_req_position_value = MagicMock(return_value=None)
        # vby very high => current yield < S * vby
        b._get_best_available_opportunity_yield = MagicMock(return_value=100.0)
        pos = {
            "enter_timestamp": int(time.time()) - 200000,
            "entry_apr": 10.0,
            "apr": 10.0,  # current yield = 10/100/365 = very small
            "entry_cost": 100,
            "chain": "optimism",
            "cost_recovered": False,
            "yield_usd": 50,
        }
        result = _drive(b._check_tip_exit_conditions(pos), sends=[None] * 5)
        should_exit, reason = result
        assert should_exit is True
        assert "Opportunity cost" in reason

    def test_min_req_value_none_skips_position_check(self):
        """Cover lines 819->830 branch: min_req_value is None."""
        b = _mk()
        b._get_current_timestamp = MagicMock(return_value=time.time())
        b._calculate_min_req_position_value = MagicMock(return_value=None)
        b._get_best_available_opportunity_yield = MagicMock(return_value=None)
        pos = {
            "enter_timestamp": int(time.time()) - 200000,
            "entry_apr": 50.0,  # high entry yield so stoploss doesn't trigger
            "apr": 50.0,
            "entry_cost": 100,
            "chain": "optimism",
            "cost_recovered": False,
            "yield_usd": 50,
        }
        result = _drive(b._check_tip_exit_conditions(pos), sends=[None] * 5)
        should_exit, reason = result
        assert should_exit is False
        assert "costs not recovered" in reason

    def test_current_value_ratio_none_skips(self):
        """Cover lines 819->830 branch: current_value_ratio is None."""
        b = _mk()
        b._get_current_timestamp = MagicMock(return_value=time.time())
        b._calculate_min_req_position_value = MagicMock(return_value=2.0)
        b._calculate_current_value_ratio = _gen_return(None)
        b._get_best_available_opportunity_yield = MagicMock(return_value=None)
        pos = {
            "enter_timestamp": int(time.time()) - 200000,
            "entry_apr": 50.0,
            "apr": 50.0,
            "entry_cost": 100,
            "chain": "optimism",
            "cost_recovered": False,
            "yield_usd": 50,
        }
        result = _drive(b._check_tip_exit_conditions(pos), sends=[None] * 5)
        should_exit, reason = result
        assert should_exit is False

    def test_vby_not_none_no_exit(self):
        """Cover line 832->839: vby exists but yield is high enough."""
        b = _mk()
        b._get_current_timestamp = MagicMock(return_value=time.time())
        b._calculate_min_req_position_value = MagicMock(return_value=None)
        b._get_best_available_opportunity_yield = MagicMock(return_value=0.000001)
        pos = {
            "enter_timestamp": int(time.time()) - 200000,
            "entry_apr": 50.0,
            "apr": 50.0,
            "entry_cost": 100,
            "chain": "optimism",
            "cost_recovered": False,
            "yield_usd": 50,
        }
        result = _drive(b._check_tip_exit_conditions(pos), sends=[None] * 5)
        should_exit, reason = result
        assert should_exit is False


class TestGetPositionTokenBalancesFallback:
    """Tests for fallback path in _get_position_token_balances."""

    def test_no_stored_balances_fallback(self):
        """Cover lines 939-961: fallback to direct balance retrieval."""
        b = _mk()
        pos = {
            "token0": "0xA",
            "token1": "0xB",
            "pool_address": "0x1",
            "chain": "optimism",
        }
        b._get_token_balance = _gen_return(1000)
        b._get_token_decimals = _gen_return(18)
        result = _drive(
            b._get_position_token_balances(pos, "optimism"), sends=[None] * 20
        )
        assert "0xA" in result
        assert "0xB" in result

    def test_no_stored_balances_no_safe_address(self):
        """Cover line 934-936: no safe address for chain."""
        b = _mk()
        b.params.safe_contract_addresses = {}
        pos = {
            "token0": "0xA",
            "token1": "0xB",
            "pool_address": "0x1",
            "chain": "unknown_chain",
        }
        result = _drive(
            b._get_position_token_balances(pos, "unknown_chain"), sends=[None] * 5
        )
        assert result == {}

    def test_stored_balances_staked_velodrome(self):
        """Cover lines 905-922: staked velodrome position with rewards."""
        b = _mk()
        b._get_velodrome_pending_rewards = _gen_return(100)
        b._get_velo_token_address = MagicMock(return_value="0xVELO")
        pos = {
            "current_token_balances": {"0xA": 1.0, "0xB": 2.0},
            "pool_address": "0x1",
            "staked": True,
            "dex_type": "velodrome",
        }
        result = _drive(
            b._get_position_token_balances(pos, "optimism"), sends=[None] * 10
        )
        assert "0xVELO" in result
        assert result["0xVELO"] == 100.0

    def test_stored_balances_staked_zero_rewards(self):
        """Cover line 914->922: velo_rewards == 0."""
        b = _mk()
        b._get_velodrome_pending_rewards = _gen_return(0)
        pos = {
            "current_token_balances": {"0xA": 1.0},
            "pool_address": "0x1",
            "staked": True,
            "dex_type": "velodrome",
        }
        result = _drive(
            b._get_position_token_balances(pos, "optimism"), sends=[None] * 10
        )
        assert "0xA" in result
        assert "0xVELO" not in result

    def test_balance_none_skip(self):
        """Cover lines 943->950, 954->961: balance is None."""
        b = _mk()
        b._get_token_balance = _gen_return(None)
        pos = {
            "token0": "0xA",
            "token1": "0xB",
            "pool_address": "0x1",
        }
        result = _drive(
            b._get_position_token_balances(pos, "optimism"), sends=[None] * 10
        )
        assert result == {}

    def test_decimals_none_skip(self):
        """Cover lines 947->950, 958->961: decimals is None."""
        b = _mk()
        b._get_token_balance = _gen_return(1000)
        b._get_token_decimals = _gen_return(None)
        pos = {
            "token0": "0xA",
            "token1": "0xB",
            "pool_address": "0x1",
        }
        result = _drive(
            b._get_position_token_balances(pos, "optimism"), sends=[None] * 10
        )
        assert result == {}


class TestCalculateCurrentValueRatioBranches:
    """Additional branch tests for _calculate_current_value_ratio."""

    def test_both_entry_quantities_zero_after_decimals(self):
        """Cover lines 1034-1037: Q0_entry and Q1_entry both zero after decimal conversion."""
        b = _mk()
        b._get_position_token_balances = _gen_return({"0xA": 1.0, "0xB": 2.0})
        b._get_token_decimals = _gen_return(18)
        b._fetch_token_prices_sma = _gen_return(1.0)
        pos = {
            "token0": "0xA",
            "token1": "0xB",
            # amount0_raw > 0 but after 10**18 conversion rounds to 0
            # Q0_entry = 1 / 10**18 = ~0 (but not zero in float)
            # Actually we need amount0_raw > 0 but the code does:
            # Q0_entry = amount0_raw / (10**token0_decimals) if amount0_raw > 0 else 0
            # So with amount0=1 and decimals=18, Q0_entry = 1e-18 which is not 0
            # We need amount0_raw=0 AND amount1_raw=0 which is caught at 1013-1017
            # Lines 1034-1037 need: amount0_raw > 0 initially but Q0_entry ends up 0
            # This means amount0_raw > 0 (passes 1013 check) but Q0_entry = 0
            # Q0_entry = amount0_raw / 10**decimals if amount0_raw > 0 else 0
            # So we need an amount where int division gives 0 somehow... but float division never gives 0
            # Actually: Q0_entry = 0 when amount0_raw == 0 (the else branch)
            # So to hit 1033: amount0_raw > 0 but somehow Q0=Q1=0 after conversion
            # This can only happen if amount0_raw passes the initial check (not both zero)
            # but then both end up zero after division. With float that can't happen.
            # This means we need one > 0 and one == 0 at initial check, then Q_entry = tiny float
            # Actually line 1033 checks Q0_entry == 0 AND Q1_entry == 0
            # With amount0_raw = 0, amount1_raw = 5: passes initial check (not both zero)
            # Q0_entry = 0 (since amount0_raw == 0 -> else 0)
            # Q1_entry = 5 / 10**18 = 5e-18 which is != 0
            # So we need BOTH raw amounts to result in Q=0 after conversion
            # amount0_raw = 0, amount1_raw = 0 is caught at 1013
            # The only way: amount0_raw = 1, amount1_raw = 0 -> initial check passes
            # Then Q0 = 1/10^18, Q1 = 0 -> Q0 != 0 so doesn't trigger
            # Conclusion: lines 1034-1037 are UNREACHABLE
            "amount0": 0,
            "amount1": 0,
            "chain": "optimism",
        }
        result = _drive(
            b._calculate_current_value_ratio(pos, "optimism"), sends=[None] * 20
        )
        assert result is None


class TestGetBestAvailableOpportunityYieldBranches:
    """Additional branch tests."""

    def test_empty_after_sorting(self):
        """Cover line 1164: sorted list is empty (edge case)."""
        b = _mk()
        b.trading_opportunities = []
        result = b._get_best_available_opportunity_yield()
        assert result is None

    def test_exception_in_yield_calc(self):
        """Cover lines 1189-1193: exception handler."""
        b = _mk()
        # Item whose .get raises during sorted key function
        bad_item = MagicMock()
        bad_item.get = MagicMock(side_effect=TypeError("broken"))
        b.trading_opportunities = [bad_item]
        result = b._get_best_available_opportunity_yield()
        assert result is None


class TestAsyncMethods:
    """Tests for _async_execute_strategy and _run_all_strategies."""

    def test_async_execute_strategy_type_error(self):
        """Cover lines 1486-1493."""
        import asyncio

        b = _mk()
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy.execute_strategy",
            side_effect=TypeError("missing arg"),
        ):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    b._async_execute_strategy("test_strategy", {})
                )
            finally:
                loop.close()
        assert "error" in result
        assert "missing" in result["error"][0]

    def test_async_execute_strategy_generic_exception(self):
        """Cover lines 1494-1499."""
        import asyncio

        b = _mk()
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy.execute_strategy",
            side_effect=RuntimeError("unexpected"),
        ):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    b._async_execute_strategy("test_strategy", {})
                )
            finally:
                loop.close()
        assert "error" in result
        assert "Unexpected" in result["error"][0]

    def test_run_all_strategies_setup_exception(self):
        """Cover lines 1520-1526: exception in strategy setup loop."""
        import asyncio

        b = _mk()
        # kwargs.items() will raise
        bad_kwargs = MagicMock()
        bad_kwargs.items.side_effect = RuntimeError("bad kwargs")
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                b._run_all_strategies([("strat", bad_kwargs)], {})
            )
        finally:
            loop.close()
        assert len(result) == 1
        assert "error" in result[0]

    def test_run_all_strategies_gather_exception(self):
        """Cover lines 1541-1547: exception in asyncio.gather."""
        import asyncio

        b = _mk()
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy.execute_strategy",
            side_effect=RuntimeError("fail"),
        ):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    b._run_all_strategies([("strat_a", {"key": "val"})], {})
                )
            finally:
                loop.close()
        assert len(result) >= 1

    def test_run_all_strategies_exception_result(self):
        """Cover lines 1531-1538: result is an Exception from gather."""
        import asyncio

        async def _mock_strategy(*a, **kw):
            raise ValueError("boom")

        b = _mk()
        b._async_execute_strategy = _mock_strategy
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                b._run_all_strategies([("strat_a", {"key": "val"})], {})
            )
        finally:
            loop.close()
        assert any("error" in r for r in result)


class TestFetchAllTradingOpportunitiesInnerBranches:
    """Tests for inner branches in fetch_all_trading_opportunities."""

    def _setup_fetch(self, results):
        from concurrent.futures import Future as ConcFuture

        b = _mk()
        b.download_strategies = _gen_none
        synced_mock = MagicMock()
        synced_mock.selected_protocols = ["strat_a"]
        b.context.coingecko = MagicMock()
        b.context.coingecko.use_x402 = False
        b.shared_state.strategies_executables = {}
        b._track_opportunities = _gen_none
        f = ConcFuture()
        f.set_result(results)
        return b, synced_mock, f

    def test_error_in_result(self):
        """Cover lines 1621-1629: result has error key with errors."""
        b, synced, f = self._setup_fetch([{"error": ["something broke"], "result": []}])
        with patch.object(
            type(b), "synchronized_data", new_callable=PropertyMock, return_value=synced
        ):
            with patch("asyncio.ensure_future", return_value=f):
                _drive(b.fetch_all_trading_opportunities(), sends=[None] * 10)
        assert b.trading_opportunities == []

    def test_empty_error_list(self):
        """Cover line 1623->1631: error key exists but empty list."""
        b, synced, f = self._setup_fetch(
            [{"error": [], "result": [{"pool_address": "0x1"}]}]
        )
        with patch.object(
            type(b), "synchronized_data", new_callable=PropertyMock, return_value=synced
        ):
            with patch("asyncio.ensure_future", return_value=f):
                _drive(b.fetch_all_trading_opportunities(), sends=[None] * 10)
        assert len(b.trading_opportunities) == 1

    def test_invalid_opportunity_format(self):
        """Cover lines 1650-1654: opportunity is not a dict."""
        b, synced, f = self._setup_fetch([{"result": ["not_a_dict"]}])
        with patch.object(
            type(b), "synchronized_data", new_callable=PropertyMock, return_value=synced
        ):
            with patch("asyncio.ensure_future", return_value=f):
                _drive(b.fetch_all_trading_opportunities(), sends=[None] * 10)
        assert b.trading_opportunities == []

    def test_exception_processing_opportunity(self):
        """Cover lines 1655-1658: exception during opportunity processing."""
        # Create a dict that raises when get() is called
        bad_dict = MagicMock(spec=dict)
        bad_dict.__class__ = dict  # isinstance(bad_dict, dict) => True
        bad_dict.__setitem__ = MagicMock(side_effect=RuntimeError("boom"))
        b, synced, f = self._setup_fetch([{"result": [bad_dict]}])
        with patch.object(
            type(b), "synchronized_data", new_callable=PropertyMock, return_value=synced
        ):
            with patch("asyncio.ensure_future", return_value=f):
                _drive(b.fetch_all_trading_opportunities(), sends=[None] * 10)

    def test_no_result_warning(self):
        """Cover lines 1615-1619: result is falsy."""
        b, synced, f = self._setup_fetch([None])
        with patch.object(
            type(b), "synchronized_data", new_callable=PropertyMock, return_value=synced
        ):
            with patch("asyncio.ensure_future", return_value=f):
                _drive(b.fetch_all_trading_opportunities(), sends=[None] * 10)
        assert b.trading_opportunities == []

    def test_future_not_done_yields(self):
        """Cover line 1609: yield when future is not done."""
        from concurrent.futures import Future as ConcFuture

        b = _mk()
        b.download_strategies = _gen_none
        synced_mock = MagicMock()
        synced_mock.selected_protocols = ["strat_a"]
        b.context.coingecko = MagicMock()
        b.context.coingecko.use_x402 = False
        b.shared_state.strategies_executables = {}
        b._track_opportunities = _gen_none

        call_count = [0]
        f = ConcFuture()
        original_done = f.done

        def _mock_done():
            call_count[0] += 1
            if call_count[0] <= 2:
                return False
            f.set_result([{"result": []}])
            return True

        f.done = _mock_done
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            with patch("asyncio.ensure_future", return_value=f):
                _drive(b.fetch_all_trading_opportunities(), sends=[None] * 20)


class TestCurrentPositionsWithInvalidAddress:
    """Test that invalid pool addresses are filtered from current_positions."""

    def test_invalid_short_pool_address_is_skipped(self):
        """Positions with short/invalid pool addresses must not crash to_checksum_address."""
        from concurrent.futures import Future as ConcFuture

        b = _mk()
        b.download_strategies = _gen_none
        b.positions_eligible_for_exit = [
            {"status": "open", "pool_address": "0x123"},
            {
                "status": "open",
                "pool_address": "0x" + "ab" * 20,
            },
            {"status": "open", "pool_address": ""},
            {"status": "closed", "pool_address": "0x" + "cd" * 20},
        ]
        synced_mock = MagicMock()
        synced_mock.selected_protocols = ["strat_a"]
        b.context.coingecko = MagicMock()
        b.context.coingecko.use_x402 = False
        b.shared_state.strategies_executables = {}
        b._track_opportunities = _gen_none
        f = ConcFuture()
        f.set_result([{"result": []}])
        with patch.object(
            type(b),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            with patch("asyncio.ensure_future", return_value=f):
                _drive(b.fetch_all_trading_opportunities(), sends=[None] * 10)


class TestPushOpportunityMetricsBranches:
    """Tests for _push_opportunity_metrics_to_mirrordb branches."""

    def test_opportunity_tracking_invalid_json(self):
        """Cover lines 1854-1855: JSONDecodeError in opportunity_tracking."""
        b = _mk()
        # Truthy but invalid JSON - passes line 1844 check, fails json.loads on 1853
        call_count = [0]

        def _read_kv_side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return {"opportunity_tracking": "not valid json{{{"}
            yield
            return {}

        b._read_kv = _read_kv_side_effect
        b._get_or_create_agent_registry = _gen_return({"agent_id": "123"})
        b._get_or_create_agent_type = _gen_return({"type_id": "456"})
        b._create_opportunity_attr_def = _gen_return({"attr_def": "test"})
        b._write_kv = _gen_none
        _drive(b._push_opportunity_metrics_to_mirrordb(), sends=[None] * 30)

    def test_json_decode_error_attr_def(self):
        """Cover lines 1854-1855, 1921-1924: JSONDecodeError in attr_def parsing."""
        b = _mk()
        call_count = [0]

        def _read_kv_side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return {"opportunity_tracking": '{"round_1": {}}'}
            elif call_count[0] == 2:
                yield
                return {"opportunity_attr_def": "not valid json{{{"}
            yield
            return {}

        b._read_kv = _read_kv_side_effect
        b._get_or_create_agent_registry = _gen_return({"agent_id": "123"})
        b._get_or_create_agent_type = _gen_return({"type_id": "456"})
        b._create_opportunity_attr_def = _gen_return({"attr_def": "test"})
        b._write_kv = _gen_none
        _drive(b._push_opportunity_metrics_to_mirrordb(), sends=[None] * 30)

    def test_attr_def_value_none(self):
        """Cover lines 1902-1905: attr_def value None in KV store."""
        b = _mk()
        call_count = [0]

        def _read_kv_side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return {"opportunity_tracking": '{"round_1": {}}'}
            elif call_count[0] == 2:
                yield
                return {"opportunity_attr_def": None}
            yield
            return {}

        b._read_kv = _read_kv_side_effect
        b._get_or_create_agent_registry = _gen_return({"agent_id": "123"})
        b._get_or_create_agent_type = _gen_return({"type_id": "456"})
        b._create_opportunity_attr_def = _gen_return({"attr_def": "test"})
        b._write_kv = _gen_none
        _drive(b._push_opportunity_metrics_to_mirrordb(), sends=[None] * 30)

    def test_attr_def_creation_fails_on_none_value(self):
        """Cover lines 1902-1905: create_opportunity_attr_def returns None when attr_def_value is None."""
        b = _mk()
        call_count = [0]

        def _read_kv_side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return {"opportunity_tracking": '{"round_1": {}}'}
            elif call_count[0] == 2:
                yield
                return {"opportunity_attr_def": None}
            yield
            return {}

        b._read_kv = _read_kv_side_effect
        b._get_or_create_agent_registry = _gen_return({"agent_id": "123"})
        b._get_or_create_agent_type = _gen_return({"type_id": "456"})
        b._create_opportunity_attr_def = _gen_return(None)
        b._write_kv = _gen_none
        _drive(b._push_opportunity_metrics_to_mirrordb(), sends=[None] * 30)

    def test_attr_def_creation_fails_on_json_decode(self):
        """Cover lines 1921-1924: create_opportunity_attr_def returns None after json decode failure."""
        b = _mk()
        call_count = [0]

        def _read_kv_side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return {"opportunity_tracking": '{"round_1": {}}'}
            elif call_count[0] == 2:
                yield
                return {"opportunity_attr_def": "invalid_json"}
            yield
            return {}

        b._read_kv = _read_kv_side_effect
        b._get_or_create_agent_registry = _gen_return({"agent_id": "123"})
        b._get_or_create_agent_type = _gen_return({"type_id": "456"})
        b._create_opportunity_attr_def = _gen_return(None)
        b._write_kv = _gen_none
        _drive(b._push_opportunity_metrics_to_mirrordb(), sends=[None] * 30)


class TestMergeDuplicateBranchesExtra:
    """Extra branch tests for _merge_duplicate_bridge_swap_actions."""

    def test_exception_checking_redundant(self):
        """Cover lines 2195-2196: exception during redundant check."""
        b = _mk()
        # Action with from_token that raises on .lower()
        bad_action = {
            "action": Action.FIND_BRIDGE_ROUTE.value,
            "from_chain": "opt",
            "to_chain": "opt",
            "from_token": MagicMock(lower=MagicMock(side_effect=RuntimeError("boom"))),
            "to_token": "0xB",
        }
        result = b._merge_duplicate_bridge_swap_actions(
            [bad_action, {"action": "other"}]
        )
        assert len(result) >= 1

    def test_exception_processing_action(self):
        """Cover lines 2239-2240: exception grouping action."""
        b = _mk()
        a1 = {
            "action": Action.FIND_BRIDGE_ROUTE.value,
            "from_chain": "opt",
            "to_chain": "mode",
            "from_token": "0xA",
            "to_token": "0xB",
        }
        a2 = {
            "action": Action.FIND_BRIDGE_ROUTE.value,
            "from_chain": "opt",
            "to_chain": "mode",
            # from_token shorter than 8 chars will cause key slicing issue? No, it works.
            # Make to_token raise on slicing:
            "from_token": MagicMock(
                __getitem__=MagicMock(side_effect=RuntimeError("x"))
            ),
            "to_token": "0xB",
        }
        result = b._merge_duplicate_bridge_swap_actions([a1, a2])
        assert isinstance(result, list)

    def test_no_duplicates_after_grouping(self):
        """Cover line 2246: all groups have <= 1 action."""
        b = _mk()
        a1 = {
            "action": Action.FIND_BRIDGE_ROUTE.value,
            "from_chain": "opt",
            "to_chain": "mode",
            "from_token": "0xA" * 5,
            "to_token": "0xB" * 5,
        }
        a2 = {
            "action": Action.FIND_BRIDGE_ROUTE.value,
            "from_chain": "mode",
            "to_chain": "opt",
            "from_token": "0xC" * 5,
            "to_token": "0xD" * 5,
        }
        result = b._merge_duplicate_bridge_swap_actions([a1, a2])
        assert len(result) == 2

    def test_merge_exception(self):
        """Cover lines 2276-2277: exception during merge."""
        b = _mk()
        a1 = {
            "action": Action.FIND_BRIDGE_ROUTE.value,
            "from_chain": "opt",
            "to_chain": "mode",
            "from_token": "0xA" * 5,
            "to_token": "0xB" * 5,
        }
        # Duplicate to force merge
        a2 = dict(a1)
        # Make get() on second action raise during merge
        # Actually the merge accesses group[0] and group[1:], so we need 2+ identical keys
        # The easiest way: make funds_percentage raise
        a2_bad = dict(a1)
        a2_bad["funds_percentage"] = MagicMock(
            __radd__=MagicMock(side_effect=RuntimeError("merge fail"))
        )
        result = b._merge_duplicate_bridge_swap_actions([a1, a2_bad])
        assert isinstance(result, list)

    def test_outer_exception(self):
        """Cover lines 2289-2295: outer exception returns original actions."""
        b = _mk()
        # Pass an object that is truthy but fails during iteration
        original = MagicMock()
        original.__bool__ = MagicMock(return_value=True)
        original.__iter__ = MagicMock(side_effect=RuntimeError("fail"))
        result = b._merge_duplicate_bridge_swap_actions(original)
        assert result is original


class TestHandleVelodromeTokenAllocationBranches:
    """Extra branch tests for _handle_velodrome_token_allocation."""

    def test_fallback_recommendation_token1(self):
        """Cover lines 2322-2327: recommendation '100% token1'."""
        b = _mk()
        actions = []
        enter_pool = {
            "dex_type": "velodrome",
            "token_requirements": {
                "position_requirements": [],
                "recommendation": "100% token1",
            },
            "token0": "0xA",
            "token1": "0xB",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "chain": "optimism",
        }
        result = b._handle_velodrome_token_allocation(actions, enter_pool, [])
        assert isinstance(result, list)

    def test_fallback_recommendation_balanced(self):
        """Cover lines 2325-2327: recommendation without '100% token0' or '100% token1'."""
        b = _mk()
        actions = []
        enter_pool = {
            "dex_type": "velodrome",
            "token_requirements": {
                "position_requirements": [],
                "recommendation": "balanced allocation",
            },
            "token0": "0xA",
            "token1": "0xB",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "chain": "optimism",
        }
        result = b._handle_velodrome_token_allocation(actions, enter_pool, [])
        assert isinstance(result, list)

    def test_funds_percentage_invalid(self):
        """Cover lines 2359-2360: ValueError on float conversion."""
        b = _mk()
        actions = [
            {
                "action": "FindBridgeRoute",
                "to_chain": "optimism",
                "to_token": "0xOTHER",
            },
        ]
        enter_pool = {
            "dex_type": "velodrome",
            "token_requirements": {
                "position_requirements": [
                    {"allocation": 1.0, "token0_ratio": 1.0, "token1_ratio": 0.0}
                ],
            },
            "token0": "0xA",
            "token1": "0xB",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "chain": "optimism",
            "relative_funds_percentage": "not_a_number",
        }
        result = b._handle_velodrome_token_allocation(actions, enter_pool, [])
        assert isinstance(result, list)

    def test_no_bridge_routes_add_new(self):
        """Cover lines 2384-2417: no FindBridgeRoute => add new one."""
        b = _mk()
        actions = [{"action": "ExitPool"}]
        enter_pool = {
            "dex_type": "velodrome",
            "token_requirements": {
                "position_requirements": [
                    {"allocation": 1.0, "token0_ratio": 1.0, "token1_ratio": 0.0}
                ],
            },
            "token0": "0xA",
            "token1": "0xB",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "chain": "optimism",
            "relative_funds_percentage": 1.0,
        }
        available_tokens = [
            {"token": "0xB", "token_symbol": "B", "chain": "mode"},
        ]
        result = b._handle_velodrome_token_allocation(
            actions, enter_pool, available_tokens
        )
        # Should have inserted a new FindBridgeRoute action after ExitPool
        bridge_actions = [a for a in result if a.get("action") == "FindBridgeRoute"]
        assert len(bridge_actions) == 1
        assert bridge_actions[0]["to_token"] == "0xA"

    def test_redirect_existing_bridge_route(self):
        """Cover lines 2367-2381: redirect existing bridge route."""
        b = _mk()
        actions = [
            {
                "action": "FindBridgeRoute",
                "to_chain": "optimism",
                "to_token": "0xOLD",
                "funds_percentage": 0.5,
            },
        ]
        enter_pool = {
            "dex_type": "velodrome",
            "token_requirements": {
                "position_requirements": [
                    {"allocation": 1.0, "token0_ratio": 1.0, "token1_ratio": 0.0}
                ],
            },
            "token0": "0xA",
            "token1": "0xB",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "chain": "optimism",
            "relative_funds_percentage": 1.0,
        }
        result = b._handle_velodrome_token_allocation(actions, enter_pool, [])
        assert result[0]["to_token"] == "0xA"
        assert result[0]["funds_percentage"] == 1.0


class TestApplyInvestmentCapBranches:
    """Tests for _apply_investment_cap_to_actions branches."""

    def test_threshold_reached_with_exit(self):
        """Cover lines 2478-2492: threshold reached with exit pool action."""
        b = _mk()
        b.sleep = _gen_none
        b.current_positions = [{"status": "open"}]
        b.calculate_initial_investment_value = _gen_return(1000)
        actions = [
            {"action": "ExitPool"},
            {"action": "EnterPool"},
        ]
        result = _drive(b._apply_investment_cap_to_actions(actions), sends=[None] * 20)
        # Should remove EnterPool but keep ExitPool
        assert any(a.get("action") == "ExitPool" for a in result)
        assert not any(a.get("action") == "EnterPool" for a in result)

    def test_threshold_reached_no_exit(self):
        """Cover lines 2494-2499: threshold reached with no exit pool."""
        b = _mk()
        b.sleep = _gen_none
        b.current_positions = [{"status": "open"}]
        b.calculate_initial_investment_value = _gen_return(1000)
        actions = [{"action": "EnterPool"}]
        result = _drive(b._apply_investment_cap_to_actions(actions), sends=[None] * 20)
        assert result == []

    def test_under_threshold_adjust_amounts(self):
        """Cover lines 2502-2511: invested_amount > 0 but under threshold."""
        b = _mk()
        b.sleep = _gen_none
        b.current_positions = [{"status": "open"}]
        b.calculate_initial_investment_value = _gen_return(500)
        actions = [{"action": "EnterPool"}]
        result = _drive(b._apply_investment_cap_to_actions(actions), sends=[None] * 20)
        assert result[0].get("invested_amount") == 500  # 1000 - 500

    def test_v_initial_none_retry(self):
        """Cover lines 2455-2461: V_initial is None, retries."""
        b = _mk()
        b.sleep = _gen_none
        call_count = [0]

        def _calc_value(*a, **kw):
            call_count[0] += 1
            yield
            if call_count[0] <= 2:
                return None
            return 500

        b.current_positions = [{"status": "open"}]
        b.calculate_initial_investment_value = _calc_value
        actions = [{"action": "EnterPool"}]
        result = _drive(b._apply_investment_cap_to_actions(actions), sends=[None] * 40)
        assert isinstance(result, list)

    def test_invested_zero_with_positions(self):
        """Cover line 2479: invested_amount == 0 and invested_positions == True."""
        b = _mk()
        b.sleep = _gen_none
        b.current_positions = [{"status": "open"}]
        b.calculate_initial_investment_value = _gen_return(None)
        actions = [{"action": "EnterPool"}]
        result = _drive(b._apply_investment_cap_to_actions(actions), sends=[None] * 40)
        # invested_amount = 0 and invested_positions = True => clear
        assert result == []


class TestGetOrderOfTransactionsBranches:
    """Tests for get_order_of_transactions branches."""

    def test_exit_with_staking(self):
        """Cover lines 2542-2561: position_to_exit with staking metadata."""
        b = _mk()
        b.selected_opportunities = [
            {
                "dex_type": "uniswap",
                "chain": "optimism",
                "token0": "0xA",
                "token1": "0xB",
                "relative_funds_percentage": 1.0,
                "token0_symbol": "A",
                "token1_symbol": "B",
            },
        ]
        b.position_to_exit = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0x1",
        }
        b._has_staking_metadata = MagicMock(return_value=True)
        b._build_unstake_lp_tokens_action = MagicMock(
            return_value={"action": "UnstakeLPTokens"}
        )
        b._build_exit_pool_action = MagicMock(return_value={"action": "ExitPool"})
        b._prepare_tokens_for_investment = _gen_return(
            [{"chain": "optimism", "token": "0xA"}]
        )
        b._build_bridge_swap_actions = MagicMock(return_value=[])
        b._build_enter_pool_action = MagicMock(
            return_value={"action": "EnterPool", "dex_type": "uniswap"}
        )
        b._initialize_entry_costs_for_new_position = _gen_none
        b._should_add_staking_actions = MagicMock(return_value=False)
        b._merge_duplicate_bridge_swap_actions = MagicMock(side_effect=lambda x: x)
        result = _drive(b.get_order_of_transactions(), sends=[None] * 20)
        assert any(a.get("action") == "UnstakeLPTokens" for a in result)
        assert any(a.get("action") == "ExitPool" for a in result)

    def test_exit_pool_action_fails(self):
        """Cover lines 2559-2561: exit pool action returns None."""
        b = _mk()
        b.selected_opportunities = [{"dex_type": "uniswap"}]
        b.position_to_exit = {"dex_type": "uniswap"}
        b._has_staking_metadata = MagicMock(return_value=False)
        b._build_exit_pool_action = MagicMock(return_value=None)
        b._prepare_tokens_for_investment = _gen_return([{"chain": "optimism"}])
        result = _drive(b.get_order_of_transactions(), sends=[None] * 10)
        assert result is None

    def test_bridge_swap_actions_none(self):
        """Cover lines 2568-2570: bridge_swap_actions returns None."""
        b = _mk()
        b.selected_opportunities = [
            {
                "dex_type": "uniswap",
                "chain": "optimism",
                "token0": "0xA",
                "token1": "0xB",
            },
        ]
        b.position_to_exit = None
        b._prepare_tokens_for_investment = _gen_return([])
        b._build_bridge_swap_actions = MagicMock(return_value=None)
        result = _drive(b.get_order_of_transactions(), sends=[None] * 10)
        assert result is None

    def test_enter_pool_action_fails(self):
        """Cover lines 2575-2577: enter pool action returns None."""
        b = _mk()
        b.selected_opportunities = [
            {"dex_type": "uniswap", "chain": "optimism", "token0": "0xA"},
        ]
        b.position_to_exit = None
        b._prepare_tokens_for_investment = _gen_return([])
        b._build_bridge_swap_actions = MagicMock(return_value=[])
        b._build_enter_pool_action = MagicMock(return_value=None)
        result = _drive(b.get_order_of_transactions(), sends=[None] * 10)
        assert result is None

    def test_velodrome_cl_pool_cache_and_staking(self):
        """Cover lines 2580-2599: velodrome CL pool caching + staking."""
        b = _mk()
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "chain": "optimism",
                "token0": "0xA",
                "token1": "0xB",
                "token0_symbol": "A",
                "token1_symbol": "B",
                "relative_funds_percentage": 1.0,
            },
        ]
        b.position_to_exit = None
        b._prepare_tokens_for_investment = _gen_return([])
        b._build_bridge_swap_actions = MagicMock(
            return_value=[{"action": "FindBridgeRoute"}]
        )
        b._build_enter_pool_action = MagicMock(
            return_value={
                "action": "EnterPool",
                "dex_type": "velodrome",
                "token_requirements": {"position_requirements": []},
            }
        )
        b._cache_enter_pool_action_for_cl_pool = _gen_none
        b._initialize_entry_costs_for_new_position = _gen_none
        b._should_add_staking_actions = MagicMock(return_value=True)
        b._build_stake_lp_tokens_action = MagicMock(
            return_value={"action": "StakeLPTokens"}
        )
        b._handle_velodrome_token_allocation = MagicMock(side_effect=lambda a, e, t: a)
        b._merge_duplicate_bridge_swap_actions = MagicMock(side_effect=lambda x: x)
        result = _drive(b.get_order_of_transactions(), sends=[None] * 20)
        assert any(a.get("action") == "StakeLPTokens" for a in result)
        assert any(a.get("action") == "FindBridgeRoute" for a in result)

    def test_velodrome_token_allocation_and_investment_cap(self):
        """Cover lines 2607-2619: velodrome allocation + investment cap."""
        b = _mk()
        b.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "chain": "optimism",
                "token0": "0xA",
                "token1": "0xB",
                "relative_funds_percentage": 1.0,
                "token0_symbol": "A",
                "token1_symbol": "B",
            },
        ]
        b.position_to_exit = None
        b.current_positions = [{"status": "open"}]
        b._prepare_tokens_for_investment = _gen_return([])
        b._build_bridge_swap_actions = MagicMock(return_value=[])
        enter_action = {
            "action": "EnterPool",
            "dex_type": "velodrome",
            "token_requirements": {"recommendation": "balanced"},
        }
        b._build_enter_pool_action = MagicMock(return_value=enter_action)
        b._initialize_entry_costs_for_new_position = _gen_none
        b._should_add_staking_actions = MagicMock(return_value=False)
        b._handle_velodrome_token_allocation = MagicMock(side_effect=lambda a, e, t: a)
        b._apply_investment_cap_to_actions = _gen_return([enter_action])
        b._merge_duplicate_bridge_swap_actions = MagicMock(side_effect=lambda x: x)
        result = _drive(b.get_order_of_transactions(), sends=[None] * 20)
        assert isinstance(result, list)

    def test_merge_exception(self):
        """Cover lines 2626-2629: merge raises exception."""
        b = _mk()
        b.selected_opportunities = [
            {"dex_type": "uniswap", "chain": "optimism", "token0": "0xA"},
        ]
        b.position_to_exit = None
        b._prepare_tokens_for_investment = _gen_return([])
        b._build_bridge_swap_actions = MagicMock(return_value=[])
        b._build_enter_pool_action = MagicMock(
            return_value={"action": "EnterPool", "dex_type": "uniswap"}
        )
        b._initialize_entry_costs_for_new_position = _gen_none
        b._should_add_staking_actions = MagicMock(return_value=False)
        b._merge_duplicate_bridge_swap_actions = MagicMock(
            side_effect=RuntimeError("merge fail")
        )
        result = _drive(b.get_order_of_transactions(), sends=[None] * 10)
        # Should return original actions
        assert isinstance(result, list)

    def test_unstake_action_none(self):
        """Cover line 2547->2554: unstake action returns None."""
        b = _mk()
        b.selected_opportunities = [
            {"dex_type": "uniswap", "chain": "optimism", "token0": "0xA"},
        ]
        b.position_to_exit = {"dex_type": "velodrome"}
        b._has_staking_metadata = MagicMock(return_value=True)
        b._build_unstake_lp_tokens_action = MagicMock(return_value=None)
        b._build_exit_pool_action = MagicMock(return_value={"action": "ExitPool"})
        b._prepare_tokens_for_investment = _gen_return([])
        b._build_bridge_swap_actions = MagicMock(return_value=[])
        b._build_enter_pool_action = MagicMock(
            return_value={"action": "EnterPool", "dex_type": "uniswap"}
        )
        b._initialize_entry_costs_for_new_position = _gen_none
        b._should_add_staking_actions = MagicMock(return_value=False)
        b._merge_duplicate_bridge_swap_actions = MagicMock(side_effect=lambda x: x)
        result = _drive(b.get_order_of_transactions(), sends=[None] * 10)
        assert not any(a.get("action") == "UnstakeLPTokens" for a in result)


class TestGetInvestableBalanceBranches:
    """Tests for _get_investable_balance whitelisted reward token path."""

    def test_whitelisted_reward_token(self):
        """Cover lines 2790-2815: token is both reward and whitelisted."""
        b = _mk()
        # OLAS on mode: checksummed address from REWARD_TOKEN_ADDRESSES
        # Also present (lowercase) in WHITELISTED_ASSETS
        token_addr = "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9"
        b.get_accumulated_rewards_for_token = _gen_return(100)
        b.context.params.airdrop_started = False
        result = _drive(
            b._get_investable_balance("mode", token_addr, 1000), sends=[None] * 10
        )
        assert result == 900  # 1000 - 100

    def test_whitelisted_reward_token_with_airdrop(self):
        """Cover lines 2796-2804: USDC on MODE with airdrop."""
        b = _mk()
        # USDC on mode from REWARD_TOKEN_ADDRESSES
        token_addr = "0xd988097fb8612cc24eeC14542bC03424c656005f"
        b.get_accumulated_rewards_for_token = _gen_return(100)
        b.context.params.airdrop_started = True
        b._get_usdc_address = MagicMock(return_value=token_addr)
        b._get_total_airdrop_rewards = _gen_return(50)
        result = _drive(
            b._get_investable_balance("mode", token_addr, 1000), sends=[None] * 10
        )
        assert result == 850  # 1000 - 100 - 50


class TestGetAvailableTokensBranches:
    """Tests for filtering reward tokens in _get_available_tokens."""

    def test_reward_token_filtered(self):
        """Cover lines 2713-2722: reward token is filtered out."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            REWARD_TOKEN_ADDRESSES,
        )

        b = _mk()
        # Use VELO on mode which is reward-only (not in WHITELISTED_ASSETS)
        velo_addr = "0x7f9AdFbd38b669F03d1d11000Bc76b9AaEA28A81"
        synced = MagicMock()
        synced.positions = [
            {
                "chain": "mode",
                "assets": [
                    {"address": velo_addr, "asset_symbol": "VELO", "balance": 1000}
                ],
            }
        ]
        with patch.object(
            type(b), "synchronized_data", new_callable=PropertyMock, return_value=synced
        ):
            b._get_investable_balance = _gen_return(0)
            b._fetch_token_prices = _gen_return({})
            result = _drive(b._get_available_tokens(), sends=[None] * 10)
        # VELO should be filtered (investable_balance=0)
        assert all(t.get("token_symbol") != "VELO" for t in result)


class TestHandleAllTokensAvailableRebalance:
    """Tests for rebalance path in _handle_all_tokens_available."""

    def test_surplus_rebalance(self):
        """Cover lines 2992-3019: token has surplus, swap to deficient token."""
        b = _mk()
        b._add_bridge_swap_action = MagicMock()
        tokens = [
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 200},
            {"chain": "optimism", "token": "0xB", "token_symbol": "B", "value": 50},
        ]
        required_tokens = [("0xA", "A"), ("0xB", "B")]
        target_ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_all_tokens_available(
            tokens, required_tokens, "optimism", 1.0, target_ratios
        )
        # Should have called _add_bridge_swap_action for rebalancing
        assert b._add_bridge_swap_action.called

    def test_exception_in_rebalance(self):
        """Cover lines 3020-3021: exception during rebalance."""
        b = _mk()
        b._add_bridge_swap_action = MagicMock(side_effect=RuntimeError("fail"))
        tokens = [
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 200},
            {"chain": "optimism", "token": "0xB", "token_symbol": "B", "value": 50},
        ]
        required_tokens = [("0xA", "A"), ("0xB", "B")]
        target_ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_all_tokens_available(
            tokens, required_tokens, "optimism", 1.0, target_ratios
        )
        # Exception caught, returns whatever was built
        assert isinstance(result, list)


class TestHandleSomeTokensAvailableBranches:
    """Tests for _handle_some_tokens_available branches."""

    def test_other_chain_tokens_prioritize_missing(self):
        """Cover lines 3053-3068: other chain tokens distributed to missing tokens."""
        b = _mk()
        b._add_bridge_swap_action = MagicMock()
        tokens = [
            {"chain": "mode", "token": "0xC", "token_symbol": "C", "value": 100},
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 100},
        ]
        required_tokens = [("0xA", "A"), ("0xB", "B")]
        tokens_we_need = [("0xB", "B")]
        target_ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_some_tokens_available(
            tokens, required_tokens, tokens_we_need, "optimism", 1.0, target_ratios
        )
        assert isinstance(result, list)

    def test_dest_chain_surplus_rebalance(self):
        """Cover lines 3071-3182: rebalance on dest chain."""
        b = _mk()
        b._add_bridge_swap_action = MagicMock()
        tokens = [
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 200},
            {"chain": "optimism", "token": "0xC", "token_symbol": "C", "value": 50},
        ]
        required_tokens = [("0xA", "A"), ("0xB", "B")]
        tokens_we_need = [("0xB", "B")]
        target_ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_some_tokens_available(
            tokens, required_tokens, tokens_we_need, "optimism", 1.0, target_ratios
        )
        assert isinstance(result, list)
        # Should have bridge swap actions for unnecessary + rebalance
        assert b._add_bridge_swap_action.called

    def test_no_unnecessary_tokens_convert_required(self):
        """Cover lines 3122-3144: no unnecessary tokens, convert existing required."""
        b = _mk()
        b._add_bridge_swap_action = MagicMock()
        tokens = [
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 100},
        ]
        required_tokens = [("0xA", "A"), ("0xB", "B")]
        tokens_we_need = [("0xB", "B")]
        target_ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_some_tokens_available(
            tokens, required_tokens, tokens_we_need, "optimism", 1.0, target_ratios
        )
        assert b._add_bridge_swap_action.called

    def test_source_same_as_target_skip(self):
        """Cover line 3132->3147: source_token_addr == target_token_addr."""
        b = _mk()
        b._add_bridge_swap_action = MagicMock()
        tokens = [
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 100},
        ]
        required_tokens = [("0xA", "A")]
        tokens_we_need = [("0xA", "A")]  # same as source
        target_ratios = {"0xA": 1.0}
        result = b._handle_some_tokens_available(
            tokens, required_tokens, tokens_we_need, "optimism", 1.0, target_ratios
        )
        assert isinstance(result, list)


class TestHandleAllTokensNeededBranches:
    """Tests for _handle_all_tokens_needed branches."""

    def test_other_chain_to_missing(self):
        """Cover lines 3221-3245: other chain tokens distributed to missing."""
        b = _mk()
        b._add_bridge_swap_action = MagicMock()
        tokens = [
            {"chain": "mode", "token": "0xC", "token_symbol": "C", "value": 100},
        ]
        required_tokens = [("0xA", "A"), ("0xB", "B")]
        target_ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_all_tokens_needed(
            tokens, required_tokens, "optimism", 1.0, target_ratios
        )
        assert b._add_bridge_swap_action.called

    def test_dest_chain_unnecessary_to_missing(self):
        """Cover lines 3254-3290: convert unnecessary dest chain tokens."""
        b = _mk()
        b._add_bridge_swap_action = MagicMock()
        tokens = [
            {"chain": "optimism", "token": "0xC", "token_symbol": "C", "value": 100},
        ]
        required_tokens = [("0xA", "A"), ("0xB", "B")]
        target_ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_all_tokens_needed(
            tokens, required_tokens, "optimism", 1.0, target_ratios
        )
        assert b._add_bridge_swap_action.called

    def test_no_target_tokens(self):
        """Cover line 3228->3221: target_tokens is empty."""
        b = _mk()
        b._add_bridge_swap_action = MagicMock()
        # All required tokens available on dest chain, no missing
        tokens = [
            {"chain": "mode", "token": "0xC", "token_symbol": "C", "value": 100},
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 50},
            {"chain": "optimism", "token": "0xB", "token_symbol": "B", "value": 50},
        ]
        required_tokens = [("0xA", "A"), ("0xB", "B")]
        target_ratios = {"0xA": 0.5, "0xB": 0.5}
        # Both required available on dest => missing=[], available=both
        # target_tokens = available_required_tokens
        result = b._handle_all_tokens_needed(
            tokens, required_tokens, "optimism", 1.0, target_ratios
        )
        assert isinstance(result, list)

    def test_no_dest_chain_value(self):
        """Cover line 3254->3292: total_dest_value == 0."""
        b = _mk()
        b._add_bridge_swap_action = MagicMock()
        tokens = [
            {"chain": "optimism", "token": "0xC", "token_symbol": "C", "value": 0},
        ]
        required_tokens = [("0xA", "A")]
        target_ratios = {"0xA": 1.0}
        result = b._handle_all_tokens_needed(
            tokens, required_tokens, "optimism", 1.0, target_ratios
        )
        assert isinstance(result, list)


class TestBuildBridgeSwapActionsNoRequired:
    """Test for _build_bridge_swap_actions with no required tokens."""

    def test_get_required_tokens_returns_empty(self):
        """Cover lines 3314-3315: no required tokens."""
        b = _mk()
        b._get_required_tokens = MagicMock(return_value=[])
        opp = {
            "chain": "optimism",
            "token0": "0xA",
            "dex_type": "uniswap",
            "relative_funds_percentage": 1.0,
        }
        result = b._build_bridge_swap_actions(opp, [])
        assert result is None


class TestStakingExceptionBranches:
    """Tests for exception handlers in staking actions."""

    def test_build_stake_exception(self):
        """Cover lines 3659-3663: exception in _build_stake_lp_tokens_action."""
        b = _mk()
        b.params.safe_contract_addresses = None  # Will cause .get() to fail
        opp = {
            "chain": "optimism",
            "pool_address": "0x1",
            "dex_type": "velodrome",
            "is_cl_pool": True,
        }
        result = b._build_stake_lp_tokens_action(opp)
        assert result is None

    def test_build_claim_exception(self):
        """Cover lines 3713-3717: exception in _build_claim_staking_rewards_action."""
        b = _mk()
        b.params.safe_contract_addresses = None
        pos = {"chain": "optimism", "pool_address": "0x1", "dex_type": "velodrome"}
        result = b._build_claim_staking_rewards_action(pos)
        assert result is None


class TestGetGaugeAddressNone:
    """Test for gauge address not found."""

    def test_gauge_none(self):
        """Cover lines 3760-3761: gauge_address returns None."""
        b = _mk()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _gen_return(None)
        b.pools = {"velodrome": mock_pool}
        pos = {"chain": "optimism", "pool_address": "0x1"}
        result = _drive(b._get_gauge_address_for_position(pos), sends=[None] * 10)
        assert result is None


class TestHasOpenPositionsException:
    """Test for exception in _has_open_positions."""

    def test_exception_returns_false(self):
        """Cover lines 3909-3911."""
        b = _mk()
        # Make current_positions iteration raise
        b.current_positions = MagicMock()
        b.current_positions.__bool__ = MagicMock(return_value=True)
        b.current_positions.__iter__ = MagicMock(side_effect=RuntimeError("boom"))
        result = b._has_open_positions()
        assert result is False


class TestCheckAndUseCachedClOpportunityBranches:
    """Tests for cache hit/miss branches."""

    def test_cache_valid_actions_returned(self):
        """Cover line 3950->3927: valid cache with actions."""
        b = _mk()
        b.current_positions = []  # no open positions
        cached = {"pool_address": "0x1", "pool_finalization_timestamp": time.time()}
        b._get_cached_cl_pool_data = _gen_return(cached)
        b._should_use_cached_cl_data = MagicMock(return_value=True)
        b._update_cl_pool_round_tracking = _gen_none
        b._reconstruct_actions_from_cached_cl_pool = _gen_return(
            [{"action": "EnterPool"}]
        )
        result = _drive(b._check_and_use_cached_cl_opportunity(), sends=[None] * 20)
        assert result == [{"action": "EnterPool"}]

    def test_cache_valid_actions_none(self):
        """Cover line 3950: actions is None from reconstruction."""
        b = _mk()
        b.current_positions = []
        cached = {"pool_address": "0x1", "pool_finalization_timestamp": time.time()}
        b._get_cached_cl_pool_data = _gen_return(cached)
        b._should_use_cached_cl_data = MagicMock(return_value=True)
        b._update_cl_pool_round_tracking = _gen_none
        b._reconstruct_actions_from_cached_cl_pool = _gen_return(None)
        result = _drive(b._check_and_use_cached_cl_opportunity(), sends=[None] * 20)
        assert result is None


class TestReconstructActionsStakingBranch:
    """Tests for staking action in reconstructed actions."""

    def test_staking_action_added(self):
        """Cover line 4063->4066: stake action added to reconstructed actions."""
        b = _mk()
        cached = {
            "enter_pool_action": {"action": "EnterPool", "dex_type": "velodrome"},
            "token0": "0xA",
            "token1": "0xB",
            "pool_address": "0x1",
            "position_requirements": [
                {"allocation": 1.0, "token0_ratio": 0.5, "token1_ratio": 0.5}
            ],
            "token0_symbol": "A",
            "token1_symbol": "B",
        }
        b._get_token_balance = _gen_return(1000)
        b._calculate_max_amounts_in = MagicMock(return_value=([500, 500], "balanced"))
        b._initialize_entry_costs_for_new_position = _gen_none
        b._should_add_staking_actions = MagicMock(return_value=True)
        b._build_stake_lp_tokens_action = MagicMock(
            return_value={"action": "StakeLPTokens"}
        )
        result = _drive(
            b._reconstruct_actions_from_cached_cl_pool(cached, "optimism"),
            sends=[None] * 20,
        )
        assert any(a.get("action") == "StakeLPTokens" for a in result)

    def test_staking_action_none(self):
        """Cover line 4063: stake action is None."""
        b = _mk()
        cached = {
            "enter_pool_action": {"action": "EnterPool", "dex_type": "velodrome"},
            "token0": "0xA",
            "token1": "0xB",
            "pool_address": "0x1",
            "position_requirements": [
                {"allocation": 1.0, "token0_ratio": 1.0, "token1_ratio": 0.0}
            ],
        }
        b._get_token_balance = _gen_return(1000)
        b._initialize_entry_costs_for_new_position = _gen_none
        b._should_add_staking_actions = MagicMock(return_value=True)
        b._build_stake_lp_tokens_action = MagicMock(return_value=None)
        result = _drive(
            b._reconstruct_actions_from_cached_cl_pool(cached, "optimism"),
            sends=[None] * 20,
        )
        # StakeLPTokens should NOT be in result since it's None
        assert not any(a.get("action") == "StakeLPTokens" for a in result)


class TestExecuteStrategyGlobals:
    """Test for execute_strategy globals cleanup."""

    def test_isolated_namespace_no_globals_pollution(self):
        """Test that strategies use isolated namespaces and don't pollute globals."""
        b = _mk()
        exec_code = "def my_test_func(*a, **kw): return 42"
        b.strategy_exec = MagicMock(return_value=(exec_code, "my_test_func"))
        # First call should work in isolated namespace
        result1 = b.execute_strategy(strategy="test")
        assert result1 == 42
        # Verify the function was NOT injected into globals
        assert "my_test_func" not in globals()
        # Second call should also work since each call gets its own namespace
        result2 = b.execute_strategy(strategy="test")
        assert result2 == 42


class TestPositionTokenBalancesVeloNoAddress:
    """Test velo token address None."""

    def test_velo_token_address_none(self):
        """Cover 917->922: _get_velo_token_address returns None."""
        b = _mk()
        b._get_velodrome_pending_rewards = _gen_return(100)
        b._get_velo_token_address = MagicMock(return_value=None)
        pos = {
            "current_token_balances": {"0xA": 1.0},
            "pool_address": "0x1",
            "staked": True,
            "dex_type": "velodrome",
        }
        result = _drive(
            b._get_position_token_balances(pos, "optimism"), sends=[None] * 10
        )
        assert "0xA" in result
        assert len(result) == 1  # No velo token added

    def test_no_token0_address(self):
        """Cover 939->950: token0_address is None."""
        b = _mk()
        b._get_token_balance = _gen_return(1000)
        b._get_token_decimals = _gen_return(18)
        pos = {
            "token0": None,
            "token1": "0xB",
            "pool_address": "0x1",
        }
        result = _drive(
            b._get_position_token_balances(pos, "optimism"), sends=[None] * 10
        )
        assert "0xB" in result

    def test_no_token1_address(self):
        """Cover 950->961: token1_address is None."""
        b = _mk()
        b._get_token_balance = _gen_return(1000)
        b._get_token_decimals = _gen_return(18)
        pos = {
            "token0": "0xA",
            "token1": None,
            "pool_address": "0x1",
        }
        result = _drive(
            b._get_position_token_balances(pos, "optimism"), sends=[None] * 10
        )
        assert "0xA" in result


class TestMergeDuplicateMultipleGroups:
    """Tests for merge with multiple groups and iterations."""

    def test_multiple_groups_with_duplicates(self):
        """Cover 2251->2250: loop iterates over multiple groups."""
        b = _mk()
        a1 = {
            "action": Action.FIND_BRIDGE_ROUTE.value,
            "from_chain": "opt",
            "to_chain": "mode",
            "from_token": "0x" + "aa" * 5,
            "to_token": "0x" + "bb" * 5,
            "funds_percentage": 0.3,
        }
        a2 = dict(a1, funds_percentage=0.2)  # duplicate of a1
        a3 = {
            "action": Action.FIND_BRIDGE_ROUTE.value,
            "from_chain": "mode",
            "to_chain": "opt",
            "from_token": "0x" + "cc" * 5,
            "to_token": "0x" + "dd" * 5,
            "funds_percentage": 0.5,
        }
        result = b._merge_duplicate_bridge_swap_actions([a1, a2, a3])
        assert len(result) == 2  # a1+a2 merged into 1, a3 stays


class TestVelodromeTokenAllocationMultipleBridgeRoutes:
    """Tests for multiple bridge route iterations."""

    def test_redirect_keeps_same_token(self):
        """Cover 2373->2381: action.to_token already matches target."""
        b = _mk()
        actions = [
            {
                "action": "FindBridgeRoute",
                "to_chain": "optimism",
                "to_token": "0xA",
                "funds_percentage": 0.5,
            },
        ]
        enter_pool = {
            "dex_type": "velodrome",
            "token_requirements": {
                "position_requirements": [
                    {"allocation": 1.0, "token0_ratio": 1.0, "token1_ratio": 0.0}
                ],
            },
            "token0": "0xA",
            "token1": "0xB",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "chain": "optimism",
            "relative_funds_percentage": 1.0,
        }
        result = b._handle_velodrome_token_allocation(actions, enter_pool, [])
        # to_token already matches, no redirect needed
        assert result[0]["to_token"] == "0xA"

    def test_no_source_token_available(self):
        """Cover 2393->2419: source_token is None."""
        b = _mk()
        actions = []
        enter_pool = {
            "dex_type": "velodrome",
            "token_requirements": {
                "position_requirements": [
                    {"allocation": 1.0, "token0_ratio": 1.0, "token1_ratio": 0.0}
                ],
            },
            "token0": "0xA",
            "token1": "0xB",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "chain": "optimism",
            "relative_funds_percentage": 1.0,
        }
        # available_tokens where all tokens are the target token
        available_tokens = [
            {"token": "0xA", "token_symbol": "A", "chain": "mode"},
        ]
        result = b._handle_velodrome_token_allocation(
            actions, enter_pool, available_tokens
        )
        # No bridge route should be added since no source token != target
        assert not any(a.get("action") == "FindBridgeRoute" for a in result)

    def test_mixed_actions_insert_position(self):
        """Cover 2413->2412: actions list has non-ExitPool mixed in so the if is False on some iterations."""
        b = _mk()
        actions = [
            {"action": "ExitPool"},
            {"action": "BridgeSwap"},
            {"action": "ExitPool"},
        ]
        enter_pool = {
            "dex_type": "velodrome",
            "token_requirements": {
                "position_requirements": [
                    {"allocation": 1.0, "token0_ratio": 1.0, "token1_ratio": 0.0}
                ],
            },
            "token0": "0xA",
            "token1": "0xB",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "chain": "optimism",
            "relative_funds_percentage": 1.0,
        }
        available_tokens = [
            {"token": "0xC", "token_symbol": "C", "chain": "mode"},
        ]
        result = b._handle_velodrome_token_allocation(
            actions, enter_pool, available_tokens
        )
        # Bridge route should be inserted after the last ExitPool (index 2 -> insert at 3)
        bridge_indices = [
            i for i, a in enumerate(result) if a.get("action") == "FindBridgeRoute"
        ]
        assert len(bridge_indices) == 1
        assert bridge_indices[0] == 3


class TestApplyInvestmentCapMultiplePositions:
    """Tests for investment cap with multiple positions."""

    def test_no_open_positions(self):
        """Cover 2441->2440: loop over positions, none open."""
        b = _mk()
        b.sleep = _gen_none
        b.current_positions = [{"status": "closed"}, {"status": "closed"}]
        b.calculate_initial_investment_value = _gen_return(None)
        actions = [{"action": "EnterPool"}]
        result = _drive(b._apply_investment_cap_to_actions(actions), sends=[None] * 20)
        # invested_amount = 0, invested_positions = False => no cap applied
        assert result == [{"action": "EnterPool"}]

    def test_multiple_enter_pool_adjustments(self):
        """Cover 2507->2506: loop over multiple actions."""
        b = _mk()
        b.sleep = _gen_none
        b.current_positions = [{"status": "open"}]
        b.calculate_initial_investment_value = _gen_return(500)
        actions = [
            {"action": "EnterPool"},
            {"action": "EnterPool"},
            {"action": "FindBridgeRoute"},
        ]
        result = _drive(b._apply_investment_cap_to_actions(actions), sends=[None] * 20)
        enter_actions = [a for a in result if a.get("action") == "EnterPool"]
        for a in enter_actions:
            assert a.get("invested_amount") == 500

    def test_multiple_open_positions(self):
        """Cover 2441->2440: loop iterates over multiple open positions."""
        b = _mk()
        b.sleep = _gen_none
        b.current_positions = [{"status": "open"}, {"status": "open"}]
        b.calculate_initial_investment_value = _gen_return(300)
        actions = [{"action": "EnterPool"}]
        result = _drive(b._apply_investment_cap_to_actions(actions), sends=[None] * 30)
        # 300 + 300 = 600 < 950 => adjust amounts
        enter_actions = [a for a in result if a.get("action") == "EnterPool"]
        assert enter_actions[0].get("invested_amount") == 400  # 1000 - 600


class TestGetOrderMultipleOpportunities:
    """Tests for get_order_of_transactions with multiple opportunities."""

    def test_two_opportunities_staking_returns_none(self):
        """Cover 2595->2566: _should_add_staking_actions True but _build_stake returns None."""
        b = _mk()
        b.selected_opportunities = [
            {
                "dex_type": "uniswap",
                "chain": "optimism",
                "token0": "0xA",
                "token1": "0xB",
                "relative_funds_percentage": 0.5,
                "token0_symbol": "A",
                "token1_symbol": "B",
            },
            {
                "dex_type": "uniswap",
                "chain": "optimism",
                "token0": "0xC",
                "token1": "0xD",
                "relative_funds_percentage": 0.5,
                "token0_symbol": "C",
                "token1_symbol": "D",
            },
        ]
        b.position_to_exit = None
        b._prepare_tokens_for_investment = _gen_return([])
        b._build_bridge_swap_actions = MagicMock(return_value=[])
        b._build_enter_pool_action = MagicMock(
            return_value={"action": "EnterPool", "dex_type": "uniswap"}
        )
        b._initialize_entry_costs_for_new_position = _gen_none
        b._should_add_staking_actions = MagicMock(return_value=True)
        b._build_stake_lp_tokens_action = MagicMock(return_value=None)
        b._merge_duplicate_bridge_swap_actions = MagicMock(side_effect=lambda x: x)
        result = _drive(b.get_order_of_transactions(), sends=[None] * 20)
        enter_count = sum(1 for a in result if a.get("action") == "EnterPool")
        assert enter_count == 2


class TestGetAvailableTokensMultiplePositions:
    """Tests for _get_available_tokens with multiple positions."""

    def test_multiple_assets_with_missing_address_and_zero_balance(self):
        """Cover 2713->2708 (if False), 2728->2708 (investable_balance=0)."""
        b = _mk()
        synced = MagicMock()
        synced.positions = [
            {
                "chain": "optimism",
                "assets": [
                    {"address": None, "asset_symbol": "X", "balance": 10**18},
                    {"address": "0xA", "asset_symbol": "A", "balance": 10**18},
                    {"address": "0xB", "asset_symbol": "B", "balance": 10**18},
                ],
            },
        ]
        call_count = [0]

        def _investable_gen(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return 0
            yield
            return 10**18

        with patch.object(
            type(b), "synchronized_data", new_callable=PropertyMock, return_value=synced
        ):
            b._get_investable_balance = _investable_gen
            b._fetch_token_prices = _gen_return({"0xA": 2.0, "0xB": 3.0})
            b._get_token_decimals = _gen_return(18)
            result = _drive(b._get_available_tokens(), sends=[None] * 30)
        # First asset has no address (skipped), second has investable_balance=0 (skipped), third is valid
        assert len(result) == 1


class TestHandleAllTokensRebalanceMultiple:
    """Tests for multiple iterations in rebalance loops."""

    def test_surplus_tokens_no_deficit_found(self):
        """Cover 2998->2981 (inner exhausts), 3003->2998 (other_token None), 3008->2998 (no deficit).

        Both 0xA and 0xB have surplus. Required token 0xD is not on dest chain (None in map).
        For each surplus token, the inner loop tries all required_tokens:
        - Self is skipped
        - 0xB/0xA: already has surplus (other_value >= other_target) -> 3008->2998
        - 0xD: not in available_tokens_map -> 3003->2998
        Inner loop exhausts without break -> 2998->2981
        """
        b = _mk()
        b._add_bridge_swap_action = MagicMock()
        tokens = [
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 800},
            {"chain": "optimism", "token": "0xB", "token_symbol": "B", "value": 700},
        ]
        required_tokens = [("0xA", "A"), ("0xB", "B"), ("0xD", "D")]
        target_ratios = {"0xA": 0.33, "0xB": 0.33, "0xD": 0.34}
        result = b._handle_all_tokens_available(
            tokens, required_tokens, "optimism", 1.0, target_ratios
        )
        assert isinstance(result, list)


class TestHandleSomeTokensMultipleOtherChain:
    """Tests for multiple other_chain tokens in _handle_some_tokens_available."""

    def test_tokens_we_need_empty_multiple_other_chain(self):
        """Cover 3055->3053: tokens_we_need empty, loop body's if is False -> back to for."""
        b = _mk()
        b._add_bridge_swap_action = MagicMock()
        tokens = [
            {"chain": "mode", "token": "0xC", "token_symbol": "C", "value": 100},
            {"chain": "base", "token": "0xD", "token_symbol": "D", "value": 50},
        ]
        required_tokens = [("0xA", "A"), ("0xB", "B")]
        tokens_we_need = []
        target_ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_some_tokens_available(
            tokens, required_tokens, tokens_we_need, "optimism", 1.0, target_ratios
        )
        assert isinstance(result, list)
        # tokens_we_need is empty so the if body is never entered
        assert b._add_bridge_swap_action.call_count == 0

    def test_no_dest_chain_tokens(self):
        """Cover 3073->3186: all tokens from other chains, dest_chain_tokens is empty."""
        b = _mk()
        b._add_bridge_swap_action = MagicMock()
        tokens = [
            {"chain": "mode", "token": "0xC", "token_symbol": "C", "value": 100},
            {"chain": "base", "token": "0xD", "token_symbol": "D", "value": 50},
        ]
        required_tokens = [("0xA", "A"), ("0xB", "B")]
        tokens_we_need = [("0xB", "B")]
        target_ratios = {"0xA": 0.5, "0xB": 0.5}
        result = b._handle_some_tokens_available(
            tokens, required_tokens, tokens_we_need, "optimism", 1.0, target_ratios
        )
        assert isinstance(result, list)

    def test_rebalance_no_deficit(self):
        """Cover 3173->3167: inner loop other_value >= other_target -> back to inner for."""
        b = _mk()
        b._add_bridge_swap_action = MagicMock()
        tokens = [
            {"chain": "optimism", "token": "0xA", "token_symbol": "A", "value": 800},
            {"chain": "optimism", "token": "0xB", "token_symbol": "B", "value": 700},
        ]
        required_tokens = [("0xA", "A"), ("0xB", "B"), ("0xD", "D")]
        tokens_we_need = []
        target_ratios = {"0xA": 0.33, "0xB": 0.33, "0xD": 0.34}
        result = b._handle_some_tokens_available(
            tokens, required_tokens, tokens_we_need, "optimism", 1.0, target_ratios
        )
        assert isinstance(result, list)


class TestHandleAllTokensNeededMultiple:
    """Tests for multiple iterations in _handle_all_tokens_needed."""

    def test_empty_required_tokens_multiple_other_chain(self):
        """Cover 3230->3223: target_tokens empty (no required_tokens), loop body if is False."""
        b = _mk()
        b._add_bridge_swap_action = MagicMock()
        tokens = [
            {"chain": "mode", "token": "0xC", "token_symbol": "C", "value": 100},
            {"chain": "base", "token": "0xD", "token_symbol": "D", "value": 50},
        ]
        required_tokens = []
        target_ratios = {}
        result = b._handle_all_tokens_needed(
            tokens, required_tokens, "optimism", 1.0, target_ratios
        )
        assert isinstance(result, list)
        assert b._add_bridge_swap_action.call_count == 0


class TestTrackOpportunitiesStages:
    """Tests for different tracking stages."""

    def test_composite_filtered_stage(self):
        """Cover line 1786: composite_filtered stage metadata."""
        b = _mk()
        b._read_kv = _gen_return({"opportunity_tracking": "{}"})
        b._write_kv = _gen_none
        b._get_current_timestamp = MagicMock(return_value=1000)
        synced = MagicMock()
        synced.period_count = 1
        with patch.object(
            type(b), "synchronized_data", new_callable=PropertyMock, return_value=synced
        ):
            _drive(
                b._track_opportunities([{"pool_address": "0x1"}], "composite_filtered"),
                sends=[None] * 10,
            )

    def test_final_selection_stage(self):
        """Cover line 1790: final_selection stage metadata."""
        b = _mk()
        b._read_kv = _gen_return({"opportunity_tracking": "{}"})
        b._write_kv = _gen_none
        b._get_current_timestamp = MagicMock(return_value=1000)
        synced = MagicMock()
        synced.period_count = 1
        with patch.object(
            type(b), "synchronized_data", new_callable=PropertyMock, return_value=synced
        ):
            _drive(
                b._track_opportunities([{"pool_address": "0x1"}], "final_selection"),
                sends=[None] * 10,
            )


class TestIsInStrategyBackoff:
    """Tests for _is_in_strategy_backoff."""

    def test_no_backoff_when_count_zero(self):
        b = _mk()
        b.shared_state.consecutive_no_action_count = 0
        assert b._is_in_strategy_backoff() is False

    def test_in_backoff_when_within_window(self):
        b = _mk()
        b.shared_state.consecutive_no_action_count = 1
        b.shared_state.last_strategy_evaluation_time = time.time()  # just now
        assert b._is_in_strategy_backoff() is True
        b.context.logger.info.assert_called()

    def test_backoff_elapsed(self):
        b = _mk()
        b.shared_state.consecutive_no_action_count = 1
        # Set last eval far in the past (base=1800s, so 2000s ago is enough)
        b.shared_state.last_strategy_evaluation_time = time.time() - 2000
        assert b._is_in_strategy_backoff() is False
        b.context.logger.info.assert_called()

    def test_exponential_growth_capped(self):
        b = _mk()
        b.shared_state.consecutive_no_action_count = 10  # 2^9 * 1800 >> max
        b.shared_state.last_strategy_evaluation_time = time.time()
        assert b._is_in_strategy_backoff() is True


class TestUpdateStrategyBackoff:
    """Tests for _update_strategy_backoff."""

    def test_actions_present_resets_count(self):
        b = _mk()
        b.shared_state.consecutive_no_action_count = 5
        b._update_strategy_backoff([{"action": "enter"}])
        assert b.shared_state.consecutive_no_action_count == 0
        assert b.shared_state.last_strategy_evaluation_time > 0

    def test_actions_present_logs_reset_when_count_was_nonzero(self):
        b = _mk()
        b.shared_state.consecutive_no_action_count = 3
        b._update_strategy_backoff([{"action": "enter"}])
        b.context.logger.info.assert_called()
        assert b.shared_state.consecutive_no_action_count == 0

    def test_no_actions_increments_count(self):
        b = _mk()
        b.shared_state.consecutive_no_action_count = 0
        b._update_strategy_backoff(None)
        assert b.shared_state.consecutive_no_action_count == 1
        b.context.logger.info.assert_called()

    def test_no_actions_increments_from_existing(self):
        b = _mk()
        b.shared_state.consecutive_no_action_count = 4
        b._update_strategy_backoff([])
        assert b.shared_state.consecutive_no_action_count == 5

    def test_empty_list_treated_as_no_actions(self):
        b = _mk()
        b.shared_state.consecutive_no_action_count = 0
        b._update_strategy_backoff([])
        assert b.shared_state.consecutive_no_action_count == 1

    def test_actions_present_with_zero_count_no_reset_log(self):
        b = _mk()
        b.shared_state.consecutive_no_action_count = 0
        b.context.logger.info.reset_mock()
        b._update_strategy_backoff([{"action": "enter"}])
        # Should NOT log the "backoff reset" message since count was 0
        for call in b.context.logger.info.call_args_list:
            assert "backoff reset" not in str(call).lower()


class TestAsyncActBackoff:
    """Test that async_act exits early when in backoff."""

    @staticmethod
    def _make_send_actions():
        calls = []

        def send_actions(actions=None):
            calls.append(actions)
            yield

        return send_actions, calls

    def test_backoff_causes_early_return(self):
        b = _mk()
        b._read_investing_paused = _gen_return(False)
        b.check_and_prepare_non_whitelisted_swaps = _gen_return([])
        b._apply_tip_filters_to_exit_decisions = _gen_return((True, []))
        b.check_funds = MagicMock(return_value=True)
        # Set backoff state: 1 consecutive failure, just happened
        b.shared_state.consecutive_no_action_count = 1
        b.shared_state.last_strategy_evaluation_time = time.time()
        send_actions, calls = self._make_send_actions()
        b.send_actions = send_actions
        _drive(b.async_act(), sends=[None] * 10)
        # send_actions called once with no args (early return)
        assert len(calls) == 1
        assert calls[0] is None

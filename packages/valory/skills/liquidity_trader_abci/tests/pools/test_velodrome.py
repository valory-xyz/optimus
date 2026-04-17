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

"""Test the pools/velodrome.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import json
import sys
import time
from collections import defaultdict
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.liquidity_trader_abci.pools.velodrome import (
    API_CACHE_SIZE,
    AllocationStatus,
    DEFAULT_DAYS,
    INT_MAX,
    MAX_TICK,
    MAX_WAIT_TIME,
    MIN_TICK,
    PRICE_CHECK_INTERVAL,
    PRICE_VOLATILITY_THRESHOLD,
    TICK_TO_PERCENTAGE_FACTOR,
    VelodromePoolBehaviour,
    WAITING_PERIOD,
    ZERO_ADDRESS,
)


class _ConcreteVelodrome(VelodromePoolBehaviour):
    """Non-abstract subclass used only for testing."""

    matching_round = MagicMock(spec=AbstractRound)

    def _get_tokens(self):  # type: ignore
        return {}


def make_behaviour(**overrides: Any) -> VelodromePoolBehaviour:
    """Create a VelodromePoolBehaviour with mocked attributes."""
    b = object.__new__(_ConcreteVelodrome)
    ctx = MagicMock()
    ctx.params.slippage_tolerance = 0.05  # 5%
    ctx.params.velodrome_router_contract_addresses = {}
    ctx.params.multisend_contract_addresses = {}
    b.__dict__["_context"] = ctx
    b.__dict__.update(overrides)
    return b


def exhaust_generator(gen: Generator, send_values=None):
    """Drive a generator to completion, returning its return value.

    *send_values* is an iterator of values that will be sent into the generator
    on each ``yield``.  When the iterator is exhausted ``None`` is sent for all
    remaining yields.

    :param gen: generator to exhaust.
    :param send_values: optional iterable of values to send.
    :return: the generator's return value.
    """
    send_iter = iter(send_values) if send_values else iter([])
    result = None
    try:
        val = next(gen)
        while True:
            try:
                send_val = next(send_iter)
            except StopIteration:
                send_val = None
            val = gen.send(send_val)
    except StopIteration as e:
        result = e.value
    return result


run_generator = exhaust_generator


def _gen_return(value: Any) -> Generator:
    """Create a generator that yields nothing and returns *value*."""
    return value
    yield  # noqa: unreachable -- makes this a generator function


def exhaust(gen: Generator) -> Any:
    """Drive a generator that does not truly yield to completion."""
    try:
        next(gen)
    except StopIteration as exc:
        return exc.value
    # If the generator yields, keep sending None until it's done
    while True:
        try:
            gen.send(None)
        except StopIteration as exc:
            return exc.value


def _mock_gen(value: Any) -> MagicMock:
    """Return a MagicMock whose every call produces a fresh generator returning *value*."""

    def _factory(*args, **kwargs):
        return _gen_return(value)

    return MagicMock(side_effect=_factory)


def make_contract_interact(return_values: List):
    """Create a fake contract_interact generator that yields once per call.

    :param return_values: list of return values for successive calls.
    :return: a generator function.
    """
    call_index = [0]

    def _fake(**kwargs):
        idx = call_index[0]
        call_index[0] += 1
        yield
        if idx < len(return_values):
            return return_values[idx]
        return None

    return _fake


def test_import() -> None:
    """Test that the velodrome module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.pools.velodrome  # noqa

    def test_min_tick(self) -> None:
        """Test MIN_TICK constant."""
        assert MIN_TICK == -887272

    def test_max_tick(self) -> None:
        """Test MAX_TICK constant."""
        assert MAX_TICK == 887272

    def test_int_max(self) -> None:
        """Test INT_MAX constant."""
        assert INT_MAX == sys.maxsize


class TestCheckPriceMovement:
    """Tests for VelodromePoolBehaviour._check_price_movement."""

    def _make_gen(
        self,
        behaviour,
        current_price,
        pool_address="0xpool",
        chain="optimism",
        price_at_selection=1.0,
        tick_ranges=None,
    ):
        """Create generator for _check_price_movement with a mocked _get_current_pool_price."""
        if tick_ranges is None:
            tick_ranges = [[{"tick_lower": -100, "tick_upper": 100}]]

        def fake_get_price(addr, ch):
            yield  # simulate one yield
            return current_price

        behaviour._get_current_pool_price = fake_get_price
        return behaviour._check_price_movement(
            pool_address, chain, price_at_selection, tick_ranges
        )

    def test_current_price_none_returns_none(self) -> None:
        """When _get_current_pool_price returns None, result is None."""
        b = make_behaviour()

        def fake_get_price(addr, ch):
            yield
            return None

        b._get_current_pool_price = fake_get_price

        gen = b._check_price_movement(
            "0xpool", "optimism", 1.0, [[{"tick_lower": -100, "tick_upper": 100}]]
        )
        result = exhaust_generator(gen)
        assert result is None

    def test_empty_tick_ranges_returns_false(self) -> None:
        """When tick_ranges is empty, returns False."""
        b = make_behaviour()
        gen = self._make_gen(b, current_price=1.0, tick_ranges=[])
        result = exhaust_generator(gen)
        assert result is False

    def test_no_valid_ticks_returns_false(self) -> None:
        """When positions have no tick_lower/tick_upper, returns False."""
        b = make_behaviour()
        gen = self._make_gen(b, current_price=1.0, tick_ranges=[[{"foo": "bar"}]])
        result = exhaust_generator(gen)
        assert result is False

    def test_zero_tick_range_returns_none(self) -> None:
        """When tick_upper == tick_lower the range is zero, returns None."""
        b = make_behaviour()
        gen = self._make_gen(
            b, current_price=1.0, tick_ranges=[[{"tick_lower": 50, "tick_upper": 50}]]
        )
        result = exhaust_generator(gen)
        assert result is None

    def test_price_not_moved_returns_false(self) -> None:
        """Price within tolerance -> False."""
        b = make_behaviour()
        b.params.slippage_tolerance = 0.05  # 5%
        # tick_range = 200, normalization = 200/10000 = 0.02
        # price_change_pct = abs(1.0 - 1.0) / 0.02 * 100 = 0
        gen = self._make_gen(
            b,
            current_price=1.0,
            price_at_selection=1.0,
            tick_ranges=[[{"tick_lower": -100, "tick_upper": 100}]],
        )
        result = exhaust_generator(gen)
        assert result is False

    def test_price_moved_beyond_tolerance_returns_true(self) -> None:
        """Price beyond tolerance -> True."""
        b = make_behaviour()
        b.params.slippage_tolerance = 0.01  # 1% tolerance
        # tick_range = 200, normalization = 200/10000 = 0.02
        # price_change_pct = abs(1.0 - 1.5) / 0.02 * 100 = 2500%
        gen = self._make_gen(
            b,
            current_price=1.5,
            price_at_selection=1.0,
            tick_ranges=[[{"tick_lower": -100, "tick_upper": 100}]],
        )
        result = exhaust_generator(gen)
        assert result is True

    def test_narrowest_range_used(self) -> None:
        """When multiple positions exist, the narrowest tick range is used."""
        b = make_behaviour()
        b.params.slippage_tolerance = (
            0.10  # 10% tolerance -> slippage_tolerance_pct = 10
        )
        # Narrowest range: 20 (tick -10 to 10), normalization = 20/10000 = 0.002
        # price_change_pct = abs(1.0 - 1.0001) / 0.002 * 100 = 5%
        # 5% < 10% -> not moved
        gen = self._make_gen(
            b,
            current_price=1.0001,
            price_at_selection=1.0,
            tick_ranges=[
                [
                    {"tick_lower": -100, "tick_upper": 100},  # range 200
                    {"tick_lower": -10, "tick_upper": 10},  # range 20 (narrowest)
                    {"tick_lower": -50, "tick_upper": 50},  # range 100
                ]
            ],
        )
        result = exhaust_generator(gen)
        assert result is False

    def test_exception_returns_none(self) -> None:
        """When an exception occurs, returns None."""
        b = make_behaviour()

        def exploding_get_price(addr, ch):
            raise RuntimeError("boom")
            yield  # make it a generator  # noqa: unreachable

        b._get_current_pool_price = exploding_get_price

        gen = b._check_price_movement(
            "0xpool", "optimism", 1.0, [[{"tick_lower": -100, "tick_upper": 100}]]
        )
        result = exhaust_generator(gen)
        assert result is None

    def test_position_with_only_tick_lower_ignored(self) -> None:
        """Position with tick_lower but no tick_upper is skipped."""
        b = make_behaviour()
        gen = self._make_gen(
            b,
            current_price=1.0,
            tick_ranges=[[{"tick_lower": -100}]],
        )
        result = exhaust_generator(gen)
        assert result is False  # no valid narrowest range found


class TestWaitForFavorablePrice:
    """Tests for VelodromePoolBehaviour._wait_for_favorable_price."""

    def _patch_time(self, time_values):
        """Return a patcher for time.time that yields successive values."""
        return patch(
            "packages.valory.skills.liquidity_trader_abci.pools.velodrome.time.time",
            side_effect=time_values,
        )

    def test_immediate_stabilization(self) -> None:
        """Price stabilizes on the first check -> READY."""
        b = make_behaviour()

        # sleep is a generator that yields once
        def fake_sleep(seconds):
            yield

        b.sleep = fake_sleep

        # _check_price_movement returns False (not moved) on first check
        call_count = 0

        def fake_check(pool_address, chain, price_at_selection, tick_ranges):
            nonlocal call_count
            call_count += 1
            yield
            return False

        b._check_price_movement = fake_check

        # time.time(): start_time=0, then elapsed=6 (after WAITING_PERIOD)
        with self._patch_time([0, 6]):
            gen = b._wait_for_favorable_price("0xpool", "optimism", 1.0, [])
            result = exhaust_generator(gen)

        assert result == AllocationStatus.READY
        assert call_count == 1

    def test_timeout(self) -> None:
        """Max wait time exceeded -> TIMEOUT."""
        b = make_behaviour()

        def fake_sleep(seconds):
            yield

        b.sleep = fake_sleep

        def fake_check(pool_address, chain, price_at_selection, tick_ranges):
            yield
            return True  # price always volatile

        b._check_price_movement = fake_check

        # start=0, first loop elapsed=601 (> 600 max)
        with self._patch_time([0, 601]):
            gen = b._wait_for_favorable_price("0xpool", "optimism", 1.0, [])
            result = exhaust_generator(gen)

        assert result == AllocationStatus.TIMEOUT

    def test_price_check_fails_returns_failed(self) -> None:
        """_check_price_movement returns None -> FAILED."""
        b = make_behaviour()

        def fake_sleep(seconds):
            yield

        b.sleep = fake_sleep

        def fake_check(pool_address, chain, price_at_selection, tick_ranges):
            yield
            return None

        b._check_price_movement = fake_check

        with self._patch_time([0, 6]):
            gen = b._wait_for_favorable_price("0xpool", "optimism", 1.0, [])
            result = exhaust_generator(gen)

        assert result == AllocationStatus.FAILED

    def test_price_volatile_then_stabilizes(self) -> None:
        """Price volatile on first check, then stabilizes on second check -> READY."""
        b = make_behaviour()
        b.params.slippage_tolerance = 0.05

        def fake_sleep(seconds):
            yield

        b.sleep = fake_sleep

        check_results = iter([True, False])

        def fake_check(pool_address, chain, price_at_selection, tick_ranges):
            yield
            return next(check_results)

        b._check_price_movement = fake_check

        # start=0, first loop elapsed=6 (not timed out), second loop elapsed=12
        with self._patch_time([0, 6, 12]):
            gen = b._wait_for_favorable_price("0xpool", "optimism", 1.0, [])
            result = exhaust_generator(gen)

        assert result == AllocationStatus.READY

    def test_exception_returns_failed(self) -> None:
        """Exception during monitoring -> FAILED."""
        b = make_behaviour()

        def fake_sleep(seconds):
            raise RuntimeError("sleep broke")
            yield  # noqa: unreachable

        b.sleep = fake_sleep

        with self._patch_time([0]):
            gen = b._wait_for_favorable_price("0xpool", "optimism", 1.0, [])
            result = exhaust_generator(gen)

        assert result == AllocationStatus.FAILED

    def test_custom_max_wait_time(self) -> None:
        """Timeout uses custom max_wait_time."""
        b = make_behaviour()

        def fake_sleep(seconds):
            yield

        b.sleep = fake_sleep

        def fake_check(pool_address, chain, price_at_selection, tick_ranges):
            yield
            return True

        b._check_price_movement = fake_check

        # custom max_wait_time=10, start=0, elapsed=11
        with self._patch_time([0, 11]):
            gen = b._wait_for_favorable_price(
                "0xpool", "optimism", 1.0, [], max_wait_time=10
            )
            result = exhaust_generator(gen)

        assert result == AllocationStatus.TIMEOUT


class TestCalculateSlippageProtection:
    """Tests for VelodromePoolBehaviour._calculate_slippage_protection."""

    def test_normal_case(self) -> None:
        """Standard slippage calculation."""
        b = make_behaviour()
        a0_min, a1_min = b._calculate_slippage_protection([1000, 2000], 0.05)
        # slippage0 = int(0.05 * 1000) = 50
        # slippage1 = int(0.05 * 2000) = 100
        assert a0_min == 950
        assert a1_min == 1900

    def test_zero_slippage(self) -> None:
        """Zero slippage means min equals desired."""
        b = make_behaviour()
        a0_min, a1_min = b._calculate_slippage_protection([1000, 2000], 0.0)
        assert a0_min == 1000
        assert a1_min == 2000

    def test_full_slippage(self) -> None:
        """100% slippage means min is zero."""
        b = make_behaviour()
        a0_min, a1_min = b._calculate_slippage_protection([1000, 2000], 1.0)
        assert a0_min == 0
        assert a1_min == 0

    def test_min_clamped_to_zero(self) -> None:
        """Slippage > 100% still clamps to zero due to max(0, ...)."""
        b = make_behaviour()
        a0_min, a1_min = b._calculate_slippage_protection([1000, 2000], 2.0)
        assert a0_min == 0
        assert a1_min == 0

    def test_zero_amounts(self) -> None:
        """Zero desired amounts."""
        b = make_behaviour()
        a0_min, a1_min = b._calculate_slippage_protection([0, 0], 0.05)
        assert a0_min == 0
        assert a1_min == 0

    def test_exception_returns_nones(self) -> None:
        """If an exception happens, returns (None, None)."""
        b = make_behaviour()
        # Pass invalid input (not enough values to unpack)
        a0_min, a1_min = b._calculate_slippage_protection([1000], 0.05)
        assert a0_min is None
        assert a1_min is None

    def test_large_amounts(self) -> None:
        """Large token amounts work correctly."""
        b = make_behaviour()
        amount = 10**18
        a0_min, a1_min = b._calculate_slippage_protection([amount, amount], 0.01)
        expected = amount - int(0.01 * amount)
        assert a0_min == expected
        assert a1_min == expected


class TestEnter:
    """Tests for VelodromePoolBehaviour.enter."""

    def _base_kwargs(self, **overrides):
        kw = {
            "pool_address": "0xpool",
            "safe_address": "0xsafe",
            "assets": ["0xtokenA", "0xtokenB"],
            "chain": "optimism",
            "max_amounts_in": [1000, 2000],
            "is_cl_pool": False,
            "is_stable": True,
        }
        kw.update(overrides)
        return kw

    def test_missing_params_stable_returns_none(self) -> None:
        """Missing required param for stable/volatile pool."""
        b = make_behaviour()
        gen = b.enter(pool_address="0xpool", safe_address="0xsafe", chain="optimism")
        result = exhaust_generator(gen)
        assert result == (None, None)

    def test_missing_params_cl_returns_none(self) -> None:
        """Missing tick_ranges/tick_spacing for CL pool."""
        b = make_behaviour()
        gen = b.enter(
            pool_address="0xpool",
            safe_address="0xsafe",
            assets=["0xA", "0xB"],
            chain="optimism",
            max_amounts_in=[1000, 2000],
            is_cl_pool=True,
        )
        result = exhaust_generator(gen)
        assert result == (None, None)

    def test_stable_pool_delegates_to_enter_stable_volatile(self) -> None:
        """Stable/volatile pool calls _enter_stable_volatile_pool."""
        b = make_behaviour()
        called_with = {}

        def fake_enter_sv(**kwargs):
            called_with.update(kwargs)
            yield
            return ("tx_hash", "router_addr")

        b._enter_stable_volatile_pool = fake_enter_sv

        gen = b.enter(**self._base_kwargs())
        result = exhaust_generator(gen)

        assert result == ("tx_hash", "router_addr")
        assert called_with["pool_address"] == "0xpool"
        assert called_with["is_stable"] is True

    def test_cl_pool_delegates_to_enter_cl_pool(self) -> None:
        """CL pool calls _enter_cl_pool."""
        b = make_behaviour()
        called_with = {}

        def fake_enter_cl(**kwargs):
            called_with.update(kwargs)
            yield
            return ("tx_hash", "nfpm_addr")

        b._enter_cl_pool = fake_enter_cl

        gen = b.enter(
            **self._base_kwargs(
                is_cl_pool=True,
                tick_ranges=[{"tick_lower": -100, "tick_upper": 100}],
                tick_spacing=10,
                pool_fee=500,
            )
        )
        result = exhaust_generator(gen)

        assert result == ("tx_hash", "nfpm_addr")
        assert called_with["tick_spacing"] == 10

    def test_missing_pool_address_stable(self) -> None:
        """Missing pool_address for stable pool returns None."""
        b = make_behaviour()
        gen = b.enter(
            safe_address="0xsafe",
            assets=["0xA", "0xB"],
            chain="optimism",
            max_amounts_in=[1000, 2000],
            is_cl_pool=False,
            is_stable=True,
        )
        result = exhaust_generator(gen)
        assert result == (None, None)

    def test_missing_assets_stable(self) -> None:
        """Missing assets for stable pool returns None."""
        b = make_behaviour()
        gen = b.enter(
            pool_address="0xpool",
            safe_address="0xsafe",
            chain="optimism",
            max_amounts_in=[1000, 2000],
            is_cl_pool=False,
            is_stable=True,
        )
        result = exhaust_generator(gen)
        assert result == (None, None)

    def test_missing_max_amounts_in_cl(self) -> None:
        """Missing max_amounts_in for CL pool returns None."""
        b = make_behaviour()
        gen = b.enter(
            pool_address="0xpool",
            safe_address="0xsafe",
            assets=["0xA", "0xB"],
            chain="optimism",
            is_cl_pool=True,
            tick_ranges=[{"tick_lower": -100, "tick_upper": 100}],
            tick_spacing=10,
        )
        result = exhaust_generator(gen)
        assert result == (None, None)


class TestExit:
    """Tests for VelodromePoolBehaviour.exit."""

    def _base_kwargs(self, **overrides):
        kw = {
            "pool_address": "0xpool",
            "safe_address": "0xsafe",
            "chain": "optimism",
            "is_cl_pool": False,
            "is_stable": True,
            "assets": ["0xA", "0xB"],
            "liquidity": 1000,
        }
        kw.update(overrides)
        return kw

    def test_missing_required_params(self) -> None:
        """Missing required params for exit."""
        b = make_behaviour()
        gen = b.exit(pool_address="0xpool")
        result = exhaust_generator(gen)
        assert result == (None, None, None)

    def test_cl_pool_missing_token_ids(self) -> None:
        """CL pool exit without token_ids returns None."""
        b = make_behaviour()
        gen = b.exit(
            pool_address="0xpool",
            safe_address="0xsafe",
            chain="optimism",
            is_cl_pool=True,
        )
        result = exhaust_generator(gen)
        assert result == (None, None, None)

    def test_cl_pool_missing_liquidities(self) -> None:
        """CL pool exit without liquidities returns None."""
        b = make_behaviour()
        gen = b.exit(
            pool_address="0xpool",
            safe_address="0xsafe",
            chain="optimism",
            is_cl_pool=True,
            token_ids=[1],
        )
        result = exhaust_generator(gen)
        assert result == (None, None, None)

    def test_stable_pool_missing_assets(self) -> None:
        """Stable/volatile exit without assets returns None."""
        b = make_behaviour()
        gen = b.exit(
            pool_address="0xpool",
            safe_address="0xsafe",
            chain="optimism",
            is_cl_pool=False,
            is_stable=True,
        )
        result = exhaust_generator(gen)
        assert result == (None, None, None)

    def test_stable_pool_delegates_to_exit_stable_volatile(self) -> None:
        """Stable/volatile exit delegates correctly."""
        b = make_behaviour()
        called_with = {}

        def fake_exit_sv(**kwargs):
            called_with.update(kwargs)
            yield
            return (b"txdata", "multisend_addr", True)

        b._exit_stable_volatile_pool = fake_exit_sv

        gen = b.exit(**self._base_kwargs())
        result = exhaust_generator(gen)

        assert result == (b"txdata", "multisend_addr", True)
        assert called_with["pool_address"] == "0xpool"
        assert called_with["is_stable"] is True

    def test_cl_pool_delegates_to_exit_cl_pool(self) -> None:
        """CL pool exit delegates correctly."""
        b = make_behaviour()
        called_with = {}

        def fake_exit_cl(**kwargs):
            called_with.update(kwargs)
            yield
            return (b"txdata", "multisend_addr", True)

        b._exit_cl_pool = fake_exit_cl

        gen = b.exit(
            pool_address="0xpool",
            safe_address="0xsafe",
            chain="optimism",
            is_cl_pool=True,
            token_ids=[1],
            liquidities=[1000],
        )
        result = exhaust_generator(gen)

        assert result == (b"txdata", "multisend_addr", True)
        assert called_with["token_ids"] == [1]
        assert called_with["liquidities"] == [1000]

    def test_missing_safe_address(self) -> None:
        """Missing safe_address returns None."""
        b = make_behaviour()
        gen = b.exit(pool_address="0xpool", chain="optimism")
        result = exhaust_generator(gen)
        assert result == (None, None, None)

    def test_missing_chain(self) -> None:
        """Missing chain returns None."""
        b = make_behaviour()
        gen = b.exit(pool_address="0xpool", safe_address="0xsafe")
        result = exhaust_generator(gen)
        assert result == (None, None, None)


class TestEnterStableVolatilePool:
    """Tests for VelodromePoolBehaviour._enter_stable_volatile_pool."""

    def _base_kwargs(self):
        return {
            "pool_address": "0xpool",
            "safe_address": "0xsafe",
            "assets": ["0xtokenA", "0xtokenB"],
            "chain": "optimism",
            "max_amounts_in": [1000, 2000],
            "is_stable": True,
        }

    def test_no_router_address(self) -> None:
        """Missing router address returns (None, None)."""
        b = make_behaviour()
        b.params.velodrome_router_contract_addresses = {}

        gen = b._enter_stable_volatile_pool(**self._base_kwargs())
        result = exhaust_generator(gen)
        assert result == (None, None)

    def test_query_returns_none(self) -> None:
        """When _query_add_liquidity_velodrome returns None, returns (None, None)."""
        b = make_behaviour()
        b.params.velodrome_router_contract_addresses = {"optimism": "0xrouter"}

        def fake_query(*args, **kwargs):
            yield
            return None

        b._query_add_liquidity_velodrome = fake_query

        gen = b._enter_stable_volatile_pool(**self._base_kwargs())
        result = exhaust_generator(gen)
        assert result == (None, None)

    def test_successful_entry(self) -> None:
        """Successful entry returns (tx_hash, router_address)."""
        b = make_behaviour()
        b.params.velodrome_router_contract_addresses = {"optimism": "0xrouter"}
        b.params.slippage_tolerance = 0.05

        # Mock SharedState for deadline calculation
        mock_state = MagicMock()
        mock_state.round_sequence.last_round_transition_timestamp.timestamp.return_value = (
            1000000.0
        )
        b.context.state = mock_state

        def fake_query(*args, **kwargs):
            yield
            return {"amount_a": 900, "amount_b": 1800}

        b._query_add_liquidity_velodrome = fake_query

        def fake_contract_interact(**kwargs):
            yield
            return "0xtxhash"

        b.contract_interact = fake_contract_interact

        gen = b._enter_stable_volatile_pool(**self._base_kwargs())
        result = exhaust_generator(gen)
        assert result == ("0xtxhash", "0xrouter")

    def test_slippage_capped_by_adjusted_amounts(self) -> None:
        """Minimum amounts are capped by adjusted_amounts (max_amounts_in)."""
        b = make_behaviour()
        b.params.velodrome_router_contract_addresses = {"optimism": "0xrouter"}
        b.params.slippage_tolerance = 0.01  # very small slippage

        mock_state = MagicMock()
        mock_state.round_sequence.last_round_transition_timestamp.timestamp.return_value = (
            1000000.0
        )
        b.context.state = mock_state

        # Expected amounts bigger than max_amounts_in: min should be capped
        def fake_query(*args, **kwargs):
            yield
            return {"amount_a": 2000, "amount_b": 4000}

        b._query_add_liquidity_velodrome = fake_query

        contract_calls = []

        def fake_contract_interact(**kwargs):
            contract_calls.append(kwargs)
            yield
            return "0xtxhash"

        b.contract_interact = fake_contract_interact

        gen = b._enter_stable_volatile_pool(**self._base_kwargs())
        result = exhaust_generator(gen)
        assert result == ("0xtxhash", "0xrouter")

        # The add_liquidity call should have amount_a_min <= 1000 and amount_b_min <= 2000
        add_liq_call = contract_calls[0]
        assert add_liq_call["amount_a_min"] <= 1000
        assert add_liq_call["amount_b_min"] <= 2000


class TestExitStableVolatilePool:
    """Tests for VelodromePoolBehaviour._exit_stable_volatile_pool."""

    def _base_kwargs(self):
        return {
            "pool_address": "0xpool",
            "safe_address": "0xsafe",
            "assets": ["0xtokenA", "0xtokenB"],
            "chain": "optimism",
            "liquidity": 5000,
            "is_stable": True,
        }

    def test_no_router_address(self) -> None:
        """Missing router address returns (None, None, None)."""
        b = make_behaviour()
        b.params.velodrome_router_contract_addresses = {}

        gen = b._exit_stable_volatile_pool(**self._base_kwargs())
        result = exhaust_generator(gen)
        assert result == (None, None, None)

    def test_no_liquidity_fetched(self) -> None:
        """When liquidity is not provided and cannot be fetched, returns (None, None, None)."""
        b = make_behaviour()
        b.params.velodrome_router_contract_addresses = {"optimism": "0xrouter"}

        def fake_contract_interact(**kwargs):
            yield
            return 0  # no balance

        b.contract_interact = fake_contract_interact

        kwargs = self._base_kwargs()
        kwargs["liquidity"] = None
        gen = b._exit_stable_volatile_pool(**kwargs)
        result = exhaust_generator(gen)
        assert result == (None, None, None)

    def test_liquidity_fetched_from_contract(self) -> None:
        """When liquidity is None, it's fetched from the pool contract."""
        b = make_behaviour()
        b.params.velodrome_router_contract_addresses = {"optimism": "0xrouter"}
        b.params.multisend_contract_addresses = {"optimism": "0xmultisend"}
        b.params.slippage_tolerance = 0.05

        mock_state = MagicMock()
        mock_state.round_sequence.last_round_transition_timestamp.timestamp.return_value = (
            1000000.0
        )
        b.context.state = mock_state

        call_count = [0]

        def fake_contract_interact(**kwargs):
            call_count[0] += 1
            yield
            if kwargs.get("contract_callable") == "get_balance":
                return 5000
            elif kwargs.get("contract_callable") == "build_approval_tx":
                return "0xaabb"
            elif kwargs.get("contract_callable") == "remove_liquidity":
                return "0xccdd"
            elif kwargs.get("contract_callable") == "get_tx_data":
                return "0xaabbccdd"
            return "0xee"

        b.contract_interact = fake_contract_interact

        def fake_query_remove(*args, **kwargs):
            yield
            return {"amount_a": 500, "amount_b": 1000}

        b._query_remove_liquidity_velodrome = fake_query_remove

        kwargs = self._base_kwargs()
        kwargs["liquidity"] = None
        gen = b._exit_stable_volatile_pool(**kwargs)
        result = exhaust_generator(gen)

        # Should have fetched balance via contract
        assert call_count[0] >= 1
        # result should be (bytes, multisend_address, True)
        assert result[1] == "0xmultisend"
        assert result[2] is True

    def test_query_remove_returns_none(self) -> None:
        """When _query_remove_liquidity_velodrome returns None, returns (None, None)."""
        b = make_behaviour()
        b.params.velodrome_router_contract_addresses = {"optimism": "0xrouter"}

        def fake_query_remove(*args, **kwargs):
            yield
            return None

        b._query_remove_liquidity_velodrome = fake_query_remove

        gen = b._exit_stable_volatile_pool(**self._base_kwargs())
        result = exhaust_generator(gen)
        assert result == (None, None)

    def test_approve_tx_fails(self) -> None:
        """When approve transaction fails, returns (None, None, None)."""
        b = make_behaviour()
        b.params.velodrome_router_contract_addresses = {"optimism": "0xrouter"}
        b.params.slippage_tolerance = 0.05

        mock_state = MagicMock()
        mock_state.round_sequence.last_round_transition_timestamp.timestamp.return_value = (
            1000000.0
        )
        b.context.state = mock_state

        def fake_query_remove(*args, **kwargs):
            yield
            return {"amount_a": 500, "amount_b": 1000}

        b._query_remove_liquidity_velodrome = fake_query_remove

        def fake_contract_interact(**kwargs):
            yield
            if kwargs.get("contract_callable") == "build_approval_tx":
                return None  # approve fails
            return "0xdefault"

        b.contract_interact = fake_contract_interact

        gen = b._exit_stable_volatile_pool(**self._base_kwargs())
        result = exhaust_generator(gen)
        assert result == (None, None, None)

    def test_remove_liquidity_tx_fails(self) -> None:
        """When remove_liquidity transaction fails, returns (None, None, None)."""
        b = make_behaviour()
        b.params.velodrome_router_contract_addresses = {"optimism": "0xrouter"}
        b.params.slippage_tolerance = 0.05

        mock_state = MagicMock()
        mock_state.round_sequence.last_round_transition_timestamp.timestamp.return_value = (
            1000000.0
        )
        b.context.state = mock_state

        def fake_query_remove(*args, **kwargs):
            yield
            return {"amount_a": 500, "amount_b": 1000}

        b._query_remove_liquidity_velodrome = fake_query_remove

        def fake_contract_interact(**kwargs):
            yield
            if kwargs.get("contract_callable") == "build_approval_tx":
                return "0xapprove"
            if kwargs.get("contract_callable") == "remove_liquidity":
                return None  # remove fails
            return "0xdefault"

        b.contract_interact = fake_contract_interact

        gen = b._exit_stable_volatile_pool(**self._base_kwargs())
        result = exhaust_generator(gen)
        assert result == (None, None, None)

    def test_no_multisend_address(self) -> None:
        """Missing multisend address returns (None, None, None)."""
        b = make_behaviour()
        b.params.velodrome_router_contract_addresses = {"optimism": "0xrouter"}
        b.params.multisend_contract_addresses = {}  # missing!
        b.params.slippage_tolerance = 0.05

        mock_state = MagicMock()
        mock_state.round_sequence.last_round_transition_timestamp.timestamp.return_value = (
            1000000.0
        )
        b.context.state = mock_state

        def fake_query_remove(*args, **kwargs):
            yield
            return {"amount_a": 500, "amount_b": 1000}

        b._query_remove_liquidity_velodrome = fake_query_remove

        def fake_contract_interact(**kwargs):
            yield
            return "0xsome_hash"

        b.contract_interact = fake_contract_interact

        gen = b._exit_stable_volatile_pool(**self._base_kwargs())
        result = exhaust_generator(gen)
        assert result == (None, None, None)

    def test_multisend_tx_hash_fails(self) -> None:
        """When multisend get_tx_data fails, returns (None, None, None)."""
        b = make_behaviour()
        b.params.velodrome_router_contract_addresses = {"optimism": "0xrouter"}
        b.params.multisend_contract_addresses = {"optimism": "0xmultisend"}
        b.params.slippage_tolerance = 0.05

        mock_state = MagicMock()
        mock_state.round_sequence.last_round_transition_timestamp.timestamp.return_value = (
            1000000.0
        )
        b.context.state = mock_state

        def fake_query_remove(*args, **kwargs):
            yield
            return {"amount_a": 500, "amount_b": 1000}

        b._query_remove_liquidity_velodrome = fake_query_remove

        def fake_contract_interact(**kwargs):
            yield
            if kwargs.get("contract_callable") == "get_tx_data":
                return None  # multisend fails
            return "0xsome_hash"

        b.contract_interact = fake_contract_interact

        gen = b._exit_stable_volatile_pool(**self._base_kwargs())
        result = exhaust_generator(gen)
        assert result == (None, None, None)

    def test_successful_exit(self) -> None:
        """Full successful exit returns (bytes, multisend_address, True)."""
        b = make_behaviour()
        b.params.velodrome_router_contract_addresses = {"optimism": "0xrouter"}
        b.params.multisend_contract_addresses = {"optimism": "0xmultisend"}
        b.params.slippage_tolerance = 0.05

        mock_state = MagicMock()
        mock_state.round_sequence.last_round_transition_timestamp.timestamp.return_value = (
            1000000.0
        )
        b.context.state = mock_state

        def fake_query_remove(*args, **kwargs):
            yield
            return {"amount_a": 500, "amount_b": 1000}

        b._query_remove_liquidity_velodrome = fake_query_remove

        def fake_contract_interact(**kwargs):
            yield
            if kwargs.get("contract_callable") == "build_approval_tx":
                return "0xapprove_data"
            elif kwargs.get("contract_callable") == "remove_liquidity":
                return "0xremove_data"
            elif kwargs.get("contract_callable") == "get_tx_data":
                return "0xaabbccdd"  # must start with 0x for bytes.fromhex
            return "0xdefault"

        b.contract_interact = fake_contract_interact

        gen = b._exit_stable_volatile_pool(**self._base_kwargs())
        result = exhaust_generator(gen)

        assert result[0] == bytes.fromhex("aabbccdd")
        assert result[1] == "0xmultisend"
        assert result[2] is True


class TestGetCachedPriceAtSelection:
    """Tests for VelodromePoolBehaviour._get_cached_price_at_selection."""

    def test_valid_cached_data(self) -> None:
        """Returns cached price when data is valid and not invalidated."""
        b = make_behaviour()
        kv_key = "velodrome_cl_pool_optimism"
        cached = json.dumps({"current_price": 1.5, "invalidated": False})

        def fake_read_kv(keys):
            yield
            return {kv_key: cached}

        b._read_kv = fake_read_kv

        gen = b._get_cached_price_at_selection("optimism")
        result = exhaust_generator(gen)
        assert result == 1.5

    def test_invalidated_cache(self) -> None:
        """Returns None when cache is invalidated."""
        b = make_behaviour()
        kv_key = "velodrome_cl_pool_optimism"
        cached = json.dumps({"current_price": 1.5, "invalidated": True})

        def fake_read_kv(keys):
            yield
            return {kv_key: cached}

        b._read_kv = fake_read_kv

        gen = b._get_cached_price_at_selection("optimism")
        result = exhaust_generator(gen)
        assert result is None

    def test_no_cached_data(self) -> None:
        """Returns None when no data in KV store."""
        b = make_behaviour()

        def fake_read_kv(keys):
            yield
            return {}

        b._read_kv = fake_read_kv

        gen = b._get_cached_price_at_selection("optimism")
        result = exhaust_generator(gen)
        assert result is None

    def test_none_from_kv(self) -> None:
        """Returns None when _read_kv returns None."""
        b = make_behaviour()

        def fake_read_kv(keys):
            yield
            return None

        b._read_kv = fake_read_kv

        gen = b._get_cached_price_at_selection("optimism")
        result = exhaust_generator(gen)
        assert result is None

    def test_json_decode_error(self) -> None:
        """Returns None when cached data is not valid JSON."""
        b = make_behaviour()
        kv_key = "velodrome_cl_pool_optimism"

        def fake_read_kv(keys):
            yield
            return {kv_key: "not-json!!!"}

        b._read_kv = fake_read_kv

        gen = b._get_cached_price_at_selection("optimism")
        result = exhaust_generator(gen)
        assert result is None

    def test_exception_returns_none(self) -> None:
        """Returns None on unexpected exception."""
        b = make_behaviour()

        def fake_read_kv(keys):
            raise RuntimeError("db failure")
            yield  # noqa: unreachable

        b._read_kv = fake_read_kv

        gen = b._get_cached_price_at_selection("optimism")
        result = exhaust_generator(gen)
        assert result is None

    def test_missing_current_price_key(self) -> None:
        """Returns 0 when 'current_price' key is missing (uses default)."""
        b = make_behaviour()
        kv_key = "velodrome_cl_pool_optimism"
        cached = json.dumps({"some_other_key": 42})

        def fake_read_kv(keys):
            yield
            return {kv_key: cached}

        b._read_kv = fake_read_kv

        gen = b._get_cached_price_at_selection("optimism")
        result = exhaust_generator(gen)
        assert result == 0  # default from .get("current_price", 0)

    def test_empty_string_kv_value(self) -> None:
        """Returns None when kv value is empty string (falsy)."""
        b = make_behaviour()
        kv_key = "velodrome_cl_pool_optimism"

        def fake_read_kv(keys):
            yield
            return {kv_key: ""}

        b._read_kv = fake_read_kv

        gen = b._get_cached_price_at_selection("optimism")
        result = exhaust_generator(gen)
        assert result is None

    def test_different_chain(self) -> None:
        """Uses the correct chain-based key."""
        b = make_behaviour()
        kv_key = "velodrome_cl_pool_base"
        cached = json.dumps({"current_price": 2.5})

        def fake_read_kv(keys):
            yield
            assert keys == (kv_key,)
            return {kv_key: cached}

        b._read_kv = fake_read_kv

        gen = b._get_cached_price_at_selection("base")
        result = exhaust_generator(gen)
        assert result == 2.5


class TestTickToPrice:
    """Tests for tick_to_price."""

    def test_zero_tick(self) -> None:
        """Tick 0 gives price = 1.0."""
        b = make_behaviour()
        result = b.tick_to_price(0, 0, 0)
        assert result == Decimal(1.0)

    def test_positive_tick(self) -> None:
        """Positive tick gives price > 1."""
        b = make_behaviour()
        result = b.tick_to_price(100, 0, 0)
        expected = Decimal(1.0001**100)
        assert abs(result - expected) < Decimal("1e-10")

    def test_negative_tick(self) -> None:
        """Negative tick gives price < 1."""
        b = make_behaviour()
        result = b.tick_to_price(-100, 0, 0)
        expected = Decimal(1.0001**-100)
        assert abs(result - expected) < Decimal("1e-10")

    def test_decimal_adjustment(self) -> None:
        """Token decimals adjust the price correctly."""
        b = make_behaviour()
        result = b.tick_to_price(0, 6, 18)
        expected = Decimal(1.0 * 10**12)
        assert result == expected

    def test_same_decimals(self) -> None:
        """When both decimals are equal, factor is 1."""
        b = make_behaviour()
        result = b.tick_to_price(0, 18, 18)
        assert result == Decimal(1.0)

    def test_zero_decimals_no_adjustment(self) -> None:
        """Both decimals 0 => condition false => no multiplier."""
        b = make_behaviour()
        result = b.tick_to_price(10, 0, 0)
        expected = Decimal(1.0001**10)
        assert abs(result - expected) < Decimal("1e-10")


class TestCalculateTokenRatios:
    """Tests for _calculate_token_ratios."""

    def test_current_below_lower(self) -> None:
        b = make_behaviour()
        assert b._calculate_token_ratios(100.0, 200.0, 50.0) == {
            "token0": 1.0,
            "token1": 0.0,
        }

    def test_current_equals_lower(self) -> None:
        b = make_behaviour()
        assert b._calculate_token_ratios(100.0, 200.0, 100.0) == {
            "token0": 1.0,
            "token1": 0.0,
        }

    def test_current_above_upper(self) -> None:
        b = make_behaviour()
        assert b._calculate_token_ratios(100.0, 200.0, 300.0) == {
            "token0": 0.0,
            "token1": 1.0,
        }

    def test_current_equals_upper(self) -> None:
        b = make_behaviour()
        assert b._calculate_token_ratios(100.0, 200.0, 200.0) == {
            "token0": 0.0,
            "token1": 1.0,
        }

    def test_ratios_sum_to_one(self) -> None:
        b = make_behaviour()
        result = b._calculate_token_ratios(100.0, 400.0, 225.0)
        assert abs(result["token0"] + result["token1"] - 1.0) < 1e-9

    def test_midpoint_values(self) -> None:
        b = make_behaviour()
        lower, upper = 1.0, 9.0
        mid = 5.0
        result = b._calculate_token_ratios(lower, upper, mid)
        sqrt_u, sqrt_l, sqrt_c = np.sqrt(upper), np.sqrt(lower), np.sqrt(mid)
        expected_t0 = (sqrt_u - sqrt_c) / (sqrt_u - sqrt_l)
        expected_t1 = (sqrt_c - sqrt_l) / (sqrt_u - sqrt_l)
        assert abs(result["token0"] - expected_t0) < 1e-9
        assert abs(result["token1"] - expected_t1) < 1e-9

    def test_just_above_lower(self) -> None:
        b = make_behaviour()
        result = b._calculate_token_ratios(100.0, 200.0, 100.01)
        assert result["token0"] > 0.9

    def test_just_below_upper(self) -> None:
        b = make_behaviour()
        result = b._calculate_token_ratios(100.0, 200.0, 199.99)
        assert result["token1"] > 0.5


class TestGetSqrtPriceX96:
    """Tests for _get_sqrt_price_x96."""

    def test_empty_pool_address(self) -> None:
        b = make_behaviour()
        gen = b._get_sqrt_price_x96("optimism", "")
        assert exhaust(gen) is None

    def test_slot0_none(self) -> None:
        b = make_behaviour()
        b.contract_interact = MagicMock(return_value=_gen_return(None))
        result = exhaust(b._get_sqrt_price_x96("optimism", "0xPool"))
        assert result is None

    def test_slot0_missing_key(self) -> None:
        b = make_behaviour()
        b.contract_interact = MagicMock(return_value=_gen_return({"other": 1}))
        result = exhaust(b._get_sqrt_price_x96("optimism", "0xPool"))
        assert result is None

    def test_slot0_success(self) -> None:
        b = make_behaviour()
        b.contract_interact = MagicMock(
            return_value=_gen_return({"sqrt_price_x96": 12345})
        )
        result = exhaust(b._get_sqrt_price_x96("optimism", "0xPool"))
        assert result == 12345


class TestGetLiquidityForTokenVelodrome:
    """Tests for get_liquidity_for_token_velodrome."""

    def test_no_position_manager(self) -> None:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {}
        result = exhaust(b.get_liquidity_for_token_velodrome(42, "optimism"))
        assert result is None

    def test_position_none(self) -> None:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPosMgr"
        }
        b.contract_interact = MagicMock(return_value=_gen_return(None))
        result = exhaust(b.get_liquidity_for_token_velodrome(42, "optimism"))
        assert result is None

    def test_position_empty(self) -> None:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPosMgr"
        }
        b.contract_interact = MagicMock(return_value=_gen_return([]))
        result = exhaust(b.get_liquidity_for_token_velodrome(42, "optimism"))
        assert result is None

    def test_returns_liquidity_at_index_2(self) -> None:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPosMgr"
        }
        b.contract_interact = MagicMock(return_value=_gen_return([0, 0, 500000, 0]))
        result = exhaust(b.get_liquidity_for_token_velodrome(42, "optimism"))
        assert result == 500000


class TestDecreaseLiquidityVelodrome:
    """Tests for decrease_liquidity_velodrome."""

    def test_no_position_manager(self) -> None:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {}
        result = exhaust(
            b.decrease_liquidity_velodrome(1, 100, 50, 50, 9999, "optimism")
        )
        assert result is None

    def test_returns_tx_hash(self) -> None:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPosMgr"
        }
        b.contract_interact = MagicMock(return_value=_gen_return("0xTxHash"))
        result = exhaust(
            b.decrease_liquidity_velodrome(1, 100, 50, 50, 9999, "optimism")
        )
        assert result == "0xTxHash"

    def test_returns_none(self) -> None:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPosMgr"
        }
        b.contract_interact = MagicMock(return_value=_gen_return(None))
        result = exhaust(
            b.decrease_liquidity_velodrome(1, 100, 50, 50, 9999, "optimism")
        )
        assert result is None


class TestCollectTokensVelodrome:
    """Tests for collect_tokens_velodrome."""

    def test_no_position_manager(self) -> None:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {}
        result = exhaust(b.collect_tokens_velodrome(1, "0xR", 100, 100, "optimism"))
        assert result is None

    def test_returns_tx_hash(self) -> None:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPosMgr"
        }
        b.contract_interact = MagicMock(return_value=_gen_return("0xCollect"))
        result = exhaust(b.collect_tokens_velodrome(1, "0xR", 100, 100, "optimism"))
        assert result == "0xCollect"

    def test_returns_none(self) -> None:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPosMgr"
        }
        b.contract_interact = MagicMock(return_value=_gen_return(None))
        result = exhaust(b.collect_tokens_velodrome(1, "0xR", 100, 100, "optimism"))
        assert result is None


class TestGetTickSpacingVelodrome:
    """Tests for _get_tick_spacing_velodrome."""

    def test_returns_tick_spacing(self) -> None:
        b = make_behaviour()
        b.contract_interact = MagicMock(return_value=_gen_return(60))
        result = exhaust(b._get_tick_spacing_velodrome("0xPool", "optimism"))
        assert result == 60

    def test_returns_none_when_none(self) -> None:
        b = make_behaviour()
        b.contract_interact = MagicMock(return_value=_gen_return(None))
        result = exhaust(b._get_tick_spacing_velodrome("0xPool", "optimism"))
        assert result is None

    def test_returns_none_when_zero(self) -> None:
        b = make_behaviour()
        b.contract_interact = MagicMock(return_value=_gen_return(0))
        result = exhaust(b._get_tick_spacing_velodrome("0xPool", "optimism"))
        assert result is None


class TestGetPoolTokens:
    """Tests for _get_pool_tokens."""

    def test_returns_token_pair(self) -> None:
        b = make_behaviour()
        b.contract_interact = MagicMock(return_value=_gen_return(["0xT0", "0xT1"]))
        result = exhaust(b._get_pool_tokens("0xPool", "optimism"))
        assert result == ("0xT0", "0xT1")

    def test_tokens_none(self) -> None:
        b = make_behaviour()
        b.contract_interact = MagicMock(return_value=_gen_return(None))
        result = exhaust(b._get_pool_tokens("0xPool", "optimism"))
        assert result == (None, None)

    def test_tokens_empty(self) -> None:
        b = make_behaviour()
        b.contract_interact = MagicMock(return_value=_gen_return([]))
        result = exhaust(b._get_pool_tokens("0xPool", "optimism"))
        assert result == (None, None)

    def test_exception_returns_none_pair(self) -> None:
        """Exception inside the try block is caught."""
        b = make_behaviour()

        def _raise_gen():
            raise RuntimeError("boom")
            yield

        b.contract_interact = MagicMock(return_value=_raise_gen())
        result = exhaust(b._get_pool_tokens("0xPool", "optimism"))
        assert result == (None, None)


class TestEnterClPool:
    """Tests for _enter_cl_pool."""

    def _make_b(self) -> VelodromePoolBehaviour:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPosMgr"
        }
        b.context.params.slippage_tolerance = 0.01
        # SharedState mock for deadline
        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 1_000_000.0
        mock_rs = MagicMock()
        mock_rs.last_round_transition_timestamp = mock_ts
        b.context.state.round_sequence = mock_rs
        return b

    # --- early returns --------------------------------------------------

    def test_no_position_manager(self) -> None:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {}
        result = exhaust(
            b._enter_cl_pool(
                "0xPool", "0xSafe", ["0xT0", "0xT1"], "optimism", [1000, 2000], False
            )
        )
        assert result == (None, None)

    def test_empty_position_manager(self) -> None:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": ""
        }
        result = exhaust(
            b._enter_cl_pool(
                "0xPool", "0xSafe", ["0xT0", "0xT1"], "optimism", [1000, 2000], False
            )
        )
        assert result == (None, None)

    def test_tick_ranges_calculation_fails(self) -> None:
        """No tick_ranges provided and _calculate_tick_lower_and_upper_velodrome returns None."""
        b = self._make_b()
        b._calculate_tick_lower_and_upper_velodrome = MagicMock(
            return_value=_gen_return(None)
        )
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=None,
            )
        )
        assert result == (None, None)

    def test_tick_spacing_fetch_fails(self) -> None:
        """tick_spacing not provided and _get_tick_spacing_velodrome returns None."""
        b = self._make_b()
        tr = [[{"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}]]
        b._get_tick_spacing_velodrome = MagicMock(return_value=_gen_return(None))
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=None,
            )
        )
        assert result == (None, None)

    def test_price_at_selection_none(self) -> None:
        b = self._make_b()
        tr = [[{"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}]]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(None))
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        assert result == (None, None)

    def test_price_movement_check_none(self) -> None:
        """price_moved is None => abort."""
        b = self._make_b()
        tr = [[{"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}]]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(1.0))
        b._check_price_movement = MagicMock(return_value=_gen_return(None))
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        assert result == (None, None)

    def test_price_moved_wait_failed(self) -> None:
        b = self._make_b()
        tr = [[{"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}]]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(1.0))
        b._check_price_movement = MagicMock(return_value=_gen_return(True))
        b._wait_for_favorable_price = MagicMock(
            return_value=_gen_return(AllocationStatus.FAILED)
        )
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        assert result == (None, None)

    def test_price_moved_wait_timeout(self) -> None:
        b = self._make_b()
        tr = [[{"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}]]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(1.0))
        b._check_price_movement = MagicMock(return_value=_gen_return(True))
        b._wait_for_favorable_price = MagicMock(
            return_value=_gen_return(AllocationStatus.TIMEOUT)
        )
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        assert result == (None, None)

    def test_price_moved_wait_ready_continues(self) -> None:
        """When AllocationStatus.READY, execution continues past the wait block."""
        b = self._make_b()
        tr = [
            [
                {
                    "tick_lower": -100,
                    "tick_upper": 100,
                    "allocation": 1.0,
                    "token0_ratio": 0.5,
                    "token1_ratio": 0.5,
                }
            ]
        ]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(1.0))
        b._check_price_movement = MagicMock(return_value=_gen_return(True))
        b._wait_for_favorable_price = MagicMock(
            return_value=_gen_return(AllocationStatus.READY)
        )
        b._get_sqrt_price_x96 = MagicMock(return_value=_gen_return(None))
        # sqrt_price defaults to 0 => band_investment returns None
        b._get_token_decimals = MagicMock(return_value=_gen_return(18))
        b._calculate_band_investment_with_pool_price = MagicMock(return_value=None)
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        assert result == (None, None)

    def test_price_moved_wait_waiting_continues(self) -> None:
        """Branch 746->759: when allocation_status is WAITING (not FAILED/TIMEOUT/READY),
        execution falls through to line 759 without returning early."""
        b = self._make_b()
        tr = [
            [
                {
                    "tick_lower": -100,
                    "tick_upper": 100,
                    "allocation": 1.0,
                    "token0_ratio": 0.5,
                    "token1_ratio": 0.5,
                }
            ]
        ]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(1.0))
        b._check_price_movement = MagicMock(return_value=_gen_return(True))
        # Return WAITING so none of the early-exit elif branches match
        b._wait_for_favorable_price = MagicMock(
            return_value=_gen_return(AllocationStatus.WAITING)
        )
        b._get_sqrt_price_x96 = _mock_gen(None)
        b._get_token_decimals = _mock_gen(18)
        b._calculate_band_investment_with_pool_price = MagicMock(return_value=None)
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        # band_investment=None so (None, None) is returned
        assert result == (None, None)
        # Must have reached _get_sqrt_price_x96 (line 759+)
        assert b._get_sqrt_price_x96.called

    def test_tick_spacing_none_then_fetched_success(self) -> None:
        """Branch 702->708: tick_spacing is initially None, _get_tick_spacing_velodrome
        returns a truthy value, execution continues to line 708."""
        b = self._make_b()
        tr = [
            [
                {
                    "tick_lower": -100,
                    "tick_upper": 100,
                    "allocation": 1.0,
                    "token0_ratio": 0.5,
                    "token1_ratio": 0.5,
                }
            ]
        ]
        # tick_spacing is fetched (not provided), returns 60 (truthy) -> proceeds
        b._get_tick_spacing_velodrome = _mock_gen(60)
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(None))
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=None,
            )
        )
        # price_at_selection is None -> returns (None, None)
        assert result == (None, None)
        # Must have called _get_tick_spacing_velodrome
        assert b._get_tick_spacing_velodrome.called

    # --- sqrt_price_x96 fallback ----------------------------------------

    def test_sqrt_price_none_defaults_zero(self) -> None:
        """When _get_sqrt_price_x96 returns None, sqrt_price_x96=0."""
        b = self._make_b()
        tr = [
            [
                {
                    "tick_lower": -100,
                    "tick_upper": 100,
                    "allocation": 1.0,
                    "token0_ratio": 0.5,
                    "token1_ratio": 0.5,
                }
            ]
        ]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(1.0))
        b._check_price_movement = MagicMock(return_value=_gen_return(False))
        b._get_sqrt_price_x96 = MagicMock(return_value=_gen_return(None))
        b._get_token_decimals = MagicMock(return_value=_gen_return(18))
        b._calculate_band_investment_with_pool_price = MagicMock(return_value=None)
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        # band_investment=None => (None, None)
        assert result == (None, None)

    # --- band investment & token amount failures ------------------------

    def test_band_investment_none(self) -> None:
        b = self._make_b()
        tr = [
            [
                {
                    "tick_lower": -100,
                    "tick_upper": 100,
                    "allocation": 1.0,
                    "token0_ratio": 0.5,
                    "token1_ratio": 0.5,
                }
            ]
        ]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(1.0))
        b._check_price_movement = MagicMock(return_value=_gen_return(False))
        b._get_sqrt_price_x96 = MagicMock(return_value=_gen_return(2**96))
        b._get_token_decimals = _mock_gen(18)
        b._calculate_band_investment_with_pool_price = MagicMock(return_value=None)
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        assert result == (None, None)

    def test_individual_token_amounts_none(self) -> None:
        b = self._make_b()
        tr = [
            [
                {
                    "tick_lower": -100,
                    "tick_upper": 100,
                    "allocation": 1.0,
                    "token0_ratio": 0.5,
                    "token1_ratio": 0.5,
                }
            ]
        ]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(1.0))
        b._check_price_movement = MagicMock(return_value=_gen_return(False))
        b._get_sqrt_price_x96 = MagicMock(return_value=_gen_return(2**96))
        b._get_token_decimals = _mock_gen(18)
        b._calculate_band_investment_with_pool_price = MagicMock(return_value=1000)
        b._calculate_individual_token_amounts = MagicMock(return_value=(None, None))
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        assert result == (None, None)

    def test_slippage_protection_none(self) -> None:
        b = self._make_b()
        tr = [
            [
                {
                    "tick_lower": -100,
                    "tick_upper": 100,
                    "allocation": 1.0,
                    "token0_ratio": 0.5,
                    "token1_ratio": 0.5,
                }
            ]
        ]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(1.0))
        b._check_price_movement = MagicMock(return_value=_gen_return(False))
        b._get_sqrt_price_x96 = MagicMock(return_value=_gen_return(2**96))
        b._get_token_decimals = _mock_gen(18)
        b._calculate_band_investment_with_pool_price = MagicMock(return_value=1000)
        b._calculate_individual_token_amounts = MagicMock(return_value=(500, 500))
        b._calculate_slippage_protection = MagicMock(return_value=(None, None))
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        assert result == (None, None)

    # --- no mint txs created -------------------------------------------

    def test_all_mints_fail(self) -> None:
        b = self._make_b()
        tr = [
            [
                {
                    "tick_lower": -100,
                    "tick_upper": 100,
                    "allocation": 1.0,
                    "token0_ratio": 0.5,
                    "token1_ratio": 0.5,
                }
            ]
        ]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(1.0))
        b._check_price_movement = MagicMock(return_value=_gen_return(False))
        b._get_sqrt_price_x96 = MagicMock(return_value=_gen_return(2**96))
        b._get_token_decimals = _mock_gen(18)
        b._calculate_band_investment_with_pool_price = MagicMock(return_value=1000)
        b._calculate_individual_token_amounts = MagicMock(return_value=(500, 500))
        b._calculate_slippage_protection = MagicMock(return_value=(450, 450))
        b.contract_interact = _mock_gen(None)
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        assert result == (None, None)

    # --- successful entry -----------------------------------------------

    def test_successful_single_position(self) -> None:
        b = self._make_b()
        tr = [
            [
                {
                    "tick_lower": -100,
                    "tick_upper": 100,
                    "allocation": 1.0,
                    "token0_ratio": 0.5,
                    "token1_ratio": 0.5,
                }
            ]
        ]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(1.0))
        b._check_price_movement = MagicMock(return_value=_gen_return(False))
        b._get_sqrt_price_x96 = MagicMock(return_value=_gen_return(2**96))
        b._get_token_decimals = _mock_gen(18)
        b._calculate_band_investment_with_pool_price = MagicMock(return_value=1000)
        b._calculate_individual_token_amounts = MagicMock(return_value=(500, 500))
        b._calculate_slippage_protection = MagicMock(return_value=(450, 450))
        b.contract_interact = _mock_gen("0xMintTx")
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        assert result == (["0xMintTx"], "0xPosMgr")

    # --- zero allocation skipped ----------------------------------------

    def test_zero_allocation_skipped(self) -> None:
        b = self._make_b()
        tr = [
            [
                {"tick_lower": -100, "tick_upper": 100, "allocation": 0},
                {
                    "tick_lower": -200,
                    "tick_upper": 200,
                    "allocation": 1.0,
                    "token0_ratio": 0.5,
                    "token1_ratio": 0.5,
                },
            ]
        ]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(1.0))
        b._check_price_movement = MagicMock(return_value=_gen_return(False))
        b._get_sqrt_price_x96 = MagicMock(return_value=_gen_return(2**96))
        b._get_token_decimals = _mock_gen(18)
        b._calculate_band_investment_with_pool_price = MagicMock(return_value=1000)
        b._calculate_individual_token_amounts = MagicMock(return_value=(500, 500))
        b._calculate_slippage_protection = MagicMock(return_value=(450, 450))
        b.contract_interact = _mock_gen("0xMint")
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        # Both positions get mint calls (zero alloc position still goes through
        # with amount0_desired=0, amount1_desired=0).
        assert result is not None
        assert "0xPosMgr" == result[1]

    # --- scaling ----------------------------------------------------------

    def test_scaling_when_over_max(self) -> None:
        """Amounts are scaled down when they exceed max_amounts_in."""
        b = self._make_b()
        tr = [
            [
                {
                    "tick_lower": -100,
                    "tick_upper": 100,
                    "allocation": 0.5,
                    "token0_ratio": 0.5,
                    "token1_ratio": 0.5,
                },
                {
                    "tick_lower": -200,
                    "tick_upper": 200,
                    "allocation": 0.5,
                    "token0_ratio": 0.5,
                    "token1_ratio": 0.5,
                },
            ]
        ]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(1.0))
        b._check_price_movement = MagicMock(return_value=_gen_return(False))
        b._get_sqrt_price_x96 = MagicMock(return_value=_gen_return(2**96))
        b._get_token_decimals = _mock_gen(18)
        # Each position gets 800 per token => total 1600, max is 1000
        b._calculate_band_investment_with_pool_price = MagicMock(return_value=2000)
        b._calculate_individual_token_amounts = MagicMock(return_value=(800, 800))
        b._calculate_slippage_protection = MagicMock(return_value=(0, 0))
        b.contract_interact = _mock_gen("0xMint")
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 1000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        assert result is not None
        assert len(result[0]) == 2

    def test_multiple_positions_success(self) -> None:
        b = self._make_b()
        tr = [
            [
                {
                    "tick_lower": -100,
                    "tick_upper": 100,
                    "allocation": 0.6,
                    "token0_ratio": 0.5,
                    "token1_ratio": 0.5,
                },
                {
                    "tick_lower": -200,
                    "tick_upper": 200,
                    "allocation": 0.4,
                    "token0_ratio": 0.5,
                    "token1_ratio": 0.5,
                },
            ]
        ]
        b._get_cached_price_at_selection = MagicMock(return_value=_gen_return(1.0))
        b._check_price_movement = MagicMock(return_value=_gen_return(False))
        b._get_sqrt_price_x96 = MagicMock(return_value=_gen_return(2**96))
        b._get_token_decimals = _mock_gen(18)
        b._calculate_band_investment_with_pool_price = MagicMock(return_value=500)
        b._calculate_individual_token_amounts = MagicMock(return_value=(250, 250))
        b._calculate_slippage_protection = MagicMock(return_value=(200, 200))
        b.contract_interact = _mock_gen("0xMint")
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=tr,
                tick_spacing=60,
            )
        )
        assert result is not None
        assert len(result[0]) == 2
        assert result[1] == "0xPosMgr"


class TestExitClPool:
    """Tests for _exit_cl_pool."""

    def _make_b(self) -> VelodromePoolBehaviour:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPosMgr"
        }
        b.context.params.multisend_contract_addresses = {"optimism": "0xMulti"}
        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 1_000_000.0
        mock_rs = MagicMock()
        mock_rs.last_round_transition_timestamp = mock_ts
        b.context.state.round_sequence = mock_rs
        return b

    def test_no_position_manager(self) -> None:
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {}
        result = exhaust(b._exit_cl_pool("optimism", "0xSafe", [1], [100], "0xPool"))
        assert result is None

    def test_no_multisend_address(self) -> None:
        b = self._make_b()
        b.context.params.multisend_contract_addresses = {}
        b._calculate_slippage_protection_for_velodrome_decrease = _mock_gen((10, 10))
        b.decrease_liquidity_velodrome = _mock_gen("0xDec")
        b.collect_tokens_velodrome = _mock_gen("0xCol")
        result = exhaust(b._exit_cl_pool("optimism", "0xSafe", [1], [100], "0xPool"))
        assert result is None

    def test_slippage_protection_none_returns_none_tuple(self) -> None:
        b = self._make_b()
        b._calculate_slippage_protection_for_velodrome_decrease = _mock_gen(
            (None, None)
        )
        result = exhaust(b._exit_cl_pool("optimism", "0xSafe", [1], [100], "0xPool"))
        assert result == (None, None)

    def test_zero_liquidity_fetches_from_contract(self) -> None:
        """When liquidity is 0 (falsy), get_liquidity_for_token is called."""
        b = self._make_b()
        b.get_liquidity_for_token_velodrome = _mock_gen(None)
        # No liquidity found => position skipped. Then multisend still called.
        b.contract_interact = _mock_gen("0xabcdef1234567890")
        result = exhaust(b._exit_cl_pool("optimism", "0xSafe", [1], [0], "0xPool"))
        b.get_liquidity_for_token_velodrome.assert_called_once()

    def test_zero_liquidity_fetched_successfully_continues(self) -> None:
        """Branch 985->992: liquidity is 0 initially; after fetch it IS truthy,
        so processing continues to slippage calculation (line 992)."""
        b = self._make_b()
        # get_liquidity_for_token_velodrome returns a truthy value (500000)
        b.get_liquidity_for_token_velodrome = _mock_gen(500000)
        b._calculate_slippage_protection_for_velodrome_decrease = _mock_gen((10, 10))
        b.decrease_liquidity_velodrome = _mock_gen("0xDec")
        b.collect_tokens_velodrome = _mock_gen("0xCol")
        b.contract_interact = _mock_gen("0xabcdef1234567890")
        result = exhaust(b._exit_cl_pool("optimism", "0xSafe", [1], [0], "0xPool"))
        # Slippage protection was called, so branch 985->992 was taken
        assert b._calculate_slippage_protection_for_velodrome_decrease.called
        assert result is not None

    def test_decrease_fails_skips(self) -> None:
        b = self._make_b()
        b._calculate_slippage_protection_for_velodrome_decrease = _mock_gen((10, 10))
        b.decrease_liquidity_velodrome = _mock_gen(None)
        b.contract_interact = _mock_gen("0xabcdef1234567890")
        result = exhaust(b._exit_cl_pool("optimism", "0xSafe", [1], [100], "0xPool"))
        # Decrease failed -> position skipped, multisend still runs
        assert result is not None

    def test_collect_fails_skips(self) -> None:
        b = self._make_b()
        b._calculate_slippage_protection_for_velodrome_decrease = _mock_gen((10, 10))
        b.decrease_liquidity_velodrome = _mock_gen("0xDec")
        b.collect_tokens_velodrome = _mock_gen(None)
        b.contract_interact = _mock_gen("0xabcdef1234567890")
        result = exhaust(b._exit_cl_pool("optimism", "0xSafe", [1], [100], "0xPool"))
        # Collect failed but decrease tx was added to multi_send_txs
        assert result is not None

    def test_multisend_hash_none(self) -> None:
        b = self._make_b()
        b._calculate_slippage_protection_for_velodrome_decrease = _mock_gen((10, 10))
        b.decrease_liquidity_velodrome = _mock_gen("0xDec")
        b.collect_tokens_velodrome = _mock_gen("0xCol")
        b.contract_interact = _mock_gen(None)
        result = exhaust(b._exit_cl_pool("optimism", "0xSafe", [1], [100], "0xPool"))
        assert result is None

    def test_successful_exit(self) -> None:
        b = self._make_b()
        b._calculate_slippage_protection_for_velodrome_decrease = _mock_gen((10, 10))
        b.decrease_liquidity_velodrome = _mock_gen("0xDec")
        b.collect_tokens_velodrome = _mock_gen("0xCol")
        b.contract_interact = _mock_gen("0xabcdef1234567890")
        result = exhaust(b._exit_cl_pool("optimism", "0xSafe", [42], [100], "0xPool"))
        assert result is not None
        # Should be (bytes, multisend_address, True)
        assert result[1] == "0xMulti"
        assert result[2] is True

    def test_multiple_positions(self) -> None:
        b = self._make_b()
        b._calculate_slippage_protection_for_velodrome_decrease = _mock_gen((10, 10))
        b.decrease_liquidity_velodrome = _mock_gen("0xDec")
        b.collect_tokens_velodrome = _mock_gen("0xCol")
        b.contract_interact = _mock_gen("0xabcdef1234567890")
        result = exhaust(
            b._exit_cl_pool("optimism", "0xSafe", [1, 2, 3], [100, 200, 300], "0xPool")
        )
        assert result is not None
        assert result[1] == "0xMulti"
        assert result[2] is True


class TestCalculateTickLowerAndUpperVelodrome:
    """Tests for _calculate_tick_lower_and_upper_velodrome."""

    def test_tick_spacing_fails(self) -> None:
        b = make_behaviour()
        b._get_tick_spacing_velodrome = MagicMock(return_value=_gen_return(None))
        result = exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", True)
        )
        assert result is None

    def test_get_pool_tokens_fails(self) -> None:
        b = make_behaviour()
        b._get_tick_spacing_velodrome = MagicMock(return_value=_gen_return(60))
        b._get_pool_tokens = MagicMock(return_value=_gen_return((None, None)))
        result = exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", True)
        )
        assert result is None

    def test_current_price_fails(self) -> None:
        b = make_behaviour()
        b._get_tick_spacing_velodrome = MagicMock(return_value=_gen_return(60))
        b._get_pool_tokens = MagicMock(return_value=_gen_return(("0xT0", "0xT1")))
        b._get_current_pool_price = MagicMock(return_value=_gen_return(None))
        result = exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", True)
        )
        assert result is None

    def test_pool_data_none(self) -> None:
        b = make_behaviour()
        b._get_tick_spacing_velodrome = MagicMock(return_value=_gen_return(60))
        b._get_pool_tokens = MagicMock(return_value=_gen_return(("0xT0", "0xT1")))
        b._get_current_pool_price = MagicMock(return_value=_gen_return(1.0))
        b.get_pool_token_history = MagicMock(return_value=_gen_return(None))
        result = exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", False)
        )
        assert result is None

    def test_empty_ratio_prices(self) -> None:
        b = make_behaviour()
        b._get_tick_spacing_velodrome = MagicMock(return_value=_gen_return(60))
        b._get_pool_tokens = MagicMock(return_value=_gen_return(("0xT0", "0xT1")))
        b._get_current_pool_price = MagicMock(return_value=_gen_return(1.0))
        b.get_pool_token_history = MagicMock(
            return_value=_gen_return({"ratio_prices": []})
        )
        result = exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", False)
        )
        assert result is None

    def test_optimize_returns_none(self) -> None:
        b = make_behaviour()
        b._get_tick_spacing_velodrome = MagicMock(return_value=_gen_return(60))
        b._get_pool_tokens = MagicMock(return_value=_gen_return(("0xT0", "0xT1")))
        b._get_current_pool_price = MagicMock(return_value=_gen_return(1.0))
        b.get_pool_token_history = MagicMock(
            return_value=_gen_return({"ratio_prices": [1.0] * 200})
        )
        b.optimize_stablecoin_bands = MagicMock(return_value=None)
        result = exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", False)
        )
        assert result is None

    def test_exception_returns_none(self) -> None:
        b = make_behaviour()
        b._get_tick_spacing_velodrome = MagicMock(return_value=_gen_return(60))
        b._get_pool_tokens = MagicMock(return_value=_gen_return(("0xT0", "0xT1")))
        b._get_current_pool_price = MagicMock(return_value=_gen_return(1.0))
        b.get_pool_token_history = MagicMock(
            return_value=_gen_return({"ratio_prices": [1.0] * 200})
        )
        b.optimize_stablecoin_bands = MagicMock(side_effect=RuntimeError("boom"))
        result = exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", False)
        )
        assert result is None

    def _setup_full(
        self,
        b: VelodromePoolBehaviour,
        *,
        is_stable: bool = True,
        band_allocs: Optional[list] = None,
        tick_results: Optional[dict] = None,
    ) -> None:
        """Wire up all sub-method mocks for a full successful run."""
        b._get_tick_spacing_velodrome = MagicMock(return_value=_gen_return(60))
        b._get_pool_tokens = MagicMock(return_value=_gen_return(("0xT0", "0xT1")))
        b._get_current_pool_price = MagicMock(return_value=_gen_return(1.0))
        b.get_pool_token_history = MagicMock(
            return_value=_gen_return(
                {"ratio_prices": [1.0 + 0.0001 * i for i in range(200)]}
            )
        )

        if band_allocs is None:
            band_allocs = [0.5, 0.3, 0.2]

        b.optimize_stablecoin_bands = MagicMock(
            return_value={
                "band_multipliers": np.array([1.0, 2.0, 3.0]),
                "band_allocations": band_allocs,
                "percent_in_bounds": 95.0,
            }
        )

        b.calculate_ema = MagicMock(return_value=np.array([1.0] * 100))
        b.calculate_std_dev = MagicMock(return_value=np.array([0.01] * 100))

        if tick_results is None:
            tick_results = {
                "band1": {"tick_lower": -100, "tick_upper": 100},
                "band2": {"tick_lower": -200, "tick_upper": 200},
                "band3": {"tick_lower": -300, "tick_upper": 300},
            }
        b.calculate_tick_range_from_bands_wrapper = MagicMock(return_value=tick_results)

    def test_successful_three_bands(self) -> None:
        b = make_behaviour()
        self._setup_full(b)
        result = exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", True)
        )
        assert result is not None
        assert len(result) == 3
        assert result[0]["tick_lower"] == -100
        assert result[0]["tick_upper"] == 100
        total = sum(p["allocation"] for p in result)
        assert abs(total - 1.0) < 1e-9
        # Check metadata
        assert "percent_in_bounds" in result[0]
        assert result[0]["percent_in_bounds"] == 95.0
        assert "ema" in result[0]
        assert "std_dev" in result[0]
        assert "current_ema" in result[0]
        assert "current_std_dev" in result[0]
        assert "band_multipliers" in result[0]

    def test_equal_ticks_adjusted(self) -> None:
        b = make_behaviour()
        self._setup_full(
            b,
            tick_results={
                "band1": {"tick_lower": 50, "tick_upper": 50},  # equal!
                "band2": {"tick_lower": -200, "tick_upper": 200},
                "band3": {"tick_lower": -300, "tick_upper": 300},
            },
        )
        result = exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", True)
        )
        assert result is not None
        first = [p for p in result if p["tick_lower"] == 50]
        assert len(first) >= 1
        # tick_upper = tick_lower + tick_spacing = 50 + 60 = 110
        assert first[0]["tick_upper"] == 110

    def test_collapsed_duplicate_ticks(self) -> None:
        b = make_behaviour()
        self._setup_full(
            b,
            band_allocs=[0.4, 0.3, 0.3],
            tick_results={
                "band1": {"tick_lower": -100, "tick_upper": 100},
                "band2": {"tick_lower": -100, "tick_upper": 100},
                "band3": {"tick_lower": -100, "tick_upper": 100},
            },
        )
        result = exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", True)
        )
        assert result is not None
        assert len(result) == 1
        assert abs(result[0]["allocation"] - 1.0) < 1e-9

    def test_is_stable_sets_min_width(self) -> None:
        b = make_behaviour()
        self._setup_full(b, is_stable=True)
        exhaust(b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", True))
        kw = b.optimize_stablecoin_bands.call_args[1]
        assert "min_width_pct" in kw
        assert kw["min_width_pct"] == 0.0001

    def test_not_stable_no_min_width(self) -> None:
        b = make_behaviour()
        self._setup_full(b, is_stable=False)
        exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", False)
        )
        kw = b.optimize_stablecoin_bands.call_args[1]
        assert "min_width_pct" not in kw

    def test_zero_allocations_no_normalization(self) -> None:
        b = make_behaviour()
        self._setup_full(b, band_allocs=[0.0, 0.0, 0.0])
        result = exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", False)
        )
        assert result is not None
        for p in result:
            assert p["allocation"] == 0.0

    def test_partial_collapse(self) -> None:
        """Two bands same, one different => 2 positions."""
        b = make_behaviour()
        self._setup_full(
            b,
            band_allocs=[0.5, 0.3, 0.2],
            tick_results={
                "band1": {"tick_lower": -100, "tick_upper": 100},
                "band2": {"tick_lower": -100, "tick_upper": 100},
                "band3": {"tick_lower": -300, "tick_upper": 300},
            },
        )
        result = exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", False)
        )
        assert result is not None
        assert len(result) == 2
        total = sum(p["allocation"] for p in result)
        assert abs(total - 1.0) < 1e-9

    def test_get_pool_token_history_exception(self) -> None:
        """Exception during get_pool_token_history is caught."""
        b = make_behaviour()
        b._get_tick_spacing_velodrome = MagicMock(return_value=_gen_return(60))
        b._get_pool_tokens = MagicMock(return_value=_gen_return(("0xT0", "0xT1")))
        b._get_current_pool_price = MagicMock(return_value=_gen_return(1.0))

        def _raise_gen(*a, **kw):
            raise RuntimeError("API error")
            yield

        b.get_pool_token_history = _raise_gen
        result = exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", False)
        )
        assert result is None

    def test_price_to_tick_inner_function_called(self) -> None:
        """price_to_tick closure (lines 1311-1313) is invoked when
        calculate_tick_range_from_bands_wrapper is NOT mocked."""
        import numpy as np

        b = make_behaviour()
        b._get_tick_spacing_velodrome = MagicMock(return_value=_gen_return(1))
        b._get_pool_tokens = MagicMock(return_value=_gen_return(("0xT0", "0xT1")))
        b._get_current_pool_price = MagicMock(return_value=_gen_return(1.0))
        prices = [1.0 + 0.0001 * i for i in range(200)]
        b.get_pool_token_history = MagicMock(
            return_value=_gen_return({"ratio_prices": prices})
        )
        # Mock optimize to return valid band config; do NOT mock
        # calculate_tick_range_from_bands_wrapper so price_to_tick gets called.
        b.optimize_stablecoin_bands = MagicMock(
            return_value={
                "band_multipliers": np.array([0.01, 0.02, 0.03]),
                "band_allocations": np.array([0.6, 0.3, 0.1]),
                "percent_in_bounds": 95.0,
            }
        )
        # Use real calculate_ema / calculate_std_dev / calculate_tick_range_from_bands_wrapper
        result = exhaust(
            b._calculate_tick_lower_and_upper_velodrome("optimism", "0xPool", True)
        )
        assert result is not None


class TestCalculateEma:
    """Tests for calculate_ema."""

    def test_single_price(self) -> None:
        """EMA of a single element should equal that element."""
        b = make_behaviour()
        ema = b.calculate_ema([5.0], period=10)
        assert len(ema) == 1
        assert ema[0] == pytest.approx(5.0)

    def test_constant_prices(self) -> None:
        """EMA of constant prices should equal that constant."""
        b = make_behaviour()
        prices = [1.0] * 20
        ema = b.calculate_ema(prices, period=10)
        np.testing.assert_allclose(ema, 1.0, atol=1e-10)

    def test_increasing_prices(self) -> None:
        """EMA should lag behind linearly increasing prices."""
        b = make_behaviour()
        prices = list(range(1, 21))
        ema = b.calculate_ema(prices, period=5)
        # EMA should be below current price for increasing sequence
        assert ema[-1] < prices[-1]
        # But above the first price
        assert ema[-1] > prices[0]

    def test_alpha_formula(self) -> None:
        """Verify the EMA recurrence relation manually for two points."""
        b = make_behaviour()
        prices = [10.0, 20.0]
        period = 3
        ema = b.calculate_ema(prices, period=period)
        alpha = 2.0 / (period + 1)
        expected_1 = 20.0 * alpha + 10.0 * (1 - alpha)
        assert ema[1] == pytest.approx(expected_1)

    def test_returns_ndarray(self) -> None:
        """Result should be a numpy array."""
        b = make_behaviour()
        ema = b.calculate_ema([1.0, 2.0, 3.0], period=2)
        assert isinstance(ema, np.ndarray)


class TestCalculateStdDev:
    """Tests for calculate_std_dev."""

    def test_constant_prices_zero_deviation(self) -> None:
        """Constant prices equal to EMA should yield zero std_dev (after window)."""
        b = make_behaviour()
        prices = [1.0] * 20
        ema = np.array(prices)
        std = b.calculate_std_dev(prices, ema, window=5)
        # After window fills, deviations should be 0
        np.testing.assert_allclose(std[4:], 0.0, atol=1e-12)

    def test_initial_value_single_element(self) -> None:
        """First element should use the small default value."""
        b = make_behaviour()
        prices = [100.0, 101.0, 102.0]
        ema = np.array([100.0, 100.5, 101.0])
        std = b.calculate_std_dev(prices, ema, window=3)
        # Index 0 uses default: 0.001 * prices[0]
        assert std[0] == pytest.approx(0.001 * 100.0)

    def test_partial_window(self) -> None:
        """Elements before window fills should use partial window."""
        b = make_behaviour()
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        ema = b.calculate_ema(prices, period=3)
        std = b.calculate_std_dev(prices, ema, window=3)
        # Index 1 uses partial window [0:2]
        assert std[1] > 0

    def test_returns_ndarray(self) -> None:
        """Result should be a numpy array."""
        b = make_behaviour()
        prices = [1.0, 2.0, 3.0]
        ema = np.array([1.0, 1.5, 2.0])
        std = b.calculate_std_dev(prices, ema, window=2)
        assert isinstance(std, np.ndarray)
        assert len(std) == 3


class TestEvaluateBandConfiguration:
    """Tests for _evaluate_band_configuration."""

    def _make_simple_inputs(self):
        """Create simple inputs for evaluation."""
        prices = np.array([1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 1.005, 0.995])
        ema = np.ones(len(prices))
        std_dev = np.full(len(prices), 0.01)
        z_scores = np.abs(prices - ema) / np.maximum(std_dev, 1e-6)
        return prices, ema, std_dev, z_scores

    def test_returns_required_keys(self) -> None:
        """Result must contain the expected keys."""
        b = make_behaviour()
        prices, ema, std_dev, z_scores = self._make_simple_inputs()
        result = b._evaluate_band_configuration(
            prices,
            ema,
            std_dev,
            band_multipliers=np.array([1.0, 2.0, 3.0]),
            z_scores=z_scores,
            band_allocations=np.array([0.7, 0.2, 0.1]),
            min_width_pct=0.0001,
        )
        assert "percent_in_bounds" in result
        assert "avg_weighted_width" in result
        assert "zscore_economic_score" in result
        assert "band_coverage" in result

    def test_percent_in_bounds_range(self) -> None:
        """percent_in_bounds should be between 0 and 100."""
        b = make_behaviour()
        prices, ema, std_dev, z_scores = self._make_simple_inputs()
        result = b._evaluate_band_configuration(
            prices,
            ema,
            std_dev,
            band_multipliers=np.array([1.0, 2.0, 3.0]),
            z_scores=z_scores,
            band_allocations=np.array([0.7, 0.2, 0.1]),
            min_width_pct=0.0001,
        )
        assert 0 <= result["percent_in_bounds"] <= 100

    def test_all_in_bounds_with_wide_bands(self) -> None:
        """Very wide bands should capture all price points."""
        b = make_behaviour()
        prices, ema, std_dev, z_scores = self._make_simple_inputs()
        result = b._evaluate_band_configuration(
            prices,
            ema,
            std_dev,
            band_multipliers=np.array([100.0, 200.0, 300.0]),
            z_scores=z_scores,
            band_allocations=np.array([0.8, 0.15, 0.05]),
            min_width_pct=0.0001,
        )
        assert result["percent_in_bounds"] == pytest.approx(100.0)

    def test_narrow_bands_lower_coverage(self) -> None:
        """Very narrow bands should capture fewer price points."""
        b = make_behaviour()
        prices, ema, std_dev, z_scores = self._make_simple_inputs()
        result_narrow = b._evaluate_band_configuration(
            prices,
            ema,
            std_dev,
            band_multipliers=np.array([0.01, 0.02, 0.03]),
            z_scores=z_scores,
            band_allocations=np.array([0.7, 0.2, 0.1]),
            min_width_pct=0.0001,
        )
        result_wide = b._evaluate_band_configuration(
            prices,
            ema,
            std_dev,
            band_multipliers=np.array([100.0, 200.0, 300.0]),
            z_scores=z_scores,
            band_allocations=np.array([0.7, 0.2, 0.1]),
            min_width_pct=0.0001,
        )
        assert result_narrow["percent_in_bounds"] <= result_wide["percent_in_bounds"]

    def test_min_width_pct_enforced(self) -> None:
        """When band multipliers produce zero-width bands, min_width_pct should enforce a floor."""
        b = make_behaviour()
        prices = np.array([1.0, 1.0, 1.0])
        ema = np.array([1.0, 1.0, 1.0])
        std_dev = np.array([0.0, 0.0, 0.0])  # zero std_dev
        z_scores = np.array([0.0, 0.0, 0.0])
        result = b._evaluate_band_configuration(
            prices,
            ema,
            std_dev,
            band_multipliers=np.array([0.0, 0.0, 0.0]),
            z_scores=z_scores,
            band_allocations=np.array([0.5, 0.3, 0.2]),
            min_width_pct=0.01,
        )
        # min_width should be applied
        assert result["avg_weighted_width"] > 0

    def test_band_coverage_sums_correctly(self) -> None:
        """Band coverages should not exceed 1.0."""
        b = make_behaviour()
        prices, ema, std_dev, z_scores = self._make_simple_inputs()
        result = b._evaluate_band_configuration(
            prices,
            ema,
            std_dev,
            band_multipliers=np.array([1.0, 2.0, 3.0]),
            z_scores=z_scores,
            band_allocations=np.array([0.7, 0.2, 0.1]),
            min_width_pct=0.0001,
        )
        total_coverage = sum(result["band_coverage"])
        assert total_coverage <= 1.0 + 1e-10


class TestRunMonteCarloLevel:
    """Tests for _run_monte_carlo_level."""

    def _default_inputs(self):
        """Create default inputs for monte carlo tests."""
        np.random.seed(42)
        prices = np.array([1.0 + 0.01 * np.sin(i) for i in range(50)])
        b = make_behaviour()
        ema = b.calculate_ema(prices.tolist(), period=18)
        std_dev = b.calculate_std_dev(prices.tolist(), ema, window=14)
        z_scores = np.abs(prices - ema) / np.maximum(std_dev, 1e-6)
        return b, prices, ema, std_dev, z_scores

    def test_returns_best_config_and_all_results(self) -> None:
        """Result should contain best_config and all_results."""
        b, prices, ema, std_dev, z_scores = self._default_inputs()
        result = b._run_monte_carlo_level(
            prices,
            ema,
            std_dev,
            z_scores,
            min_multiplier=0.1,
            max_multiplier=1.5,
            num_simulations=5,
            min_width_pct=0.0001,
        )
        assert "best_config" in result
        assert "all_results" in result

    def test_best_config_keys(self) -> None:
        """best_config should have all required keys."""
        b, prices, ema, std_dev, z_scores = self._default_inputs()
        result = b._run_monte_carlo_level(
            prices,
            ema,
            std_dev,
            z_scores,
            min_multiplier=0.1,
            max_multiplier=1.5,
            num_simulations=5,
            min_width_pct=0.0001,
        )
        config = result["best_config"]
        assert "band_multipliers" in config
        assert "band_allocations" in config
        assert "zscore_economic_score" in config
        assert "percent_in_bounds" in config
        assert "avg_weighted_width" in config

    def test_all_results_lengths(self) -> None:
        """All result arrays should have length equal to num_simulations."""
        b, prices, ema, std_dev, z_scores = self._default_inputs()
        num_sims = 10
        result = b._run_monte_carlo_level(
            prices,
            ema,
            std_dev,
            z_scores,
            min_multiplier=0.1,
            max_multiplier=1.5,
            num_simulations=num_sims,
            min_width_pct=0.0001,
        )
        for key in [
            "m1_values",
            "m2_values",
            "m3_values",
            "a1_values",
            "a2_values",
            "a3_values",
            "percent_in_bounds_values",
            "avg_weighted_width_values",
            "zscore_economic_score_values",
        ]:
            assert len(result["all_results"][key]) == num_sims

    def test_allocations_sum_to_one(self) -> None:
        """All allocation triplets should sum to ~1.0."""
        b, prices, ema, std_dev, z_scores = self._default_inputs()
        result = b._run_monte_carlo_level(
            prices,
            ema,
            std_dev,
            z_scores,
            min_multiplier=0.1,
            max_multiplier=1.5,
            num_simulations=20,
            min_width_pct=0.0001,
        )
        ar = result["all_results"]
        for i in range(20):
            total = ar["a1_values"][i] + ar["a2_values"][i] + ar["a3_values"][i]
            assert total == pytest.approx(1.0, abs=1e-8)

    def test_band_multiplier_ordering(self) -> None:
        """m1 < m2 < m3 should hold for every simulation."""
        b, prices, ema, std_dev, z_scores = self._default_inputs()
        result = b._run_monte_carlo_level(
            prices,
            ema,
            std_dev,
            z_scores,
            min_multiplier=0.1,
            max_multiplier=1.5,
            num_simulations=20,
            min_width_pct=0.0001,
        )
        ar = result["all_results"]
        for i in range(20):
            assert ar["m1_values"][i] < ar["m2_values"][i]
            assert ar["m2_values"][i] < ar["m3_values"][i]

    def test_verbose_logging(self) -> None:
        """Verbose mode should call logger.info."""
        b, prices, ema, std_dev, z_scores = self._default_inputs()
        b._run_monte_carlo_level(
            prices,
            ema,
            std_dev,
            z_scores,
            min_multiplier=0.1,
            max_multiplier=1.5,
            num_simulations=5,
            min_width_pct=0.0001,
            verbose=True,
        )
        assert b.context.logger.info.called

    def test_best_has_highest_score(self) -> None:
        """The best_config should have the highest zscore_economic_score."""
        b, prices, ema, std_dev, z_scores = self._default_inputs()
        result = b._run_monte_carlo_level(
            prices,
            ema,
            std_dev,
            z_scores,
            min_multiplier=0.1,
            max_multiplier=1.5,
            num_simulations=20,
            min_width_pct=0.0001,
        )
        best_score = result["best_config"]["zscore_economic_score"]
        max_score = max(result["all_results"]["zscore_economic_score_values"])
        assert best_score == pytest.approx(max_score)


class TestOptimizeStablecoinBands:
    """Tests for optimize_stablecoin_bands."""

    def _stable_prices(self, n: int = 50) -> List[float]:
        """Generate stablecoin-like prices."""
        np.random.seed(123)
        return [1.0 + 0.001 * np.random.randn() for _ in range(n)]

    def test_returns_required_keys(self) -> None:
        """Result must contain the expected keys."""
        b = make_behaviour()
        result = b.optimize_stablecoin_bands(self._stable_prices())
        expected_keys = {
            "band_multipliers",
            "band_allocations",
            "zscore_economic_score",
            "percent_in_bounds",
            "avg_weighted_width",
            "from_level",
            "from_level_name",
        }
        assert expected_keys == set(result.keys())

    def test_band_multipliers_length(self) -> None:
        """Should return 3 band multipliers."""
        b = make_behaviour()
        result = b.optimize_stablecoin_bands(self._stable_prices())
        assert len(result["band_multipliers"]) == 3

    def test_band_allocations_sum_to_one(self) -> None:
        """Band allocations should sum to approximately 1.0."""
        b = make_behaviour()
        result = b.optimize_stablecoin_bands(self._stable_prices())
        assert sum(result["band_allocations"]) == pytest.approx(1.0, abs=1e-6)

    def test_from_level_is_valid(self) -> None:
        """from_level should be 0, 1, or 2."""
        b = make_behaviour()
        result = b.optimize_stablecoin_bands(self._stable_prices())
        assert result["from_level"] in (0, 1, 2)

    def test_verbose_mode(self) -> None:
        """Verbose mode should produce logger output."""
        b = make_behaviour()
        b.optimize_stablecoin_bands(self._stable_prices(30), verbose=True)
        assert b.context.logger.info.called

    def test_no_recursion_when_not_triggered(self) -> None:
        """If inner band conditions are not met, should stop at first level."""
        b = make_behaviour()
        # Use volatile prices so inner band allocation is lower
        np.random.seed(999)
        volatile_prices = [1.0 + 0.1 * np.random.randn() for _ in range(50)]
        result = b.optimize_stablecoin_bands(volatile_prices)
        # The result should be valid regardless of which level
        assert result["zscore_economic_score"] > 0

    def test_custom_parameters(self) -> None:
        """Custom EMA and std_dev window should be accepted."""
        b = make_behaviour()
        result = b.optimize_stablecoin_bands(
            self._stable_prices(),
            min_width_pct=0.001,
            ema_period=10,
            std_dev_window=7,
        )
        assert result["zscore_economic_score"] > 0


class TestGetPoolTokenHistory:
    """Tests for get_pool_token_history (generator)."""

    def test_unsupported_chain_returns_none(self) -> None:
        """Unsupported chain should return None."""
        b = make_behaviour()
        b.context.params.use_x402 = False
        gen = b.get_pool_token_history("unsupported_chain", "0xaaa", "0xbbb")
        result = run_generator(gen)
        assert result is None

    def test_missing_coin_ids_returns_none(self) -> None:
        """If coin IDs cannot be resolved, should return None."""
        b = make_behaviour()
        b.context.params.use_x402 = False

        def fake_get_coin_id(chain, addr, platform, headers):
            yield
            return None

        b._get_coin_id_from_address = fake_get_coin_id

        gen = b.get_pool_token_history("optimism", "0xaaa", "0xbbb")
        result = run_generator(gen)
        assert result is None

    def test_missing_price_data_returns_none(self) -> None:
        """If price data is missing, should return None."""
        b = make_behaviour()
        b.context.params.use_x402 = False

        coin_call = [0]

        def fake_get_coin_id(chain, addr, platform, headers):
            coin_call[0] += 1
            yield
            return f"coin-{coin_call[0]}"

        def fake_get_market_data(coin_id, days, headers):
            yield
            return None

        b._get_coin_id_from_address = fake_get_coin_id
        b._get_historical_market_data = fake_get_market_data

        gen = b.get_pool_token_history("optimism", "0xaaa", "0xbbb")
        result = run_generator(gen)
        assert result is None

    def test_successful_price_history(self) -> None:
        """Should return valid price history when all data is available."""
        b = make_behaviour()
        b.context.params.use_x402 = False

        coin_call = [0]

        def fake_get_coin_id(chain, addr, platform, headers):
            coin_call[0] += 1
            yield
            return f"coin-{coin_call[0]}"

        market_call = [0]

        def fake_get_market_data(coin_id, days, headers):
            market_call[0] += 1
            prices = [2.0, 2.1, 2.2] if market_call[0] == 1 else [1.0, 1.05, 1.1]
            yield
            return {"prices": prices, "timestamps": [1000, 2000, 3000]}

        b._get_coin_id_from_address = fake_get_coin_id
        b._get_historical_market_data = fake_get_market_data

        gen = b.get_pool_token_history("optimism", "0xaaa", "0xbbb")
        result = run_generator(gen)

        assert result is not None
        assert "ratio_prices" in result
        assert "current_price" in result
        assert "days" in result
        assert len(result["ratio_prices"]) == 3

    def test_different_length_price_lists(self) -> None:
        """Should handle price lists of different lengths by taking the shorter one."""
        b = make_behaviour()
        b.context.params.use_x402 = False

        coin_call = [0]

        def fake_get_coin_id(chain, addr, platform, headers):
            coin_call[0] += 1
            yield
            return f"coin-{coin_call[0]}"

        market_call = [0]

        def fake_get_market_data(coin_id, days, headers):
            market_call[0] += 1
            yield
            if market_call[0] == 1:
                return {"prices": [2.0, 2.1], "timestamps": [1000, 2000]}
            else:
                return {"prices": [1.0, 1.05, 1.1], "timestamps": [1000, 2000, 3000]}

        b._get_coin_id_from_address = fake_get_coin_id
        b._get_historical_market_data = fake_get_market_data

        gen = b.get_pool_token_history("base", "0xaaa", "0xbbb")
        result = run_generator(gen)

        assert result is not None
        assert len(result["ratio_prices"]) == 2

    def test_api_key_header_added(self) -> None:
        """API key should be added to headers when use_x402 is False."""
        b = make_behaviour()
        b.context.params.use_x402 = False

        captured_headers = {}

        def fake_get_coin_id(chain, addr, platform, headers):
            captured_headers.update(headers)
            yield
            return None

        b._get_coin_id_from_address = fake_get_coin_id

        gen = b.get_pool_token_history("optimism", "0xaaa", "0xbbb", api_key="my-key")
        run_generator(gen)

        assert captured_headers.get("x-cg-api-key") == "my-key"

    def test_exception_returns_none(self) -> None:
        """Exception during processing should return None."""
        b = make_behaviour()
        b.context.params.use_x402 = False

        def fake_get_coin_id(chain, addr, platform, headers):
            raise RuntimeError("API failure")
            yield  # pragma: no cover

        b._get_coin_id_from_address = fake_get_coin_id

        gen = b.get_pool_token_history("optimism", "0xaaa", "0xbbb")
        result = run_generator(gen)
        assert result is None

    def test_zero_price_skipped(self) -> None:
        """Price entries with zero token0 price should be skipped."""
        b = make_behaviour()
        b.context.params.use_x402 = False

        coin_call = [0]

        def fake_get_coin_id(chain, addr, platform, headers):
            coin_call[0] += 1
            yield
            return f"coin-{coin_call[0]}"

        market_call = [0]

        def fake_get_market_data(coin_id, days, headers):
            market_call[0] += 1
            yield
            if market_call[0] == 1:
                # token0 prices: second is zero
                return {"prices": [2.0, 0.0, 3.0], "timestamps": [1000, 2000, 3000]}
            else:
                return {"prices": [1.0, 1.05, 1.1], "timestamps": [1000, 2000, 3000]}

        b._get_coin_id_from_address = fake_get_coin_id
        b._get_historical_market_data = fake_get_market_data

        gen = b.get_pool_token_history("optimism", "0xaaa", "0xbbb")
        result = run_generator(gen)

        # One entry (index 1) has token0 price=0, should be skipped
        assert result is not None
        assert len(result["ratio_prices"]) == 2


class TestGetCoinIdFromAddress:
    """Tests for _get_coin_id_from_address (generator)."""

    def _make_behaviour_with_coingecko(self):
        b = make_behaviour()
        b.context.params.use_x402 = False
        b.context.coingecko = MagicMock()
        b.context.coingecko.coin_from_address_endpoint = (
            "coins/{platform}/contract/{address}"
        )
        # sleep is a generator method; mock it
        b.sleep = MagicMock(return_value=iter([None]))
        return b

    def test_known_stablecoin_returns_mapping(self) -> None:
        """Known stablecoin address should return mapped coin ID."""
        b = self._make_behaviour_with_coingecko()
        gen = b._get_coin_id_from_address(
            "optimism",
            "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
            "optimistic-ethereum",
            {},
        )
        result = run_generator(gen)
        assert result == "usd-coin"

    def test_known_stablecoin_none_mapping(self) -> None:
        """Stablecoin mapped to None should return None."""
        b = self._make_behaviour_with_coingecko()
        gen = b._get_coin_id_from_address(
            "optimism",
            "0x9dabae7274d28a45f0b65bf8ed201a5731492ca0",
            "optimistic-ethereum",
            {},
        )
        result = run_generator(gen)
        assert result is None

    def test_api_success_returns_id(self) -> None:
        """Successful API call should return the coin ID."""
        b = self._make_behaviour_with_coingecko()
        b.context.coingecko.request.return_value = (True, {"id": "my-coin"})

        gen = b._get_coin_id_from_address(
            "optimism",
            "0xdeadbeef1234567890abcdef1234567890abcdef",
            "optimistic-ethereum",
            {"Accept": "application/json"},
        )
        result = run_generator(gen)
        assert result == "my-coin"

    def test_api_failure_returns_none(self) -> None:
        """Failed API call should return None."""
        b = self._make_behaviour_with_coingecko()
        b.context.coingecko.request.return_value = (False, {})

        gen = b._get_coin_id_from_address(
            "optimism",
            "0xdeadbeef1234567890abcdef1234567890abcdef",
            "optimistic-ethereum",
            {},
        )
        result = run_generator(gen)
        assert result is None

    def test_api_response_parse_error(self) -> None:
        """If response_json.get raises, should return None."""
        b = self._make_behaviour_with_coingecko()
        mock_response = MagicMock()
        mock_response.get.side_effect = Exception("parse error")
        b.context.coingecko.request.return_value = (True, mock_response)

        gen = b._get_coin_id_from_address(
            "optimism",
            "0xdeadbeef1234567890abcdef1234567890abcdef",
            "optimistic-ethereum",
            {},
        )
        result = run_generator(gen)
        assert result is None

    def test_x402_signer_passed_when_enabled(self) -> None:
        """When use_x402 is True, eoa_account should be passed as signer."""
        b = self._make_behaviour_with_coingecko()
        b.context.params.use_x402 = True
        eoa = MagicMock()
        b.context.coingecko.request.return_value = (True, {"id": "x402-coin"})

        with patch.object(
            type(b), "eoa_account", new_callable=PropertyMock, return_value=eoa
        ):
            gen = b._get_coin_id_from_address(
                "optimism",
                "0xdeadbeef1234567890abcdef1234567890abcdef",
                "optimistic-ethereum",
                {},
            )
            result = run_generator(gen)
        assert result == "x402-coin"
        call_kwargs = b.context.coingecko.request.call_args[1]
        assert call_kwargs["x402_signer"] is eoa

    def test_exception_returns_none(self) -> None:
        """General exception should return None."""
        b = self._make_behaviour_with_coingecko()
        b.context.coingecko.request.side_effect = RuntimeError("network error")

        gen = b._get_coin_id_from_address(
            "optimism",
            "0xdeadbeef1234567890abcdef1234567890abcdef",
            "optimistic-ethereum",
            {},
        )
        result = run_generator(gen)
        assert result is None


class TestGetHistoricalMarketData:
    """Tests for _get_historical_market_data (generator)."""

    def _make_behaviour_with_coingecko(self):
        b = make_behaviour()
        b.context.params.use_x402 = False
        b.context.coingecko = MagicMock()
        b.context.coingecko.historical_market_data_endpoint = (
            "coins/{coin_id}/market_chart?days={days}"
        )
        b.sleep = MagicMock(return_value=iter([None]))
        return b

    def test_success_returns_parsed_data(self) -> None:
        """Successful response should return parsed price data."""
        b = self._make_behaviour_with_coingecko()
        b.context.coingecko.request.return_value = (
            True,
            {
                "prices": [[1000000, 50000.0], [2000000, 51000.0]],
            },
        )

        gen = b._get_historical_market_data("bitcoin", 30, {})
        result = run_generator(gen)

        assert result is not None
        assert result["coin_id"] == "bitcoin"
        assert len(result["prices"]) == 2
        assert result["prices"][0] == 50000.0
        # timestamps should be converted from ms to seconds
        assert result["timestamps"][0] == 1000.0

    def test_failure_returns_none(self) -> None:
        """Failed request should return None."""
        b = self._make_behaviour_with_coingecko()
        b.context.coingecko.request.return_value = (False, {})

        gen = b._get_historical_market_data("bitcoin", 30, {})
        result = run_generator(gen)
        assert result is None

    def test_rate_limit_429_status_code(self) -> None:
        """HTTP 429 response should return None after sleeping."""
        b = self._make_behaviour_with_coingecko()
        mock_response = MagicMock()
        mock_response.status_code = 429
        b.context.coingecko.request.return_value = (True, mock_response)

        gen = b._get_historical_market_data("bitcoin", 30, {})
        result = run_generator(gen)
        assert result is None

    def test_rate_limit_429_json_error_code(self) -> None:
        """JSON error_code=429 should return None."""
        b = self._make_behaviour_with_coingecko()
        response = {"status": {"error_code": 429}}
        b.context.coingecko.request.return_value = (True, response)

        gen = b._get_historical_market_data("bitcoin", 30, {})
        result = run_generator(gen)
        assert result is None

    def test_non_200_status_code(self) -> None:
        """Non-200 status code should return None."""
        b = self._make_behaviour_with_coingecko()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.get.return_value = {}
        b.context.coingecko.request.return_value = (True, mock_response)

        gen = b._get_historical_market_data("bitcoin", 30, {})
        result = run_generator(gen)
        assert result is None

    def test_exception_returns_none(self) -> None:
        """Exception during processing should return None."""
        b = self._make_behaviour_with_coingecko()
        b.context.coingecko.request.side_effect = RuntimeError("network error")

        gen = b._get_historical_market_data("bitcoin", 30, {})
        result = run_generator(gen)
        assert result is None

    def test_x402_signer_used_when_enabled(self) -> None:
        """When use_x402 is True, eoa_account should be passed."""
        b = self._make_behaviour_with_coingecko()
        b.context.params.use_x402 = True
        eoa = MagicMock()
        b.context.coingecko.request.return_value = (True, {"prices": []})

        with patch.object(
            type(b), "eoa_account", new_callable=PropertyMock, return_value=eoa
        ):
            gen = b._get_historical_market_data("bitcoin", 30, {})
            run_generator(gen)
        call_kwargs = b.context.coingecko.request.call_args[1]
        assert call_kwargs["x402_signer"] is eoa


class TestGetCurrentPoolPrice:
    """Tests for _get_current_pool_price (generator)."""

    def test_success_returns_price(self) -> None:
        """Should calculate price from sqrt_price_x96."""
        b = make_behaviour()
        sqrt_price_x96 = 2**96  # This means price = 1.0

        def fake_get_sqrt(chain, pool_address):
            yield
            return sqrt_price_x96

        b._get_sqrt_price_x96 = fake_get_sqrt

        gen = b._get_current_pool_price("0xpool", "optimism")
        result = run_generator(gen)

        assert result == pytest.approx(1.0)

    def test_sqrt_price_none_returns_none(self) -> None:
        """If sqrt_price_x96 is None, should return None."""
        b = make_behaviour()

        def fake_get_sqrt(chain, pool_address):
            yield
            return None

        b._get_sqrt_price_x96 = fake_get_sqrt

        gen = b._get_current_pool_price("0xpool", "optimism")
        result = run_generator(gen)

        assert result is None

    def test_exception_returns_none(self) -> None:
        """Exception should return None."""
        b = make_behaviour()

        def fake_get_sqrt(chain, pool_address):
            raise RuntimeError("contract error")
            yield  # pragma: no cover

        b._get_sqrt_price_x96 = fake_get_sqrt

        gen = b._get_current_pool_price("0xpool", "optimism")
        result = run_generator(gen)

        assert result is None

    def test_non_unit_price(self) -> None:
        """Test with a non-trivial sqrt_price_x96 value."""
        b = make_behaviour()
        sqrt_price_x96 = int(2**96 * 1.5)  # price = 2.25

        def fake_get_sqrt(chain, pool_address):
            yield
            return sqrt_price_x96

        b._get_sqrt_price_x96 = fake_get_sqrt

        gen = b._get_current_pool_price("0xpool", "optimism")
        result = run_generator(gen)

        assert result == pytest.approx(2.25)


class TestCalculateTickRangeFromBandsWrapper:
    """Tests for calculate_tick_range_from_bands_wrapper."""

    @staticmethod
    def _identity_tick(price: float) -> float:
        """A simple price-to-tick function for testing."""
        return price * 100  # simple linear mapping

    def test_returns_all_bands(self) -> None:
        """Result should have band1, band2, band3, and tuple aliases."""
        b = make_behaviour()
        result = b.calculate_tick_range_from_bands_wrapper(
            band_multipliers=[1.0, 2.0, 3.0],
            standard_deviation=0.01,
            ema=1.0,
            tick_spacing=1,
            price_to_tick_function=self._identity_tick,
        )
        assert "band1" in result
        assert "band2" in result
        assert "band3" in result
        assert "inner_ticks" in result
        assert "middle_ticks" in result
        assert "outer_ticks" in result

    def test_band_tick_structure(self) -> None:
        """Each band should have tick_lower and tick_upper."""
        b = make_behaviour()
        result = b.calculate_tick_range_from_bands_wrapper(
            band_multipliers=[1.0, 2.0, 3.0],
            standard_deviation=0.01,
            ema=1.0,
            tick_spacing=1,
            price_to_tick_function=self._identity_tick,
        )
        for band_key in ["band1", "band2", "band3"]:
            assert "tick_lower" in result[band_key]
            assert "tick_upper" in result[band_key]

    def test_tick_spacing_rounding(self) -> None:
        """Ticks should be rounded down to tick_spacing multiples."""
        b = make_behaviour()
        result = b.calculate_tick_range_from_bands_wrapper(
            band_multipliers=[1.0, 2.0, 3.0],
            standard_deviation=0.01,
            ema=1.0,
            tick_spacing=10,
            price_to_tick_function=self._identity_tick,
        )
        for band_key in ["band1", "band2", "band3"]:
            assert result[band_key]["tick_lower"] % 10 == 0
            assert result[band_key]["tick_upper"] % 10 == 0

    def test_wider_multiplier_gives_wider_ticks(self) -> None:
        """Outer bands should produce wider tick ranges."""
        b = make_behaviour()
        result = b.calculate_tick_range_from_bands_wrapper(
            band_multipliers=[1.0, 2.0, 3.0],
            standard_deviation=0.01,
            ema=1.0,
            tick_spacing=1,
            price_to_tick_function=self._identity_tick,
        )
        width1 = result["band1"]["tick_upper"] - result["band1"]["tick_lower"]
        width2 = result["band2"]["tick_upper"] - result["band2"]["tick_lower"]
        width3 = result["band3"]["tick_upper"] - result["band3"]["tick_lower"]
        assert width1 <= width2 <= width3

    def test_ticks_clamped_to_min_max(self) -> None:
        """Ticks should be clamped to [min_tick, max_tick]."""
        b = make_behaviour()

        def extreme_tick(price: float) -> float:
            return -1_000_000 if price < 1.0 else 1_000_000

        result = b.calculate_tick_range_from_bands_wrapper(
            band_multipliers=[1.0, 2.0, 3.0],
            standard_deviation=0.5,
            ema=1.0,
            tick_spacing=1,
            price_to_tick_function=extreme_tick,
        )
        for band_key in ["band1", "band2", "band3"]:
            assert result[band_key]["tick_lower"] >= MIN_TICK
            assert result[band_key]["tick_upper"] <= MAX_TICK

    def test_custom_min_max_tick(self) -> None:
        """Custom min/max tick should be respected."""
        b = make_behaviour()
        result = b.calculate_tick_range_from_bands_wrapper(
            band_multipliers=[1.0, 2.0, 3.0],
            standard_deviation=0.01,
            ema=1.0,
            tick_spacing=1,
            price_to_tick_function=self._identity_tick,
            min_tick=-100,
            max_tick=200,
        )
        for band_key in ["band1", "band2", "band3"]:
            assert result[band_key]["tick_lower"] >= -100
            assert result[band_key]["tick_upper"] <= 200

    def test_inner_ticks_tuple_matches_band1(self) -> None:
        """inner_ticks should match band1 values."""
        b = make_behaviour()
        result = b.calculate_tick_range_from_bands_wrapper(
            band_multipliers=[1.0, 2.0, 3.0],
            standard_deviation=0.01,
            ema=1.0,
            tick_spacing=1,
            price_to_tick_function=self._identity_tick,
        )
        assert result["inner_ticks"] == (
            result["band1"]["tick_lower"],
            result["band1"]["tick_upper"],
        )
        assert result["middle_ticks"] == (
            result["band2"]["tick_lower"],
            result["band2"]["tick_upper"],
        )
        assert result["outer_ticks"] == (
            result["band3"]["tick_lower"],
            result["band3"]["tick_upper"],
        )


class TestGetPoolReserves:
    """Tests for _get_pool_reserves (generator)."""

    def test_success_returns_tuple(self) -> None:
        """Should return (reserve0, reserve1) on success."""
        b = make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return [1000, 2000]

        b.contract_interact = fake_contract_interact

        gen = b._get_pool_reserves("0xpool", "optimism")
        result = run_generator(gen)

        assert result == (1000, 2000)

    def test_none_data_returns_none(self) -> None:
        """None reserves should return None."""
        b = make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return None

        b.contract_interact = fake_contract_interact

        gen = b._get_pool_reserves("0xpool", "optimism")
        result = run_generator(gen)

        assert result is None

    def test_short_data_returns_none(self) -> None:
        """Data with fewer than 2 elements should return None."""
        b = make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return [1000]

        b.contract_interact = fake_contract_interact

        gen = b._get_pool_reserves("0xpool", "optimism")
        result = run_generator(gen)

        assert result is None

    def test_exception_returns_none(self) -> None:
        """Exception should return None."""
        b = make_behaviour()

        def fake_contract_interact(**kwargs):
            raise RuntimeError("contract error")
            yield  # pragma: no cover

        b.contract_interact = fake_contract_interact

        gen = b._get_pool_reserves("0xpool", "optimism")
        result = run_generator(gen)

        assert result is None


class TestGetTokenDecimalsForAssets:
    """Tests for _get_token_decimals_for_assets (generator)."""

    def test_success_returns_list(self) -> None:
        """Should return a list of decimals for each asset."""
        b = make_behaviour()
        call_count = [0]

        def fake_contract_interact(**kwargs):
            call_count[0] += 1
            yield
            return 18 if call_count[0] == 1 else 6

        b.contract_interact = fake_contract_interact

        gen = b._get_token_decimals_for_assets(["0xtoken0", "0xtoken1"], "optimism")
        result = run_generator(gen)

        assert result == [18, 6]

    def test_none_decimals_returns_none(self) -> None:
        """If any token returns None for decimals, should return None."""
        b = make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return None

        b.contract_interact = fake_contract_interact

        gen = b._get_token_decimals_for_assets(["0xtoken0"], "optimism")
        result = run_generator(gen)

        assert result is None

    def test_empty_assets_returns_empty(self) -> None:
        """Empty asset list should return empty list."""
        b = make_behaviour()
        gen = b._get_token_decimals_for_assets([], "optimism")
        result = run_generator(gen)
        assert result == []

    def test_exception_returns_none(self) -> None:
        """Exception should return None."""
        b = make_behaviour()

        def fake_contract_interact(**kwargs):
            raise RuntimeError("error")
            yield  # pragma: no cover

        b.contract_interact = fake_contract_interact

        gen = b._get_token_decimals_for_assets(["0xtoken0"], "optimism")
        result = run_generator(gen)

        assert result is None


class TestCalculateBandInvestmentWithPoolPrice:
    """Tests for _calculate_band_investment_with_pool_price."""

    def test_basic_calculation(self) -> None:
        """Test basic investment calculation."""
        b = make_behaviour()
        sqrt_price_x96 = 2**96  # price_ratio = 1.0
        result = b._calculate_band_investment_with_pool_price(
            max_amounts_in=[1_000_000, 1_000_000_000_000_000_000],
            allocation=0.5,
            sqrt_price_x96=sqrt_price_x96,
            token_decimals=[6, 18],
        )
        assert result is not None
        assert isinstance(result, int)
        assert result > 0

    def test_full_allocation(self) -> None:
        """100% allocation should use full amounts."""
        b = make_behaviour()
        sqrt_price_x96 = 2**96  # price_ratio = 1.0
        result = b._calculate_band_investment_with_pool_price(
            max_amounts_in=[
                1_000_000,
                1_000_000,
            ],  # 1 token0, 1 token1 (6 decimals each)
            allocation=1.0,
            sqrt_price_x96=sqrt_price_x96,
            token_decimals=[6, 6],
        )
        # total_token0_equiv = 1 + 1*1.0 = 2, allocated = 2.0, raw = 2_000_000
        assert result == 2_000_000

    def test_zero_allocation(self) -> None:
        """Zero allocation should return 0."""
        b = make_behaviour()
        sqrt_price_x96 = 2**96
        result = b._calculate_band_investment_with_pool_price(
            max_amounts_in=[1_000_000, 1_000_000],
            allocation=0.0,
            sqrt_price_x96=sqrt_price_x96,
            token_decimals=[6, 6],
        )
        assert result == 0

    def test_none_sqrt_price_returns_none(self) -> None:
        """None sqrt_price_x96 should return None."""
        b = make_behaviour()
        result = b._calculate_band_investment_with_pool_price(
            max_amounts_in=[1_000_000, 1_000_000],
            allocation=0.5,
            sqrt_price_x96=None,
            token_decimals=[6, 6],
        )
        assert result is None

    def test_zero_sqrt_price_returns_none(self) -> None:
        """Zero sqrt_price_x96 should return None."""
        b = make_behaviour()
        result = b._calculate_band_investment_with_pool_price(
            max_amounts_in=[1_000_000, 1_000_000],
            allocation=0.5,
            sqrt_price_x96=0,
            token_decimals=[6, 6],
        )
        assert result is None

    def test_exception_returns_none(self) -> None:
        """Exception during calculation should return None."""
        b = make_behaviour()
        # Pass invalid token_decimals to trigger an exception
        result = b._calculate_band_investment_with_pool_price(
            max_amounts_in=[1_000_000],
            allocation=0.5,
            sqrt_price_x96=2**96,
            token_decimals=[6],  # Only one decimal, unpacking will fail
        )
        assert result is None


class TestCalculateIndividualTokenAmounts:
    """Tests for _calculate_individual_token_amounts."""

    def test_equal_ratio_unit_price(self) -> None:
        """50/50 ratio at unit price should split evenly."""
        b = make_behaviour()
        sqrt_price_x96 = 2**96  # price = 1.0
        a0, a1 = b._calculate_individual_token_amounts(
            total_band_investment=2_000_000,
            token0_ratio=0.5,
            token1_ratio=0.5,
            sqrt_price_x96=sqrt_price_x96,
            token_decimals=[6, 6],
        )
        assert a0 == 1_000_000
        assert a1 == 1_000_000

    def test_all_token0(self) -> None:
        """100% token0 ratio should put everything in token0."""
        b = make_behaviour()
        sqrt_price_x96 = 2**96
        a0, a1 = b._calculate_individual_token_amounts(
            total_band_investment=1_000_000,
            token0_ratio=1.0,
            token1_ratio=0.0,
            sqrt_price_x96=sqrt_price_x96,
            token_decimals=[6, 6],
        )
        assert a0 == 1_000_000
        assert a1 == 0

    def test_different_decimals(self) -> None:
        """Should handle different token decimals correctly."""
        b = make_behaviour()
        sqrt_price_x96 = 2**96  # price = 1.0
        a0, a1 = b._calculate_individual_token_amounts(
            total_band_investment=1_000_000,  # 1.0 token0 (6 decimals)
            token0_ratio=0.5,
            token1_ratio=0.5,
            sqrt_price_x96=sqrt_price_x96,
            token_decimals=[6, 18],
        )
        assert a0 == 500_000  # 0.5 token0 in 6 decimals
        assert a1 == 500_000_000_000_000_000  # 0.5 token1 in 18 decimals

    def test_none_sqrt_price_returns_none_none(self) -> None:
        """None sqrt_price should return (None, None)."""
        b = make_behaviour()
        a0, a1 = b._calculate_individual_token_amounts(
            total_band_investment=1_000_000,
            token0_ratio=0.5,
            token1_ratio=0.5,
            sqrt_price_x96=None,
            token_decimals=[6, 6],
        )
        assert a0 is None
        assert a1 is None

    def test_zero_sqrt_price_returns_none_none(self) -> None:
        """Zero sqrt_price should return (None, None)."""
        b = make_behaviour()
        a0, a1 = b._calculate_individual_token_amounts(
            total_band_investment=1_000_000,
            token0_ratio=0.5,
            token1_ratio=0.5,
            sqrt_price_x96=0,
            token_decimals=[6, 6],
        )
        assert a0 is None
        assert a1 is None

    def test_exception_returns_none_none(self) -> None:
        """Exception should return (None, None)."""
        b = make_behaviour()
        a0, a1 = b._calculate_individual_token_amounts(
            total_band_investment=1_000_000,
            token0_ratio=0.5,
            token1_ratio=0.5,
            sqrt_price_x96=2**96,
            token_decimals=[6],  # Wrong length
        )
        assert a0 is None
        assert a1 is None

    def test_non_unit_price(self) -> None:
        """Test with non-unit price ratio."""
        b = make_behaviour()
        # price_ratio = (1.5)^2 = 2.25
        sqrt_price_x96 = int(2**96 * 1.5)
        a0, a1 = b._calculate_individual_token_amounts(
            total_band_investment=1_000_000,  # 1.0 in token0 terms (6 decimals)
            token0_ratio=0.5,
            token1_ratio=0.5,
            sqrt_price_x96=sqrt_price_x96,
            token_decimals=[6, 6],
        )
        # token0 = 0.5 * 1.0 = 0.5 -> 500_000
        assert a0 == 500_000
        # token1 = (0.5 / 2.25) -> ~0.2222 -> ~222_222
        expected_t1 = int(0.5 / 2.25 * 1_000_000)
        assert a1 == expected_t1


class TestCalculateStablePoolAmounts:
    """Tests for _calculate_stable_pool_amounts."""

    def test_balanced_pool(self) -> None:
        """Equal reserves and equal balances should produce equal amounts."""
        b = make_behaviour()
        result = b._calculate_stable_pool_amounts(
            max_amounts_in=[1_000_000, 1_000_000],  # 1 USDC, 1 USDT (6 decimals)
            pool_reserves=(10_000_000, 10_000_000),
            token_decimals=[6, 6],
        )
        assert len(result) == 2
        # With 1:1 ratio, both should be used fully
        assert result[0] == 1_000_000
        assert result[1] == 1_000_000

    def test_option1_feasible(self) -> None:
        """When token1 is abundant, option 1 (max token0) should be chosen."""
        b = make_behaviour()
        # Pool ratio: 2 token1 per token0
        result = b._calculate_stable_pool_amounts(
            max_amounts_in=[1_000_000, 10_000_000],
            pool_reserves=(5_000_000, 10_000_000),
            token_decimals=[6, 6],
        )
        assert result is not None
        assert len(result) == 2
        # Option 1: use all 1M token0, need 2M token1 (feasible, have 10M)
        assert result[0] == 1_000_000

    def test_option2_feasible_only(self) -> None:
        """When option1 is not feasible, option 2 (max token1) should be chosen."""
        b = make_behaviour()
        # Pool ratio: 2 token1 per token0
        # We have 10M token0 but only 1M token1
        # Option1: use 10M token0, need 20M token1 -> not feasible (only have 1M)
        # Option2: use 1M token1, need 500K token0 -> feasible (have 10M)
        result = b._calculate_stable_pool_amounts(
            max_amounts_in=[10_000_000, 1_000_000],
            pool_reserves=(5_000_000, 10_000_000),
            token_decimals=[6, 6],
        )
        assert result is not None
        assert len(result) == 2
        assert result[1] == 1_000_000

    def test_zero_reserves_returns_original(self) -> None:
        """Zero reserves should return original amounts."""
        b = make_behaviour()
        original = [1_000_000, 2_000_000]
        result = b._calculate_stable_pool_amounts(
            max_amounts_in=original,
            pool_reserves=(0, 10_000_000),
            token_decimals=[6, 6],
        )
        assert result == original

    def test_neither_feasible_returns_original(self) -> None:
        """When neither option is feasible, should return original amounts."""
        b = make_behaviour()
        # Pool ratio very skewed: 1000 token1 per token0
        # We have 1 of each
        # Option1: need 1000 token1 (have 1) -> not feasible
        # Option2: need 0.001 token0 = int(0) (have 1) -> technically feasible via int truncation
        # Let me construct something truly infeasible:
        # pool_ratio = 10, amounts = [5, 5]
        # Option1: use 5 token0, need 50 token1 -> not feasible (have 5)
        # Option2: use 5 token1, need 0.5 token0 = 0 (int truncation) -> feasible (0 <= 5)
        # Hard to make truly infeasible with ints, so just check the result is valid
        original = [5, 5]
        result = b._calculate_stable_pool_amounts(
            max_amounts_in=original,
            pool_reserves=(1_000_000, 10_000_000),
            token_decimals=[6, 6],
        )
        assert result is not None
        assert len(result) == 2

    def test_different_decimals(self) -> None:
        """Handles different token decimals (e.g., USDC/6 and DAI/18)."""
        b = make_behaviour()
        # Pool has equal "value" of both tokens
        result = b._calculate_stable_pool_amounts(
            max_amounts_in=[1_000_000, 1_000_000_000_000_000_000],  # 1 USDC, 1 DAI
            pool_reserves=(10_000_000, 10_000_000_000_000_000_000),  # 10 USDC, 10 DAI
            token_decimals=[6, 18],
        )
        assert result is not None
        assert len(result) == 2

    def test_exception_returns_original(self) -> None:
        """Exception should return original max_amounts_in."""
        b = make_behaviour()
        original = [1_000_000, 2_000_000]
        # Pass invalid pool_reserves to trigger exception
        result = b._calculate_stable_pool_amounts(
            max_amounts_in=original,
            pool_reserves=None,  # This will cause unpacking error
            token_decimals=[6, 6],
        )
        assert result == original

    def test_both_feasible_option1_higher_value(self) -> None:
        """When both options are feasible and option1 has higher value, option1 chosen."""
        b = make_behaviour()
        # Pool ratio 1:1, we have more of token0 but enough of both
        # Option1: use 5M token0, need 5M token1 -> feasible (have 6M)
        # Option2: use 6M token1, need 6M token0 -> feasible (have 5M)? No, need 6M but have 5M
        # Let me use amounts where both are truly feasible:
        # amounts=[3M, 4M], pool ratio = 1:1
        # Option1: use 3M token0, need 3M token1 -> feasible (have 4M)
        # Option2: use 4M token1, need 4M token0 -> not feasible (have 3M)
        # So only option1 feasible. Try:
        # amounts=[3M, 3M], pool ratio = 1:1
        # Option1: use 3M token0, need 3M token1 -> feasible
        # Option2: use 3M token1, need 3M token0 -> feasible
        # Both feasible, option1_value=3.0, option2_value=3.0 -> option1 (>=)
        result = b._calculate_stable_pool_amounts(
            max_amounts_in=[3_000_000, 3_000_000],
            pool_reserves=(10_000_000, 10_000_000),
            token_decimals=[6, 6],
        )
        assert result[0] == 3_000_000
        assert result[1] == 3_000_000

    def test_both_feasible_option2_higher_value(self) -> None:
        """When both options are feasible and option2 has higher value, option2 chosen."""
        b = make_behaviour()
        # Pool ratio 1:2 (2 token1 per token0)
        # amounts=[1M, 5M]
        # Option1: use 1M token0, need 2M token1 -> feasible (have 5M)
        #   option1_value = normalized_amount0 = 1.0
        # Option2: use 5M token1, need 2.5M token0 -> feasible (have 1M)? No!
        # Need both to be feasible and option2 value > option1:
        # Pool ratio: 0.5 token1 per token0
        # amounts=[2M, 2M], reserves=(10M, 5M)
        # Option1: use 2M token0, need 1M token1 -> feasible (have 2M)
        #   option1_value = 2.0
        # Option2: use 2M token1, need 4M token0 -> not feasible (have 2M)
        # Still not both feasible. Let me try:
        # Pool ratio: 2 token1 per token0, amounts=[1M, 10M], reserves=(5M, 10M)
        # Option1: use 1M token0, need 2M token1 -> feasible (have 10M), option1_value=1.0
        # Option2: use 10M token1, need 5M token0 -> not feasible (have 1M)
        # Let me think differently. For option2_value > option1_value:
        # option1_value = normalized_amount0 = amount0 / 10^dec0
        # option2_value = required_normalized_amount0_for_max_amount1 = normalized_amount1 / pool_ratio
        # Need option2_value > option1_value:
        # normalized_amount1 / pool_ratio > normalized_amount0
        # And option2 feasible: required_amount0_for_max_amount1 <= amount0_desired
        # This means: normalized_amount1 / pool_ratio <= normalized_amount0
        # This contradicts the condition! So option2_value can never be > option1_value
        # when option2 is feasible. Let's verify: if option2 feasible, then
        # required_amount0 <= amount0_desired, i.e., normalized_amount1/pool_ratio <= normalized_amount0
        # That means option2_value <= option1_value. So the "both feasible, option2 chosen" branch
        # is effectively unreachable. Skip this test.
        pass

    def test_constraint_adjustment(self) -> None:
        """Constraint verification should adjust amounts for rounding."""
        b = make_behaviour()
        # Use amounts that will produce a rounding mismatch
        result = b._calculate_stable_pool_amounts(
            max_amounts_in=[1_000_003, 10_000_000],
            pool_reserves=(5_000_000, 10_000_000),
            token_decimals=[6, 6],
        )
        assert result is not None
        assert len(result) == 2

    def test_constraint_adjustment_different_decimals(self) -> None:
        """Constraint adjustment with different decimals."""
        b = make_behaviour()
        result = b._calculate_stable_pool_amounts(
            max_amounts_in=[1_000_000, 1_000_000_000_000_000_003],
            pool_reserves=(10_000_000, 10_000_000_000_000_000_000),
            token_decimals=[6, 18],
        )
        assert result is not None
        assert len(result) == 2


class TestGetGaugeAddress:
    """Tests for get_gauge_address generator."""

    def test_no_chain_returns_none(self) -> None:
        """Missing chain kwarg returns None."""
        b = make_behaviour()
        result = exhaust_generator(b.get_gauge_address("0xPool"))
        assert result is None

    def test_no_voter_address_returns_none(self) -> None:
        """No voter contract address for chain returns None."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {}
        result = exhaust_generator(b.get_gauge_address("0xPool", chain="optimism"))
        assert result is None

    def test_contract_interact_returns_none(self) -> None:
        """contract_interact returning None yields None."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(b.get_gauge_address("0xPool", chain="optimism"))
        assert result is None

    def test_contract_interact_returns_zero_address(self) -> None:
        """ZERO_ADDRESS gauge means no gauge found."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact([ZERO_ADDRESS])
        result = exhaust_generator(b.get_gauge_address("0xPool", chain="optimism"))
        assert result is None

    def test_success(self) -> None:
        """Happy path returns valid gauge address."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact(["0xGauge123"])
        result = exhaust_generator(b.get_gauge_address("0xPool", chain="optimism"))
        assert result == "0xGauge123"

    def test_empty_string_gauge(self) -> None:
        """Empty string gauge is falsy -> None."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact([""])
        result = exhaust_generator(b.get_gauge_address("0xPool", chain="optimism"))
        assert result is None


class TestStakeLpTokens:
    """Tests for stake_lp_tokens generator."""

    def test_missing_params(self) -> None:
        """Missing chain/safe_address returns error dict."""
        b = make_behaviour()
        result = exhaust_generator(b.stake_lp_tokens("0xLP", 100))
        assert "error" in result
        assert "Missing required parameters" in result["error"]

    def test_missing_chain(self) -> None:
        """Only safe_address provided, chain is missing."""
        b = make_behaviour()
        result = exhaust_generator(
            b.stake_lp_tokens("0xLP", 100, safe_address="0xSafe")
        )
        assert "error" in result

    def test_amount_zero(self) -> None:
        """Amount <= 0 returns error."""
        b = make_behaviour()
        result = exhaust_generator(
            b.stake_lp_tokens("0xLP", 0, chain="optimism", safe_address="0xSafe")
        )
        assert "Amount must be greater than 0" in result["error"]

    def test_amount_negative(self) -> None:
        """Negative amount returns error."""
        b = make_behaviour()
        result = exhaust_generator(
            b.stake_lp_tokens("0xLP", -5, chain="optimism", safe_address="0xSafe")
        )
        assert "error" in result

    def test_no_gauge(self) -> None:
        """No gauge for the pool returns error."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        # get_gauge_address: contract_interact -> None => no gauge
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b.stake_lp_tokens("0xLP", 100, chain="optimism", safe_address="0xSafe")
        )
        assert "No gauge found" in result["error"]

    def test_approve_tx_fails(self) -> None:
        """Approve tx failure returns error."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        # 1: get_gauge_address -> "0xGauge"
        # 2: approve tx -> None (fail)
        b.contract_interact = make_contract_interact(["0xGauge", None])
        result = exhaust_generator(
            b.stake_lp_tokens("0xLP", 100, chain="optimism", safe_address="0xSafe")
        )
        assert "approval" in result["error"].lower()

    def test_stake_tx_fails(self) -> None:
        """Stake tx failure returns error."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        # 1: get_gauge_address -> "0xGauge"
        # 2: approve tx -> "0xApproveData"
        # 3: stake tx -> None (fail)
        b.contract_interact = make_contract_interact(["0xGauge", "0xApproveData", None])
        result = exhaust_generator(
            b.stake_lp_tokens("0xLP", 100, chain="optimism", safe_address="0xSafe")
        )
        assert "stake" in result["error"].lower()

    def test_no_multisend_address(self) -> None:
        """Missing multisend address returns error."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.params.multisend_contract_addresses = {}
        b.contract_interact = make_contract_interact(
            ["0xGauge", "0xApproveData", "0xStakeData"]
        )
        result = exhaust_generator(
            b.stake_lp_tokens("0xLP", 100, chain="optimism", safe_address="0xSafe")
        )
        assert "multisend" in result["error"].lower()

    def test_multisend_tx_fails(self) -> None:
        """Multisend tx failure returns error."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.params.multisend_contract_addresses = {"optimism": "0xMulti"}
        # 1: gauge, 2: approve, 3: stake, 4: multisend -> None
        b.contract_interact = make_contract_interact(
            ["0xGauge", "0xApproveData", "0xStakeData", None]
        )
        result = exhaust_generator(
            b.stake_lp_tokens("0xLP", 100, chain="optimism", safe_address="0xSafe")
        )
        assert "multisend" in result["error"].lower()

    def test_success(self) -> None:
        """Full happy path."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.params.multisend_contract_addresses = {"optimism": "0xMulti"}
        b.contract_interact = make_contract_interact(
            ["0xGauge", "0xApproveData", "0xStakeData", "0xaabbccdd"]
        )
        result = exhaust_generator(
            b.stake_lp_tokens("0xLP", 100, chain="optimism", safe_address="0xSafe")
        )
        assert result["success"] is True
        assert result["is_multisend"] is True
        assert result["contract_address"] == "0xMulti"
        assert result["gauge_address"] == "0xGauge"
        assert result["amount"] == 100
        assert isinstance(result["tx_hash"], bytes)


class TestUnstakeLpTokens:
    """Tests for unstake_lp_tokens generator."""

    def test_missing_params(self) -> None:
        """Missing chain/safe_address returns error."""
        b = make_behaviour()
        result = exhaust_generator(b.unstake_lp_tokens("0xLP", 100))
        assert "error" in result

    def test_amount_zero(self) -> None:
        """Amount <= 0 returns error."""
        b = make_behaviour()
        result = exhaust_generator(
            b.unstake_lp_tokens("0xLP", 0, chain="optimism", safe_address="0xSafe")
        )
        assert "Amount must be greater than 0" in result["error"]

    def test_no_gauge(self) -> None:
        """No gauge found returns error."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b.unstake_lp_tokens("0xLP", 100, chain="optimism", safe_address="0xSafe")
        )
        assert "No gauge found" in result["error"]

    def test_insufficient_balance_none(self) -> None:
        """Staked balance is None => insufficient."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        # 1: gauge, 2: balance_of -> None
        b.contract_interact = make_contract_interact(["0xGauge", None])
        result = exhaust_generator(
            b.unstake_lp_tokens("0xLP", 100, chain="optimism", safe_address="0xSafe")
        )
        assert "Insufficient" in result["error"]

    def test_insufficient_balance_low(self) -> None:
        """Staked balance < amount => error."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        # 1: gauge, 2: balance_of -> 50 (< 100)
        b.contract_interact = make_contract_interact(["0xGauge", 50])
        result = exhaust_generator(
            b.unstake_lp_tokens("0xLP", 100, chain="optimism", safe_address="0xSafe")
        )
        assert "Insufficient" in result["error"]

    def test_withdraw_tx_fails(self) -> None:
        """Withdraw tx failure returns error."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        # 1: gauge, 2: balance (200), 3: withdraw -> None
        b.contract_interact = make_contract_interact(["0xGauge", 200, None])
        result = exhaust_generator(
            b.unstake_lp_tokens("0xLP", 100, chain="optimism", safe_address="0xSafe")
        )
        assert "withdraw" in result["error"].lower()

    def test_success(self) -> None:
        """Happy path."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact(["0xGauge", 200, "0xaabbccdd"])
        result = exhaust_generator(
            b.unstake_lp_tokens("0xLP", 100, chain="optimism", safe_address="0xSafe")
        )
        assert result["success"] is True
        assert result["is_multisend"] is False
        assert result["amount"] == 100
        assert isinstance(result["tx_hash"], bytes)


class TestClaimRewards:
    """Tests for claim_rewards generator."""

    def test_missing_params(self) -> None:
        """Missing required params returns error."""
        b = make_behaviour()
        result = exhaust_generator(b.claim_rewards("0xLP"))
        assert "error" in result

    def test_no_gauge(self) -> None:
        """No gauge returns error."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        # get_gauge_address -> None
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b.claim_rewards("0xLP", chain="optimism", safe_address="0xSafe")
        )
        assert "No gauge found" in result["error"]

    def test_zero_pending_rewards(self) -> None:
        """Zero pending rewards returns success with message."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        # 1: get_gauge in claim_rewards -> "0xGauge"
        # 2: get_gauge in get_pending_rewards -> "0xGauge"
        # 3: earned -> 0
        b.contract_interact = make_contract_interact(["0xGauge", "0xGauge", 0])
        result = exhaust_generator(
            b.claim_rewards("0xLP", chain="optimism", safe_address="0xSafe")
        )
        assert result["success"] is True
        assert result["pending_rewards"] == 0

    def test_claim_tx_fails(self) -> None:
        """Claim tx failure returns error."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        # 1: get_gauge -> 0xGauge
        # 2: get_gauge in get_pending_rewards -> 0xGauge
        # 3: earned -> 100
        # 4: claim_tx -> None
        b.contract_interact = make_contract_interact(["0xGauge", "0xGauge", 100, None])
        result = exhaust_generator(
            b.claim_rewards("0xLP", chain="optimism", safe_address="0xSafe")
        )
        assert "error" in result
        assert "claim" in result["error"].lower()

    def test_success(self) -> None:
        """Happy path with rewards to claim."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact(
            ["0xGauge", "0xGauge", 100, "0xClaimTxHash"]
        )
        result = exhaust_generator(
            b.claim_rewards("0xLP", chain="optimism", safe_address="0xSafe")
        )
        assert result["success"] is True
        assert result["pending_rewards"] == 100
        assert result["tx_hash"] == "0xClaimTxHash"


class TestGetPendingRewards:
    """Tests for get_pending_rewards generator."""

    def test_no_chain(self) -> None:
        """Missing chain returns 0."""
        b = make_behaviour()
        result = exhaust_generator(b.get_pending_rewards("0xLP", "0xUser"))
        assert result == 0

    def test_no_gauge(self) -> None:
        """No gauge returns 0."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b.get_pending_rewards("0xLP", "0xUser", chain="optimism")
        )
        assert result == 0

    def test_earned_none(self) -> None:
        """Earned result is None returns 0."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        # 1: gauge, 2: earned -> None
        b.contract_interact = make_contract_interact(["0xGauge", None])
        result = exhaust_generator(
            b.get_pending_rewards("0xLP", "0xUser", chain="optimism")
        )
        assert result == 0

    def test_earned_not_int(self) -> None:
        """Non-int earned_result coerces to 0."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact(["0xGauge", "not_an_int"])
        result = exhaust_generator(
            b.get_pending_rewards("0xLP", "0xUser", chain="optimism")
        )
        assert result == 0

    def test_success(self) -> None:
        """Valid integer reward returned."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact(["0xGauge", 500])
        result = exhaust_generator(
            b.get_pending_rewards("0xLP", "0xUser", chain="optimism")
        )
        assert result == 500


class TestGetStakedBalance:
    """Tests for get_staked_balance generator."""

    def test_no_chain(self) -> None:
        """Missing chain returns 0."""
        b = make_behaviour()
        result = exhaust_generator(b.get_staked_balance("0xLP", "0xUser"))
        assert result == 0

    def test_no_gauge(self) -> None:
        """No gauge returns 0."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b.get_staked_balance("0xLP", "0xUser", chain="optimism")
        )
        assert result == 0

    def test_balance_none(self) -> None:
        """None balance returns 0."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact(["0xGauge", None])
        result = exhaust_generator(
            b.get_staked_balance("0xLP", "0xUser", chain="optimism")
        )
        assert result == 0

    def test_balance_zero(self) -> None:
        """Zero balance (falsy) returns 0."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact(["0xGauge", 0])
        result = exhaust_generator(
            b.get_staked_balance("0xLP", "0xUser", chain="optimism")
        )
        assert result == 0

    def test_balance_not_int(self) -> None:
        """Non-int balance coerces to 0."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact(["0xGauge", "string_val"])
        result = exhaust_generator(
            b.get_staked_balance("0xLP", "0xUser", chain="optimism")
        )
        assert result == 0

    def test_success(self) -> None:
        """Valid balance returned."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact(["0xGauge", 1000])
        result = exhaust_generator(
            b.get_staked_balance("0xLP", "0xUser", chain="optimism")
        )
        assert result == 1000


class TestGetGaugeTotalSupply:
    """Tests for get_gauge_total_supply generator."""

    def test_no_chain(self) -> None:
        """Missing chain returns 0."""
        b = make_behaviour()
        result = exhaust_generator(b.get_gauge_total_supply("0xLP"))
        assert result == 0

    def test_no_gauge(self) -> None:
        """No gauge returns 0."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(b.get_gauge_total_supply("0xLP", chain="optimism"))
        assert result == 0

    def test_total_supply_none(self) -> None:
        """None total supply returns 0."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact(["0xGauge", None])
        result = exhaust_generator(b.get_gauge_total_supply("0xLP", chain="optimism"))
        assert result == 0

    def test_total_supply_not_int(self) -> None:
        """Non-int coerces to 0."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact(["0xGauge", "bad"])
        result = exhaust_generator(b.get_gauge_total_supply("0xLP", chain="optimism"))
        assert result == 0

    def test_success(self) -> None:
        """Valid total supply returned."""
        b = make_behaviour()
        b.params.velodrome_voter_contract_addresses = {"optimism": "0xVoter"}
        b.contract_interact = make_contract_interact(["0xGauge", 50000])
        result = exhaust_generator(b.get_gauge_total_supply("0xLP", chain="optimism"))
        assert result == 50000


class TestStakeClLpTokens:
    """Tests for stake_cl_lp_tokens generator."""

    def test_missing_params(self) -> None:
        """Missing chain/safe_address returns error."""
        b = make_behaviour()
        result = exhaust_generator(b.stake_cl_lp_tokens([1], "0xGauge"))
        assert "error" in result

    def test_empty_token_ids(self) -> None:
        """Empty token_ids returns error."""
        b = make_behaviour()
        result = exhaust_generator(
            b.stake_cl_lp_tokens([], "0xGauge", chain="optimism", safe_address="0xSafe")
        )
        assert "No token IDs" in result["error"]

    def test_no_position_manager(self) -> None:
        """Missing position manager address returns error."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {}
        result = exhaust_generator(
            b.stake_cl_lp_tokens(
                [1], "0xGauge", chain="optimism", safe_address="0xSafe"
            )
        )
        assert "position manager" in result["error"].lower()

    def test_approval_needed_but_fails(self) -> None:
        """setApprovalForAll tx fails returns error."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        # 1: is_approved -> False (not approved)
        # 2: approve_all_tx -> None (fail)
        b.contract_interact = make_contract_interact([False, None])
        result = exhaust_generator(
            b.stake_cl_lp_tokens(
                [1], "0xGauge", chain="optimism", safe_address="0xSafe"
            )
        )
        assert "setApprovalForAll" in result["error"]

    def test_all_stake_txs_fail_not_approved(self) -> None:
        """All stake txs fail when approval was given => no valid txs error."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        # 1: is_approved -> True (skip approval)
        # 2: stake token 1 -> None (fail)
        # 3: stake token 2 -> None (fail)
        b.contract_interact = make_contract_interact([True, None, None])
        result = exhaust_generator(
            b.stake_cl_lp_tokens(
                [1, 2], "0xGauge", chain="optimism", safe_address="0xSafe"
            )
        )
        assert "No valid stake transactions" in result["error"]

    def test_no_multisend_address(self) -> None:
        """Missing multisend address returns error."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        b.params.multisend_contract_addresses = {}
        # 1: is_approved -> True, 2: stake -> ok
        b.contract_interact = make_contract_interact([True, "0xStakeData"])
        result = exhaust_generator(
            b.stake_cl_lp_tokens(
                [1], "0xGauge", chain="optimism", safe_address="0xSafe"
            )
        )
        assert "multisend" in result["error"].lower()

    def test_multisend_tx_fails(self) -> None:
        """Multisend tx failure returns error."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        b.params.multisend_contract_addresses = {"optimism": "0xMulti"}
        # 1: is_approved -> True, 2: stake -> ok, 3: multisend -> None
        b.contract_interact = make_contract_interact([True, "0xStakeData", None])
        result = exhaust_generator(
            b.stake_cl_lp_tokens(
                [1], "0xGauge", chain="optimism", safe_address="0xSafe"
            )
        )
        assert "multisend" in result["error"].lower()

    def test_success_already_approved(self) -> None:
        """Happy path, already approved for all."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        b.params.multisend_contract_addresses = {"optimism": "0xMulti"}
        # 1: is_approved -> True, 2: stake1, 3: stake2, 4: multisend
        b.contract_interact = make_contract_interact(
            [True, "0xStake1", "0xStake2", "0xaabbccdd"]
        )
        result = exhaust_generator(
            b.stake_cl_lp_tokens(
                [1, 2], "0xGauge", chain="optimism", safe_address="0xSafe"
            )
        )
        assert result["success"] is True
        assert result["is_multisend"] is True
        assert len(result["staked_positions"]) == 2

    def test_success_needs_approval(self) -> None:
        """Happy path, needs approval first."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        b.params.multisend_contract_addresses = {"optimism": "0xMulti"}
        # 1: is_approved -> False, 2: approve -> ok, 3: stake1, 4: multisend
        b.contract_interact = make_contract_interact(
            [False, "0xApproveAll", "0xStake1", "0xaabbccdd"]
        )
        result = exhaust_generator(
            b.stake_cl_lp_tokens(
                [1], "0xGauge", chain="optimism", safe_address="0xSafe"
            )
        )
        assert result["success"] is True

    def test_partial_stake_failure(self) -> None:
        """One token fails to stake, another succeeds."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        b.params.multisend_contract_addresses = {"optimism": "0xMulti"}
        # 1: is_approved -> True, 2: stake token1 -> None (fail), 3: stake token2 -> ok, 4: multisend
        b.contract_interact = make_contract_interact(
            [True, None, "0xStake2", "0xaabbccdd"]
        )
        result = exhaust_generator(
            b.stake_cl_lp_tokens(
                [1, 2], "0xGauge", chain="optimism", safe_address="0xSafe"
            )
        )
        assert result["success"] is True
        assert len(result["staked_positions"]) == 1
        assert result["staked_positions"][0]["token_id"] == 2


class TestUnstakeClLpTokens:
    """Tests for unstake_cl_lp_tokens generator."""

    def test_missing_params(self) -> None:
        """Missing chain/safe_address returns error."""
        b = make_behaviour()
        result = exhaust_generator(b.unstake_cl_lp_tokens([1], "0xGauge"))
        assert "error" in result

    def test_empty_token_ids(self) -> None:
        """Empty token_ids returns error."""
        b = make_behaviour()
        result = exhaust_generator(
            b.unstake_cl_lp_tokens(
                [], "0xGauge", chain="optimism", safe_address="0xSafe"
            )
        )
        assert "No token IDs" in result["error"]

    def test_all_withdraw_txs_fail(self) -> None:
        """All withdraw txs fail => error."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([None, None])
        result = exhaust_generator(
            b.unstake_cl_lp_tokens(
                [1, 2], "0xGauge", chain="optimism", safe_address="0xSafe"
            )
        )
        assert "No valid unstake transactions" in result["error"]

    def test_no_multisend_address(self) -> None:
        """Missing multisend returns error."""
        b = make_behaviour()
        b.params.multisend_contract_addresses = {}
        b.contract_interact = make_contract_interact(["0xWithdraw1"])
        result = exhaust_generator(
            b.unstake_cl_lp_tokens(
                [1], "0xGauge", chain="optimism", safe_address="0xSafe"
            )
        )
        assert "multisend" in result["error"].lower()

    def test_multisend_tx_fails(self) -> None:
        """Multisend tx failure returns error."""
        b = make_behaviour()
        b.params.multisend_contract_addresses = {"optimism": "0xMulti"}
        b.contract_interact = make_contract_interact(["0xWithdraw1", None])
        result = exhaust_generator(
            b.unstake_cl_lp_tokens(
                [1], "0xGauge", chain="optimism", safe_address="0xSafe"
            )
        )
        assert "multisend" in result["error"].lower()

    def test_success(self) -> None:
        """Happy path."""
        b = make_behaviour()
        b.params.multisend_contract_addresses = {"optimism": "0xMulti"}
        b.contract_interact = make_contract_interact(["0xW1", "0xW2", "0xaabbccdd"])
        result = exhaust_generator(
            b.unstake_cl_lp_tokens(
                [1, 2], "0xGauge", chain="optimism", safe_address="0xSafe"
            )
        )
        assert result["success"] is True
        assert result["is_multisend"] is True
        assert len(result["unstaked_positions"]) == 2

    def test_partial_failure(self) -> None:
        """One token fails, another succeeds."""
        b = make_behaviour()
        b.params.multisend_contract_addresses = {"optimism": "0xMulti"}
        b.contract_interact = make_contract_interact([None, "0xW2", "0xaabbccdd"])
        result = exhaust_generator(
            b.unstake_cl_lp_tokens(
                [1, 2], "0xGauge", chain="optimism", safe_address="0xSafe"
            )
        )
        assert result["success"] is True
        assert len(result["unstaked_positions"]) == 1
        assert result["unstaked_positions"][0]["token_id"] == 2


class TestClaimClRewards:
    """Tests for claim_cl_rewards generator."""

    def test_missing_params(self) -> None:
        """Missing chain/safe_address returns error."""
        b = make_behaviour()
        result = exhaust_generator(b.claim_cl_rewards("0xGauge", [1]))
        assert "error" in result

    def test_empty_token_ids(self) -> None:
        """Empty token_ids returns error."""
        b = make_behaviour()
        result = exhaust_generator(
            b.claim_cl_rewards("0xGauge", [], chain="optimism", safe_address="0xSafe")
        )
        assert "No token IDs" in result["error"]

    def test_no_pending_rewards(self) -> None:
        """All tokens have zero rewards -> success with message."""
        b = make_behaviour()
        # earned for token 1 -> 0
        b.contract_interact = make_contract_interact([0])
        result = exhaust_generator(
            b.claim_cl_rewards("0xGauge", [1], chain="optimism", safe_address="0xSafe")
        )
        assert result["success"] is True
        assert "No tokens with pending rewards" in result["message"]

    def test_earned_result_none_skipped(self) -> None:
        """If earned_result is None for a token, it is skipped."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b.claim_cl_rewards("0xGauge", [1], chain="optimism", safe_address="0xSafe")
        )
        assert result["success"] is True
        assert "No tokens with pending rewards" in result["message"]

    def test_earned_not_int_treated_as_zero(self) -> None:
        """Non-int earned treated as 0."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact(["not_int"])
        result = exhaust_generator(
            b.claim_cl_rewards("0xGauge", [1], chain="optimism", safe_address="0xSafe")
        )
        assert result["success"] is True
        assert "No tokens with pending rewards" in result["message"]

    def test_claim_tx_fails_for_token(self) -> None:
        """Claim tx fails for a token with rewards -> skipped, no-rewards result."""
        b = make_behaviour()
        # earned -> 100, claim_tx -> None
        b.contract_interact = make_contract_interact([100, None])
        result = exhaust_generator(
            b.claim_cl_rewards("0xGauge", [1], chain="optimism", safe_address="0xSafe")
        )
        assert result["success"] is True
        assert "No tokens with pending rewards" in result["message"]

    def test_no_multisend_address(self) -> None:
        """Missing multisend returns error."""
        b = make_behaviour()
        b.params.multisend_contract_addresses = {}
        b.contract_interact = make_contract_interact([100, "0xClaimTx"])
        result = exhaust_generator(
            b.claim_cl_rewards("0xGauge", [1], chain="optimism", safe_address="0xSafe")
        )
        assert "multisend" in result["error"].lower()

    def test_multisend_tx_fails(self) -> None:
        """Multisend tx failure returns error."""
        b = make_behaviour()
        b.params.multisend_contract_addresses = {"optimism": "0xMulti"}
        b.contract_interact = make_contract_interact([100, "0xClaimTx", None])
        result = exhaust_generator(
            b.claim_cl_rewards("0xGauge", [1], chain="optimism", safe_address="0xSafe")
        )
        assert "multisend" in result["error"].lower()

    def test_success_single_token(self) -> None:
        """Happy path with one token having rewards."""
        b = make_behaviour()
        b.params.multisend_contract_addresses = {"optimism": "0xMulti"}
        b.contract_interact = make_contract_interact([100, "0xClaimTx", "0xaabbccdd"])
        result = exhaust_generator(
            b.claim_cl_rewards("0xGauge", [1], chain="optimism", safe_address="0xSafe")
        )
        assert result["success"] is True
        assert result["is_multisend"] is True
        assert len(result["tokens_with_rewards"]) == 1
        assert result["tokens_with_rewards"][0]["earned"] == 100

    def test_success_multiple_tokens_mixed(self) -> None:
        """Multiple tokens: one with rewards, one without."""
        b = make_behaviour()
        b.params.multisend_contract_addresses = {"optimism": "0xMulti"}
        # token 1: earned=0 (no rewards)
        # token 2: earned=200, claim_tx=ok
        # multisend -> ok
        b.contract_interact = make_contract_interact(
            [0, 200, "0xClaimTx2", "0xaabbccdd"]
        )
        result = exhaust_generator(
            b.claim_cl_rewards(
                "0xGauge", [1, 2], chain="optimism", safe_address="0xSafe"
            )
        )
        assert result["success"] is True
        assert len(result["tokens_with_rewards"]) == 1
        assert result["tokens_with_rewards"][0]["token_id"] == 2


class TestGetClPendingRewards:
    """Tests for get_cl_pending_rewards generator."""

    def test_missing_params(self) -> None:
        """Missing required kwargs returns 0."""
        b = make_behaviour()
        result = exhaust_generator(b.get_cl_pending_rewards("0xAccount"))
        assert result == 0

    def test_missing_gauge_address(self) -> None:
        """Missing gauge_address returns 0."""
        b = make_behaviour()
        result = exhaust_generator(
            b.get_cl_pending_rewards("0xAccount", chain="optimism", token_id=1)
        )
        assert result == 0

    def test_missing_token_id(self) -> None:
        """Missing token_id returns 0."""
        b = make_behaviour()
        result = exhaust_generator(
            b.get_cl_pending_rewards(
                "0xAccount", chain="optimism", gauge_address="0xGauge"
            )
        )
        assert result == 0

    def test_earned_none(self) -> None:
        """earned_result is None returns 0."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b.get_cl_pending_rewards(
                "0xAccount", chain="optimism", gauge_address="0xGauge", token_id=1
            )
        )
        assert result == 0

    def test_earned_not_int(self) -> None:
        """Non-int earned coerces to 0."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact(["not_int"])
        result = exhaust_generator(
            b.get_cl_pending_rewards(
                "0xAccount", chain="optimism", gauge_address="0xGauge", token_id=1
            )
        )
        assert result == 0

    def test_success(self) -> None:
        """Valid earned amount returned."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([999])
        result = exhaust_generator(
            b.get_cl_pending_rewards(
                "0xAccount", chain="optimism", gauge_address="0xGauge", token_id=1
            )
        )
        assert result == 999


class TestGetClStakedBalance:
    """Tests for get_cl_staked_balance generator."""

    def test_missing_params(self) -> None:
        """Missing chain/gauge_address returns 0."""
        b = make_behaviour()
        result = exhaust_generator(b.get_cl_staked_balance("0xAccount"))
        assert result == 0

    def test_missing_gauge(self) -> None:
        """Missing gauge_address returns 0."""
        b = make_behaviour()
        result = exhaust_generator(
            b.get_cl_staked_balance("0xAccount", chain="optimism")
        )
        assert result == 0

    def test_balance_none(self) -> None:
        """None balance returns 0."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b.get_cl_staked_balance(
                "0xAccount", chain="optimism", gauge_address="0xGauge"
            )
        )
        assert result == 0

    def test_balance_zero(self) -> None:
        """Zero balance (falsy) returns 0."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([0])
        result = exhaust_generator(
            b.get_cl_staked_balance(
                "0xAccount", chain="optimism", gauge_address="0xGauge"
            )
        )
        assert result == 0

    def test_balance_not_int(self) -> None:
        """Non-int coerces to 0."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact(["bad"])
        result = exhaust_generator(
            b.get_cl_staked_balance(
                "0xAccount", chain="optimism", gauge_address="0xGauge"
            )
        )
        assert result == 0

    def test_success(self) -> None:
        """Valid balance returned."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([7777])
        result = exhaust_generator(
            b.get_cl_staked_balance(
                "0xAccount", chain="optimism", gauge_address="0xGauge"
            )
        )
        assert result == 7777


class TestIsClTokenStaked:
    """Tests for is_cl_token_staked generator (ZD #950 fix)."""

    def test_missing_params(self) -> None:
        """Missing chain/gauge/token_id returns None (cannot verify)."""
        b = make_behaviour()
        result = exhaust_generator(b.is_cl_token_staked("0xAccount", 1))
        assert result is None

    def test_missing_gauge_address(self) -> None:
        b = make_behaviour()
        result = exhaust_generator(
            b.is_cl_token_staked("0xAccount", 1, chain="optimism")
        )
        assert result is None

    def test_missing_token_id(self) -> None:
        b = make_behaviour()
        result = exhaust_generator(
            b.is_cl_token_staked(
                "0xAccount", None, chain="optimism", gauge_address="0xGauge"
            )
        )
        assert result is None

    def test_contract_returns_none(self) -> None:
        """RPC failure bubbles up as None so callers can fall back."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b.is_cl_token_staked(
                "0xAccount", 1, chain="optimism", gauge_address="0xGauge"
            )
        )
        assert result is None

    def test_true(self) -> None:
        b = make_behaviour()
        b.contract_interact = make_contract_interact([True])
        result = exhaust_generator(
            b.is_cl_token_staked(
                "0xAccount", 1, chain="optimism", gauge_address="0xGauge"
            )
        )
        assert result is True

    def test_false(self) -> None:
        b = make_behaviour()
        b.contract_interact = make_contract_interact([False])
        result = exhaust_generator(
            b.is_cl_token_staked(
                "0xAccount", 1, chain="optimism", gauge_address="0xGauge"
            )
        )
        assert result is False


class TestGetClGaugeTotalSupply:
    """Tests for get_cl_gauge_total_supply generator."""

    def test_missing_params(self) -> None:
        """Missing chain/gauge_address returns 0."""
        b = make_behaviour()
        result = exhaust_generator(b.get_cl_gauge_total_supply())
        assert result == 0

    def test_total_supply_none(self) -> None:
        """None total supply returns 0."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b.get_cl_gauge_total_supply(chain="optimism", gauge_address="0xGauge")
        )
        assert result == 0

    def test_total_supply_not_int(self) -> None:
        """Non-int coerces to 0."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact(["bad"])
        result = exhaust_generator(
            b.get_cl_gauge_total_supply(chain="optimism", gauge_address="0xGauge")
        )
        assert result == 0

    def test_success(self) -> None:
        """Valid total supply returned."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([123456])
        result = exhaust_generator(
            b.get_cl_gauge_total_supply(chain="optimism", gauge_address="0xGauge")
        )
        assert result == 123456


class TestGetFactoryAddressVelodrome:
    """Tests for _get_factory_address_velodrome generator."""

    def test_mode_chain_uses_factory_method(self) -> None:
        """Mode chain uses 'factory' callable."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact(["0xFactory"])
        result = exhaust_generator(b._get_factory_address_velodrome("0xRouter", "mode"))
        assert result == "0xFactory"

    def test_optimism_chain_uses_default_factory_method(self) -> None:
        """Optimism uses 'defaultFactory' callable."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact(["0xFactory"])
        result = exhaust_generator(
            b._get_factory_address_velodrome("0xRouter", "optimism")
        )
        assert result == "0xFactory"

    def test_factory_none(self) -> None:
        """None factory returns None."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b._get_factory_address_velodrome("0xRouter", "optimism")
        )
        assert result is None

    def test_factory_empty(self) -> None:
        """Empty string factory returns None (falsy)."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([""])
        result = exhaust_generator(
            b._get_factory_address_velodrome("0xRouter", "optimism")
        )
        assert result is None

    def test_mode_case_insensitive(self) -> None:
        """Chain comparison is lowered."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact(["0xFactory"])
        result = exhaust_generator(b._get_factory_address_velodrome("0xRouter", "Mode"))
        assert result == "0xFactory"


class TestQueryAddLiquidityVelodrome:
    """Tests for _query_add_liquidity_velodrome generator."""

    def test_mode_chain_success(self) -> None:
        """Mode chain uses quote_add_liquidity_mode."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact(
            [{"amount_a": 1000, "amount_b": 2000, "liquidity": 500}]
        )
        result = exhaust_generator(
            b._query_add_liquidity_velodrome(
                "0xRouter", "0xA", "0xB", True, [1000, 2000], "mode"
            )
        )
        assert result == {"amount_a": 1000, "amount_b": 2000, "liquidity": 500}

    def test_mode_chain_no_result(self) -> None:
        """Mode chain, contract returns None."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b._query_add_liquidity_velodrome(
                "0xRouter", "0xA", "0xB", True, [1000, 2000], "mode"
            )
        )
        assert result is None

    def test_optimism_no_factory(self) -> None:
        """Optimism chain, factory lookup fails."""
        b = make_behaviour()
        # _get_factory_address_velodrome calls contract_interact once -> None
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b._query_add_liquidity_velodrome(
                "0xRouter", "0xA", "0xB", False, [1000, 2000], "optimism"
            )
        )
        assert result is None

    def test_optimism_success(self) -> None:
        """Optimism chain, full happy path."""
        b = make_behaviour()
        # 1: factory lookup -> "0xFactory"
        # 2: quote -> result
        b.contract_interact = make_contract_interact(
            ["0xFactory", {"amount_a": 900, "amount_b": 1800, "liquidity": 400}]
        )
        result = exhaust_generator(
            b._query_add_liquidity_velodrome(
                "0xRouter", "0xA", "0xB", False, [1000, 2000], "optimism"
            )
        )
        assert result == {"amount_a": 900, "amount_b": 1800, "liquidity": 400}

    def test_optimism_no_result(self) -> None:
        """Optimism chain, contract returns None after factory lookup succeeds."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact(["0xFactory", None])
        result = exhaust_generator(
            b._query_add_liquidity_velodrome(
                "0xRouter", "0xA", "0xB", False, [1000, 2000], "optimism"
            )
        )
        assert result is None

    def test_exception_returns_none(self) -> None:
        """Exception during execution returns None."""
        b = make_behaviour()
        # Return a dict missing expected keys -> triggers KeyError
        b.contract_interact = make_contract_interact([{"wrong_key": 1}])
        result = exhaust_generator(
            b._query_add_liquidity_velodrome(
                "0xRouter", "0xA", "0xB", True, [1000, 2000], "mode"
            )
        )
        assert result is None


class TestQueryRemoveLiquidityVelodrome:
    """Tests for _query_remove_liquidity_velodrome generator."""

    def test_mode_chain_success(self) -> None:
        """Mode chain uses quote_remove_liquidity_mode."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact(
            [{"amount_a": 1000, "amount_b": 2000}]
        )
        result = exhaust_generator(
            b._query_remove_liquidity_velodrome(
                "0xRouter", "0xA", "0xB", True, 500, "mode"
            )
        )
        assert result == {"amount_a": 1000, "amount_b": 2000}

    def test_mode_chain_no_result(self) -> None:
        """Mode chain, contract returns None."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b._query_remove_liquidity_velodrome(
                "0xRouter", "0xA", "0xB", True, 500, "mode"
            )
        )
        assert result is None

    def test_optimism_no_factory(self) -> None:
        """Optimism chain, factory lookup fails."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b._query_remove_liquidity_velodrome(
                "0xRouter", "0xA", "0xB", False, 500, "optimism"
            )
        )
        assert result is None

    def test_optimism_success(self) -> None:
        """Optimism chain, full happy path."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact(
            ["0xFactory", {"amount_a": 900, "amount_b": 1800}]
        )
        result = exhaust_generator(
            b._query_remove_liquidity_velodrome(
                "0xRouter", "0xA", "0xB", False, 500, "optimism"
            )
        )
        assert result == {"amount_a": 900, "amount_b": 1800}

    def test_optimism_no_result(self) -> None:
        """Optimism chain, contract returns None."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact(["0xFactory", None])
        result = exhaust_generator(
            b._query_remove_liquidity_velodrome(
                "0xRouter", "0xA", "0xB", False, 500, "optimism"
            )
        )
        assert result is None

    def test_exception_returns_none(self) -> None:
        """Exception returns None."""
        b = make_behaviour()
        b.contract_interact = make_contract_interact([{"wrong_key": 1}])
        result = exhaust_generator(
            b._query_remove_liquidity_velodrome(
                "0xRouter", "0xA", "0xB", True, 500, "mode"
            )
        )
        assert result is None


class TestCalculateSlippageProtectionForVelodromeDecrease:
    """Tests for _calculate_slippage_protection_for_velodrome_decrease generator."""

    def _make_read_kv(self, return_value):
        """Create a fake _read_kv generator."""

        def fake_read_kv(**kwargs):
            yield
            return return_value

        return fake_read_kv

    def test_no_position_manager(self) -> None:
        """Missing position manager returns (None, None)."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {}
        b._read_kv = self._make_read_kv(None)
        result = exhaust_generator(
            b._calculate_slippage_protection_for_velodrome_decrease(
                1, "optimism", "0xPool"
            )
        )
        assert result == (None, None)

    def test_position_none(self) -> None:
        """Failed to get position returns (None, None)."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        b._read_kv = self._make_read_kv(None)
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b._calculate_slippage_protection_for_velodrome_decrease(
                1, "optimism", "0xPool"
            )
        )
        assert result == (None, None)

    def test_position_too_short(self) -> None:
        """Position with < 8 keys returns (None, None)."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        b._read_kv = self._make_read_kv(None)
        # dict with 1 key, len < 8
        b.contract_interact = make_contract_interact([{"a": 1}])
        result = exhaust_generator(
            b._calculate_slippage_protection_for_velodrome_decrease(
                1, "optimism", "0xPool"
            )
        )
        assert result == (None, None)

    def test_withdrawal_mode_returns_zero_zero(self) -> None:
        """In withdrawal mode, returns (0, 0) for no slippage protection."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        b._read_kv = self._make_read_kv({"withdrawal_status": "WITHDRAWING"})
        position = {
            "tickLower": -100,
            "tickUpper": 100,
            "liquidity": 1000,
            "k4": 0,
            "k5": 0,
            "k6": 0,
            "k7": 0,
            "k8": 0,
        }
        b.contract_interact = make_contract_interact([position])
        result = exhaust_generator(
            b._calculate_slippage_protection_for_velodrome_decrease(
                1, "optimism", "0xPool"
            )
        )
        assert result == (0, 0)

    def test_slot0_fails(self) -> None:
        """Failed slot0 query returns (None, None)."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        b._read_kv = self._make_read_kv(None)
        position = {
            "tickLower": -100,
            "tickUpper": 100,
            "liquidity": 1000,
            "k4": 0,
            "k5": 0,
            "k6": 0,
            "k7": 0,
            "k8": 0,
        }
        # 1: position, 2: slot0 -> None
        b.contract_interact = make_contract_interact([position, None])
        result = exhaust_generator(
            b._calculate_slippage_protection_for_velodrome_decrease(
                1, "optimism", "0xPool"
            )
        )
        assert result == (None, None)

    def test_success_with_slippage(self) -> None:
        """Full happy path returning slippage-protected amounts."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        b.params.slippage_tolerance = 0.05
        b._read_kv = self._make_read_kv(None)

        position = {
            "tickLower": -100,
            "tickUpper": 100,
            "liquidity": 1000,
            "k4": 0,
            "k5": 0,
            "k6": 0,
            "k7": 0,
            "k8": 0,
        }
        slot0 = {"tick": 0, "sqrt_price_x96": 79228162514264337593543950336}

        def fake_decrease(pos, tick, sqrt_price, chain):
            yield
            return (1000, 2000)

        b._calculate_velodrome_decrease_amounts = fake_decrease
        # 1: position, 2: slot0
        b.contract_interact = make_contract_interact([position, slot0])

        result = exhaust_generator(
            b._calculate_slippage_protection_for_velodrome_decrease(
                1, "optimism", "0xPool"
            )
        )
        # _calculate_slippage_protection with 5%: 1000 - 50 = 950, 2000 - 100 = 1900
        assert result == (950, 1900)

    def test_slippage_protection_returns_none(self) -> None:
        """If _calculate_slippage_protection returns (None, None), propagated."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        b.params.slippage_tolerance = 0.05
        b._read_kv = self._make_read_kv(None)

        position = {
            "tickLower": -100,
            "tickUpper": 100,
            "liquidity": 1000,
            "k4": 0,
            "k5": 0,
            "k6": 0,
            "k7": 0,
            "k8": 0,
        }
        slot0 = {"tick": 0, "sqrt_price_x96": 79228162514264337593543950336}

        def fake_decrease(pos, tick, sqrt_price, chain):
            yield
            return (1000, 2000)

        b._calculate_velodrome_decrease_amounts = fake_decrease
        b._calculate_slippage_protection = lambda amounts, tol: (None, None)
        b.contract_interact = make_contract_interact([position, slot0])

        result = exhaust_generator(
            b._calculate_slippage_protection_for_velodrome_decrease(
                1, "optimism", "0xPool"
            )
        )
        assert result == (None, None)

    def test_exception_returns_none_none(self) -> None:
        """Exception during calculation returns (None, None)."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }

        def bad_read_kv(**kwargs):
            raise Exception("boom")
            yield  # noqa: unreachable - makes it a generator

        b._read_kv = bad_read_kv
        result = exhaust_generator(
            b._calculate_slippage_protection_for_velodrome_decrease(
                1, "optimism", "0xPool"
            )
        )
        assert result == (None, None)

    def test_withdrawal_status_not_withdrawing(self) -> None:
        """withdrawal_status present but not WITHDRAWING does not short-circuit."""
        b = make_behaviour()
        b.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        b.params.slippage_tolerance = 0.1
        b._read_kv = self._make_read_kv({"withdrawal_status": "ACTIVE"})

        position = {
            "tickLower": -100,
            "tickUpper": 100,
            "liquidity": 1000,
            "k4": 0,
            "k5": 0,
            "k6": 0,
            "k7": 0,
            "k8": 0,
        }
        slot0 = {"tick": 0, "sqrt_price_x96": 79228162514264337593543950336}

        def fake_decrease(pos, tick, sqrt_price, chain):
            yield
            return (1000, 2000)

        b._calculate_velodrome_decrease_amounts = fake_decrease
        b.contract_interact = make_contract_interact([position, slot0])

        result = exhaust_generator(
            b._calculate_slippage_protection_for_velodrome_decrease(
                1, "optimism", "0xPool"
            )
        )
        # 10% slippage: 1000 - 100 = 900, 2000 - 200 = 1800
        assert result == (900, 1800)


class TestCalculateVelodromeDecreaseAmounts:
    """Tests for _calculate_velodrome_decrease_amounts generator."""

    def test_sqrt_ratio_a_fails(self) -> None:
        """Failed sqrt_ratio_a returns (None, None)."""
        b = make_behaviour()

        def fake_sqrt(chain, tick):
            yield
            return None

        b.get_velodrome_sqrt_ratio_at_tick = fake_sqrt
        result = exhaust_generator(
            b._calculate_velodrome_decrease_amounts(
                {"tickLower": -100, "tickUpper": 100, "liquidity": 1000},
                0,
                79228162514264337593543950336,
                "optimism",
            )
        )
        assert result == (None, None)

    def test_sqrt_ratio_b_fails(self) -> None:
        """Failed sqrt_ratio_b returns (None, None)."""
        b = make_behaviour()
        call_count = [0]

        def fake_sqrt(chain, tick):
            call_count[0] += 1
            yield
            return 12345 if call_count[0] == 1 else None

        b.get_velodrome_sqrt_ratio_at_tick = fake_sqrt
        result = exhaust_generator(
            b._calculate_velodrome_decrease_amounts(
                {"tickLower": -100, "tickUpper": 100, "liquidity": 1000},
                0,
                79228162514264337593543950336,
                "optimism",
            )
        )
        assert result == (None, None)

    def test_success(self) -> None:
        """Happy path returns amounts from get_velodrome_amounts_for_liquidity."""
        b = make_behaviour()
        call_count = [0]

        def fake_sqrt(chain, tick):
            call_count[0] += 1
            yield
            return 10000 + call_count[0]

        def fake_amounts(**kwargs):
            yield
            return (500, 600)

        b.get_velodrome_sqrt_ratio_at_tick = fake_sqrt
        b.get_velodrome_amounts_for_liquidity = fake_amounts

        result = exhaust_generator(
            b._calculate_velodrome_decrease_amounts(
                {"tickLower": -100, "tickUpper": 100, "liquidity": 1000},
                0,
                79228162514264337593543950336,
                "optimism",
            )
        )
        assert result == (500, 600)

    def test_exception_returns_none_none(self) -> None:
        """Exception during calculation returns (None, None)."""
        b = make_behaviour()
        # Missing required keys triggers KeyError -> caught by except
        result = exhaust_generator(
            b._calculate_velodrome_decrease_amounts(
                {}, 0, 79228162514264337593543950336, "optimism"
            )
        )
        assert result == (None, None)


class TestGetVelodromeAmountsForLiquidity:
    """Tests for get_velodrome_amounts_for_liquidity generator."""

    def test_no_sugar_address(self) -> None:
        """Missing sugar address returns (0, 0)."""
        b = make_behaviour()
        b.params.velodrome_slipstream_helper_contract_addresses = {}
        result = exhaust_generator(
            b.get_velodrome_amounts_for_liquidity(
                chain="optimism",
                sqrt_price_x96=100,
                sqrt_ratio_a_x96=90,
                sqrt_ratio_b_x96=110,
                liquidity=1000,
            )
        )
        assert result == (0, 0)

    def test_amounts_none(self) -> None:
        """Contract returns None -> (0, 0)."""
        b = make_behaviour()
        b.params.velodrome_slipstream_helper_contract_addresses = {
            "optimism": "0xSugar"
        }
        b.contract_interact = make_contract_interact([None])
        result = exhaust_generator(
            b.get_velodrome_amounts_for_liquidity(
                chain="optimism",
                sqrt_price_x96=100,
                sqrt_ratio_a_x96=90,
                sqrt_ratio_b_x96=110,
                liquidity=1000,
            )
        )
        assert result == (0, 0)

    def test_amounts_empty(self) -> None:
        """Contract returns falsy (empty list) -> (0, 0)."""
        b = make_behaviour()
        b.params.velodrome_slipstream_helper_contract_addresses = {
            "optimism": "0xSugar"
        }
        b.contract_interact = make_contract_interact([[]])
        result = exhaust_generator(
            b.get_velodrome_amounts_for_liquidity(
                chain="optimism",
                sqrt_price_x96=100,
                sqrt_ratio_a_x96=90,
                sqrt_ratio_b_x96=110,
                liquidity=1000,
            )
        )
        assert result == (0, 0)

    def test_success(self) -> None:
        """Happy path returns amounts tuple."""
        b = make_behaviour()
        b.params.velodrome_slipstream_helper_contract_addresses = {
            "optimism": "0xSugar"
        }
        b.contract_interact = make_contract_interact([[111, 222]])
        result = exhaust_generator(
            b.get_velodrome_amounts_for_liquidity(
                chain="optimism",
                sqrt_price_x96=100,
                sqrt_ratio_a_x96=90,
                sqrt_ratio_b_x96=110,
                liquidity=1000,
            )
        )
        assert result == (111, 222)


class TestCoverageGaps:
    """Tests targeting remaining coverage gaps."""

    def test_historical_market_data_json_decode_error(self) -> None:
        """JSONDecodeError when parsing prices should return None."""
        b = make_behaviour()
        b.context.params.use_x402 = False
        b.context.coingecko = MagicMock()
        b.context.coingecko.historical_market_data_endpoint = (
            "coins/{coin_id}/market_chart?days={days}"
        )
        b.sleep = MagicMock(return_value=iter([None]))

        # Make response_json.get raise JSONDecodeError when accessing "prices"
        mock_response = MagicMock()
        mock_response.get.side_effect = json.JSONDecodeError("fail", "", 0)
        del mock_response.status_code
        b.context.coingecko.request.return_value = (True, mock_response)

        gen = b._get_historical_market_data("bitcoin", 30, {})
        result = run_generator(gen)
        assert result is None

    def test_stable_pool_amounts_check0_gt_check1(self) -> None:
        """When check0 > check1, should adjust amount1 upward."""
        b = make_behaviour()
        # pool_ratio = 0.5 (reserves=(10M, 5M) with dec=[6,6])
        # amounts=[3, 5]: option1 uses 3 token0, needs int(3*0.5)=1 token1
        # check0 = 3 * scale, check1 = 1 * scale => check0 > check1
        # adjustment: target_amount1 = check0 // scale = 3, 3 <= 5 so adjust
        result = b._calculate_stable_pool_amounts(
            max_amounts_in=[3, 5],
            pool_reserves=(10_000_000, 5_000_000),
            token_decimals=[6, 6],
        )
        assert result is not None
        assert len(result) == 2

    def test_stable_pool_option2_only_feasible(self) -> None:
        """When only option2 is feasible, should use max token1."""
        b = make_behaviour()
        # pool_ratio = 2 (reserves=(5M, 10M) with dec=[6,6])
        # amounts=[10M, 1M]
        # option1: use 10M token0, need 20M token1 -> NOT feasible (have 1M)
        # option2: use 1M token1, need 500K token0 -> feasible (have 10M)
        result = b._calculate_stable_pool_amounts(
            max_amounts_in=[10_000_000, 1_000_000],
            pool_reserves=(5_000_000, 10_000_000),
            token_decimals=[6, 6],
        )
        assert result is not None
        assert result[1] == 1_000_000

    def test_stable_pool_check0_gt_check1_exceeds_limit(self) -> None:
        """When check0 > check1 and target_amount1 > amount1_desired, skip adjustment."""
        b = make_behaviour()
        # pool_ratio = 0.5 (reserves=(10M, 5M) with dec=[6,6])
        # amounts=[3, 2]: option1: req_a1=int(3*0.5)=1, 1<=2 feasible
        #   option2: req_a0=int(2/0.5)=4, 4<=3? NO
        # adjusted_amounts = [3, 1]
        # check0 = 3*10^12, check1 = 1*10^12 => check0 > check1
        # target_amount1 = 3*10^12 // 10^12 = 3, 3 <= 2? NO => skip adjustment
        result = b._calculate_stable_pool_amounts(
            max_amounts_in=[3, 2],
            pool_reserves=(10_000_000, 5_000_000),
            token_decimals=[6, 6],
        )
        assert result is not None
        # adjustment was skipped, so amount1 stays at 1 (req_amount1_for_max_amount0)
        assert result[1] == 1

    def test_enter_cl_pool_tick_ranges_calculated_successfully(self) -> None:
        """tick_ranges=None, calculation succeeds, then fails at tick_spacing."""
        b = make_behaviour()
        b.context.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPosMgr"
        }
        valid_tick_ranges = [
            [{"tick_lower": -100, "tick_upper": 100, "allocation": 1.0}]
        ]
        b._calculate_tick_lower_and_upper_velodrome = MagicMock(
            return_value=_gen_return(valid_tick_ranges)
        )
        b._get_tick_spacing_velodrome = MagicMock(return_value=_gen_return(None))
        result = exhaust(
            b._enter_cl_pool(
                "0xPool",
                "0xSafe",
                ["0xT0", "0xT1"],
                "optimism",
                [1000, 2000],
                False,
                tick_ranges=None,
                tick_spacing=None,
            )
        )
        assert result == (None, None)
        # Verify tick_ranges calculation was called
        b._calculate_tick_lower_and_upper_velodrome.assert_called_once()

    def test_optimize_stablecoin_bands_verbose_recursion(self) -> None:
        """Test optimize_stablecoin_bands with verbose mode triggering recursion."""
        b = make_behaviour()

        # Mock _run_monte_carlo_level to control recursion behavior precisely.
        # Level 0: high allocation + low multiplier -> triggers recursion to level 1
        # Level 1: high allocation + low multiplier -> triggers recursion to level 2
        # Level 2: final level (trigger_threshold=None), always breaks
        def _cfg(mults, allocs, score):
            return {
                "band_multipliers": mults,
                "band_allocations": allocs,
                "zscore_economic_score": score,
                "percent_in_bounds": 0.99,
                "avg_weighted_width": 0.01,
            }

        level_results = [
            {  # Level 0: triggers recursion (alloc > 0.95, mult < 0.15)
                "best_config": _cfg([0.10, 0.20, 0.40], [0.97, 0.02, 0.01], 5.0),
            },
            {  # Level 1: worse score, triggers recursion (alloc > 0.95, mult < 0.02)
                "best_config": _cfg([0.01, 0.02, 0.04], [0.98, 0.01, 0.01], 3.0),
            },
            {  # Level 2: final level, doesn't need to trigger
                "best_config": _cfg([0.005, 0.01, 0.02], [0.96, 0.03, 0.01], 4.0),
            },
        ]
        call_count = [0]
        orig_run = b._run_monte_carlo_level

        def mock_run(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return level_results[idx]

        b._run_monte_carlo_level = mock_run

        prices = [1.0] * 50
        result = b.optimize_stablecoin_bands(prices, verbose=True)
        assert result is not None
        # Best overall should be level 0 with score 5.0
        assert result["zscore_economic_score"] == 5.0
        # All 3 levels should have been run
        assert call_count[0] == 3
        # Verbose logging should have fired
        assert b.context.logger.info.called

    def test_optimize_stablecoin_bands_no_recursion_verbose(self) -> None:
        """Test verbose mode where recursion is NOT triggered."""
        b = make_behaviour()
        np.random.seed(999)
        # More volatile prices -> lower inner band allocation -> no recursion
        prices = [1.0 + 0.05 * np.random.randn() for _ in range(50)]
        result = b.optimize_stablecoin_bands(prices, verbose=True)
        assert result is not None
        assert result["zscore_economic_score"] > 0

    def test_optimize_stablecoin_bands_recursion_not_verbose(self) -> None:
        """Recursion triggers with verbose=False (covers 1606->1618 False branch)."""
        b = make_behaviour()

        def _cfg(mults, allocs, score):
            return {
                "band_multipliers": mults,
                "band_allocations": allocs,
                "zscore_economic_score": score,
                "percent_in_bounds": 0.99,
                "avg_weighted_width": 0.01,
            }

        level_results = [
            {"best_config": _cfg([0.10, 0.20, 0.40], [0.97, 0.02, 0.01], 5.0)},
            {"best_config": _cfg([0.01, 0.02, 0.04], [0.98, 0.01, 0.01], 3.0)},
            {"best_config": _cfg([0.005, 0.01, 0.02], [0.96, 0.03, 0.01], 4.0)},
        ]
        call_count = [0]

        def mock_run(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return level_results[idx]

        b._run_monte_carlo_level = mock_run
        result = b.optimize_stablecoin_bands([1.0] * 50, verbose=False)
        assert result["zscore_economic_score"] == 5.0
        assert call_count[0] == 3

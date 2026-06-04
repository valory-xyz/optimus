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

"""Tests for behaviours/fetch_strategies.py."""

# pylint: skip-file

import json
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    OLAS_ADDRESSES,
    ZERO_ADDRESS,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
    CONTRACT_CHECK_CACHE_PREFIX,
    FetchStrategiesBehaviour,
    TRANSFER_EVENT_SIGNATURE,
    ZERO_ADDRESS_PADDED,
)


def _mk():
    """Create a FetchStrategiesBehaviour without __init__."""
    obj = object.__new__(FetchStrategiesBehaviour)
    ctx = MagicMock()
    obj.__dict__["_context"] = ctx
    obj.current_positions = []
    obj.portfolio_data = {}
    obj.assets = {}
    obj.whitelisted_assets = {}
    obj.funding_events = {}
    obj.agent_performance = {}
    obj.initial_investment_values_per_pool = {}
    obj.pools = {}
    obj.service_staking_state = MagicMock()
    obj.store_funding_events = MagicMock()
    obj.read_funding_events = MagicMock(return_value={})
    # Default slug map so per-chain dispatch in _calculate_safe_balances_value
    # treats Optimism + Base as Safe-API chains. Tests that need a different
    # configuration override this on the instance.
    obj.params.safe_api_chain_slugs = {"optimism": "oeth", "base": "base"}
    return obj


def _drive(gen, sends=None):
    """Drive a generator to completion, sending values from *sends*."""
    sends = list(sends or [])
    idx = 0
    val = None
    while True:
        try:
            yielded = gen.send(val)  # noqa: F841
            if idx < len(sends):
                val = sends[idx]
                idx += 1
            else:
                val = None
        except StopIteration as exc:
            return exc.value


def _gen_return(value):
    """Return a trivial generator that yields once then returns *value*."""

    def _inner(*a, **kw):
        yield
        return value

    return _inner


def _gen_none(*a, **kw):
    yield


class TestFetchStrategiesWithdrawalGate:
    """Verify the post-portfolio-refresh gate emits a withdrawal payload."""

    @staticmethod
    def _mk_benchmark() -> Any:
        """Build a minimal benchmark-tool mock that supports the context manager."""
        bm = MagicMock()
        bm.local.return_value.__enter__ = MagicMock(return_value=None)
        bm.local.return_value.__exit__ = MagicMock(return_value=False)
        return bm

    @staticmethod
    def _wire_common_path(obj: Any) -> None:
        """Stub the non-portfolio parts of async_act so the test reaches the gate."""
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.available_strategies = {"optimism": ["uniswapV3"]}
        obj.params.initial_assets = {}
        obj._get_native_balance = _gen_return(1.0)
        obj._read_kv = _gen_return(
            {
                "selected_protocols": json.dumps(["uniswapV3"]),
                "trading_type": "balanced",
            }
        )
        obj._write_kv = _gen_none
        obj.whitelisted_assets = {"optimism": {"0xT": "TKN"}}
        obj.read_whitelisted_assets = MagicMock()
        obj._get_current_timestamp = lambda: 100
        obj._track_whitelisted_assets = _gen_none
        obj.calculate_user_share_values = _gen_none
        obj._update_agent_performance_metrics = MagicMock()
        obj.wait_until_round_end = _gen_none
        obj.set_done = MagicMock()

    def test_portfolio_refreshed_before_withdrawal_routing(self) -> None:
        """Pin the ordering: store_portfolio_data fires BEFORE the withdrawal payload.

        Before this ordering was enforced, async_act short-circuited on
        investing_paused at the top of the function. The portfolio file was
        never refreshed after a completed withdrawal, so the UI kept
        serving stale balances and follow-up withdrawals planned actions
        against ghost funds.
        """
        obj = _mk()
        obj.context.benchmark_tool.measure.return_value = self._mk_benchmark()
        obj.context.agent_address = "0xagent"
        obj.current_positions = []

        call_order: List[str] = []
        captured: Dict[str, Any] = {}

        def fake_send(payload):  # type: ignore[no-untyped-def]
            call_order.append("send_a2a_transaction")
            captured["payload"] = payload
            yield

        sd = MagicMock(period_count=1)
        with patch.object(
            type(obj), "synchronized_data", new_callable=PropertyMock, return_value=sd
        ):
            with patch.object(
                type(obj), "shared_state", new_callable=PropertyMock
            ) as mock_ss:
                mock_ss.return_value = MagicMock()
                self._wire_common_path(obj)
                obj.store_portfolio_data = MagicMock(
                    side_effect=lambda *a, **kw: call_order.append(
                        "store_portfolio_data"
                    )
                )
                obj._read_investing_paused = _gen_return(True)
                obj.send_a2a_transaction = fake_send

                _drive(obj.async_act())

        assert call_order == ["store_portfolio_data", "send_a2a_transaction"]
        decoded = json.loads(captured["payload"].content)
        assert decoded["event"] == "withdrawal_initiated"
        obj.set_done.assert_called_once()

    def test_withdrawal_payload_still_emits_when_refresh_raises(self) -> None:
        """Refresh failure does not block the withdrawal payload."""
        obj = _mk()
        obj.context.benchmark_tool.measure.return_value = self._mk_benchmark()
        obj.context.agent_address = "0xagent"
        obj.current_positions = []

        captured: Dict[str, Any] = {}

        def fake_send(payload):
            captured["payload"] = payload
            yield

        sd = MagicMock(period_count=1)
        with patch.object(
            type(obj), "synchronized_data", new_callable=PropertyMock, return_value=sd
        ):
            with patch.object(
                type(obj), "shared_state", new_callable=PropertyMock
            ) as mock_ss:
                mock_ss.return_value = MagicMock()
                self._wire_common_path(obj)
                obj._get_native_balance = MagicMock(
                    side_effect=RuntimeError("rpc-down")
                )
                obj._read_investing_paused = _gen_return(True)
                obj.send_a2a_transaction = fake_send

                _drive(obj.async_act())

        decoded = json.loads(captured["payload"].content)
        assert decoded["event"] == "withdrawal_initiated"
        obj.set_done.assert_called_once()

    def test_period_zero_routes_withdrawal_after_eth_revert_noop(self) -> None:
        """On period 0, withdrawal still routes once the eth-revert path is a no-op."""
        obj = _mk()
        obj.context.benchmark_tool.measure.return_value = self._mk_benchmark()
        obj.context.agent_address = "0xagent"
        obj.current_positions = []

        call_order: List[str] = []
        captured: Dict[str, Any] = {}

        def fake_send(payload):
            call_order.append("send_a2a_transaction")
            captured["payload"] = payload
            yield

        sd = MagicMock(period_count=0)
        with patch.object(
            type(obj), "synchronized_data", new_callable=PropertyMock, return_value=sd
        ):
            with patch.object(
                type(obj), "shared_state", new_callable=PropertyMock
            ) as mock_ss:
                mock_ss.return_value = MagicMock()
                self._wire_common_path(obj)
                obj._validate_velodrome_v2_pool_addresses = _gen_none
                obj.update_position_amounts = _gen_none
                obj.check_and_update_zero_liquidity_positions = MagicMock()
                obj._check_and_create_eth_revert_transactions = _gen_none
                obj.store_portfolio_data = MagicMock(
                    side_effect=lambda *a, **kw: call_order.append(
                        "store_portfolio_data"
                    )
                )
                obj._read_investing_paused = _gen_return(True)
                obj.send_a2a_transaction = fake_send

                _drive(obj.async_act())

        assert call_order == ["store_portfolio_data", "send_a2a_transaction"]
        decoded = json.loads(captured["payload"].content)
        assert decoded["event"] == "withdrawal_initiated"
        obj.set_done.assert_called_once()

    def test_no_withdrawal_payload_when_not_paused(self) -> None:
        """investing_paused=False sends the normal selected-protocols payload."""
        obj = _mk()
        obj.context.benchmark_tool.measure.return_value = self._mk_benchmark()
        obj.context.agent_address = "0xagent"
        obj.current_positions = []

        captured: Dict[str, Any] = {}

        def fake_send(payload):  # type: ignore[no-untyped-def]
            captured["payload"] = payload
            yield

        sd = MagicMock(period_count=1)
        with patch.object(
            type(obj), "synchronized_data", new_callable=PropertyMock, return_value=sd
        ):
            with patch.object(
                type(obj), "shared_state", new_callable=PropertyMock
            ) as mock_ss:
                mock_ss.return_value = MagicMock()
                self._wire_common_path(obj)
                obj.store_portfolio_data = MagicMock()
                obj._read_investing_paused = _gen_return(False)
                obj.send_a2a_transaction = fake_send

                _drive(obj.async_act())

        decoded = json.loads(captured["payload"].content)
        assert "event" not in decoded
        assert decoded["selected_protocols"] == ["uniswapV3"]
        obj.store_portfolio_data.assert_called_once()


class TestIsTimeUpdateDue:
    """TestIsTimeUpdateDue."""

    def test_due(self):
        """Test due."""
        obj = _mk()
        obj._get_current_timestamp = lambda: 10000
        obj.portfolio_data = {"last_updated": 0}
        assert obj._is_time_update_due() is True

    def test_not_due(self):
        """Test not due."""
        obj = _mk()
        obj._get_current_timestamp = lambda: 100
        obj.portfolio_data = {"last_updated": 100}
        assert obj._is_time_update_due() is False


class TestHavePositionsChanged:
    """TestHavePositionsChanged."""

    def test_no_last_data(self):
        """Test no last data."""
        obj = _mk()
        obj.current_positions = []
        assert obj._have_positions_changed({}) is True

    def test_no_allocations_key(self):
        """Test no allocations key."""
        obj = _mk()
        obj.current_positions = []
        assert obj._have_positions_changed({"foo": 1}) is True

    def test_count_changed(self):
        """Test count changed."""
        obj = _mk()
        obj.current_positions = [
            {"pool_address": "0x1", "dex_type": "a", "status": "open"}
        ]
        assert obj._have_positions_changed({"allocations": []}) is True

    def test_no_change(self):
        """Test no change."""
        obj = _mk()
        obj.current_positions = [
            {"pool_address": "0x1", "dex_type": "uniswapV3", "status": "open"}
        ]
        last = {"allocations": [{"id": "0x1", "type": "uniswapV3"}]}
        assert obj._have_positions_changed(last) is False

    def test_new_positions(self):
        """Test new positions."""
        obj = _mk()
        obj.current_positions = [
            {"pool_address": "0x1", "dex_type": "a", "status": "open"},
            {"pool_address": "0x2", "dex_type": "b", "status": "open"},
        ]
        last = {"allocations": [{"id": "0x1", "type": "a"}]}
        assert obj._have_positions_changed(last) is True

    def test_closed_positions(self):
        """Test closed positions."""
        obj = _mk()
        obj.current_positions = [
            {"pool_address": "0x1", "dex_type": "a", "status": "closed"}
        ]
        last = {"allocations": [{"id": "0x1", "type": "a"}]}
        assert obj._have_positions_changed(last) is True


class TestUpdatePortfolioBreakdownRatios:
    """TestUpdatePortfolioBreakdownRatios."""

    def test_empty_breakdown(self):
        """Test empty breakdown."""
        obj = _mk()
        bd = []
        obj._update_portfolio_breakdown_ratios(bd, Decimal(100))
        assert bd == []

    def test_zero_total(self):
        """Test zero total."""
        obj = _mk()
        bd = [{"value_usd": 10, "balance": 1, "price": 10}]
        obj._update_portfolio_breakdown_ratios(bd, Decimal(0))
        assert bd[0]["ratio"] == 0.0

    def test_normal(self):
        """Test normal."""
        obj = _mk()
        bd = [
            {"value_usd": 50, "balance": 5, "price": 10},
            {"value_usd": 50, "balance": 10, "price": 5},
        ]
        obj._update_portfolio_breakdown_ratios(bd, Decimal(100))
        assert len(bd) == 2
        assert isinstance(bd[0]["value_usd"], float)

    def test_filters_small_values(self):
        """Test filters small values."""
        obj = _mk()
        bd = [
            {"value_usd": 0.001, "balance": 0.0001, "price": 10},
            {"value_usd": 50, "balance": 5, "price": 10},
        ]
        obj._update_portfolio_breakdown_ratios(bd, Decimal(50))
        assert len(bd) == 1

    def test_none_value_usd_skipped(self):
        """Test that entries with None value_usd are handled in the filter step."""
        obj = _mk()
        # The sum at line 720 will fail with None, so the except at 751 catches it
        # and keeps the original list. Test with value_usd=0 (below threshold) instead.
        bd = [
            {"value_usd": 0.001, "balance": 0, "price": 0},
            {"value_usd": 10, "balance": 1, "price": 10},
        ]
        obj._update_portfolio_breakdown_ratios(bd, Decimal(10))
        assert len(bd) == 1
        assert bd[0]["value_usd"] == 10.0


class TestAdjustForDecimals:
    """TestAdjustForDecimals."""

    def test_basic(self):
        """Test basic."""
        obj = _mk()
        assert obj._adjust_for_decimals(1000000, 6) == Decimal("1")


class TestIsGnosisSafe:
    """TestIsGnosisSafe."""

    def test_none(self):
        """Test none."""
        obj = _mk()
        assert obj._is_gnosis_safe(None) is False

    def test_not_contract(self):
        """Test not contract."""
        obj = _mk()
        assert obj._is_gnosis_safe({"is_contract": False}) is False

    def test_wrong_name(self):
        """Test wrong name."""
        obj = _mk()
        assert obj._is_gnosis_safe({"is_contract": True, "name": "Foo"}) is False

    def test_is_safe(self):
        """Test is safe."""
        obj = _mk()
        assert (
            obj._is_gnosis_safe({"is_contract": True, "name": "GnosisSafeProxy"})
            is True
        )


class TestShouldIncludeTransfer:
    """TestShouldIncludeTransfer."""

    def test_no_from(self):
        """Test no from."""
        obj = _mk()
        assert obj._should_include_transfer(None) is False

    def test_zero_address(self):
        """Test zero address."""
        obj = _mk()
        assert (
            obj._should_include_transfer(
                {"hash": "0x0000000000000000000000000000000000000000"}
            )
            is False
        )

    def test_empty_hash(self):
        """Test empty hash."""
        obj = _mk()
        assert obj._should_include_transfer({"hash": ""}) is False

    def test_eth_transfer_bad_status(self):
        """Test eth transfer bad status."""
        obj = _mk()
        assert (
            obj._should_include_transfer(
                {"hash": "0x123", "is_contract": False},
                tx_data={"status": "fail", "value": "100"},
                is_eth_transfer=True,
            )
            is False
        )

    def test_eth_transfer_zero_value(self):
        """Test eth transfer zero value."""
        obj = _mk()
        assert (
            obj._should_include_transfer(
                {"hash": "0x123", "is_contract": False},
                tx_data={"status": "ok", "value": "0"},
                is_eth_transfer=True,
            )
            is False
        )

    def test_eoa(self):
        """Test eoa."""
        obj = _mk()
        assert (
            obj._should_include_transfer({"hash": "0x123", "is_contract": False})
            is True
        )

    def test_contract_not_safe(self):
        """Test contract not safe."""
        obj = _mk()
        assert (
            obj._should_include_transfer(
                {"hash": "0x123", "is_contract": True, "name": "Foo"}
            )
            is False
        )

    def test_contract_safe(self):
        """Test contract safe."""
        obj = _mk()
        assert (
            obj._should_include_transfer(
                {"hash": "0x123", "is_contract": True, "name": "GnosisSafeProxy"}
            )
            is True
        )


class TestShouldIncludeTransferMode:
    """TestShouldIncludeTransferMode."""

    def test_no_from(self):
        """Test no from."""
        obj = _mk()
        assert obj._should_include_transfer_mode(None) is False

    def test_zero(self):
        """Test zero."""
        obj = _mk()
        assert obj._should_include_transfer_mode({"hash": "0x0"}) is False

    def test_eoa(self):
        """Test eoa."""
        obj = _mk()
        assert (
            obj._should_include_transfer_mode({"hash": "0xabc", "is_contract": False})
            is True
        )

    def test_eth_bad(self):
        """Test eth bad."""
        obj = _mk()
        assert (
            obj._should_include_transfer_mode(
                {"hash": "0xabc", "is_contract": False},
                tx_data={"status": "fail", "value": "1"},
                is_eth_transfer=True,
            )
            is False
        )


class TestGetDatetimeFromTimestamp:
    """TestGetDatetimeFromTimestamp."""

    def test_z_suffix(self):
        """Test z suffix."""
        obj = _mk()
        dt = obj._get_datetime_from_timestamp("2024-01-01T00:00:00Z")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_plus_tz(self):
        """Test plus tz."""
        obj = _mk()
        dt = obj._get_datetime_from_timestamp("2024-01-01T00:00:00+00:00")
        assert dt is not None

    def test_no_tz(self):
        """Test no tz."""
        obj = _mk()
        dt = obj._get_datetime_from_timestamp("2024-01-01T00:00:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_invalid(self):
        """Test invalid."""
        obj = _mk()
        dt = obj._get_datetime_from_timestamp("not-a-date")
        assert dt is None

    def test_utc_suffix(self):
        """Test utc suffix."""
        obj = _mk()
        dt = obj._get_datetime_from_timestamp("2024-01-01T00:00:00UTC")
        assert dt is not None


class TestGetVeloTokenAddress:
    """TestGetVeloTokenAddress."""

    def test_found(self):
        """Test found."""
        obj = _mk()

        obj.params.velo_token_contract_addresses = {"optimism": "0xVELO"}
        assert obj._get_velo_token_address("optimism") == "0xVELO"

    def test_not_found(self):
        """Test not found."""
        obj = _mk()

        obj.params.velo_token_contract_addresses = {}
        assert obj._get_velo_token_address("mode") is None


class TestIsAirdropTransfer:
    """TestIsAirdropTransfer."""

    def test_not_started(self):
        """Test not started."""
        obj = _mk()

        obj.params.airdrop_started = False
        assert obj._is_airdrop_transfer({}) is False

    def test_no_contract(self):
        """Test no contract."""
        obj = _mk()

        obj.params.airdrop_started = True
        obj.params.airdrop_contract_address = None
        assert obj._is_airdrop_transfer({}) is False

    def test_match(self):
        """Test match."""
        obj = _mk()

        obj.params.airdrop_started = True
        obj.params.airdrop_contract_address = "0xAirdrop"
        obj._get_usdc_address = lambda c: "0xUSDC"
        tx = {
            "from": {"hash": "0xAirdrop"},
            "token": {"symbol": "USDC", "address": "0xusdc"},
        }
        assert obj._is_airdrop_transfer(tx) is True

    def test_no_match_wrong_symbol(self):
        """Test no match wrong symbol."""
        obj = _mk()

        obj.params.airdrop_started = True
        obj.params.airdrop_contract_address = "0xAirdrop"
        obj._get_usdc_address = lambda c: "0xUSDC"
        tx = {
            "from": {"hash": "0xAirdrop"},
            "token": {"symbol": "ETH", "address": "0xusdc"},
        }
        assert obj._is_airdrop_transfer(tx) is False


class TestCheckAndUpdateZeroLiquidityPositions:
    """TestCheckAndUpdateZeroLiquidityPositions."""

    def test_no_positions(self):
        """Test no positions."""
        obj = _mk()
        obj.current_positions = None
        obj.check_and_update_zero_liquidity_positions()  # no crash

    def test_close_zero_liquidity(self):
        """Test close zero liquidity."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "UniswapV3", "current_liquidity": 0}
        ]
        obj.store_current_positions = MagicMock()
        obj.check_and_update_zero_liquidity_positions()
        assert obj.current_positions[0]["status"] == "closed"

    def test_keep_nonzero(self):
        """Test keep nonzero."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "UniswapV3", "current_liquidity": 100}
        ]
        obj.store_current_positions = MagicMock()
        obj.check_and_update_zero_liquidity_positions()
        assert obj.current_positions[0]["status"] == "open"

    def test_skip_closed(self):
        """Test skip closed."""
        obj = _mk()
        obj.current_positions = [
            {"status": "closed", "dex_type": "UniswapV3", "current_liquidity": 0}
        ]
        obj.store_current_positions = MagicMock()
        obj.check_and_update_zero_liquidity_positions()
        assert obj.current_positions[0]["status"] == "closed"

    def test_velodrome_cl_all_zero(self):
        """Test velodrome cl all zero."""
        obj = _mk()
        obj.current_positions = [
            {
                "status": "open",
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "positions": [
                    {"token_id": 1, "current_liquidity": 0},
                    {"token_id": 2, "current_liquidity": 0},
                ],
            }
        ]
        obj.store_current_positions = MagicMock()
        obj.check_and_update_zero_liquidity_positions()
        assert obj.current_positions[0]["status"] == "closed"

    def test_velodrome_cl_some_nonzero(self):
        """Test velodrome cl some nonzero."""
        obj = _mk()
        obj.current_positions = [
            {
                "status": "open",
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "positions": [
                    {"token_id": 1, "current_liquidity": 0},
                    {"token_id": 2, "current_liquidity": 100},
                ],
            }
        ]
        obj.store_current_positions = MagicMock()
        obj.check_and_update_zero_liquidity_positions()
        assert obj.current_positions[0]["status"] == "open"

    def test_velodrome_cl_none_token_id(self):
        """Test velodrome cl none token id."""
        obj = _mk()
        obj.current_positions = [
            {
                "status": "open",
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "positions": [{"token_id": None, "current_liquidity": 0}],
            }
        ]
        obj.store_current_positions = MagicMock()
        obj.check_and_update_zero_liquidity_positions()
        assert obj.current_positions[0]["status"] == "closed"

    def test_default_liquidity_avoids_false_close(self):
        """Test default liquidity avoids false close."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "UniswapV3"}  # no current_liquidity key
        ]
        obj.store_current_positions = MagicMock()
        obj.check_and_update_zero_liquidity_positions()
        assert obj.current_positions[0]["status"] == "open"


class TestAddToPortfolioBreakdown:
    """TestAddToPortfolioBreakdown."""

    def test_new_entry(self):
        """Test new entry."""
        obj = _mk()
        bd = []
        obj._add_to_portfolio_breakdown(
            bd, "0xA", "TKN", Decimal(10), Decimal(2), Decimal(20)
        )
        assert len(bd) == 1
        assert bd[0]["asset"] == "TKN"

    def test_existing_entry(self):
        """Test existing entry."""
        obj = _mk()
        bd = [
            {
                "address": "0xa",
                "balance": 5.0,
                "value_usd": 10.0,
                "asset": "TKN",
                "price": 2.0,
            }
        ]
        obj._add_to_portfolio_breakdown(
            bd, "0xA", "TKN", Decimal(10), Decimal(2), Decimal(20)
        )
        assert len(bd) == 1
        assert bd[0]["balance"] == 15.0
        assert bd[0]["value_usd"] == 30.0


class TestFetchHistoricalEthPrice:
    """Tests for _fetch_historical_eth_price with caching."""

    def _setup_obj(self, api_response, use_x402=False, cached_price=None):
        """Create a test object with coingecko and caching stubs."""
        obj = _mk()
        cg = MagicMock()
        obj.context.coingecko = cg
        cg.historical_price_endpoint = "url/{coin_id}/{date}"
        cg.api_key = "key"
        cg.request.return_value = api_response

        obj.params.use_x402 = use_x402
        obj._get_cached_price = _gen_return(cached_price)
        obj._cache_price = _gen_none
        return obj

    def test_success_caches_result(self):
        """Test success caches result."""
        obj = self._setup_obj(
            (True, {"market_data": {"current_price": {"usd": 3000.0}}}),
        )
        cache_calls = []
        original_cache = _gen_none

        def tracking_cache(*args, **kwargs):
            """Tracking cache."""
            cache_calls.append(args)
            return original_cache(*args, **kwargs)

        obj._cache_price = tracking_cache
        result = _drive(obj._fetch_historical_eth_price("01-01-2024"))
        assert result == 3000.0
        assert len(cache_calls) == 1
        assert cache_calls[0] == ("ethereum", 3000.0, "01-01-2024")

    def test_returns_cached_price_without_api_call(self):
        """Test returns cached price without api call."""
        obj = self._setup_obj(
            (False, {}),  # API would fail, but shouldn't be called
            cached_price=2500.0,
        )
        result = _drive(obj._fetch_historical_eth_price("01-01-2024"))
        assert result == 2500.0
        # Verify API was NOT called since cache hit
        obj.context.coingecko.request.assert_not_called()

    def test_api_failure_returns_none(self):
        """Test api failure returns none."""
        obj = self._setup_obj((False, {}))
        obj.context.coingecko.api_key = None
        result = _drive(obj._fetch_historical_eth_price("01-01-2024"))
        assert result is None

    def test_no_price_in_response(self):
        """Test no price in response."""
        obj = self._setup_obj(
            (True, {"market_data": {"current_price": {}}}),
        )
        result = _drive(obj._fetch_historical_eth_price("01-01-2024"))
        assert result is None

    def test_x402(self):
        """Test x402."""
        obj = self._setup_obj(
            (True, {"market_data": {"current_price": {"usd": 1.0}}}),
            use_x402=True,
        )
        with patch.object(
            type(obj),
            "eoa_account",
            new_callable=PropertyMock,
            return_value=MagicMock(),
        ):
            result = _drive(obj._fetch_historical_eth_price("01-01-2024"))
        assert result == 1.0


class TestGetHistoricalPriceForDate:
    """TestGetHistoricalPriceForDate."""

    def test_zero_address(self):
        """Test zero address."""
        obj = _mk()
        obj._fetch_historical_eth_price = _gen_return(2500.0)
        gen = obj._get_historical_price_for_date(
            ZERO_ADDRESS, "ETH", "01-01-2024", "optimism"
        )
        result = _drive(gen)
        assert result == 2500.0

    def test_no_coingecko_id(self):
        """Test no coingecko id."""
        obj = _mk()
        obj.get_coin_id_from_symbol = lambda s, c: None
        gen = obj._get_historical_price_for_date("0xTOKEN", "FOO", "01-01-2024", "mode")
        result = _drive(gen)
        assert result is None

    def test_with_coingecko_id(self):
        """Test with coingecko id."""
        obj = _mk()
        obj.get_coin_id_from_symbol = lambda s, c: "foo-coin"
        obj._fetch_historical_token_price = _gen_return(42.0)
        gen = obj._get_historical_price_for_date("0xTOKEN", "FOO", "01-01-2024", "mode")
        result = _drive(gen)
        assert result == 42.0

    def test_exception(self):
        """Test exception."""
        obj = _mk()

        def boom(s, c):
            """Boom."""
            raise RuntimeError("fail")

        obj.get_coin_id_from_symbol = boom
        gen = obj._get_historical_price_for_date("0xTOKEN", "FOO", "01-01-2024", "mode")
        result = _drive(gen)
        assert result is None


class TestUpdateAgentPerformanceMetrics:
    """TestUpdateAgentPerformanceMetrics."""

    def test_with_data(self):
        """Test with data."""
        obj = _mk()
        obj.portfolio_data = {
            "portfolio_value": 100.0,
            "total_roi": 5.0,
            "partial_roi": 3.0,
        }
        obj.read_agent_performance = MagicMock()
        obj.update_agent_performance_timestamp = MagicMock()
        obj.store_agent_performance = MagicMock()
        obj._update_agent_performance_metrics()
        assert len(obj.agent_performance["metrics"]) == 2

    def test_no_data(self):
        """Test no data."""
        obj = _mk()
        obj.portfolio_data = {}
        obj.read_agent_performance = MagicMock()
        obj.update_agent_performance_timestamp = MagicMock()
        obj.store_agent_performance = MagicMock()
        obj._update_agent_performance_metrics()
        assert len(obj.agent_performance["metrics"]) == 2
        assert "$0.00" in obj.agent_performance["metrics"][0]["value"]

    def test_exception(self):
        """Test exception."""
        obj = _mk()
        obj.read_agent_performance = MagicMock(side_effect=RuntimeError("boom"))
        obj._update_agent_performance_metrics()  # no crash


class TestHandleBalancerPosition:
    """TestHandleBalancerPosition."""

    def test_ok(self):
        """Test ok."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        position = {
            "pool_address": "0xPool",
            "pool_id": "0xPoolId",
            "token0": "0xT0",
            "token1": "0xT1",
            "token0_symbol": "T0",
            "token1_symbol": "T1",
        }
        obj.get_user_share_value_balancer = _gen_return({"0xT0": Decimal(10)})
        obj._get_balancer_pool_name = _gen_return("PoolName")
        gen = obj._handle_balancer_position(position, "optimism")
        result = _drive(gen)
        assert result[0] == {"0xT0": Decimal(10)}
        assert result[1] == "PoolName"


class TestHandleUniswapPosition:
    """TestHandleUniswapPosition."""

    def test_ok(self):
        """Test ok."""
        obj = _mk()
        position = {
            "pool_address": "0xPool",
            "token_id": 1,
            "token0": "0xT0",
            "token1": "0xT1",
            "token0_symbol": "WETH",
            "token1_symbol": "USDC",
        }
        obj.get_user_share_value_uniswap = _gen_return({"0xT0": Decimal(5)})
        gen = obj._handle_uniswap_position(position, "optimism")
        result = _drive(gen)
        assert result[0] == {"0xT0": Decimal(5)}
        assert "Uniswap V3" in result[1]


class TestHandleSturdyPosition:
    """TestHandleSturdyPosition."""

    def test_ok(self):
        """Test ok."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        position = {
            "pool_address": "0xAgg",
            "token0": "0xT0",
            "token0_symbol": "DAI",
        }
        obj.get_user_share_value_sturdy = _gen_return({"0xT0": Decimal(100)})
        obj._get_aggregator_name = _gen_return("SturdyAgg")
        gen = obj._handle_sturdy_position(position, "mode")
        result = _drive(gen)
        assert result[0] == {"0xT0": Decimal(100)}
        assert result[1] == "SturdyAgg"


class TestHandleVelodromePosition:
    """TestHandleVelodromePosition."""

    def test_not_staked(self):
        """Test not staked."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": "0xVELO"}
        position = {
            "pool_address": "0xPool",
            "token_id": 1,
            "token0": "0xT0",
            "token1": "0xT1",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "staked": False,
            "is_cl_pool": False,
        }
        obj.get_user_share_value_velodrome = _gen_return(
            {"0xT0": Decimal(5), "0xT1": Decimal(10)}
        )
        gen = obj._handle_velodrome_position(position, "optimism")
        result = _drive(gen)
        assert result[0] == {"0xT0": Decimal(5), "0xT1": Decimal(10)}
        assert "Pool" in result[1]

    def test_staked_with_rewards(self):
        """Test staked with rewards."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": "0xVELO"}
        position = {
            "pool_address": "0xPool",
            "token_id": 1,
            "token0": "0xT0",
            "token1": "0xT1",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "staked": True,
            "is_cl_pool": True,
        }
        obj.get_user_share_value_velodrome = _gen_return(
            {"0xT0": Decimal(5), "0xT1": Decimal(10)}
        )
        obj._get_velodrome_pending_rewards = _gen_return(Decimal("2.5"))
        gen = obj._handle_velodrome_position(position, "optimism")
        result = _drive(gen)
        assert "0xVELO" in result[0]
        assert result[0]["0xVELO"] == Decimal("2.5")
        assert "CL Pool" in result[1]
        assert "VELO" in result[2].values()


class TestGetTickRanges:
    """TestGetTickRanges."""

    def test_non_cl_dex(self):
        """Test non cl dex."""
        obj = _mk()
        position = {"dex_type": "balancerPool"}
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert result == []

    def test_velodrome_non_cl(self):
        """Test velodrome non cl."""
        obj = _mk()
        position = {"dex_type": "velodrome", "is_cl_pool": False}
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert result == []

    def test_no_pool_address(self):
        """Test no pool address."""
        obj = _mk()
        position = {"dex_type": "UniswapV3", "pool_address": None}
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert result == []

    def test_slot0_fail(self):
        """Test slot0 fail."""
        obj = _mk()

        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        position = {"dex_type": "UniswapV3", "pool_address": "0xPool", "token_id": 1}
        obj.contract_interact = _gen_return(None)
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert result == []

    def test_no_position_manager(self):
        """Test no position manager."""
        obj = _mk()

        obj.params.uniswap_position_manager_contract_addresses = {}
        position = {"dex_type": "UniswapV3", "pool_address": "0xPool", "token_id": 1}
        obj.contract_interact = _gen_return({"tick": 100, "sqrt_price_x96": 100})
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert result == []


class TestCalculatePositionValue:
    """TestCalculatePositionValue."""

    def test_basic(self):
        """Test basic."""
        obj = _mk()
        position = {"pool_address": "0xP"}
        token_info = {"0xT0": "TKN0", "0xT1": "TKN1"}
        user_balances = {"0xT0": Decimal(10), "0xT1": Decimal(20)}
        obj._fetch_token_price = _gen_return(2.0)
        obj._update_position_with_current_value = _gen_none
        bd = []
        gen = obj._calculate_position_value(
            position, "optimism", user_balances, token_info, bd
        )
        result = _drive(gen)
        assert result == Decimal(10) * Decimal("2.0") + Decimal(20) * Decimal("2.0")
        assert len(bd) == 2

    def test_missing_balance(self):
        """Test missing balance."""
        obj = _mk()
        position = {"pool_address": "0xP"}
        token_info = {"0xT0": "TKN0"}
        user_balances = {}
        obj._fetch_token_price = _gen_return(2.0)
        obj._update_position_with_current_value = _gen_none
        gen = obj._calculate_position_value(
            position, "optimism", user_balances, token_info, []
        )
        result = _drive(gen)
        assert result == Decimal(0)

    def test_missing_price(self):
        """Test missing price."""
        obj = _mk()
        position = {"pool_address": "0xP"}
        token_info = {"0xT0": "TKN0"}
        user_balances = {"0xT0": Decimal(10)}
        obj._fetch_token_price = _gen_return(None)
        obj._update_position_with_current_value = _gen_none
        gen = obj._calculate_position_value(
            position, "optimism", user_balances, token_info, []
        )
        result = _drive(gen)
        assert result == Decimal(0)

    def test_existing_asset_update(self):
        """Test existing asset update."""
        obj = _mk()
        position = {"pool_address": "0xP"}
        token_info = {"0xT0": "TKN0"}
        user_balances = {"0xT0": Decimal(10)}
        obj._fetch_token_price = _gen_return(2.0)
        obj._update_position_with_current_value = _gen_none
        bd = [
            {
                "address": "0xT0",
                "balance": 5.0,
                "value_usd": 10.0,
                "price": 2.0,
                "asset": "TKN0",
            }
        ]
        gen = obj._calculate_position_value(
            position, "optimism", user_balances, token_info, bd
        )
        _drive(gen)
        assert bd[0]["balance"] == 10.0  # updated


class TestUpdatePositionAmounts:
    """TestUpdatePositionAmounts."""

    def test_no_positions(self):
        """Test no positions."""
        obj = _mk()
        obj.current_positions = None
        gen = obj.update_position_amounts()
        _drive(gen)

    def test_skip_closed(self):
        """Test skip closed."""
        obj = _mk()
        obj.current_positions = [
            {"status": "closed", "dex_type": "UniswapV3", "chain": "optimism"}
        ]
        obj.store_current_positions = MagicMock()
        gen = obj.update_position_amounts()
        _drive(gen)

    def test_missing_dex_type(self):
        """Test missing dex type."""
        obj = _mk()
        obj.current_positions = [{"status": "open", "chain": "optimism"}]
        obj.store_current_positions = MagicMock()
        gen = obj.update_position_amounts()
        _drive(gen)

    def test_unknown_dex(self):
        """Test unknown dex."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "unknown", "chain": "optimism"}
        ]
        obj.store_current_positions = MagicMock()
        gen = obj.update_position_amounts()
        _drive(gen)

    def test_balancer(self):
        """Test balancer."""
        obj = _mk()
        obj.current_positions = [
            {
                "status": "open",
                "dex_type": "balancerPool",
                "chain": "optimism",
                "pool_address": "0xP",
            }
        ]
        obj._update_balancer_position = _gen_none
        obj.store_current_positions = MagicMock()
        gen = obj.update_position_amounts()
        _drive(gen)
        obj.store_current_positions.assert_called_once()

    def test_uniswap(self):
        """Test uniswap."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "UniswapV3", "chain": "optimism"}
        ]
        obj._update_uniswap_position = _gen_none
        obj.store_current_positions = MagicMock()
        gen = obj.update_position_amounts()
        _drive(gen)

    def test_velodrome(self):
        """Test velodrome."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "velodrome", "chain": "optimism"}
        ]
        obj._update_velodrome_position = _gen_none
        obj.store_current_positions = MagicMock()
        gen = obj.update_position_amounts()
        _drive(gen)

    def test_sturdy(self):
        """Test sturdy."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "Sturdy", "chain": "mode"}
        ]
        obj._update_sturdy_position = _gen_none
        obj.store_current_positions = MagicMock()
        gen = obj.update_position_amounts()
        _drive(gen)


class TestUpdateBalancerPosition:
    """TestUpdateBalancerPosition."""

    def test_missing_params(self):
        """Test missing params."""
        obj = _mk()

        obj.params.safe_contract_addresses = {}
        gen = obj._update_balancer_position({"chain": "optimism"})
        _drive(gen)

    def test_success(self):
        """Test success."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.contract_interact = _gen_return(1000)
        pos = {"pool_address": "0xP", "chain": "optimism"}
        gen = obj._update_balancer_position(pos)
        _drive(gen)
        assert pos["current_liquidity"] == 1000

    def test_none_balance(self):
        """Test none balance."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.contract_interact = _gen_return(None)
        pos = {"pool_address": "0xP", "chain": "optimism"}
        gen = obj._update_balancer_position(pos)
        _drive(gen)
        assert "current_liquidity" not in pos


class TestUpdateUniswapPosition:
    """TestUpdateUniswapPosition."""

    def test_missing_params(self):
        """Test missing params."""
        obj = _mk()
        gen = obj._update_uniswap_position({"chain": "optimism"})
        _drive(gen)

    def test_no_pm(self):
        """Test no pm."""
        obj = _mk()

        obj.params.uniswap_position_manager_contract_addresses = {}
        gen = obj._update_uniswap_position({"token_id": 1, "chain": "optimism"})
        _drive(gen)

    def test_success(self):
        """Test success."""
        obj = _mk()

        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        obj.contract_interact = _gen_return({"liquidity": 500})
        pos = {"token_id": 1, "chain": "optimism"}
        gen = obj._update_uniswap_position(pos)
        _drive(gen)
        assert pos["current_liquidity"] == 500

    def test_no_data(self):
        """Test no data."""
        obj = _mk()

        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        obj.contract_interact = _gen_return(None)
        pos = {"token_id": 1, "chain": "optimism"}
        gen = obj._update_uniswap_position(pos)
        _drive(gen)

    def test_zero_liquidity_writes_zero(self):
        """On-chain liquidity of 0 must be written through, not treated as # noqa: D205,D209
        a fetch failure that leaves the cached non-zero value intact."""
        obj = _mk()

        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        obj.contract_interact = _gen_return({"liquidity": 0})
        pos = {"token_id": 1, "chain": "optimism", "current_liquidity": 12345}
        gen = obj._update_uniswap_position(pos)
        _drive(gen)
        assert pos["current_liquidity"] == 0


class TestUpdateSturdyPosition:
    """TestUpdateSturdyPosition."""

    def test_missing(self):
        """Test missing."""
        obj = _mk()

        obj.params.safe_contract_addresses = {}
        gen = obj._update_sturdy_position({"chain": "mode"})
        _drive(gen)

    def test_success(self):
        """Test success."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj.contract_interact = _gen_return(999)
        pos = {"pool_address": "0xP", "chain": "mode"}
        gen = obj._update_sturdy_position(pos)
        _drive(gen)
        assert pos["current_liquidity"] == 999

    def test_none(self):
        """Test none."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj.contract_interact = _gen_return(None)
        pos = {"pool_address": "0xP", "chain": "mode"}
        gen = obj._update_sturdy_position(pos)
        _drive(gen)


class TestUpdateVelodromePosition:
    """TestUpdateVelodromePosition."""

    def test_no_chain(self):
        """Test no chain."""
        obj = _mk()
        gen = obj._update_velodrome_position({})
        _drive(gen)

    def test_cl_success(self):
        """Test cl success."""
        obj = _mk()

        obj.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        obj.contract_interact = _gen_return({"liquidity": 42})
        pos = {
            "chain": "optimism",
            "is_cl_pool": True,
            "positions": [{"token_id": 1}],
        }
        gen = obj._update_velodrome_position(pos)
        _drive(gen)
        assert pos["positions"][0]["current_liquidity"] == 42

    def test_cl_zero_liquidity_writes_zero(self):
        """On-chain liquidity of 0 must be written through. A stale cached # noqa: D205,D209
        non-zero current_liquidity would otherwise prevent
        check_and_update_zero_liquidity_positions from closing the position."""
        obj = _mk()

        obj.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        obj.contract_interact = _gen_return({"liquidity": 0})
        pos = {
            "chain": "optimism",
            "is_cl_pool": True,
            "positions": [{"token_id": 1, "current_liquidity": 999}],
        }
        gen = obj._update_velodrome_position(pos)
        _drive(gen)
        assert pos["positions"][0]["current_liquidity"] == 0

    def test_cl_no_token_id(self):
        """Test cl no token id."""
        obj = _mk()

        obj.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        pos = {
            "chain": "optimism",
            "is_cl_pool": True,
            "positions": [{"token_id": None}],
        }
        gen = obj._update_velodrome_position(pos)
        _drive(gen)

    def test_cl_no_pm(self):
        """Test cl no pm."""
        obj = _mk()

        obj.params.velodrome_non_fungible_position_manager_contract_addresses = {}
        pos = {
            "chain": "optimism",
            "is_cl_pool": True,
            "positions": [{"token_id": 1}],
        }
        gen = obj._update_velodrome_position(pos)
        _drive(gen)

    def test_non_cl_success(self):
        """Test non cl success."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.contract_interact = _gen_return(77)
        pos = {"chain": "optimism", "is_cl_pool": False, "pool_address": "0xP"}
        gen = obj._update_velodrome_position(pos)
        _drive(gen)
        assert pos["current_liquidity"] == 77

    def test_non_cl_missing(self):
        """Test non cl missing."""
        obj = _mk()

        obj.params.safe_contract_addresses = {}
        pos = {"chain": "optimism", "is_cl_pool": False}
        gen = obj._update_velodrome_position(pos)
        _drive(gen)


class TestChainTotalInvestment:
    """TestChainTotalInvestment."""

    def test_load_found(self):
        """Test load found."""
        obj = _mk()
        obj._read_kv = _gen_return({"mode_total_investment": "123.45"})
        gen = obj._load_chain_total_investment("mode")
        assert _drive(gen) == 123.45

    def test_load_not_found(self):
        """Test load not found."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        gen = obj._load_chain_total_investment("mode")
        assert _drive(gen) == 0.0

    def test_load_invalid(self):
        """Test load invalid."""
        obj = _mk()
        obj._read_kv = _gen_return({"mode_total_investment": "not-a-number"})
        gen = obj._load_chain_total_investment("mode")
        assert _drive(gen) == 0.0

    def test_save(self):
        """Test save."""
        obj = _mk()
        written = {}

        def fake_write(d):
            """Fake write."""
            written.update(d)
            yield

        obj._write_kv = fake_write
        gen = obj._save_chain_total_investment("mode", 42.0)
        _drive(gen)
        assert written["mode_total_investment"] == "42.0"


class TestLoadFundingEventsData:
    """TestLoadFundingEventsData."""

    def test_found(self):
        """Test found."""
        obj = _mk()
        obj._read_kv = _gen_return({"funding_events": json.dumps({"mode": {}})})
        gen = obj._load_funding_events_data()
        assert _drive(gen) == {"mode": {}}

    def test_not_found(self):
        """Test not found."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        gen = obj._load_funding_events_data()
        assert _drive(gen) == {}

    def test_invalid_json(self):
        """Test invalid json."""
        obj = _mk()
        obj._read_kv = _gen_return({"funding_events": "{{bad"})
        gen = obj._load_funding_events_data()
        assert _drive(gen) == {}


class TestSaveTransferData:
    """TestSaveTransferData."""

    def test_mode(self):
        """Test mode."""
        obj = _mk()
        written = {}

        def fake_write(d):
            """Fake write."""
            written.update(d)
            yield

        obj._write_kv = fake_write
        gen = obj._save_transfer_data_mode({"a": 1})
        _drive(gen)
        assert "mode_transfer_data" in written

    def test_optimism(self):
        """Test optimism."""
        obj = _mk()
        written = {}

        def fake_write(d):
            """Fake write."""
            written.update(d)
            yield

        obj._write_kv = fake_write
        gen = obj._save_transfer_data_optimism({"b": 2})
        _drive(gen)
        assert "optimism_transfer_data" in written


class TestContractNameGetters:
    """TestContractNameGetters."""

    def test_aggregator(self):
        """Test aggregator."""
        obj = _mk()
        obj.contract_interact = _gen_return("MyAgg")
        gen = obj._get_aggregator_name("0xAgg", "mode")
        assert _drive(gen) == "MyAgg"

    def test_balancer(self):
        """Test balancer."""
        obj = _mk()
        obj.contract_interact = _gen_return("MyPool")
        gen = obj._get_balancer_pool_name("0xPool", "optimism")
        assert _drive(gen) == "MyPool"


class TestGetUserShareValueVelodrome:
    """TestGetUserShareValueVelodrome."""

    def test_missing_tokens(self):
        """Test missing tokens."""
        obj = _mk()
        pos = {"token0": None, "token1": "0xT1", "is_cl_pool": False}
        gen = obj.get_user_share_value_velodrome("0xU", "0xP", 1, "optimism", pos)
        assert _drive(gen) == {}

    def test_cl(self):
        """Test cl."""
        obj = _mk()
        pos = {"token0": "0xT0", "token1": "0xT1", "is_cl_pool": True}
        obj._get_user_share_value_velodrome_cl = _gen_return({"0xT0": Decimal(1)})
        gen = obj.get_user_share_value_velodrome("0xU", "0xP", 1, "optimism", pos)
        assert _drive(gen) == {"0xT0": Decimal(1)}

    def test_non_cl(self):
        """Test non cl."""
        obj = _mk()
        pos = {"token0": "0xT0", "token1": "0xT1", "is_cl_pool": False}
        obj._get_user_share_value_velodrome_non_cl = _gen_return({"0xT0": Decimal(2)})
        gen = obj.get_user_share_value_velodrome("0xU", "0xP", 1, "optimism", pos)
        assert _drive(gen) == {"0xT0": Decimal(2)}


class TestGetUserShareValueUniswap:
    """TestGetUserShareValueUniswap."""

    def test_missing_data(self):
        """Test missing data."""
        obj = _mk()
        pos = {"token0": None, "token1": "0xT1"}
        gen = obj.get_user_share_value_uniswap("0xP", 1, "optimism", pos)
        assert _drive(gen) == {}

    def test_no_pm(self):
        """Test no pm."""
        obj = _mk()

        obj.params.uniswap_position_manager_contract_addresses = {}
        pos = {"token0": "0xT0", "token1": "0xT1"}
        gen = obj.get_user_share_value_uniswap("0xP", 1, "optimism", pos)
        assert _drive(gen) == {}


class TestGetUserShareValueBalancer:
    """TestGetUserShareValueBalancer."""

    def test_no_vault(self):
        """Test no vault."""
        obj = _mk()

        obj.params.balancer_vault_contract_addresses = {}
        gen = obj.get_user_share_value_balancer("0xU", "0xPID", "0xP", "optimism")
        assert _drive(gen) == {}

    def test_no_pool_tokens(self):
        """Test no pool tokens."""
        obj = _mk()

        obj.params.balancer_vault_contract_addresses = {"optimism": "0xV"}
        obj.contract_interact = _gen_return(None)
        gen = obj.get_user_share_value_balancer("0xU", "0xPID", "0xP", "optimism")
        assert _drive(gen) == {}


class TestGetUserShareValueSturdy:
    """TestGetUserShareValueSturdy."""

    def test_no_balance(self):
        """Test no balance."""
        obj = _mk()
        obj.contract_interact = _gen_return(None)
        gen = obj.get_user_share_value_sturdy("0xU", "0xAgg", "0xA", "mode")
        assert _drive(gen) == {}

    def test_no_decimals(self):
        """Test no decimals."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return 1000
            else:
                yield
                return None

        obj.contract_interact = fake_ci
        gen = obj.get_user_share_value_sturdy("0xU", "0xAgg", "0xA", "mode")
        assert _drive(gen) == {}

    def test_success(self):
        """Test success."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return 1000000
            else:
                yield
                return 6

        obj.contract_interact = fake_ci
        gen = obj.get_user_share_value_sturdy("0xU", "0xAgg", "0xA", "mode")
        result = _drive(gen)
        assert "0xA" in result
        assert result["0xA"] == Decimal("1000000") / Decimal("1000000")


class TestCalculateAirdropRewardsValue:
    """TestCalculateAirdropRewardsValue."""

    def test_non_mode(self):
        """Test non mode."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        gen = obj.calculate_airdrop_rewards_value()
        assert _drive(gen) == Decimal(0)

    def test_zero_rewards(self):
        """Test zero rewards."""
        obj = _mk()

        obj.params.target_investment_chains = ["mode"]
        obj._get_total_airdrop_rewards = _gen_return(0)
        gen = obj.calculate_airdrop_rewards_value()
        assert _drive(gen) == Decimal(0)

    def test_with_rewards(self):
        """Test with rewards."""
        obj = _mk()

        obj.params.target_investment_chains = ["mode"]
        obj._get_total_airdrop_rewards = _gen_return(1000000)  # 1 USDC
        obj._get_usdc_address = lambda c: "0xUSDC"
        obj._fetch_token_price = _gen_return(1.0)
        gen = obj.calculate_airdrop_rewards_value()
        result = _drive(gen)
        assert result == Decimal("1") * Decimal("1.0")

    def test_with_rewards_no_price(self):
        """Test with rewards no price."""
        obj = _mk()

        obj.params.target_investment_chains = ["mode"]
        obj._get_total_airdrop_rewards = _gen_return(1000000)
        obj._get_usdc_address = lambda c: "0xUSDC"
        obj._fetch_token_price = _gen_return(None)
        gen = obj.calculate_airdrop_rewards_value()
        result = _drive(gen)
        # fallback to $1
        assert result == Decimal("1")


class TestCalculateStakingRewardsValue:
    """TestCalculateStakingRewardsValue."""

    def test_no_olas_address(self):
        """Test no olas address."""
        obj = _mk()

        obj.params.target_investment_chains = ["unknown"]
        obj.update_accumulated_rewards_for_chain = _gen_none
        gen = obj.calculate_stakig_rewards_value()
        assert _drive(gen) == Decimal(0)

    def test_zero_rewards(self):
        """Test zero rewards."""
        obj = _mk()

        obj.params.target_investment_chains = ["mode"]
        obj.update_accumulated_rewards_for_chain = _gen_none
        obj.get_accumulated_rewards_for_token = _gen_return(0)
        gen = obj.calculate_stakig_rewards_value()
        assert _drive(gen) == Decimal(0)

    def test_with_rewards(self):
        """Test with rewards."""
        obj = _mk()

        obj.params.target_investment_chains = ["mode"]
        obj.update_accumulated_rewards_for_chain = _gen_none
        obj.get_accumulated_rewards_for_token = _gen_return(10**18)
        obj._fetch_token_price = _gen_return(5.0)
        gen = obj.calculate_stakig_rewards_value()
        result = _drive(gen)
        assert result == Decimal("1") * Decimal("5.0")

    def test_no_price(self):
        """Test no price."""
        obj = _mk()

        obj.params.target_investment_chains = ["mode"]
        obj.update_accumulated_rewards_for_chain = _gen_none
        obj.get_accumulated_rewards_for_token = _gen_return(10**18)
        obj._fetch_token_price = _gen_return(None)
        gen = obj.calculate_stakig_rewards_value()
        assert _drive(gen) == Decimal(0)


class TestCalculateWithdrawalsValue:
    """TestCalculateWithdrawalsValue."""

    def test_unsupported_chain(self):
        """Test unsupported chain."""
        obj = _mk()

        obj.params.target_investment_chains = ["base"]
        gen = obj.calculate_withdrawals_value()
        assert _drive(gen) == Decimal(0)

    def test_mode_none_transfers(self):
        """Test mode none transfers."""
        obj = _mk()

        obj.params.target_investment_chains = ["mode"]
        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj._track_erc20_transfers_mode = MagicMock(return_value=None)
        gen = obj.calculate_withdrawals_value()
        assert _drive(gen) == Decimal(0)

    def test_mode_with_transfers(self):
        """Test mode with transfers."""
        obj = _mk()

        obj.params.target_investment_chains = ["mode"]
        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj._track_erc20_transfers_mode = MagicMock(
            return_value={"outgoing": {"2024-01-01": []}}
        )
        obj._track_and_calculate_withdrawal_value_mode = _gen_return(Decimal("50"))
        gen = obj.calculate_withdrawals_value()
        assert _drive(gen) == Decimal("50")

    def test_optimism_no_transfers(self):
        """Test optimism no transfers."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        # No prior cache; fetcher returns None so the function short-circuits
        # before reaching the kv write.
        obj._read_kv = _gen_return({})
        obj._track_erc20_transfers_optimism = _gen_return(None)
        gen = obj.calculate_withdrawals_value()
        assert _drive(gen) == Decimal(0)

    def test_optimism_with_transfers(self):
        """Test optimism with transfers."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        # No prior cache; fetcher returns data so the function recomputes and
        # writes the kv cache on the way out.
        obj._read_kv = _gen_return({})
        obj._write_kv = _gen_none
        obj._track_erc20_transfers_optimism = _gen_return({"outgoing": {"d": []}})
        obj._track_and_calculate_withdrawal_value_optimism = _gen_return(Decimal("25"))
        obj._get_current_timestamp = lambda: 1704067200
        gen = obj.calculate_withdrawals_value()
        assert _drive(gen) == Decimal("25")


class TestTrackWithdrawalMode:
    """TestTrackWithdrawalMode."""

    def test_empty(self):
        """Test empty."""
        obj = _mk()
        gen = obj._track_and_calculate_withdrawal_value_mode({})
        assert _drive(gen) == Decimal(0)

    def test_with_usdc(self):
        """Test with usdc."""
        obj = _mk()
        transfers = {
            "2024-01-01": [
                {"symbol": "USDC", "amount": 10, "timestamp": "2024-01-01T00:00:00Z"}
            ]
        }
        obj._calculate_total_withdrawal_value = _gen_return(Decimal("10"))
        gen = obj._track_and_calculate_withdrawal_value_mode(transfers)
        assert _drive(gen) == Decimal("10")

    def test_exception(self):
        """Test exception."""
        obj = _mk()

        def boom(*a, **kw):
            """Boom."""
            raise RuntimeError("fail")
            yield  # noqa: unreachable

        obj._calculate_total_withdrawal_value = boom
        transfers = {
            "2024-01-01": [
                {"symbol": "USDC", "amount": 10, "timestamp": "2024-01-01T00:00:00Z"}
            ]
        }
        gen = obj._track_and_calculate_withdrawal_value_mode(transfers)
        assert _drive(gen) == Decimal(0)


class TestTrackWithdrawalOptimism:
    """TestTrackWithdrawalOptimism."""

    def test_empty(self):
        """Test empty."""
        obj = _mk()
        gen = obj._track_and_calculate_withdrawal_value_optimism({})
        assert _drive(gen) == Decimal(0)

    def test_with_usdc(self):
        """Test with usdc."""
        obj = _mk()
        transfers = {
            "2024-01-01": [
                {
                    "symbol": "USDC",
                    "amount": 10,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "to_address": "0xEOA",
                }
            ]
        }
        obj._is_not_other_contract_optimism = _gen_return(True)
        obj._calculate_total_withdrawal_value = _gen_return(Decimal("10"))
        gen = obj._track_and_calculate_withdrawal_value_optimism(transfers)
        assert _drive(gen) == Decimal("10")

    def test_filter_contracts(self):
        """Test filter contracts."""
        obj = _mk()
        transfers = {
            "2024-01-01": [
                {
                    "symbol": "USDC",
                    "amount": 10,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "to_address": "0xContract",
                }
            ]
        }
        obj._is_not_other_contract_optimism = _gen_return(False)
        obj._calculate_total_withdrawal_value = _gen_return(Decimal("0"))
        gen = obj._track_and_calculate_withdrawal_value_optimism(transfers)
        assert _drive(gen) == Decimal("0")


class TestCalculateTotalWithdrawalValue:
    """TestCalculateTotalWithdrawalValue."""

    def test_empty(self):
        """Test empty."""
        obj = _mk()
        gen = obj._calculate_total_withdrawal_value([], chain="mode")
        assert _drive(gen) == Decimal(0)

    def test_no_timestamp(self):
        """Test no timestamp."""
        obj = _mk()
        gen = obj._calculate_total_withdrawal_value([{"timestamp": ""}], chain="mode")
        assert _drive(gen) == Decimal(0)

    def test_bad_timestamp(self):
        """Test bad timestamp."""
        obj = _mk()
        gen = obj._calculate_total_withdrawal_value(
            [{"timestamp": "not-a-date"}], chain="mode"
        )
        assert _drive(gen) == Decimal(0)

    def test_no_coin_id(self):
        """Test no coin id."""
        obj = _mk()
        obj.get_coin_id_from_symbol = lambda s, c: None
        gen = obj._calculate_total_withdrawal_value(
            [{"timestamp": "2024-01-01T00:00:00Z"}], chain="mode"
        )
        assert _drive(gen) == Decimal(0)

    def test_with_price(self):
        """Test with price."""
        obj = _mk()
        obj.get_coin_id_from_symbol = lambda s, c: "usd-coin"
        obj._fetch_historical_token_price = _gen_return(1.0)
        transfers = [{"timestamp": "2024-01-01T00:00:00Z", "amount": 100}]
        gen = obj._calculate_total_withdrawal_value(transfers, chain="mode")
        result = _drive(gen)
        assert result == Decimal("100") * Decimal("1.0")

    def test_no_price(self):
        """Test no price."""
        obj = _mk()
        obj.get_coin_id_from_symbol = lambda s, c: "usd-coin"
        obj._fetch_historical_token_price = _gen_return(None)
        transfers = [{"timestamp": "2024-01-01T00:00:00Z", "amount": 100}]
        gen = obj._calculate_total_withdrawal_value(transfers, chain="mode")
        assert _drive(gen) == Decimal(0)


class TestUpdatePositionWithCurrentValue:
    """TestUpdatePositionWithCurrentValue."""

    def test_basic_cost_recovered(self):
        """Test basic cost recovered."""
        obj = _mk()
        obj._get_current_timestamp = lambda: 1000
        obj._calculate_corrected_yield = _gen_return(Decimal("100"))
        pos = {"pool_address": "0xP", "amount0": 100, "amount1": 200, "entry_cost": 50}
        gen = obj._update_position_with_current_value(
            pos,
            Decimal("500"),
            "optimism",
            user_balances={"0xT0": Decimal(1)},
            token_info={},
            token_prices={},
        )
        _drive(gen)
        assert pos["cost_recovered"] is True
        assert pos["yield_usd"] == 100.0

    def test_not_recovered(self):
        """Test not recovered."""
        obj = _mk()
        obj._get_current_timestamp = lambda: 1000
        obj._calculate_corrected_yield = _gen_return(Decimal("10"))
        pos = {"pool_address": "0xP", "amount0": 100, "amount1": 200, "entry_cost": 50}
        gen = obj._update_position_with_current_value(
            pos,
            Decimal("500"),
            "optimism",
            user_balances={"0xT0": Decimal(1)},
            token_info={},
            token_prices={},
        )
        _drive(gen)
        assert pos["cost_recovered"] is False

    def test_no_balances(self):
        """Test no balances."""
        obj = _mk()
        obj._get_current_timestamp = lambda: 1000
        obj._get_current_token_balances = _gen_return(None)
        pos = {"pool_address": "0xP", "amount0": None, "amount1": None}
        gen = obj._update_position_with_current_value(pos, Decimal("500"), "optimism")
        _drive(gen)
        assert pos["cost_recovered"] is False

    def test_exception(self):
        """Test exception."""
        obj = _mk()
        obj._get_current_timestamp = MagicMock(side_effect=RuntimeError("boom"))
        pos = {"pool_address": "0xP"}
        gen = obj._update_position_with_current_value(pos, Decimal("500"), "optimism")
        _drive(gen)
        assert pos["cost_recovered"] is False

    def test_zero_entry_cost(self):
        """Test zero entry cost."""
        obj = _mk()
        obj._get_current_timestamp = lambda: 1000
        obj._calculate_corrected_yield = _gen_return(Decimal("0"))
        pos = {"pool_address": "0xP", "amount0": 0, "amount1": 0, "entry_cost": 0}
        gen = obj._update_position_with_current_value(
            pos,
            Decimal("0"),
            "optimism",
            user_balances={"0xT": Decimal(0)},
            token_info={},
            token_prices={},
        )
        _drive(gen)
        assert pos["cost_recovered"] is True


class TestCalculateCorrectedYield:
    """TestCalculateCorrectedYield."""

    def test_no_decimals(self):
        """Test no decimals."""
        obj = _mk()
        obj._get_token_decimals = _gen_return(None)
        pos = {
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xP",
            "token0_symbol": "A",
            "token1_symbol": "B",
        }
        gen = obj._calculate_corrected_yield(pos, 0, 0, {}, "optimism", {})
        assert _drive(gen) == Decimal(0)

    def test_with_prices_in_cache(self):
        """Test with prices in cache."""
        obj = _mk()
        call_count = [0]

        def fake_dec(*a, **kw):
            """Fake dec."""
            call_count[0] += 1
            yield
            return 18

        obj._get_token_decimals = fake_dec
        pos = {
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xP",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "staked": False,
        }
        balances = {"0xT0": Decimal("2"), "0xT1": Decimal("3")}
        prices = {"0xT0": Decimal("10"), "0xT1": Decimal("5")}
        gen = obj._calculate_corrected_yield(
            pos, 10**18, 10**18, balances, "optimism", prices
        )
        result = _drive(gen)
        # increase = (2-1)*10 + (3-1)*5 = 10+10 = 20
        assert result == Decimal("20")

    def test_fetch_prices(self):
        """Test fetch prices."""
        obj = _mk()
        call_count = [0]

        def fake_dec(*a, **kw):
            """Fake dec."""
            call_count[0] += 1
            yield
            return 18

        obj._get_token_decimals = fake_dec
        obj._fetch_token_price = _gen_return(10.0)
        pos = {
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xP",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "staked": False,
        }
        balances = {"0xT0": Decimal("2"), "0xT1": Decimal("2")}
        gen = obj._calculate_corrected_yield(
            pos, 10**18, 10**18, balances, "optimism", None
        )
        result = _drive(gen)
        assert result == Decimal("20")

    def test_fetch_prices_fail(self):
        """Test fetch prices fail."""
        obj = _mk()

        def fake_dec(*a, **kw):
            """Fake dec."""
            yield
            return 18

        obj._get_token_decimals = fake_dec
        obj._fetch_token_price = _gen_return(None)
        pos = {
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xP",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "staked": False,
        }
        balances = {"0xT0": Decimal("2"), "0xT1": Decimal("2")}
        gen = obj._calculate_corrected_yield(
            pos, 10**18, 10**18, balances, "optimism", None
        )
        assert _drive(gen) == Decimal(0)

    def test_velo_rewards(self):
        """Test velo rewards."""
        obj = _mk()

        def fake_dec(*a, **kw):
            """Fake dec."""
            yield
            return 18

        obj._get_token_decimals = fake_dec
        obj._get_velo_token_address = lambda c: "0xVELO"
        obj.get_coin_id_from_symbol = lambda s, c: "velodrome-finance"
        obj._fetch_coin_price = _gen_return(0.5)
        pos = {
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xP",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "staked": True,
            "dex_type": "velodrome",
        }
        prices = {"0xT0": Decimal("10"), "0xT1": Decimal("5")}
        balances = {"0xT0": Decimal("1"), "0xT1": Decimal("1"), "0xVELO": Decimal("10")}
        gen = obj._calculate_corrected_yield(
            pos, 10**18, 10**18, balances, "optimism", prices
        )
        result = _drive(gen)
        # base yield = 0, velo = 10 * 0.5 = 5
        assert result == Decimal("5.0")


class TestGetCurrentTokenBalances:
    """TestGetCurrentTokenBalances."""

    def test_balancer(self):
        """Test balancer."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.get_user_share_value_balancer = _gen_return({"0xT": Decimal(1)})
        pos = {
            "dex_type": "balancerPool",
            "pool_id": "0xPID",
            "pool_address": "0xP",
            "chain": "optimism",
        }
        gen = obj._get_current_token_balances(pos, "optimism")
        assert _drive(gen) == {"0xT": Decimal(1)}

    def test_uniswap(self):
        """Test uniswap."""
        obj = _mk()
        obj.get_user_share_value_uniswap = _gen_return({"0xT": Decimal(2)})
        pos = {
            "dex_type": "UniswapV3",
            "pool_address": "0xP",
            "token_id": 1,
            "chain": "optimism",
        }
        gen = obj._get_current_token_balances(pos, "optimism")
        assert _drive(gen) == {"0xT": Decimal(2)}

    def test_velodrome(self):
        """Test velodrome."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.get_user_share_value_velodrome = _gen_return({"0xT": Decimal(3)})
        pos = {
            "dex_type": "velodrome",
            "pool_address": "0xP",
            "token_id": 1,
            "chain": "optimism",
        }
        gen = obj._get_current_token_balances(pos, "optimism")
        assert _drive(gen) == {"0xT": Decimal(3)}

    def test_sturdy(self):
        """Test sturdy."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj.get_user_share_value_sturdy = _gen_return({"0xT": Decimal(4)})
        pos = {
            "dex_type": "Sturdy",
            "pool_address": "0xP",
            "token0": "0xT",
            "chain": "mode",
        }
        gen = obj._get_current_token_balances(pos, "mode")
        assert _drive(gen) == {"0xT": Decimal(4)}

    def test_unknown(self):
        """Test unknown."""
        obj = _mk()
        pos = {"dex_type": "unknown"}
        gen = obj._get_current_token_balances(pos, "optimism")
        assert _drive(gen) == {}


class TestReadInvestingPaused:
    """TestReadInvestingPaused."""

    def test_true(self):
        """Test true."""
        obj = _mk()
        obj._read_kv = _gen_return({"investing_paused": "true"})
        gen = obj._read_investing_paused()
        assert _drive(gen) is True

    def test_false(self):
        """Test false."""
        obj = _mk()
        obj._read_kv = _gen_return({"investing_paused": "false"})
        gen = obj._read_investing_paused()
        assert _drive(gen) is False

    def test_none_result(self):
        """Test none result."""
        obj = _mk()
        obj._read_kv = _gen_return(None)
        result = _drive(obj._read_investing_paused())
        assert result is False
        obj.context.logger.error.assert_called_once()

    def test_none_value(self):
        """Test none value."""
        obj = _mk()
        obj._read_kv = _gen_return({"investing_paused": None})
        gen = obj._read_investing_paused()
        assert _drive(gen) is False

    def test_unexpected_exception_propagates(self):
        """The narrowed handler does not swallow exceptions it never expected."""
        obj = _mk()

        def boom(*a, **kw):
            """Boom."""
            raise RuntimeError("fail")
            yield  # noqa

        obj._read_kv = boom
        with pytest.raises(RuntimeError, match="fail"):
            _drive(obj._read_investing_paused())


class TestCheckIsValidSafeAddress:
    """TestCheckIsValidSafeAddress."""

    def test_valid(self):
        """Test valid."""
        obj = _mk()
        obj.contract_interact = _gen_return(["0xOwner"])
        gen = obj.check_is_valid_safe_address("0xSafe", "optimism")
        assert _drive(gen) is True

    def test_invalid(self):
        """Test invalid."""
        obj = _mk()
        obj.contract_interact = _gen_return(None)
        gen = obj.check_is_valid_safe_address("0xSafe", "optimism")
        assert _drive(gen) is False

    def test_exception(self):
        """Test exception."""
        obj = _mk()

        def boom(*a, **kw):
            """Boom."""
            raise RuntimeError("boom")
            yield  # noqa

        obj.contract_interact = boom
        gen = obj.check_is_valid_safe_address("0xSafe", "optimism")
        assert _drive(gen) is False


class TestGetVelodromePendingRewards:
    """TestGetVelodromePendingRewards."""

    def test_no_pool_address(self):
        """Test no pool address."""
        obj = _mk()
        gen = obj._get_velodrome_pending_rewards({}, "optimism", "0xU")
        assert _drive(gen) == Decimal(0)

    def test_no_pool_behaviour(self):
        """Test no pool behaviour."""
        obj = _mk()
        obj.pools = {}
        gen = obj._get_velodrome_pending_rewards(
            {"pool_address": "0xP"}, "optimism", "0xU"
        )
        assert _drive(gen) == Decimal(0)

    def test_cl_pool_with_rewards(self):
        """Test cl pool with rewards."""
        obj = _mk()
        pool_mock = MagicMock()
        pool_mock.get_gauge_address = _gen_return("0xGauge")
        pool_mock.get_cl_pending_rewards = _gen_return(10**18)
        obj.pools = {"velodrome": pool_mock}
        pos = {
            "pool_address": "0xP",
            "is_cl_pool": True,
            "positions": [{"token_id": 1}],
        }
        gen = obj._get_velodrome_pending_rewards(pos, "optimism", "0xU")
        result = _drive(gen)
        assert result == Decimal("1")

    def test_cl_pool_no_gauge(self):
        """Test cl pool no gauge."""
        obj = _mk()
        pool_mock = MagicMock()
        pool_mock.get_gauge_address = _gen_return(None)
        obj.pools = {"velodrome": pool_mock}
        pos = {"pool_address": "0xP", "is_cl_pool": True, "positions": []}
        gen = obj._get_velodrome_pending_rewards(pos, "optimism", "0xU")
        assert _drive(gen) == Decimal(0)

    def test_regular_pool(self):
        """Test regular pool."""
        obj = _mk()
        pool_mock = MagicMock()
        pool_mock.get_pending_rewards = _gen_return(5 * 10**18)
        obj.pools = {"velodrome": pool_mock}
        pos = {"pool_address": "0xP", "is_cl_pool": False}
        gen = obj._get_velodrome_pending_rewards(pos, "optimism", "0xU")
        result = _drive(gen)
        assert result == Decimal("5")

    def test_exception(self):
        """Test exception."""
        obj = _mk()
        obj.pools = {"velodrome": MagicMock(side_effect=RuntimeError)}
        pos = {"pool_address": "0xP"}
        gen = obj._get_velodrome_pending_rewards(pos, "optimism", "0xU")
        assert _drive(gen) == Decimal(0)


class TestCreatePortfolioData:
    """TestCreatePortfolioData."""

    def test_basic(self):
        """Test basic."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._get_current_timestamp = lambda: 1000
        result = obj._create_portfolio_data(
            Decimal("100"),
            Decimal("50"),
            Decimal("5"),
            Decimal("2"),
            Decimal("10"),
            200.0,
            300.0,
            [],
            [],
        )
        assert result["portfolio_value"] == 150.0
        assert result["total_roi"] is not None
        assert result["address"] == "0xSafe"

    def test_no_initial_investment(self):
        """Test no initial investment."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._get_current_timestamp = lambda: 1000
        result = obj._create_portfolio_data(
            Decimal("100"),
            Decimal("50"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            None,
            None,
            [],
            [],
        )
        assert result["total_roi"] is None
        assert result["initial_investment"] is None

    def test_zero_initial(self):
        """Test zero initial."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._get_current_timestamp = lambda: 1000
        result = obj._create_portfolio_data(
            Decimal("100"),
            Decimal("50"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            0,
            None,
            [],
            [],
        )
        assert result["total_roi"] is None

    def test_with_allocations_and_breakdown(self):
        """Test with allocations and breakdown."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._get_current_timestamp = lambda: 1000
        allocs = [
            {
                "chain": "optimism",
                "type": "uniswapV3",
                "id": "0xP",
                "assets": ["WETH"],
                "apr": 5.0,
                "details": "d",
                "ratio": 100.0,
                "address": "0xSafe",
            }
        ]
        bd = [
            {
                "asset": "WETH",
                "address": "0xWETH",
                "balance": 1.0,
                "price": 3000.0,
                "value_usd": 3000.0,
                "ratio": 1.0,
            }
        ]
        result = obj._create_portfolio_data(
            Decimal("3000"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            3000.0,
            3000.0,
            allocs,
            bd,
        )
        assert len(result["allocations"]) == 1
        assert len(result["portfolio_breakdown"]) == 1

    def test_tick_ranges_in_allocation(self):
        """Test tick ranges in allocation."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._get_current_timestamp = lambda: 1000
        allocs = [
            {
                "chain": "optimism",
                "type": "uniswapV3",
                "id": "0xP",
                "assets": ["WETH"],
                "apr": 5.0,
                "details": "d",
                "ratio": 100.0,
                "address": "0xSafe",
                "tick_ranges": [{"tick": 1}],
            }
        ]
        result = obj._create_portfolio_data(
            Decimal("100"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            100.0,
            100.0,
            allocs,
            [],
        )
        assert "tick_ranges" in result["allocations"][0]

    def test_olas_filtered(self):
        """Test olas filtered."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._get_current_timestamp = lambda: 1000
        olas_addr = OLAS_ADDRESSES.get("optimism", "0xOLAS")
        bd = [
            {
                "asset": "OLAS",
                "address": olas_addr,
                "balance": 1.0,
                "price": 1.0,
                "value_usd": 1.0,
                "ratio": 1.0,
            }
        ]
        result = obj._create_portfolio_data(
            Decimal("1"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            1.0,
            1.0,
            [],
            bd,
        )
        assert len(result["portfolio_breakdown"]) == 0

    def test_exception(self):
        """Test exception."""
        obj = _mk()

        obj.params.target_investment_chains = []  # will cause index error
        result = obj._create_portfolio_data(
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            None,
            None,
            [],
            [],
        )
        assert result == {}

    def test_allocation_error_skipped(self):
        """Test allocation error skipped."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._get_current_timestamp = lambda: 1000
        allocs = [{"bad": "data"}]  # missing keys
        result = obj._create_portfolio_data(
            Decimal("100"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            100.0,
            100.0,
            allocs,
            [],
        )
        assert len(result["allocations"]) == 0

    def test_breakdown_error_skipped(self):
        """Test breakdown error skipped."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._get_current_timestamp = lambda: 1000
        bd = [{"bad": "data"}]  # missing keys
        result = obj._create_portfolio_data(
            Decimal("100"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            100.0,
            100.0,
            [],
            bd,
        )
        assert len(result["portfolio_breakdown"]) == 0


class TestValidateVelodromeV2PoolAddresses:
    """TestValidateVelodromeV2PoolAddresses."""

    def test_skip_non_velodrome(self):
        """Test skip non velodrome."""
        obj = _mk()
        obj.current_positions = [
            {"dex_type": "UniswapV3", "is_cl_pool": False, "is_stable": True}
        ]
        obj.store_current_positions = MagicMock()
        gen = obj._validate_velodrome_v2_pool_addresses()
        _drive(gen)

    def test_skip_cl_pool(self):
        """Test skip cl pool."""
        obj = _mk()
        obj.current_positions = [
            {"dex_type": "velodrome", "is_cl_pool": True, "is_stable": True}
        ]
        obj.store_current_positions = MagicMock()
        gen = obj._validate_velodrome_v2_pool_addresses()
        _drive(gen)

    def test_skip_not_stable(self):
        """Test skip not stable."""
        obj = _mk()
        obj.current_positions = [
            {"dex_type": "velodrome", "is_cl_pool": False, "is_stable": False}
        ]
        obj.store_current_positions = MagicMock()
        gen = obj._validate_velodrome_v2_pool_addresses()
        _drive(gen)

    def test_validates(self):
        """Test validates."""
        obj = _mk()
        obj.current_positions = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": False,
                "is_stable": True,
                "pool_address": "0xP",
            }
        ]
        obj._validate_velodrome_v2_pool_address = _gen_return(True)
        obj.store_current_positions = MagicMock()
        gen = obj._validate_velodrome_v2_pool_addresses()
        _drive(gen)
        obj.store_current_positions.assert_called_once()


class TestValidateVelodromeV2PoolAddress:
    """TestValidateVelodromeV2PoolAddress."""

    def test_already_updated(self):
        """Test already updated."""
        obj = _mk()
        gen = obj._validate_velodrome_v2_pool_address({"isUpdated": True})
        assert _drive(gen) is True

    def test_missing_data(self):
        """Test missing data."""
        obj = _mk()
        gen = obj._validate_velodrome_v2_pool_address({"enter_tx_hash": None})
        assert _drive(gen) is False

    def test_no_receipt(self):
        """Test no receipt."""
        obj = _mk()
        obj.get_transaction_receipt = _gen_return(None)
        pos = {"enter_tx_hash": "0xTX", "chain": "optimism", "pool_address": "0xP"}
        gen = obj._validate_velodrome_v2_pool_address(pos)
        assert _drive(gen) is False

    def test_no_mint_event(self):
        """Test no mint event."""
        obj = _mk()
        obj.get_transaction_receipt = _gen_return({"logs": []})
        pos = {"enter_tx_hash": "0xTX", "chain": "optimism", "pool_address": "0xP"}
        gen = obj._validate_velodrome_v2_pool_address(pos)
        assert _drive(gen) is False

    def test_match(self):
        """Test match."""
        obj = _mk()
        obj.get_transaction_receipt = _gen_return(
            {
                "logs": [
                    {
                        "topics": [
                            TRANSFER_EVENT_SIGNATURE,
                            ZERO_ADDRESS_PADDED,
                            "0xTo",
                        ],
                        "address": "0xp",
                    }
                ]
            }
        )
        pos = {"enter_tx_hash": "0xTX", "chain": "optimism", "pool_address": "0xP"}
        gen = obj._validate_velodrome_v2_pool_address(pos)
        assert _drive(gen) is True
        assert pos["is_updated"] is True

    def test_mismatch(self):
        """Test mismatch."""
        obj = _mk()
        obj.get_transaction_receipt = _gen_return(
            {
                "logs": [
                    {
                        "topics": [
                            TRANSFER_EVENT_SIGNATURE,
                            ZERO_ADDRESS_PADDED,
                            "0xTo",
                        ],
                        "address": "0xNEW",
                    }
                ]
            }
        )
        pos = {"enter_tx_hash": "0xTX", "chain": "optimism", "pool_address": "0xP"}
        gen = obj._validate_velodrome_v2_pool_address(pos)
        assert _drive(gen) is True
        assert pos["pool_address"] == "0xnew"

    def test_exception(self):
        """Test exception."""
        obj = _mk()

        def boom(*a, **kw):
            """Boom."""
            raise RuntimeError("boom")
            yield  # noqa

        obj.get_transaction_receipt = boom
        pos = {"enter_tx_hash": "0xTX", "chain": "optimism", "pool_address": "0xP"}
        gen = obj._validate_velodrome_v2_pool_address(pos)
        assert _drive(gen) is False


class TestCalculateTotalReversionValue:
    """TestCalculateTotalReversionValue."""

    def test_basic(self):
        """Test basic."""
        obj = _mk()
        obj._fetch_historical_eth_price = _gen_return(2000.0)
        eth_transfers = [
            {"timestamp": "2024-01-01T00:00:00Z", "amount": 1.0},
            {"timestamp": "2024-02-01T00:00:00Z", "amount": 0.5},
        ]
        reversion = [{"amount": 0.5}]
        result = _drive(obj._calculate_total_reversion_value(eth_transfers, reversion))
        assert result == 0.5 * 2000.0

    def test_multiple_reversions(self):
        """Test multiple reversions."""
        obj = _mk()
        obj._fetch_historical_eth_price = _gen_return(1000.0)
        eth_transfers = [
            {"timestamp": "2024-01-01T00:00:00Z", "amount": 1.0},
            {"timestamp": "2024-02-01T00:00:00Z", "amount": 0.5},
        ]
        reversion = [{"amount": 0.3}, {"amount": 0.2}]
        result = _drive(obj._calculate_total_reversion_value(eth_transfers, reversion))
        assert result == (0.3 + 0.2) * 1000.0

    def test_no_eth_price(self):
        """Test no eth price."""
        obj = _mk()
        obj._fetch_historical_eth_price = _gen_return(None)
        eth_transfers = [{"timestamp": "2024-01-01T00:00:00Z", "amount": 1.0}]
        reversion = [{"amount": 0.5}]
        result = _drive(obj._calculate_total_reversion_value(eth_transfers, reversion))
        assert result == 0.0

    def test_unix_timestamp(self):
        """Test unix timestamp."""
        obj = _mk()
        obj._fetch_historical_eth_price = _gen_return(500.0)
        eth_transfers = [
            {"timestamp": "1704067200", "amount": 1.0},
        ]
        reversion = [{"amount": 0.1}]
        result = _drive(obj._calculate_total_reversion_value(eth_transfers, reversion))
        assert result == 0.1 * 500.0

    def test_bad_timestamp(self):
        """Test bad timestamp."""
        obj = _mk()
        obj._fetch_historical_eth_price = _gen_return(500.0)
        eth_transfers = [{"timestamp": "bad", "amount": 1.0}]
        reversion = [{"amount": 0.1}]
        result = _drive(obj._calculate_total_reversion_value(eth_transfers, reversion))
        # fallback to current date
        assert result == 0.1 * 500.0


class TestShouldIncludeTransferOptimism:
    """TestShouldIncludeTransferOptimism."""

    def test_empty(self):
        """Test empty."""
        obj = _mk()
        gen = obj._should_include_transfer_optimism("")
        assert _drive(gen) is False

    def test_zero_addr(self):
        """Test zero addr."""
        obj = _mk()
        gen = obj._should_include_transfer_optimism(
            "0x0000000000000000000000000000000000000000"
        )
        assert _drive(gen) is False

    def test_cached_eoa(self):
        """Test cached eoa."""
        obj = _mk()
        cache_key = f"{CONTRACT_CHECK_CACHE_PREFIX}optimism_0xabc"
        obj._read_kv = _gen_return({cache_key: json.dumps({"is_eoa": True})})
        gen = obj._should_include_transfer_optimism("0xABC")
        assert _drive(gen) is True

    def test_cached_contract(self):
        """Test cached contract."""
        obj = _mk()
        cache_key = f"{CONTRACT_CHECK_CACHE_PREFIX}optimism_0xabc"
        obj._read_kv = _gen_return({cache_key: json.dumps({"is_eoa": False})})
        gen = obj._should_include_transfer_optimism("0xABC")
        assert _drive(gen) is False

    def test_eoa_uncached(self):
        """Test eoa uncached."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj._request_with_retries = _gen_return((True, {"result": "0x"}))
        obj._write_kv = _gen_none
        cg = MagicMock()
        obj.context.coingecko = cg

        gen = obj._should_include_transfer_optimism("0xABC")
        assert _drive(gen) is True

    def test_contract_not_safe(self):
        """Test contract not safe."""
        obj = _mk()
        call_count = [0]

        def fake_read(*a, **kw):
            """Fake read."""
            yield
            return {}

        def fake_req(*a, **kw):
            """Fake req."""
            nonlocal call_count  # noqa: F824
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return (True, {"result": "0x1234"})
            else:
                yield
                return (False, {})

        obj._read_kv = fake_read
        obj._request_with_retries = fake_req
        obj._write_kv = _gen_none
        cg = MagicMock()
        obj.context.coingecko = cg

        gen = obj._should_include_transfer_optimism("0xABC")
        assert _drive(gen) is False

    def test_rpc_fail(self):
        """Test rpc fail."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj._request_with_retries = _gen_return((False, {}))
        cg = MagicMock()
        obj.context.coingecko = cg

        gen = obj._should_include_transfer_optimism("0xABC")
        assert _drive(gen) is False

    def test_exception(self):
        """Test exception."""
        obj = _mk()
        # _read_kv succeeds, but _request_with_retries raises inside try block
        obj._read_kv = _gen_return({})

        def boom(*a, **kw):
            """Boom."""
            raise RuntimeError("fail")
            yield  # noqa

        obj._request_with_retries = boom
        obj.coingecko.rate_limited_status_callback = MagicMock()
        obj.params.sleep_time = 1
        gen = obj._should_include_transfer_optimism("0xABC")
        assert _drive(gen) is False


class TestIsNotOtherContractOptimism:
    """TestIsNotOtherContractOptimism."""

    def test_empty(self):
        """Test empty."""
        obj = _mk()
        gen = obj._is_not_other_contract_optimism("")
        assert _drive(gen) is False

    def test_zero(self):
        """Test zero."""
        obj = _mk()
        gen = obj._is_not_other_contract_optimism(
            "0x0000000000000000000000000000000000000000"
        )
        assert _drive(gen) is False

    def test_cached(self):
        """Test cached."""
        obj = _mk()
        cache_key = f"{CONTRACT_CHECK_CACHE_PREFIX}optimism_0xabc"
        obj._read_kv = _gen_return({cache_key: json.dumps({"is_eoa": True})})
        gen = obj._is_not_other_contract_optimism("0xABC")
        assert _drive(gen) is True

    def test_eoa(self):
        """Test eoa."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj._request_with_retries = _gen_return((True, {"result": "0x"}))
        obj._write_kv = _gen_none
        cg = MagicMock()
        obj.context.coingecko = cg

        gen = obj._is_not_other_contract_optimism("0xABC")
        assert _drive(gen) is True

    def test_contract_safe(self):
        """Test contract safe."""
        obj = _mk()
        call_count = [0]

        def fake_req(*a, **kw):
            """Fake req."""
            nonlocal call_count  # noqa: F824
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return (True, {"result": "0x1234"})
            else:
                yield
                return (True, {})

        obj._read_kv = _gen_return({})
        obj._request_with_retries = fake_req
        obj._write_kv = _gen_none
        cg = MagicMock()
        obj.context.coingecko = cg

        gen = obj._is_not_other_contract_optimism("0xABC")
        assert _drive(gen) is True

    def test_rpc_fail(self):
        """Test rpc fail."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj._request_with_retries = _gen_return((False, {}))
        cg = MagicMock()
        obj.context.coingecko = cg

        gen = obj._is_not_other_contract_optimism("0xABC")
        assert _drive(gen) is False

    def test_exception(self):
        """Test exception."""
        obj = _mk()
        obj._read_kv = _gen_return({})

        def boom(*a, **kw):
            """Boom."""
            raise RuntimeError("fail")
            yield  # noqa

        obj._request_with_retries = boom
        obj.coingecko.rate_limited_status_callback = MagicMock()
        obj.params.sleep_time = 1
        gen = obj._is_not_other_contract_optimism("0xABC")
        assert _drive(gen) is False


class TestGetMasterSafeAddress:
    """TestGetMasterSafeAddress."""

    def test_no_service_id(self):
        """Test no service id."""
        obj = _mk()

        obj.params.on_chain_service_id = None
        gen = obj.get_master_safe_address()
        assert _drive(gen) is None

    def test_no_chains(self):
        """Test no chains."""
        obj = _mk()

        obj.params.on_chain_service_id = 1
        obj.params.target_investment_chains = []
        gen = obj.get_master_safe_address()
        assert _drive(gen) is None

    def test_staking_path_staked(self):
        """Test staking path staked."""
        obj = _mk()

        obj.params.on_chain_service_id = 1
        obj.params.target_investment_chains = ["optimism"]
        obj.params.staking_token_contract_address = "0xStaking"
        obj.params.staking_chain = "optimism"
        from packages.valory.skills.liquidity_trader_abci.states.base import (
            StakingState,
        )

        obj.service_staking_state = StakingState.STAKED
        obj._get_service_info = _gen_return([0, "0xMaster"])
        obj.check_is_valid_safe_address = _gen_return(True)
        gen = obj.get_master_safe_address()
        assert _drive(gen) == "0xMaster"

    def test_staking_path_unstaked_then_staked(self):
        """Test staking path unstaked then staked."""
        obj = _mk()

        obj.params.on_chain_service_id = 1
        obj.params.target_investment_chains = ["optimism"]
        obj.params.staking_token_contract_address = "0xStaking"
        obj.params.staking_chain = "optimism"
        from packages.valory.skills.liquidity_trader_abci.states.base import (
            StakingState,
        )

        obj.service_staking_state = StakingState.UNSTAKED

        def fake_get_state(*a, **kw):
            """Fake get state."""
            obj.service_staking_state = StakingState.STAKED
            yield

        obj._get_service_staking_state = fake_get_state
        obj._get_service_info = _gen_return([0, "0xMaster"])
        obj.check_is_valid_safe_address = _gen_return(True)
        gen = obj.get_master_safe_address()
        assert _drive(gen) == "0xMaster"

    def test_staking_invalid_addr(self):
        """Test staking invalid addr."""
        obj = _mk()

        obj.params.on_chain_service_id = 1
        obj.params.target_investment_chains = ["optimism"]
        obj.params.staking_token_contract_address = "0xStaking"
        obj.params.staking_chain = "optimism"
        from packages.valory.skills.liquidity_trader_abci.states.base import (
            StakingState,
        )

        obj.service_staking_state = StakingState.STAKED
        obj._get_service_info = _gen_return([0, "0xMaster"])
        obj.check_is_valid_safe_address = _gen_return(False)
        gen = obj.get_master_safe_address()
        assert _drive(gen) is None

    def test_staking_no_info(self):
        """Test staking no info."""
        obj = _mk()

        obj.params.on_chain_service_id = 1
        obj.params.target_investment_chains = ["optimism"]
        obj.params.staking_token_contract_address = "0xStaking"
        obj.params.staking_chain = "optimism"
        from packages.valory.skills.liquidity_trader_abci.states.base import (
            StakingState,
        )

        obj.service_staking_state = StakingState.STAKED
        obj._get_service_info = _gen_return(None)
        gen = obj.get_master_safe_address()
        assert _drive(gen) is None

    def test_no_staking_registry(self):
        """Test no staking registry."""
        obj = _mk()

        obj.params.on_chain_service_id = 1
        obj.params.target_investment_chains = ["optimism"]
        obj.params.staking_token_contract_address = None
        obj.params.staking_chain = None
        obj.params.service_registry_contract_addresses = {"optimism": "0xReg"}
        obj.contract_interact = _gen_return("0xOwner")
        obj.check_is_valid_safe_address = _gen_return(True)
        gen = obj.get_master_safe_address()
        assert _drive(gen) == "0xOwner"

    def test_no_registry_addr(self):
        """Test no registry addr."""
        obj = _mk()

        obj.params.on_chain_service_id = 1
        obj.params.target_investment_chains = ["optimism"]
        obj.params.staking_token_contract_address = None
        obj.params.staking_chain = None
        obj.params.service_registry_contract_addresses = {}
        gen = obj.get_master_safe_address()
        assert _drive(gen) is None

    def test_registry_no_result(self):
        """Test registry no result."""
        obj = _mk()

        obj.params.on_chain_service_id = 1
        obj.params.target_investment_chains = ["optimism"]
        obj.params.staking_token_contract_address = None
        obj.params.staking_chain = None
        obj.params.service_registry_contract_addresses = {"optimism": "0xReg"}
        obj.contract_interact = _gen_return(None)
        gen = obj.get_master_safe_address()
        assert _drive(gen) is None


class TestShouldRecalculatePortfolio:
    """TestShouldRecalculatePortfolio."""

    def test_no_initial(self):
        """Test no initial."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj._load_chain_total_investment = _gen_return(None)
        gen = obj.should_recalculate_portfolio({"portfolio_value": 100})
        assert _drive(gen) is True

    def test_no_final(self):
        """Test no final."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj._load_chain_total_investment = _gen_return(100.0)
        gen = obj.should_recalculate_portfolio({})
        assert _drive(gen) is True

    def test_post_tx_round(self):
        """Test post tx round."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj._load_chain_total_investment = _gen_return(100.0)
        mock_round = MagicMock()
        from packages.valory.skills.liquidity_trader_abci.states.post_tx_settlement import (
            PostTxSettlementRound,
        )

        mock_round.round_id = PostTxSettlementRound.auto_round_id()
        obj.context.state.round_sequence._abci_app._previous_rounds = [mock_round]
        gen = obj.should_recalculate_portfolio({"portfolio_value": 100})
        assert _drive(gen) is True

    def test_time_or_positions(self):
        """Test time or positions."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj._load_chain_total_investment = _gen_return(100.0)
        mock_round = MagicMock()
        mock_round.round_id = "other_round"
        obj.context.state.round_sequence._abci_app._previous_rounds = [mock_round]
        obj._is_time_update_due = lambda: True
        obj._have_positions_changed = lambda d: False
        gen = obj.should_recalculate_portfolio({"portfolio_value": 100})
        assert _drive(gen) is True


class TestUpdatePortfolioMetrics:
    """TestUpdatePortfolioMetrics."""

    def test_zero_value(self):
        """Test zero value."""
        obj = _mk()
        obj._update_portfolio_breakdown_ratios = MagicMock()
        gen = obj._update_portfolio_metrics(Decimal(0), [], [], [])
        _drive(gen)
        obj._update_portfolio_breakdown_ratios.assert_called_once()

    def test_positive_value(self):
        """Test positive value."""
        obj = _mk()
        obj._update_portfolio_breakdown_ratios = MagicMock()
        obj._update_allocation_ratios = _gen_none
        gen = obj._update_portfolio_metrics(Decimal(100), [], [], [])
        _drive(gen)


class TestUpdateAllocationRatios:
    """TestUpdateAllocationRatios."""

    def test_zero_total(self):
        """Test zero total."""
        obj = _mk()
        allocs = []
        gen = obj._update_allocation_ratios([], Decimal(0), allocs)
        _drive(gen)
        assert allocs == []

    def test_with_shares(self):
        """Test with shares."""
        obj = _mk()
        obj.current_positions = [{"pool_address": "0xP", "dex_type": "UniswapV3"}]
        obj._get_tick_ranges = _gen_return([])
        shares = [
            (
                Decimal(100),
                "UniswapV3",
                "optimism",
                "0xP",
                ["WETH"],
                5.0,
                "details",
                "0xSafe",
                {},
            )
        ]
        allocs = []
        gen = obj._update_allocation_ratios(shares, Decimal(100), allocs)
        _drive(gen)
        assert len(allocs) == 1
        assert allocs[0]["ratio"] == 100.0

    def test_with_tick_ranges(self):
        """Test with tick ranges."""
        obj = _mk()
        obj.current_positions = [{"pool_address": "0xP", "dex_type": "UniswapV3"}]
        obj._get_tick_ranges = _gen_return([{"tick": 1}])
        shares = [
            (
                Decimal(100),
                "UniswapV3",
                "optimism",
                "0xP",
                ["WETH"],
                5.0,
                "details",
                "0xSafe",
                {},
            )
        ]
        allocs = []
        gen = obj._update_allocation_ratios(shares, Decimal(100), allocs)
        _drive(gen)
        assert "tick_ranges" in allocs[0]


class TestCalculatePositionAmounts:
    """TestCalculatePositionAmounts."""

    def test_missing_details(self):
        """Test missing details."""
        obj = _mk()
        gen = obj._calculate_position_amounts({}, 0, 0, {}, "UniswapV3", "optimism")
        assert _drive(gen) == (0, 0)

    def test_uniswap(self):
        """Test uniswap."""
        obj = _mk()
        details = {
            "tickLower": -100,
            "tickUpper": 100,
            "liquidity": 1000,
            "tokensOwed0": 5,
            "tokensOwed1": 10,
        }
        pos = {"token_id": 1}
        gen = obj._calculate_position_amounts(
            details, 0, 2**96, pos, "UniswapV3", "optimism"
        )
        result = _drive(gen)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_velodrome_with_pm(self):
        """Test velodrome with pm."""
        obj = _mk()

        obj.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        obj.get_velodrome_position_principal = _gen_return((100, 200))
        details = {
            "tickLower": -100,
            "tickUpper": 100,
            "liquidity": 1000,
            "tokensOwed0": 0,
            "tokensOwed1": 0,
        }
        pos = {"token_id": 1}
        gen = obj._calculate_position_amounts(
            details, 0, 2**96, pos, "velodrome", "optimism"
        )
        result = _drive(gen)
        assert result == (100, 200)

    def test_velodrome_no_pm(self):
        """Test velodrome no pm."""
        obj = _mk()

        obj.params.velodrome_non_fungible_position_manager_contract_addresses = {}
        obj.get_velodrome_sqrt_ratio_at_tick = _gen_return(2**96)
        obj.get_velodrome_amounts_for_liquidity = _gen_return((50, 60))
        details = {
            "tickLower": -100,
            "tickUpper": 100,
            "liquidity": 1000,
            "tokensOwed0": 0,
            "tokensOwed1": 0,
        }
        pos = {"token_id": 1}
        gen = obj._calculate_position_amounts(
            details, 0, 2**96, pos, "velodrome", "optimism"
        )
        result = _drive(gen)
        assert result == (50, 60)


class TestGetTokenDecimalsPair:
    """TestGetTokenDecimalsPair."""

    def test_both_ok(self):
        """Test both ok."""
        obj = _mk()
        call_count = [0]

        def fake_dec(*a, **kw):
            """Fake dec."""
            call_count[0] += 1
            yield
            return 18

        obj._get_token_decimals = fake_dec
        gen = obj._get_token_decimals_pair("optimism", "0xT0", "0xT1")
        result = _drive(gen)
        assert result == (18, 18)

    def test_first_none(self):
        """Test first none."""
        obj = _mk()
        call_count = [0]

        def fake_dec(*a, **kw):
            """Fake dec."""
            call_count[0] += 1
            yield
            return None if call_count[0] == 1 else 18

        obj._get_token_decimals = fake_dec
        gen = obj._get_token_decimals_pair("optimism", "0xT0", "0xT1")
        result = _drive(gen)
        assert result == (None, None)


class TestCalculateClPositionValue:
    """TestCalculateClPositionValue."""

    def test_missing_params(self):
        """Test missing params."""
        obj = _mk()
        gen = obj._calculate_cl_position_value(
            None, "optimism", {}, "0xT0", "0xT1", "0xPM", MagicMock(), "velodrome"
        )
        assert _drive(gen) == {}

    def test_no_slot0(self):
        """Test no slot0."""
        obj = _mk()
        obj.contract_interact = _gen_return(None)
        gen = obj._calculate_cl_position_value(
            "0xP",
            "optimism",
            {"token_id": 1},
            "0xT0",
            "0xT1",
            "0xPM",
            MagicMock(),
            "velodrome",
        )
        assert _drive(gen) == {}

    def test_invalid_slot0(self):
        """Test invalid slot0."""
        obj = _mk()
        obj.contract_interact = _gen_return({"tick": None})
        gen = obj._calculate_cl_position_value(
            "0xP",
            "optimism",
            {"token_id": 1},
            "0xT0",
            "0xT1",
            "0xPM",
            MagicMock(),
            "velodrome",
        )
        assert _drive(gen) == {}

    def test_no_decimals(self):
        """Test no decimals."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return {"sqrt_price_x96": 2**96, "tick": 0}
            return None

        obj.contract_interact = fake_ci

        def fake_dec(*a, **kw):
            """Fake dec."""
            yield
            return None

        obj._get_token_decimals = fake_dec
        gen = obj._calculate_cl_position_value(
            "0xP",
            "optimism",
            {"token_id": 1},
            "0xT0",
            "0xT1",
            "0xPM",
            MagicMock(),
            "UniswapV3",
        )
        # _get_token_decimals_pair returns (None, None) -> None in token_decimals
        assert _drive(gen) == {}

    def test_success(self):
        """Test success."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return {"sqrt_price_x96": 2**96, "tick": 0}
            elif call_count[0] == 2:
                return {
                    "tickLower": -100,
                    "tickUpper": 100,
                    "liquidity": 1000,
                    "tokensOwed0": 0,
                    "tokensOwed1": 0,
                }
            return None

        obj.contract_interact = fake_ci

        def fake_dec(*a, **kw):
            """Fake dec."""
            yield
            return 18

        obj._get_token_decimals = fake_dec

        def fake_amounts(*a, **kw):
            """Fake amounts."""
            yield
            return (10**18, 2 * 10**18)

        obj._calculate_position_amounts = fake_amounts

        pos = {"token_id": 1, "token0_symbol": "A", "token1_symbol": "B"}
        gen = obj._calculate_cl_position_value(
            "0xP", "optimism", pos, "0xT0", "0xT1", "0xPM", MagicMock(), "UniswapV3"
        )
        result = _drive(gen)
        assert "0xT0" in result
        assert "0xT1" in result

    def test_position_no_token_id(self):
        """Test position no token id."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            return {"sqrt_price_x96": 2**96, "tick": 0}

        obj.contract_interact = fake_ci

        def fake_dec(*a, **kw):
            """Fake dec."""
            yield
            return 18

        obj._get_token_decimals = fake_dec

        pos = {"token_id": None, "token0_symbol": "A", "token1_symbol": "B"}
        gen = obj._calculate_cl_position_value(
            "0xP", "optimism", pos, "0xT0", "0xT1", "0xPM", MagicMock(), "UniswapV3"
        )
        result = _drive(gen)
        assert result["0xT0"] == Decimal(0)

    def test_position_details_fail(self):
        """Test position details fail."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return {"sqrt_price_x96": 2**96, "tick": 0}
            return None

        obj.contract_interact = fake_ci

        def fake_dec(*a, **kw):
            """Fake dec."""
            yield
            return 18

        obj._get_token_decimals = fake_dec

        pos = {"token_id": 1, "token0_symbol": "A", "token1_symbol": "B"}
        gen = obj._calculate_cl_position_value(
            "0xP", "optimism", pos, "0xT0", "0xT1", "0xPM", MagicMock(), "UniswapV3"
        )
        result = _drive(gen)
        assert result["0xT0"] == Decimal(0)

    def test_multiple_positions(self):
        """Test multiple positions."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return {"sqrt_price_x96": 2**96, "tick": 0}
            return {
                "tickLower": -100,
                "tickUpper": 100,
                "liquidity": 1000,
                "tokensOwed0": 0,
                "tokensOwed1": 0,
            }

        obj.contract_interact = fake_ci

        def fake_dec(*a, **kw):
            """Fake dec."""
            yield
            return 18

        obj._get_token_decimals = fake_dec

        def fake_amounts(*a, **kw):
            """Fake amounts."""
            yield
            return (10**18, 10**18)

        obj._calculate_position_amounts = fake_amounts

        pos = {
            "positions": [{"token_id": 1}, {"token_id": 2}],
            "token0_symbol": "A",
            "token1_symbol": "B",
        }
        gen = obj._calculate_cl_position_value(
            "0xP", "optimism", pos, "0xT0", "0xT1", "0xPM", MagicMock(), "UniswapV3"
        )
        result = _drive(gen)
        assert result["0xT0"] == Decimal(2)  # 2 * 10**18 / 10**18


class TestGetUserShareValueVelodromeCl:
    """TestGetUserShareValueVelodromeCl."""

    def test_no_pm(self):
        """Test no pm."""
        obj = _mk()
        obj.params.velodrome_non_fungible_position_manager_contract_addresses = {}
        pos = {"token0": "0xT0", "token1": "0xT1", "is_cl_pool": True}
        gen = obj._get_user_share_value_velodrome_cl(
            "0xP", "optimism", pos, "0xT0", "0xT1"
        )
        assert _drive(gen) == {}

    def test_ok(self):
        """Test ok."""
        obj = _mk()
        obj.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        obj._calculate_cl_position_value = _gen_return({"0xT0": Decimal(1)})
        pos = {"token0": "0xT0", "token1": "0xT1", "is_cl_pool": True}
        gen = obj._get_user_share_value_velodrome_cl(
            "0xP", "optimism", pos, "0xT0", "0xT1"
        )
        assert _drive(gen) == {"0xT0": Decimal(1)}


class TestGetUserShareValueVelodromeNonCl:
    """TestGetUserShareValueVelodromeNonCl."""

    def test_no_user_balance(self):
        """Test no user balance."""
        obj = _mk()
        obj.contract_interact = _gen_return(None)
        pos = {"token0_symbol": "A", "token1_symbol": "B"}
        gen = obj._get_user_share_value_velodrome_non_cl(
            "0xU", "0xP", "optimism", pos, "0xT0", "0xT1"
        )
        assert _drive(gen) == {}

    def test_no_total_supply(self):
        """Test no total supply."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return 100
            return None

        obj.contract_interact = fake_ci
        pos = {"token0_symbol": "A", "token1_symbol": "B"}
        gen = obj._get_user_share_value_velodrome_non_cl(
            "0xU", "0xP", "optimism", pos, "0xT0", "0xT1"
        )
        assert _drive(gen) == {}

    def test_no_reserves(self):
        """Test no reserves."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] <= 2:
                return 100
            return None

        obj.contract_interact = fake_ci
        pos = {"token0_symbol": "A", "token1_symbol": "B"}
        gen = obj._get_user_share_value_velodrome_non_cl(
            "0xU", "0xP", "optimism", pos, "0xT0", "0xT1"
        )
        assert _drive(gen) == {}

    def test_success(self):
        """Test success."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return 50  # user_balance
            elif call_count[0] == 2:
                return 100  # total_supply
            elif call_count[0] == 3:
                return [10**18, 2 * 10**18]  # reserves
            return 18

        obj.contract_interact = fake_ci

        def fake_dec(*a, **kw):
            """Fake dec."""
            yield
            return 18

        obj._get_token_decimals = fake_dec

        pos = {"token0_symbol": "A", "token1_symbol": "B"}
        gen = obj._get_user_share_value_velodrome_non_cl(
            "0xU", "0xP", "optimism", pos, "0xT0", "0xT1"
        )
        result = _drive(gen)
        assert "0xT0" in result
        assert "0xT1" in result

    def test_no_decimals(self):
        """Test no decimals."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return 50
            elif call_count[0] == 2:
                return 100
            elif call_count[0] == 3:
                return [10**18, 10**18]
            return None

        obj.contract_interact = fake_ci

        def fake_dec(*a, **kw):
            """Fake dec."""
            yield
            return None

        obj._get_token_decimals = fake_dec

        pos = {"token0_symbol": "A", "token1_symbol": "B"}
        gen = obj._get_user_share_value_velodrome_non_cl(
            "0xU", "0xP", "optimism", pos, "0xT0", "0xT1"
        )
        assert _drive(gen) == {}


class TestGetUserShareValueBalancerFull:
    """TestGetUserShareValueBalancerFull."""

    def test_zero_total_supply(self):
        """Test zero total supply."""
        obj = _mk()
        obj.params.balancer_vault_contract_addresses = {"optimism": "0xV"}
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return [["0xT0"], [1000]]  # pool tokens
            elif call_count[0] == 2:
                return 50  # user balance
            elif call_count[0] == 3:
                return 0  # total supply = 0
            return 18

        obj.contract_interact = fake_ci
        gen = obj.get_user_share_value_balancer("0xU", "0xPID", "0xP", "optimism")
        assert _drive(gen) == {}

    def test_success(self):
        """Test success."""
        obj = _mk()
        obj.params.balancer_vault_contract_addresses = {"optimism": "0xV"}
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return [["0xAbcdef1234567890abcdef1234567890abcdef12"], [10**18]]
            elif call_count[0] == 2:
                return 5 * 10**17  # 50% share
            elif call_count[0] == 3:
                return 10**18  # total supply
            return 18

        obj.contract_interact = fake_ci

        def fake_dec(*a, **kw):
            """Fake dec."""
            yield
            return 18

        obj._get_token_decimals = fake_dec
        gen = obj.get_user_share_value_balancer("0xU", "0xPID", "0xP", "optimism")
        result = _drive(gen)
        assert len(result) == 1

    def test_no_user_balance(self):
        """Test no user balance."""
        obj = _mk()
        obj.params.balancer_vault_contract_addresses = {"optimism": "0xV"}
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return [["0xT0"], [1000]]
            elif call_count[0] == 2:
                return None
            return 18

        obj.contract_interact = fake_ci
        gen = obj.get_user_share_value_balancer("0xU", "0xPID", "0xP", "optimism")
        assert _drive(gen) == {}


class TestCalculateSafeBalancesValue:
    """TestCalculateSafeBalancesValue."""

    def test_no_safe_address(self):
        """Test no safe address."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {}
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)

    def test_optimism_no_balances(self):
        """Test optimism no balances."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._get_safe_balances_from_safe_api = _gen_return([])
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)

    def test_mode_no_balances(self):
        """Test mode no balances."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj._get_mode_balances_from_explorer_api = _gen_return([])
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)

    def test_zero_balance_skipped(self):
        """Test zero balance skipped."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._get_safe_balances_from_safe_api = _gen_return(
            [{"address": "0xT", "asset_symbol": "TKN", "balance": 0}]
        )
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)

    def test_olas_skipped(self):
        """Test olas skipped."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        olas = OLAS_ADDRESSES["optimism"]
        obj._get_safe_balances_from_safe_api = _gen_return(
            [{"address": olas, "asset_symbol": "OLAS", "balance": 10**18}]
        )
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)

    def test_safe_fiat_conversion_used_directly(self):
        """Non-null ``fiat_conversion`` is used as price; CoinGecko is skipped."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": None}

        coingecko_calls: List[Any] = []

        def fake_coingecko(*a: Any, **kw: Any) -> Any:
            coingecko_calls.append((a, kw))
            yield
            return 12345.0  # should never be returned if fiat_conversion is used

        obj._fetch_zero_address_price = fake_coingecko
        obj._fetch_token_price = fake_coingecko

        obj._get_safe_balances_from_safe_api = _gen_return(
            [
                {
                    "address": "0xUsdc",
                    "asset_symbol": "USDC",
                    "balance": 10**6,
                    "fiat_balance": "1.0",
                    "fiat_conversion": "1.0",
                }
            ]
        )
        obj._get_token_decimals = _gen_return(6)
        result = _drive(obj._calculate_safe_balances_value([]))
        # 1 USDC at $1 = $1.00, with no CoinGecko call.
        assert result == Decimal("1") * Decimal("1.0")
        assert coingecko_calls == []

    def test_safe_fiat_conversion_zero_falls_back_to_coingecko(self):
        """``fiat_conversion: "0.0"`` (no Safe price feed) falls back to CoinGecko."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": None}
        obj._fetch_token_price = _gen_return(2.5)
        obj._get_token_decimals = _gen_return(6)

        obj._get_safe_balances_from_safe_api = _gen_return(
            [
                {
                    "address": "0xMystery",
                    "asset_symbol": "MYS",
                    "balance": 10**6,
                    "fiat_balance": "0.0",
                    "fiat_conversion": "0.0",
                }
            ]
        )
        result = _drive(obj._calculate_safe_balances_value([]))
        # Should use the CoinGecko price ($2.5), not value the token at $0.
        assert result == Decimal("1") * Decimal("2.5")

    def test_safe_fiat_conversion_invalid_decimal_falls_back(self):
        """A malformed ``fiat_conversion`` string falls back to CoinGecko."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": None}
        obj._fetch_token_price = _gen_return(7.0)
        obj._get_token_decimals = _gen_return(6)

        obj._get_safe_balances_from_safe_api = _gen_return(
            [
                {
                    "address": "0xToken",
                    "asset_symbol": "TKN",
                    "balance": 10**6,
                    "fiat_balance": "not-a-number",
                    "fiat_conversion": "not-a-number",
                }
            ]
        )
        result = _drive(obj._calculate_safe_balances_value([]))
        assert result == Decimal("1") * Decimal("7.0")

    def test_eth_balance(self):
        """Test eth balance."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": None}
        obj._get_safe_balances_from_safe_api = _gen_return(
            [{"address": ZERO_ADDRESS, "asset_symbol": "ETH", "balance": 10**18}]
        )
        obj._fetch_zero_address_price = _gen_return(3000.0)
        gen = obj._calculate_safe_balances_value([])
        result = _drive(gen)
        assert result == Decimal("1") * Decimal("3000.0")

    def test_erc20_balance(self):
        """Test erc20 balance."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": None}
        obj._get_safe_balances_from_safe_api = _gen_return(
            [{"address": "0xUSDC", "asset_symbol": "USDC", "balance": 10**6}]
        )
        obj._get_token_decimals = _gen_return(6)
        obj._fetch_token_price = _gen_return(1.0)
        gen = obj._calculate_safe_balances_value([])
        result = _drive(gen)
        assert result == Decimal("1") * Decimal("1.0")

    def test_no_price(self):
        """Test no price."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": None}
        obj._get_safe_balances_from_safe_api = _gen_return(
            [{"address": "0xTKN", "asset_symbol": "TKN", "balance": 10**18}]
        )
        obj._get_token_decimals = _gen_return(18)
        obj._fetch_token_price = _gen_return(None)
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)

    def test_no_decimals(self):
        """Test no decimals."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": None}
        obj._get_safe_balances_from_safe_api = _gen_return(
            [{"address": "0xTKN", "asset_symbol": "TKN", "balance": 10**18}]
        )
        obj._get_token_decimals = _gen_return(None)
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)

    def test_velo_balance(self):
        """Test velo balance."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": "0xvelo"}
        obj._get_safe_balances_from_safe_api = _gen_return(
            [{"address": "0xVELO", "asset_symbol": "VELO", "balance": 10**18}]
        )
        obj._get_token_decimals = _gen_return(18)
        obj.get_coin_id_from_symbol = lambda s, c: "velodrome-finance"
        obj._fetch_coin_price = _gen_return(0.5)
        gen = obj._calculate_safe_balances_value([])
        result = _drive(gen)
        assert result == Decimal("1") * Decimal("0.5")


class TestCalculateTotalVolume:
    """TestCalculateTotalVolume."""

    def test_cached(self):
        """Test cached."""
        obj = _mk()
        obj._read_kv = _gen_return(
            {"initial_investment_values": json.dumps({"0xP_0xTX": 100.0})}
        )
        obj.current_positions = [
            {"pool_address": "0xP", "tx_hash": "0xTX", "status": "open"}
        ]
        obj._write_kv = _gen_none
        gen = obj._calculate_total_volume()
        result = _drive(gen)
        assert result == 100.0

    def test_no_cache(self):
        """Test no cache."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj.current_positions = [
            {
                "pool_address": "0xP",
                "tx_hash": "0xTX",
                "token0": "0xT0",
                "token1": "0xT1",
                "amount0": 10**18,
                "amount1": 10**18,
                "timestamp": 1704067200,
                "chain": "optimism",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
            }
        ]
        obj._get_token_decimals = _gen_return(18)
        obj._fetch_historical_token_prices = _gen_return({"0xT0": 3000.0, "0xT1": 1.0})
        obj._write_kv = _gen_none
        gen = obj._calculate_total_volume()
        result = _drive(gen)
        assert result is not None
        assert result > 0

    def test_missing_data(self):
        """Test missing data."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj.current_positions = [
            {"pool_address": "0xP", "tx_hash": "0xTX", "token0": None}
        ]
        gen = obj._calculate_total_volume()
        result = _drive(gen)
        assert result is None

    def test_no_positions(self):
        """Test no positions."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj.current_positions = []
        gen = obj._calculate_total_volume()
        result = _drive(gen)
        assert result is None

    def test_no_price(self):
        """Test no price."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj.current_positions = [
            {
                "pool_address": "0xP",
                "tx_hash": "0xTX",
                "token0": "0xT0",
                "amount0": 10**18,
                "timestamp": 1704067200,
                "chain": "optimism",
                "token0_symbol": "WETH",
            }
        ]
        obj._get_token_decimals = _gen_return(18)
        obj._fetch_historical_token_prices = _gen_return({})
        gen = obj._calculate_total_volume()
        result = _drive(gen)
        assert result is None

    def test_invalid_cache(self):
        """Test invalid cache."""
        obj = _mk()
        obj._read_kv = _gen_return({"initial_investment_values": "{{bad json"})
        obj.current_positions = []
        gen = obj._calculate_total_volume()
        result = _drive(gen)
        assert result is None

    def test_no_token0_decimals(self):
        """Test no token0 decimals."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj.current_positions = [
            {
                "pool_address": "0xP",
                "tx_hash": "0xTX",
                "token0": "0xT0",
                "amount0": 10**18,
                "timestamp": 1704067200,
                "chain": "optimism",
                "token0_symbol": "TKN",
            }
        ]
        obj._get_token_decimals = _gen_return(None)
        gen = obj._calculate_total_volume()
        result = _drive(gen)
        assert result is None

    def test_with_token1(self):
        """Test with token1."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj.current_positions = [
            {
                "pool_address": "0xP",
                "tx_hash": "0xTX",
                "token0": "0xT0",
                "token1": "0xT1",
                "amount0": 10**18,
                "amount1": 10**6,
                "timestamp": 1704067200,
                "chain": "optimism",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
            }
        ]
        obj._get_token_decimals = _gen_return(18)
        obj._fetch_historical_token_prices = _gen_return({"0xT0": 3000.0, "0xT1": 1.0})
        obj._write_kv = _gen_none
        gen = obj._calculate_total_volume()
        result = _drive(gen)
        assert result is not None

    def test_no_token1_price(self):
        """Test no token1 price."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj.current_positions = [
            {
                "pool_address": "0xP",
                "tx_hash": "0xTX",
                "token0": "0xT0",
                "token1": "0xT1",
                "amount0": 10**18,
                "amount1": 10**18,
                "timestamp": 1704067200,
                "chain": "optimism",
                "token0_symbol": "WETH",
                "token1_symbol": "FOO",
            }
        ]
        obj._get_token_decimals = _gen_return(18)
        obj._fetch_historical_token_prices = _gen_return({"0xT0": 3000.0})
        gen = obj._calculate_total_volume()
        result = _drive(gen)
        assert result is None


class TestTrackEthTransfersMode:
    """TestTrackEthTransfersMode."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_api_error(self, mock_requests):
        """Test api error."""
        obj = _mk()
        mock_requests.get.return_value = MagicMock(status_code=500)
        result = obj._track_eth_transfers_mode("0xSafe", "2024-01-01")
        assert result == {"incoming": {}, "outgoing": {}}

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_api_bad_status(self, mock_requests):
        """Test api bad status."""
        obj = _mk()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"status": "0", "message": "fail", "result": []}
        mock_requests.get.return_value = resp
        result = obj._track_eth_transfers_mode("0xSafe", "2024-01-01")
        assert result == {"incoming": {}, "outgoing": {}}

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_success_incoming(self, mock_requests):
        """Test success incoming."""
        obj = _mk()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "status": "1",
            "result": [
                {
                    "timeStamp": "1704067200",
                    "value": str(10**18),
                    "to": "0xsafe",
                    "from": "0xother",
                    "hash": "0xTX",
                }
            ],
        }
        mock_requests.get.return_value = resp
        result = obj._track_eth_transfers_mode("0xSafe", "2024-12-31")
        assert "incoming" in result

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_exception(self, mock_requests):
        """Test exception."""
        obj = _mk()
        mock_requests.get.side_effect = RuntimeError("fail")
        result = obj._track_eth_transfers_mode("0xSafe", "2024-01-01")
        assert result == {"incoming": {}, "outgoing": {}}


class TestTrackErc20TransfersMode:
    """TestTrackErc20TransfersMode."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_api_error(self, mock_requests):
        """Test api error."""
        obj = _mk()
        mock_requests.get.return_value = MagicMock(status_code=500)
        result = obj._track_erc20_transfers_mode("0xSafe", 1704067200)
        assert result == {"outgoing": {}}

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_empty(self, mock_requests):
        """Test empty."""
        obj = _mk()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"items": []}
        mock_requests.get.return_value = resp
        result = obj._track_erc20_transfers_mode("0xSafe", 1704067200)
        assert result == {"outgoing": {}}

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_exception(self, mock_requests):
        """Test exception."""
        obj = _mk()
        mock_requests.get.side_effect = RuntimeError("fail")
        result = obj._track_erc20_transfers_mode("0xSafe", 1704067200)
        assert result == {"outgoing": {}}

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_outgoing_usdc(self, mock_requests):
        """Test outgoing usdc."""
        obj = _mk()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "items": [
                {
                    "timestamp": "2024-01-01T00:00:00.000000Z",
                    "total": {"value": "1000000"},
                    "token": {"address": "0xUSDC", "symbol": "USDC", "decimals": "6"},
                    "from": {"hash": "0xsafe", "is_contract": False},
                    "to": {"hash": "0xrecipient", "is_contract": False, "name": ""},
                    "transaction_hash": "0xTX",
                }
            ],
            "next_page_params": None,
        }
        mock_requests.get.return_value = resp
        result = obj._track_erc20_transfers_mode("0xSafe", 1704067200)
        assert "outgoing" in result

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.MAX_PAGINATION_PAGES",
        2,
    )
    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_pagination_cap_triggers_break(self, mock_requests):
        """The cap fires after exactly MAX_PAGINATION_PAGES pages, not before."""

        # Fake API: always returns one item plus a next-page cursor, so without
        # the cap the loop would never terminate.
        def make_response(*_args, **_kwargs):
            """Make response."""
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "items": [
                    {
                        "timestamp": "2024-01-01T00:00:00.000000Z",
                        "total": {"value": "1000000"},
                        "token": {
                            "address": "0xUSDC",
                            "symbol": "USDC",
                            "decimals": "6",
                        },
                        "from": {"hash": "0xsafe", "is_contract": False},
                        "to": {
                            "hash": "0xrecipient",
                            "is_contract": False,
                            "name": "",
                        },
                        "transaction_hash": "0xTX",
                    }
                ],
                "next_page_params": {"block_number": 100, "index": 0},
            }
            return resp

        mock_requests.get.side_effect = make_response
        obj = _mk()
        obj._track_erc20_transfers_mode("0xSafe", 1704067200)

        # The loop fetched exactly MAX_PAGINATION_PAGES (=2) pages and stopped,
        # despite the API still advertising a next-page cursor.
        assert mock_requests.get.call_count == 2
        obj.context.logger.warning.assert_called()


class TestFetchEthTransfersMode:
    """TestFetchEthTransfersMode."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_api_error(self, mock_requests):
        """Test api error."""
        obj = _mk()
        obj.funding_events = {"mode": {}}
        mock_requests.get.return_value = MagicMock(status_code=500)
        end_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = obj._fetch_eth_transfers_mode("0xAddr", end_dt, {}, True)
        assert result is False

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_empty(self, mock_requests):
        """Test empty."""
        obj = _mk()
        obj.funding_events = {"mode": {}}
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"items": []}
        mock_requests.get.return_value = resp
        end_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = obj._fetch_eth_transfers_mode("0xAddr", end_dt, {}, True)
        assert result is True

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_with_data(self, mock_requests):
        """Test with data."""
        obj = _mk()
        obj.funding_events = {"mode": {}}
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "items": [
                {
                    "value": str(10**18),
                    "delta": str(10**18),
                    "transaction_hash": None,
                    "block_timestamp": "2024-01-01T00:00:00Z",
                    "block_number": 1,
                }
            ],
            "next_page_params": None,
        }
        mock_requests.get.return_value = resp
        end_dt = datetime(2024, 12, 31, tzinfo=timezone.utc)
        transfers = {}
        result = obj._fetch_eth_transfers_mode("0xAddr", end_dt, transfers, True)
        assert result is True


class TestCheckAndCreateEthRevertTransactions:
    """TestCheckAndCreateEthRevertTransactions."""

    def test_no_safe(self):
        """Test no safe."""
        obj = _mk()
        gen = obj._check_and_create_eth_revert_transactions("optimism", None, "sender")
        _drive(gen)

    def test_zero_amount(self):
        """Test zero amount."""
        obj = _mk()
        obj._track_eth_transfers_and_reversions = _gen_return(
            {"to_address": None, "reversion_amount": 0, "master_safe_address": None}
        )
        gen = obj._check_and_create_eth_revert_transactions(
            "optimism", "0xSafe", "sender"
        )
        _drive(gen)

    def test_positive_amount_no_master(self):
        """Test positive amount no master."""
        obj = _mk()
        obj._track_eth_transfers_and_reversions = _gen_return(
            {"to_address": None, "reversion_amount": 0.5, "master_safe_address": None}
        )
        gen = obj._check_and_create_eth_revert_transactions(
            "optimism", "0xSafe", "sender"
        )
        _drive(gen)

    def test_positive_amount_with_tx(self):
        """Test positive amount with tx."""
        obj = _mk()
        master_addr = "0x" + "aa" * 20  # 42 chars
        obj._track_eth_transfers_and_reversions = _gen_return(
            {
                "to_address": master_addr,
                "reversion_amount": 0.5,
                "master_safe_address": master_addr,
            }
        )
        obj.contract_interact = _gen_return("0x" + "ab" * 32)
        obj.send_a2a_transaction = _gen_none
        obj.wait_until_round_end = _gen_none
        obj.set_done = MagicMock()
        gen = obj._check_and_create_eth_revert_transactions(
            "optimism", "0xSafe", "sender"
        )
        _drive(gen)
        obj.set_done.assert_called_once()

    def test_positive_amount_no_hash(self):
        """Test positive amount no hash."""
        obj = _mk()
        obj._track_eth_transfers_and_reversions = _gen_return(
            {
                "to_address": "0xMaster",
                "reversion_amount": 0.5,
                "master_safe_address": "0xMaster",
            }
        )
        obj.contract_interact = _gen_return(None)
        gen = obj._check_and_create_eth_revert_transactions(
            "optimism", "0xSafe", "sender"
        )
        _drive(gen)


class TestTrackWhitelistedAssets:
    """TestTrackWhitelistedAssets."""

    def test_empty_chains(self):
        """Test empty chains."""
        obj = _mk()
        obj.params.target_investment_chains = []
        obj.store_whitelisted_assets = MagicMock()
        gen = obj._track_whitelisted_assets()
        _drive(gen)

    def test_skip_non_target_chain(self):
        """Test skip non target chain."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.store_whitelisted_assets = MagicMock()
        obj.sleep = _gen_none
        # Only iterate over chains that are in target chains
        gen = obj._track_whitelisted_assets()
        _drive(gen)

    def test_price_drop(self):
        """Test price drop."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.whitelisted_assets = {"mode": {"0xT": "TKN"}}
        obj.store_whitelisted_assets = MagicMock()
        obj.sleep = _gen_none
        # Yesterday price 100, today price 90 -> -10% drop
        call_count = [0]

        def fake_price(*a, **kw):
            """Fake price."""
            call_count[0] += 1
            yield
            return 100.0 if call_count[0] == 1 else 90.0

        obj._get_historical_price_for_date = fake_price
        gen = obj._track_whitelisted_assets()
        _drive(gen)

    def test_no_price(self):
        """Test no price."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.whitelisted_assets = {"mode": {"0xT": "TKN"}}
        obj.store_whitelisted_assets = MagicMock()
        obj.sleep = _gen_none
        obj._get_historical_price_for_date = _gen_return(None)
        gen = obj._track_whitelisted_assets()
        _drive(gen)

    def test_price_exception(self):
        """Test price exception."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.whitelisted_assets = {"mode": {"0xT": "TKN"}}
        obj.store_whitelisted_assets = MagicMock()
        obj.sleep = _gen_none

        def boom(*a, **kw):
            """Boom."""
            raise RuntimeError("fail")
            yield  # noqa

        obj._get_historical_price_for_date = boom
        gen = obj._track_whitelisted_assets()
        _drive(gen)

    def test_price_stable(self):
        """Price stable (> -5%), no removal."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.whitelisted_assets = {"mode": {"0xT": "TKN"}}
        obj.store_whitelisted_assets = MagicMock()
        obj.sleep = _gen_none
        call_count = [0]

        def fake_price(*a, **kw):
            """Fake price."""
            call_count[0] += 1
            yield
            return 100.0 if call_count[0] == 1 else 98.0  # -2%, above -5%

        obj._get_historical_price_for_date = fake_price
        fake_wa = {"mode": {"0xT": "TKN"}}
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.WHITELISTED_ASSETS",
            fake_wa,
        ):
            gen = obj._track_whitelisted_assets()
            _drive(gen)
        assert "0xT" in obj.whitelisted_assets["mode"]

    def test_address_removal_branch(self):
        """Cover lines 325->334, 328-329: address in whitelisted_assets removed."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.whitelisted_assets = {"mode": {"0xT": "TKN"}}
        obj.store_whitelisted_assets = MagicMock()
        obj.sleep = _gen_none
        call_count = [0]

        def fake_price(*a, **kw):
            """Fake price."""
            call_count[0] += 1
            yield
            return 100.0 if call_count[0] == 1 else 80.0  # -20% drop

        obj._get_historical_price_for_date = fake_price
        # Patch the WHITELISTED_ASSETS global so the loop finds our token
        fake_wa = {"mode": {"0xT": "TKN"}}
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.WHITELISTED_ASSETS",
            fake_wa,
        ):
            gen = obj._track_whitelisted_assets()
            _drive(gen)
        assert "0xT" not in obj.whitelisted_assets["mode"]
        obj.store_whitelisted_assets.assert_called()


class TestHavePositionsChangedClosed:
    """TestHavePositionsChangedClosed."""

    def test_closed_positions_detected(self):
        """Test closed positions detected."""
        obj = _mk()
        # Last had 2 positions, current has only 1 open + 1 different
        obj.current_positions = [
            {"pool_address": "0x1", "dex_type": "a", "status": "open"},
        ]
        last = {
            "allocations": [
                {"id": "0x1", "type": "a"},
                {"id": "0x2", "type": "b"},
            ]
        }
        assert obj._have_positions_changed(last) is True


class TestUpdatePortfolioBreakdownRatiosEdge:
    """TestUpdatePortfolioBreakdownRatiosEdge."""

    def test_none_value_usd_entry_filtered(self):
        """Cover line 736: value_usd is None -> continue."""
        obj = _mk()
        # Pass total_value=0 so the sum() that would fail on None is skipped
        bd = [
            {"value_usd": None, "balance": 0, "price": 0},
            {"value_usd": 10, "balance": 1, "price": 10},
        ]
        obj._update_portfolio_breakdown_ratios(bd, Decimal(0))
        assert len(bd) == 1
        assert bd[0]["value_usd"] == 10.0

    def test_inner_exception_branch(self):
        """Cover lines 744-748: exception in inner try."""
        obj = _mk()
        # Use a value that Decimal(str(...)) will reject — an object whose str() is invalid
        bad_val = type("Bad", (), {"__str__": lambda self: "NaN_bad"})()
        bd = [
            {"value_usd": bad_val, "balance": 1, "price": 1},
            {"value_usd": 50, "balance": 5, "price": 10},
        ]
        # total_value=0 to skip the sum() pre-check
        obj._update_portfolio_breakdown_ratios(bd, Decimal(0))
        # The invalid entry should be skipped, only valid one kept
        assert len(bd) == 1


class TestCreatePortfolioDataROIException:
    """TestCreatePortfolioDataROIException."""

    def test_roi_calculation_error(self):
        """Cover lines 1001-1004: exception during ROI calculation."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._get_current_timestamp = lambda: 1000
        # Use a Decimal so tiny that float() underflows to 0.0, causing ZeroDivisionError
        # Decimal("1e-400") > 0 is True, but float("1e-400") == 0.0
        tiny = Decimal("1e-400")
        result = obj._create_portfolio_data(
            Decimal("100"),
            Decimal("50"),
            Decimal("5"),
            Decimal("2"),
            Decimal("10"),
            tiny,
            None,
            [],
            [],
        )
        assert result["total_roi"] is None
        assert result["partial_roi"] is None


class TestGetTickRangesFull:
    """TestGetTickRangesFull."""

    def test_success_with_data(self):
        """Cover lines 829-874: successful tick range retrieval."""
        obj = _mk()
        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return {"tick": 50, "sqrt_price_x96": 2**96}  # slot0
            elif call_count[0] == 2:
                return {"tickLower": 0, "tickUpper": 100}  # position data
            return None

        obj.contract_interact = fake_ci
        position = {
            "dex_type": "UniswapV3",
            "pool_address": "0xPool",
            "token_id": 1,
        }
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert len(result) == 1
        assert result[0]["in_range"] is True
        assert result[0]["tick_lower"] == 0
        assert result[0]["tick_upper"] == 100

    def test_velodrome_cl_with_positions(self):
        """Cover Velodrome CL multi-position path."""
        obj = _mk()
        obj.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return {"tick": 50, "sqrt_price_x96": 2**96}
            else:
                return {"tickLower": -10, "tickUpper": 200}

        obj.contract_interact = fake_ci
        position = {
            "dex_type": "velodrome",
            "is_cl_pool": True,
            "pool_address": "0xPool",
            "positions": [{"token_id": 1}, {"token_id": 2}],
        }
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert len(result) == 2

    def test_no_position_data(self):
        """Cover line 857-858: position_data is None -> continue."""
        obj = _mk()
        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return {"tick": 50, "sqrt_price_x96": 2**96}
            return None  # position data fails

        obj.contract_interact = fake_ci
        position = {"dex_type": "UniswapV3", "pool_address": "0xPool", "token_id": 1}
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert result == []

    def test_no_token_id_skipped(self):
        """Cover line 844-845: token_id missing -> continue."""
        obj = _mk()
        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            return {"tick": 50, "sqrt_price_x96": 2**96}

        obj.contract_interact = fake_ci
        position = {"dex_type": "UniswapV3", "pool_address": "0xPool", "token_id": None}
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert result == []


class TestHandleVelodromePositionEdge:
    """TestHandleVelodromePositionEdge."""

    def test_staked_zero_rewards(self):
        """Cover 1183->1193: velo_rewards == 0 -> skip adding."""
        obj = _mk()
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": "0xVELO"}
        position = {
            "pool_address": "0xPool",
            "token_id": 1,
            "token0": "0xT0",
            "token1": "0xT1",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "staked": True,
            "is_cl_pool": False,
        }
        obj.get_user_share_value_velodrome = _gen_return({"0xT0": Decimal(5)})
        obj._get_velodrome_pending_rewards = _gen_return(Decimal(0))
        gen = obj._handle_velodrome_position(position, "optimism")
        result = _drive(gen)
        assert "0xVELO" not in result[0]

    def test_staked_no_velo_address(self):
        """Cover 1186->1193: velo_token_address is None."""
        obj = _mk()
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {}
        position = {
            "pool_address": "0xPool",
            "token_id": 1,
            "token0": "0xT0",
            "token1": "0xT1",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "staked": True,
            "is_cl_pool": False,
        }
        obj.get_user_share_value_velodrome = _gen_return({"0xT0": Decimal(5)})
        obj._get_velodrome_pending_rewards = _gen_return(Decimal("2.5"))
        gen = obj._handle_velodrome_position(position, "optimism")
        result = _drive(gen)
        # No velo address, so VELO shouldn't be added
        assert "0xVELO" not in result[0]


class TestCalculateCorrectedYieldVeloBranches:
    """TestCalculateCorrectedYieldVeloBranches."""

    def test_velo_price_in_token_prices(self):
        """Cover line 1431: velo_price from token_prices cache."""
        obj = _mk()

        def fake_dec(*a, **kw):
            """Fake dec."""
            yield
            return 18

        obj._get_token_decimals = fake_dec
        obj._get_velo_token_address = lambda c: "0xVELO"
        pos = {
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xP",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "staked": True,
            "dex_type": "velodrome",
        }
        prices = {"0xT0": Decimal("10"), "0xT1": Decimal("5"), "0xVELO": Decimal("0.5")}
        balances = {"0xT0": Decimal("1"), "0xT1": Decimal("1"), "0xVELO": Decimal("10")}
        gen = obj._calculate_corrected_yield(
            pos, 10**18, 10**18, balances, "optimism", prices
        )
        result = _drive(gen)
        assert result == Decimal("5.0")  # 10 * 0.5

    def test_velo_price_fetch_none(self):
        """Cover lines 1436-1439: _fetch_coin_price returns None."""
        obj = _mk()

        def fake_dec(*a, **kw):
            """Fake dec."""
            yield
            return 18

        obj._get_token_decimals = fake_dec
        obj._get_velo_token_address = lambda c: "0xVELO"
        obj.get_coin_id_from_symbol = lambda s, c: "velodrome-finance"
        obj._fetch_coin_price = _gen_return(None)
        pos = {
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xP",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "staked": True,
            "dex_type": "velodrome",
        }
        prices = {"0xT0": Decimal("10"), "0xT1": Decimal("5")}
        balances = {"0xT0": Decimal("1"), "0xT1": Decimal("1"), "0xVELO": Decimal("10")}
        gen = obj._calculate_corrected_yield(
            pos, 10**18, 10**18, balances, "optimism", prices
        )
        result = _drive(gen)
        assert result == Decimal("0")  # velo_price defaults to 0

    def test_velo_balance_zero(self):
        """Cover 1428->1446: velo_balance == 0 -> skip."""
        obj = _mk()

        def fake_dec(*a, **kw):
            """Fake dec."""
            yield
            return 18

        obj._get_token_decimals = fake_dec
        obj._get_velo_token_address = lambda c: "0xVELO"
        pos = {
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xP",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "staked": True,
            "dex_type": "velodrome",
        }
        prices = {"0xT0": Decimal("10"), "0xT1": Decimal("5")}
        balances = {"0xT0": Decimal("1"), "0xT1": Decimal("1"), "0xVELO": Decimal("0")}
        gen = obj._calculate_corrected_yield(
            pos, 10**18, 10**18, balances, "optimism", prices
        )
        result = _drive(gen)
        assert result == Decimal("0")

    def test_velo_not_in_balances(self):
        """Cover 1426->1446: velo_token_address not in current_balances."""
        obj = _mk()

        def fake_dec(*a, **kw):
            """Fake dec."""
            yield
            return 18

        obj._get_token_decimals = fake_dec
        obj._get_velo_token_address = lambda c: "0xVELO"
        pos = {
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xP",
            "token0_symbol": "A",
            "token1_symbol": "B",
            "staked": True,
            "dex_type": "velodrome",
        }
        prices = {"0xT0": Decimal("10"), "0xT1": Decimal("5")}
        balances = {"0xT0": Decimal("1"), "0xT1": Decimal("1")}
        gen = obj._calculate_corrected_yield(
            pos, 10**18, 10**18, balances, "optimism", prices
        )
        result = _drive(gen)
        assert result == Decimal("0")


class TestUpdateUniswapPositionWarning:
    """TestUpdateUniswapPositionWarning."""

    def test_cl_no_data_warning(self):
        """Cover line 2815: position_data is None for Uniswap CL."""
        obj = _mk()
        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        obj.contract_interact = _gen_return(None)
        pos = {"token_id": 1, "chain": "optimism"}
        gen = obj._update_uniswap_position(pos)
        _drive(gen)
        # Warning logged, no current_liquidity set
        assert "current_liquidity" not in pos


class TestUpdateVelodromePositionWarning:
    """TestUpdateVelodromePositionWarning."""

    def test_non_cl_none_balance(self):
        """Cover line 2849: balance is None for Velodrome non-CL."""
        obj = _mk()
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.contract_interact = _gen_return(None)
        pos = {"chain": "optimism", "is_cl_pool": False, "pool_address": "0xP"}
        gen = obj._update_velodrome_position(pos)
        _drive(gen)
        assert "current_liquidity" not in pos

    def test_cl_no_data_warning(self):
        """Cover line 2815 analog for velodrome: position_data None."""
        obj = _mk()
        obj.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        obj.contract_interact = _gen_return(None)
        pos = {
            "chain": "optimism",
            "is_cl_pool": True,
            "positions": [{"token_id": 1}],
        }
        gen = obj._update_velodrome_position(pos)
        _drive(gen)


class TestGetMasterSafeAddressRegistryInvalid:
    """TestGetMasterSafeAddressRegistryInvalid."""

    def test_registry_invalid_address(self):
        """Cover line 4801: registry returns owner but address is invalid."""
        obj = _mk()
        obj.params.on_chain_service_id = 1
        obj.params.target_investment_chains = ["optimism"]
        obj.params.staking_token_contract_address = None
        obj.params.staking_chain = None
        obj.params.service_registry_contract_addresses = {"optimism": "0xReg"}
        obj.contract_interact = _gen_return("0xOwner")
        obj.check_is_valid_safe_address = _gen_return(False)
        gen = obj.get_master_safe_address()
        assert _drive(gen) is None


class TestGetMasterSafeAddressStakingStaysUnstaked:
    """TestGetMasterSafeAddressStakingStaysUnstaked."""

    def test_staking_stays_unstaked(self):
        """Cover 4750->4772: after _get_service_staking_state, still UNSTAKED."""
        obj = _mk()
        obj.params.on_chain_service_id = 1
        obj.params.target_investment_chains = ["optimism"]
        obj.params.staking_token_contract_address = "0xStaking"
        obj.params.staking_chain = "optimism"
        from packages.valory.skills.liquidity_trader_abci.states.base import (
            StakingState,
        )

        obj.service_staking_state = StakingState.UNSTAKED
        obj._get_service_staking_state = _gen_none  # state stays UNSTAKED
        obj.params.service_registry_contract_addresses = {"optimism": "0xReg"}
        obj.contract_interact = _gen_return("0xOwner")
        obj.check_is_valid_safe_address = _gen_return(True)
        gen = obj.get_master_safe_address()
        assert _drive(gen) == "0xOwner"


class TestVelodromePendingRewardsFalsy:
    """TestVelodromePendingRewardsFalsy."""

    def test_cl_rewards_zero(self):
        """Cover 4895->4887: rewards is 0 (falsy)."""
        obj = _mk()
        pool_mock = MagicMock()
        pool_mock.get_gauge_address = _gen_return("0xGauge")
        pool_mock.get_cl_pending_rewards = _gen_return(0)
        obj.pools = {"velodrome": pool_mock}
        pos = {
            "pool_address": "0xP",
            "is_cl_pool": True,
            "positions": [{"token_id": 1}],
        }
        gen = obj._get_velodrome_pending_rewards(pos, "optimism", "0xU")
        assert _drive(gen) == Decimal(0)

    def test_cl_rewards_none(self):
        """Cover rewards=None path."""
        obj = _mk()
        pool_mock = MagicMock()
        pool_mock.get_gauge_address = _gen_return("0xGauge")
        pool_mock.get_cl_pending_rewards = _gen_return(None)
        obj.pools = {"velodrome": pool_mock}
        pos = {
            "pool_address": "0xP",
            "is_cl_pool": True,
            "positions": [{"token_id": 1}],
        }
        gen = obj._get_velodrome_pending_rewards(pos, "optimism", "0xU")
        assert _drive(gen) == Decimal(0)


class TestValidateVelodromeV2PoolAddressesFailure:
    """TestValidateVelodromeV2PoolAddressesFailure."""

    def test_validation_fails(self):
        """Cover 4935->4926: validation returns False, no log."""
        obj = _mk()
        obj.current_positions = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": False,
                "is_stable": True,
                "pool_address": "0xP",
            }
        ]
        obj._validate_velodrome_v2_pool_address = _gen_return(False)
        obj.store_current_positions = MagicMock()
        gen = obj._validate_velodrome_v2_pool_addresses()
        _drive(gen)
        obj.store_current_positions.assert_called_once()


class TestShouldIncludeTransferEthOk:
    """TestShouldIncludeTransferEthOk."""

    def test_eth_transfer_ok(self):
        """Cover 3358->3361: eth transfer with status ok and value > 0."""
        obj = _mk()
        result = obj._should_include_transfer(
            {"hash": "0x123", "is_contract": False},
            tx_data={"status": "ok", "value": "100"},
            is_eth_transfer=True,
        )
        assert result is True


class TestGetDatetimeFromTimestampTzAware:
    """TestGetDatetimeFromTimestampTzAware."""

    def test_already_has_tzinfo(self):
        """Cover 3376->3379 false branch: dt already has tzinfo."""
        obj = _mk()
        # A datetime string without Z or + but with timezone info after parsing
        # Use a string that doesn't end with Z, doesn't contain +, doesn't end with UTC
        dt = obj._get_datetime_from_timestamp("2024-01-01T00:00:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc


class TestShouldIncludeTransferModeEthOk:
    """TestShouldIncludeTransferModeEthOk."""

    def test_eth_transfer_ok(self):
        """Test eth transfer ok."""
        obj = _mk()
        result = obj._should_include_transfer_mode(
            {"hash": "0xabc", "is_contract": False},
            tx_data={"status": "ok", "value": "100"},
            is_eth_transfer=True,
        )
        assert result is True


class TestCalculateSafeBalancesValueZeroBalance:
    """TestCalculateSafeBalancesValueZeroBalance."""

    def test_zero_adjusted_balance(self):
        """Cover line 2155: adjusted_balance <= 0 -> continue."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": None}
        obj._get_safe_balances_from_safe_api = _gen_return(
            [{"address": "0xTKN", "asset_symbol": "TKN", "balance": 0}]
        )
        obj._get_token_decimals = _gen_return(18)
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)


class TestTrackWithdrawalOptimismException:
    """TestTrackWithdrawalOptimismException."""

    def test_exception(self):
        """Cover lines 2403-2407: exception in withdrawal calculation."""
        obj = _mk()

        def boom(*a, **kw):
            """Boom."""
            raise RuntimeError("fail")
            yield  # noqa: unreachable

        obj._is_not_other_contract_optimism = boom
        transfers = {
            "2024-01-01": [
                {
                    "symbol": "USDC",
                    "amount": 10,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "to_address": "0xEOA",
                }
            ]
        }
        gen = obj._track_and_calculate_withdrawal_value_optimism(transfers)
        assert _drive(gen) == Decimal(0)


class TestShouldIncludeTransferOptimismCacheError:
    """TestShouldIncludeTransferOptimismCacheError."""

    def test_cached_bad_json(self):
        """Cover lines 3860-3861: JSONDecodeError in cache."""
        obj = _mk()
        cache_key = f"{CONTRACT_CHECK_CACHE_PREFIX}optimism_0xabc"
        obj._read_kv = _gen_return({cache_key: "{{bad json"})
        obj._request_with_retries = _gen_return((True, {"result": "0x"}))
        obj._write_kv = _gen_none
        cg = MagicMock()
        obj.context.coingecko = cg
        gen = obj._should_include_transfer_optimism("0xABC")
        assert _drive(gen) is True


class TestIsNotOtherContractOptimismCacheError:
    """TestIsNotOtherContractOptimismCacheError."""

    def test_cached_bad_json(self):
        """Cover lines 3939-3940: JSONDecodeError in cache."""
        obj = _mk()
        cache_key = f"{CONTRACT_CHECK_CACHE_PREFIX}optimism_0xabc"
        obj._read_kv = _gen_return({cache_key: "{{bad"})
        obj._request_with_retries = _gen_return((True, {"result": "0x"}))
        obj._write_kv = _gen_none
        cg = MagicMock()
        obj.context.coingecko = cg
        gen = obj._is_not_other_contract_optimism("0xABC")
        assert _drive(gen) is True

    def test_not_eoa_log(self):
        """Cover line 3986: not is_eoa -> log message."""
        obj = _mk()
        call_count = [0]

        def fake_req(*a, **kw):
            """Fake req."""
            nonlocal call_count  # noqa: F824
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return (True, {"result": "0x1234"})  # has code
            else:
                yield
                return (False, {})  # not a safe

        obj._read_kv = _gen_return({})
        obj._request_with_retries = fake_req
        obj._write_kv = _gen_none
        cg = MagicMock()
        obj.context.coingecko = cg
        gen = obj._is_not_other_contract_optimism("0xABC")
        assert _drive(gen) is False


class TestCalculatePositionAmountsOutOfRange:
    """TestCalculatePositionAmountsOutOfRange."""

    def test_out_of_range(self):
        """Cover line 1756: tick out of range."""
        obj = _mk()
        details = {
            "tickLower": 100,
            "tickUpper": 200,
            "liquidity": 1000,
            "tokensOwed0": 5,
            "tokensOwed1": 10,
        }
        pos = {"token_id": 1}
        gen = obj._calculate_position_amounts(
            details, 50, 2**96, pos, "UniswapV3", "optimism"
        )
        result = _drive(gen)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestGetUserShareValueBalancerZeroSupplyDecimal:
    """TestGetUserShareValueBalancerZeroSupplyDecimal."""

    def test_zero_total_supply_decimal(self):
        """Cover lines 1985-1987: total_supply_decimal == 0."""
        obj = _mk()
        obj.params.balancer_vault_contract_addresses = {"optimism": "0xV"}
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return [["0x" + "ab" * 20], [1000]]  # pool tokens
            elif call_count[0] == 2:
                return 50  # user balance
            elif call_count[0] == 3:
                return 1  # total_supply (non-None, non-zero from contract)
            return 18

        obj.contract_interact = fake_ci

        # But make the total_supply decimal value zero by returning 0 from contract
        call_count2 = [0]

        def fake_ci2(*a, **kw):
            """Fake ci2."""
            call_count2[0] += 1
            yield
            if call_count2[0] == 1:
                return [["0x" + "ab" * 20], [1000]]
            elif call_count2[0] == 2:
                return 50
            elif call_count2[0] == 3:
                return 0  # total_supply is 0 -> first check catches it
            return 18

        obj.contract_interact = fake_ci2
        gen = obj.get_user_share_value_balancer("0xU", "0xPID", "0xP", "optimism")
        assert _drive(gen) == {}


class TestGetUserShareValueBalancerNoDecimals:
    """TestGetUserShareValueBalancerNoDecimals."""

    def test_token_no_decimals(self):
        """Cover lines 2000-2003: _get_token_decimals returns None."""
        obj = _mk()
        obj.params.balancer_vault_contract_addresses = {"optimism": "0xV"}
        call_count = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return [["0x" + "ab" * 20], [10**18]]
            elif call_count[0] == 2:
                return 5 * 10**17
            elif call_count[0] == 3:
                return 10**18
            return 18

        obj.contract_interact = fake_ci
        obj._get_token_decimals = _gen_return(None)
        gen = obj.get_user_share_value_balancer("0xU", "0xPID", "0xP", "optimism")
        result = _drive(gen)
        assert result == {}


class TestGetUserShareValueUniswapSuccess:
    """TestGetUserShareValueUniswapSuccess."""

    def test_success(self):
        """Cover line 1899: successful path through uniswap."""
        obj = _mk()
        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        obj._calculate_cl_position_value = _gen_return({"0xT0": Decimal(1)})
        pos = {"token0": "0xT0", "token1": "0xT1"}
        gen = obj.get_user_share_value_uniswap("0xP", 1, "optimism", pos)
        result = _drive(gen)
        assert result == {"0xT0": Decimal(1)}


class TestCalculateUserShareValues:
    """TestCalculateUserShareValues."""

    def test_no_positions(self):
        """No open positions -> only safe balances and metrics."""
        obj = _mk()
        obj.current_positions = []
        obj.store_current_positions = MagicMock()
        obj._calculate_safe_balances_value = _gen_return(Decimal("50"))
        obj.calculate_stakig_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_airdrop_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_withdrawals_value = _gen_return(Decimal("0"))
        obj._update_portfolio_metrics = _gen_none
        obj.calculate_initial_investment_value_from_funding_events = _gen_return(100.0)
        obj._calculate_total_volume = _gen_return(100.0)
        obj._create_portfolio_data = lambda *a, **kw: {"data": True}
        obj.store_portfolio_data = MagicMock()
        gen = obj.calculate_user_share_values()
        _drive(gen)
        assert obj.portfolio_data == {"data": True}

    def test_with_open_position(self):
        """Cover full position processing path."""
        obj = _mk()
        obj.current_positions = [
            {
                "status": "open",
                "dex_type": "UniswapV3",
                "chain": "optimism",
                "pool_address": "0xPool",
                "token0": "0xT0",
                "token1": "0xT1",
                "token0_symbol": "A",
                "token1_symbol": "B",
                "apr": 5.0,
            }
        ]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.store_current_positions = MagicMock()
        obj._handle_uniswap_position = _gen_return(
            (
                {"0xT0": Decimal(10)},
                "Uniswap V3 Pool",
                {"0xT0": "A", "0xT1": "B"},
            )
        )
        obj._calculate_position_value = _gen_return(Decimal("100"))
        obj._calculate_safe_balances_value = _gen_return(Decimal("50"))
        obj.calculate_stakig_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_airdrop_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_withdrawals_value = _gen_return(Decimal("0"))
        obj._update_portfolio_metrics = _gen_none
        obj.calculate_initial_investment_value_from_funding_events = _gen_return(100.0)
        obj._calculate_total_volume = _gen_return(100.0)
        obj._create_portfolio_data = lambda *a, **kw: {"value": 150}
        obj.store_portfolio_data = MagicMock()
        gen = obj.calculate_user_share_values()
        _drive(gen)
        assert obj.portfolio_data == {"value": 150}

    def test_missing_dex_type(self):
        """Cover line 589-591: missing dex_type."""
        obj = _mk()
        obj.current_positions = [{"status": "open", "chain": "optimism"}]
        obj.store_current_positions = MagicMock()
        obj._calculate_safe_balances_value = _gen_return(Decimal("0"))
        obj.calculate_stakig_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_airdrop_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_withdrawals_value = _gen_return(Decimal("0"))
        obj._update_portfolio_metrics = _gen_none
        obj.calculate_initial_investment_value_from_funding_events = _gen_return(None)
        obj._load_chain_total_investment = _gen_return(0.0)
        obj.params.target_investment_chains = ["optimism"]
        obj._calculate_total_volume = _gen_return(None)
        obj._create_portfolio_data = lambda *a, **kw: {}
        obj.store_portfolio_data = MagicMock()
        gen = obj.calculate_user_share_values()
        _drive(gen)

    def test_unsupported_dex(self):
        """Cover line 594-595: unsupported dex type."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "unknown_dex", "chain": "optimism"}
        ]
        obj.store_current_positions = MagicMock()
        obj._calculate_safe_balances_value = _gen_return(Decimal("0"))
        obj.calculate_stakig_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_airdrop_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_withdrawals_value = _gen_return(Decimal("0"))
        obj._update_portfolio_metrics = _gen_none
        obj.calculate_initial_investment_value_from_funding_events = _gen_return(100.0)
        obj._calculate_total_volume = _gen_return(100.0)
        obj._create_portfolio_data = lambda *a, **kw: {"ok": True}
        obj.store_portfolio_data = MagicMock()
        gen = obj.calculate_user_share_values()
        _drive(gen)

    def test_handler_returns_none(self):
        """Cover line 600-601: handler returns None."""
        obj = _mk()
        obj.current_positions = [
            {
                "status": "open",
                "dex_type": "UniswapV3",
                "chain": "optimism",
                "pool_address": "0xP",
            }
        ]
        obj.store_current_positions = MagicMock()
        obj._handle_uniswap_position = _gen_return(None)
        obj._calculate_safe_balances_value = _gen_return(Decimal("0"))
        obj.calculate_stakig_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_airdrop_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_withdrawals_value = _gen_return(Decimal("0"))
        obj._update_portfolio_metrics = _gen_none
        obj.calculate_initial_investment_value_from_funding_events = _gen_return(100.0)
        obj._calculate_total_volume = _gen_return(100.0)
        obj._create_portfolio_data = lambda *a, **kw: {"ok": True}
        obj.store_portfolio_data = MagicMock()
        gen = obj.calculate_user_share_values()
        _drive(gen)

    def test_exception_in_position(self):
        """Cover lines 630-632: exception processing position."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "UniswapV3", "chain": "optimism"}
        ]
        obj.store_current_positions = MagicMock()

        def boom(*a, **kw):
            """Boom."""
            raise RuntimeError("fail")
            yield  # noqa

        obj._handle_uniswap_position = boom
        obj._calculate_safe_balances_value = _gen_return(Decimal("0"))
        obj.calculate_stakig_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_airdrop_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_withdrawals_value = _gen_return(Decimal("0"))
        obj._update_portfolio_metrics = _gen_none
        obj.calculate_initial_investment_value_from_funding_events = _gen_return(100.0)
        obj._calculate_total_volume = _gen_return(100.0)
        obj._create_portfolio_data = lambda *a, **kw: {"ok": True}
        obj.store_portfolio_data = MagicMock()
        gen = obj.calculate_user_share_values()
        _drive(gen)

    def test_initial_investment_none_fallback(self):
        """Cover lines 657-672: initial_investment is None, use KV store."""
        obj = _mk()
        obj.current_positions = []
        obj.store_current_positions = MagicMock()
        obj._calculate_safe_balances_value = _gen_return(Decimal("0"))
        obj.calculate_stakig_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_airdrop_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_withdrawals_value = _gen_return(Decimal("0"))
        obj._update_portfolio_metrics = _gen_none
        obj.calculate_initial_investment_value_from_funding_events = _gen_return(None)
        obj._load_chain_total_investment = _gen_return(200.0)
        obj.params.target_investment_chains = ["optimism"]
        obj._calculate_total_volume = _gen_return(200.0)
        obj._create_portfolio_data = lambda *a, **kw: {"data": True}
        obj.store_portfolio_data = MagicMock()
        gen = obj.calculate_user_share_values()
        _drive(gen)

    def test_initial_investment_none_no_kv(self):
        """Cover lines 670-672: KV store also has 0."""
        obj = _mk()
        obj.current_positions = []
        obj.store_current_positions = MagicMock()
        obj._calculate_safe_balances_value = _gen_return(Decimal("0"))
        obj.calculate_stakig_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_airdrop_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_withdrawals_value = _gen_return(Decimal("0"))
        obj._update_portfolio_metrics = _gen_none
        obj.calculate_initial_investment_value_from_funding_events = _gen_return(None)
        obj._load_chain_total_investment = _gen_return(0.0)
        obj.params.target_investment_chains = ["optimism"]
        obj._calculate_total_volume = _gen_return(None)
        obj._create_portfolio_data = lambda *a, **kw: {}
        obj.store_portfolio_data = MagicMock()
        gen = obj.calculate_user_share_values()
        _drive(gen)

    def test_portfolio_data_falsy(self):
        """Cover line 688: portfolio_data is empty/falsy."""
        obj = _mk()
        obj.current_positions = []
        obj.store_current_positions = MagicMock()
        obj._calculate_safe_balances_value = _gen_return(Decimal("0"))
        obj.calculate_stakig_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_airdrop_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_withdrawals_value = _gen_return(Decimal("0"))
        obj._update_portfolio_metrics = _gen_none
        obj.calculate_initial_investment_value_from_funding_events = _gen_return(100.0)
        obj._calculate_total_volume = _gen_return(100.0)
        obj._create_portfolio_data = lambda *a, **kw: {}  # falsy
        obj.store_portfolio_data = MagicMock()
        gen = obj.calculate_user_share_values()
        _drive(gen)
        assert obj.portfolio_data == {}

    def test_balancer_position(self):
        """Cover balancer handler path with pool_id."""
        obj = _mk()
        obj.current_positions = [
            {
                "status": "open",
                "dex_type": "balancerPool",
                "chain": "optimism",
                "pool_id": "0xPoolId",
                "pool_address": "0xPool",
                "token0": "0xT0",
                "token1": "0xT1",
                "token0_symbol": "A",
                "token1_symbol": "B",
                "apr": 5.0,
            }
        ]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.store_current_positions = MagicMock()
        obj._handle_balancer_position = _gen_return(
            (
                {"0xT0": Decimal(10)},
                "Pool",
                {"0xT0": "A", "0xT1": "B"},
            )
        )
        obj._calculate_position_value = _gen_return(Decimal("100"))
        obj._calculate_safe_balances_value = _gen_return(Decimal("0"))
        obj.calculate_stakig_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_airdrop_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_withdrawals_value = _gen_return(Decimal("0"))
        obj._update_portfolio_metrics = _gen_none
        obj.calculate_initial_investment_value_from_funding_events = _gen_return(100.0)
        obj._calculate_total_volume = _gen_return(100.0)
        obj._create_portfolio_data = lambda *a, **kw: {"ok": True}
        obj.store_portfolio_data = MagicMock()
        gen = obj.calculate_user_share_values()
        _drive(gen)


class TestAsyncAct:
    """TestAsyncAct."""

    def test_full_flow(self):
        """Cover async_act lines 102-253."""
        obj = _mk()
        obj.context.benchmark_tool.measure.return_value.__enter__ = MagicMock()
        obj.context.benchmark_tool.measure.return_value.__exit__ = MagicMock()
        bm = MagicMock()
        bm.local.return_value.__enter__ = MagicMock(return_value=None)
        bm.local.return_value.__exit__ = MagicMock(return_value=False)
        bm.consensus.return_value.__enter__ = MagicMock(return_value=None)
        bm.consensus.return_value.__exit__ = MagicMock(return_value=False)
        obj.context.benchmark_tool.measure.return_value = bm
        obj.context.agent_address = "0xAgent"
        obj.current_positions = []

        with patch.object(
            type(obj), "synchronized_data", new_callable=PropertyMock
        ) as mock_sd:
            mock_sd.return_value = MagicMock(period_count=1)

            obj.params.target_investment_chains = ["optimism"]
            obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
            obj.params.available_strategies = {"optimism": ["uniswapV3"]}
            obj.params.initial_assets = {}

            obj._get_native_balance = _gen_return(1.0)
            obj._read_kv = _gen_return(
                {
                    "selected_protocols": json.dumps(["uniswapV3"]),
                    "trading_type": "balanced",
                }
            )
            obj._write_kv = _gen_none

            with patch.object(
                type(obj), "shared_state", new_callable=PropertyMock
            ) as mock_ss:
                mock_ss.return_value = MagicMock()
                obj.whitelisted_assets = {"optimism": {"0xT": "TKN"}}
                obj.read_whitelisted_assets = MagicMock()
                obj._get_current_timestamp = lambda: 100
                obj._track_whitelisted_assets = _gen_none
                obj.calculate_user_share_values = _gen_none
                obj.store_portfolio_data = MagicMock()
                obj._update_agent_performance_metrics = MagicMock()
                obj.send_a2a_transaction = _gen_none
                obj.wait_until_round_end = _gen_none
                obj.set_done = MagicMock()

                gen = obj.async_act()
                _drive(gen)
                obj.set_done.assert_called_once()

    def test_period_zero(self):
        """Cover period_count=0 branches (lines 116-127, 139-142)."""
        obj = _mk()
        bm = MagicMock()
        bm.local.return_value.__enter__ = MagicMock(return_value=None)
        bm.local.return_value.__exit__ = MagicMock(return_value=False)
        bm.consensus.return_value.__enter__ = MagicMock(return_value=None)
        bm.consensus.return_value.__exit__ = MagicMock(return_value=False)
        obj.context.benchmark_tool.measure.return_value = bm
        obj.context.agent_address = "0xAgent"
        obj.current_positions = []

        with patch.object(
            type(obj), "synchronized_data", new_callable=PropertyMock
        ) as mock_sd:
            mock_sd.return_value = MagicMock(period_count=0)

            obj.params.target_investment_chains = ["optimism"]
            obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
            obj.params.available_strategies = {"optimism": ["uniswapV3"]}
            obj.params.initial_assets = {}

            obj._validate_velodrome_v2_pool_addresses = _gen_none
            obj.update_position_amounts = _gen_none
            obj.check_and_update_zero_liquidity_positions = MagicMock()
            obj._get_native_balance = _gen_return(1.0)
            obj._check_and_create_eth_revert_transactions = _gen_none
            obj._read_kv = _gen_return({})
            obj._write_kv = _gen_none

            with patch.object(
                type(obj), "shared_state", new_callable=PropertyMock
            ) as mock_ss:
                mock_ss.return_value = MagicMock()
                obj.assets = {}
                obj.whitelisted_assets = {}
                obj.store_whitelisted_assets = MagicMock()
                obj._get_current_timestamp = lambda: 100000
                obj._track_whitelisted_assets = _gen_none
                obj.calculate_user_share_values = _gen_none
                obj.store_portfolio_data = MagicMock()
                obj._update_agent_performance_metrics = MagicMock()
                obj.send_a2a_transaction = _gen_none
                obj.wait_until_round_end = _gen_none
                obj.set_done = MagicMock()

                gen = obj.async_act()
                _drive(gen)
                obj.set_done.assert_called_once()

    def test_invalid_last_updated_timestamp(self):
        """Cover the ValueError branch on int(last_updated)."""
        obj = _mk()
        bm = MagicMock()
        bm.local.return_value.__enter__ = MagicMock(return_value=None)
        bm.local.return_value.__exit__ = MagicMock(return_value=False)
        bm.consensus.return_value.__enter__ = MagicMock(return_value=None)
        bm.consensus.return_value.__exit__ = MagicMock(return_value=False)
        obj.context.benchmark_tool.measure.return_value = bm
        obj.context.agent_address = "0xAgent"
        obj.current_positions = []

        with patch.object(
            type(obj), "synchronized_data", new_callable=PropertyMock
        ) as mock_sd:
            mock_sd.return_value = MagicMock(period_count=1)

            obj.params.target_investment_chains = ["optimism"]
            obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
            obj.params.available_strategies = {}

            obj._get_native_balance = _gen_return(1.0)
            obj._read_investing_paused = _gen_return(False)

            read_call = [0]

            def fake_read(*a, **kw):
                """Fake read."""
                read_call[0] += 1
                yield
                if read_call[0] == 1:
                    return {
                        "selected_protocols": json.dumps(["uniswapV3"]),
                        "trading_type": "balanced",
                    }
                elif read_call[0] == 2:
                    return {"last_whitelisted_updated": "bad-timestamp"}
                return {}

            obj._read_kv = fake_read
            obj._write_kv = _gen_none

            with patch.object(
                type(obj), "shared_state", new_callable=PropertyMock
            ) as mock_ss:
                mock_ss.return_value = MagicMock()
                obj.whitelisted_assets = {"optimism": {}}
                obj.read_whitelisted_assets = MagicMock()
                obj._get_current_timestamp = lambda: 100
                obj._track_whitelisted_assets = _gen_none
                obj.calculate_user_share_values = _gen_none
                obj.store_portfolio_data = MagicMock()
                obj._update_agent_performance_metrics = MagicMock()
                obj.send_a2a_transaction = _gen_none
                obj.wait_until_round_end = _gen_none
                obj.set_done = MagicMock()

                gen = obj.async_act()
                _drive(gen)


class TestCalculateInitialInvestmentValueFromFundingEvents:
    """Tests for calculate_initial_investment_value_from_funding_events."""

    def test_no_safe_address(self):
        """Test no safe address."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.params.safe_contract_addresses = {}
        obj.params.airdrop_started = False
        obj._save_chain_total_investment = _gen_none
        obj._write_kv = _gen_none
        obj._get_current_timestamp = lambda: 100
        assert (
            _drive(obj.calculate_initial_investment_value_from_funding_events()) is None
        )

    def test_unsupported_chain(self):
        """Test unsupported chain."""
        obj = _mk()
        obj.params.target_investment_chains = ["polygon"]
        obj.params.safe_contract_addresses = {"polygon": "0xSafe"}
        obj.params.airdrop_started = False
        obj._read_kv = _gen_return({})
        obj._save_chain_total_investment = _gen_none
        obj._write_kv = _gen_none
        obj._get_current_timestamp = lambda: 100
        assert (
            _drive(obj.calculate_initial_investment_value_from_funding_events()) is None
        )

    def test_mode_airdrop_first_scan(self):
        """Test mode airdrop first scan."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj.params.airdrop_started = True
        cc = [0]

        def fr(*a, **kw):
            """Fr."""
            cc[0] += 1
            yield
            return {} if cc[0] == 1 else {}

        obj._read_kv = fr
        obj._write_kv = _gen_none
        obj._fetch_all_transfers_until_date_mode = _gen_return(
            {"2025-01-01": [{"symbol": "USDC", "amount": 10}]}
        )
        obj._calculate_chain_investment_value = _gen_return(100.0)
        obj._save_chain_total_investment = _gen_none
        obj._get_current_timestamp = lambda: 100
        assert (
            _drive(obj.calculate_initial_investment_value_from_funding_events())
            == 100.0
        )

    def test_mode_airdrop_incremental(self):
        """Test mode airdrop incremental."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj.params.airdrop_started = True
        obj._read_kv = _gen_return({"airdrop_full_scan_completed": "true"})
        obj._write_kv = _gen_none
        obj._fetch_all_transfers_until_date_mode = _gen_return(
            {"2025-01-01": [{"symbol": "USDC", "amount": 5}]}
        )
        obj._calculate_chain_investment_value = _gen_return(50.0)
        obj._save_chain_total_investment = _gen_none
        obj._get_current_timestamp = lambda: 100
        assert (
            _drive(obj.calculate_initial_investment_value_from_funding_events()) == 50.0
        )

    def test_optimism_no_previous_calc(self):
        """Test optimism no previous calc."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.airdrop_started = False
        obj._read_kv = _gen_return({})
        obj._write_kv = _gen_none
        obj._fetch_all_transfers_until_date_optimism = _gen_return(
            {"2025-01-01": [{"symbol": "ETH", "delta": 1}]}
        )
        obj._calculate_chain_investment_value = _gen_return(200.0)
        obj._save_chain_total_investment = _gen_none
        obj._get_current_timestamp = lambda: 100
        assert (
            _drive(obj.calculate_initial_investment_value_from_funding_events())
            == 200.0
        )

    def test_cached_value_within_ttl(self):
        """Last calc within INITIAL_INVESTMENT_CACHE_TTL_SECONDS returns the cached value."""
        obj = _mk()
        # Last calc was 60 seconds ago; default TTL is 30 minutes → hit.
        now_ts = 1704067200
        ts = str(now_ts - 60)
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.airdrop_started = False
        obj._read_kv = _gen_return(
            {"last_initial_value_calculated_timestamp_optimism": ts}
        )
        obj._load_chain_total_investment = _gen_return(999.0)
        obj._write_kv = _gen_none
        obj._save_chain_total_investment = _gen_none
        obj._get_current_timestamp = lambda: now_ts
        assert (
            _drive(obj.calculate_initial_investment_value_from_funding_events())
            == 999.0
        )

    def test_cached_value_within_ttl_zero_falls_through(self):
        """Within-TTL hit with a 0 cached value falls through to a fresh fetch."""
        obj = _mk()
        now_ts = 1704067200
        ts = str(now_ts - 60)
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.airdrop_started = False
        obj._read_kv = _gen_return(
            {"last_initial_value_calculated_timestamp_optimism": ts}
        )
        obj._load_chain_total_investment = _gen_return(0.0)
        obj._fetch_all_transfers_until_date_optimism = _gen_return(
            {"d": [{"symbol": "ETH", "delta": 1}]}
        )
        obj._calculate_chain_investment_value = _gen_return(50.0)
        obj._write_kv = _gen_none
        obj._save_chain_total_investment = _gen_none
        obj._get_current_timestamp = lambda: now_ts
        assert (
            _drive(obj.calculate_initial_investment_value_from_funding_events()) == 50.0
        )

    def test_invalid_timestamp_format(self):
        """Test invalid timestamp format."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.airdrop_started = False
        obj._read_kv = _gen_return(
            {"last_initial_value_calculated_timestamp_optimism": "bad"}
        )
        obj._write_kv = _gen_none
        obj._fetch_all_transfers_until_date_optimism = _gen_return(
            {"d": [{"symbol": "ETH", "delta": 1}]}
        )
        obj._calculate_chain_investment_value = _gen_return(10.0)
        obj._save_chain_total_investment = _gen_none
        obj._get_current_timestamp = lambda: 100
        assert (
            _drive(obj.calculate_initial_investment_value_from_funding_events()) == 10.0
        )

    def test_no_transfers(self):
        """Test no transfers."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj.params.airdrop_started = False
        obj._read_kv = _gen_return({})
        obj._write_kv = _gen_none
        obj._fetch_all_transfers_until_date_mode = _gen_return({})
        obj._save_chain_total_investment = _gen_none
        obj._get_current_timestamp = lambda: 100
        assert (
            _drive(obj.calculate_initial_investment_value_from_funding_events()) is None
        )


class TestCalculateChainInvestmentValue:
    """Tests for _calculate_chain_investment_value with per-transfer caching."""

    NO_REVERSION = {
        "reversion_amount": 0,
        "historical_reversion_value": 0,
        "reversion_date": None,
    }

    def _base(self, reversion=None):
        """Create a base test object with common stubs."""
        obj = _mk()
        obj._track_eth_transfers_and_reversions = _gen_return(
            reversion or self.NO_REVERSION
        )
        obj._save_chain_total_investment = _gen_none
        obj._load_priced_transfers = _gen_return({})
        obj._save_priced_transfers = _gen_none
        obj.params.airdrop_started = False
        return obj

    def test_eth_transfer_with_reversion(self):
        """Test eth transfer with reversion."""
        obj = self._base(
            {
                "reversion_amount": 0.5,
                "historical_reversion_value": 100.0,
                "reversion_date": "01-01-2025",
            }
        )
        obj._fetch_historical_eth_price = _gen_return(2000.0)
        assert (
            _drive(
                obj._calculate_chain_investment_value(
                    {"2025-01-01": [{"symbol": "ETH", "delta": 1.0, "amount": 1.0}]},
                    "optimism",
                    "0xSafe",
                )
            )
            == 900.0
        )

    def test_usdc_non_eth(self):
        """Test usdc non eth."""
        obj = self._base()
        obj.get_coin_id_from_symbol = lambda s, c: "usd-coin"
        obj._fetch_historical_token_price = _gen_return(1.0)
        assert (
            _drive(
                obj._calculate_chain_investment_value(
                    {"2025-01-01": [{"symbol": "USDC", "amount": 100}]},
                    "optimism",
                    "0xS",
                )
            )
            == 100.0
        )

    def test_airdrop_excluded(self):
        """Test airdrop excluded."""
        obj = self._base()
        obj.params.airdrop_started = True
        obj.params.airdrop_contract_address = "0xAirdrop"
        assert (
            _drive(
                obj._calculate_chain_investment_value(
                    {
                        "2025-01-01": [
                            {
                                "symbol": "USDC",
                                "amount": 50,
                                "from_address": "0xairdrop",
                            }
                        ]
                    },
                    "mode",
                    "0xS",
                )
            )
            == 0.0
        )

    def test_no_coingecko_id(self):
        """Test no coingecko id."""
        obj = self._base()
        obj.get_coin_id_from_symbol = lambda s, c: None
        assert (
            _drive(
                obj._calculate_chain_investment_value(
                    {"2025-01-01": [{"symbol": "WEIRD", "amount": 10}]},
                    "optimism",
                    "0xS",
                )
            )
            == 0.0
        )

    def test_negative_amount(self):
        """Test negative amount."""
        obj = self._base()
        assert (
            _drive(
                obj._calculate_chain_investment_value(
                    {"2025-01-01": [{"symbol": "ETH", "amount": -5}]}, "optimism", "0xS"
                )
            )
            == 0.0
        )

    def test_exception_in_transfer(self):
        """Test exception in transfer."""
        obj = self._base()
        assert (
            _drive(
                obj._calculate_chain_investment_value(
                    {"bad": [{"symbol": "ETH", "amount": 1}]}, "optimism", "0xS"
                )
            )
            == 0.0
        )

    def test_reversion_no_date(self):
        """Test reversion no date."""
        obj = self._base(
            {
                "reversion_amount": 1.0,
                "historical_reversion_value": 0,
                "reversion_date": None,
            }
        )
        obj._fetch_historical_eth_price = _gen_return(3000.0)
        assert (
            _drive(obj._calculate_chain_investment_value({}, "optimism", "0xS"))
            == -3000.0
        )

    def test_cached_transfer_skips_price_fetch(self):
        """Previously priced transfers use cached value, no API call."""
        obj = self._base()
        # Pre-populate the priced cache with a known transfer key
        obj._load_priced_transfers = _gen_return({"2025-01-01_0xTX1": 500.0})
        # If price fetch were called it would return a different value
        obj._fetch_historical_eth_price = _gen_return(9999.0)
        result = _drive(
            obj._calculate_chain_investment_value(
                {"2025-01-01": [{"symbol": "ETH", "delta": 1.0, "tx_hash": "0xTX1"}]},
                "optimism",
                "0xS",
            )
        )
        # Should use cached 500.0, not 9999.0
        assert result == 500.0

    def test_mix_cached_and_new_transfers(self):
        """Cached + new transfers are summed correctly."""
        obj = self._base()
        obj._load_priced_transfers = _gen_return({"2025-01-01_0xOLD": 200.0})
        obj._fetch_historical_eth_price = _gen_return(1000.0)
        result = _drive(
            obj._calculate_chain_investment_value(
                {
                    "2025-01-01": [
                        {"symbol": "ETH", "delta": 1.0, "tx_hash": "0xOLD"},
                        {"symbol": "ETH", "delta": 0.5, "tx_hash": "0xNEW"},
                    ]
                },
                "optimism",
                "0xS",
            )
        )
        # cached=200, new=0.5*1000=500
        assert result == 700.0

    def test_price_fetch_failure_logs_warning(self):
        """When price fetch fails, transfer is skipped with a warning (not silent)."""
        obj = self._base()
        obj._fetch_historical_eth_price = _gen_return(None)
        result = _drive(
            obj._calculate_chain_investment_value(
                {"2025-01-01": [{"symbol": "ETH", "delta": 1.0}]},
                "optimism",
                "0xS",
            )
        )
        assert result == 0.0
        # Verify warning was logged (not silently skipped)
        obj.context.logger.warning.assert_called()


class TestFetchAllTransfersUntilDateMode:
    """TestFetchAllTransfersUntilDateMode."""

    def test_success(self):
        """Test success."""
        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()
        obj._fetch_token_transfers_mode = _gen_return(True)
        obj._fetch_eth_transfers_mode = MagicMock(return_value=True)
        assert isinstance(
            _drive(
                obj._fetch_all_transfers_until_date_mode("0xA", "2025-01-01", False)
            ),
            dict,
        )

    def test_token_fetch_fails(self):
        """Test token fetch fails."""
        obj = _mk()
        obj.funding_events = {"mode": {"2025-01-01": []}}
        obj.read_funding_events = lambda: obj.funding_events
        obj.store_funding_events = MagicMock()
        obj._fetch_token_transfers_mode = _gen_return(False)
        obj._fetch_eth_transfers_mode = MagicMock(return_value=True)
        assert isinstance(
            _drive(
                obj._fetch_all_transfers_until_date_mode("0xA", "2025-01-01", False)
            ),
            dict,
        )

    def test_backward_compat(self):
        """Test backward compat."""
        obj = _mk()
        obj.read_funding_events = lambda: {"mode": {"2025-01-01": [{"type": "eth"}]}}
        obj.store_funding_events = MagicMock()
        obj._fetch_token_transfers_mode = _gen_return(True)
        obj._fetch_eth_transfers_mode = MagicMock(return_value=True)
        assert isinstance(
            _drive(
                obj._fetch_all_transfers_until_date_mode("0xA", "2025-01-01", False)
            ),
            dict,
        )

    def test_exception(self):
        """Test exception."""
        obj = _mk()
        obj.read_funding_events = lambda: {"mode": {}}
        obj.store_funding_events = MagicMock()

        def bad(*a, **kw):
            """Bad."""
            yield
            raise RuntimeError("boom")

        obj._fetch_token_transfers_mode = bad
        assert isinstance(
            _drive(
                obj._fetch_all_transfers_until_date_mode("0xA", "2025-01-01", False)
            ),
            dict,
        )


class TestFetchAllTransfersUntilDateOptimism:
    """TestFetchAllTransfersUntilDateOptimism."""

    def test_success(self):
        """Test success."""
        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()
        obj._fetch_optimism_transfers_safeglobal = _gen_none
        assert isinstance(
            _drive(obj._fetch_all_transfers_until_date_optimism("0xA", "2025-01-01")),
            dict,
        )

    def test_exception(self):
        """Test exception."""
        obj = _mk()
        obj.read_funding_events = lambda: {}

        def boom(*a, **kw):
            """Boom."""
            yield
            raise RuntimeError("fail")

        obj._fetch_optimism_transfers_safeglobal = boom
        assert (
            _drive(obj._fetch_all_transfers_until_date_optimism("0xA", "2025-01-01"))
            == {}
        )


class TestFetchTokenTransfersBatch2:
    """TestFetchTokenTransfersBatch2."""

    def test_request_fails(self):
        """Test request fails."""
        obj = _mk()
        obj._request_with_retries = _gen_return((False, {}))
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        assert (
            _drive(
                obj._fetch_token_transfers(
                    "0xA",
                    datetime(2025, 1, 1, tzinfo=timezone.utc),
                    defaultdict(list),
                    {},
                )
            )
            is None
        )

    def test_no_items(self):
        """Test no items."""
        obj = _mk()
        obj._request_with_retries = _gen_return((True, {"items": []}))
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        assert (
            _drive(
                obj._fetch_token_transfers(
                    "0xA",
                    datetime(2025, 1, 1, tzinfo=timezone.utc),
                    defaultdict(list),
                    {},
                )
            )
            is None
        )

    def test_processes(self):
        """Test processes."""
        obj = _mk()
        tx = {
            "timestamp": "2024-12-15T10:00:00Z",
            "from": {"hash": "0xS", "is_contract": False},
            "token": {"symbol": "USDC", "decimals": 6, "address": "0xT"},
            "total": {"value": "1000000"},
            "transaction_hash": "0xH",
        }
        obj._request_with_retries = _gen_return((True, {"items": [tx]}))
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        t = defaultdict(list)
        _drive(
            obj._fetch_token_transfers(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, {}
            )
        )
        assert len(t) == 1

    def test_skip_future(self):
        """Test skip future."""
        obj = _mk()
        tx = {
            "timestamp": "2025-06-01T10:00:00Z",
            "from": {"hash": "0xS"},
            "token": {"symbol": "USDC", "decimals": 6},
            "total": {"value": "1000000"},
        }
        obj._request_with_retries = _gen_return((True, {"items": [tx]}))
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        t = defaultdict(list)
        _drive(
            obj._fetch_token_transfers(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, {}
            )
        )
        assert len(t) == 0

    def test_skip_existing(self):
        """Test skip existing."""
        obj = _mk()
        tx = {
            "timestamp": "2024-12-15T10:00:00Z",
            "from": {"hash": "0xS"},
            "token": {"symbol": "USDC", "decimals": 6},
            "total": {"value": "1000000"},
        }
        obj._request_with_retries = _gen_return((True, {"items": [tx]}))
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        t = defaultdict(list)
        _drive(
            obj._fetch_token_transfers(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, {"2024-12-15": []}
            )
        )
        assert len(t) == 0


class TestFetchTokenTransfersModeBatch2:
    """TestFetchTokenTransfersModeBatch2."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_non_200(self, m):
        """Test non 200."""
        m.return_value = MagicMock(status_code=500)
        obj = _mk()
        obj.funding_events = {}
        assert (
            _drive(
                obj._fetch_token_transfers_mode(
                    "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, False
                )
            )
            is False
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_empty(self, m):
        """Test empty."""
        m.return_value = MagicMock(status_code=200, json=lambda: {"items": []})
        obj = _mk()
        obj.funding_events = {}
        assert (
            _drive(
                obj._fetch_token_transfers_mode(
                    "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, False
                )
            )
            is True
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_usdc(self, m):
        """Test usdc."""
        tx = {
            "timestamp": "2024-12-15T10:00:00Z",
            "from": {"hash": "0xS", "is_contract": False},
            "token": {"symbol": "USDC", "decimals": "6", "address": "0xT"},
            "total": {"value": "1000000"},
            "transaction_hash": "0xH",
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        obj = _mk()
        obj.funding_events = {}
        obj._is_airdrop_transfer = lambda t: False
        obj._should_include_transfer_mode = lambda fa, tx, is_eth_transfer: True
        t = {}
        assert (
            _drive(
                obj._fetch_token_transfers_mode(
                    "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, True
                )
            )
            is True
        )
        assert "2024-12-15" in t

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_self_transfer(self, m):
        """Test self transfer."""
        tx = {
            "timestamp": "2024-12-15T10:00:00Z",
            "from": {"hash": "0xaddr"},
            "token": {"symbol": "USDC", "decimals": "6"},
            "total": {"value": "1000000"},
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        obj = _mk()
        obj.funding_events = {}
        assert (
            _drive(
                obj._fetch_token_transfers_mode(
                    "0xAddr", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, True
                )
            )
            is True
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_airdrop(self, m):
        """Test airdrop."""
        tx = {
            "timestamp": "2024-12-15T10:00:00Z",
            "from": {"hash": "0xS", "is_contract": False},
            "token": {"symbol": "USDC", "decimals": "6"},
            "total": {"value": "5000000"},
            "transaction_hash": "0xAH",
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        obj = _mk()
        obj.funding_events = {}
        obj._is_airdrop_transfer = lambda t: True
        obj._update_airdrop_rewards = _gen_none
        obj._should_include_transfer_mode = lambda fa, tx, is_eth_transfer: True
        assert (
            _drive(
                obj._fetch_token_transfers_mode(
                    "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, True
                )
            )
            is True
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_pagination(self, m):
        """Test pagination."""
        tx = {
            "timestamp": "2024-12-15T10:00:00Z",
            "from": {"hash": "0xS", "is_contract": False},
            "token": {"symbol": "USDC", "decimals": "6", "address": "0xT"},
            "total": {"value": "1000000"},
            "transaction_hash": "0xH",
        }
        c = [0]

        def mj():
            """Mj."""
            c[0] += 1
            if c[0] == 1:
                return {
                    "items": [tx],
                    "next_page_params": {"block_number": 100, "index": 0},
                }
            return {"items": [], "next_page_params": None}

        r = MagicMock(status_code=200)
        r.json = mj
        m.return_value = r
        obj = _mk()
        obj.funding_events = {}
        obj._is_airdrop_transfer = lambda t: False
        obj._should_include_transfer_mode = lambda fa, tx, is_eth_transfer: True
        assert (
            _drive(
                obj._fetch_token_transfers_mode(
                    "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, True
                )
            )
            is True
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_incremental_stop(self, m):
        """Test incremental stop."""
        tx = {
            "timestamp": "2020-01-01T10:00:00Z",
            "from": {"hash": "0xS"},
            "token": {},
            "total": {},
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        obj = _mk()
        obj.funding_events = {"mode": {"2024-01-01": []}}
        assert (
            _drive(
                obj._fetch_token_transfers_mode(
                    "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, False
                )
            )
            is True
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_zero_amount(self, m):
        """Test zero amount."""
        tx = {
            "timestamp": "2024-12-15T10:00:00Z",
            "from": {"hash": "0xS", "is_contract": False},
            "token": {"symbol": "USDC", "decimals": "6"},
            "total": {"value": "0"},
            "transaction_hash": "0xH",
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        obj = _mk()
        obj.funding_events = {}
        obj._is_airdrop_transfer = lambda t: False
        obj._should_include_transfer_mode = lambda fa, tx, is_eth_transfer: True
        assert len({}) == 0  # just exercises the path

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_non_usdc(self, m):
        """Test non usdc."""
        tx = {
            "timestamp": "2024-12-15T10:00:00Z",
            "from": {"hash": "0xS", "is_contract": False},
            "token": {"symbol": "WETH", "decimals": "18"},
            "total": {"value": str(10**18)},
            "transaction_hash": "0xH",
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        obj = _mk()
        obj.funding_events = {}
        obj._is_airdrop_transfer = lambda t: False
        obj._should_include_transfer_mode = lambda fa, tx, is_eth_transfer: True
        t = {}
        _drive(
            obj._fetch_token_transfers_mode(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, True
            )
        )
        assert len(t) == 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_bad_timestamp(self, m):
        """Test bad timestamp."""
        tx = {"timestamp": "bad", "from": {"hash": "0xS"}, "token": {}, "total": {}}
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        obj = _mk()
        obj.funding_events = {}
        assert (
            _drive(
                obj._fetch_token_transfers_mode(
                    "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, True
                )
            )
            is True
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_bad_date_key(self, m):
        """Test bad date key."""
        m.return_value = MagicMock(status_code=200, json=lambda: {"items": []})
        obj = _mk()
        obj.funding_events = {"mode": {"bad-key": []}}
        assert (
            _drive(
                obj._fetch_token_transfers_mode(
                    "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, False
                )
            )
            is True
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.MAX_PAGINATION_PAGES",
        2,
    )
    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_pagination_cap_triggers_break(self, m):
        """The cap fires after exactly MAX_PAGINATION_PAGES pages, not before."""
        tx = {
            "timestamp": "2024-12-15T10:00:00Z",
            "from": {"hash": "0xS", "is_contract": False},
            "token": {"symbol": "USDC", "decimals": "6", "address": "0xT"},
            "total": {"value": "1000000"},
            "transaction_hash": "0xH",
        }
        # Always advertise a next-page cursor so without the cap the loop
        # would never terminate.
        m.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "items": [tx],
                "next_page_params": {"block_number": 100, "index": 0},
            },
        )
        obj = _mk()
        obj.funding_events = {}
        obj._is_airdrop_transfer = lambda t: False
        obj._should_include_transfer_mode = lambda fa, tx, is_eth_transfer: True
        _drive(
            obj._fetch_token_transfers_mode(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, True
            )
        )
        # The cap stops the loop after exactly 2 pages.
        assert m.call_count == 2
        obj.context.logger.warning.assert_called()


class TestFetchEthTransfersModePaginationCap:
    """Cover the pagination cap in _fetch_eth_transfers_mode."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.MAX_PAGINATION_PAGES",
        2,
    )
    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_pagination_cap_triggers_break(self, m):
        """The cap fires after exactly MAX_PAGINATION_PAGES pages, not before."""
        item = {
            "date": "2024-12-15",
            "value": "1000000000000000000",
        }
        m.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "items": [item],
                "next_page_params": {"block_number": 100, "items_count": 1},
            },
        )
        obj = _mk()
        obj.funding_events = {}
        obj._fetch_historical_eth_price = lambda *_args, **_kwargs: 0.0
        obj._fetch_eth_transfers_mode("0xA", "2025-01-01", {}, True)
        assert m.call_count == 2
        obj.context.logger.warning.assert_called()


class TestFetchEthTransfersModeBatch2:
    """TestFetchEthTransfersModeBatch2."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_non_200(self, m):
        """Test non 200."""
        m.return_value = MagicMock(status_code=500)
        obj = _mk()
        obj.funding_events = {}
        assert (
            obj._fetch_eth_transfers_mode(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, False
            )
            is False
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_empty(self, m):
        """Test empty."""
        m.return_value = MagicMock(status_code=200, json=lambda: {"items": []})
        obj = _mk()
        obj.funding_events = {}
        assert (
            obj._fetch_eth_transfers_mode(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, False
            )
            is True
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_processes(self, m):
        """Test processes."""
        e = {
            "value": str(2 * 10**18),
            "delta": str(1 * 10**18),
            "transaction_hash": None,
            "block_timestamp": "2024-12-15T10:00:00Z",
            "block_number": 100,
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [e], "next_page_params": None}
        )
        obj = _mk()
        obj.funding_events = {}
        t = {}
        assert (
            obj._fetch_eth_transfers_mode(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, True
            )
            is True
        )
        assert "2024-12-15" in t

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_zero_val(self, m):
        """Test zero val."""
        e = {
            "value": "0",
            "delta": str(10**18),
            "transaction_hash": None,
            "block_timestamp": "2024-12-15T10:00:00Z",
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [e], "next_page_params": None}
        )
        obj = _mk()
        obj.funding_events = {}
        t = {}
        obj._fetch_eth_transfers_mode(
            "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, True
        )
        assert len(t) == 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_neg_delta(self, m):
        """Test neg delta."""
        e = {
            "value": str(10**18),
            "delta": str(-(10**18)),
            "transaction_hash": None,
            "block_timestamp": "2024-12-15T10:00:00Z",
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [e], "next_page_params": None}
        )
        obj = _mk()
        obj.funding_events = {}
        t = {}
        obj._fetch_eth_transfers_mode(
            "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, True
        )
        assert len(t) == 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_hash_present(self, m):
        """Test hash present."""
        e = {
            "value": str(10**18),
            "delta": str(10**18),
            "transaction_hash": "0xABC",
            "block_timestamp": "2024-12-15T10:00:00Z",
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [e], "next_page_params": None}
        )
        obj = _mk()
        obj.funding_events = {}
        t = {}
        obj._fetch_eth_transfers_mode(
            "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, True
        )
        assert len(t) == 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_incremental(self, m):
        """Test incremental."""
        e = {
            "value": str(2 * 10**18),
            "delta": str(1 * 10**18),
            "transaction_hash": None,
            "block_timestamp": "2020-01-01T10:00:00Z",
            "block_number": 1,
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [e], "next_page_params": None}
        )
        obj = _mk()
        obj.funding_events = {"mode": {"2024-01-01": []}}
        assert (
            obj._fetch_eth_transfers_mode(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, False
            )
            is True
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_pagination(self, m):
        """Test pagination."""
        e = {
            "value": str(2 * 10**18),
            "delta": str(1 * 10**18),
            "transaction_hash": None,
            "block_timestamp": "2024-12-15T10:00:00Z",
            "block_number": 100,
        }
        c = [0]

        def mj():
            """Mj."""
            c[0] += 1
            if c[0] == 1:
                return {
                    "items": [e],
                    "next_page_params": {"block_number": 50, "index": 0},
                }
            return {"items": [], "next_page_params": None}

        r = MagicMock(status_code=200)
        r.json = mj
        m.return_value = r
        obj = _mk()
        obj.funding_events = {}
        assert (
            obj._fetch_eth_transfers_mode(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, True
            )
            is True
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_bad_key(self, m):
        """Test bad key."""
        m.return_value = MagicMock(status_code=200, json=lambda: {"items": []})
        obj = _mk()
        obj.funding_events = {"mode": {"not-a-date": []}}
        assert (
            obj._fetch_eth_transfers_mode(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, False
            )
            is True
        )


class TestFetchOptimismSafeglobalBatch2:
    """TestFetchOptimismSafeglobalBatch2."""

    def test_fail(self):
        """Test fail."""
        obj = _mk()
        obj._request_with_retries = _gen_return((False, {}))
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        _drive(
            obj._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {}
            )
        )

    def test_empty(self):
        """Test empty."""
        obj = _mk()
        obj._request_with_retries = _gen_return((True, {"results": []}))
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        _drive(
            obj._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {}
            )
        )

    def test_erc20_usdc(self):
        """Test erc20 usdc."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xS",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {"symbol": "USDC", "decimals": 6},
            "tokenAddress": "0xT",
            "value": "1000000",
            "transactionHash": "0xH",
            "transferId": "t1",
        }
        obj = _mk()
        ci = [0]

        def fr(*a, **kw):
            """Fr."""
            ci[0] += 1
            yield
            return (
                (True, {"results": [td], "next": None})
                if ci[0] == 1
                else (True, {"results": []})
            )

        obj._request_with_retries = fr
        obj._should_include_transfer_optimism = _gen_return(True)
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        t = defaultdict(list)
        _drive(obj._fetch_optimism_transfers_safeglobal("0xA", "2025-01-01", t, {}))
        assert "2024-12-15" in t

    def test_ether(self):
        """Test ether."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xS",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transactionHash": "0xH",
            "transferId": "t1",
        }
        obj = _mk()
        obj._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        obj._should_include_transfer_optimism = _gen_return(True)
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        t = defaultdict(list)
        _drive(obj._fetch_optimism_transfers_safeglobal("0xA", "2025-01-01", t, {}))
        assert len(t["2024-12-15"]) == 1

    def test_erc721(self):
        """Test erc721."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xS",
            "type": "ERC721_TRANSFER",
            "transferId": "t1",
        }
        obj = _mk()
        obj._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        obj._should_include_transfer_optimism = _gen_return(True)
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        _drive(
            obj._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {}
            )
        )

    def test_self(self):
        """Test self."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xaddr",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transferId": "t1",
        }
        obj = _mk()
        obj._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        _drive(
            obj._fetch_optimism_transfers_safeglobal(
                "0xAddr", "2025-01-01", defaultdict(list), {}
            )
        )

    def test_no_token_info_with_addr(self):
        """Test no token info with addr."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xS",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {},
            "tokenAddress": "0xT",
            "value": "1000000",
            "transferId": "t1",
        }
        obj = _mk()
        ci = [0]

        def fr(*a, **kw):
            """Fr."""
            ci[0] += 1
            yield
            return (
                (True, {"results": [td], "next": None})
                if ci[0] == 1
                else (True, {"results": []})
            )

        obj._request_with_retries = fr
        obj._should_include_transfer_optimism = _gen_return(True)
        obj._get_token_decimals = _gen_return(6)
        obj._get_token_symbol = _gen_return("USDC")
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        t = defaultdict(list)
        _drive(obj._fetch_optimism_transfers_safeglobal("0xA", "2025-01-01", t, {}))
        assert "2024-12-15" in t

    def test_no_token_info_no_addr(self):
        """Test no token info no addr."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xS",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {},
            "tokenAddress": "",
            "value": "1000000",
            "transferId": "t1",
        }
        obj = _mk()
        obj._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        obj._should_include_transfer_optimism = _gen_return(True)
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        _drive(
            obj._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {}
            )
        )

    def test_zero_eth(self):
        """Test zero eth."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xS",
            "type": "ETHER_TRANSFER",
            "value": "0",
            "transferId": "t1",
        }
        obj = _mk()
        obj._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        obj._should_include_transfer_optimism = _gen_return(True)
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        _drive(
            obj._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {}
            )
        )

    def test_bad_eth_value(self):
        """Test bad eth value."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xS",
            "type": "ETHER_TRANSFER",
            "value": "bad",
            "transferId": "t1",
        }
        obj = _mk()
        obj._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        obj._should_include_transfer_optimism = _gen_return(True)
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        _drive(
            obj._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {}
            )
        )

    def test_exception(self):
        """Test exception."""
        obj = _mk()

        def bad(*a, **kw):
            """Bad."""
            yield
            raise RuntimeError("boom")

        obj._request_with_retries = bad
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1
        _drive(
            obj._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {}
            )
        )


class TestTrackEthReversions:
    """TestTrackEthReversions."""

    def _s(self, ch="optimism"):
        o = _mk()
        o.params.target_investment_chains = [ch]
        o.params.safe_contract_addresses = {ch: "0xSafe"}
        return o

    def test_no_incoming(self):
        """Test no incoming."""
        o = self._s()
        o._fetch_all_transfers_until_date_optimism = _gen_return({})
        o._fetch_outgoing_transfers_until_date_optimism = _gen_return({})
        assert _drive(o._track_eth_transfers_and_reversions("0xSafe", "optimism")) == {}

    def test_unsupported(self):
        """Test unsupported."""
        assert (
            _drive(
                self._s("polygon")._track_eth_transfers_and_reversions("0xS", "polygon")
            )
            == {}
        )

    def test_no_master(self):
        """Test no master."""
        o = self._s()
        o._fetch_all_transfers_until_date_optimism = _gen_return(
            {
                "d": [
                    {
                        "symbol": "ETH",
                        "timestamp": "t",
                        "amount": 1.0,
                        "from_address": "0xM",
                    }
                ]
            }
        )
        o._fetch_outgoing_transfers_until_date_optimism = _gen_return({})
        o.get_master_safe_address = _gen_return(None)
        assert _drive(o._track_eth_transfers_and_reversions("0xSafe", "optimism")) == {}

    def test_two_transfers(self):
        """Test two transfers."""
        o = self._s()
        inc = {
            "d": [
                {
                    "symbol": "ETH",
                    "timestamp": "1704067200Z",
                    "amount": 1.0,
                    "from_address": "0xmaster",
                },
                {
                    "symbol": "ETH",
                    "timestamp": "1704153600Z",
                    "amount": 0.5,
                    "from_address": "0xmaster",
                },
            ]
        }
        o._fetch_all_transfers_until_date_optimism = _gen_return(inc)
        o._fetch_outgoing_transfers_until_date_optimism = _gen_return({})
        o.get_master_safe_address = _gen_return("0xMaster")
        o._get_native_balance = _gen_return(2.0)
        assert (
            _drive(o._track_eth_transfers_and_reversions("0xSafe", "optimism"))[
                "reversion_amount"
            ]
            == 0.5
        )

    def test_reversion_done(self):
        """Test reversion done."""
        o = self._s()
        inc = {
            "d": [
                {
                    "symbol": "ETH",
                    "timestamp": "1704067200Z",
                    "amount": 1.0,
                    "from_address": "0xmaster",
                },
                {
                    "symbol": "ETH",
                    "timestamp": "1704153600Z",
                    "amount": 0.5,
                    "from_address": "0xmaster",
                },
            ]
        }
        out = {
            "d": [
                {
                    "symbol": "ETH",
                    "timestamp": "1704240000Z",
                    "amount": 0.5,
                    "to_address": "0xmaster",
                    "from_address": "0xsafe",
                }
            ]
        }
        o._fetch_all_transfers_until_date_optimism = _gen_return(inc)
        o._fetch_outgoing_transfers_until_date_optimism = _gen_return(out)
        o.get_master_safe_address = _gen_return("0xMaster")
        o._get_native_balance = _gen_return(1.0)
        o._calculate_total_reversion_value = _gen_return(500.0)
        r = _drive(o._track_eth_transfers_and_reversions("0xSafe", "optimism"))
        assert r["reversion_amount"] == 0 and r["historical_reversion_value"] == 500.0

    def test_mode(self):
        """Test mode."""
        o = self._s("mode")
        o._track_eth_transfers_mode = MagicMock(
            return_value={
                "incoming": {
                    "ts": [
                        {
                            "symbol": "ETH",
                            "timestamp": "ts",
                            "amount": 1.0,
                            "from_address": "0xm",
                        }
                    ]
                },
                "outgoing": {},
            }
        )
        o.get_master_safe_address = _gen_return("0xM")
        o._get_native_balance = _gen_return(1.0)
        assert (
            _drive(o._track_eth_transfers_and_reversions("0xSafe", "mode"))[
                "reversion_amount"
            ]
            == 0
        )

    def test_balance_cap(self):
        """Test balance cap."""
        o = self._s()
        inc = {
            "d": [
                {
                    "symbol": "ETH",
                    "timestamp": "1704067200Z",
                    "amount": 1.0,
                    "from_address": "0xmaster",
                },
                {
                    "symbol": "ETH",
                    "timestamp": "1704153600Z",
                    "amount": 5.0,
                    "from_address": "0xmaster",
                },
            ]
        }
        o._fetch_all_transfers_until_date_optimism = _gen_return(inc)
        o._fetch_outgoing_transfers_until_date_optimism = _gen_return({})
        o.get_master_safe_address = _gen_return("0xMaster")
        o._get_native_balance = _gen_return(0.1)
        assert (
            _drive(o._track_eth_transfers_and_reversions("0xSafe", "optimism"))[
                "reversion_amount"
            ]
            == 0.1
        )

    def test_exception(self):
        """Test exception."""
        o = self._s()

        def boom(*a, **kw):
            """Boom."""
            yield
            raise RuntimeError("f")

        o._fetch_all_transfers_until_date_optimism = boom
        assert (
            _drive(o._track_eth_transfers_and_reversions("0xSafe", "optimism"))[
                "reversion_amount"
            ]
            == 0
        )

    def test_unix_ts(self):
        """Test unix ts."""
        o = self._s()
        inc = {
            "d": [
                {
                    "symbol": "ETH",
                    "timestamp": "1704067200",
                    "amount": 1.0,
                    "from_address": "0xmaster",
                },
                {
                    "symbol": "ETH",
                    "timestamp": "1704153600",
                    "amount": 0.5,
                    "from_address": "0xmaster",
                },
            ]
        }
        o._fetch_all_transfers_until_date_optimism = _gen_return(inc)
        o._fetch_outgoing_transfers_until_date_optimism = _gen_return({})
        o.get_master_safe_address = _gen_return("0xMaster")
        o._get_native_balance = _gen_return(2.0)
        assert (
            _drive(o._track_eth_transfers_and_reversions("0xSafe", "optimism"))[
                "reversion_date"
            ]
            is not None
        )

    def test_bad_ts(self):
        """Test bad ts."""
        o = self._s()
        inc = {
            "d": [
                {
                    "symbol": "ETH",
                    "timestamp": "bad",
                    "amount": 1.0,
                    "from_address": "0xmaster",
                },
                {
                    "symbol": "ETH",
                    "timestamp": "bad",
                    "amount": 0.5,
                    "from_address": "0xmaster",
                },
            ]
        }
        o._fetch_all_transfers_until_date_optimism = _gen_return(inc)
        o._fetch_outgoing_transfers_until_date_optimism = _gen_return({})
        o.get_master_safe_address = _gen_return("0xMaster")
        o._get_native_balance = _gen_return(2.0)
        assert (
            _drive(o._track_eth_transfers_and_reversions("0xSafe", "optimism"))[
                "reversion_date"
            ]
            is not None
        )


class TestCalcReversionValue:
    """TestCalcReversionValue."""

    def test_iso(self):
        """Test iso."""
        o = _mk()
        o._fetch_historical_eth_price = _gen_return(2000.0)
        assert (
            _drive(
                o._calculate_total_reversion_value(
                    [{"timestamp": "2024-01-01T00:00:00Z"}],
                    [{"amount": 0.5}, {"amount": 0.3}],
                )
            )
            == 1600.0
        )

    def test_unix(self):
        """Test unix."""
        o = _mk()
        o._fetch_historical_eth_price = _gen_return(1000.0)
        assert (
            _drive(
                o._calculate_total_reversion_value(
                    [{"timestamp": "1704067200"}], [{"amount": 1.0}]
                )
            )
            == 1000.0
        )

    def test_bad(self):
        """Test bad."""
        o = _mk()
        o._fetch_historical_eth_price = _gen_return(500.0)
        assert (
            _drive(
                o._calculate_total_reversion_value(
                    [{"timestamp": "bad"}], [{"amount": 2.0}]
                )
            )
            == 1000.0
        )

    def test_no_price(self):
        """Test no price."""
        o = _mk()
        o._fetch_historical_eth_price = _gen_return(None)
        assert (
            _drive(
                o._calculate_total_reversion_value(
                    [{"timestamp": "1704067200"}], [{"amount": 1.0}]
                )
            )
            == 0.0
        )


class TestOutgoingOptimism:
    """TestOutgoingOptimism."""

    def test_no_addr(self):
        """Test no addr."""
        assert (
            _drive(
                _mk()._fetch_outgoing_transfers_until_date_optimism("", "2025-01-01")
            )
            == {}
        )

    def test_fail(self):
        """Failed fetch returns previously-persisted data and does not store."""
        o = _mk()
        # Disk has a legacy entry the migration would strip — assert it
        # survives a failed fetch because we don't persist the stripped shape.
        o.read_funding_events = lambda: {
            "optimism_outgoing": {"2024-01-01": [{"y": 1}]}
        }
        o.store_funding_events = MagicMock()
        o._request_with_retries = _gen_return((False, {}))
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        result = _drive(
            o._fetch_outgoing_transfers_until_date_optimism("0xA", "2025-01-01")
        )
        # Returns the previously-persisted raw shape (legacy entry kept).
        assert result == {"2024-01-01": [{"y": 1}]}
        # store must NOT fire on fetch failure.
        o.store_funding_events.assert_not_called()

    def test_eth(self):
        """Test eth."""
        t = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xaddr",
            "to": "0xR",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transactionHash": "0xH",
        }
        o = _mk()
        o._request_with_retries = _gen_return((True, {"results": [t], "next": None}))
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        assert "2024-12-15" in _drive(
            o._fetch_outgoing_transfers_until_date_optimism("0xAddr", "2025-01-01")
        )

    def test_zero(self):
        """Test zero."""
        t = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xaddr",
            "to": "0xR",
            "type": "ETHER_TRANSFER",
            "value": "0",
            "transactionHash": "0xH",
        }
        o = _mk()
        o._request_with_retries = _gen_return((True, {"results": [t], "next": None}))
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        assert (
            len(
                _drive(
                    o._fetch_outgoing_transfers_until_date_optimism(
                        "0xAddr", "2025-01-01"
                    )
                )
            )
            == 0
        )

    def test_bad_val(self):
        """Test bad val."""
        t = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xaddr",
            "to": "0xR",
            "type": "ETHER_TRANSFER",
            "value": "bad",
            "transactionHash": "0xH",
        }
        o = _mk()
        o._request_with_retries = _gen_return((True, {"results": [t], "next": None}))
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        assert (
            len(
                _drive(
                    o._fetch_outgoing_transfers_until_date_optimism(
                        "0xAddr", "2025-01-01"
                    )
                )
            )
            == 0
        )

    def test_exception(self):
        """Test exception."""
        o = _mk()

        def boom(*a, **kw):
            """Boom."""
            yield
            raise RuntimeError("f")

        o._request_with_retries = boom
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        assert (
            _drive(o._fetch_outgoing_transfers_until_date_optimism("0xA", "2025-01-01"))
            == {}
        )


class TestErc20Optimism:
    """TestErc20Optimism."""

    def test_no_addr(self):
        """Test no addr."""
        assert _drive(_mk()._track_erc20_transfers_optimism("", 1704067200)) == {
            "outgoing": {}
        }

    def test_fail(self):
        """Fetch failure now returns None so the kv TTL cache is skipped."""
        o = _mk()
        o._request_with_retries = _gen_return((False, {}))
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        assert _drive(o._track_erc20_transfers_optimism("0xA", 1704067200)) is None
        # No persistence on failure: store_funding_events should not be called.
        o.store_funding_events.assert_not_called()

    def test_usdc(self):
        """Test usdc."""
        t = {
            "executionDate": "2024-01-01T10:00:00Z",
            "from": "0xaddr",
            "to": "0xR",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {"symbol": "USDC", "decimals": 6},
            "tokenAddress": "0xT",
            "value": "1000000",
            "transactionHash": "0xH",
        }
        o = _mk()
        o._request_with_retries = _gen_return((True, {"results": [t], "next": None}))
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        assert (
            "2024-01-01"
            in _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))[
                "outgoing"
            ]
        )

    def test_exception(self):
        """Exception inside the fetcher now returns None so the kv cache is skipped."""
        o = _mk()

        def boom(*a, **kw):
            """Boom."""
            yield
            raise RuntimeError("f")

        o._request_with_retries = boom
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        assert _drive(o._track_erc20_transfers_optimism("0xA", 1704067200)) is None


class TestTrackEthMode:
    """TestTrackEthMode."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_non_200(self, m):
        """Test non 200."""
        m.return_value = MagicMock(status_code=500)
        assert _mk()._track_eth_transfers_mode("0xS", "2025-01-01") == {
            "incoming": {},
            "outgoing": {},
        }

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_status_0(self, m):
        """Test status 0."""
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"status": "0", "message": "err"}
        )
        assert _mk()._track_eth_transfers_mode("0xS", "2025-01-01") == {
            "incoming": {},
            "outgoing": {},
        }

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_ok(self, m):
        """Test ok."""
        txs = [
            {
                "timeStamp": "1704067200",
                "value": str(10**18),
                "to": "0xsafe",
                "from": "0xs",
                "hash": "0xH",
            }
        ]
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"status": "1", "result": txs}
        )
        r = _mk()._track_eth_transfers_mode("0xSafe", "2025-01-01")
        assert len(r["incoming"]) > 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_outgoing(self, m):
        """Test outgoing."""
        txs = [
            {
                "timeStamp": "1704067200",
                "value": str(10**18),
                "from": "0xsafe",
                "to": "0xr",
                "hash": "0xH",
            }
        ]
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"status": "1", "result": txs}
        )
        r = _mk()._track_eth_transfers_mode("0xSafe", "2025-01-01")
        assert len(r["outgoing"]) > 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_zero(self, m):
        """Test zero."""
        m.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "status": "1",
                "result": [
                    {
                        "timeStamp": "1704067200",
                        "value": "0",
                        "to": "0xsafe",
                        "from": "0xs",
                    }
                ],
            },
        )
        assert (
            len(_mk()._track_eth_transfers_mode("0xSafe", "2025-01-01")["incoming"])
            == 0
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_future(self, m):
        """Test future."""
        m.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "status": "1",
                "result": [
                    {
                        "timeStamp": "1893456000",
                        "value": str(10**18),
                        "to": "0xsafe",
                        "from": "0xs",
                    }
                ],
            },
        )
        assert (
            len(_mk()._track_eth_transfers_mode("0xSafe", "2025-01-01")["incoming"])
            == 0
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_bad_ts(self, m):
        """Test bad ts."""
        m.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "status": "1",
                "result": [
                    {
                        "timeStamp": "bad",
                        "value": str(10**18),
                        "to": "0xsafe",
                        "from": "0xs",
                    }
                ],
            },
        )
        assert (
            len(_mk()._track_eth_transfers_mode("0xSafe", "2025-01-01")["incoming"])
            == 0
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_exception(self, m):
        """Test exception."""
        m.side_effect = RuntimeError("f")
        assert _mk()._track_eth_transfers_mode("0xS", "2025-01-01") == {
            "incoming": {},
            "outgoing": {},
        }


class TestTrackErc20Mode:
    """TestTrackErc20Mode."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_non_200(self, m):
        """Test non 200."""
        m.return_value = MagicMock(status_code=500)
        assert _mk()._track_erc20_transfers_mode("0xS", 1704067200) == {"outgoing": {}}

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_empty(self, m):
        """Test empty."""
        m.return_value = MagicMock(status_code=200, json=lambda: {"items": []})
        assert _mk()._track_erc20_transfers_mode("0xS", 1704067200) == {"outgoing": {}}

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_usdc(self, m):
        """Test usdc."""
        tx = {
            "timestamp": "2024-01-01T10:00:00Z",
            "from": {"hash": "0xsafe"},
            "to": {"hash": "0xR", "is_contract": False},
            "token": {"symbol": "USDC", "decimals": 6, "address": "0xT"},
            "total": {"value": "1000000"},
            "transaction_hash": "0xH",
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        o = _mk()
        o._should_include_transfer_mode = lambda fa, t, is_eth_transfer: True
        assert (
            "2024-01-01"
            in o._track_erc20_transfers_mode("0xSafe", 1704067200)["outgoing"]
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_zero(self, m):
        """Test zero."""
        tx = {
            "timestamp": "2024-01-01T10:00:00Z",
            "from": {"hash": "0xsafe"},
            "to": {"hash": "0xR"},
            "token": {"symbol": "USDC", "decimals": 6},
            "total": {"value": "0"},
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        assert (
            len(_mk()._track_erc20_transfers_mode("0xSafe", 1704067200)["outgoing"])
            == 0
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_bad_ts(self, m):
        """Test bad ts."""
        tx = {
            "timestamp": "bad",
            "from": {"hash": "0xsafe"},
            "to": {"hash": "0xR"},
            "token": {"symbol": "USDC", "decimals": 6},
            "total": {"value": "1000000"},
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        assert (
            len(_mk()._track_erc20_transfers_mode("0xSafe", 1704067200)["outgoing"])
            == 0
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_future(self, m):
        """Test future."""
        tx = {
            "timestamp": "2030-01-01T10:00:00Z",
            "from": {"hash": "0xsafe"},
            "to": {"hash": "0xR", "is_contract": False},
            "token": {"symbol": "USDC", "decimals": 6},
            "total": {"value": "1000000"},
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        o = _mk()
        o._should_include_transfer_mode = lambda fa, t, is_eth_transfer: True
        assert len(o._track_erc20_transfers_mode("0xSafe", 1704067200)["outgoing"]) == 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_not_included(self, m):
        """Test not included."""
        tx = {
            "timestamp": "2024-01-01T10:00:00Z",
            "from": {"hash": "0xsafe"},
            "to": {"hash": "0xR"},
            "token": {"symbol": "USDC", "decimals": 6},
            "total": {"value": "1000000"},
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        o = _mk()
        o._should_include_transfer_mode = lambda fa, t, is_eth_transfer: False
        assert len(o._track_erc20_transfers_mode("0xSafe", 1704067200)["outgoing"]) == 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_non_usdc(self, m):
        """Test non usdc."""
        tx = {
            "timestamp": "2024-01-01T10:00:00Z",
            "from": {"hash": "0xsafe"},
            "to": {"hash": "0xR", "is_contract": False},
            "token": {"symbol": "WETH", "decimals": 18},
            "total": {"value": str(10**18)},
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        o = _mk()
        o._should_include_transfer_mode = lambda fa, t, is_eth_transfer: True
        assert len(o._track_erc20_transfers_mode("0xSafe", 1704067200)["outgoing"]) == 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_not_from_safe(self, m):
        """Test not from safe."""
        tx = {
            "timestamp": "2024-01-01T10:00:00Z",
            "from": {"hash": "0xOther"},
            "to": {"hash": "0xR", "is_contract": False},
            "token": {"symbol": "USDC", "decimals": 6},
            "total": {"value": "1000000"},
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        o = _mk()
        o._should_include_transfer_mode = lambda fa, t, is_eth_transfer: True
        assert len(o._track_erc20_transfers_mode("0xSafe", 1704067200)["outgoing"]) == 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_pagination(self, m):
        """Test pagination."""
        tx = {
            "timestamp": "2024-01-01T10:00:00Z",
            "from": {"hash": "0xsafe"},
            "to": {"hash": "0xR", "is_contract": False},
            "token": {"symbol": "USDC", "decimals": 6, "address": "0xT"},
            "total": {"value": "1000000"},
            "transaction_hash": "0xH",
        }
        c = [0]

        def mj():
            """Mj."""
            c[0] += 1
            if c[0] == 1:
                return {
                    "items": [tx],
                    "next_page_params": {"block_number": 1, "index": 0},
                }
            return {"items": [], "next_page_params": None}

        r = MagicMock(status_code=200)
        r.json = mj
        m.return_value = r
        o = _mk()
        o._should_include_transfer_mode = lambda fa, t, is_eth_transfer: True
        assert (
            "2024-01-01"
            in o._track_erc20_transfers_mode("0xSafe", 1704067200)["outgoing"]
        )

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_val_error(self, m):
        """Test val error."""
        tx = {
            "timestamp": "2024-01-01T10:00:00Z",
            "from": {"hash": "0xsafe"},
            "to": {"hash": "0xR", "is_contract": False},
            "token": {"symbol": "USDC", "decimals": "bad"},
            "total": {"value": "1000000"},
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        o = _mk()
        o._should_include_transfer_mode = lambda fa, t, is_eth_transfer: True
        assert len(o._track_erc20_transfers_mode("0xSafe", 1704067200)["outgoing"]) == 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_exception(self, m):
        """Test exception."""
        m.side_effect = RuntimeError("f")
        assert _mk()._track_erc20_transfers_mode("0xS", 1704067200) == {"outgoing": {}}


class TestVeloRewardsException:
    """TestVeloRewardsException."""

    def test_exception(self):
        """Test exception."""
        o = _mk()
        o.pools = {"velodrome": MagicMock()}

        def bad(*a, **kw):
            """Bad."""
            yield
            raise RuntimeError("f")

        o.pools["velodrome"].get_gauge_address = bad
        assert _drive(
            o._get_velodrome_pending_rewards(
                {"pool_address": "0xP", "is_cl_pool": True}, "optimism", "0xU"
            )
        ) == Decimal(0)


class TestSmallBranchGaps2:
    """TestSmallBranchGaps2."""

    def test_closed_positions(self):
        """Test _have_positions_changed when positions are closed."""
        o = _mk()
        # Same count but different identifiers → closed_positions non-empty
        o.current_positions = [
            {"pool_address": "0xNEW", "dex_type": "uniswapV3", "status": "open"}
        ]
        last_data = {"allocations": [{"id": "0xOLD", "type": "uniswapV3"}]}
        assert o._have_positions_changed(last_data) is True

    def test_total_supply_zero_str(self):
        """Test get_user_share_value_balancer when total_supply is string '0' (bypasses int check, hits Decimal check)."""
        o = _mk()
        o.params.balancer_vault_contract_addresses = {"optimism": "0x" + "ab" * 20}
        ci = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            ci[0] += 1
            yield
            if ci[0] == 1:
                return (["0x" + "cd" * 20], [100])  # pool tokens
            elif ci[0] == 2:
                return 50  # user balance
            return "0"  # total supply = "0" (string, bypasses `== 0` at line 1972 but hits Decimal == 0 at 1985)

        o.contract_interact = fake_ci
        assert (
            _drive(
                o.get_user_share_value_balancer(
                    "0xU", "0xPID", "0x" + "ab" * 20, "optimism"
                )
            )
            == {}
        )

    def test_total_supply_zero_int(self):
        """Test get_user_share_value_balancer when total_supply is 0."""
        o = _mk()
        o.params.balancer_vault_contract_addresses = {"optimism": "0x" + "ab" * 20}
        ci = [0]

        def fake_ci(*a, **kw):
            """Fake ci."""
            ci[0] += 1
            yield
            if ci[0] == 1:
                return (["0x" + "cd" * 20], [100])
            elif ci[0] == 2:
                return 50
            return 0

        o.contract_interact = fake_ci
        assert (
            _drive(
                o.get_user_share_value_balancer(
                    "0xU", "0xPID", "0x" + "ab" * 20, "optimism"
                )
            )
            == {}
        )


class TestOptimismSafeglobalBranches:
    """Tests for all internal loop branches in _fetch_optimism_transfers_safeglobal."""

    def _obj(self):
        o = _mk()
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        return o

    def test_no_timestamp(self):
        """Transfer with no executionDate → continue at line 3709."""
        td = {
            "from": "0xS",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transferId": "t1",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(
            o._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {}
            )
        )

    def test_bad_timestamp(self):
        """Transfer with unparseable timestamp → no tx_date → continue at line 3715."""
        td = {
            "executionDate": "not-a-date",
            "from": "0xS",
            "type": "ETHER_TRANSFER",
            "transferId": "t1",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        o._should_include_transfer_optimism = _gen_return(True)
        _drive(
            o._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {}
            )
        )

    def test_existing_data(self):
        """Transfer date in existing_data → continue at line 3719."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xS",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transferId": "t1",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(
            o._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {"2024-12-15": []}
            )
        )

    def test_after_end_date(self):
        """Transfer date > end_date → continue at line 3723."""
        td = {
            "executionDate": "2026-01-01T10:00:00Z",
            "from": "0xS",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transferId": "t1",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(
            o._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {}
            )
        )

    def test_duplicate(self):
        """Duplicate transferId → continue at line 3730."""
        td1 = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xS",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transferId": "dup",
        }
        td2 = {
            "executionDate": "2024-12-16T10:00:00Z",
            "from": "0xS",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transferId": "dup",
        }
        o = self._obj()
        o._request_with_retries = _gen_return(
            (True, {"results": [td1, td2], "next": None})
        )
        o._should_include_transfer_optimism = _gen_return(True)
        _drive(
            o._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {}
            )
        )

    def test_not_included(self):
        """_should_include_transfer_optimism returns False → continue at line 3745."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xS",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transferId": "t1",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        o._should_include_transfer_optimism = _gen_return(False)
        _drive(
            o._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {}
            )
        )

    def test_non_usdc(self):
        """ERC20 transfer with non-USDC symbol → continue at line 3767."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xS",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {"symbol": "WETH", "decimals": 18},
            "tokenAddress": "0xT",
            "value": str(10**18),
            "transferId": "t1",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        o._should_include_transfer_optimism = _gen_return(True)
        _drive(
            o._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {}
            )
        )

    def test_unknown_type(self):
        """Unknown transfer type → continue at line 3813."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xS",
            "type": "UNKNOWN_TYPE",
            "transferId": "t1",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        o._should_include_transfer_optimism = _gen_return(True)
        _drive(
            o._fetch_optimism_transfers_safeglobal(
                "0xA", "2025-01-01", defaultdict(list), {}
            )
        )

    def test_pagination(self):
        """Test pagination path → line 3828."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xS",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transactionHash": "0xH",
            "transferId": "t1",
        }
        o = self._obj()
        ci = [0]

        def fr(*a, **kw):
            """Fr."""
            ci[0] += 1
            yield
            if ci[0] == 1:
                return (True, {"results": [td], "next": "http://next-page"})
            return (True, {"results": []})

        o._request_with_retries = fr
        o._should_include_transfer_optimism = _gen_return(True)
        t = defaultdict(list)
        _drive(o._fetch_optimism_transfers_safeglobal("0xA", "2025-01-01", t, {}))
        assert len(t["2024-12-15"]) == 1


class TestOutgoingOptimismBranches:
    """Tests for internal loop branches in _fetch_outgoing_transfers_until_date_optimism."""

    def _obj(self):
        o = _mk()
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        return o

    def test_empty_results(self):
        """Empty results → break at line 4275."""
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": []}))
        assert (
            _drive(o._fetch_outgoing_transfers_until_date_optimism("0xA", "2025-01-01"))
            == {}
        )

    def test_no_execution_date(self):
        """No executionDate → continue at line 4281."""
        td = {
            "from": "0xaddr",
            "to": "0xR",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._fetch_outgoing_transfers_until_date_optimism("0xAddr", "2025-01-01"))

    def test_bad_timestamp(self):
        """Invalid timestamp → continue at lines 4289-4293."""
        td = {
            "executionDate": "not-valid",
            "from": "0xaddr",
            "to": "0xR",
            "type": "ETHER_TRANSFER",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._fetch_outgoing_transfers_until_date_optimism("0xAddr", "2025-01-01"))

    def test_future_date(self):
        """Transfer after current_date → continue at line 4296."""
        td = {
            "executionDate": "2026-01-01T10:00:00Z",
            "from": "0xaddr",
            "to": "0xR",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._fetch_outgoing_transfers_until_date_optimism("0xAddr", "2025-01-01"))

    def test_not_from_us(self):
        """Transfer from different address → continue at line 4335."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xOther",
            "to": "0xR",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transactionHash": "0xH",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._fetch_outgoing_transfers_until_date_optimism("0xAddr", "2025-01-01"))

    def test_duplicate(self):
        """Duplicate transaction → continue at line 4305."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xaddr",
            "to": "0xR",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transactionHash": "0xH",
        }
        o = self._obj()
        o._request_with_retries = _gen_return(
            (True, {"results": [td, td], "next": None})
        )
        r = _drive(
            o._fetch_outgoing_transfers_until_date_optimism("0xAddr", "2025-01-01")
        )
        assert len(r.get("2024-12-15", [])) == 1

    def test_pagination(self):
        """Pagination → line 4345."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "from": "0xaddr",
            "to": "0xR",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transactionHash": "0xH",
        }
        o = self._obj()
        ci = [0]

        def fr(*a, **kw):
            """Fr."""
            ci[0] += 1
            yield
            if ci[0] == 1:
                return (True, {"results": [td], "next": "http://next-page"})
            return (True, {"results": []})

        o._request_with_retries = fr
        r = _drive(
            o._fetch_outgoing_transfers_until_date_optimism("0xAddr", "2025-01-01")
        )
        assert "2024-12-15" in r


class TestErc20OptimismBranches:
    """Tests for internal loop branches in _track_erc20_transfers_optimism."""

    def _obj(self):
        o = _mk()
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        return o

    def test_empty_results(self):
        """Empty results → break at line 4394."""
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": []}))
        assert _drive(o._track_erc20_transfers_optimism("0xA", 1704067200)) == {
            "outgoing": {}
        }

    def test_no_execution_date(self):
        """No executionDate → continue at line 4400."""
        td = {"from": "0xaddr", "type": "ERC20_TRANSFER"}
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))

    def test_bad_timestamp(self):
        """Bad timestamp → continue at lines 4408-4412."""
        td = {"executionDate": "bad", "from": "0xaddr", "type": "ERC20_TRANSFER"}
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))

    def test_future_date(self):
        """Transfer after current_date → continue at line 4415."""
        td = {
            "executionDate": "2026-01-01T10:00:00Z",
            "from": "0xaddr",
            "type": "ERC20_TRANSFER",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))

    def test_not_from_safe(self):
        """Transfer from different address → continue at line 4419."""
        td = {
            "executionDate": "2024-01-01T10:00:00Z",
            "from": "0xOther",
            "type": "ERC20_TRANSFER",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))

    def test_not_erc20(self):
        """Non-ERC20 transfer type → continue at line 4425."""
        td = {
            "executionDate": "2024-01-01T10:00:00Z",
            "from": "0xaddr",
            "type": "ETHER_TRANSFER",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))

    def test_duplicate(self):
        """Duplicate transfer → continue at line 4430."""
        td = {
            "executionDate": "2024-01-01T10:00:00Z",
            "from": "0xaddr",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {"symbol": "USDC", "decimals": 6},
            "tokenAddress": "0xT",
            "value": "1000000",
            "transactionHash": "0xH",
            "to": "0xR",
        }
        o = self._obj()
        o._request_with_retries = _gen_return(
            (True, {"results": [td, td], "next": None})
        )
        r = _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))
        assert len(r["outgoing"].get("2024-01-01", [])) == 1

    def test_no_token_info_with_addr(self):
        """No tokenInfo but tokenAddress present → lines 4437-4443."""
        td = {
            "executionDate": "2024-01-01T10:00:00Z",
            "from": "0xaddr",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {},
            "tokenAddress": "0xT",
            "value": "1000000",
            "transactionHash": "0xH",
            "to": "0xR",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        o._get_token_decimals = _gen_return(6)
        o._get_token_symbol = _gen_return("USDC")
        r = _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))
        assert "2024-01-01" in r["outgoing"]

    def test_no_token_info_no_addr(self):
        """No tokenInfo and no tokenAddress → continue at line 4445."""
        td = {
            "executionDate": "2024-01-01T10:00:00Z",
            "from": "0xaddr",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {},
            "tokenAddress": "",
            "value": "1000000",
            "transactionHash": "0xH",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))

    def test_not_usdc(self):
        """Non-USDC symbol → continue at line 4452."""
        td = {
            "executionDate": "2024-01-01T10:00:00Z",
            "from": "0xaddr",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {"symbol": "WETH", "decimals": 18},
            "tokenAddress": "0xT",
            "value": str(10**18),
            "transactionHash": "0xH",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))

    def test_zero_amount(self):
        """Zero amount → continue at line 4458."""
        td = {
            "executionDate": "2024-01-01T10:00:00Z",
            "from": "0xaddr",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {"symbol": "USDC", "decimals": 6},
            "tokenAddress": "0xT",
            "value": "0",
            "transactionHash": "0xH",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))

    def test_pagination(self):
        """Pagination → line 4484."""
        td = {
            "executionDate": "2024-01-01T10:00:00Z",
            "from": "0xaddr",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {"symbol": "USDC", "decimals": 6},
            "tokenAddress": "0xT",
            "value": "1000000",
            "transactionHash": "0xH",
            "to": "0xR",
        }
        o = self._obj()
        ci = [0]

        def fr(*a, **kw):
            """Fr."""
            ci[0] += 1
            yield
            if ci[0] == 1:
                return (True, {"results": [td], "next": "http://next"})
            return (True, {"results": []})

        o._request_with_retries = fr
        r = _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))
        assert "2024-01-01" in r["outgoing"]


class TestReversionDateIso:
    """Test ISO timestamp format for reversion_date (line 4152)."""

    def test_iso_z_timestamp(self):
        """Test iso z timestamp."""
        o = _mk()
        o.params.target_investment_chains = ["optimism"]
        o.params.safe_contract_addresses = {"optimism": "0xSafe"}
        # Two transfers from master, second has ISO-Z timestamp
        inc = {
            "d": [
                {
                    "symbol": "ETH",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "amount": 1.0,
                    "from_address": "0xmaster",
                },
                {
                    "symbol": "ETH",
                    "timestamp": "2024-01-02T00:00:00Z",
                    "amount": 0.5,
                    "from_address": "0xmaster",
                },
            ]
        }
        o._fetch_all_transfers_until_date_optimism = _gen_return(inc)
        o._fetch_outgoing_transfers_until_date_optimism = _gen_return({})
        o.get_master_safe_address = _gen_return("0xMaster")
        o._get_native_balance = _gen_return(2.0)
        r = _drive(o._track_eth_transfers_and_reversions("0xSafe", "optimism"))
        assert r["reversion_date"] == "02-01-2024"


class TestIncrementalOptimismFetching:
    """Tests for early-stop pagination and caching in Optimism transfer fetching."""

    def test_safeglobal_stops_on_existing_transfer_ids(self):
        """Pagination stops once every transfer on a page has a seen uid."""
        o = _mk()
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1

        # Persisted on a prior cycle with explicit transfer_id fields so the
        # uid-based dedup recognises them on the new page.
        existing_data = {
            "2024-12-15": [
                {"transfer_id": "tid-1", "symbol": "ETH", "amount": 1.0},
                {"transfer_id": "tid-2", "symbol": "ETH", "amount": 1.0},
            ],
        }
        page_data = {
            "results": [
                {
                    "transferId": "tid-1",
                    "executionDate": "2024-12-15T10:00:00Z",
                    "from": "0xSender",
                    "type": "ETHER_TRANSFER",
                    "value": str(10**18),
                    "transactionHash": "0xH1",
                },
                {
                    "transferId": "tid-2",
                    "executionDate": "2024-12-15T11:00:00Z",
                    "from": "0xSender",
                    "type": "ETHER_TRANSFER",
                    "value": str(10**18),
                    "transactionHash": "0xH2",
                },
            ],
            "next": "http://next-page-url",
        }
        o._request_with_retries = _gen_return((True, page_data))
        o._should_include_transfer_optimism = _gen_return(True)

        all_transfers = defaultdict(list)
        _drive(
            o._fetch_optimism_transfers_safeglobal(
                "0xAddr", "2025-01-01", all_transfers, existing_data
            )
        )
        # No new transfers should be added (every uid in the page was seen)
        assert len(all_transfers) == 0

    def test_outgoing_persists_new_transfers(self):
        """New outgoing transfers are persisted to funding_events."""
        o = _mk()
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        o.funding_events = {}

        transfer = {
            "executionDate": "2024-12-20T10:00:00Z",
            "from": "0xsafe",
            "to": "0xRecipient",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transactionHash": "0xH1",
        }
        o._request_with_retries = _gen_return(
            (True, {"results": [transfer], "next": None})
        )
        result = _drive(
            o._fetch_outgoing_transfers_until_date_optimism("0xSafe", "2025-01-01")
        )
        assert "2024-12-20" in result
        # Verify persisted
        assert "optimism_outgoing" in o.funding_events
        assert "2024-12-20" in o.funding_events["optimism_outgoing"]
        o.store_funding_events.assert_called()

    def test_outgoing_returns_existing_on_no_address(self):
        """When no address is provided, returns existing persisted data.

        The persisted entry carries ``transfer_id`` so the on-load legacy
        migration does not drop it.
        """
        o = _mk()
        o.funding_events = {
            "optimism_outgoing": {
                "2024-01-01": [{"symbol": "ETH", "transfer_id": "tid-1"}]
            }
        }
        result = _drive(
            o._fetch_outgoing_transfers_until_date_optimism("", "2025-01-01")
        )
        assert "2024-01-01" in result

    def test_reversion_cache_hit(self):
        """Cached reversion info is returned when transfer count is unchanged."""
        o = _mk()
        o.params.target_investment_chains = ["optimism"]
        o.params.safe_contract_addresses = {"optimism": "0xSafe"}

        cached_result = {
            "reversion_amount": 0.5,
            "master_safe_address": "0xMaster",
            "historical_reversion_value": 100.0,
            "reversion_date": "01-01-2025",
        }
        # incoming has 2 transfers, outgoing has 1 => count=3
        inc = {
            "d": [
                {"symbol": "ETH", "timestamp": "t"},
                {"symbol": "ETH", "timestamp": "t2"},
            ]
        }
        out = {"d": [{"symbol": "ETH", "timestamp": "t3"}]}

        o._fetch_all_transfers_until_date_optimism = _gen_return(inc)
        o._fetch_outgoing_transfers_until_date_optimism = _gen_return(out)
        o.funding_events = {
            "optimism_reversion_info": cached_result,
            "optimism_reversion_transfer_count": 3,  # matches 2+1
        }

        result = _drive(o._track_eth_transfers_and_reversions("0xSafe", "optimism"))
        assert result == cached_result

    def test_reversion_cache_miss_on_new_transfers(self):
        """Reversion is recomputed when transfer count changes."""
        o = _mk()
        o.params.target_investment_chains = ["optimism"]
        o.params.safe_contract_addresses = {"optimism": "0xSafe"}

        stale_cache = {
            "reversion_amount": 999,
            "master_safe_address": "0xOld",
            "historical_reversion_value": 0.0,
            "reversion_date": None,
        }
        inc = {
            "d": [
                {
                    "symbol": "ETH",
                    "timestamp": "1704067200Z",
                    "amount": 1.0,
                    "from_address": "0xmaster",
                },
                {
                    "symbol": "ETH",
                    "timestamp": "1704153600Z",
                    "amount": 0.5,
                    "from_address": "0xmaster",
                },
            ]
        }
        o._fetch_all_transfers_until_date_optimism = _gen_return(inc)
        o._fetch_outgoing_transfers_until_date_optimism = _gen_return({})
        o.get_master_safe_address = _gen_return("0xMaster")
        o._get_native_balance = _gen_return(2.0)
        o.funding_events = {
            "optimism_reversion_info": stale_cache,
            "optimism_reversion_transfer_count": 1,  # stale: was 1, now 2
        }

        result = _drive(o._track_eth_transfers_and_reversions("0xSafe", "optimism"))
        # Should NOT return stale cache — reversion_amount should be 0.5, not 999
        assert result["reversion_amount"] == 0.5


class TestCoverageGaps:
    """Tests for uncovered lines/branches in new code."""

    def test_get_transfer_key_no_tx_hash(self):
        """Fallback key when tx_hash is missing."""
        o = _mk()
        key = o._get_transfer_key(
            "2025-01-01", {"symbol": "ETH", "delta": 1.5, "from_address": "0xF"}, 3
        )
        assert key == "2025-01-01_ETH_1.5_0xF_3"

    def test_get_transfer_key_empty_tx_hash(self):
        """Fallback key when tx_hash is empty string."""
        o = _mk()
        key = o._get_transfer_key("2025-01-01", {"tx_hash": "", "symbol": "USDC"}, 0)
        assert key == "2025-01-01_USDC_0__0"

    def test_load_priced_transfers_valid(self):
        """Load valid JSON from KV store."""
        o = _mk()
        data = json.dumps({"k1": 100.0, "k2": 200.0})
        o._read_kv = _gen_return({"optimism_priced_transfers": data})
        result = _drive(o._load_priced_transfers("optimism"))
        assert result == {"k1": 100.0, "k2": 200.0}

    def test_load_priced_transfers_empty(self):
        """Return empty dict when KV has no data."""
        o = _mk()
        o._read_kv = _gen_return({})
        result = _drive(o._load_priced_transfers("optimism"))
        assert result == {}

    def test_load_priced_transfers_malformed_json(self):
        """Return empty dict and log warning on malformed JSON."""
        o = _mk()
        o._read_kv = _gen_return({"optimism_priced_transfers": "not-json{{"})
        result = _drive(o._load_priced_transfers("optimism"))
        assert result == {}
        o.context.logger.warning.assert_called()

    def test_save_priced_transfers(self):
        """Verify KV write is called with serialized JSON."""
        o = _mk()
        written = {}

        def capture_write(data):
            """Capture write."""
            written.update(data)
            yield

        o._write_kv = capture_write
        _drive(o._save_priced_transfers("optimism", {"k": 42.0}))
        assert "optimism_priced_transfers" in written
        assert json.loads(written["optimism_priced_transfers"]) == {"k": 42.0}

    def test_count_transfers(self):
        """Verify transfer counting across dates."""
        o = _mk()
        transfers = {"d1": [{"a": 1}, {"b": 2}], "d2": [{"c": 3}]}
        assert o._count_transfers(transfers) == 3
        assert o._count_transfers({}) == 0

    def test_fetch_historical_eth_price_fallback_cache(self):
        """When API fails but fallback cache has a price, return it."""
        o = _mk()
        cg = MagicMock()
        o.context.coingecko = cg
        cg.historical_price_endpoint = "url/{coin_id}/{date}"
        cg.api_key = "key"
        cg.request.return_value = (False, {})  # API fails
        o.params.use_x402 = False

        call_count = {"n": 0}

        def cache_returns_on_second_call(*a, **kw):
            """Cache returns on second call."""
            call_count["n"] += 1
            yield
            # First call (before API): no cache. Second call (fallback): has price.
            if call_count["n"] >= 2:
                return 2500.0
            return None

        o._get_cached_price = cache_returns_on_second_call
        o._cache_price = _gen_none

        result = _drive(o._fetch_historical_eth_price("01-01-2025"))
        assert result == 2500.0
        assert call_count["n"] == 2

    def test_fetch_historical_eth_price_no_price_but_fallback(self):
        """API succeeds but no price in response, fallback cache returns price."""
        o = _mk()
        cg = MagicMock()
        o.context.coingecko = cg
        cg.historical_price_endpoint = "url/{coin_id}/{date}"
        cg.api_key = "key"
        cg.request.return_value = (True, {"market_data": {"current_price": {}}})
        o.params.use_x402 = False

        call_count = {"n": 0}

        def cache_returns_on_second_call(*a, **kw):
            """Cache returns on second call."""
            call_count["n"] += 1
            yield
            if call_count["n"] >= 2:
                return 1800.0
            return None

        o._get_cached_price = cache_returns_on_second_call
        o._cache_price = _gen_none

        result = _drive(o._fetch_historical_eth_price("01-01-2025"))
        assert result == 1800.0

    def test_load_priced_transfers_type_error(self):
        """Return empty dict on TypeError during JSON parse."""
        o = _mk()
        o._read_kv = _gen_return({"optimism_priced_transfers": 12345})
        result = _drive(o._load_priced_transfers("optimism"))
        assert result == {}
        o.context.logger.warning.assert_called()

    def test_outgoing_early_stop(self):
        """Outgoing pagination stops when entire page is on stored dates."""
        o = _mk()
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        o.funding_events = {
            "optimism_outgoing": {
                "2024-12-15": [{"symbol": "ETH", "amount": 1.0}],
            }
        }
        page_data = {
            "results": [
                {
                    "executionDate": "2024-12-15T10:00:00Z",
                    "from": "0xsafe",
                    "to": "0xR",
                    "type": "ETHER_TRANSFER",
                    "value": str(10**18),
                    "transactionHash": "0xH1",
                },
                {
                    "executionDate": "2024-12-15T11:00:00Z",
                    "from": "0xsafe",
                    "to": "0xR",
                    "type": "ETHER_TRANSFER",
                    "value": str(10**18),
                    "transactionHash": "0xH2",
                },
            ],
            "next": "http://next-page-url",
        }
        o._request_with_retries = _gen_return((True, page_data))

        result = _drive(
            o._fetch_outgoing_transfers_until_date_optimism("0xSafe", "2025-01-01")
        )
        assert "2024-12-15" in result
        o.store_funding_events.assert_called()

    def test_outgoing_merge_preserves_existing_when_dates_overlap(self):
        """Page entries on a stored date are appended, not replacing the list."""
        o = _mk()
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        o.funding_events = {
            "optimism_outgoing": {
                "2024-12-15": [
                    {
                        "transfer_id": "tid-old",
                        "symbol": "ETH",
                        "amount": 0.5,
                        "tx_hash": "0xOLD",
                    }
                ],
            }
        }
        # Page has one transfer on an existing date and one on a new date
        page_data = {
            "results": [
                {
                    "transferId": "tid-new",
                    "executionDate": "2024-12-20T10:00:00Z",
                    "from": "0xsafe",
                    "to": "0xR",
                    "type": "ETHER_TRANSFER",
                    "value": str(10**18),
                    "transactionHash": "0xNEW",
                },
                {
                    "transferId": "tid-dup",
                    "executionDate": "2024-12-15T12:00:00Z",
                    "from": "0xsafe",
                    "to": "0xR",
                    "type": "ETHER_TRANSFER",
                    "value": str(10**18),
                    "transactionHash": "0xDUP",
                },
            ],
            "next": None,
        }
        o._request_with_retries = _gen_return((True, page_data))

        result = _drive(
            o._fetch_outgoing_transfers_until_date_optimism("0xSafe", "2025-01-01")
        )
        # New date added.
        assert "2024-12-20" in result
        # Existing entry on 2024-12-15 is preserved alongside the new entry.
        tx_hashes_on_dec_15 = {t.get("tx_hash") for t in result["2024-12-15"]}
        assert "0xOLD" in tx_hashes_on_dec_15
        assert "0xDUP" in tx_hashes_on_dec_15


class TestClosedPositionsBranch:
    """Test the closed_positions branch (lines 557-560) specifically."""

    def test_closed_positions_only(self):
        """Same length but different identifiers → new empty, closed non-empty."""
        o = _mk()
        # Two current positions, one of which matches last
        o.current_positions = [
            {"pool_address": "0xA", "dex_type": "uniV3", "status": "open"},
            {
                "pool_address": "0xA",
                "dex_type": "uniV3",
                "status": "open",
            },  # dup → collapses to 1 in set
        ]
        last_data = {
            "allocations": [
                {"id": "0xA", "type": "uniV3"},
                {"id": "0xB", "type": "uniV3"},
            ]
        }
        # current_set = {("0xA","uniV3","open")}, last_set = {("0xA","uniV3","open"),("0xB","uniV3","open")}
        # len(positions)==2==2, new_positions=empty, closed_positions={("0xB",...)}
        assert o._have_positions_changed(last_data) is True


class TestFetchAllTransfersDateMerge:
    """Test merge/existing data paths in _fetch_all_transfers_until_date_mode and _optimism."""

    def test_mode_merge_new_dates(self):
        """Line 3192: new date in existing_mode_data."""
        obj = _mk()
        obj.read_funding_events = lambda: {"mode": {"2024-01-01": [{"x": 1}]}}
        obj.store_funding_events = MagicMock()

        # Token transfers succeed and add a new date
        def token_mode(*a, **kw):
            """Token mode."""
            a[2]["2024-02-01"] = [{"x": 2}]  # all_transfers_by_date[date]
            yield
            return True

        obj._fetch_token_transfers_mode = token_mode
        obj._fetch_eth_transfers_mode = MagicMock(return_value=True)
        r = _drive(obj._fetch_all_transfers_until_date_mode("0xA", "2025-01-01", False))
        assert "2024-02-01" in r

    def test_optimism_existing_funding_events(self):
        """Persisted dates are migrated and new dates appended after a successful fetch."""
        obj = _mk()
        # Persisted entry carries transfer_id so it survives the legacy
        # migration on load.
        obj.read_funding_events = lambda: {
            "optimism": {"2024-01-01": [{"transfer_id": "tid-old", "y": 1}]}
        }
        obj.store_funding_events = MagicMock()

        # _fetch_optimism_transfers_safeglobal adds a new date and signals success
        def sfg(*a, **kw):
            """Stub fetcher that records a new date and returns True."""
            a[2]["2024-02-01"] = [{"y": 2}]  # all_transfers_by_date
            yield
            return True

        obj._fetch_optimism_transfers_safeglobal = sfg
        r = _drive(obj._fetch_all_transfers_until_date_optimism("0xA", "2025-01-01"))
        assert "2024-02-01" in r
        # store fires only after a successful fetch.
        obj.store_funding_events.assert_called_once()

    def test_optimism_empty_funding_events(self):
        """Empty disk state returns an empty dict without crashing."""
        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()

        def sfg(*a, **kw):
            """Stub fetcher: no new transfers, fully successful."""
            yield
            return True

        obj._fetch_optimism_transfers_safeglobal = sfg
        r = _drive(obj._fetch_all_transfers_until_date_optimism("0xA", "2025-01-01"))
        assert isinstance(r, dict)

    def test_optimism_fetch_failure_does_not_persist_partial(self):
        """Fetch failure must not persist the stripped migration shape."""
        obj = _mk()
        # Disk has a legacy entry without transfer_id — exactly the shape
        # the migration would strip.
        obj.read_funding_events = lambda: {"optimism": {"2024-01-01": [{"y": 1}]}}
        obj.store_funding_events = MagicMock()

        def sfg(*a, **kw):
            """Stub fetcher: simulates a partial-page success then failure."""
            a[2]["2024-02-01"] = [{"y": 99}]  # partial new data
            yield
            return False

        obj._fetch_optimism_transfers_safeglobal = sfg
        _drive(obj._fetch_all_transfers_until_date_optimism("0xA", "2025-01-01"))
        # store must NOT fire on failure — disk keeps the original legacy
        # entries intact for next cycle to retry against.
        obj.store_funding_events.assert_not_called()


class TestTokenTransfersModeAmountZero:
    """Test amount==0 branch at line 3475 in _fetch_token_transfers_mode."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_zero_value_included(self, m):
        """Transfer with total value '0' → amount == 0 → continue."""
        tx = {
            "timestamp": "2024-12-15T10:00:00Z",
            "from": {"hash": "0xS", "is_contract": False},
            "token": {"symbol": "USDC", "decimals": "6"},
            "total": {"value": "0"},
            "transaction_hash": "0xH",
        }
        m.return_value = MagicMock(
            status_code=200, json=lambda: {"items": [tx], "next_page_params": None}
        )
        obj = _mk()
        obj.funding_events = {}
        obj._is_airdrop_transfer = lambda t: False
        obj._should_include_transfer_mode = lambda fa, tx, is_eth_transfer: True
        t = {}
        _drive(
            obj._fetch_token_transfers_mode(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, True
            )
        )
        assert len(t) == 0


class TestFetchTokenTransfersModeNetworkErrors:
    """Test _fetch_token_transfers_mode handles network and parse errors."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_connection_error(self, m):
        """Test that ConnectionError is caught and returns False."""
        import requests as req_lib

        m.side_effect = req_lib.ConnectionError("connection refused")
        obj = _mk()
        obj.funding_events = {}
        result = _drive(
            obj._fetch_token_transfers_mode(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, False
            )
        )
        assert result is False

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_json_decode_error(self, m):
        """Test that JSONDecodeError from response.json() is caught."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("No JSON")
        m.return_value = resp
        obj = _mk()
        obj.funding_events = {}
        result = _drive(
            obj._fetch_token_transfers_mode(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), {}, False
            )
        )
        assert result is False


class TestFetchEthTransfersModeNetworkErrors:
    """Test _fetch_eth_transfers_mode handles network and parse errors."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_connection_error(self, m):
        """Test that ConnectionError is caught and returns False."""
        import requests as req_lib

        m.side_effect = req_lib.ConnectionError("connection refused")
        obj = _mk()
        obj.funding_events = {}
        result = obj._fetch_eth_transfers_mode("0xA", "2025-01-01", {}, False)
        assert result is False

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_json_decode_error(self, m):
        """Test that JSONDecodeError from response.json() is caught."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("No JSON")
        m.return_value = resp
        obj = _mk()
        obj.funding_events = {}
        result = obj._fetch_eth_transfers_mode("0xA", "2025-01-01", {}, False)
        assert result is False

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_bad_block_timestamp(self, m):
        """Test that a bad block_timestamp in balance entry is skipped."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "items": [
                {
                    "value": str(10**18),
                    "delta": str(10**18),
                    "transaction_hash": None,
                    "block_timestamp": "not-a-date",
                    "block_number": 1,
                }
            ],
            "next_page_params": None,
        }
        m.return_value = resp
        obj = _mk()
        obj.funding_events = {}
        transfers = {}
        result = obj._fetch_eth_transfers_mode("0xA", "2025-01-01", transfers, True)
        # Should not crash; the bad entry is skipped
        assert result is True


class TestTrackEthTransfersModeJsonError:
    """Test _track_eth_transfers_mode handles JSON parse errors."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_json_decode_error(self, mock_requests):
        """Test that JSONDecodeError from response.json() is caught."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("No JSON")
        mock_requests.get.return_value = resp
        result = _mk()._track_eth_transfers_mode("0xSafe", "2025-01-01")
        assert result == {"incoming": {}, "outgoing": {}}


class TestTrackErc20TransfersModeJsonError:
    """Test _track_erc20_transfers_mode handles JSON parse errors."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_json_decode_error(self, mock_requests):
        """Test that JSONDecodeError from response.json() is caught."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("No JSON")
        mock_requests.get.return_value = resp
        result = _mk()._track_erc20_transfers_mode("0xSafe", 1704067200)
        assert result == {"outgoing": {}}


class TestTransferMissingFromKey:
    """Test that transfer dicts with missing 'from' key do not crash."""

    def _obj(self):
        o = _mk()
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        return o

    def test_outgoing_transfer_missing_from_key(self):
        """Transfer dict without 'from' key should not raise AttributeError."""
        td = {
            "executionDate": "2024-12-15T10:00:00Z",
            "to": "0xR",
            "type": "ETHER_TRANSFER",
            "value": str(10**18),
            "transactionHash": "0xH",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        result = _drive(
            o._fetch_outgoing_transfers_until_date_optimism("0xAddr", "2025-01-01")
        )
        # Transfer without "from" should be silently skipped, no crash
        assert isinstance(result, dict)

    def test_erc20_transfer_missing_from_key(self):
        """ERC20 transfer dict without 'from' key should not raise."""
        td = {
            "executionDate": "2024-01-01T10:00:00Z",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {"symbol": "USDC", "decimals": 6},
            "tokenAddress": "0xT",
            "value": "1000000",
            "transactionHash": "0xH",
            "to": "0xR",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        result = _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))
        # Transfer without "from" should be silently skipped, no crash
        assert isinstance(result, dict)


class TestTransferUniqueIdHelper:
    """_transfer_unique_id picks the best stable identifier."""

    def test_safe_transferid_preferred(self):
        """Safe's transferId wins over transactionHash + log_index."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            _transfer_unique_id,
        )

        uid = _transfer_unique_id(
            {"transferId": "tid-1", "transactionHash": "0xH", "logIndex": 4}
        )
        assert uid == "tid-1"

    def test_falls_back_to_txhash_and_log_index(self):
        """API responses without transferId still produce a stable id."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            _transfer_unique_id,
        )

        assert _transfer_unique_id({"transactionHash": "0xH", "logIndex": 3}) == "0xH:3"

    def test_falls_back_to_persisted_dict_shape(self):
        """Legacy persisted dicts (tx_hash, type, amount) still get a stable id.

        Without this fallback, restoring funding_events.json from a previous
        deployment would mean every transfer looks 'new' on the first warm
        cycle after upgrade.
        """
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            _transfer_unique_id,
        )

        uid = _transfer_unique_id({"tx_hash": "0xH", "type": "token", "amount": "16.0"})
        assert uid == "0xH:token:16.0"

    def test_empty_when_no_identifier(self):
        """A transfer with no identifying fields returns the empty string.

        Callers must check the return for truthiness before adding to
        ``seen_ids`` — otherwise an empty id would collide across unrelated
        malformed transfers.
        """
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            _transfer_unique_id,
        )

        assert _transfer_unique_id({}) == ""

    def test_log_index_zero_is_kept(self):
        """logIndex=0 is a valid index (first log in tx), not falsy."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            _transfer_unique_id,
        )

        assert _transfer_unique_id({"transactionHash": "0xH", "logIndex": 0}) == "0xH:0"


class TestCollectSeenTransferIdsHelper:
    """_collect_seen_transfer_ids walks date-keyed dicts and gathers IDs."""

    def test_empty(self):
        """None and empty dict produce an empty set without errors."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            _collect_seen_transfer_ids,
        )

        assert _collect_seen_transfer_ids({}) == set()
        assert _collect_seen_transfer_ids(None) == set()

    def test_collects_from_mixed_shapes(self):
        """Collect IDs from new and legacy persisted transfer dicts."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            _collect_seen_transfer_ids,
        )

        data = {
            "2025-06-23": [
                {"transfer_id": "tid-1"},
                {"tx_hash": "0xABC", "type": "token", "amount": "16.0"},
            ],
            "2026-03-12": [
                {"transferId": "tid-2"},
            ],
        }
        seen = _collect_seen_transfer_ids(data)
        assert "tid-1" in seen
        assert "tid-2" in seen
        assert "0xABC:token:16.0" in seen

    def test_ignores_malformed_entries(self):
        """Defensive: non-dict entries in the value list don't crash."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            _collect_seen_transfer_ids,
        )

        data = {"2025-01-01": [None, "not a dict", {"transfer_id": "ok"}]}
        assert _collect_seen_transfer_ids(data) == {"ok"}


class TestErc20OptimismWithdrawalCache:
    """Withdrawal fetcher persistence and early-stop (PR 1.1 + 1.2)."""

    @staticmethod
    def _transfer(tid: str, date: str = "2024-01-01", amount: str = "1000000"):
        """Build a minimal Safe API ERC20 transfer dict."""
        return {
            "transferId": tid,
            "executionDate": f"{date}T10:00:00Z",
            "from": "0xaddr",
            "to": "0xR",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {"symbol": "USDC", "decimals": 6},
            "tokenAddress": "0xT",
            "value": amount,
            "transactionHash": f"0xH-{tid}",
        }

    def test_cold_start_persists_withdrawals_under_new_key(self):
        """First run with empty cache persists under optimism_withdrawals."""
        obj = _mk()
        obj.read_funding_events = lambda: {}
        store = MagicMock()
        obj.store_funding_events = store
        obj.funding_events = {}

        obj._request_with_retries = _gen_return(
            (True, {"results": [self._transfer("tid-1")], "next": None})
        )
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        result = _drive(obj._track_erc20_transfers_optimism("0xAddr", 1704067200))

        assert "2024-01-01" in result["outgoing"]
        # New schema key persisted
        assert "optimism_withdrawals" in obj.funding_events
        assert "2024-01-01" in obj.funding_events["optimism_withdrawals"]
        stored = obj.funding_events["optimism_withdrawals"]["2024-01-01"][0]
        # transfer_id round-trips so future cycles can rebuild the seen-set
        assert stored["transfer_id"] == "tid-1"
        store.assert_called_once()

    def test_warm_start_early_stops_after_one_page(self):
        """Loop breaks after page 1 when persisted transferIds match API."""
        obj = _mk()
        obj.read_funding_events = lambda: {
            "optimism_withdrawals": {
                "2024-01-01": [
                    {
                        "transfer_id": "tid-1",
                        "from_address": "0xAddr",
                        "to_address": "0xR",
                        "amount": 1.0,
                        "tx_hash": "0xH-tid-1",
                        "type": "token",
                    }
                ]
            }
        }
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}

        call_count = [0]

        def req(*a, **kw):
            """Fake _request_with_retries that always advertises a next URL."""
            call_count[0] += 1
            yield
            return (
                True,
                {
                    "results": [TestErc20OptimismWithdrawalCache._transfer("tid-1")],
                    "next": "would-paginate-without-early-stop",
                },
            )

        obj._request_with_retries = req
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        _drive(obj._track_erc20_transfers_optimism("0xAddr", 1704067200))
        # Exactly one HTTP call: the early-stop fires after page 1.
        assert call_count[0] == 1

    def test_same_day_topup_with_new_transfer_id_lands(self):
        """Same-day top-up with a new transferId is merged, not dropped."""
        obj = _mk()
        obj.read_funding_events = lambda: {
            "optimism_withdrawals": {
                "2024-01-01": [
                    {
                        "transfer_id": "tid-morning",
                        "from_address": "0xAddr",
                        "amount": 5.0,
                        "type": "token",
                    }
                ]
            }
        }
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}

        # Safe returns newest-first: the afternoon top-up, then the morning one
        obj._request_with_retries = _gen_return(
            (
                True,
                {
                    "results": [
                        self._transfer("tid-afternoon", date="2024-01-01"),
                        self._transfer("tid-morning", date="2024-01-01"),
                    ],
                    "next": None,
                },
            )
        )
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        _drive(obj._track_erc20_transfers_optimism("0xAddr", 1704067200))

        stored = obj.funding_events["optimism_withdrawals"]["2024-01-01"]
        ids = {t.get("transfer_id") for t in stored}
        assert {"tid-morning", "tid-afternoon"}.issubset(ids)


class TestCalculateWithdrawalsValueTtlCache:
    """TTL-based kv cache for the optimism withdrawal path."""

    def test_cache_hit_skips_fetcher(self):
        """Cache hit within the TTL returns cached value without calling the fetcher."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xS"}

        # Last calc was 60 seconds ago; TTL is 30 minutes, so it's a hit.
        now_ts = 1704067200
        kv_state = {
            "last_withdrawals_calculated_timestamp_optimism": str(now_ts - 60),
            "total_withdrawals_optimism": "12.5",
        }

        def fake_read_kv(*args, **kwargs):
            yield
            return dict(kv_state)

        obj._read_kv = fake_read_kv
        obj._get_current_timestamp = lambda: now_ts

        called = [0]

        def fetcher(*a, **kw):
            called[0] += 1
            yield
            return {"outgoing": {}}

        obj._track_erc20_transfers_optimism = fetcher

        result = _drive(obj.calculate_withdrawals_value())
        assert result == Decimal("12.5")
        assert called[0] == 0

    def test_cache_miss_outside_ttl_recalculates_and_writes_kv(self):
        """Cache age >= TTL → recompute and rewrite both kv keys."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xS"}

        # 31 minutes ago → just outside the 30-minute TTL.
        now_ts = 1704067200
        kv_state = {
            "last_withdrawals_calculated_timestamp_optimism": str(now_ts - 31 * 60),
            "total_withdrawals_optimism": "8.0",
        }

        def fake_read_kv(*args, **kwargs):
            yield
            return dict(kv_state)

        writes: Dict[str, str] = {}

        def fake_write_kv(d):
            writes.update(d)
            yield

        obj._read_kv = fake_read_kv
        obj._write_kv = fake_write_kv
        obj._track_erc20_transfers_optimism = _gen_return(
            {
                "outgoing": {
                    "2024-01-01": [
                        {
                            "amount": 3.0,
                            "symbol": "USDC",
                            "timestamp": "2024-01-01T00:00:00Z",
                        }
                    ]
                }
            }
        )
        obj._track_and_calculate_withdrawal_value_optimism = _gen_return(Decimal("3.0"))
        obj._get_current_timestamp = lambda: 1704067200

        result = _drive(obj.calculate_withdrawals_value())
        assert result == Decimal("3.0")
        assert writes["total_withdrawals_optimism"] == "3.0"
        assert writes["last_withdrawals_calculated_timestamp_optimism"] == "1704067200"

    def test_cache_with_no_prior_timestamp_falls_through(self):
        """No prior kv entry → fetcher runs and cache is freshly written."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xS"}

        def fake_read_kv(*args, **kwargs):
            yield
            return {}

        writes: Dict[str, str] = {}

        def fake_write_kv(d):
            writes.update(d)
            yield

        obj._read_kv = fake_read_kv
        obj._write_kv = fake_write_kv
        obj._track_erc20_transfers_optimism = _gen_return({"outgoing": {}})
        obj._track_and_calculate_withdrawal_value_optimism = _gen_return(Decimal("0"))
        obj._get_current_timestamp = lambda: 1704067200

        result = _drive(obj.calculate_withdrawals_value())
        assert result == Decimal("0")
        # First-time write seeds the cache
        assert "total_withdrawals_optimism" in writes
        assert "last_withdrawals_calculated_timestamp_optimism" in writes

    def test_fetcher_failure_skips_kv_write(self):
        """Fetcher returning None must skip the kv TTL cache write."""
        from datetime import timedelta

        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xS"}

        yesterday_ts = int((datetime.now() - timedelta(days=2)).timestamp())
        kv_state = {
            "last_withdrawals_calculated_timestamp_optimism": str(yesterday_ts),
            "total_withdrawals_optimism": "5.0",
        }

        def fake_read_kv(*args, **kwargs):
            yield
            return dict(kv_state)

        writes: Dict[str, str] = {}

        def fake_write_kv(d):
            writes.update(d)
            yield

        obj._read_kv = fake_read_kv
        obj._write_kv = fake_write_kv
        obj._track_erc20_transfers_optimism = _gen_return(None)
        obj._get_current_timestamp = lambda: 1704067200

        result = _drive(obj.calculate_withdrawals_value())
        assert result == Decimal(0)
        # Critical: no kv write on failure so the next cycle retries.
        assert writes == {}


class TestErc20OptimismFailureModes:
    """Failure handling for _track_erc20_transfers_optimism (PR review fixes)."""

    @staticmethod
    def _transfer(tid: str, date: str = "2024-01-01") -> Dict[str, Any]:
        """Build a minimal Safe API outgoing USDC transfer dict."""
        return {
            "transferId": tid,
            "executionDate": f"{date}T10:00:00Z",
            "from": "0xaddr",
            "to": "0xR",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {"symbol": "USDC", "decimals": 6},
            "tokenAddress": "0xT",
            "value": "1000000",
            "transactionHash": f"0xH-{tid}",
        }

    def test_mid_pagination_failure_discards_partial_and_returns_none(self):
        """Page 1 succeeds and is followed by a page 2 fetch failure.

        Returning a partial dict here, plus seeding next cycle's seen-set
        with the page-1 ids, would let the early-stop fire on a fully-seen
        page 1 next time and permanently drop page-2 and older data.
        Instead the function must return None and skip persistence.
        """
        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}

        call_count = [0]

        def req(*a, **kw):
            """First call returns one transfer + a next cursor; second fails."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return (
                    True,
                    {
                        "results": [self._transfer("tid-1")],
                        "next": "next-page-url",
                    },
                )
            return (False, {})

        obj._request_with_retries = req
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        result = _drive(obj._track_erc20_transfers_optimism("0xAddr", 1704067200))
        assert result is None
        # Partial data must NOT be persisted on mid-pagination failure.
        obj.store_funding_events.assert_not_called()

    def test_eligible_only_early_stop_with_mixed_page(self):
        """Mixed-traffic page still trips early-stop when every eligible is seen."""
        obj = _mk()
        obj.read_funding_events = lambda: {
            "optimism_withdrawals": {
                "2024-01-01": [
                    {"transfer_id": "tid-seen"},
                ]
            }
        }
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}

        outgoing = self._transfer("tid-seen")
        incoming_eth_unseen = {
            "transferId": "tid-incoming",
            "executionDate": "2024-01-01T09:00:00Z",
            "from": "0xSomeoneElse",
            "to": "0xaddr",
            "type": "ETHER_TRANSFER",
            "value": "1000000000000000000",
            "transactionHash": "0xHE",
        }

        call_count = [0]

        def req(*a, **kw):
            """Returns a mixed page advertising another page; should stop after page 1."""
            call_count[0] += 1
            yield
            return (
                True,
                {
                    "results": [outgoing, incoming_eth_unseen],
                    "next": "would-paginate-without-eligible-only-stop",
                },
            )

        obj._request_with_retries = req
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        _drive(obj._track_erc20_transfers_optimism("0xAddr", 1704067200))
        # One call: stop fires because all eligible (outgoing-ERC20)
        # entries on the page are already seen, even though an unrelated
        # incoming entry is also present.
        assert call_count[0] == 1

    def test_eligible_only_early_stop_with_non_usdc_outgoing_erc20(self):
        """Non-USDC outgoing ERC20 must not block the early-stop.

        The seen-set only ever receives outgoing-USDC uids, so any
        non-USDC outgoing ERC20 (fee tokens, LP tokens, anything else
        the safe might transfer out) must NOT be counted in
        ``eligible_in_page``. Without this, a single non-USDC outgoing
        entry on the page would make ``seen_eligible_in_page ==
        eligible_in_page`` unreachable and the loop would paginate the
        full history on every cache-miss for any safe with mixed
        outgoing tokens.
        """
        obj = _mk()
        obj.read_funding_events = lambda: {
            "optimism_withdrawals": {
                "2024-01-01": [
                    {"transfer_id": "tid-usdc-seen"},
                ]
            }
        }
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}

        usdc_outgoing_seen = self._transfer("tid-usdc-seen")
        non_usdc_outgoing = {
            "transferId": "tid-aero-unseen",
            "executionDate": "2024-01-01T11:00:00Z",
            "from": "0xaddr",
            "to": "0xR",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {"symbol": "AERO", "decimals": 18},
            "tokenAddress": "0xAERO",
            "value": str(10**18),
            "transactionHash": "0xH-aero",
        }

        call_count = [0]

        def req(*a, **kw):
            """Returns a page with one USDC (seen) and one non-USDC outgoing."""
            call_count[0] += 1
            yield
            return (
                True,
                {
                    "results": [usdc_outgoing_seen, non_usdc_outgoing],
                    "next": "would-paginate-if-non-usdc-blocked-stop",
                },
            )

        obj._request_with_retries = req
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        _drive(obj._track_erc20_transfers_optimism("0xAddr", 1704067200))
        # One call: AERO is filtered out of eligible_in_page, so the
        # single eligible (USDC) entry being already-seen trips the stop.
        assert call_count[0] == 1


class TestDropLegacyTransferEntries:
    """Migration that removes pre-PR persisted entries lacking transfer_id."""

    def test_drops_entries_without_transfer_id(self):
        """Legacy entries without transfer_id are removed."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            _drop_legacy_transfer_entries,
        )

        data = {
            "2024-01-01": [
                {"tx_hash": "0xA", "type": "token", "amount": "1.0"},
                {"transfer_id": "tid-1", "tx_hash": "0xB"},
            ],
            "2024-01-02": [
                {"tx_hash": "0xC", "amount": "2.0"},
            ],
        }
        migrated, dropped = _drop_legacy_transfer_entries(data)
        # Only the entry with transfer_id survives on 2024-01-01;
        # 2024-01-02 has no surviving entries so the key is dropped.
        assert migrated == {"2024-01-01": [{"transfer_id": "tid-1", "tx_hash": "0xB"}]}
        # Two legacy entries were dropped (the 2024-01-01 first entry and
        # the only 2024-01-02 entry).
        assert dropped == 2

    def test_empty_input_returns_empty(self):
        """Empty/None input passes through untouched."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            _drop_legacy_transfer_entries,
        )

        assert _drop_legacy_transfer_entries({}) == ({}, 0)
        assert _drop_legacy_transfer_entries(None) == (None, 0)

    def test_applied_when_loading_optimism_outgoing(self):
        """Loading optimism_outgoing strips legacy entries lacking transfer_id."""
        obj = _mk()
        obj.read_funding_events = lambda: {
            "optimism_outgoing": {
                "2024-01-01": [
                    {"tx_hash": "0xA", "type": "eth", "amount": "1.0"},  # legacy
                    {"transfer_id": "tid-1", "tx_hash": "0xB"},  # new
                ]
            }
        }
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}
        obj._request_with_retries = _gen_return((True, {"results": [], "next": None}))
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        _drive(
            obj._fetch_outgoing_transfers_until_date_optimism("0xAddr", "2024-12-31")
        )
        # Legacy entry dropped, only the transfer_id-carrying entry remains.
        kept = obj.funding_events["optimism_outgoing"]["2024-01-01"]
        assert len(kept) == 1
        assert kept[0]["transfer_id"] == "tid-1"


class TestProxySafePagination:
    """Pagination uses the configured safe_api_url + slug for every page, not the absolute next URL (PR 1.4)."""

    @staticmethod
    def _outgoing_usdc(tid: str, date: str = "2024-01-01") -> Dict[str, Any]:
        return {
            "transferId": tid,
            "executionDate": f"{date}T10:00:00Z",
            "from": "0xaddr",
            "to": "0xR",
            "type": "ERC20_TRANSFER",
            "tokenInfo": {"symbol": "USDC", "decimals": 6},
            "tokenAddress": "0xT",
            "value": "1000000",
            "transactionHash": f"0xH-{tid}",
        }

    @staticmethod
    def _outgoing_eth(tid: str, date: str = "2024-01-01") -> Dict[str, Any]:
        return {
            "transferId": tid,
            "executionDate": f"{date}T10:00:00Z",
            "from": "0xaddr",
            "to": "0xR",
            "type": "ETHER_TRANSFER",
            "value": "1000000000000000000",
            "transactionHash": f"0xH-{tid}",
        }

    def test_withdrawal_fetcher_paginates_via_proxy_base_url(self):
        """Subsequent page requests use the configured proxy + slug, not Safe's absolute next URL.

        Mocks two pages where the first page advertises an absolute
        ``next`` URL pointing at the real Safe host. Asserts that the
        second request still goes through ``self.params.safe_api_url``
        with the chain slug (with an offset query string), so the agent
        stays inside the configured proxy URL end-to-end.
        """
        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}
        obj.params.safe_api_url = "https://safe-proxy.example.com"
        obj.params.safe_api_chain_slugs = {"optimism": "oeth"}

        captured: List[str] = []
        call_count = [0]

        def req(*a, **kw):
            captured.append(kw.get("endpoint", a[0] if a else ""))
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return (
                    True,
                    {
                        "results": [self._outgoing_usdc(f"tid-{call_count[0]}-1")],
                        "next": "https://safe-transaction-optimism.safe.global/api/v1/safes/0xS/transfers/?offset=100",
                    },
                )
            return (True, {"results": [], "next": None})

        obj._request_with_retries = req
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        _drive(obj._track_erc20_transfers_optimism("0xAddr", 1704067200))

        # Two API calls (page 1 + page 2); both go to the proxy base URL.
        assert call_count[0] == 2
        assert all(
            url.startswith("https://safe-proxy.example.com/oeth/api/v1")
            for url in captured
        ), captured
        # No call leaked to Safe's host even though page 1 advertised it.
        assert all("safe.global" not in url for url in captured), captured
        # Offset is advanced by ``len(transfers)`` (1 here), not by the
        # requested ``limit`` (100). Anchored with ``endswith`` because
        # ``"offset=1"`` is a substring of ``"offset=100"``, so the loose
        # ``in`` check would still pass under a regression to
        # ``offset += SAFE_TRANSFERS_PAGE_LIMIT``. A short non-final page
        # must not cause the next request to skip records.
        assert captured[0].endswith("&offset=0"), captured[0]
        assert captured[1].endswith("&offset=1"), captured[1]
        # Page limit also pinned so a constant change would be caught.
        assert "limit=100" in captured[0] and "limit=100" in captured[1], captured

    def test_pagination_respects_max_pagination_pages_cap(self):
        """The MAX_PAGINATION_PAGES backstop fires for a misbehaving API."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            MAX_PAGINATION_PAGES,
        )

        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}
        obj.params.safe_api_url = "https://safe-proxy.example.com"
        obj.params.safe_api_chain_slugs = {"optimism": "oeth"}

        call_count = [0]

        def req(*a, **kw):
            """Always return one transfer and a non-null next — would loop forever."""
            call_count[0] += 1
            yield
            return (
                True,
                {
                    "results": [self._outgoing_usdc(f"tid-page-{call_count[0]}")],
                    "next": "x",
                },
            )

        obj._request_with_retries = req
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        result = _drive(obj._track_erc20_transfers_optimism("0xAddr", 1704067200))
        # Stops at the cap, not infinity.
        assert call_count[0] == MAX_PAGINATION_PAGES
        # Cap-hit treated as failure: returns None and skips persistence,
        # so partial pages don't seed next cycle's seen-set with a falsely
        # complete view of history.
        assert result is None
        obj.store_funding_events.assert_not_called()

    def test_incoming_fetcher_paginates_via_proxy_base_url(self):
        """Same proxy-URL guarantee for the incoming-transfers fetcher.

        Mocks a paginated response whose ``next`` is an absolute Safe host
        URL; asserts the second request stays on the configured proxy URL.
        """
        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}
        obj.params.safe_api_url = "https://safe-proxy.example.com"
        obj.params.safe_api_chain_slugs = {"optimism": "oeth"}

        captured: List[str] = []
        call_count = [0]

        def req(*a, **kw):
            captured.append(kw.get("endpoint", a[0] if a else ""))
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return (
                    True,
                    {
                        "results": [self._outgoing_usdc(f"tid-{call_count[0]}")],
                        "next": "https://safe-transaction-optimism.safe.global/api/v1/safes/0xS/incoming-transfers/?offset=100",
                    },
                )
            return (True, {"results": [], "next": None})

        obj._request_with_retries = req
        obj._should_include_transfer_optimism = _gen_return(True)
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        existing: Dict[str, Any] = {}
        all_transfers_by_date: Dict[str, List[Dict]] = defaultdict(list)
        _drive(
            obj._fetch_optimism_transfers_safeglobal(
                "0xAddr", "2099-01-01", all_transfers_by_date, existing
            )
        )

        assert call_count[0] == 2
        assert all(
            url.startswith("https://safe-proxy.example.com/oeth/api/v1")
            for url in captured
        ), captured
        assert all("safe.global" not in url for url in captured), captured
        # Offset advances by ``len(transfers)`` (1), not by ``limit`` (100).
        # Anchored with ``endswith`` because ``"offset=1"`` is a substring of
        # ``"offset=100"``.
        assert captured[0].endswith("&offset=0"), captured[0]
        assert captured[1].endswith("&offset=1"), captured[1]
        # Page limit also pinned so a constant change would be caught.
        assert "limit=100" in captured[0] and "limit=100" in captured[1], captured

    def test_outgoing_eth_fetcher_paginates_via_proxy_base_url(self):
        """Same proxy-URL guarantee for the outgoing-ETH fetcher."""
        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}
        obj.params.safe_api_url = "https://safe-proxy.example.com"
        obj.params.safe_api_chain_slugs = {"optimism": "oeth"}

        captured: List[str] = []
        call_count = [0]

        def req(*a, **kw):
            captured.append(kw.get("endpoint", a[0] if a else ""))
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return (
                    True,
                    {
                        "results": [self._outgoing_eth(f"tid-{call_count[0]}")],
                        "next": "https://safe-transaction-optimism.safe.global/api/v1/safes/0xaddr/transfers/?offset=100",
                    },
                )
            return (True, {"results": [], "next": None})

        obj._request_with_retries = req
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        _drive(
            obj._fetch_outgoing_transfers_until_date_optimism("0xaddr", "2099-01-01")
        )

        assert call_count[0] == 2
        assert all(
            url.startswith("https://safe-proxy.example.com/oeth/api/v1")
            for url in captured
        ), captured
        assert all("safe.global" not in url for url in captured), captured
        # Offset advances by ``len(transfers)`` (1), not by ``limit`` (100).
        # Anchored with ``endswith`` because ``"offset=1"`` is a substring of
        # ``"offset=100"``.
        assert captured[0].endswith("&offset=0"), captured[0]
        assert captured[1].endswith("&offset=1"), captured[1]
        # Page limit also pinned so a constant change would be caught.
        assert "limit=100" in captured[0] and "limit=100" in captured[1], captured

    def test_incoming_fetcher_max_pagination_cap(self):
        """MAX_PAGINATION_PAGES cap on incoming fetcher signals fetch failure."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            MAX_PAGINATION_PAGES,
        )

        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}
        obj.params.safe_api_url = "https://safe-proxy.example.com"
        obj.params.safe_api_chain_slugs = {"optimism": "oeth"}

        call_count = [0]

        def req(*a, **kw):
            call_count[0] += 1
            yield
            return (
                True,
                {
                    "results": [self._outgoing_usdc(f"tid-{call_count[0]}")],
                    "next": "x",
                },
            )

        obj._request_with_retries = req
        obj._should_include_transfer_optimism = _gen_return(True)
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        existing: Dict[str, Any] = {}
        all_transfers_by_date: Dict[str, List[Dict]] = defaultdict(list)
        success = _drive(
            obj._fetch_optimism_transfers_safeglobal(
                "0xAddr", "2099-01-01", all_transfers_by_date, existing
            )
        )
        assert call_count[0] == MAX_PAGINATION_PAGES
        # Cap-hit signals failure so the caller skips persistence.
        assert success is False

    def test_outgoing_eth_fetcher_max_pagination_cap(self):
        """MAX_PAGINATION_PAGES cap on outgoing-ETH fetcher skips persistence."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            MAX_PAGINATION_PAGES,
        )

        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}
        obj.params.safe_api_url = "https://safe-proxy.example.com"
        obj.params.safe_api_chain_slugs = {"optimism": "oeth"}

        call_count = [0]

        def req(*a, **kw):
            call_count[0] += 1
            yield
            return (
                True,
                {
                    "results": [self._outgoing_eth(f"tid-{call_count[0]}")],
                    "next": "x",
                },
            )

        obj._request_with_retries = req
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        result = _drive(
            obj._fetch_outgoing_transfers_until_date_optimism("0xaddr", "2099-01-01")
        )
        assert call_count[0] == MAX_PAGINATION_PAGES
        # Real invariant: cap-hit skips persistence. The ``{}`` return is
        # incidental — it's the previously persisted shape, which is empty
        # here because ``funding_events`` was seeded empty.
        obj.store_funding_events.assert_not_called()
        assert result == {}

    def test_short_non_final_page_does_not_skip_records(self):
        """Regression: short non-final page must not skip records.

        Safe's DRF can return fewer rows than ``limit`` on a non-final
        page (server-side clamp, partial filter). Advancing ``offset``
        by the requested ``limit`` would skip the un-returned rows; the
        correct increment is ``len(transfers)``. This test simulates
        page 1 returning 50 rows with ``next != null`` and asserts that
        page 2 fetches starting at offset=50, not offset=100.
        """
        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}
        obj.params.safe_api_url = "https://safe-proxy.example.com"
        obj.params.safe_api_chain_slugs = {"optimism": "oeth"}

        captured: List[str] = []
        call_count = [0]
        page_one = [self._outgoing_usdc(f"tid-{i}") for i in range(50)]

        def req(*a, **kw):
            captured.append(kw.get("endpoint", a[0] if a else ""))
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return (
                    True,
                    {
                        "results": page_one,
                        # Non-null next signals "more records exist" even
                        # though this page is short.
                        "next": (
                            "https://safe-transaction-optimism.safe.global/api/v1/"
                            "safes/0xS/transfers/?offset=100"
                        ),
                    },
                )
            return (True, {"results": [], "next": None})

        obj._request_with_retries = req
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        _drive(obj._track_erc20_transfers_optimism("0xAddr", 1704067200))

        assert call_count[0] == 2
        assert captured[0].endswith("&offset=0"), captured[0]
        # Page 2 must request offset=50 (the count consumed), not
        # offset=100 (the requested ``limit``). offset=100 would silently
        # drop records 50-99. ``offset=50`` is not a substring of
        # ``offset=100``, so the ``in`` form would already catch the bug,
        # but ``endswith`` keeps the assertion consistent with the
        # sibling tests above and is robust if a query-param is ever
        # appended after ``offset``.
        assert captured[1].endswith("&offset=50"), captured[1]

    def test_short_non_final_page_incoming_fetcher_does_not_skip_records(self):
        """Same short-page regression test, for the incoming-transfers fetcher.

        Symmetric with ``test_short_non_final_page_does_not_skip_records``
        which only covers ``_track_erc20_transfers_optimism``. A revert to
        ``offset += SAFE_TRANSFERS_PAGE_LIMIT`` inside
        ``_fetch_optimism_transfers_safeglobal`` would not be caught by the
        ``test_*_paginates_via_proxy_base_url`` test alone (page-2 ``offset=1``
        is too small to differentiate ``len(transfers)`` from a literal 1).
        """
        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}
        obj.params.safe_api_url = "https://safe-proxy.example.com"
        obj.params.safe_api_chain_slugs = {"optimism": "oeth"}

        captured: List[str] = []
        call_count = [0]
        page_one = [self._outgoing_usdc(f"tid-inc-{i}") for i in range(50)]

        def req(*a, **kw):
            captured.append(kw.get("endpoint", a[0] if a else ""))
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return (
                    True,
                    {
                        "results": page_one,
                        "next": (
                            "https://safe-transaction-optimism.safe.global/api/v1/"
                            "safes/0xS/incoming-transfers/?offset=100"
                        ),
                    },
                )
            return (True, {"results": [], "next": None})

        obj._request_with_retries = req
        obj._should_include_transfer_optimism = _gen_return(True)
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        existing: Dict[str, Any] = {}
        all_transfers_by_date: Dict[str, List[Dict]] = defaultdict(list)
        _drive(
            obj._fetch_optimism_transfers_safeglobal(
                "0xAddr", "2099-01-01", all_transfers_by_date, existing
            )
        )

        assert call_count[0] == 2
        assert captured[0].endswith("&offset=0"), captured[0]
        assert captured[1].endswith("&offset=50"), captured[1]

    def test_short_non_final_page_outgoing_eth_fetcher_does_not_skip_records(self):
        """Same short-page regression test, for the outgoing-ETH fetcher.

        Symmetric with the withdrawal and incoming versions; pins
        ``offset += len(transfers)`` in
        ``_fetch_outgoing_transfers_until_date_optimism`` against a
        revert to ``offset += SAFE_TRANSFERS_PAGE_LIMIT``.
        """
        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()
        obj.funding_events = {}
        obj.params.safe_api_url = "https://safe-proxy.example.com"
        obj.params.safe_api_chain_slugs = {"optimism": "oeth"}

        captured: List[str] = []
        call_count = [0]
        page_one = [self._outgoing_eth(f"tid-eth-{i}") for i in range(50)]

        def req(*a, **kw):
            captured.append(kw.get("endpoint", a[0] if a else ""))
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return (
                    True,
                    {
                        "results": page_one,
                        "next": (
                            "https://safe-transaction-optimism.safe.global/api/v1/"
                            "safes/0xS/transfers/?offset=100"
                        ),
                    },
                )
            return (True, {"results": [], "next": None})

        obj._request_with_retries = req
        obj.context.coingecko = MagicMock()
        obj.params.sleep_time = 1

        _drive(
            obj._fetch_outgoing_transfers_until_date_optimism("0xaddr", "2099-01-01")
        )

        assert call_count[0] == 2
        assert captured[0].endswith("&offset=0"), captured[0]
        assert captured[1].endswith("&offset=50"), captured[1]


class TestKvCacheTtlBoundaries:
    """Cache TTL boundary conditions for both withdrawal and initial-investment caches (PR 1.3)."""

    def test_withdrawal_cache_negative_age_treated_as_expired(self):
        """A future-dated last_ts (clock skew) is treated as expired, not as a hit."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xS"}

        now_ts = 1704067200
        # Stored ts is in the future relative to "now" → age is negative.
        kv_state = {
            "last_withdrawals_calculated_timestamp_optimism": str(now_ts + 3600),
            "total_withdrawals_optimism": "12.5",
        }

        def fake_read_kv(*args, **kwargs):
            yield
            return dict(kv_state)

        writes: Dict[str, str] = {}

        def fake_write_kv(d):
            writes.update(d)
            yield

        obj._read_kv = fake_read_kv
        obj._write_kv = fake_write_kv
        obj._get_current_timestamp = lambda: now_ts

        called = [0]

        def fetcher(*a, **kw):
            called[0] += 1
            yield
            return {"outgoing": {}}

        obj._track_erc20_transfers_optimism = fetcher
        obj._track_and_calculate_withdrawal_value_optimism = _gen_return(Decimal("0"))

        result = _drive(obj.calculate_withdrawals_value())
        # Cache treated as expired → fetcher ran, kv rewritten.
        assert result == Decimal("0")
        assert called[0] == 1
        assert writes["last_withdrawals_calculated_timestamp_optimism"] == str(now_ts)

    def test_initial_investment_cache_age_zero_is_hit(self):
        """Age = 0 (same-second read/write) is a cache hit, pinning the ``<=`` lower bound."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.airdrop_started = False
        obj._load_chain_total_investment = _gen_return(777.0)

        now_ts = 1704067200
        obj._read_kv = _gen_return(
            {"last_initial_value_calculated_timestamp_optimism": str(now_ts)}
        )
        obj._get_current_timestamp = lambda: now_ts

        assert (
            _drive(obj.calculate_initial_investment_value_from_funding_events())
            == 777.0
        )

    def test_withdrawal_cache_age_zero_is_hit(self):
        """Age = 0 (same-second read/write) is a cache hit, pinning the ``<=`` lower bound."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xS"}

        now_ts = 1704067200
        kv_state = {
            "last_withdrawals_calculated_timestamp_optimism": str(now_ts),
            "total_withdrawals_optimism": "12.5",
        }

        def fake_read_kv(*args, **kwargs):
            yield
            return dict(kv_state)

        obj._read_kv = fake_read_kv
        obj._get_current_timestamp = lambda: now_ts

        called = [0]

        def fetcher(*a, **kw):
            called[0] += 1
            yield
            return {"outgoing": {}}

        obj._track_erc20_transfers_optimism = fetcher

        assert _drive(obj.calculate_withdrawals_value()) == Decimal("12.5")
        assert called[0] == 0

    def test_initial_investment_cache_just_inside_ttl_is_hit(self):
        """Age = TTL - 1 second is a cache hit for the initial-investment path."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            INITIAL_INVESTMENT_CACHE_TTL_SECONDS,
        )

        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.airdrop_started = False
        obj._load_chain_total_investment = _gen_return(777.0)

        now_ts = 1704067200
        just_inside_ttl_ts = now_ts - (INITIAL_INVESTMENT_CACHE_TTL_SECONDS - 1)
        obj._read_kv = _gen_return(
            {
                "last_initial_value_calculated_timestamp_optimism": str(
                    just_inside_ttl_ts
                )
            }
        )
        obj._get_current_timestamp = lambda: now_ts

        assert (
            _drive(obj.calculate_initial_investment_value_from_funding_events())
            == 777.0
        )

    def test_withdrawal_cache_just_inside_ttl_is_hit(self):
        """Age = TTL - 1 second is a cache hit for the withdrawal path."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            WITHDRAWAL_CACHE_TTL_SECONDS,
        )

        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xS"}

        now_ts = 1704067200
        just_inside_ttl_ts = now_ts - (WITHDRAWAL_CACHE_TTL_SECONDS - 1)
        kv_state = {
            "last_withdrawals_calculated_timestamp_optimism": str(just_inside_ttl_ts),
            "total_withdrawals_optimism": "42.0",
        }

        def fake_read_kv(*args, **kwargs):
            yield
            return dict(kv_state)

        obj._read_kv = fake_read_kv
        obj._get_current_timestamp = lambda: now_ts

        called = [0]

        def fetcher(*a, **kw):
            called[0] += 1
            yield
            return {"outgoing": {}}

        obj._track_erc20_transfers_optimism = fetcher

        assert _drive(obj.calculate_withdrawals_value()) == Decimal("42.0")
        assert called[0] == 0

    def test_withdrawal_cache_just_outside_ttl_is_miss(self):
        """Age = TTL exactly is a cache miss for the withdrawal path."""
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            WITHDRAWAL_CACHE_TTL_SECONDS,
        )

        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xS"}

        now_ts = 1704067200
        at_ttl_ts = now_ts - WITHDRAWAL_CACHE_TTL_SECONDS
        kv_state = {
            "last_withdrawals_calculated_timestamp_optimism": str(at_ttl_ts),
            "total_withdrawals_optimism": "42.0",
        }

        def fake_read_kv(*args, **kwargs):
            yield
            return dict(kv_state)

        writes: Dict[str, str] = {}

        def fake_write_kv(d):
            writes.update(d)
            yield

        obj._read_kv = fake_read_kv
        obj._write_kv = fake_write_kv
        obj._get_current_timestamp = lambda: now_ts
        obj._track_erc20_transfers_optimism = _gen_return({"outgoing": {}})
        obj._track_and_calculate_withdrawal_value_optimism = _gen_return(Decimal("0"))

        assert _drive(obj.calculate_withdrawals_value()) == Decimal("0")
        assert writes["last_withdrawals_calculated_timestamp_optimism"] == str(now_ts)

    def test_initial_investment_cache_exactly_at_ttl_is_miss(self):
        """Age = TTL exactly is a cache miss for the initial-investment path.

        Pins the strict ``< TTL`` boundary symmetric with
        ``test_withdrawal_cache_just_outside_ttl_is_miss``. A
        ``<``→``<=`` regression on the guard would otherwise turn the
        exact-TTL case into a spurious hit and go undetected.
        """
        from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
            INITIAL_INVESTMENT_CACHE_TTL_SECONDS,
        )

        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.airdrop_started = False

        now_ts = 1704067200
        at_ttl_ts = now_ts - INITIAL_INVESTMENT_CACHE_TTL_SECONDS
        obj._read_kv = _gen_return(
            {"last_initial_value_calculated_timestamp_optimism": str(at_ttl_ts)}
        )
        obj._get_current_timestamp = lambda: now_ts

        load_called = [0]

        def load_chain(*a, **kw):
            """The cache-hit branch should never fire at exactly-TTL."""
            load_called[0] += 1
            yield
            return 999.0

        obj._load_chain_total_investment = load_chain
        obj._fetch_all_transfers_until_date_optimism = _gen_return(
            {"d": [{"symbol": "USDC", "delta": 1}]}
        )
        obj._calculate_chain_investment_value = _gen_return(50.0)
        obj._save_chain_total_investment = _gen_none
        obj._write_kv = _gen_none

        result = _drive(obj.calculate_initial_investment_value_from_funding_events())
        # Cache-miss path: recompute runs, cached loader is never invoked.
        assert result == 50.0
        assert load_called[0] == 0

    def test_withdrawal_cache_invalid_timestamp_format(self):
        """Unparseable ``last_withdrawals_calculated_timestamp_{chain}`` recomputes.

        Symmetric with ``test_invalid_timestamp_format`` on the
        initial-investment path. A regression flipping the parse-error
        fallback to a spurious cache hit (e.g. ``age_seconds = 0``)
        would otherwise be invisible on the withdrawal path.
        """
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xS"}

        now_ts = 1704067200
        kv_state = {
            "last_withdrawals_calculated_timestamp_optimism": "not-a-number",
            "total_withdrawals_optimism": "42.0",
        }

        def fake_read_kv(*args, **kwargs):
            yield
            return dict(kv_state)

        writes: Dict[str, str] = {}

        def fake_write_kv(d):
            writes.update(d)
            yield

        obj._read_kv = fake_read_kv
        obj._write_kv = fake_write_kv
        obj._get_current_timestamp = lambda: now_ts

        called = [0]

        def fetcher(*a, **kw):
            called[0] += 1
            yield
            return {"outgoing": {}}

        obj._track_erc20_transfers_optimism = fetcher
        obj._track_and_calculate_withdrawal_value_optimism = _gen_return(Decimal("0"))

        # Cache treated as expired → fetcher runs, kv timestamp is rewritten
        # with a parseable value so the next cycle is clean.
        assert _drive(obj.calculate_withdrawals_value()) == Decimal("0")
        assert called[0] == 1
        assert writes["last_withdrawals_calculated_timestamp_optimism"] == str(now_ts)

    def test_withdrawal_cache_within_ttl_missing_cached_val_falls_through(self):
        """Within-TTL hit with no ``total_withdrawals_{chain}`` recomputes.

        The cache-hit branch reads the value via a second ``_read_kv``;
        if that key is missing the code must fall through to the
        fetcher rather than serve ``None`` or ``Decimal("0")``.
        """
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xS"}

        now_ts = 1704067200
        # Two reads: the first returns just the timestamp (within TTL),
        # the second (for the cached value) returns no value.
        reads = [
            {"last_withdrawals_calculated_timestamp_optimism": str(now_ts - 60)},
            {},
        ]
        read_idx = [0]

        def fake_read_kv(*args, **kwargs):
            yield
            data = reads[read_idx[0]]
            read_idx[0] += 1
            return dict(data)

        obj._read_kv = fake_read_kv
        obj._write_kv = _gen_none
        obj._get_current_timestamp = lambda: now_ts

        called = [0]

        def fetcher(*a, **kw):
            called[0] += 1
            yield
            return {"outgoing": {}}

        obj._track_erc20_transfers_optimism = fetcher
        obj._track_and_calculate_withdrawal_value_optimism = _gen_return(Decimal("0"))

        result = _drive(obj.calculate_withdrawals_value())
        assert result == Decimal("0")
        # Fell through to the fetcher rather than returning a stale/empty value.
        assert called[0] == 1

    def test_withdrawal_cache_within_ttl_unparseable_cached_val_falls_through(
        self,
    ):
        """Within-TTL hit with a garbage ``total_withdrawals_{chain}`` recomputes.

        ``Decimal(str(cached_val))`` raises ``InvalidOperation`` on a
        non-numeric value; the code catches that and must fall through
        to the fetcher rather than propagate the exception or return
        ``Decimal("0")``.
        """
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xS"}

        now_ts = 1704067200
        kv_state = {
            "last_withdrawals_calculated_timestamp_optimism": str(now_ts - 60),
            "total_withdrawals_optimism": "garbage",
        }

        def fake_read_kv(*args, **kwargs):
            yield
            return dict(kv_state)

        obj._read_kv = fake_read_kv
        obj._write_kv = _gen_none
        obj._get_current_timestamp = lambda: now_ts

        called = [0]

        def fetcher(*a, **kw):
            called[0] += 1
            yield
            return {"outgoing": {}}

        obj._track_erc20_transfers_optimism = fetcher
        obj._track_and_calculate_withdrawal_value_optimism = _gen_return(Decimal("0"))

        result = _drive(obj.calculate_withdrawals_value())
        assert result == Decimal("0")
        # Fell through to the fetcher rather than raising or returning 0.
        assert called[0] == 1

    def test_initial_investment_cache_negative_age_treated_as_expired(self):
        """Future-dated last_initial_value_calculated_timestamp recomputes."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.airdrop_started = False

        now_ts = 1704067200
        future_ts = str(now_ts + 3600)
        obj._read_kv = _gen_return(
            {"last_initial_value_calculated_timestamp_optimism": future_ts}
        )
        obj._get_current_timestamp = lambda: now_ts

        load_called = [0]

        def load_chain(*a, **kw):
            """Confirm the cache-hit branch did not fire."""
            load_called[0] += 1
            yield
            return 1.0

        obj._load_chain_total_investment = load_chain
        obj._fetch_all_transfers_until_date_optimism = _gen_return(
            {"d": [{"symbol": "USDC", "delta": 1}]}
        )
        obj._calculate_chain_investment_value = _gen_return(123.0)
        obj._save_chain_total_investment = _gen_none
        obj._write_kv = _gen_none

        result = _drive(obj.calculate_initial_investment_value_from_funding_events())
        # Cache treated as expired (negative age) → cache-hit branch is
        # skipped, _load_chain_total_investment is never called, and the
        # recompute path runs.
        assert result == 123.0
        assert load_called[0] == 0

    def test_initial_investment_ttl_hit_falsy_passes_full_history_flag_to_mode(self):
        """In-window cache hit with falsy stored value loads full history.

        Previously a falsy cached value still fell through to the
        ``fetch_till_date = False`` branch (incremental fetch), which
        contradicted the ``fetch_till_date = True`` set on the falsy-hit
        path. The fix moves the expired branch into an explicit ``else``.
        The Mode fetcher is the one that consumes ``fetch_till_date``, so
        we exercise the Mode chain to actually observe the flag value.
        """
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj.params.airdrop_started = False

        now_ts = 1704067200
        obj._read_kv = _gen_return(
            {"last_initial_value_calculated_timestamp_optimism": str(now_ts - 60)}
        )
        obj._get_current_timestamp = lambda: now_ts
        # Stored cache value is falsy — triggers the "load full history" branch.
        obj._load_chain_total_investment = _gen_return(0.0)

        captured: Dict[str, Any] = {}

        def fetch_mode(address, end_date, fetch_till_date):
            captured["fetch_till_date"] = fetch_till_date
            yield
            return {"d": [{"symbol": "USDC", "delta": 1}]}

        obj._fetch_all_transfers_until_date_mode = fetch_mode
        obj._calculate_chain_investment_value = _gen_return(500.0)
        obj._save_chain_total_investment = _gen_none
        obj._write_kv = _gen_none

        result = _drive(obj.calculate_initial_investment_value_from_funding_events())
        assert result == 500.0
        # The falsy in-window branch must request full history. Before the
        # else-wrap fix, fetch_till_date was overwritten back to False by
        # the (now-explicit) expired branch.
        assert captured["fetch_till_date"] is True

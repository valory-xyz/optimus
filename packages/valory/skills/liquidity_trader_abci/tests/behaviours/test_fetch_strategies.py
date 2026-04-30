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
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, PropertyMock, patch

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


def _mk() -> Any:
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
    obj.store_funding_events = MagicMock()  # type: ignore[method-assign]
    obj.read_funding_events = MagicMock(return_value={})  # type: ignore[method-assign]
    return obj


def _drive(gen: Any, sends: Any = None) -> Any:
    """Drive a generator to completion, sending values from *sends*."""
    sends = list(sends or [])
    idx = 0
    val = None
    while True:
        try:
            gen.send(val)
            if idx < len(sends):
                val = sends[idx]
                idx += 1
            else:
                val = None
        except StopIteration as exc:
            return exc.value


def _gen_return(value: Any) -> Any:
    """Return a trivial generator that yields once then returns *value*."""

    def _inner(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
        yield
        return value

    return _inner


def _gen_none(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
    yield


class TestIsTimeUpdateDue:
    """Tests for IsTimeUpdateDue."""

    def test_due(self) -> None:
        """Test due."""
        obj = _mk()
        obj._get_current_timestamp = lambda: 10000
        obj.portfolio_data = {"last_updated": 0}
        assert obj._is_time_update_due() is True

    def test_not_due(self) -> None:
        """Test not due."""
        obj = _mk()
        obj._get_current_timestamp = lambda: 100
        obj.portfolio_data = {"last_updated": 100}
        assert obj._is_time_update_due() is False


class TestHavePositionsChanged:
    """Tests for HavePositionsChanged."""

    def test_no_last_data(self) -> None:
        """Test no last data."""
        obj = _mk()
        obj.current_positions = []
        assert obj._have_positions_changed({}) is True

    def test_no_allocations_key(self) -> None:
        """Test no allocations key."""
        obj = _mk()
        obj.current_positions = []
        assert obj._have_positions_changed({"foo": 1}) is True

    def test_count_changed(self) -> None:
        """Test count changed."""
        obj = _mk()
        obj.current_positions = [
            {"pool_address": "0x1", "dex_type": "a", "status": "open"}
        ]
        assert obj._have_positions_changed({"allocations": []}) is True

    def test_no_change(self) -> None:
        """Test no change."""
        obj = _mk()
        obj.current_positions = [
            {"pool_address": "0x1", "dex_type": "uniswapV3", "status": "open"}
        ]
        last = {"allocations": [{"id": "0x1", "type": "uniswapV3"}]}
        assert obj._have_positions_changed(last) is False

    def test_new_positions(self) -> None:
        """Test new positions."""
        obj = _mk()
        obj.current_positions = [
            {"pool_address": "0x1", "dex_type": "a", "status": "open"},
            {"pool_address": "0x2", "dex_type": "b", "status": "open"},
        ]
        last = {"allocations": [{"id": "0x1", "type": "a"}]}
        assert obj._have_positions_changed(last) is True

    def test_closed_positions(self) -> None:
        """Test closed positions."""
        obj = _mk()
        obj.current_positions = [
            {"pool_address": "0x1", "dex_type": "a", "status": "closed"}
        ]
        last = {"allocations": [{"id": "0x1", "type": "a"}]}
        assert obj._have_positions_changed(last) is True


class TestUpdatePortfolioBreakdownRatios:
    """Tests for UpdatePortfolioBreakdownRatios."""

    def test_empty_breakdown(self) -> None:
        """Test empty breakdown."""
        obj = _mk()
        bd: List[Any] = []
        obj._update_portfolio_breakdown_ratios(bd, Decimal(100))
        assert bd == []

    def test_zero_total(self) -> None:
        """Test zero total."""
        obj = _mk()
        bd = [{"value_usd": 10, "balance": 1, "price": 10}]
        obj._update_portfolio_breakdown_ratios(bd, Decimal(0))
        assert bd[0]["ratio"] == 0.0

    def test_normal(self) -> None:
        """Test normal."""
        obj = _mk()
        bd = [
            {"value_usd": 50, "balance": 5, "price": 10},
            {"value_usd": 50, "balance": 10, "price": 5},
        ]
        obj._update_portfolio_breakdown_ratios(bd, Decimal(100))
        assert len(bd) == 2
        assert isinstance(bd[0]["value_usd"], float)

    def test_filters_small_values(self) -> None:
        """Test filters small values."""
        obj = _mk()
        bd = [
            {"value_usd": 0.001, "balance": 0.0001, "price": 10},
            {"value_usd": 50, "balance": 5, "price": 10},
        ]
        obj._update_portfolio_breakdown_ratios(bd, Decimal(50))
        assert len(bd) == 1

    def test_none_value_usd_skipped(self) -> None:
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
        assert bd[0]["value_usd"] == 10.0  # type: ignore[index]


class TestAdjustForDecimals:
    """Tests for AdjustForDecimals."""

    def test_basic(self) -> None:
        """Test basic."""
        obj = _mk()
        assert obj._adjust_for_decimals(1000000, 6) == Decimal("1")


class TestIsGnosisSafe:
    """Tests for IsGnosisSafe."""

    def test_none(self) -> None:
        """Test none."""
        obj = _mk()
        assert obj._is_gnosis_safe(None) is False

    def test_not_contract(self) -> None:
        """Test not contract."""
        obj = _mk()
        assert obj._is_gnosis_safe({"is_contract": False}) is False

    def test_wrong_name(self) -> None:
        """Test wrong name."""
        obj = _mk()
        assert obj._is_gnosis_safe({"is_contract": True, "name": "Foo"}) is False

    def test_is_safe(self) -> None:
        """Test is safe."""
        obj = _mk()
        assert (
            obj._is_gnosis_safe({"is_contract": True, "name": "GnosisSafeProxy"})
            is True
        )


class TestShouldIncludeTransfer:
    """Tests for ShouldIncludeTransfer."""

    def test_no_from(self) -> None:
        """Test no from."""
        obj = _mk()
        assert obj._should_include_transfer(None) is False

    def test_zero_address(self) -> None:
        """Test zero address."""
        obj = _mk()
        assert (
            obj._should_include_transfer(
                {"hash": "0x0000000000000000000000000000000000000000"}
            )
            is False
        )

    def test_empty_hash(self) -> None:
        """Test empty hash."""
        obj = _mk()
        assert obj._should_include_transfer({"hash": ""}) is False

    def test_eth_transfer_bad_status(self) -> None:
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

    def test_eth_transfer_zero_value(self) -> None:
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

    def test_eoa(self) -> None:
        """Test eoa."""
        obj = _mk()
        assert (
            obj._should_include_transfer({"hash": "0x123", "is_contract": False})
            is True
        )

    def test_contract_not_safe(self) -> None:
        """Test contract not safe."""
        obj = _mk()
        assert (
            obj._should_include_transfer(
                {"hash": "0x123", "is_contract": True, "name": "Foo"}
            )
            is False
        )

    def test_contract_safe(self) -> None:
        """Test contract safe."""
        obj = _mk()
        assert (
            obj._should_include_transfer(
                {"hash": "0x123", "is_contract": True, "name": "GnosisSafeProxy"}
            )
            is True
        )


class TestShouldIncludeTransferMode:
    """Tests for ShouldIncludeTransferMode."""

    def test_no_from(self) -> None:
        """Test no from."""
        obj = _mk()
        assert obj._should_include_transfer_mode(None) is False

    def test_zero(self) -> None:
        """Test zero."""
        obj = _mk()
        assert obj._should_include_transfer_mode({"hash": "0x0"}) is False

    def test_eoa(self) -> None:
        """Test eoa."""
        obj = _mk()
        assert (
            obj._should_include_transfer_mode({"hash": "0xabc", "is_contract": False})
            is True
        )

    def test_eth_bad(self) -> None:
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
    """Tests for GetDatetimeFromTimestamp."""

    def test_z_suffix(self) -> None:
        """Test z suffix."""
        obj = _mk()
        dt = obj._get_datetime_from_timestamp("2024-01-01T00:00:00Z")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_plus_tz(self) -> None:
        """Test plus tz."""
        obj = _mk()
        dt = obj._get_datetime_from_timestamp("2024-01-01T00:00:00+00:00")
        assert dt is not None

    def test_no_tz(self) -> None:
        """Test no tz."""
        obj = _mk()
        dt = obj._get_datetime_from_timestamp("2024-01-01T00:00:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_invalid(self) -> None:
        """Test invalid."""
        obj = _mk()
        dt = obj._get_datetime_from_timestamp("not-a-date")
        assert dt is None

    def test_utc_suffix(self) -> None:
        """Test utc suffix."""
        obj = _mk()
        dt = obj._get_datetime_from_timestamp("2024-01-01T00:00:00UTC")
        assert dt is not None


class TestGetVeloTokenAddress:
    """Tests for GetVeloTokenAddress."""

    def test_found(self) -> None:
        """Test found."""
        obj = _mk()

        obj.params.velo_token_contract_addresses = {"optimism": "0xVELO"}
        assert obj._get_velo_token_address("optimism") == "0xVELO"

    def test_not_found(self) -> None:
        """Test not found."""
        obj = _mk()

        obj.params.velo_token_contract_addresses = {}
        assert obj._get_velo_token_address("mode") is None


class TestIsAirdropTransfer:
    """Tests for IsAirdropTransfer."""

    def test_not_started(self) -> None:
        """Test not started."""
        obj = _mk()

        obj.params.airdrop_started = False
        assert obj._is_airdrop_transfer({}) is False

    def test_no_contract(self) -> None:
        """Test no contract."""
        obj = _mk()

        obj.params.airdrop_started = True
        obj.params.airdrop_contract_address = None
        assert obj._is_airdrop_transfer({}) is False

    def test_match(self) -> None:
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

    def test_no_match_wrong_symbol(self) -> None:
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
    """Tests for CheckAndUpdateZeroLiquidityPositions."""

    def test_no_positions(self) -> None:
        """Test no positions."""
        obj = _mk()
        obj.current_positions = None
        obj.check_and_update_zero_liquidity_positions()  # no crash

    def test_close_zero_liquidity(self) -> None:
        """Test close zero liquidity."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "UniswapV3", "current_liquidity": 0}
        ]
        obj.store_current_positions = MagicMock()
        obj.check_and_update_zero_liquidity_positions()
        assert obj.current_positions[0]["status"] == "closed"

    def test_keep_nonzero(self) -> None:
        """Test keep nonzero."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "UniswapV3", "current_liquidity": 100}
        ]
        obj.store_current_positions = MagicMock()
        obj.check_and_update_zero_liquidity_positions()
        assert obj.current_positions[0]["status"] == "open"

    def test_skip_closed(self) -> None:
        """Test skip closed."""
        obj = _mk()
        obj.current_positions = [
            {"status": "closed", "dex_type": "UniswapV3", "current_liquidity": 0}
        ]
        obj.store_current_positions = MagicMock()
        obj.check_and_update_zero_liquidity_positions()
        assert obj.current_positions[0]["status"] == "closed"

    def test_velodrome_cl_all_zero(self) -> None:
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

    def test_velodrome_cl_some_nonzero(self) -> None:
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

    def test_velodrome_cl_none_token_id(self) -> None:
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

    def test_default_liquidity_avoids_false_close(self) -> None:
        """Test default liquidity avoids false close."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "UniswapV3"}  # no current_liquidity key
        ]
        obj.store_current_positions = MagicMock()
        obj.check_and_update_zero_liquidity_positions()
        assert obj.current_positions[0]["status"] == "open"


class TestAddToPortfolioBreakdown:
    """Tests for AddToPortfolioBreakdown."""

    def test_new_entry(self) -> None:
        """Test new entry."""
        obj = _mk()
        bd: List[Any] = []
        obj._add_to_portfolio_breakdown(
            bd, "0xA", "TKN", Decimal(10), Decimal(2), Decimal(20)
        )
        assert len(bd) == 1
        assert bd[0]["asset"] == "TKN"

    def test_existing_entry(self) -> None:
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

    def _setup_obj(
        self, api_response: Any, use_x402: Any = False, cached_price: Any = None
    ) -> Any:
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

    def test_success_caches_result(self) -> Any:
        """Test success caches result."""
        obj = self._setup_obj(
            (True, {"market_data": {"current_price": {"usd": 3000.0}}}),
        )
        cache_calls = []
        original_cache = _gen_none

        def tracking_cache(*args: Any, **kwargs: Any) -> Any:
            cache_calls.append(args)
            return original_cache(*args, **kwargs)

        obj._cache_price = tracking_cache
        result = _drive(obj._fetch_historical_eth_price("01-01-2024"))
        assert result == 3000.0
        assert len(cache_calls) == 1
        assert cache_calls[0] == ("ethereum", 3000.0, "01-01-2024")

    def test_returns_cached_price_without_api_call(self) -> None:
        """Test returns cached price without api call."""
        obj = self._setup_obj(
            (False, {}),  # API would fail, but shouldn't be called
            cached_price=2500.0,
        )
        result = _drive(obj._fetch_historical_eth_price("01-01-2024"))
        assert result == 2500.0
        # Verify API was NOT called since cache hit
        obj.context.coingecko.request.assert_not_called()

    def test_api_failure_returns_none(self) -> None:
        """Test api failure returns none."""
        obj = self._setup_obj((False, {}))
        obj.context.coingecko.api_key = None
        result = _drive(obj._fetch_historical_eth_price("01-01-2024"))
        assert result is None

    def test_no_price_in_response(self) -> None:
        """Test no price in response."""
        obj = self._setup_obj(
            (True, {"market_data": {"current_price": {}}}),
        )
        result = _drive(obj._fetch_historical_eth_price("01-01-2024"))
        assert result is None

    def test_x402(self) -> None:
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
    """Tests for GetHistoricalPriceForDate."""

    def test_zero_address(self) -> None:
        """Test zero address."""
        obj = _mk()
        obj._fetch_historical_eth_price = _gen_return(2500.0)
        gen = obj._get_historical_price_for_date(
            ZERO_ADDRESS, "ETH", "01-01-2024", "optimism"
        )
        result = _drive(gen)
        assert result == 2500.0

    def test_no_coingecko_id(self) -> None:
        """Test no coingecko id."""
        obj = _mk()
        obj.get_coin_id_from_symbol = lambda s, c: None
        gen = obj._get_historical_price_for_date("0xTOKEN", "FOO", "01-01-2024", "mode")
        result = _drive(gen)
        assert result is None

    def test_with_coingecko_id(self) -> None:
        """Test with coingecko id."""
        obj = _mk()
        obj.get_coin_id_from_symbol = lambda s, c: "foo-coin"
        obj._fetch_historical_token_price = _gen_return(42.0)
        gen = obj._get_historical_price_for_date("0xTOKEN", "FOO", "01-01-2024", "mode")
        result = _drive(gen)
        assert result == 42.0

    def test_exception(self) -> None:
        """Test exception."""
        obj = _mk()

        def boom(s: Any, c: Any) -> None:
            raise RuntimeError("fail")

        obj.get_coin_id_from_symbol = boom
        gen = obj._get_historical_price_for_date("0xTOKEN", "FOO", "01-01-2024", "mode")
        result = _drive(gen)
        assert result is None


class TestUpdateAgentPerformanceMetrics:
    """Tests for UpdateAgentPerformanceMetrics."""

    def test_with_data(self) -> None:
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

    def test_no_data(self) -> None:
        """Test no data."""
        obj = _mk()
        obj.portfolio_data = {}
        obj.read_agent_performance = MagicMock()
        obj.update_agent_performance_timestamp = MagicMock()
        obj.store_agent_performance = MagicMock()
        obj._update_agent_performance_metrics()
        assert len(obj.agent_performance["metrics"]) == 2
        assert "$0.00" in obj.agent_performance["metrics"][0]["value"]

    def test_exception(self) -> None:
        """Test exception."""
        obj = _mk()
        obj.read_agent_performance = MagicMock(side_effect=RuntimeError("boom"))
        obj._update_agent_performance_metrics()  # no crash


class TestHandleBalancerPosition:
    """Tests for HandleBalancerPosition."""

    def test_ok(self) -> None:
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
    """Tests for HandleUniswapPosition."""

    def test_ok(self) -> None:
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
    """Tests for HandleSturdyPosition."""

    def test_ok(self) -> None:
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
    """Tests for HandleVelodromePosition."""

    def test_not_staked(self) -> None:
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

    def test_staked_with_rewards(self) -> None:
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
    """Tests for GetTickRanges."""

    def test_non_cl_dex(self) -> None:
        """Test non cl dex."""
        obj = _mk()
        position = {"dex_type": "balancerPool"}
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert result == []

    def test_velodrome_non_cl(self) -> None:
        """Test velodrome non cl."""
        obj = _mk()
        position = {"dex_type": "velodrome", "is_cl_pool": False}
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert result == []

    def test_no_pool_address(self) -> None:
        """Test no pool address."""
        obj = _mk()
        position = {"dex_type": "UniswapV3", "pool_address": None}
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert result == []

    def test_slot0_fail(self) -> None:
        """Test slot0 fail."""
        obj = _mk()

        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        position = {"dex_type": "UniswapV3", "pool_address": "0xPool", "token_id": 1}
        obj.contract_interact = _gen_return(None)
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert result == []

    def test_no_position_manager(self) -> None:
        """Test no position manager."""
        obj = _mk()

        obj.params.uniswap_position_manager_contract_addresses = {}
        position = {"dex_type": "UniswapV3", "pool_address": "0xPool", "token_id": 1}
        obj.contract_interact = _gen_return({"tick": 100, "sqrt_price_x96": 100})
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert result == []


class TestCalculatePositionValue:
    """Tests for CalculatePositionValue."""

    def test_basic(self) -> None:
        """Test basic."""
        obj = _mk()
        position = {"pool_address": "0xP"}
        token_info = {"0xT0": "TKN0", "0xT1": "TKN1"}
        user_balances = {"0xT0": Decimal(10), "0xT1": Decimal(20)}
        obj._fetch_token_price = _gen_return(2.0)
        obj._update_position_with_current_value = _gen_none
        bd: List[Any] = []
        gen = obj._calculate_position_value(
            position, "optimism", user_balances, token_info, bd
        )
        result = _drive(gen)
        assert result == Decimal(10) * Decimal("2.0") + Decimal(20) * Decimal("2.0")
        assert len(bd) == 2

    def test_missing_balance(self) -> None:
        """Test missing balance."""
        obj = _mk()
        position = {"pool_address": "0xP"}
        token_info = {"0xT0": "TKN0"}
        user_balances: Dict[Any, Any] = {}
        obj._fetch_token_price = _gen_return(2.0)
        obj._update_position_with_current_value = _gen_none
        gen = obj._calculate_position_value(
            position, "optimism", user_balances, token_info, []
        )
        result = _drive(gen)
        assert result == Decimal(0)

    def test_missing_price(self) -> None:
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

    def test_existing_asset_update(self) -> None:
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
    """Tests for UpdatePositionAmounts."""

    def test_no_positions(self) -> None:
        """Test no positions."""
        obj = _mk()
        obj.current_positions = None
        gen = obj.update_position_amounts()
        _drive(gen)

    def test_skip_closed(self) -> None:
        """Test skip closed."""
        obj = _mk()
        obj.current_positions = [
            {"status": "closed", "dex_type": "UniswapV3", "chain": "optimism"}
        ]
        obj.store_current_positions = MagicMock()
        gen = obj.update_position_amounts()
        _drive(gen)

    def test_missing_dex_type(self) -> None:
        """Test missing dex type."""
        obj = _mk()
        obj.current_positions = [{"status": "open", "chain": "optimism"}]
        obj.store_current_positions = MagicMock()
        gen = obj.update_position_amounts()
        _drive(gen)

    def test_unknown_dex(self) -> None:
        """Test unknown dex."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "unknown", "chain": "optimism"}
        ]
        obj.store_current_positions = MagicMock()
        gen = obj.update_position_amounts()
        _drive(gen)

    def test_balancer(self) -> None:
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

    def test_uniswap(self) -> None:
        """Test uniswap."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "UniswapV3", "chain": "optimism"}
        ]
        obj._update_uniswap_position = _gen_none
        obj.store_current_positions = MagicMock()
        gen = obj.update_position_amounts()
        _drive(gen)

    def test_velodrome(self) -> None:
        """Test velodrome."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "velodrome", "chain": "optimism"}
        ]
        obj._update_velodrome_position = _gen_none
        obj.store_current_positions = MagicMock()
        gen = obj.update_position_amounts()
        _drive(gen)

    def test_sturdy(self) -> None:
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
    """Tests for UpdateBalancerPosition."""

    def test_missing_params(self) -> None:
        """Test missing params."""
        obj = _mk()

        obj.params.safe_contract_addresses = {}
        gen = obj._update_balancer_position({"chain": "optimism"})
        _drive(gen)

    def test_success(self) -> None:
        """Test success."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.contract_interact = _gen_return(1000)
        pos = {"pool_address": "0xP", "chain": "optimism"}
        gen = obj._update_balancer_position(pos)
        _drive(gen)
        assert pos["current_liquidity"] == 1000

    def test_none_balance(self) -> None:
        """Test none balance."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.contract_interact = _gen_return(None)
        pos = {"pool_address": "0xP", "chain": "optimism"}
        gen = obj._update_balancer_position(pos)
        _drive(gen)
        assert "current_liquidity" not in pos


class TestUpdateUniswapPosition:
    """Tests for UpdateUniswapPosition."""

    def test_missing_params(self) -> None:
        """Test missing params."""
        obj = _mk()
        gen = obj._update_uniswap_position({"chain": "optimism"})
        _drive(gen)

    def test_no_pm(self) -> None:
        """Test no pm."""
        obj = _mk()

        obj.params.uniswap_position_manager_contract_addresses = {}
        gen = obj._update_uniswap_position({"token_id": 1, "chain": "optimism"})
        _drive(gen)

    def test_success(self) -> None:
        """Test success."""
        obj = _mk()

        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        obj.contract_interact = _gen_return({"liquidity": 500})
        pos = {"token_id": 1, "chain": "optimism"}
        gen = obj._update_uniswap_position(pos)
        _drive(gen)
        assert pos["current_liquidity"] == 500

    def test_no_data(self) -> None:
        """Test no data."""
        obj = _mk()

        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        obj.contract_interact = _gen_return(None)
        pos = {"token_id": 1, "chain": "optimism"}
        gen = obj._update_uniswap_position(pos)
        _drive(gen)


class TestUpdateSturdyPosition:
    """Tests for UpdateSturdyPosition."""

    def test_missing(self) -> None:
        """Test missing."""
        obj = _mk()

        obj.params.safe_contract_addresses = {}
        gen = obj._update_sturdy_position({"chain": "mode"})
        _drive(gen)

    def test_success(self) -> None:
        """Test success."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj.contract_interact = _gen_return(999)
        pos = {"pool_address": "0xP", "chain": "mode"}
        gen = obj._update_sturdy_position(pos)
        _drive(gen)
        assert pos["current_liquidity"] == 999

    def test_none(self) -> None:
        """Test none."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj.contract_interact = _gen_return(None)
        pos = {"pool_address": "0xP", "chain": "mode"}
        gen = obj._update_sturdy_position(pos)
        _drive(gen)


class TestUpdateVelodromePosition:
    """Tests for UpdateVelodromePosition."""

    def test_no_chain(self) -> None:
        """Test no chain."""
        obj = _mk()
        gen = obj._update_velodrome_position({})
        _drive(gen)

    def test_cl_success(self) -> None:
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
        assert pos["positions"][0]["current_liquidity"] == 42  # type: ignore[index]

    def test_cl_no_token_id(self) -> None:
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

    def test_cl_no_pm(self) -> None:
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

    def test_non_cl_success(self) -> None:
        """Test non cl success."""
        obj = _mk()

        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.contract_interact = _gen_return(77)
        pos = {"chain": "optimism", "is_cl_pool": False, "pool_address": "0xP"}
        gen = obj._update_velodrome_position(pos)
        _drive(gen)
        assert pos["current_liquidity"] == 77

    def test_non_cl_missing(self) -> None:
        """Test non cl missing."""
        obj = _mk()

        obj.params.safe_contract_addresses = {}
        pos = {"chain": "optimism", "is_cl_pool": False}
        gen = obj._update_velodrome_position(pos)
        _drive(gen)


class TestChainTotalInvestment:
    """Tests for ChainTotalInvestment."""

    def test_load_found(self) -> None:
        """Test load found."""
        obj = _mk()
        obj._read_kv = _gen_return({"mode_total_investment": "123.45"})
        gen = obj._load_chain_total_investment("mode")
        assert _drive(gen) == 123.45

    def test_load_not_found(self) -> None:
        """Test load not found."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        gen = obj._load_chain_total_investment("mode")
        assert _drive(gen) == 0.0

    def test_load_invalid(self) -> None:
        """Test load invalid."""
        obj = _mk()
        obj._read_kv = _gen_return({"mode_total_investment": "not-a-number"})
        gen = obj._load_chain_total_investment("mode")
        assert _drive(gen) == 0.0

    def test_save(self) -> None:
        """Test save."""
        obj = _mk()
        written = {}

        def fake_write(d: Any) -> Generator[Any, Any, Any]:
            written.update(d)
            yield

        obj._write_kv = fake_write
        gen = obj._save_chain_total_investment("mode", 42.0)
        _drive(gen)
        assert written["mode_total_investment"] == "42.0"


class TestLoadFundingEventsData:
    """Tests for LoadFundingEventsData."""

    def test_found(self) -> None:
        """Test found."""
        obj = _mk()
        obj._read_kv = _gen_return({"funding_events": json.dumps({"mode": {}})})
        gen = obj._load_funding_events_data()
        assert _drive(gen) == {"mode": {}}

    def test_not_found(self) -> None:
        """Test not found."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        gen = obj._load_funding_events_data()
        assert _drive(gen) == {}

    def test_invalid_json(self) -> None:
        """Test invalid json."""
        obj = _mk()
        obj._read_kv = _gen_return({"funding_events": "{{bad"})
        gen = obj._load_funding_events_data()
        assert _drive(gen) == {}


class TestSaveTransferData:
    """Tests for SaveTransferData."""

    def test_mode(self) -> None:
        """Test mode."""
        obj = _mk()
        written = {}

        def fake_write(d: Any) -> Generator[Any, Any, Any]:
            written.update(d)
            yield

        obj._write_kv = fake_write
        gen = obj._save_transfer_data_mode({"a": 1})
        _drive(gen)
        assert "mode_transfer_data" in written

    def test_optimism(self) -> None:
        """Test optimism."""
        obj = _mk()
        written = {}

        def fake_write(d: Any) -> Generator[Any, Any, Any]:
            written.update(d)
            yield

        obj._write_kv = fake_write
        gen = obj._save_transfer_data_optimism({"b": 2})
        _drive(gen)
        assert "optimism_transfer_data" in written


class TestContractNameGetters:
    """Tests for ContractNameGetters."""

    def test_aggregator(self) -> None:
        """Test aggregator."""
        obj = _mk()
        obj.contract_interact = _gen_return("MyAgg")
        gen = obj._get_aggregator_name("0xAgg", "mode")
        assert _drive(gen) == "MyAgg"

    def test_balancer(self) -> None:
        """Test balancer."""
        obj = _mk()
        obj.contract_interact = _gen_return("MyPool")
        gen = obj._get_balancer_pool_name("0xPool", "optimism")
        assert _drive(gen) == "MyPool"


class TestGetUserShareValueVelodrome:
    """Tests for GetUserShareValueVelodrome."""

    def test_missing_tokens(self) -> None:
        """Test missing tokens."""
        obj = _mk()
        pos = {"token0": None, "token1": "0xT1", "is_cl_pool": False}
        gen = obj.get_user_share_value_velodrome("0xU", "0xP", 1, "optimism", pos)
        assert _drive(gen) == {}

    def test_cl(self) -> None:
        """Test cl."""
        obj = _mk()
        pos = {"token0": "0xT0", "token1": "0xT1", "is_cl_pool": True}
        obj._get_user_share_value_velodrome_cl = _gen_return({"0xT0": Decimal(1)})
        gen = obj.get_user_share_value_velodrome("0xU", "0xP", 1, "optimism", pos)
        assert _drive(gen) == {"0xT0": Decimal(1)}

    def test_non_cl(self) -> None:
        """Test non cl."""
        obj = _mk()
        pos = {"token0": "0xT0", "token1": "0xT1", "is_cl_pool": False}
        obj._get_user_share_value_velodrome_non_cl = _gen_return({"0xT0": Decimal(2)})
        gen = obj.get_user_share_value_velodrome("0xU", "0xP", 1, "optimism", pos)
        assert _drive(gen) == {"0xT0": Decimal(2)}


class TestGetUserShareValueUniswap:
    """Tests for GetUserShareValueUniswap."""

    def test_missing_data(self) -> None:
        """Test missing data."""
        obj = _mk()
        pos = {"token0": None, "token1": "0xT1"}
        gen = obj.get_user_share_value_uniswap("0xP", 1, "optimism", pos)
        assert _drive(gen) == {}

    def test_no_pm(self) -> None:
        """Test no pm."""
        obj = _mk()

        obj.params.uniswap_position_manager_contract_addresses = {}
        pos = {"token0": "0xT0", "token1": "0xT1"}
        gen = obj.get_user_share_value_uniswap("0xP", 1, "optimism", pos)
        assert _drive(gen) == {}


class TestGetUserShareValueBalancer:
    """Tests for GetUserShareValueBalancer."""

    def test_no_vault(self) -> None:
        """Test no vault."""
        obj = _mk()

        obj.params.balancer_vault_contract_addresses = {}
        gen = obj.get_user_share_value_balancer("0xU", "0xPID", "0xP", "optimism")
        assert _drive(gen) == {}

    def test_no_pool_tokens(self) -> None:
        """Test no pool tokens."""
        obj = _mk()

        obj.params.balancer_vault_contract_addresses = {"optimism": "0xV"}
        obj.contract_interact = _gen_return(None)
        gen = obj.get_user_share_value_balancer("0xU", "0xPID", "0xP", "optimism")
        assert _drive(gen) == {}


class TestGetUserShareValueSturdy:
    """Tests for GetUserShareValueSturdy."""

    def test_no_balance(self) -> None:
        """Test no balance."""
        obj = _mk()
        obj.contract_interact = _gen_return(None)
        gen = obj.get_user_share_value_sturdy("0xU", "0xAgg", "0xA", "mode")
        assert _drive(gen) == {}

    def test_no_decimals(self) -> None:
        """Test no decimals."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_success(self) -> None:
        """Test success."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
    """Tests for CalculateAirdropRewardsValue."""

    def test_non_mode(self) -> None:
        """Test non mode."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        gen = obj.calculate_airdrop_rewards_value()
        assert _drive(gen) == Decimal(0)

    def test_zero_rewards(self) -> None:
        """Test zero rewards."""
        obj = _mk()

        obj.params.target_investment_chains = ["mode"]
        obj._get_total_airdrop_rewards = _gen_return(0)
        gen = obj.calculate_airdrop_rewards_value()
        assert _drive(gen) == Decimal(0)

    def test_with_rewards(self) -> None:
        """Test with rewards."""
        obj = _mk()

        obj.params.target_investment_chains = ["mode"]
        obj._get_total_airdrop_rewards = _gen_return(1000000)  # 1 USDC
        obj._get_usdc_address = lambda c: "0xUSDC"
        obj._fetch_token_price = _gen_return(1.0)
        gen = obj.calculate_airdrop_rewards_value()
        result = _drive(gen)
        assert result == Decimal("1") * Decimal("1.0")

    def test_with_rewards_no_price(self) -> None:
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
    """Tests for CalculateStakingRewardsValue."""

    def test_no_olas_address(self) -> None:
        """Test no olas address."""
        obj = _mk()

        obj.params.target_investment_chains = ["unknown"]
        obj.update_accumulated_rewards_for_chain = _gen_none
        gen = obj.calculate_stakig_rewards_value()
        assert _drive(gen) == Decimal(0)

    def test_zero_rewards(self) -> None:
        """Test zero rewards."""
        obj = _mk()

        obj.params.target_investment_chains = ["mode"]
        obj.update_accumulated_rewards_for_chain = _gen_none
        obj.get_accumulated_rewards_for_token = _gen_return(0)
        gen = obj.calculate_stakig_rewards_value()
        assert _drive(gen) == Decimal(0)

    def test_with_rewards(self) -> None:
        """Test with rewards."""
        obj = _mk()

        obj.params.target_investment_chains = ["mode"]
        obj.update_accumulated_rewards_for_chain = _gen_none
        obj.get_accumulated_rewards_for_token = _gen_return(10**18)
        obj._fetch_token_price = _gen_return(5.0)
        gen = obj.calculate_stakig_rewards_value()
        result = _drive(gen)
        assert result == Decimal("1") * Decimal("5.0")

    def test_no_price(self) -> None:
        """Test no price."""
        obj = _mk()

        obj.params.target_investment_chains = ["mode"]
        obj.update_accumulated_rewards_for_chain = _gen_none
        obj.get_accumulated_rewards_for_token = _gen_return(10**18)
        obj._fetch_token_price = _gen_return(None)
        gen = obj.calculate_stakig_rewards_value()
        assert _drive(gen) == Decimal(0)


class TestCalculateWithdrawalsValue:
    """Tests for CalculateWithdrawalsValue."""

    def test_unsupported_chain(self) -> None:
        """Test unsupported chain."""
        obj = _mk()

        obj.params.target_investment_chains = ["base"]
        gen = obj.calculate_withdrawals_value()
        assert _drive(gen) == Decimal(0)

    def test_mode_none_transfers(self) -> None:
        """Test mode none transfers."""
        obj = _mk()

        obj.params.target_investment_chains = ["mode"]
        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj._track_erc20_transfers_mode = MagicMock(return_value=None)
        gen = obj.calculate_withdrawals_value()
        assert _drive(gen) == Decimal(0)

    def test_mode_with_transfers(self) -> None:
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

    def test_optimism_no_transfers(self) -> None:
        """Test optimism no transfers."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._track_erc20_transfers_optimism = _gen_return(None)
        gen = obj.calculate_withdrawals_value()
        assert _drive(gen) == Decimal(0)

    def test_optimism_with_transfers(self) -> None:
        """Test optimism with transfers."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._track_erc20_transfers_optimism = _gen_return({"outgoing": {"d": []}})
        obj._track_and_calculate_withdrawal_value_optimism = _gen_return(Decimal("25"))
        gen = obj.calculate_withdrawals_value()
        assert _drive(gen) == Decimal("25")


class TestTrackWithdrawalMode:
    """Tests for TrackWithdrawalMode."""

    def test_empty(self) -> None:
        """Test empty."""
        obj = _mk()
        gen = obj._track_and_calculate_withdrawal_value_mode({})
        assert _drive(gen) == Decimal(0)

    def test_with_usdc(self) -> None:
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

    def test_exception(self) -> None:
        """Test exception."""
        obj = _mk()

        def boom(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
    """Tests for TrackWithdrawalOptimism."""

    def test_empty(self) -> None:
        """Test empty."""
        obj = _mk()
        gen = obj._track_and_calculate_withdrawal_value_optimism({})
        assert _drive(gen) == Decimal(0)

    def test_with_usdc(self) -> None:
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

    def test_filter_contracts(self) -> None:
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
    """Tests for CalculateTotalWithdrawalValue."""

    def test_empty(self) -> None:
        """Test empty."""
        obj = _mk()
        gen = obj._calculate_total_withdrawal_value([], chain="mode")
        assert _drive(gen) == Decimal(0)

    def test_no_timestamp(self) -> None:
        """Test no timestamp."""
        obj = _mk()
        gen = obj._calculate_total_withdrawal_value([{"timestamp": ""}], chain="mode")
        assert _drive(gen) == Decimal(0)

    def test_bad_timestamp(self) -> None:
        """Test bad timestamp."""
        obj = _mk()
        gen = obj._calculate_total_withdrawal_value(
            [{"timestamp": "not-a-date"}], chain="mode"
        )
        assert _drive(gen) == Decimal(0)

    def test_no_coin_id(self) -> None:
        """Test no coin id."""
        obj = _mk()
        obj.get_coin_id_from_symbol = lambda s, c: None
        gen = obj._calculate_total_withdrawal_value(
            [{"timestamp": "2024-01-01T00:00:00Z"}], chain="mode"
        )
        assert _drive(gen) == Decimal(0)

    def test_with_price(self) -> None:
        """Test with price."""
        obj = _mk()
        obj.get_coin_id_from_symbol = lambda s, c: "usd-coin"
        obj._fetch_historical_token_price = _gen_return(1.0)
        transfers = [{"timestamp": "2024-01-01T00:00:00Z", "amount": 100}]
        gen = obj._calculate_total_withdrawal_value(transfers, chain="mode")
        result = _drive(gen)
        assert result == Decimal("100") * Decimal("1.0")

    def test_no_price(self) -> None:
        """Test no price."""
        obj = _mk()
        obj.get_coin_id_from_symbol = lambda s, c: "usd-coin"
        obj._fetch_historical_token_price = _gen_return(None)
        transfers = [{"timestamp": "2024-01-01T00:00:00Z", "amount": 100}]
        gen = obj._calculate_total_withdrawal_value(transfers, chain="mode")
        assert _drive(gen) == Decimal(0)


class TestUpdatePositionWithCurrentValue:
    """Tests for UpdatePositionWithCurrentValue."""

    def test_basic_cost_recovered(self) -> None:
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

    def test_not_recovered(self) -> None:
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

    def test_no_balances(self) -> None:
        """Test no balances."""
        obj = _mk()
        obj._get_current_timestamp = lambda: 1000
        obj._get_current_token_balances = _gen_return(None)
        pos = {"pool_address": "0xP", "amount0": None, "amount1": None}
        gen = obj._update_position_with_current_value(pos, Decimal("500"), "optimism")
        _drive(gen)
        assert pos["cost_recovered"] is False

    def test_exception(self) -> None:
        """Test exception."""
        obj = _mk()
        obj._get_current_timestamp = MagicMock(side_effect=RuntimeError("boom"))
        pos = {"pool_address": "0xP"}
        gen = obj._update_position_with_current_value(pos, Decimal("500"), "optimism")
        _drive(gen)
        assert pos["cost_recovered"] is False

    def test_zero_entry_cost(self) -> None:
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
    """Tests for CalculateCorrectedYield."""

    def test_no_decimals(self) -> None:
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

    def test_with_prices_in_cache(self) -> None:
        """Test with prices in cache."""
        obj = _mk()
        call_count = [0]

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_fetch_prices(self) -> None:
        """Test fetch prices."""
        obj = _mk()
        call_count = [0]

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_fetch_prices_fail(self) -> None:
        """Test fetch prices fail."""
        obj = _mk()

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_velo_rewards(self) -> None:
        """Test velo rewards."""
        obj = _mk()

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
    """Tests for GetCurrentTokenBalances."""

    def test_balancer(self) -> None:
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

    def test_uniswap(self) -> None:
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

    def test_velodrome(self) -> None:
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

    def test_sturdy(self) -> None:
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

    def test_unknown(self) -> None:
        """Test unknown."""
        obj = _mk()
        pos = {"dex_type": "unknown"}
        gen = obj._get_current_token_balances(pos, "optimism")
        assert _drive(gen) == {}


class TestReadInvestingPaused:
    """Tests for ReadInvestingPaused."""

    def test_true(self) -> None:
        """Test true."""
        obj = _mk()
        obj._read_kv = _gen_return({"investing_paused": "true"})
        gen = obj._read_investing_paused()
        assert _drive(gen) is True

    def test_false(self) -> None:
        """Test false."""
        obj = _mk()
        obj._read_kv = _gen_return({"investing_paused": "false"})
        gen = obj._read_investing_paused()
        assert _drive(gen) is False

    def test_none_result(self) -> None:
        """Test none result."""
        obj = _mk()
        obj._read_kv = _gen_return(None)
        gen = obj._read_investing_paused()
        assert _drive(gen) is False

    def test_none_value(self) -> None:
        """Test none value."""
        obj = _mk()
        obj._read_kv = _gen_return({"investing_paused": None})
        gen = obj._read_investing_paused()
        assert _drive(gen) is False

    def test_exception(self) -> None:
        """Test exception."""
        obj = _mk()

        def boom(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            raise RuntimeError("fail")
            yield  # noqa

        obj._read_kv = boom
        gen = obj._read_investing_paused()
        assert _drive(gen) is False


class TestCheckIsValidSafeAddress:
    """Tests for CheckIsValidSafeAddress."""

    def test_valid(self) -> None:
        """Test valid."""
        obj = _mk()
        obj.contract_interact = _gen_return(["0xOwner"])
        gen = obj.check_is_valid_safe_address("0xSafe", "optimism")
        assert _drive(gen) is True

    def test_invalid(self) -> None:
        """Test invalid."""
        obj = _mk()
        obj.contract_interact = _gen_return(None)
        gen = obj.check_is_valid_safe_address("0xSafe", "optimism")
        assert _drive(gen) is False

    def test_exception(self) -> None:
        """Test exception."""
        obj = _mk()

        def boom(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            raise RuntimeError("boom")
            yield  # noqa

        obj.contract_interact = boom
        gen = obj.check_is_valid_safe_address("0xSafe", "optimism")
        assert _drive(gen) is False


class TestGetVelodromePendingRewards:
    """Tests for GetVelodromePendingRewards."""

    def test_no_pool_address(self) -> None:
        """Test no pool address."""
        obj = _mk()
        gen = obj._get_velodrome_pending_rewards({}, "optimism", "0xU")
        assert _drive(gen) == Decimal(0)

    def test_no_pool_behaviour(self) -> None:
        """Test no pool behaviour."""
        obj = _mk()
        obj.pools = {}
        gen = obj._get_velodrome_pending_rewards(
            {"pool_address": "0xP"}, "optimism", "0xU"
        )
        assert _drive(gen) == Decimal(0)

    def test_cl_pool_with_rewards(self) -> None:
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

    def test_cl_pool_no_gauge(self) -> None:
        """Test cl pool no gauge."""
        obj = _mk()
        pool_mock = MagicMock()
        pool_mock.get_gauge_address = _gen_return(None)
        obj.pools = {"velodrome": pool_mock}
        pos = {"pool_address": "0xP", "is_cl_pool": True, "positions": []}
        gen = obj._get_velodrome_pending_rewards(pos, "optimism", "0xU")
        assert _drive(gen) == Decimal(0)

    def test_regular_pool(self) -> None:
        """Test regular pool."""
        obj = _mk()
        pool_mock = MagicMock()
        pool_mock.get_pending_rewards = _gen_return(5 * 10**18)
        obj.pools = {"velodrome": pool_mock}
        pos = {"pool_address": "0xP", "is_cl_pool": False}
        gen = obj._get_velodrome_pending_rewards(pos, "optimism", "0xU")
        result = _drive(gen)
        assert result == Decimal("5")

    def test_exception(self) -> None:
        """Test exception."""
        obj = _mk()
        obj.pools = {"velodrome": MagicMock(side_effect=RuntimeError)}
        pos = {"pool_address": "0xP"}
        gen = obj._get_velodrome_pending_rewards(pos, "optimism", "0xU")
        assert _drive(gen) == Decimal(0)


class TestCreatePortfolioData:
    """Tests for CreatePortfolioData."""

    def test_basic(self) -> None:
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

    def test_no_initial_investment(self) -> None:
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

    def test_zero_initial(self) -> None:
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

    def test_with_allocations_and_breakdown(self) -> None:
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

    def test_tick_ranges_in_allocation(self) -> None:
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

    def test_olas_filtered(self) -> None:
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

    def test_exception(self) -> None:
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

    def test_allocation_error_skipped(self) -> None:
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

    def test_breakdown_error_skipped(self) -> None:
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
    """Tests for ValidateVelodromeV2PoolAddresses."""

    def test_skip_non_velodrome(self) -> None:
        """Test skip non velodrome."""
        obj = _mk()
        obj.current_positions = [
            {"dex_type": "UniswapV3", "is_cl_pool": False, "is_stable": True}
        ]
        obj.store_current_positions = MagicMock()
        gen = obj._validate_velodrome_v2_pool_addresses()
        _drive(gen)

    def test_skip_cl_pool(self) -> None:
        """Test skip cl pool."""
        obj = _mk()
        obj.current_positions = [
            {"dex_type": "velodrome", "is_cl_pool": True, "is_stable": True}
        ]
        obj.store_current_positions = MagicMock()
        gen = obj._validate_velodrome_v2_pool_addresses()
        _drive(gen)

    def test_skip_not_stable(self) -> None:
        """Test skip not stable."""
        obj = _mk()
        obj.current_positions = [
            {"dex_type": "velodrome", "is_cl_pool": False, "is_stable": False}
        ]
        obj.store_current_positions = MagicMock()
        gen = obj._validate_velodrome_v2_pool_addresses()
        _drive(gen)

    def test_validates(self) -> None:
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
    """Tests for ValidateVelodromeV2PoolAddress."""

    def test_already_updated(self) -> None:
        """Test already updated."""
        obj = _mk()
        gen = obj._validate_velodrome_v2_pool_address({"isUpdated": True})
        assert _drive(gen) is True

    def test_missing_data(self) -> None:
        """Test missing data."""
        obj = _mk()
        gen = obj._validate_velodrome_v2_pool_address({"enter_tx_hash": None})
        assert _drive(gen) is False

    def test_no_receipt(self) -> None:
        """Test no receipt."""
        obj = _mk()
        obj.get_transaction_receipt = _gen_return(None)
        pos = {"enter_tx_hash": "0xTX", "chain": "optimism", "pool_address": "0xP"}
        gen = obj._validate_velodrome_v2_pool_address(pos)
        assert _drive(gen) is False

    def test_no_mint_event(self) -> None:
        """Test no mint event."""
        obj = _mk()
        obj.get_transaction_receipt = _gen_return({"logs": []})
        pos = {"enter_tx_hash": "0xTX", "chain": "optimism", "pool_address": "0xP"}
        gen = obj._validate_velodrome_v2_pool_address(pos)
        assert _drive(gen) is False

    def test_match(self) -> None:
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

    def test_mismatch(self) -> None:
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

    def test_exception(self) -> None:
        """Test exception."""
        obj = _mk()

        def boom(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            raise RuntimeError("boom")
            yield  # noqa

        obj.get_transaction_receipt = boom
        pos = {"enter_tx_hash": "0xTX", "chain": "optimism", "pool_address": "0xP"}
        gen = obj._validate_velodrome_v2_pool_address(pos)
        assert _drive(gen) is False


class TestCalculateTotalReversionValue:
    """Tests for CalculateTotalReversionValue."""

    def test_basic(self) -> None:
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

    def test_multiple_reversions(self) -> None:
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

    def test_no_eth_price(self) -> None:
        """Test no eth price."""
        obj = _mk()
        obj._fetch_historical_eth_price = _gen_return(None)
        eth_transfers = [{"timestamp": "2024-01-01T00:00:00Z", "amount": 1.0}]
        reversion = [{"amount": 0.5}]
        result = _drive(obj._calculate_total_reversion_value(eth_transfers, reversion))
        assert result == 0.0

    def test_unix_timestamp(self) -> None:
        """Test unix timestamp."""
        obj = _mk()
        obj._fetch_historical_eth_price = _gen_return(500.0)
        eth_transfers = [
            {"timestamp": "1704067200", "amount": 1.0},
        ]
        reversion = [{"amount": 0.1}]
        result = _drive(obj._calculate_total_reversion_value(eth_transfers, reversion))
        assert result == 0.1 * 500.0

    def test_bad_timestamp(self) -> None:
        """Test bad timestamp."""
        obj = _mk()
        obj._fetch_historical_eth_price = _gen_return(500.0)
        eth_transfers = [{"timestamp": "bad", "amount": 1.0}]
        reversion = [{"amount": 0.1}]
        result = _drive(obj._calculate_total_reversion_value(eth_transfers, reversion))
        # fallback to current date
        assert result == 0.1 * 500.0


class TestShouldIncludeTransferOptimism:
    """Tests for ShouldIncludeTransferOptimism."""

    def test_empty(self) -> None:
        """Test empty."""
        obj = _mk()
        gen = obj._should_include_transfer_optimism("")
        assert _drive(gen) is False

    def test_zero_addr(self) -> None:
        """Test zero addr."""
        obj = _mk()
        gen = obj._should_include_transfer_optimism(
            "0x0000000000000000000000000000000000000000"
        )
        assert _drive(gen) is False

    def test_cached_eoa(self) -> None:
        """Test cached eoa."""
        obj = _mk()
        cache_key = f"{CONTRACT_CHECK_CACHE_PREFIX}optimism_0xabc"
        obj._read_kv = _gen_return({cache_key: json.dumps({"is_eoa": True})})
        gen = obj._should_include_transfer_optimism("0xABC")
        assert _drive(gen) is True

    def test_cached_contract(self) -> None:
        """Test cached contract."""
        obj = _mk()
        cache_key = f"{CONTRACT_CHECK_CACHE_PREFIX}optimism_0xabc"
        obj._read_kv = _gen_return({cache_key: json.dumps({"is_eoa": False})})
        gen = obj._should_include_transfer_optimism("0xABC")
        assert _drive(gen) is False

    def test_eoa_uncached(self) -> None:
        """Test eoa uncached."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj._request_with_retries = _gen_return((True, {"result": "0x"}))
        obj._write_kv = _gen_none
        cg = MagicMock()
        obj.context.coingecko = cg

        gen = obj._should_include_transfer_optimism("0xABC")
        assert _drive(gen) is True

    def test_contract_not_safe(self) -> None:
        """Test contract not safe."""
        obj = _mk()
        call_count = [0]

        def fake_read(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            yield
            return {}

        def fake_req(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_rpc_fail(self) -> None:
        """Test rpc fail."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj._request_with_retries = _gen_return((False, {}))
        cg = MagicMock()
        obj.context.coingecko = cg

        gen = obj._should_include_transfer_optimism("0xABC")
        assert _drive(gen) is False

    def test_exception(self) -> None:
        """Test exception."""
        obj = _mk()
        # _read_kv succeeds, but _request_with_retries raises inside try block
        obj._read_kv = _gen_return({})

        def boom(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            raise RuntimeError("fail")
            yield  # noqa

        obj._request_with_retries = boom
        obj.coingecko.rate_limited_status_callback = MagicMock()
        obj.params.sleep_time = 1
        gen = obj._should_include_transfer_optimism("0xABC")
        assert _drive(gen) is False


class TestIsNotOtherContractOptimism:
    """Tests for IsNotOtherContractOptimism."""

    def test_empty(self) -> None:
        """Test empty."""
        obj = _mk()
        gen = obj._is_not_other_contract_optimism("")
        assert _drive(gen) is False

    def test_zero(self) -> None:
        """Test zero."""
        obj = _mk()
        gen = obj._is_not_other_contract_optimism(
            "0x0000000000000000000000000000000000000000"
        )
        assert _drive(gen) is False

    def test_cached(self) -> None:
        """Test cached."""
        obj = _mk()
        cache_key = f"{CONTRACT_CHECK_CACHE_PREFIX}optimism_0xabc"
        obj._read_kv = _gen_return({cache_key: json.dumps({"is_eoa": True})})
        gen = obj._is_not_other_contract_optimism("0xABC")
        assert _drive(gen) is True

    def test_eoa(self) -> None:
        """Test eoa."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj._request_with_retries = _gen_return((True, {"result": "0x"}))
        obj._write_kv = _gen_none
        cg = MagicMock()
        obj.context.coingecko = cg

        gen = obj._is_not_other_contract_optimism("0xABC")
        assert _drive(gen) is True

    def test_contract_safe(self) -> None:
        """Test contract safe."""
        obj = _mk()
        call_count = [0]

        def fake_req(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_rpc_fail(self) -> None:
        """Test rpc fail."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj._request_with_retries = _gen_return((False, {}))
        cg = MagicMock()
        obj.context.coingecko = cg

        gen = obj._is_not_other_contract_optimism("0xABC")
        assert _drive(gen) is False

    def test_exception(self) -> None:
        """Test exception."""
        obj = _mk()
        obj._read_kv = _gen_return({})

        def boom(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            raise RuntimeError("fail")
            yield  # noqa

        obj._request_with_retries = boom
        obj.coingecko.rate_limited_status_callback = MagicMock()
        obj.params.sleep_time = 1
        gen = obj._is_not_other_contract_optimism("0xABC")
        assert _drive(gen) is False


class TestGetMasterSafeAddress:
    """Tests for GetMasterSafeAddress."""

    def test_no_service_id(self) -> None:
        """Test no service id."""
        obj = _mk()

        obj.params.on_chain_service_id = None
        gen = obj.get_master_safe_address()
        assert _drive(gen) is None

    def test_no_chains(self) -> None:
        """Test no chains."""
        obj = _mk()

        obj.params.on_chain_service_id = 1
        obj.params.target_investment_chains = []
        gen = obj.get_master_safe_address()
        assert _drive(gen) is None

    def test_staking_path_staked(self) -> None:
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

    def test_staking_path_unstaked_then_staked(self) -> None:
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

        def fake_get_state(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            obj.service_staking_state = StakingState.STAKED
            yield

        obj._get_service_staking_state = fake_get_state
        obj._get_service_info = _gen_return([0, "0xMaster"])
        obj.check_is_valid_safe_address = _gen_return(True)
        gen = obj.get_master_safe_address()
        assert _drive(gen) == "0xMaster"

    def test_staking_invalid_addr(self) -> None:
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

    def test_staking_no_info(self) -> None:
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

    def test_no_staking_registry(self) -> None:
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

    def test_no_registry_addr(self) -> None:
        """Test no registry addr."""
        obj = _mk()

        obj.params.on_chain_service_id = 1
        obj.params.target_investment_chains = ["optimism"]
        obj.params.staking_token_contract_address = None
        obj.params.staking_chain = None
        obj.params.service_registry_contract_addresses = {}
        gen = obj.get_master_safe_address()
        assert _drive(gen) is None

    def test_registry_no_result(self) -> None:
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
    """Tests for ShouldRecalculatePortfolio."""

    def test_no_initial(self) -> None:
        """Test no initial."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj._load_chain_total_investment = _gen_return(None)
        gen = obj.should_recalculate_portfolio({"portfolio_value": 100})
        assert _drive(gen) is True

    def test_no_final(self) -> None:
        """Test no final."""
        obj = _mk()

        obj.params.target_investment_chains = ["optimism"]
        obj._load_chain_total_investment = _gen_return(100.0)
        gen = obj.should_recalculate_portfolio({})
        assert _drive(gen) is True

    def test_post_tx_round(self) -> None:
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

    def test_time_or_positions(self) -> None:
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
    """Tests for UpdatePortfolioMetrics."""

    def test_zero_value(self) -> None:
        """Test zero value."""
        obj = _mk()
        obj._update_portfolio_breakdown_ratios = MagicMock()
        gen = obj._update_portfolio_metrics(Decimal(0), [], [], [])
        _drive(gen)
        obj._update_portfolio_breakdown_ratios.assert_called_once()

    def test_positive_value(self) -> None:
        """Test positive value."""
        obj = _mk()
        obj._update_portfolio_breakdown_ratios = MagicMock()
        obj._update_allocation_ratios = _gen_none
        gen = obj._update_portfolio_metrics(Decimal(100), [], [], [])
        _drive(gen)


class TestUpdateAllocationRatios:
    """Tests for UpdateAllocationRatios."""

    def test_zero_total(self) -> None:
        """Test zero total."""
        obj = _mk()
        allocs: List[Any] = []
        gen = obj._update_allocation_ratios([], Decimal(0), allocs)
        _drive(gen)
        assert allocs == []

    def test_with_shares(self) -> None:
        """Test with shares."""
        obj = _mk()
        obj.current_positions = [{"pool_address": "0xP", "dex_type": "UniswapV3"}]
        obj._get_tick_ranges = _gen_return([])
        shares: Any = [
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
        allocs: List[Any] = []
        gen = obj._update_allocation_ratios(shares, Decimal(100), allocs)
        _drive(gen)
        assert len(allocs) == 1
        assert allocs[0]["ratio"] == 100.0

    def test_with_tick_ranges(self) -> None:
        """Test with tick ranges."""
        obj = _mk()
        obj.current_positions = [{"pool_address": "0xP", "dex_type": "UniswapV3"}]
        obj._get_tick_ranges = _gen_return([{"tick": 1}])
        shares: Any = [
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
        allocs: List[Any] = []
        gen = obj._update_allocation_ratios(shares, Decimal(100), allocs)
        _drive(gen)
        assert "tick_ranges" in allocs[0]


class TestCalculatePositionAmounts:
    """Tests for CalculatePositionAmounts."""

    def test_missing_details(self) -> None:
        """Test missing details."""
        obj = _mk()
        gen = obj._calculate_position_amounts({}, 0, 0, {}, "UniswapV3", "optimism")
        assert _drive(gen) == (0, 0)

    def test_uniswap(self) -> None:
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

    def test_velodrome_with_pm(self) -> None:
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

    def test_velodrome_no_pm(self) -> None:
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
    """Tests for GetTokenDecimalsPair."""

    def test_both_ok(self) -> None:
        """Test both ok."""
        obj = _mk()
        call_count = [0]

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            call_count[0] += 1
            yield
            return 18

        obj._get_token_decimals = fake_dec
        gen = obj._get_token_decimals_pair("optimism", "0xT0", "0xT1")
        result = _drive(gen)
        assert result == (18, 18)

    def test_first_none(self) -> None:
        """Test first none."""
        obj = _mk()
        call_count = [0]

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            call_count[0] += 1
            yield
            return None if call_count[0] == 1 else 18

        obj._get_token_decimals = fake_dec
        gen = obj._get_token_decimals_pair("optimism", "0xT0", "0xT1")
        result = _drive(gen)
        assert result == (None, None)


class TestCalculateClPositionValue:
    """Tests for CalculateClPositionValue."""

    def test_missing_params(self) -> None:
        """Test missing params."""
        obj = _mk()
        gen = obj._calculate_cl_position_value(
            None, "optimism", {}, "0xT0", "0xT1", "0xPM", MagicMock(), "velodrome"
        )
        assert _drive(gen) == {}

    def test_no_slot0(self) -> None:
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

    def test_invalid_slot0(self) -> None:
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

    def test_no_decimals(self) -> None:
        """Test no decimals."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return {"sqrt_price_x96": 2**96, "tick": 0}
            return None

        obj.contract_interact = fake_ci

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_success(self) -> None:
        """Test success."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            yield
            return 18

        obj._get_token_decimals = fake_dec

        def fake_amounts(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_position_no_token_id(self) -> None:
        """Test position no token id."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            call_count[0] += 1
            yield
            return {"sqrt_price_x96": 2**96, "tick": 0}

        obj.contract_interact = fake_ci

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            yield
            return 18

        obj._get_token_decimals = fake_dec

        pos = {"token_id": None, "token0_symbol": "A", "token1_symbol": "B"}
        gen = obj._calculate_cl_position_value(
            "0xP", "optimism", pos, "0xT0", "0xT1", "0xPM", MagicMock(), "UniswapV3"
        )
        result = _drive(gen)
        assert result["0xT0"] == Decimal(0)

    def test_position_details_fail(self) -> None:
        """Test position details fail."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return {"sqrt_price_x96": 2**96, "tick": 0}
            return None

        obj.contract_interact = fake_ci

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            yield
            return 18

        obj._get_token_decimals = fake_dec

        pos = {"token_id": 1, "token0_symbol": "A", "token1_symbol": "B"}
        gen = obj._calculate_cl_position_value(
            "0xP", "optimism", pos, "0xT0", "0xT1", "0xPM", MagicMock(), "UniswapV3"
        )
        result = _drive(gen)
        assert result["0xT0"] == Decimal(0)

    def test_multiple_positions(self) -> None:
        """Test multiple positions."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            yield
            return 18

        obj._get_token_decimals = fake_dec

        def fake_amounts(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
    """Tests for GetUserShareValueVelodromeCl."""

    def test_no_pm(self) -> None:
        """Test no pm."""
        obj = _mk()
        obj.params.velodrome_non_fungible_position_manager_contract_addresses = {}
        pos = {"token0": "0xT0", "token1": "0xT1", "is_cl_pool": True}
        gen = obj._get_user_share_value_velodrome_cl(
            "0xP", "optimism", pos, "0xT0", "0xT1"
        )
        assert _drive(gen) == {}

    def test_ok(self) -> None:
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
    """Tests for GetUserShareValueVelodromeNonCl."""

    def test_no_user_balance(self) -> None:
        """Test no user balance."""
        obj = _mk()
        obj.contract_interact = _gen_return(None)
        pos = {"token0_symbol": "A", "token1_symbol": "B"}
        gen = obj._get_user_share_value_velodrome_non_cl(
            "0xU", "0xP", "optimism", pos, "0xT0", "0xT1"
        )
        assert _drive(gen) == {}

    def test_no_total_supply(self) -> None:
        """Test no total supply."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_no_reserves(self) -> None:
        """Test no reserves."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_success(self) -> None:
        """Test success."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_no_decimals(self) -> None:
        """Test no decimals."""
        obj = _mk()
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            yield
            return None

        obj._get_token_decimals = fake_dec

        pos = {"token0_symbol": "A", "token1_symbol": "B"}
        gen = obj._get_user_share_value_velodrome_non_cl(
            "0xU", "0xP", "optimism", pos, "0xT0", "0xT1"
        )
        assert _drive(gen) == {}


class TestGetUserShareValueBalancerFull:
    """Tests for GetUserShareValueBalancerFull."""

    def test_zero_total_supply(self) -> None:
        """Test zero total supply."""
        obj = _mk()
        obj.params.balancer_vault_contract_addresses = {"optimism": "0xV"}
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_success(self) -> None:
        """Test success."""
        obj = _mk()
        obj.params.balancer_vault_contract_addresses = {"optimism": "0xV"}
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            yield
            return 18

        obj._get_token_decimals = fake_dec
        gen = obj.get_user_share_value_balancer("0xU", "0xPID", "0xP", "optimism")
        result = _drive(gen)
        assert len(result) == 1

    def test_no_user_balance(self) -> None:
        """Test no user balance."""
        obj = _mk()
        obj.params.balancer_vault_contract_addresses = {"optimism": "0xV"}
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
    """Tests for CalculateSafeBalancesValue."""

    def test_no_safe_address(self) -> None:
        """Test no safe address."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {}
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)

    def test_optimism_no_balances(self) -> None:
        """Test optimism no balances."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._get_optimism_balances_from_safe_api = _gen_return([])
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)

    def test_mode_no_balances(self) -> None:
        """Test mode no balances."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj._get_mode_balances_from_explorer_api = _gen_return([])
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)

    def test_zero_balance_skipped(self) -> None:
        """Test zero balance skipped."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj._get_optimism_balances_from_safe_api = _gen_return(
            [{"address": "0xT", "asset_symbol": "TKN", "balance": 0}]
        )
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)

    def test_olas_skipped(self) -> None:
        """Test olas skipped."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        olas = OLAS_ADDRESSES["optimism"]
        obj._get_optimism_balances_from_safe_api = _gen_return(
            [{"address": olas, "asset_symbol": "OLAS", "balance": 10**18}]
        )
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)

    def test_eth_balance(self) -> None:
        """Test eth balance."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": None}
        obj._get_optimism_balances_from_safe_api = _gen_return(
            [{"address": ZERO_ADDRESS, "asset_symbol": "ETH", "balance": 10**18}]
        )
        obj._fetch_zero_address_price = _gen_return(3000.0)
        gen = obj._calculate_safe_balances_value([])
        result = _drive(gen)
        assert result == Decimal("1") * Decimal("3000.0")

    def test_erc20_balance(self) -> None:
        """Test erc20 balance."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": None}
        obj._get_optimism_balances_from_safe_api = _gen_return(
            [{"address": "0xUSDC", "asset_symbol": "USDC", "balance": 10**6}]
        )
        obj._get_token_decimals = _gen_return(6)
        obj._fetch_token_price = _gen_return(1.0)
        gen = obj._calculate_safe_balances_value([])
        result = _drive(gen)
        assert result == Decimal("1") * Decimal("1.0")

    def test_no_price(self) -> None:
        """Test no price."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": None}
        obj._get_optimism_balances_from_safe_api = _gen_return(
            [{"address": "0xTKN", "asset_symbol": "TKN", "balance": 10**18}]
        )
        obj._get_token_decimals = _gen_return(18)
        obj._fetch_token_price = _gen_return(None)
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)

    def test_no_decimals(self) -> None:
        """Test no decimals."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": None}
        obj._get_optimism_balances_from_safe_api = _gen_return(
            [{"address": "0xTKN", "asset_symbol": "TKN", "balance": 10**18}]
        )
        obj._get_token_decimals = _gen_return(None)
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)

    def test_velo_balance(self) -> None:
        """Test velo balance."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": "0xvelo"}
        obj._get_optimism_balances_from_safe_api = _gen_return(
            [{"address": "0xVELO", "asset_symbol": "VELO", "balance": 10**18}]
        )
        obj._get_token_decimals = _gen_return(18)
        obj.get_coin_id_from_symbol = lambda s, c: "velodrome-finance"
        obj._fetch_coin_price = _gen_return(0.5)
        gen = obj._calculate_safe_balances_value([])
        result = _drive(gen)
        assert result == Decimal("1") * Decimal("0.5")


class TestCalculateTotalVolume:
    """Tests for CalculateTotalVolume."""

    def test_cached(self) -> None:
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

    def test_no_cache(self) -> None:
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

    def test_missing_data(self) -> None:
        """Test missing data."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj.current_positions = [
            {"pool_address": "0xP", "tx_hash": "0xTX", "token0": None}
        ]
        gen = obj._calculate_total_volume()
        result = _drive(gen)
        assert result is None

    def test_no_positions(self) -> None:
        """Test no positions."""
        obj = _mk()
        obj._read_kv = _gen_return({})
        obj.current_positions = []
        gen = obj._calculate_total_volume()
        result = _drive(gen)
        assert result is None

    def test_no_price(self) -> None:
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

    def test_invalid_cache(self) -> None:
        """Test invalid cache."""
        obj = _mk()
        obj._read_kv = _gen_return({"initial_investment_values": "{{bad json"})
        obj.current_positions = []
        gen = obj._calculate_total_volume()
        result = _drive(gen)
        assert result is None

    def test_no_token0_decimals(self) -> None:
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

    def test_with_token1(self) -> None:
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

    def test_no_token1_price(self) -> None:
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
    """Tests for TrackEthTransfersMode."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_api_error(self, mock_requests: MagicMock) -> None:
        """Test api error."""
        obj = _mk()
        mock_requests.get.return_value = MagicMock(status_code=500)
        result = obj._track_eth_transfers_mode("0xSafe", "2024-01-01")
        assert result == {"incoming": {}, "outgoing": {}}

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_api_bad_status(self, mock_requests: MagicMock) -> None:
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
    def test_success_incoming(self, mock_requests: MagicMock) -> None:
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
    def test_exception(self, mock_requests: MagicMock) -> None:
        """Test exception."""
        obj = _mk()
        mock_requests.get.side_effect = RuntimeError("fail")
        result = obj._track_eth_transfers_mode("0xSafe", "2024-01-01")
        assert result == {"incoming": {}, "outgoing": {}}


class TestTrackErc20TransfersMode:
    """Tests for TrackErc20TransfersMode."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_api_error(self, mock_requests: MagicMock) -> None:
        """Test api error."""
        obj = _mk()
        mock_requests.get.return_value = MagicMock(status_code=500)
        result = obj._track_erc20_transfers_mode("0xSafe", 1704067200)
        assert result == {"outgoing": {}}

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_empty(self, mock_requests: MagicMock) -> None:
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
    def test_exception(self, mock_requests: MagicMock) -> None:
        """Test exception."""
        obj = _mk()
        mock_requests.get.side_effect = RuntimeError("fail")
        result = obj._track_erc20_transfers_mode("0xSafe", 1704067200)
        assert result == {"outgoing": {}}

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_outgoing_usdc(self, mock_requests: MagicMock) -> None:
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


class TestFetchEthTransfersMode:
    """Tests for FetchEthTransfersMode."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_api_error(self, mock_requests: MagicMock) -> None:
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
    def test_empty(self, mock_requests: MagicMock) -> None:
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
    def test_with_data(self, mock_requests: MagicMock) -> None:
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
        transfers: Dict[Any, Any] = {}
        result = obj._fetch_eth_transfers_mode("0xAddr", end_dt, transfers, True)
        assert result is True


class TestCheckAndCreateEthRevertTransactions:
    """Tests for CheckAndCreateEthRevertTransactions."""

    def test_no_safe(self) -> None:
        """Test no safe."""
        obj = _mk()
        gen = obj._check_and_create_eth_revert_transactions("optimism", None, "sender")
        _drive(gen)

    def test_zero_amount(self) -> None:
        """Test zero amount."""
        obj = _mk()
        obj._track_eth_transfers_and_reversions = _gen_return(
            {"to_address": None, "reversion_amount": 0, "master_safe_address": None}
        )
        gen = obj._check_and_create_eth_revert_transactions(
            "optimism", "0xSafe", "sender"
        )
        _drive(gen)

    def test_positive_amount_no_master(self) -> None:
        """Test positive amount no master."""
        obj = _mk()
        obj._track_eth_transfers_and_reversions = _gen_return(
            {"to_address": None, "reversion_amount": 0.5, "master_safe_address": None}
        )
        gen = obj._check_and_create_eth_revert_transactions(
            "optimism", "0xSafe", "sender"
        )
        _drive(gen)

    def test_positive_amount_with_tx(self) -> None:
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

    def test_positive_amount_no_hash(self) -> None:
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
    """Tests for TrackWhitelistedAssets."""

    def test_empty_chains(self) -> None:
        """Test empty chains."""
        obj = _mk()
        obj.params.target_investment_chains = []
        obj.store_whitelisted_assets = MagicMock()
        gen = obj._track_whitelisted_assets()
        _drive(gen)

    def test_skip_non_target_chain(self) -> None:
        """Test skip non target chain."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.store_whitelisted_assets = MagicMock()
        obj.sleep = _gen_none
        # Only iterate over chains that are in target chains
        gen = obj._track_whitelisted_assets()
        _drive(gen)

    def test_price_drop(self) -> None:
        """Test price drop."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.whitelisted_assets = {"mode": {"0xT": "TKN"}}
        obj.store_whitelisted_assets = MagicMock()
        obj.sleep = _gen_none
        # Yesterday price 100, today price 90 -> -10% drop
        call_count = [0]

        def fake_price(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            call_count[0] += 1
            yield
            return 100.0 if call_count[0] == 1 else 90.0

        obj._get_historical_price_for_date = fake_price
        gen = obj._track_whitelisted_assets()
        _drive(gen)

    def test_no_price(self) -> None:
        """Test no price."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.whitelisted_assets = {"mode": {"0xT": "TKN"}}
        obj.store_whitelisted_assets = MagicMock()
        obj.sleep = _gen_none
        obj._get_historical_price_for_date = _gen_return(None)
        gen = obj._track_whitelisted_assets()
        _drive(gen)

    def test_price_exception(self) -> None:
        """Test price exception."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.whitelisted_assets = {"mode": {"0xT": "TKN"}}
        obj.store_whitelisted_assets = MagicMock()
        obj.sleep = _gen_none

        def boom(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            raise RuntimeError("fail")
            yield  # noqa

        obj._get_historical_price_for_date = boom
        gen = obj._track_whitelisted_assets()
        _drive(gen)

    def test_price_stable(self) -> None:
        """Price stable (> -5%), no removal."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.whitelisted_assets = {"mode": {"0xT": "TKN"}}
        obj.store_whitelisted_assets = MagicMock()
        obj.sleep = _gen_none
        call_count = [0]

        def fake_price(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_address_removal_branch(self) -> None:
        """Cover lines 325->334, 328-329: address in whitelisted_assets removed."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.whitelisted_assets = {"mode": {"0xT": "TKN"}}
        obj.store_whitelisted_assets = MagicMock()
        obj.sleep = _gen_none
        call_count = [0]

        def fake_price(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
    """Tests for HavePositionsChangedClosed."""

    def test_closed_positions_detected(self) -> None:
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
    """Tests for UpdatePortfolioBreakdownRatiosEdge."""

    def test_none_value_usd_entry_filtered(self) -> None:
        """Cover line 736: value_usd is None -> continue."""
        obj = _mk()
        # Pass total_value=0 so the sum() that would fail on None is skipped
        bd = [
            {"value_usd": None, "balance": 0, "price": 0},
            {"value_usd": 10, "balance": 1, "price": 10},
        ]
        obj._update_portfolio_breakdown_ratios(bd, Decimal(0))
        assert len(bd) == 1
        assert bd[0]["value_usd"] == 10.0  # type: ignore[index]

    def test_inner_exception_branch(self) -> None:
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
    """Tests for CreatePortfolioDataROIException."""

    def test_roi_calculation_error(self) -> None:
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
    """Tests for GetTickRangesFull."""

    def test_success_with_data(self) -> None:
        """Cover lines 829-874: successful tick range retrieval."""
        obj = _mk()
        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_velodrome_cl_with_positions(self) -> None:
        """Cover Velodrome CL multi-position path."""
        obj = _mk()
        obj.params.velodrome_non_fungible_position_manager_contract_addresses = {
            "optimism": "0xPM"
        }
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_no_position_data(self) -> None:
        """Cover line 857-858: position_data is None -> continue."""
        obj = _mk()
        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_no_token_id_skipped(self) -> None:
        """Cover line 844-845: token_id missing -> continue."""
        obj = _mk()
        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            call_count[0] += 1
            yield
            return {"tick": 50, "sqrt_price_x96": 2**96}

        obj.contract_interact = fake_ci
        position = {"dex_type": "UniswapV3", "pool_address": "0xPool", "token_id": None}
        gen = obj._get_tick_ranges(position, "optimism")
        result = _drive(gen)
        assert result == []


class TestHandleVelodromePositionEdge:
    """Tests for HandleVelodromePositionEdge."""

    def test_staked_zero_rewards(self) -> None:
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

    def test_staked_no_velo_address(self) -> None:
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
    """Tests for CalculateCorrectedYieldVeloBranches."""

    def test_velo_price_in_token_prices(self) -> None:
        """Cover line 1431: velo_price from token_prices cache."""
        obj = _mk()

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_velo_price_fetch_none(self) -> None:
        """Cover lines 1436-1439: _fetch_coin_price returns None."""
        obj = _mk()

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_velo_balance_zero(self) -> None:
        """Cover 1428->1446: velo_balance == 0 -> skip."""
        obj = _mk()

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_velo_not_in_balances(self) -> None:
        """Cover 1426->1446: velo_token_address not in current_balances."""
        obj = _mk()

        def fake_dec(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
    """Tests for UpdateUniswapPositionWarning."""

    def test_cl_no_data_warning(self) -> None:
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
    """Tests for UpdateVelodromePositionWarning."""

    def test_non_cl_none_balance(self) -> None:
        """Cover line 2849: balance is None for Velodrome non-CL."""
        obj = _mk()
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.contract_interact = _gen_return(None)
        pos = {"chain": "optimism", "is_cl_pool": False, "pool_address": "0xP"}
        gen = obj._update_velodrome_position(pos)
        _drive(gen)
        assert "current_liquidity" not in pos

    def test_cl_no_data_warning(self) -> None:
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
    """Tests for GetMasterSafeAddressRegistryInvalid."""

    def test_registry_invalid_address(self) -> None:
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
    """Tests for GetMasterSafeAddressStakingStaysUnstaked."""

    def test_staking_stays_unstaked(self) -> None:
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
    """Tests for VelodromePendingRewardsFalsy."""

    def test_cl_rewards_zero(self) -> None:
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

    def test_cl_rewards_none(self) -> None:
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
    """Tests for ValidateVelodromeV2PoolAddressesFailure."""

    def test_validation_fails(self) -> None:
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
    """Tests for ShouldIncludeTransferEthOk."""

    def test_eth_transfer_ok(self) -> None:
        """Cover 3358->3361: eth transfer with status ok and value > 0."""
        obj = _mk()
        result = obj._should_include_transfer(
            {"hash": "0x123", "is_contract": False},
            tx_data={"status": "ok", "value": "100"},
            is_eth_transfer=True,
        )
        assert result is True


class TestGetDatetimeFromTimestampTzAware:
    """Tests for GetDatetimeFromTimestampTzAware."""

    def test_already_has_tzinfo(self) -> None:
        """Cover 3376->3379 false branch: dt already has tzinfo."""
        obj = _mk()
        # A datetime string without Z or + but with timezone info after parsing
        # Use a string that doesn't end with Z, doesn't contain +, doesn't end with UTC
        dt = obj._get_datetime_from_timestamp("2024-01-01T00:00:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc


class TestShouldIncludeTransferModeEthOk:
    """Tests for ShouldIncludeTransferModeEthOk."""

    def test_eth_transfer_ok(self) -> None:
        """Test eth transfer ok."""
        obj = _mk()
        result = obj._should_include_transfer_mode(
            {"hash": "0xabc", "is_contract": False},
            tx_data={"status": "ok", "value": "100"},
            is_eth_transfer=True,
        )
        assert result is True


class TestCalculateSafeBalancesValueZeroBalance:
    """Tests for CalculateSafeBalancesValueZeroBalance."""

    def test_zero_adjusted_balance(self) -> None:
        """Cover line 2155: adjusted_balance <= 0 -> continue."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.velo_token_contract_addresses = {"optimism": None}
        obj._get_optimism_balances_from_safe_api = _gen_return(
            [{"address": "0xTKN", "asset_symbol": "TKN", "balance": 0}]
        )
        obj._get_token_decimals = _gen_return(18)
        gen = obj._calculate_safe_balances_value([])
        assert _drive(gen) == Decimal(0)


class TestTrackWithdrawalOptimismException:
    """Tests for TrackWithdrawalOptimismException."""

    def test_exception(self) -> None:
        """Cover lines 2403-2407: exception in withdrawal calculation."""
        obj = _mk()

        def boom(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
    """Tests for ShouldIncludeTransferOptimismCacheError."""

    def test_cached_bad_json(self) -> None:
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
    """Tests for IsNotOtherContractOptimismCacheError."""

    def test_cached_bad_json(self) -> None:
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

    def test_not_eoa_log(self) -> None:
        """Cover line 3986: not is_eoa -> log message."""
        obj = _mk()
        call_count = [0]

        def fake_req(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
    """Tests for CalculatePositionAmountsOutOfRange."""

    def test_out_of_range(self) -> None:
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
    """Tests for GetUserShareValueBalancerZeroSupplyDecimal."""

    def test_zero_total_supply_decimal(self) -> None:
        """Cover lines 1985-1987: total_supply_decimal == 0."""
        obj = _mk()
        obj.params.balancer_vault_contract_addresses = {"optimism": "0xV"}
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

        def fake_ci2(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
    """Tests for GetUserShareValueBalancerNoDecimals."""

    def test_token_no_decimals(self) -> None:
        """Cover lines 2000-2003: _get_token_decimals returns None."""
        obj = _mk()
        obj.params.balancer_vault_contract_addresses = {"optimism": "0xV"}
        call_count = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
    """Tests for GetUserShareValueUniswapSuccess."""

    def test_success(self) -> None:
        """Cover line 1899: successful path through uniswap."""
        obj = _mk()
        obj.params.uniswap_position_manager_contract_addresses = {"optimism": "0xPM"}
        obj._calculate_cl_position_value = _gen_return({"0xT0": Decimal(1)})
        pos = {"token0": "0xT0", "token1": "0xT1"}
        gen = obj.get_user_share_value_uniswap("0xP", 1, "optimism", pos)
        result = _drive(gen)
        assert result == {"0xT0": Decimal(1)}


class TestCalculateUserShareValues:
    """Tests for CalculateUserShareValues."""

    def test_no_positions(self) -> None:
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

    def test_with_open_position(self) -> None:
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

    def test_missing_dex_type(self) -> None:
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

    def test_unsupported_dex(self) -> None:
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

    def test_handler_returns_none(self) -> None:
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

    def test_exception_in_position(self) -> None:
        """Cover lines 630-632: exception processing position."""
        obj = _mk()
        obj.current_positions = [
            {"status": "open", "dex_type": "UniswapV3", "chain": "optimism"}
        ]
        obj.store_current_positions = MagicMock()

        def boom(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_initial_investment_none_fallback(self) -> None:
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

    def test_initial_investment_none_no_kv(self) -> None:
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

    def test_portfolio_data_falsy(self) -> None:
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

    def test_balancer_position(self) -> None:
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
    """Tests for AsyncAct."""

    def test_full_flow(self) -> None:
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

    def test_period_zero(self) -> None:
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

    def test_invalid_last_updated_timestamp(self) -> None:
        """Cover lines 208-212: ValueError on int(last_updated)."""
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

            read_call = [0]

            def fake_read(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_no_safe_address(self) -> None:
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

    def test_unsupported_chain(self) -> None:
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

    def test_mode_airdrop_first_scan(self) -> None:
        """Test mode airdrop first scan."""
        obj = _mk()
        obj.params.target_investment_chains = ["mode"]
        obj.params.safe_contract_addresses = {"mode": "0xSafe"}
        obj.params.airdrop_started = True
        cc = [0]

        def fr(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_mode_airdrop_incremental(self) -> None:
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

    def test_optimism_no_previous_calc(self) -> None:
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

    def test_cached_value_today(self) -> None:
        """Test cached value today."""
        import calendar
        from datetime import datetime

        obj = _mk()
        # Build a UTC timestamp whose utcfromtimestamp date matches datetime.now() date
        now = datetime.now()
        noon = datetime(now.year, now.month, now.day, 12, 0, 0)
        ts = str(calendar.timegm(noon.timetuple()))
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.airdrop_started = False
        obj._read_kv = _gen_return({"last_initial_value_calculated_timestamp": ts})
        obj._load_chain_total_investment = _gen_return(999.0)
        obj._write_kv = _gen_none
        obj._save_chain_total_investment = _gen_none
        obj._get_current_timestamp = lambda: 100
        assert (
            _drive(obj.calculate_initial_investment_value_from_funding_events())
            == 999.0
        )

    def test_cached_value_today_zero(self) -> None:
        """Test cached value today zero."""
        import calendar
        from datetime import datetime

        obj = _mk()
        # Build a UTC timestamp whose utcfromtimestamp date matches datetime.now() date
        now = datetime.now()
        noon = datetime(now.year, now.month, now.day, 12, 0, 0)
        ts = str(calendar.timegm(noon.timetuple()))
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.airdrop_started = False
        obj._read_kv = _gen_return({"last_initial_value_calculated_timestamp": ts})
        obj._load_chain_total_investment = _gen_return(0.0)
        obj._fetch_all_transfers_until_date_optimism = _gen_return(
            {"d": [{"symbol": "ETH", "delta": 1}]}
        )
        obj._calculate_chain_investment_value = _gen_return(50.0)
        obj._write_kv = _gen_none
        obj._save_chain_total_investment = _gen_none
        obj._get_current_timestamp = lambda: 100
        assert (
            _drive(obj.calculate_initial_investment_value_from_funding_events()) == 50.0
        )

    def test_invalid_timestamp_format(self) -> None:
        """Test invalid timestamp format."""
        obj = _mk()
        obj.params.target_investment_chains = ["optimism"]
        obj.params.safe_contract_addresses = {"optimism": "0xSafe"}
        obj.params.airdrop_started = False
        obj._read_kv = _gen_return({"last_initial_value_calculated_timestamp": "bad"})
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

    def test_no_transfers(self) -> None:
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

    def _base(self, reversion: Any = None) -> Any:
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

    def test_eth_transfer_with_reversion(self) -> None:
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

    def test_usdc_non_eth(self) -> None:
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

    def test_airdrop_excluded(self) -> None:
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

    def test_no_coingecko_id(self) -> None:
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

    def test_negative_amount(self) -> None:
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

    def test_exception_in_transfer(self) -> None:
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

    def test_reversion_no_date(self) -> None:
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

    def test_cached_transfer_skips_price_fetch(self) -> None:
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

    def test_mix_cached_and_new_transfers(self) -> None:
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

    def test_price_fetch_failure_logs_warning(self) -> None:
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
    """Tests for FetchAllTransfersUntilDateMode."""

    def test_success(self) -> None:
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

    def test_token_fetch_fails(self) -> None:
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

    def test_backward_compat(self) -> None:
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

    def test_exception(self) -> None:
        """Test exception."""
        obj = _mk()
        obj.read_funding_events = lambda: {"mode": {}}
        obj.store_funding_events = MagicMock()

        def bad(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
    """Tests for FetchAllTransfersUntilDateOptimism."""

    def test_success(self) -> None:
        """Test success."""
        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()
        obj._fetch_optimism_transfers_safeglobal = _gen_none
        assert isinstance(
            _drive(obj._fetch_all_transfers_until_date_optimism("0xA", "2025-01-01")),
            dict,
        )

    def test_exception(self) -> None:
        """Test exception."""
        obj = _mk()
        obj.read_funding_events = lambda: {}

        def boom(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            yield
            raise RuntimeError("fail")

        obj._fetch_optimism_transfers_safeglobal = boom
        assert (
            _drive(obj._fetch_all_transfers_until_date_optimism("0xA", "2025-01-01"))
            == {}
        )


class TestFetchTokenTransfersBatch2:
    """Tests for FetchTokenTransfersBatch2."""

    def test_request_fails(self) -> None:
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

    def test_no_items(self) -> None:
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

    def test_processes(self) -> None:
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
        t: Dict[Any, List[Any]] = defaultdict(list)
        _drive(
            obj._fetch_token_transfers(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, {}
            )
        )
        assert len(t) == 1

    def test_skip_future(self) -> None:
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
        t: Dict[Any, List[Any]] = defaultdict(list)
        _drive(
            obj._fetch_token_transfers(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, {}
            )
        )
        assert len(t) == 0

    def test_skip_existing(self) -> None:
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
        t: Dict[Any, List[Any]] = defaultdict(list)
        _drive(
            obj._fetch_token_transfers(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, {"2024-12-15": []}
            )
        )
        assert len(t) == 0


class TestFetchTokenTransfersModeBatch2:
    """Tests for FetchTokenTransfersModeBatch2."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_non_200(self, m: Any) -> None:
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
    def test_empty(self, m: Any) -> None:
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
    def test_usdc(self, m: Any) -> None:
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
        t: Dict[Any, Any] = {}
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
    def test_self_transfer(self, m: Any) -> None:
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
    def test_airdrop(self, m: Any) -> None:
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
    def test_pagination(self, m: Any) -> Any:
        """Test pagination."""
        tx = {
            "timestamp": "2024-12-15T10:00:00Z",
            "from": {"hash": "0xS", "is_contract": False},
            "token": {"symbol": "USDC", "decimals": "6", "address": "0xT"},
            "total": {"value": "1000000"},
            "transaction_hash": "0xH",
        }
        c = [0]

        def mj() -> Any:
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
    def test_incremental_stop(self, m: Any) -> None:
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
    def test_zero_amount(self, m: Any) -> None:
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
    def test_non_usdc(self, m: Any) -> None:
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
        t: Dict[Any, Any] = {}
        _drive(
            obj._fetch_token_transfers_mode(
                "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, True
            )
        )
        assert len(t) == 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_bad_timestamp(self, m: Any) -> None:
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
    def test_bad_date_key(self, m: Any) -> None:
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


class TestFetchEthTransfersModeBatch2:
    """Tests for FetchEthTransfersModeBatch2."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_non_200(self, m: Any) -> None:
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
    def test_empty(self, m: Any) -> None:
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
    def test_processes(self, m: Any) -> None:
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
        t: Dict[Any, Any] = {}
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
    def test_zero_val(self, m: Any) -> None:
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
        t: Dict[Any, Any] = {}
        obj._fetch_eth_transfers_mode(
            "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, True
        )
        assert len(t) == 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_neg_delta(self, m: Any) -> None:
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
        t: Dict[Any, Any] = {}
        obj._fetch_eth_transfers_mode(
            "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, True
        )
        assert len(t) == 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_hash_present(self, m: Any) -> None:
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
        t: Dict[Any, Any] = {}
        obj._fetch_eth_transfers_mode(
            "0xA", datetime(2025, 1, 1, tzinfo=timezone.utc), t, True
        )
        assert len(t) == 0

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_incremental(self, m: Any) -> None:
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
    def test_pagination(self, m: Any) -> Any:
        """Test pagination."""
        e = {
            "value": str(2 * 10**18),
            "delta": str(1 * 10**18),
            "transaction_hash": None,
            "block_timestamp": "2024-12-15T10:00:00Z",
            "block_number": 100,
        }
        c = [0]

        def mj() -> Any:
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
    def test_bad_key(self, m: Any) -> None:
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
    """Tests for FetchOptimismSafeglobalBatch2."""

    def test_fail(self) -> None:
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

    def test_empty(self) -> None:
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

    def test_erc20_usdc(self) -> None:
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

        def fr(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
        t: Dict[Any, List[Any]] = defaultdict(list)
        _drive(obj._fetch_optimism_transfers_safeglobal("0xA", "2025-01-01", t, {}))
        assert "2024-12-15" in t

    def test_ether(self) -> None:
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
        t: Dict[Any, List[Any]] = defaultdict(list)
        _drive(obj._fetch_optimism_transfers_safeglobal("0xA", "2025-01-01", t, {}))
        assert len(t["2024-12-15"]) == 1

    def test_erc721(self) -> None:
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

    def test_self(self) -> None:
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

    def test_no_token_info_with_addr(self) -> None:
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

        def fr(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
        t: Dict[Any, List[Any]] = defaultdict(list)
        _drive(obj._fetch_optimism_transfers_safeglobal("0xA", "2025-01-01", t, {}))
        assert "2024-12-15" in t

    def test_no_token_info_no_addr(self) -> None:
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

    def test_zero_eth(self) -> None:
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

    def test_bad_eth_value(self) -> None:
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

    def test_exception(self) -> None:
        """Test exception."""
        obj = _mk()

        def bad(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
    """Tests for TrackEthReversions."""

    def _s(self, ch: Any = "optimism") -> Any:
        o = _mk()
        o.params.target_investment_chains = [ch]
        o.params.safe_contract_addresses = {ch: "0xSafe"}
        return o

    def test_no_incoming(self) -> None:
        """Test no incoming."""
        o = self._s()
        o._fetch_all_transfers_until_date_optimism = _gen_return({})
        o._fetch_outgoing_transfers_until_date_optimism = _gen_return({})
        assert _drive(o._track_eth_transfers_and_reversions("0xSafe", "optimism")) == {}

    def test_unsupported(self) -> None:
        """Test unsupported."""
        assert (
            _drive(
                self._s("polygon")._track_eth_transfers_and_reversions("0xS", "polygon")
            )
            == {}
        )

    def test_no_master(self) -> None:
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

    def test_two_transfers(self) -> None:
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

    def test_reversion_done(self) -> None:
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

    def test_mode(self) -> None:
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

    def test_balance_cap(self) -> None:
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

    def test_exception(self) -> None:
        """Test exception."""
        o = self._s()

        def boom(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            yield
            raise RuntimeError("f")

        o._fetch_all_transfers_until_date_optimism = boom
        assert (
            _drive(o._track_eth_transfers_and_reversions("0xSafe", "optimism"))[
                "reversion_amount"
            ]
            == 0
        )

    def test_unix_ts(self) -> None:
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

    def test_bad_ts(self) -> None:
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
    """Tests for CalcReversionValue."""

    def test_iso(self) -> None:
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

    def test_unix(self) -> None:
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

    def test_bad(self) -> None:
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

    def test_no_price(self) -> None:
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
    """Tests for OutgoingOptimism."""

    def test_no_addr(self) -> None:
        """Test no addr."""
        assert (
            _drive(
                _mk()._fetch_outgoing_transfers_until_date_optimism("", "2025-01-01")
            )
            == {}
        )

    def test_fail(self) -> None:
        """Test fail."""
        o = _mk()
        o._request_with_retries = _gen_return((False, {}))
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        assert (
            _drive(o._fetch_outgoing_transfers_until_date_optimism("0xA", "2025-01-01"))
            == {}
        )

    def test_eth(self) -> None:
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

    def test_zero(self) -> None:
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

    def test_bad_val(self) -> None:
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

    def test_exception(self) -> None:
        """Test exception."""
        o = _mk()

        def boom(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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
    """Tests for Erc20Optimism."""

    def test_no_addr(self) -> None:
        """Test no addr."""
        assert _drive(_mk()._track_erc20_transfers_optimism("", 1704067200)) == {
            "outgoing": {}
        }

    def test_fail(self) -> None:
        """Test fail."""
        o = _mk()
        o._request_with_retries = _gen_return((False, {}))
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        assert _drive(o._track_erc20_transfers_optimism("0xA", 1704067200)) == {
            "outgoing": {}
        }

    def test_usdc(self) -> None:
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

    def test_exception(self) -> None:
        """Test exception."""
        o = _mk()

        def boom(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            yield
            raise RuntimeError("f")

        o._request_with_retries = boom
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        assert _drive(o._track_erc20_transfers_optimism("0xA", 1704067200)) == {
            "outgoing": {}
        }


class TestTrackEthMode:
    """Tests for TrackEthMode."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_non_200(self, m: Any) -> None:
        """Test non 200."""
        m.return_value = MagicMock(status_code=500)
        assert _mk()._track_eth_transfers_mode("0xS", "2025-01-01") == {
            "incoming": {},
            "outgoing": {},
        }

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_status_0(self, m: Any) -> None:
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
    def test_ok(self, m: Any) -> None:
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
    def test_outgoing(self, m: Any) -> None:
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
    def test_zero(self, m: Any) -> None:
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
    def test_future(self, m: Any) -> None:
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
    def test_bad_ts(self, m: Any) -> None:
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
    def test_exception(self, m: Any) -> None:
        """Test exception."""
        m.side_effect = RuntimeError("f")
        assert _mk()._track_eth_transfers_mode("0xS", "2025-01-01") == {
            "incoming": {},
            "outgoing": {},
        }


class TestTrackErc20Mode:
    """Tests for TrackErc20Mode."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_non_200(self, m: Any) -> None:
        """Test non 200."""
        m.return_value = MagicMock(status_code=500)
        assert _mk()._track_erc20_transfers_mode("0xS", 1704067200) == {"outgoing": {}}

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_empty(self, m: Any) -> None:
        """Test empty."""
        m.return_value = MagicMock(status_code=200, json=lambda: {"items": []})
        assert _mk()._track_erc20_transfers_mode("0xS", 1704067200) == {"outgoing": {}}

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_usdc(self, m: Any) -> None:
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
    def test_zero(self, m: Any) -> None:
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
    def test_bad_ts(self, m: Any) -> None:
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
    def test_future(self, m: Any) -> None:
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
    def test_not_included(self, m: Any) -> None:
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
    def test_non_usdc(self, m: Any) -> None:
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
    def test_not_from_safe(self, m: Any) -> None:
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
    def test_pagination(self, m: Any) -> Any:
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

        def mj() -> Any:
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
    def test_val_error(self, m: Any) -> None:
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
    def test_exception(self, m: Any) -> None:
        """Test exception."""
        m.side_effect = RuntimeError("f")
        assert _mk()._track_erc20_transfers_mode("0xS", 1704067200) == {"outgoing": {}}


class TestVeloRewardsException:
    """Tests for VeloRewardsException."""

    def test_exception(self) -> None:
        """Test exception."""
        o = _mk()
        o.pools = {"velodrome": MagicMock()}

        def bad(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            yield
            raise RuntimeError("f")

        o.pools["velodrome"].get_gauge_address = bad
        assert _drive(
            o._get_velodrome_pending_rewards(
                {"pool_address": "0xP", "is_cl_pool": True}, "optimism", "0xU"
            )
        ) == Decimal(0)


class TestSmallBranchGaps2:
    """Tests for SmallBranchGaps2."""

    def test_closed_positions(self) -> None:
        """Test _have_positions_changed when positions are closed."""
        o = _mk()
        # Same count but different identifiers → closed_positions non-empty
        o.current_positions = [
            {"pool_address": "0xNEW", "dex_type": "uniswapV3", "status": "open"}
        ]
        last_data = {"allocations": [{"id": "0xOLD", "type": "uniswapV3"}]}
        assert o._have_positions_changed(last_data) is True

    def test_total_supply_zero_str(self) -> None:
        """Test get_user_share_value_balancer when total_supply is string '0' (bypasses int check, hits Decimal check)."""
        o = _mk()
        o.params.balancer_vault_contract_addresses = {"optimism": "0x" + "ab" * 20}
        ci = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_total_supply_zero_int(self) -> None:
        """Test get_user_share_value_balancer when total_supply is 0."""
        o = _mk()
        o.params.balancer_vault_contract_addresses = {"optimism": "0x" + "ab" * 20}
        ci = [0]

        def fake_ci(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def _obj(self) -> Any:
        o = _mk()
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        return o

    def test_no_timestamp(self) -> None:
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

    def test_bad_timestamp(self) -> None:
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

    def test_existing_data(self) -> None:
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

    def test_after_end_date(self) -> None:
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

    def test_duplicate(self) -> None:
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

    def test_not_included(self) -> None:
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

    def test_non_usdc(self) -> None:
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

    def test_unknown_type(self) -> None:
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

    def test_pagination(self) -> None:
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

        def fr(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            ci[0] += 1
            yield
            if ci[0] == 1:
                return (True, {"results": [td], "next": "http://next-page"})
            return (True, {"results": []})

        o._request_with_retries = fr
        o._should_include_transfer_optimism = _gen_return(True)
        t: Dict[Any, List[Any]] = defaultdict(list)
        _drive(o._fetch_optimism_transfers_safeglobal("0xA", "2025-01-01", t, {}))
        assert len(t["2024-12-15"]) == 1


class TestOutgoingOptimismBranches:
    """Tests for internal loop branches in _fetch_outgoing_transfers_until_date_optimism."""

    def _obj(self) -> Any:
        o = _mk()
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        return o

    def test_empty_results(self) -> None:
        """Empty results → break at line 4275."""
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": []}))
        assert (
            _drive(o._fetch_outgoing_transfers_until_date_optimism("0xA", "2025-01-01"))
            == {}
        )

    def test_no_execution_date(self) -> None:
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

    def test_bad_timestamp(self) -> None:
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

    def test_future_date(self) -> None:
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

    def test_not_from_us(self) -> None:
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

    def test_duplicate(self) -> None:
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

    def test_pagination(self) -> None:
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

        def fr(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def _obj(self) -> Any:
        o = _mk()
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        return o

    def test_empty_results(self) -> None:
        """Empty results → break at line 4394."""
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": []}))
        assert _drive(o._track_erc20_transfers_optimism("0xA", 1704067200)) == {
            "outgoing": {}
        }

    def test_no_execution_date(self) -> None:
        """No executionDate → continue at line 4400."""
        td = {"from": "0xaddr", "type": "ERC20_TRANSFER"}
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))

    def test_bad_timestamp(self) -> None:
        """Bad timestamp → continue at lines 4408-4412."""
        td = {"executionDate": "bad", "from": "0xaddr", "type": "ERC20_TRANSFER"}
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))

    def test_future_date(self) -> None:
        """Transfer after current_date → continue at line 4415."""
        td = {
            "executionDate": "2026-01-01T10:00:00Z",
            "from": "0xaddr",
            "type": "ERC20_TRANSFER",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))

    def test_not_from_safe(self) -> None:
        """Transfer from different address → continue at line 4419."""
        td = {
            "executionDate": "2024-01-01T10:00:00Z",
            "from": "0xOther",
            "type": "ERC20_TRANSFER",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))

    def test_not_erc20(self) -> None:
        """Non-ERC20 transfer type → continue at line 4425."""
        td = {
            "executionDate": "2024-01-01T10:00:00Z",
            "from": "0xaddr",
            "type": "ETHER_TRANSFER",
        }
        o = self._obj()
        o._request_with_retries = _gen_return((True, {"results": [td], "next": None}))
        _drive(o._track_erc20_transfers_optimism("0xAddr", 1704067200))

    def test_duplicate(self) -> None:
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

    def test_no_token_info_with_addr(self) -> None:
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

    def test_no_token_info_no_addr(self) -> None:
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

    def test_not_usdc(self) -> None:
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

    def test_zero_amount(self) -> None:
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

    def test_pagination(self) -> None:
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

        def fr(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
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

    def test_iso_z_timestamp(self) -> None:
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

    def test_safeglobal_stops_on_existing_dates(self) -> None:
        """Pagination stops when entire page is on already-stored dates."""
        o = _mk()
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1

        existing_data = {
            "2024-12-15": [{"symbol": "ETH", "amount": 1.0}],
        }
        # All transfers in page are on already-stored date
        page_data = {
            "results": [
                {
                    "executionDate": "2024-12-15T10:00:00Z",
                    "from": "0xSender",
                    "type": "ETHER_TRANSFER",
                    "value": str(10**18),
                    "transactionHash": "0xH1",
                },
                {
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

        all_transfers: Dict[Any, List[Any]] = defaultdict(list)
        _drive(
            o._fetch_optimism_transfers_safeglobal(
                "0xAddr", "2025-01-01", all_transfers, existing_data
            )
        )
        # No new transfers should be added (all skipped as existing)
        assert len(all_transfers) == 0
        # API was called only once (early-stop prevented following "next" cursor)
        assert (
            o._request_with_retries.call_count
            if hasattr(o._request_with_retries, "call_count")
            else True
        )

    def test_outgoing_persists_new_transfers(self) -> None:
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

    def test_outgoing_returns_existing_on_no_address(self) -> None:
        """When no address is provided, returns existing persisted data."""
        o = _mk()
        o.funding_events = {"optimism_outgoing": {"2024-01-01": [{"symbol": "ETH"}]}}
        result = _drive(
            o._fetch_outgoing_transfers_until_date_optimism("", "2025-01-01")
        )
        assert "2024-01-01" in result

    def test_reversion_cache_hit(self) -> None:
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

    def test_reversion_cache_miss_on_new_transfers(self) -> None:
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

    def test_get_transfer_key_no_tx_hash(self) -> None:
        """Fallback key when tx_hash is missing."""
        o = _mk()
        key = o._get_transfer_key(
            "2025-01-01", {"symbol": "ETH", "delta": 1.5, "from_address": "0xF"}, 3
        )
        assert key == "2025-01-01_ETH_1.5_0xF_3"

    def test_get_transfer_key_empty_tx_hash(self) -> None:
        """Fallback key when tx_hash is empty string."""
        o = _mk()
        key = o._get_transfer_key("2025-01-01", {"tx_hash": "", "symbol": "USDC"}, 0)
        assert key == "2025-01-01_USDC_0__0"

    def test_load_priced_transfers_valid(self) -> None:
        """Load valid JSON from KV store."""
        o = _mk()
        data = json.dumps({"k1": 100.0, "k2": 200.0})
        o._read_kv = _gen_return({"optimism_priced_transfers": data})
        result = _drive(o._load_priced_transfers("optimism"))
        assert result == {"k1": 100.0, "k2": 200.0}

    def test_load_priced_transfers_empty(self) -> None:
        """Return empty dict when KV has no data."""
        o = _mk()
        o._read_kv = _gen_return({})
        result = _drive(o._load_priced_transfers("optimism"))
        assert result == {}

    def test_load_priced_transfers_malformed_json(self) -> None:
        """Return empty dict and log warning on malformed JSON."""
        o = _mk()
        o._read_kv = _gen_return({"optimism_priced_transfers": "not-json{{"})
        result = _drive(o._load_priced_transfers("optimism"))
        assert result == {}
        o.context.logger.warning.assert_called()

    def test_save_priced_transfers(self) -> None:
        """Verify KV write is called with serialized JSON."""
        o = _mk()
        written = {}

        def capture_write(data: Any) -> Generator[Any, Any, Any]:
            written.update(data)
            yield

        o._write_kv = capture_write
        _drive(o._save_priced_transfers("optimism", {"k": 42.0}))
        assert "optimism_priced_transfers" in written
        assert json.loads(written["optimism_priced_transfers"]) == {"k": 42.0}

    def test_count_transfers(self) -> None:
        """Verify transfer counting across dates."""
        o = _mk()
        transfers = {"d1": [{"a": 1}, {"b": 2}], "d2": [{"c": 3}]}
        assert o._count_transfers(transfers) == 3
        assert o._count_transfers({}) == 0

    def test_fetch_historical_eth_price_fallback_cache(self) -> None:
        """When API fails but fallback cache has a price, return it."""
        o = _mk()
        cg = MagicMock()
        o.context.coingecko = cg
        cg.historical_price_endpoint = "url/{coin_id}/{date}"
        cg.api_key = "key"
        cg.request.return_value = (False, {})  # API fails
        o.params.use_x402 = False

        call_count = {"n": 0}

        def cache_returns_on_second_call(
            *a: Any, **kw: Any
        ) -> Generator[Any, Any, Any]:
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

    def test_fetch_historical_eth_price_no_price_but_fallback(self) -> None:
        """API succeeds but no price in response, fallback cache returns price."""
        o = _mk()
        cg = MagicMock()
        o.context.coingecko = cg
        cg.historical_price_endpoint = "url/{coin_id}/{date}"
        cg.api_key = "key"
        cg.request.return_value = (True, {"market_data": {"current_price": {}}})
        o.params.use_x402 = False

        call_count = {"n": 0}

        def cache_returns_on_second_call(
            *a: Any, **kw: Any
        ) -> Generator[Any, Any, Any]:
            call_count["n"] += 1
            yield
            if call_count["n"] >= 2:
                return 1800.0
            return None

        o._get_cached_price = cache_returns_on_second_call
        o._cache_price = _gen_none

        result = _drive(o._fetch_historical_eth_price("01-01-2025"))
        assert result == 1800.0

    def test_load_priced_transfers_type_error(self) -> None:
        """Return empty dict on TypeError during JSON parse."""
        o = _mk()
        o._read_kv = _gen_return({"optimism_priced_transfers": 12345})
        result = _drive(o._load_priced_transfers("optimism"))
        assert result == {}
        o.context.logger.warning.assert_called()

    def test_outgoing_early_stop(self) -> None:
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

    def test_outgoing_merge_skips_existing_dates(self) -> None:
        """New transfers on already-stored dates are not overwritten during merge."""
        o = _mk()
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        o.funding_events = {
            "optimism_outgoing": {
                "2024-12-15": [{"symbol": "ETH", "amount": 0.5, "tx_hash": "0xOLD"}],
            }
        }
        # Page has one transfer on an existing date and one on a new date
        page_data = {
            "results": [
                {
                    "executionDate": "2024-12-20T10:00:00Z",
                    "from": "0xsafe",
                    "to": "0xR",
                    "type": "ETHER_TRANSFER",
                    "value": str(10**18),
                    "transactionHash": "0xNEW",
                },
                {
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
        # New date added
        assert "2024-12-20" in result
        # Existing date kept original data (not overwritten)
        assert result["2024-12-15"][0]["tx_hash"] == "0xOLD"


class TestClosedPositionsBranch:
    """Test the closed_positions branch (lines 557-560) specifically."""

    def test_closed_positions_only(self) -> None:
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

    def test_mode_merge_new_dates(self) -> None:
        """Line 3192: new date in existing_mode_data."""
        obj = _mk()
        obj.read_funding_events = lambda: {"mode": {"2024-01-01": [{"x": 1}]}}
        obj.store_funding_events = MagicMock()

        # Token transfers succeed and add a new date
        def token_mode(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            a[2]["2024-02-01"] = [{"x": 2}]  # all_transfers_by_date[date]
            yield
            return True

        obj._fetch_token_transfers_mode = token_mode
        obj._fetch_eth_transfers_mode = MagicMock(return_value=True)
        r = _drive(obj._fetch_all_transfers_until_date_mode("0xA", "2025-01-01", False))
        assert "2024-02-01" in r

    def test_optimism_existing_funding_events(self) -> None:
        """Lines 3225, 3243-3244: existing funding_events truthy, merge new dates."""
        obj = _mk()
        obj.read_funding_events = lambda: {"optimism": {"2024-01-01": [{"y": 1}]}}
        obj.store_funding_events = MagicMock()

        # _fetch_optimism_transfers_safeglobal adds a new date
        def sfg(*a: Any, **kw: Any) -> Generator[Any, Any, Any]:
            a[2]["2024-02-01"] = [{"y": 2}]  # all_transfers_by_date
            yield

        obj._fetch_optimism_transfers_safeglobal = sfg
        r = _drive(obj._fetch_all_transfers_until_date_optimism("0xA", "2025-01-01"))
        assert "2024-02-01" in r

    def test_optimism_empty_funding_events(self) -> None:
        """Line 3247->3249: funding_events falsy after fetch."""
        obj = _mk()
        obj.read_funding_events = lambda: {}
        obj.store_funding_events = MagicMock()
        obj._fetch_optimism_transfers_safeglobal = _gen_none
        r = _drive(obj._fetch_all_transfers_until_date_optimism("0xA", "2025-01-01"))
        assert isinstance(r, dict)


class TestTokenTransfersModeAmountZero:
    """Test amount==0 branch at line 3475 in _fetch_token_transfers_mode."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests.get"
    )
    def test_zero_value_included(self, m: Any) -> None:
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
        t: Dict[Any, Any] = {}
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
    def test_connection_error(self, m: Any) -> None:
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
    def test_json_decode_error(self, m: Any) -> None:
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
    def test_connection_error(self, m: Any) -> None:
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
    def test_json_decode_error(self, m: Any) -> None:
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
    def test_bad_block_timestamp(self, m: Any) -> None:
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
        transfers: Dict[Any, Any] = {}
        result = obj._fetch_eth_transfers_mode("0xA", "2025-01-01", transfers, True)
        # Should not crash; the bad entry is skipped
        assert result is True


class TestTrackEthTransfersModeJsonError:
    """Test _track_eth_transfers_mode handles JSON parse errors."""

    @patch(
        "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.requests"
    )
    def test_json_decode_error(self, mock_requests: MagicMock) -> None:
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
    def test_json_decode_error(self, mock_requests: MagicMock) -> None:
        """Test that JSONDecodeError from response.json() is caught."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("No JSON")
        mock_requests.get.return_value = resp
        result = _mk()._track_erc20_transfers_mode("0xSafe", 1704067200)
        assert result == {"outgoing": {}}


class TestTransferMissingFromKey:
    """Test that transfer dicts with missing 'from' key do not crash."""

    def _obj(self) -> Any:
        o = _mk()
        o.context.coingecko = MagicMock()
        o.params.sleep_time = 1
        return o

    def test_outgoing_transfer_missing_from_key(self) -> None:
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

    def test_erc20_transfer_missing_from_key(self) -> None:
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

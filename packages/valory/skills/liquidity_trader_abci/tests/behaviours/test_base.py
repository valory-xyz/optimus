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

"""Test the behaviours/base.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import json
import os
import tempfile
import time
from typing import Any, Dict, Generator, List, Optional, Tuple
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    AIRDROP_TOTAL_KEY,
    Action,
    ETH_REMAINING_KEY,
    GasCostTracker,
    LiquidityTraderBaseBehaviour,
    MIN_TIME_IN_POSITION,
    OLAS_ADDRESSES,
    PRICE_CACHE_KEY_PREFIX,
    REWARD_UPDATE_INTERVAL,
    REWARD_UPDATE_KEY_PREFIX,
    ZERO_ADDRESS,
    execute_strategy,
)
from packages.valory.skills.liquidity_trader_abci.states.base import StakingState


def _make_behaviour(**overrides: Any) -> LiquidityTraderBaseBehaviour:
    """Create a LiquidityTraderBaseBehaviour without calling __init__."""
    b = object.__new__(LiquidityTraderBaseBehaviour)

    params = MagicMock()
    params.safe_contract_addresses = {
        "optimism": "0x" + "aa" * 20,
        "mode": "0x" + "bb" * 20,
    }
    params.multisend_contract_addresses = {
        "optimism": "0x" + "cc" * 20,
        "mode": "0x" + "dd" * 20,
    }
    params.target_investment_chains = ["optimism", "mode"]
    params.safe_api_base_url = "https://safe.optimism.io/api/v1/safes"
    params.mode_explorer_api_base_url = "https://explorer.mode.network"
    params.slippage_tolerance = 0.05
    params.velodrome_router_contract_addresses = {}
    params.velodrome_slipstream_helper_contract_addresses = {}
    params.staking_token_contract_address = "0x" + "11" * 20
    params.staking_activity_checker_contract_address = "0x" + "22" * 20
    params.staking_chain = "optimism"
    params.on_chain_service_id = 42
    params.staking_subgraph_endpoints = {"optimism": "https://subgraph.example.com"}
    params.use_x402 = False
    params.store_path = "/tmp"
    params.pool_info_filename = "pool.json"
    params.portfolio_info_filename = "portfolio.json"
    params.whitelisted_assets_filename = "assets.json"
    params.funding_events_filename = "funding.json"
    params.agent_performance_filename = "perf.json"
    params.gas_cost_info_filename = "gas.json"

    synced = MagicMock()
    synced.service_staking_state = StakingState.STAKED.value
    synced.min_num_of_safe_tx_required = 5
    synced.period_count = 1

    coingecko = MagicMock()
    coingecko.api_key = "test_key"
    coingecko.chain_to_platform_id_mapping = {
        "optimism": "optimistic-ethereum",
        "mode": "mode",
    }
    coingecko.token_price_endpoint = "simple/token_price/{asset_platform_id}?contract_addresses={token_address}&vs_currencies=usd"
    coingecko.coin_price_endpoint = "simple/price?ids={coin_id}&vs_currencies=usd"
    coingecko.historical_price_endpoint = "coins/{coin_id}/history?date={date}"
    coingecko.historical_market_data_endpoint = (
        "coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    )

    ctx = MagicMock()
    ctx.agent_address = "test_agent"
    ctx.logger = MagicMock()
    ctx.params = params
    ctx.coingecko = coingecko

    shared_state = MagicMock()
    shared_state.synchronized_data = synced
    ctx.state = shared_state

    round_seq = MagicMock()
    round_seq.last_round_transition_timestamp.timestamp.return_value = 1700000000.0

    b.__dict__.update(
        {
            "_context": ctx,
            "current_positions": [],
            "pools": {},
            "portfolio_data": {},
            "whitelisted_assets": {},
            "funding_events": {},
            "agent_performance": {},
            "_current_entry_costs": 0.0,
            "service_staking_state": StakingState.UNSTAKED,
            "gas_cost_tracker": GasCostTracker(file_path="/tmp/gas.json"),
            "initial_investment_values_per_pool": {},
            "_inflight_strategy_req": None,
            "round_sequence": round_seq,
            "current_positions_filepath": "/tmp/pool.json",
            "portfolio_data_filepath": "/tmp/portfolio.json",
            "whitelisted_assets_filepath": "/tmp/assets.json",
            "funding_events_filepath": "/tmp/funding.json",
            "agent_performance_filepath": "/tmp/perf.json",
        }
    )

    b.__dict__.update(overrides)
    return b


def _exhaust(gen, sends=None):
    """Drive a generator to completion, returning its return value."""
    if not hasattr(gen, "__next__"):
        return gen
    sends = sends or []
    try:
        next(gen)
        for s in sends:
            gen.send(s)
        while True:
            gen.send(None)
    except StopIteration as e:
        return e.value


def _make_gen(return_value):
    """Create a generator function that yields once and returns return_value."""

    def method(*args, **kwargs):
        yield
        return return_value

    return method


def _make_gen_none():
    """Generator that returns None."""

    def method(*args, **kwargs):
        if False:
            yield
        return None

    return method


def test_import() -> None:
    """Test that the behaviours base module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.behaviours.base  # noqa


class TestGasCostTracker:
    """Test GasCostTracker class."""

    def test_init(self) -> None:
        tracker = GasCostTracker(file_path="/tmp/gas_costs.json")
        assert tracker.file_path == "/tmp/gas_costs.json"
        assert tracker.data == {}

    def test_log_gas_usage_new_chain(self) -> None:
        tracker = GasCostTracker(file_path="/tmp/gas_costs.json")
        tracker.log_gas_usage("1", 1000, "0xhash", 21000, 50)
        assert "1" in tracker.data
        assert len(tracker.data["1"]) == 1
        assert tracker.data["1"][0]["tx_hash"] == "0xhash"
        assert tracker.data["1"][0]["gas_used"] == 21000
        assert tracker.data["1"][0]["gas_price"] == 50
        assert tracker.data["1"][0]["timestamp"] == 1000

    def test_log_gas_usage_existing_chain(self) -> None:
        tracker = GasCostTracker(file_path="/tmp/gas_costs.json")
        tracker.log_gas_usage("1", 1000, "0xhash1", 21000, 50)
        tracker.log_gas_usage("1", 2000, "0xhash2", 22000, 55)
        assert len(tracker.data["1"]) == 2

    def test_log_gas_usage_max_records(self) -> None:
        tracker = GasCostTracker(file_path="/tmp/gas_costs.json")
        for i in range(25):
            tracker.log_gas_usage("1", i * 100, f"0xhash{i}", 21000, 50)
        assert len(tracker.data["1"]) == GasCostTracker.MAX_RECORDS
        assert tracker.data["1"][0]["timestamp"] == 500
        assert tracker.data["1"][-1]["timestamp"] == 2400

    def test_update_data(self) -> None:
        tracker = GasCostTracker(file_path="/tmp/gas_costs.json")
        tracker.log_gas_usage("1", 1000, "0xhash", 21000, 50)
        new_data = {"2": [{"timestamp": 2000, "tx_hash": "0xnew"}]}
        tracker.update_data(new_data)
        assert tracker.data == new_data
        assert "1" not in tracker.data


class TestProperties:
    """Test synchronized_data, params, shared_state properties."""

    def test_synchronized_data(self) -> None:
        b = _make_behaviour()
        assert b.synchronized_data is not None

    def test_params(self) -> None:
        b = _make_behaviour()
        assert b.params is not None

    def test_shared_state(self) -> None:
        b = _make_behaviour()
        assert b.shared_state is not None


class TestErrorMethods:
    """Test error logging methods."""

    def test_default_error_logs(self) -> None:
        b = _make_behaviour()
        b.default_error("contract_id", "callable_name", MagicMock())
        b.context.logger.error.assert_called_once()

    def test_contract_interaction_error_info(self) -> None:
        b = _make_behaviour()
        msg = MagicMock()
        msg.raw_transaction.body = {"info": "informational"}
        b.contract_interaction_error("cid", "call", msg)
        b.context.logger.info.assert_called()

    def test_contract_interaction_error_warning(self) -> None:
        b = _make_behaviour()
        msg = MagicMock()
        msg.raw_transaction.body = {"warning": "watch out"}
        b.contract_interaction_error("cid", "call", msg)
        b.context.logger.warning.assert_called()

    def test_contract_interaction_error_error(self) -> None:
        b = _make_behaviour()
        msg = MagicMock()
        msg.raw_transaction.body = {"error": "something broke"}
        b.contract_interaction_error("cid", "call", msg)
        b.context.logger.error.assert_called()

    def test_contract_interaction_error_fallback(self) -> None:
        b = _make_behaviour()
        msg = MagicMock()
        msg.raw_transaction.body = {}
        b.contract_interaction_error("cid", "call", msg)
        b.context.logger.error.assert_called()


class TestConvertToTokenUnits:
    """Test _convert_to_token_units."""

    def test_standard_18_decimals(self) -> None:
        b = _make_behaviour()
        result = b._convert_to_token_units(10**18, 18)
        assert result == f"{1.0:.18f}"

    def test_6_decimals(self) -> None:
        b = _make_behaviour()
        result = b._convert_to_token_units(1_000_000, 6)
        assert result == f"{1.0:.6f}"

    def test_none_amount(self) -> None:
        b = _make_behaviour()
        result = b._convert_to_token_units(None, 18)
        assert result is None

    def test_none_decimal(self) -> None:
        b = _make_behaviour()
        result = b._convert_to_token_units(1000, None)
        assert result is None


class TestGetPriceCacheKey:
    """Test _get_price_cache_key."""

    def test_with_date(self) -> None:
        b = _make_behaviour()
        key = b._get_price_cache_key("0xABC", "01-01-2025")
        assert key == f"{PRICE_CACHE_KEY_PREFIX}0xabc_01-01-2025"

    def test_with_none_date(self) -> None:
        b = _make_behaviour()
        key = b._get_price_cache_key("0xABC", None)
        assert key == f"{PRICE_CACHE_KEY_PREFIX}0xabc_None"


class TestGetBalance:
    """Test _get_balance."""

    def test_no_positions(self) -> None:
        b = _make_behaviour()
        assert b._get_balance("optimism", "0xToken", None) is None

    def test_empty_positions(self) -> None:
        b = _make_behaviour()
        assert b._get_balance("optimism", "0xToken", []) is None

    def test_matching_token(self) -> None:
        b = _make_behaviour()
        positions = [
            {
                "chain": "optimism",
                "assets": [{"address": "0xToken", "balance": 1000}],
            }
        ]
        assert b._get_balance("optimism", "0xToken", positions) == 1000

    def test_no_match(self) -> None:
        b = _make_behaviour()
        positions = [
            {
                "chain": "optimism",
                "assets": [{"address": "0xOther", "balance": 500}],
            }
        ]
        assert b._get_balance("optimism", "0xToken", positions) is None

    def test_case_insensitive(self) -> None:
        b = _make_behaviour()
        positions = [
            {
                "chain": "optimism",
                "assets": [{"address": "0xABCD", "balance": 999}],
            }
        ]
        assert b._get_balance("optimism", "0xabcd", positions) == 999


class TestGetActiveLpAddresses:
    """Test _get_active_lp_addresses."""

    def test_no_positions(self) -> None:
        b = _make_behaviour()
        result = b._get_active_lp_addresses()
        assert result == set()

    def test_open_positions(self) -> None:
        b = _make_behaviour(
            current_positions=[
                {"status": "open", "pool_address": "0xPool1"},
                {"status": "closed", "pool_address": "0xPool2"},
            ]
        )
        result = b._get_active_lp_addresses()
        assert "0xpool1" in result
        assert "0xpool2" not in result

    def test_no_pool_address(self) -> None:
        b = _make_behaviour(current_positions=[{"status": "open"}])
        result = b._get_active_lp_addresses()
        assert len(result) == 0


class TestGetRewardTokenAddresses:
    """Test _get_reward_token_addresses."""

    def test_known_chain(self) -> None:
        b = _make_behaviour()
        result = b._get_reward_token_addresses("optimism")
        assert len(result) > 0

    def test_unknown_chain(self) -> None:
        b = _make_behaviour()
        result = b._get_reward_token_addresses("unknown_chain")
        assert len(result) == 0


class TestGetCoinIdFromSymbol:
    """Test get_coin_id_from_symbol."""

    def test_known_symbol(self) -> None:
        b = _make_behaviour()
        # USDC is in COIN_ID_MAPPING for both chains
        result = b.get_coin_id_from_symbol("usdc", "optimism")
        assert result is not None

    def test_unknown_symbol(self) -> None:
        b = _make_behaviour()
        result = b.get_coin_id_from_symbol("XYZTOKEN", "optimism")
        assert result is None

    def test_unknown_chain(self) -> None:
        b = _make_behaviour()
        result = b.get_coin_id_from_symbol("usdc", "nonexistent")
        assert result is None


class TestChainAddresses:
    """Test _get_usdc_address and _get_olas_address."""

    def test_usdc_optimism(self) -> None:
        b = _make_behaviour()
        addr = b._get_usdc_address("optimism")
        assert addr is not None

    def test_usdc_mode(self) -> None:
        b = _make_behaviour()
        addr = b._get_usdc_address("mode")
        assert addr is not None

    def test_usdc_unknown_chain(self) -> None:
        b = _make_behaviour()
        addr = b._get_usdc_address("unknown")
        assert addr is None

    def test_olas_optimism(self) -> None:
        b = _make_behaviour()
        addr = b._get_olas_address("optimism")
        assert addr is not None

    def test_olas_mode(self) -> None:
        b = _make_behaviour()
        addr = b._get_olas_address("mode")
        assert addr is not None

    def test_olas_unknown(self) -> None:
        b = _make_behaviour()
        addr = b._get_olas_address("unknown")
        assert addr is None


class TestHasStakingMetadata:
    """Test _has_staking_metadata."""

    def test_with_gauge_address(self) -> None:
        b = _make_behaviour()
        assert b._has_staking_metadata({"gauge_address": "0x123"}) is True

    def test_with_staked_true(self) -> None:
        b = _make_behaviour()
        assert b._has_staking_metadata({"staked": True}) is True

    def test_with_staked_amount(self) -> None:
        b = _make_behaviour()
        assert b._has_staking_metadata({"staked_amount": 100}) is True

    def test_no_metadata(self) -> None:
        b = _make_behaviour()
        assert b._has_staking_metadata({}) is False

    def test_zero_staked_amount(self) -> None:
        b = _make_behaviour()
        assert b._has_staking_metadata({"staked_amount": 0}) is False


class TestIsAirdropTransfer:
    """Test _is_airdrop_transfer."""

    def test_valid_airdrop(self) -> None:
        b = _make_behaviour()
        transfer = {
            "from_address": "0xAirdropContract",
            "token_address": OLAS_ADDRESSES["mode"],
            "symbol": "OLAS",
        }
        assert b._is_airdrop_transfer(transfer, "0xAirdropContract") is True

    def test_wrong_sender(self) -> None:
        b = _make_behaviour()
        transfer = {
            "from_address": "0xOther",
            "token_address": OLAS_ADDRESSES["mode"],
            "symbol": "OLAS",
        }
        assert b._is_airdrop_transfer(transfer, "0xAirdropContract") is False

    def test_wrong_symbol(self) -> None:
        b = _make_behaviour()
        transfer = {
            "from_address": "0xAirdropContract",
            "token_address": OLAS_ADDRESSES["mode"],
            "symbol": "USDC",
        }
        assert b._is_airdrop_transfer(transfer, "0xAirdropContract") is False

    def test_no_airdrop_contract(self) -> None:
        b = _make_behaviour()
        assert b._is_airdrop_transfer({}, "") is False
        assert b._is_airdrop_transfer({}, None) is False


class TestTimeCalculations:
    """Test time-related methods."""

    def test_calculate_days_since_entry(self) -> None:
        b = _make_behaviour()
        b._get_current_timestamp = lambda: 1700000000 + 3 * 24 * 3600
        result = b._calculate_days_since_entry(1700000000)
        assert abs(result - 3.0) < 0.01

    def test_check_minimum_time_met_no_timestamp(self) -> None:
        b = _make_behaviour()
        assert b._check_minimum_time_met({}) is True

    def test_check_minimum_time_met_zero_days(self) -> None:
        b = _make_behaviour()
        assert (
            b._check_minimum_time_met({"enter_timestamp": 100, "min_hold_days": 0})
            is True
        )

    def test_check_minimum_time_met_true(self) -> None:
        b = _make_behaviour()
        b._get_current_timestamp = lambda: 1700000000 + 30 * 24 * 3600
        result = b._check_minimum_time_met(
            {"enter_timestamp": 1700000000, "min_hold_days": 21.0}
        )
        assert result is True

    def test_check_minimum_time_met_false(self) -> None:
        b = _make_behaviour()
        b._get_current_timestamp = lambda: 1700000000 + 1 * 24 * 3600
        result = b._check_minimum_time_met(
            {"enter_timestamp": 1700000000, "min_hold_days": 21.0}
        )
        assert result is False


class TestGetEntryCostsKey:
    """Test _get_entry_costs_key."""

    def test_format(self) -> None:
        b = _make_behaviour()
        key = b._get_entry_costs_key("optimism", "pool123")
        assert key == "entry_costs_optimism_pool123"


class TestGetCurrentTimestamp:
    """Test _get_current_timestamp."""

    def test_returns_int(self) -> None:
        b = _make_behaviour()
        ts = b._get_current_timestamp()
        assert isinstance(ts, int)
        assert ts > 0


class TestNameGeneration:
    """Test name generation methods."""

    def test_generate_phonetic_syllable(self) -> None:
        b = _make_behaviour()
        result = b.generate_phonetic_syllable(0)
        assert isinstance(result, str)
        assert len(result) >= 2

    def test_generate_phonetic_syllable_wraps(self) -> None:
        b = _make_behaviour()
        r1 = b.generate_phonetic_syllable(0)
        r2 = b.generate_phonetic_syllable(1000)
        # Wrapping via modulo
        assert isinstance(r2, str)

    def test_generate_phonetic_name(self) -> None:
        b = _make_behaviour()
        # Needs hex address string of sufficient length
        address = "0x" + "a1b2c3d4" * 10
        result = b.generate_phonetic_name(address, 2, 2)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_name(self) -> None:
        b = _make_behaviour()
        address = "0x" + "a1b2c3d4" * 10
        result = b.generate_name(address)
        assert isinstance(result, str)
        assert "-" in result  # first_name-last_name_prefixNN format

    def test_generate_name_deterministic(self) -> None:
        b = _make_behaviour()
        address = "0x1234567890abcdef1234567890abcdef12345678"
        r1 = b.generate_name(address)
        r2 = b.generate_name(address)
        assert r1 == r2


class TestStoreReadData:
    """Test _store_data and _read_data."""

    def test_store_data_none(self) -> None:
        b = _make_behaviour()
        b._store_data(None, "test", "/tmp/test.json")
        b.context.logger.warning.assert_called()

    def test_store_and_read_data(self) -> None:
        b = _make_behaviour()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            b._store_data({"key": "value"}, "test", filepath)
            b.test = None
            b._read_data("test", filepath)
            assert b.test == {"key": "value"}
        finally:
            os.unlink(filepath)

    def test_read_data_file_not_found(self) -> None:
        b = _make_behaviour()
        with tempfile.TemporaryDirectory() as td:
            filepath = os.path.join(td, "nonexistent_test_file_abc123.json")
            b._read_data("current_positions", filepath)
            # Should create the file with []
            assert os.path.exists(filepath)
            with open(filepath) as f:
                assert json.load(f) == []

    def test_read_data_json_decode_error(self) -> None:
        b = _make_behaviour()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{{{")
            filepath = f.name
        try:
            b._read_data("test_attr", filepath)
            b.context.logger.error.assert_called()
        finally:
            os.unlink(filepath)

    def test_read_data_attribute_not_exist(self) -> None:
        b = _make_behaviour()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "val"}, f)
            filepath = f.name
        try:
            b._read_data("nonexistent_attribute_xyz", filepath)
            b.context.logger.warning.assert_called()
        finally:
            os.unlink(filepath)

    def test_read_data_class_object(self) -> None:
        """Test _read_data with class_object=True calls update_data."""
        b = _make_behaviour()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"chain1": [{"tx": "0x1"}]}, f)
            filepath = f.name
        try:
            b._read_data("gas_cost_tracker", filepath, class_object=True)
            assert b.gas_cost_tracker.data == {"chain1": [{"tx": "0x1"}]}
        finally:
            os.unlink(filepath)

    def test_store_data_io_error(self) -> None:
        b = _make_behaviour()
        b._store_data({"key": "val"}, "test", "/nonexistent_dir/file.json")
        b.context.logger.error.assert_called()


class TestStoreReadConvenience:
    """Test store_* and read_* methods."""

    def test_store_current_positions(self) -> None:
        b = _make_behaviour()
        b.current_positions = [{"pool": "test"}]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            b.current_positions_filepath = f.name
        try:
            b.store_current_positions()
            with open(b.current_positions_filepath) as f:
                assert json.load(f) == [{"pool": "test"}]
        finally:
            os.unlink(b.current_positions_filepath)

    def test_store_whitelisted_assets(self) -> None:
        b = _make_behaviour()
        b.whitelisted_assets = {"mode": {"0x1": "USDC"}}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            b.whitelisted_assets_filepath = f.name
        try:
            b.store_whitelisted_assets()
            with open(b.whitelisted_assets_filepath) as f:
                assert json.load(f) == {"mode": {"0x1": "USDC"}}
        finally:
            os.unlink(b.whitelisted_assets_filepath)

    def test_store_funding_events(self) -> None:
        b = _make_behaviour()
        b.funding_events = {"event": "test"}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            b.funding_events_filepath = f.name
        try:
            b.store_funding_events()
        finally:
            os.unlink(b.funding_events_filepath)

    def test_store_gas_costs(self) -> None:
        b = _make_behaviour()
        b.gas_cost_tracker.data = {"1": []}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            b.gas_cost_tracker.file_path = f.name
        try:
            b.store_gas_costs()
        finally:
            os.unlink(b.gas_cost_tracker.file_path)

    def test_store_portfolio_data(self) -> None:
        b = _make_behaviour()
        b.portfolio_data = {"total": 100}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            b.portfolio_data_filepath = f.name
        try:
            b.store_portfolio_data()
        finally:
            os.unlink(b.portfolio_data_filepath)

    def test_read_current_positions(self) -> None:
        b = _make_behaviour()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"pool": "test"}], f)
            b.current_positions_filepath = f.name
        try:
            b.read_current_positions()
            assert b.current_positions == [{"pool": "test"}]
        finally:
            os.unlink(b.current_positions_filepath)

    def test_read_gas_costs(self) -> None:
        b = _make_behaviour()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"chain1": []}, f)
            b.gas_cost_tracker.file_path = f.name
        try:
            b.read_gas_costs()
            assert b.gas_cost_tracker.data == {"chain1": []}
        finally:
            os.unlink(b.gas_cost_tracker.file_path)


class TestAgentPerformance:
    """Test agent performance methods."""

    def test_initialize(self) -> None:
        b = _make_behaviour()
        b.initialize_agent_performance()
        assert b.agent_performance["timestamp"] is None
        assert b.agent_performance["metrics"] == []

    def test_update_timestamp(self) -> None:
        b = _make_behaviour()
        b.agent_performance = {"timestamp": None}
        b.update_agent_performance_timestamp()
        assert b.agent_performance["timestamp"] is not None
        assert isinstance(b.agent_performance["timestamp"], int)

    def test_read_agent_performance_empty(self) -> None:
        b = _make_behaviour()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            b.agent_performance_filepath = f.name
        try:
            b.read_agent_performance()
            # Empty dict -> should initialize
            assert "timestamp" in b.agent_performance
        finally:
            os.unlink(b.agent_performance_filepath)

    def test_read_agent_performance_exception(self) -> None:
        b = _make_behaviour()
        b.agent_performance_filepath = "/tmp/nonexistent_dir_xyz/perf.json"
        b._read_data = MagicMock(side_effect=Exception("fail"))
        b.read_agent_performance()
        assert "timestamp" in b.agent_performance


class TestCalculateRateLimitWaitTime:
    """Test _calculate_rate_limit_wait_time."""

    def test_no_rate_limiter(self) -> None:
        b = _make_behaviour()
        del b.context.coingecko.rate_limiter
        assert b._calculate_rate_limit_wait_time() == 0

    def test_no_credits(self) -> None:
        b = _make_behaviour()
        b.context.coingecko.rate_limiter.no_credits = True
        assert b._calculate_rate_limit_wait_time() == 0

    def test_rate_limited(self) -> None:
        b = _make_behaviour()
        b.context.coingecko.rate_limiter.no_credits = False
        b.context.coingecko.rate_limiter.rate_limited = True
        b.context.coingecko.rate_limiter.last_request_time = time.time() - 30
        result = b._calculate_rate_limit_wait_time()
        assert 0 <= result <= 60

    def test_not_rate_limited(self) -> None:
        b = _make_behaviour()
        b.context.coingecko.rate_limiter.no_credits = False
        b.context.coingecko.rate_limiter.rate_limited = False
        assert b._calculate_rate_limit_wait_time() == 0


class TestContractInteract:
    """Test contract_interact generator."""

    def test_success(self) -> None:
        b = _make_behaviour()
        from packages.valory.protocols.contract_api import ContractApiMessage

        resp = MagicMock()
        resp.performative = ContractApiMessage.Performative.RAW_TRANSACTION
        resp.raw_transaction.body = {"data": "result_value"}
        b.get_contract_api_response = _make_gen(resp)
        result = _exhaust(
            b.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address="0x123",
                contract_public_id=MagicMock(),
                contract_callable="test_method",
                data_key="data",
            )
        )
        assert result == "result_value"

    def test_wrong_performative(self) -> None:
        b = _make_behaviour()
        from packages.valory.protocols.contract_api import ContractApiMessage

        resp = MagicMock()
        resp.performative = ContractApiMessage.Performative.ERROR
        b.get_contract_api_response = _make_gen(resp)
        result = _exhaust(
            b.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address="0x123",
                contract_public_id=MagicMock(),
                contract_callable="test_method",
                data_key="data",
            )
        )
        assert result is None

    def test_missing_data_key(self) -> None:
        b = _make_behaviour()
        from packages.valory.protocols.contract_api import ContractApiMessage

        resp = MagicMock()
        resp.performative = ContractApiMessage.Performative.RAW_TRANSACTION
        resp.raw_transaction.body = {"other_key": "val"}
        b.get_contract_api_response = _make_gen(resp)
        result = _exhaust(
            b.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address="0x123",
                contract_public_id=MagicMock(),
                contract_callable="test_method",
                data_key="data",
            )
        )
        assert result is None


class TestGetPositions:
    """Test get_positions generator."""

    def test_optimism_and_mode(self) -> None:
        b = _make_behaviour()
        b.params.target_investment_chains = ["optimism", "mode"]
        b._get_optimism_balances_from_safe_api = _make_gen(
            [{"asset_symbol": "USDC", "balance": 100}]
        )
        b._get_mode_balances_from_explorer_api = _make_gen(
            [{"asset_symbol": "ETH", "balance": 200}]
        )
        result = _exhaust(b.get_positions())
        assert len(result) == 2

    def test_empty_balances(self) -> None:
        b = _make_behaviour()
        b.params.target_investment_chains = ["optimism"]
        b._get_optimism_balances_from_safe_api = _make_gen([])
        result = _exhaust(b.get_positions())
        assert result == []


class TestGetOptimismBalances:
    """Test _get_optimism_balances_from_safe_api."""

    def test_no_safe_address(self) -> None:
        b = _make_behaviour()
        b.params.safe_contract_addresses = {}
        result = _exhaust(b._get_optimism_balances_from_safe_api())
        assert result == []

    def test_with_balances(self) -> None:
        b = _make_behaviour()
        token_addr = "0x" + "ab" * 20
        b._fetch_safe_balances_with_pagination = _make_gen(
            [
                {"tokenAddress": None, "balance": "1000"},
                {
                    "tokenAddress": token_addr,
                    "token": {"symbol": "USDC"},
                    "balance": "500",
                },
            ]
        )
        b._fetch_reward_balances = _make_gen([])
        b._fetch_ousdt_balance = _make_gen(None)
        result = _exhaust(b._get_optimism_balances_from_safe_api())
        assert len(result) == 2
        assert result[0]["asset_symbol"] == "ETH"
        assert result[1]["asset_symbol"] == "USDC"

    def test_skips_filtered_token(self) -> None:
        b = _make_behaviour()
        b._fetch_safe_balances_with_pagination = _make_gen(
            [
                {
                    "tokenAddress": "0xfAf87e196A29969094bE35DfB0Ab9d0b8518dB84",
                    "token": {"symbol": "SKIP"},
                    "balance": "100",
                },
            ]
        )
        b._fetch_reward_balances = _make_gen([])
        b._fetch_ousdt_balance = _make_gen(None)
        result = _exhaust(b._get_optimism_balances_from_safe_api())
        assert len(result) == 0

    def test_skips_token_without_info(self) -> None:
        """ERC-20 token with no token info is skipped."""
        b = _make_behaviour()
        b._fetch_safe_balances_with_pagination = _make_gen(
            [
                {
                    "tokenAddress": "0x" + "ab" * 20,
                    "token": None,
                    "balance": "100",
                },
            ]
        )
        b._fetch_reward_balances = _make_gen([])
        b._fetch_ousdt_balance = _make_gen(None)
        result = _exhaust(b._get_optimism_balances_from_safe_api())
        assert len(result) == 0

    def test_with_reward_and_ousdt(self) -> None:
        b = _make_behaviour()
        b._fetch_safe_balances_with_pagination = _make_gen([])
        b._fetch_reward_balances = _make_gen([{"asset_symbol": "VELO", "balance": 100}])
        b._fetch_ousdt_balance = _make_gen({"asset_symbol": "oUSDT", "balance": 50})
        result = _exhaust(b._get_optimism_balances_from_safe_api())
        assert len(result) == 2


class TestGetModeBalances:
    """Test _get_mode_balances_from_explorer_api."""

    def test_no_safe_address(self) -> None:
        b = _make_behaviour()
        b.params.safe_contract_addresses = {}
        result = _exhaust(b._get_mode_balances_from_explorer_api())
        assert result == []

    def test_with_eth_and_tokens(self) -> None:
        b = _make_behaviour()
        b._get_native_balance = _make_gen(10**18)
        b._fetch_mode_token_balances = _make_gen(
            [{"asset_symbol": "USDC", "balance": 1000}]
        )
        result = _exhaust(b._get_mode_balances_from_explorer_api())
        assert len(result) == 2
        assert result[0]["asset_symbol"] == "ETH"

    def test_zero_eth(self) -> None:
        b = _make_behaviour()
        b._get_native_balance = _make_gen(0)
        b._fetch_mode_token_balances = _make_gen([])
        result = _exhaust(b._get_mode_balances_from_explorer_api())
        assert len(result) == 0


class TestFetchModeTokenBalances:
    """Test _fetch_mode_token_balances."""

    def test_renames_xvelo(self) -> None:
        b = _make_behaviour()
        b.current_positions = []
        token_addr = "0x" + "cd" * 20
        b._fetch_mode_tokens_with_pagination = _make_gen(
            [{"token": {"address": token_addr, "symbol": "XVELO"}, "value": "1000"}]
        )
        result = _exhaust(b._fetch_mode_token_balances("0xSafe"))
        assert len(result) == 1
        assert result[0]["asset_symbol"] == "VELO"

    def test_filters_lp_tokens(self) -> None:
        lp_addr = "0x" + "ee" * 20
        b = _make_behaviour()
        b.current_positions = [{"status": "open", "pool_address": lp_addr}]
        b._fetch_mode_tokens_with_pagination = _make_gen(
            [{"token": {"address": lp_addr, "symbol": "LP"}, "value": "1000"}]
        )
        result = _exhaust(b._fetch_mode_token_balances("0xSafe"))
        assert len(result) == 0

    def test_filters_zero_balance(self) -> None:
        b = _make_behaviour()
        b.current_positions = []
        token_addr = "0x" + "dd" * 20
        b._fetch_mode_tokens_with_pagination = _make_gen(
            [{"token": {"address": token_addr, "symbol": "TOK"}, "value": "0"}]
        )
        result = _exhaust(b._fetch_mode_token_balances("0xSafe"))
        assert len(result) == 0

    def test_invalid_balance_value(self) -> None:
        b = _make_behaviour()
        b.current_positions = []
        token_addr = "0x" + "ff" * 20
        b._fetch_mode_tokens_with_pagination = _make_gen(
            [
                {
                    "token": {"address": token_addr, "symbol": "TOK"},
                    "value": "not_a_number",
                }
            ]
        )
        result = _exhaust(b._fetch_mode_token_balances("0xSafe"))
        assert len(result) == 0

    def test_negative_balance(self) -> None:
        """Negative balance after int conversion → not appended."""
        b = _make_behaviour()
        b.current_positions = []
        token_addr = "0x" + "cc" * 20
        b._fetch_mode_tokens_with_pagination = _make_gen(
            [{"token": {"address": token_addr, "symbol": "TOK"}, "value": "-1"}]
        )
        result = _exhaust(b._fetch_mode_token_balances("0xSafe"))
        assert len(result) == 0


class TestIsStakingKpiMet:
    """Test _is_staking_kpi_met generator."""

    def test_not_staked(self) -> None:
        b = _make_behaviour()
        b.synchronized_data.service_staking_state = StakingState.UNSTAKED.value
        result = _exhaust(b._is_staking_kpi_met())
        assert result is None

    def test_no_min_tx(self) -> None:
        b = _make_behaviour()
        b.synchronized_data.min_num_of_safe_tx_required = None
        result = _exhaust(b._is_staking_kpi_met())
        assert result is None

    def test_kpi_met(self) -> None:
        b = _make_behaviour()
        b._get_multisig_nonces_since_last_cp = _make_gen(10)
        b.synchronized_data.min_num_of_safe_tx_required = 5
        result = _exhaust(b._is_staking_kpi_met())
        assert result is True

    def test_kpi_not_met(self) -> None:
        b = _make_behaviour()
        b._get_multisig_nonces_since_last_cp = _make_gen(2)
        b.synchronized_data.min_num_of_safe_tx_required = 5
        result = _exhaust(b._is_staking_kpi_met())
        assert result is False

    def test_nonces_none(self) -> None:
        b = _make_behaviour()
        b._get_multisig_nonces_since_last_cp = _make_gen(None)
        result = _exhaust(b._is_staking_kpi_met())
        assert result is None


class TestMultisigNonces:
    """Test multisig nonce methods."""

    def test_get_nonces_success(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen([42])
        result = _exhaust(b._get_multisig_nonces("optimism", "0xMultisig"))
        assert result == 42

    def test_get_nonces_none(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(None)
        result = _exhaust(b._get_multisig_nonces("optimism", "0xMultisig"))
        assert result is None

    def test_get_nonces_empty(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen([])
        result = _exhaust(b._get_multisig_nonces("optimism", "0xMultisig"))
        assert result is None

    def test_nonces_since_last_cp_success(self) -> None:
        b = _make_behaviour()
        b._get_multisig_nonces = _make_gen(50)
        b._get_service_info = _make_gen((0, 0, (40,)))
        result = _exhaust(
            b._get_multisig_nonces_since_last_cp("optimism", "0xMultisig")
        )
        assert result == 10

    def test_nonces_since_last_cp_nonces_none(self) -> None:
        b = _make_behaviour()
        b._get_multisig_nonces = _make_gen(None)
        result = _exhaust(
            b._get_multisig_nonces_since_last_cp("optimism", "0xMultisig")
        )
        assert result is None

    def test_nonces_since_last_cp_service_info_none(self) -> None:
        b = _make_behaviour()
        b._get_multisig_nonces = _make_gen(50)
        b._get_service_info = _make_gen(None)
        result = _exhaust(
            b._get_multisig_nonces_since_last_cp("optimism", "0xMultisig")
        )
        assert result is None


class TestServiceInfo:
    """Test service info methods."""

    def test_get_service_info_no_service_id(self) -> None:
        b = _make_behaviour()
        b.params.on_chain_service_id = None
        result = _exhaust(b._get_service_info("optimism"))
        assert result is None

    def test_get_service_info_success(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen((1, 2, (3, 4)))
        result = _exhaust(b._get_service_info("optimism"))
        assert result == (1, 2, (3, 4))

    def test_get_staking_state_no_service_id(self) -> None:
        b = _make_behaviour()
        b.params.on_chain_service_id = None
        _exhaust(b._get_service_staking_state("optimism"))
        assert b.service_staking_state == StakingState.UNSTAKED

    def test_get_staking_state_none_response(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(None)
        _exhaust(b._get_service_staking_state("optimism"))
        assert b.service_staking_state == StakingState.UNSTAKED

    def test_get_staking_state_staked(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(StakingState.STAKED.value)
        _exhaust(b._get_service_staking_state("optimism"))
        assert b.service_staking_state == StakingState.STAKED


class TestLiveness:
    """Test liveness methods."""

    def test_liveness_ratio_valid(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(1000)
        result = _exhaust(b._get_liveness_ratio("optimism"))
        assert result == 1000

    def test_liveness_ratio_zero(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(0)
        result = _exhaust(b._get_liveness_ratio("optimism"))
        assert result == 0
        b.context.logger.error.assert_called()

    def test_liveness_ratio_none(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(None)
        result = _exhaust(b._get_liveness_ratio("optimism"))
        assert result is None

    def test_liveness_period_valid(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(86400)
        result = _exhaust(b._get_liveness_period("optimism"))
        assert result == 86400

    def test_liveness_period_zero(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(0)
        result = _exhaust(b._get_liveness_period("optimism"))
        assert result == 0
        b.context.logger.error.assert_called()


class TestPriceCache:
    """Test price caching methods."""

    def test_get_cached_price_no_result(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        result = _exhaust(b._get_cached_price("0xToken", "01-01-2025"))
        assert result is None

    def test_get_cached_price_historical(self) -> None:
        b = _make_behaviour()
        cache_key = b._get_price_cache_key("0xToken", "01-01-2025")
        b._read_kv = _make_gen({cache_key: json.dumps({"01-01-2025": 1.5})})
        result = _exhaust(b._get_cached_price("0xToken", "01-01-2025"))
        assert result == 1.5

    def test_get_cached_price_invalid_json(self) -> None:
        b = _make_behaviour()
        cache_key = b._get_price_cache_key("0xToken", "01-01-2025")
        b._read_kv = _make_gen({cache_key: "not json"})
        result = _exhaust(b._get_cached_price("0xToken", "01-01-2025"))
        assert result is None

    def test_cache_price_new(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen({})
        b._write_kv = _make_gen(True)
        _exhaust(b._cache_price("0xToken", 1.5, "01-01-2025"))
        # No error should be raised

    def test_cache_price_existing(self) -> None:
        b = _make_behaviour()
        cache_key = b._get_price_cache_key("0xToken", "01-01-2025")
        b._read_kv = _make_gen({cache_key: json.dumps({"old_date": 1.0})})
        b._write_kv = _make_gen(True)
        _exhaust(b._cache_price("0xToken", 1.5, "01-01-2025"))

    def test_cache_price_invalid_existing(self) -> None:
        b = _make_behaviour()
        cache_key = b._get_price_cache_key("0xToken", "01-01-2025")
        b._read_kv = _make_gen({cache_key: "bad json"})
        b._write_kv = _make_gen(True)
        _exhaust(b._cache_price("0xToken", 1.5, "01-01-2025"))
        b.context.logger.error.assert_called()


class TestFetchTokenPrices:
    """Test _fetch_token_prices generator."""

    def test_fetches_normal_and_zero_address(self) -> None:
        b = _make_behaviour()
        b._fetch_zero_address_price = _make_gen(3000.0)
        b._fetch_token_price = _make_gen(1.0)
        tokens = [
            {"token": ZERO_ADDRESS, "chain": "optimism"},
            {"token": "0xUSDC", "chain": "optimism"},
        ]
        result = _exhaust(b._fetch_token_prices(tokens))
        assert ZERO_ADDRESS in result
        assert "0xUSDC" in result

    def test_missing_chain(self) -> None:
        b = _make_behaviour()
        tokens = [{"token": "0xToken"}]
        result = _exhaust(b._fetch_token_prices(tokens))
        assert result == {}

    def test_price_none_skipped(self) -> None:
        b = _make_behaviour()
        b._fetch_token_price = _make_gen(None)
        tokens = [{"token": "0xToken", "chain": "optimism"}]
        result = _exhaust(b._fetch_token_prices(tokens))
        assert result == {}


class TestBuildExitPoolAction:
    """Test _build_exit_pool_action_base."""

    def test_no_position(self) -> None:
        b = _make_behaviour()
        result = b._build_exit_pool_action_base(None)
        assert result is None

    def test_empty_position(self) -> None:
        b = _make_behaviour()
        result = b._build_exit_pool_action_base({})
        assert result is None


class TestBuildUnstakeAction:
    """Test _build_unstake_lp_tokens_action."""

    def test_non_velodrome_dex(self) -> None:
        b = _make_behaviour()
        result = b._build_unstake_lp_tokens_action({"dex_type": "uniswap"})
        assert result is None

    def test_missing_params(self) -> None:
        b = _make_behaviour()
        result = b._build_unstake_lp_tokens_action({"dex_type": "velodrome"})
        assert result is None

    def test_no_safe_address(self) -> None:
        b = _make_behaviour()
        b.params.safe_contract_addresses = {}
        result = b._build_unstake_lp_tokens_action(
            {"dex_type": "velodrome", "chain": "optimism", "pool_address": "0xPool"}
        )
        assert result is None


class TestEthRemainingAmount:
    """Test ETH remaining amount methods."""

    def test_get_not_in_kv(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        b.reset_eth_remaining_amount = _make_gen(5000)
        result = _exhaust(b.get_eth_remaining_amount())
        assert result == 5000

    def test_get_synced_with_chain(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen({ETH_REMAINING_KEY: "1000"})
        b._get_native_balance = _make_gen(2000)
        b._write_kv = _make_gen(True)
        result = _exhaust(b.get_eth_remaining_amount())
        assert result == 2000

    def test_get_cached_matches_chain(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen({ETH_REMAINING_KEY: "1000"})
        b._get_native_balance = _make_gen(1000)
        result = _exhaust(b.get_eth_remaining_amount())
        assert result == 1000

    def test_get_invalid_value(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen({ETH_REMAINING_KEY: "not_a_number"})
        b.reset_eth_remaining_amount = _make_gen(0)
        result = _exhaust(b.get_eth_remaining_amount())
        assert result == 0

    def test_reset(self) -> None:
        b = _make_behaviour()
        b._get_native_balance = _make_gen(5000)
        b._write_kv = _make_gen(True)
        result = _exhaust(b.reset_eth_remaining_amount())
        assert result == 5000

    def test_reset_none_balance(self) -> None:
        b = _make_behaviour()
        b._get_native_balance = _make_gen(None)
        b._write_kv = _make_gen(True)
        result = _exhaust(b.reset_eth_remaining_amount())
        assert result == 0

    def test_update(self) -> None:
        b = _make_behaviour()
        b.get_eth_remaining_amount = _make_gen(1000)
        b._write_kv = _make_gen(True)
        _exhaust(b.update_eth_remaining_amount(300))
        # Should have written 700


class TestShouldUpdateRewards:
    """Test should_update_rewards_from_subgraph."""

    def test_period_zero(self) -> None:
        b = _make_behaviour()
        b.synchronized_data.period_count = 0
        result = _exhaust(b.should_update_rewards_from_subgraph("optimism"))
        assert result is True

    def test_no_previous_update(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        result = _exhaust(b.should_update_rewards_from_subgraph("optimism"))
        assert result is True

    def test_recent_update(self) -> None:
        b = _make_behaviour()
        b._get_current_timestamp = lambda: 1700000000
        update_key = f"{REWARD_UPDATE_KEY_PREFIX}optimism"
        b._read_kv = _make_gen({update_key: str(1700000000 - 100)})
        result = _exhaust(b.should_update_rewards_from_subgraph("optimism"))
        assert result is False

    def test_old_update(self) -> None:
        b = _make_behaviour()
        b._get_current_timestamp = lambda: 1700000000
        update_key = f"{REWARD_UPDATE_KEY_PREFIX}optimism"
        b._read_kv = _make_gen(
            {update_key: str(1700000000 - REWARD_UPDATE_INTERVAL - 1)}
        )
        result = _exhaust(b.should_update_rewards_from_subgraph("optimism"))
        assert result is True

    def test_invalid_timestamp(self) -> None:
        b = _make_behaviour()
        update_key = f"{REWARD_UPDATE_KEY_PREFIX}optimism"
        b._read_kv = _make_gen({update_key: "not_a_number"})
        result = _exhaust(b.should_update_rewards_from_subgraph("optimism"))
        assert result is True


class TestGetAccumulatedRewards:
    """Test get_accumulated_rewards_for_token."""

    def test_no_result(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        result = _exhaust(b.get_accumulated_rewards_for_token("optimism", "0xToken"))
        assert result == 0

    def test_valid_value(self) -> None:
        b = _make_behaviour()
        key = "accumulated_rewards_optimism_0xtoken"
        b._read_kv = _make_gen({key: "12345"})
        result = _exhaust(b.get_accumulated_rewards_for_token("optimism", "0xToken"))
        assert result == 12345

    def test_none_value(self) -> None:
        b = _make_behaviour()
        key = "accumulated_rewards_optimism_0xtoken"
        b._read_kv = _make_gen({key: None})
        result = _exhaust(b.get_accumulated_rewards_for_token("optimism", "0xToken"))
        assert result == 0

    def test_invalid_value(self) -> None:
        b = _make_behaviour()
        key = "accumulated_rewards_optimism_0xtoken"
        b._read_kv = _make_gen({key: "not_int"})
        result = _exhaust(b.get_accumulated_rewards_for_token("optimism", "0xToken"))
        assert result == 0


class TestEntryCosts:
    """Test entry cost methods."""

    def test_get_all_entry_costs_empty(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        result = _exhaust(b._get_all_entry_costs())
        assert result == {}

    def test_get_all_entry_costs_valid(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(
            {"entry_costs_dict": json.dumps({"key1": 1.5, "key2": 2.0})}
        )
        result = _exhaust(b._get_all_entry_costs())
        assert result == {"key1": 1.5, "key2": 2.0}

    def test_store_entry_costs(self) -> None:
        b = _make_behaviour()
        b._get_all_entry_costs = _make_gen({})
        b._write_kv = _make_gen(True)
        _exhaust(b._store_entry_costs("optimism", "pool1", 5.0))

    def test_store_entry_costs_exception(self) -> None:
        b = _make_behaviour()
        b._get_all_entry_costs = _make_gen(None)  # Will cause error
        # This should not raise
        _exhaust(b._store_entry_costs("optimism", "pool1", 5.0))
        b.context.logger.error.assert_called()


class TestAirdropRewards:
    """Test airdrop reward methods."""

    def test_get_total_zero(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        result = _exhaust(b._get_total_airdrop_rewards("mode"))
        assert result == 0

    def test_get_total_valid(self) -> None:
        b = _make_behaviour()
        key = f"{AIRDROP_TOTAL_KEY}_mode"
        b._read_kv = _make_gen({key: "100"})
        result = _exhaust(b._get_total_airdrop_rewards("mode"))
        assert result == 100

    def test_get_total_invalid(self) -> None:
        b = _make_behaviour()
        key = f"{AIRDROP_TOTAL_KEY}_mode"
        b._read_kv = _make_gen({key: "bad"})
        result = _exhaust(b._get_total_airdrop_rewards("mode"))
        assert result == 0

    def test_update_no_dedup(self) -> None:
        b = _make_behaviour()
        b._get_total_airdrop_rewards = _make_gen(50)
        b._write_kv = _make_gen(True)
        _exhaust(b._update_airdrop_rewards(25, "mode"))

    def test_update_with_dedup_new(self) -> None:
        b = _make_behaviour()
        # First read for dedup check returns nothing
        call_count = [0]

        def fake_read_kv(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return {}  # No processed marker
            yield
            return {}

        b._read_kv = fake_read_kv
        b._get_total_airdrop_rewards = _make_gen(0)
        b._write_kv = _make_gen(True)
        _exhaust(b._update_airdrop_rewards(100, "mode", tx_hash="0xabc"))

    def test_update_with_dedup_already_processed(self) -> None:
        b = _make_behaviour()
        processed_key = "airdrop_processed_mode_0xabc"
        b._read_kv = _make_gen({processed_key: "true"})
        _exhaust(b._update_airdrop_rewards(100, "mode", tx_hash="0xabc"))
        # Should return early without updating


class TestSignMessage:
    """Test sign_message generator."""

    def test_success(self) -> None:
        b = _make_behaviour()
        b.get_signature = _make_gen("0xdeadbeef")
        result = _exhaust(b.sign_message("hello"))
        assert result == "deadbeef"

    def test_no_signature(self) -> None:
        b = _make_behaviour()
        b.get_signature = _make_gen(None)
        result = _exhaust(b.sign_message("hello"))
        assert result is None


class TestCalculateInitialInvestment:
    """Test calculate_initial_investment generator."""

    def test_no_open_positions(self) -> None:
        b = _make_behaviour()
        b.current_positions = [{"status": "closed"}]
        result = _exhaust(b.calculate_initial_investment())
        assert result is None

    def test_with_positions(self) -> None:
        b = _make_behaviour()
        b.current_positions = [
            {"status": "open", "pool_address": "0xPool", "tx_hash": "0xHash"}
        ]
        b.calculate_initial_investment_value = _make_gen(100.0)
        result = _exhaust(b.calculate_initial_investment())
        assert result == 100.0

    def test_null_value_skipped(self) -> None:
        b = _make_behaviour()
        b.current_positions = [
            {"status": "open", "id": "p1"},
            {"status": "open", "pool_address": "0xP", "tx_hash": "0xH"},
        ]
        call_count = [0]

        def fake_calc(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return None
            yield
            return 50.0

        b.calculate_initial_investment_value = fake_calc
        result = _exhaust(b.calculate_initial_investment())
        assert result == 50.0


class TestExecuteStrategy:
    """Test the execute_strategy module-level function."""

    def test_no_executable(self) -> None:
        result = execute_strategy("unknown", {})
        assert result is None

    def test_no_callable_in_exec(self) -> None:
        result = execute_strategy("test", {"test": ("x = 1", "nonexistent_method")})
        assert result is None

    def test_valid_strategy(self) -> None:
        code = "def test_func(**kwargs):\n    return {'result': 42}"
        result = execute_strategy("strat", {"strat": (code, "test_func")})
        assert result == {"result": 42}

    def test_generator_strategy(self) -> None:
        code = "def gen_func(**kwargs):\n    yield 1\n    yield 2"
        result = execute_strategy("strat", {"strat": (code, "gen_func")})
        assert result == [1, 2]


class TestFetchRewardBalances:
    """Test _fetch_reward_balances generator."""

    def test_no_safe_address(self) -> None:
        b = _make_behaviour()
        b.params.safe_contract_addresses = {}
        result = _exhaust(b._fetch_reward_balances("optimism"))
        assert result == []

    def test_no_reward_tokens(self) -> None:
        b = _make_behaviour()
        result = _exhaust(b._fetch_reward_balances("unknown_chain"))
        assert result == []

    def test_with_balances(self) -> None:
        b = _make_behaviour()
        b._get_token_balance = _make_gen(1000)
        result = _exhaust(b._fetch_reward_balances("optimism"))
        assert len(result) > 0

    def test_zero_balance(self) -> None:
        b = _make_behaviour()
        b._get_token_balance = _make_gen(0)
        result = _exhaust(b._fetch_reward_balances("optimism"))
        assert result == []


class TestFetchOusdtBalance:
    """Test _fetch_ousdt_balance generator."""

    def test_no_safe_address(self) -> None:
        b = _make_behaviour()
        b.params.safe_contract_addresses = {}
        result = _exhaust(b._fetch_ousdt_balance("optimism"))
        assert result is None

    def test_positive_balance(self) -> None:
        b = _make_behaviour()
        b._get_token_balance = _make_gen(5000)
        result = _exhaust(b._fetch_ousdt_balance("optimism"))
        assert result is not None
        assert result["asset_symbol"] == "oUSDT"
        assert result["balance"] == 5000

    def test_zero_balance(self) -> None:
        b = _make_behaviour()
        b._get_token_balance = _make_gen(0)
        result = _exhaust(b._fetch_ousdt_balance("optimism"))
        assert result is None

    def test_exception(self) -> None:
        b = _make_behaviour()

        def bad_gen(*a, **kw):
            raise Exception("fail")
            yield  # noqa: unreachable

        b._get_token_balance = bad_gen
        result = _exhaust(b._fetch_ousdt_balance("optimism"))
        assert result is None


class TestAdjustPositions:
    """Test _adjust_current_positions_for_backward_compatibility."""

    def test_dict_input(self) -> None:
        b = _make_behaviour()
        b.store_current_positions = MagicMock()
        b._get_token_symbol = _make_gen("USDC")
        data = {"pool_address": "0xPool", "status": "open"}
        _exhaust(b._adjust_current_positions_for_backward_compatibility(data))
        assert len(b.current_positions) == 1
        assert b.current_positions[0]["entry_cost"] == 0.0

    def test_list_with_address_and_assets(self) -> None:
        b = _make_behaviour()
        b.store_current_positions = MagicMock()
        b._get_token_symbol = _make_gen("TOKEN")
        data = [{"address": "0xPool", "assets": ["0xT0", "0xT1"], "chain": "optimism"}]
        _exhaust(b._adjust_current_positions_for_backward_compatibility(data))
        assert b.current_positions[0]["pool_address"] == "0xPool"
        assert "address" not in b.current_positions[0]
        assert b.current_positions[0]["token0"] == "0xT0"
        assert b.current_positions[0]["token1"] == "0xT1"

    def test_adds_default_fields(self) -> None:
        b = _make_behaviour()
        b.store_current_positions = MagicMock()
        data = [{"pool_address": "0x1", "enter_timestamp": 100}]
        _exhaust(b._adjust_current_positions_for_backward_compatibility(data))
        pos = b.current_positions[0]
        assert pos["status"] == "open"
        assert pos["entry_cost"] == 0.0
        assert pos["min_hold_days"] == MIN_TIME_IN_POSITION
        assert pos["cost_recovered"] is False
        assert pos["principal_usd"] == 0.0

    def test_entry_apr_from_apr(self) -> None:
        b = _make_behaviour()
        b.store_current_positions = MagicMock()
        data = [{"apr": 5.0}]
        _exhaust(b._adjust_current_positions_for_backward_compatibility(data))
        assert b.current_positions[0]["entry_apr"] == 5.0

    def test_unexpected_format(self) -> None:
        b = _make_behaviour()
        _exhaust(b._adjust_current_positions_for_backward_compatibility("invalid"))
        assert b.current_positions == []


class TestCalculateMinSafeTx:
    """Test _calculate_min_num_of_safe_tx_required."""

    def test_no_liveness_ratio(self) -> None:
        b = _make_behaviour()
        b._get_liveness_ratio = _make_gen(None)
        b._get_liveness_period = _make_gen(86400)
        result = _exhaust(b._calculate_min_num_of_safe_tx_required("optimism"))
        assert result is None

    def test_no_liveness_period(self) -> None:
        b = _make_behaviour()
        b._get_liveness_ratio = _make_gen(1000)
        b._get_liveness_period = _make_gen(None)
        result = _exhaust(b._calculate_min_num_of_safe_tx_required("optimism"))
        assert result is None

    def test_no_checkpoint(self) -> None:
        b = _make_behaviour()
        b._get_liveness_ratio = _make_gen(10**18)
        b._get_liveness_period = _make_gen(86400)
        b._get_ts_checkpoint = _make_gen(None)
        result = _exhaust(b._calculate_min_num_of_safe_tx_required("optimism"))
        assert result is None

    def test_success(self) -> None:
        b = _make_behaviour()
        b._get_liveness_ratio = _make_gen(10**18)
        b._get_liveness_period = _make_gen(86400)
        b._get_ts_checkpoint = _make_gen(1700000000 - 86400)
        result = _exhaust(b._calculate_min_num_of_safe_tx_required("optimism"))
        assert isinstance(result, int)
        assert result > 0


class TestInvalidateClPoolCache:
    """Test _invalidate_cl_pool_cache."""

    def test_success(self) -> None:
        b = _make_behaviour()
        b._write_kv = _make_gen(True)
        _exhaust(b._invalidate_cl_pool_cache("optimism"))
        b.context.logger.info.assert_called()

    def test_exception(self) -> None:
        b = _make_behaviour()

        def bad_write(*a, **kw):
            raise Exception("fail")
            yield  # noqa: unreachable

        b._write_kv = bad_write
        _exhaust(b._invalidate_cl_pool_cache("optimism"))
        b.context.logger.error.assert_called()


class TestStoreReadMethods:
    """Test store_*/read_* convenience methods that delegate to _store_data/_read_data."""

    def test_store_whitelisted_assets(self) -> None:
        b = _make_behaviour()
        b.whitelisted_assets = {"tok": "val"}
        with patch.object(b, "_store_data") as mock_store:
            b.store_whitelisted_assets()
            mock_store.assert_called_once()

    def test_read_whitelisted_assets(self) -> None:
        b = _make_behaviour()
        with patch.object(b, "_read_data") as mock_read:
            b.read_whitelisted_assets()
            mock_read.assert_called_once()

    def test_store_funding_events(self) -> None:
        b = _make_behaviour()
        b.funding_events = {"e": 1}
        with patch.object(b, "_store_data") as mock_store:
            b.store_funding_events()
            mock_store.assert_called_once()

    def test_read_funding_events(self) -> None:
        b = _make_behaviour()
        with patch.object(b, "_read_data") as mock_read:
            b.read_funding_events()
            mock_read.assert_called_once()

    def test_store_portfolio_data(self) -> None:
        b = _make_behaviour()
        b.portfolio_data = {"k": "v"}
        with patch.object(b, "_store_data") as mock_store:
            b.store_portfolio_data()
            mock_store.assert_called_once()

    def test_read_portfolio_data(self) -> None:
        b = _make_behaviour()
        with patch.object(b, "_read_data") as mock_read:
            b.read_portfolio_data()
            mock_read.assert_called_once()

    def test_store_agent_performance(self) -> None:
        b = _make_behaviour()
        b.agent_performance = {"ts": 1}
        with patch.object(b, "_store_data") as mock_store:
            b.store_agent_performance()
            mock_store.assert_called_once()

    def test_store_gas_costs(self) -> None:
        b = _make_behaviour()
        with patch.object(b, "_store_data") as mock_store:
            b.store_gas_costs()
            mock_store.assert_called_once()

    def test_read_gas_costs(self) -> None:
        b = _make_behaviour()
        with patch.object(b, "_read_data") as mock_read:
            b.read_gas_costs()
            mock_read.assert_called_once()


class TestReadDataFilePaths:
    """Test _read_data with various file conditions."""

    def test_read_data_file_not_found_creates_file(self) -> None:
        b = _make_behaviour()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "nonexistent.json")
            b._read_data("current_positions", path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data == []

    def test_read_data_permission_error(self) -> None:
        b = _make_behaviour()
        with patch("builtins.open", side_effect=PermissionError("denied")):
            b._read_data("current_positions", "/fake/path.json")
            b.context.logger.error.assert_called()

    def test_read_data_json_decode_error(self) -> None:
        b = _make_behaviour()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            f.flush()
            fname = f.name
        try:
            b._read_data("current_positions", fname)
            b.context.logger.error.assert_called()
        finally:
            os.unlink(fname)


class TestUpdateAgentPerformanceTimestampException:
    """Test update_agent_performance_timestamp exception path."""

    def test_exception(self) -> None:
        b = _make_behaviour()
        b.agent_performance = None  # Will cause TypeError on key access
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.datetime"
        ) as mock_dt:
            mock_dt.now.side_effect = Exception("boom")
            b.update_agent_performance_timestamp()
            b.context.logger.error.assert_called()


class TestGetTokenDecimals:
    """Test _get_token_decimals."""

    def test_returns_decimals(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(18)
        result = _exhaust(b._get_token_decimals("optimism", "0x" + "ab" * 20))
        assert result == 18


class TestGetTokenSymbol:
    """Test _get_token_symbol."""

    def test_returns_symbol(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen("USDC")
        result = _exhaust(b._get_token_symbol("optimism", "0x" + "ab" * 20))
        assert result == "USDC"


class TestGetNativeBalance:
    """Test _get_native_balance."""

    def test_success(self) -> None:
        b = _make_behaviour()
        mock_response = MagicMock()
        mock_response.performative = MagicMock()
        mock_response.state.body = {"get_balance_result": 1000}
        # Patch LedgerApiMessage so performative check passes
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.LedgerApiMessage"
        ) as mock_msg:
            mock_msg.Performative.STATE = mock_response.performative
            b.get_ledger_api_response = _make_gen(mock_response)
            result = _exhaust(b._get_native_balance("optimism", "0xSafe"))
            assert result == 1000

    def test_failure(self) -> None:
        b = _make_behaviour()
        mock_response = MagicMock()
        mock_response.performative = "FAILURE"
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.LedgerApiMessage"
        ) as mock_msg:
            mock_msg.Performative.STATE = "STATE"
            b.get_ledger_api_response = _make_gen(mock_response)
            result = _exhaust(b._get_native_balance("optimism", "0xSafe"))
            assert result is None


class TestGetTokenBalance:
    """Test _get_token_balance."""

    def test_returns_balance(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(500)
        result = _exhaust(b._get_token_balance("optimism", "0xSafe", "0xToken"))
        assert result == 500


class TestGetNextCheckpoint:
    """Test _get_next_checkpoint."""

    def test_returns_timestamp(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(1700000100)
        result = _exhaust(b._get_next_checkpoint("optimism"))
        assert result == 1700000100


class TestGetTsCheckpoint:
    """Test _get_ts_checkpoint."""

    def test_returns_timestamp(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(1700000050)
        result = _exhaust(b._get_ts_checkpoint("optimism"))
        assert result == 1700000050


class TestGetLivenessRatio:
    """Test _get_liveness_ratio."""

    def test_valid(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(100)
        result = _exhaust(b._get_liveness_ratio("optimism"))
        assert result == 100

    def test_zero_logs_error(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(0)
        result = _exhaust(b._get_liveness_ratio("optimism"))
        assert result == 0
        b.context.logger.error.assert_called()


class TestGetLivenessPeriod:
    """Test _get_liveness_period."""

    def test_valid(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(3600)
        result = _exhaust(b._get_liveness_period("optimism"))
        assert result == 3600


class TestFetchTokenNameFromContract:
    """Test _fetch_token_name_from_contract."""

    def test_returns_name(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen("Wrapped Ether")
        result = _exhaust(
            b._fetch_token_name_from_contract("optimism", "0x" + "ab" * 20)
        )
        assert result == "Wrapped Ether"


class TestFetchTokenPriceFull:
    """Test _fetch_token_price method."""

    def test_cached_price(self) -> None:
        b = _make_behaviour()
        b._get_cached_price = _make_gen(1.5)
        result = _exhaust(b._fetch_token_price("0x" + "ab" * 20, "optimism"))
        assert result == 1.5

    def test_success_with_price(self) -> None:
        b = _make_behaviour()
        token_addr = "0x" + "ab" * 20
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(
            return_value=(True, {token_addr.lower(): {"usd": 2.5}})
        )
        b._cache_price = _make_gen(None)
        b._get_last_known_price = _make_gen(None)
        result = _exhaust(b._fetch_token_price(token_addr, "optimism"))
        assert result == 2.5

    def test_success_no_price_data(self) -> None:
        b = _make_behaviour()
        token_addr = "0x" + "ab" * 20
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(
            return_value=(True, {token_addr.lower(): {"usd": 0}})
        )
        b._get_last_known_price = _make_gen(None)
        result = _exhaust(b._fetch_token_price(token_addr, "optimism"))
        assert result == 0

    def test_success_processing_exception(self) -> None:
        b = _make_behaviour()
        token_addr = "0x" + "ab" * 20
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(
            return_value=(True, None)  # Will cause AttributeError on .get()
        )
        b._get_last_known_price = _make_gen(None)
        result = _exhaust(b._fetch_token_price(token_addr, "optimism"))
        assert result is None

    def test_failure_with_last_known(self) -> None:
        b = _make_behaviour()
        token_addr = "0x" + "ab" * 20
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(return_value=(False, {}))
        b._get_last_known_price = _make_gen(3.0)
        result = _exhaust(b._fetch_token_price(token_addr, "optimism"))
        assert result == 3.0

    def test_failure_no_last_known(self) -> None:
        b = _make_behaviour()
        token_addr = "0x" + "ab" * 20
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(return_value=(False, {}))
        b._get_last_known_price = _make_gen(None)
        result = _exhaust(b._fetch_token_price(token_addr, "optimism"))
        assert result is None

    def test_no_platform_id(self) -> None:
        b = _make_behaviour()
        b._get_cached_price = _make_gen(None)
        b.coingecko.chain_to_platform_id_mapping = {}
        result = _exhaust(b._fetch_token_price("0x" + "ab" * 20, "optimism"))
        assert result is None


class TestFetchCoinPrice:
    """Test _fetch_coin_price method."""

    def test_cached(self) -> None:
        b = _make_behaviour()
        b._get_cached_price = _make_gen(1800.0)
        result = _exhaust(b._fetch_coin_price("ethereum"))
        assert result == 1800.0

    def test_success(self) -> None:
        b = _make_behaviour()
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(
            return_value=(True, {"ethereum": {"usd": 1900.0}})
        )
        b._cache_price = _make_gen(None)
        result = _exhaust(b._fetch_coin_price("ethereum"))
        assert result == 1900.0

    def test_success_no_price(self) -> None:
        b = _make_behaviour()
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(return_value=(True, {"ethereum": {"usd": 0}}))
        result = _exhaust(b._fetch_coin_price("ethereum"))
        assert result == 0

    def test_success_exception(self) -> None:
        b = _make_behaviour()
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(return_value=(True, None))
        result = _exhaust(b._fetch_coin_price("ethereum"))
        assert result is None

    def test_failure(self) -> None:
        b = _make_behaviour()
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(return_value=(False, {}))
        result = _exhaust(b._fetch_coin_price("ethereum"))
        assert result is None


class TestFetchTokenPrices:
    """Test _fetch_token_prices method."""

    def test_with_zero_address(self) -> None:
        b = _make_behaviour()
        b._fetch_zero_address_price = _make_gen(1800.0)
        b._fetch_token_price = _make_gen(None)
        token_balances = [{"token": ZERO_ADDRESS, "chain": "optimism"}]
        result = _exhaust(b._fetch_token_prices(token_balances))
        assert result[ZERO_ADDRESS] == 1800.0

    def test_missing_chain(self) -> None:
        b = _make_behaviour()
        token_balances = [{"token": "0x" + "ab" * 20}]
        result = _exhaust(b._fetch_token_prices(token_balances))
        assert result == {}

    def test_with_erc20(self) -> None:
        b = _make_behaviour()
        addr = "0x" + "ab" * 20
        b._fetch_token_price = _make_gen(2.5)
        token_balances = [{"token": addr, "chain": "optimism"}]
        result = _exhaust(b._fetch_token_prices(token_balances))
        assert result[addr] == 2.5


class TestGetCachedPriceEdge:
    """Test _get_cached_price edge cases."""

    def test_no_result(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        result = _exhaust(b._get_cached_price("0xToken", "01-01-2024"))
        assert result is None

    def test_with_date_returns_price(self) -> None:
        b = _make_behaviour()
        cache_data = json.dumps({"01-01-2024": 5.0})
        key = b._get_price_cache_key("0xToken", "01-01-2024")
        b._read_kv = _make_gen({key: cache_data})
        result = _exhaust(b._get_cached_price("0xToken", "01-01-2024"))
        assert result == 5.0

    def test_invalid_json(self) -> None:
        b = _make_behaviour()
        key = b._get_price_cache_key("0xToken", "01-01-2024")
        b._read_kv = _make_gen({key: "not json"})
        result = _exhaust(b._get_cached_price("0xToken", "01-01-2024"))
        assert result is None


class TestCachePrice:
    """Test _cache_price."""

    def test_with_date(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        b._write_kv = _make_gen(True)
        _exhaust(b._cache_price("ethereum", 1800.0, "01-01-2024"))

    def test_existing_cache_invalid_json(self) -> None:
        b = _make_behaviour()
        key = b._get_price_cache_key("ethereum", "01-01-2024")
        b._read_kv = _make_gen({key: "not json"})
        b._write_kv = _make_gen(True)
        _exhaust(b._cache_price("ethereum", 1800.0, "01-01-2024"))
        b.context.logger.error.assert_called()


class TestRequestWithRetries:
    """Test _request_with_retries."""

    def _mock_response(self, status_code, body):
        resp = MagicMock()
        resp.status_code = status_code
        resp.body = json.dumps(body).encode("utf-8") if isinstance(body, dict) else body
        return resp

    def test_success(self) -> None:
        b = _make_behaviour()
        resp = self._mock_response(200, {"result": "ok"})
        b.get_http_response = _make_gen(resp)
        success, data = _exhaust(
            b._request_with_retries(
                endpoint="https://example.com",
                method="GET",
                rate_limited_callback=lambda: None,
                max_retries=1,
            )
        )
        assert success is True
        assert data["result"] == "ok"

    def test_rate_limited_exhausts_retries(self) -> None:
        b = _make_behaviour()
        resp = self._mock_response(429, {"error": "rate limited"})
        b.get_http_response = _make_gen(resp)
        b.sleep = _make_gen(None)
        success, data = _exhaust(
            b._request_with_retries(
                endpoint="https://example.com",
                method="GET",
                rate_limited_callback=lambda: None,
                max_retries=1,
            )
        )
        assert success is False

    def test_503_exhausts_retries(self) -> None:
        b = _make_behaviour()
        resp = self._mock_response(503, {"error": "unavailable"})
        b.get_http_response = _make_gen(resp)
        b.sleep = _make_gen(None)
        success, data = _exhaust(
            b._request_with_retries(
                endpoint="https://example.com",
                method="GET",
                rate_limited_callback=lambda: None,
                max_retries=1,
            )
        )
        assert success is False

    def test_other_error_exhausts_retries(self) -> None:
        b = _make_behaviour()
        resp = self._mock_response(500, {"error": "server"})
        b.get_http_response = _make_gen(resp)
        b.sleep = _make_gen(None)
        success, data = _exhaust(
            b._request_with_retries(
                endpoint="https://example.com",
                method="GET",
                rate_limited_callback=lambda: None,
                max_retries=1,
                retry_wait=0,
            )
        )
        assert success is False

    def test_json_decode_error(self) -> None:
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = b"not json"
        b.get_http_response = _make_gen(resp)
        b.sleep = _make_gen(None)
        success, data = _exhaust(
            b._request_with_retries(
                endpoint="https://example.com",
                method="GET",
                rate_limited_callback=lambda: None,
                max_retries=1,
                retry_wait=0,
            )
        )
        assert success is False
        assert "exception" in data

    def test_coingecko_adds_delay(self) -> None:
        b = _make_behaviour()
        resp = self._mock_response(200, {"result": "ok"})
        b.get_http_response = _make_gen(resp)
        b.sleep = _make_gen(None)
        success, _ = _exhaust(
            b._request_with_retries(
                endpoint="https://api.coingecko.com/v3/simple/price",
                method="GET",
                rate_limited_callback=lambda: None,
                max_retries=1,
            )
        )
        assert success is True


class TestDoConnectionRequest:
    """Test _do_connection_request."""

    def test_returns_response(self) -> None:
        b = _make_behaviour()
        b._get_request_nonce_from_dialogue = MagicMock(return_value="nonce_1")
        b.get_callback_request = MagicMock(return_value=lambda: None)
        b.context.outbox = MagicMock()
        mock_requests = MagicMock()
        mock_requests.request_id_to_callback = {}
        b.context.requests = mock_requests
        expected = MagicMock()
        b.wait_for_message = _make_gen(expected)
        result = _exhaust(b._do_connection_request(MagicMock(), MagicMock()))
        assert result == expected


class TestCallMirrordb:
    """Test _call_mirrordb."""

    def test_success(self) -> None:
        b = _make_behaviour()
        response_mock = MagicMock()
        response_mock.payload = json.dumps({"response": {"id": 1}})
        b._do_connection_request = _make_gen(response_mock)
        srr_dialogues = MagicMock()
        srr_dialogues.create.return_value = (MagicMock(), MagicMock())
        b.context.srr_dialogues = srr_dialogues
        result = _exhaust(b._call_mirrordb("read_", endpoint="test"))
        assert result == {"id": 1}

    def test_error_response(self) -> None:
        b = _make_behaviour()
        response_mock = MagicMock()
        response_mock.payload = json.dumps({"error": "not found"})
        b._do_connection_request = _make_gen(response_mock)
        srr_dialogues = MagicMock()
        srr_dialogues.create.return_value = (MagicMock(), MagicMock())
        b.context.srr_dialogues = srr_dialogues
        result = _exhaust(b._call_mirrordb("read_", endpoint="test"))
        assert result is None

    def test_exception(self) -> None:
        b = _make_behaviour()
        srr_dialogues = MagicMock()
        srr_dialogues.create.side_effect = Exception("connection error")
        b.context.srr_dialogues = srr_dialogues
        result = _exhaust(b._call_mirrordb("read_", endpoint="test"))
        assert result is None


class TestReadWriteKV:
    """Test _read_kv and _write_kv."""

    def test_read_kv_success(self) -> None:
        b = _make_behaviour()
        mock_response = MagicMock()
        mock_response.performative = MagicMock()
        mock_response.data = {"key1": "val1"}
        # Patch the KvStoreMessage check
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.KvStoreMessage"
        ) as kv_msg:
            kv_msg.Performative.READ_RESPONSE = mock_response.performative
            kv_dialogues = MagicMock()
            kv_dialogues.create.return_value = (MagicMock(), MagicMock())
            b.context.kv_store_dialogues = kv_dialogues
            b._do_connection_request = _make_gen(mock_response)
            result = _exhaust(b._read_kv(("key1",)))
            assert result == {"key1": "val1"}

    def test_read_kv_wrong_performative(self) -> None:
        b = _make_behaviour()
        mock_response = MagicMock()
        mock_response.performative = "WRONG"
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.KvStoreMessage"
        ) as kv_msg:
            kv_msg.Performative.READ_RESPONSE = "READ_RESPONSE"
            kv_dialogues = MagicMock()
            kv_dialogues.create.return_value = (MagicMock(), MagicMock())
            b.context.kv_store_dialogues = kv_dialogues
            b._do_connection_request = _make_gen(mock_response)
            result = _exhaust(b._read_kv(("key1",)))
            assert result is None

    def test_write_kv(self) -> None:
        b = _make_behaviour()
        mock_response = MagicMock()
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.KvStoreMessage"
        ) as kv_msg:
            kv_msg.Performative.SUCCESS = "SUCCESS"
            kv_msg.Performative.CREATE_OR_UPDATE_REQUEST = "COU"
            kv_dialogues = MagicMock()
            kv_dialogues.create.return_value = (MagicMock(), MagicMock())
            b.context.kv_store_dialogues = kv_dialogues
            b._do_connection_request = _make_gen(mock_response)
            _exhaust(b._write_kv({"key1": "val1"}))


class TestFetchSafeBalancesWithPagination:
    """Test _fetch_safe_balances_with_pagination."""

    def test_success_single_page(self) -> None:
        b = _make_behaviour()
        page_data = {"results": [{"tokenAddress": None}], "next": None}
        b._request_with_retries = _make_gen((True, page_data))
        result = _exhaust(b._fetch_safe_balances_with_pagination("0xSafe"))
        assert len(result) == 1

    def test_failure(self) -> None:
        b = _make_behaviour()
        b._request_with_retries = _make_gen((False, {"error": "fail"}))
        result = _exhaust(b._fetch_safe_balances_with_pagination("0xSafe"))
        assert result == []

    def test_empty_results(self) -> None:
        b = _make_behaviour()
        b._request_with_retries = _make_gen((True, {"results": []}))
        result = _exhaust(b._fetch_safe_balances_with_pagination("0xSafe"))
        assert result == []


class TestFetchModeTokensWithPagination:
    """Test _fetch_mode_tokens_with_pagination."""

    def test_success_single_page(self) -> None:
        b = _make_behaviour()
        page_data = {"items": [{"token": {"address": "0xA"}}], "next_page_params": None}
        b._request_with_retries = _make_gen((True, page_data))
        result = _exhaust(b._fetch_mode_tokens_with_pagination("0xSafe"))
        assert len(result) == 1

    def test_failure(self) -> None:
        b = _make_behaviour()
        b._request_with_retries = _make_gen((False, {"error": "fail"}))
        result = _exhaust(b._fetch_mode_tokens_with_pagination("0xSafe"))
        assert result == []

    def test_empty_items(self) -> None:
        b = _make_behaviour()
        b._request_with_retries = _make_gen((True, {"items": []}))
        result = _exhaust(b._fetch_mode_tokens_with_pagination("0xSafe"))
        assert result == []


class TestAdjustCurrentPositions:
    """Test _adjust_current_positions_for_backward_compatibility."""

    def test_dict_input_converted_to_list(self) -> None:
        b = _make_behaviour()
        with patch.object(b, "store_current_positions"):
            b._get_token_symbol = _make_gen("ETH")
            _exhaust(
                b._adjust_current_positions_for_backward_compatibility(
                    {"pool_address": "0xPool", "chain": "optimism"}
                )
            )
            assert len(b.current_positions) == 1
            assert b.current_positions[0]["status"] == "open"

    def test_list_with_assets(self) -> None:
        b = _make_behaviour()
        with patch.object(b, "store_current_positions"):
            b._get_token_symbol = _make_gen("SYM")
            _exhaust(
                b._adjust_current_positions_for_backward_compatibility(
                    [
                        {
                            "address": "0xPool",
                            "chain": "optimism",
                            "assets": ["0xToken0", "0xToken1"],
                        }
                    ]
                )
            )
            assert b.current_positions[0]["pool_address"] == "0xPool"
            assert "token0" in b.current_positions[0]
            assert "token1" in b.current_positions[0]

    def test_list_with_one_asset(self) -> None:
        b = _make_behaviour()
        with patch.object(b, "store_current_positions"):
            b._get_token_symbol = _make_gen("SYM")
            _exhaust(
                b._adjust_current_positions_for_backward_compatibility(
                    [{"chain": "optimism", "assets": ["0xToken0"]}]
                )
            )
            assert "token0" in b.current_positions[0]
            assert "token1" not in b.current_positions[0]

    def test_unexpected_format(self) -> None:
        b = _make_behaviour()
        _exhaust(b._adjust_current_positions_for_backward_compatibility("invalid"))
        assert b.current_positions == []
        b.context.logger.warning.assert_called()

    def test_backward_compat_fields_added(self) -> None:
        b = _make_behaviour()
        with patch.object(b, "store_current_positions"):
            _exhaust(
                b._adjust_current_positions_for_backward_compatibility(
                    [{"chain": "optimism", "enter_timestamp": 1700000000}]
                )
            )
            pos = b.current_positions[0]
            assert "entry_cost" in pos
            assert "min_hold_days" in pos
            assert "cost_recovered" in pos
            assert "principal_usd" in pos

    def test_entry_apr_from_apr(self) -> None:
        b = _make_behaviour()
        with patch.object(b, "store_current_positions"):
            _exhaust(
                b._adjust_current_positions_for_backward_compatibility(
                    [{"chain": "optimism", "apr": 15.5}]
                )
            )
            assert b.current_positions[0]["entry_apr"] == 15.5


class TestCalculateInitialInvestmentValue:
    """Test calculate_initial_investment_value."""

    def test_missing_data(self) -> None:
        b = _make_behaviour()
        result = _exhaust(b.calculate_initial_investment_value({}))
        assert result is None

    def test_no_token0_decimals(self) -> None:
        b = _make_behaviour()
        b._get_token_decimals = _make_gen(None)
        pos = {
            "token0": "0xT0",
            "amount0": 1000,
            "timestamp": 1700000000,
            "chain": "optimism",
        }
        result = _exhaust(b.calculate_initial_investment_value(pos))
        assert result is None

    def test_single_token(self) -> None:
        b = _make_behaviour()
        b._get_token_decimals = _make_gen(6)
        b._fetch_historical_token_prices = _make_gen({"0xT0": 1.0})
        pos = {
            "token0": "0xT0",
            "token0_symbol": "USDC",
            "amount0": 1_000_000,
            "timestamp": 1700000000,
            "chain": "optimism",
        }
        result = _exhaust(b.calculate_initial_investment_value(pos))
        assert result == 1.0

    def test_two_tokens(self) -> None:
        b = _make_behaviour()
        b._get_token_decimals = _make_gen(18)
        b._fetch_historical_token_prices = _make_gen({"0xT0": 1.0, "0xT1": 2.0})
        pos = {
            "token0": "0xT0",
            "token0_symbol": "A",
            "token1": "0xT1",
            "token1_symbol": "B",
            "amount0": 10**18,
            "amount1": 10**18,
            "timestamp": 1700000000,
            "chain": "optimism",
        }
        result = _exhaust(b.calculate_initial_investment_value(pos))
        assert result == 3.0  # 1.0 * 1.0 + 1.0 * 2.0

    def test_no_historical_prices(self) -> None:
        b = _make_behaviour()
        b._get_token_decimals = _make_gen(18)
        b._fetch_historical_token_prices = _make_gen({})
        pos = {
            "token0": "0xT0",
            "token0_symbol": "A",
            "amount0": 10**18,
            "timestamp": 1700000000,
            "chain": "optimism",
        }
        result = _exhaust(b.calculate_initial_investment_value(pos))
        assert result is None

    def test_no_price_for_token0(self) -> None:
        b = _make_behaviour()
        b._get_token_decimals = _make_gen(18)
        b._fetch_historical_token_prices = _make_gen({"0xOther": 1.0})
        pos = {
            "token0": "0xT0",
            "token0_symbol": "A",
            "amount0": 10**18,
            "timestamp": 1700000000,
            "chain": "optimism",
        }
        result = _exhaust(b.calculate_initial_investment_value(pos))
        assert result is None


class TestCalculateInitialInvestment:
    """Test calculate_initial_investment."""

    def test_no_open_positions(self) -> None:
        b = _make_behaviour()
        b.current_positions = [{"status": "closed"}]
        b.calculate_initial_investment_value = _make_gen(None)
        result = _exhaust(b.calculate_initial_investment())
        assert result is None

    def test_with_values(self) -> None:
        b = _make_behaviour()
        b.current_positions = [
            {"status": "open", "pool_address": "0xPool", "tx_hash": "0xhash"},
        ]
        b.calculate_initial_investment_value = _make_gen(100.0)
        result = _exhaust(b.calculate_initial_investment())
        assert result == 100.0

    def test_null_position_value(self) -> None:
        b = _make_behaviour()
        b.current_positions = [
            {"status": "open", "pool_address": "0xP", "tx_hash": "0xh", "id": "x"},
        ]
        b.calculate_initial_investment_value = _make_gen(None)
        result = _exhaust(b.calculate_initial_investment())
        assert result is None


class TestFetchHistoricalTokenPrices:
    """Test _fetch_historical_token_prices."""

    def test_success(self) -> None:
        b = _make_behaviour()
        b._fetch_historical_token_price = _make_gen(5.0)
        with patch.object(b, "get_coin_id_from_symbol", return_value="usd-coin"):
            result = _exhaust(
                b._fetch_historical_token_prices(
                    [["USDC", "0xAddr"]], "01-01-2024", "optimism"
                )
            )
            assert result["0xAddr"] == 5.0

    def test_no_coingecko_id(self) -> None:
        b = _make_behaviour()
        with patch.object(b, "get_coin_id_from_symbol", return_value=None):
            result = _exhaust(
                b._fetch_historical_token_prices(
                    [["UNKNOWN", "0xAddr"]], "01-01-2024", "optimism"
                )
            )
            assert result == {}


class TestFetchHistoricalTokenPrice:
    """Test _fetch_historical_token_price."""

    def test_cached(self) -> None:
        b = _make_behaviour()
        b._get_cached_price = _make_gen(10.0)
        result = _exhaust(b._fetch_historical_token_price("usd-coin", "01-01-2024"))
        assert result == 10.0

    def test_success(self) -> None:
        b = _make_behaviour()
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(
            return_value=(
                True,
                {"market_data": {"current_price": {"usd": 1.0}}},
            )
        )
        b._cache_price = _make_gen(None)
        result = _exhaust(b._fetch_historical_token_price("usd-coin", "01-01-2024"))
        assert result == 1.0

    def test_success_no_price(self) -> None:
        b = _make_behaviour()
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(
            return_value=(True, {"market_data": {"current_price": {}}})
        )
        result = _exhaust(b._fetch_historical_token_price("usd-coin", "01-01-2024"))
        assert result is None

    def test_success_exception(self) -> None:
        b = _make_behaviour()
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(return_value=(True, None))
        result = _exhaust(b._fetch_historical_token_price("usd-coin", "01-01-2024"))
        assert result is None

    def test_failure(self) -> None:
        b = _make_behaviour()
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(return_value=(False, {}))
        result = _exhaust(b._fetch_historical_token_price("usd-coin", "01-01-2024"))
        assert result is None


class TestFetchTokenPricesSma:
    """Test _fetch_token_prices_sma."""

    def test_no_symbol(self) -> None:
        b = _make_behaviour()
        b._get_token_symbol = _make_gen(None)
        result = _exhaust(b._fetch_token_prices_sma("0xToken", "optimism"))
        assert result is None

    def test_no_coingecko_id(self) -> None:
        b = _make_behaviour()
        b._get_token_symbol = _make_gen("TOK")
        with patch.object(b, "get_coin_id_from_symbol", return_value=None):
            result = _exhaust(b._fetch_token_prices_sma("0xToken", "optimism"))
            assert result is None

    def test_success(self) -> None:
        b = _make_behaviour()
        b._get_token_symbol = _make_gen("ETH")
        b.sleep = _make_gen(None)
        with patch.object(b, "get_coin_id_from_symbol", return_value="ethereum"):
            b.coingecko.request = MagicMock(
                return_value=(
                    True,
                    {"prices": [[1700000000, 100.0], [1700003600, 200.0]]},
                )
            )
            result = _exhaust(b._fetch_token_prices_sma("0xToken", "optimism"))
            assert result == 150.0

    def test_failure(self) -> None:
        b = _make_behaviour()
        b._get_token_symbol = _make_gen("ETH")
        b.sleep = _make_gen(None)
        with patch.object(b, "get_coin_id_from_symbol", return_value="ethereum"):
            b.coingecko.request = MagicMock(return_value=(False, {}))
            result = _exhaust(b._fetch_token_prices_sma("0xToken", "optimism"))
            assert result is None

    def test_rate_limited(self) -> None:
        b = _make_behaviour()
        b._get_token_symbol = _make_gen("ETH")
        b.sleep = _make_gen(None)
        with patch.object(b, "get_coin_id_from_symbol", return_value="ethereum"):
            b.coingecko.request = MagicMock(
                return_value=(True, {"status": {"error_code": 429}})
            )
            result = _exhaust(b._fetch_token_prices_sma("0xToken", "optimism"))
            assert result is None

    def test_empty_prices(self) -> None:
        b = _make_behaviour()
        b._get_token_symbol = _make_gen("ETH")
        b.sleep = _make_gen(None)
        with patch.object(b, "get_coin_id_from_symbol", return_value="ethereum"):
            b.coingecko.request = MagicMock(return_value=(True, {"prices": []}))
            result = _exhaust(b._fetch_token_prices_sma("0xToken", "optimism"))
            assert result is None

    def test_exception(self) -> None:
        b = _make_behaviour()
        b._get_token_symbol = _make_gen("ETH")
        b.sleep = _make_gen(None)
        with patch.object(b, "get_coin_id_from_symbol", side_effect=Exception("err")):
            result = _exhaust(b._fetch_token_prices_sma("0xToken", "optimism"))
            assert result is None


class TestQueryServiceRewards:
    """Test query_service_rewards."""

    def test_success(self) -> None:
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps(
            {"data": {"service": {"olasRewardsEarned": "100"}}}
        ).encode("utf-8")
        b.get_http_response = _make_gen(resp)
        result = _exhaust(b.query_service_rewards("optimism"))
        assert result["olasRewardsEarned"] == "100"

    def test_no_service_data(self) -> None:
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps({"data": {"service": None}}).encode("utf-8")
        b.get_http_response = _make_gen(resp)
        result = _exhaust(b.query_service_rewards("optimism"))
        assert result is None

    def test_failure(self) -> None:
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 500
        resp.body = b"error"
        b.get_http_response = _make_gen(resp)
        result = _exhaust(b.query_service_rewards("optimism"))
        assert result is None

    def test_json_decode_error(self) -> None:
        """Test query_service_rewards with malformed JSON body."""
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = b"not valid json {"
        b.get_http_response = _make_gen(resp)
        result = _exhaust(b.query_service_rewards("optimism"))
        assert result is None
        b.context.logger.error.assert_called()

    def test_data_null_response(self) -> None:
        """Test query_service_rewards when data is null in response."""
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps({"data": None}).encode("utf-8")
        b.get_http_response = _make_gen(resp)
        result = _exhaust(b.query_service_rewards("optimism"))
        assert result is None


class TestUpdateAccumulatedRewardsForChain:
    """Test update_accumulated_rewards_for_chain."""

    def test_skip_update(self) -> None:
        b = _make_behaviour()
        b.should_update_rewards_from_subgraph = _make_gen(False)
        _exhaust(b.update_accumulated_rewards_for_chain("optimism"))

    def test_no_service_data(self) -> None:
        b = _make_behaviour()
        b.should_update_rewards_from_subgraph = _make_gen(True)
        b.query_service_rewards = _make_gen(None)
        _exhaust(b.update_accumulated_rewards_for_chain("optimism"))

    def test_success(self) -> None:
        b = _make_behaviour()
        b.should_update_rewards_from_subgraph = _make_gen(True)
        b.query_service_rewards = _make_gen({"olasRewardsEarned": "500"})
        b._write_kv = _make_gen(True)
        _exhaust(b.update_accumulated_rewards_for_chain("optimism"))

    def test_invalid_rewards_value(self) -> None:
        b = _make_behaviour()
        b.should_update_rewards_from_subgraph = _make_gen(True)
        b.query_service_rewards = _make_gen({"olasRewardsEarned": "not_a_number"})
        b._write_kv = _make_gen(True)
        _exhaust(b.update_accumulated_rewards_for_chain("optimism"))


class TestShouldUpdateRewardsFromSubgraph:
    """Test should_update_rewards_from_subgraph."""

    def test_period_zero(self) -> None:
        b = _make_behaviour()
        b.synchronized_data.period_count = 0
        result = _exhaust(b.should_update_rewards_from_subgraph("optimism"))
        assert result is True

    def test_no_kv_data(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        result = _exhaust(b.should_update_rewards_from_subgraph("optimism"))
        assert result is True

    def test_invalid_timestamp(self) -> None:
        b = _make_behaviour()
        key = f"{REWARD_UPDATE_KEY_PREFIX}optimism"
        b._read_kv = _make_gen({key: "not_a_number"})
        result = _exhaust(b.should_update_rewards_from_subgraph("optimism"))
        assert result is True

    def test_recent_update(self) -> None:
        b = _make_behaviour()
        key = f"{REWARD_UPDATE_KEY_PREFIX}optimism"
        b._read_kv = _make_gen({key: str(time.time())})
        result = _exhaust(b.should_update_rewards_from_subgraph("optimism"))
        assert result is False


class TestGetAllEntryCostsException:
    """Test _get_all_entry_costs exception path."""

    def test_exception(self) -> None:
        b = _make_behaviour()

        def bad_read(*a, **kw):
            raise Exception("fail")
            yield  # noqa: unreachable

        b._read_kv = bad_read
        result = _exhaust(b._get_all_entry_costs())
        assert result == {}


class TestBuildExitPoolActionBaseFull:
    """Test _build_exit_pool_action_base with all branches."""

    def test_velodrome_cl_multi_position(self) -> None:
        b = _make_behaviour()
        pos = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPool",
            "pool_type": "cl",
            "is_stable": False,
            "is_cl_pool": True,
            "positions": [
                {"token_id": 1, "liquidity": 100},
                {"token_id": 2, "liquidity": 200},
            ],
        }
        result = b._build_exit_pool_action_base(pos)
        assert result["token_ids"] == [1, 2]
        assert result["liquidities"] == [100, 200]

    def test_single_token_id(self) -> None:
        b = _make_behaviour()
        pos = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPool",
            "pool_type": "cl",
            "is_stable": False,
            "is_cl_pool": False,
            "token_id": 42,
            "liquidity": 999,
        }
        result = b._build_exit_pool_action_base(pos)
        assert result["token_id"] == 42
        assert result["liquidity"] == 999

    def test_sturdy_action_type(self) -> None:
        b = _make_behaviour()
        pos = {
            "dex_type": "Sturdy",
            "chain": "optimism",
            "pool_address": "0xPool",
            "pool_type": "lending",
            "is_stable": False,
            "is_cl_pool": False,
        }
        result = b._build_exit_pool_action_base(pos)
        assert result["action"] == Action.WITHDRAW.value

    def test_with_tokens(self) -> None:
        b = _make_behaviour()
        pos = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPool",
            "pool_type": "stable",
            "is_stable": True,
            "is_cl_pool": False,
        }
        tokens = [{"token": "0xA"}, {"token": "0xB"}]
        result = b._build_exit_pool_action_base(pos, tokens)
        assert result["assets"] == ["0xA", "0xB"]


class TestBuildSwapToUsdcAction:
    """Test _build_swap_to_usdc_action."""

    def test_success(self) -> None:
        b = _make_behaviour()
        result = b._build_swap_to_usdc_action("optimism", "0x" + "ab" * 20, "WETH")
        assert result is not None
        assert result["action"] == Action.FIND_BRIDGE_ROUTE.value

    def test_already_usdc(self) -> None:
        b = _make_behaviour()
        usdc = "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85"
        result = b._build_swap_to_usdc_action("optimism", usdc, "USDC")
        assert result is None

    def test_skip_olas(self) -> None:
        b = _make_behaviour()
        olas = "0xfc2e6e6bcbd49ccf3a5f029c79984372dcbfe527"
        result = b._build_swap_to_usdc_action("optimism", olas, "OLAS")
        assert result is None

    def test_no_usdc_address(self) -> None:
        b = _make_behaviour()
        result = b._build_swap_to_usdc_action("unknown_chain", "0x" + "ab" * 20, "TOK")
        assert result is None

    def test_with_description(self) -> None:
        b = _make_behaviour()
        result = b._build_swap_to_usdc_action(
            "optimism", "0x" + "ab" * 20, "WETH", description="swap for exit"
        )
        assert result["description"] == "swap for exit"

    def test_exception(self) -> None:
        b = _make_behaviour()
        with patch.object(b, "_get_usdc_address", side_effect=Exception("boom")):
            result = b._build_swap_to_usdc_action("optimism", "0x" + "ab" * 20, "WETH")
            assert result is None


class TestGetUsdcAddressException:
    """Test _get_usdc_address exception path."""

    def test_exception(self) -> None:
        b = _make_behaviour()
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.to_checksum_address",
            side_effect=Exception("bad address"),
        ):
            result = b._get_usdc_address("optimism")
            assert result is None


class TestVelodromePositionPrincipal:
    """Test get_velodrome_position_principal."""

    def test_success(self) -> None:
        b = _make_behaviour()
        b.params.velodrome_slipstream_helper_contract_addresses = {
            "optimism": "0x" + "55" * 20
        }
        b.contract_interact = _make_gen([100, 200])
        result = _exhaust(
            b.get_velodrome_position_principal(
                "optimism", "0xPM", 1, 79228162514264337593543950336
            )
        )
        assert result == (100, 200)

    def test_no_sugar_address(self) -> None:
        b = _make_behaviour()
        b.params.velodrome_slipstream_helper_contract_addresses = {}
        result = _exhaust(
            b.get_velodrome_position_principal(
                "optimism", "0xPM", 1, 79228162514264337593543950336
            )
        )
        assert result == (0, 0)

    def test_no_amounts(self) -> None:
        b = _make_behaviour()
        b.params.velodrome_slipstream_helper_contract_addresses = {
            "optimism": "0x" + "55" * 20
        }
        b.contract_interact = _make_gen(None)
        result = _exhaust(
            b.get_velodrome_position_principal(
                "optimism", "0xPM", 1, 79228162514264337593543950336
            )
        )
        assert result == (0, 0)


class TestVelodromeAmountsForLiquidity:
    """Test get_velodrome_amounts_for_liquidity."""

    def test_success(self) -> None:
        b = _make_behaviour()
        b.params.velodrome_slipstream_helper_contract_addresses = {
            "optimism": "0x" + "55" * 20
        }
        b.contract_interact = _make_gen([300, 400])
        result = _exhaust(
            b.get_velodrome_amounts_for_liquidity("optimism", 1000, 500, 2000, 100)
        )
        assert result == (300, 400)

    def test_no_sugar(self) -> None:
        b = _make_behaviour()
        b.params.velodrome_slipstream_helper_contract_addresses = {}
        result = _exhaust(
            b.get_velodrome_amounts_for_liquidity("optimism", 1000, 500, 2000, 100)
        )
        assert result == (0, 0)

    def test_no_amounts(self) -> None:
        b = _make_behaviour()
        b.params.velodrome_slipstream_helper_contract_addresses = {
            "optimism": "0x" + "55" * 20
        }
        b.contract_interact = _make_gen(None)
        result = _exhaust(
            b.get_velodrome_amounts_for_liquidity("optimism", 1000, 500, 2000, 100)
        )
        assert result == (0, 0)


class TestVelodromeSqrtRatioAtTick:
    """Test get_velodrome_sqrt_ratio_at_tick."""

    def test_success(self) -> None:
        b = _make_behaviour()
        b.params.velodrome_slipstream_helper_contract_addresses = {
            "optimism": "0x" + "55" * 20
        }
        b.contract_interact = _make_gen(79228162514264337593543950336)
        result = _exhaust(b.get_velodrome_sqrt_ratio_at_tick("optimism", 0))
        assert result == 79228162514264337593543950336

    def test_no_sugar(self) -> None:
        b = _make_behaviour()
        b.params.velodrome_slipstream_helper_contract_addresses = {}
        result = _exhaust(b.get_velodrome_sqrt_ratio_at_tick("optimism", 0))
        assert result == 0

    def test_no_result(self) -> None:
        b = _make_behaviour()
        b.params.velodrome_slipstream_helper_contract_addresses = {
            "optimism": "0x" + "55" * 20
        }
        b.contract_interact = _make_gen(None)
        result = _exhaust(b.get_velodrome_sqrt_ratio_at_tick("optimism", 0))
        assert result == 0


class TestBuildUnstakeActionFull:
    """Test _build_unstake_lp_tokens_action full coverage."""

    def test_cl_pool_with_positions(self) -> None:
        b = _make_behaviour()
        pos = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPool",
            "is_cl_pool": True,
            "gauge_address": "0xGauge",
            "positions": [{"token_id": 1}, {"token_id": 2}],
        }
        result = b._build_unstake_lp_tokens_action(pos)
        assert result["is_cl_pool"] is True
        assert result["token_ids"] == [1, 2]
        assert result["gauge_address"] == "0xGauge"

    def test_cl_pool_single_token_id(self) -> None:
        b = _make_behaviour()
        pos = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPool",
            "is_cl_pool": True,
            "positions": [],
            "token_id": 42,
        }
        result = b._build_unstake_lp_tokens_action(pos)
        assert result["token_ids"] == [42]

    def test_cl_pool_no_token_ids(self) -> None:
        b = _make_behaviour()
        pos = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPool",
            "is_cl_pool": True,
            "positions": [],
        }
        result = b._build_unstake_lp_tokens_action(pos)
        assert result is None

    def test_regular_pool(self) -> None:
        b = _make_behaviour()
        pos = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPool",
            "is_cl_pool": False,
        }
        result = b._build_unstake_lp_tokens_action(pos)
        assert result["is_cl_pool"] is False

    def test_non_velodrome(self) -> None:
        b = _make_behaviour()
        pos = {
            "dex_type": "UniswapV3",
            "chain": "optimism",
            "pool_address": "0xPool",
        }
        result = b._build_unstake_lp_tokens_action(pos)
        assert result is None

    def test_missing_params(self) -> None:
        b = _make_behaviour()
        pos = {"dex_type": "velodrome"}
        result = b._build_unstake_lp_tokens_action(pos)
        assert result is None

    def test_no_safe_address(self) -> None:
        b = _make_behaviour()
        b.params.safe_contract_addresses = {}
        pos = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPool",
            "is_cl_pool": False,
        }
        result = b._build_unstake_lp_tokens_action(pos)
        assert result is None

    def test_exception(self) -> None:
        b = _make_behaviour()
        pos = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPool",
            "is_cl_pool": True,
            "positions": None,  # Will cause TypeError
        }
        result = b._build_unstake_lp_tokens_action(pos)
        assert result is None


class TestAgentRegistryMethods:
    """Test MirrorDB agent registry generator methods."""

    def test_get_agent_type_by_name(self) -> None:
        b = _make_behaviour()
        b._call_mirrordb = _make_gen({"type_id": "t1", "type_name": "optimus"})
        result = _exhaust(b.get_agent_type_by_name("optimus"))
        assert result["type_id"] == "t1"

    def test_create_agent_type(self) -> None:
        b = _make_behaviour()
        b._call_mirrordb = _make_gen({"type_id": "t2"})
        result = _exhaust(b.create_agent_type("optimus", "desc"))
        assert result["type_id"] == "t2"

    def test_get_attr_def_by_name(self) -> None:
        b = _make_behaviour()
        b._call_mirrordb = _make_gen({"attr_id": "a1"})
        result = _exhaust(b.get_attr_def_by_name("metrics"))
        assert result["attr_id"] == "a1"

    def test_create_attribute_definition(self) -> None:
        b = _make_behaviour()
        b._call_mirrordb = _make_gen({"attr_id": "a2"})
        b.sign_message = _make_gen("sig_hex")
        result = _exhaust(
            b.create_attribute_definition("t1", "metrics", "json", True, "{}", "agent1")
        )
        assert result["attr_id"] == "a2"

    def test_create_attribute_definition_no_signature(self) -> None:
        b = _make_behaviour()
        b.sign_message = _make_gen(None)
        result = _exhaust(
            b.create_attribute_definition("t1", "metrics", "json", True, "{}", "agent1")
        )
        assert result is None

    def test_get_agent_registry_by_address(self) -> None:
        b = _make_behaviour()
        b._call_mirrordb = _make_gen({"agent_id": "ag1"})
        result = _exhaust(b.get_agent_registry_by_address("0xAgent"))
        assert result["agent_id"] == "ag1"

    def test_create_agent_registry(self) -> None:
        b = _make_behaviour()
        b._call_mirrordb = _make_gen({"agent_id": "ag2"})
        result = _exhaust(b.create_agent_registry("agent_name", "t1", "0xAgent"))
        assert result["agent_id"] == "ag2"

    def test_create_agent_attribute(self) -> None:
        b = _make_behaviour()
        b._call_mirrordb = _make_gen({"attr_val_id": "av1"})
        b.sign_message = _make_gen("sig_hex")
        result = _exhaust(b.create_agent_attribute("ag1", "a1", json_value='{"k":"v"}'))
        assert result["attr_val_id"] == "av1"

    def test_create_agent_attribute_no_signature(self) -> None:
        b = _make_behaviour()
        b.sign_message = _make_gen(None)
        result = _exhaust(b.create_agent_attribute("ag1", "a1"))
        assert result is None


class TestGetOrCreateAgentType:
    """Test _get_or_create_agent_type."""

    def test_cached(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen({"agent_type": json.dumps({"type_id": "t1"})})
        result = _exhaust(b._get_or_create_agent_type("0xAddr"))
        assert result["type_id"] == "t1"

    def test_fetch_existing(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        b.get_agent_type_by_name = _make_gen({"type_id": "t2"})
        result = _exhaust(b._get_or_create_agent_type("0xAddr"))
        assert result["type_id"] == "t2"

    def test_create_new(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        b.get_agent_type_by_name = _make_gen(None)
        b.create_agent_type = _make_gen({"type_id": "t3"})
        b._write_kv = _make_gen(True)
        result = _exhaust(b._get_or_create_agent_type("0xAddr"))
        assert result["type_id"] == "t3"


class TestGetOrCreateAttrDef:
    """Test _get_or_create_attr_def."""

    def test_cached(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen({"attr_def": json.dumps({"attr_id": "a1"})})
        result = _exhaust(b._get_or_create_attr_def("t1", "ag1"))
        assert result["attr_id"] == "a1"

    def test_fetch_existing(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        b.get_attr_def_by_name = _make_gen({"attr_id": "a2"})
        result = _exhaust(b._get_or_create_attr_def("t1", "ag1"))
        assert result["attr_id"] == "a2"

    def test_create_new(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        b.get_attr_def_by_name = _make_gen(None)
        b.create_attribute_definition = _make_gen({"attr_id": "a3"})
        b._write_kv = _make_gen(True)
        result = _exhaust(b._get_or_create_attr_def("t1", "ag1"))
        assert result["attr_id"] == "a3"


class TestGetOrCreateAgentRegistry:
    """Test _get_or_create_agent_registry."""

    def test_cached(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen({"agent_registry": json.dumps({"agent_id": "ag1"})})
        result = _exhaust(b._get_or_create_agent_registry())
        assert result["agent_id"] == "ag1"

    def test_fetch_existing(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        b._get_or_create_agent_type = _make_gen({"type_id": "t1"})
        b.get_agent_registry_by_address = _make_gen({"agent_id": "ag2"})
        result = _exhaust(b._get_or_create_agent_registry())
        assert result["agent_id"] == "ag2"

    def test_create_new(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        b._get_or_create_agent_type = _make_gen({"type_id": "t1"})
        b.get_agent_registry_by_address = _make_gen(None)
        b.create_agent_registry = _make_gen({"agent_id": "ag3"})
        b._write_kv = _make_gen(True)
        with patch.object(b, "generate_name", return_value="test-name01"):
            result = _exhaust(b._get_or_create_agent_registry())
            assert result["agent_id"] == "ag3"

    def test_no_agent_type(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        b._get_or_create_agent_type = _make_gen(None)
        result = _exhaust(b._get_or_create_agent_registry())
        assert result is None

    def test_exception(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)

        def bad_gen(*a, **kw):
            raise Exception("fail")
            yield  # noqa: unreachable

        b._get_or_create_agent_type = bad_gen
        result = _exhaust(b._get_or_create_agent_registry())
        assert result is None


class TestUpdateAirdropRewardsException:
    """Test _update_airdrop_rewards exception path."""

    def test_exception(self) -> None:
        b = _make_behaviour()

        def bad_read(*a, **kw):
            raise Exception("fail")
            yield  # noqa: unreachable

        b._read_kv = bad_read
        _exhaust(b._update_airdrop_rewards(100, "optimism", "0xhash"))
        b.context.logger.error.assert_called()


class TestFetchRewardBalances:
    """Test _fetch_reward_balances."""

    def test_no_safe_address(self) -> None:
        b = _make_behaviour()
        b.params.safe_contract_addresses = {}
        result = _exhaust(b._fetch_reward_balances("optimism"))
        assert result == []

    def test_no_reward_tokens(self) -> None:
        b = _make_behaviour()
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.REWARD_TOKEN_ADDRESSES",
            {},
        ):
            result = _exhaust(b._fetch_reward_balances("optimism"))
            assert result == []

    def test_with_balance(self) -> None:
        b = _make_behaviour()
        b._get_token_balance = _make_gen(1000)
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.REWARD_TOKEN_ADDRESSES",
            {"optimism": {"0x" + "11" * 20: "OLAS"}},
        ):
            result = _exhaust(b._fetch_reward_balances("optimism"))
            assert len(result) == 1
            assert result[0]["asset_symbol"] == "OLAS"

    def test_zero_balance(self) -> None:
        b = _make_behaviour()
        b._get_token_balance = _make_gen(0)
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.REWARD_TOKEN_ADDRESSES",
            {"optimism": {"0x" + "11" * 20: "OLAS"}},
        ):
            result = _exhaust(b._fetch_reward_balances("optimism"))
            assert result == []

    def test_token_exception(self) -> None:
        b = _make_behaviour()

        def bad_balance(*a, **kw):
            raise Exception("fail")
            yield  # noqa: unreachable

        b._get_token_balance = bad_balance
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.REWARD_TOKEN_ADDRESSES",
            {"optimism": {"0x" + "11" * 20: "OLAS"}},
        ):
            result = _exhaust(b._fetch_reward_balances("optimism"))
            assert result == []

    def test_outer_exception(self) -> None:
        b = _make_behaviour()
        b.params.safe_contract_addresses = {"optimism": "0x" + "aa" * 20}
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.REWARD_TOKEN_ADDRESSES",
            side_effect=Exception("boom"),
        ):
            result = _exhaust(b._fetch_reward_balances("optimism"))
            assert result == []


class TestGetLastKnownPrice:
    """Test _get_last_known_price."""

    def test_found_current_price(self) -> None:
        b = _make_behaviour()
        cache_data = json.dumps({"current": [5.0, time.time()]})
        key = b._get_price_cache_key("0xToken", None)

        # We need to mock _read_kv to return data for any cache key
        def mock_read(*args, **kwargs):
            keys = args[0] if args else kwargs.get("keys", ())
            k = keys[0]
            yield
            return {k: cache_data}

        b._read_kv = mock_read
        result = _exhaust(b._get_last_known_price("0xToken"))
        assert result == 5.0

    def test_not_found(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        result = _exhaust(b._get_last_known_price("0xToken"))
        assert result is None

    def test_exception(self) -> None:
        b = _make_behaviour()

        def bad_read(*a, **kw):
            raise Exception("fail")
            yield  # noqa: unreachable

        b._read_kv = bad_read
        result = _exhaust(b._get_last_known_price("0xToken"))
        assert result is None

    def test_invalid_cache_data(self) -> None:
        b = _make_behaviour()

        def mock_read(*args, **kwargs):
            keys = args[0] if args else kwargs.get("keys", ())
            k = keys[0]
            yield
            return {k: "not json"}

        b._read_kv = mock_read
        result = _exhaust(b._get_last_known_price("0xToken"))
        # Should continue to next dates and eventually return None
        assert result is None


class TestFetchOusdtBalance:
    """Test _fetch_ousdt_balance."""

    def test_no_safe_address(self) -> None:
        b = _make_behaviour()
        b.params.safe_contract_addresses = {}
        result = _exhaust(b._fetch_ousdt_balance("optimism"))
        assert result is None

    def test_with_balance(self) -> None:
        b = _make_behaviour()
        b._get_token_balance = _make_gen(1000)
        result = _exhaust(b._fetch_ousdt_balance("optimism"))
        assert result["asset_symbol"] == "oUSDT"
        assert result["balance"] == 1000

    def test_zero_balance(self) -> None:
        b = _make_behaviour()
        b._get_token_balance = _make_gen(0)
        result = _exhaust(b._fetch_ousdt_balance("optimism"))
        assert result is None

    def test_exception(self) -> None:
        b = _make_behaviour()

        def bad_balance(*a, **kw):
            raise Exception("fail")
            yield  # noqa: unreachable

        b._get_token_balance = bad_balance
        result = _exhaust(b._fetch_ousdt_balance("optimism"))
        assert result is None


class TestGetServiceStakingState:
    """Test _get_service_staking_state."""

    def test_no_service_id(self) -> None:
        b = _make_behaviour()
        b.params.on_chain_service_id = None
        _exhaust(b._get_service_staking_state("optimism"))
        assert b.service_staking_state == StakingState.UNSTAKED

    def test_returns_state(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(StakingState.STAKED.value)
        _exhaust(b._get_service_staking_state("optimism"))
        assert b.service_staking_state == StakingState.STAKED

    def test_none_state(self) -> None:
        b = _make_behaviour()
        b.contract_interact = _make_gen(None)
        _exhaust(b._get_service_staking_state("optimism"))
        assert b.service_staking_state == StakingState.UNSTAKED


class TestGetServiceInfo:
    """Test _get_service_info."""

    def test_no_service_id(self) -> None:
        b = _make_behaviour()
        b.params.on_chain_service_id = None
        result = _exhaust(b._get_service_info("optimism"))
        assert result is None

    def test_returns_info(self) -> None:
        b = _make_behaviour()
        expected = (1, 2, (3, 4))
        b.contract_interact = _make_gen(expected)
        result = _exhaust(b._get_service_info("optimism"))
        assert result == expected


class TestGetEthRemainingAmountEdge:
    """Test get_eth_remaining_amount edge cases."""

    def test_on_chain_matches_cached(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen({ETH_REMAINING_KEY: "1000"})
        b._get_native_balance = _make_gen(1000)
        result = _exhaust(b.get_eth_remaining_amount())
        assert result == 1000

    def test_on_chain_none_returns_cached(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen({ETH_REMAINING_KEY: "1000"})
        b._get_native_balance = _make_gen(None)
        result = _exhaust(b.get_eth_remaining_amount())
        assert result == 1000

    def test_on_chain_mismatch_syncs(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen({ETH_REMAINING_KEY: "500"})
        b._get_native_balance = _make_gen(1000)
        b._write_kv = _make_gen(True)
        result = _exhaust(b.get_eth_remaining_amount())
        assert result == 1000


class TestUpdateEthRemainingAmount:
    """Test update_eth_remaining_amount."""

    def test_update(self) -> None:
        b = _make_behaviour()
        b.get_eth_remaining_amount = _make_gen(1000)
        b._write_kv = _make_gen(True)
        _exhaust(b.update_eth_remaining_amount(300))


class TestResetEthRemainingAmount:
    """Test reset_eth_remaining_amount."""

    def test_reset(self) -> None:
        b = _make_behaviour()
        b._get_native_balance = _make_gen(5000)
        b._write_kv = _make_gen(True)
        result = _exhaust(b.reset_eth_remaining_amount())
        assert result == 5000

    def test_reset_none_balance(self) -> None:
        b = _make_behaviour()
        b._get_native_balance = _make_gen(None)
        b._write_kv = _make_gen(True)
        result = _exhaust(b.reset_eth_remaining_amount())
        assert result == 0


class TestStoreEntryCosts:
    """Test _store_entry_costs."""

    def test_success(self) -> None:
        b = _make_behaviour()
        b._get_all_entry_costs = _make_gen({})
        b._write_kv = _make_gen(True)
        _exhaust(b._store_entry_costs("optimism", "pos1", 0.5))

    def test_exception(self) -> None:
        b = _make_behaviour()

        def bad_gen(*a, **kw):
            raise Exception("fail")
            yield  # noqa: unreachable

        b._get_all_entry_costs = bad_gen
        _exhaust(b._store_entry_costs("optimism", "pos1", 0.5))
        b.context.logger.error.assert_called()


class TestGetModeBalancesFromExplorerApi:
    """Test _get_mode_balances_from_explorer_api."""

    def test_no_safe_address(self) -> None:
        b = _make_behaviour()
        b.params.safe_contract_addresses = {}
        result = _exhaust(b._get_mode_balances_from_explorer_api())
        assert result == []

    def test_with_eth_and_tokens(self) -> None:
        b = _make_behaviour()
        b._get_native_balance = _make_gen(1000)
        b._fetch_mode_token_balances = _make_gen(
            [
                {
                    "asset_symbol": "USDC",
                    "asset_type": "erc_20",
                    "address": "0xA",
                    "balance": 500,
                }
            ]
        )
        result = _exhaust(b._get_mode_balances_from_explorer_api())
        assert len(result) == 2  # ETH + USDC

    def test_zero_eth(self) -> None:
        b = _make_behaviour()
        b._get_native_balance = _make_gen(0)
        b._fetch_mode_token_balances = _make_gen([])
        result = _exhaust(b._get_mode_balances_from_explorer_api())
        assert result == []


class TestUpdateAirdropRewardsFull:
    """Test _update_airdrop_rewards full paths."""

    def test_dedup_skips(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen({"airdrop_processed_optimism_0xhash": "true"})
        _exhaust(b._update_airdrop_rewards(100, "optimism", "0xhash"))

    def test_new_tx(self) -> None:
        b = _make_behaviour()
        # First call returns no processed key, second returns 0 total
        call_count = [0]

        def mock_read(*args, **kwargs):
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return {}  # not processed yet
            return {}  # no total yet

        b._read_kv = mock_read
        b._write_kv = _make_gen(True)
        _exhaust(b._update_airdrop_rewards(100, "optimism", "0xhash"))

    def test_no_tx_hash(self) -> None:
        b = _make_behaviour()
        b._get_total_airdrop_rewards = _make_gen(50)
        b._write_kv = _make_gen(True)
        _exhaust(b._update_airdrop_rewards(100, "optimism"))


class TestStoreDataIOError:
    """Test _store_data with IOError during json.dump."""

    def test_io_error(self) -> None:
        b = _make_behaviour()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            with patch("json.dump", side_effect=IOError("disk full")):
                b._store_data({"key": "val"}, "test_attr", fname)
                b.context.logger.error.assert_called()
        finally:
            os.unlink(fname)


class TestGetCachedPriceCurrentPrice:
    """Test _get_cached_price with current price (no date key)."""

    def test_current_price_valid_ttl(self) -> None:
        b = _make_behaviour()
        # The _get_cached_price with date="" triggers the else branch at line 1357
        # But line 1354 checks `if date:` — we need to pass an empty string for date
        # Actually, looking at code: when date is truthy, it returns price_data.get(date)
        # The else branch at 1357 is for when date is falsy
        cache_key = b._get_price_cache_key("0xToken", "")
        cache_data = json.dumps({"current": [5.0, time.time()]})
        b._read_kv = _make_gen({cache_key: cache_data})
        # Need to mock _get_current_timestamp to return recent time
        with patch.object(b, "_get_current_timestamp", return_value=time.time()):
            result = _exhaust(b._get_cached_price("0xToken", ""))
            assert result == 5.0

    def test_current_price_expired_ttl(self) -> None:
        b = _make_behaviour()
        cache_key = b._get_price_cache_key("0xToken", "")
        cache_data = json.dumps({"current": [5.0, 0]})  # old timestamp
        b._read_kv = _make_gen({cache_key: cache_data})
        with patch.object(b, "_get_current_timestamp", return_value=time.time()):
            result = _exhaust(b._get_cached_price("0xToken", ""))
            assert result is None

    def test_current_price_no_current_data(self) -> None:
        b = _make_behaviour()
        cache_key = b._get_price_cache_key("0xToken", "")
        cache_data = json.dumps({})
        b._read_kv = _make_gen({cache_key: cache_data})
        result = _exhaust(b._get_cached_price("0xToken", ""))
        assert result is None


class TestCachePriceNoDate:
    """Test _cache_price without date (current price mode)."""

    def test_cache_current_price(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        b._write_kv = _make_gen(True)
        with patch.object(b, "_get_current_timestamp", return_value=1700000000):
            _exhaust(b._cache_price("0xToken", 5.0, ""))


class TestFetchZeroAddressPrice:
    """Test _fetch_zero_address_price."""

    def test_returns_eth_price(self) -> None:
        b = _make_behaviour()
        b._fetch_coin_price = _make_gen(1800.0)
        result = _exhaust(b._fetch_zero_address_price())
        assert result == 1800.0


class TestCalcInitInvestEdge:
    """Test calculate_initial_investment_value additional edge cases."""

    def test_token1_decimals_none(self) -> None:
        b = _make_behaviour()
        call_count = [0]

        def mock_decimals(*args, **kwargs):
            call_count[0] += 1
            yield
            return 18 if call_count[0] == 1 else None

        b._get_token_decimals = mock_decimals
        pos = {
            "token0": "0xT0",
            "token0_symbol": "A",
            "token1": "0xT1",
            "token1_symbol": "B",
            "amount0": 10**18,
            "amount1": 10**18,
            "timestamp": 1700000000,
            "chain": "optimism",
        }
        result = _exhaust(b.calculate_initial_investment_value(pos))
        assert result is None

    def test_token1_no_historical_price(self) -> None:
        b = _make_behaviour()
        b._get_token_decimals = _make_gen(18)
        b._fetch_historical_token_prices = _make_gen({"0xT0": 1.0})
        pos = {
            "token0": "0xT0",
            "token0_symbol": "A",
            "token1": "0xT1",
            "token1_symbol": "B",
            "amount0": 10**18,
            "amount1": 10**18,
            "timestamp": 1700000000,
            "chain": "optimism",
        }
        result = _exhaust(b.calculate_initial_investment_value(pos))
        assert result is None

    def test_enter_timestamp_fallback(self) -> None:
        b = _make_behaviour()
        b._get_token_decimals = _make_gen(6)
        b._fetch_historical_token_prices = _make_gen({"0xT0": 1.0})
        pos = {
            "token0": "0xT0",
            "token0_symbol": "USDC",
            "amount0": 1_000_000,
            "enter_timestamp": 1700000000,  # Using enter_timestamp instead of timestamp
            "chain": "optimism",
        }
        result = _exhaust(b.calculate_initial_investment_value(pos))
        assert result == 1.0


class TestFetchTokenPricesSmaEdge:
    """Test _fetch_token_prices_sma edge cases."""

    def test_prices_truncated_to_num_hours(self) -> None:
        """Test that prices list is truncated when longer than num_hours."""
        b = _make_behaviour()
        b._get_token_symbol = _make_gen("ETH")
        b.sleep = _make_gen(None)
        # Create 30 data points; num_hours=2 should truncate to last 2
        prices = [[i * 3600, 100 + i] for i in range(30)]
        with patch.object(b, "get_coin_id_from_symbol", return_value="ethereum"):
            b.coingecko.request = MagicMock(return_value=(True, {"prices": prices}))
            result = _exhaust(
                b._fetch_token_prices_sma("0xToken", "optimism", num_hours=2)
            )
            # Last 2 prices: 128 and 129
            assert result == (128.0 + 129.0) / 2


class TestGetOrCreateAgentTypeFailure:
    """Test _get_or_create_agent_type when creation fails."""

    def test_create_failure_raises(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        b.get_agent_type_by_name = _make_gen(None)
        b.create_agent_type = _make_gen(None)  # creation fails
        with pytest.raises(Exception, match="Failed to create agent type"):
            _exhaust(b._get_or_create_agent_type("0xAddr"))


class TestGetOrCreateAttrDefFailure:
    """Test _get_or_create_attr_def when creation fails."""

    def test_create_failure_raises(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        b.get_attr_def_by_name = _make_gen(None)
        b.create_attribute_definition = _make_gen(None)  # creation fails
        with pytest.raises(Exception, match="Failed to create attribute definition"):
            _exhaust(b._get_or_create_attr_def("t1", "ag1"))


class TestGetOrCreateAgentRegistryCreateFails:
    """Test _get_or_create_agent_registry when create_agent_registry returns None."""

    def test_create_fails(self) -> None:
        b = _make_behaviour()
        b._read_kv = _make_gen(None)
        b._get_or_create_agent_type = _make_gen({"type_id": "t1"})
        b.get_agent_registry_by_address = _make_gen(None)
        b.create_agent_registry = _make_gen(None)  # creation fails
        with patch.object(b, "generate_name", return_value="test-name01"):
            result = _exhaust(b._get_or_create_agent_registry())
            assert result is None


class TestGetLastKnownPriceHistorical:
    """Test _get_last_known_price finding historical price."""

    def test_found_historical_price(self) -> None:
        b = _make_behaviour()
        from datetime import datetime

        # Mock _get_current_timestamp to return a fixed time
        with patch.object(b, "_get_current_timestamp", return_value=1700000000):
            date_str = datetime.utcfromtimestamp(1700000000).strftime("%d-%m-%Y")
            cache_data = json.dumps({date_str: 42.5})

            def mock_read(*args, **kwargs):
                keys = args[0] if args else kwargs.get("keys", ())
                k = keys[0]
                yield
                return {k: cache_data}

            b._read_kv = mock_read
            result = _exhaust(b._get_last_known_price("0xToken"))
            assert result == 42.5


class TestFetchRewardBalancesOuterException:
    """Test _fetch_reward_balances outer exception."""

    def test_outer_exception_catches(self) -> None:
        b = _make_behaviour()
        # Make safe_contract_addresses.get raise an exception
        b.params.safe_contract_addresses = MagicMock()
        b.params.safe_contract_addresses.get.side_effect = Exception("boom")
        result = _exhaust(b._fetch_reward_balances("optimism"))
        assert result == []


class TestPaginationMultiPage:
    """Test pagination methods with multiple pages."""

    def test_safe_balances_two_pages(self) -> None:
        b = _make_behaviour()
        call_count = [0]

        def mock_request(*args, **kwargs):
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return (True, {"results": [{"tokenAddress": None}], "next": "page2"})
            return (
                True,
                {"results": [{"tokenAddress": "0x" + "ab" * 20}], "next": None},
            )

        b._request_with_retries = mock_request
        result = _exhaust(b._fetch_safe_balances_with_pagination("0xSafe"))
        assert len(result) == 2

    def test_mode_tokens_two_pages(self) -> None:
        b = _make_behaviour()
        call_count = [0]

        def mock_request(*args, **kwargs):
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return (
                    True,
                    {
                        "items": [{"token": {"address": "0xA"}}],
                        "next_page_params": {"page": 2},
                    },
                )
            return (
                True,
                {"items": [{"token": {"address": "0xB"}}], "next_page_params": None},
            )

        b._request_with_retries = mock_request
        result = _exhaust(b._fetch_mode_tokens_with_pagination("0xSafe"))
        assert len(result) == 2


class TestRequestWithRetriesMultiIteration:
    """Test _request_with_retries with retry loops (multi-iteration)."""

    def test_rate_limited_then_success(self) -> None:
        """Rate limited on first attempt, succeeds on second."""
        b = _make_behaviour()
        call_count = [0]

        def mock_http(*args, **kwargs):
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                resp = MagicMock()
                resp.status_code = 429
                resp.body = json.dumps({"error": "rate limited"}).encode("utf-8")
                return resp
            resp = MagicMock()
            resp.status_code = 200
            resp.body = json.dumps({"result": "ok"}).encode("utf-8")
            return resp

        b.get_http_response = mock_http
        b.sleep = _make_gen(None)
        success, data = _exhaust(
            b._request_with_retries(
                endpoint="https://example.com",
                method="GET",
                rate_limited_callback=lambda: None,
                max_retries=3,
            )
        )
        assert success is True

    def test_503_then_success(self) -> None:
        """503 on first attempt, succeeds on second."""
        b = _make_behaviour()
        call_count = [0]

        def mock_http(*args, **kwargs):
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                resp = MagicMock()
                resp.status_code = 503
                resp.body = json.dumps({"error": "unavailable"}).encode("utf-8")
                return resp
            resp = MagicMock()
            resp.status_code = 200
            resp.body = json.dumps({"result": "ok"}).encode("utf-8")
            return resp

        b.get_http_response = mock_http
        b.sleep = _make_gen(None)
        success, data = _exhaust(
            b._request_with_retries(
                endpoint="https://example.com",
                method="GET",
                rate_limited_callback=lambda: None,
                max_retries=3,
            )
        )
        assert success is True

    def test_other_error_then_success(self) -> None:
        """500 error on first attempt, succeeds on second."""
        b = _make_behaviour()
        call_count = [0]

        def mock_http(*args, **kwargs):
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                resp = MagicMock()
                resp.status_code = 500
                resp.body = json.dumps({"error": "server"}).encode("utf-8")
                return resp
            resp = MagicMock()
            resp.status_code = 200
            resp.body = json.dumps({"result": "ok"}).encode("utf-8")
            return resp

        b.get_http_response = mock_http
        b.sleep = _make_gen(None)
        success, data = _exhaust(
            b._request_with_retries(
                endpoint="https://example.com",
                method="GET",
                rate_limited_callback=lambda: None,
                max_retries=3,
                retry_wait=0,
            )
        )
        assert success is True


class TestFetchTokenPricesSmaEmptyAfterFilter:
    """Test SMA with entries that produce empty prices after filter."""

    def test_empty_after_filter(self) -> None:
        b = _make_behaviour()
        b._get_token_symbol = _make_gen("ETH")
        b.sleep = _make_gen(None)
        # Entries with len < 2 will be filtered out
        with patch.object(b, "get_coin_id_from_symbol", return_value="ethereum"):
            b.coingecko.request = MagicMock(
                return_value=(True, {"prices": [[1700000000]]})  # single element entry
            )
            result = _exhaust(b._fetch_token_prices_sma("0xToken", "optimism"))
            assert result is None


class TestExecuteStrategyIsolatedNamespace:
    """Test execute_strategy uses isolated namespaces instead of globals."""

    def test_isolated_namespace(self) -> None:
        import packages.valory.skills.liquidity_trader_abci.behaviours.base as base_mod

        method_name = "_test_strategy_method"
        strategy_code = f"def {method_name}(**kwargs):\n    return {{'result': 'new'}}"
        executables = {"test_strat": (strategy_code, method_name)}

        result = execute_strategy("test_strat", executables)
        assert result["result"] == "new"

        # Verify the function was NOT injected into module globals
        assert method_name not in base_mod.__dict__


class TestFetchOusdtBalanceExceptionInner:
    """Test _fetch_ousdt_balance with exception during balance fetch."""

    def test_exception_during_get_balance(self) -> None:
        b = _make_behaviour()

        def bad_gen(*a, **kw):
            raise Exception("contract error")
            yield  # noqa: unreachable

        b._get_token_balance = bad_gen
        result = _exhaust(b._fetch_ousdt_balance("optimism"))
        assert result is None


class TestAdjustCurrentPositionsFalseBranches:
    """Test _adjust_current_positions_for_backward_compatibility false branches."""

    def test_fields_already_present(self) -> None:
        """When all backward-compat fields already exist, the if-not-in checks are False."""
        b = _make_behaviour()
        with patch.object(b, "store_current_positions"):
            _exhaust(
                b._adjust_current_positions_for_backward_compatibility(
                    [
                        {
                            "chain": "optimism",
                            "status": "open",
                            "entry_cost": 1.0,
                            "min_hold_days": 7,
                            "cost_recovered": True,
                            "principal_usd": 100.0,
                            "entry_apr": 10.0,
                        }
                    ]
                )
            )
            pos = b.current_positions[0]
            assert pos["entry_cost"] == 1.0
            assert pos["min_hold_days"] == 7
            assert pos["cost_recovered"] is True
            assert pos["principal_usd"] == 100.0
            assert pos["entry_apr"] == 10.0

    def test_assets_not_list(self) -> None:
        """When assets key exists but is not a list."""
        b = _make_behaviour()
        with patch.object(b, "store_current_positions"):
            _exhaust(
                b._adjust_current_positions_for_backward_compatibility(
                    [{"chain": "optimism", "assets": "not_a_list"}]
                )
            )
            assert "token0" not in b.current_positions[0]

    def test_assets_empty_list(self) -> None:
        """When assets key is an empty list."""
        b = _make_behaviour()
        with patch.object(b, "store_current_positions"):
            _exhaust(
                b._adjust_current_positions_for_backward_compatibility(
                    [{"chain": "optimism", "assets": []}]
                )
            )
            assert "token0" not in b.current_positions[0]


class TestReadAgentPerformanceFalseBranch:
    """Test read_agent_performance when data already exists."""

    def test_with_existing_data(self) -> None:
        b = _make_behaviour()
        b.agent_performance = {"timestamp": 123, "metrics": [], "agent_behavior": None}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(b.agent_performance, f)
            fname = f.name
        try:
            b.agent_performance_filepath = fname
            b.read_agent_performance()
            assert b.agent_performance["timestamp"] == 123
        finally:
            os.unlink(fname)


class TestGetBalanceChainMatch:
    """Test _get_balance with matching chain and token."""

    def test_matching_chain_and_token(self) -> None:
        b = _make_behaviour()
        token = "0x" + "ab" * 20
        positions = [
            {
                "chain": "optimism",
                "assets": [
                    {"address": token, "balance": 1000},
                ],
            }
        ]
        result = b._get_balance("optimism", token, positions)
        assert result == 1000

    def test_no_matching_chain(self) -> None:
        b = _make_behaviour()
        positions = [{"chain": "mode", "assets": []}]
        result = b._get_balance("optimism", "0xToken", positions)
        assert result is None


class TestFetchTokenPricesPriceNone:
    """Test _fetch_token_prices when price is None (not added to dict)."""

    def test_price_none_not_added(self) -> None:
        b = _make_behaviour()
        b._fetch_token_price = _make_gen(None)
        token_balances = [{"token": "0x" + "ab" * 20, "chain": "optimism"}]
        result = _exhaust(b._fetch_token_prices(token_balances))
        assert result == {}


class TestFetchHistoricalTokenPricesFalsy:
    """Test _fetch_historical_token_prices when price is falsy."""

    def test_price_zero_not_added(self) -> None:
        b = _make_behaviour()
        b._fetch_historical_token_price = _make_gen(0)
        with patch.object(b, "get_coin_id_from_symbol", return_value="usd-coin"):
            result = _exhaust(
                b._fetch_historical_token_prices(
                    [["USDC", "0xAddr"]], "01-01-2024", "optimism"
                )
            )
            assert result == {}


class TestUseX402True:
    """Test methods with use_x402=True to cover the other branch."""

    def test_fetch_token_price_x402(self) -> None:
        b = _make_behaviour()
        b.params.use_x402 = True
        b._get_cached_price = _make_gen(None)
        token_addr = "0x" + "ab" * 20
        b.coingecko.request = MagicMock(
            return_value=(True, {token_addr.lower(): {"usd": 1.0}})
        )
        b._cache_price = _make_gen(None)
        b._get_last_known_price = _make_gen(None)
        # Need eoa_account property mock
        with patch.object(
            type(b), "eoa_account", new_callable=PropertyMock, return_value=MagicMock()
        ):
            result = _exhaust(b._fetch_token_price(token_addr, "optimism"))
            assert result == 1.0

    def test_fetch_coin_price_x402(self) -> None:
        b = _make_behaviour()
        b.params.use_x402 = True
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(
            return_value=(True, {"ethereum": {"usd": 1800.0}})
        )
        b._cache_price = _make_gen(None)
        with patch.object(
            type(b), "eoa_account", new_callable=PropertyMock, return_value=MagicMock()
        ):
            result = _exhaust(b._fetch_coin_price("ethereum"))
            assert result == 1800.0

    def test_fetch_historical_price_x402(self) -> None:
        b = _make_behaviour()
        b.params.use_x402 = True
        b._get_cached_price = _make_gen(None)
        b.coingecko.request = MagicMock(
            return_value=(True, {"market_data": {"current_price": {"usd": 1.0}}})
        )
        b._cache_price = _make_gen(None)
        with patch.object(
            type(b), "eoa_account", new_callable=PropertyMock, return_value=MagicMock()
        ):
            result = _exhaust(b._fetch_historical_token_price("usd-coin", "01-01-2024"))
            assert result == 1.0

    def test_fetch_sma_x402(self) -> None:
        b = _make_behaviour()
        b.params.use_x402 = True
        b._get_token_symbol = _make_gen("ETH")
        b.sleep = _make_gen(None)
        with patch.object(b, "get_coin_id_from_symbol", return_value="ethereum"):
            b.coingecko.request = MagicMock(
                return_value=(True, {"prices": [[1, 100.0]]})
            )
            with patch.object(
                type(b),
                "eoa_account",
                new_callable=PropertyMock,
                return_value=MagicMock(),
            ):
                result = _exhaust(b._fetch_token_prices_sma("0xToken", "optimism"))
                assert result == 100.0


class TestFetchSmaResponseNotDict:
    """Test SMA when response is not a dict (non-dict check at 1953)."""

    def test_response_is_list(self) -> None:
        b = _make_behaviour()
        b._get_token_symbol = _make_gen("ETH")
        b.sleep = _make_gen(None)
        with patch.object(b, "get_coin_id_from_symbol", return_value="ethereum"):
            # Return a list instead of dict - should still extract prices if it has "prices" key
            b.coingecko.request = MagicMock(
                return_value=(True, {"prices": [[1, 100.0]]})
            )
            result = _exhaust(b._fetch_token_prices_sma("0xToken", "optimism"))
            assert result == 100.0


class TestUpdateAccumulatedRewardsNoOlas:
    """Test update_accumulated_rewards_for_chain when no OLAS address."""

    def test_no_olas_address(self) -> None:
        b = _make_behaviour()
        b.should_update_rewards_from_subgraph = _make_gen(True)
        b.query_service_rewards = _make_gen({"olasRewardsEarned": "500"})
        b._write_kv = _make_gen(True)
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.OLAS_ADDRESSES",
            {},
        ):
            _exhaust(b.update_accumulated_rewards_for_chain("optimism"))


class TestBuildExitPoolActionBaseEmptyPositions:
    """Test _build_exit_pool_action_base with CL pool but empty positions."""

    def test_cl_pool_empty_positions(self) -> None:
        b = _make_behaviour()
        pos = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPool",
            "pool_type": "cl",
            "is_stable": False,
            "is_cl_pool": True,
            "positions": [],  # empty
        }
        result = b._build_exit_pool_action_base(pos)
        # token_ids and liquidities will be empty, so the if check is False
        assert "token_ids" not in result


class TestGetLastKnownPriceHistoricalBranch:
    """Test _get_last_known_price finding historical price (not current)."""

    def test_historical_only(self) -> None:
        """Test when cache has historical price but no current price."""
        b = _make_behaviour()
        from datetime import datetime

        with patch.object(b, "_get_current_timestamp", return_value=1700000000):
            date_str = datetime.utcfromtimestamp(1700000000).strftime("%d-%m-%Y")
            # Cache without "current" key, only historical
            cache_data = json.dumps({date_str: 42.5})

            def mock_read(*args, **kwargs):
                keys = args[0] if args else kwargs.get("keys", ())
                k = keys[0]
                yield
                return {k: cache_data}

            b._read_kv = mock_read
            result = _exhaust(b._get_last_known_price("0xToken"))
            assert result == 42.5

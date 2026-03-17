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

"""Tests for asset_lending custom component."""

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

import packages.valory.customs.asset_lending.asset_lending as al_module
from packages.valory.customs.asset_lending.asset_lending import (
    REQUIRED_FIELDS,
    apply_composite_pre_filter,
    calculate_daily_returns,
    calculate_il_risk_score_for_lending,
    calculate_il_risk_score_for_silos,
    calculate_metrics,
    calculate_sharpe_ratio,
    check_missing_fields,
    fetch_aggregators,
    fetch_historical_data,
    fetch_token_id,
    filter_aggregators,
    format_aggregator,
    get_best_opportunities,
    get_coin_list,
    get_errors,
    get_sharpe_ratio_for_address,
    is_pro_api_key,
    remove_irrelevant_fields,
    run,
    standardize_metrics,
    throttled_request,
    analyze_vault_liquidity,
)


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level caches and thread-local state before each test."""
    al_module._coin_list_cache = None
    al_module._aggregators_cache = None
    al_module._historical_data_cache = None
    al_module._last_request_time.clear()
    if hasattr(al_module._thread_local, "errors"):
        al_module._thread_local.errors = []
    yield
    al_module._coin_list_cache = None
    al_module._aggregators_cache = None
    al_module._historical_data_cache = None
    al_module._last_request_time.clear()
    if hasattr(al_module._thread_local, "errors"):
        al_module._thread_local.errors = []


class TestGetErrors:
    """Tests for get_errors function."""

    def test_initializes_empty_list(self):
        """Test that get_errors initializes empty list on first call."""
        if hasattr(al_module._thread_local, "errors"):
            delattr(al_module._thread_local, "errors")
        result = get_errors()
        assert result == []

    def test_returns_existing_list(self):
        """Test that get_errors returns existing errors list."""
        al_module._thread_local.errors = ["error1"]
        result = get_errors()
        assert result == ["error1"]


class TestThrottledRequest:
    """Tests for throttled_request function."""

    @patch("packages.valory.customs.asset_lending.asset_lending.requests.get")
    def test_first_request_no_throttle(self, mock_get):
        """Test that first request is not throttled."""
        mock_get.return_value = MagicMock()
        throttled_request("https://example.com")
        mock_get.assert_called_once_with("https://example.com", timeout=30)

    @patch("packages.valory.customs.asset_lending.asset_lending.requests.get")
    @patch("packages.valory.customs.asset_lending.asset_lending.time.sleep")
    def test_rapid_requests_are_throttled(self, mock_sleep, mock_get):
        """Test that rapid requests are throttled when elapsed < min_interval."""
        mock_get.return_value = MagicMock()
        # Set last request time to now so elapsed ~0 < min_interval
        al_module._last_request_time["https://example.com"] = time.time()
        throttled_request("https://example.com", min_interval=10.0)
        # Sleep should be called because elapsed (~0) < min_interval (10.0)
        mock_sleep.assert_called_once()

    @patch("packages.valory.customs.asset_lending.asset_lending.requests.get")
    def test_request_after_interval_elapsed(self, mock_get):
        """Test that no throttle occurs when enough time has elapsed (73->75 False branch)."""
        mock_get.return_value = MagicMock()
        # Set last request time far in the past so elapsed > min_interval
        al_module._last_request_time["https://example.com"] = time.time() - 100
        throttled_request("https://example.com", min_interval=0.01)
        mock_get.assert_called_once_with("https://example.com", timeout=30)


class TestGetCoinList:
    """Tests for get_coin_list function."""

    @patch("packages.valory.customs.asset_lending.asset_lending.throttled_request")
    def test_fetches_and_caches(self, mock_req):
        """Test that coin list is fetched and cached."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"id": "bitcoin", "symbol": "btc"}]
        mock_response.raise_for_status = MagicMock()
        mock_req.return_value = mock_response
        result = get_coin_list()
        assert len(result) == 1
        assert result[0]["id"] == "bitcoin"

    @patch("packages.valory.customs.asset_lending.asset_lending.throttled_request")
    def test_returns_cache_on_second_call(self, mock_req):
        """Test that second call returns cached data."""
        al_module._coin_list_cache = [{"id": "cached"}]
        result = get_coin_list()
        assert result[0]["id"] == "cached"
        mock_req.assert_not_called()

    @patch("packages.valory.customs.asset_lending.asset_lending.throttled_request")
    def test_error_returns_empty_list(self, mock_req):
        """Test that request error returns empty list."""
        import requests as req_lib

        mock_req.side_effect = req_lib.RequestException("fail")
        result = get_coin_list()
        assert result == []

    @patch("packages.valory.customs.asset_lending.asset_lending.throttled_request")
    def test_json_decode_error_returns_empty_list(self, mock_req):
        """Test that ValueError from response.json() returns empty list."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = ValueError("No JSON object")
        mock_req.return_value = mock_response
        result = get_coin_list()
        assert result == []


class TestFetchTokenId:
    """Tests for fetch_token_id function."""

    def test_known_mapping(self):
        """Test that known mappings are returned directly."""
        assert fetch_token_id("weth") == "weth"
        assert fetch_token_id("WETH") == "weth"

    @patch("packages.valory.customs.asset_lending.asset_lending.get_coin_list")
    def test_found_in_coin_list(self, mock_list):
        """Test that symbol is found in coin list."""
        mock_list.return_value = [{"id": "bitcoin", "symbol": "btc"}]
        assert fetch_token_id("btc") == "bitcoin"

    @patch("packages.valory.customs.asset_lending.asset_lending.get_coin_list")
    def test_not_found(self, mock_list):
        """Test that None is returned when symbol is not found."""
        mock_list.return_value = [{"id": "bitcoin", "symbol": "btc"}]
        result = fetch_token_id("unknown_coin")
        assert result is None


class TestFetchHistoricalData:
    """Tests for fetch_historical_data function."""

    @patch("packages.valory.customs.asset_lending.asset_lending.throttled_request")
    def test_fetches_data(self, mock_req):
        """Test that historical data is fetched."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"timestamp": 1}]
        mock_req.return_value = mock_response
        result = fetch_historical_data()
        assert result == [{"timestamp": 1}]

    @patch("packages.valory.customs.asset_lending.asset_lending.throttled_request")
    def test_returns_cache(self, mock_req):
        """Test that cached data is returned."""
        al_module._historical_data_cache = [{"cached": True}]
        result = fetch_historical_data()
        assert result == [{"cached": True}]
        mock_req.assert_not_called()

    @patch("packages.valory.customs.asset_lending.asset_lending.throttled_request")
    def test_error_returns_none(self, mock_req):
        """Test that error returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_req.return_value = mock_response
        result = fetch_historical_data()
        assert result is None


class TestCalculateDailyReturns:
    """Tests for calculate_daily_returns function."""

    def test_basic_calculation(self):
        """Test basic daily returns calculation."""
        result = calculate_daily_returns(0.10)
        assert result > 0

    def test_with_reward_apy(self):
        """Test with reward APY."""
        result = calculate_daily_returns(0.10, reward_apy=0.05)
        assert result > calculate_daily_returns(0.10)


class TestCalculateSharpeRatio:
    """Tests for calculate_sharpe_ratio function."""

    def test_with_valid_data(self):
        """Test Sharpe ratio with valid returns data."""
        returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01])
        result = calculate_sharpe_ratio(returns)
        assert not np.isnan(result)

    def test_insufficient_data(self):
        """Test that insufficient data returns NaN."""
        returns = pd.Series([0.01])
        result = calculate_sharpe_ratio(returns)
        assert np.isnan(result)


class TestGetSharpeRatioForAddress:
    """Tests for get_sharpe_ratio_for_address function."""

    def test_no_records(self):
        """Test with no matching records."""
        historical = [{"timestamp": 1000, "doc": {"key_0x123": {"baseAPY": 0.1}}}]
        result = get_sharpe_ratio_for_address(historical, "0xnotfound")
        assert np.isnan(result)

    def test_with_matching_records(self):
        """Test with matching records."""
        ts = 1704067200000  # 2024-01-01 in ms
        historical = []
        for i in range(10):
            historical.append(
                {
                    "timestamp": ts + i * 86400000,
                    "doc": {
                        f"data_0xaddr": {"baseAPY": 0.1 + i * 0.01, "rewardsAPY": 0.02},
                        "data_0xaddr": {"baseAPY": 0.1 + i * 0.01, "rewardsAPY": 0.02},
                    },
                }
            )
        result = get_sharpe_ratio_for_address(historical, "0xaddr")
        assert isinstance(result, float)

    def test_entry_split_less_than_2(self):
        """Test that entries with less than 2 parts after split are skipped."""
        historical = [{"timestamp": 1000, "doc": {"badkey": {"baseAPY": 0.1}}}]
        result = get_sharpe_ratio_for_address(historical, "0xaddr")
        assert np.isnan(result)


class TestCheckMissingFields:
    """Tests for check_missing_fields function."""

    def test_no_missing(self):
        """Test with all fields present."""
        kwargs = {f: "v" for f in REQUIRED_FIELDS}
        assert check_missing_fields(kwargs) == []

    def test_all_missing(self):
        """Test with all fields missing."""
        assert len(check_missing_fields({})) == len(REQUIRED_FIELDS)


class TestRemoveIrrelevantFields:
    """Tests for remove_irrelevant_fields function."""

    def test_removes_extras(self):
        """Test that extra fields are removed."""
        result = remove_irrelevant_fields({"a": 1, "b": 2}, ("a",))
        assert result == {"a": 1}


class TestFetchAggregators:
    """Tests for fetch_aggregators function."""

    @patch("packages.valory.customs.asset_lending.asset_lending.throttled_request")
    def test_fetches_successfully(self, mock_req):
        """Test successful fetch."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"address": "0x1"}]
        mock_resp.raise_for_status = MagicMock()
        mock_req.return_value = mock_resp
        result = fetch_aggregators()
        assert len(result) == 1

    @patch("packages.valory.customs.asset_lending.asset_lending.throttled_request")
    def test_returns_cache(self, mock_req):
        """Test that cached data is returned."""
        al_module._aggregators_cache = [{"cached": True}]
        result = fetch_aggregators()
        assert result[0]["cached"]
        mock_req.assert_not_called()

    @patch("packages.valory.customs.asset_lending.asset_lending.throttled_request")
    def test_api_errors_in_result(self, mock_req):
        """Test when API returns errors field."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"errors": ["bad"]}
        mock_resp.raise_for_status = MagicMock()
        mock_req.return_value = mock_resp
        result = fetch_aggregators()
        assert result == []

    @patch("packages.valory.customs.asset_lending.asset_lending.throttled_request")
    def test_request_exception(self, mock_req):
        """Test request exception handling."""
        import requests as req_lib

        mock_req.side_effect = req_lib.RequestException("fail")
        result = fetch_aggregators()
        assert result == []


class TestStandardizeMetrics:
    """Tests for standardize_metrics function."""

    def test_empty_pools(self):
        """Test with empty pools returns early."""
        assert standardize_metrics([]) == []

    def test_single_pool(self):
        """Test with single pool (std dev = 0)."""
        pools = [{"total_apr": 10, "tvl": 1000}]
        result = standardize_metrics(pools)
        assert "composite_score" in result[0]

    def test_multiple_pools(self):
        """Test with multiple pools."""
        pools = [
            {"total_apr": 10, "tvl": 1000},
            {"total_apr": 20, "tvl": 2000},
        ]
        result = standardize_metrics(pools)
        assert result[1]["composite_score"] > result[0]["composite_score"]


class TestApplyCompositePreFilter:
    """Tests for apply_composite_pre_filter function."""

    def test_empty_pools(self):
        """Test with empty pools."""
        assert apply_composite_pre_filter([]) == []

    def test_disabled_filter(self):
        """Test with composite filter disabled."""
        pools = [{"tvl": 5000, "total_apr": 10}]
        result = apply_composite_pre_filter(pools, use_composite_filter=False)
        assert len(result) == 1

    def test_disabled_filter_empty_pools(self):
        """Test with disabled filter and empty pools."""
        result = apply_composite_pre_filter([], use_composite_filter=False)
        assert result == []

    def test_below_tvl_threshold(self):
        """Test that pools below TVL threshold are excluded."""
        pools = [{"tvl": 100, "total_apr": 10, "address": "0x1"}]
        result = apply_composite_pre_filter(pools, min_tvl_threshold=1000)
        assert result == []

    def test_invalid_tvl_values(self):
        """Test with invalid TVL values."""
        pools = [{"tvl": "invalid", "total_apr": 10, "address": "0x1"}]
        result = apply_composite_pre_filter(pools, min_tvl_threshold=0)
        assert result == []

    def test_top_n_limit(self):
        """Test top_n limit."""
        pools = [{"tvl": 5000, "total_apr": 10 + i} for i in range(5)]
        result = apply_composite_pre_filter(pools, top_n=2, min_tvl_threshold=0)
        assert len(result) == 2


class TestFilterAggregators:
    """Tests for filter_aggregators function."""

    @patch(
        "packages.valory.customs.asset_lending.asset_lending.apply_composite_pre_filter"
    )
    def test_filters_by_chain_and_asset(self, mock_filter):
        """Test filtering by chain and asset."""
        mock_filter.side_effect = lambda x, **kw: x
        aggs = [
            {
                "chainName": "optimism",
                "address": "0x1234567890abcdef1234567890abcdef12345678",
                "asset": {"address": "0xasset"},
                "apy": {"total": 0.1},
                "tvl": 5000,
            }
        ]
        result = filter_aggregators(["optimism"], aggs, "0xasset", [], top_n=10)
        assert len(result) == 1

    @patch(
        "packages.valory.customs.asset_lending.asset_lending.apply_composite_pre_filter"
    )
    def test_excludes_current_positions(self, mock_filter):
        """Test that current positions are excluded."""
        mock_filter.side_effect = lambda x, **kw: x
        addr = "0x1234567890abcDEF1234567890abcDEF12345678"
        from web3 import Web3

        checksum = Web3.to_checksum_address(addr)
        aggs = [
            {
                "chainName": "optimism",
                "address": addr,
                "asset": {"address": "0xasset"},
                "apy": {"total": 0.1},
                "tvl": 5000,
            }
        ]
        result = filter_aggregators(["optimism"], aggs, "0xasset", [checksum])
        assert len(result) == 0

    def test_empty_result(self):
        """Test with no matching aggregators."""
        result = filter_aggregators(["optimism"], [], "0xasset", [])
        assert result == []


class TestCalculateIlRiskScoreForLending:
    """Tests for calculate_il_risk_score_for_lending function."""

    def test_missing_tokens(self):
        """Test with missing tokens."""
        result = calculate_il_risk_score_for_lending("", "token2", "key")
        assert result is None

    def test_missing_token2(self):
        """Test with missing second token."""
        result = calculate_il_risk_score_for_lending("token1", "", "key")
        assert result is None

    @patch("packages.valory.customs.asset_lending.asset_lending.is_pro_api_key")
    @patch("packages.valory.customs.asset_lending.asset_lending.CoinGeckoAPI")
    def test_successful_calculation(self, mock_cg_class, mock_is_pro):
        """Test successful IL risk score calculation."""
        mock_is_pro.return_value = False
        mock_cg = MagicMock()
        mock_cg.get_coin_market_chart_range_by_id.side_effect = [
            {"prices": [[i, 100 + i * 0.5] for i in range(100)]},
            {"prices": [[i, 200 + i * 0.3] for i in range(100)]},
        ]
        mock_cg_class.return_value = mock_cg
        result = calculate_il_risk_score_for_lending("token1", "token2", "key")
        assert isinstance(result, float)

    @patch("packages.valory.customs.asset_lending.asset_lending.is_pro_api_key")
    @patch("packages.valory.customs.asset_lending.asset_lending.CoinGeckoAPI")
    def test_api_exception(self, mock_cg_class, mock_is_pro):
        """Test API exception handling."""
        mock_is_pro.return_value = False
        mock_cg = MagicMock()
        mock_cg.get_coin_market_chart_range_by_id.side_effect = Exception("API error")
        mock_cg_class.return_value = mock_cg
        result = calculate_il_risk_score_for_lending("token1", "token2", "key")
        assert result is None

    @patch("packages.valory.customs.asset_lending.asset_lending.is_pro_api_key")
    @patch("packages.valory.customs.asset_lending.asset_lending.CoinGeckoAPI")
    def test_empty_prices(self, mock_cg_class, mock_is_pro):
        """Test with empty price data."""
        mock_is_pro.return_value = False
        mock_cg = MagicMock()
        mock_cg.get_coin_market_chart_range_by_id.side_effect = [
            {"prices": []},
            {"prices": [[1, 100]]},
        ]
        mock_cg_class.return_value = mock_cg
        result = calculate_il_risk_score_for_lending("token1", "token2", "key")
        assert result is None

    @patch("packages.valory.customs.asset_lending.asset_lending.is_pro_api_key")
    @patch("packages.valory.customs.asset_lending.asset_lending.CoinGeckoAPI")
    def test_pro_api_key(self, mock_cg_class, mock_is_pro):
        """Test with pro API key."""
        mock_is_pro.return_value = True
        mock_cg = MagicMock()
        mock_cg.get_coin_market_chart_range_by_id.side_effect = [
            {"prices": [[i, 100 + i] for i in range(10)]},
            {"prices": [[i, 200 + i] for i in range(10)]},
        ]
        mock_cg_class.return_value = mock_cg
        result = calculate_il_risk_score_for_lending("t1", "t2", "pro_key")
        assert isinstance(result, float)


class TestCalculateIlRiskScoreForSilos:
    """Tests for calculate_il_risk_score_for_silos function."""

    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_token_id")
    def test_no_token0_id(self, mock_fetch):
        """Test when token0 id is not found."""
        mock_fetch.return_value = None
        result = calculate_il_risk_score_for_silos("unknown", [], "key")
        assert result is None

    @patch(
        "packages.valory.customs.asset_lending.asset_lending.calculate_il_risk_score_for_lending"
    )
    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_token_id")
    def test_successful_calculation(self, mock_fetch, mock_il):
        """Test successful silo IL risk calculation."""
        mock_fetch.side_effect = lambda s: f"id_{s.lower()}"
        mock_il.return_value = -0.05
        silos = [{"collateral": "TOKEN1"}, {"collateral": "TOKEN2"}]
        result = calculate_il_risk_score_for_silos("WETH", silos, "key")
        assert isinstance(result, float)

    @patch(
        "packages.valory.customs.asset_lending.asset_lending.calculate_il_risk_score_for_lending"
    )
    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_token_id")
    def test_silo_token_not_found(self, mock_fetch, mock_il):
        """Test when silo collateral token is not found."""

        def side_effect(s):
            return "id_weth" if s.lower() == "weth" else None

        mock_fetch.side_effect = side_effect
        silos = [{"collateral": "UNKNOWN"}]
        result = calculate_il_risk_score_for_silos("WETH", silos, "key")
        assert result is None

    @patch(
        "packages.valory.customs.asset_lending.asset_lending.calculate_il_risk_score_for_lending"
    )
    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_token_id")
    def test_il_risk_score_returns_none(self, mock_fetch, mock_il):
        """Test when IL risk score calculation returns None."""
        mock_fetch.side_effect = lambda s: f"id_{s.lower()}"
        mock_il.return_value = None
        silos = [{"collateral": "TOKEN1"}]
        result = calculate_il_risk_score_for_silos("WETH", silos, "key")
        assert result is None

    @patch(
        "packages.valory.customs.asset_lending.asset_lending.calculate_il_risk_score_for_lending"
    )
    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_token_id")
    def test_il_risk_score_returns_zero(self, mock_fetch, mock_il):
        """Test when IL risk score returns 0 (falsy but valid)."""
        mock_fetch.side_effect = lambda s: f"id_{s.lower()}"
        mock_il.return_value = 0  # Falsy
        silos = [{"collateral": "TOKEN1"}]
        result = calculate_il_risk_score_for_silos("WETH", silos, "key")
        # Returns None because `not il_risk_score` is True when il_risk_score == 0
        assert result is None


class TestAnalyzeVaultLiquidity:
    """Tests for analyze_vault_liquidity function."""

    def test_with_valid_data(self):
        """Test with valid TVL and total assets."""
        agg = {"tvl": 1000000, "totalAssets": 500000, "address": "0x1"}
        depth, max_pos = analyze_vault_liquidity(agg)
        assert depth > 0
        assert max_pos > 0

    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_aggregators")
    def test_missing_data_fetches_from_aggregators(self, mock_fetch):
        """Test that missing TVL/totalAssets triggers re-fetch."""
        mock_fetch.return_value = [
            {"address": "0x1", "tvl": 1000000, "totalAssets": 500000}
        ]
        agg = {"tvl": 0, "totalAssets": 0, "address": "0x1"}
        depth, max_pos = analyze_vault_liquidity(agg)
        assert depth > 0

    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_aggregators")
    def test_missing_data_not_found_in_aggregators(self, mock_fetch):
        """Test when data is not found in aggregators either."""
        mock_fetch.return_value = []
        agg = {"tvl": 0, "totalAssets": 0, "address": "0x1"}
        depth, max_pos = analyze_vault_liquidity(agg)
        assert np.isnan(depth)
        assert np.isnan(max_pos)

    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_aggregators")
    def test_pool_address_match(self, mock_fetch):
        """Test matching by pool_address field."""
        mock_fetch.return_value = [
            {"address": "0x2", "tvl": 1000000, "totalAssets": 500000}
        ]
        agg = {"tvl": 0, "totalAssets": 0, "address": "0x1", "pool_address": "0x2"}
        depth, max_pos = analyze_vault_liquidity(agg)
        assert depth > 0


class TestFormatAggregator:
    """Tests for format_aggregator function."""

    def test_formats_correctly(self):
        """Test correct formatting."""
        agg = {
            "chainName": "optimism",
            "address": "0x1",
            "asset": {"symbol": "WETH", "address": "0xtoken"},
            "total_apr": 0.1,
            "whitelistedSilos": [],
            "il_risk_score": -0.05,
            "sharpe_ratio": 1.5,
            "depth_score": 100,
            "max_position_size": 5000,
            "type": "lending",
        }
        result = format_aggregator(agg)
        assert result["chain"] == "optimism"
        assert result["apr"] == 10.0  # 0.1 * 100
        assert result["dex_type"] == "Sturdy"


class TestIsProApiKey:
    """Tests for is_pro_api_key function."""

    @patch("packages.valory.customs.asset_lending.asset_lending.CoinGeckoAPI")
    def test_pro_key(self, mock_cg_class):
        """Test that a pro key returns True."""
        mock_cg = MagicMock()
        mock_cg.get_coin_market_chart_range_by_id.return_value = {"prices": []}
        mock_cg_class.return_value = mock_cg
        assert is_pro_api_key("pro_key") is True

    @patch("packages.valory.customs.asset_lending.asset_lending.CoinGeckoAPI")
    def test_non_pro_key(self, mock_cg_class):
        """Test that a non-pro key returns False."""
        mock_cg = MagicMock()
        mock_cg.get_coin_market_chart_range_by_id.side_effect = Exception(
            "unauthorized"
        )
        mock_cg_class.return_value = mock_cg
        assert is_pro_api_key("bad_key") is False

    @patch("packages.valory.customs.asset_lending.asset_lending.CoinGeckoAPI")
    def test_empty_response(self, mock_cg_class):
        """Test that an empty/falsy response returns False."""
        mock_cg = MagicMock()
        mock_cg.get_coin_market_chart_range_by_id.return_value = {}
        mock_cg_class.return_value = mock_cg
        assert is_pro_api_key("key") is False


class TestGetBestOpportunities:
    """Tests for get_best_opportunities function."""

    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_aggregators")
    def test_no_aggregators(self, mock_fetch):
        """Test when no aggregators are available."""
        mock_fetch.return_value = []
        result = get_best_opportunities(["optimism"], "0xasset", [], "key")
        assert "error" in result

    @patch("packages.valory.customs.asset_lending.asset_lending.filter_aggregators")
    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_aggregators")
    def test_no_filtered_aggregators(self, mock_fetch, mock_filter):
        """Test when no aggregators pass filtering."""
        mock_fetch.return_value = [{"address": "0x1"}]
        mock_filter.return_value = []
        result = get_best_opportunities(["optimism"], "0xasset", [], "key")
        assert "error" in result

    @patch(
        "packages.valory.customs.asset_lending.asset_lending.analyze_vault_liquidity"
    )
    @patch(
        "packages.valory.customs.asset_lending.asset_lending.get_sharpe_ratio_for_address"
    )
    @patch(
        "packages.valory.customs.asset_lending.asset_lending.calculate_il_risk_score_for_silos"
    )
    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_historical_data")
    @patch("packages.valory.customs.asset_lending.asset_lending.filter_aggregators")
    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_aggregators")
    def test_no_historical_data(
        self, mock_fetch, mock_filter, mock_hist, mock_il, mock_sharpe, mock_vault
    ):
        """Test when historical data is unavailable."""
        mock_fetch.return_value = [{"address": "0x1"}]
        mock_filter.return_value = [
            {"address": "0x1", "asset": {"symbol": "WETH"}, "whitelistedSilos": []}
        ]
        mock_hist.return_value = None
        result = get_best_opportunities(["optimism"], "0xasset", [], "key")
        assert "error" in result

    @patch("packages.valory.customs.asset_lending.asset_lending.format_aggregator")
    @patch(
        "packages.valory.customs.asset_lending.asset_lending.analyze_vault_liquidity"
    )
    @patch(
        "packages.valory.customs.asset_lending.asset_lending.get_sharpe_ratio_for_address"
    )
    @patch(
        "packages.valory.customs.asset_lending.asset_lending.calculate_il_risk_score_for_silos"
    )
    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_historical_data")
    @patch("packages.valory.customs.asset_lending.asset_lending.filter_aggregators")
    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_aggregators")
    def test_successful_flow(
        self,
        mock_fetch,
        mock_filter,
        mock_hist,
        mock_il,
        mock_sharpe,
        mock_vault,
        mock_format,
    ):
        """Test successful end-to-end flow."""
        mock_fetch.return_value = [{"address": "0x1"}]
        mock_filter.return_value = [
            {"address": "0x1", "asset": {"symbol": "WETH"}, "whitelistedSilos": []}
        ]
        mock_hist.return_value = [{"timestamp": 1, "doc": {}}]
        mock_il.return_value = -0.05
        mock_sharpe.return_value = 1.5
        mock_vault.return_value = (100, 5000)
        mock_format.return_value = {"pool_address": "0x1"}
        result = get_best_opportunities(["optimism"], "0xasset", [], "key")
        assert isinstance(result, list)
        assert len(result) == 1


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    @patch(
        "packages.valory.customs.asset_lending.asset_lending.analyze_vault_liquidity"
    )
    @patch(
        "packages.valory.customs.asset_lending.asset_lending.get_sharpe_ratio_for_address"
    )
    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_historical_data")
    @patch(
        "packages.valory.customs.asset_lending.asset_lending.calculate_il_risk_score_for_silos"
    )
    def test_successful(self, mock_il, mock_hist, mock_sharpe, mock_vault):
        """Test successful metrics calculation."""
        mock_il.return_value = -0.05
        mock_hist.return_value = [{"timestamp": 1}]
        mock_sharpe.return_value = 1.5
        mock_vault.return_value = (100, 5000)
        position = {
            "token0_symbol": "WETH",
            "whitelistedSilos": [],
            "pool_address": "0x1",
        }
        result = calculate_metrics(position, "key")
        assert result["il_risk_score"] == -0.05

    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_historical_data")
    @patch(
        "packages.valory.customs.asset_lending.asset_lending.calculate_il_risk_score_for_silos"
    )
    def test_no_historical_data(self, mock_il, mock_hist):
        """Test when historical data is unavailable."""
        mock_il.return_value = -0.05
        mock_hist.return_value = None
        position = {
            "token0_symbol": "WETH",
            "whitelistedSilos": [],
            "pool_address": "0x1",
        }
        result = calculate_metrics(position, "key")
        assert "error" in result


class TestRun:
    """Tests for the run function."""

    def test_missing_fields(self):
        """Test with missing required fields."""
        result = run()
        assert "error" in result

    @patch("packages.valory.customs.asset_lending.asset_lending.calculate_metrics")
    def test_get_metrics_mode(self, mock_calc):
        """Test get_metrics mode."""
        mock_calc.return_value = {"il_risk_score": -0.05}
        result = run(
            chains=["optimism"],
            lending_asset="0xasset",
            current_positions=[],
            coingecko_api_key="key",
            get_metrics=True,
            position={
                "pool_address": "0x1",
                "token0_symbol": "WETH",
                "whitelistedSilos": [],
            },
        )
        assert result == {"il_risk_score": -0.05}

    @patch("packages.valory.customs.asset_lending.asset_lending.calculate_metrics")
    def test_get_metrics_returns_none(self, mock_calc):
        """Test get_metrics mode when metrics is None."""
        mock_calc.return_value = None
        result = run(
            chains=["optimism"],
            lending_asset="0xasset",
            current_positions=[],
            coingecko_api_key="key",
            get_metrics=True,
            position={"pool_address": "0x1"},
        )
        assert "error" in result

    @patch("packages.valory.customs.asset_lending.asset_lending.get_best_opportunities")
    def test_opportunity_search_mode(self, mock_opp):
        """Test opportunity search mode."""
        mock_opp.return_value = [{"pool": "data"}]
        result = run(
            chains=["optimism"],
            lending_asset="0xasset",
            current_positions=[],
            coingecko_api_key="key",
        )
        assert "result" in result
        assert "error" in result

    @patch("packages.valory.customs.asset_lending.asset_lending.get_best_opportunities")
    def test_opportunity_search_error(self, mock_opp):
        """Test opportunity search returning error."""
        mock_opp.return_value = {"error": ["some error"]}
        result = run(
            chains=["optimism"],
            lending_asset="0xasset",
            current_positions=[],
            coingecko_api_key="key",
        )
        assert "result" in result

    @patch("packages.valory.customs.asset_lending.asset_lending.get_best_opportunities")
    def test_opportunity_search_empty(self, mock_opp):
        """Test opportunity search returning empty."""
        mock_opp.return_value = []
        result = run(
            chains=["optimism"],
            lending_asset="0xasset",
            current_positions=[],
            coingecko_api_key="key",
        )
        assert "result" in result

    @patch("packages.valory.customs.asset_lending.asset_lending.get_best_opportunities")
    def test_get_metrics_with_errors(self, mock_opp):
        """Test that errors are included in the response."""
        mock_opp.return_value = [{"pool": "data"}]
        al_module._thread_local.errors = []
        result = run(
            chains=["optimism"],
            lending_asset="0xasset",
            current_positions=[],
            coingecko_api_key="key",
        )
        assert isinstance(result["error"], list)


class TestBranchCoverage:
    """Additional tests for full branch coverage."""

    @patch(
        "packages.valory.customs.asset_lending.asset_lending.apply_composite_pre_filter"
    )
    def test_filter_aggregators_asset_mismatch(self, mock_filter):
        """Test filter_aggregators where chain matches but asset address does not (376->370)."""
        mock_filter.side_effect = lambda x, **kw: x
        aggs = [
            {
                "chainName": "optimism",
                "address": "0x1234567890abcdef1234567890abcdef12345678",
                "asset": {"address": "0xWRONG_ASSET"},
                "apy": {"total": 0.1},
                "tvl": 5000,
            },
            {
                "chainName": "optimism",
                "address": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                "asset": {"address": "0xRIGHT_ASSET"},
                "apy": {"total": 0.2},
                "tvl": 8000,
            },
        ]
        result = filter_aggregators(["optimism"], aggs, "0xRIGHT_ASSET", [])
        # Only the second aggregator should pass
        assert len(result) == 1

    @patch(
        "packages.valory.customs.asset_lending.asset_lending.calculate_il_risk_score_for_lending"
    )
    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_token_id")
    def test_silos_token_id_cache_hit(self, mock_fetch, mock_il):
        """Test calculate_il_risk_score_for_silos where token_id_cache is hit (line 483)."""
        mock_fetch.side_effect = lambda s: f"id_{s.lower()}"
        mock_il.return_value = -0.05
        # Two silos with the SAME collateral token to trigger cache hit on second lookup
        silos = [{"collateral": "TOKEN1"}, {"collateral": "TOKEN1"}]
        result = calculate_il_risk_score_for_silos("WETH", silos, "key")
        assert isinstance(result, float)
        # fetch_token_id should only be called once for TOKEN1 (second uses cache)
        # token0 (WETH) is in coingecko_name_to_id so fetch_token_id is only called for TOKEN1
        assert mock_fetch.call_count == 1

    @patch("packages.valory.customs.asset_lending.asset_lending.fetch_aggregators")
    def test_analyze_vault_liquidity_non_matching_items(self, mock_fetch):
        """Test analyze_vault_liquidity where aggregators list has non-matching items (532->531)."""
        mock_fetch.return_value = [
            {"address": "0xNOTMATCH1", "tvl": 999, "totalAssets": 999},
            {"address": "0xNOTMATCH2", "tvl": 888, "totalAssets": 888},
            {"address": "0xTARGET", "tvl": 1000000, "totalAssets": 500000},
        ]
        agg = {"tvl": 0, "totalAssets": 0, "address": "0xTARGET"}
        depth, max_pos = analyze_vault_liquidity(agg)
        assert depth > 0
        assert max_pos > 0


class TestNetworkResilience:
    """Tests for network resilience in asset_lending."""

    @patch("packages.valory.customs.asset_lending.asset_lending.requests.get")
    def test_throttled_request_passes_timeout(self, mock_get):
        """Test that throttled_request passes timeout=30 to requests.get."""
        mock_get.return_value = MagicMock()
        throttled_request("https://example.com")
        _, kwargs = mock_get.call_args
        assert kwargs.get("timeout") == 30

    @patch("packages.valory.customs.asset_lending.asset_lending.throttled_request")
    def test_fetch_historical_data_connection_error(self, mock_req):
        """Test that ConnectionError in fetch_historical_data returns None."""
        import requests as req_lib

        mock_req.side_effect = req_lib.ConnectionError("connection refused")
        result = fetch_historical_data()
        assert result is None

    @patch("packages.valory.customs.asset_lending.asset_lending.throttled_request")
    def test_fetch_historical_data_json_decode_error(self, mock_req):
        """Test that JSONDecodeError in fetch_historical_data returns None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("No JSON")
        mock_req.return_value = mock_resp
        result = fetch_historical_data()
        assert result is None

    @patch("packages.valory.customs.asset_lending.asset_lending.throttled_request")
    def test_fetch_aggregators_json_decode_error(self, mock_req):
        """Test that ValueError from response.json() in fetch_aggregators is caught."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("No JSON object could be decoded")
        mock_req.return_value = mock_resp
        result = fetch_aggregators()
        assert result == []
        errors = get_errors()
        assert any(
            "failed" in str(e).lower() or "error" in str(e).lower() for e in errors
        )

    @patch("packages.valory.customs.asset_lending.asset_lending.is_pro_api_key")
    @patch("packages.valory.customs.asset_lending.asset_lending.CoinGeckoAPI")
    def test_il_risk_score_missing_prices_key(self, mock_cg_class, mock_is_pro):
        """Test that missing 'prices' key in CoinGecko response returns None."""
        mock_is_pro.return_value = False
        mock_cg = MagicMock()
        # Response missing "prices" key
        mock_cg.get_coin_market_chart_range_by_id.side_effect = [
            {"market_caps": []},
            {"prices": [[i, 200 + i] for i in range(10)]},
        ]
        mock_cg_class.return_value = mock_cg
        result = calculate_il_risk_score_for_lending("token1", "token2", "key")
        assert result is None

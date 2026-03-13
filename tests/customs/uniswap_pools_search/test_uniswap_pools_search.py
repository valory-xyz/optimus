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

"""Tests for uniswap_pools_search custom component."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import packages.valory.customs.uniswap_pools_search.uniswap_pools_search as uni_mod
from packages.valory.customs.uniswap_pools_search.uniswap_pools_search import (
    REQUIRED_FIELDS,
    apply_composite_pre_filter,
    assess_pool_liquidity,
    calculate_apr,
    calculate_il_impact,
    calculate_il_risk_score,
    calculate_metrics,
    calculate_metrics_liquidity_risk,
    check_missing_fields,
    fetch_graphql_data,
    fetch_pool_data,
    format_pool_data,
    get_coin_id_from_symbol,
    get_errors,
    get_filtered_pools_for_uniswap,
    get_opportunities_for_uniswap,
    get_uniswap_pool_sharpe_ratio,
    is_pro_api_key,
    remove_irrelevant_fields,
    run,
    run_query,
    standardize_metrics,
    _reset_x402_adapter,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset module state before each test."""
    if hasattr(uni_mod._thread_local, "errors"):
        uni_mod._thread_local.errors = []
    yield
    if hasattr(uni_mod._thread_local, "errors"):
        uni_mod._thread_local.errors = []


class TestGetErrors:
    """Tests for get_errors function."""

    def test_initializes_empty(self):
        """Test initialization of errors list."""
        if hasattr(uni_mod._thread_local, "errors"):
            delattr(uni_mod._thread_local, "errors")
        assert get_errors() == []

    def test_returns_existing(self):
        """Test returning existing errors."""
        uni_mod._thread_local.errors = ["err"]
        assert get_errors() == ["err"]


class TestCheckMissingFields:
    """Tests for check_missing_fields function."""

    def test_no_missing(self):
        """Test with all fields present."""
        kwargs = {f: "v" for f in REQUIRED_FIELDS}
        assert check_missing_fields(kwargs) == []

    def test_all_missing(self):
        """Test with all fields missing."""
        assert len(check_missing_fields({})) == len(REQUIRED_FIELDS)

    def test_none_value(self):
        """Test None value counts as missing."""
        kwargs = {f: "v" for f in REQUIRED_FIELDS}
        kwargs["chains"] = None
        assert "chains" in check_missing_fields(kwargs)


class TestRemoveIrrelevantFields:
    """Tests for remove_irrelevant_fields function."""

    def test_removes_extras(self):
        """Test removing extra fields."""
        result = remove_irrelevant_fields({"a": 1, "b": 2}, ("a",))
        assert result == {"a": 1}


class TestGetCoinIdFromSymbol:
    """Tests for get_coin_id_from_symbol function."""

    def test_found(self):
        """Test finding coin ID."""
        mapping = {"optimism": {"weth": "weth-id"}}
        assert get_coin_id_from_symbol(mapping, "WETH", "optimism") == "weth-id"

    def test_not_found(self):
        """Test when coin ID is not found."""
        mapping = {"optimism": {"weth": "weth-id"}}
        assert get_coin_id_from_symbol(mapping, "unknown", "optimism") is None

    def test_chain_not_in_mapping(self):
        """Test when chain is not in mapping."""
        mapping = {}
        assert get_coin_id_from_symbol(mapping, "weth", "optimism") is None


class TestRunQuery:
    """Tests for run_query function."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_successful_query(self, mock_post):
        """Test successful GraphQL query."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"pools": []}}
        mock_post.return_value = mock_resp
        result = run_query("{pools{id}}", "https://api.example.com")
        assert result == {"pools": []}

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_error_status(self, mock_post):
        """Test error status code."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp
        result = run_query("{pools{id}}", "https://api.example.com")
        assert "error" in result

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_graphql_errors(self, mock_post):
        """Test GraphQL errors in response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"errors": [{"message": "bad"}]}
        mock_post.return_value = mock_resp
        result = run_query("{pools{id}}", "https://api.example.com")
        assert "error" in result

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_with_variables(self, mock_post):
        """Test query with variables."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"result": "ok"}}
        mock_post.return_value = mock_resp
        result = run_query("query", "url", variables={"key": "val"})
        assert result == {"result": "ok"}


class TestCalculateApr:
    """Tests for calculate_apr function."""

    def test_normal_calculation(self):
        """Test normal APR calculation."""
        result = calculate_apr(1000, 100000, 0.003)
        assert result > 0

    def test_zero_tvl(self):
        """Test zero TVL returns 0."""
        assert calculate_apr(1000, 0, 0.003) == 0


class TestStandardizeMetrics:
    """Tests for standardize_metrics function."""

    def test_empty_pools(self):
        """Test with empty pools returns early."""
        assert standardize_metrics([]) == []

    def test_single_pool(self):
        """Test single pool (zero std dev)."""
        pools = [{"apr": 10, "tvl": 1000}]
        result = standardize_metrics(pools)
        assert "composite_score" in result[0]

    def test_multiple_pools(self):
        """Test multiple pools."""
        pools = [{"apr": 10, "tvl": 1000}, {"apr": 20, "tvl": 2000}]
        result = standardize_metrics(pools)
        assert result[1]["composite_score"] > result[0]["composite_score"]

    def test_invalid_values(self):
        """Test with invalid apr/tvl values."""
        pools = [{"apr": "invalid", "tvl": "bad"}]
        result = standardize_metrics(pools)
        assert "composite_score" in result[0]


class TestApplyCompositePreFilter:
    """Tests for apply_composite_pre_filter function."""

    def test_empty_pools(self):
        """Test with empty pools."""
        assert apply_composite_pre_filter([]) == []

    def test_disabled_filter(self):
        """Test with filter disabled."""
        pools = [{"tvl": 5000, "apr": 10}]
        result = apply_composite_pre_filter(pools, use_composite_filter=False)
        assert len(result) == 1

    def test_disabled_filter_empty(self):
        """Test disabled filter with empty pools."""
        assert apply_composite_pre_filter([], use_composite_filter=False) == []

    def test_below_tvl_threshold(self):
        """Test pools below TVL threshold."""
        pools = [{"tvl": 100, "apr": 10}]
        result = apply_composite_pre_filter(pools, min_tvl_threshold=1000)
        assert result == []

    def test_invalid_tvl(self):
        """Test with invalid TVL."""
        pools = [{"tvl": "invalid", "apr": 10}]
        result = apply_composite_pre_filter(pools, min_tvl_threshold=0)
        assert result == []

    def test_top_n(self):
        """Test top_n limit."""
        pools = [{"tvl": 5000, "apr": 10 + i} for i in range(5)]
        result = apply_composite_pre_filter(pools, top_n=2, min_tvl_threshold=0)
        assert len(result) == 2


class TestGetFilteredPoolsForUniswap:
    """Tests for get_filtered_pools_for_uniswap function."""

    def _make_pool(
        self,
        pool_id="0x1234567890abcdef1234567890abcdef12345678",
        chain="optimism",
        fee_tier="3000",
        tvl="100000",
        volume="50000",
        token0_id="0xtoken0",
        token1_id="0xtoken1",
    ):
        """Create a test pool."""
        return {
            "id": pool_id,
            "chain": chain,
            "feeTier": fee_tier,
            "totalValueLockedUSD": tvl,
            "volumeUSD": volume,
            "token0": {"id": token0_id, "symbol": "TK0"},
            "token1": {"id": token1_id, "symbol": "TK1"},
        }

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.apply_composite_pre_filter"
    )
    def test_basic_filtering(self, mock_filter):
        """Test basic pool filtering."""
        mock_filter.side_effect = lambda x, **kw: x
        pools = [self._make_pool()]
        result = get_filtered_pools_for_uniswap(
            pools, [], {"optimism": {}}, max_apr_threshold=99999
        )
        assert len(result) == 1

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.apply_composite_pre_filter"
    )
    def test_excludes_current_positions(self, mock_filter):
        """Test exclusion of current positions."""
        mock_filter.side_effect = lambda x, **kw: x
        from web3 import Web3

        pool_id = "0x1234567890abcdef1234567890abcdef12345678"
        checksum = Web3.to_checksum_address(pool_id)
        pools = [self._make_pool(pool_id=pool_id)]
        result = get_filtered_pools_for_uniswap(
            pools, [checksum], {"optimism": {}}, max_apr_threshold=99999
        )
        assert len(result) == 0

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.apply_composite_pre_filter"
    )
    def test_apr_threshold_filter(self, mock_filter):
        """Test APR threshold filtering."""
        mock_filter.side_effect = lambda x, **kw: x
        pools = [self._make_pool(volume="10000000")]  # High volume -> high APR
        result = get_filtered_pools_for_uniswap(
            pools, [], {"optimism": {}}, max_apr_threshold=0.001
        )
        assert len(result) == 0

    def test_empty_result(self):
        """Test empty result when no pools qualify."""
        result = get_filtered_pools_for_uniswap([], [], {"optimism": {}})
        assert result == []

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.apply_composite_pre_filter"
    )
    def test_whitelisted_tokens_check(self, mock_filter):
        """Test whitelisted tokens filtering."""
        mock_filter.side_effect = lambda x, **kw: x
        pools = [self._make_pool()]
        # Whitelist requires specific tokens
        whitelisted = {"optimism": {"0xtoken0": "TK0", "0xtoken1": "TK1"}}
        result = get_filtered_pools_for_uniswap(
            pools, [], whitelisted, max_apr_threshold=99999
        )
        assert len(result) == 1

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.apply_composite_pre_filter"
    )
    def test_chain_not_whitelisted(self, mock_filter):
        """Test pool chain not in whitelisted assets."""
        mock_filter.side_effect = lambda x, **kw: x
        pools = [self._make_pool(chain="base")]
        result = get_filtered_pools_for_uniswap(
            pools, [], {"optimism": {}}, max_apr_threshold=99999
        )
        assert len(result) == 0


class TestFetchGraphqlData:
    """Tests for fetch_graphql_data function."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.run_query"
    )
    def test_fetches_pools(self, mock_query):
        """Test successful pool fetching."""
        mock_query.return_value = {"pools": [{"id": "0x1"}]}
        result = fetch_graphql_data(
            ["optimism"], {"optimism": "https://api.example.com"}
        )
        assert len(result) == 1
        assert result[0]["chain"] == "optimism"

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.run_query"
    )
    def test_no_endpoint_for_chain(self, mock_query):
        """Test when no endpoint exists for chain."""
        result = fetch_graphql_data(["optimism"], {})
        assert result == []
        mock_query.assert_not_called()

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.run_query"
    )
    def test_query_error(self, mock_query):
        """Test query error handling."""
        mock_query.return_value = {"error": "query failed"}
        result = fetch_graphql_data(
            ["optimism"], {"optimism": "https://api.example.com"}
        )
        assert result == []


class TestGetUniswapPoolSharpeRatio:
    """Tests for get_uniswap_pool_sharpe_ratio function."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_successful_calculation(self, mock_post):
        """Test successful Sharpe ratio calculation."""
        data = [
            {
                "date": str(1700000000 + i * 86400),
                "tvlUSD": "100000",
                "feesUSD": str(100 + i),
            }
            for i in range(30)
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"poolDayDatas": data}}
        mock_post.return_value = mock_resp
        result = get_uniswap_pool_sharpe_ratio("0x1", "https://api.example.com")
        assert isinstance(result, float)

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_no_data(self, mock_post):
        """Test with no data."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"poolDayDatas": []}}
        mock_post.return_value = mock_resp
        result = get_uniswap_pool_sharpe_ratio("0x1", "https://api.example.com")
        assert np.isnan(result)


class TestCalculateIlImpact:
    """Tests for calculate_il_impact function."""

    def test_equal_prices(self):
        """Test with equal initial and final prices."""
        result = calculate_il_impact(1.0, 1.0)
        assert abs(result) < 1e-10

    def test_price_increase(self):
        """Test with price increase."""
        result = calculate_il_impact(1.0, 4.0)
        assert result < 0


class TestIsProApiKey:
    """Tests for is_pro_api_key function."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_pro_key(self, mock_cg):
        """Test pro API key detection."""
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.return_value = {"prices": []}
        mock_cg.return_value = inst
        assert is_pro_api_key("key") is True

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_non_pro_key(self, mock_cg):
        """Test non-pro API key."""
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = Exception("fail")
        mock_cg.return_value = inst
        assert is_pro_api_key("key") is False

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_empty_response(self, mock_cg):
        """Test empty response."""
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.return_value = {}
        mock_cg.return_value = inst
        assert is_pro_api_key("key") is False


class TestCalculateIlRiskScore:
    """Tests for calculate_il_risk_score function."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_successful_calculation(self, mock_cg, mock_is_pro):
        """Test successful IL risk score calculation."""
        mock_is_pro.return_value = False
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = [
            {"prices": [[i, 100 + i * 0.5] for i in range(100)]},
            {"prices": [[i, 200 + i * 0.3] for i in range(100)]},
        ]
        mock_cg.return_value = inst
        result = calculate_il_risk_score("t0", "t1", "key")
        assert isinstance(result, float)

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_with_x402(self, mock_cg):
        """Test with x402 session."""
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = [
            {"prices": [[i, 100 + i] for i in range(10)]},
            {"prices": [[i, 200 + i] for i in range(10)]},
        ]
        mock_cg.return_value = inst
        result = calculate_il_risk_score(
            "t0", "t1", "key", x402_session=MagicMock(), x402_proxy="https://proxy.com"
        )
        assert isinstance(result, float)

    def test_no_api_key_no_x402(self):
        """Test with no API key and no x402."""
        result = calculate_il_risk_score("t0", "t1", "")
        assert result is None

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_exception(self, mock_cg):
        """Test exception handling."""
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = Exception("fail")
        mock_cg.return_value = inst
        result = calculate_il_risk_score(
            "t0", "t1", "", x402_session=MagicMock(), x402_proxy="https://p.com"
        )
        assert result is None

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_insufficient_data(self, mock_cg, mock_is_pro):
        """Test with insufficient data points."""
        mock_is_pro.return_value = False
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = [
            {"prices": [[1, 100]]},
            {"prices": [[1, 200]]},
        ]
        mock_cg.return_value = inst
        result = calculate_il_risk_score("t0", "t1", "key")
        assert result is None

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_pro_api_key_used(self, mock_cg, mock_is_pro):
        """Test with pro API key."""
        mock_is_pro.return_value = True
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = [
            {"prices": [[i, 100 + i] for i in range(10)]},
            {"prices": [[i, 200 + i] for i in range(10)]},
        ]
        mock_cg.return_value = inst
        result = calculate_il_risk_score("t0", "t1", "key")
        assert isinstance(result, float)


class TestFetchPoolData:
    """Tests for fetch_pool_data function."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_successful(self, mock_post):
        """Test successful pool data fetch."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"pool": {"id": "0x1"}}}
        mock_post.return_value = mock_resp
        result = fetch_pool_data("0x1", "https://api.example.com")
        assert result == {"id": "0x1"}

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_error_status(self, mock_post):
        """Test error status code."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {}
        mock_post.return_value = mock_resp
        result = fetch_pool_data("0x1", "https://api.example.com")
        assert result is None

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_exception(self, mock_post):
        """Test exception handling."""
        mock_post.side_effect = Exception("network error")
        result = fetch_pool_data("0x1", "https://api.example.com")
        assert result is None


class TestCalculateMetricsLiquidityRisk:
    """Tests for calculate_metrics_liquidity_risk function."""

    def test_valid_data(self):
        """Test with valid data."""
        pool_data = {
            "totalValueLockedUSD": "100000",
            "totalValueLockedToken0": "50000",
            "totalValueLockedToken1": "50000",
        }
        depth, max_pos = calculate_metrics_liquidity_risk(pool_data)
        assert depth > 0
        assert max_pos > 0

    def test_zero_tokens(self):
        """Test with zero token values."""
        pool_data = {
            "totalValueLockedUSD": "100000",
            "totalValueLockedToken0": "0",
            "totalValueLockedToken1": "0",
        }
        depth, max_pos = calculate_metrics_liquidity_risk(pool_data)
        assert depth == 0

    def test_exception(self):
        """Test exception handling returns nan values."""
        pool_data = {
            "totalValueLockedUSD": "not_a_number",
            "totalValueLockedToken0": "bad",
            "totalValueLockedToken1": "bad",
        }
        depth, max_pos = calculate_metrics_liquidity_risk(pool_data)
        assert np.isnan(depth)
        assert np.isnan(max_pos)


class TestAssessPoolLiquidity:
    """Tests for assess_pool_liquidity function."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.fetch_pool_data"
    )
    def test_no_pool_data(self, mock_fetch):
        """Test when pool data is None."""
        mock_fetch.return_value = None
        depth, max_pos = assess_pool_liquidity("0x1", "https://api.example.com")
        assert np.isnan(depth)
        assert np.isnan(max_pos)

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.fetch_pool_data"
    )
    def test_with_pool_data(self, mock_fetch):
        """Test with valid pool data."""
        mock_fetch.return_value = {
            "totalValueLockedUSD": "100000",
            "totalValueLockedToken0": "50000",
            "totalValueLockedToken1": "50000",
        }
        depth, max_pos = assess_pool_liquidity("0x1", "https://api.example.com")
        assert depth > 0


class TestFormatPoolData:
    """Tests for format_pool_data function."""

    def test_formats_correctly(self):
        """Test correct formatting."""
        pool = {
            "chain": "optimism",
            "apr": 10.5,
            "id": "0x1",
            "token0": {"id": "0xt0", "symbol": "TK0"},
            "token1": {"id": "0xt1", "symbol": "TK1"},
            "il_risk_score": -0.05,
            "sharpe_ratio": 1.5,
            "depth_score": 100,
            "max_position_size": 5000,
            "type": "lp",
        }
        result = format_pool_data(pool)
        assert result["dex_type"] == "UniswapV3"
        assert result["chain"] == "optimism"
        assert result["pool_address"] == "0x1"


class TestGetOpportunitiesForUniswap:
    """Tests for get_opportunities_for_uniswap function."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.fetch_graphql_data"
    )
    def test_graphql_error(self, mock_fetch):
        """Test GraphQL error."""
        mock_fetch.return_value = {"error": "query failed"}
        result = get_opportunities_for_uniswap(
            ["optimism"], {"optimism": "url"}, [], "key", {}, {}, None, None
        )
        assert "error" in result

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.get_filtered_pools_for_uniswap"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.fetch_graphql_data"
    )
    def test_no_filtered_pools(self, mock_fetch, mock_filter):
        """Test no pools after filtering."""
        mock_fetch.return_value = [{"id": "0x1"}]
        mock_filter.return_value = []
        result = get_opportunities_for_uniswap(
            ["optimism"], {"optimism": "url"}, [], "key", {}, {}, None, None
        )
        assert "error" in result

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.assess_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.get_uniswap_pool_sharpe_ratio"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.calculate_il_risk_score"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.get_filtered_pools_for_uniswap"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.fetch_graphql_data"
    )
    def test_successful_flow(
        self, mock_fetch, mock_filter, mock_il, mock_sharpe, mock_liquidity
    ):
        """Test successful end-to-end flow."""
        pool = {
            "id": "0x1",
            "chain": "optimism",
            "apr": 10,
            "token0": {"id": "0xt0", "symbol": "TK0"},
            "token1": {"id": "0xt1", "symbol": "TK1"},
        }
        mock_fetch.return_value = [pool]
        mock_filter.return_value = [pool]
        mock_il.return_value = -0.05
        mock_sharpe.return_value = 1.5
        mock_liquidity.return_value = (100, 5000)
        coin_id_mapping = {"optimism": {"tk0": "tk0-id", "tk1": "tk1-id"}}
        result = get_opportunities_for_uniswap(
            ["optimism"],
            {"optimism": "url"},
            [],
            "key",
            {},
            coin_id_mapping,
            None,
            None,
        )
        assert isinstance(result, list)
        assert len(result) == 1

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.assess_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.get_uniswap_pool_sharpe_ratio"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.calculate_il_risk_score"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.get_filtered_pools_for_uniswap"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.fetch_graphql_data"
    )
    def test_token_id_not_found(
        self, mock_fetch, mock_filter, mock_il, mock_sharpe, mock_liquidity
    ):
        """Test when token IDs are not found in mapping."""
        pool = {
            "id": "0x1",
            "chain": "optimism",
            "apr": 10,
            "token0": {"id": "0xt0", "symbol": "UNKNOWN0"},
            "token1": {"id": "0xt1", "symbol": "UNKNOWN1"},
        }
        mock_fetch.return_value = [pool]
        mock_filter.return_value = [pool]
        mock_sharpe.return_value = 1.5
        mock_liquidity.return_value = (100, 5000)
        result = get_opportunities_for_uniswap(
            ["optimism"], {"optimism": "url"}, [], "key", {}, {}, None, None
        )
        assert isinstance(result, list)

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.assess_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.get_uniswap_pool_sharpe_ratio"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.calculate_il_risk_score"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.get_filtered_pools_for_uniswap"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.fetch_graphql_data"
    )
    def test_token_id_cache(
        self, mock_fetch, mock_filter, mock_il, mock_sharpe, mock_liquidity
    ):
        """Test that token IDs are cached between pools."""
        pool1 = {
            "id": "0x1",
            "chain": "optimism",
            "apr": 10,
            "token0": {"id": "0xt0", "symbol": "TK0"},
            "token1": {"id": "0xt1", "symbol": "TK1"},
        }
        pool2 = {
            "id": "0x2",
            "chain": "optimism",
            "apr": 15,
            "token0": {"id": "0xt0", "symbol": "TK0"},
            "token1": {"id": "0xt1", "symbol": "TK1"},
        }
        mock_fetch.return_value = [pool1, pool2]
        mock_filter.return_value = [pool1, pool2]
        mock_il.return_value = -0.05
        mock_sharpe.return_value = 1.5
        mock_liquidity.return_value = (100, 5000)
        coin_id_mapping = {"optimism": {"tk0": "tk0-id", "tk1": "tk1-id"}}
        result = get_opportunities_for_uniswap(
            ["optimism"],
            {"optimism": "url"},
            [],
            "key",
            {},
            coin_id_mapping,
            None,
            None,
        )
        assert len(result) == 2


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.assess_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.get_uniswap_pool_sharpe_ratio"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.calculate_il_risk_score"
    )
    def test_successful(self, mock_il, mock_sharpe, mock_liquidity):
        """Test successful metrics calculation."""
        mock_il.return_value = -0.05
        mock_sharpe.return_value = 1.5
        mock_liquidity.return_value = (100, 5000)
        position = {
            "token0_symbol": "TK0",
            "token1_symbol": "TK1",
            "chain": "optimism",
            "pool_address": "0x1",
        }
        result = calculate_metrics(
            position,
            "key",
            {"optimism": "url"},
            {"optimism": {"tk0": "tk0-id", "tk1": "tk1-id"}},
            None,
            None,
        )
        assert result["il_risk_score"] == -0.05

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.assess_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.get_uniswap_pool_sharpe_ratio"
    )
    def test_no_token_ids(self, mock_sharpe, mock_liquidity):
        """Test when token IDs are not found."""
        mock_sharpe.return_value = 1.5
        mock_liquidity.return_value = (100, 5000)
        position = {
            "token0_symbol": "UNKNOWN0",
            "token1_symbol": "UNKNOWN1",
            "chain": "optimism",
            "pool_address": "0x1",
        }
        result = calculate_metrics(position, "key", {"optimism": "url"}, {}, None, None)
        assert result["il_risk_score"] is None


class TestRun:
    """Tests for the run function."""

    def test_missing_fields(self):
        """Test with missing fields."""
        result = run()
        assert "error" in result

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.calculate_metrics"
    )
    def test_get_metrics_mode(self, mock_calc):
        """Test get_metrics mode."""
        mock_calc.return_value = {"il_risk_score": -0.05}
        result = run(
            chains=["optimism"],
            graphql_endpoints={"optimism": "url"},
            current_positions=[],
            whitelisted_assets={},
            get_metrics=True,
            position={"pool_address": "0x1"},
            coingecko_api_key="key",
            coin_id_mapping={},
            x402_session=None,
            x402_proxy=None,
        )
        assert result == {"il_risk_score": -0.05}

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.calculate_metrics"
    )
    def test_get_metrics_returns_none(self, mock_calc):
        """Test get_metrics mode when metrics is None."""
        mock_calc.return_value = None
        result = run(
            chains=["optimism"],
            graphql_endpoints={"optimism": "url"},
            current_positions=[],
            whitelisted_assets={},
            get_metrics=True,
            position={"pool_address": "0x1"},
            coingecko_api_key="key",
            coin_id_mapping={},
            x402_session=None,
            x402_proxy=None,
        )
        assert "error" in result

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.get_opportunities_for_uniswap"
    )
    def test_opportunity_search(self, mock_opp):
        """Test opportunity search mode."""
        mock_opp.return_value = [{"pool": "data"}]
        result = run(
            chains=["optimism"],
            graphql_endpoints={"optimism": "url"},
            current_positions=[],
            whitelisted_assets={},
        )
        assert "result" in result

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.get_opportunities_for_uniswap"
    )
    def test_opportunity_search_error(self, mock_opp):
        """Test opportunity search error."""
        mock_opp.return_value = {"error": "no pools"}
        result = run(
            chains=["optimism"],
            graphql_endpoints={"optimism": "url"},
            current_positions=[],
            whitelisted_assets={},
        )
        assert "result" in result

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.get_opportunities_for_uniswap"
    )
    def test_opportunity_search_empty(self, mock_opp):
        """Test empty opportunity search."""
        mock_opp.return_value = []
        result = run(
            chains=["optimism"],
            graphql_endpoints={"optimism": "url"},
            current_positions=[],
            whitelisted_assets={},
        )
        assert "result" in result


class TestResetX402Adapter:
    """Tests for _reset_x402_adapter helper."""

    def test_none_session(self):
        """No error when session is None."""
        _reset_x402_adapter(None)

    def test_session_with_retry_flag(self):
        """Resets _is_retry on adapters that have it."""
        adapter = MagicMock()
        adapter._is_retry = True
        session = MagicMock()
        session.adapters = {"https://": adapter}
        _reset_x402_adapter(session)
        assert adapter._is_retry is False

    def test_session_without_retry_flag(self):
        """No error when adapter lacks _is_retry."""
        adapter = object()
        session = MagicMock()
        session.adapters = {"https://": adapter}
        _reset_x402_adapter(session)

    def test_empty_adapters(self):
        """No error when session has no adapters."""
        session = MagicMock()
        session.adapters = {}
        _reset_x402_adapter(session)

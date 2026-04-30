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

import time
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import packages.valory.customs.uniswap_pools_search.uniswap_pools_search as uni_mod
from packages.valory.customs.uniswap_pools_search.uniswap_pools_search import (
    REQUIRED_FIELDS,
    _reset_x402_adapter,
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
    get_cached_price,
    get_coin_id_from_symbol,
    get_errors,
    get_filtered_pools_for_uniswap,
    get_opportunities_for_uniswap,
    get_uniswap_pool_sharpe_ratio,
    is_pro_api_key,
    remove_irrelevant_fields,
    run,
    run_query,
    set_cached_price,
    standardize_metrics,
)


@pytest.fixture(autouse=True)
def reset_state() -> Generator[Any, Any, Any]:
    """Reset module state before each test.

    :yield: TODO
    """
    if hasattr(uni_mod._thread_local, "errors"):
        uni_mod._thread_local.errors = []
    yield
    if hasattr(uni_mod._thread_local, "errors"):
        uni_mod._thread_local.errors = []


class TestGetErrors:
    """Tests for get_errors function."""

    def test_initializes_empty(self) -> None:
        """Test initialization of errors list."""
        if hasattr(uni_mod._thread_local, "errors"):
            del uni_mod._thread_local.errors
        assert get_errors() == []

    def test_returns_existing(self) -> None:
        """Test returning existing errors."""
        uni_mod._thread_local.errors = ["err"]
        assert get_errors() == ["err"]


class TestCheckMissingFields:
    """Tests for check_missing_fields function."""

    def test_no_missing(self) -> None:
        """Test with all fields present."""
        kwargs = {f: "v" for f in REQUIRED_FIELDS}
        assert check_missing_fields(kwargs) == []

    def test_all_missing(self) -> None:
        """Test with all fields missing."""
        assert len(check_missing_fields({})) == len(REQUIRED_FIELDS)

    def test_none_value(self) -> None:
        """Test None value counts as missing."""
        kwargs = {f: "v" for f in REQUIRED_FIELDS}
        kwargs["chains"] = None  # type: ignore[assignment]
        assert "chains" in check_missing_fields(kwargs)


class TestRemoveIrrelevantFields:
    """Tests for remove_irrelevant_fields function."""

    def test_removes_extras(self) -> None:
        """Test removing extra fields."""
        result = remove_irrelevant_fields({"a": 1, "b": 2}, ("a",))
        assert result == {"a": 1}


class TestGetCoinIdFromSymbol:
    """Tests for get_coin_id_from_symbol function."""

    def test_found(self) -> None:
        """Test finding coin ID."""
        mapping = {"optimism": {"weth": "weth-id"}}
        assert get_coin_id_from_symbol(mapping, "WETH", "optimism") == "weth-id"

    def test_not_found(self) -> None:
        """Test when coin ID is not found."""
        mapping = {"optimism": {"weth": "weth-id"}}
        assert get_coin_id_from_symbol(mapping, "unknown", "optimism") is None

    def test_chain_not_in_mapping(self) -> None:
        """Test when chain is not in mapping."""
        mapping: Dict[Any, Any] = {}
        assert get_coin_id_from_symbol(mapping, "weth", "optimism") is None


class TestRunQuery:
    """Tests for run_query function."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_successful_query(self, mock_post: MagicMock) -> None:
        """Test successful GraphQL query.

        :param mock_post: TODO
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"pools": []}}
        mock_post.return_value = mock_resp
        result = run_query("{pools{id}}", "https://api.example.com")
        assert result == {"pools": []}

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_error_status(self, mock_post: MagicMock) -> None:
        """Test error status code.

        :param mock_post: TODO
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp
        result = run_query("{pools{id}}", "https://api.example.com")
        assert "error" in result

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_graphql_errors(self, mock_post: MagicMock) -> None:
        """Test GraphQL errors in response.

        :param mock_post: TODO
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"errors": [{"message": "bad"}]}
        mock_post.return_value = mock_resp
        result = run_query("{pools{id}}", "https://api.example.com")
        assert "error" in result

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_null_data_response(self, mock_post: MagicMock) -> None:
        """Test that {"data": null} response does not crash.

        :param mock_post: TODO
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": None}
        mock_post.return_value = mock_resp
        result = run_query("{pools{id}}", "https://api.example.com")
        assert result == {}

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_with_variables(self, mock_post: MagicMock) -> None:
        """Test query with variables.

        :param mock_post: TODO
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"result": "ok"}}
        mock_post.return_value = mock_resp
        result = run_query("query", "url", variables={"key": "val"})
        assert result == {"result": "ok"}


class TestCalculateApr:
    """Tests for calculate_apr function."""

    def test_normal_calculation(self) -> None:
        """Test normal APR calculation."""
        result = calculate_apr(1000, 100000, 0.003)
        assert result > 0

    def test_zero_tvl(self) -> None:
        """Test zero TVL returns 0."""
        assert calculate_apr(1000, 0, 0.003) == 0


class TestStandardizeMetrics:
    """Tests for standardize_metrics function."""

    def test_empty_pools(self) -> None:
        """Test with empty pools returns early."""
        assert standardize_metrics([]) == []

    def test_single_pool(self) -> None:
        """Test single pool (zero std dev)."""
        pools = [{"apr": 10, "tvl": 1000}]
        result = standardize_metrics(pools)
        assert "composite_score" in result[0]

    def test_multiple_pools(self) -> None:
        """Test multiple pools."""
        pools = [{"apr": 10, "tvl": 1000}, {"apr": 20, "tvl": 2000}]
        result = standardize_metrics(pools)
        assert result[1]["composite_score"] > result[0]["composite_score"]

    def test_invalid_values(self) -> None:
        """Test with invalid apr/tvl values."""
        pools = [{"apr": "invalid", "tvl": "bad"}]
        result = standardize_metrics(pools)
        assert "composite_score" in result[0]


class TestApplyCompositePreFilter:
    """Tests for apply_composite_pre_filter function."""

    def test_empty_pools(self) -> None:
        """Test with empty pools."""
        assert apply_composite_pre_filter([]) == []

    def test_disabled_filter(self) -> None:
        """Test with filter disabled."""
        pools = [{"tvl": 5000, "apr": 10}]
        result = apply_composite_pre_filter(pools, use_composite_filter=False)
        assert len(result) == 1

    def test_disabled_filter_empty(self) -> None:
        """Test disabled filter with empty pools."""
        assert apply_composite_pre_filter([], use_composite_filter=False) == []

    def test_below_tvl_threshold(self) -> None:
        """Test pools below TVL threshold."""
        pools = [{"tvl": 100, "apr": 10}]
        result = apply_composite_pre_filter(pools, min_tvl_threshold=1000)
        assert result == []

    def test_invalid_tvl(self) -> None:
        """Test with invalid TVL."""
        pools = [{"tvl": "invalid", "apr": 10}]
        result = apply_composite_pre_filter(pools, min_tvl_threshold=0)
        assert result == []

    def test_top_n(self) -> None:
        """Test top_n limit."""
        pools = [{"tvl": 5000, "apr": 10 + i} for i in range(5)]
        result = apply_composite_pre_filter(pools, top_n=2, min_tvl_threshold=0)
        assert len(result) == 2


class TestGetFilteredPoolsForUniswap:
    """Tests for get_filtered_pools_for_uniswap function."""

    def _make_pool(
        self,
        pool_id: Any = "0x1234567890abcdef1234567890abcdef12345678",
        chain: Any = "optimism",
        fee_tier: Any = "3000",
        tvl: Any = "100000",
        volume: Any = "50000",
        token0_id: Any = "0xtoken0",
        token1_id: Any = "0xtoken1",
    ) -> Any:
        """Create a test pool.

        :param chain: TODO
        :param fee_tier: TODO
        :param pool_id: TODO
        :param token0_id: TODO
        :param token1_id: TODO
        :param tvl: TODO
        :param volume: TODO
        :return: TODO
        """
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
    def test_basic_filtering(self, mock_filter: MagicMock) -> None:
        """Test basic pool filtering.

        :param mock_filter: TODO
        """
        mock_filter.side_effect = lambda x, **kw: x
        pools = [self._make_pool()]
        result = get_filtered_pools_for_uniswap(
            pools, [], {"optimism": {}}, max_apr_threshold=99999
        )
        assert len(result) == 1

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.apply_composite_pre_filter"
    )
    def test_excludes_current_positions(self, mock_filter: MagicMock) -> None:
        """Test exclusion of current positions.

        :param mock_filter: TODO
        """
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
    def test_apr_threshold_filter(self, mock_filter: MagicMock) -> None:
        """Test APR threshold filtering.

        :param mock_filter: TODO
        """
        mock_filter.side_effect = lambda x, **kw: x
        pools = [self._make_pool(volume="10000000")]  # High volume -> high APR
        result = get_filtered_pools_for_uniswap(
            pools, [], {"optimism": {}}, max_apr_threshold=0.001
        )
        assert len(result) == 0

    def test_empty_result(self) -> None:
        """Test empty result when no pools qualify."""
        result = get_filtered_pools_for_uniswap([], [], {"optimism": {}})
        assert result == []

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.apply_composite_pre_filter"
    )
    def test_whitelisted_tokens_check(self, mock_filter: MagicMock) -> None:
        """Test whitelisted tokens filtering.

        :param mock_filter: TODO
        """
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
    def test_chain_not_whitelisted(self, mock_filter: MagicMock) -> None:
        """Test pool chain not in whitelisted assets.

        :param mock_filter: TODO
        """
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
    def test_fetches_pools(self, mock_query: MagicMock) -> None:
        """Test successful pool fetching.

        :param mock_query: TODO
        """
        mock_query.return_value = {"pools": [{"id": "0x1"}]}
        result = fetch_graphql_data(
            ["optimism"], {"optimism": "https://api.example.com"}
        )
        assert len(result) == 1
        assert result[0]["chain"] == "optimism"  # type: ignore[index]

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.run_query"
    )
    def test_no_endpoint_for_chain(self, mock_query: MagicMock) -> None:
        """Test when no endpoint exists for chain.

        :param mock_query: TODO
        """
        result = fetch_graphql_data(["optimism"], {})
        assert result == []
        mock_query.assert_not_called()

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.run_query"
    )
    def test_query_error(self, mock_query: MagicMock) -> None:
        """Test query error handling.

        :param mock_query: TODO
        """
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
    def test_successful_calculation(self, mock_post: MagicMock) -> None:
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
    def test_no_data(self, mock_post: MagicMock) -> None:
        """Test with no data.

        :param mock_post: TODO
        """
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"poolDayDatas": []}}
        mock_post.return_value = mock_resp
        result = get_uniswap_pool_sharpe_ratio("0x1", "https://api.example.com")
        assert np.isnan(result)

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_null_data_response(self, mock_post: MagicMock) -> None:
        """Test that {"data": null} response does not crash.

        :param mock_post: TODO
        """
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": None}
        mock_post.return_value = mock_resp
        result = get_uniswap_pool_sharpe_ratio("0x1", "https://api.example.com")
        assert np.isnan(result)


class TestCalculateIlImpact:
    """Tests for calculate_il_impact function."""

    def test_equal_prices(self) -> None:
        """Test with equal initial and final prices."""
        result = calculate_il_impact(1.0, 1.0)
        assert abs(result) < 1e-10

    def test_price_increase(self) -> None:
        """Test with price increase."""
        result = calculate_il_impact(1.0, 4.0)
        assert result < 0


class TestIsProApiKey:
    """Tests for is_pro_api_key function."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_pro_key(self, mock_cg: MagicMock) -> None:
        """Test pro API key detection.

        :param mock_cg: TODO
        """
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.return_value = {"prices": []}
        mock_cg.return_value = inst
        assert is_pro_api_key("key") is True

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_non_pro_key(self, mock_cg: MagicMock) -> None:
        """Test non-pro API key.

        :param mock_cg: TODO
        """
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = Exception("fail")
        mock_cg.return_value = inst
        assert is_pro_api_key("key") is False

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_empty_response(self, mock_cg: MagicMock) -> None:
        """Test empty response.

        :param mock_cg: TODO
        """
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
    def test_successful_calculation(
        self, mock_cg: MagicMock, mock_is_pro: MagicMock
    ) -> None:
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
    def test_with_x402(self, mock_cg: MagicMock) -> None:
        """Test with x402 session.

        :param mock_cg: TODO
        """
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

    def test_no_api_key_no_x402(self) -> None:
        """Test with no API key and no x402."""
        result = calculate_il_risk_score("t0", "t1", "")
        assert result is None

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_exception(self, mock_cg: MagicMock) -> None:
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
    def test_insufficient_data(
        self, mock_cg: MagicMock, mock_is_pro: MagicMock
    ) -> None:
        """Test with insufficient data points.

        :param mock_cg: TODO
        :param mock_is_pro: TODO
        """
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
    def test_pro_api_key_used(self, mock_cg: MagicMock, mock_is_pro: MagicMock) -> None:
        """Test with pro API key.

        :param mock_cg: TODO
        :param mock_is_pro: TODO
        """
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
    def test_successful(self, mock_post: MagicMock) -> None:
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
    def test_error_status(self, mock_post: MagicMock) -> None:
        """Test error status code.

        :param mock_post: TODO
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {}
        mock_post.return_value = mock_resp
        result = fetch_pool_data("0x1", "https://api.example.com")
        assert result is None

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_exception(self, mock_post: MagicMock) -> None:
        """Test exception handling."""
        mock_post.side_effect = Exception("network error")
        result = fetch_pool_data("0x1", "https://api.example.com")
        assert result is None


class TestCalculateMetricsLiquidityRisk:
    """Tests for calculate_metrics_liquidity_risk function."""

    def test_valid_data(self) -> None:
        """Test with valid data."""
        pool_data = {
            "totalValueLockedUSD": "100000",
            "totalValueLockedToken0": "50000",
            "totalValueLockedToken1": "50000",
        }
        depth, max_pos = calculate_metrics_liquidity_risk(pool_data)
        assert depth > 0
        assert max_pos > 0

    def test_zero_tokens(self) -> None:
        """Test with zero token values."""
        pool_data = {
            "totalValueLockedUSD": "100000",
            "totalValueLockedToken0": "0",
            "totalValueLockedToken1": "0",
        }
        depth, max_pos = calculate_metrics_liquidity_risk(pool_data)
        assert depth == 0

    def test_exception(self) -> None:
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
    def test_no_pool_data(self, mock_fetch: MagicMock) -> None:
        """Test when pool data is None.

        :param mock_fetch: TODO
        """
        mock_fetch.return_value = None
        depth, max_pos = assess_pool_liquidity("0x1", "https://api.example.com")
        assert np.isnan(depth)
        assert np.isnan(max_pos)

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.fetch_pool_data"
    )
    def test_with_pool_data(self, mock_fetch: MagicMock) -> None:
        """Test with valid pool data.

        :param mock_fetch: TODO
        """
        mock_fetch.return_value = {
            "totalValueLockedUSD": "100000",
            "totalValueLockedToken0": "50000",
            "totalValueLockedToken1": "50000",
        }
        depth, max_pos = assess_pool_liquidity("0x1", "https://api.example.com")
        assert depth > 0


class TestFormatPoolData:
    """Tests for format_pool_data function."""

    def test_formats_correctly(self) -> None:
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
    def test_graphql_error(self, mock_fetch: MagicMock) -> None:
        """Test GraphQL error.

        :param mock_fetch: TODO
        """
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
    def test_no_filtered_pools(
        self, mock_fetch: MagicMock, mock_filter: MagicMock
    ) -> None:
        """Test no pools after filtering.

        :param mock_fetch: TODO
        :param mock_filter: TODO
        """
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
        self,
        mock_fetch: MagicMock,
        mock_filter: MagicMock,
        mock_il: MagicMock,
        mock_sharpe: MagicMock,
        mock_liquidity: MagicMock,
    ) -> None:
        """Test successful end-to-end flow.

        :param mock_fetch: TODO
        :param mock_filter: TODO
        :param mock_il: TODO
        :param mock_liquidity: TODO
        :param mock_sharpe: TODO
        """
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
        self,
        mock_fetch: MagicMock,
        mock_filter: MagicMock,
        mock_il: MagicMock,
        mock_sharpe: MagicMock,
        mock_liquidity: MagicMock,
    ) -> None:
        """Test when token IDs are not found in mapping.

        :param mock_fetch: TODO
        :param mock_filter: TODO
        :param mock_il: TODO
        :param mock_liquidity: TODO
        :param mock_sharpe: TODO
        """
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
        self,
        mock_fetch: MagicMock,
        mock_filter: MagicMock,
        mock_il: MagicMock,
        mock_sharpe: MagicMock,
        mock_liquidity: MagicMock,
    ) -> None:
        """Test that token IDs are cached between pools.

        :param mock_fetch: TODO
        :param mock_filter: TODO
        :param mock_il: TODO
        :param mock_liquidity: TODO
        :param mock_sharpe: TODO
        """
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
    def test_successful(
        self, mock_il: MagicMock, mock_sharpe: MagicMock, mock_liquidity: MagicMock
    ) -> None:
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
        assert result["il_risk_score"] == -0.05  # type: ignore[index]

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.assess_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.get_uniswap_pool_sharpe_ratio"
    )
    def test_no_token_ids(
        self, mock_sharpe: MagicMock, mock_liquidity: MagicMock
    ) -> None:
        """Test when token IDs are not found.

        :param mock_liquidity: TODO
        :param mock_sharpe: TODO
        """
        mock_sharpe.return_value = 1.5
        mock_liquidity.return_value = (100, 5000)
        position = {
            "token0_symbol": "UNKNOWN0",
            "token1_symbol": "UNKNOWN1",
            "chain": "optimism",
            "pool_address": "0x1",
        }
        result = calculate_metrics(position, "key", {"optimism": "url"}, {}, None, None)
        assert result["il_risk_score"] is None  # type: ignore[index]


class TestRun:
    """Tests for the run function."""

    def test_missing_fields(self) -> None:
        """Test with missing fields."""
        result = run()
        assert "error" in result

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.calculate_metrics"
    )
    def test_get_metrics_mode(self, mock_calc: MagicMock) -> None:
        """Test get_metrics mode.

        :param mock_calc: TODO
        """
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
    def test_get_metrics_returns_none(self, mock_calc: MagicMock) -> None:
        """Test get_metrics mode when metrics is None.

        :param mock_calc: TODO
        """
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
    def test_opportunity_search(self, mock_opp: MagicMock) -> None:
        """Test opportunity search mode.

        :param mock_opp: TODO
        """
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
    def test_opportunity_search_error(self, mock_opp: MagicMock) -> None:
        """Test opportunity search error.

        :param mock_opp: TODO
        """
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
    def test_opportunity_search_empty(self, mock_opp: MagicMock) -> None:
        """Test empty opportunity search.

        :param mock_opp: TODO
        """
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

    def test_none_session(self) -> None:
        """No error when session is None."""
        _reset_x402_adapter(None)

    def test_session_with_retry_flag(self) -> None:
        """Resets _is_retry on adapters that have it."""
        adapter = MagicMock()
        adapter._is_retry = True
        session = MagicMock()
        session.adapters = {"https://": adapter}
        _reset_x402_adapter(session)
        assert adapter._is_retry is False

    def test_session_without_retry_flag(self) -> None:
        """No error when adapter lacks _is_retry."""
        adapter = object()
        session = MagicMock()
        session.adapters = {"https://": adapter}
        _reset_x402_adapter(session)

    def test_empty_adapters(self) -> None:
        """No error when session has no adapters."""
        session = MagicMock()
        session.adapters = {}
        _reset_x402_adapter(session)


class TestRunQueryNetworkResilience:
    """Tests for run_query handling network errors and parse failures."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_connection_error(self, mock_post: MagicMock) -> None:
        """Test that ConnectionError is caught and returns error dict.

        :param mock_post: TODO
        """
        import requests as req_lib

        mock_post.side_effect = req_lib.ConnectionError("connection refused")
        result = run_query("{pools{id}}", "https://api.example.com")
        assert "error" in result

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_timeout_error(self, mock_post: MagicMock) -> None:
        """Test that Timeout is caught and returns error dict.

        :param mock_post: TODO
        """
        import requests as req_lib

        mock_post.side_effect = req_lib.Timeout("timed out")
        result = run_query("{pools{id}}", "https://api.example.com")
        assert "error" in result

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_json_decode_error(self, mock_post: MagicMock) -> None:
        """Test that JSONDecodeError from response.json() returns error dict.

        :param mock_post: TODO
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("No JSON")
        mock_post.return_value = mock_resp
        result = run_query("{pools{id}}", "https://api.example.com")
        assert "error" in result

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_timeout_kwarg_passed(self, mock_post: MagicMock) -> None:
        """Test that timeout=30 is passed to requests.post.

        :param mock_post: TODO
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"pools": []}}
        mock_post.return_value = mock_resp
        run_query("{pools{id}}", "https://api.example.com")
        _, kwargs = mock_post.call_args
        assert kwargs.get("timeout") == 30

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_data_null_returns_empty_dict(self, mock_post: MagicMock) -> None:
        """Test that {"data": null} returns empty dict.

        :param mock_post: TODO
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": None}
        mock_post.return_value = mock_resp
        result = run_query("{pools{id}}", "https://api.example.com")
        assert result == {}


class TestGetUniswapPoolSharpeRatioNetworkResilience:
    """Tests for get_uniswap_pool_sharpe_ratio handling network errors."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_connection_error(self, mock_post: MagicMock) -> None:
        """Test that ConnectionError returns NaN.

        :param mock_post: TODO
        """
        import requests as req_lib

        mock_post.side_effect = req_lib.ConnectionError("refused")
        result = get_uniswap_pool_sharpe_ratio("0x1", "https://api.example.com")
        assert np.isnan(result)

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_timeout_kwarg_passed(self, mock_post: MagicMock) -> None:
        """Test that timeout=30 is passed to requests.post.

        :param mock_post: TODO
        """
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"poolDayDatas": []}}
        mock_post.return_value = mock_resp
        get_uniswap_pool_sharpe_ratio("0x1", "https://api.example.com")
        _, kwargs = mock_post.call_args
        assert kwargs.get("timeout") == 30


class TestFetchPoolDataNetworkResilience:
    """Tests for fetch_pool_data handling timeout parameter."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.requests.post"
    )
    def test_timeout_kwarg_passed(self, mock_post: MagicMock) -> None:
        """Test that timeout=30 is passed to requests.post.

        :param mock_post: TODO
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"pool": {"id": "0x1"}}}
        mock_post.return_value = mock_resp
        fetch_pool_data("0x1", "https://api.example.com")
        _, kwargs = mock_post.call_args
        assert kwargs.get("timeout") == 30


class TestCalculateIlRiskScoreMissingPricesKey:
    """Tests for calculate_il_risk_score with missing prices key."""

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_missing_prices_key_returns_none(
        self, mock_cg: MagicMock, mock_is_pro: MagicMock
    ) -> None:
        """Test that missing 'prices' key in response returns None.

        :param mock_cg: TODO
        :param mock_is_pro: TODO
        """
        mock_is_pro.return_value = False
        inst = MagicMock()
        # Response dict lacks "prices" key
        inst.get_coin_market_chart_range_by_id.side_effect = [
            {"market_caps": []},
            {"prices": [[i, 200 + i] for i in range(10)]},
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
    def test_none_response_returns_none(
        self, mock_cg: MagicMock, mock_is_pro: MagicMock
    ) -> None:
        """Test that None response (TypeError on key access) returns None.

        :param mock_cg: TODO
        :param mock_is_pro: TODO
        """
        mock_is_pro.return_value = False
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = [
            None,
            {"prices": [[i, 200 + i] for i in range(10)]},
        ]
        mock_cg.return_value = inst
        result = calculate_il_risk_score("t0", "t1", "key")
        assert result is None


class TestGetCachedPrice:
    """Tests for get_cached_price function."""

    def test_cache_miss_returns_none(self) -> None:
        """Test that missing cache key returns None."""
        assert get_cached_price("token", 90, {}, 1800) is None

    def test_cache_hit_returns_data(self) -> None:
        """Test that valid (non-expired) cache entry returns data."""
        cache: Dict[Any, Any] = {}
        set_cached_price("token", 90, {"prices": [[0, 100]]}, cache)
        result = get_cached_price("token", 90, cache, 1800)
        assert result == {"prices": [[0, 100]]}

    def test_cache_expired_returns_none(self) -> None:
        """Test that expired cache entry returns None."""
        cache = {
            "il_range_token_90": {
                "data": {"prices": [[0, 100]]},
                "timestamp": time.time() - 3600,
            }
        }
        result = get_cached_price("token", 90, cache, 1800)
        assert result is None


class TestCalculateIlRiskScoreWithCache:
    """Tests for calculate_il_risk_score with pre-populated cache."""

    def test_both_tokens_cached(self) -> None:
        """Test that cached data is used for both tokens without calling API."""
        prices_t0 = {"prices": [[i, 100 + i * 0.5] for i in range(100)]}
        prices_t1 = {"prices": [[i, 200 + i * 0.3] for i in range(100)]}
        cache: Dict[Any, Any] = {}
        set_cached_price("t0", 90, prices_t0, cache)
        set_cached_price("t1", 90, prices_t1, cache)
        result = calculate_il_risk_score("t0", "t1", "key", price_cache=cache)
        assert isinstance(result, float)

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_one_token_cached_one_fetched(
        self, mock_cg: MagicMock, mock_is_pro: MagicMock
    ) -> None:
        """Test partial cache: one token cached, other fetched from API.

        :param mock_cg: TODO
        :param mock_is_pro: TODO
        """
        mock_is_pro.return_value = False
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.return_value = {
            "prices": [[i, 200 + i * 0.3] for i in range(100)]
        }
        mock_cg.return_value = inst

        cache: Dict[Any, Any] = {}
        set_cached_price(
            "t0", 90, {"prices": [[i, 100 + i * 0.5] for i in range(100)]}, cache
        )
        result = calculate_il_risk_score("t0", "t1", "key", price_cache=cache)
        assert isinstance(result, float)
        # Only t1 should have been fetched
        assert inst.get_coin_market_chart_range_by_id.call_count == 1

    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.uniswap_pools_search.uniswap_pools_search.CoinGeckoAPI"
    )
    def test_second_token_cached_first_fetched(
        self, mock_cg: MagicMock, mock_is_pro: MagicMock
    ) -> None:
        """Test partial cache: second token cached, first fetched from API.

        :param mock_cg: TODO
        :param mock_is_pro: TODO
        """
        mock_is_pro.return_value = False
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.return_value = {
            "prices": [[i, 100 + i * 0.5] for i in range(100)]
        }
        mock_cg.return_value = inst

        cache: Dict[Any, Any] = {}
        set_cached_price(
            "t1", 90, {"prices": [[i, 200 + i * 0.3] for i in range(100)]}, cache
        )
        result = calculate_il_risk_score("t0", "t1", "key", price_cache=cache)
        assert isinstance(result, float)
        # Only t0 should have been fetched
        assert inst.get_coin_market_chart_range_by_id.call_count == 1

    def test_price_cache_none_defaults_to_empty_dict(self) -> None:
        """Test that passing price_cache=None does not crash."""
        result = calculate_il_risk_score("t0", "t1", "", price_cache=None)
        assert result is None


class TestRunWithPriceCache:
    """Tests for run() with explicit price_cache dict."""

    def test_run_with_explicit_price_cache(self) -> None:
        """Test that run() accepts a non-None price_cache without error."""
        result = run(price_cache={}, price_cache_ttl=600)
        assert "error" in result

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

"""Tests for balancer_pools_search custom component."""

import json
import statistics
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

import packages.valory.customs.balancer_pools_search.balancer_pools_search as bal_mod
from packages.valory.customs.balancer_pools_search.balancer_pools_search import (
    EXCLUDED_APR_TYPES,
    REQUIRED_FIELDS,
    SUPPORTED_POOL_TYPES,
    analyze_pool_liquidity,
    apply_composite_pre_filter,
    calculate_differential_investment,
    calculate_il_impact_multi,
    calculate_il_risk_score_multi,
    calculate_metrics,
    calculate_single_pool_investment,
    check_missing_fields,
    create_graphql_client,
    create_pool_snapshots_query,
    create_web3_connection,
    fetch_liquidity_metrics,
    fetch_token_name_from_contract,
    filter_valid_investment_pools,
    format_pool_data,
    get_balancer_pool_sharpe_ratio,
    get_balancer_pools,
    get_cached_price,
    get_coin_id_from_symbol,
    get_errors,
    get_filtered_pools_for_balancer,
    get_opportunities_for_balancer,
    get_pool_token_prices,
    get_token_investments_multi,
    get_total_apr,
    get_underlying_token_symbol,
    is_pro_api_key,
    normalize_token_symbol,
    remove_irrelevant_fields,
    run,
    run_query,
    set_cached_price,
    standardize_metrics,
    _reset_x402_adapter,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset module state before each test."""
    if hasattr(bal_mod._thread_local, "errors"):
        bal_mod._thread_local.errors = []
    bal_mod.create_web3_connection.cache_clear()
    bal_mod.fetch_token_name_from_contract.cache_clear()
    yield
    if hasattr(bal_mod._thread_local, "errors"):
        bal_mod._thread_local.errors = []


class TestGetErrors:
    """Tests for get_errors function."""

    def test_initializes_empty(self):
        """Test errors list initialization."""
        if hasattr(bal_mod._thread_local, "errors"):
            delattr(bal_mod._thread_local, "errors")
        assert get_errors() == []

    def test_returns_existing(self):
        """Test returning existing errors list."""
        bal_mod._thread_local.errors = ["err1"]
        assert get_errors() == ["err1"]


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
        """Test removing extra fields."""
        result = remove_irrelevant_fields({"a": 1, "b": 2}, ("a",))
        assert result == {"a": 1}


class TestRunQuery:
    """Tests for run_query function."""

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_successful(self, mock_post):
        """Test successful query."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"pools": []}}
        mock_post.return_value = mock_resp
        result = run_query("query", "url")
        assert result == {"pools": []}

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_error_status(self, mock_post):
        """Test error status code."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp
        result = run_query("query", "url")
        assert "error" in result

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_graphql_errors(self, mock_post):
        """Test GraphQL errors."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"errors": [{"message": "bad"}]}
        mock_post.return_value = mock_resp
        result = run_query("query", "url")
        assert "error" in result


class TestGetTotalApr:
    """Tests for get_total_apr function."""

    def test_normal(self):
        """Test normal APR calculation."""
        pool = {
            "dynamicData": {
                "aprItems": [
                    {"type": "REWARD", "apr": 0.05},
                    {"type": "SWAP_FEE", "apr": 0.02},
                ]
            }
        }
        result = get_total_apr(pool)
        assert result == 0.05  # SWAP_FEE is excluded

    def test_empty_apr_items(self):
        """Test with empty APR items."""
        pool = {"dynamicData": {"aprItems": []}}
        assert get_total_apr(pool) == 0

    def test_missing_dynamic_data(self):
        """Test with missing dynamicData."""
        assert get_total_apr({}) == 0


class TestStandardizeMetrics:
    """Tests for standardize_metrics function."""

    def test_empty_pools(self):
        """Test with empty pools returns early."""
        assert standardize_metrics([]) == []

    def test_single_pool(self):
        """Test with single pool (triggers apr_std==0 and tvl_std==0)."""
        pools = [{"apr": 10, "tvl": 1000}]
        result = standardize_metrics(pools)
        assert "composite_score" in result[0]

    def test_invalid_values(self):
        """Test with invalid values triggers ValueError/TypeError branch."""
        pools = [{"apr": "bad", "tvl": "bad"}]
        result = standardize_metrics(pools)
        assert "composite_score" in result[0]

    def test_multiple_pools(self):
        """Test with multiple pools to get non-zero std deviations."""
        pools = [
            {"apr": 10, "tvl": 1000},
            {"apr": 20, "tvl": 5000},
            {"apr": 30, "tvl": 10000},
        ]
        result = standardize_metrics(pools)
        assert len(result) == 3
        assert all("composite_score" in p for p in result)
        # With multiple different values, std should not be zero
        # so the real standardization path is taken

    def test_all_same_values(self):
        """Test pools with identical values (std == 0 both branches)."""
        pools = [
            {"apr": 10, "tvl": 1000},
            {"apr": 10, "tvl": 1000},
        ]
        result = standardize_metrics(pools)
        assert len(result) == 2

    def test_mixed_valid_invalid(self):
        """Test mix of valid and invalid pool values."""
        pools = [
            {"apr": 10, "tvl": 1000},
            {"apr": "invalid", "tvl": None},
        ]
        result = standardize_metrics(pools)
        assert len(result) == 2
        assert "composite_score" in result[0]
        assert "composite_score" in result[1]


class TestApplyCompositePreFilter:
    """Tests for apply_composite_pre_filter function."""

    def test_empty(self):
        """Test with empty pools."""
        assert apply_composite_pre_filter([]) == []

    def test_disabled(self):
        """Test disabled filter returns pools[:top_n]."""
        pools = [{"tvl": 5000, "apr": 10}]
        result = apply_composite_pre_filter(pools, use_composite_filter=False)
        assert len(result) == 1

    def test_disabled_empty(self):
        """Test disabled filter with empty returns []."""
        assert apply_composite_pre_filter([], use_composite_filter=False) == []

    def test_below_tvl(self):
        """Test all pools below TVL threshold returns empty."""
        pools = [{"tvl": 100, "apr": 10}]
        result = apply_composite_pre_filter(pools, min_tvl_threshold=1000)
        assert result == []

    def test_invalid_tvl(self):
        """Test invalid TVL value (ValueError/TypeError branch)."""
        pools = [{"tvl": "invalid", "apr": 10}]
        result = apply_composite_pre_filter(pools, min_tvl_threshold=0)
        assert result == []

    def test_successful_filtering(self):
        """Test successful composite pre-filtering with valid pools above threshold."""
        pools = [
            {"tvl": 5000, "apr": 10},
            {"tvl": 8000, "apr": 20},
            {"tvl": 3000, "apr": 5},
        ]
        result = apply_composite_pre_filter(pools, top_n=2, min_tvl_threshold=1000)
        assert len(result) == 2
        # Should be sorted by composite score descending
        assert result[0].get("composite_score", 0) >= result[1].get(
            "composite_score", 0
        )

    def test_top_n_limit(self):
        """Test top_n limits output count."""
        pools = [{"tvl": 5000, "apr": 10 + i} for i in range(5)]
        result = apply_composite_pre_filter(pools, top_n=3, min_tvl_threshold=1000)
        assert len(result) == 3

    def test_none_pools(self):
        """Test with not pools (falsy) returns empty."""
        assert apply_composite_pre_filter(None) == []


class TestCreateWeb3Connection:
    """Tests for create_web3_connection function."""

    def test_unknown_chain(self):
        """Test with unknown chain returns None."""
        result = create_web3_connection("unknown_chain_xyz")
        assert result is None

    @patch("packages.valory.customs.balancer_pools_search.balancer_pools_search.Web3")
    def test_known_chain_connected(self, mock_web3):
        """Test with known chain that connects successfully."""
        mock_instance = MagicMock()
        mock_instance.is_connected.return_value = True
        mock_web3.return_value = mock_instance
        mock_web3.HTTPProvider = MagicMock()
        bal_mod.create_web3_connection.cache_clear()
        result = create_web3_connection("optimism")
        assert result == mock_instance

    @patch("packages.valory.customs.balancer_pools_search.balancer_pools_search.Web3")
    def test_known_chain_not_connected(self, mock_web3):
        """Test with known chain that fails to connect returns None."""
        mock_instance = MagicMock()
        mock_instance.is_connected.return_value = False
        mock_web3.return_value = mock_instance
        mock_web3.HTTPProvider = MagicMock()
        bal_mod.create_web3_connection.cache_clear()
        result = create_web3_connection("optimism")
        assert result is None


class TestFetchTokenNameFromContract:
    """Tests for fetch_token_name_from_contract function."""

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.create_web3_connection"
    )
    def test_no_web3(self, mock_conn):
        """Test returns None when web3 connection fails."""
        mock_conn.return_value = None
        bal_mod.fetch_token_name_from_contract.cache_clear()
        result = fetch_token_name_from_contract("optimism", "0x1234")
        assert result is None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.create_web3_connection"
    )
    def test_successful(self, mock_conn):
        """Test successful token name fetch."""
        mock_web3 = MagicMock()
        mock_contract = MagicMock()
        mock_contract.functions.name.return_value.call.return_value = "TestToken"
        mock_web3.eth.contract.return_value = mock_contract
        mock_conn.return_value = mock_web3
        bal_mod.fetch_token_name_from_contract.cache_clear()
        result = fetch_token_name_from_contract(
            "optimism", "0x1234567890abcdef1234567890abcdef12345678"
        )
        assert result == "TestToken"

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.create_web3_connection"
    )
    def test_contract_exception(self, mock_conn):
        """Test returns None on contract call exception."""
        mock_web3 = MagicMock()
        mock_contract = MagicMock()
        mock_contract.functions.name.return_value.call.side_effect = Exception("fail")
        mock_web3.eth.contract.return_value = mock_contract
        mock_conn.return_value = mock_web3
        bal_mod.fetch_token_name_from_contract.cache_clear()
        result = fetch_token_name_from_contract(
            "optimism", "0x1234567890abcdef1234567890abcdef12345678"
        )
        assert result is None


class TestGetBalancerPools:
    """Tests for get_balancer_pools function."""

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.run_query"
    )
    def test_success(self, mock_query):
        """Test successful pool fetch."""
        mock_query.return_value = {"poolGetPools": [{"id": "1"}]}
        result = get_balancer_pools(["optimism"], "https://api.example.com")
        assert len(result) == 1

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.run_query"
    )
    def test_error(self, mock_query):
        """Test error in query."""
        mock_query.return_value = {"error": "fail"}
        result = get_balancer_pools(["optimism"], "https://api.example.com")
        assert "error" in result


class TestGetFilteredPoolsForBalancer:
    """Tests for get_filtered_pools_for_balancer function."""

    def _make_pool(
        self,
        pool_type="WEIGHTED",
        address="0x1234567890abcdef1234567890abcdef12345678",
        chain="OPTIMISM",
    ):
        """Create a test pool."""
        return {
            "type": pool_type,
            "address": address,
            "chain": chain,
            "poolTokens": [
                {"address": "0xtoken0", "symbol": "TK0"},
                {"address": "0xtoken1", "symbol": "TK1"},
            ],
            "dynamicData": {
                "totalLiquidity": 100000,
                "aprItems": [{"type": "REWARD", "apr": 0.1}],
            },
        }

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.apply_composite_pre_filter"
    )
    def test_basic_filtering(self, mock_filter):
        """Test basic filtering."""
        mock_filter.side_effect = lambda x, **kw: x
        pools = [self._make_pool()]
        result = get_filtered_pools_for_balancer(pools, [], {"optimism": {}})
        assert len(result) == 1

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.apply_composite_pre_filter"
    )
    def test_unsupported_type(self, mock_filter):
        """Test unsupported pool type."""
        mock_filter.side_effect = lambda x, **kw: x
        pools = [self._make_pool(pool_type="UNKNOWN")]
        result = get_filtered_pools_for_balancer(pools, [], {"optimism": {}})
        assert len(result) == 0

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.apply_composite_pre_filter"
    )
    def test_excludes_current_positions(self, mock_filter):
        """Test excluding current positions."""
        mock_filter.side_effect = lambda x, **kw: x
        from web3 import Web3

        addr = "0x1234567890abcdef1234567890abcdef12345678"
        pools = [self._make_pool(address=addr)]
        result = get_filtered_pools_for_balancer(
            pools, [Web3.to_checksum_address(addr)], {"optimism": {}}
        )
        assert len(result) == 0

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.apply_composite_pre_filter"
    )
    def test_chain_not_in_whitelisted(self, mock_filter):
        """Test pool chain not in whitelisted_assets."""
        mock_filter.side_effect = lambda x, **kw: x
        pools = [self._make_pool(chain="UNKNOWN_CHAIN")]
        result = get_filtered_pools_for_balancer(pools, [], {"optimism": {}})
        assert len(result) == 0

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.apply_composite_pre_filter"
    )
    def test_whitelisted_tokens_filtering(self, mock_filter):
        """Test filtering with whitelisted tokens where tokens match."""
        mock_filter.side_effect = lambda x, **kw: x
        pools = [self._make_pool()]
        # Whitelisted with specific tokens that match
        result = get_filtered_pools_for_balancer(
            pools, [], {"optimism": {"0xtoken0": True, "0xtoken1": True}}
        )
        assert len(result) == 1

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.apply_composite_pre_filter"
    )
    def test_whitelisted_tokens_mismatch(self, mock_filter):
        """Test filtering with whitelisted tokens where tokens don't match."""
        mock_filter.side_effect = lambda x, **kw: x
        pools = [self._make_pool()]
        # Whitelisted with specific tokens that DON'T match pool tokens
        result = get_filtered_pools_for_balancer(
            pools, [], {"optimism": {"0xother0": True, "0xother1": True}}
        )
        assert len(result) == 0

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.apply_composite_pre_filter"
    )
    def test_kwargs_forwarded(self, mock_filter):
        """Test that kwargs are forwarded to apply_composite_pre_filter."""
        mock_filter.side_effect = lambda x, **kw: x
        pools = [self._make_pool()]
        get_filtered_pools_for_balancer(
            pools,
            [],
            {"optimism": {}},
            top_n=5,
            apr_weight=0.8,
            tvl_weight=0.2,
            min_tvl_threshold=500,
        )
        call_kwargs = mock_filter.call_args[1]
        assert call_kwargs["top_n"] == 5


class TestGetCoinIdFromSymbol:
    """Tests for get_coin_id_from_symbol function."""

    def test_found(self):
        """Test found coin ID."""
        assert (
            get_coin_id_from_symbol({"optimism": {"weth": "id"}}, "WETH", "optimism")
            == "id"
        )

    def test_not_found(self):
        """Test not found."""
        assert get_coin_id_from_symbol({}, "weth", "optimism") is None


class TestCalculateIlImpactMulti:
    """Tests for calculate_il_impact_multi function."""

    def test_empty_prices(self):
        """Test with empty price lists."""
        assert calculate_il_impact_multi([], []) == 0

    def test_length_mismatch(self):
        """Test with mismatched lengths."""
        assert calculate_il_impact_multi([1, 2], [1]) == 0

    def test_equal_prices(self):
        """Test with equal prices gives zero IL."""
        result = calculate_il_impact_multi([1.0, 1.0], [1.0, 1.0])
        assert abs(result) < 1e-10

    def test_with_weights(self):
        """Test with custom weights."""
        result = calculate_il_impact_multi([1.0, 1.0], [2.0, 0.5], weights=[0.5, 0.5])
        assert isinstance(result, float)

    def test_wrong_weights_length(self):
        """Test with wrong weights length returns 0."""
        result = calculate_il_impact_multi([1.0, 1.0], [2.0, 0.5], weights=[0.5])
        assert result == 0

    def test_no_initial_prices(self):
        """Test with no initial prices."""
        assert calculate_il_impact_multi([], [1.0]) == 0

    def test_no_final_prices(self):
        """Test with no final prices."""
        assert calculate_il_impact_multi([1.0], []) == 0

    def test_default_equal_weights(self):
        """Test default equal weights path with 3 tokens."""
        result = calculate_il_impact_multi([1.0, 1.0, 1.0], [2.0, 0.5, 1.5])
        assert isinstance(result, float)


class TestCalculateIlRiskScoreMulti:
    """Tests for calculate_il_risk_score_multi function."""

    def test_insufficient_token_ids(self):
        """Test with less than 2 valid token IDs."""
        assert calculate_il_risk_score_multi(["id1", None], "key") is None

    def test_only_none_ids(self):
        """Test with all None token IDs."""
        assert calculate_il_risk_score_multi([None, None], "key") is None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_successful(self, mock_cg, mock_pro, mock_sleep):
        """Test successful calculation with demo key (is_pro=False)."""
        mock_pro.return_value = False
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = [
            {"prices": [[i, 100 + i * 0.5] for i in range(50)]},
            {"prices": [[i, 200 + i * 0.3] for i in range(50)]},
        ]
        mock_cg.return_value = inst
        result = calculate_il_risk_score_multi(["t0", "t1"], "key")
        assert isinstance(result, float)

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_successful_pro_key(self, mock_cg, mock_pro, mock_sleep):
        """Test successful calculation with pro API key (is_pro=True)."""
        mock_pro.return_value = True
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = [
            {"prices": [[i, 100 + i * 0.5] for i in range(50)]},
            {"prices": [[i, 200 + i * 0.3] for i in range(50)]},
        ]
        mock_cg.return_value = inst
        result = calculate_il_risk_score_multi(["t0", "t1"], "key")
        assert isinstance(result, float)

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_with_x402(self, mock_cg, mock_sleep):
        """Test with x402 session."""
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = [
            {"prices": [[i, 100 + i] for i in range(10)]},
            {"prices": [[i, 200 + i] for i in range(10)]},
        ]
        mock_cg.return_value = inst
        result = calculate_il_risk_score_multi(
            ["t0", "t1"], "key", x402_session=MagicMock(), x402_proxy="https://p.com"
        )
        assert isinstance(result, float)

    def test_no_api_key_no_x402(self):
        """Test with no API key and no x402."""
        result = calculate_il_risk_score_multi(["t0", "t1"], "")
        assert result is None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_api_exception(self, mock_cg, mock_sleep):
        """Test inner API exception (per-token)."""
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = Exception("fail")
        mock_cg.return_value = inst
        result = calculate_il_risk_score_multi(
            ["t0", "t1"], "", x402_session=MagicMock(), x402_proxy="https://p.com"
        )
        assert result is None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_insufficient_data(self, mock_cg, mock_pro, mock_sleep):
        """Test with insufficient price data (min_length < 2)."""
        mock_pro.return_value = False
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = [
            {"prices": [[1, 100]]},
            {"prices": [[1, 200]]},
        ]
        mock_cg.return_value = inst
        result = calculate_il_risk_score_multi(["t0", "t1"], "key")
        assert result is None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_outer_exception(self, mock_cg, mock_pro, mock_sleep):
        """Test outer exception handling (line 447-449).

        This triggers the outer except by making np.corrcoef raise after
        the per-token loop completes successfully.
        """
        mock_pro.return_value = False
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = [
            {"prices": [[i, 100 + i * 0.5] for i in range(50)]},
            {"prices": [[i, 200 + i * 0.3] for i in range(50)]},
        ]
        mock_cg.return_value = inst
        with patch(
            "packages.valory.customs.balancer_pools_search.balancer_pools_search.np.corrcoef",
            side_effect=Exception("corrcoef fail"),
        ):
            result = calculate_il_risk_score_multi(["t0", "t1"], "key")
            assert result is None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_inner_exception_cg_constructor(self, mock_cg, mock_pro, mock_sleep):
        """Test inner exception when CoinGeckoAPI constructor raises."""
        mock_pro.return_value = False
        mock_cg.side_effect = Exception("constructor fail")
        result = calculate_il_risk_score_multi(["t0", "t1"], "key")
        assert result is None


class TestCreateGraphqlClient:
    """Tests for create_graphql_client function."""

    @patch("packages.valory.customs.balancer_pools_search.balancer_pools_search.Client")
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.RequestsHTTPTransport"
    )
    def test_creates_client(self, mock_transport, mock_client):
        """Test client creation."""
        result = create_graphql_client()
        mock_transport.assert_called_once()
        mock_client.assert_called_once()


class TestCreatePoolSnapshotsQuery:
    """Tests for create_pool_snapshots_query function."""

    def test_creates_query(self):
        """Test query creation."""
        result = create_pool_snapshots_query("pool1", "OPTIMISM")
        assert result is not None


class TestFetchLiquidityMetrics:
    """Tests for fetch_liquidity_metrics function (the second/overriding definition)."""

    def test_no_snapshots(self):
        """Test with no snapshots returns None."""
        mock_client = MagicMock()
        mock_client.execute.return_value = {"poolGetSnapshots": []}
        result = fetch_liquidity_metrics("pool1", "OPTIMISM", client=mock_client)
        assert result is None

    def test_successful(self):
        """Test successful metric computation."""
        mock_client = MagicMock()
        mock_client.execute.return_value = {
            "poolGetSnapshots": [
                {"totalLiquidity": "100000", "volume24h": "5000"},
                {"totalLiquidity": "120000", "volume24h": "6000"},
            ]
        }
        result = fetch_liquidity_metrics("pool1", "OPTIMISM", client=mock_client)
        assert result is not None
        assert "Depth Score" in result

    def test_extreme_values_filtered(self):
        """Test that extreme values are filtered out."""
        mock_client = MagicMock()
        mock_client.execute.return_value = {
            "poolGetSnapshots": [
                {"totalLiquidity": "1e17", "volume24h": "5000"},
            ]
        }
        result = fetch_liquidity_metrics("pool1", "OPTIMISM", client=mock_client)
        assert result is None

    def test_invalid_snapshot_values(self):
        """Test with invalid snapshot values (ValueError/TypeError)."""
        mock_client = MagicMock()
        mock_client.execute.return_value = {
            "poolGetSnapshots": [
                {"totalLiquidity": "invalid", "volume24h": "5000"},
            ]
        }
        result = fetch_liquidity_metrics("pool1", "OPTIMISM", client=mock_client)
        assert result is None

    def test_exception(self):
        """Test outer exception handling."""
        mock_client = MagicMock()
        mock_client.execute.side_effect = Exception("network error")
        result = fetch_liquidity_metrics("pool1", "OPTIMISM", client=mock_client)
        assert result is None

    def test_small_price_impact(self):
        """Test with very small price impact (below 0.001 threshold)."""
        mock_client = MagicMock()
        mock_client.execute.return_value = {
            "poolGetSnapshots": [
                {"totalLiquidity": "100000", "volume24h": "5000"},
            ]
        }
        result = fetch_liquidity_metrics(
            "pool1", "OPTIMISM", client=mock_client, price_impact=0.0001
        )
        assert result is not None

    def test_zero_avg_tvl_and_volume(self):
        """Test with zero TVL and volume (depth_score == 0 branch)."""
        mock_client = MagicMock()
        mock_client.execute.return_value = {
            "poolGetSnapshots": [
                {"totalLiquidity": "0", "volume24h": "0"},
            ]
        }
        result = fetch_liquidity_metrics("pool1", "OPTIMISM", client=mock_client)
        assert result is not None
        assert result["Depth Score"] == 0
        assert result["Liquidity Risk Multiplier"] == 0

    def test_no_client_creates_one(self):
        """Test that None client triggers create_graphql_client."""
        with patch(
            "packages.valory.customs.balancer_pools_search.balancer_pools_search.create_graphql_client"
        ) as mock_create:
            mock_client = MagicMock()
            mock_client.execute.return_value = {"poolGetSnapshots": []}
            mock_create.return_value = mock_client
            result = fetch_liquidity_metrics("pool1", "OPTIMISM", client=None)
            mock_create.assert_called_once()
            assert result is None

    def test_missing_poolGetSnapshots_key(self):
        """Test with missing poolGetSnapshots key returns empty list default."""
        mock_client = MagicMock()
        mock_client.execute.return_value = {}
        result = fetch_liquidity_metrics("pool1", "OPTIMISM", client=mock_client)
        assert result is None

    def test_statistics_error(self):
        """Test statistics.StatisticsError branch (lines 589-591).

        When filtered_snapshots are valid but statistics.mean raises.
        """
        mock_client = MagicMock()
        # Valid snapshots that pass the extreme value filter
        mock_client.execute.return_value = {
            "poolGetSnapshots": [
                {"totalLiquidity": "100000", "volume24h": "5000"},
            ]
        }
        # Monkey-patch statistics.mean to raise StatisticsError
        with patch(
            "packages.valory.customs.balancer_pools_search.balancer_pools_search.statistics.mean",
            side_effect=statistics.StatisticsError("no data"),
        ):
            result = fetch_liquidity_metrics("pool1", "OPTIMISM", client=mock_client)
            assert result is None

    def test_depth_score_calculation_error(self):
        """Test ZeroDivisionError/OverflowError/ValueError in depth calculation (lines 627-630).

        Note: This except block catches ZeroDivisionError, OverflowError, ValueError
        during the depth score / liquidity risk multiplier / max position calculation.
        These are very hard to trigger naturally since the code uses max() to prevent
        negative values and has other guards. We force it by making min() raise.
        """
        mock_client = MagicMock()
        mock_client.execute.return_value = {
            "poolGetSnapshots": [
                {"totalLiquidity": "100000", "volume24h": "5000"},
            ]
        }
        # Patch the built-in min to raise ValueError when called with the specific
        # arguments used in depth_score capping
        original_min = min
        call_count = [0]

        def bad_min(*args, **kwargs):
            call_count[0] += 1
            # The first min calls are from filtered_snapshots and mean calcs;
            # We want to fail during depth_score = min(depth_score, 1e6) which is
            # the 1st or 2nd call to min in the try block
            if call_count[0] >= 3:
                raise ValueError("forced error")
            return original_min(*args, **kwargs)

        with patch("builtins.min", side_effect=bad_min):
            result = fetch_liquidity_metrics("pool1", "OPTIMISM", client=mock_client)
            assert result is None


class TestAnalyzePoolLiquidity:
    """Tests for analyze_pool_liquidity function."""

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.fetch_liquidity_metrics"
    )
    def test_no_metrics(self, mock_fetch):
        """Test when metrics are None."""
        mock_fetch.return_value = None
        depth, max_pos = analyze_pool_liquidity("pool1", "OPTIMISM")
        assert np.isnan(depth)

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.fetch_liquidity_metrics"
    )
    def test_with_metrics(self, mock_fetch):
        """Test with valid metrics."""
        mock_fetch.return_value = {"Depth Score": 100, "Maximum Position Size": 5000}
        depth, max_pos = analyze_pool_liquidity("pool1", "OPTIMISM")
        assert depth == 100

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.fetch_liquidity_metrics"
    )
    def test_exception(self, mock_fetch):
        """Test exception handling."""
        mock_fetch.side_effect = Exception("fail")
        depth, max_pos = analyze_pool_liquidity("pool1", "OPTIMISM")
        assert np.isnan(depth)


class TestGetBalancerPoolSharpeRatio:
    """Tests for get_balancer_pool_sharpe_ratio function."""

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_no_data(self, mock_post):
        """Test with no data returns None."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"poolGetSnapshots": []}}
        mock_post.return_value = mock_resp
        result = get_balancer_pool_sharpe_ratio("pool1", "OPTIMISM")
        assert result is None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_successful(self, mock_post):
        """Test successful Sharpe ratio calculation returns capped value."""
        data = [
            {
                "timestamp": 1700000000 + i * 86400,
                "sharePrice": 1.0 + i * 0.001,
                "fees24h": 100 + i,
                "totalLiquidity": 100000,
            }
            for i in range(30)
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"poolGetSnapshots": data}}
        mock_post.return_value = mock_resp
        result = get_balancer_pool_sharpe_ratio("pool1", "OPTIMISM")
        assert result is not None
        assert -10 <= result <= 10

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_exception(self, mock_post):
        """Test outer exception handling."""
        mock_post.side_effect = Exception("fail")
        result = get_balancer_pool_sharpe_ratio("pool1", "OPTIMISM")
        assert result is None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_insufficient_returns(self, mock_post):
        """Test with less than 5 valid returns."""
        data = [
            {
                "timestamp": 1700000000 + i * 86400,
                "sharePrice": str(1.0),
                "fees24h": str(0),
                "totalLiquidity": str(100000),
            }
            for i in range(3)
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"poolGetSnapshots": data}}
        mock_post.return_value = mock_resp
        result = get_balancer_pool_sharpe_ratio("pool1", "OPTIMISM")
        assert result is None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_extreme_values_filtered(self, mock_post):
        """Test extreme values are replaced with NaN and warning is logged."""
        data = []
        for i in range(30):
            sp = 1e17 if i == 0 else 1.0 + i * 0.001
            data.append(
                {
                    "timestamp": 1700000000 + i * 86400,
                    "sharePrice": sp,
                    "fees24h": 100 + i,
                    "totalLiquidity": 100000,
                }
            )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"poolGetSnapshots": data}}
        mock_post.return_value = mock_resp
        result = get_balancer_pool_sharpe_ratio("pool1", "OPTIMISM")
        # Should still compute a result with remaining valid data
        assert result is not None
        assert -10 <= result <= 10

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_nan_sharpe_ratio(self, mock_post):
        """Test NaN/Inf Sharpe ratio returns None."""
        # All identical share prices with no fees -> std=0 -> NaN Sharpe
        data = [
            {
                "timestamp": 1700000000 + i * 86400,
                "sharePrice": str(1.0),
                "fees24h": str(0),
                "totalLiquidity": str(0),
            }
            for i in range(30)
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"poolGetSnapshots": data}}
        mock_post.return_value = mock_resp
        result = get_balancer_pool_sharpe_ratio("pool1", "OPTIMISM")
        assert result is None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_inner_exception(self, mock_post):
        """Test inner processing exception."""
        mock_resp = MagicMock()
        # Return data that will cause processing errors
        mock_resp.json.return_value = {"data": {"poolGetSnapshots": [{"bad": "data"}]}}
        mock_post.return_value = mock_resp
        result = get_balancer_pool_sharpe_ratio("pool1", "OPTIMISM")
        assert result is None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_infinite_returns_replaced(self, mock_post):
        """Test that infinite values in returns are handled and warning is logged."""
        data = []
        for i in range(30):
            sp = (
                1.0 + i * 0.001 if i != 5 else 0.0
            )  # Zero sharePrice creates inf pct_change
            data.append(
                {
                    "timestamp": 1700000000 + i * 86400,
                    "sharePrice": sp,
                    "fees24h": 100,
                    "totalLiquidity": 100000,
                }
            )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"poolGetSnapshots": data}}
        mock_post.return_value = mock_resp
        result = get_balancer_pool_sharpe_ratio("pool1", "OPTIMISM")
        # Should handle inf values and still return a result
        assert result is not None
        assert -10 <= result <= 10

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_missing_column(self, mock_post):
        """Test with data where fees24h is missing (line 693->692 branch: col not in df.columns).

        When fees24h column doesn't exist, the `if col in df.columns` check is False
        for that column, causing the loop to continue without processing it.
        """
        # Use numeric timestamps so pd.to_datetime(unit='s') works
        # But omit fees24h to trigger the col-not-in-df branch
        data = [
            {
                "timestamp": 1700000000 + i * 86400,
                "sharePrice": 1.0 + i * 0.001,
                "totalLiquidity": 100000,
            }
            for i in range(30)
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"poolGetSnapshots": data}}
        mock_post.return_value = mock_resp
        result = get_balancer_pool_sharpe_ratio("pool1", "OPTIMISM")
        # Should handle missing column and still produce result
        assert result is None or isinstance(result, (int, float))

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_sharpe_capped_at_bounds(self, mock_post):
        """Test Sharpe ratio is capped between -10 and 10."""
        # Create data with consistently increasing prices -> high positive Sharpe
        data = [
            {
                "timestamp": 1700000000 + i * 86400,
                "sharePrice": 1.0 + i * 0.1,  # Large consistent increases
                "fees24h": 1000,
                "totalLiquidity": 100000,
            }
            for i in range(30)
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"poolGetSnapshots": data}}
        mock_post.return_value = mock_resp
        result = get_balancer_pool_sharpe_ratio("pool1", "OPTIMISM")
        assert result is not None
        assert -10 <= result <= 10


class TestGetUnderlyingTokenSymbol:
    """Tests for get_underlying_token_symbol function."""

    def test_known_mapping(self):
        """Test known token mapping."""
        assert get_underlying_token_symbol("weth") == "ethereum"
        assert get_underlying_token_symbol("WETH") == "ethereum"

    def test_unknown_symbol(self):
        """Test unknown symbol returns itself."""
        assert get_underlying_token_symbol("unknowntoken") == "unknowntoken"

    def test_additional_mappings(self):
        """Test additional known mappings."""
        assert get_underlying_token_symbol("csusdc") == "usdc"
        assert get_underlying_token_symbol("wsteth") == "steth"
        assert get_underlying_token_symbol("ausdc") == "usdc"
        assert get_underlying_token_symbol("dai") == "dai"


class TestNormalizeTokenSymbol:
    """Tests for normalize_token_symbol function."""

    def test_removes_prefix(self):
        """Test removing known prefixes."""
        assert normalize_token_symbol("wETH") == "ETH"

    def test_no_prefix(self):
        """Test with no known prefix returns as-is."""
        # 'ETH' starts with no matching prefix in lowercase
        # Actually 'e' is not in prefixes list
        result = normalize_token_symbol("ETH")
        assert result == "ETH"

    def test_various_prefixes(self):
        """Test various prefix removals."""
        assert normalize_token_symbol("csUSDC") == "USDC"
        assert normalize_token_symbol("waETH") == "ETH"
        assert normalize_token_symbol("aDAI") == "DAI"
        assert normalize_token_symbol("cDAI") == "DAI"
        assert normalize_token_symbol("vTOKEN") == "TOKEN"
        assert normalize_token_symbol("xSUSHI") == "SUSHI"


class TestGetPoolTokenPrices:
    """Tests for get_pool_token_prices function."""

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_with_x402_successful(self, mock_cg, mock_pro, mock_sleep):
        """Test price fetching with x402 session."""
        inst = MagicMock()
        inst.get_price.return_value = {"weth": {"usd": 2000.0}}
        mock_cg.return_value = inst
        result = get_pool_token_prices(
            ["WETH"], x402_session=MagicMock(), x402_proxy="https://proxy.com"
        )
        assert result is not None
        assert "WETH" in result

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_no_api_key_no_x402_returns_none(self, mock_cg, mock_pro, mock_sleep):
        """Test with no API key and no x402 returns None."""
        inst = MagicMock()
        mock_cg.return_value = inst
        result = get_pool_token_prices(["WETH"], coingecko_api_key="")
        assert result is None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_pro_key(self, mock_cg, mock_pro, mock_sleep):
        """Test with pro API key."""
        mock_pro.return_value = True
        inst = MagicMock()
        inst.get_price.return_value = {"weth": {"usd": 2000.0}}
        mock_cg.return_value = inst
        result = get_pool_token_prices(["WETH"], coingecko_api_key="prokey")
        # Note: first attempt gets 'ethereum' (from get_underlying_token_symbol) not 'weth'
        assert result is not None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_demo_key(self, mock_cg, mock_pro, mock_sleep):
        """Test with demo API key."""
        mock_pro.return_value = False
        inst = MagicMock()
        inst.get_price.return_value = {"ethereum": {"usd": 2000.0}}
        mock_cg.return_value = inst
        result = get_pool_token_prices(["WETH"], coingecko_api_key="demokey")
        assert result is not None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_price_not_found_usd_pegged(self, mock_cg, mock_pro, mock_sleep):
        """Test USD-pegged token gets price 1.0 when not found via API."""
        mock_pro.return_value = False
        inst = MagicMock()
        inst.get_price.return_value = {}  # No price found
        mock_cg.return_value = inst
        # The search fallback uses undefined coingecko_api -> NameError -> caught by except
        result = get_pool_token_prices(["USDC"], coingecko_api_key="key")
        assert result is not None
        assert result.get("USDC") == 1.0

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_price_not_found_no_usd(self, mock_cg, mock_pro, mock_sleep):
        """Test non-USD token with no price found gets 0.0."""
        mock_pro.return_value = False
        inst = MagicMock()
        inst.get_price.return_value = {}  # No price found
        mock_cg.return_value = inst
        result = get_pool_token_prices(["RANDOMTOKEN"], coingecko_api_key="key")
        assert result is not None
        assert result.get("RANDOMTOKEN") == 0.0

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_get_price_exception_tries_next(self, mock_cg, mock_pro, mock_sleep):
        """Test that exception in get_price tries next possible ID."""
        mock_pro.return_value = False
        inst = MagicMock()
        inst.get_price.side_effect = Exception("rate limit")
        mock_cg.return_value = inst
        result = get_pool_token_prices(["WETH"], coingecko_api_key="key")
        assert result is not None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_outer_exception(self, mock_cg, mock_sleep):
        """Test outer exception returns fallback dict with all prices 0.0."""
        # The first time.sleep(3) raises, entering the outer except
        mock_sleep.side_effect = Exception("outer fail")
        result = get_pool_token_prices(["WETH"], coingecko_api_key="key")
        assert result == {"WETH": 0.0}

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_price_found_break(self, mock_cg, mock_pro, mock_sleep):
        """Test that once price is found, inner loop breaks."""
        mock_pro.return_value = False
        inst = MagicMock()
        inst.get_price.return_value = {"weth": {"usd": 2000.0}}
        mock_cg.return_value = inst
        result = get_pool_token_prices(["WETH"], coingecko_api_key="key")
        assert result is not None
        assert "WETH" in result

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_inner_per_symbol_exception(self, mock_cg, mock_pro, mock_sleep):
        """Test per-symbol exception handler sets price to 0.0."""
        mock_pro.return_value = False
        inst = MagicMock()
        # get_price raises for both symbols; search also raises => per-symbol except triggered
        inst.get_price.side_effect = Exception("rate limit")
        inst.search.side_effect = Exception("search fail too")
        mock_cg.return_value = inst
        # Force the per-symbol except by raising at the symbol-processing level.
        # We need an exception that escapes the inner try blocks but is caught by the
        # per-symbol except. We can achieve this by making get_underlying_token_symbol raise.
        with patch(
            "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_underlying_token_symbol",
            side_effect=RuntimeError("forced"),
        ):
            result = get_pool_token_prices(["WETH", "DAI"], coingecko_api_key="key")
        assert result is not None
        assert result["WETH"] == 0.0
        assert result["DAI"] == 0.0

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_search_fallback_finds_price(self, mock_cg, mock_pro, mock_sleep):
        """Test the search fallback path when direct get_price fails."""
        mock_pro.return_value = False
        inst = MagicMock()
        # All direct get_price calls return empty (no match)
        inst.get_price.side_effect = [
            {},  # 1st possible ID
            {},  # 2nd possible ID
            {},  # 3rd possible ID
            {},  # 4th possible ID
            {},  # 5th possible ID
            {"ethereum": {"usd": 2500.0}},  # search fallback get_price call
        ]
        # search returns a coin
        inst.search.return_value = {"coins": [{"id": "ethereum"}]}
        mock_cg.return_value = inst
        result = get_pool_token_prices(["WETH"], coingecko_api_key="key")
        assert result is not None
        assert result["WETH"] == 2500.0

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_search_fallback_price_data_missing_coin_id(
        self, mock_cg, mock_pro, mock_sleep
    ):
        """Test search fallback when get_price returns data without the expected coin_id."""
        mock_pro.return_value = False
        inst = MagicMock()
        # All direct get_price calls return empty
        inst.get_price.return_value = {}
        # search returns a coin, but the subsequent get_price returns wrong key
        inst.search.return_value = {"coins": [{"id": "ethereum"}]}
        mock_cg.return_value = inst
        result = get_pool_token_prices(["RANDOMTOKEN"], coingecko_api_key="key")
        assert result is not None
        # Price should be 0.0 since coin_id not in price_data and token is not USD-pegged
        assert result["RANDOMTOKEN"] == 0.0

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_search_fallback_exception(self, mock_cg, mock_pro, mock_sleep):
        """Test search fallback when search raises an exception (except branch)."""
        mock_pro.return_value = False
        inst = MagicMock()
        # All direct get_price calls return empty
        inst.get_price.return_value = {}
        # search raises an exception
        inst.search.side_effect = Exception("search API error")
        mock_cg.return_value = inst
        result = get_pool_token_prices(["RANDOMTOKEN"], coingecko_api_key="key")
        assert result is not None
        # Price should be 0.0 since search failed and token is not USD-pegged
        assert result["RANDOMTOKEN"] == 0.0


class TestGetTokenInvestmentsMulti:
    """Tests for get_token_investments_multi function."""

    def test_zero_investment(self):
        """Test with zero investment."""
        assert get_token_investments_multi(0, {"TK0": 100}) == []

    def test_negative_investment(self):
        """Test with negative investment."""
        assert get_token_investments_multi(-100, {"TK0": 100}) == []

    def test_empty_prices(self):
        """Test with empty prices."""
        assert get_token_investments_multi(100, {}) == []

    def test_single_token(self):
        """Test with single token (less than 2)."""
        assert get_token_investments_multi(100, {"TK0": 50}) == []

    def test_both_zero_prices(self):
        """Test with both token prices zero."""
        assert get_token_investments_multi(100, {"TK0": 0, "TK1": 0}) == []

    def test_one_zero_price_first(self):
        """Test with first token price zero."""
        result = get_token_investments_multi(100, {"TK0": 0, "TK1": 50})
        assert len(result) >= 2
        assert result[0] == 0
        assert result[1] > 0

    def test_one_zero_price_second(self):
        """Test with second token price zero."""
        result = get_token_investments_multi(100, {"TK0": 50, "TK1": 0})
        assert len(result) >= 2
        assert result[0] > 0
        assert result[1] == 0

    def test_normal(self):
        """Test normal calculation with both valid prices."""
        result = get_token_investments_multi(100, {"TK0": 50, "TK1": 100})
        assert len(result) == 2
        assert result[0] > 0
        assert result[1] > 0

    def test_three_tokens(self):
        """Test with three tokens (third gets zero)."""
        result = get_token_investments_multi(100, {"TK0": 50, "TK1": 100, "TK2": 200})
        assert len(result) == 3
        assert result[0] > 0
        assert result[1] > 0
        assert result[2] == 0  # Only first two get investment


class TestCalculateSinglePoolInvestment:
    """Tests for calculate_single_pool_investment function."""

    def test_below_min_apr(self):
        """Test APR below minimum threshold."""
        assert calculate_single_pool_investment(0.01, 100000) == 0.0

    def test_normal(self):
        """Test normal calculation above min APR."""
        result = calculate_single_pool_investment(0.10, 100000)
        assert result > 0
        assert result <= 1000.0

    def test_small_investment_below_min(self):
        """Test investment below minimum threshold of 100."""
        result = calculate_single_pool_investment(0.021, 10)
        assert result == 0.0

    def test_high_apr_capped(self):
        """Test high APR investment is capped at max_investment."""
        result = calculate_single_pool_investment(1.0, 1000000)
        assert result <= 1000.0


class TestCalculateDifferentialInvestment:
    """Tests for calculate_differential_investment function."""

    def test_zero_tvl(self):
        """Test with zero TVL."""
        assert calculate_differential_investment(0.1, 0.05, 0) == 0.0

    def test_single_pool(self):
        """Test single pool mode."""
        result = calculate_differential_investment(0.1, 0, 100000, is_single_pool=True)
        assert result >= 0

    def test_low_apr(self):
        """Test with very low current APR."""
        assert calculate_differential_investment(0.005, 0, 100000) == 0.0

    def test_zero_base_apr(self):
        """Test with zero base APR uses 75% of current APR."""
        result = calculate_differential_investment(0.1, 0, 100000)
        assert result >= 0

    def test_ratio_lte_1(self):
        """Test when ratio <= 1 returns 0."""
        assert calculate_differential_investment(0.05, 0.1, 100000) == 0.0

    def test_normal(self):
        """Test normal differential investment calculation."""
        result = calculate_differential_investment(0.2, 0.1, 100000)
        assert result > 0
        assert result <= 1000.0

    def test_small_diff_below_min(self):
        """Test small differential below min threshold."""
        result = calculate_differential_investment(0.02, 0.019, 100)
        assert result == 0.0

    def test_negative_tvl(self):
        """Test with negative TVL."""
        assert calculate_differential_investment(0.1, 0.05, -100) == 0.0

    def test_negative_base_apr(self):
        """Test with negative base APR triggers apr_base <= 0 branch."""
        result = calculate_differential_investment(0.1, -0.01, 100000)
        assert result >= 0

    def test_single_pool_capped(self):
        """Test single pool investment is capped at 1000."""
        result = calculate_differential_investment(1.0, 0, 1000000, is_single_pool=True)
        assert result <= 1000.0


class TestFilterValidInvestmentPools:
    """Tests for filter_valid_investment_pools function."""

    def test_filters_zero_investment(self):
        """Test filtering out zero investment pools."""
        pools = [{"max_investment_usd": 0, "max_investment_amounts": [1, 1]}]
        assert filter_valid_investment_pools(pools) == []

    def test_filters_missing_amounts(self):
        """Test filtering out pools with insufficient amounts."""
        pools = [{"max_investment_usd": 100, "max_investment_amounts": [1]}]
        assert filter_valid_investment_pools(pools) == []

    def test_filters_zero_token0(self):
        """Test filtering out pools with zero token0 amount."""
        pools = [{"max_investment_usd": 100, "max_investment_amounts": [0, 1]}]
        assert filter_valid_investment_pools(pools) == []

    def test_filters_zero_token1(self):
        """Test filtering out pools with zero token1 amount."""
        pools = [{"max_investment_usd": 100, "max_investment_amounts": [1, 0]}]
        assert filter_valid_investment_pools(pools) == []

    def test_valid_pool(self):
        """Test valid pool passes filter."""
        pools = [{"max_investment_usd": 100, "max_investment_amounts": [1, 1]}]
        assert len(filter_valid_investment_pools(pools)) == 1

    def test_empty_list(self):
        """Test empty list."""
        assert filter_valid_investment_pools([]) == []

    def test_no_amounts_key(self):
        """Test pool missing max_investment_amounts key."""
        pools = [{"max_investment_usd": 100}]
        assert filter_valid_investment_pools(pools) == []


class TestFormatPoolData:
    """Tests for format_pool_data function."""

    def _make_pool(self, apr=0.1, tvl=100000, pool_id="pool1", chain="OPTIMISM"):
        """Create a test pool for format_pool_data."""
        return {
            "id": pool_id,
            "address": "0xaddr",
            "chain": chain,
            "type": "Weighted",
            "apr": apr,
            "poolTokens": [
                {"address": "0xtk0", "symbol": "TK0"},
                {"address": "0xtk1", "symbol": "TK1"},
            ],
            "dynamicData": {"totalLiquidity": str(tvl)},
            "il_risk_score": -0.05,
            "sharpe_ratio": 1.5,
            "depth_score": 100,
            "max_position_size": 5000,
            "trading_type": "lp",
        }

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_pool_token_prices"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_single_pool(self, mock_cg, mock_pro, mock_prices, mock_sleep):
        """Test formatting a single pool (is_single_pool=True branch)."""
        mock_pro.return_value = False
        mock_prices.return_value = {"TK0": 50.0, "TK1": 100.0}
        pools = [self._make_pool(apr=0.1)]
        result = format_pool_data(pools, "key")
        assert len(result) == 1
        assert result[0]["dex_type"] == "balancerPool"

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_pool_token_prices"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_multiple_pools(self, mock_cg, mock_pro, mock_prices, mock_sleep):
        """Test formatting multiple pools (is_single_pool=False branch)."""
        mock_pro.return_value = False
        mock_prices.return_value = {"TK0": 50.0, "TK1": 100.0}
        pools = [
            self._make_pool(apr=0.2, pool_id="pool1"),
            self._make_pool(apr=0.1, pool_id="pool2"),
        ]
        result = format_pool_data(pools, "key")
        assert len(result) == 2

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_pool_token_prices"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_low_apr_pool(self, mock_cg, mock_pro, mock_prices, mock_sleep):
        """Test pool with very low APR gets empty investment."""
        mock_pro.return_value = False
        mock_prices.return_value = {"TK0": 50.0, "TK1": 100.0}
        pools = [self._make_pool(apr=0.005)]
        result = format_pool_data(pools, "key")
        assert result[0]["max_investment_usd"] == 0.0
        assert result[0]["max_investment_amounts"] == []

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_pool_token_prices"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_zero_diff_investment(self, mock_cg, mock_pro, mock_prices, mock_sleep):
        """Test when diff_investment is 0 due to very low APR."""
        mock_pro.return_value = False
        mock_prices.return_value = {"TK0": 50.0, "TK1": 100.0}
        pools = [
            self._make_pool(apr=0.005, pool_id="pool1"),  # Below 0.01 threshold -> skip
            self._make_pool(apr=0.003, pool_id="pool2"),  # Below 0.01 threshold -> skip
        ]
        result = format_pool_data(pools, "key")
        # Both APRs are below 0.01 so all get empty investment
        assert all(p["max_investment_usd"] == 0.0 for p in result)

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_pool_token_prices"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_no_token_prices(self, mock_cg, mock_pro, mock_prices, mock_sleep):
        """Test when token prices are all zero."""
        mock_pro.return_value = False
        mock_prices.return_value = {"TK0": 0.0, "TK1": 0.0}
        pools = [self._make_pool(apr=0.2)]
        result = format_pool_data(pools, "key")
        assert result[0]["max_investment_usd"] == 0.0

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.time.sleep"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_pool_token_prices"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.is_pro_api_key"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_with_x402(self, mock_cg, mock_pro, mock_prices, mock_sleep):
        """Test with x402 session and proxy."""
        mock_pro.return_value = False
        mock_prices.return_value = {"TK0": 50.0, "TK1": 100.0}
        pools = [self._make_pool(apr=0.1)]
        result = format_pool_data(
            pools, "key", x402_session=MagicMock(), x402_proxy="https://proxy.com"
        )
        assert len(result) == 1


class TestGetOpportunitiesForBalancer:
    """Tests for get_opportunities_for_balancer function."""

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.filter_valid_investment_pools"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.format_pool_data"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.analyze_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_balancer_pool_sharpe_ratio"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.calculate_il_risk_score_multi"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_filtered_pools_for_balancer"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_balancer_pools"
    )
    def test_successful(
        self,
        mock_get,
        mock_filter,
        mock_il,
        mock_sharpe,
        mock_liq,
        mock_format,
        mock_valid,
    ):
        """Test successful opportunity search."""
        pool = {
            "id": "pool1",
            "chain": "OPTIMISM",
            "type": "Weighted",
            "poolTokens": [
                {"address": "0xt0", "symbol": "TK0"},
                {"address": "0xt1", "symbol": "TK1"},
            ],
        }
        mock_get.return_value = [pool]
        mock_filter.return_value = [pool]
        mock_il.return_value = -0.05
        mock_sharpe.return_value = 1.5
        mock_liq.return_value = (100, 5000)
        mock_format.return_value = [{"dex_type": "balancerPool"}]
        mock_valid.return_value = [{"dex_type": "balancerPool"}]

        result = get_opportunities_for_balancer(
            ["optimism"],
            "url",
            [],
            "key",
            {"optimism": {}},
            {"optimism": {"tk0": "id0", "tk1": "id1"}},
            None,
            None,
        )
        assert isinstance(result, list)
        assert len(result) == 1

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_balancer_pools"
    )
    def test_error_from_get_pools(self, mock_get):
        """Test error returned from get_balancer_pools."""
        mock_get.return_value = {"error": "API failure"}
        result = get_opportunities_for_balancer(
            ["optimism"], "url", [], "key", {}, {}, None, None
        )
        assert "error" in result

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_filtered_pools_for_balancer"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_balancer_pools"
    )
    def test_no_filtered_pools(self, mock_get, mock_filter):
        """Test when no pools pass filtering."""
        mock_get.return_value = [{"id": "1"}]
        mock_filter.return_value = []
        result = get_opportunities_for_balancer(
            ["optimism"], "url", [], "key", {}, {}, None, None
        )
        assert "error" in result

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.filter_valid_investment_pools"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.format_pool_data"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.analyze_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_balancer_pool_sharpe_ratio"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.calculate_il_risk_score_multi"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_filtered_pools_for_balancer"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_balancer_pools"
    )
    def test_insufficient_valid_token_ids(
        self,
        mock_get,
        mock_filter,
        mock_il,
        mock_sharpe,
        mock_liq,
        mock_format,
        mock_valid,
    ):
        """Test when pool has insufficient valid token IDs for IL calculation."""
        pool = {
            "id": "pool1",
            "chain": "OPTIMISM",
            "type": "Weighted",
            "poolTokens": [
                {"address": "0xt0", "symbol": "UNKNOWN0"},
                {"address": "0xt1", "symbol": "UNKNOWN1"},
            ],
        }
        mock_get.return_value = [pool]
        mock_filter.return_value = [pool]
        mock_sharpe.return_value = 1.5
        mock_liq.return_value = (100, 5000)
        mock_format.return_value = [{"dex_type": "balancerPool"}]
        mock_valid.return_value = [{"dex_type": "balancerPool"}]

        result = get_opportunities_for_balancer(
            ["optimism"],
            "url",
            [],
            "key",
            {"optimism": {}},
            {},  # empty coin_id_mapping
            None,
            None,
        )
        # Should still return results but with il_risk_score = None
        mock_il.assert_not_called()


class TestIsProApiKey:
    """Tests for is_pro_api_key function."""

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_pro(self, mock_cg):
        """Test pro key returns True."""
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.return_value = {"prices": []}
        mock_cg.return_value = inst
        assert is_pro_api_key("key") is True

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_not_pro(self, mock_cg):
        """Test non-pro key returns False."""
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.side_effect = Exception("fail")
        mock_cg.return_value = inst
        assert is_pro_api_key("key") is False

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_empty_response(self, mock_cg):
        """Test empty response (falsy) returns False."""
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.return_value = {}
        mock_cg.return_value = inst
        assert is_pro_api_key("key") is False

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.CoinGeckoAPI"
    )
    def test_none_response(self, mock_cg):
        """Test None response returns False."""
        inst = MagicMock()
        inst.get_coin_market_chart_range_by_id.return_value = None
        mock_cg.return_value = inst
        assert is_pro_api_key("key") is False


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.analyze_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_balancer_pool_sharpe_ratio"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.calculate_il_risk_score_multi"
    )
    def test_with_token_count(self, mock_il, mock_sharpe, mock_liq):
        """Test with token_count in position (token_count > 0 branch)."""
        mock_il.return_value = -0.05
        mock_sharpe.return_value = 1.5
        mock_liq.return_value = (100, 5000)
        position = {
            "pool_id": "p1",
            "chain": "optimism",
            "token_count": 2,
            "token0": "0xt0",
            "token0_symbol": "TK0",
            "token1": "0xt1",
            "token1_symbol": "TK1",
        }
        result = calculate_metrics(
            position, "key", {"optimism": {"tk0": "id0", "tk1": "id1"}}
        )
        assert result["il_risk_score"] == -0.05

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.analyze_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_balancer_pool_sharpe_ratio"
    )
    def test_fallback_token_format(self, mock_sharpe, mock_liq):
        """Test fallback to token0/token1 format (token_count=0)."""
        mock_sharpe.return_value = 1.5
        mock_liq.return_value = (100, 5000)
        position = {
            "pool_id": "p1",
            "chain": "optimism",
            "token_count": 0,
            "token0": "0xt0",
            "token0_symbol": "TK0",
            "token1": "0xt1",
            "token1_symbol": "TK1",
        }
        result = calculate_metrics(position, "key", {})
        assert result["il_risk_score"] is None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.analyze_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_balancer_pool_sharpe_ratio"
    )
    def test_insufficient_valid_ids(self, mock_sharpe, mock_liq):
        """Test with insufficient valid token IDs."""
        mock_sharpe.return_value = 1.5
        mock_liq.return_value = (100, 5000)
        position = {
            "pool_id": "p1",
            "chain": "optimism",
            "token_count": 2,
            "token0": "0xt0",
            "token0_symbol": "UNKNOWN0",
            "token1": "0xt1",
            "token1_symbol": "UNKNOWN1",
        }
        result = calculate_metrics(position, "key", {})
        assert result["il_risk_score"] is None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.analyze_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_balancer_pool_sharpe_ratio"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.calculate_il_risk_score_multi"
    )
    def test_fallback_with_valid_ids(self, mock_il, mock_sharpe, mock_liq):
        """Test fallback token0/token1 format with valid IDs >= 2."""
        mock_il.return_value = -0.03
        mock_sharpe.return_value = 1.5
        mock_liq.return_value = (100, 5000)
        position = {
            "pool_id": "p1",
            "chain": "optimism",
            "token_count": 0,
            "token0": "0xt0",
            "token0_symbol": "TK0",
            "token1": "0xt1",
            "token1_symbol": "TK1",
        }
        result = calculate_metrics(
            position, "key", {"optimism": {"tk0": "id0", "tk1": "id1"}}
        )
        assert result["il_risk_score"] == -0.03

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.analyze_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_balancer_pool_sharpe_ratio"
    )
    def test_no_token_count_key(self, mock_sharpe, mock_liq):
        """Test with missing token_count key (defaults to 0)."""
        mock_sharpe.return_value = 1.5
        mock_liq.return_value = (100, 5000)
        position = {
            "pool_id": "p1",
            "chain": "optimism",
            "token0": "0xt0",
            "token0_symbol": "TK0",
            "token1": "0xt1",
            "token1_symbol": "TK1",
        }
        result = calculate_metrics(
            position, "key", {"optimism": {"tk0": "id0", "tk1": "id1"}}
        )
        assert result is not None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.analyze_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_balancer_pool_sharpe_ratio"
    )
    def test_token_count_with_missing_keys(self, mock_sharpe, mock_liq):
        """Test token_count > 0 but some token keys missing."""
        mock_sharpe.return_value = 1.5
        mock_liq.return_value = (100, 5000)
        position = {
            "pool_id": "p1",
            "chain": "optimism",
            "token_count": 3,
            "token0": "0xt0",
            "token0_symbol": "TK0",
            "token1": "0xt1",
            "token1_symbol": "TK1",
            # token2 and token2_symbol are missing
        }
        result = calculate_metrics(
            position, "key", {"optimism": {"tk0": "id0", "tk1": "id1"}}
        )
        assert result is not None

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.analyze_pool_liquidity"
    )
    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_balancer_pool_sharpe_ratio"
    )
    def test_no_tokens_at_all(self, mock_sharpe, mock_liq):
        """Test with token_count=0 and no token0/token1 keys (elif False branch, line 1244->1254)."""
        mock_sharpe.return_value = 1.5
        mock_liq.return_value = (100, 5000)
        position = {
            "pool_id": "p1",
            "chain": "optimism",
            "token_count": 0,
            # No token0 or token1 keys at all
        }
        result = calculate_metrics(position, "key", {})
        assert result is not None
        assert result["il_risk_score"] is None


class TestRun:
    """Tests for the run function."""

    def test_missing_fields(self):
        """Test with missing fields."""
        result = run()
        assert "error" in result

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.calculate_metrics"
    )
    def test_get_metrics(self, mock_calc):
        """Test get_metrics mode with successful result."""
        mock_calc.return_value = {"il_risk_score": -0.05}
        result = run(
            chains=["optimism"],
            graphql_endpoint="url",
            current_positions=[],
            whitelisted_assets={},
            coin_id_mapping={},
            get_metrics=True,
            position={"pool_id": "p1"},
            coingecko_api_key="key",
            x402_session=None,
            x402_proxy=None,
        )
        assert result == {"il_risk_score": -0.05}

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.calculate_metrics"
    )
    def test_get_metrics_none(self, mock_calc):
        """Test get_metrics returning None."""
        mock_calc.return_value = None
        result = run(
            chains=["optimism"],
            graphql_endpoint="url",
            current_positions=[],
            whitelisted_assets={},
            coin_id_mapping={},
            get_metrics=True,
            position={"pool_id": "p1"},
            coingecko_api_key="key",
            x402_session=None,
            x402_proxy=None,
        )
        assert "error" in result

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_opportunities_for_balancer"
    )
    def test_opportunity_search(self, mock_opp):
        """Test opportunity search mode with results."""
        mock_opp.return_value = [{"pool": "data"}]
        result = run(
            chains=["optimism"],
            graphql_endpoint="url",
            current_positions=[],
            whitelisted_assets={},
            coin_id_mapping={},
        )
        assert "result" in result

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_opportunities_for_balancer"
    )
    def test_opportunity_search_error(self, mock_opp):
        """Test opportunity search returning error dict."""
        mock_opp.return_value = {"error": "fail"}
        result = run(
            chains=["optimism"],
            graphql_endpoint="url",
            current_positions=[],
            whitelisted_assets={},
            coin_id_mapping={},
        )
        assert "error" in result

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_opportunities_for_balancer"
    )
    def test_opportunity_search_empty(self, mock_opp):
        """Test empty opportunity search result."""
        mock_opp.return_value = []
        result = run(
            chains=["optimism"],
            graphql_endpoint="url",
            current_positions=[],
            whitelisted_assets={},
            coin_id_mapping={},
        )
        assert "error" in result

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.get_opportunities_for_balancer"
    )
    def test_opportunity_search_no_errors(self, mock_opp):
        """Test opportunity search with no errors produces clean result."""
        mock_opp.return_value = [{"pool": "data"}]
        result = run(
            chains=["optimism"],
            graphql_endpoint="url",
            current_positions=[],
            whitelisted_assets={},
            coin_id_mapping={},
        )
        assert result["error"] == []
        assert result["result"] == [{"pool": "data"}]

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.calculate_metrics"
    )
    def test_get_metrics_with_errors(self, mock_calc):
        """Test get_metrics when errors are accumulated."""
        mock_calc.return_value = {"il_risk_score": -0.05}
        bal_mod._thread_local.errors = ["some previous error"]
        result = run(
            chains=["optimism"],
            graphql_endpoint="url",
            current_positions=[],
            whitelisted_assets={},
            coin_id_mapping={},
            get_metrics=True,
            position={"pool_id": "p1"},
            coingecko_api_key="key",
            x402_session=None,
            x402_proxy=None,
        )
        assert "error" in result


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


class TestSharpeRatioNullData:
    """Tests for get_balancer_pool_sharpe_ratio with null data response."""

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_null_data_response(self, mock_post):
        """Test that {"data": null} response does not crash."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": None}
        mock_post.return_value = mock_resp
        result = get_balancer_pool_sharpe_ratio("pool1", "OPTIMISM")
        assert result is None


class TestRunQueryNetworkResilience:
    """Tests for run_query handling network errors and parse failures."""

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_connection_error(self, mock_post):
        """Test that ConnectionError is caught and returns error dict."""
        import requests as req_lib

        mock_post.side_effect = req_lib.ConnectionError("connection refused")
        result = run_query("query", "https://api.example.com")
        assert "error" in result

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_timeout_error(self, mock_post):
        """Test that Timeout is caught and returns error dict."""
        import requests as req_lib

        mock_post.side_effect = req_lib.Timeout("timed out")
        result = run_query("query", "https://api.example.com")
        assert "error" in result

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_json_decode_error(self, mock_post):
        """Test that JSONDecodeError from response.json() returns error dict."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("No JSON")
        mock_post.return_value = mock_resp
        result = run_query("query", "https://api.example.com")
        assert "error" in result

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_timeout_kwarg_passed(self, mock_post):
        """Test that timeout=30 is passed to requests.post."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"pools": []}}
        mock_post.return_value = mock_resp
        run_query("query", "https://api.example.com")
        _, kwargs = mock_post.call_args
        assert kwargs.get("timeout") == 30

    @patch(
        "packages.valory.customs.balancer_pools_search.balancer_pools_search.requests.post"
    )
    def test_data_null_returns_empty_dict(self, mock_post):
        """Test that {"data": null} returns empty dict instead of crashing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": None}
        mock_post.return_value = mock_resp
        result = run_query("query", "https://api.example.com")
        assert result == {}


class TestGetCachedPrice:
    """Tests for get_cached_price function."""

    def test_cache_miss_returns_none(self):
        """Test that missing cache key returns None."""
        assert get_cached_price("token", 90, {}, 1800) is None

    def test_cache_hit_returns_data(self):
        """Test that valid (non-expired) cache entry returns data."""
        cache = {}
        set_cached_price("token", 90, {"prices": [[0, 100]]}, cache)
        result = get_cached_price("token", 90, cache, 1800)
        assert result == {"prices": [[0, 100]]}

    def test_cache_expired_returns_none(self):
        """Test that expired cache entry returns None."""
        cache = {
            "il_range_token_90": {
                "data": {"prices": [[0, 100]]},
                "timestamp": time.time() - 3600,
            }
        }
        result = get_cached_price("token", 90, cache, 1800)
        assert result is None


class TestCalculateIlRiskScoreMultiWithCache:
    """Tests for calculate_il_risk_score_multi with pre-populated cache."""

    def test_all_tokens_cached(self):
        """Test that cached price data is used without calling CoinGeckoAPI."""
        prices_t0 = {"prices": [[i, 100 + i * 0.5] for i in range(50)]}
        prices_t1 = {"prices": [[i, 200 + i * 0.3] for i in range(50)]}
        cache = {}
        set_cached_price("t0", 90, prices_t0, cache)
        set_cached_price("t1", 90, prices_t1, cache)
        result = calculate_il_risk_score_multi(["t0", "t1"], "key", price_cache=cache)
        assert isinstance(result, float)

    def test_price_cache_none_defaults_to_empty_dict(self):
        """Test that passing price_cache=None does not crash."""
        result = calculate_il_risk_score_multi(["t0", None], "key", price_cache=None)
        assert result is None


class TestRunWithPriceCache:
    """Tests for run() with explicit price_cache dict."""

    def test_run_with_explicit_price_cache(self):
        """Test that run() accepts a non-None price_cache without error."""
        result = run(price_cache={}, price_cache_ttl=600)
        assert "error" in result

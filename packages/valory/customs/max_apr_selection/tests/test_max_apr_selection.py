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

"""Tests for max_apr_selection custom component."""

import math
from unittest.mock import patch

import pytest

from packages.valory.customs.max_apr_selection.max_apr_selection import (
    DEPTH_SCORE_THRESHOLD,
    IL_RISK_SCORE_THRESHOLD,
    REQUIRED_FIELDS,
    SHARPE_RATIO_THRESHOLD,
    apply_risk_thresholds_and_select_optimal_strategy,
    calculate_composite_score,
    calculate_relative_percentages,
    check_missing_fields,
    get_max_values,
    il_risk_descriptor,
    logs,
    remove_irrelevant_fields,
    run,
)


@pytest.fixture(autouse=True)
def clear_logs():
    """Clear the global logs list before each test."""
    logs.clear()
    yield
    logs.clear()


class TestCheckMissingFields:
    """Tests for the check_missing_fields function."""

    def test_no_missing_fields(self):
        """Test that no missing fields are returned when all fields are present."""
        kwargs = {field: "value" for field in REQUIRED_FIELDS}
        assert check_missing_fields(kwargs) == []

    def test_all_missing_fields(self):
        """Test that all fields are returned when none are present."""
        result = check_missing_fields({})
        assert set(result) == set(REQUIRED_FIELDS)

    def test_none_value_counts_as_missing(self):
        """Test that fields with None value count as missing."""
        kwargs = {field: "value" for field in REQUIRED_FIELDS}
        kwargs["trading_opportunities"] = None
        result = check_missing_fields(kwargs)
        assert "trading_opportunities" in result


class TestRemoveIrrelevantFields:
    """Tests for the remove_irrelevant_fields function."""

    def test_removes_irrelevant_fields(self):
        """Test that irrelevant fields are removed."""
        kwargs = {field: "value" for field in REQUIRED_FIELDS}
        kwargs["extra"] = "should_go"
        result = remove_irrelevant_fields(kwargs)
        assert "extra" not in result

    def test_empty_kwargs(self):
        """Test with empty kwargs."""
        assert remove_irrelevant_fields({}) == {}


class TestCalculateCompositeScore:
    """Tests for the calculate_composite_score function."""

    def test_normal_calculation(self):
        """Test composite score with valid values."""
        pool = {"sharpe_ratio": 2.0, "depth_score": 100.0, "il_risk_score": -0.3}
        max_values = {"sharpe_ratio": 4.0, "depth_score": 200.0, "il_risk_score": -0.6}
        score = calculate_composite_score(pool, max_values)
        assert isinstance(score, float)
        assert score > 0

    def test_none_metrics_returns_zero(self):
        """Test that None metrics return score of 0."""
        pool = {"sharpe_ratio": None, "depth_score": 100.0, "il_risk_score": -0.3}
        max_values = {"sharpe_ratio": 4.0, "depth_score": 200.0, "il_risk_score": -0.6}
        assert calculate_composite_score(pool, max_values) == 0

    def test_none_depth_score_returns_zero(self):
        """Test that None depth_score returns score of 0."""
        pool = {"sharpe_ratio": 2.0, "depth_score": None, "il_risk_score": -0.3}
        max_values = {"sharpe_ratio": 4.0, "depth_score": 200.0, "il_risk_score": -0.6}
        assert calculate_composite_score(pool, max_values) == 0

    def test_none_il_risk_score_returns_zero(self):
        """Test that None il_risk_score returns score of 0."""
        pool = {"sharpe_ratio": 2.0, "depth_score": 100.0, "il_risk_score": None}
        max_values = {"sharpe_ratio": 4.0, "depth_score": 200.0, "il_risk_score": -0.6}
        assert calculate_composite_score(pool, max_values) == 0

    def test_nan_metrics_returns_zero(self):
        """Test that NaN metrics return score of 0."""
        pool = {"sharpe_ratio": math.nan, "depth_score": 100.0, "il_risk_score": -0.3}
        max_values = {"sharpe_ratio": 4.0, "depth_score": 200.0, "il_risk_score": -0.6}
        assert calculate_composite_score(pool, max_values) == 0

    def test_nan_depth_score_returns_zero(self):
        """Test that NaN depth_score returns score of 0."""
        pool = {"sharpe_ratio": 2.0, "depth_score": math.nan, "il_risk_score": -0.3}
        max_values = {"sharpe_ratio": 4.0, "depth_score": 200.0, "il_risk_score": -0.6}
        assert calculate_composite_score(pool, max_values) == 0

    def test_nan_il_risk_score_returns_zero(self):
        """Test that NaN il_risk_score returns score of 0."""
        pool = {"sharpe_ratio": 2.0, "depth_score": 100.0, "il_risk_score": math.nan}
        max_values = {"sharpe_ratio": 4.0, "depth_score": 200.0, "il_risk_score": -0.6}
        assert calculate_composite_score(pool, max_values) == 0

    def test_zero_il_risk_max_value(self):
        """Test when max IL risk score is 0."""
        pool = {"sharpe_ratio": 2.0, "depth_score": 100.0, "il_risk_score": 0}
        max_values = {"sharpe_ratio": 4.0, "depth_score": 200.0, "il_risk_score": 0}
        score = calculate_composite_score(pool, max_values)
        assert score > 0

    def test_missing_fields_default_to_nan(self):
        """Test that missing fields default to nan."""
        pool = {}
        max_values = {"sharpe_ratio": 4.0, "depth_score": 200.0, "il_risk_score": -0.6}
        assert calculate_composite_score(pool, max_values) == 0

    def test_il_risk_normalization_clamped(self):
        """Test that normalized IL risk score is clamped between 0 and 1."""
        pool = {"sharpe_ratio": 2.0, "depth_score": 100.0, "il_risk_score": -0.1}
        max_values = {"sharpe_ratio": 4.0, "depth_score": 200.0, "il_risk_score": -0.6}
        score = calculate_composite_score(pool, max_values)
        assert score > 0


class TestGetMaxValues:
    """Tests for the get_max_values function."""

    def test_normal_pools(self):
        """Test max values with normal pool data."""
        pools = [
            {"sharpe_ratio": 1.0, "depth_score": 50.0, "il_risk_score": -0.1},
            {"sharpe_ratio": 3.0, "depth_score": 150.0, "il_risk_score": -0.5},
        ]
        max_vals = get_max_values(pools)
        assert max_vals["sharpe_ratio"] == 3.0
        assert max_vals["depth_score"] == 150.0
        assert max_vals["il_risk_score"] == -0.5

    def test_missing_fields_default_zero(self):
        """Test that missing fields default to 0."""
        pools = [{}]
        max_vals = get_max_values(pools)
        assert max_vals["sharpe_ratio"] == 0
        assert max_vals["depth_score"] == 0
        assert max_vals["il_risk_score"] == 0

    def test_il_risk_max_by_abs_value(self):
        """Test that IL risk score max is by absolute value (most negative)."""
        pools = [
            {"il_risk_score": -0.1, "sharpe_ratio": 1, "depth_score": 1},
            {"il_risk_score": -0.9, "sharpe_ratio": 1, "depth_score": 1},
        ]
        max_vals = get_max_values(pools)
        assert max_vals["il_risk_score"] == -0.9


class TestIlRiskDescriptor:
    """Tests for the il_risk_descriptor function."""

    def test_none_value(self):
        """Test with None value."""
        assert il_risk_descriptor(None) == "Unknown"

    def test_high_risk(self):
        """Test high risk (< -0.5)."""
        assert il_risk_descriptor(-0.8) == "High"

    def test_moderate_risk(self):
        """Test moderate risk (-0.5 to -0.2)."""
        assert il_risk_descriptor(-0.3) == "Moderate"

    def test_low_risk(self):
        """Test low risk (-0.2 to 0)."""
        assert il_risk_descriptor(-0.1) == "Low"

    def test_minimal_risk(self):
        """Test minimal risk (>= 0)."""
        assert il_risk_descriptor(0.1) == "Minimal"

    def test_boundary_minus_0_5(self):
        """Test boundary value -0.5."""
        assert il_risk_descriptor(-0.5) == "Moderate"

    def test_boundary_minus_0_2(self):
        """Test boundary value -0.2."""
        assert il_risk_descriptor(-0.2) == "Low"

    def test_boundary_zero(self):
        """Test boundary value 0."""
        assert il_risk_descriptor(0) == "Minimal"


class TestCalculateRelativePercentages:
    """Tests for the calculate_relative_percentages function."""

    def test_equal_percentages(self):
        """Test with equal percentages."""
        result = calculate_relative_percentages([50, 50])
        assert len(result) == 2
        assert abs(result[0] - 0.5) < 1e-10

    def test_single_value(self):
        """Test with a single value."""
        result = calculate_relative_percentages([100])
        assert len(result) == 1
        assert abs(result[0] - 1.0) < 1e-10

    def test_zero_total_returns_empty(self):
        """Test with all zeros returns empty list."""
        result = calculate_relative_percentages([0, 0, 0])
        assert result == []
        assert any("zero" in log.lower() for log in logs)

    def test_multiple_values(self):
        """Test with multiple values."""
        result = calculate_relative_percentages([60, 30, 10])
        assert len(result) == 3
        assert abs(result[0] - 0.6) < 1e-10


class TestApplyRiskThresholdsAndSelectOptimalStrategy:
    """Tests for the apply_risk_thresholds_and_select_optimal_strategy function."""

    def _make_opportunity(self, sharpe=1.0, depth=50.0, il_risk=-0.1, apr=10.0,
                          dex_type="UniswapV3", pool_address="0xpool"):
        """Create a test opportunity."""
        return {
            "sharpe_ratio": sharpe,
            "depth_score": depth,
            "il_risk_score": il_risk,
            "apr": apr,
            "dex_type": dex_type,
            "pool_address": pool_address,
            "token0_symbol": "TOKEN0",
            "token1_symbol": "TOKEN1",
            "advertised_apr": apr,
        }

    def test_no_opportunities_meeting_thresholds(self):
        """Test with opportunities that fail risk thresholds."""
        opps = [self._make_opportunity(sharpe=0)]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=0.1
        )
        assert result["optimal_strategies"] == []
        assert result["position_to_exit"] == {}

    def test_sharpe_ratio_below_threshold(self):
        """Test that pools with sharpe_ratio <= threshold are excluded."""
        opps = [self._make_opportunity(sharpe=SHARPE_RATIO_THRESHOLD)]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=0.1
        )
        assert result["optimal_strategies"] == []

    def test_depth_score_below_threshold(self):
        """Test that pools with depth_score <= threshold are excluded."""
        opps = [self._make_opportunity(depth=DEPTH_SCORE_THRESHOLD)]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=0.1
        )
        assert result["optimal_strategies"] == []

    def test_il_risk_above_threshold(self):
        """Test that pools with il_risk_score > threshold are excluded."""
        opps = [self._make_opportunity(il_risk=IL_RISK_SCORE_THRESHOLD + 0.1)]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=0.1
        )
        assert result["optimal_strategies"] == []

    def test_invalid_metric_types(self):
        """Test with invalid metric types (non-numeric)."""
        opp = self._make_opportunity()
        opp["sharpe_ratio"] = "invalid"
        result = apply_risk_thresholds_and_select_optimal_strategy(
            [opp], composite_score_threshold=0.1
        )
        assert result["optimal_strategies"] == []

    def test_invalid_depth_score_type(self):
        """Test with non-numeric depth_score."""
        opp = self._make_opportunity()
        opp["depth_score"] = "invalid"
        result = apply_risk_thresholds_and_select_optimal_strategy(
            [opp], composite_score_threshold=0.1
        )
        assert result["optimal_strategies"] == []

    def test_invalid_il_risk_type(self):
        """Test with non-numeric il_risk_score."""
        opp = self._make_opportunity()
        opp["il_risk_score"] = "invalid"
        result = apply_risk_thresholds_and_select_optimal_strategy(
            [opp], composite_score_threshold=0.1
        )
        assert result["optimal_strategies"] == []

    def test_new_entry_without_positions(self):
        """Test optimal strategy selection for new entry (no current positions)."""
        opps = [
            self._make_opportunity(sharpe=2.0, depth=100.0, il_risk=-0.1, apr=20.0),
            self._make_opportunity(sharpe=1.5, depth=80.0, il_risk=-0.2, apr=15.0,
                                   pool_address="0xpool2"),
        ]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=0.0, max_pools=2
        )
        assert len(result["optimal_strategies"]) > 0
        assert result["position_to_exit"] == {}
        assert "reasoning" in result

    def test_new_entry_single_pool(self):
        """Test selection for new entry with max_pools=1."""
        opps = [
            self._make_opportunity(sharpe=2.0, depth=100.0, il_risk=-0.1, apr=20.0),
        ]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=0.0, max_pools=1
        )
        assert len(result["optimal_strategies"]) == 1
        assert "relative_funds_percentage" in result["optimal_strategies"][0]

    def test_new_entry_no_pools_meet_composite_threshold(self):
        """Test when no opportunities meet the composite score threshold for new entry."""
        opps = [
            self._make_opportunity(sharpe=0.5, depth=10.0, il_risk=-0.1, apr=5.0),
        ]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=999.0, max_pools=1
        )
        assert result["optimal_strategies"] == []

    def test_with_current_positions_better_opportunity(self):
        """Test with current positions and a better opportunity found."""
        opps = [
            self._make_opportunity(sharpe=5.0, depth=200.0, il_risk=-0.05, apr=30.0),
        ]
        current = [
            self._make_opportunity(sharpe=1.0, depth=50.0, il_risk=-0.3, apr=10.0),
        ]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=0.0, current_positions=current
        )
        assert len(result["optimal_strategies"]) == 1
        assert result["position_to_exit"] is not None

    def test_with_current_positions_no_better_opportunity(self):
        """Test with current positions when no better opportunity is found."""
        opps = [
            self._make_opportunity(sharpe=0.5, depth=10.0, il_risk=-0.1, apr=5.0),
        ]
        current = [
            self._make_opportunity(sharpe=3.0, depth=100.0, il_risk=-0.1, apr=20.0),
        ]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=999.0, current_positions=current
        )
        assert result["optimal_strategies"] == []

    def test_metrics_improved_comparison(self):
        """Test that metrics comparison between old and new pools works."""
        opps = [
            self._make_opportunity(sharpe=5.0, depth=200.0, il_risk=-0.01, apr=30.0),
        ]
        current = [
            self._make_opportunity(sharpe=1.0, depth=50.0, il_risk=-0.3, apr=10.0),
        ]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=0.0, current_positions=current
        )
        assert "reasoning" in result
        assert len(result["optimal_strategies"]) > 0

    def test_il_risk_none_in_comparison(self):
        """Test metrics comparison when il_risk_score is None."""
        opps = [
            self._make_opportunity(sharpe=5.0, depth=200.0, il_risk=-0.01, apr=30.0),
        ]
        current_pos = self._make_opportunity(sharpe=1.0, depth=50.0, il_risk=-0.3, apr=10.0)
        current_pos["il_risk_score"] = None
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=0.0, current_positions=[current_pos]
        )
        assert "reasoning" in result

    def test_top_opp_il_risk_none_in_comparison(self):
        """Test metrics comparison when top opp il_risk_score is None."""
        opp = self._make_opportunity(sharpe=5.0, depth=200.0, il_risk=-0.01, apr=30.0)
        opp["il_risk_score"] = None
        current = [
            self._make_opportunity(sharpe=1.0, depth=50.0, il_risk=-0.3, apr=10.0),
        ]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            [opp], composite_score_threshold=0.0, current_positions=current
        )
        assert "reasoning" in result

    def test_multiple_current_positions_least_performing(self):
        """Test least performing pool is identified from multiple current positions."""
        opps = [
            self._make_opportunity(sharpe=5.0, depth=200.0, il_risk=-0.05, apr=30.0),
        ]
        current = [
            self._make_opportunity(sharpe=3.0, depth=100.0, il_risk=-0.1, apr=20.0,
                                   pool_address="0xgood"),
            self._make_opportunity(sharpe=0.5, depth=10.0, il_risk=-0.4, apr=5.0,
                                   pool_address="0xbad"),
        ]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=0.0, current_positions=current
        )
        if result["optimal_strategies"]:
            assert result["position_to_exit"]["pool_address"] == "0xbad"

    def test_relative_funds_percentage_with_positions(self):
        """Test relative_funds_percentage is set to 1.0 when current positions exist."""
        opps = [
            self._make_opportunity(sharpe=5.0, depth=200.0, il_risk=-0.05, apr=30.0),
        ]
        current = [
            self._make_opportunity(sharpe=1.0, depth=50.0, il_risk=-0.3, apr=10.0),
        ]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=0.0, current_positions=current
        )
        if result["optimal_strategies"]:
            assert result["optimal_strategies"][0]["relative_funds_percentage"] == 1.0


class TestRun:
    """Tests for the run function."""

    def test_missing_required_fields(self):
        """Test run with missing fields."""
        result = run()
        assert "error" in result

    def test_partial_missing_fields(self):
        """Test run with partial fields."""
        result = run(trading_opportunities=[])
        assert "error" in result

    @patch("packages.valory.customs.max_apr_selection.max_apr_selection.apply_risk_thresholds_and_select_optimal_strategy")
    def test_run_delegates_correctly(self, mock_fn):
        """Test that run delegates to apply_risk_thresholds_and_select_optimal_strategy."""
        mock_fn.return_value = {"optimal_strategies": [], "position_to_exit": {},
                                "reasoning": "test"}
        kwargs = {
            "trading_opportunities": [],
            "current_positions": [],
            "max_pools": 1,
            "composite_score_threshold": 0.5,
        }
        result = run(**kwargs)
        mock_fn.assert_called_once()
        assert "logs" in result

    @patch("packages.valory.customs.max_apr_selection.max_apr_selection.apply_risk_thresholds_and_select_optimal_strategy")
    def test_run_strips_irrelevant_kwargs(self, mock_fn):
        """Test that run strips extra kwargs."""
        mock_fn.return_value = {"optimal_strategies": [], "position_to_exit": {},
                                "reasoning": "test"}
        kwargs = {
            "trading_opportunities": [],
            "current_positions": [],
            "max_pools": 1,
            "composite_score_threshold": 0.5,
            "extra_field": "should_be_removed",
        }
        result = run(**kwargs)
        call_kwargs = mock_fn.call_args[1]
        assert "extra_field" not in call_kwargs

    def test_run_with_positional_args(self):
        """Test that positional args are ignored."""
        result = run("positional", trading_opportunities=[], current_positions=[],
                     max_pools=1, composite_score_threshold=0.5)
        assert "logs" in result


class TestBranchCoverage:
    """Tests specifically targeting uncovered branches."""

    def _make_opportunity(self, sharpe=1.0, depth=50.0, il_risk=-0.1, apr=10.0,
                          dex_type="UniswapV3", pool_address="0xpool"):
        """Create a test opportunity."""
        return {
            "sharpe_ratio": sharpe,
            "depth_score": depth,
            "il_risk_score": il_risk,
            "apr": apr,
            "dex_type": dex_type,
            "pool_address": pool_address,
            "token0_symbol": "TOKEN0",
            "token1_symbol": "TOKEN1",
            "advertised_apr": apr,
        }

    def test_position_to_exit_empty_dict(self):
        """Test branch 220->231: position_to_exit is an empty dict (falsy).

        When a current position is an empty dict, calculate_composite_score
        returns 0 (fields default to nan), making it the least performing pool.
        Since {} is falsy, `if position_to_exit:` is False, skipping the
        reasoning block about the current position.
        """
        opps = [
            self._make_opportunity(sharpe=2.0, depth=100.0, il_risk=-0.1, apr=20.0),
        ]
        # Use an empty dict as one current position so it becomes least performing
        current = [{}]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=0.0, current_positions=current
        )
        # The function should still return results; the empty-dict position
        # means the "if position_to_exit:" block is skipped
        assert "reasoning" in result
        assert len(result["optimal_strategies"]) == 1

    def test_new_opp_worse_sharpe_and_depth(self):
        """Test branches 235->237 and 237->241: new opportunity has worse sharpe and depth.

        When the top opportunity has a lower sharpe_ratio and depth_score than
        the position_to_exit, both 'if' branches at lines 235 and 237 take
        the false path.
        """
        # New opportunity: lower sharpe and depth, but better IL risk and high APR
        opps = [
            self._make_opportunity(sharpe=1.0, depth=30.0, il_risk=-0.01, apr=50.0),
        ]
        # Current position: higher sharpe and depth, but worse IL risk
        current = [
            self._make_opportunity(sharpe=5.0, depth=200.0, il_risk=-0.4, apr=10.0),
        ]
        result = apply_risk_thresholds_and_select_optimal_strategy(
            opps, composite_score_threshold=0.0, current_positions=current
        )
        assert "reasoning" in result
        assert len(result["optimal_strategies"]) == 1
        # Verify that "higher risk-adjusted returns" and "better market liquidity"
        # are NOT in the reasoning (since sharpe and depth are worse)
        reasoning = result["reasoning"]
        assert "higher risk-adjusted returns" not in reasoning
        assert "better market liquidity" not in reasoning

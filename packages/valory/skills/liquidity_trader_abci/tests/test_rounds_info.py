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

"""Test the rounds_info.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from packages.valory.skills.liquidity_trader_abci.rounds_info import ROUNDS_INFO


def test_import() -> None:
    """Test that the rounds_info module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.rounds_info  # noqa


def test_rounds_info_is_dict() -> None:
    """Test ROUNDS_INFO is a dict."""
    assert isinstance(ROUNDS_INFO, dict)


def test_rounds_info_not_empty() -> None:
    """Test ROUNDS_INFO is not empty."""
    assert len(ROUNDS_INFO) > 0


def test_rounds_info_entries_have_required_keys() -> None:
    """Test each entry in ROUNDS_INFO has required keys."""
    for round_name, info in ROUNDS_INFO.items():
        assert "name" in info, f"Missing 'name' in {round_name}"
        assert "description" in info, f"Missing 'description' in {round_name}"
        assert "transitions" in info, f"Missing 'transitions' in {round_name}"


def test_rounds_info_contains_key_rounds() -> None:
    """Test ROUNDS_INFO contains important round names."""
    expected_rounds = [
        "APRPopulationRound",
        "CallCheckpointRound",
        "CheckStakingKPIMetRound",
        "DecisionMakingRound",
        "EvaluateStrategyRound",
        "FetchStrategiesRound",
        "GetPositionsRound",
        "PostTxSettlementRound",
        "WithdrawFundsRound",
    ]
    for round_name in expected_rounds:
        assert round_name in ROUNDS_INFO, f"Missing {round_name}"

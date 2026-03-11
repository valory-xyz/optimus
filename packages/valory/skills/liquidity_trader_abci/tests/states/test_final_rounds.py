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

"""Test the states/final_rounds.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from packages.valory.skills.abstract_round_abci.base import DegenerateRound
from packages.valory.skills.liquidity_trader_abci.states.final_rounds import (
    FailedMultiplexerRound,
    FinishedCallCheckpointRound,
    FinishedCheckStakingKPIMetRound,
    FinishedDecisionMakingRound,
    FinishedEvaluateStrategyRound,
    FinishedTxPreparationRound,
)


def test_import() -> None:
    """Test that the final_rounds module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.states.final_rounds  # noqa


def test_failed_multiplexer_round_is_degenerate() -> None:
    """Test FailedMultiplexerRound is a DegenerateRound."""
    assert issubclass(FailedMultiplexerRound, DegenerateRound)


def test_finished_call_checkpoint_round_is_degenerate() -> None:
    """Test FinishedCallCheckpointRound is a DegenerateRound."""
    assert issubclass(FinishedCallCheckpointRound, DegenerateRound)


def test_finished_check_staking_kpi_met_round_is_degenerate() -> None:
    """Test FinishedCheckStakingKPIMetRound is a DegenerateRound."""
    assert issubclass(FinishedCheckStakingKPIMetRound, DegenerateRound)


def test_finished_decision_making_round_is_degenerate() -> None:
    """Test FinishedDecisionMakingRound is a DegenerateRound."""
    assert issubclass(FinishedDecisionMakingRound, DegenerateRound)


def test_finished_evaluate_strategy_round_is_degenerate() -> None:
    """Test FinishedEvaluateStrategyRound is a DegenerateRound."""
    assert issubclass(FinishedEvaluateStrategyRound, DegenerateRound)


def test_finished_tx_preparation_round_is_degenerate() -> None:
    """Test FinishedTxPreparationRound is a DegenerateRound."""
    assert issubclass(FinishedTxPreparationRound, DegenerateRound)

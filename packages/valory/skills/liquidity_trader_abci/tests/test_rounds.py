# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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

"""This module contains the test for rounds of  liquidity trader"""
from unittest.mock import MagicMock

import pytest
import logging  # noqa: F401
from typing import cast

from packages.valory.skills.abstract_round_abci.base import (
    AbciAppDB
)

from packages.valory.skills.liquidity_trader_abci.rounds import (
    CallCheckpointRound,
    CheckStakingKPIMetRound,
    DecisionMakingRound,
    EvaluateStrategyRound,
    Event,
    FinishedCallCheckpointRound,
    FinishedCheckStakingKPIMetRound,
    FinishedEvaluateStrategyRound,
    FinishedDecisionMakingRound,
    FinishedTxPreparationRound,
    FailedMultiplexerRound,
    GetPositionsRound,
    LiquidityTraderAbciApp,
    PostTxSettlementRound,
    StakingState,
    SynchronizedData,
)

@pytest.fixture
def setup_app() -> LiquidityTraderAbciApp:
    """Set up the initial app instance for testing."""
    # Create mock objects for the required arguments
    synchronized_data = MagicMock(spec=SynchronizedData)
    logger = MagicMock()  # Mock logger
    context = MagicMock()  # Mock context

    # Initialize the app with the mocked dependencies
    return LiquidityTraderAbciApp(synchronized_data, logger, context)


def test_initial_state(setup_app: LiquidityTraderAbciApp) -> None:
    """Test the initial round of the application."""
    app = setup_app
    assert app.initial_round_cls == CallCheckpointRound
    assert CallCheckpointRound in app.initial_states


def test_call_check_point_round_transition(setup_app: LiquidityTraderAbciApp) -> None:
    """Test transitions from CallCheckpointRound."""
    app = setup_app
    transition_function = app.transition_function[CallCheckpointRound]

    # Transition on done
    assert (
        transition_function[Event.DONE] == CheckStakingKPIMetRound
    )

    # Transition on next checkpoint
    assert (
        transition_function[Event.NEXT_CHECKPOINT_NOT_REACHED_YET]
        == CheckStakingKPIMetRound
    )

    # Test no settle
    assert transition_function[Event.SETTLE] == FinishedCallCheckpointRound
    
    # Test no service evicted
    assert transition_function[Event.SERVICE_EVICTED] == GetPositionsRound

    # Test no majority
    assert transition_function[Event.NO_MAJORITY] == CallCheckpointRound


def test_check_staking_kpi_round_transition(setup_app: LiquidityTraderAbciApp) -> None:
    """Test transitions from CheckStakingKPIMetRound."""
    app = setup_app
    transition_function = app.transition_function[CheckStakingKPIMetRound]

    # Transition on done
    assert transition_function[Event.DONE] == GetPositionsRound

    # Transition on settle
    assert transition_function[Event.SETTLE] == FinishedCheckStakingKPIMetRound

    # Test none and no majority
    assert transition_function[Event.NONE] == CheckStakingKPIMetRound
    assert transition_function[Event.NO_MAJORITY] == CheckStakingKPIMetRound


def test_get_positions_round_transition(setup_app: LiquidityTraderAbciApp) -> None:
    """Test transitions from GetPositionsRound."""
    app = setup_app
    transition_function = app.transition_function[GetPositionsRound]

    # Transition on done
    assert transition_function[Event.DONE] == EvaluateStrategyRound

    # Test no majority
    assert transition_function[Event.NO_MAJORITY] == GetPositionsRound
    

def test_evaluate_strategy_round_transition(setup_app: LiquidityTraderAbciApp) -> None:
    """Test transitions from EvaluateStrategyRound."""
    app = setup_app
    transition_function = app.transition_function[EvaluateStrategyRound]

    # Test transition on done
    assert transition_function[Event.DONE] == DecisionMakingRound

    # Test transition on wait
    assert transition_function[Event.WAIT] == FinishedEvaluateStrategyRound


def test_decision_making_round_transition(setup_app: LiquidityTraderAbciApp) -> None:
    """Test transitions from DecisionMakingRound."""
    app = setup_app
    transition_function = app.transition_function[DecisionMakingRound]

    # Transition on done
    assert transition_function[Event.DONE] == FinishedDecisionMakingRound

    # Test no majority
    assert transition_function[Event.NO_MAJORITY] == DecisionMakingRound

    # Test no settle
    assert transition_function[Event.SETTLE] == FinishedTxPreparationRound


def test_post_tx_settlement_round_transition(setup_app: LiquidityTraderAbciApp) -> None:
    """Test transitions from PostTxSettlementRound."""
    app = setup_app
    transition_function = app.transition_function[PostTxSettlementRound]

    # Test transition on done
    assert transition_function[Event.DONE] == PostTxSettlementRound

    # Test transition on action execute
    assert transition_function[Event.ACTION_EXECUTED] == DecisionMakingRound

    # Test transition on checkpoint tx executed
    assert transition_function[Event.CHECKPOINT_TX_EXECUTED] == CallCheckpointRound

    # Test transition on vanity tx executed
    assert transition_function[Event.VANITY_TX_EXECUTED] == CheckStakingKPIMetRound

    # Test transition on unrecognized
    assert transition_function[Event.UNRECOGNIZED] == FailedMultiplexerRound


def test_final_states(setup_app: LiquidityTraderAbciApp) -> None:
    """Test the final states of the application."""
    app = setup_app
    assert FinishedEvaluateStrategyRound in app.final_states
    assert FinishedDecisionMakingRound in app.final_states
    assert FinishedTxPreparationRound in app.final_states
    assert FinishedCallCheckpointRound in app.final_states
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

"""Test the behaviours/__init__.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from packages.valory.skills.liquidity_trader_abci.behaviours import (
    Action,
    CheckStakingKPIMetBehaviour,
    Decision,
    DecisionMakingBehaviour,
    DexType,
    EvaluateStrategyBehaviour,
    GasCostTracker,
    GetPositionsBehaviour,
    PositionStatus,
    PostTxSettlementBehaviour,
    WithdrawFundsBehaviour,
    __all__,
)


def test_import() -> None:
    """Test that the behaviours __init__ module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.behaviours  # noqa


def test_all_exports() -> None:
    """Test __all__ contains expected exports."""
    expected = [
        "Action",
        "CheckStakingKPIMetBehaviour",
        "Decision",
        "DecisionMakingBehaviour",
        "DexType",
        "EvaluateStrategyBehaviour",
        "GasCostTracker",
        "GetPositionsBehaviour",
        "PositionStatus",
        "PostTxSettlementBehaviour",
        "WithdrawFundsBehaviour",
    ]
    assert sorted(__all__) == sorted(expected)


def test_action_is_enum() -> None:
    """Test Action is available."""
    assert hasattr(Action, "EXIT_POOL")


def test_decision_is_enum() -> None:
    """Test Decision is available."""
    assert hasattr(Decision, "CONTINUE")


def test_dex_type_is_enum() -> None:
    """Test DexType is available."""
    assert hasattr(DexType, "BALANCER")


def test_position_status_is_enum() -> None:
    """Test PositionStatus is available."""
    assert hasattr(PositionStatus, "OPEN")


def test_gas_cost_tracker_is_class() -> None:
    """Test GasCostTracker is available."""
    assert callable(GasCostTracker)

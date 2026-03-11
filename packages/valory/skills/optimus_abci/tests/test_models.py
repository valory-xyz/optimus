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

"""Test the models.py module of the optimus_abci skill."""

# pylint: skip-file

from unittest.mock import MagicMock, patch

from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.tests.data.dummy_abci.models import (
    RandomnessApi as BaseRandomnessApi,
)
from packages.valory.skills.liquidity_trader_abci.models import Coingecko as BaseCoingecko
from packages.valory.skills.optimus_abci.composition import OptimusAbciApp
from packages.valory.skills.optimus_abci.models import (
    MARGIN,
    MULTIPLIER,
    BenchmarkTool,
    Coingecko,
    Params,
    RandomnessApi,
    Requests,
    SharedState,
)


def test_import() -> None:
    """Test that the models module can be imported."""
    import packages.valory.skills.optimus_abci.models  # noqa


def test_coingecko_alias() -> None:
    """Test Coingecko is an alias."""
    assert Coingecko is BaseCoingecko


def test_requests_alias() -> None:
    """Test Requests is an alias."""
    assert Requests is BaseRequests


def test_benchmark_tool_alias() -> None:
    """Test BenchmarkTool is an alias."""
    assert BenchmarkTool is BaseBenchmarkTool


def test_randomness_api_alias() -> None:
    """Test RandomnessApi is an alias."""
    assert RandomnessApi is BaseRandomnessApi


def test_margin_constant() -> None:
    """Test MARGIN constant value."""
    assert MARGIN == 5


def test_multiplier_constant() -> None:
    """Test MULTIPLIER constant value."""
    assert MULTIPLIER == 40


class TestParams:
    """Test Params class."""

    def test_init_extracts_service_endpoint_base(self) -> None:
        """Test Params __init__ extracts service_endpoint_base."""
        mock_context = MagicMock()
        mock_context.skill_id = "test_skill/test:0.1.0"
        # Mock _ensure to capture the call and super().__init__ to avoid complex parent init
        with patch.object(
            Params, "_ensure", return_value="http://localhost:8000"
        ) as mock_ensure, patch.object(
            Params.__bases__[0], "__init__", return_value=None
        ):
            params = Params.__new__(Params)
            params.__init__(skill_context=mock_context, service_endpoint_base="http://localhost:8000")
            mock_ensure.assert_called_once_with(
                "service_endpoint_base",
                {"skill_context": mock_context, "service_endpoint_base": "http://localhost:8000"},
                str,
            )
            assert params.service_endpoint_base == "http://localhost:8000"


class TestSharedState:
    """Test SharedState class."""

    def test_abci_app_cls(self) -> None:
        """Test that abci_app_cls is OptimusAbciApp."""
        assert SharedState.abci_app_cls == OptimusAbciApp

    def test_initialization(self) -> None:
        """Test SharedState initialization."""
        mock_context = MagicMock()
        mock_context.params.round_timeout_seconds = 10.0
        mock_context.params.reset_pause_duration = 10
        mock_context.params.validate_timeout = 10.0
        state = SharedState(name="state", skill_context=mock_context)
        assert state is not None

    def test_params_property(self) -> None:
        """Test params property returns context params."""
        mock_context = MagicMock()
        state = SharedState(name="state", skill_context=mock_context)
        # The params property casts context.params to Params
        result = state.params
        assert result is mock_context.params

    def test_setup(self) -> None:
        """Test setup configures event_to_timeout."""
        mock_context = MagicMock()
        mock_context.params.round_timeout_seconds = 10.0
        mock_context.params.reset_pause_duration = 20
        mock_context.params.validate_timeout = 15.0

        state = SharedState(name="state", skill_context=mock_context)

        with patch.object(
            type(state).__bases__[0], "setup"
        ):
            state.setup()

        # Verify that event_to_timeout was populated
        from packages.valory.skills.liquidity_trader_abci.rounds import (
            Event as LiquidityTraderEvent,
        )
        from packages.valory.skills.reset_pause_abci.rounds import (
            Event as ResetPauseEvent,
        )
        from packages.valory.skills.transaction_settlement_abci.rounds import (
            Event as TransactionSettlementEvent,
        )

        assert (
            OptimusAbciApp.event_to_timeout[TransactionSettlementEvent.VALIDATE_TIMEOUT]
            == 15.0
        )
        assert (
            OptimusAbciApp.event_to_timeout[LiquidityTraderEvent.ROUND_TIMEOUT]
            == 10.0 * MULTIPLIER
        )
        assert (
            OptimusAbciApp.event_to_timeout[
                ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT
            ]
            == 20 + MARGIN
        )

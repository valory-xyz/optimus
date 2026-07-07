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

from typing import Any
from unittest.mock import MagicMock, call, patch

from packages.valory.skills.optimus_abci.composition import OptimusAbciApp
from packages.valory.skills.optimus_abci.models import (
    MARGIN,
    MULTIPLIER,
    Params,
    SharedState,
)


def test_import() -> None:
    """Test that the models module can be imported."""
    import packages.valory.skills.optimus_abci.models  # noqa


class TestParams:
    """Test Params class."""

    def test_init_extracts_service_endpoint_base_and_mech_timeout(self) -> None:
        """Test Params __init__ extracts service_endpoint_base and mech_interact_round_timeout_seconds."""
        mock_context = MagicMock()
        mock_context.skill_id = "test_skill/test:0.1.0"

        def _ensure_side_effect(name: str, _kwargs: Any, type_: Any) -> Any:
            if name == "service_endpoint_base":
                return "http://localhost:8000"
            if name == "mech_interact_round_timeout_seconds":
                return 900
            raise AssertionError(f"Unexpected _ensure call for {name}")

        with (
            patch.object(
                Params, "_ensure", side_effect=_ensure_side_effect
            ) as mock_ensure,
            patch.object(Params.__bases__[0], "__init__", return_value=None),
        ):
            params = Params.__new__(Params)
            params.__init__(  # type: ignore[misc]
                skill_context=mock_context,
                service_endpoint_base="http://localhost:8000",
                mech_interact_round_timeout_seconds=900,
            )
            expected_kwargs = {
                "skill_context": mock_context,
                "service_endpoint_base": "http://localhost:8000",
                "mech_interact_round_timeout_seconds": 900,
            }
            assert mock_ensure.call_args_list == [
                call("service_endpoint_base", expected_kwargs, str),
                call(
                    "mech_interact_round_timeout_seconds",
                    expected_kwargs,
                    type_=int,
                ),
            ]
            assert params.service_endpoint_base == "http://localhost:8000"
            assert params.mech_interact_round_timeout_seconds == 900


class TestSharedState:
    """Test SharedState class."""

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
        result = state.params
        assert result is mock_context.params

    def test_setup(self) -> None:
        """Test setup configures event_to_timeout."""
        mock_context = MagicMock()
        mock_context.params.round_timeout_seconds = 10.0
        mock_context.params.reset_pause_duration = 20
        mock_context.params.validate_timeout = 15.0

        state = SharedState(name="state", skill_context=mock_context)

        with patch.object(type(state).__bases__[0], "setup"):
            state.setup()

        from packages.valory.skills.liquidity_trader_abci.rounds import (
            Event as LiquidityTraderEvent,
        )
        from packages.valory.skills.registration_abci.rounds import (
            Event as RegistrationEvent,
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
            OptimusAbciApp.event_to_timeout[ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT]
            == 20 + MARGIN
        )
        assert OptimusAbciApp.event_to_timeout[RegistrationEvent.ROUND_TIMEOUT] == 10.0

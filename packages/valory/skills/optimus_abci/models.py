# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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

"""This module contains the shared state for the abci skill of OptimusAbciApp."""

from typing import Any, Dict, Type, Union, cast

from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.tests.data.dummy_abci.models import (
    RandomnessApi as BaseRandomnessApi,
)
from packages.valory.skills.liquidity_trader_abci.models import (
    Coingecko,
)
from packages.valory.skills.liquidity_trader_abci.models import (
    Params as LiquidityTraderParams,
)
from packages.valory.skills.liquidity_trader_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.liquidity_trader_abci.rounds import (
    Event as LiquidityTraderEvent,
)
from packages.valory.skills.mech_interact_abci.models import (
    MechResponseSpecs as BaseMechResponseSpecs,
)
from packages.valory.skills.mech_interact_abci.models import (
    MechToolsSpecs as BaseMechToolsSpecs,
)
from packages.valory.skills.mech_interact_abci.models import (
    MechsSubgraph as BaseMechsSubgraph,
)
from packages.valory.skills.mech_interact_abci.models import Params as MechParams
from packages.valory.skills.mech_interact_abci.models import (
    SharedState as MechSharedState,
)
from packages.valory.skills.optimus_abci.composition import OptimusAbciApp
from packages.valory.skills.registration_abci.rounds import Event as RegistrationEvent
from packages.valory.skills.reset_pause_abci.rounds import Event as ResetPauseEvent
from packages.valory.skills.termination_abci.models import TerminationParams
from packages.valory.skills.transaction_settlement_abci.rounds import (
    Event as TransactionSettlementEvent,
)

EventType = Union[
    Type[LiquidityTraderEvent],
    Type[TransactionSettlementEvent],
    Type[ResetPauseEvent],
    Type[RegistrationEvent],
]
EventToTimeoutMappingType = Dict[
    Union[
        LiquidityTraderEvent,
        TransactionSettlementEvent,
        ResetPauseEvent,
        RegistrationEvent,
    ],
    float,
]

Coingecko = Coingecko  # re-export: skill.yaml looks up `Coingecko` on this module
Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool

RandomnessApi = BaseRandomnessApi

# Re-exports so skill.yaml can resolve the mech_interact_abci API-spec models on
# this composition skill's `models` module (the mech behaviours read them via
# ``self.context.mech_response`` / ``mech_tools`` / ``mechs_subgraph``).
MechResponseSpecs = BaseMechResponseSpecs
MechToolsSpecs = BaseMechToolsSpecs
MechsSubgraph = BaseMechsSubgraph

MARGIN = 5
MULTIPLIER = 40


class Params(  # pylint: disable=too-many-ancestors
    TerminationParams,
    LiquidityTraderParams,
    MechParams,
):
    """A model to represent params for multiple abci apps.

    Also mixes in ``MechParams`` so the composed ``mech_interact_abci``
    behaviours can read their marketplace/request config off the single shared
    ``self.context.params`` in this composition skill.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init"""
        self.service_endpoint_base = self._ensure("service_endpoint_base", kwargs, str)
        super().__init__(*args, **kwargs)


class SharedState(BaseSharedState, MechSharedState):
    """Keep the current shared state of the skill.

    Mixes in the ``mech_interact_abci`` shared state so the composed mech
    behaviours can use ``penalized_mechs`` / ``last_called_mech`` /
    ``last_failure_reason`` off the single live ``self.context.state``.
    """

    abci_app_cls = OptimusAbciApp  # type: ignore

    @property
    def params(self) -> Params:
        """Get the parameters."""
        return cast(Params, self.context.params)

    def setup(self) -> None:
        """Set up."""
        super().setup()

        events = (
            LiquidityTraderEvent,
            TransactionSettlementEvent,
            ResetPauseEvent,
            RegistrationEvent,
        )
        round_timeout = self.params.round_timeout_seconds
        round_timeout_overrides = {
            cast(EventType, event).ROUND_TIMEOUT: round_timeout for event in events
        }
        reset_pause_timeout = self.params.reset_pause_duration + MARGIN
        event_to_timeout_overrides: EventToTimeoutMappingType = {
            **round_timeout_overrides,
            TransactionSettlementEvent.RESET_TIMEOUT: round_timeout,
            TransactionSettlementEvent.VALIDATE_TIMEOUT: self.params.validate_timeout,
            LiquidityTraderEvent.ROUND_TIMEOUT: self.params.round_timeout_seconds
            * MULTIPLIER,
            ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT: reset_pause_timeout,
        }

        for event, override in event_to_timeout_overrides.items():
            OptimusAbciApp.event_to_timeout[event] = override

# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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

"""This module contains the behaviour for writing apr related data to database for the 'liquidity_trader_abci' skill."""

from typing import Generator, Type

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    LiquidityTraderBaseBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.payloads import APRPopulationPayload
from packages.valory.skills.liquidity_trader_abci.states.apr_population import (
    APRPopulationRound,
)


class APRPopulationBehaviour(LiquidityTraderBaseBehaviour):
    """Behavior for APR population round - APR now comes from subgraph."""

    matching_round: Type[AbstractRound] = APRPopulationRound

    def async_act(self) -> Generator:
        """Async act - simplified since APR now comes from subgraph."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            payload_context = "APR Population"

            # APR calculation is now handled by the subgraph
            # This round is kept for compatibility but does minimal work
            self.context.logger.info(
                "APR Population round - APR now calculated by subgraph"
            )

            payload = APRPopulationPayload(sender=sender, context=payload_context)

            with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
                yield from self.send_a2a_transaction(payload)
                yield from self.wait_until_round_end()

            self.set_done()

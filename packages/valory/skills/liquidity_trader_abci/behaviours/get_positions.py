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

"""This module contains the behaviour for fetching the positions for the 'liquidity_trader_abci' skill."""

import json
from typing import Generator, Type

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    LiquidityTraderBaseBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.states.get_positions import (
    GetPositionsPayload,
    GetPositionsRound,
)


class GetPositionsBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that gets the balances of the assets of agent safes."""

    matching_round: Type[AbstractRound] = GetPositionsRound
    current_positions = None

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            if not self.assets:
                self.assets = self.params.initial_assets
                self.store_assets()

            positions = yield from self.get_positions()
            yield from self._adjust_current_positions_for_backward_compatibility(
                self.current_positions
            )

            self.context.logger.info(f"POSITIONS: {positions}")
            sender = self.context.agent_address

            if positions is None:
                positions = GetPositionsRound.ERROR_PAYLOAD

            serialized_positions = json.dumps(positions, sort_keys=True)
            payload = GetPositionsPayload(sender=sender, positions=serialized_positions)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

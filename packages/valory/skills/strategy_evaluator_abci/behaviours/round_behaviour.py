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

"""This module contains the round behaviour for the 'strategy_evaluator_abci' skill."""

from typing import Set, Type

from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.strategy_evaluator_abci.behaviours.backtesting import (
    BacktestBehaviour,
)
from packages.valory.skills.strategy_evaluator_abci.behaviours.prepare_swap_tx import (
    PrepareEvmSwapBehaviour,
    PrepareSwapBehaviour,
)
from packages.valory.skills.strategy_evaluator_abci.behaviours.proxy_swap_queue import (
    ProxySwapQueueBehaviour,
)
from packages.valory.skills.strategy_evaluator_abci.behaviours.strategy_exec import (
    StrategyExecBehaviour,
)
from packages.valory.skills.strategy_evaluator_abci.behaviours.swap_queue import (
    SwapQueueBehaviour,
)
from packages.valory.skills.strategy_evaluator_abci.rounds import (
    StrategyEvaluatorAbciApp,
)


class AgentStrategyEvaluatorRoundBehaviour(AbstractRoundBehaviour):
    """This behaviour manages the consensus stages for the strategy evaluation."""

    initial_behaviour_cls = StrategyExecBehaviour
    abci_app_cls = StrategyEvaluatorAbciApp
    behaviours: Set[Type[BaseBehaviour]] = {
        StrategyExecBehaviour,  # type: ignore
        PrepareSwapBehaviour,  # type: ignore
        PrepareEvmSwapBehaviour,  # type: ignore
        SwapQueueBehaviour,  # type: ignore
        ProxySwapQueueBehaviour,  # type: ignore
        BacktestBehaviour,  # type: ignore
    }

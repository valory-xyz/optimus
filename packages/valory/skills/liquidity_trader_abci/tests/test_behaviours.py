# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2023 Valory AG
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

"""Tests for valory/liquidity_trader_abci skill's behaviours."""

# pylint: skip-file

import json
import time
from pathlib import Path
from typing import Any, Type, cast
from unittest import mock


from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.abstract_round_abci.behaviour_utils import BaseBehaviour
from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.registration_abci.behaviours import (
    RegistrationBaseBehaviour,
)
from packages.valory.skills.reset_pause_abci.behaviours import (
    ResetAndPauseBehaviour,
)
from packages.valory.skills.liquidity_trader_abci import PUBLIC_ID
from packages.valory.skills.liquidity_trader_abci.behaviours import (
    CallCheckpointBehaviour,
    CheckStakingKPIMetBehaviour,
    GetPositionsBehaviour,
    EvaluateStrategyBehaviour,
    DecisionMakingBehaviour,
    PostTxSettlementBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.rounds import Event, SynchronizedData


PACKAGE_DIR = Path(__file__).parent.parent


def test_skill_public_id() -> None:
    """Test skill module public ID"""

    assert PUBLIC_ID.name == Path(__file__).parents[1].name
    assert PUBLIC_ID.author == Path(__file__).parents[3].name


class LiquidityTraderAbciFSMBehaviourBaseCase(FSMBehaviourBaseCase):
    """Base case for testing FSMBehaviour."""

    path_to_skill = PACKAGE_DIR

    def setup(self, **kwargs: Any) -> None:
        """
        Set up the test method.

        Called each time before a test method is called.

        :param kwargs: the keyword arguments passed to _prepare_skill
        """
        super().setup(**kwargs)
        self.synchronized_data = SynchronizedData(
            AbciAppDB(
                
            )
        )

    def end_round(  # type: ignore
        self,
    ) -> None:
        """Ends round early to cover `wait_for_end` generator."""
        super().end_round(Event.DONE)


class BaseCallCheckpointBehaviourTest(LiquidityTraderAbciFSMBehaviourBaseCase):
    """Test Behaviour."""

    behaviour_class: Type[BaseBehaviour]
    next_behaviour_class: Type[BaseBehaviour]


class TestRegistrationBehaviour(LiquidityTraderAbciFSMBehaviourBaseCase):
    """Test case to test RegistrationBehaviour."""

    def test_registration(self) -> None:
        """Test registration."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            RegistrationBaseBehaviour.auto_behaviour_id(),
            self.synchronized_data,
        )
        assert (
            cast(
                BaseBehaviour,
                cast(BaseBehaviour, self.behaviour.current_behaviour),
            ).behaviour_id
            == RegistrationBaseBehaviour.auto_behaviour_id()
        )
        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()

        self.end_round()
        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == CallCheckpointBehaviour.auto_behaviour_id()


class TestCallCheckpointBehaviour(BaseCallCheckpointBehaviourTest):
    """Test CallCheckpointBehaviour."""

    call_checkpoint_behaviour_class = CallCheckpointBehaviour
    next_behaviour_class = CheckStakingKPIMetBehaviour


class TestResetAndPauseBehaviour(LiquidityTraderAbciFSMBehaviourBaseCase):
    """Test ResetBehaviour."""

    behaviour_class = ResetAndPauseBehaviour
    next_behaviour_class = CallCheckpointBehaviour

    def test_reset_behaviour(
        self,
    ) -> None:
        """Test reset behaviour."""
        self.fast_forward_to_behaviour(
            behaviour=self.behaviour,
            behaviour_id=self.behaviour_class.auto_behaviour_id(),
            synchronized_data=self.synchronized_data,
        )
        assert self.behaviour.current_behaviour is not None
        self.behaviour.current_behaviour.pause = False
        assert (
            cast(
                BaseBehaviour,
                cast(BaseBehaviour, self.behaviour.current_behaviour),
            ).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )
        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round()
        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.next_behaviour_class.auto_behaviour_id()
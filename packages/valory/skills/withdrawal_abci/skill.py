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

"""This module contains the withdrawal skill for the 'optimus' agent."""

from typing import Any

from aea.configurations.base import PublicId
from aea.skills.base import Skill

from packages.valory.skills.withdrawal_abci.behaviours import WithdrawalBehaviour
from packages.valory.skills.withdrawal_abci.handlers import WithdrawalHttpHandler
from packages.valory.skills.withdrawal_abci.models import WithdrawalParams, WithdrawalSharedState


class WithdrawalSkill(Skill):
    """Non-ABCI skill for handling withdrawal requests independently."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the withdrawal skill."""
        super().__init__(**kwargs)
        self.withdrawal_behaviour = WithdrawalBehaviour(
            name="withdrawal_behaviour",
            skill_context=self.context,
        )
        self.withdrawal_http_handler = WithdrawalHttpHandler(
            name="withdrawal_http_handler",
            skill_context=self.context,
        )
        self.withdrawal_params = WithdrawalParams(**kwargs)
        self.withdrawal_shared_state = WithdrawalSharedState(**kwargs)

    @property
    def behaviours(self) -> set:
        """Get the behaviours."""
        return {self.withdrawal_behaviour}

    @property
    def handlers(self) -> set:
        """Get the handlers."""
        return {self.withdrawal_http_handler}

    @property
    def models(self) -> set:
        """Get the models."""
        return {self.withdrawal_params, self.withdrawal_shared_state}

    def setup(self) -> None:
        """Set up the skill."""
        self.context.logger.info("Withdrawal skill setup complete")

    def teardown(self) -> None:
        """Tear down the skill."""
        self.context.logger.info("Withdrawal skill teardown complete") 
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

"""This package contains the implemenatation of the StrategyBehaviour interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator

from packages.valory.skills.abstract_round_abci.behaviours import BaseBehaviour


class StrategyBehaviour(BaseBehaviour, ABC):
    """StrategyBehaviour"""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize `StrategyBehaviour`."""
        super().__init__(**kwargs)

    @abstractmethod
    def get_decision(self) -> Dict[str, str]:
        """Get the decision on whether to enter a pool or not"""
        pass

    def async_act(self) -> Generator[Any, None, None]:
        """Async act"""
        pass

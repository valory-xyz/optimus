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

"""This module contains the shared state for the skill of LlmInteraction."""
from typing import Any, Dict, Callable, Tuple

from aea.exceptions import enforce
from aea.skills.base import Model

from packages.valory.skills.abstract_round_abci.utils import check_type

class Params(Model):
    """Parameters"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init"""
        self.in_flight_req: bool = False
        self.req_to_callback: Dict[str, Tuple[Callable, Dict[str, Any]]] = {}
        self.request_count: int = 0
        self.request_queue = []
        self.cleanup_freq = kwargs.get("cleanup_freq", 50)
        self.waiting_time = self._ensure_get(
            "waiting_time", kwargs, int
        )
        self.available_strategies = self._ensure_get(
            "available_strategies", kwargs, list
        )
        self.service_endpoint_base = self._ensure_get("service_endpoint_base", kwargs, str)
        super().__init__(*args, **kwargs)

    @classmethod
    def _ensure_get(cls, key: str, kwargs: Dict, type_: Any) -> Any:
        """Ensure that the parameters are set, and return them without popping the key."""
        enforce("skill_context" in kwargs, "Only use on models!")
        skill_id = kwargs["skill_context"].skill_id
        enforce(
            key in kwargs,
            f"{key!r} of type {type_!r} required, but it is not set in `models.params.args` of `skill.yaml` of `{skill_id}`",
        )
        value = kwargs.get(key, None)
        try:
            check_type(key, value, type_)
        except TypeError:  # pragma: nocover
            enforce(
                False,
                f"{key!r} must be a {type_}, but type {type(value)} was found in `models.params.args` of `skill.yaml` of `{skill_id}`",
            )
        return value
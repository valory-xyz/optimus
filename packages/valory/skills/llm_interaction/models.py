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

from aea.skills.base import Model

from packages.valory.skills.abstract_round_abci.utils import check_type
from aea.exceptions import enforce


class LlmInteractionParams(Model):
    """Parameters"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init"""
        self.in_flight_req: bool = False
        self.req_to_callback: Dict[str, Tuple[Callable, Dict[str, Any]]] = {}
        self.request_count: int = 0
        self.request_queue = []
        self.cleanup_freq = kwargs.get("cleanup_freq", 50)

        store_path = kwargs.get("store_path", None)
        enforce(store_path is not None, "store path not specified!")
        self.store_path = store_path

        service_endpoint_base = kwargs.get("service_endpoint_base", None)
        enforce(service_endpoint_base is not None, "service_endpoint_base not specified!")
        self.service_endpoint_base = service_endpoint_base

        available_strategies = kwargs.get("available_strategies", None)
        enforce(available_strategies is not None, "available_strategies not specified!")
        self.available_strategies = available_strategies

        portfolio_info_filename = kwargs.get("portfolio_info_filename", None)
        enforce(portfolio_info_filename is not None, "portfolio_info_filename not specified!")
        self.portfolio_info_filename = portfolio_info_filename

        default_acceptance_time = kwargs.get("default_acceptance_time", None)
        enforce(default_acceptance_time is not None, "default_acceptance_time not specified!")
        self.default_acceptance_time = service_endpoint_base

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
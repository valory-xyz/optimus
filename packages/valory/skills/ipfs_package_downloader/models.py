# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2024 Valory AG
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

"""This module contains the shared state for the skill."""
from collections import defaultdict
from typing import Any, Callable, Dict, List, cast

from aea.exceptions import enforce
from aea.skills.base import Model


class Params(Model):
    """A model to represent params for multiple abci apps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""

        self.in_flight_req: bool = False
        self.req_to_callback: Dict[str, Callable] = {}
        self.file_hash_to_id: Dict[
            str, List[str]
        ] = self._nested_list_todict_workaround(
            kwargs,
            "file_hash_to_id",
        )
        self.request_count: int = 0
        self.cleanup_freq = kwargs.get("cleanup_freq", 50)
        self.timeout_limit = kwargs.get("timeout_limit", None)
        self.component_yaml_filename = kwargs.get("component_yaml_filename", None)
        self.entry_point_key = kwargs.get("entry_point_key", None)
        self.callable_keys = kwargs.get("callable_keys", None)
        enforce(self.timeout_limit is not None, "'timeout_limit' must be set!")
        enforce(
            self.component_yaml_filename is not None,
            "'component_yaml_filename' must be set!",
        )
        enforce(self.entry_point_key is not None, "'entry_point_key' must be set!")
        enforce(self.callable_keys is not None, "'callable_keys' must be set!")
        # maps the request id to the number of times it has timed out
        self.request_id_to_num_timeouts: Dict[int, int] = defaultdict(lambda: 0)
        super().__init__(*args, **kwargs)

    def _nested_list_todict_workaround(
        self,
        kwargs: Dict,
        key: str,
    ) -> Dict:
        """Get a nested list from the kwargs and convert it to a dictionary."""
        values = cast(List, kwargs.get(key))
        if len(values) == 0:
            raise ValueError(f"No {key} specified!")
        return {value[0]: value[1] for value in values}

# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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

"""This package contains the implementation of a custom component's management."""

import time
from asyncio import Future
from typing import Any, Callable, Dict, Optional, Tuple, cast

import yaml
from aea.protocols.base import Message
from aea.protocols.dialogue.base import Dialogue
from aea.skills.behaviours import SimpleBehaviour

from packages.valory.connections.ipfs.connection import IpfsDialogues
from packages.valory.connections.ipfs.connection import PUBLIC_ID as IPFS_CONNECTION_ID
from packages.valory.protocols.ipfs import IpfsMessage
from packages.valory.protocols.ipfs.dialogues import IpfsDialogue
from packages.valory.skills.ipfs_package_downloader.models import Params


COMPONENT_YAML_STORE_KEY = "component_yaml"
ENTRY_POINT_STORE_KEY = "entry_point"
CALLABLES_STORE_KEY = "callables"


class IpfsPackageDownloader(SimpleBehaviour):
    """A class to download packages from IPFS."""

    def __init__(self, **kwargs: Any):
        """Initialise the agent."""
        super().__init__(**kwargs)
        self._executing_task: Optional[Dict[str, Optional[float]]] = None
        self._packages_to_file_hash: Dict[str, str] = {}
        self._all_packages: Dict[str, Dict[str, str]] = {}
        self._inflight_package_req: Optional[str] = None
        self._last_polling: Optional[float] = None
        self._invalid_request = False
        self._async_result: Optional[Future] = None

    def setup(self) -> None:
        """Implement the setup."""
        self.context.logger.info("Setting up IpfsPackageDownloader")
        self._packages_to_file_hash = {
            value: key
            for key, values in self.params.file_hash_to_id.items()
            for value in values
        }

    def act(self) -> None:
        """Implement the act."""
        self._download_packages()

    @property
    def params(self) -> Params:
        """Get the parameters."""
        return cast(Params, self.context.params)

    @property
    def request_id_to_num_timeouts(self) -> Dict[int, int]:
        """Maps the request id to the number of times it has timed out."""
        return self.params.request_id_to_num_timeouts

    def count_timeout(self, request_id: int) -> None:
        """Increase the timeout for a request."""
        self.request_id_to_num_timeouts[request_id] += 1

    def timeout_limit_reached(self, request_id: int) -> bool:
        """Check if the timeout limit has been reached."""
        return self.params.timeout_limit <= self.request_id_to_num_timeouts[request_id]

    def _has_executing_task_timed_out(self) -> bool:
        """Check if the executing task timed out."""
        if self._executing_task is None:
            return False
        timeout_deadline = self._executing_task.get("timeout_deadline", None)
        if timeout_deadline is None:
            return False
        return timeout_deadline <= time.time()

    def _download_packages(self) -> None:
        """Download packages."""
        if self._inflight_package_req is not None:
            # there already is a req in flight
            return
        if len(self._packages_to_file_hash) == len(self._all_packages):
            # we already have all the packages
            return
        for package, file_hash in self._packages_to_file_hash.items():
            if package in self._all_packages:
                continue
            # read one at a time
            ipfs_msg, message = self._build_ipfs_get_file_req(file_hash)
            self._inflight_package_req = package
            self.send_message(ipfs_msg, message, self._handle_get_package)
            return

    def load_custom_component(
        self, serialized_objects: Dict[str, str]
    ) -> Dict[str, Any]:
        """Load a custom component package.

        :param serialized_objects: the serialized objects.
        :return: the component.yaml, entry_point.py and callable as tuple.
        """
        # the package MUST contain a component.yaml file
        if self.params.component_yaml_filename not in serialized_objects:
            self.context.logger.error(
                "Invalid component package. "
                f"The package MUST contain a {self.params.component_yaml_filename}."
            )
            return {}
        # load the component.yaml file
        component_yaml = yaml.safe_load(
            serialized_objects[self.params.component_yaml_filename]
        )
        if self.params.entry_point_key not in component_yaml or not all(
            callable_key in component_yaml for callable_key in self.params.callable_keys
        ):
            self.context.logger.error(
                f"Invalid component package. The {self.params.component_yaml_filename} file MUST contain the "
                f"{self.params.entry_point_key} and {self.params.callable_keys} keys."
            )
            return {}
        # the name of the script that needs to be executed
        entry_point_name = component_yaml[self.params.entry_point_key]
        # load the script
        if entry_point_name not in serialized_objects:
            self.context.logger.error(
                f"Invalid component package. "
                f"The entry point {entry_point_name!r} is not present in the component package."
            )
            return {}
        entry_point = serialized_objects[entry_point_name]
        # initialize with the methods that need to be called
        component = {
            callable_key: component_yaml[callable_key]
            for callable_key in self.params.callable_keys
        }
        component.update(
            {
                COMPONENT_YAML_STORE_KEY: component_yaml,
                ENTRY_POINT_STORE_KEY: entry_point,
            }
        )
        return component

    def _handle_get_package(self, message: IpfsMessage, _dialogue: Dialogue) -> None:
        """Handle get package response"""
        package_req = cast(str, self._inflight_package_req)
        self._all_packages[package_req] = message.files
        self.context.shared_state[package_req] = self.load_custom_component(
            message.files
        )
        self._inflight_package_req = None

    def send_message(
        self, msg: Message, dialogue: Dialogue, callback: Callable
    ) -> None:
        """Send message."""
        self.context.outbox.put_message(message=msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        self.params.req_to_callback[nonce] = callback
        self.params.in_flight_req = True

    def _build_ipfs_message(
        self,
        performative: IpfsMessage.Performative,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[IpfsMessage, IpfsDialogue]:
        """Builds an IPFS message."""
        ipfs_dialogues = cast(IpfsDialogues, self.context.ipfs_dialogues)
        message, dialogue = ipfs_dialogues.create(
            counterparty=str(IPFS_CONNECTION_ID),
            performative=performative,
            timeout=timeout,
            **kwargs,
        )
        return message, dialogue

    def _build_ipfs_get_file_req(
        self,
        ipfs_hash: str,
        timeout: Optional[float] = None,
    ) -> Tuple[IpfsMessage, IpfsDialogue]:
        """
        Builds a GET_FILES IPFS request.

        :param ipfs_hash: the ipfs hash of the file/dir to download.
        :param timeout: timeout for the request.
        :returns: the ipfs message, and its corresponding dialogue.
        """
        message, dialogue = self._build_ipfs_message(
            performative=IpfsMessage.Performative.GET_FILES,  # type: ignore
            ipfs_hash=ipfs_hash,
            timeout=timeout,
        )
        return message, dialogue

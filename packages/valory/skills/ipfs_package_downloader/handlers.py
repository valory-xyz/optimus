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

"""This package contains a scaffold of a handler."""

from typing import cast

from aea.protocols.base import Message
from aea.skills.base import Handler

from packages.valory.protocols.ipfs import IpfsMessage
from packages.valory.skills.ipfs_package_downloader.models import Params


class BaseHandler(Handler):
    """Base Handler"""

    def setup(self) -> None:
        """Set up the handler."""
        self.context.logger.info(f"{self.__class__.__name__}: setup method called.")

    def cleanup_dialogues(self) -> None:
        """Clean up all dialogues."""
        for handler_name in self.context.handlers.__dict__.keys():
            dialogues_name = handler_name.replace("_handler", "_dialogues")
            dialogues = getattr(self.context, dialogues_name)
            dialogues.cleanup()

    @property
    def params(self) -> Params:
        """Get the parameters."""
        return cast(Params, self.context.params)

    def teardown(self) -> None:
        """Teardown the handler."""
        self.context.logger.info(f"{self.__class__.__name__}: teardown called.")

    def on_message_handled(self, _message: Message) -> None:
        """Callback after a message has been handled."""
        self.params.request_count += 1
        if self.params.request_count % self.params.cleanup_freq == 0:
            self.context.logger.info(
                f"{self.params.request_count} requests processed. Cleaning up dialogues."
            )
            self.cleanup_dialogues()


class IpfsHandler(BaseHandler):
    """IPFS API message handler."""

    SUPPORTED_PROTOCOL = IpfsMessage.protocol_id

    def handle(self, message: Message) -> None:
        """
        Implement the reaction to an IPFS message.

        :param message: the message
        """
        self.context.logger.info(f"Received message: {message}")
        ipfs_msg = cast(IpfsMessage, message)
        if ipfs_msg.performative == IpfsMessage.Performative.ERROR:
            self.context.logger.warning(
                f"IPFS Message performative not recognized: {ipfs_msg.performative}"
            )
            self.params.in_flight_req = False
            return

        dialogue = self.context.ipfs_dialogues.update(ipfs_msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        callback = self.params.req_to_callback.pop(nonce)
        callback(ipfs_msg, dialogue)
        self.params.in_flight_req = False
        self.on_message_handled(message)

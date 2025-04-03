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

"""This module contains the handlers for the skill of LiquidityTraderAbciApp."""

from typing import Optional, cast

from aea.configurations.data_types import PublicId
from aea.protocols.base import Message

from packages.dvilela.protocols.kv_store.message import KvStoreMessage
from packages.valory.protocols.srr.message import SrrMessage
from packages.valory.protocols.ipfs import IpfsMessage
from packages.valory.skills.abstract_round_abci.handlers import (
    ABCIRoundHandler as BaseABCIRoundHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import AbstractResponseHandler
from packages.valory.skills.abstract_round_abci.handlers import (
    ContractApiHandler as BaseContractApiHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    HttpHandler as BaseHttpHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    LedgerApiHandler as BaseLedgerApiHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    SigningHandler as BaseSigningHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    TendermintHandler as BaseTendermintHandler,
)
from packages.valory.skills.liquidity_trader_abci.models import SharedState


ABCIHandler = BaseABCIRoundHandler
HttpHandler = BaseHttpHandler
SigningHandler = BaseSigningHandler
LedgerApiHandler = BaseLedgerApiHandler
ContractApiHandler = BaseContractApiHandler
TendermintHandler = BaseTendermintHandler


class IpfsHandler(AbstractResponseHandler):
    """IPFS message handler."""

    SUPPORTED_PROTOCOL = IpfsMessage.protocol_id
    allowed_response_performatives = frozenset({IpfsMessage.Performative.IPFS_HASH})
    custom_support_performative = IpfsMessage.Performative.FILES

    @property
    def shared_state(self) -> SharedState:
        """Get the parameters."""
        return cast(SharedState, self.context.state)

    def handle(self, message: IpfsMessage) -> None:
        """
        Implement the reaction to an IPFS message.

        :param message: the message
        :return: None
        """
        self.context.logger.debug(f"Received message: {message}")
        self.shared_state.in_flight_req = False

        if message.performative != self.custom_support_performative:
            return super().handle(message)

        dialogue = self.context.ipfs_dialogues.update(message)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        callback = self.shared_state.req_to_callback.pop(nonce)
        callback(message, dialogue)


class SrrHandler(AbstractResponseHandler):
    """Handler for managing Srr messages."""

    SUPPORTED_PROTOCOL: Optional[PublicId] = SrrMessage.protocol_id
    allowed_response_performatives = frozenset(
        {
            SrrMessage.Performative.REQUEST,
            SrrMessage.Performative.RESPONSE,
        }
    )

    def handle(self, message: Message) -> None:
        """
        React to an SRR message.

        :param message: the SrrMessage instance
        """
        self.context.logger.info(f"Received Srr message: {message}")
        srr_msg = cast(SrrMessage, message)
        if srr_msg.performative not in self.allowed_response_performatives:
            self.context.logger.warning(
                f"SRR performative not recognized: {srr_msg.performative}"
            )
            self.context.state.in_flight_req = False
            return

        dialogue = self.context.srr_dialogues.update(srr_msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        callback, kwargs = self.context.state.req_to_callback.pop(nonce)
        callback(srr_msg, dialogue, **kwargs)

        self.context.state.in_flight_req = False
        self.on_message_handled(message)
        

class KvStoreHandler(AbstractResponseHandler):
    """A class for handling KeyValue messages."""

    SUPPORTED_PROTOCOL: Optional[PublicId] = KvStoreMessage.protocol_id
    allowed_response_performatives = frozenset(
        {
            KvStoreMessage.Performative.READ_REQUEST,
            KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST,
            KvStoreMessage.Performative.READ_RESPONSE,
            KvStoreMessage.Performative.SUCCESS,
            KvStoreMessage.Performative.ERROR,
        }
    )

    def handle(self, message: KvStoreMessage) -> None:
        """
        React to an KvStore message.

        :param message: the KvStoreMessage instance
        """
        self.context.logger.info(f"Received KvStore message: {message}")
        kv_store_msg = cast(KvStoreMessage, message)

        if kv_store_msg.performative not in self.allowed_response_performatives:
            self.context.logger.warning(
                f"KvStore performative not recognized: {kv_store_msg.performative}"
            )
            self.context.state.in_flight_req = False
            return

        if kv_store_msg.performative == KvStoreMessage.Performative.SUCCESS:
            dialogue = self.context.kv_store_dialogues.update(kv_store_msg)
            nonce = dialogue.dialogue_label.dialogue_reference[0]
            callback, kwargs = self.context.state.req_to_callback.pop(nonce)
            callback(kv_store_msg, dialogue, **kwargs)

            self.context.state.in_flight_req = False
            return
        else:
            super().handle(message)

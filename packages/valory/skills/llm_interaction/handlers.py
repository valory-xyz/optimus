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

"""This module contains the handlers for the skill of LlmInteraction."""

import json
import threading
import time
import re
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union, cast, Generator
from urllib.parse import urlparse

from aea.protocols.base import Message
from aea.skills.base import Handler

from aea.configurations.data_types import PublicId
from packages.valory.connections.http_server.connection import (
    PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
)
from packages.valory.protocols.http.message import HttpMessage
from packages.valory.skills.abstract_round_abci.handlers import (
    HttpHandler as BaseHttpHandler,
)
from packages.dvilela.connections.kv_store.connection import (
    PUBLIC_ID as KV_STORE_CONNECTION_PUBLIC_ID,
)
from packages.dvilela.protocols.kv_store.dialogues import (
    KvStoreDialogue,
    KvStoreDialogues,
)
from packages.dvilela.protocols.kv_store.message import KvStoreMessage
from packages.valory.skills.liquidity_trader_abci.rounds import SynchronizedData
from packages.valory.skills.llm_interaction.dialogues import HttpDialogue, HttpDialogues
from packages.valory.skills.optimus_abci.models import SharedState
from packages.valory.skills.abstract_round_abci.models import Requests
from packages.valory.skills.llm_interaction.dialogues import LlmDialogue, LlmDialogues
from packages.valory.skills.llm_interaction.models import Params
from packages.valory.protocols.llm.message import LlmMessage
from packages.valory.skills.abstract_round_abci.handlers import AbstractResponseHandler
from packages.valory.connections.openai.connection import (
    PUBLIC_ID as LLM_CONNECTION_PUBLIC_ID,
)
from aea.protocols.dialogue.base import Dialogue

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

class KvStoreHandler(BaseHandler):
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

    def handle(self, message: Message) -> None:
        """
        Implement the reaction to an KVStore message.

        :param message: the message
        """
        self.context.logger.info(f"Received message: {message}")
        kv_store_msg = cast(KvStoreMessage, message)

        if kv_store_msg.performative not in self.allowed_response_performatives:
            self.context.logger.warning(
                f"KV Store Message performative not recognized: {kv_store_msg.performative}"
            )
            self.params.in_flight_req = False
            return

        dialogue = self.context.kv_store_dialogues.update(kv_store_msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        callback, kwargs = self.params.req_to_callback.pop(nonce)
        callback(kv_store_msg, dialogue, **kwargs)
        self.params.in_flight_req = False
        self.on_message_handled(message)

class HttpCode(Enum):
    """Http codes"""

    OK_CODE = 200
    NOT_FOUND_CODE = 404
    BAD_REQUEST_CODE = 400
    NOT_READY = 503


class HttpMethod(Enum):
    """Http methods"""

    GET = "get"
    HEAD = "head"
    POST = "post"


class HttpHandler(BaseHttpHandler):
    """This implements the HTTP handler."""

    SUPPORTED_PROTOCOL = HttpMessage.protocol_id

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return SynchronizedData(
            db=self.context.state.round_sequence.latest_synchronized_data.db
        )

    def setup(self) -> None:
        """Implement the setup."""

        # Custom hostname (set via params)
        service_endpoint_base = urlparse(
            self.context.params.service_endpoint_base
        ).hostname

        # Propel hostname regex
        propel_uri_base_hostname = (
            r"https?:\/\/[a-zA-Z0-9]{16}.agent\.propel\.(staging\.)?autonolas\.tech"
        )

        # Route regexes
        hostname_regex = rf".*({service_endpoint_base}|{propel_uri_base_hostname}|localhost|127.0.0.1|0.0.0.0)(:\d+)?"
        self.handler_url_regex = rf"{hostname_regex}\/.*"
        health_url_regex = rf"{hostname_regex}\/healthcheck"

        self.routes = {
            (HttpMethod.POST.value,): [
                (rf"{hostname_regex}\/process_prompt", self._handle_post_process_prompt),
            ],
            (HttpMethod.GET.value, HttpMethod.HEAD.value): [
                (health_url_regex, self._handle_get_health),
            ],
        }

        self.json_content_header = "Content-Type: application/json\n"

    def _handle_post_process_prompt(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) ->  Generator[None, None, None]:
        """
        Handle a POST request to process a user prompt using OpenAI.

        :param http_msg: the http message
        :param http_dialogue: the http dialogue
        """
        try:
            prompt_template = self.context.params.llm_prompt
            
            # Parse the request body
            request_data = json.loads(http_msg.body.decode("utf-8"))
            user_prompt = request_data.get("prompt", "")
            # formatted_str = prompt_template.format(user_prompt=user_prompt)
            # self.context.logger.info(f"{formatted_str=}")
            
            if not user_prompt:
                raise ValueError("Prompt is required.")

            # Create LLM request message
            llm_dialogues = cast(LlmDialogues, self.context.llm_dialogues)
            request_llm_message, llm_dialogue = llm_dialogues.create(
                counterparty=str(LLM_CONNECTION_PUBLIC_ID),
                performative=LlmMessage.Performative.REQUEST,
                prompt_template=prompt_template,
                prompt_values={},
            )
            self.context.logger.info(f"LLM request message created. {request_llm_message} {llm_dialogue}")
            kwargs = {
                "http_msg": http_msg,
                "http_dialogue": http_dialogue
            }
            self.send_message(request_llm_message, llm_dialogue, self._handle_llm_response, kwargs)

        except Exception as e:
            self.context.logger.error(f"Error processing prompt: {str(e)}")
            self._handle_bad_request(http_msg, http_dialogue)
    
    def send_message(
        self, msg: Message, dialogue: Dialogue, callback: Callable, kwargs = {}
    ) -> None:
        """Send message."""
        self.context.outbox.put_message(message=msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        self.context.params.req_to_callback[nonce] = (callback, kwargs)
        self.context.params.in_flight_req = True

    def _handle_llm_response(self, llm_response_message: LlmMessage, dialogue: Dialogue, http_msg: HttpMessage, http_dialogue: HttpDialogue) -> None:
        """Handle get tool response"""
        self.context.logger.info(f"{llm_response_message=} {dialogue=}")
        response_data = json.loads(llm_response_message.value)
        strategies = response_data.get("words", [])
        self._send_ok_response(http_msg, http_dialogue, {"response": strategies})
        
        # Use threading to avoid blocking
        threading.Thread(target=self._delayed_write_kv, args=(strategies,)).start()

    def _delayed_write_kv(self, strategies: List[str]) -> None:
        """Write to KV store after a delay."""
        time.sleep(self.context.params.waiting_time)
        self._write_kv({"strategies": json.dumps(strategies, sort_keys=True)})

    def _get_handler(self, url: str, method: str) -> Tuple[Optional[Callable], Dict]:
        """Check if an url is meant to be handled in this handler

        We expect url to match the pattern {hostname}/.*,
        where hostname is allowed to be localhost, 127.0.0.1 or the token_uri_base's hostname.
        Examples:
            localhost:8000/0
            127.0.0.1:8000/100
            https://pfp.staging.autonolas.tech/45
            http://pfp.staging.autonolas.tech/120

        :param url: the url to check
        :param method: the http method
        :returns: the handling method if the message is intended to be handled by this handler, None otherwise, and the regex captures
        """
        # Check base url
        if not re.match(self.handler_url_regex, url):
            self.context.logger.info(
                f"The url {url} does not match the HttpHandler's pattern"
            )
            return None, {}

        # Check if there is a route for this request
        for methods, routes in self.routes.items():
            if method not in methods:
                continue

            for route in routes:  # type: ignore
                # Routes are tuples like (route_regex, handle_method)
                m = re.match(route[0], url)
                if m:
                    return route[1], m.groupdict()

        # No route found
        self.context.logger.info(
            f"The message [{method}] {url} is intended for the HttpHandler but did not match any valid pattern"
        )
        return self._handle_bad_request, {}

    def handle(self, message: Message) -> None:
        """Implement the reaction to an envelope."""
        http_msg = cast(HttpMessage, message)

        if (
            http_msg.performative != HttpMessage.Performative.REQUEST
            or message.sender != str(HTTP_SERVER_PUBLIC_ID.without_hash())
        ):
            super().handle(message)
            return

        handler, kwargs = self._get_handler(http_msg.url, http_msg.method)

        if not handler:
            super().handle(message)
            return

        http_dialogues = cast(HttpDialogues, self.context.http_dialogues)
        http_dialogue = cast(HttpDialogue, http_dialogues.update(http_msg))

        if http_dialogue is None:
            return

        handler(http_msg, http_dialogue, **kwargs)

    def _handle_bad_request(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle a Http bad request.

        :param http_msg: the http message
        :param http_dialogue: the http dialogue
        """
        http_response = http_dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=http_msg,
            version=http_msg.version,
            status_code=HttpCode.BAD_REQUEST_CODE.value,
            status_text="Bad request",
            headers=http_msg.headers,
            body=b"",
        )

        # Send response
        self.context.logger.info("Responding with: {}".format(http_response))
        self.context.outbox.put_message(message=http_response)

    def _send_ok_response(
        self,
        http_msg: HttpMessage,
        http_dialogue: HttpDialogue,
        data: Union[Dict, List],
    ) -> None:
        """Send an OK response with the provided data"""
        http_response = http_dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=http_msg,
            version=http_msg.version,
            status_code=HttpCode.OK_CODE.value,
            status_text="Success",
            headers=f"{self.json_content_header}{http_msg.headers}",
            body=json.dumps(data).encode("utf-8"),
        )

        # Send response
        self.context.logger.info("Responding with: {}".format(http_response))
        self.context.outbox.put_message(message=http_response)

    def _handle_get_health(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle a Http request of verb GET.

        :param http_msg: the http message
        :param http_dialogue: the http dialogue
        """
        seconds_since_last_transition = None
        is_tm_unhealthy = None
        is_transitioning_fast = None
        current_round = None
        rounds = None

        round_sequence = cast(SharedState, self.context.state).round_sequence

        if round_sequence._last_round_transition_timestamp:
            is_tm_unhealthy = cast(
                SharedState, self.context.state
            ).round_sequence.block_stall_deadline_expired

            current_time = datetime.now().timestamp()
            seconds_since_last_transition = current_time - datetime.timestamp(
                round_sequence._last_round_transition_timestamp
            )

            is_transitioning_fast = (
                not is_tm_unhealthy
                and seconds_since_last_transition
                < 2 * self.context.params.reset_pause_duration
            )

        if round_sequence._abci_app:
            current_round = round_sequence._abci_app.current_round.round_id
            rounds = [
                r.round_id for r in round_sequence._abci_app._previous_rounds[-25:]
            ]
            rounds.append(current_round)

        data = {
            "seconds_since_last_transition": seconds_since_last_transition,
            "is_tm_healthy": not is_tm_unhealthy,
            "period": self.synchronized_data.period_count,
            "reset_pause_duration": self.context.params.reset_pause_duration,
            "rounds": rounds,
            "is_transitioning_fast": is_transitioning_fast,
        }

        self._send_ok_response(http_msg, http_dialogue, data)
    
    def _write_kv(
        self,
        data: Dict[str, List[str]],
    ) -> Generator[None, None, bool]:
        """Send a request message from the skill context."""
        self.context.logger.info(f"Preparing to write data to KV store: {data}")
        kv_store_dialogues = cast(KvStoreDialogues, self.context.kv_store_dialogues)
        kv_store_message, srr_dialogue = kv_store_dialogues.create(
            counterparty=str(KV_STORE_CONNECTION_PUBLIC_ID),
            performative=KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST,
            data=data,
        )
        kv_store_message = cast(KvStoreMessage, kv_store_message)
        kv_store_dialogue = cast(KvStoreDialogue, srr_dialogue)
        self.context.logger.info("Sending KV store request message.")
        self.send_message(kv_store_message, kv_store_dialogue, self._handle_kv_store_response)

    def _handle_kv_store_response(self, kv_store_response_message: KvStoreMessage, dialogue: Dialogue) -> None:
        """Handle get tool response"""
        self.context.logger.info(f"{kv_store_response_message=} {dialogue=}")
        success = kv_store_response_message.performative == KvStoreMessage.Performative.SUCCESS
        if success:
            self.context.logger.info("KV store update successful.")
        else:
            self.context.logger.error("KV store update failed.")

class LlmHandler(BaseHandler):
    """A class for handling LLLM messages."""

    SUPPORTED_PROTOCOL: Optional[PublicId] = LlmMessage.protocol_id
    allowed_response_performatives = frozenset(
        {
            LlmMessage.Performative.REQUEST,
            LlmMessage.Performative.RESPONSE,
        }
    )

    def handle(self, message: Message) -> None:
        """
        Implement the reaction to an LLM message.

        :param message: the message
        """
        self.context.logger.info(f"Received message: {message}")
        llm_msg = cast(LlmMessage, message)

        if llm_msg.performative not in self.allowed_response_performatives:
            self.context.logger.warning(
                f"LLM Message performative not recognized: {llm_msg.performative}"
            )
            self.params.in_flight_req = False
            return

        dialogue = self.context.llm_dialogues.update(llm_msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        callback, kwargs = self.params.req_to_callback.pop(nonce)
        callback(llm_msg, dialogue, **kwargs)
        self.params.in_flight_req = False
        self.on_message_handled(message)
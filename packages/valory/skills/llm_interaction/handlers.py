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
"""This module contains the handlers for the LLM interaction skill."""

import json
import re
import threading
import time
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union, cast
from urllib.parse import urlparse

from aea.configurations.data_types import PublicId
from aea.protocols.base import Message
from aea.protocols.dialogue.base import Dialogue
from aea.skills.base import Handler

from packages.dvilela.connections.kv_store.connection import (
    PUBLIC_ID as KV_STORE_CONNECTION_PUBLIC_ID,
)
from packages.dvilela.protocols.kv_store.dialogues import (
    KvStoreDialogue,
    KvStoreDialogues,
)
from packages.dvilela.protocols.kv_store.message import KvStoreMessage
from packages.valory.connections.http_server.connection import (
    PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
)
from packages.valory.connections.openai.connection import (
    PUBLIC_ID as LLM_CONNECTION_PUBLIC_ID,
)
from packages.valory.protocols.http.message import HttpMessage
from packages.valory.protocols.llm.message import LlmMessage
from packages.valory.skills.abstract_round_abci.handlers import (
    HttpHandler as BaseHttpHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import AbstractResponseHandler
from packages.valory.skills.abstract_round_abci.models import Requests
from packages.valory.skills.liquidity_trader_abci.rounds import SynchronizedData
from packages.valory.skills.llm_interaction.dialogues import (
    HttpDialogue,
    HttpDialogues,
    LlmDialogue,
    LlmDialogues,
)
from packages.valory.skills.llm_interaction.models import Params
from packages.valory.skills.optimus_abci.models import SharedState


class BaseHandler(Handler):
    """Base handler providing shared utilities for all handlers."""

    def setup(self) -> None:
        """Set up the handler."""
        self.context.logger.info(f"{self.__class__.__name__}: setup method called.")

    def teardown(self) -> None:
        """Teardown the handler."""
        self.context.logger.info(f"{self.__class__.__name__}: teardown called.")

    @property
    def params(self) -> Params:
        """Shortcut to access the parameters."""
        return cast(Params, self.context.params)

    def cleanup_dialogues(self) -> None:
        """Clean up all dialogues."""
        for handler_name in self.context.handlers.__dict__.keys():
            dialogues_name = handler_name.replace("_handler", "_dialogues")
            dialogues = getattr(self.context, dialogues_name, None)
            if dialogues is not None:
                dialogues.cleanup()

    def on_message_handled(self, _message: Message) -> None:
        """Callback triggered once a message is handled."""
        self.params.request_count += 1
        if self.params.request_count % self.params.cleanup_freq == 0:
            self.context.logger.info(
                f"{self.params.request_count} requests processed. Cleaning up dialogues."
            )
            self.cleanup_dialogues()


class KvStoreHandler(BaseHandler):
    """Handler for managing KV Store messages."""

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
        React to a KvStoreMessage.

        :param message: the KvStoreMessage instance
        """
        self.context.logger.info(f"Received KvStore message: {message}")
        kv_store_msg = cast(KvStoreMessage, message)

        if kv_store_msg.performative not in self.allowed_response_performatives:
            self.context.logger.warning(
                f"KV Store performative not recognized: {kv_store_msg.performative}"
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
    """HTTP status codes enumeration."""

    OK_CODE = 200
    NOT_FOUND_CODE = 404
    BAD_REQUEST_CODE = 400
    NOT_READY = 503


class HttpMethod(Enum):
    """HTTP methods enumeration."""

    GET = "get"
    HEAD = "head"
    POST = "post"


class HttpHandler(BaseHttpHandler):
    """Handler for managing incoming HTTP messages."""

    SUPPORTED_PROTOCOL = HttpMessage.protocol_id

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return SynchronizedData(
            db=self.context.state.round_sequence.latest_synchronized_data.db
        )

    def setup(self) -> None:
        """Set up the HTTP handler and define routes."""
        super().setup()  # Not strictly necessary if BaseHttpHandler implements no logic

        # Prepare hostname regex from parameters
        service_endpoint_base = urlparse(self.context.params.service_endpoint_base).hostname
        propel_uri_base_hostname = (
            r"https?:\/\/[a-zA-Z0-9]{16}.agent\.propel\.(staging\.)?autonolas\.tech"
        )
        hostname_regex = (
            rf".*({service_endpoint_base}|{propel_uri_base_hostname}|localhost|127.0.0.1|0.0.0.0)(:\d+)?"
        )
        health_url_regex = rf"{hostname_regex}\/healthcheck"

        # Define routes
        self.routes = {
            (HttpMethod.POST.value,): [
                (rf"{hostname_regex}\/process_prompt", self._handle_post_process_prompt),
            ],
            (HttpMethod.GET.value, HttpMethod.HEAD.value): [
                (health_url_regex, self._handle_get_health),
            ],
        }
        self.handler_url_regex = rf"{hostname_regex}\/.*"
        self.json_content_header = "Content-Type: application/json\n"

    def handle(self, message: Message) -> None:
        """
        React to an incoming HTTP message.

        :param message: the HttpMessage instance
        """
        http_msg = cast(HttpMessage, message)

        # Check if it's a request from our HTTP server
        if (
            http_msg.performative != HttpMessage.Performative.REQUEST
            or message.sender != str(HTTP_SERVER_PUBLIC_ID.without_hash())
        ):
            super().handle(message)
            return

        # Match handler
        handler, kwargs = self._get_handler(http_msg.url, http_msg.method)
        if not handler:
            super().handle(message)
            return

        # Retrieve or create a dialogue
        http_dialogues = cast(HttpDialogues, self.context.http_dialogues)
        http_dialogue = http_dialogues.update(http_msg)
        if http_dialogue is None:
            return

        # Call matched handler
        handler(http_msg, http_dialogue, **kwargs)

    def _get_handler(self, url: str, method: str) -> Tuple[Optional[Callable], Dict]:
        """
        Determine which route handler should be used.

        :param url: the request URL
        :param method: the request method
        :return: the route handler and regex captures
        """
        if not re.match(self.handler_url_regex, url):
            self.context.logger.info(
                f"The URL {url} does not match the HttpHandler's pattern."
            )
            return None, {}

        for methods, routes in self.routes.items():
            if method not in methods:
                continue
            for route_regex, route_handler in routes:
                match_obj = re.match(route_regex, url)
                if match_obj:
                    return route_handler, match_obj.groupdict()

        self.context.logger.info(
            f"No valid route found for [{method}] {url}."
        )
        return self._handle_bad_request, {}

    def _handle_post_process_prompt(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> Generator[None, None, None]:
        """
        Handle POST requests to process user prompts.

        :param http_msg: the HttpMessage instance
        :param http_dialogue: the HttpDialogue instance
        """
        request_id = http_dialogue.dialogue_label.dialogue_reference[0]
        self.context.params.request_queue.append(request_id)

        try:
            # Parse incoming data
            data = json.loads(http_msg.body.decode("utf-8"))
            user_prompt = data.get("prompt", "")
            if not user_prompt:
                raise ValueError("Prompt is required.")

            # Format the prompt
            prompt_template = self.context.params.llm_prompt
            formatted_prompt = prompt_template.format(
                USER_PROMPT=user_prompt,
                STRATEGIES=self.context.params.available_strategies,
            )

            # Create LLM request
            llm_dialogues = cast(LlmDialogues, self.context.llm_dialogues)
            request_llm_message, llm_dialogue = llm_dialogues.create(
                counterparty=str(LLM_CONNECTION_PUBLIC_ID),
                performative=LlmMessage.Performative.REQUEST,
                prompt_template=formatted_prompt,
                prompt_values={},
            )

            # Prepare callback args
            callback_kwargs = {"http_msg": http_msg, "http_dialogue": http_dialogue}
            self._send_message(
                request_llm_message, llm_dialogue, self._handle_llm_response, callback_kwargs
            )

        except (json.JSONDecodeError, ValueError) as e:
            self.context.logger.error(f"Error processing prompt: {str(e)}")
            self._handle_bad_request(http_msg, http_dialogue)

    def _handle_llm_response(
        self,
        llm_response_message: LlmMessage,
        dialogue: Dialogue,
        http_msg: HttpMessage,
        http_dialogue: HttpDialogue,
    ) -> None:
        """
        Handle the response from the LLM.

        :param llm_response_message: the LlmMessage with the LLM output
        :param dialogue: the LlmDialogue
        :param http_msg: the original HttpMessage
        :param http_dialogue: the original HttpDialogue
        """
        try:
            response_data = json.loads(llm_response_message.value)
            strategies = response_data.get("strategies", [])
            if not strategies:
                # No suitable strategies found
                fallback_message = {
                    "response": (
                        f"No suitable strategies found. "
                        f"Falling back to default list: {self.context.params.available_strategies}"
                    )
                }
                self._send_ok_response(http_msg, http_dialogue, fallback_message)
                strategies = self.context.params.available_strategies
            else:
                self._send_ok_response(http_msg, http_dialogue, {"response": strategies})

            # Offload KV store update to a separate thread
            threading.Thread(target=self._delayed_write_kv, args=(strategies,)).start()

        except json.JSONDecodeError as e:
            self.context.logger.error(
                f"Failed to decode LLM response: {str(e)}"
            )
            self._handle_bad_request(http_msg, http_dialogue)

    def _delayed_write_kv(self, strategies: List[str]) -> None:
        """
        Write to the KV store after a delay if this was the only request in queue.

        :param strategies: list of chosen strategies
        """
        time.sleep(self.context.params.waiting_time)
        if len(self.context.params.request_queue) == 1:
            self._write_kv({"strategies": json.dumps(strategies)})
        self.context.params.request_queue.pop()

    def _write_kv(self, data: Dict[str, str]) -> Generator[None, None, bool]:
        """
        Create or update data in the KV store.

        :param data: key-value data to store
        :return: success flag
        """
        kv_store_dialogues = cast(KvStoreDialogues, self.context.kv_store_dialogues)
        kv_store_message, srr_dialogue = kv_store_dialogues.create(
            counterparty=str(KV_STORE_CONNECTION_PUBLIC_ID),
            performative=KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST,
            data=data,
        )

        kv_store_dialogue = cast(KvStoreDialogue, srr_dialogue)
        self._send_message(
            kv_store_message, kv_store_dialogue, self._handle_kv_store_response
        )

    def _handle_kv_store_response(self, kv_store_response_message: KvStoreMessage, dialogue: Dialogue) -> None:
        """
        Handle KV store response messages.

        :param kv_store_response_message: the KvStoreMessage response
        :param dialogue: the KvStoreDialogue
        """
        success = kv_store_response_message.performative == KvStoreMessage.Performative.SUCCESS
        if success:
            self.context.logger.info("KV store update successful.")
        else:
            self.context.logger.error("KV store update failed.")

    def _handle_get_health(self, http_msg: HttpMessage, http_dialogue: HttpDialogue) -> None:
        """
        Handle GET /healthcheck to provide system health information.

        :param http_msg: the HttpMessage instance
        :param http_dialogue: the HttpDialogue instance
        """
        round_sequence = cast(SharedState, self.context.state).round_sequence
        last_transition_timestamp = round_sequence._last_round_transition_timestamp

        # Compute relevant health info
        if last_transition_timestamp is not None:
            current_time = datetime.now().timestamp()
            seconds_since_last_transition = current_time - datetime.timestamp(last_transition_timestamp)
            is_tm_unhealthy = round_sequence.block_stall_deadline_expired
            is_transitioning_fast = (
                not is_tm_unhealthy
                and seconds_since_last_transition < 2 * self.context.params.reset_pause_duration
            )
        else:
            seconds_since_last_transition = None
            is_tm_unhealthy = None
            is_transitioning_fast = None

        current_round = None
        rounds_list = None
        if round_sequence._abci_app:
            current_round = round_sequence._abci_app.current_round.round_id
            rounds_list = [r.round_id for r in round_sequence._abci_app._previous_rounds[-25:]]
            rounds_list.append(current_round)

        data = {
            "seconds_since_last_transition": seconds_since_last_transition,
            "is_tm_healthy": not is_tm_unhealthy if is_tm_unhealthy is not None else None,
            "period": self.synchronized_data.period_count,
            "reset_pause_duration": self.context.params.reset_pause_duration,
            "rounds": rounds_list,
            "is_transitioning_fast": is_transitioning_fast,
        }
        self._send_ok_response(http_msg, http_dialogue, data)

    def _handle_bad_request(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Respond with BAD REQUEST if an error occurs or no route matches.

        :param http_msg: the HttpMessage instance
        :param http_dialogue: the HttpDialogue instance
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
        self.context.logger.info(f"Responding with: {http_response}")
        self.context.outbox.put_message(message=http_response)

    def _send_ok_response(
        self,
        http_msg: HttpMessage,
        http_dialogue: HttpDialogue,
        data: Union[Dict, List],
    ) -> None:
        """
        Send a 200 OK response with JSON data.

        :param http_msg: the original HttpMessage
        :param http_dialogue: the HttpDialogue
        :param data: the response data
        """
        http_response = http_dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=http_msg,
            version=http_msg.version,
            status_code=HttpCode.OK_CODE.value,
            status_text="Success",
            headers=f"{self.json_content_header}{http_msg.headers}",
            body=json.dumps(data).encode("utf-8"),
        )
        self.context.logger.info(f"Responding with: {http_response}")
        self.context.outbox.put_message(message=http_response)

    def _send_message(
        self,
        message: Message,
        dialogue: Dialogue,
        callback: Callable,
        callback_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Send a message and set up a callback for the response.

        :param message: the Message to send
        :param dialogue: the Dialogue context
        :param callback: the callback function upon response
        :param callback_kwargs: optional kwargs for the callback
        """
        self.context.outbox.put_message(message=message)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        self.context.params.req_to_callback[nonce] = (callback, callback_kwargs or {})
        self.context.params.in_flight_req = True


class LlmHandler(BaseHandler):
    """Handler for managing LLM messages."""

    SUPPORTED_PROTOCOL: Optional[PublicId] = LlmMessage.protocol_id
    allowed_response_performatives = frozenset(
        {
            LlmMessage.Performative.REQUEST,
            LlmMessage.Performative.RESPONSE,
        }
    )

    def handle(self, message: Message) -> None:
        """
        React to an LLM message.

        :param message: the LlmMessage instance
        """
        self.context.logger.info(f"Received LLM message: {message}")
        llm_msg = cast(LlmMessage, message)

        if llm_msg.performative not in self.allowed_response_performatives:
            self.context.logger.warning(
                f"LLM performative not recognized: {llm_msg.performative}"
            )
            self.params.in_flight_req = False
            return

        dialogue = self.context.llm_dialogues.update(llm_msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        callback, kwargs = self.params.req_to_callback.pop(nonce)
        callback(llm_msg, dialogue, **kwargs)

        self.params.in_flight_req = False
        self.on_message_handled(message)
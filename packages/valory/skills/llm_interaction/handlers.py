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
import yaml
from datetime import datetime
from enum import Enum
from pathlib import Path
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
from packages.valory.skills.liquidity_trader_abci.rounds_info import ROUNDS_INFO
from packages.valory.skills.liquidity_trader_abci.rounds import SynchronizedData
from packages.valory.skills.llm_interaction.dialogues import (
    HttpDialogue,
    HttpDialogues,
    LlmDialogue,
    LlmDialogues,
)
from packages.valory.skills.llm_interaction.models import Params
from packages.valory.skills.optimus_abci.models import SharedState
from packages.valory.skills.llm_interaction.prompts import PROMPT
from packages.valory.skills.liquidity_trader_abci.behaviours import THRESHOLDS, TradingType

def load_fsm_spec() -> Dict:
    """Load the chained FSM spec"""
    with open(
        Path(__file__).parent.parent / "optimus_abci" / "fsm_specification.yaml",
        "r",
        encoding="utf-8",
    ) as spec_file:
        return yaml.safe_load(spec_file)


def camel_to_snake(camel_str: str) -> str:
    """Converts from CamelCase to snake_case."""
    import re

    snake_str = re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str).lower()
    return snake_str


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
        portfolio_url_regex = rf"{hostname_regex}\/portfolio"
        static_files_regex = rf"{hostname_regex}\/(.*)" 
        process_prompt_regex = rf"{hostname_regex}\/configure_strategies"

        # Define routes
        self.routes = {
            (HttpMethod.POST.value,): [
                (process_prompt_regex, self._handle_post_process_prompt),
            ],
            (HttpMethod.GET.value, HttpMethod.HEAD.value): [
                (health_url_regex, self._handle_get_health),
                (portfolio_url_regex, self._handle_get_portfolio),
                (static_files_regex, self._handle_get_static_file),
            ],
        }
        fsm = load_fsm_spec()

        self.rounds_info: Dict = {  # pylint: disable=attribute-defined-outside-init
            camel_to_snake(k): v for k, v in ROUNDS_INFO.items()
        }
        for source_info, target_round in fsm["transition_func"].items():
            source_round, event = source_info[1:-1].split(", ")
            self.rounds_info[camel_to_snake(source_round)]["transitions"][
                event.lower()
            ] = camel_to_snake(target_round)

        self.handler_url_regex = rf"{hostname_regex}\/.*"
        self.json_content_header = "Content-Type: application/json\n"
        self.html_content_header = "Content-Type: text/html\n"

    def handle(self, message: Message) -> None:
        """
        Implement the reaction to an envelope.

        :param message: the message
        """
        http_msg = cast(HttpMessage, message)

        # Check if this is a request sent from the http_server skill
        if (
            http_msg.performative != HttpMessage.Performative.REQUEST
            or message.sender != str(HTTP_SERVER_PUBLIC_ID.without_hash())
        ):
            super().handle(message)
            return

        # Check if this message is for this skill. If not, send to super()
        handler, kwargs = self._get_handler(http_msg.url, http_msg.method)
        if not handler:
            super().handle(message)
            return

        # Retrieve dialogues
        http_dialogues = cast(HttpDialogues, self.context.http_dialogues)
        http_dialogue = cast(HttpDialogue, http_dialogues.update(http_msg))

        # Invalid message
        if http_dialogue is None:
            self.context.logger.info(
                "Received invalid http message={}, unidentified dialogue.".format(
                    http_msg
                )
            )
            return

        # Handle message
        self.context.logger.info(
            "Received http request with method={}, url={} and body={!r}".format(
                http_msg.method,
                http_msg.url,
                http_msg.body,
            )
        )
        handler(http_msg, http_dialogue, **kwargs)

    def _handle_get_static_file(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle a HTTP GET request for a static file.

        :param http_msg: the HTTP message
        :param http_dialogue: the HTTP dialogue
        """
        try:
            # Extract the requested path from the URL
            requested_path = urlparse(http_msg.url).path.lstrip("/")
            self.context.logger.debug(f"Requested path: {requested_path}")

            # Construct the file path
            file_path = Path(Path(__file__).parent, "modius-ui-build", requested_path)
            self.context.logger.debug(f"Constructed file path: {file_path}")

            # If the file exists and is a file, send it as a response
            if file_path.exists() and file_path.is_file():
                self.context.logger.debug(f"File found: {file_path}")
                with open(file_path, "rb") as file:
                    file_content = file.read()

                # Send the file content as a response
                self._send_ok_response(http_msg, http_dialogue, file_content)
            else:
                self.context.logger.debug(f"File not found or is not a file: {file_path}")
                # If the file doesn't exist or is not a file, return the index.html file
                index_file_path = Path(Path(__file__).parent, "modius-ui-build", "index.html")
                self.context.logger.debug(f"Returning index.html from: {index_file_path}")
                with open(index_file_path, "r", encoding="utf-8") as file:
                    index_html = file.read()

                # Send the HTML response
                self._send_ok_response(http_msg, http_dialogue, index_html)
        except FileNotFoundError as e:
            self.context.logger.error(f"FileNotFoundError: {e}")
            self._handle_not_found(http_msg, http_dialogue)
    
    def _handle_not_found(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle a HTTP request for a resource that was not found.

        :param http_msg: the HTTP message
        :param http_dialogue: the HTTP dialogue
        """
        http_response = http_dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=http_msg,
            version=http_msg.version,
            status_code=HttpCode.NOT_FOUND_CODE.value,
            status_text="Not Found",
            headers=http_msg.headers,
            body=b"",
        )

        # Send response
        self.context.logger.info("Responding with: {}".format(http_response))
        self.context.outbox.put_message(message=http_response)

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

    def _handle_get_portfolio(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle a Http request to get portfolio values.

        :param http_msg: the http message
        :param http_dialogue: the http dialogue
        """
        # Define the path to the portfolio data file
        portfolio_data_filepath: str = (
            self.context.params.store_path / self.context.params.portfolio_info_filename
        )

        # Read the portfolio data from the file
        try:
            with open(portfolio_data_filepath, "r", encoding="utf-8") as file:
                portfolio_data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.context.logger.error(f"Error reading portfolio data: {str(e)}")
            portfolio_data = {"error": "Could not read portfolio data"}

        shared_state = cast(SharedState, self.context.state)
        selected_protocols = shared_state.selected_protocols
        trading_type = shared_state.trading_type
        portfolio_data.update({"selected_protocols": selected_protocols, "trading_type": trading_type})

        portfolio_data_json = json.dumps(portfolio_data)
        # Send the portfolio data as a response
        self._send_ok_response(http_msg, http_dialogue, portfolio_data_json)

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

            shared_state = cast(SharedState, self.context.state)
            previous_trading_type = shared_state.trading_type
            
            available_trading_types = [trading_type.value for trading_type in TradingType]
            last_selected_threshold = THRESHOLDS.get(previous_trading_type)
            # Format the prompt
            prompt_template = PROMPT.format(
                USER_PROMPT=user_prompt,
                PREVIOUS_TRADING_TYPE=previous_trading_type,
                TRADING_TYPES=available_trading_types,
                AVAILABLE_PROTOCOLS=self.context.params.available_strategies,
                LAST_THRESHOLD=last_selected_threshold
            )

            # Create LLM request
            llm_dialogues = cast(LlmDialogues, self.context.llm_dialogues)
            request_llm_message, llm_dialogue = llm_dialogues.create(
                counterparty=str(LLM_CONNECTION_PUBLIC_ID),
                performative=LlmMessage.Performative.REQUEST,
                prompt_template=prompt_template,
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
            selected_protocols = response_data.get("selected_protocols", [])
            trading_type = response_data.get("trading_type", "")
            reasoning = response_data.get("reasoning", "")
            previous_trading_type = cast(SharedState, self.context.state).trading_type

            if not strategies or not trading_type or not reasoning:
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
                response_data = {
                    "selected_protocols": selected_protocols,
                    "trading_type": trading_type,
                    "reasoning": reasoning,
                    "previous_trading_type": previous_trading_type
                }
                self._send_ok_response(http_msg, http_dialogue, response_data)

            # Offload KV store update to a separate thread
            threading.Thread(target=self._delayed_write_kv, args=(selected_protocols, trading_type)).start()

        except json.JSONDecodeError as e:
            self.context.logger.error(
                f"Failed to decode LLM response: {str(e)}"
            )
            self._handle_bad_request(http_msg, http_dialogue)

    def _delayed_write_kv(self, selected_protocols: List[str], trading_type: str) -> None:
        """
        Write to the KV store after a delay if this was the only request in queue.

        :param strategies: list of chosen strategies
        """
        time.sleep(self.context.params.waiting_time)
        if len(self.context.params.request_queue) == 1:
            self._write_kv({"selected_protocols": json.dumps(selected_protocols), "trading_type": trading_type})
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
            "rounds_info": self.rounds_info,
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
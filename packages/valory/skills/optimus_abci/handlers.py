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

"""This module contains the handlers for the skill of OptimusAbciApp."""

import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, cast
from urllib.parse import urlparse

import yaml
from aea.protocols.base import Message
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
from packages.dvilela.connections.genai.connection import (
    PUBLIC_ID as GENAI_CONNECTION_PUBLIC_ID,
)
from packages.valory.protocols.http.message import HttpMessage
from packages.valory.protocols.srr.message import SrrMessage
from packages.valory.skills.abstract_round_abci.handlers import (
    HttpHandler as BaseHttpHandler,
)
from packages.valory.skills.abstract_round_abci.models import Requests
from packages.valory.skills.liquidity_trader_abci.rounds_info import ROUNDS_INFO
from packages.valory.skills.liquidity_trader_abci.rounds import SynchronizedData
from packages.valory.skills.optimus_abci.dialogues import (
    HttpDialogue,
    HttpDialogues,
    SrrDialogue,
    SrrDialogues,
)
from packages.valory.skills.optimus_abci.models import Params
from packages.valory.skills.optimus_abci.models import SharedState
from packages.valory.skills.optimus_abci.prompts import PROMPT
from packages.valory.skills.liquidity_trader_abci.behaviours import THRESHOLDS, TradingType
from packages.valory.connections.http_server.connection import (
    PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
)
from packages.valory.protocols.http.message import HttpMessage
from packages.valory.skills.abstract_round_abci.handlers import (
    ABCIRoundHandler as BaseABCIRoundHandler,
)
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
from packages.valory.skills.liquidity_trader_abci.handlers import (
    IpfsHandler as BaseIpfsHandler,
)
from packages.valory.skills.liquidity_trader_abci.rounds import SynchronizedData
from packages.valory.skills.liquidity_trader_abci.rounds_info import ROUNDS_INFO
from packages.valory.skills.optimus_abci.dialogues import HttpDialogue, HttpDialogues
from packages.valory.skills.optimus_abci.models import SharedState


ABCIHandler = BaseABCIRoundHandler
SigningHandler = BaseSigningHandler
LedgerApiHandler = BaseLedgerApiHandler
ContractApiHandler = BaseContractApiHandler
TendermintHandler = BaseTendermintHandler
IpfsHandler = BaseIpfsHandler


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
        self.context.state.request_count += 1
        if self.context.state.request_count % self.context.params.cleanup_freq == 0:
            self.context.logger.info(
                f"{self.context.state.request_count} requests processed. Cleaning up dialogues."
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
            self.context.state.in_flight_req = False
            return

        dialogue = self.context.kv_store_dialogues.update(kv_store_msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        callback, kwargs = self.context.state.req_to_callback.pop(nonce)
        callback(kv_store_msg, dialogue, **kwargs)

        self.context.state.in_flight_req = False
        self.on_message_handled(message)

class HttpHandler(BaseHttpHandler):
    """This implements the HTTP handler."""

    SUPPORTED_PROTOCOL = HttpMessage.protocol_id

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
        portfolio_url_regex = rf"{hostname_regex}\/portfolio"
        static_files_regex = (
            rf"{hostname_regex}\/(.*)"  # New regex for serving static files
        )
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

        self.json_content_header = "Content-Type: application/json\n"
        self.html_content_header = "Content-Type: text/html\n"
        fsm = load_fsm_spec()

        self.rounds_info: Dict = {  # pylint: disable=attribute-defined-outside-init
            camel_to_snake(k): v for k, v in ROUNDS_INFO.items()
        }

        for source_info, target_round in fsm["transition_func"].items():
            source_round, event = source_info[1:-1].split(", ")
            self.rounds_info[camel_to_snake(source_round)]["transitions"][
                event.lower()
            ] = camel_to_snake(target_round)
        self.json_content_header = "Content-Type: application/json\n"
        self.html_content_header = "Content-Type: text/html\n"
        self.function_entry_count = 0

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return SynchronizedData(
            db=self.context.state.round_sequence.latest_synchronized_data.db
        )

    @property
    def shared_state(self) -> SharedState:
        """Get the parameters."""
        return cast(SharedState, self.context.state)

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

            # Construct the file path
            file_path = Path(Path(__file__).parent, "modius-ui-build", requested_path)

            # If the file exists and is a file, send it as a response
            if file_path.exists() and file_path.is_file():
                with open(file_path, "rb") as file:
                    file_content = file.read()

                # Send the file content as a response
                self._send_ok_response(http_msg, http_dialogue, file_content)
            else:
                # If the file doesn't exist or is not a file, return the index.html file
                with open(
                    Path(Path(__file__).parent, "modius-ui-build", "index.html"),
                    "r",
                    encoding="utf-8",
                ) as file:
                    index_html = file.read()

                # Send the HTML response
                self._send_ok_response(http_msg, http_dialogue, index_html)
        except FileNotFoundError:
            self._handle_not_found(http_msg, http_dialogue)

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
        data: Union[str, Dict, List, bytes],
    ) -> None:
        """Send an OK response with the provided data"""
        headers = (
            self.json_content_header
            if isinstance(data, (dict, list))
            else self.html_content_header
        )
        headers += http_msg.headers

        # Convert dictionary or list to JSON string
        if isinstance(data, (dict, list)):
            data = json.dumps(data)

        http_response = http_dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=http_msg,
            version=http_msg.version,
            status_code=HttpCode.OK_CODE.value,
            status_text="Success",
            headers=headers,
            body=data.encode("utf-8") if isinstance(data, str) else data,
        )

        # Send response
        self.context.logger.info("Responding with: {}".format(http_response))
        self.context.outbox.put_message(message=http_response)

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

        portfolio_data_json = json.dumps(portfolio_data)
        # Send the portfolio data as a response
        self._send_ok_response(http_msg, http_dialogue, portfolio_data_json)

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

        # Update evaluate strategy round description with agent reasoning if available
        agent_reasoning = cast(SharedState, self.context.state).agent_reasoning
        if agent_reasoning and "evaluate_strategy_round" in self.rounds_info:
            self.rounds_info["evaluate_strategy_round"]["description"] = agent_reasoning

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

    def _handle_get_static_js(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle a HTTP GET request for the main.js file.

        :param http_msg: the HTTP message
        :param http_dialogue: the HTTP dialogue
        """
        try:
            # Read the main.js file
            with open(
                Path(
                    Path(__file__).parent,
                    "modius-ui-build",
                    "static",
                    "js",
                    "main.d1485dfa.js",
                ),
                "rb",
            ) as file:
                file_content = file.read()

            # Send the file content as a response
            self._send_ok_response(http_msg, http_dialogue, file_content)
        except FileNotFoundError:
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

    def _handle_post_process_prompt(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> Generator[None, None, None]:
        """
        Handle POST requests to process user prompts.

        :param http_msg: the HttpMessage instance
        :param http_dialogue: the HttpDialogue instance
        """
        request_id = http_dialogue.dialogue_label.dialogue_reference[0]
        self.context.state.request_queue.append(request_id)

        # Increment the function entry counter
        self.function_entry_count += 1
        self.context.logger.info(f"------------------REQUEST NUMBER - {self.function_entry_count} TIMESTAMP - {time.time()}.")

        try:
            # Parse incoming data
            data = json.loads(http_msg.body.decode("utf-8"))
            user_prompt = data.get("prompt", "")
            if not user_prompt:
                raise ValueError("Prompt is required.")

            previous_trading_type = self.context.state.trading_type
            if not previous_trading_type:
                previous_trading_type = TradingType.BALANCED.value
            
            previous_selected_protocols = self.context.state.selected_protocols
            if not previous_selected_protocols:
                previous_selected_protocols = self.context.params.available_strategies

            available_trading_types = [trading_type.value for trading_type in TradingType]
            last_selected_threshold = THRESHOLDS.get(previous_trading_type)
            # Format the prompt
            prompt_template = PROMPT.format(
                USER_PROMPT=user_prompt,
                PREVIOUS_TRADING_TYPE=previous_trading_type,
                TRADING_TYPES=available_trading_types,
                AVAILABLE_PROTOCOLS=self.context.params.available_strategies,
                LAST_THRESHOLD=last_selected_threshold,
                PREVIOUS_SELECTED_PROTOCOLS=previous_selected_protocols,
                THRESHOLDS=THRESHOLDS
            )

            # Prepare payload data
            payload_data = {"prompt": prompt_template}

            # Create LLM request
            srr_dialogues = cast(SrrDialogues, self.context.srr_dialogues)
            request_srr_message, srr_dialogue = srr_dialogues.create(
                counterparty=str(GENAI_CONNECTION_PUBLIC_ID),
                performative=SrrMessage.Performative.REQUEST,
                payload=json.dumps(payload_data),
            )

            # Prepare callback args
            callback_kwargs = {"http_msg": http_msg, "http_dialogue": http_dialogue}
            self._send_message(
                request_srr_message, srr_dialogue, self._handle_llm_response, callback_kwargs
            )

        except (json.JSONDecodeError, ValueError) as e:
            self.context.logger.error(f"Error processing prompt: {str(e)}")
            self._handle_bad_request(http_msg, http_dialogue)

    def _handle_llm_response(
        self,
        llm_response_message: SrrMessage,
        dialogue: Dialogue,
        http_msg: HttpMessage,
        http_dialogue: HttpDialogue,
    ) -> None:
        """
        Handle the response from the LLM.

        :param srr_response_message: the SrrMessage with the LLM output
        :param dialogue: the Dialogue
        :param http_msg: the original HttpMessage
        :param http_dialogue: the original HttpDialogue
        """
        selected_protocols, trading_type, reasoning, previous_trading_type = self._parse_llm_response(llm_response_message)

        response_data = {
            "selected_protocols": selected_protocols,
            "trading_type": trading_type,
            "reasoning": reasoning,
            "previous_trading_type": previous_trading_type,
        }

        self._send_ok_response(http_msg, http_dialogue, response_data)

        # Offload KV store update to a separate thread
        threading.Thread(target=self._delayed_write_kv, args=(selected_protocols, trading_type)).start()

    def _parse_llm_response(
        self,
        llm_response_message: SrrMessage,
    ) -> Tuple[List[str], str, str, str]:
        """
        Parse the LLM response and return the selected protocols, trading type, reasoning, and previous trading type.

        :param llm_response_message: the SrrMessage with the LLM output
        :param http_msg: the original HttpMessage
        :param http_dialogue: the original HttpDialogue
        :return: a tuple containing the selected protocols, trading type, reasoning, and previous trading type
        """
        try:
            payload = json.loads(llm_response_message.payload)
            response_data = json.loads(payload.get("response", ""))
        except json.JSONDecodeError as e:
            response = json.loads(llm_response_message.payload).get("response")
            if response.strip().startswith("```json") and response.strip().endswith("```"):
                # Strip the triple backticks and attempt to parse the JSON content
                json_content = response.strip()[7:-3]
                try:
                    response_data = json.loads(json_content)
                except json.JSONDecodeError:
                    selected_protocols = self.context.state.selected_protocols
                    if not selected_protocols:
                        selected_protocols = self.context.params.available_strategies
                    trading_type = self.context.state.trading_type
                    if not trading_type:
                        trading_type = TradingType.BALANCED.value

                    reasoning = f"Failed to parse LLM response. The response is not in valid JSON format. Falling back to previous strategies: {selected_protocols} and trading type: {trading_type}"
                    return selected_protocols, trading_type, reasoning, trading_type
            else:
                reasoning = "Failed to parse LLM response. Falling back to default strategies."
                selected_protocols = self.context.params.available_strategies
                trading_type = self.context.state.trading_type
                if not trading_type:
                    trading_type = TradingType.BALANCED.value
                return selected_protocols, trading_type, reasoning, trading_type

        self.context.logger.info(f"Received LLM response: {response_data}")
        selected_protocols = response_data.get("selected_protocols", [])
        trading_type = response_data.get("trading_type", "")
        reasoning = response_data.get("reasoning", "")
        previous_trading_type = self.context.state.trading_type

        if not selected_protocols or not trading_type or not reasoning:
            selected_protocols = self.context.state.selected_protocols
            if not selected_protocols:
                selected_protocols = self.context.params.available_strategies
            trading_type = self.context.state.trading_type
            if not trading_type:
                trading_type = TradingType.BALANCED.value
            reasoning = ""

        return selected_protocols, trading_type, reasoning, previous_trading_type


    def _delayed_write_kv(self, selected_protocols: List[str], trading_type: str) -> None:
        """
        Write to the KV store after a delay if this was the only request in queue.

        :param strategies: list of chosen strategies
        """
        self.context.logger.info("Waiting for default acceptance time...")
        time.sleep(self.context.params.default_acceptance_time)
        if len(self.context.state.request_queue) == 1:
            self._write_kv({"selected_protocols": json.dumps(selected_protocols), "trading_type": trading_type})
        self.context.logger.info("KV store update delayed.")
        self.context.state.request_queue.pop()

    def _write_kv(self, data: Dict[str, str]) -> Generator[None, None, bool]:
        """
        Create or update data in the KV store.

        :param data: key-value data to store
        :return: success flag
        """
        self.context.logger.info("Writing to KV store...")
        kv_store_dialogues = cast(KvStoreDialogues, self.context.kv_store_dialogues)
        kv_store_message, srr_dialogue = kv_store_dialogues.create(
            counterparty=str(KV_STORE_CONNECTION_PUBLIC_ID),
            performative=KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST,
            data=data,
        )
        self.context.logger.info(f"KV store message: {kv_store_message}")
        kv_store_dialogue = cast(KvStoreDialogue, srr_dialogue)
        self._send_message(
            kv_store_message, kv_store_dialogue, self._handle_kv_store_response
        )
        self.context.logger.info(f"KV store update sent.{self.context.state.req_to_callback}")

    def _handle_kv_store_response(self, kv_store_response_message: KvStoreMessage, dialogue: Dialogue) -> None:
        """
        Handle KV store response messages.

        :param kv_store_response_message: the KvStoreMessage response
        :param dialogue: the KvStoreDialogue
        """
        self.context.logger.info(f"Received KV store response: {kv_store_response_message}")
        success = kv_store_response_message.performative == KvStoreMessage.Performative.SUCCESS
        if success:
            self.context.logger.info("KV store update successful.")
        else:
            self.context.logger.error("KV store update failed.")

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
        self.context.state.req_to_callback[nonce] = (callback, callback_kwargs or {})
        self.context.state.in_flight_req = True


class SrrHandler(BaseHandler):
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

from packages.valory.skills.liquidity_trader_abci.handlers import (
    KvStoreHandler as BaseKvStoreHandler,
)
KvStoreHandler = BaseKvStoreHandler
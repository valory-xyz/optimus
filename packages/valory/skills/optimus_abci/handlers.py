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
import math
import re
import threading
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, cast
from urllib.parse import urlparse

import yaml
from aea.configurations.data_types import PublicId
from aea.protocols.base import Message
from aea.protocols.dialogue.base import Dialogue
from aea.skills.base import Handler

from packages.dvilela.connections.genai.connection import (
    PUBLIC_ID as GENAI_CONNECTION_PUBLIC_ID,
)
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
from packages.valory.protocols.http.message import HttpMessage
from packages.valory.protocols.srr.message import SrrMessage
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
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    THRESHOLDS,
    TradingType,
)
from packages.valory.skills.liquidity_trader_abci.handlers import (
    IpfsHandler as BaseIpfsHandler,
)
from packages.valory.skills.liquidity_trader_abci.rounds import SynchronizedData
from packages.valory.skills.liquidity_trader_abci.rounds_info import ROUNDS_INFO
from packages.valory.skills.liquidity_trader_abci.utils import (
    validate_and_fix_protocols,
)
from packages.valory.skills.optimus_abci.dialogues import (
    HttpDialogue,
    HttpDialogues,
    SrrDialogues,
)
from packages.valory.skills.optimus_abci.models import Params, SharedState
from packages.valory.skills.optimus_abci.prompts import PROMPT


ABCIHandler = BaseABCIRoundHandler
SigningHandler = BaseSigningHandler
LedgerApiHandler = BaseLedgerApiHandler
ContractApiHandler = BaseContractApiHandler
TendermintHandler = BaseTendermintHandler
IpfsHandler = BaseIpfsHandler


# Strategy to protocol name mapping
STRATEGY_TO_PROTOCOL = {
    "balancer_pools_search": "balancerPool",
    "asset_lending": "sturdy",
    "uniswap_pools_search": "uniswapV3",
    "velodrome_pools_search": "velodrome",
}
# Reverse mapping for converting protocol names back to strategy names
PROTOCOL_TO_STRATEGY = {v: k for k, v in STRATEGY_TO_PROTOCOL.items()}
PROTOCOL_DEFINITIONS = {
    "balancerPool": "protocol for investing in liquidity positions",
    "sturdy": "protocol for lending assets",
    "uniswapV3": "protocol for investing in liquidity positions",
    "velodrome": "protocol for investing in liquidity positions",
}
MODIUS_AGENT_PROFILE_PATH = "modius-ui-build"
OPTIMUS_AGENT_PROFILE_PATH = "optimus-ui-build"

OK_CODE = 200
NOT_FOUND_CODE = 404
BAD_REQUEST_CODE = 400
AVERAGE_PERIOD_SECONDS = 10


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


class KvStoreHandler(AbstractResponseHandler):
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
                f"KvStore performative not recognized: {kv_store_msg.performative}"
            )
            self.context.state.in_flight_req = False
            return

        if kv_store_msg.performative in [
            KvStoreMessage.Performative.SUCCESS,
            KvStoreMessage.Performative.READ_RESPONSE,
        ]:
            nonce = kv_store_msg.dialogue_reference[0]
            callback, kwargs = self.context.state.req_to_callback.pop(nonce, (None, {}))

            if callback:
                dialogue = self.context.kv_store_dialogues.update(kv_store_msg)
                callback(kv_store_msg, dialogue, **kwargs)
                self.context.state.in_flight_req = False
            else:
                super().handle(message)
        else:
            super().handle(message)


class HttpHandler(BaseHttpHandler):
    """This implements the HTTP handler."""

    SUPPORTED_PROTOCOL = HttpMessage.protocol_id
    A = 8.786e-1  # 8.786×10^-1
    B = 8.272e-1  # 8.272×10^-1
    C = 2.44
    D = -7.552
    # Define bounds for the CS
    MIN_CS = 0.20
    MAX_CS = 0.50

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
        features_url_regex = (
            rf"{hostname_regex}\/features"  # New regex for /features endpoint
        )
        # Withdrawal endpoints
        withdrawal_amount_regex = rf"{hostname_regex}\/withdrawal\/amount"
        withdrawal_initiate_regex = rf"{hostname_regex}\/withdrawal\/initiate"
        withdrawal_status_regex = (
            rf"{hostname_regex}\/withdrawal\/status\/(?P<withdrawal_id>[a-zA-Z0-9\-]+)"
        )

        # Define routes
        self.routes = {
            (HttpMethod.POST.value,): [
                (process_prompt_regex, self._handle_post_process_prompt),
                (withdrawal_initiate_regex, self._handle_post_withdrawal_initiate),
            ],
            (HttpMethod.GET.value, HttpMethod.HEAD.value): [
                (health_url_regex, self._handle_get_health),
                (portfolio_url_regex, self._handle_get_portfolio),
                (features_url_regex, self._handle_get_features),  # Add new route
                (withdrawal_amount_regex, self._handle_get_withdrawal_amount),
                (withdrawal_status_regex, self._handle_get_withdrawal_status),
                (
                    static_files_regex,
                    self._handle_get_static_file,
                ),
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
        self.available_strategies = []
        for chain in self.context.params.target_investment_chains:
            chain_strategies = self.context.params.available_strategies.get(chain, [])
            self.available_strategies.extend(chain_strategies)

        self.agent_profile_path = (
            OPTIMUS_AGENT_PROFILE_PATH
            if self.context.params.target_investment_chains[0] == "optimism"
            else MODIUS_AGENT_PROFILE_PATH
        )

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

    @property
    def params(self) -> Params:
        """Shortcut to access the parameters."""
        return cast(Params, self.context.params)

    def _handle_get_features(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle a GET request to check if chat feature is enabled.

        :param http_msg: the HTTP message
        :param http_dialogue: the HTTP dialogue
        """
        # Check if GENAI_API_KEY is set
        api_key = self.context.params.genai_api_key

        is_chat_enabled = (
            api_key is not None
            and isinstance(api_key, str)
            and api_key.strip() != ""
            and api_key != "${str:}"
            and api_key != '""'
        )

        data = {"isChatEnabled": is_chat_enabled}

        self._send_ok_response(http_msg, http_dialogue, data)

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

            # If the path is empty or just "/", serve index.html
            if not requested_path or requested_path == "":
                requested_path = "index.html"

            self.context.logger.info(f"Serving static file: {requested_path}")

            # Construct the file path
            file_path = Path(
                Path(__file__).parent, self.agent_profile_path, requested_path
            )

            self.context.logger.info(f"Resolved file path: {file_path}")

            # If the file exists and is a file, send it as a response
            if file_path.exists() and file_path.is_file():
                # Determine content type based on file extension
                content_type = self._get_content_type(file_path.suffix)

                with open(file_path, "rb") as file:
                    file_content = file.read()

                self.context.logger.info(
                    f"Serving file {file_path} with content-type: {content_type}"
                )
                # Send the file content as a response
                self._send_ok_response(
                    http_msg, http_dialogue, file_content, content_type
                )
            else:
                self.context.logger.info(
                    f"File not found: {file_path}, serving index.html as fallback"
                )
                # If the file doesn't exist or is not a file, return the index.html file
                index_path = Path(
                    Path(__file__).parent, self.agent_profile_path, "index.html"
                )
                with open(index_path, "r", encoding="utf-8") as file:
                    index_html = file.read()

                # Send the HTML response
                self._send_ok_response(http_msg, http_dialogue, index_html, "text/html")
        except FileNotFoundError as e:
            self.context.logger.error(
                f"FileNotFoundError when serving static file: {e}"
            )
            self._handle_not_found(http_msg, http_dialogue)
        except Exception as e:
            self.context.logger.error(f"Error serving static file: {e}")
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
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue, error_msg=None
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
            body=b"" if not error_msg else error_msg.encode("utf-8"),
        )

        # Send response
        self.context.logger.info("Responding with: {}".format(http_response))
        self.context.outbox.put_message(message=http_response)

    def _send_ok_response(
        self,
        http_msg: HttpMessage,
        http_dialogue: HttpDialogue,
        data: Union[Dict, List, str, bytes],
        content_type: Optional[str] = None,
    ) -> None:
        """Send an OK response with the provided data"""

        body_bytes: bytes
        headers: str

        if isinstance(data, bytes):
            body_bytes = data
            header_content_type = (
                f"Content-Type: {content_type}\n"
                if content_type
                else self.json_content_header
            )
            headers = f"{header_content_type}{http_msg.headers}"
        elif isinstance(data, str):
            body_bytes = data.encode("utf-8")
            header_content_type = (
                f"Content-Type: {content_type}\n"
                if content_type
                else self.html_content_header
            )
            headers = f"{header_content_type}{http_msg.headers}"
        else:
            body_bytes = json.dumps(data).encode("utf-8")
            headers = f"{self.json_content_header}{http_msg.headers}"

        http_response = http_dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=http_msg,
            version=http_msg.version,
            status_code=OK_CODE,
            status_text="Success",
            headers=headers,
            body=body_bytes,
        )

        # Send response
        self.context.logger.info(f"Responding with {OK_CODE}")
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
            self.context.logger.info(f"Error reading portfolio data: {str(e)}")
            portfolio_data = {"error": "Could not read portfolio data"}

        # Get selected protocols (strategy names) from state
        selected_protocols = (
            json.loads(self.context.state.selected_protocols)
            if isinstance(self.context.state.selected_protocols, str)
            else self.context.state.selected_protocols or self.available_strategies
        )

        # Convert strategy names to protocol names
        selected_protocol_names = [
            STRATEGY_TO_PROTOCOL.get(strategy, strategy)
            for strategy in selected_protocols
        ]

        # Get trading type from state
        trading_type = self.context.state.trading_type
        if not trading_type:
            trading_type = TradingType.BALANCED.value

        # Add selected protocol names and trading type to portfolio data
        portfolio_data["selected_protocols"] = selected_protocol_names
        portfolio_data["trading_type"] = trading_type

        portfolio_data_json = json.dumps(portfolio_data)
        self.context.logger.info(f"Portfolio Response - {portfolio_data}")
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

    def _get_content_type(self, file_extension: str) -> str:
        """
        Get the appropriate content type based on file extension.

        :param file_extension: the file extension (including the dot)
        :return: the content type string
        """
        content_types = {
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
            ".txt": "text/plain",
            ".pdf": "application/pdf",
            ".woff": "font/woff",
            ".woff2": "font/woff2",
            ".ttf": "font/ttf",
            ".eot": "application/vnd.ms-fontobject",
        }
        return content_types.get(file_extension.lower(), "application/octet-stream")

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

        try:
            # Parse incoming data
            data = json.loads(http_msg.body.decode("utf-8"))
            user_prompt = data.get("prompt", "")
            self.context.logger.info(f"user_prompt from the user {user_prompt}")

            if not user_prompt:
                raise ValueError("Prompt is required.")

            previous_trading_type = self.context.state.trading_type
            if not previous_trading_type:
                previous_trading_type = TradingType.BALANCED.value

            previous_selected_protocols = (
                json.loads(self.context.state.selected_protocols)
                if isinstance(self.context.state.selected_protocols, str)
                else self.context.state.selected_protocols or self.available_strategies
            )

            available_trading_types = [
                trading_type.value for trading_type in TradingType
            ]
            last_selected_threshold = THRESHOLDS.get(previous_trading_type)

            # Convert strategy names to protocol names for LLM
            available_protocols = [
                STRATEGY_TO_PROTOCOL.get(strategy, strategy)
                for strategy in self.available_strategies
            ]

            available_protocols_definitons = {
                protocol: PROTOCOL_DEFINITIONS.get(protocol, "")
                for protocol in available_protocols
            }

            # Convert previous selected protocols to their protocol names
            previous_protocols_for_llm = [
                STRATEGY_TO_PROTOCOL.get(strategy, strategy)
                for strategy in previous_selected_protocols
            ]

            # Format the prompt
            prompt_template = PROMPT.format(
                USER_PROMPT=user_prompt,
                PREVIOUS_TRADING_TYPE=previous_trading_type,
                TRADING_TYPES=available_trading_types,
                AVAILABLE_PROTOCOLS=available_protocols_definitons,
                LAST_THRESHOLD=last_selected_threshold,
                PREVIOUS_SELECTED_PROTOCOLS=previous_protocols_for_llm,
                THRESHOLDS=THRESHOLDS,
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
                request_srr_message,
                srr_dialogue,
                self._handle_llm_response,
                callback_kwargs,
            )

        except (json.JSONDecodeError, ValueError) as e:
            self.context.logger.info(f"Error processing prompt: {str(e)}")
            self._handle_bad_request(http_msg, http_dialogue)

    def _parse_llm_response(
        self,
        llm_response_message: SrrMessage,
    ) -> Tuple[List[str], str, float, str, str]:
        """
        Parse the LLM response and return the selected protocols, trading type, max loss percentage, reasoning, and previous trading type.

        :param llm_response_message: the SrrMessage with the LLM output
        :return: a tuple containing the selected protocols, trading type, max loss percentage, reasoning, and previous trading type
        """
        try:
            payload = json.loads(llm_response_message.payload)
            self.context.logger.info(f"response from the geminia {payload}")

            if "error" in payload:
                # Extract the specific error message
                error_details = payload["error"]
                # Try to extract a more specific error message, but fall back to the full error if parsing fails
                try:
                    if 'message: "' in error_details:
                        error_message = error_details.split('message: "')[1].split('"')[
                            0
                        ]
                    else:
                        # If the expected pattern isn't found, use the full error details
                        error_message = error_details
                except (IndexError, AttributeError):
                    # Fallback to full error details if parsing fails
                    error_message = error_details

                reasoning = f"LLM Error: {error_message}"
                self.context.logger.error(f"Error from LLM: {reasoning}")
                return [], "", 10.0, reasoning, self.context.state.trading_type

            response_data = json.loads(payload.get("response", ""))
        except json.JSONDecodeError:
            # Attempt to handle JSON content wrapped in triple backticks
            try:
                response = json.loads(llm_response_message.payload).get("response", "")
                if response.strip().startswith("```json") and response.strip().endswith(
                    "```"
                ):
                    # Strip the triple backticks and attempt to parse the JSON content
                    json_content = response.strip()[7:-3]
                    response_data = json.loads(json_content)
                else:
                    raise ValueError("Response is not in valid JSON format.")
            except (json.JSONDecodeError, ValueError):
                reasoning = (
                    "Failed to parse LLM response. Falling back to default strategies."
                )
                self.context.logger.error(reasoning)
                return self._fallback_to_previous_strategy_with_loss()

        self.context.logger.info(f"Received LLM response: {response_data}")

        # Extract the main fields
        selected_protocols = response_data.get("selected_protocols", [])
        trading_type = response_data.get("trading_type", "")
        reasoning = response_data.get("reasoning", "")
        previous_trading_type = self.context.state.trading_type

        # Extract and validate max_loss_percentage
        max_loss_percentage = response_data.get("max_loss_percentage", None)

        self.context.logger.info(f"max_loss_percentage: {max_loss_percentage}")

        # Validate max_loss_percentage
        if max_loss_percentage is None or not isinstance(
            max_loss_percentage, (int, float)
        ):
            # Provide default based on trading type
            max_loss_percentage = (
                15.0 if trading_type == TradingType.RISKY.value else 5.0
            )
            self.context.logger.warning(
                f"Missing or invalid max_loss_percentage in LLM response. Using default: {max_loss_percentage}"
            )
        else:
            # Ensure the max_loss_percentage is within acceptable bounds
            max_loss_percentage = float(max_loss_percentage)
            if max_loss_percentage < 1.0:
                self.context.logger.warning(
                    f"max_loss_percentage too low: {max_loss_percentage}. Setting to minimum: 1.0"
                )
                max_loss_percentage = 1.0
            elif max_loss_percentage > 30.0:
                self.context.logger.warning(
                    f"max_loss_percentage too high: {max_loss_percentage}. Setting to maximum: 30.0"
                )
                max_loss_percentage = 30.0

        if not selected_protocols or not trading_type or not reasoning:
            return self._fallback_to_previous_strategy_with_loss()

        return (
            selected_protocols,
            trading_type,
            max_loss_percentage,
            reasoning,
            previous_trading_type,
        )

    def _fallback_to_previous_strategy_with_loss(
        self,
    ) -> Tuple[List[str], str, float, str, str]:
        """Fallback to previous strategy in case of parsing errors, including default loss percentage."""
        selected_protocols = (
            json.loads(self.context.state.selected_protocols)
            if isinstance(self.context.state.selected_protocols, str)
            else self.context.state.selected_protocols
            or self.context.params.available_strategies
        )

        selected_protocols = [
            STRATEGY_TO_PROTOCOL.get(strategy, strategy)
            for strategy in selected_protocols
        ]

        trading_type = self.context.state.trading_type
        if not trading_type:
            trading_type = TradingType.BALANCED.value

        # Default loss percentage based on trading type
        max_loss_percentage = 15.0 if trading_type == TradingType.RISKY.value else 5.0

        reasoning = "Failed to parse LLM response. Falling back to previous strategies."

        return (
            selected_protocols,
            trading_type,
            max_loss_percentage,
            reasoning,
            trading_type,
        )

    def calculate_composite_score_from_var(
        self, var: float, correlation_coefficient: float = 1.0
    ) -> float:
        """Calculate the composite score threshold based on user's risk tolerance (VaR)."""
        # Constants from the formula
        A = 8.786e-1  # 8.786×10^-1
        B = 8.272e-1  # 8.272×10^-1
        C = 2.44
        D = -7.552

        # Define bounds for the CS
        MIN_CS = 0.20
        MAX_CS = 0.50

        # Apply the equation: CS = ln((A/(VaR + B)) - C)/D
        try:
            # Calculate the numerator
            numerator = A

            # Calculate the denominator
            denominator = var + B

            # Apply natural logarithm to the ratio
            ln_term = math.log(numerator / denominator)

            # Calculate the composite score
            cs = (ln_term - C) / D

            # Apply correlation coefficient adjustment
            adjusted_cs = cs * correlation_coefficient

            # Apply bounds
            bounded_cs = max(MIN_CS, min(MAX_CS, adjusted_cs))

            self.context.logger.info(
                f"Calculated CS: {adjusted_cs}, Bounded CS: {bounded_cs} for VaR: {var}"
            )

            return bounded_cs
        except (ValueError, ZeroDivisionError) as e:
            self.context.logger.error(f"Error calculating composite score: {e}")
            # Return a default value if calculation fails
            return THRESHOLDS.get(TradingType.BALANCED.value, 0.3374)

    def _handle_llm_response(
        self,
        llm_response_message: SrrMessage,
        dialogue: Dialogue,
        http_msg: HttpMessage,
        http_dialogue: HttpDialogue,
    ) -> None:
        """
        Handle the response from the LLM.

        :param llm_response_message: the SrrMessage with the LLM output
        :param dialogue: the Dialogue
        :param http_msg: the original HttpMessage
        :param http_dialogue: the original HttpDialogue
        """
        (
            selected_protocol_names,
            trading_type,
            max_loss_percentage,
            reasoning,
            previous_trading_type,
        ) = self._parse_llm_response(llm_response_message)

        # Convert percentage to VaR (negative decimal)
        var_value = -max_loss_percentage / 100.0
        # Calculate composite score using the formula
        composite_score = self.calculate_composite_score_from_var(var_value)

        self.context.logger.info(f"composite_score: {composite_score}")

        # Validate and fix protocol names first (before converting to strategies)
        validated_protocol_names = validate_and_fix_protocols(
            selected_protocol_names,
            self.params.target_investment_chains,
            self.params.available_strategies,
        )

        # Convert validated protocol names to strategies for storage
        selected_protocols = [
            PROTOCOL_TO_STRATEGY.get(protocol, protocol)
            for protocol in validated_protocol_names
        ]

        response_data = {
            "selected_protocols": selected_protocol_names,
            "trading_type": trading_type,
            "max_loss_percentage": max_loss_percentage,
            "composite_score": round(composite_score, 4),
            "reasoning": reasoning,
            "previous_trading_type": previous_trading_type,
        }

        self.context.logger.info(f"response_data: {response_data}")
        self._send_ok_response(http_msg, http_dialogue, response_data)
        # Store the calculated composite score in the state
        self.context.logger.info(f"trading_type: {trading_type}")

        storage_data = {
            "selected_protocols": json.dumps(selected_protocols),
            "trading_type": trading_type,
            "composite_score": str(composite_score),
            "max_loss_percentage": str(max_loss_percentage),
        }

        # Offload KV store update to a separate thread
        threading.Thread(
            target=self._delayed_write_kv_extended, args=(storage_data,)
        ).start()

    def _fallback_to_previous_strategy(self) -> Tuple[List[str], str, str, str]:
        """Fallback to previous strategy in case of parsing errors."""
        selected_protocols = selected_protocols = (
            json.loads(self.context.state.selected_protocols)
            if isinstance(self.context.state.selected_protocols, str)
            else self.context.state.selected_protocols or self.available_strategies
        )

        selected_protocols = [
            STRATEGY_TO_PROTOCOL.get(strategy, strategy)
            for strategy in selected_protocols
        ]

        # Validate and fix protocols in fallback using shared utility
        selected_protocols = validate_and_fix_protocols(
            selected_protocols,
            self.params.target_investment_chains,
            self.params.available_strategies,
        )

        trading_type = self.context.state.trading_type
        if not trading_type:
            trading_type = TradingType.BALANCED.value
        reasoning = "Failed to parse LLM response. Falling back to previous strategies."

        return selected_protocols, trading_type, reasoning, trading_type

    def _delayed_write_kv_extended(self, data: Dict[str, str]) -> None:
        """
        Write to the KV store after a delay if this was the only request in queue.

        :param data: Dictionary of data to store
        """
        self.context.logger.info("Waiting for default acceptance time...")
        time.sleep(self.context.params.default_acceptance_time)

        if len(self.context.state.request_queue) == 1:
            self._write_kv(data)

            # Also update the state values
            if "selected_protocols" in data:
                self.context.state.selected_protocols = json.loads(
                    data["selected_protocols"]
                )
            if "trading_type" in data:
                self.context.state.trading_type = data["trading_type"]
            if "composite_score" in data:
                # Update the appropriate threshold based on trading type
                self.context.logger.info(
                    f"KV composite_score {(data['composite_score'])}"
                )
                self.context.state.composite_score = float(data["composite_score"])

        self.context.state.request_queue.pop()

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
        self.context.logger.info(f"Writing to KV store... {kv_store_message}")
        kv_store_dialogue = cast(KvStoreDialogue, srr_dialogue)
        self._send_message(
            kv_store_message, kv_store_dialogue, self._handle_kv_store_response
        )

    def _handle_kv_store_response(
        self, kv_store_response_message: KvStoreMessage, dialogue: Dialogue
    ) -> None:
        """
        Handle KV store response messages.

        :param kv_store_response_message: the KvStoreMessage response
        :param dialogue: the KvStoreDialogue
        """
        success = (
            kv_store_response_message.performative
            == KvStoreMessage.Performative.SUCCESS
        )
        if success:
            self.context.logger.info("KV store update successful.")
        else:
            self.context.logger.info("KV store update failed.")

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

    def _handle_get_withdrawal_amount(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle GET request to get withdrawal amount information.

        :param http_msg: the HttpMessage instance
        :param http_dialogue: the HttpDialogue instance
        """
        # Define the path to the portfolio data file
        portfolio_data_filepath: str = (
            self.context.params.store_path / self.context.params.portfolio_info_filename
        )

        # Read the portfolio data from the file (same as portfolio endpoint)
        try:
            with open(portfolio_data_filepath, "r", encoding="utf-8") as file:
                portfolio_data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.context.logger.info(f"Error reading portfolio data: {str(e)}")
            response = {"error": "Portfolio data not available", "status_code": 404}
            self._send_ok_response(http_msg, http_dialogue, response)
            return

        # Extract withdrawal amount data
        operating_chain = self.context.params.target_investment_chains[0]
        asset_breakdown = portfolio_data.get("portfolio_breakdown", [])

        # Filter out OLAS from asset breakdown
        filtered_breakdown = []
        for asset in asset_breakdown:
            asset_symbol = asset.get("asset", "")
            if asset_symbol == "OLAS":  # nosec B105
                continue  # Skip OLAS
            filtered_breakdown.append(asset)

        # Calculate total value excluding OLAS
        total_value = sum(asset.get("value_usd", 0) for asset in filtered_breakdown)

        # Transform asset_breakdown to match the expected format
        formatted_breakdown = []
        for asset in filtered_breakdown:
            formatted_breakdown.append(
                {
                    "token_symbol": asset.get("asset", "Unknown"),
                    "value": asset.get("balance", 0),
                    "value_usd": asset.get("value_usd", 0),
                }
            )

        response = {
            "amount": int(total_value * 10**6),  # Convert to USDC decimals (6)
            "total_value_usd": float(total_value),
            "chain": operating_chain,
            "asset_breakdown": formatted_breakdown,
            "status_code": 200,
        }

        self._send_ok_response(http_msg, http_dialogue, response)

    def _handle_post_withdrawal_initiate(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle POST request to initiate withdrawal.

        :param http_msg: the HttpMessage instance
        :param http_dialogue: the HttpDialogue instance
        """
        try:
            # Parse incoming data
            data = json.loads(http_msg.body.decode("utf-8"))
            target_address = data.get("target_address", "")

            # Validate target address
            if not target_address or not self._is_valid_ethereum_address(
                target_address
            ):
                error_response = {"error": "Invalid target address format"}
                self._send_ok_response(http_msg, http_dialogue, error_response)
                return

            # Check if there are funds available
            portfolio_data_filepath: str = (
                self.context.params.store_path
                / self.context.params.portfolio_info_filename
            )

            try:
                with open(portfolio_data_filepath, "r", encoding="utf-8") as file:
                    portfolio_data = json.load(file)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self.context.logger.info(f"Error reading portfolio data: {str(e)}")
                error_response = {"error": "Portfolio data not available"}
                self._send_ok_response(http_msg, http_dialogue, error_response)
                return

            total_value = portfolio_data.get("portfolio_value", 0)
            if total_value <= 0:
                error_response = {"error": "No funds available for withdrawal"}
                self._send_ok_response(http_msg, http_dialogue, error_response)
                return

            # Generate withdrawal ID
            withdrawal_id = str(uuid.uuid4())

            # Get current timestamp
            current_timestamp = int(time.time())

            # Prepare withdrawal data with correct initial status and message
            withdrawal_data = {
                "withdrawal_id": withdrawal_id,
                "withdrawal_status": "INITIATED",
                "withdrawal_target_address": target_address,
                "withdrawal_message": "Withdrawal Initiated. Preparing your funds...",
                "withdrawal_requested_at": str(current_timestamp),
                "withdrawal_estimated_value_usd": str(total_value),
                "withdrawal_chain": self.context.params.target_investment_chains[0],
                "withdrawal_safe_address": self.context.params.safe_contract_addresses.get(
                    self.context.params.target_investment_chains[0]
                ),
                "withdrawal_current_step": "EXIT_POOLS",
                "investing_paused": "true",
            }

            # Store withdrawal data in KV store
            threading.Thread(
                target=self._write_withdrawal_data, args=(withdrawal_data,)
            ).start()

            # Prepare response
            response = {
                "id": withdrawal_id,
                "status": "initiated",
                "estimated_value_usd": float(total_value),
                "chain": self.context.params.target_investment_chains[0],
            }

            self._send_ok_response(http_msg, http_dialogue, response)

        except (json.JSONDecodeError, ValueError) as e:
            self.context.logger.info(
                f"Error processing withdrawal initiation: {str(e)}"
            )
            error_response = {"error": "Invalid request format"}
            self._send_ok_response(http_msg, http_dialogue, error_response)

    def _handle_get_withdrawal_status(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue, withdrawal_id: str
    ) -> None:
        """
        Handle GET request to get withdrawal status.

        :param http_msg: the HttpMessage instance
        :param http_dialogue: the HttpDialogue instance
        :param withdrawal_id: the withdrawal ID from URL
        """
        try:
            # Read withdrawal data from KV store
            withdrawal_data = self._read_withdrawal_data()

            if (
                not withdrawal_data
                or withdrawal_data.get("withdrawal_id") != withdrawal_id
            ):
                time.sleep(5)
                withdrawal_data = self._read_withdrawal_data()
                if (
                    not withdrawal_data
                    or withdrawal_data.get("withdrawal_id") != withdrawal_id
                ):
                    # Return UNKNOWN status when no withdrawal data found
                    response = {
                        "status": "unknown",
                        "message": "",
                        "chain": "",
                        "requested_at": "",
                        "estimated_value_usd": 0.0,
                        "transaction_link": "",
                    }
                    self._send_ok_response(http_msg, http_dialogue, response)
                    return

            # Get transaction links based on chain
            transaction_links = []
            if withdrawal_data.get("withdrawal_transaction_hashes"):
                tx_hashes = json.loads(withdrawal_data["withdrawal_transaction_hashes"])
                chain = withdrawal_data.get("withdrawal_chain", "optimism")
                for tx_hash in tx_hashes:
                    transaction_links.append(self._get_transaction_link(chain, tx_hash))

            # Prepare response based on status
            status = withdrawal_data.get("withdrawal_status", "UNKNOWN")
            response = {
                "status": status.lower(),
                "message": withdrawal_data.get("withdrawal_message", ""),
                "chain": withdrawal_data.get("withdrawal_chain", ""),
                "requested_at": withdrawal_data.get("withdrawal_requested_at", ""),
                "estimated_value_usd": float(
                    withdrawal_data.get("withdrawal_estimated_value_usd", 0)
                ),
                "transaction_link": transaction_links[0] if transaction_links else "",
            }

            # Add status-specific fields
            if status == "COMPLETED":
                response["completed_at"] = withdrawal_data.get(
                    "withdrawal_completed_at", ""
                )
                response["transaction_hashes"] = json.loads(
                    withdrawal_data.get("withdrawal_transaction_hashes", "[]")
                )
                response["transaction_count"] = len(response["transaction_hashes"])

            self._send_ok_response(http_msg, http_dialogue, response)

        except Exception as e:
            self.context.logger.error(f"Error getting withdrawal status: {str(e)}")
            error_response = {"error": "Failed to retrieve withdrawal status"}
            self._send_ok_response(http_msg, http_dialogue, error_response)

    def _is_valid_ethereum_address(self, address: str) -> bool:
        """
        Validate if the given string is a valid Ethereum address.

        :param address: the address to validate
        :return: True if valid, False otherwise
        """
        import re

        # Check if it matches Ethereum address pattern
        pattern = re.compile(r"^0x[a-fA-F0-9]{40}$")
        return bool(pattern.match(address))

    def _write_withdrawal_data(self, data: Dict[str, str]) -> None:
        """
        Write withdrawal data to KV store.

        :param data: the withdrawal data to store
        """
        try:
            # Use the existing _write_kv method
            self._write_kv(data)
            self.context.logger.info(f"Withdrawal data written to KV store: {data}")
        except Exception as e:
            self.context.logger.error(f"Error writing withdrawal data: {str(e)}")

    def _read_withdrawal_data(self) -> Optional[Dict[str, str]]:
        """
        Read withdrawal data from KV store.

        :return: withdrawal data or None if not found
        """
        try:
            # Read all withdrawal-related keys from KV store
            keys = (
                "withdrawal_id",
                "withdrawal_status",
                "withdrawal_target_address",
                "withdrawal_message",
                "withdrawal_requested_at",
                "withdrawal_completed_at",
                "withdrawal_estimated_value_usd",
                "withdrawal_chain",
                "withdrawal_safe_address",
                "withdrawal_transaction_hashes",
                "withdrawal_current_step",
            )

            result = self._read_kv(keys)
            return result if result else None
        except Exception as e:
            self.context.logger.error(f"Error reading withdrawal data: {str(e)}")
            return None

    def _get_transaction_link(self, chain: str, tx_hash: str) -> str:
        """
        Get transaction explorer link based on chain.

        :param chain: the blockchain chain
        :param tx_hash: the transaction hash
        :return: the transaction explorer link
        """
        chain_explorers = {
            "optimism": f"https://optimistic.etherscan.io/tx/{tx_hash}",
            "base": f"https://basescan.org/tx/{tx_hash}",
            "mode": f"https://explorer-mode-mainnet-0.t.conduit.xyz/tx/{tx_hash}",
            "ethereum": f"https://etherscan.io/tx/{tx_hash}",
        }
        return chain_explorers.get(chain, f"https://etherscan.io/tx/{tx_hash}")

    def _read_kv(self, keys: tuple) -> Dict[str, Any]:
        """
        Read data from the KV store.

        :param keys: tuple of keys to read
        :return: dictionary with key-value pairs
        """
        try:
            kv_store_dialogues = cast(KvStoreDialogues, self.context.kv_store_dialogues)
            kv_store_message, srr_dialogue = kv_store_dialogues.create(
                counterparty=str(KV_STORE_CONNECTION_PUBLIC_ID),
                performative=KvStoreMessage.Performative.READ_REQUEST,
                keys=keys,
            )
            self.context.logger.info(f"Reading from KV store... {kv_store_message}")
            kv_store_dialogue = cast(KvStoreDialogue, srr_dialogue)
            self._send_message(
                kv_store_message, kv_store_dialogue, self._handle_kv_read_response
            )
            # Return cached data if available, otherwise empty dict
            return getattr(self.context.state, "last_kv_read_data", {})

        except Exception as e:
            self.context.logger.error(f"Error reading from KV store: {e}")
            return {}

    def _handle_kv_read_response(
        self, kv_store_response_message: KvStoreMessage, dialogue: Dialogue
    ) -> None:
        """Handle KV store read response messages."""
        success = (
            kv_store_response_message.performative
            == KvStoreMessage.Performative.READ_RESPONSE
        )
        if success:
            self.context.logger.info("KV store read successful.")
            # Store the read data for later use
            if hasattr(kv_store_response_message, "data"):
                self.context.state.last_kv_read_data = kv_store_response_message.data
        else:
            self.context.logger.info("KV store read failed.")
            self.context.state.last_kv_read_data = {}
        self.context.state.in_flight_req = False


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

        nonce = srr_msg.dialogue_reference[
            0
        ]  # Assuming dialogue_reference is accessible
        callback, kwargs = self.context.state.req_to_callback.pop(nonce, (None, {}))

        if callback is None:
            super().handle(message)
        else:
            dialogue = self.context.srr_dialogues.update(srr_msg)
            callback(srr_msg, dialogue, **kwargs)
            self.context.state.in_flight_req = False

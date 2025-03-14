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

        # Routes
        self.routes = {
            (HttpMethod.POST.value,): [],
            (HttpMethod.GET.value, HttpMethod.HEAD.value): [
                (health_url_regex, self._handle_get_health),
                (portfolio_url_regex, self._handle_get_portfolio),
                (
                    static_files_regex,
                    self._handle_get_static_file,
                ),  # New route for serving static files
            ],
        }

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

    def _handle_get_static_css(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle a HTTP GET request for the main.css file.

        :param http_msg: the HTTP message
        :param http_dialogue: the HTTP dialogue
        """
        try:
            # Read the main.css file
            with open(
                Path(
                    Path(__file__).parent,
                    "modius-ui-build",
                    "static",
                    "css",
                    "main.972e2d60.css",
                ),
                "rb",
            ) as file:
                file_content = file.read()

            # Send the file content as a response
            self._send_ok_response(http_msg, http_dialogue, file_content)
        except FileNotFoundError:
            self._handle_not_found(http_msg, http_dialogue)

    def _handle_get_manifest(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle a HTTP GET request for the manifest.json file.

        :param http_msg: the HTTP message
        :param http_dialogue: the HTTP dialogue
        """
        try:
            # Read the manifest.json file
            with open(
                Path(Path(__file__).parent, "modius-ui-build", "manifest.json"), "rb"
            ) as file:
                file_content = file.read()

            # Send the file content as a response
            self._send_ok_response(http_msg, http_dialogue, file_content)
        except FileNotFoundError:
            self._handle_not_found(http_msg, http_dialogue)

    def _handle_get_favicon(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle a HTTP GET request for the favicon.ico file.

        :param http_msg: the HTTP message
        :param http_dialogue: the HTTP dialogue
        """
        try:
            # Read the favicon.ico file
            with open(
                Path(Path(__file__).parent, "modius-ui-build", "favicon.ico"), "rb"
            ) as file:
                favicon_content = file.read()

            # Send the favicon content as a response
            self._send_ok_response(http_msg, http_dialogue, favicon_content)
        except FileNotFoundError:
            self._handle_not_found(http_msg, http_dialogue)

    def _handle_get_mode_network_logo(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle a HTTP GET request for the mode-network.png file.

        :param http_msg: the HTTP message
        :param http_dialogue: the HTTP dialogue
        """
        try:
            # Read the mode-network.png file
            with open(
                Path(
                    Path(__file__).parent,
                    "modius-ui-build",
                    "logos",
                    "protocols",
                    "mode-network.png",
                ),
                "rb",
            ) as file:
                file_content = file.read()

            # Send the file content as a response
            self._send_ok_response(http_msg, http_dialogue, file_content)
        except FileNotFoundError:
            self._handle_not_found(http_msg, http_dialogue)

    def _handle_get_modius_logo(
        self, http_msg: HttpMessage, http_dialogue: HttpDialogue
    ) -> None:
        """
        Handle a HTTP GET request for the modius.png file.

        :param http_msg: the HTTP message
        :param http_dialogue: the HTTP dialogue
        """
        try:
            # Read the modius.png file
            with open(
                Path(Path(__file__).parent, "modius-ui-build", "logos", "modius.png"),
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

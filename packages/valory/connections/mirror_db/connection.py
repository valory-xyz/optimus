#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2026 Valory AG
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

"""GenericMirrorDB connection for interacting with REST APIs."""

import asyncio
import json
import logging
import ssl
from functools import wraps
from typing import Any, Dict, Optional, Union, cast

import aiohttp
import certifi
from aea.configurations.base import PublicId
from aea.connections.base import Connection, ConnectionStates
from aea.mail.base import Envelope
from aea.protocols.base import Address, Message
from aea.protocols.dialogue.base import Dialogue
from aea.protocols.dialogue.base import Dialogue as BaseDialogue

from packages.valory.protocols.srr.dialogues import SrrDialogue
from packages.valory.protocols.srr.dialogues import SrrDialogues as BaseSrrDialogues
from packages.valory.protocols.srr.message import SrrMessage


PUBLIC_ID = PublicId.from_str("valory/mirror_db:0.1.0")

_MODULE_LOGGER = logging.getLogger(__name__)


class MirrorDBHTTPError(Exception):
    """HTTP error raised by mirror_db with the response status code attached."""

    def __init__(self, status_code: int, message: str) -> None:
        """Initialize with a status code and message."""
        super().__init__(message)
        self.status_code = status_code


_RETRYABLE_NETWORK_EXCEPTIONS: tuple = (
    aiohttp.ClientConnectorError,
    aiohttp.ServerTimeoutError,
    asyncio.TimeoutError,
)


def _is_retryable(exc: BaseException) -> bool:
    """Return True if the exception is a transient failure worth retrying."""
    if isinstance(exc, MirrorDBHTTPError):
        return (
            exc.status_code in (408, 429)
            or 500 <= exc.status_code < 600
        )
    return isinstance(exc, _RETRYABLE_NETWORK_EXCEPTIONS)


def retry_with_exponential_backoff(max_retries=5, initial_delay=1, backoff_factor=2):  # type: ignore
    """Retry a function with exponential backoff on transient failures.

    Retries 408, 429, 5xx, and transient network errors (ClientConnectorError,
    ServerTimeoutError, asyncio.TimeoutError). All other exceptions propagate
    immediately.
    """

    def decorator(func):  # type: ignore
        @wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore
            delay = initial_delay
            last_exception: Optional[Exception] = None
            logger = getattr(args[0], "logger", None) if args else None
            if logger is None:
                logger = _MODULE_LOGGER
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if not _is_retryable(e):
                        raise
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Retrying mirror_db request in {delay}s "
                            f"(attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
            logger.error(
                f"Max retries ({max_retries}) reached for mirror_db request"
            )
            raise last_exception  # type: ignore

        return wrapper

    return decorator


class SrrDialogues(BaseSrrDialogues):
    """A class to keep track of SRR dialogues."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize dialogues.

        :param kwargs: keyword arguments
        """

        def role_from_first_message(  # pylint: disable=unused-argument
            message: Message, receiver_address: Address
        ) -> Dialogue.Role:
            """Infer the role of the agent from an incoming/outgoing first message

            :param message: an incoming/outgoing first message
            :param receiver_address: the address of the receiving agent
            :return: The role of the agent
            """
            return SrrDialogue.Role.CONNECTION

        BaseSrrDialogues.__init__(
            self,
            self_address=str(kwargs.pop("connection_id")),
            role_from_first_message=role_from_first_message,
            **kwargs,
        )


class GenericMirrorDBConnection(Connection):
    """Generic proxy to interact with REST API backend services."""

    connection_id = PUBLIC_ID

    # Registry of valid endpoint patterns (using regex patterns)
    _VALID_ENDPOINTS = {
        r"^api/[a-zA-Z0-9_-]+/?$",  # Base resources
        r"^api/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/?$",  # Specific resource by ID
        r"^api/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/?$",  # Nested resources
    }

    # List of allowed methods that can be called via the SRR protocol
    _ALLOWED_METHODS = {
        "create_",
        "read_",
        "update_",
        "delete_",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the connection.

        :param args: positional arguments passed to component base
        :param kwargs: keyword arguments passed to component base
        """
        super().__init__(*args, **kwargs)
        self.base_url = self.configuration.config.get("mirror_db_base_url")
        # self.api_key: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.dialogues = SrrDialogues(connection_id=PUBLIC_ID)
        self._response_envelopes: Optional[asyncio.Queue] = None
        self.task_to_request: Dict[asyncio.Future, Envelope] = {}
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

        # Store all configuration in a single dictionary
        self._config = {
            # "api_key": self.api_key,
            "base_url": self.base_url,
            # Add other default configs here
        }

    @property
    def response_envelopes(self) -> asyncio.Queue:
        """
        Returns the response envelopes queue.

        :return: The queue of response envelopes
        :raises ValueError: If the queue is not initialized
        """
        if self._response_envelopes is None:
            raise ValueError(
                "`GenericMirrorDBConnection.response_envelopes` is not yet initialized. Is the connection setup?"
            )
        return self._response_envelopes

    async def connect(self) -> None:
        """
        Connect to the backend service.

        Sets up the response queue and initializes the HTTP session.
        """
        self._response_envelopes = asyncio.Queue()
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=self.ssl_context),
            timeout=aiohttp.ClientTimeout(total=30),
        )
        self.state = ConnectionStates.connected

    async def disconnect(self) -> None:
        """
        Disconnect from the backend service.

        Closes the HTTP session and cleans up resources.
        """
        if self.is_disconnected:
            return

        self.state = ConnectionStates.disconnecting

        for task in self.task_to_request.keys():
            if not task.cancelled():
                task.cancel()
        self._response_envelopes = None

        if self.session is not None:
            await self.session.close()
            self.session = None

        self.state = ConnectionStates.disconnected

    async def receive(
        self, *args: Any, **kwargs: Any
    ) -> Optional[Union["Envelope", None]]:
        """
        Receive an envelope.

        :param args: Arguments for the receive method
        :param kwargs: Keyword arguments for the receive method
        :return: The received envelope or None
        """
        return await self.response_envelopes.get()

    async def send(self, envelope: Envelope) -> None:
        """
        Send an envelope.

        :param envelope: The envelope to send
        """
        task = self._handle_envelope(envelope)
        task.add_done_callback(self._handle_done_task)
        self.task_to_request[task] = envelope

    def _handle_envelope(self, envelope: Envelope) -> asyncio.Task:
        """
        Handle incoming envelopes by dispatching background tasks.

        :param envelope: The envelope to handle
        :return: The task handling the envelope
        """
        message = cast(SrrMessage, envelope.message)
        dialogue = self.dialogues.update(message)
        task = self.loop.create_task(self._get_response(message, dialogue))
        return task

    def prepare_error_message(
        self, srr_message: SrrMessage, dialogue: Optional[BaseDialogue], error: str
    ) -> SrrMessage:
        """
        Prepare error message.

        :param srr_message: The original message
        :param dialogue: The dialogue
        :param error: The error message
        :return: The error response message
        """
        response_message = cast(
            SrrMessage,
            dialogue.reply(  # type: ignore
                performative=SrrMessage.Performative.RESPONSE,
                target_message=srr_message,
                payload=json.dumps({"error": error}, ensure_ascii=True),
                error=True,
            ),
        )
        return response_message

    def _handle_done_task(self, task: asyncio.Future) -> None:
        """
        Process a completed task.

        :param task: The completed task
        """
        request = None
        try:
            request = self.task_to_request.pop(task)
            try:
                response_message: Optional[Message] = task.result()
            except Exception as e:  # pylint: disable=broad-except
                self.logger.exception(
                    "Unhandled exception in mirror_db task "
                    f"(to={request.to}, sender={request.sender}): {e}"
                )
                response_message = self._build_typed_error_message(request, e)

            envelope: Optional[Envelope] = None
            if response_message is not None:
                envelope = Envelope(
                    to=request.sender,
                    sender=request.to,
                    message=response_message,
                    context=request.context,
                )
            self.response_envelopes.put_nowait(envelope)
        except Exception as outer_exc:  # pylint: disable=broad-except
            self.logger.exception(
                f"Unrecoverable failure in _handle_done_task: {outer_exc}"
            )
            self.response_envelopes.put_nowait(None)

    def _build_typed_error_message(
        self, request: Envelope, exc: Exception
    ) -> Optional[Message]:
        """Build an SRR error reply for a failed task."""
        try:
            srr_message = cast(SrrMessage, request.message)
            dialogue = self.dialogues.update(srr_message)
            return self.prepare_error_message(
                srr_message,
                dialogue,
                f"Backend task failed: {exc}",
            )
        except Exception as build_exc:  # pylint: disable=broad-except
            self.logger.exception(
                f"Failed to build typed error message: {build_exc}"
            )
            return None

    async def _get_response(
        self, srr_message: SrrMessage, dialogue: Optional[BaseDialogue]
    ) -> SrrMessage:
        """
        Get response from the backend service.

        :param srr_message: The request message
        :param dialogue: The dialogue
        :return: The response message
        """
        if srr_message.performative != SrrMessage.Performative.REQUEST:
            return self.prepare_error_message(
                srr_message,
                dialogue,
                f"Performative `{srr_message.performative.value}` is not supported.",
            )

        try:
            payload = json.loads(srr_message.payload)
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.error(
                f"Failed to parse mirror_db request payload: {e}"
            )
            return self.prepare_error_message(
                srr_message,
                dialogue,
                f"Failed to parse request payload: {e}",
            )
        method_name = payload.get("method")

        if method_name not in self._ALLOWED_METHODS:
            return self.prepare_error_message(
                srr_message,
                dialogue,
                f"Method {method_name} is not allowed or present.",
            )

        method = getattr(self, method_name, None)

        if method is None:
            return self.prepare_error_message(
                srr_message,
                dialogue,
                f"Method {method_name} is not available.",
            )

        # Log endpoint and payload (safe log that handles Unicode characters)
        endpoint = payload.get("kwargs", {}).get("endpoint")
        try:
            self.logger.info(f"endpoint,payload : {endpoint,payload}")
        except UnicodeEncodeError:
            safe_payload = str(payload).encode("ascii", "replace").decode("ascii")
            self.logger.info(f"endpoint,payload : {endpoint},{safe_payload}")

        try:
            response = await method(**payload.get("kwargs", {}))
            response_message = cast(
                SrrMessage,
                dialogue.reply(  # type: ignore
                    performative=SrrMessage.Performative.RESPONSE,
                    target_message=srr_message,
                    payload=json.dumps({"response": response}, ensure_ascii=True),
                    error=False,
                ),
            )
            return response_message

        except Exception as e:
            return self.prepare_error_message(
                srr_message, dialogue, f"Exception while calling backend service:\n{e}"
            )

    async def _raise_for_response(
        self, response: aiohttp.ClientResponse, action: str
    ) -> None:
        """
        Raise exception with relevant message based on the HTTP status code.

        :param response: The HTTP response
        :param action: The action being performed (for error messages)
        :raises MirrorDBHTTPError: If the response status is not 200
        """
        if response.status == 200:
            return
        error_content = await response.json()
        detail = error_content.get("detail", error_content)
        raise MirrorDBHTTPError(
            response.status,
            f"Error {action}: {detail} (HTTP {response.status})",
        )

    @retry_with_exponential_backoff()
    async def create_(self, method_name: str, endpoint: str, data: Dict) -> Dict:
        """
        Create a resource using a POST request.

        :param method_name: Name of the method for logging and error reporting
        :param endpoint: API endpoint to call
        :param data: The data to send in the request body
        :return: Response from the API
        """
        async with self.session.post(  # type: ignore
            f"{self.base_url}/{endpoint}",
            json=data,
            # headers={"access-token": f"{self.api_key}"},
        ) as response:
            await self._raise_for_response(
                response, f"creating resource via {method_name}"
            )
            return await response.json()

    @retry_with_exponential_backoff()
    async def read_(self, method_name: str, endpoint: str) -> Dict:
        """
        Read a resource using a GET request.

        :param method_name: Name of the method for logging and error reporting
        :param endpoint: API endpoint to call
        :return: Response from the API
        """
        async with self.session.get(  # type: ignore
            f"{self.base_url}/{endpoint}",
            # headers={"access-token": f"{self.api_key}"},
        ) as response:
            await self._raise_for_response(
                response, f"reading resource via {method_name}"
            )
            return await response.json()

    @retry_with_exponential_backoff()
    async def update_(self, method_name: str, endpoint: str, data: Dict) -> Dict:
        """
        Update a resource using a PUT request.

        :param method_name: Name of the method for logging and error reporting
        :param endpoint: API endpoint to call
        :param data: The data to send in the request body
        :return: Response from the API
        """
        async with self.session.put(  # type: ignore
            f"{self.base_url}/{endpoint}",
            json=data,
            # headers={"access-token": f"{self.api_key}"},
        ) as response:
            await self._raise_for_response(
                response, f"updating resource via {method_name}"
            )
            return await response.json()

    @retry_with_exponential_backoff()
    async def delete_(self, method_name: str, endpoint: str) -> Dict:
        """
        Delete a resource using a DELETE request.

        :param method_name: Name of the method for logging and error reporting
        :param endpoint: API endpoint to call
        :return: Response from the API
        """
        async with self.session.delete(  # type: ignore
            f"{self.base_url}/{endpoint}",
            # headers={"access-token": f"{self.api_key}"},
        ) as response:
            await self._raise_for_response(
                response, f"deleting resource via {method_name}"
            )
            return await response.json()

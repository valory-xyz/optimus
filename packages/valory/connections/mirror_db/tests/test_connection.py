# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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
# pylint: disable=protected-access,attribute-defined-outside-init

"""Tests for mirror_db connection."""

import asyncio
import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aea.configurations.base import ConnectionConfig
from aea.connections.base import ConnectionStates
from aea.mail.base import Envelope
from aea.protocols.base import Message

from packages.valory.connections.mirror_db.connection import (
    PUBLIC_ID,
    GenericMirrorDBConnection,
    SrrDialogues,
    retry_with_exponential_backoff,
)
from packages.valory.protocols.srr.dialogues import SrrDialogue
from packages.valory.protocols.srr.message import SrrMessage


ANY_SKILL = "skill/any:0.1.0"
BASE_URL = "https://test-mirror-db.example.com"


def _make_connection(base_url: str = BASE_URL) -> GenericMirrorDBConnection:
    """Create a GenericMirrorDBConnection with a mocked configuration."""
    configuration = ConnectionConfig(
        mirror_db_base_url=base_url,
        connection_id=GenericMirrorDBConnection.connection_id,
    )
    connection = GenericMirrorDBConnection(
        configuration=configuration,
        data_dir=MagicMock(),
    )
    return connection


def _make_srr_request_message(payload: dict) -> SrrMessage:
    """Create an SrrMessage with REQUEST performative."""
    return SrrMessage(
        performative=SrrMessage.Performative.REQUEST,
        payload=json.dumps(payload),
    )


def _make_mock_dialogue() -> MagicMock:
    """Create a mock dialogue that supports reply()."""
    dialogue = MagicMock()

    def mock_reply(**kwargs):
        """Build a real SrrMessage from reply kwargs."""
        return SrrMessage(
            performative=kwargs["performative"],
            dialogue_reference=("test", "test"),
            message_id=2,
            target=1,
            payload=kwargs["payload"],
            error=kwargs["error"],
        )

    dialogue.reply.side_effect = mock_reply
    return dialogue


class TestRetryWithExponentialBackoff:
    """Tests for the retry_with_exponential_backoff decorator."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self) -> None:
        """Test that a successful call returns immediately without retry."""
        call_count = 0

        @retry_with_exponential_backoff(max_retries=3, initial_delay=0.01)
        async def successful_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_retry_then_success(self) -> None:
        """Test retry on rate limit error then eventual success."""
        call_count = 0

        @retry_with_exponential_backoff(
            max_retries=3, initial_delay=0.01, backoff_factor=2
        )
        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Rate limit exceeded")
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_rate_limit_max_retries_exceeded(self) -> None:
        """Test that max retries raises the exception."""

        @retry_with_exponential_backoff(max_retries=2, initial_delay=0.01)
        async def always_rate_limited() -> str:
            raise Exception("Rate limit exceeded")

        with pytest.raises(Exception, match="Rate limit exceeded"):
            await always_rate_limited()

    @pytest.mark.asyncio
    async def test_non_rate_limit_error_raises_immediately(self) -> None:
        """Test non-rate-limit errors raise immediately without retrying."""
        call_count = 0

        @retry_with_exponential_backoff(max_retries=5, initial_delay=0.01)
        async def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Some other error")

        with pytest.raises(ValueError, match="Some other error"):
            await failing_func()
        assert call_count == 1


class TestSrrDialogues:
    """Tests for the SrrDialogues class."""

    def test_role_from_first_message_returns_connection_role(self) -> None:
        """Test the role_from_first_message callable returns CONNECTION."""
        dialogues = SrrDialogues(connection_id=PUBLIC_ID)
        # Access the role_from_first_message via the dialogues internals
        role = dialogues._role_from_first_message(
            MagicMock(spec=Message), "some_address"
        )
        assert role == SrrDialogue.Role.CONNECTION


class TestGenericMirrorDBConnectionInit:
    """Tests for GenericMirrorDBConnection initialization."""

    def test_init_attributes(self) -> None:
        """Test that __init__ sets all attributes correctly."""
        connection = _make_connection()
        assert connection.base_url == BASE_URL
        assert connection.session is None
        assert connection._response_envelopes is None
        assert connection.task_to_request == {}
        assert connection.ssl_context is not None
        assert connection._config["base_url"] == BASE_URL


class TestResponseEnvelopesProperty:
    """Tests for the response_envelopes property."""

    def test_raises_when_not_initialized(self) -> None:
        """Test accessing response_envelopes before connect raises ValueError."""
        connection = _make_connection()
        with pytest.raises(ValueError, match="not yet initialized"):
            _ = connection.response_envelopes

    @pytest.mark.asyncio
    async def test_returns_queue_after_connect(self) -> None:
        """Test that response_envelopes returns the queue after connect."""
        connection = _make_connection()
        await connection.connect()
        try:
            assert isinstance(connection.response_envelopes, asyncio.Queue)
        finally:
            await connection.disconnect()


class TestConnect:
    """Tests for the connect method."""

    @pytest.mark.asyncio
    async def test_connect(self) -> None:
        """Test that connect sets up session and queue."""
        connection = _make_connection()
        await connection.connect()
        try:
            assert connection.session is not None
            assert connection._response_envelopes is not None
            assert connection.state == ConnectionStates.connected
        finally:
            await connection.disconnect()

    @pytest.mark.asyncio
    async def test_connect_session_has_timeout(self) -> None:
        """Test that connect sets a timeout on the aiohttp session."""
        connection = _make_connection()
        await connection.connect()
        try:
            assert connection.session is not None
            assert connection.session.timeout.total == 30
        finally:
            await connection.disconnect()


class TestDisconnect:
    """Tests for the disconnect method."""

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """Test disconnect closes session and cleans up."""
        connection = _make_connection()
        await connection.connect()
        await connection.disconnect()
        assert connection.session is None
        assert connection._response_envelopes is None
        assert connection.state == ConnectionStates.disconnected

    @pytest.mark.asyncio
    async def test_disconnect_when_already_disconnected(self) -> None:
        """Test disconnect is a no-op when already disconnected."""
        connection = _make_connection()
        connection.state = ConnectionStates.disconnected
        await connection.disconnect()
        assert connection.state == ConnectionStates.disconnected

    @pytest.mark.asyncio
    async def test_disconnect_cancels_pending_tasks(self) -> None:
        """Test disconnect cancels pending tasks."""
        connection = _make_connection()
        await connection.connect()

        mock_future = MagicMock(spec=asyncio.Future)
        mock_future.cancelled.return_value = False
        connection.task_to_request[mock_future] = MagicMock()

        await connection.disconnect()
        mock_future.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_skips_already_cancelled_tasks(self) -> None:
        """Test disconnect skips already cancelled tasks."""
        connection = _make_connection()
        await connection.connect()

        mock_future = MagicMock(spec=asyncio.Future)
        mock_future.cancelled.return_value = True
        connection.task_to_request[mock_future] = MagicMock()

        await connection.disconnect()
        mock_future.cancel.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_with_no_session(self) -> None:
        """Test disconnect when session is already None."""
        connection = _make_connection()
        await connection.connect()
        connection.session = None
        await connection.disconnect()
        assert connection.state == ConnectionStates.disconnected


class TestReceive:
    """Tests for the receive method."""

    @pytest.mark.asyncio
    async def test_receive_returns_envelope_from_queue(self) -> None:
        """Test that receive returns the envelope from the queue."""
        connection = _make_connection()
        await connection.connect()
        try:
            mock_envelope = MagicMock(spec=Envelope)
            connection.response_envelopes.put_nowait(mock_envelope)
            result = await connection.receive()
            assert result is mock_envelope
        finally:
            await connection.disconnect()


class TestSend:
    """Tests for the send method."""

    @pytest.mark.asyncio
    async def test_send_creates_task(self) -> None:
        """Test that send creates a task and adds it to task_to_request."""
        connection = _make_connection()
        await connection.connect()
        try:
            envelope = MagicMock(spec=Envelope)
            mock_task = MagicMock(spec=asyncio.Task)
            with patch.object(connection, "_handle_envelope", return_value=mock_task):
                await connection.send(envelope)

            assert mock_task in connection.task_to_request
            assert connection.task_to_request[mock_task] is envelope
            mock_task.add_done_callback.assert_called_once()
        finally:
            await connection.disconnect()


class TestHandleEnvelope:
    """Tests for the _handle_envelope method."""

    @pytest.mark.asyncio
    async def test_handle_envelope_creates_task(self) -> None:
        """Test that _handle_envelope creates an asyncio task."""
        connection = _make_connection()
        await connection.connect()
        try:
            mock_message = MagicMock(spec=SrrMessage)
            mock_envelope = MagicMock(spec=Envelope)
            mock_envelope.message = mock_message

            with patch.object(
                connection.dialogues, "update", return_value=MagicMock()
            ), patch.object(
                connection,
                "_get_response",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ):
                task = connection._handle_envelope(mock_envelope)
                assert isinstance(task, asyncio.Task)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        finally:
            await connection.disconnect()


class TestPrepareErrorMessage:
    """Tests for the prepare_error_message method."""

    def test_prepare_error_message(self) -> None:
        """Test that prepare_error_message creates a proper error response."""
        connection = _make_connection()
        mock_dialogue = _make_mock_dialogue()
        msg = MagicMock(spec=SrrMessage)

        error_msg = connection.prepare_error_message(msg, mock_dialogue, "Test error")
        assert error_msg.performative == SrrMessage.Performative.RESPONSE
        payload = json.loads(error_msg.payload)
        assert payload["error"] == "Test error"
        assert error_msg.error is True


class TestHandleDoneTask:
    """Tests for the _handle_done_task method."""

    @pytest.mark.asyncio
    async def test_handle_done_task_with_response(self) -> None:
        """Test _handle_done_task with a response message from the task."""
        connection = _make_connection()
        await connection.connect()
        try:
            mock_message = MagicMock(spec=Message)
            mock_task = MagicMock(spec=asyncio.Future)
            mock_task.result.return_value = mock_message

            request_envelope = MagicMock(
                to=str(PUBLIC_ID),
                sender=ANY_SKILL,
                context=MagicMock(),
            )
            connection.task_to_request[mock_task] = request_envelope

            mock_envelope_instance = MagicMock(spec=Envelope)
            with patch(
                "packages.valory.connections.mirror_db.connection.Envelope",
                return_value=mock_envelope_instance,
            ) as mock_envelope_cls:
                connection._handle_done_task(mock_task)

                mock_envelope_cls.assert_called_once_with(
                    to=ANY_SKILL,
                    sender=str(PUBLIC_ID),
                    message=mock_message,
                    context=request_envelope.context,
                )

            assert mock_task not in connection.task_to_request
            result = connection.response_envelopes.get_nowait()
            assert result is mock_envelope_instance
        finally:
            await connection.disconnect()

    @pytest.mark.asyncio
    async def test_handle_done_task_with_none_response(self) -> None:
        """Test _handle_done_task when the task returns None."""
        connection = _make_connection()
        await connection.connect()
        try:
            mock_task = MagicMock(spec=asyncio.Future)
            mock_task.result.return_value = None

            request_envelope = MagicMock(
                to=str(PUBLIC_ID),
                sender=ANY_SKILL,
                context=MagicMock(),
            )
            connection.task_to_request[mock_task] = request_envelope

            connection._handle_done_task(mock_task)

            assert mock_task not in connection.task_to_request
            envelope = connection.response_envelopes.get_nowait()
            assert envelope is None
        finally:
            await connection.disconnect()


class TestGetResponse:
    """Tests for the _get_response method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.connection = _make_connection()

    def _make_request_and_dialogue(self, payload: dict):
        """Create a mock SrrMessage with REQUEST performative and a mock dialogue."""
        msg = MagicMock(spec=SrrMessage)
        msg.performative = SrrMessage.Performative.REQUEST
        msg.payload = json.dumps(payload)
        dialogue = _make_mock_dialogue()
        return msg, dialogue

    @pytest.mark.asyncio
    async def test_wrong_performative(self) -> None:
        """Test _get_response with wrong performative."""
        msg = MagicMock(spec=SrrMessage)
        wrong_perf = MagicMock()
        wrong_perf.value = "response"
        msg.performative = wrong_perf
        dialogue = _make_mock_dialogue()

        result = await self.connection._get_response(msg, dialogue)
        payload = json.loads(result.payload)
        assert "error" in payload
        assert "not supported" in payload["error"]

    @pytest.mark.asyncio
    async def test_disallowed_method(self) -> None:
        """Test _get_response with a method not in _ALLOWED_METHODS."""
        msg, dialogue = self._make_request_and_dialogue(
            {"method": "not_allowed", "kwargs": {}}
        )

        result = await self.connection._get_response(msg, dialogue)
        payload = json.loads(result.payload)
        assert "error" in payload
        assert "not allowed or present" in payload["error"]

    @pytest.mark.asyncio
    async def test_method_not_found_on_instance(self) -> None:
        """Test _get_response when method is allowed but not on the instance."""
        self.connection._ALLOWED_METHODS = {"nonexistent_method"}
        msg, dialogue = self._make_request_and_dialogue(
            {"method": "nonexistent_method", "kwargs": {}}
        )

        result = await self.connection._get_response(msg, dialogue)
        payload = json.loads(result.payload)
        assert "error" in payload
        assert "not available" in payload["error"]

    @pytest.mark.asyncio
    async def test_successful_read_method_call(self) -> None:
        """Test _get_response with a successful read_ method call."""
        msg, dialogue = self._make_request_and_dialogue(
            {
                "method": "read_",
                "kwargs": {"method_name": "test", "endpoint": "api/test"},
            }
        )

        mock_result = {"data": "value"}
        with patch.object(
            self.connection,
            "read_",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await self.connection._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert payload["response"] == mock_result
        assert result.error is False

    @pytest.mark.asyncio
    async def test_method_raises_exception(self) -> None:
        """Test _get_response when the method raises an exception."""
        msg, dialogue = self._make_request_and_dialogue(
            {
                "method": "read_",
                "kwargs": {"method_name": "test", "endpoint": "api/test"},
            }
        )

        with patch.object(
            self.connection,
            "read_",
            new_callable=AsyncMock,
            side_effect=Exception("Connection failed"),
        ):
            result = await self.connection._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert "error" in payload
        assert "Connection failed" in payload["error"]

    @pytest.mark.asyncio
    async def test_endpoint_print_unicode_error(self) -> None:
        """Test _get_response handles UnicodeEncodeError in print."""
        msg, dialogue = self._make_request_and_dialogue(
            {
                "method": "read_",
                "kwargs": {"method_name": "test", "endpoint": "api/test"},
            }
        )

        mock_result = {"data": "value"}

        call_count = 0

        def mock_print_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise UnicodeEncodeError("utf-8", "test", 0, 1, "err")
            # second call succeeds

        with patch.object(
            self.connection,
            "read_",
            new_callable=AsyncMock,
            return_value=mock_result,
        ), patch(
            "builtins.print",
            side_effect=mock_print_fn,
        ):
            result = await self.connection._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert payload["response"] == mock_result

    @pytest.mark.asyncio
    async def test_create_method_call(self) -> None:
        """Test _get_response with create_ method."""
        msg, dialogue = self._make_request_and_dialogue(
            {
                "method": "create_",
                "kwargs": {
                    "method_name": "test",
                    "endpoint": "api/test",
                    "data": {"key": "val"},
                },
            }
        )

        mock_result = {"id": 1}
        with patch.object(
            self.connection,
            "create_",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await self.connection._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert payload["response"] == mock_result

    @pytest.mark.asyncio
    async def test_update_method_call(self) -> None:
        """Test _get_response with update_ method."""
        msg, dialogue = self._make_request_and_dialogue(
            {
                "method": "update_",
                "kwargs": {
                    "method_name": "test",
                    "endpoint": "api/test",
                    "data": {"key": "val"},
                },
            }
        )

        mock_result = {"updated": True}
        with patch.object(
            self.connection,
            "update_",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await self.connection._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert payload["response"] == mock_result

    @pytest.mark.asyncio
    async def test_delete_method_call(self) -> None:
        """Test _get_response with delete_ method."""
        msg, dialogue = self._make_request_and_dialogue(
            {
                "method": "delete_",
                "kwargs": {"method_name": "test", "endpoint": "api/test"},
            }
        )

        mock_result = {"deleted": True}
        with patch.object(
            self.connection,
            "delete_",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await self.connection._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert payload["response"] == mock_result

    @pytest.mark.asyncio
    async def test_no_kwargs_key(self) -> None:
        """Test _get_response when payload has no kwargs key."""
        msg, dialogue = self._make_request_and_dialogue({"method": "read_"})

        with patch.object(
            self.connection,
            "read_",
            new_callable=AsyncMock,
            return_value={"data": "ok"},
        ):
            result = await self.connection._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert payload["response"] == {"data": "ok"}


class TestRaiseForResponse:
    """Tests for the _raise_for_response method."""

    @pytest.mark.asyncio
    async def test_status_200_returns_none(self) -> None:
        """Test that a 200 response returns None."""
        connection = _make_connection()
        mock_response = MagicMock()
        mock_response.status = 200

        result = await connection._raise_for_response(mock_response, "test")
        assert result is None

    @pytest.mark.asyncio
    async def test_non_200_raises_exception(self) -> None:
        """Test that a non-200 response raises an Exception."""
        connection = _make_connection()
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json.return_value = {"detail": "Bad request"}

        with pytest.raises(Exception, match="Error test: Bad request \\(HTTP 400\\)"):
            await connection._raise_for_response(mock_response, "test")

    @pytest.mark.asyncio
    async def test_non_200_without_detail_key(self) -> None:
        """Test non-200 response where json has no 'detail' key."""
        connection = _make_connection()
        mock_response = AsyncMock()
        mock_response.status = 500
        error_body = {"message": "Internal Server Error"}
        mock_response.json.return_value = error_body

        with pytest.raises(Exception, match="Internal Server Error"):
            await connection._raise_for_response(mock_response, "action")


class TestCRUDMethods:
    """Tests for the CRUD methods (create_, read_, update_, delete_)."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.connection = _make_connection()

    def _mock_session_method(
        self,
        method_name: str,
        status: int = 200,
        json_data: Optional[dict] = None,
    ):
        """Create a mock context manager for a session HTTP method."""
        mock_response = AsyncMock()
        mock_response.status = status
        mock_response.json.return_value = json_data or {}

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_response
        mock_cm.__aexit__.return_value = False

        mock_session = MagicMock()
        getattr(mock_session, method_name).return_value = mock_cm

        return mock_session, mock_response

    @pytest.mark.asyncio
    async def test_create_success(self) -> None:
        """Test create_ with a successful response."""
        mock_session, _ = self._mock_session_method("post", 200, {"id": 1})
        self.connection.session = mock_session

        result = await self.connection.create_(
            method_name="test_create",
            endpoint="api/items",
            data={"name": "test"},
        )
        assert result == {"id": 1}
        mock_session.post.assert_called_once_with(
            f"{BASE_URL}/api/items",
            json={"name": "test"},
        )

    @pytest.mark.asyncio
    async def test_create_error(self) -> None:
        """Test create_ with an error response."""
        mock_session, _ = self._mock_session_method("post", 400, {"detail": "Bad data"})
        self.connection.session = mock_session

        with pytest.raises(Exception, match="Bad data"):
            await self.connection.create_(
                method_name="test_create",
                endpoint="api/items",
                data={"name": "test"},
            )

    @pytest.mark.asyncio
    async def test_read_success(self) -> None:
        """Test read_ with a successful response."""
        mock_session, _ = self._mock_session_method("get", 200, {"data": "value"})
        self.connection.session = mock_session

        result = await self.connection.read_(
            method_name="test_read", endpoint="api/items"
        )
        assert result == {"data": "value"}
        mock_session.get.assert_called_once_with(
            f"{BASE_URL}/api/items",
        )

    @pytest.mark.asyncio
    async def test_read_error(self) -> None:
        """Test read_ with an error response."""
        mock_session, _ = self._mock_session_method("get", 404, {"detail": "Not found"})
        self.connection.session = mock_session

        with pytest.raises(Exception, match="Not found"):
            await self.connection.read_(method_name="test_read", endpoint="api/items")

    @pytest.mark.asyncio
    async def test_update_success(self) -> None:
        """Test update_ with a successful response."""
        mock_session, _ = self._mock_session_method("put", 200, {"updated": True})
        self.connection.session = mock_session

        result = await self.connection.update_(
            method_name="test_update",
            endpoint="api/items/1",
            data={"name": "updated"},
        )
        assert result == {"updated": True}
        mock_session.put.assert_called_once_with(
            f"{BASE_URL}/api/items/1",
            json={"name": "updated"},
        )

    @pytest.mark.asyncio
    async def test_update_error(self) -> None:
        """Test update_ with an error response."""
        mock_session, _ = self._mock_session_method(
            "put", 500, {"detail": "Server error"}
        )
        self.connection.session = mock_session

        with pytest.raises(Exception, match="Server error"):
            await self.connection.update_(
                method_name="test_update",
                endpoint="api/items/1",
                data={"name": "updated"},
            )

    @pytest.mark.asyncio
    async def test_delete_success(self) -> None:
        """Test delete_ with a successful response."""
        mock_session, _ = self._mock_session_method("delete", 200, {"deleted": True})
        self.connection.session = mock_session

        result = await self.connection.delete_(
            method_name="test_delete", endpoint="api/items/1"
        )
        assert result == {"deleted": True}
        mock_session.delete.assert_called_once_with(
            f"{BASE_URL}/api/items/1",
        )

    @pytest.mark.asyncio
    async def test_delete_error(self) -> None:
        """Test delete_ with an error response."""
        mock_session, _ = self._mock_session_method(
            "delete", 403, {"detail": "Forbidden"}
        )
        self.connection.session = mock_session

        with pytest.raises(Exception, match="Forbidden"):
            await self.connection.delete_(
                method_name="test_delete", endpoint="api/items/1"
            )

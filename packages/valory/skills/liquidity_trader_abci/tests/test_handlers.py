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

"""Test the handlers.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from unittest.mock import MagicMock, patch

from packages.dvilela.protocols.kv_store.message import KvStoreMessage
from packages.valory.protocols.ipfs import IpfsMessage
from packages.valory.skills.liquidity_trader_abci.handlers import (
    IpfsHandler,
    KvStoreHandler,
)


def test_import() -> None:
    """Test that the handlers module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.handlers  # noqa


class TestIpfsHandler:
    """Test IpfsHandler class."""

    def _make_handler(self) -> IpfsHandler:
        """Create an IpfsHandler using object.__new__ and __dict__ injection."""
        handler = object.__new__(IpfsHandler)
        mock_context = MagicMock()
        handler.__dict__["_context"] = mock_context
        return handler

    def test_supported_protocol(self) -> None:
        """Test SUPPORTED_PROTOCOL attribute."""
        assert IpfsHandler.SUPPORTED_PROTOCOL == IpfsMessage.protocol_id

    def test_allowed_response_performatives(self) -> None:
        """Test allowed_response_performatives attribute."""
        assert (
            IpfsMessage.Performative.IPFS_HASH
            in IpfsHandler.allowed_response_performatives
        )

    def test_custom_support_performative(self) -> None:
        """Test custom_support_performative attribute."""
        assert IpfsHandler.custom_support_performative == IpfsMessage.Performative.FILES

    def test_shared_state_property(self) -> None:
        """Test shared_state property."""
        handler = self._make_handler()
        result = handler.shared_state
        assert result is handler.context.state

    def test_handle_ipfs_hash_calls_super(self) -> None:
        """Test handle delegates to super for non-FILES performative."""
        handler = self._make_handler()
        message = MagicMock(spec=IpfsMessage)
        message.performative = IpfsMessage.Performative.IPFS_HASH

        with patch.object(IpfsHandler.__bases__[0], "handle") as mock_super_handle:
            handler.handle(message)

        assert handler.context.state.in_flight_req is False
        mock_super_handle.assert_called_once_with(message)

    def test_handle_files_calls_callback(self) -> None:
        """Test handle with FILES performative calls callback."""
        handler = self._make_handler()
        message = MagicMock(spec=IpfsMessage)
        message.performative = IpfsMessage.Performative.FILES

        mock_dialogue = MagicMock()
        mock_dialogue.dialogue_label.dialogue_reference = ("nonce_123", "")
        handler.context.ipfs_dialogues.update.return_value = mock_dialogue

        callback = MagicMock()
        handler.context.state.req_to_callback = {"nonce_123": callback}

        handler.handle(message)

        assert handler.context.state.in_flight_req is False
        handler.context.ipfs_dialogues.update.assert_called_once_with(message)
        callback.assert_called_once_with(message, mock_dialogue)


class TestKvStoreHandler:
    """Test KvStoreHandler class."""

    def _make_handler(self) -> KvStoreHandler:
        """Create a KvStoreHandler using object.__new__ and __dict__ injection."""
        handler = object.__new__(KvStoreHandler)
        mock_context = MagicMock()
        handler.__dict__["_context"] = mock_context
        return handler

    def test_supported_protocol(self) -> None:
        """Test SUPPORTED_PROTOCOL attribute."""
        assert KvStoreHandler.SUPPORTED_PROTOCOL == KvStoreMessage.protocol_id

    def test_allowed_response_performatives(self) -> None:
        """Test allowed_response_performatives includes expected performatives."""
        expected = frozenset(
            {
                KvStoreMessage.Performative.READ_REQUEST,
                KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST,
                KvStoreMessage.Performative.READ_RESPONSE,
                KvStoreMessage.Performative.SUCCESS,
                KvStoreMessage.Performative.ERROR,
            }
        )
        assert KvStoreHandler.allowed_response_performatives == expected

    def test_handle_unrecognized_performative(self) -> None:
        """Test handle with unrecognized performative logs warning."""
        handler = self._make_handler()
        message = MagicMock(spec=KvStoreMessage)
        # Use a performative not in allowed set
        message.performative = "unrecognized_perf"

        handler.handle(message)

        handler.context.logger.warning.assert_called_once()
        assert handler.context.state.in_flight_req is False

    def test_handle_success_performative_calls_callback(self) -> None:
        """Test handle with SUCCESS performative calls callback."""
        handler = self._make_handler()
        message = MagicMock(spec=KvStoreMessage)
        message.performative = KvStoreMessage.Performative.SUCCESS

        mock_dialogue = MagicMock()
        mock_dialogue.dialogue_label.dialogue_reference = ("nonce_456", "")
        handler.context.kv_store_dialogues.update.return_value = mock_dialogue

        callback = MagicMock()
        kwargs = {"key": "value"}
        handler.context.state.req_to_callback = {"nonce_456": (callback, kwargs)}

        handler.handle(message)

        handler.context.kv_store_dialogues.update.assert_called_once_with(message)
        callback.assert_called_once_with(message, mock_dialogue, key="value")
        assert handler.context.state.in_flight_req is False

    def test_handle_other_allowed_performative_calls_super(self) -> None:
        """Test handle with non-SUCCESS allowed performative calls super."""
        handler = self._make_handler()
        message = MagicMock(spec=KvStoreMessage)
        message.performative = KvStoreMessage.Performative.READ_RESPONSE

        with patch.object(KvStoreHandler.__bases__[0], "handle") as mock_super_handle:
            handler.handle(message)

        mock_super_handle.assert_called_once_with(message)

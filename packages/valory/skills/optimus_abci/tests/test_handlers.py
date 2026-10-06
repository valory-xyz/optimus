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

"""Test the handlers.py module of the optimus_abci skill."""

# pylint: skip-file

import json
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import requests
from web3 import Web3
from web3.exceptions import BadFunctionCallOutput, ContractLogicError

import packages.valory.skills.optimus_abci.handlers as handlers_module
from packages.valory.skills.liquidity_trader_abci.behaviours.base import ZERO_ADDRESS
from packages.valory.skills.optimus_abci.handlers import (
    BASIUS_AGENT_PROFILE_PATH,
    BaseHandler,
    ESTIMATED_GAS_PER_TX,
    HttpCode,
    HttpHandler,
    KvStoreHandler,
    MODIUS_AGENT_PROFILE_PATH,
    OK_CODE,
    OPTIMUS_AGENT_PROFILE_PATH,
    SrrHandler,
    camel_to_snake,
    load_fsm_spec,
)


def test_import() -> None:
    """Test that the handlers module can be imported."""
    import packages.valory.skills.optimus_abci.handlers  # noqa


def test_camel_to_snake_simple() -> None:
    """Test camel_to_snake with simple CamelCase."""
    assert camel_to_snake("CamelCase") == "camel_case"


def test_camel_to_snake_multiple_words() -> None:
    """Test camel_to_snake with multiple words."""
    assert camel_to_snake("FetchStrategiesRound") == "fetch_strategies_round"


def test_camel_to_snake_single_word() -> None:
    """Test camel_to_snake with a single lowercase word."""
    assert camel_to_snake("hello") == "hello"


def test_camel_to_snake_already_snake() -> None:
    """Test camel_to_snake with already snake_case input."""
    assert camel_to_snake("already_snake") == "already_snake"


def test_camel_to_snake_single_upper() -> None:
    """Test camel_to_snake with a single uppercase letter at the start."""
    assert camel_to_snake("A") == "a"


def test_load_fsm_spec() -> None:
    """Test load_fsm_spec returns a dict with expected keys."""
    spec = load_fsm_spec()
    assert isinstance(spec, dict)
    assert "transition_func" in spec
    assert "alphabet_in" in spec


def _make_concrete_base_handler() -> Any:
    """Create a concrete subclass of BaseHandler for testing."""

    class ConcreteHandler(BaseHandler):
        """ConcreteHandler."""

        SUPPORTED_PROTOCOL = None

        def handle(self, message: Any) -> None:
            """Handle."""

    handler = ConcreteHandler.__new__(ConcreteHandler)
    mock_context = MagicMock()
    # Bypass the property by setting _context on the instance
    object.__setattr__(handler, "_context", mock_context)
    return handler, mock_context


class TestBaseHandler:
    """Test BaseHandler class."""

    def test_setup(self) -> None:
        """Test setup method logs info."""
        handler, ctx = _make_concrete_base_handler()
        handler.setup()
        ctx.logger.info.assert_called_once()

    def test_teardown(self) -> None:
        """Test teardown method logs info."""
        handler, ctx = _make_concrete_base_handler()
        handler.teardown()
        ctx.logger.info.assert_called_once()

    def test_params_property(self) -> None:
        """Test params property returns context params."""
        handler, ctx = _make_concrete_base_handler()
        result = handler.params
        assert result is ctx.params

    def test_cleanup_dialogues(self) -> None:
        """Test cleanup_dialogues calls cleanup on found dialogues."""
        handler, ctx = _make_concrete_base_handler()
        mock_dialogues = MagicMock()
        ctx.handlers.__dict__ = {"http_handler": MagicMock()}
        ctx.http_dialogues = mock_dialogues
        handler.cleanup_dialogues()
        mock_dialogues.cleanup.assert_called_once()

    def test_cleanup_dialogues_no_matching_dialogues(self) -> None:
        """Test cleanup_dialogues when no matching dialogues exist."""
        handler, ctx = _make_concrete_base_handler()
        ctx.handlers.__dict__ = {"some_handler": MagicMock()}
        ctx.some_dialogues = None
        # Should not raise
        handler.cleanup_dialogues()

    def test_on_message_handled_increments_count(self) -> None:
        """Test on_message_handled increments request count."""
        handler, ctx = _make_concrete_base_handler()
        ctx.state.request_count = 0
        ctx.params.cleanup_freq = 100
        handler.on_message_handled(MagicMock())
        assert ctx.state.request_count == 1

    def test_on_message_handled_triggers_cleanup(self) -> None:
        """Test on_message_handled triggers cleanup at cleanup_freq."""
        handler, ctx = _make_concrete_base_handler()
        ctx.state.request_count = 99
        ctx.params.cleanup_freq = 100
        mock_dialogues = MagicMock()
        ctx.handlers.__dict__ = {"http_handler": MagicMock()}
        ctx.http_dialogues = mock_dialogues
        handler.on_message_handled(MagicMock())
        assert ctx.state.request_count == 100
        mock_dialogues.cleanup.assert_called_once()

    def test_on_message_handled_no_cleanup_before_freq(self) -> None:
        """Test on_message_handled does not trigger cleanup before freq."""
        handler, ctx = _make_concrete_base_handler()
        ctx.state.request_count = 98
        ctx.params.cleanup_freq = 100
        handler.on_message_handled(MagicMock())
        assert ctx.state.request_count == 99
        # cleanup_dialogues is not called because 99 % 100 != 0


class TestKvStoreHandler:
    """Test KvStoreHandler class."""

    def test_supported_protocol(self) -> None:
        """Test SUPPORTED_PROTOCOL is set."""
        from packages.valory.protocols.kv_store.message import KvStoreMessage

        assert KvStoreHandler.SUPPORTED_PROTOCOL == KvStoreMessage.protocol_id

    def test_allowed_response_performatives(self) -> None:
        """Test allowed_response_performatives is set."""
        from packages.valory.protocols.kv_store.message import KvStoreMessage

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
        """Test handle with unrecognized performative."""
        handler = KvStoreHandler.__new__(KvStoreHandler)
        mock_context = MagicMock()
        object.__setattr__(handler, "_context", mock_context)
        mock_context.state.in_flight_req = True

        msg = MagicMock()
        msg.performative = "unknown_performative"
        handler.handle(msg)
        assert mock_context.state.in_flight_req is False

    def test_handle_success_with_callback(self) -> None:
        """Test handle SUCCESS with callback in req_to_callback."""
        from packages.valory.protocols.kv_store.message import KvStoreMessage

        handler = KvStoreHandler.__new__(KvStoreHandler)
        mock_context = MagicMock()
        object.__setattr__(handler, "_context", mock_context)

        callback = MagicMock()
        mock_context.state.req_to_callback = {"nonce1": (callback, {"k": "v"})}
        mock_context.state.in_flight_req = True

        msg = MagicMock()
        msg.performative = KvStoreMessage.Performative.SUCCESS
        msg.dialogue_reference = ("nonce1", "")

        handler.handle(msg)
        callback.assert_called_once()
        assert mock_context.state.in_flight_req is False

    def test_handle_success_without_callback(self) -> None:
        """Test handle SUCCESS without callback delegates to super."""
        from packages.valory.protocols.kv_store.message import KvStoreMessage

        handler = KvStoreHandler.__new__(KvStoreHandler)
        mock_context = MagicMock()
        object.__setattr__(handler, "_context", mock_context)
        mock_context.state.req_to_callback = {}

        msg = MagicMock()
        msg.performative = KvStoreMessage.Performative.SUCCESS
        msg.dialogue_reference = ("nonce_missing", "")

        with patch.object(KvStoreHandler.__bases__[0], "handle") as mock_super_handle:
            handler.handle(msg)
            mock_super_handle.assert_called_once()

    def test_handle_read_response_with_callback(self) -> None:
        """Test handle READ_RESPONSE with callback."""
        from packages.valory.protocols.kv_store.message import KvStoreMessage

        handler = KvStoreHandler.__new__(KvStoreHandler)
        mock_context = MagicMock()
        object.__setattr__(handler, "_context", mock_context)

        callback = MagicMock()
        mock_context.state.req_to_callback = {"nonce2": (callback, {})}
        mock_context.state.in_flight_req = True

        msg = MagicMock()
        msg.performative = KvStoreMessage.Performative.READ_RESPONSE
        msg.dialogue_reference = ("nonce2", "")

        handler.handle(msg)
        callback.assert_called_once()
        assert mock_context.state.in_flight_req is False

    def test_handle_error_delegates_to_super(self) -> None:
        """Test handle ERROR delegates to super."""
        from packages.valory.protocols.kv_store.message import KvStoreMessage

        handler = KvStoreHandler.__new__(KvStoreHandler)
        mock_context = MagicMock()
        object.__setattr__(handler, "_context", mock_context)

        msg = MagicMock()
        msg.performative = KvStoreMessage.Performative.ERROR

        with patch.object(KvStoreHandler.__bases__[0], "handle") as mock_super_handle:
            handler.handle(msg)
            mock_super_handle.assert_called_once()


class TestSrrHandler:
    """Test SrrHandler class."""

    def test_supported_protocol(self) -> None:
        """Test SUPPORTED_PROTOCOL is set."""
        from packages.valory.protocols.srr.message import SrrMessage

        assert SrrHandler.SUPPORTED_PROTOCOL == SrrMessage.protocol_id

    def test_allowed_response_performatives(self) -> None:
        """Test allowed_response_performatives is set."""
        from packages.valory.protocols.srr.message import SrrMessage

        expected = frozenset(
            {
                SrrMessage.Performative.REQUEST,
                SrrMessage.Performative.RESPONSE,
            }
        )
        assert SrrHandler.allowed_response_performatives == expected

    def test_handle_unrecognized_performative(self) -> None:
        """Test handle with unrecognized performative."""
        handler = SrrHandler.__new__(SrrHandler)
        mock_context = MagicMock()
        object.__setattr__(handler, "_context", mock_context)
        mock_context.state.in_flight_req = True

        msg = MagicMock()
        msg.performative = "unrecognized"

        handler.handle(msg)
        assert mock_context.state.in_flight_req is False

    def test_handle_with_callback(self) -> None:
        """Test handle RESPONSE with callback."""
        from packages.valory.protocols.srr.message import SrrMessage

        handler = SrrHandler.__new__(SrrHandler)
        mock_context = MagicMock()
        object.__setattr__(handler, "_context", mock_context)

        callback = MagicMock()
        mock_context.state.req_to_callback = {"nonce_srr": (callback, {"x": 1})}
        mock_context.state.in_flight_req = True

        msg = MagicMock(spec=SrrMessage)
        msg.performative = SrrMessage.Performative.RESPONSE
        msg.dialogue_reference = ("nonce_srr", "")

        handler.handle(msg)
        callback.assert_called_once()
        assert mock_context.state.in_flight_req is False

    def test_handle_without_callback(self) -> None:
        """Test handle RESPONSE without callback delegates to super."""
        from packages.valory.protocols.srr.message import SrrMessage

        handler = SrrHandler.__new__(SrrHandler)
        mock_context = MagicMock()
        object.__setattr__(handler, "_context", mock_context)
        mock_context.state.req_to_callback = {}

        msg = MagicMock(spec=SrrMessage)
        msg.performative = SrrMessage.Performative.RESPONSE
        msg.dialogue_reference = ("no_such_nonce", "")

        with patch.object(SrrHandler.__bases__[0], "handle") as mock_super_handle:
            handler.handle(msg)
            mock_super_handle.assert_called_once()


def _make_http_handler() -> Any:
    """Create an HttpHandler instance with mocked dependencies."""
    handler = HttpHandler.__new__(HttpHandler)
    mock_context = MagicMock()
    object.__setattr__(handler, "_context", mock_context)
    handler.json_content_header = "Content-Type: application/json\n"
    handler.html_content_header = "Content-Type: text/html\n"
    handler.available_strategies = ["balancer_pools_search", "velodrome_pools_search"]
    handler.agent_profile_path = OPTIMUS_AGENT_PROFILE_PATH
    handler.rounds_info = {}
    handler.handler_url_regex = r".*localhost(:\d+)?\/.*"
    handler.routes = {}
    # Off by default, as the config has it. Left as a MagicMock this reads
    # truthy and silently puts every test on the facilitator route.
    mock_context.coingecko.use_mech_facilitator = False
    return handler, mock_context


class TestHttpHandlerMethods:
    """Test individual methods of the HttpHandler."""

    def test_get_content_type_html(self) -> None:
        """Test _get_content_type for html."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".html") == "text/html"

    def test_get_content_type_css(self) -> None:
        """Test _get_content_type for css."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".css") == "text/css"

    def test_get_content_type_js(self) -> None:
        """Test _get_content_type for javascript."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".js") == "application/javascript"

    def test_get_content_type_json(self) -> None:
        """Test _get_content_type for json."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".json") == "application/json"

    def test_get_content_type_png(self) -> None:
        """Test _get_content_type for png."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".png") == "image/png"

    def test_get_content_type_jpg(self) -> None:
        """Test _get_content_type for jpg."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".jpg") == "image/jpeg"

    def test_get_content_type_jpeg(self) -> None:
        """Test _get_content_type for jpeg."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".jpeg") == "image/jpeg"

    def test_get_content_type_gif(self) -> None:
        """Test _get_content_type for gif."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".gif") == "image/gif"

    def test_get_content_type_svg(self) -> None:
        """Test _get_content_type for svg."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".svg") == "image/svg+xml"

    def test_get_content_type_ico(self) -> None:
        """Test _get_content_type for ico."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".ico") == "image/x-icon"

    def test_get_content_type_txt(self) -> None:
        """Test _get_content_type for txt."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".txt") == "text/plain"

    def test_get_content_type_pdf(self) -> None:
        """Test _get_content_type for pdf."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".pdf") == "application/pdf"

    def test_get_content_type_woff(self) -> None:
        """Test _get_content_type for woff."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".woff") == "font/woff"

    def test_get_content_type_woff2(self) -> None:
        """Test _get_content_type for woff2."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".woff2") == "font/woff2"

    def test_get_content_type_ttf(self) -> None:
        """Test _get_content_type for ttf."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".ttf") == "font/ttf"

    def test_get_content_type_eot(self) -> None:
        """Test _get_content_type for eot."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".eot") == "application/vnd.ms-fontobject"

    def test_get_content_type_unknown(self) -> None:
        """Test _get_content_type for unknown extension."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".xyz") == "application/octet-stream"

    def test_get_content_type_case_insensitive(self) -> None:
        """Test _get_content_type is case insensitive."""
        handler, _ = _make_http_handler()
        assert handler._get_content_type(".HTML") == "text/html"

    def test_is_valid_ethereum_address_valid(self) -> None:
        """Test _is_valid_ethereum_address with valid address."""
        handler, _ = _make_http_handler()
        assert handler._is_valid_ethereum_address(
            "0x1234567890abcdef1234567890abcdef12345678"
        )

    def test_is_valid_ethereum_address_valid_mixed_case(self) -> None:
        """Test _is_valid_ethereum_address with mixed case address."""
        handler, _ = _make_http_handler()
        assert handler._is_valid_ethereum_address(
            "0xAbCdEf1234567890AbCdEf1234567890AbCdEf12"
        )

    def test_is_valid_ethereum_address_invalid_no_prefix(self) -> None:
        """Test _is_valid_ethereum_address with no 0x prefix."""
        handler, _ = _make_http_handler()
        assert not handler._is_valid_ethereum_address(
            "1234567890abcdef1234567890abcdef12345678"
        )

    def test_is_valid_ethereum_address_invalid_short(self) -> None:
        """Test _is_valid_ethereum_address with short address."""
        handler, _ = _make_http_handler()
        assert not handler._is_valid_ethereum_address("0x1234")

    def test_is_valid_ethereum_address_invalid_long(self) -> None:
        """Test _is_valid_ethereum_address with long address."""
        handler, _ = _make_http_handler()
        assert not handler._is_valid_ethereum_address(
            "0x1234567890abcdef1234567890abcdef1234567890"
        )

    def test_is_valid_ethereum_address_empty(self) -> None:
        """Test _is_valid_ethereum_address with empty string."""
        handler, _ = _make_http_handler()
        assert not handler._is_valid_ethereum_address("")

    def test_is_valid_ethereum_address_invalid_chars(self) -> None:
        """Test _is_valid_ethereum_address with invalid characters."""
        handler, _ = _make_http_handler()
        assert not handler._is_valid_ethereum_address(
            "0xGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
        )

    def test_get_transaction_link_optimism(self) -> None:
        """Test _get_transaction_link for optimism."""
        handler, _ = _make_http_handler()
        link = handler._get_transaction_link("optimism", "0xabc")
        assert link == "https://optimistic.etherscan.io/tx/0xabc"

    def test_get_transaction_link_base(self) -> None:
        """Test _get_transaction_link for base."""
        handler, _ = _make_http_handler()
        link = handler._get_transaction_link("base", "0xdef")
        assert link == "https://basescan.org/tx/0xdef"

    def test_get_transaction_link_mode(self) -> None:
        """Test _get_transaction_link for mode."""
        handler, _ = _make_http_handler()
        link = handler._get_transaction_link("mode", "0x123")
        assert link == "https://explorer-mode-mainnet-0.t.conduit.xyz/tx/0x123"

    def test_get_transaction_link_ethereum(self) -> None:
        """Test _get_transaction_link for ethereum."""
        handler, _ = _make_http_handler()
        link = handler._get_transaction_link("ethereum", "0x456")
        assert link == "https://etherscan.io/tx/0x456"

    def test_get_transaction_link_unknown_chain(self) -> None:
        """Test _get_transaction_link for unknown chain defaults to etherscan."""
        handler, _ = _make_http_handler()
        link = handler._get_transaction_link("polygon", "0x789")
        assert link == "https://etherscan.io/tx/0x789"

    def test_calculate_composite_score_from_var_balanced(self) -> None:
        """Test calculate_composite_score_from_var with a balanced VaR."""
        handler, _ = _make_http_handler()
        # VaR = -5/100 = -0.05
        score = handler.calculate_composite_score_from_var(-0.05)
        assert 0.20 <= score <= 0.50

    def test_calculate_composite_score_from_var_risky(self) -> None:
        """Test calculate_composite_score_from_var with a risky VaR."""
        handler, _ = _make_http_handler()
        # VaR = -15/100 = -0.15
        score = handler.calculate_composite_score_from_var(-0.15)
        assert 0.20 <= score <= 0.50

    def test_calculate_composite_score_from_var_with_correlation(self) -> None:
        """Test calculate_composite_score_from_var with a correlation coefficient."""
        handler, _ = _make_http_handler()
        score = handler.calculate_composite_score_from_var(-0.10, 0.5)
        assert 0.20 <= score <= 0.50

    def test_calculate_composite_score_from_var_bounds_min(self) -> None:
        """Test calculate_composite_score_from_var returns MIN_CS when score is too low."""
        handler, _ = _make_http_handler()
        # Use very small correlation to push CS below minimum
        score = handler.calculate_composite_score_from_var(-0.01, 0.01)
        assert score == 0.20

    def test_calculate_composite_score_from_var_bounds_max(self) -> None:
        """Test calculate_composite_score_from_var returns MAX_CS when score is too high."""
        handler, _ = _make_http_handler()
        # Large correlation to push CS above MAX
        score = handler.calculate_composite_score_from_var(-0.05, 10.0)
        assert score == 0.50

    def test_calculate_composite_score_from_var_division_by_zero(self) -> None:
        """Test calculate_composite_score_from_var handles division by zero."""
        handler, _ = _make_http_handler()
        # var + B = 0 means var = -B = -0.8272
        score = handler.calculate_composite_score_from_var(-0.8272)
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            THRESHOLDS,
            TradingType,
        )

        assert score == THRESHOLDS.get(TradingType.BALANCED.value, 0.3374)

    def test_calculate_composite_score_from_var_value_error(self) -> None:
        """Test calculate_composite_score_from_var handles negative log argument."""
        handler, _ = _make_http_handler()
        # When var + B < 0 and A/(var+B) < 0, log of negative number raises ValueError
        score = handler.calculate_composite_score_from_var(-1.0)
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            THRESHOLDS,
            TradingType,
        )

        assert score == THRESHOLDS.get(TradingType.BALANCED.value, 0.3374)

    def test_calculate_withdrawal_funding_deficit_sufficient_balance(self) -> None:
        """Test _calculate_withdrawal_funding_deficit when balance is sufficient."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        actions = [{"action": "exit"}, {"action": "swap"}]
        # balance > total_gas_needed
        result = handler._calculate_withdrawal_funding_deficit(actions, 10**15)
        assert result == {}

    def test_calculate_withdrawal_funding_deficit_insufficient_balance(self) -> None:
        """Test _calculate_withdrawal_funding_deficit when balance is insufficient."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        actions = [{"action": "exit"}, {"action": "swap"}]
        result = handler._calculate_withdrawal_funding_deficit(actions, 0)
        assert result != {}
        assert "optimism" in result
        assert "0xagent" in result["optimism"]

    def test_calculate_withdrawal_funding_deficit_exact_balance(self) -> None:
        """Test _calculate_withdrawal_funding_deficit when balance equals needed."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        actions = [{"action": "exit"}]
        total_gas_needed = int(ESTIMATED_GAS_PER_TX * 1 * 1.2)
        result = handler._calculate_withdrawal_funding_deficit(
            actions, total_gas_needed
        )
        assert result == {}

    def test_get_handler_no_match(self) -> None:
        """Test _get_handler when URL doesn't match base pattern."""
        handler, _ = _make_http_handler()
        handler.handler_url_regex = r".*localhost(:\d+)?\/.*"
        result, kwargs = handler._get_handler("https://example.com/test", "get")
        assert result is None
        assert kwargs == {}

    def test_get_handler_matches_route(self) -> None:
        """Test _get_handler when URL matches a route."""
        handler, _ = _make_http_handler()
        handler.handler_url_regex = r".*localhost(:\d+)?\/.*"
        mock_handler_fn = MagicMock()
        handler.routes = {
            ("get", "head"): [
                (r".*localhost(:\d+)?\/health", mock_handler_fn),
            ]
        }
        result, kwargs = handler._get_handler("http://localhost:8000/health", "get")
        assert result is mock_handler_fn

    def test_get_handler_no_route_match(self) -> None:
        """Test _get_handler when URL matches base but no specific route."""
        handler, _ = _make_http_handler()
        handler.handler_url_regex = r".*localhost(:\d+)?\/.*"
        handler.routes = {
            ("get",): [
                (r".*localhost(:\d+)?\/health", MagicMock()),
            ]
        }
        result, kwargs = handler._get_handler("http://localhost:8000/unknown", "get")
        # Should return _handle_bad_request
        assert result is not None

    def test_get_handler_wrong_method(self) -> None:
        """Test _get_handler when method doesn't match."""
        handler, _ = _make_http_handler()
        handler.handler_url_regex = r".*localhost(:\d+)?\/.*"
        handler.routes = {
            ("get",): [
                (r".*localhost(:\d+)?\/health", MagicMock()),
            ]
        }
        result, kwargs = handler._get_handler("http://localhost:8000/health", "post")
        # Method 'post' doesn't match 'get' routes; falls through to bad_request
        assert result is not None

    def test_has_deficit_true(self) -> None:
        """Test _has_deficit returns True when deficit > 0."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        deficit = {
            "optimism": {
                "0xagent": {
                    "0x0000000000000000000000000000000000000000": {"deficit": 100}
                }
            }
        }
        assert handler._has_deficit(deficit) is True

    def test_has_deficit_false(self) -> None:
        """Test _has_deficit returns False when deficit is 0."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        deficit = {
            "optimism": {
                "0xagent": {
                    "0x0000000000000000000000000000000000000000": {"deficit": 0}
                }
            }
        }
        assert handler._has_deficit(deficit) is False

    def test_has_deficit_missing_chain(self) -> None:
        """Test _has_deficit returns False when chain is missing."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        assert handler._has_deficit({}) is False

    def test_has_deficit_invalid_value(self) -> None:
        """Test _has_deficit returns False when deficit value is invalid."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        deficit = {
            "optimism": {
                "0xagent": {
                    "0x0000000000000000000000000000000000000000": {
                        "deficit": "not_a_number"
                    }
                }
            }
        }
        assert handler._has_deficit(deficit) is False

    def test_is_in_withdrawal_mode_true(self) -> None:
        """Test _is_in_withdrawal_mode returns True when investing is paused."""
        handler, _ = _make_http_handler()
        handler._read_withdrawal_data = MagicMock(
            return_value={"investing_paused": "true"}
        )
        assert handler._is_in_withdrawal_mode() is True

    def test_is_in_withdrawal_mode_false(self) -> None:
        """Test _is_in_withdrawal_mode returns False when investing is not paused."""
        handler, _ = _make_http_handler()
        handler._read_withdrawal_data = MagicMock(
            return_value={"investing_paused": "false"}
        )
        assert handler._is_in_withdrawal_mode() is False

    def test_is_in_withdrawal_mode_no_data(self) -> None:
        """Test _is_in_withdrawal_mode returns falsy when no withdrawal data."""
        handler, _ = _make_http_handler()
        handler._read_withdrawal_data = MagicMock(return_value=None)
        assert not handler._is_in_withdrawal_mode()

    def test_is_in_withdrawal_mode_empty_dict(self) -> None:
        """Test _is_in_withdrawal_mode returns falsy with empty dict."""
        handler, _ = _make_http_handler()
        handler._read_withdrawal_data = MagicMock(return_value={})
        assert not handler._is_in_withdrawal_mode()

    def test_get_withdrawal_actions_success(self) -> None:
        """Test _get_withdrawal_actions returns actions from synced data."""
        handler, ctx = _make_http_handler()
        actions = [{"action": "exit"}, {"action": "swap"}]
        mock_synced = MagicMock()
        mock_synced.db.get.return_value = json.dumps(actions)
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_synced,
        ):
            result = handler._get_withdrawal_actions()
        assert result == actions

    def test_get_withdrawal_actions_empty(self) -> None:
        """Test _get_withdrawal_actions returns empty list when no actions."""
        handler, ctx = _make_http_handler()
        mock_synced = MagicMock()
        mock_synced.db.get.return_value = "[]"
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_synced,
        ):
            result = handler._get_withdrawal_actions()
        assert result == []

    def test_get_withdrawal_actions_exception(self) -> None:
        """Test _get_withdrawal_actions returns empty list on exception."""
        handler, ctx = _make_http_handler()
        mock_synced = MagicMock()
        mock_synced.db.get.side_effect = Exception("DB error")
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_synced,
        ):
            result = handler._get_withdrawal_actions()
        assert result == []

    def test_get_withdrawal_actions_none(self) -> None:
        """Test _get_withdrawal_actions returns empty list when db returns None."""
        handler, ctx = _make_http_handler()
        mock_synced = MagicMock()
        mock_synced.db.get.return_value = None
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_synced,
        ):
            result = handler._get_withdrawal_actions()
        assert result == []

    def test_get_withdrawal_actions_pre_fsm_attribute_error(self) -> None:
        """Test _get_withdrawal_actions returns [] before FSM init."""
        handler, ctx = _make_http_handler()
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            side_effect=AttributeError("FSM not initialized"),
        ):
            result = handler._get_withdrawal_actions()
        assert result == []

    def test_get_withdrawal_actions_pre_fsm_value_error(self) -> None:
        """Test _get_withdrawal_actions returns [] on ValueError."""
        handler, ctx = _make_http_handler()
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            side_effect=ValueError("no round"),
        ):
            result = handler._get_withdrawal_actions()
        assert result == []

    def test_get_password_from_args_with_flag(self) -> None:
        """Test _get_password_from_args with --password flag."""
        handler, _ = _make_http_handler()
        with patch("sys.argv", ["cmd", "--password", "secret123"]):
            result = handler._get_password_from_args()
        assert result == "secret123"

    def test_get_password_from_args_with_equals(self) -> None:
        """Test _get_password_from_args with --password=value format."""
        handler, _ = _make_http_handler()
        with patch("sys.argv", ["cmd", "--password=secret123"]):
            result = handler._get_password_from_args()
        assert result == "secret123"

    def test_get_password_from_args_no_password(self) -> None:
        """Test _get_password_from_args returns None when no password."""
        handler, _ = _make_http_handler()
        with patch("sys.argv", ["cmd"]):
            result = handler._get_password_from_args()
        assert result is None

    def test_get_password_from_args_flag_at_end(self) -> None:
        """Test _get_password_from_args when --password is last arg."""
        handler, _ = _make_http_handler()
        with patch("sys.argv", ["cmd", "--password"]):
            result = handler._get_password_from_args()
        # password_index + 1 is not < len(args), so falls through
        assert result is None

    def test_send_message(self) -> None:
        """Test _send_message puts message in outbox and stores callback."""
        handler, ctx = _make_http_handler()
        ctx.state.req_to_callback = {}
        ctx.state.in_flight_req = False
        message = MagicMock()
        dialogue = MagicMock()
        dialogue.dialogue_label.dialogue_reference = ("nonce123", "")
        callback = MagicMock()

        handler._send_message(message, dialogue, callback, {"key": "val"})

        ctx.outbox.put_message.assert_called_once_with(message=message)
        assert "nonce123" in ctx.state.req_to_callback
        assert ctx.state.in_flight_req is True

    def test_send_message_no_kwargs(self) -> None:
        """Test _send_message with no callback kwargs."""
        handler, ctx = _make_http_handler()
        ctx.state.req_to_callback = {}
        ctx.state.in_flight_req = False
        message = MagicMock()
        dialogue = MagicMock()
        dialogue.dialogue_label.dialogue_reference = ("nonce456", "")
        callback = MagicMock()

        handler._send_message(message, dialogue, callback)

        assert ctx.state.req_to_callback["nonce456"] == (callback, {})

    def test_handle_kv_store_response_success(self) -> None:
        """Test _handle_kv_store_response logs success."""
        from packages.valory.protocols.kv_store.message import KvStoreMessage

        handler, _ = _make_http_handler()
        msg = MagicMock()
        msg.performative = KvStoreMessage.Performative.SUCCESS
        handler._handle_kv_store_response(msg, MagicMock())

    def test_handle_kv_store_response_failure(self) -> None:
        """Test _handle_kv_store_response logs failure."""
        from packages.valory.protocols.kv_store.message import KvStoreMessage

        handler, _ = _make_http_handler()
        msg = MagicMock()
        msg.performative = KvStoreMessage.Performative.ERROR
        handler._handle_kv_store_response(msg, MagicMock())

    def test_handle_kv_read_response_success(self) -> None:
        """Test _handle_kv_read_response stores data on success."""
        from packages.valory.protocols.kv_store.message import KvStoreMessage

        handler, ctx = _make_http_handler()
        msg = MagicMock()
        msg.performative = KvStoreMessage.Performative.READ_RESPONSE
        msg.data = {"key": "value"}
        handler._handle_kv_read_response(msg, MagicMock())
        assert ctx.state.last_kv_read_data == {"key": "value"}
        assert ctx.state.in_flight_req is False

    def test_handle_kv_read_response_success_no_data_attr(self) -> None:
        """Test _handle_kv_read_response when msg has no data attribute."""
        from packages.valory.protocols.kv_store.message import KvStoreMessage

        handler, ctx = _make_http_handler()
        msg = MagicMock(spec=[])
        msg.performative = KvStoreMessage.Performative.READ_RESPONSE
        handler._handle_kv_read_response(msg, MagicMock())
        # hasattr(msg, 'data') is False since we used spec=[]
        assert ctx.state.in_flight_req is False

    def test_handle_kv_read_response_failure(self) -> None:
        """Test _handle_kv_read_response sets empty data on failure."""
        from packages.valory.protocols.kv_store.message import KvStoreMessage

        handler, ctx = _make_http_handler()
        msg = MagicMock()
        msg.performative = KvStoreMessage.Performative.ERROR
        handler._handle_kv_read_response(msg, MagicMock())
        assert ctx.state.last_kv_read_data == {}
        assert ctx.state.in_flight_req is False

    def test_handle_get_features_x402_enabled(self) -> None:
        """Test _handle_get_features when x402 is enabled."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = True
        handler._send_ok_response = MagicMock()
        mock_msg = MagicMock()
        mock_dialogue = MagicMock()
        handler._handle_get_features(mock_msg, mock_dialogue)
        handler._send_ok_response.assert_called_once_with(
            mock_msg, mock_dialogue, {"isChatEnabled": True}
        )

    def test_handle_get_features_api_key_set(self) -> None:
        """Test _handle_get_features when API key is set."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.params.genai_api_key = "valid_api_key"
        handler._send_ok_response = MagicMock()
        mock_msg = MagicMock()
        mock_dialogue = MagicMock()
        handler._handle_get_features(mock_msg, mock_dialogue)
        handler._send_ok_response.assert_called_once_with(
            mock_msg, mock_dialogue, {"isChatEnabled": True}
        )

    def test_handle_get_features_api_key_empty(self) -> None:
        """Test _handle_get_features when API key is empty."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.params.genai_api_key = ""
        handler._send_ok_response = MagicMock()
        mock_msg = MagicMock()
        mock_dialogue = MagicMock()
        handler._handle_get_features(mock_msg, mock_dialogue)
        handler._send_ok_response.assert_called_once_with(
            mock_msg, mock_dialogue, {"isChatEnabled": False}
        )

    def test_handle_get_features_api_key_none(self) -> None:
        """Test _handle_get_features when API key is None."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.params.genai_api_key = None
        handler._send_ok_response = MagicMock()
        mock_msg = MagicMock()
        mock_dialogue = MagicMock()
        handler._handle_get_features(mock_msg, mock_dialogue)
        handler._send_ok_response.assert_called_once_with(
            mock_msg, mock_dialogue, {"isChatEnabled": False}
        )

    def test_handle_get_features_api_key_placeholder(self) -> None:
        """Test _handle_get_features when API key is a placeholder."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.params.genai_api_key = "${str:}"
        handler._send_ok_response = MagicMock()
        mock_msg = MagicMock()
        mock_dialogue = MagicMock()
        handler._handle_get_features(mock_msg, mock_dialogue)
        handler._send_ok_response.assert_called_once_with(
            mock_msg, mock_dialogue, {"isChatEnabled": False}
        )

    def test_handle_get_features_api_key_quoted_empty(self) -> None:
        """Test _handle_get_features when API key is double-quoted empty."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.params.genai_api_key = '""'
        handler._send_ok_response = MagicMock()
        mock_msg = MagicMock()
        mock_dialogue = MagicMock()
        handler._handle_get_features(mock_msg, mock_dialogue)
        handler._send_ok_response.assert_called_once_with(
            mock_msg, mock_dialogue, {"isChatEnabled": False}
        )

    def test_handle_get_features_api_key_not_string(self) -> None:
        """Test _handle_get_features when API key is not a string."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.params.genai_api_key = 12345
        handler._send_ok_response = MagicMock()
        mock_msg = MagicMock()
        mock_dialogue = MagicMock()
        handler._handle_get_features(mock_msg, mock_dialogue)
        handler._send_ok_response.assert_called_once_with(
            mock_msg, mock_dialogue, {"isChatEnabled": False}
        )

    def test_send_ok_response_dict(self) -> None:
        """Test _send_ok_response with dict data."""
        handler, _ = _make_http_handler()
        mock_msg = MagicMock()
        mock_msg.version = "1.1"
        mock_msg.headers = "Host: localhost"
        mock_dialogue = MagicMock()

        handler._send_ok_response(mock_msg, mock_dialogue, {"key": "value"})

        mock_dialogue.reply.assert_called_once()
        call_kwargs = mock_dialogue.reply.call_args[1]
        assert call_kwargs["status_code"] == OK_CODE
        assert json.loads(call_kwargs["body"].decode()) == {"key": "value"}

    def test_send_ok_response_string(self) -> None:
        """Test _send_ok_response with string data."""
        handler, _ = _make_http_handler()
        mock_msg = MagicMock()
        mock_msg.version = "1.1"
        mock_msg.headers = "Host: localhost"
        mock_dialogue = MagicMock()

        handler._send_ok_response(mock_msg, mock_dialogue, "<html></html>")

        mock_dialogue.reply.assert_called_once()
        call_kwargs = mock_dialogue.reply.call_args[1]
        assert call_kwargs["body"] == b"<html></html>"
        assert "text/html" in call_kwargs["headers"]

    def test_send_ok_response_bytes(self) -> None:
        """Test _send_ok_response with bytes data."""
        handler, _ = _make_http_handler()
        mock_msg = MagicMock()
        mock_msg.version = "1.1"
        mock_msg.headers = "Host: localhost"
        mock_dialogue = MagicMock()

        handler._send_ok_response(mock_msg, mock_dialogue, b"\x89PNG", "image/png")

        mock_dialogue.reply.assert_called_once()
        call_kwargs = mock_dialogue.reply.call_args[1]
        assert call_kwargs["body"] == b"\x89PNG"
        assert "image/png" in call_kwargs["headers"]

    def test_send_ok_response_bytes_no_content_type(self) -> None:
        """Test _send_ok_response with bytes data and no content type."""
        handler, _ = _make_http_handler()
        mock_msg = MagicMock()
        mock_msg.version = "1.1"
        mock_msg.headers = "Host: localhost"
        mock_dialogue = MagicMock()

        handler._send_ok_response(mock_msg, mock_dialogue, b"\x89PNG")

        mock_dialogue.reply.assert_called_once()
        call_kwargs = mock_dialogue.reply.call_args[1]
        assert "application/json" in call_kwargs["headers"]

    def test_send_ok_response_string_with_content_type(self) -> None:
        """Test _send_ok_response with string data and custom content type."""
        handler, _ = _make_http_handler()
        mock_msg = MagicMock()
        mock_msg.version = "1.1"
        mock_msg.headers = "Host: localhost"
        mock_dialogue = MagicMock()

        handler._send_ok_response(mock_msg, mock_dialogue, "<html></html>", "text/html")

        mock_dialogue.reply.assert_called_once()
        call_kwargs = mock_dialogue.reply.call_args[1]
        assert "text/html" in call_kwargs["headers"]

    def test_send_ok_response_string_no_content_type(self) -> None:
        """Test _send_ok_response with string data and no content type uses html."""
        handler, _ = _make_http_handler()
        mock_msg = MagicMock()
        mock_msg.version = "1.1"
        mock_msg.headers = "Host: localhost"
        mock_dialogue = MagicMock()

        handler._send_ok_response(mock_msg, mock_dialogue, "<html></html>")

        call_kwargs = mock_dialogue.reply.call_args[1]
        assert "text/html" in call_kwargs["headers"]

    def test_send_ok_response_list(self) -> None:
        """Test _send_ok_response with list data."""
        handler, _ = _make_http_handler()
        mock_msg = MagicMock()
        mock_msg.version = "1.1"
        mock_msg.headers = "Host: localhost"
        mock_dialogue = MagicMock()

        handler._send_ok_response(mock_msg, mock_dialogue, [1, 2, 3])

        mock_dialogue.reply.assert_called_once()
        call_kwargs = mock_dialogue.reply.call_args[1]
        assert json.loads(call_kwargs["body"].decode()) == [1, 2, 3]

    def test_handle_bad_request(self) -> None:
        """Test _handle_bad_request sends 400 response."""
        handler, _ = _make_http_handler()
        mock_msg = MagicMock()
        mock_msg.version = "1.1"
        mock_msg.headers = "Host: localhost"
        mock_dialogue = MagicMock()

        handler._handle_bad_request(mock_msg, mock_dialogue)

        mock_dialogue.reply.assert_called_once()
        call_kwargs = mock_dialogue.reply.call_args[1]
        assert call_kwargs["status_code"] == HttpCode.BAD_REQUEST_CODE.value
        assert call_kwargs["body"] == b""

    def test_handle_bad_request_with_error_msg(self) -> None:
        """Test _handle_bad_request sends 400 response with error message."""
        handler, _ = _make_http_handler()
        mock_msg = MagicMock()
        mock_msg.version = "1.1"
        mock_msg.headers = "Host: localhost"
        mock_dialogue = MagicMock()

        handler._handle_bad_request(mock_msg, mock_dialogue, error_msg="test error")

        mock_dialogue.reply.assert_called_once()
        call_kwargs = mock_dialogue.reply.call_args[1]
        assert call_kwargs["body"] == b"test error"

    def test_handle_not_found(self) -> None:
        """Test _handle_not_found sends 404 response."""
        handler, _ = _make_http_handler()
        mock_msg = MagicMock()
        mock_msg.version = "1.1"
        mock_msg.headers = "Host: localhost"
        mock_dialogue = MagicMock()

        handler._handle_not_found(mock_msg, mock_dialogue)

        mock_dialogue.reply.assert_called_once()
        call_kwargs = mock_dialogue.reply.call_args[1]
        assert call_kwargs["status_code"] == HttpCode.NOT_FOUND_CODE.value

    def test_handle_internal_error_sends_500(self) -> None:
        """_handle_internal_error sends an HTTP 500 response."""
        handler, _ = _make_http_handler()
        mock_msg = MagicMock()
        mock_msg.version = "1.1"
        mock_msg.headers = "Host: localhost"
        mock_dialogue = MagicMock()

        handler._handle_internal_error(mock_msg, mock_dialogue, error_msg="boom")

        mock_dialogue.reply.assert_called_once()
        call_kwargs = mock_dialogue.reply.call_args[1]
        assert call_kwargs["status_code"] == HttpCode.INTERNAL_SERVER_ERROR.value
        assert call_kwargs["status_text"] == "Internal server error"
        assert call_kwargs["body"] == b"boom"

    def test_handle_dispatch_catches_handler_exception(self) -> None:
        """A handler that raises must trigger an HTTP 500 reply, not propagate."""
        from packages.valory.connections.http_server.connection import (
            PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
        )
        from packages.valory.protocols.http.message import HttpMessage

        handler, ctx = _make_http_handler()

        raising_handler = MagicMock(side_effect=RuntimeError("kaboom"))
        handler._get_handler = MagicMock(return_value=(raising_handler, {}))

        http_msg = MagicMock(spec=HttpMessage)
        http_msg.performative = HttpMessage.Performative.REQUEST
        http_msg.method = "GET"
        http_msg.url = "http://localhost/test"
        http_msg.body = b""
        http_msg.version = "1.1"
        http_msg.headers = "Host: localhost"
        http_msg.sender = str(HTTP_SERVER_PUBLIC_ID.without_hash())

        mock_dialogue = MagicMock()
        ctx.http_dialogues.update.return_value = mock_dialogue

        with patch.object(handler, "_handle_internal_error") as mock_internal_error:
            handler.handle(http_msg)

        raising_handler.assert_called_once()
        mock_internal_error.assert_called_once()
        call_args = mock_internal_error.call_args
        assert call_args.args[0] is http_msg
        assert call_args.args[1] is mock_dialogue
        # Body must be generic — no exception text leaks to the HTTP caller.
        assert call_args.kwargs["error_msg"] == "Internal server error"
        # The exception detail is preserved in the server-side log only.
        log_args = ctx.logger.exception.call_args_list[0]
        assert "kaboom" in log_args.args[0]

    def test_handle_dispatch_writes_500_to_outbox_end_to_end(self) -> None:
        """A raising handler results in a 500 envelope on outbox.put_message."""
        from packages.valory.connections.http_server.connection import (
            PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
        )
        from packages.valory.protocols.http.message import HttpMessage

        handler, ctx = _make_http_handler()

        raising_handler = MagicMock(side_effect=RuntimeError("internal-detail"))
        handler._get_handler = MagicMock(return_value=(raising_handler, {}))

        http_msg = MagicMock(spec=HttpMessage)
        http_msg.performative = HttpMessage.Performative.REQUEST
        http_msg.method = "GET"
        http_msg.url = "http://localhost/test"
        http_msg.body = b""
        http_msg.version = "1.1"
        http_msg.headers = "Host: localhost"
        http_msg.sender = str(HTTP_SERVER_PUBLIC_ID.without_hash())

        reply_msg = MagicMock()
        mock_dialogue = MagicMock()
        mock_dialogue.reply.return_value = reply_msg
        ctx.http_dialogues.update.return_value = mock_dialogue

        handler.handle(http_msg)

        mock_dialogue.reply.assert_called_once()
        reply_kwargs = mock_dialogue.reply.call_args.kwargs
        assert reply_kwargs["status_code"] == HttpCode.INTERNAL_SERVER_ERROR.value
        assert reply_kwargs["body"] == b"Internal server error"
        # The internal exception text is NOT echoed in the response body.
        assert b"internal-detail" not in reply_kwargs["body"]
        ctx.outbox.put_message.assert_called_once_with(message=reply_msg)

    def test_handle_dispatch_swallows_error_reply_exception(self) -> None:
        """If _handle_internal_error itself raises, the failure is logged, not propagated."""
        from packages.valory.connections.http_server.connection import (
            PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
        )
        from packages.valory.protocols.http.message import HttpMessage

        handler, ctx = _make_http_handler()

        raising_handler = MagicMock(side_effect=RuntimeError("kaboom"))
        handler._get_handler = MagicMock(return_value=(raising_handler, {}))

        http_msg = MagicMock(spec=HttpMessage)
        http_msg.performative = HttpMessage.Performative.REQUEST
        http_msg.method = "GET"
        http_msg.url = "http://localhost/test"
        http_msg.body = b""
        http_msg.version = "1.1"
        http_msg.headers = "Host: localhost"
        http_msg.sender = str(HTTP_SERVER_PUBLIC_ID.without_hash())

        mock_dialogue = MagicMock()
        ctx.http_dialogues.update.return_value = mock_dialogue

        with patch.object(
            handler,
            "_handle_internal_error",
            side_effect=RuntimeError("reply failed"),
        ):
            handler.handle(http_msg)

        # Two .exception calls: one for the original handler error, one for the reply error.
        assert ctx.logger.exception.call_count == 2
        log_messages = [call.args[0] for call in ctx.logger.exception.call_args_list]
        assert any("Failed to send error reply" in m for m in log_messages)
        assert any("reply failed" in m for m in log_messages)

    def test_synchronized_data_property(self) -> None:
        """Test synchronized_data property."""
        handler, ctx = _make_http_handler()
        mock_db = MagicMock()
        ctx.state.round_sequence.latest_synchronized_data.db = mock_db
        result = handler.synchronized_data
        assert result is not None

    def test_shared_state_property(self) -> None:
        """Test shared_state property."""
        handler, ctx = _make_http_handler()
        result = handler.shared_state
        assert result is ctx.state

    def test_params_property(self) -> None:
        """Test params property."""
        handler, ctx = _make_http_handler()
        result = handler.params
        assert result is ctx.params

    def test_funds_status_property(self) -> None:
        """Test funds_status property."""
        handler, ctx = _make_http_handler()
        from packages.valory.skills.funds_manager.behaviours import (
            GET_FUNDS_STATUS_METHOD_NAME,
        )

        mock_fn = MagicMock()
        ctx.shared_state = {GET_FUNDS_STATUS_METHOD_NAME: mock_fn}
        _ = handler.funds_status  # noqa: F841 — property access triggers mock_fn
        mock_fn.assert_called_once()

    def test_read_kv(self) -> None:
        """Test _read_kv sends message and returns cached data."""
        handler, ctx = _make_http_handler()
        ctx.kv_store_dialogues.create.return_value = (MagicMock(), MagicMock())
        ctx.state.last_kv_read_data = {"cached": "data"}
        ctx.state.req_to_callback = {}
        ctx.state.in_flight_req = False
        result = handler._read_kv(("key1", "key2"))
        assert result == {"cached": "data"}

    def test_read_kv_exception(self) -> None:
        """Test _read_kv returns empty dict on exception."""
        handler, ctx = _make_http_handler()
        ctx.kv_store_dialogues.create.side_effect = Exception("KV error")
        result = handler._read_kv(("key1",))
        assert result == {}

    def test_write_kv(self) -> None:
        """Test _write_kv creates message and sends it."""
        handler, ctx = _make_http_handler()
        ctx.kv_store_dialogues.create.return_value = (MagicMock(), MagicMock())
        ctx.state.req_to_callback = {}
        ctx.state.in_flight_req = False
        handler._write_kv({"key": "value"})
        ctx.outbox.put_message.assert_called_once()

    def test_write_withdrawal_data(self) -> None:
        """Test _write_withdrawal_data calls _write_kv."""
        handler, ctx = _make_http_handler()
        handler._write_kv = MagicMock()
        data = {"withdrawal_id": "123"}
        handler._write_withdrawal_data(data)
        handler._write_kv.assert_called_once_with(data)

    def test_write_withdrawal_data_exception(self) -> None:
        """Test _write_withdrawal_data handles exception."""
        handler, ctx = _make_http_handler()
        handler._write_kv = MagicMock(side_effect=Exception("Write error"))
        handler._write_withdrawal_data({"withdrawal_id": "123"})
        ctx.logger.error.assert_called()

    def test_write_withdrawal_data_skips_when_lock_held(self) -> None:
        """A duplicate concurrent withdrawal write must short-circuit."""
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        handler, ctx = _make_http_handler()
        handler._write_kv = MagicMock()
        # Hold the module-level lock so the call sees a duplicate in flight
        acquired = handlers_mod._WITHDRAWAL_WRITE_LOCK.acquire(blocking=False)
        try:
            assert acquired is True
            handler._write_withdrawal_data({"withdrawal_id": "123"})
        finally:
            handlers_mod._WITHDRAWAL_WRITE_LOCK.release()
        handler._write_kv.assert_not_called()
        ctx.logger.info.assert_called()

    def test_write_withdrawal_data_releases_lock_after_exception(self) -> None:
        """Wrapped function raising must not leak the lock to subsequent callers."""
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        handler, _ = _make_http_handler()
        handler._write_kv = MagicMock(side_effect=RuntimeError("boom"))
        # First call hits the wrapped function, which raises. The except
        # branch swallows it. Lock must be released on the way out.
        handler._write_withdrawal_data({"withdrawal_id": "1"})
        # The lock should be free now — try acquiring without blocking.
        acquired = handlers_mod._WITHDRAWAL_WRITE_LOCK.acquire(blocking=False)
        try:
            assert acquired is True, "lock leaked across exception"
        finally:
            if acquired:
                handlers_mod._WITHDRAWAL_WRITE_LOCK.release()

    def test_submit_background_returns_before_task_completes(self) -> None:
        """The caller of _submit_background must not block on the task."""
        import threading

        handler, _ = _make_http_handler()
        release = threading.Event()
        observed = {"started": False, "done": False}

        def slow_task() -> None:
            observed["started"] = True
            release.wait(timeout=2.0)
            observed["done"] = True

        try:
            future = handler._submit_background(slow_task)
            # Submission must return immediately; the task is still pending.
            assert future.done() is False
            # Allow the task to complete and confirm the executor ran it.
            release.set()
            future.result(timeout=2.0)
            assert observed["started"] is True
            assert observed["done"] is True
        finally:
            release.set()
            handler.teardown()

    def test_submit_background_logs_exception_via_done_callback(self) -> None:
        """A background task that raises must surface via the done-callback log."""
        handler, ctx = _make_http_handler()

        def failing_task() -> None:
            raise RuntimeError("background-boom")

        try:
            future = handler._submit_background(failing_task)
            # Drain the future. .exception() blocks until the task and the
            # done-callback finish, so by the time it returns the logger must
            # already have been called.
            assert isinstance(future.exception(timeout=2.0), RuntimeError)
            assert any(
                "background-task-failed" in str(call.args)
                and "background-boom" in str(call.args)
                for call in ctx.logger.error.call_args_list
            )
        finally:
            handler.teardown()

    def test_teardown_shuts_down_background_executor(self) -> None:
        """teardown() must release the long-lived executor's worker threads."""
        handler, _ = _make_http_handler()
        # Force lazy creation.
        handler._submit_background(lambda: None).result(timeout=2.0)
        executor = handler._background_executor
        handler.teardown()
        # Re-submission after teardown is rejected — proves the executor
        # was actually shut down rather than left dangling.
        with pytest.raises(RuntimeError):
            executor.submit(lambda: None)
        # And the handler attribute is cleared so a future setup() lazily
        # creates a fresh executor instead of reusing the dead one.
        assert not hasattr(handler, "_background_executor")

    def test_read_withdrawal_data(self) -> None:
        """Test _read_withdrawal_data reads from KV store."""
        handler, ctx = _make_http_handler()
        handler._read_kv = MagicMock(return_value={"withdrawal_id": "test"})
        result = handler._read_withdrawal_data()
        assert result == {"withdrawal_id": "test"}

    def test_read_withdrawal_data_empty(self) -> None:
        """Test _read_withdrawal_data returns None when empty."""
        handler, ctx = _make_http_handler()
        handler._read_kv = MagicMock(return_value={})
        result = handler._read_withdrawal_data()
        assert result is None

    def test_read_withdrawal_data_exception(self) -> None:
        """Test _read_withdrawal_data returns None on exception."""
        handler, ctx = _make_http_handler()
        handler._read_kv = MagicMock(side_effect=Exception("Read error"))
        result = handler._read_withdrawal_data()
        assert result is None

    def test_get_web3_instance(self) -> None:
        """Test _get_web3_instance returns Web3 or None."""
        handler, ctx = _make_http_handler()
        ctx.params.optimism_ledger_rpc = ""
        result = handler._get_web3_instance("optimism")
        assert result is None

    def test_get_web3_instance_exception(self) -> None:
        """Test _get_web3_instance returns None on exception."""
        handler, ctx = _make_http_handler()
        ctx.params.optimism_ledger_rpc = "invalid://url"
        with patch(
            "packages.valory.skills.optimus_abci.handlers.Web3",
            side_effect=Exception("Web3 error"),
        ):
            result = handler._get_web3_instance("optimism")
        assert result is None

    def test_get_web3_instance_valid_rpc(self) -> None:
        """Test _get_web3_instance with valid rpc."""
        handler, ctx = _make_http_handler()
        ctx.params.optimism_ledger_rpc = "https://rpc.example.com"
        mock_web3 = MagicMock()
        with patch(
            "packages.valory.skills.optimus_abci.handlers.Web3",
            return_value=mock_web3,
        ):
            result = handler._get_web3_instance("optimism")
        assert result is mock_web3

    def test_get_web3_instance_passes_timeout_to_provider(self) -> None:
        """Web3 provider must be constructed with a request timeout."""
        from packages.valory.skills.optimus_abci.handlers import (
            WEB3_HTTP_TIMEOUT_SECONDS,
        )

        handler, ctx = _make_http_handler()
        ctx.params.optimism_ledger_rpc = "https://rpc.example.com"
        with patch(
            "packages.valory.skills.optimus_abci.handlers.Web3"
        ) as mock_web3_cls:
            handler._get_web3_instance("optimism")

        provider_call = mock_web3_cls.HTTPProvider.call_args
        assert provider_call.args[0] == "https://rpc.example.com"
        assert provider_call.kwargs["request_kwargs"] == {
            "timeout": WEB3_HTTP_TIMEOUT_SECONDS
        }
        assert WEB3_HTTP_TIMEOUT_SECONDS == 30

    def test_is_transient_web3_error_classifications(self) -> None:
        """Transient errors retry, deterministic ones propagate immediately."""
        from packages.valory.skills.optimus_abci.handlers import (
            _is_transient_web3_error,
        )

        # Typed transient
        assert _is_transient_web3_error(requests.exceptions.Timeout("slow")) is True
        assert (
            _is_transient_web3_error(requests.exceptions.ConnectionError("dns")) is True
        )
        # HTTP statuses matched only on word boundaries.
        # Includes Cloudflare-fronted RPC range (520-525).
        for status in (408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525):
            assert (
                _is_transient_web3_error(Exception(f"HTTP {status} from rpc")) is True
            ), f"status {status} should retry"
        # Deterministic — never retry
        assert _is_transient_web3_error(Exception("execution reverted: x")) is False
        assert (
            _is_transient_web3_error(Exception("Return amount is not enough")) is False
        )
        assert _is_transient_web3_error(ValueError("nonsense")) is False
        # A digit substring that is not a standalone 5xx code must not retry —
        # e.g. a contract revert reason like "value 5039 is invalid".
        assert _is_transient_web3_error(Exception("value 5039 is invalid")) is False
        # Untyped exception whose message just happens to contain "timeout"
        # without a transient HTTP status no longer auto-retries.
        assert (
            _is_transient_web3_error(Exception("config timeout setting wrong")) is False
        )

    def test_call_with_web3_retries_succeeds_on_second_attempt(self) -> None:
        """A transient error is retried; the second-attempt success returns."""
        from packages.valory.skills.optimus_abci.handlers import _call_with_web3_retries

        call_count = {"n": 0}

        def flaky() -> str:
            """Flaky."""
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise requests.exceptions.Timeout("slow")
            return "ok"

        with patch("packages.valory.skills.optimus_abci.handlers.time.sleep"):
            result = _call_with_web3_retries(flaky, max_retries=3, initial_delay=0.0)
        assert result == "ok"
        assert call_count["n"] == 2

    def test_call_with_web3_retries_propagates_terminal_error(self) -> None:
        """A non-transient error raises immediately without retrying."""
        from packages.valory.skills.optimus_abci.handlers import _call_with_web3_retries

        call_count = {"n": 0}

        def reverting() -> None:
            """Reverting."""
            call_count["n"] += 1
            raise Exception("execution reverted")

        with pytest.raises(Exception, match="execution reverted"):
            _call_with_web3_retries(reverting, max_retries=5, initial_delay=0.0)
        assert call_count["n"] == 1

    def test_call_with_web3_retries_exhausts_attempts(self) -> None:
        """After max_retries transient failures, the last exception propagates."""
        from packages.valory.skills.optimus_abci.handlers import _call_with_web3_retries

        call_count = {"n": 0}

        def always_timeout() -> None:
            """Always timeout."""
            call_count["n"] += 1
            raise requests.exceptions.Timeout("slow")

        with patch("packages.valory.skills.optimus_abci.handlers.time.sleep"):
            with pytest.raises(requests.exceptions.Timeout):
                _call_with_web3_retries(
                    always_timeout, max_retries=3, initial_delay=0.0
                )
        assert call_count["n"] == 3

    def test_call_web3_with_breaker_short_circuits_when_open(self) -> None:
        """When the breaker is open, the helper raises CircuitBreakerOpenError."""
        from packages.valory.skills.liquidity_trader_abci.models import (
            CircuitBreakerOpenError,
            EndpointCircuitBreaker,
        )

        handler, _ = _make_http_handler()
        breaker = EndpointCircuitBreaker(
            failure_threshold=1, recovery_timeout_seconds=10.0
        )
        breaker.on_failure()  # transitions to OPEN

        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_ss.get_circuit_breaker.return_value = breaker
            mock_shared.return_value = mock_ss
            with pytest.raises(CircuitBreakerOpenError):
                handler._call_web3_with_breaker("optimism", lambda: "never")

    def test_call_web3_with_breaker_records_failure(self) -> None:
        """Underlying failures count against the breaker."""
        from packages.valory.skills.liquidity_trader_abci.models import (
            EndpointCircuitBreaker,
        )

        handler, _ = _make_http_handler()
        breaker = EndpointCircuitBreaker(
            failure_threshold=2, recovery_timeout_seconds=10.0
        )

        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_ss.get_circuit_breaker.return_value = breaker
            mock_shared.return_value = mock_ss

            def revert() -> None:
                """Revert."""
                raise Exception("execution reverted")

            for _ in range(2):
                with pytest.raises(Exception):
                    handler._call_web3_with_breaker("optimism", revert)

        assert breaker.state.value == "open"

    def test_call_web3_with_breaker_records_success(self) -> None:
        """A successful call is recorded against the breaker and returns the value."""
        from packages.valory.skills.liquidity_trader_abci.models import (
            EndpointCircuitBreaker,
        )

        handler, _ = _make_http_handler()
        breaker = EndpointCircuitBreaker(
            failure_threshold=2, recovery_timeout_seconds=10.0
        )
        breaker.on_failure()  # one prior failure recorded

        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_ss.get_circuit_breaker.return_value = breaker
            mock_shared.return_value = mock_ss
            result = handler._call_web3_with_breaker("optimism", lambda: 42)
        assert result == 42
        assert breaker.state.value == "closed"

    def test_check_usdc_balance_no_web3(self) -> None:
        """Test _check_usdc_balance returns None when no web3 instance."""
        handler, ctx = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        result = handler._check_usdc_balance("0xaddr", "optimism", "0xusdc")
        assert result is None

    def test_check_usdc_balance_exception(self) -> None:
        """Test _check_usdc_balance returns None on a transient network error."""
        handler, ctx = _make_http_handler()
        handler._get_web3_instance = MagicMock(
            side_effect=requests.exceptions.ConnectionError("Connection error")
        )
        result = handler._check_usdc_balance("0xaddr", "optimism", "0xusdc")
        assert result is None

    def test_check_usdc_balance_propagates_circuit_breaker_open(self) -> None:
        """CircuitBreakerOpenError must propagate; the catch-all does not eat it."""  # noqa: D403
        from packages.valory.skills.liquidity_trader_abci.models import (
            CircuitBreakerOpenError,
        )

        handler, _ = _make_http_handler()
        mock_w3 = MagicMock()
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        handler._call_web3_with_breaker = MagicMock(
            side_effect=CircuitBreakerOpenError("optimism")
        )
        with pytest.raises(CircuitBreakerOpenError):
            handler._check_usdc_balance("0x" + "0" * 40, "optimism", "0x" + "0" * 40)

    def test_get_nonce_and_gas_web3_no_web3(self) -> None:
        """Test _get_nonce_and_gas_web3 returns None, None when no web3."""
        handler, ctx = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        nonce, gas = handler._get_nonce_and_gas_web3("0xaddr", "optimism")
        assert nonce is None
        assert gas is None

    def test_get_nonce_and_gas_web3_exception(self) -> None:
        """Test _get_nonce_and_gas_web3 returns None, None on exception."""
        handler, ctx = _make_http_handler()
        handler._get_web3_instance = MagicMock(side_effect=Exception("Nonce error"))
        nonce, gas = handler._get_nonce_and_gas_web3("0xaddr", "optimism")
        assert nonce is None
        assert gas is None

    def test_sign_and_submit_tx_web3_no_web3(self) -> None:
        """Test _sign_and_submit_tx_web3 returns None when no web3."""
        handler, ctx = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        result = handler._sign_and_submit_tx_web3({}, "optimism", MagicMock())
        assert result is None

    def test_sign_and_submit_tx_web3_exception(self) -> None:
        """Test _sign_and_submit_tx_web3 returns None on exception."""
        handler, ctx = _make_http_handler()
        handler._get_web3_instance = MagicMock(side_effect=Exception("Submit error"))
        result = handler._sign_and_submit_tx_web3({}, "optimism", MagicMock())
        assert result is None

    def test_check_transaction_status_no_web3(self) -> None:
        """Test _check_transaction_status returns False when no web3."""
        handler, ctx = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        result = handler._check_transaction_status("0xhash", "optimism")
        assert result is False

    def test_check_transaction_status_success(self) -> None:
        """Test _check_transaction_status returns True on success."""
        handler, ctx = _make_http_handler()
        mock_w3 = MagicMock()
        mock_receipt = MagicMock()
        mock_receipt.status = 1
        mock_w3.eth.wait_for_transaction_receipt.return_value = mock_receipt
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        result = handler._check_transaction_status("0xhash", "optimism")
        assert result is True

    def test_check_transaction_status_failure(self) -> None:
        """Test _check_transaction_status returns False on failed tx."""
        handler, ctx = _make_http_handler()
        mock_w3 = MagicMock()
        mock_receipt = MagicMock()
        mock_receipt.status = 0
        mock_w3.eth.wait_for_transaction_receipt.return_value = mock_receipt
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        result = handler._check_transaction_status("0xhash", "optimism")
        assert result is False

    def test_check_transaction_status_exception(self) -> None:
        """Test _check_transaction_status returns False on exception."""
        handler, ctx = _make_http_handler()
        handler._get_web3_instance = MagicMock(side_effect=Exception("Timeout"))
        result = handler._check_transaction_status("0xhash", "optimism")
        assert result is False

    def test_estimate_gas_no_web3(self) -> None:
        """Test _estimate_gas reports no gas and no funds failure without web3."""
        handler, ctx = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        result = handler._estimate_gas(
            {"value": "0x0", "to": "0x0", "data": "0x"}, "0xaddr", "optimism"
        )
        assert result == (None, False)

    def test_estimate_gas_exception_return_amount(self) -> None:
        """Test _estimate_gas classifies a 'Return amount' error as a funds failure."""
        handler, ctx = _make_http_handler()
        mock_w3 = MagicMock()
        mock_w3.eth.estimate_gas.side_effect = Exception("Return amount is not enough")
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        result = handler._estimate_gas(
            {"value": 0, "to": "0x" + "0" * 40, "data": "0x"},
            "0x" + "0" * 40,
            "optimism",
        )
        assert result == (None, True)

    def test_estimate_gas_exception_execution_reverted(self) -> None:
        """Test _estimate_gas classifies 'execution reverted' as a funds failure."""
        handler, ctx = _make_http_handler()
        mock_w3 = MagicMock()
        mock_w3.eth.estimate_gas.side_effect = Exception("execution reverted")
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        result = handler._estimate_gas(
            {"value": "0x123", "to": "0x" + "0" * 40, "data": "0x"},
            "0x" + "0" * 40,
            "optimism",
        )
        assert result == (None, True)

    def test_estimate_gas_generic_exception(self) -> None:
        """Test _estimate_gas classifies an unrecognised error as infrastructure."""
        handler, ctx = _make_http_handler()
        mock_w3 = MagicMock()
        mock_w3.eth.estimate_gas.side_effect = Exception("Unknown error")
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        result = handler._estimate_gas(
            {"value": 0, "to": "0x" + "0" * 40, "data": "0x"},
            "0x" + "0" * 40,
            "optimism",
        )
        assert result == (None, False)

    def test_estimate_gas_success_hex_value(self) -> None:
        """Test _estimate_gas with hex value."""
        handler, ctx = _make_http_handler()
        mock_w3 = MagicMock()
        mock_w3.eth.estimate_gas.return_value = 100000
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        result = handler._estimate_gas(
            {"value": "0x100", "to": "0x" + "0" * 40, "data": "0x"},
            "0x" + "0" * 40,
            "optimism",
        )
        assert result == (int(100000 * 1.2), False)

    def test_estimate_gas_success_int_value(self) -> None:
        """Test _estimate_gas with int value."""
        handler, ctx = _make_http_handler()
        mock_w3 = MagicMock()
        mock_w3.eth.estimate_gas.return_value = 100000
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        result = handler._estimate_gas(
            {"value": 256, "to": "0x" + "0" * 40, "data": "0x"},
            "0x" + "0" * 40,
            "optimism",
        )
        assert result == (int(100000 * 1.2), False)

    def test_get_eoa_account_no_password_success(self) -> None:
        """Test _get_eoa_account with no password and valid key."""
        handler, ctx = _make_http_handler()
        handler._get_password_from_args = MagicMock(return_value=None)
        ctx.default_ledger_id = "ethereum"
        ctx.data_dir = "/tmp/test_data"
        mock_account = MagicMock()
        with (
            patch("builtins.open", MagicMock()),
            patch(
                "packages.valory.skills.optimus_abci.handlers.Account.from_key",
                return_value=mock_account,
            ),
            patch("pathlib.Path.open", MagicMock()),
        ):
            result = handler._get_eoa_account()
        assert result is mock_account

    def test_get_eoa_account_no_password_failure(self) -> None:
        """Test _get_eoa_account with no password and failed read."""
        handler, ctx = _make_http_handler()
        handler._get_password_from_args = MagicMock(return_value=None)
        ctx.default_ledger_id = "ethereum"
        ctx.data_dir = "/tmp/test_data"
        with patch("pathlib.Path.open", side_effect=Exception("File not found")):
            result = handler._get_eoa_account()
        assert result is None

    def test_get_eoa_account_with_password_failure(self) -> None:
        """Test _get_eoa_account with password and decryption failure."""
        handler, ctx = _make_http_handler()
        handler._get_password_from_args = MagicMock(return_value="mypassword")
        ctx.default_ledger_id = "ethereum"
        ctx.data_dir = "/tmp/test_data"
        with patch(
            "packages.valory.skills.optimus_abci.handlers.EthereumCrypto",
            side_effect=Exception("Decrypt error"),
        ):
            result = handler._get_eoa_account()
        assert result is None

    def test_get_lifi_quote_sync_no_chain_id(self) -> None:
        """Test _get_lifi_quote_sync when chain_id is not found."""
        handler, ctx = _make_http_handler()
        ctx.params.chain_to_chain_id_mapping = {}
        result = handler._get_lifi_quote_sync("0xaddr", "unknown", "0xusdc", "1000")
        assert result is None

    def test_get_lifi_quote_sync_success(self) -> None:
        """Test _get_lifi_quote_sync returns quote on success."""
        handler, ctx = _make_http_handler()
        ctx.params.chain_to_chain_id_mapping = {"optimism": 10}
        ctx.params.slippage_for_swap = 0.01
        ctx.params.lifi_quote_to_amount_url = "https://api.example.com/quote"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"quote": "data"}
        with patch(
            "packages.valory.skills.optimus_abci.handlers.requests.get",
            return_value=mock_response,
        ):
            result = handler._get_lifi_quote_sync(
                "0xaddr", "optimism", "0xusdc", "1000"
            )
        assert result == {"quote": "data"}

    def test_get_lifi_quote_sync_non_200(self) -> None:
        """Test _get_lifi_quote_sync returns None on non-200."""
        handler, ctx = _make_http_handler()
        ctx.params.chain_to_chain_id_mapping = {"optimism": 10}
        ctx.params.slippage_for_swap = 0.01
        ctx.params.lifi_quote_to_amount_url = "https://api.example.com/quote"
        mock_response = MagicMock()
        mock_response.status_code = 500
        with patch(
            "packages.valory.skills.optimus_abci.handlers.requests.get",
            return_value=mock_response,
        ):
            result = handler._get_lifi_quote_sync(
                "0xaddr", "optimism", "0xusdc", "1000"
            )
        assert result is None

    def test_get_lifi_quote_sync_exception(self) -> None:
        """Test _get_lifi_quote_sync returns None on a typed transient error."""
        handler, ctx = _make_http_handler()
        ctx.params.chain_to_chain_id_mapping = {"optimism": 10}
        ctx.params.slippage_for_swap = 0.01
        ctx.params.lifi_quote_to_amount_url = "https://api.example.com/quote"
        with patch(
            "packages.valory.skills.optimus_abci.handlers.requests.get",
            side_effect=requests.exceptions.ConnectionError("Network error"),
        ):
            result = handler._get_lifi_quote_sync(
                "0xaddr", "optimism", "0xusdc", "1000"
            )
        assert result is None

    def test_get_lifi_quote_sync_propagates_unexpected_exception(self) -> None:
        """An unexpected exception (e.g. AttributeError) must propagate.

        The catch-all that previously turned every exception into a silent
        None hid programmer errors from the global dispatch wrapper, which
        is what surfaces them as HTTP 500.
        """
        handler, ctx = _make_http_handler()
        ctx.params.chain_to_chain_id_mapping = {"optimism": 10}
        ctx.params.slippage_for_swap = 0.01
        ctx.params.lifi_quote_to_amount_url = "https://api.example.com/quote"
        with patch(
            "packages.valory.skills.optimus_abci.handlers.requests.get",
            side_effect=AttributeError("programmer-bug"),
        ):
            with pytest.raises(AttributeError, match="programmer-bug"):
                handler._get_lifi_quote_sync("0xaddr", "optimism", "0xusdc", "1000")

    def test_get_lifi_quote_breaker_opens_after_consecutive_failures(self) -> None:
        """Sustained LiFi failures open the breaker; subsequent calls short-circuit."""
        from packages.valory.skills.liquidity_trader_abci.models import (
            CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            EndpointCircuitBreaker,
        )

        handler, ctx = _make_http_handler()
        ctx.params.chain_to_chain_id_mapping = {"optimism": 10}
        ctx.params.slippage_for_swap = 0.01
        ctx.params.lifi_quote_to_amount_url = "https://api.example.com/quote"

        # Use a real breaker; the default mock returns fresh MagicMocks
        # each call which would not maintain state across attempts.
        real_breaker = EndpointCircuitBreaker()
        with patch.object(
            type(handler),
            "shared_state",
            new_callable=PropertyMock,
        ) as mock_shared:
            mock_shared.return_value.get_circuit_breaker.return_value = real_breaker

            with patch(
                "packages.valory.skills.optimus_abci.handlers.requests.get",
                side_effect=requests.exceptions.ConnectionError("upstream down"),
            ) as mock_get:
                for _ in range(CIRCUIT_BREAKER_FAILURE_THRESHOLD):
                    result = handler._get_lifi_quote_sync(
                        "0xaddr", "optimism", "0xusdc", "1000"
                    )
                    assert result is None
                calls_at_open = mock_get.call_count

                # The next call must short-circuit without invoking requests.get.
                short_circuited = handler._get_lifi_quote_sync(
                    "0xaddr", "optimism", "0xusdc", "1000"
                )
                assert short_circuited is None
                assert mock_get.call_count == calls_at_open

    def test_get_lifi_quote_breaker_opens_on_sustained_5xx(self) -> None:
        """Sustained 5xx responses (no transport error) must open the breaker.

        ``requests.get`` returns a Response object on 5xx — only transport
        errors raise. The wrapper must promote retriable status codes to
        breaker failures so the breaker is symmetric with the Web3 path.
        """
        from packages.valory.skills.liquidity_trader_abci.models import (
            CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            EndpointCircuitBreaker,
        )

        handler, ctx = _make_http_handler()
        ctx.params.chain_to_chain_id_mapping = {"optimism": 10}
        ctx.params.slippage_for_swap = 0.01
        ctx.params.lifi_quote_to_amount_url = "https://api.example.com/quote"

        real_breaker = EndpointCircuitBreaker()
        bad_response = MagicMock()
        bad_response.status_code = 503
        bad_response.json.return_value = {"error": "service unavailable"}

        with patch.object(
            type(handler),
            "shared_state",
            new_callable=PropertyMock,
        ) as mock_shared:
            mock_shared.return_value.get_circuit_breaker.return_value = real_breaker

            with patch(
                "packages.valory.skills.optimus_abci.handlers.requests.get",
                return_value=bad_response,
            ) as mock_get:
                for _ in range(CIRCUIT_BREAKER_FAILURE_THRESHOLD):
                    result = handler._get_lifi_quote_sync(
                        "0xaddr", "optimism", "0xusdc", "1000"
                    )
                    assert result is None
                calls_at_open = mock_get.call_count

                # The next call must short-circuit without invoking requests.get.
                assert (
                    handler._get_lifi_quote_sync("0xaddr", "optimism", "0xusdc", "1000")
                    is None
                )
                assert mock_get.call_count == calls_at_open

    def test_get_lifi_quote_breaker_recovers_after_cooldown(self) -> None:
        """After the recovery window, a successful probe closes the breaker."""
        from packages.valory.skills.liquidity_trader_abci.models import (
            CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            CIRCUIT_BREAKER_RECOVERY_SECONDS,
            EndpointCircuitBreaker,
        )

        handler, ctx = _make_http_handler()
        ctx.params.chain_to_chain_id_mapping = {"optimism": 10}
        ctx.params.slippage_for_swap = 0.01
        ctx.params.lifi_quote_to_amount_url = "https://api.example.com/quote"

        real_breaker = EndpointCircuitBreaker()
        with patch.object(
            type(handler),
            "shared_state",
            new_callable=PropertyMock,
        ) as mock_shared:
            mock_shared.return_value.get_circuit_breaker.return_value = real_breaker

            # Open the breaker.
            with patch(
                "packages.valory.skills.optimus_abci.handlers.requests.get",
                side_effect=requests.exceptions.ConnectionError("upstream down"),
            ):
                for _ in range(CIRCUIT_BREAKER_FAILURE_THRESHOLD):
                    handler._get_lifi_quote_sync("0xaddr", "optimism", "0xusdc", "1000")

            # Fast-forward past the recovery window so the next allow() flips
            # the breaker into HALF_OPEN.
            real_breaker._opened_at -= CIRCUIT_BREAKER_RECOVERY_SECONDS + 1

            # Probe succeeds: breaker closes, response is forwarded.
            ok_response = MagicMock()
            ok_response.status_code = 200
            ok_response.json.return_value = {"transactionRequest": {}}
            with patch(
                "packages.valory.skills.optimus_abci.handlers.requests.get",
                return_value=ok_response,
            ):
                result = handler._get_lifi_quote_sync(
                    "0xaddr", "optimism", "0xusdc", "1000"
                )
            assert result == {"transactionRequest": {}}

    def test_check_usdc_balance_success(self) -> None:
        """Test _check_usdc_balance returns balance on success."""
        handler, ctx = _make_http_handler()
        mock_w3 = MagicMock()
        mock_contract = MagicMock()
        mock_contract.functions.balanceOf.return_value.call.return_value = 1000000
        mock_w3.eth.contract.return_value = mock_contract
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        with patch(
            "packages.valory.skills.optimus_abci.handlers.Web3"
        ) as mock_web3_cls:
            mock_web3_cls.to_checksum_address = lambda x: x
            result = handler._check_usdc_balance("0xaddr", "optimism", "0xusdc")
        assert result == 1000000

    def test_get_nonce_and_gas_web3_success(self) -> None:
        """Test _get_nonce_and_gas_web3 returns nonce and gas price."""
        handler, ctx = _make_http_handler()
        mock_w3 = MagicMock()
        mock_w3.eth.get_transaction_count.return_value = 42
        mock_w3.eth.gas_price = 1000000000
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        with patch(
            "packages.valory.skills.optimus_abci.handlers.Web3"
        ) as mock_web3_cls:
            mock_web3_cls.to_checksum_address = lambda x: x
            nonce, gas = handler._get_nonce_and_gas_web3("0xaddr", "optimism")
        assert nonce == 42
        assert gas == 1000000000

    def test_sign_and_submit_tx_web3_success(self) -> None:
        """Test _sign_and_submit_tx_web3 returns tx hash on success."""
        handler, ctx = _make_http_handler()
        mock_w3 = MagicMock()
        mock_tx_hash = MagicMock()
        mock_tx_hash.to_0x_hex.return_value = "0xdeadbeef"
        mock_w3.eth.send_raw_transaction.return_value = mock_tx_hash
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        mock_account = MagicMock()
        mock_signed = MagicMock()
        mock_account.sign_transaction.return_value = mock_signed
        result = handler._sign_and_submit_tx_web3(
            {"data": "0x"}, "optimism", mock_account
        )
        assert result == "0xdeadbeef"

    def test_get_eoa_account_with_password_success(self) -> None:
        """Test _get_eoa_account with password and successful decryption."""
        handler, ctx = _make_http_handler()
        handler._get_password_from_args = MagicMock(return_value="mypassword")
        ctx.default_ledger_id = "ethereum"
        ctx.data_dir = "/tmp/test_data"
        mock_crypto = MagicMock()
        mock_crypto.private_key = "0xprivkey"
        mock_account = MagicMock()
        with (
            patch(
                "packages.valory.skills.optimus_abci.handlers.EthereumCrypto",
                return_value=mock_crypto,
            ),
            patch(
                "packages.valory.skills.optimus_abci.handlers.Account.from_key",
                return_value=mock_account,
            ),
        ):
            result = handler._get_eoa_account()
        assert result is mock_account

    def test_handle_main_not_request(self) -> None:
        """Test handle when message is not a REQUEST."""
        from packages.valory.protocols.http.message import HttpMessage

        handler, ctx = _make_http_handler()
        msg = MagicMock(spec=HttpMessage)
        msg.performative = HttpMessage.Performative.RESPONSE
        with patch.object(HttpHandler.__bases__[0], "handle") as mock_super:
            handler.handle(msg)
            mock_super.assert_called_once()

    def test_handle_main_wrong_sender(self) -> None:
        """Test handle when sender is not http_server."""
        from packages.valory.protocols.http.message import HttpMessage

        handler, ctx = _make_http_handler()
        msg = MagicMock(spec=HttpMessage)
        msg.performative = HttpMessage.Performative.REQUEST
        msg.sender = "wrong_sender"
        with patch.object(HttpHandler.__bases__[0], "handle") as mock_super:
            handler.handle(msg)
            mock_super.assert_called_once()

    def test_handle_main_no_handler_found(self) -> None:
        """Test handle when no handler is found for URL."""
        from packages.valory.connections.http_server.connection import (
            PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
        )
        from packages.valory.protocols.http.message import HttpMessage

        handler, ctx = _make_http_handler()
        msg = MagicMock(spec=HttpMessage)
        msg.performative = HttpMessage.Performative.REQUEST
        msg.sender = str(HTTP_SERVER_PUBLIC_ID.without_hash())
        msg.url = "https://example.com/test"
        msg.method = "get"
        handler._get_handler = MagicMock(return_value=(None, {}))
        with patch.object(HttpHandler.__bases__[0], "handle") as mock_super:
            handler.handle(msg)
            mock_super.assert_called_once()

    def test_handle_main_invalid_dialogue(self) -> None:
        """Test handle when dialogue update returns None."""
        from packages.valory.connections.http_server.connection import (
            PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
        )
        from packages.valory.protocols.http.message import HttpMessage

        handler, ctx = _make_http_handler()
        msg = MagicMock(spec=HttpMessage)
        msg.performative = HttpMessage.Performative.REQUEST
        msg.sender = str(HTTP_SERVER_PUBLIC_ID.without_hash())
        msg.url = "http://localhost:8000/test"
        msg.method = "get"
        handler._get_handler = MagicMock(return_value=(MagicMock(), {}))
        ctx.http_dialogues.update.return_value = None
        handler.handle(msg)

    def test_handle_main_valid_route(self) -> None:
        """Test handle calls the route handler for valid request."""
        from packages.valory.connections.http_server.connection import (
            PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
        )
        from packages.valory.protocols.http.message import HttpMessage

        handler, ctx = _make_http_handler()
        msg = MagicMock(spec=HttpMessage)
        msg.performative = HttpMessage.Performative.REQUEST
        msg.sender = str(HTTP_SERVER_PUBLIC_ID.without_hash())
        msg.url = "http://localhost:8000/health"
        msg.method = "get"
        msg.body = b""
        mock_route_handler = MagicMock()
        handler._get_handler = MagicMock(
            return_value=(mock_route_handler, {"key": "val"})
        )
        mock_dialogue = MagicMock()
        ctx.http_dialogues.update.return_value = mock_dialogue
        handler.handle(msg)
        mock_route_handler.assert_called_once()

    def test_handle_get_portfolio(self) -> None:
        """Test _handle_get_portfolio reads and returns portfolio data."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(
            return_value="/tmp/portfolio.json"
        )
        ctx.params.portfolio_info_filename = "portfolio.json"
        ctx.state.selected_protocols = json.dumps(["balancer_pools_search"])
        ctx.state.trading_type = "balanced"
        portfolio = {"portfolio_value": 100, "portfolio_breakdown": []}
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value=portfolio):
                handler._handle_get_portfolio(MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()

    def test_handle_get_portfolio_file_not_found(self) -> None:
        """Test _handle_get_portfolio when file not found."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(
            return_value="/tmp/portfolio.json"
        )
        ctx.params.portfolio_info_filename = "portfolio.json"
        ctx.state.selected_protocols = None
        ctx.state.trading_type = None
        with patch("builtins.open", side_effect=FileNotFoundError("not found")):
            handler._handle_get_portfolio(MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()

    def test_handle_get_portfolio_selected_protocols_list(self) -> None:
        """Test _handle_get_portfolio when selected_protocols is a list."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(
            return_value="/tmp/portfolio.json"
        )
        ctx.params.portfolio_info_filename = "portfolio.json"
        ctx.state.selected_protocols = ["velodrome_pools_search"]
        ctx.state.trading_type = ""
        portfolio = {"portfolio_value": 100}
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value=portfolio):
                handler._handle_get_portfolio(MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()

    def test_handle_get_portfolio_selected_protocols_none(self) -> None:
        """Test _handle_get_portfolio when selected_protocols is None."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(
            return_value="/tmp/portfolio.json"
        )
        ctx.params.portfolio_info_filename = "portfolio.json"
        ctx.state.selected_protocols = None
        ctx.state.trading_type = "balanced"
        portfolio = {"portfolio_value": 100}
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value=portfolio):
                handler._handle_get_portfolio(MagicMock(), MagicMock())

    def test_handle_get_health(self) -> None:
        """Test _handle_get_health returns health data."""
        from datetime import datetime

        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_round_seq = MagicMock()
        mock_round_seq._last_round_transition_timestamp = datetime.now()
        mock_round_seq.block_stall_deadline_expired = False
        mock_app = MagicMock()
        mock_app.current_round.round_id = "test_round"
        mock_app._previous_rounds = []
        mock_round_seq._abci_app = mock_app
        ctx.state.round_sequence = mock_round_seq
        ctx.state.agent_reasoning = None
        ctx.params.reset_pause_duration = 10
        mock_synced = MagicMock()
        mock_synced.period_count = 5
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_synced,
        ):
            handler._handle_get_health(MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()

    def test_handle_get_health_agent_health_propagates(self) -> None:
        """The four staking-activity keys propagate verbatim into the health JSON."""
        from datetime import datetime

        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_round_seq = MagicMock()
        mock_round_seq._last_round_transition_timestamp = datetime.now()
        mock_round_seq.block_stall_deadline_expired = False
        mock_round_seq._abci_app = None
        ctx.state.round_sequence = mock_round_seq
        ctx.state.agent_reasoning = None
        ctx.params.reset_pause_duration = 10
        mock_synced = MagicMock()
        mock_synced.period_count = 5
        mock_synced.is_staking_kpi_met = True
        mock_synced.is_activity_target_met = True
        mock_synced.activity_target = 1
        mock_synced.activity_completed = 3
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_synced,
        ):
            handler._handle_get_health(MagicMock(), MagicMock())
        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data["agent_health"] == {
            "is_staking_kpi_met": True,
            "is_activity_target_met": True,
            "activity_target": 1,
            "activity_completed": 3,
        }

    def test_handle_get_health_agent_health_new_regime_not_yet_met(self) -> None:
        """New regime, KPI behind: ``is_activity_target_met=False`` reaches Pearl.

        Pearl reads ``is_activity_target_met`` to decide rotation; a ``False``
        here must propagate so it keeps the agent running rather than rotating.
        """
        from datetime import datetime

        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_round_seq = MagicMock()
        mock_round_seq._last_round_transition_timestamp = datetime.now()
        mock_round_seq.block_stall_deadline_expired = False
        mock_round_seq._abci_app = None
        ctx.state.round_sequence = mock_round_seq
        ctx.state.agent_reasoning = None
        ctx.params.reset_pause_duration = 10
        mock_synced = MagicMock()
        mock_synced.period_count = 5
        mock_synced.is_staking_kpi_met = False
        mock_synced.is_activity_target_met = False
        mock_synced.activity_target = 1
        mock_synced.activity_completed = 0
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_synced,
        ):
            handler._handle_get_health(MagicMock(), MagicMock())
        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data["agent_health"]["is_activity_target_met"] is False
        assert call_data["agent_health"]["activity_completed"] == 0

    def test_handle_get_health_no_transition(self) -> None:
        """Test _handle_get_health when no transition timestamp."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_round_seq = MagicMock()
        mock_round_seq._last_round_transition_timestamp = None
        mock_round_seq._abci_app = None
        ctx.state.round_sequence = mock_round_seq
        ctx.state.agent_reasoning = None
        mock_synced = MagicMock()
        mock_synced.period_count = 0
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_synced,
        ):
            handler._handle_get_health(MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()

    def test_handle_get_health_with_reasoning(self) -> None:
        """Test _handle_get_health updates reasoning in rounds_info."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler.rounds_info = {"evaluate_strategy_round": {"description": "old"}}
        mock_round_seq = MagicMock()
        mock_round_seq._last_round_transition_timestamp = None
        mock_round_seq._abci_app = None
        ctx.state.round_sequence = mock_round_seq
        ctx.state.agent_reasoning = "New strategy reasoning"
        mock_synced = MagicMock()
        mock_synced.period_count = 0
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_synced,
        ):
            handler._handle_get_health(MagicMock(), MagicMock())
        assert (
            handler.rounds_info["evaluate_strategy_round"]["description"]
            == "New strategy reasoning"
        )

    def test_handle_get_health_during_fsm_startup_window(self) -> None:
        """During startup, _abci_app may be set before current_round is bound.

        The handler must return 200 with current_round=None / rounds=None
        instead of raising AttributeError, which the global dispatch wrapper
        would translate to HTTP 500 and flap the probe.
        """

        class _StartingAbciApp:
            """Mimics the partial-init state seen during FSM startup."""

            @property
            def current_round(self):  # type: ignore[no-untyped-def]
                raise AttributeError("current_round not yet bound")

            _previous_rounds: list = []

        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_round_seq = MagicMock()
        mock_round_seq._last_round_transition_timestamp = None
        mock_round_seq._abci_app = _StartingAbciApp()
        ctx.state.round_sequence = mock_round_seq
        ctx.state.agent_reasoning = None
        ctx.params.reset_pause_duration = 10
        mock_synced = MagicMock()
        mock_synced.period_count = 0
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_synced,
        ):
            handler._handle_get_health(MagicMock(), MagicMock())
        # Healthcheck completes successfully despite the AttributeError.
        handler._send_ok_response.assert_called_once()
        sent_payload = handler._send_ok_response.call_args.args[2]
        assert sent_payload["rounds"] is None

    def test_handle_get_health_slow_transition(self) -> None:
        """Test _handle_get_health when transition is slow."""
        from datetime import datetime

        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_round_seq = MagicMock()
        # Timestamp long ago to make transition slow
        mock_round_seq._last_round_transition_timestamp = datetime(2020, 1, 1)
        mock_round_seq.block_stall_deadline_expired = False
        mock_round_seq._abci_app = None
        ctx.state.round_sequence = mock_round_seq
        ctx.state.agent_reasoning = None
        ctx.params.reset_pause_duration = 10
        mock_synced = MagicMock()
        mock_synced.period_count = 0
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_synced,
        ):
            handler._handle_get_health(MagicMock(), MagicMock())

    def test_handle_get_health_tm_unhealthy(self) -> None:
        """Test _handle_get_health when TM is unhealthy."""
        from datetime import datetime

        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_round_seq = MagicMock()
        mock_round_seq._last_round_transition_timestamp = datetime(2020, 1, 1)
        mock_round_seq.block_stall_deadline_expired = True
        mock_round_seq._abci_app = None
        ctx.state.round_sequence = mock_round_seq
        ctx.state.agent_reasoning = None
        ctx.params.reset_pause_duration = 10
        mock_synced = MagicMock()
        mock_synced.period_count = 0
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_synced,
        ):
            handler._handle_get_health(MagicMock(), MagicMock())

    def test_handle_get_static_file_exists(self) -> None:
        """Test _handle_get_static_file when file exists."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.url = "http://localhost:8000/style.css"
        mock_dialogue = MagicMock()
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = True
        mock_path.suffix = ".css"
        with (
            patch(
                "packages.valory.skills.optimus_abci.handlers.Path",
                return_value=mock_path,
            ),
            patch(
                "packages.valory.skills.optimus_abci.handlers.urlparse"
            ) as mock_urlparse,
            patch("builtins.open", MagicMock()),
        ):
            mock_urlparse.return_value.path = "/style.css"
            handler._handle_get_static_file(mock_msg, mock_dialogue)
        handler._send_ok_response.assert_called_once()

    def test_handle_get_static_file_not_found_fallback(self) -> None:
        """Test _handle_get_static_file falls back to index.html."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.url = "http://localhost:8000/nonexistent"
        mock_dialogue = MagicMock()
        with (
            patch(
                "packages.valory.skills.optimus_abci.handlers.urlparse"
            ) as mock_urlparse,
            patch("packages.valory.skills.optimus_abci.handlers.Path") as mock_path_cls,
        ):
            mock_urlparse.return_value.path = "/nonexistent"
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = False
            mock_path_cls.return_value = mock_path_instance
            mock_path_cls.__truediv__ = MagicMock()
            with patch("builtins.open", MagicMock()):
                handler._handle_get_static_file(mock_msg, mock_dialogue)
        handler._send_ok_response.assert_called_once()

    def test_handle_get_static_file_index_html_empty_path(self) -> None:
        """Test _handle_get_static_file serves index.html for empty path."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.url = "http://localhost:8000/"
        mock_dialogue = MagicMock()
        with (
            patch(
                "packages.valory.skills.optimus_abci.handlers.urlparse"
            ) as mock_urlparse,
            patch("packages.valory.skills.optimus_abci.handlers.Path") as mock_path_cls,
            patch("builtins.open", MagicMock()),
        ):
            mock_urlparse.return_value.path = "/"
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.suffix = ".html"
            mock_path_cls.return_value = mock_path_instance
            handler._handle_get_static_file(mock_msg, mock_dialogue)
        handler._send_ok_response.assert_called_once()

    def test_handle_get_static_file_file_not_found_exception(self) -> None:
        """Test _handle_get_static_file handles FileNotFoundError."""
        handler, ctx = _make_http_handler()
        handler._handle_not_found = MagicMock()
        mock_msg = MagicMock()
        mock_msg.url = "http://localhost:8000/missing.txt"
        with patch(
            "packages.valory.skills.optimus_abci.handlers.urlparse"
        ) as mock_urlparse:
            mock_urlparse.return_value.path = "/missing.txt"
            with patch(
                "packages.valory.skills.optimus_abci.handlers.Path",
                side_effect=FileNotFoundError("not found"),
            ):
                handler._handle_get_static_file(mock_msg, MagicMock())
        handler._handle_not_found.assert_called_once()

    def test_handle_get_static_file_generic_exception(self) -> None:
        """Test _handle_get_static_file handles generic exception."""
        handler, ctx = _make_http_handler()
        handler._handle_not_found = MagicMock()
        mock_msg = MagicMock()
        mock_msg.url = "http://localhost:8000/error"
        with patch(
            "packages.valory.skills.optimus_abci.handlers.urlparse"
        ) as mock_urlparse:
            mock_urlparse.return_value.path = "/error"
            with patch(
                "packages.valory.skills.optimus_abci.handlers.Path",
                side_effect=Exception("error"),
            ):
                handler._handle_get_static_file(mock_msg, MagicMock())
        handler._handle_not_found.assert_called_once()

    def test_handle_post_process_prompt_success(self) -> None:
        """Test _handle_post_process_prompt processes prompt successfully."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.state.request_queue = []
        ctx.state.trading_type = "balanced"
        ctx.state.selected_protocols = json.dumps(["balancer_pools_search"])
        ctx.state.req_to_callback = {}
        ctx.state.in_flight_req = False
        ctx.srr_dialogues.create.return_value = (MagicMock(), MagicMock())
        mock_msg = MagicMock()
        mock_msg.body = json.dumps({"prompt": "invest conservatively"}).encode()
        mock_dialogue = MagicMock()
        mock_dialogue.dialogue_label.dialogue_reference = ("req1", "")
        handler._handle_post_process_prompt(mock_msg, mock_dialogue)
        ctx.outbox.put_message.assert_called_once()

    def test_handle_post_process_prompt_empty_prompt(self) -> None:
        """Test _handle_post_process_prompt with empty prompt."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.state.request_queue = []
        handler._handle_bad_request = MagicMock()
        mock_msg = MagicMock()
        mock_msg.body = json.dumps({"prompt": ""}).encode()
        mock_dialogue = MagicMock()
        mock_dialogue.dialogue_label.dialogue_reference = ("req1", "")
        handler._handle_post_process_prompt(mock_msg, mock_dialogue)
        handler._handle_bad_request.assert_called_once()

    def test_handle_post_process_prompt_invalid_json(self) -> None:
        """Test _handle_post_process_prompt with invalid JSON."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.state.request_queue = []
        handler._handle_bad_request = MagicMock()
        mock_msg = MagicMock()
        mock_msg.body = b"not json"
        mock_dialogue = MagicMock()
        mock_dialogue.dialogue_label.dialogue_reference = ("req1", "")
        handler._handle_post_process_prompt(mock_msg, mock_dialogue)
        handler._handle_bad_request.assert_called_once()

    def test_handle_post_process_prompt_x402_insufficient_funds(self) -> None:
        """Test _handle_post_process_prompt with x402 and insufficient funds."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = True
        ctx.state.request_queue = []
        handler._send_ok_response = MagicMock()
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_shared_val = MagicMock()
            mock_shared_val.sufficient_funds_for_x402_payments = False
            mock_shared.return_value = mock_shared_val
            mock_msg = MagicMock()
            mock_dialogue = MagicMock()
            mock_dialogue.dialogue_label.dialogue_reference = ("req1", "")
            handler._handle_post_process_prompt(mock_msg, mock_dialogue)
        handler._send_ok_response.assert_called_once()

    def test_handle_post_process_prompt_x402_sufficient_funds(self) -> None:
        """Test _handle_post_process_prompt with x402 enabled and sufficient funds.

        This covers the branch 1373->1381 where use_x402 is True but
        sufficient_funds_for_x402_payments is also True, so we skip
        the early return and proceed to the try block.
        """
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = True
        ctx.state.request_queue = []
        ctx.state.trading_type = "balanced"
        ctx.state.selected_protocols = json.dumps(["balancer_pools_search"])
        ctx.state.req_to_callback = {}
        ctx.state.in_flight_req = False
        ctx.srr_dialogues.create.return_value = (MagicMock(), MagicMock())
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_shared_val = MagicMock()
            mock_shared_val.sufficient_funds_for_x402_payments = True
            mock_shared.return_value = mock_shared_val
            mock_msg = MagicMock()
            mock_msg.body = json.dumps({"prompt": "test prompt"}).encode()
            mock_dialogue = MagicMock()
            mock_dialogue.dialogue_label.dialogue_reference = ("req1", "")
            handler._handle_post_process_prompt(mock_msg, mock_dialogue)

    def test_handle_post_process_prompt_selected_protocols_none(self) -> None:
        """Test _handle_post_process_prompt when selected_protocols is None."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.state.request_queue = []
        ctx.state.trading_type = None
        ctx.state.selected_protocols = None
        ctx.state.req_to_callback = {}
        ctx.state.in_flight_req = False
        ctx.srr_dialogues.create.return_value = (MagicMock(), MagicMock())
        mock_msg = MagicMock()
        mock_msg.body = json.dumps({"prompt": "test"}).encode()
        mock_dialogue = MagicMock()
        mock_dialogue.dialogue_label.dialogue_reference = ("req1", "")
        handler._handle_post_process_prompt(mock_msg, mock_dialogue)

    def test_handle_llm_response_success(self) -> None:
        """Test _handle_llm_response with valid response."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._delayed_write_kv_extended = MagicMock()
        ctx.state.selected_protocols = json.dumps(["balancer_pools_search"])
        ctx.state.trading_type = "balanced"
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.available_strategies = {"optimism": ["balancer_pools_search"]}

        response_data = {
            "selected_protocols": ["balancerPool"],
            "trading_type": "balanced",
            "max_loss_percentage": 5.0,
            "reasoning": "Test reasoning",
        }
        llm_msg = MagicMock()
        llm_msg.payload = json.dumps({"response": json.dumps(response_data)})

        with (
            patch(
                "packages.valory.skills.optimus_abci.handlers.validate_and_fix_protocols",
                return_value=["balancerPool"],
            ),
            patch("packages.valory.skills.optimus_abci.handlers.ThreadPoolExecutor"),
        ):
            handler._handle_llm_response(llm_msg, MagicMock(), MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()

    def test_handle_llm_response_error(self) -> None:
        """Test _handle_llm_response with error in response."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()

        llm_msg = MagicMock()
        llm_msg.payload = json.dumps({"error": "API rate limit exceeded"})

        handler._handle_llm_response(llm_msg, MagicMock(), MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()
        call_args = handler._send_ok_response.call_args[0]
        assert "error" in call_args[2]

    def test_handle_llm_response_json_error(self) -> None:
        """Test _handle_llm_response with JSON decode error."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()

        llm_msg = MagicMock()
        llm_msg.payload = "not valid json"

        handler._handle_llm_response(llm_msg, MagicMock(), MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()

    def test_handle_llm_response_generic_exception(self) -> None:
        """Test _handle_llm_response with generic exception."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()

        llm_msg = MagicMock()
        llm_msg.payload = json.dumps({"response": "invalid json {"})

        handler._handle_llm_response(llm_msg, MagicMock(), MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()

    def test_handle_llm_response_filtered_protocols(self) -> None:
        """Test _handle_llm_response when some protocols are filtered."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        ctx.state.selected_protocols = json.dumps([])
        ctx.state.trading_type = "balanced"
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.available_strategies = {"optimism": ["balancer_pools_search"]}

        response_data = {
            "selected_protocols": ["balancerPool", "uniswapV3"],
            "trading_type": "balanced",
            "max_loss_percentage": 5.0,
            "reasoning": "Test",
        }
        llm_msg = MagicMock()
        llm_msg.payload = json.dumps({"response": json.dumps(response_data)})

        with (
            patch(
                "packages.valory.skills.optimus_abci.handlers.validate_and_fix_protocols",
                return_value=["balancerPool"],
            ),
            patch("packages.valory.skills.optimus_abci.handlers.ThreadPoolExecutor"),
        ):
            handler._handle_llm_response(llm_msg, MagicMock(), MagicMock(), MagicMock())

    def test_fallback_to_previous_strategy_with_loss(self) -> None:
        """Test _fallback_to_previous_strategy_with_loss."""
        handler, ctx = _make_http_handler()
        ctx.state.selected_protocols = json.dumps(["balancer_pools_search"])
        ctx.state.trading_type = "balanced"
        ctx.params.available_strategies = {"optimism": ["balancer_pools_search"]}
        result = handler._fallback_to_previous_strategy_with_loss()
        assert len(result) == 5

    def test_fallback_to_previous_strategy_with_loss_risky(self) -> None:
        """Test _fallback_to_previous_strategy_with_loss with risky type."""
        handler, ctx = _make_http_handler()
        ctx.state.selected_protocols = None
        ctx.state.trading_type = "risky"
        ctx.params.available_strategies = ["balancer_pools_search"]
        result = handler._fallback_to_previous_strategy_with_loss()
        assert result[2] == 15.0

    def test_fallback_to_previous_strategy_with_loss_no_type(self) -> None:
        """Test _fallback_to_previous_strategy_with_loss with no trading type."""
        handler, ctx = _make_http_handler()
        ctx.state.selected_protocols = None
        ctx.state.trading_type = None
        ctx.params.available_strategies = ["balancer_pools_search"]
        result = handler._fallback_to_previous_strategy_with_loss()
        assert result[1] == "balanced"
        assert result[2] == 5.0

    def test_fallback_to_previous_strategy(self) -> None:
        """Test _fallback_to_previous_strategy."""
        handler, ctx = _make_http_handler()
        ctx.state.selected_protocols = json.dumps(["balancer_pools_search"])
        ctx.state.trading_type = "balanced"
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.available_strategies = {"optimism": ["balancer_pools_search"]}
        with patch(
            "packages.valory.skills.optimus_abci.handlers.validate_and_fix_protocols",
            return_value=["balancerPool"],
        ):
            result = handler._fallback_to_previous_strategy()
        assert len(result) == 4

    def test_fallback_to_previous_strategy_no_type(self) -> None:
        """Test _fallback_to_previous_strategy with no trading type."""
        handler, ctx = _make_http_handler()
        ctx.state.selected_protocols = None
        ctx.state.trading_type = None
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.available_strategies = {"optimism": ["balancer_pools_search"]}
        handler.available_strategies = ["balancer_pools_search"]
        with patch(
            "packages.valory.skills.optimus_abci.handlers.validate_and_fix_protocols",
            return_value=["balancerPool"],
        ):
            result = handler._fallback_to_previous_strategy()
        assert result[1] == "balanced"

    def test_delayed_write_kv_extended_single_request(self) -> None:
        """Test _delayed_write_kv_extended with one request in queue."""
        handler, ctx = _make_http_handler()
        ctx.params.default_acceptance_time = 0
        ctx.state.request_queue = ["req1"]
        handler._write_kv = MagicMock()
        handler._update_agent_performance_chat = MagicMock()
        data = {
            "selected_protocols": json.dumps(["balancer_pools_search"]),
            "trading_type": "balanced",
            "composite_score": "0.35",
        }
        with patch("packages.valory.skills.optimus_abci.handlers.time.sleep"):
            handler._delayed_write_kv_extended(data)
        handler._write_kv.assert_called_once()
        assert ctx.state.selected_protocols == ["balancer_pools_search"]
        assert ctx.state.trading_type == "balanced"

    def test_delayed_write_kv_extended_multiple_requests(self) -> None:
        """Test _delayed_write_kv_extended with multiple requests."""
        handler, ctx = _make_http_handler()
        ctx.params.default_acceptance_time = 0
        ctx.state.request_queue = ["req1", "req2"]
        handler._write_kv = MagicMock()
        handler._update_agent_performance_chat = MagicMock()
        data = {"selected_protocols": json.dumps(["x"]), "trading_type": "balanced"}
        with patch("packages.valory.skills.optimus_abci.handlers.time.sleep"):
            handler._delayed_write_kv_extended(data)
        handler._write_kv.assert_not_called()

    def test_delayed_write_kv_extended_skips_when_lock_held(self) -> None:
        """A duplicate concurrent KV write must short-circuit."""
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        handler, ctx = _make_http_handler()
        handler._write_kv = MagicMock()
        ctx.state.request_queue = ["req1"]
        ctx.params.default_acceptance_time = 0
        acquired = handlers_mod._KV_WRITE_LOCK.acquire(blocking=False)
        try:
            assert acquired is True
            handler._delayed_write_kv_extended({"trading_type": "balanced"})
        finally:
            handlers_mod._KV_WRITE_LOCK.release()
        handler._write_kv.assert_not_called()
        ctx.logger.info.assert_called()

    def test_delayed_write_kv_extended_releases_lock_after_exception(
        self,
    ) -> None:
        """The KV-write lock is released even when the wrapped body raises."""
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        handler, ctx = _make_http_handler()
        handler._write_kv = MagicMock(side_effect=RuntimeError("boom"))
        ctx.state.request_queue = ["req1"]
        ctx.params.default_acceptance_time = 0
        with patch("packages.valory.skills.optimus_abci.handlers.time.sleep"):
            with pytest.raises(RuntimeError):
                handler._delayed_write_kv_extended({"trading_type": "balanced"})
        acquired = handlers_mod._KV_WRITE_LOCK.acquire(blocking=False)
        try:
            assert acquired is True, "lock leaked across exception"
        finally:
            if acquired:
                handlers_mod._KV_WRITE_LOCK.release()

    def test_handle_get_withdrawal_amount_success(self) -> None:
        """Test _handle_get_withdrawal_amount with valid portfolio data."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(
            return_value="/tmp/portfolio.json"
        )
        ctx.params.portfolio_info_filename = "portfolio.json"
        ctx.params.target_investment_chains = ["optimism"]
        portfolio = {
            "portfolio_breakdown": [
                {"asset": "ETH", "balance": 1.0, "value_usd": 3000.0},
                {"asset": "OLAS", "balance": 100.0, "value_usd": 500.0},
                {"asset": "USDC", "balance": 1000.0, "value_usd": 1000.0},
            ]
        }
        with (
            patch("builtins.open", MagicMock()),
            patch("json.load", return_value=portfolio),
        ):
            handler._handle_get_withdrawal_amount(MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()
        call_data = handler._send_ok_response.call_args[0][2]
        # OLAS should be filtered out
        assert call_data["total_value_usd"] == 4000.0

    def test_handle_get_withdrawal_amount_file_not_found(self) -> None:
        """Test _handle_get_withdrawal_amount when portfolio file not found."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(
            return_value="/tmp/portfolio.json"
        )
        ctx.params.portfolio_info_filename = "portfolio.json"
        with patch("builtins.open", side_effect=FileNotFoundError("not found")):
            handler._handle_get_withdrawal_amount(MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()
        call_data = handler._send_ok_response.call_args[0][2]
        assert "error" in call_data

    def test_handle_post_withdrawal_initiate_success(self) -> None:
        """Test _handle_post_withdrawal_initiate with valid request."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._is_valid_ethereum_address = MagicMock(return_value=True)
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.safe_contract_addresses = {"optimism": "0xsafe"}
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(
            return_value="/tmp/portfolio.json"
        )
        ctx.params.portfolio_info_filename = "portfolio.json"
        portfolio = {"portfolio_value": 1000}
        mock_msg = MagicMock()
        mock_msg.body = json.dumps({"target_address": "0x" + "a" * 40}).encode()
        with (
            patch("builtins.open", MagicMock()),
            patch("json.load", return_value=portfolio),
            patch("packages.valory.skills.optimus_abci.handlers.ThreadPoolExecutor"),
        ):
            handler._handle_post_withdrawal_initiate(mock_msg, MagicMock())
        handler._send_ok_response.assert_called_once()

    def test_handle_post_withdrawal_initiate_invalid_address(self) -> None:
        """Test _handle_post_withdrawal_initiate with invalid address."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.body = json.dumps({"target_address": "invalid"}).encode()
        handler._handle_post_withdrawal_initiate(mock_msg, MagicMock())
        call_data = handler._send_ok_response.call_args[0][2]
        assert "error" in call_data

    def test_handle_post_withdrawal_initiate_no_address(self) -> None:
        """Test _handle_post_withdrawal_initiate with no address."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.body = json.dumps({}).encode()
        handler._handle_post_withdrawal_initiate(mock_msg, MagicMock())

    def test_handle_post_withdrawal_initiate_no_funds(self) -> None:
        """Test _handle_post_withdrawal_initiate when no funds available."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._is_valid_ethereum_address = MagicMock(return_value=True)
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(
            return_value="/tmp/portfolio.json"
        )
        ctx.params.portfolio_info_filename = "portfolio.json"
        portfolio = {"portfolio_value": 0}
        mock_msg = MagicMock()
        mock_msg.body = json.dumps({"target_address": "0x" + "a" * 40}).encode()
        with (
            patch("builtins.open", MagicMock()),
            patch("json.load", return_value=portfolio),
        ):
            handler._handle_post_withdrawal_initiate(mock_msg, MagicMock())
        call_data = handler._send_ok_response.call_args[0][2]
        assert "error" in call_data

    def test_handle_post_withdrawal_initiate_portfolio_not_found(self) -> None:
        """Test _handle_post_withdrawal_initiate with missing portfolio."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._is_valid_ethereum_address = MagicMock(return_value=True)
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(
            return_value="/tmp/portfolio.json"
        )
        ctx.params.portfolio_info_filename = "portfolio.json"
        mock_msg = MagicMock()
        mock_msg.body = json.dumps({"target_address": "0x" + "a" * 40}).encode()
        with patch("builtins.open", side_effect=FileNotFoundError("not found")):
            handler._handle_post_withdrawal_initiate(mock_msg, MagicMock())
        call_data = handler._send_ok_response.call_args[0][2]
        assert "error" in call_data

    def test_handle_post_withdrawal_initiate_invalid_json(self) -> None:
        """Test _handle_post_withdrawal_initiate with invalid JSON."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.body = b"not json"
        handler._handle_post_withdrawal_initiate(mock_msg, MagicMock())
        call_data = handler._send_ok_response.call_args[0][2]
        assert "error" in call_data

    def test_handle_get_withdrawal_status_found(self) -> None:
        """Test _handle_get_withdrawal_status when withdrawal found."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._read_withdrawal_data = MagicMock(
            return_value={
                "withdrawal_id": "abc-123",
                "withdrawal_status": "INITIATED",
                "withdrawal_message": "Processing",
                "withdrawal_chain": "optimism",
                "withdrawal_requested_at": "1234567890",
                "withdrawal_estimated_value_usd": "1000",
            }
        )
        handler._handle_get_withdrawal_status(MagicMock(), MagicMock(), "abc-123")
        handler._send_ok_response.assert_called_once()

    def test_handle_get_withdrawal_status_completed(self) -> None:
        """Test _handle_get_withdrawal_status when withdrawal is completed."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._read_withdrawal_data = MagicMock(
            return_value={
                "withdrawal_id": "abc-123",
                "withdrawal_status": "COMPLETED",
                "withdrawal_message": "Done",
                "withdrawal_chain": "optimism",
                "withdrawal_requested_at": "1234567890",
                "withdrawal_estimated_value_usd": "1000",
                "withdrawal_completed_at": "1234567891",
                "withdrawal_transaction_hashes": json.dumps(["0xhash1", "0xhash2"]),
            }
        )
        handler._handle_get_withdrawal_status(MagicMock(), MagicMock(), "abc-123")
        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data["status"] == "completed"
        assert len(call_data["transaction_hashes"]) == 2

    def test_handle_get_withdrawal_status_not_found(self) -> None:
        """Test _handle_get_withdrawal_status when withdrawal not found."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._read_withdrawal_data = MagicMock(return_value=None)
        with patch("packages.valory.skills.optimus_abci.handlers.time.sleep"):
            handler._handle_get_withdrawal_status(
                MagicMock(), MagicMock(), "unknown-id"
            )
        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data["status"] == "unknown"

    def test_handle_get_withdrawal_status_wrong_id(self) -> None:
        """Test _handle_get_withdrawal_status when withdrawal has wrong id."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._read_withdrawal_data = MagicMock(
            return_value={
                "withdrawal_id": "other-id",
            }
        )
        with patch("packages.valory.skills.optimus_abci.handlers.time.sleep"):
            handler._handle_get_withdrawal_status(MagicMock(), MagicMock(), "abc-123")
        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data["status"] == "unknown"

    def test_handle_get_withdrawal_status_exception(self) -> None:
        """Test _handle_get_withdrawal_status handles exception."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._read_withdrawal_data = MagicMock(side_effect=Exception("Error"))
        handler._handle_get_withdrawal_status(MagicMock(), MagicMock(), "abc-123")
        call_data = handler._send_ok_response.call_args[0][2]
        assert "error" in call_data

    def test_update_agent_performance_chat_success(self) -> None:
        """Test _update_agent_performance_chat with valid chat."""
        handler, ctx = _make_http_handler()
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(return_value="/tmp/perf.json")
        ctx.params.agent_performance_filename = "agent_perf.json"
        existing_perf = {
            "timestamp": 123,
            "metrics": [],
            "last_activity": None,
            "agent_behavior": None,
        }
        with (
            patch("builtins.open", MagicMock()),
            patch("json.load", return_value=existing_perf),
            patch("json.dump") as mock_dump,
        ):
            handler._update_agent_performance_chat("New behavior info")
        mock_dump.assert_called_once()

    def test_update_agent_performance_chat_none(self) -> None:
        """Test _update_agent_performance_chat with None chat."""
        handler, ctx = _make_http_handler()
        handler._update_agent_performance_chat(None)

    def test_update_agent_performance_chat_file_not_found(self) -> None:
        """Test _update_agent_performance_chat when file not found."""
        handler, ctx = _make_http_handler()
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(return_value="/tmp/perf.json")
        ctx.params.agent_performance_filename = "agent_perf.json"
        with (
            patch(
                "builtins.open",
                side_effect=[FileNotFoundError("not found"), MagicMock()],
            ),
            patch("json.dump"),
        ):
            handler._update_agent_performance_chat("New behavior")

    def test_update_agent_performance_chat_llm_error(self) -> None:
        """Test _update_agent_performance_chat with LLM error message."""
        handler, ctx = _make_http_handler()
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(return_value="/tmp/perf.json")
        ctx.params.agent_performance_filename = "agent_perf.json"
        existing_perf = {
            "timestamp": 123,
            "metrics": [],
            "last_activity": None,
            "agent_behavior": None,
        }
        with (
            patch("builtins.open", MagicMock()),
            patch("json.load", return_value=existing_perf),
        ):
            handler._update_agent_performance_chat("LLM Error: something went wrong")
        # Should return early when message starts with "LLM Error:"

    def test_update_agent_performance_chat_html_tags(self) -> None:
        """Test _update_agent_performance_chat cleans HTML tags."""
        handler, ctx = _make_http_handler()
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(return_value="/tmp/perf.json")
        ctx.params.agent_performance_filename = "agent_perf.json"
        existing_perf = {
            "timestamp": 123,
            "metrics": [],
            "last_activity": None,
            "agent_behavior": None,
        }
        with (
            patch("builtins.open", MagicMock()),
            patch("json.load", return_value=existing_perf),
            patch("json.dump") as mock_dump,
        ):
            handler._update_agent_performance_chat("<b>Bold</b>&nbsp;text&lt;")
        mock_dump.assert_called_once()

    def test_update_agent_performance_chat_outer_exception(self) -> None:
        """Test _update_agent_performance_chat handles outer exception."""
        handler, ctx = _make_http_handler()
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(
            side_effect=Exception("Path error")
        )
        ctx.params.agent_performance_filename = "agent_perf.json"
        handler._update_agent_performance_chat("chat msg")
        ctx.logger.error.assert_called()

    def test_update_agent_performance_chat_inner_generic_exception(self) -> None:
        """Test _update_agent_performance_chat handles inner generic exception."""
        handler, ctx = _make_http_handler()
        ctx.params.store_path = MagicMock()
        ctx.params.store_path.__truediv__ = MagicMock(return_value="/tmp/perf.json")
        ctx.params.agent_performance_filename = "agent_perf.json"
        # First open succeeds for read but json.load raises a generic Exception
        with (
            patch("builtins.open", MagicMock()),
            patch("json.load", side_effect=[Exception("Generic error"), None]),
            patch("json.dump"),
        ):
            handler._update_agent_performance_chat("chat msg")

    def test_handle_get_funds_status_normal_mode(self) -> None:
        """Test _handle_get_funds_status in normal mode."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._is_in_withdrawal_mode = MagicMock(return_value=False)
        ctx.params.use_x402 = False
        mock_fund_req = MagicMock()
        mock_fund_req.get_response_body.return_value = {"optimism": {"deficit": 0}}
        with patch.object(
            type(handler),
            "funds_status",
            new_callable=PropertyMock,
            return_value=mock_fund_req,
        ):
            handler._handle_get_funds_status(MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()

    def test_handle_get_funds_status_withdrawal_no_deficit(self) -> None:
        """Test _handle_get_funds_status in withdrawal mode with no deficit."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._is_in_withdrawal_mode = MagicMock(return_value=True)
        handler._has_deficit = MagicMock(return_value=False)
        ctx.params.use_x402 = False
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        mock_fund_req = MagicMock()
        mock_fund_req.get_response_body.return_value = {}
        with patch.object(
            type(handler),
            "funds_status",
            new_callable=PropertyMock,
            return_value=mock_fund_req,
        ):
            handler._handle_get_funds_status(MagicMock(), MagicMock())
        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data == {}

    def test_handle_get_funds_status_withdrawal_with_deficit_no_actions(self) -> None:
        """Test _handle_get_funds_status in withdrawal mode with deficit but no actions."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._is_in_withdrawal_mode = MagicMock(return_value=True)
        handler._has_deficit = MagicMock(return_value=True)
        handler._get_withdrawal_actions = MagicMock(return_value=[])
        ctx.params.use_x402 = False
        mock_fund_req = MagicMock()
        mock_fund_req.get_response_body.return_value = {"optimism": {}}
        with patch.object(
            type(handler),
            "funds_status",
            new_callable=PropertyMock,
            return_value=mock_fund_req,
        ):
            handler._handle_get_funds_status(MagicMock(), MagicMock())
        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data == {}

    def test_handle_get_funds_status_withdrawal_with_actions(self) -> None:
        """Test _handle_get_funds_status in withdrawal mode with deficit and actions."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._is_in_withdrawal_mode = MagicMock(return_value=True)
        handler._has_deficit = MagicMock(return_value=True)
        handler._get_withdrawal_actions = MagicMock(return_value=[{"action": "exit"}])
        handler._read_withdrawal_data = MagicMock(
            return_value={"withdrawal_transaction_hashes": "[]"}
        )
        handler._calculate_withdrawal_funding_deficit = MagicMock(
            return_value={"deficit": 100}
        )
        ctx.params.use_x402 = False
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            ZERO_ADDRESS,
        )

        mock_fund_req = MagicMock()
        mock_fund_req.get_response_body.return_value = {
            "optimism": {
                "0xagent": {ZERO_ADDRESS: {"balance": "1000", "deficit": "500"}}
            }
        }
        with patch.object(
            type(handler),
            "funds_status",
            new_callable=PropertyMock,
            return_value=mock_fund_req,
        ):
            handler._handle_get_funds_status(MagicMock(), MagicMock())

    def test_handle_get_funds_status_withdrawal_all_executed(self) -> None:
        """Test _handle_get_funds_status when all withdrawal actions executed."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._is_in_withdrawal_mode = MagicMock(return_value=True)
        handler._has_deficit = MagicMock(return_value=True)
        handler._get_withdrawal_actions = MagicMock(return_value=[{"action": "exit"}])
        handler._read_withdrawal_data = MagicMock(
            return_value={"withdrawal_transaction_hashes": json.dumps(["0xhash1"])}
        )
        ctx.params.use_x402 = False
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            ZERO_ADDRESS,
        )

        mock_fund_req = MagicMock()
        mock_fund_req.get_response_body.return_value = {
            "optimism": {"0xagent": {ZERO_ADDRESS: {"balance": "1000"}}}
        }
        with patch.object(
            type(handler),
            "funds_status",
            new_callable=PropertyMock,
            return_value=mock_fund_req,
        ):
            handler._handle_get_funds_status(MagicMock(), MagicMock())
        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data == {}

    def test_handle_get_funds_status_withdrawal_error_reading(self) -> None:
        """Test _handle_get_funds_status when error reading withdrawal data."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._is_in_withdrawal_mode = MagicMock(return_value=True)
        handler._has_deficit = MagicMock(return_value=True)
        handler._get_withdrawal_actions = MagicMock(
            return_value=[{"action": "exit"}, {"action": "swap"}]
        )
        handler._read_withdrawal_data = MagicMock(side_effect=Exception("Read error"))
        handler._calculate_withdrawal_funding_deficit = MagicMock(return_value={})
        ctx.params.use_x402 = False
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            ZERO_ADDRESS,
        )

        mock_fund_req = MagicMock()
        mock_fund_req.get_response_body.return_value = {
            "optimism": {"0xagent": {ZERO_ADDRESS: {"balance": "1000"}}}
        }
        with patch.object(
            type(handler),
            "funds_status",
            new_callable=PropertyMock,
            return_value=mock_fund_req,
        ):
            handler._handle_get_funds_status(MagicMock(), MagicMock())

    def test_handle_get_funds_status_x402(self) -> None:
        """Test _handle_get_funds_status with x402."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._is_in_withdrawal_mode = MagicMock(return_value=False)
        handler._ensure_sufficient_funds_for_x402_payments = MagicMock()
        ctx.params.use_x402 = True
        mock_fund_req = MagicMock()
        mock_fund_req.get_response_body.return_value = {}
        with (
            patch.object(
                type(handler),
                "funds_status",
                new_callable=PropertyMock,
                return_value=mock_fund_req,
            ),
            patch("packages.valory.skills.optimus_abci.handlers.ThreadPoolExecutor"),
        ):
            handler._handle_get_funds_status(MagicMock(), MagicMock())

    def test_handle_get_funds_status_balance_value_error(self) -> None:
        """Test _handle_get_funds_status when balance value cannot be parsed."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._is_in_withdrawal_mode = MagicMock(return_value=True)
        handler._has_deficit = MagicMock(return_value=True)
        handler._get_withdrawal_actions = MagicMock(return_value=[{"action": "exit"}])
        handler._read_withdrawal_data = MagicMock(
            return_value={"withdrawal_transaction_hashes": "[]"}
        )
        handler._calculate_withdrawal_funding_deficit = MagicMock(return_value={})
        ctx.params.use_x402 = False
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            ZERO_ADDRESS,
        )

        mock_fund_req = MagicMock()
        mock_fund_req.get_response_body.return_value = {
            "optimism": {"0xagent": {ZERO_ADDRESS: {"balance": "not_a_number"}}}
        }
        with patch.object(
            type(handler),
            "funds_status",
            new_callable=PropertyMock,
            return_value=mock_fund_req,
        ):
            handler._handle_get_funds_status(MagicMock(), MagicMock())

    def test_ensure_sufficient_funds_skips_when_lock_held(self) -> None:
        """A duplicate concurrent x402 topup must short-circuit."""
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        handler, ctx = _make_http_handler()
        handler._get_eoa_account = MagicMock()
        acquired = handlers_mod._X402_TOPUP_LOCK.acquire(blocking=False)
        try:
            assert acquired is True
            handler._ensure_sufficient_funds_for_x402_payments()
        finally:
            handlers_mod._X402_TOPUP_LOCK.release()
        # The function returned before doing any work.
        handler._get_eoa_account.assert_not_called()
        ctx.logger.info.assert_called()

    def test_ensure_sufficient_funds_releases_lock_after_exception(self) -> None:
        """The x402 topup lock is released even when the wrapped body raises."""
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        handler._get_eoa_account = MagicMock(side_effect=RuntimeError("boom"))
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_shared.return_value = MagicMock()
            handler._ensure_sufficient_funds_for_x402_payments()
        acquired = handlers_mod._X402_TOPUP_LOCK.acquire(blocking=False)
        try:
            assert acquired is True, "lock leaked across exception"
        finally:
            if acquired:
                handlers_mod._X402_TOPUP_LOCK.release()

    def test_ensure_sufficient_funds_no_eoa(self) -> None:
        """Test _ensure_sufficient_funds_for_x402_payments when no EOA account."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        handler._get_eoa_account = MagicMock(return_value=None)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            result = handler._ensure_sufficient_funds_for_x402_payments()
        assert result is False

    def test_ensure_sufficient_funds_no_usdc_address(self) -> None:
        """Test _ensure_sufficient_funds_for_x402_payments when no USDC address."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["unknown_chain"]
        mock_account = MagicMock()
        mock_account.address = "0xaddr"
        handler._get_eoa_account = MagicMock(return_value=mock_account)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
            assert mock_ss.sufficient_funds_for_x402_payments is False

    def test_ensure_sufficient_funds_balance_check_fails(self) -> None:
        """Test _ensure_sufficient_funds_for_x402_payments when balance check fails."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        mock_account = MagicMock()
        mock_account.address = "0xaddr"
        handler._get_eoa_account = MagicMock(return_value=mock_account)
        handler._check_usdc_balance = MagicMock(return_value=None)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
            assert mock_ss.sufficient_funds_for_x402_payments is True

    def test_ensure_sufficient_funds_breaker_open_marks_insufficient(self) -> None:
        """A breaker-open during balance check flips sufficient to False."""
        from packages.valory.skills.liquidity_trader_abci.models import (
            CircuitBreakerOpenError,
        )

        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        mock_account = MagicMock()
        mock_account.address = "0xaddr"
        handler._get_eoa_account = MagicMock(return_value=mock_account)
        handler._check_usdc_balance = MagicMock(
            side_effect=CircuitBreakerOpenError("optimism")
        )
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
            assert mock_ss.sufficient_funds_for_x402_payments is False

    def test_ensure_sufficient_funds_balance_sufficient(self) -> None:
        """Test _ensure_sufficient_funds_for_x402_payments when balance is sufficient."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.x402_payment_requirements = {"threshold": 1000, "topup": 5000}
        mock_account = MagicMock()
        mock_account.address = "0xaddr"
        handler._get_eoa_account = MagicMock(return_value=mock_account)
        handler._check_usdc_balance = MagicMock(return_value=2000)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
            assert mock_ss.sufficient_funds_for_x402_payments is True

    def test_ensure_sufficient_funds_swap_needed_no_quote(self) -> None:
        """Test _ensure_sufficient_funds when swap needed but no quote."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.x402_payment_requirements = {"threshold": 1000, "topup": 5000}
        mock_account = MagicMock()
        mock_account.address = "0xaddr"
        handler._get_eoa_account = MagicMock(return_value=mock_account)
        handler._check_usdc_balance = MagicMock(return_value=500)
        handler._get_lifi_quote_sync = MagicMock(return_value=None)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
            assert mock_ss.sufficient_funds_for_x402_payments is False

    def test_ensure_sufficient_funds_swap_no_tx_request(self) -> None:
        """Test _ensure_sufficient_funds when quote has no transactionRequest."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.x402_payment_requirements = {"threshold": 1000, "topup": 5000}
        mock_account = MagicMock()
        mock_account.address = "0xaddr"
        handler._get_eoa_account = MagicMock(return_value=mock_account)
        handler._check_usdc_balance = MagicMock(return_value=500)
        handler._get_lifi_quote_sync = MagicMock(return_value={"data": "some"})
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
            assert mock_ss.sufficient_funds_for_x402_payments is False

    def test_ensure_sufficient_funds_swap_no_nonce(self) -> None:
        """Test _ensure_sufficient_funds when nonce/gas retrieval fails."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.x402_payment_requirements = {"threshold": 1000, "topup": 5000}
        mock_account = MagicMock()
        mock_account.address = "0xaddr"
        handler._get_eoa_account = MagicMock(return_value=mock_account)
        handler._check_usdc_balance = MagicMock(return_value=500)
        handler._get_lifi_quote_sync = MagicMock(
            return_value={
                "transactionRequest": {"to": "0x1", "data": "0x", "value": "0x0"}
            }
        )
        handler._get_nonce_and_gas_web3 = MagicMock(return_value=(None, None))
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
            assert mock_ss.sufficient_funds_for_x402_payments is False

    def test_ensure_sufficient_funds_swap_no_gas_estimate(self) -> None:
        """Test _ensure_sufficient_funds when gas estimation fails."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.x402_payment_requirements = {"threshold": 1000, "topup": 5000}
        mock_account = MagicMock()
        mock_account.address = "0xaddr"
        handler._get_eoa_account = MagicMock(return_value=mock_account)
        handler._check_usdc_balance = MagicMock(return_value=500)
        handler._get_lifi_quote_sync = MagicMock(
            return_value={
                "transactionRequest": {"to": "0x1", "data": "0x", "value": "0x0"}
            }
        )
        handler._get_nonce_and_gas_web3 = MagicMock(return_value=(1, 1000))
        handler._estimate_gas = MagicMock(return_value=(None, False))
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
            assert mock_ss.sufficient_funds_for_x402_payments is False

    def test_ensure_sufficient_funds_swap_tx_fail(self) -> None:
        """Test _ensure_sufficient_funds when tx submission fails stores ETH deficit."""
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.x402_payment_requirements = {"threshold": 1000, "topup": 5000}
        ctx.params.chain_to_chain_id_mapping = {"optimism": 10}
        mock_account = MagicMock()
        mock_account.address = "0xaddr"
        handler._get_eoa_account = MagicMock(return_value=mock_account)
        handler._check_usdc_balance = MagicMock(return_value=500)
        handler._get_lifi_quote_sync = MagicMock(
            return_value={
                "transactionRequest": {"to": "0x1", "data": "0x", "value": "0x100"}
            }
        )
        handler._get_nonce_and_gas_web3 = MagicMock(return_value=(1, 1000))
        handler._estimate_gas = MagicMock(return_value=(21000, False))
        handler._sign_and_submit_tx_web3 = MagicMock(return_value=None)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            with patch(
                "packages.valory.skills.optimus_abci.handlers.Web3"
            ) as mock_web3:
                mock_web3.to_checksum_address = lambda x: x
                handler._ensure_sufficient_funds_for_x402_payments()
            assert mock_ss.sufficient_funds_for_x402_payments is False
            # value=0x100=256, gas=21000, gasPrice=1000 => total=21000*1000+256.
            # Three cycles of that is far below the floor, so the floor wins.
            assert mock_ss.x402_eth_deficit == handlers_mod.X402_ETH_DEFICIT_FLOOR_WEI

    def test_ensure_sufficient_funds_swap_tx_not_successful(self) -> None:
        """Test _ensure_sufficient_funds when tx is not successful stores ETH deficit."""
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.x402_payment_requirements = {"threshold": 1000, "topup": 5000}
        ctx.params.chain_to_chain_id_mapping = {"optimism": 10}
        mock_account = MagicMock()
        mock_account.address = "0xaddr"
        handler._get_eoa_account = MagicMock(return_value=mock_account)
        handler._check_usdc_balance = MagicMock(return_value=500)
        handler._get_lifi_quote_sync = MagicMock(
            return_value={
                "transactionRequest": {"to": "0x1", "data": "0x", "value": 256}
            }
        )
        handler._get_nonce_and_gas_web3 = MagicMock(return_value=(1, 1000))
        handler._estimate_gas = MagicMock(return_value=(21000, False))
        handler._sign_and_submit_tx_web3 = MagicMock(return_value="0xhash")
        handler._check_transaction_status = MagicMock(return_value=False)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            with patch(
                "packages.valory.skills.optimus_abci.handlers.Web3"
            ) as mock_web3:
                mock_web3.to_checksum_address = lambda x: x
                handler._ensure_sufficient_funds_for_x402_payments()
            assert mock_ss.sufficient_funds_for_x402_payments is False
            # value=256, gas=21000, gasPrice=1000 => total=21000*1000+256.
            # Three cycles of that is far below the floor, so the floor wins.
            assert mock_ss.x402_eth_deficit == handlers_mod.X402_ETH_DEFICIT_FLOOR_WEI

    def test_ensure_sufficient_funds_swap_success(self) -> None:
        """Test _ensure_sufficient_funds when swap succeeds clears deficit."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.x402_payment_requirements = {"threshold": 1000, "topup": 5000}
        ctx.params.chain_to_chain_id_mapping = {"optimism": 10}
        mock_account = MagicMock()
        mock_account.address = "0xaddr"
        handler._get_eoa_account = MagicMock(return_value=mock_account)
        handler._check_usdc_balance = MagicMock(return_value=500)
        handler._get_lifi_quote_sync = MagicMock(
            return_value={
                "transactionRequest": {"to": "0x1", "data": "0x", "value": 256}
            }
        )
        handler._get_nonce_and_gas_web3 = MagicMock(return_value=(1, 1000))
        handler._estimate_gas = MagicMock(return_value=(21000, False))
        handler._sign_and_submit_tx_web3 = MagicMock(return_value="0xhash")
        handler._check_transaction_status = MagicMock(return_value=True)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            with patch(
                "packages.valory.skills.optimus_abci.handlers.Web3"
            ) as mock_web3:
                mock_web3.to_checksum_address = lambda x: x
                handler._ensure_sufficient_funds_for_x402_payments()
            assert mock_ss.sufficient_funds_for_x402_payments is True
            assert mock_ss.x402_eth_deficit == 0

    def test_ensure_sufficient_funds_exception(self) -> None:
        """Test _ensure_sufficient_funds handles exception."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        handler._get_eoa_account = MagicMock(side_effect=Exception("Unexpected"))
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
            assert mock_ss.sufficient_funds_for_x402_payments is False

    # -- OPE-1940: deficit reporting on unaffordable x402 top-ups ------------

    @staticmethod
    def _x402_swap_handler(gas_price: int = 1000, tx_value: Any = 256) -> Any:
        """Build a handler primed to reach the gas-estimate step of a swap."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.x402_payment_requirements = {"threshold": 1000, "topup": 5000}
        ctx.params.chain_to_chain_id_mapping = {"optimism": 10}
        mock_account = MagicMock()
        mock_account.address = "0xaddr"
        handler._get_eoa_account = MagicMock(return_value=mock_account)
        handler._check_usdc_balance = MagicMock(return_value=500)
        handler._get_lifi_quote_sync = MagicMock(
            return_value={
                "transactionRequest": {"to": "0x1", "data": "0x", "value": tx_value}
            }
        )
        handler._get_nonce_and_gas_web3 = MagicMock(return_value=(1, gas_price))
        # An empty EOA, so a funds-classified estimate failure is a real one.
        handler._get_native_balance = MagicMock(return_value=0)
        return handler, ctx

    def test_estimate_gas_classifies_out_of_funds(self) -> None:
        """_estimate_gas must recognise the OutOfFunds text from ZD#1239."""
        handler, _ = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=MagicMock())
        handler._call_web3_with_breaker = MagicMock(
            side_effect=Exception("EVM error: OutOfFunds")
        )
        with patch("packages.valory.skills.optimus_abci.handlers.Web3") as mock_web3:
            mock_web3.to_checksum_address = lambda x: x
            tx_gas, insufficient = handler._estimate_gas(
                {"to": "0x1", "data": "0x", "value": 0}, "0xaddr", "optimism"
            )
        assert tx_gas is None
        assert insufficient is True

    @pytest.mark.parametrize(
        "error_text, expected_funds_failure",
        [
            ("EVM error: OutOfFunds", True),
            ("Return amount is not enough", True),
            ("execution reverted", True),
            ("insufficient funds for gas * price + value", True),
            ("Max retries exceeded with url: /rpc", False),
        ],
    )
    def test_estimate_gas_error_classification(
        self, error_text: str, expected_funds_failure: bool
    ) -> None:
        """The classifier separates funding shortfalls from infra failures."""
        handler, _ = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=MagicMock())
        handler._call_web3_with_breaker = MagicMock(side_effect=Exception(error_text))
        with patch("packages.valory.skills.optimus_abci.handlers.Web3") as mock_web3:
            mock_web3.to_checksum_address = lambda x: x
            tx_gas, insufficient = handler._estimate_gas(
                {"to": "0x1", "data": "0x", "value": 0}, "0xaddr", "optimism"
            )
        assert tx_gas is None
        assert insufficient is expected_funds_failure

    def test_estimate_gas_no_web3_is_not_a_funds_failure(self) -> None:
        """A missing Web3 instance is infrastructure, not a funding shortfall."""
        handler, _ = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        assert handler._estimate_gas({}, "0xaddr", "optimism") == (None, False)

    def test_ensure_sufficient_funds_out_of_funds_sets_deficit(self) -> None:
        """Regression for OPE-1940: the reported failure must set a deficit.

        The gas estimate is driven through the real classifier with the exact
        error text from the ZD#1239 bundle, so this fails if the match set
        regresses rather than silently passing on a stand-in message.
        """
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        handler, _ = self._x402_swap_handler()
        handler._get_web3_instance = MagicMock(return_value=MagicMock())
        handler._call_web3_with_breaker = MagicMock(
            side_effect=Exception("EVM error: OutOfFunds")
        )
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_ss.x402_eth_deficit = 0
            mock_shared.return_value = mock_ss
            with patch(
                "packages.valory.skills.optimus_abci.handlers.Web3"
            ) as mock_web3:
                mock_web3.to_checksum_address = lambda x: x
                handler._ensure_sufficient_funds_for_x402_payments()
        assert mock_ss.sufficient_funds_for_x402_payments is False
        assert mock_ss.x402_eth_deficit > 0
        expected = max(
            (handlers_mod.X402_SWAP_FALLBACK_GAS * 1000 + 256)
            * handlers_mod.X402_SWAP_CYCLES_OF_HEADROOM,
            handlers_mod.X402_ETH_DEFICIT_FLOOR_WEI,
        )
        assert mock_ss.x402_eth_deficit == expected

    def test_ensure_sufficient_funds_applies_swap_cycle_headroom(self) -> None:
        """The reported figure covers N swap cycles, not one.

        Uses a gas price high enough that the headroom multiple clears the
        floor, so this asserts the multiplication rather than the clamp.
        """
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        gas_price = 10**9
        tx_value = 10**14
        handler, _ = self._x402_swap_handler(gas_price=gas_price, tx_value=tx_value)
        handler._estimate_gas = MagicMock(return_value=(None, True))
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
        single_cycle = handlers_mod.X402_SWAP_FALLBACK_GAS * gas_price + tx_value
        assert single_cycle * handlers_mod.X402_SWAP_CYCLES_OF_HEADROOM > (
            handlers_mod.X402_ETH_DEFICIT_FLOOR_WEI
        ), "test inputs must clear the floor for this assertion to mean anything"
        assert (
            mock_ss.x402_eth_deficit
            == single_cycle * handlers_mod.X402_SWAP_CYCLES_OF_HEADROOM
        )

    def test_x402_fallback_deficit_is_dimensionally_sane(self) -> None:
        """The fallback figure lands near the ~0.0003 ETH that restored the agent.

        A loose bound on purpose: the point is to fail loudly on a unit error,
        such as multiplying a wei cost by a gas price.
        """
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        # Representative Optimism values: ~0.001 gwei L2 gas price, and a
        # 0.25 USDC swap worth ~0.00009 ETH.
        gas_price = 10**6
        tx_value = 90_000_000_000_000
        handler, _ = self._x402_swap_handler(gas_price=gas_price, tx_value=tx_value)
        handler._estimate_gas = MagicMock(return_value=(None, True))
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
        field_confirmed = 300_000_000_000_000  # 0.0003 ETH
        assert field_confirmed <= mock_ss.x402_eth_deficit <= field_confirmed * 10
        assert mock_ss.x402_eth_deficit >= handlers_mod.X402_ETH_DEFICIT_FLOOR_WEI

    def test_ensure_sufficient_funds_prefers_lifi_gas_limit(self) -> None:
        """The route-specific gasLimit from LiFi is used when the quote has one."""
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        gas_price = 10**9
        tx_value = 10**14
        lifi_gas_limit = 750000
        handler, _ = self._x402_swap_handler(gas_price=gas_price, tx_value=tx_value)
        handler._get_lifi_quote_sync = MagicMock(
            return_value={
                "transactionRequest": {
                    "to": "0x1",
                    "data": "0x",
                    "value": tx_value,
                    "gasLimit": hex(lifi_gas_limit),
                }
            }
        )
        handler._estimate_gas = MagicMock(return_value=(None, True))
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
        expected = (
            lifi_gas_limit * gas_price + tx_value
        ) * handlers_mod.X402_SWAP_CYCLES_OF_HEADROOM
        assert mock_ss.x402_eth_deficit == expected

    def test_ensure_sufficient_funds_gas_infra_failure_reports_nothing(self) -> None:
        """An infrastructure gas failure must not ask the user for money."""
        handler, _ = self._x402_swap_handler()
        handler._estimate_gas = MagicMock(return_value=(None, False))
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_ss.x402_eth_deficit = 4242
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
        assert mock_ss.sufficient_funds_for_x402_payments is False
        assert mock_ss.x402_eth_deficit == 4242

    def test_funds_classified_failure_on_funded_agent_reports_nothing(self) -> None:
        """A revert on an EOA that can afford the swap is not a funds failure.

        "execution reverted" and LiFi's slippage revert match the funds
        classifier but can hit a fully funded EOA; asking the user for ETH
        would not fix them, so the deficit must be left untouched.
        """
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        handler, _ = self._x402_swap_handler()
        handler._estimate_gas = MagicMock(return_value=(None, True))
        single_cycle = handlers_mod.X402_SWAP_FALLBACK_GAS * 1000 + 256
        handler._get_native_balance = MagicMock(return_value=single_cycle)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_ss.x402_eth_deficit = 4242
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
        assert mock_ss.sufficient_funds_for_x402_payments is False
        assert mock_ss.x402_eth_deficit == 4242

    def test_funds_classified_failure_with_unreadable_balance_reports(self) -> None:
        """When the balance cannot be read the classifier's verdict stands."""
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        handler, _ = self._x402_swap_handler()
        handler._estimate_gas = MagicMock(return_value=(None, True))
        handler._get_native_balance = MagicMock(return_value=None)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_ss.x402_eth_deficit = 0
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
        expected = max(
            (handlers_mod.X402_SWAP_FALLBACK_GAS * 1000 + 256)
            * handlers_mod.X402_SWAP_CYCLES_OF_HEADROOM,
            handlers_mod.X402_ETH_DEFICIT_FLOOR_WEI,
        )
        assert mock_ss.x402_eth_deficit == expected

    def test_ensure_sufficient_funds_sufficient_balance_clears_stale_deficit(
        self,
    ) -> None:
        """A sufficient balance clears a deficit reported by an earlier cycle.

        This is the field-confirmed workaround on OPE-1940: the user sends USDC
        directly to the Agent Signer, so the swap never runs again and the
        deficit would otherwise never be cleared.
        """
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.x402_payment_requirements = {"threshold": 1000, "topup": 5000}
        mock_account = MagicMock()
        mock_account.address = "0xaddr"
        handler._get_eoa_account = MagicMock(return_value=mock_account)
        handler._check_usdc_balance = MagicMock(return_value=2000)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_ss.x402_eth_deficit = 999999
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
        assert mock_ss.sufficient_funds_for_x402_payments is True
        assert mock_ss.x402_eth_deficit == 0

    @pytest.mark.parametrize("native_balance", [None, 10**18])
    def test_quote_failure_on_funded_agent_reports_nothing(
        self, native_balance: Any
    ) -> None:
        """A LiFi outage on a funded agent is not a funding shortfall."""
        handler, _ = self._x402_swap_handler()
        handler._get_lifi_quote_sync = MagicMock(return_value=None)
        handler._get_native_balance = MagicMock(return_value=native_balance)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_ss.x402_eth_deficit = 7
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
        assert mock_ss.sufficient_funds_for_x402_payments is False
        assert mock_ss.x402_eth_deficit == 7

    def test_quote_failure_on_unfunded_agent_reports_the_floor(self) -> None:
        """A LiFi outage on an agent below the floor does report the floor."""
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        handler, _ = self._x402_swap_handler()
        handler._get_lifi_quote_sync = MagicMock(return_value=None)
        handler._get_native_balance = MagicMock(return_value=1)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
        assert mock_ss.x402_eth_deficit == handlers_mod.X402_ETH_DEFICIT_FLOOR_WEI

    def test_quote_without_tx_request_uses_the_same_floor_rule(self) -> None:
        """A malformed quote follows the quote-failure rule, not a bare return."""
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        handler, _ = self._x402_swap_handler()
        handler._get_lifi_quote_sync = MagicMock(return_value={"data": "some"})
        handler._get_native_balance = MagicMock(return_value=1)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
        assert mock_ss.x402_eth_deficit == handlers_mod.X402_ETH_DEFICIT_FLOOR_WEI

    def test_infrastructure_failures_leave_the_deficit_untouched(self) -> None:
        """Breaker-open, unreadable balance and missing nonce report nothing."""
        from packages.valory.skills.liquidity_trader_abci.models import (
            CircuitBreakerOpenError,
        )

        cases = [
            (
                "breaker",
                lambda h: setattr(
                    h,
                    "_check_usdc_balance",
                    MagicMock(side_effect=CircuitBreakerOpenError("optimism")),
                ),
            ),
            (
                "none_balance",
                lambda h: setattr(
                    h, "_check_usdc_balance", MagicMock(return_value=None)
                ),
            ),
            (
                "no_nonce",
                lambda h: setattr(
                    h, "_get_nonce_and_gas_web3", MagicMock(return_value=(None, None))
                ),
            ),
        ]
        for name, prime in cases:
            handler, _ = self._x402_swap_handler()
            prime(handler)
            with patch.object(
                type(handler), "shared_state", new_callable=PropertyMock
            ) as mock_shared:
                mock_ss = MagicMock()
                mock_ss.x402_eth_deficit = 1234
                mock_shared.return_value = mock_ss
                handler._ensure_sufficient_funds_for_x402_payments()
            assert mock_ss.x402_eth_deficit == 1234, name

    def test_get_native_balance_paths(self) -> None:
        """_get_native_balance returns the balance, or None when unavailable."""
        from packages.valory.skills.liquidity_trader_abci.models import (
            CircuitBreakerOpenError,
        )

        handler, _ = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        assert handler._get_native_balance("0xaddr", "optimism") is None

        handler._get_web3_instance = MagicMock(return_value=MagicMock())
        with patch("packages.valory.skills.optimus_abci.handlers.Web3") as mock_web3:
            mock_web3.to_checksum_address = lambda x: x
            handler._call_web3_with_breaker = MagicMock(return_value=42)
            assert handler._get_native_balance("0xaddr", "optimism") == 42
            handler._call_web3_with_breaker = MagicMock(
                side_effect=CircuitBreakerOpenError("optimism")
            )
            assert handler._get_native_balance("0xaddr", "optimism") is None
            handler._call_web3_with_breaker = MagicMock(side_effect=Exception("boom"))
            assert handler._get_native_balance("0xaddr", "optimism") is None

    def test_tx_request_gas_limit_parsing(self) -> None:
        """The gasLimit from LiFi is optional and defensively parsed."""
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        parse = handlers_mod._tx_request_gas_limit
        assert parse({}) is None
        assert parse({"gasLimit": None}) is None
        assert parse({"gasLimit": "0x186a0"}) == 100000
        assert parse({"gasLimit": 100000}) == 100000
        assert parse({"gasLimit": "not-a-number"}) is None
        assert parse({"gasLimit": 0}) is None
        # A bogus upstream value must not flow into the reported deficit.
        cap = handlers_mod.X402_LIFI_GAS_LIMIT_CAP
        assert parse({"gasLimit": cap}) == cap
        assert parse({"gasLimit": cap + 1}) is None
        assert parse({"gasLimit": hex(cap + 1)}) is None

    def test_oversized_lifi_gas_limit_falls_back_to_constant(self) -> None:
        """A LiFi gasLimit above the cap is ignored in favour of the constant."""
        from packages.valory.skills.optimus_abci import handlers as handlers_mod

        handler, _ = self._x402_swap_handler()
        handler._get_lifi_quote_sync = MagicMock(
            return_value={
                "transactionRequest": {
                    "to": "0x1",
                    "data": "0x",
                    "value": 256,
                    "gasLimit": handlers_mod.X402_LIFI_GAS_LIMIT_CAP * 1000,
                }
            }
        )
        handler._estimate_gas = MagicMock(return_value=(None, True))
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_ss.x402_eth_deficit = 0
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
        expected = max(
            (handlers_mod.X402_SWAP_FALLBACK_GAS * 1000 + 256)
            * handlers_mod.X402_SWAP_CYCLES_OF_HEADROOM,
            handlers_mod.X402_ETH_DEFICIT_FLOOR_WEI,
        )
        assert mock_ss.x402_eth_deficit == expected

    def test_funds_status_surfaces_the_x402_deficit(self) -> None:
        """The endpoint reports the shortfall the reported agent never showed.

        Drives _handle_get_funds_status with a standard deficit of "0" - the
        exact shape from the ZD#1239 bundle - and a non-zero x402 deficit.
        """
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            ZERO_ADDRESS,
        )

        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._is_in_withdrawal_mode = MagicMock(return_value=False)
        ctx.params.use_x402 = False
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"

        mock_fund_req = MagicMock()
        mock_fund_req.get_response_body.return_value = {
            "optimism": {
                "0xagent": {ZERO_ADDRESS: {"balance": "63882783972811", "deficit": "0"}}
            }
        }
        with (
            patch.object(
                type(handler),
                "funds_status",
                new_callable=PropertyMock,
                return_value=mock_fund_req,
            ),
            patch.object(
                type(handler), "shared_state", new_callable=PropertyMock
            ) as mock_shared,
        ):
            mock_ss = MagicMock()
            mock_ss.x402_eth_deficit = 300_000_000_000_000
            mock_shared.return_value = mock_ss
            handler._handle_get_funds_status(MagicMock(), MagicMock())

        body = handler._send_ok_response.call_args[0][2]
        token_data = body["optimism"]["0xagent"][ZERO_ADDRESS]
        assert int(token_data["deficit"]) == 300_000_000_000_000 - 63882783972811
        assert int(token_data["deficit"]) > 0

    def test_inject_x402_eth_deficit_into_empty_response(self) -> None:
        """Test _inject_x402_eth_deficit adds deficit to empty response."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        result = handler._inject_x402_eth_deficit({}, 1000000)
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            ZERO_ADDRESS,
        )

        assert result["optimism"]["0xagent"][ZERO_ADDRESS]["deficit"] == str(1000000)

    def test_inject_x402_eth_deficit_overrides_smaller_deficit(self) -> None:
        """Test _inject_x402_eth_deficit overrides when x402 deficit is larger."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            ZERO_ADDRESS,
        )

        existing = {
            "optimism": {"0xagent": {ZERO_ADDRESS: {"balance": "100", "deficit": "50"}}}
        }
        result = handler._inject_x402_eth_deficit(existing, 5000)
        # x402 needs 5000, balance is 100, so deficit = 5000-100 = 4900 > existing 50
        assert result["optimism"]["0xagent"][ZERO_ADDRESS]["deficit"] == str(4900)

    def test_inject_x402_eth_deficit_keeps_larger_existing(self) -> None:
        """Test _inject_x402_eth_deficit keeps existing deficit when it's larger."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            ZERO_ADDRESS,
        )

        existing = {
            "optimism": {
                "0xagent": {ZERO_ADDRESS: {"balance": "0", "deficit": "99999"}}
            }
        }
        result = handler._inject_x402_eth_deficit(existing, 1000)
        assert result["optimism"]["0xagent"][ZERO_ADDRESS]["deficit"] == str(99999)

    def test_inject_x402_eth_deficit_balance_exceeds_need(self) -> None:
        """Test _inject_x402_eth_deficit when balance already covers the deficit."""
        handler, ctx = _make_http_handler()
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            ZERO_ADDRESS,
        )

        existing = {
            "optimism": {"0xagent": {ZERO_ADDRESS: {"balance": "5000", "deficit": "0"}}}
        }
        result = handler._inject_x402_eth_deficit(existing, 1000)
        # balance=5000 > eth_deficit=1000, so new_deficit = max(0, 1000-5000) = 0
        # deficit stays "0" (not updated)
        assert result["optimism"]["0xagent"][ZERO_ADDRESS]["deficit"] == "0"

    def test_handle_get_funds_status_injects_x402_deficit(self) -> None:
        """Test _handle_get_funds_status injects x402_eth_deficit when present."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._is_in_withdrawal_mode = MagicMock(return_value=False)
        handler._ensure_sufficient_funds_for_x402_payments = MagicMock()
        ctx.params.use_x402 = True
        ctx.params.target_investment_chains = ["optimism"]
        ctx.agent_address = "0xagent"
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
            ZERO_ADDRESS,
        )

        mock_fund_req = MagicMock()
        mock_fund_req.get_response_body.return_value = {
            "optimism": {"0xagent": {ZERO_ADDRESS: {"balance": "100", "deficit": "0"}}}
        }

        mock_ss = MagicMock()
        mock_ss.x402_eth_deficit = 50000

        with (
            patch.object(
                type(handler),
                "funds_status",
                new_callable=PropertyMock,
                return_value=mock_fund_req,
            ),
            patch.object(
                type(handler),
                "shared_state",
                new_callable=PropertyMock,
                return_value=mock_ss,
            ),
            patch("packages.valory.skills.optimus_abci.handlers.ThreadPoolExecutor"),
        ):
            handler._handle_get_funds_status(MagicMock(), MagicMock())

        # Verify the response included the injected deficit
        call_args = handler._send_ok_response.call_args
        response = call_args[0][2]
        assert int(response["optimism"]["0xagent"][ZERO_ADDRESS]["deficit"]) == 49900

    def test_parse_llm_response_valid(self) -> None:
        """Test _parse_llm_response with valid response."""
        handler, ctx = _make_http_handler()
        ctx.state.trading_type = "balanced"
        response = {
            "selected_protocols": ["balancerPool"],
            "trading_type": "balanced",
            "max_loss_percentage": 5.0,
            "reasoning": "Good strategy",
        }
        msg = MagicMock()
        msg.payload = json.dumps({"response": json.dumps(response)})
        result = handler._parse_llm_response(msg)
        assert result[0] == ["balancerPool"]
        assert result[1] == "balanced"
        assert result[2] == 5.0

    def test_parse_llm_response_error(self) -> None:
        """Test _parse_llm_response with error in response."""
        handler, ctx = _make_http_handler()
        ctx.state.trading_type = "balanced"
        msg = MagicMock()
        msg.payload = json.dumps({"error": "API error"})
        result = handler._parse_llm_response(msg)
        assert result[0] == []
        assert "LLM Error" in result[3]

    def test_parse_llm_response_error_with_message_pattern(self) -> None:
        """Test _parse_llm_response with error containing message pattern."""
        handler, ctx = _make_http_handler()
        ctx.state.trading_type = "balanced"
        msg = MagicMock()
        msg.payload = json.dumps({"error": 'Some error message: "Rate limit exceeded"'})
        result = handler._parse_llm_response(msg)
        assert "Rate limit exceeded" in result[3]

    def test_parse_llm_response_no_max_loss(self) -> None:
        """Test _parse_llm_response with missing max_loss_percentage."""
        handler, ctx = _make_http_handler()
        ctx.state.trading_type = "balanced"
        response = {
            "selected_protocols": ["balancerPool"],
            "trading_type": "risky",
            "reasoning": "Strategy",
        }
        msg = MagicMock()
        msg.payload = json.dumps({"response": json.dumps(response)})
        result = handler._parse_llm_response(msg)
        assert result[2] == 15.0  # Default for risky

    def test_parse_llm_response_max_loss_too_low(self) -> None:
        """Test _parse_llm_response with max_loss_percentage below 1."""
        handler, ctx = _make_http_handler()
        ctx.state.trading_type = "balanced"
        response = {
            "selected_protocols": ["balancerPool"],
            "trading_type": "balanced",
            "max_loss_percentage": 0.5,
            "reasoning": "Strategy",
        }
        msg = MagicMock()
        msg.payload = json.dumps({"response": json.dumps(response)})
        result = handler._parse_llm_response(msg)
        assert result[2] == 1.0

    def test_parse_llm_response_max_loss_too_high(self) -> None:
        """Test _parse_llm_response with max_loss_percentage above 30."""
        handler, ctx = _make_http_handler()
        ctx.state.trading_type = "balanced"
        response = {
            "selected_protocols": ["balancerPool"],
            "trading_type": "balanced",
            "max_loss_percentage": 50.0,
            "reasoning": "Strategy",
        }
        msg = MagicMock()
        msg.payload = json.dumps({"response": json.dumps(response)})
        result = handler._parse_llm_response(msg)
        assert result[2] == 30.0

    def test_parse_llm_response_missing_fields(self) -> None:
        """Test _parse_llm_response with missing required fields."""
        handler, ctx = _make_http_handler()
        ctx.state.selected_protocols = None
        ctx.state.trading_type = None
        ctx.params.available_strategies = ["balancer_pools_search"]
        response = {
            "selected_protocols": [],
            "trading_type": "",
            "max_loss_percentage": 5.0,
            "reasoning": "",
        }
        msg = MagicMock()
        msg.payload = json.dumps({"response": json.dumps(response)})
        result = handler._parse_llm_response(msg)
        # Should fall back to previous strategy
        assert "Falling back" in result[3]

    def test_parse_llm_response_json_in_backticks(self) -> None:
        """Test _parse_llm_response with JSON wrapped in triple backticks."""
        handler, ctx = _make_http_handler()
        ctx.state.trading_type = "balanced"
        response = {
            "selected_protocols": ["balancerPool"],
            "trading_type": "balanced",
            "max_loss_percentage": 5.0,
            "reasoning": "Good strategy",
        }
        inner_json = "```json\n" + json.dumps(response) + "\n```"
        msg = MagicMock()
        msg.payload = json.dumps({"response": inner_json})
        result = handler._parse_llm_response(msg)
        assert result[0] == ["balancerPool"]

    def test_parse_llm_response_invalid_json(self) -> None:
        """Test _parse_llm_response with completely invalid JSON."""
        handler, ctx = _make_http_handler()
        ctx.state.selected_protocols = None
        ctx.state.trading_type = None
        ctx.params.available_strategies = ["balancer_pools_search"]
        msg = MagicMock()
        msg.payload = json.dumps({"response": "not valid json at all"})
        result = handler._parse_llm_response(msg)
        assert "Falling back" in result[3]

    def test_parse_llm_response_max_loss_not_number(self) -> None:
        """Test _parse_llm_response with max_loss_percentage not a number."""
        handler, ctx = _make_http_handler()
        ctx.state.trading_type = "balanced"
        response = {
            "selected_protocols": ["balancerPool"],
            "trading_type": "balanced",
            "max_loss_percentage": "not a number",
            "reasoning": "Strategy",
        }
        msg = MagicMock()
        msg.payload = json.dumps({"response": json.dumps(response)})
        result = handler._parse_llm_response(msg)
        assert result[2] == 5.0  # Default for balanced

    def test_setup_http_handler(self) -> None:
        """Test HttpHandler.setup configures routes and loads FSM spec."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.params.service_endpoint_base = "http://localhost:8000"
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.available_strategies = {"optimism": ["balancer_pools_search"]}
        with (
            patch(
                "packages.valory.skills.optimus_abci.handlers.load_fsm_spec"
            ) as mock_load,
            patch(
                "packages.valory.skills.optimus_abci.handlers.ROUNDS_INFO",
                {"TestRound": {"transitions": {}}},
            ),
        ):
            mock_load.return_value = {
                "transition_func": {
                    "(TestRound, DONE)": "OtherRound",
                }
            }
            handler.setup()
        assert hasattr(handler, "routes")
        assert handler.agent_profile_path == OPTIMUS_AGENT_PROFILE_PATH
        assert "done" in handler.rounds_info.get("test_round", {}).get(
            "transitions", {}
        )

    def test_setup_builds_every_round_the_real_fsm_declares(self) -> None:
        """The real FSM and the real ROUNDS_INFO have to agree.

        ``setup`` indexes ``rounds_info`` by every source round in the FSM, so a
        round added to the FSM without an entry raises ``KeyError`` and the agent
        dies before it starts. The other setup tests patch both the spec and
        ROUNDS_INFO, so they cannot see that; this one patches neither.
        """
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.params.service_endpoint_base = "http://localhost:8000"
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.available_strategies = {"optimism": ["balancer_pools_search"]}

        handler.setup()

        # Every source round resolved, and transitions were recorded against it.
        assert handler.rounds_info
        assert any(info.get("transitions") for info in handler.rounds_info.values())

    def test_setup_http_handler_x402(self) -> None:
        """Test HttpHandler.setup with x402 enabled."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = True
        ctx.params.service_endpoint_base = "http://localhost:8000"
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.available_strategies = {"optimism": ["balancer_pools_search"]}
        with (
            patch(
                "packages.valory.skills.optimus_abci.handlers.load_fsm_spec"
            ) as mock_load,
            patch("packages.valory.skills.optimus_abci.handlers.ROUNDS_INFO", {}),
            patch("packages.valory.skills.optimus_abci.handlers.ThreadPoolExecutor"),
            patch.object(
                type(handler), "shared_state", new_callable=PropertyMock
            ) as mock_shared,
        ):
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            mock_load.return_value = {"transition_func": {}}
            handler.setup()

    def test_setup_http_handler_basius(self) -> None:
        """Test HttpHandler.setup with base chain selects the basius UI."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.params.service_endpoint_base = "http://localhost:8000"
        ctx.params.target_investment_chains = ["base"]
        ctx.params.available_strategies = {"base": []}
        with (
            patch(
                "packages.valory.skills.optimus_abci.handlers.load_fsm_spec"
            ) as mock_load,
            patch("packages.valory.skills.optimus_abci.handlers.ROUNDS_INFO", {}),
        ):
            mock_load.return_value = {"transition_func": {}}
            handler.setup()
        assert handler.agent_profile_path == BASIUS_AGENT_PROFILE_PATH

    def test_setup_http_handler_modius_fallback(self) -> None:
        """Test HttpHandler.setup falls back to modius UI for unmapped chains."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.params.service_endpoint_base = "http://localhost:8000"
        ctx.params.target_investment_chains = ["mode"]
        ctx.params.available_strategies = {"mode": []}
        with (
            patch(
                "packages.valory.skills.optimus_abci.handlers.load_fsm_spec"
            ) as mock_load,
            patch("packages.valory.skills.optimus_abci.handlers.ROUNDS_INFO", {}),
        ):
            mock_load.return_value = {"transition_func": {}}
            handler.setup()
        assert handler.agent_profile_path == MODIUS_AGENT_PROFILE_PATH

    def test_handle_get_withdrawal_status_found_then_not_after_retry(self) -> None:
        """Test _handle_get_withdrawal_status when first read finds wrong id, retry also wrong."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        # First call returns wrong id, second also wrong id
        handler._read_withdrawal_data = MagicMock(
            side_effect=[
                {"withdrawal_id": "wrong"},
                {"withdrawal_id": "still_wrong"},
            ]
        )
        with patch("packages.valory.skills.optimus_abci.handlers.time.sleep"):
            handler._handle_get_withdrawal_status(MagicMock(), MagicMock(), "abc-123")
        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data["status"] == "unknown"

    def test_parse_llm_response_error_attribute_error(self) -> None:
        r"""Test _parse_llm_response when error parsing triggers AttributeError.

        When error_details is a dict with 'message: \"' as a key, the `in` check passes
        but .split() raises AttributeError since dict has no split method.
        """
        handler, ctx = _make_http_handler()
        ctx.state.trading_type = "balanced"
        # Use a dict with the sentinel key to trigger the `in` check,
        # then .split() raises AttributeError since dict doesn't have .split()
        msg = MagicMock()
        msg.payload = json.dumps({"error": {'message: "': "some error detail"}})
        result = handler._parse_llm_response(msg)
        assert "LLM Error" in result[3]

    def test_delayed_write_kv_extended_no_selected_protocols(self) -> None:
        """Test _delayed_write_kv_extended with only trading_type and composite_score."""
        handler, ctx = _make_http_handler()
        ctx.params.default_acceptance_time = 0
        ctx.state.request_queue = ["req1"]
        handler._write_kv = MagicMock()
        handler._update_agent_performance_chat = MagicMock()
        data = {"trading_type": "risky"}
        with patch("packages.valory.skills.optimus_abci.handlers.time.sleep"):
            handler._delayed_write_kv_extended(data)
        assert ctx.state.trading_type == "risky"

    def test_delayed_write_kv_extended_only_composite_score(self) -> None:
        """Test _delayed_write_kv_extended with only composite_score."""
        handler, ctx = _make_http_handler()
        ctx.params.default_acceptance_time = 0
        ctx.state.request_queue = ["req1"]
        handler._write_kv = MagicMock()
        handler._update_agent_performance_chat = MagicMock()
        data = {"composite_score": "0.42"}
        with patch("packages.valory.skills.optimus_abci.handlers.time.sleep"):
            handler._delayed_write_kv_extended(data)
        assert ctx.state.composite_score == 0.42

    def test_delayed_write_kv_extended_empty_data(self) -> None:
        """Test _delayed_write_kv_extended with no matching keys."""
        handler, ctx = _make_http_handler()
        ctx.params.default_acceptance_time = 0
        ctx.state.request_queue = ["req1"]
        handler._write_kv = MagicMock()
        handler._update_agent_performance_chat = MagicMock()
        data = {"other_key": "value"}
        with patch("packages.valory.skills.optimus_abci.handlers.time.sleep"):
            handler._delayed_write_kv_extended(data)

    def test_handle_llm_response_inner_json_parse_error(self) -> None:
        """Test _handle_llm_response when inner response is not valid JSON (JSONDecodeError)."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        llm_msg = MagicMock()
        # Valid outer JSON but inner "response" is not valid JSON
        llm_msg.payload = json.dumps({"response": "not json {"})
        handler._handle_llm_response(llm_msg, MagicMock(), MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()
        call_args = handler._send_ok_response.call_args[0]
        assert "error" in call_args[2]
        assert "Failed to parse LLM response" in call_args[2]["error"]

    def test_handle_llm_response_value_error_in_float(self) -> None:
        """Test _handle_llm_response when a non-JSON exception occurs (generic Exception path).

        The code at line 1662 does float(strategy_data.get("max_loss_percentage", 10)).
        If max_loss_percentage is a non-numeric string, float() raises ValueError,
        caught by the generic except Exception block at lines 1681-1688.
        """
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        ctx.state.trading_type = "balanced"

        llm_msg = MagicMock()
        # Valid JSON, valid inner JSON, but max_loss_percentage is non-numeric
        response_data = {
            "selected_protocols": ["balancerPool"],
            "trading_type": "balanced",
            "max_loss_percentage": "not_a_number",
            "reasoning": "Test reasoning",
        }
        llm_msg.payload = json.dumps({"response": json.dumps(response_data)})

        handler._handle_llm_response(llm_msg, MagicMock(), MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()
        call_args = handler._send_ok_response.call_args[0]
        assert "error" in call_args[2]
        assert "Failed to process LLM response" in call_args[2]["error"]

    def test_get_lifi_quote_sync_chain_id_none(self) -> None:
        """Test _get_lifi_quote_sync when chain is not in mapping (returns None)."""
        handler, ctx = _make_http_handler()
        # Chain not in mapping, .get() returns None, which is falsy
        ctx.params.chain_to_chain_id_mapping = {}
        result = handler._get_lifi_quote_sync("0xaddr", "optimism", "0xusdc", "1000")
        assert result is None

    def test_get_lifi_quote_sync_chain_id_explicit_none(self) -> None:
        """Test _get_lifi_quote_sync when chain_id is explicitly None."""
        handler, ctx = _make_http_handler()
        ctx.params.chain_to_chain_id_mapping = {"optimism": None}
        result = handler._get_lifi_quote_sync("0xaddr", "optimism", "0xusdc", "1000")
        assert result is None

    def test_handle_get_withdrawal_status_found_on_retry(self) -> None:
        """Test _handle_get_withdrawal_status when found on retry."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._read_withdrawal_data = MagicMock(
            side_effect=[
                None,
                {
                    "withdrawal_id": "abc-123",
                    "withdrawal_status": "INITIATED",
                    "withdrawal_message": "Processing",
                    "withdrawal_chain": "optimism",
                    "withdrawal_requested_at": "123",
                    "withdrawal_estimated_value_usd": "1000",
                },
            ]
        )
        with patch("packages.valory.skills.optimus_abci.handlers.time.sleep"):
            handler._handle_get_withdrawal_status(MagicMock(), MagicMock(), "abc-123")
        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data["status"] == "initiated"

    def test_handle_get_health_pre_fsm_attribute_error(self) -> None:
        """Test _handle_get_health when synchronized_data raises AttributeError (pre-FSM)."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_round_seq = MagicMock()
        mock_round_seq._last_round_transition_timestamp = None
        mock_round_seq._abci_app = None
        ctx.state.round_sequence = mock_round_seq
        ctx.state.agent_reasoning = None
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            side_effect=AttributeError("no synchronized_data yet"),
        ):
            handler._handle_get_health(MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()
        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data["period"] is None
        assert call_data["agent_health"] == {
            "is_staking_kpi_met": None,
            "is_activity_target_met": None,
            "activity_target": None,
            "activity_completed": None,
        }

    def test_handle_get_health_pre_fsm_value_error(self) -> None:
        """Test _handle_get_health when synchronized_data raises ValueError (pre-FSM)."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_round_seq = MagicMock()
        mock_round_seq._last_round_transition_timestamp = None
        mock_round_seq._abci_app = None
        ctx.state.round_sequence = mock_round_seq
        ctx.state.agent_reasoning = None
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            side_effect=ValueError("not ready"),
        ):
            handler._handle_get_health(MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()
        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data["period"] is None
        assert call_data["agent_health"] == {
            "is_staking_kpi_met": None,
            "is_activity_target_met": None,
            "activity_target": None,
            "activity_completed": None,
        }

    def test_handle_get_health_tm_unhealthy_is_none(self) -> None:
        """Test _handle_get_health when is_tm_unhealthy is None reports is_tm_healthy as None."""
        handler, ctx = _make_http_handler()
        handler._send_ok_response = MagicMock()
        mock_round_seq = MagicMock()
        # No transition timestamp means is_tm_unhealthy stays None
        mock_round_seq._last_round_transition_timestamp = None
        mock_round_seq._abci_app = None
        ctx.state.round_sequence = mock_round_seq
        ctx.state.agent_reasoning = None
        mock_synced = MagicMock()
        mock_synced.period_count = 0
        with patch.object(
            type(handler),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_synced,
        ):
            handler._handle_get_health(MagicMock(), MagicMock())
        handler._send_ok_response.assert_called_once()
        call_data = handler._send_ok_response.call_args[0][2]
        # When is_tm_unhealthy is None, is_tm_healthy should be None, not True
        assert call_data["is_tm_healthy"] is None


_PD_CHAIN = "optimism"
_PD_CHAIN_ID = 10
_PD_SAFE = "0xabcdef0123456789abcdef0123456789abcdef01"
_PD_EOA = "0xfedcba9876543210fedcba9876543210fedcba98"
_PD_MARKETPLACE = "0xdeadbeefcafebabedeadbeefcafebabedeadbeef"
_PD_TRACKER = "0xfacefeedfacefeedfacefeedfacefeedfacefeed"
_PD_TOKEN = "0x0b2c639c533813f4aa9d7837caf62653d097ff85"
_PD_ZERO = "0x0000000000000000000000000000000000000000"
# FixedPriceTokenUSDC, padded to the 32 bytes the marketplace keys trackers by.
_PD_PAYMENT_TYPE_HEX = "0x" + "6406bb5f".ljust(64, "0")
_PD_PAYMENT_TYPE = bytes.fromhex(_PD_PAYMENT_TYPE_HEX[2:])
_PD_FLOOR = 150000
_PD_TARGET = 500000
_PD_CAP = 500000
_PD_GAS_ESTIMATE = 100000


def _make_predeposit_handler(**overrides: Any) -> Any:
    """Create an HttpHandler configured for the mech pre-deposit path.

    :param overrides: values to change from the defaults.
    :return: the handler and its mocked context.
    """
    handler, ctx = _make_http_handler()
    ctx.coingecko.mech_chain = overrides.get("mech_chain", _PD_CHAIN)
    ctx.coingecko.use_mech_facilitator = overrides.get("use_mech_facilitator", True)
    ctx.coingecko.mech_facilitator_base_url = overrides.get(
        "base_url", "https://facilitator.example/"
    )
    ctx.coingecko.mech_marketplace_addresses = overrides.get(
        "marketplaces", {_PD_CHAIN: _PD_MARKETPLACE}
    )
    ctx.coingecko.mech_pre_deposit_floor = overrides.get("floor", _PD_FLOOR)
    ctx.coingecko.mech_pre_deposit_target = overrides.get("target", _PD_TARGET)
    ctx.coingecko.mech_pre_deposit_cap = overrides.get("cap", _PD_CAP)
    ctx.params.chain_to_chain_id_mapping = overrides.get(
        "chain_ids", {_PD_CHAIN: _PD_CHAIN_ID}
    )
    ctx.params.safe_contract_addresses = overrides.get("safes", {_PD_CHAIN: _PD_SAFE})
    return handler, ctx


def _make_fake_contract(**call_results: Any) -> Any:
    """Return a contract stand-in whose reads resolve to ``call_results``.

    :param call_results: a return value per contract function name. A value
        that is an ``Exception`` is raised from ``call()`` instead.
    :return: the contract stand-in.
    """
    contract = MagicMock()

    def _function(name: str) -> Any:
        def _bind(*_args: Any, **_kwargs: Any) -> Any:
            result = call_results[name]
            if isinstance(result, Exception):
                return SimpleNamespace(call=MagicMock(side_effect=result))
            return SimpleNamespace(call=MagicMock(return_value=result))

        return _bind

    contract.functions = SimpleNamespace(
        **{name: _function(name) for name in call_results}
    )
    contract.encode_abi = MagicMock(side_effect=lambda **kw: f"0x{kw['args']!r}")
    return contract


def _install_rpc(
    handler: Any,
    contracts: Optional[dict] = None,
    native_balance: int = 0,
    gas_estimate: Any = _PD_GAS_ESTIMATE,
) -> Any:
    """Point the handler's RPC at fake contracts and return the fake web3.

    :param handler: the handler to wire.
    :param contracts: a contract stand-in per address.
    :param native_balance: what the EOA holds, in wei.
    :param gas_estimate: the gas estimate, or an exception to raise.
    :return: the fake web3 instance.
    """
    by_address = {
        Web3.to_checksum_address(address): contract
        for address, contract in (contracts or {}).items()
    }
    w3 = MagicMock()
    w3.eth.contract = MagicMock(side_effect=lambda address, abi: by_address[address])
    w3.eth.get_transaction_count = MagicMock(return_value=7)
    w3.eth.gas_price = 1000
    w3.eth.get_balance = MagicMock(return_value=native_balance)
    if isinstance(gas_estimate, Exception):
        w3.eth.estimate_gas = MagicMock(side_effect=gas_estimate)
    else:
        w3.eth.estimate_gas = MagicMock(return_value=gas_estimate)
    handler._get_web3_instance = MagicMock(return_value=w3)
    return w3


def _make_response(status_code: int = 200, payload: Any = None) -> Any:
    """Return a facilitator response stand-in.

    :param status_code: the HTTP status to report.
    :param payload: what ``json()`` returns.
    :return: the response stand-in.
    """
    return SimpleNamespace(
        status_code=status_code, json=MagicMock(return_value=payload)
    )


class TestContractBinding:
    """Cover binding a web3 contract for the pre-deposit reads."""

    def test_contract_is_none_without_an_rpc(self) -> None:
        """No RPC for the chain means no contract to read."""
        handler, _ = _make_predeposit_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        assert handler._contract(_PD_CHAIN, _PD_TRACKER, handler._TRACKER_ABI) is None

    def test_contract_is_bound_to_the_checksummed_address(self) -> None:
        """The address is checksummed before web3 sees it."""
        handler, _ = _make_predeposit_handler()
        tracker = _make_fake_contract(token=_PD_TOKEN)
        w3 = _install_rpc(handler, {_PD_TRACKER: tracker})
        assert (
            handler._contract(
                _PD_CHAIN, _PD_TRACKER.upper().replace("0X", "0x"), handler._TRACKER_ABI
            )
            is tracker
        )
        assert w3.eth.contract.call_args.kwargs["address"] == Web3.to_checksum_address(
            _PD_TRACKER
        )


class TestFacilitatorPaymentType:
    """Cover reading which asset the facilitator charges in."""

    def test_no_base_url_configured(self) -> None:
        """Without a facilitator URL there is no payment type to read."""
        handler, _ = _make_predeposit_handler(base_url="")
        assert handler._read_facilitator_payment_type(_PD_CHAIN, _PD_SAFE) is None

    def test_the_safe_is_checksummed_into_the_url(self) -> None:
        """The requester path carries the checksummed Safe."""
        handler, _ = _make_predeposit_handler()
        response = _make_response(payload={"payment_type": _PD_PAYMENT_TYPE_HEX})
        with patch.object(
            handlers_module.requests, "get", return_value=response
        ) as get:
            handler._read_facilitator_payment_type(_PD_CHAIN, _PD_SAFE)
        url = get.call_args[0][0]
        assert url == (
            f"https://facilitator.example/mech/{_PD_CHAIN}/requester/"
            f"{Web3.to_checksum_address(_PD_SAFE)}"
        )

    def test_a_hex_payment_type_is_decoded(self) -> None:
        """A 32-byte hex payment type comes back as raw bytes."""
        handler, _ = _make_predeposit_handler()
        response = _make_response(payload={"payment_type": _PD_PAYMENT_TYPE_HEX})
        with patch.object(handlers_module.requests, "get", return_value=response):
            result = handler._read_facilitator_payment_type(_PD_CHAIN, _PD_SAFE)
        assert result == _PD_PAYMENT_TYPE

    def test_a_payment_type_without_the_0x_prefix_is_accepted(self) -> None:
        """The prefix is optional; the 32 bytes are what matter."""
        handler, _ = _make_predeposit_handler()
        response = _make_response(payload={"payment_type": _PD_PAYMENT_TYPE_HEX[2:]})
        with patch.object(handlers_module.requests, "get", return_value=response):
            result = handler._read_facilitator_payment_type(_PD_CHAIN, _PD_SAFE)
        assert result == _PD_PAYMENT_TYPE

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"payment_type": None},
            {"payment_type": 42},
            {"payment_type": "0xnothex"},
            {"payment_type": "0x1234"},
            {"payment_type": "0x" + "ab" * 33},
            {"payment_type": ""},
        ],
    )
    def test_an_unusable_payment_type_is_refused(self, payload: Any) -> None:
        """Anything that is not 32 bytes of hex yields no payment type."""
        handler, _ = _make_predeposit_handler()
        with patch.object(
            handlers_module.requests,
            "get",
            return_value=_make_response(payload=payload),
        ):
            assert handler._read_facilitator_payment_type(_PD_CHAIN, _PD_SAFE) is None

    @pytest.mark.parametrize("status_code", [400, 404, 429, 500, 503])
    def test_a_non_200_response_is_refused(self, status_code: int) -> None:
        """A facilitator error skips the check rather than guessing."""
        handler, _ = _make_predeposit_handler()
        with patch.object(
            handlers_module.requests,
            "get",
            return_value=_make_response(status_code=status_code),
        ):
            assert handler._read_facilitator_payment_type(_PD_CHAIN, _PD_SAFE) is None

    def test_a_transport_failure_is_refused(self) -> None:
        """An unreachable facilitator skips the check."""
        handler, _ = _make_predeposit_handler()
        with patch.object(
            handlers_module.requests,
            "get",
            side_effect=requests.exceptions.ConnectionError("down"),
        ):
            assert handler._read_facilitator_payment_type(_PD_CHAIN, _PD_SAFE) is None

    def test_an_unparseable_body_is_refused(self) -> None:
        """A response that is not JSON skips the check."""
        handler, _ = _make_predeposit_handler()
        response = SimpleNamespace(
            status_code=200, json=MagicMock(side_effect=ValueError("not json"))
        )
        with patch.object(handlers_module.requests, "get", return_value=response):
            assert handler._read_facilitator_payment_type(_PD_CHAIN, _PD_SAFE) is None


class TestBalanceTrackerResolution:
    """Cover resolving the tracker that holds the pre-deposit."""

    def test_no_marketplace_configured_for_the_chain(self) -> None:
        """The marketplace is configured, so an unconfigured chain is refused."""
        handler, _ = _make_predeposit_handler(marketplaces={})
        assert handler._resolve_balance_tracker(_PD_CHAIN, _PD_PAYMENT_TYPE) is None

    def test_no_rpc_for_the_chain(self) -> None:
        """Without an RPC the marketplace cannot be read."""
        handler, _ = _make_predeposit_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        assert handler._resolve_balance_tracker(_PD_CHAIN, _PD_PAYMENT_TYPE) is None

    def test_the_configured_marketplace_is_the_one_read(self) -> None:
        """The tracker is resolved from the configured marketplace only."""
        handler, _ = _make_predeposit_handler()
        marketplace = _make_fake_contract(mapPaymentTypeBalanceTrackers=_PD_TRACKER)
        w3 = _install_rpc(handler, {_PD_MARKETPLACE: marketplace})
        result = handler._resolve_balance_tracker(_PD_CHAIN, _PD_PAYMENT_TYPE)
        assert result == Web3.to_checksum_address(_PD_TRACKER)
        assert w3.eth.contract.call_args.kwargs["address"] == Web3.to_checksum_address(
            _PD_MARKETPLACE
        )

    @pytest.mark.parametrize("tracker", [_PD_ZERO, "", None])
    def test_an_unregistered_payment_type_is_refused(self, tracker: Any) -> None:
        """A marketplace that maps the type to nothing yields no tracker."""
        handler, _ = _make_predeposit_handler()
        marketplace = _make_fake_contract(mapPaymentTypeBalanceTrackers=tracker)
        _install_rpc(handler, {_PD_MARKETPLACE: marketplace})
        assert handler._resolve_balance_tracker(_PD_CHAIN, _PD_PAYMENT_TYPE) is None

    def test_a_reverting_marketplace_is_refused(self) -> None:
        """A read that reverts yields no tracker rather than an exception."""
        handler, _ = _make_predeposit_handler()
        marketplace = _make_fake_contract(
            mapPaymentTypeBalanceTrackers=ValueError("reverted")
        )
        _install_rpc(handler, {_PD_MARKETPLACE: marketplace})
        assert handler._resolve_balance_tracker(_PD_CHAIN, _PD_PAYMENT_TYPE) is None


class TestPreDepositReads:
    """Cover reading the deposit and the tracker's token."""

    def test_the_deposit_is_read_for_the_safe(self) -> None:
        """The balance read is keyed by the Safe the marketplace debits."""
        handler, _ = _make_predeposit_handler()
        tracker = _make_fake_contract(mapRequesterBalances=4321)
        _install_rpc(handler, {_PD_TRACKER: tracker})
        assert handler._read_pre_deposit(_PD_CHAIN, _PD_TRACKER, _PD_SAFE) == 4321

    def test_a_zero_deposit_is_distinct_from_an_unreadable_one(self) -> None:
        """An empty pot reads as 0, not as a failure."""
        handler, _ = _make_predeposit_handler()
        tracker = _make_fake_contract(mapRequesterBalances=0)
        _install_rpc(handler, {_PD_TRACKER: tracker})
        assert handler._read_pre_deposit(_PD_CHAIN, _PD_TRACKER, _PD_SAFE) == 0

    def test_an_unreadable_deposit_is_none(self) -> None:
        """A reverting read yields None so the caller can skip."""
        handler, _ = _make_predeposit_handler()
        tracker = _make_fake_contract(mapRequesterBalances=ValueError("reverted"))
        _install_rpc(handler, {_PD_TRACKER: tracker})
        assert handler._read_pre_deposit(_PD_CHAIN, _PD_TRACKER, _PD_SAFE) is None

    def test_no_rpc_means_no_deposit_reading(self) -> None:
        """Without an RPC the deposit cannot be read."""
        handler, _ = _make_predeposit_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        assert handler._read_pre_deposit(_PD_CHAIN, _PD_TRACKER, _PD_SAFE) is None

    def test_a_token_tracker_reports_its_token(self) -> None:
        """A tracker that exposes token() takes that ERC20."""
        handler, _ = _make_predeposit_handler()
        tracker = _make_fake_contract(token=_PD_TOKEN)
        _install_rpc(handler, {_PD_TRACKER: tracker})
        assert handler._tracker_token(_PD_CHAIN, _PD_TRACKER) == (
            True,
            Web3.to_checksum_address(_PD_TOKEN),
        )

    @pytest.mark.parametrize(
        "token",
        [
            _PD_ZERO,
            "",
            None,
            ContractLogicError("execution reverted"),
            BadFunctionCallOutput("no data returned"),
        ],
    )
    def test_a_native_tracker_reports_no_token(self, token: Any) -> None:
        """The contract answering that it has no token() is what marks it native."""
        handler, _ = _make_predeposit_handler()
        tracker = _make_fake_contract(token=token)
        _install_rpc(handler, {_PD_TRACKER: tracker})
        assert handler._tracker_token(_PD_CHAIN, _PD_TRACKER) == (True, None)

    @pytest.mark.parametrize(
        "error",
        [
            requests.exceptions.Timeout("rpc timed out"),
            requests.exceptions.ConnectionError("rpc unreachable"),
            ValueError("malformed response"),
        ],
    )
    def test_a_read_that_never_reached_the_tracker_is_not_native(
        self, error: Exception
    ) -> None:
        """An RPC failure says nothing about which asset the tracker takes."""
        handler, _ = _make_predeposit_handler()
        tracker = _make_fake_contract(token=error)
        _install_rpc(handler, {_PD_TRACKER: tracker})
        assert handler._tracker_token(_PD_CHAIN, _PD_TRACKER) == (False, None)

    def test_no_rpc_means_no_token_reading(self) -> None:
        """Without an RPC the token cannot be read."""
        handler, _ = _make_predeposit_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        assert handler._tracker_token(_PD_CHAIN, _PD_TRACKER) == (False, None)


class TestSendFromEoa:
    """Cover sending one deposit transaction from the agent EOA."""

    def test_the_tx_is_signed_for_the_given_chain_id(self) -> None:
        """The chain id, nonce and gas all come from the live chain."""
        handler, _ = _make_predeposit_handler()
        _install_rpc(handler)
        handler._sign_and_submit_tx_web3 = MagicMock(return_value="0xhash")
        handler._check_transaction_status = MagicMock(return_value=True)
        assert handler._send_from_eoa(
            _PD_CHAIN,
            _PD_CHAIN_ID,
            SimpleNamespace(address=_PD_EOA),
            _PD_TRACKER,
            "0xdata",
            value=11,
        )
        tx = handler._sign_and_submit_tx_web3.call_args[0][0]
        assert tx["chainId"] == _PD_CHAIN_ID
        assert tx["nonce"] == 7
        assert tx["value"] == 11
        assert tx["to"] == Web3.to_checksum_address(_PD_TRACKER)

    def test_the_gas_limit_carries_headroom_over_the_estimate(self) -> None:
        """The estimate did not see the state the tx will land in."""
        handler, _ = _make_predeposit_handler()
        _install_rpc(handler)
        handler._sign_and_submit_tx_web3 = MagicMock(return_value="0xhash")
        handler._check_transaction_status = MagicMock(return_value=True)
        handler._send_from_eoa(
            _PD_CHAIN, _PD_CHAIN_ID, SimpleNamespace(address=_PD_EOA), _PD_TRACKER, "0x"
        )
        tx = handler._sign_and_submit_tx_web3.call_args[0][0]
        assert tx["gas"] == int(
            _PD_GAS_ESTIMATE * handlers_module.GAS_ESTIMATE_HEADROOM
        )
        assert tx["gas"] > _PD_GAS_ESTIMATE

    def test_no_rpc_sends_nothing(self) -> None:
        """Without an RPC no transaction is built."""
        handler, _ = _make_predeposit_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        handler._sign_and_submit_tx_web3 = MagicMock()
        assert not handler._send_from_eoa(
            _PD_CHAIN, _PD_CHAIN_ID, SimpleNamespace(address=_PD_EOA), _PD_TRACKER, "0x"
        )
        handler._sign_and_submit_tx_web3.assert_not_called()

    def test_a_failed_estimate_sends_nothing(self) -> None:
        """A tx that cannot be estimated is not broadcast blind."""
        handler, _ = _make_predeposit_handler()
        _install_rpc(handler, gas_estimate=ValueError("reverted"))
        handler._sign_and_submit_tx_web3 = MagicMock()
        assert not handler._send_from_eoa(
            _PD_CHAIN, _PD_CHAIN_ID, SimpleNamespace(address=_PD_EOA), _PD_TRACKER, "0x"
        )
        handler._sign_and_submit_tx_web3.assert_not_called()

    def test_a_rejected_submission_is_not_waited_on(self) -> None:
        """Nothing was broadcast, so there is no receipt to wait for."""
        handler, _ = _make_predeposit_handler()
        _install_rpc(handler)
        handler._sign_and_submit_tx_web3 = MagicMock(return_value=None)
        handler._check_transaction_status = MagicMock()
        assert not handler._send_from_eoa(
            _PD_CHAIN, _PD_CHAIN_ID, SimpleNamespace(address=_PD_EOA), _PD_TRACKER, "0x"
        )
        handler._check_transaction_status.assert_not_called()

    def test_a_reverted_tx_is_reported_as_failed(self) -> None:
        """A mined-but-reverted deposit is not a success."""
        handler, _ = _make_predeposit_handler()
        _install_rpc(handler)
        handler._sign_and_submit_tx_web3 = MagicMock(return_value="0xhash")
        handler._check_transaction_status = MagicMock(return_value=False)
        assert not handler._send_from_eoa(
            _PD_CHAIN, _PD_CHAIN_ID, SimpleNamespace(address=_PD_EOA), _PD_TRACKER, "0x"
        )


class TestTokenDeposit:
    """Cover approving and depositing an ERC20 on the Safe's behalf."""

    def test_the_approve_precedes_the_deposit(self) -> None:
        """The tracker pulls the tokens, so the allowance has to exist first."""
        handler, _ = _make_predeposit_handler()
        _install_rpc(
            handler,
            {
                _PD_TOKEN: _make_fake_contract(),
                _PD_TRACKER: _make_fake_contract(),
            },
        )
        handler._check_usdc_balance = MagicMock(return_value=_PD_TARGET)
        handler._send_from_eoa = MagicMock(return_value=True)
        assert handlers_module.MechDepositOutcome.SUFFICIENT is handler._deposit_token(
            _PD_CHAIN,
            _PD_CHAIN_ID,
            SimpleNamespace(address=_PD_EOA),
            _PD_SAFE,
            _PD_TRACKER,
            _PD_TOKEN,
            _PD_TARGET,
        )
        targets = [call[0][3] for call in handler._send_from_eoa.call_args_list]
        assert targets == [_PD_TOKEN, _PD_TRACKER]

    def test_the_allowance_is_granted_to_the_tracker(self) -> None:
        """Approving anyone else leaves the tracker unable to pull the tokens."""
        handler, _ = _make_predeposit_handler()
        token = _make_fake_contract()
        _install_rpc(handler, {_PD_TOKEN: token, _PD_TRACKER: _make_fake_contract()})
        handler._check_usdc_balance = MagicMock(return_value=_PD_TARGET)
        handler._send_from_eoa = MagicMock(return_value=True)
        handler._deposit_token(
            _PD_CHAIN,
            _PD_CHAIN_ID,
            SimpleNamespace(address=_PD_EOA),
            _PD_SAFE,
            _PD_TRACKER,
            _PD_TOKEN,
            _PD_TARGET,
        )
        assert token.encode_abi.call_args.kwargs["args"][0] == (
            Web3.to_checksum_address(_PD_TRACKER)
        )

    def test_a_failed_deposit_after_a_successful_approve_is_not_a_shortfall(
        self,
    ) -> None:
        """The EOA held the token, so a failed send says nothing about its funds."""
        handler, _ = _make_predeposit_handler()
        _install_rpc(
            handler,
            {_PD_TOKEN: _make_fake_contract(), _PD_TRACKER: _make_fake_contract()},
        )
        handler._check_usdc_balance = MagicMock(return_value=_PD_TARGET)
        handler._send_from_eoa = MagicMock(side_effect=[True, False])
        assert (
            handler._deposit_token(
                _PD_CHAIN,
                _PD_CHAIN_ID,
                SimpleNamespace(address=_PD_EOA),
                _PD_SAFE,
                _PD_TRACKER,
                _PD_TOKEN,
                _PD_TARGET,
            )
            is handlers_module.MechDepositOutcome.UNAVAILABLE
        )
        assert handler._send_from_eoa.call_count == 2

    def test_the_deposit_credits_the_safe_not_the_payer(self) -> None:
        """The EOA pays but the marketplace debits the Safe, so the Safe is credited."""
        handler, _ = _make_predeposit_handler()
        tracker = _make_fake_contract()
        _install_rpc(handler, {_PD_TOKEN: _make_fake_contract(), _PD_TRACKER: tracker})
        handler._check_usdc_balance = MagicMock(return_value=_PD_TARGET)
        handler._send_from_eoa = MagicMock(return_value=True)
        handler._deposit_token(
            _PD_CHAIN,
            _PD_CHAIN_ID,
            SimpleNamespace(address=_PD_EOA),
            _PD_SAFE,
            _PD_TRACKER,
            _PD_TOKEN,
            _PD_TARGET,
        )
        assert tracker.encode_abi.call_args.kwargs["args"] == [
            Web3.to_checksum_address(_PD_SAFE),
            _PD_TARGET,
        ]

    def test_the_deposit_is_clamped_to_what_the_eoa_holds(self) -> None:
        """A swap can deliver less than it quoted, so deposit what is there."""
        handler, _ = _make_predeposit_handler()
        token = _make_fake_contract()
        tracker = _make_fake_contract()
        _install_rpc(handler, {_PD_TOKEN: token, _PD_TRACKER: tracker})
        handler._check_usdc_balance = MagicMock(return_value=120000)
        handler._send_from_eoa = MagicMock(return_value=True)
        handler._deposit_token(
            _PD_CHAIN,
            _PD_CHAIN_ID,
            SimpleNamespace(address=_PD_EOA),
            _PD_SAFE,
            _PD_TRACKER,
            _PD_TOKEN,
            _PD_TARGET,
        )
        assert token.encode_abi.call_args.kwargs["args"][1] == 120000
        assert tracker.encode_abi.call_args.kwargs["args"][1] == 120000

    def test_a_failed_approve_does_not_attempt_the_deposit(self) -> None:
        """A deposit without its allowance reverts, so it is not sent."""
        handler, _ = _make_predeposit_handler()
        _install_rpc(
            handler,
            {_PD_TOKEN: _make_fake_contract(), _PD_TRACKER: _make_fake_contract()},
        )
        handler._check_usdc_balance = MagicMock(return_value=_PD_TARGET)
        handler._send_from_eoa = MagicMock(return_value=False)
        assert handlers_module.MechDepositOutcome.UNAVAILABLE is handler._deposit_token(
            _PD_CHAIN,
            _PD_CHAIN_ID,
            SimpleNamespace(address=_PD_EOA),
            _PD_SAFE,
            _PD_TRACKER,
            _PD_TOKEN,
            _PD_TARGET,
        )
        assert handler._send_from_eoa.call_count == 1

    @pytest.mark.parametrize(
        "held,expected",
        [
            # An unreadable balance is not a shortfall: the agent does not know
            # whether it can pay, so it must not ask the user for money.
            (None, "UNAVAILABLE"),
            (0, "UNDERFUNDED"),
        ],
    )
    def test_nothing_to_deposit_sends_nothing(self, held: Any, expected: str) -> None:
        """An empty balance is a shortfall; an unreadable one is not."""
        handler, _ = _make_predeposit_handler()
        _install_rpc(handler)
        handler._check_usdc_balance = MagicMock(return_value=held)
        handler._send_from_eoa = MagicMock()
        assert handler._deposit_token(
            _PD_CHAIN,
            _PD_CHAIN_ID,
            SimpleNamespace(address=_PD_EOA),
            _PD_SAFE,
            _PD_TRACKER,
            _PD_TOKEN,
            _PD_TARGET,
        ) is getattr(handlers_module.MechDepositOutcome, expected)
        handler._send_from_eoa.assert_not_called()

    def test_no_rpc_sends_nothing(self) -> None:
        """Without an RPC neither call can be encoded."""
        handler, _ = _make_predeposit_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        handler._check_usdc_balance = MagicMock(return_value=_PD_TARGET)
        handler._send_from_eoa = MagicMock()
        assert handlers_module.MechDepositOutcome.UNAVAILABLE is handler._deposit_token(
            _PD_CHAIN,
            _PD_CHAIN_ID,
            SimpleNamespace(address=_PD_EOA),
            _PD_SAFE,
            _PD_TRACKER,
            _PD_TOKEN,
            _PD_TARGET,
        )
        handler._send_from_eoa.assert_not_called()


class TestNativeDeposit:
    """Cover depositing native value on the Safe's behalf."""

    def test_the_gas_reserve_is_left_behind(self) -> None:
        """A deposit must not leave the agent unable to pay for gas."""
        handler, _ = _make_predeposit_handler()
        balance = handlers_module.X402_ETH_DEFICIT_FLOOR_WEI + 500
        _install_rpc(
            handler, {_PD_TRACKER: _make_fake_contract()}, native_balance=balance
        )
        handler._send_from_eoa = MagicMock(return_value=True)
        assert handler._deposit_native(
            _PD_CHAIN,
            _PD_CHAIN_ID,
            SimpleNamespace(address=_PD_EOA),
            _PD_SAFE,
            _PD_TRACKER,
            10**18,
        )
        assert handler._send_from_eoa.call_args.kwargs["value"] == 500

    def test_a_failed_native_send_is_not_a_shortfall(self) -> None:
        """The balance covered the deposit, so the failure is not about funds."""
        handler, _ = _make_predeposit_handler()
        balance = handlers_module.X402_ETH_DEFICIT_FLOOR_WEI + 10**18
        _install_rpc(
            handler, {_PD_TRACKER: _make_fake_contract()}, native_balance=balance
        )
        handler._send_from_eoa = MagicMock(return_value=False)
        assert (
            handler._deposit_native(
                _PD_CHAIN,
                _PD_CHAIN_ID,
                SimpleNamespace(address=_PD_EOA),
                _PD_SAFE,
                _PD_TRACKER,
                4242,
            )
            is handlers_module.MechDepositOutcome.UNAVAILABLE
        )

    def test_the_native_deposit_credits_the_safe(self) -> None:
        """The EOA sends the value but the Safe is the requester credited."""
        handler, _ = _make_predeposit_handler()
        tracker = _make_fake_contract()
        balance = handlers_module.X402_ETH_DEFICIT_FLOOR_WEI + 10**18
        _install_rpc(handler, {_PD_TRACKER: tracker}, native_balance=balance)
        handler._send_from_eoa = MagicMock(return_value=True)
        handler._deposit_native(
            _PD_CHAIN,
            _PD_CHAIN_ID,
            SimpleNamespace(address=_PD_EOA),
            _PD_SAFE,
            _PD_TRACKER,
            4242,
        )
        assert tracker.encode_abi.call_args.kwargs["args"] == [
            Web3.to_checksum_address(_PD_SAFE)
        ]

    def test_a_deposit_under_the_target_is_not_inflated(self) -> None:
        """A spendable balance above the shortfall deposits only the shortfall."""
        handler, _ = _make_predeposit_handler()
        balance = handlers_module.X402_ETH_DEFICIT_FLOOR_WEI + 10**18
        _install_rpc(
            handler, {_PD_TRACKER: _make_fake_contract()}, native_balance=balance
        )
        handler._send_from_eoa = MagicMock(return_value=True)
        handler._deposit_native(
            _PD_CHAIN,
            _PD_CHAIN_ID,
            SimpleNamespace(address=_PD_EOA),
            _PD_SAFE,
            _PD_TRACKER,
            4242,
        )
        assert handler._send_from_eoa.call_args.kwargs["value"] == 4242

    @pytest.mark.parametrize("offset", [-1, 0])
    def test_a_balance_at_or_under_the_reserve_deposits_nothing(
        self, offset: int
    ) -> None:
        """At the reserve there is nothing spendable left."""
        handler, _ = _make_predeposit_handler()
        balance = handlers_module.X402_ETH_DEFICIT_FLOOR_WEI + offset
        _install_rpc(handler, native_balance=balance)
        handler._send_from_eoa = MagicMock()
        assert (
            handler._deposit_native(
                _PD_CHAIN,
                _PD_CHAIN_ID,
                SimpleNamespace(address=_PD_EOA),
                _PD_SAFE,
                _PD_TRACKER,
                10**18,
            )
            is handlers_module.MechDepositOutcome.UNDERFUNDED
        )
        handler._send_from_eoa.assert_not_called()

    def test_no_rpc_deposits_nothing(self) -> None:
        """Without an RPC the balance cannot be read."""
        handler, _ = _make_predeposit_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        handler._send_from_eoa = MagicMock()
        assert (
            handler._deposit_native(
                _PD_CHAIN,
                _PD_CHAIN_ID,
                SimpleNamespace(address=_PD_EOA),
                _PD_SAFE,
                _PD_TRACKER,
                10**18,
            )
            is handlers_module.MechDepositOutcome.UNAVAILABLE
        )
        handler._send_from_eoa.assert_not_called()

    def test_a_tracker_that_cannot_be_bound_deposits_nothing(self) -> None:
        """A tracker with no contract to encode against deposits nothing."""
        handler, _ = _make_predeposit_handler()
        balance = handlers_module.X402_ETH_DEFICIT_FLOOR_WEI + 10**18
        _install_rpc(handler, native_balance=balance)
        handler._contract = MagicMock(return_value=None)
        handler._send_from_eoa = MagicMock()
        assert (
            handlers_module.MechDepositOutcome.UNAVAILABLE
            is handler._deposit_native(
                _PD_CHAIN,
                _PD_CHAIN_ID,
                SimpleNamespace(address=_PD_EOA),
                _PD_SAFE,
                _PD_TRACKER,
                10**18,
            )
        )
        handler._send_from_eoa.assert_not_called()


class TestTopUpSizing:
    """Cover when a top-up happens and how large it is."""

    def _handler_at(self, deposited: int, **overrides: Any) -> Any:
        """Return a handler whose tracker already holds ``deposited``.

        :param deposited: what the tracker reports for the Safe.
        :param overrides: threshold overrides.
        :return: the handler, the token contract and the tracker contract.
        """
        handler, _ = _make_predeposit_handler(**overrides)
        token = _make_fake_contract()
        tracker = _make_fake_contract(token=_PD_TOKEN)
        # First read is the pot before the deposit; the read-back after a
        # deposit sees it landed in full.
        reads = SimpleNamespace(call=MagicMock(side_effect=[deposited, _PD_TARGET]))
        tracker.functions.mapRequesterBalances = lambda *_: reads
        marketplace = _make_fake_contract(mapPaymentTypeBalanceTrackers=_PD_TRACKER)
        _install_rpc(
            handler,
            {
                _PD_MARKETPLACE: marketplace,
                _PD_TRACKER: tracker,
                _PD_TOKEN: token,
            },
        )
        handler._check_usdc_balance = MagicMock(return_value=_PD_CAP)
        handler._send_from_eoa = MagicMock(return_value=True)
        return handler, token, tracker

    def _top_up(self, handler: Any) -> bool:
        """Run one top-up with the standard arguments.

        :param handler: the handler under test.
        :return: whether the pre-deposit is sufficient.
        """
        return handler._top_up_mech_pre_deposit(
            _PD_CHAIN,
            _PD_CHAIN_ID,
            SimpleNamespace(address=_PD_EOA),
            _PD_SAFE,
            _PD_PAYMENT_TYPE,
        )

    @pytest.mark.parametrize("deposited", [_PD_FLOOR, _PD_FLOOR + 1, _PD_TARGET])
    def test_at_or_above_the_floor_nothing_is_sent(self, deposited: int) -> None:
        """The floor is when to act, so at it there is nothing to do."""
        handler, _, _ = self._handler_at(deposited)
        assert handlers_module.MechDepositOutcome.SUFFICIENT is self._top_up(handler)
        handler._send_from_eoa.assert_not_called()

    def test_below_the_floor_the_deposit_reaches_the_target(self) -> None:
        """A top-up fills to the target, not merely back over the floor."""
        handler, _, tracker = self._handler_at(_PD_FLOOR - 1)
        assert handlers_module.MechDepositOutcome.SUFFICIENT is self._top_up(handler)
        assert tracker.encode_abi.call_args.kwargs["args"][1] == (
            _PD_TARGET - (_PD_FLOOR - 1)
        )

    def test_the_cap_bounds_one_top_up(self) -> None:
        """A misconfigured target cannot drain the EOA in one go."""
        handler, _, tracker = self._handler_at(0, target=10**9, cap=1000)
        assert handlers_module.MechDepositOutcome.SUFFICIENT is self._top_up(handler)
        assert tracker.encode_abi.call_args.kwargs["args"][1] == 1000

    def test_a_target_under_the_deposit_sends_nothing(self) -> None:
        """A target below what is held leaves a non-positive amount."""
        handler, _, _ = self._handler_at(100, floor=200, target=50)
        assert handlers_module.MechDepositOutcome.SUFFICIENT is self._top_up(handler)
        handler._send_from_eoa.assert_not_called()

    def test_a_deposit_that_leaves_the_pot_below_the_floor_is_not_sufficient(
        self,
    ) -> None:
        """Leftover USDC on the EOA must not open chat on a near-empty pot.

        The deposit is clamped to what the EOA holds, so a sent deposit can be
        one base unit. Only the pot's balance after the deposit says whether
        calls can be paid for.
        """
        handler, _, tracker = self._handler_at(0)
        handler._check_usdc_balance = MagicMock(return_value=1)
        reads = SimpleNamespace(call=MagicMock(side_effect=[0, 1]))
        tracker.functions.mapRequesterBalances = lambda *_: reads
        assert self._top_up(handler) is handlers_module.MechDepositOutcome.UNDERFUNDED
        assert handler._send_from_eoa.call_count == 2

    def test_a_deposit_that_reaches_the_floor_is_sufficient(self) -> None:
        """The pot is read back after the deposit, not assumed from the send."""
        handler, _, tracker = self._handler_at(0)
        reads = SimpleNamespace(call=MagicMock(side_effect=[0, _PD_TARGET]))
        tracker.functions.mapRequesterBalances = lambda *_: reads
        assert self._top_up(handler) is handlers_module.MechDepositOutcome.SUFFICIENT

    def test_an_unreadable_pot_after_a_deposit_is_unavailable(self) -> None:
        """A sent deposit with no readable result is not evidence either way."""
        handler, _, tracker = self._handler_at(0)
        reads = SimpleNamespace(call=MagicMock(side_effect=[0, ValueError("rpc")]))
        tracker.functions.mapRequesterBalances = lambda *_: reads
        assert self._top_up(handler) is handlers_module.MechDepositOutcome.UNAVAILABLE

    def test_a_native_tracker_takes_the_native_route(self) -> None:
        """A tracker with no token() is funded with value, not an approve."""
        handler, _ = _make_predeposit_handler()
        tracker = _make_fake_contract(token=ContractLogicError("execution reverted"))
        reads = SimpleNamespace(call=MagicMock(side_effect=[0, _PD_TARGET]))
        tracker.functions.mapRequesterBalances = lambda *_: reads
        marketplace = _make_fake_contract(mapPaymentTypeBalanceTrackers=_PD_TRACKER)
        _install_rpc(
            handler,
            {_PD_MARKETPLACE: marketplace, _PD_TRACKER: tracker},
            native_balance=handlers_module.X402_ETH_DEFICIT_FLOOR_WEI + 10**18,
        )
        handler._send_from_eoa = MagicMock(return_value=True)
        assert handlers_module.MechDepositOutcome.SUFFICIENT is self._top_up(handler)
        assert handler._send_from_eoa.call_args.kwargs["value"] == _PD_CAP

    def test_an_unresolvable_tracker_sends_nothing(self) -> None:
        """No tracker means nothing to deposit into."""
        handler, _ = _make_predeposit_handler(marketplaces={})
        handler._send_from_eoa = MagicMock()
        assert handlers_module.MechDepositOutcome.UNAVAILABLE is self._top_up(handler)
        handler._send_from_eoa.assert_not_called()

    def test_an_unreadable_deposit_sends_nothing(self) -> None:
        """An unreadable pot is not assumed empty."""
        handler, _ = _make_predeposit_handler()
        tracker = _make_fake_contract(mapRequesterBalances=ValueError("reverted"))
        marketplace = _make_fake_contract(mapPaymentTypeBalanceTrackers=_PD_TRACKER)
        _install_rpc(handler, {_PD_MARKETPLACE: marketplace, _PD_TRACKER: tracker})
        handler._send_from_eoa = MagicMock()
        assert handlers_module.MechDepositOutcome.UNAVAILABLE is self._top_up(handler)
        handler._send_from_eoa.assert_not_called()


class TestEnsureMechPreDeposit:
    """Cover the entry point that resolves the chain, Safe and EOA."""

    def _wire(self, handler: Any) -> None:
        """Give the handler a readable chain and a funded EOA.

        :param handler: the handler to wire.
        """
        marketplace = _make_fake_contract(mapPaymentTypeBalanceTrackers=_PD_TRACKER)
        tracker = _make_fake_contract(mapRequesterBalances=_PD_TARGET, token=_PD_TOKEN)
        _install_rpc(handler, {_PD_MARKETPLACE: marketplace, _PD_TRACKER: tracker})
        handler._get_eoa_account = MagicMock(
            return_value=SimpleNamespace(address=_PD_EOA)
        )

    def test_a_funded_pre_deposit_reports_sufficient(self) -> None:
        """Nothing to do is still a success."""
        handler, _ = _make_predeposit_handler()
        self._wire(handler)
        with patch.object(
            handlers_module.requests,
            "get",
            return_value=_make_response(payload={"payment_type": _PD_PAYMENT_TYPE_HEX}),
        ):
            assert (
                handlers_module.MechDepositOutcome.SUFFICIENT
                is handler._ensure_mech_pre_deposit()
            )

    def test_the_mech_chain_decides_which_safe_pays(self) -> None:
        """The Safe is the one on the chain the marketplace charges."""
        handler, _ = _make_predeposit_handler(
            safes={_PD_CHAIN: _PD_SAFE, "base": _PD_EOA}
        )
        self._wire(handler)
        with patch.object(
            handlers_module.requests,
            "get",
            return_value=_make_response(payload={"payment_type": _PD_PAYMENT_TYPE_HEX}),
        ) as get:
            handler._ensure_mech_pre_deposit()
        assert Web3.to_checksum_address(_PD_SAFE) in get.call_args[0][0]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"chain_ids": {}},
            {"safes": {}},
            {"safes": {_PD_CHAIN: ""}},
            {"mech_chain": "unconfigured"},
        ],
    )
    def test_an_unconfigured_chain_deposits_nothing(self, overrides: Any) -> None:
        """A chain with no id or no Safe cannot be funded."""
        handler, _ = _make_predeposit_handler(**overrides)
        self._wire(handler)
        handler._send_from_eoa = MagicMock()
        assert (
            handlers_module.MechDepositOutcome.UNAVAILABLE
            is handler._ensure_mech_pre_deposit()
        )
        handler._send_from_eoa.assert_not_called()

    def test_no_eoa_deposits_nothing(self) -> None:
        """Without a key there is nobody to pay the deposit."""
        handler, _ = _make_predeposit_handler()
        self._wire(handler)
        handler._get_eoa_account = MagicMock(return_value=None)
        handler._send_from_eoa = MagicMock()
        assert (
            handlers_module.MechDepositOutcome.UNAVAILABLE
            is handler._ensure_mech_pre_deposit()
        )
        handler._send_from_eoa.assert_not_called()

    def test_an_unreadable_payment_type_deposits_nothing(self) -> None:
        """Without a payment type no tracker can be resolved."""
        handler, _ = _make_predeposit_handler()
        self._wire(handler)
        handler._send_from_eoa = MagicMock()
        with patch.object(
            handlers_module.requests,
            "get",
            return_value=_make_response(status_code=503),
        ):
            assert (
                handlers_module.MechDepositOutcome.UNAVAILABLE
                is handler._ensure_mech_pre_deposit()
            )
        handler._send_from_eoa.assert_not_called()

    def test_an_unexpected_error_is_contained(self) -> None:
        """A background step must not take the handler down with it."""
        handler, _ = _make_predeposit_handler()
        handler._get_eoa_account = MagicMock(side_effect=RuntimeError("boom"))
        assert (
            handlers_module.MechDepositOutcome.UNAVAILABLE
            is handler._ensure_mech_pre_deposit()
        )

    def test_a_concurrent_top_up_is_not_duplicated(self) -> None:
        """Two top-ups would re-broadcast the same approve from one nonce."""
        handler, _ = _make_predeposit_handler()
        handler._get_eoa_account = MagicMock()
        handlers_module._MECH_PRE_DEPOSIT_LOCK.acquire()
        try:
            assert (
                handlers_module.MechDepositOutcome.UNAVAILABLE
                is handler._ensure_mech_pre_deposit()
            )
        finally:
            handlers_module._MECH_PRE_DEPOSIT_LOCK.release()
        handler._get_eoa_account.assert_not_called()

    def test_the_lock_is_released_after_an_error(self) -> None:
        """A failed top-up must not block every later one."""
        handler, _ = _make_predeposit_handler()
        handler._get_eoa_account = MagicMock(side_effect=RuntimeError("boom"))
        handler._ensure_mech_pre_deposit()
        assert handlers_module._MECH_PRE_DEPOSIT_LOCK.acquire(blocking=False)
        handlers_module._MECH_PRE_DEPOSIT_LOCK.release()


class TestPaidCallFundingMaintenance:
    """Cover which funding steps run on each route."""

    def test_the_plain_x402_route_only_keeps_the_eoa_funded(self) -> None:
        """With no facilitator the EOA pays each call from its own balance."""
        handler, _ = _make_predeposit_handler(use_mech_facilitator=False)
        handler._ensure_sufficient_funds_for_x402_payments = MagicMock()
        handler._ensure_mech_pre_deposit = MagicMock()
        handler._maintain_paid_call_funding()
        handler._ensure_sufficient_funds_for_x402_payments.assert_called_once_with()
        handler._ensure_mech_pre_deposit.assert_not_called()

    def test_the_facilitator_route_swaps_before_it_deposits(self) -> None:
        """The deposit spends the token the swap puts on the EOA."""
        handler, _ = _make_predeposit_handler(use_mech_facilitator=True)
        order = []
        handler._ensure_sufficient_funds_for_x402_payments = MagicMock(
            side_effect=lambda: order.append("swap")
        )
        handler._ensure_mech_pre_deposit = MagicMock(
            side_effect=lambda: order.append("deposit")
        )
        handler._maintain_paid_call_funding()
        assert order == ["swap", "deposit"]


class TestPaidCallChain:
    """Cover which chain the agent keeps its payment balance on."""

    def test_the_facilitator_route_uses_the_mech_chain(self) -> None:
        """The marketplace debits a pot on mech_chain, so the token is needed there."""
        handler, ctx = _make_predeposit_handler(use_mech_facilitator=True)
        ctx.params.target_investment_chains = ["base"]
        assert handler._paid_call_chain() == _PD_CHAIN

    def test_the_plain_route_uses_the_first_trading_chain(self) -> None:
        """Off the facilitator the EOA pays each call from its own balance."""
        handler, ctx = _make_predeposit_handler(use_mech_facilitator=False)
        ctx.params.target_investment_chains = ["base"]
        assert handler._paid_call_chain() == "base"

    def test_the_swap_and_the_deposit_cannot_target_different_chains(self) -> None:
        """The swap funds the EOA on whichever chain the deposit will spend on.

        With the swap reading target_investment_chains and the deposit reading
        mech_chain, the shipped service config pointed them at base and optimism
        respectively, so the EOA was funded on a chain the deposit never touched.
        """
        handler, ctx = _make_predeposit_handler(use_mech_facilitator=True)
        ctx.params.target_investment_chains = ["base", "optimism", "mode"]
        ctx.params.x402_payment_requirements = {"threshold": 1000, "topup": 5000}
        handler._get_eoa_account = MagicMock(
            return_value=SimpleNamespace(address=_PD_EOA)
        )
        handler._check_usdc_balance = MagicMock(return_value=2000)
        with patch.object(type(handler), "shared_state", new_callable=PropertyMock):
            handler._ensure_sufficient_funds_for_x402_payments()
        swap_chain = handler._check_usdc_balance.call_args[0][1]
        assert swap_chain == handler._paid_call_chain() == _PD_CHAIN

    def test_the_swap_buys_the_token_the_tracker_takes(self) -> None:
        """The deposit spends USDC on the mech chain, so that is what is bought."""
        handler, ctx = _make_predeposit_handler(use_mech_facilitator=True)
        ctx.params.target_investment_chains = ["base"]
        ctx.params.x402_payment_requirements = {"threshold": 1000, "topup": 5000}
        handler._get_eoa_account = MagicMock(
            return_value=SimpleNamespace(address=_PD_EOA)
        )
        handler._check_usdc_balance = MagicMock(return_value=2000)
        with patch.object(type(handler), "shared_state", new_callable=PropertyMock):
            handler._ensure_sufficient_funds_for_x402_payments()
        assert handler._check_usdc_balance.call_args[0][2] == (
            handlers_module.USDC_ADDRESSES[_PD_CHAIN]
        )


class TestPendingNonce:
    """Cover that in-flight transactions still consume their nonce."""

    def test_the_deposit_nonce_counts_pending_transactions(self) -> None:
        """A deposit signed against "latest" can take a pending swap's nonce."""
        handler, _ = _make_predeposit_handler()
        w3 = _install_rpc(handler)
        handler._sign_and_submit_tx_web3 = MagicMock(return_value="0xhash")
        handler._check_transaction_status = MagicMock(return_value=True)
        handler._send_from_eoa(
            _PD_CHAIN, _PD_CHAIN_ID, SimpleNamespace(address=_PD_EOA), _PD_TRACKER, "0x"
        )
        assert w3.eth.get_transaction_count.call_args[0][1] == "pending"

    def test_the_swap_nonce_counts_pending_transactions(self) -> None:
        """The swap shares the EOA with the deposit, so it counts the same way."""
        handler, _ = _make_predeposit_handler()
        w3 = _install_rpc(handler)
        handler._call_web3_with_breaker = MagicMock(
            side_effect=lambda _c, fn, *a: fn(*a)
        )
        handler._get_nonce_and_gas_web3(_PD_EOA, _PD_CHAIN)
        assert w3.eth.get_transaction_count.call_args[0][1] == "pending"


class TestMechDepositReporting:
    """Cover what a failed pre-deposit tells /funds-status."""

    def test_a_funded_pre_deposit_overrides_a_failed_swap(self) -> None:
        """The pot pays for calls, so a full pot means prompts can be served."""
        handler, _ = _make_predeposit_handler()
        handler._record_x402_topup_outcome = MagicMock()
        handler._record_mech_pre_deposit_outcome(
            handlers_module.MechDepositOutcome.SUFFICIENT
        )
        sufficient, deficit, _ = handler._record_x402_topup_outcome.call_args[0]
        assert sufficient is True
        assert deficit is None

    def test_a_funded_pre_deposit_keeps_the_swap_deficit(self) -> None:
        """The ETH the swap asked for is still what the next refill needs."""
        handler, _ = _make_predeposit_handler()
        state = SimpleNamespace(
            sufficient_funds_for_x402_payments=False, x402_eth_deficit=777
        )
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock, return_value=state
        ):
            handler._record_mech_pre_deposit_outcome(
                handlers_module.MechDepositOutcome.SUFFICIENT
            )
        assert state.sufficient_funds_for_x402_payments is True
        assert state.x402_eth_deficit == 777
        assert state.x402_funding_checked is True

    def test_an_unknown_pre_deposit_leaves_the_swap_verdict_alone(self) -> None:
        """Recording anything here would refuse chat on one failed read."""
        handler, _ = _make_predeposit_handler()
        handler._record_x402_topup_outcome = MagicMock()
        handler._record_mech_pre_deposit_outcome(
            handlers_module.MechDepositOutcome.UNAVAILABLE
        )
        handler._record_x402_topup_outcome.assert_not_called()

    @pytest.mark.parametrize(
        "sends",
        [
            pytest.param([False], id="approve fails"),
            pytest.param([True, False], id="deposit fails after approve"),
        ],
    )
    def test_a_deposit_that_keeps_failing_does_not_gate_chat(self, sends: Any) -> None:
        """A reverting or timed-out send on a funded agent is not a funding verdict."""
        handler, _ = _make_predeposit_handler()
        tracker = _make_fake_contract(mapRequesterBalances=0, token=_PD_TOKEN)
        marketplace = _make_fake_contract(mapPaymentTypeBalanceTrackers=_PD_TRACKER)
        _install_rpc(
            handler,
            {
                _PD_MARKETPLACE: marketplace,
                _PD_TRACKER: tracker,
                _PD_TOKEN: _make_fake_contract(),
            },
        )
        handler._check_usdc_balance = MagicMock(return_value=_PD_CAP)
        handler._send_from_eoa = MagicMock(side_effect=sends)
        handler._record_x402_topup_outcome = MagicMock()

        outcome = handler._top_up_mech_pre_deposit(
            _PD_CHAIN,
            _PD_CHAIN_ID,
            SimpleNamespace(address=_PD_EOA),
            _PD_SAFE,
            _PD_PAYMENT_TYPE,
        )
        handler._record_mech_pre_deposit_outcome(outcome)

        assert outcome is handlers_module.MechDepositOutcome.UNAVAILABLE
        handler._record_x402_topup_outcome.assert_not_called()

    def test_an_unreadable_tracker_token_does_not_gate_chat(self) -> None:
        """An RPC failure mid-top-up must not turn into a funding verdict.

        Read as "native", it would send the deposit down the value route,
        fail there and be recorded as the agent being unable to pay.
        """
        handler, _ = _make_predeposit_handler()
        tracker = _make_fake_contract(
            mapRequesterBalances=0, token=requests.exceptions.Timeout("rpc")
        )
        marketplace = _make_fake_contract(mapPaymentTypeBalanceTrackers=_PD_TRACKER)
        _install_rpc(handler, {_PD_MARKETPLACE: marketplace, _PD_TRACKER: tracker})
        handler._send_from_eoa = MagicMock()
        handler._record_x402_topup_outcome = MagicMock()

        outcome = handler._top_up_mech_pre_deposit(
            _PD_CHAIN,
            _PD_CHAIN_ID,
            SimpleNamespace(address=_PD_EOA),
            _PD_SAFE,
            _PD_PAYMENT_TYPE,
        )
        handler._record_mech_pre_deposit_outcome(outcome)

        assert outcome is handlers_module.MechDepositOutcome.UNAVAILABLE
        handler._send_from_eoa.assert_not_called()
        handler._record_x402_topup_outcome.assert_not_called()

    def test_an_unaffordable_deposit_on_an_empty_eoa_reports_the_floor(self) -> None:
        """This is what makes Pearl ask the user to top the agent up."""
        handler, _ = _make_predeposit_handler()
        handler._get_eoa_account = MagicMock(
            return_value=SimpleNamespace(address=_PD_EOA)
        )
        handler._get_native_balance = MagicMock(return_value=0)
        handler._record_x402_topup_outcome = MagicMock()
        handler._record_mech_pre_deposit_outcome(
            handlers_module.MechDepositOutcome.UNDERFUNDED
        )
        sufficient, deficit, _ = handler._record_x402_topup_outcome.call_args[0]
        assert sufficient is False
        assert deficit == handlers_module.X402_ETH_DEFICIT_FLOOR_WEI

    def test_an_unaffordable_deposit_on_a_funded_eoa_reports_no_deficit(self) -> None:
        """More gas would not help, so the failure is not a funding one."""
        handler, _ = _make_predeposit_handler()
        handler._get_eoa_account = MagicMock(
            return_value=SimpleNamespace(address=_PD_EOA)
        )
        handler._get_native_balance = MagicMock(
            return_value=handlers_module.X402_ETH_DEFICIT_FLOOR_WEI * 10
        )
        handler._record_x402_topup_outcome = MagicMock()
        handler._record_mech_pre_deposit_outcome(
            handlers_module.MechDepositOutcome.UNDERFUNDED
        )
        sufficient, deficit, _ = handler._record_x402_topup_outcome.call_args[0]
        assert sufficient is False
        assert deficit is None

    def test_an_unavailable_eoa_reports_no_deficit(self) -> None:
        """With no key there is no balance to size a shortfall from."""
        handler, _ = _make_predeposit_handler()
        handler._get_eoa_account = MagicMock(return_value=None)
        handler._record_x402_topup_outcome = MagicMock()
        handler._record_mech_pre_deposit_outcome(
            handlers_module.MechDepositOutcome.UNDERFUNDED
        )
        assert handler._record_x402_topup_outcome.call_args[0][1] is None

    def test_the_deposit_outcome_reaches_the_reporting(self) -> None:
        """The result used to be discarded, so a failure never showed up."""
        handler, _ = _make_predeposit_handler(use_mech_facilitator=True)
        handler._ensure_sufficient_funds_for_x402_payments = MagicMock()
        handler._ensure_mech_pre_deposit = MagicMock(
            return_value=handlers_module.MechDepositOutcome.UNDERFUNDED
        )
        handler._record_mech_pre_deposit_outcome = MagicMock()
        handler._maintain_paid_call_funding()
        handler._record_mech_pre_deposit_outcome.assert_called_once_with(
            handlers_module.MechDepositOutcome.UNDERFUNDED
        )


class TestPaidCallFundingLock:
    """Cover that the swap and the deposit are serialized together."""

    def test_a_second_run_does_neither_step(self) -> None:
        """A deposit started mid-swap would be signed against the swap's nonce."""
        handler, _ = _make_predeposit_handler(use_mech_facilitator=True)
        handler._ensure_sufficient_funds_for_x402_payments = MagicMock()
        handler._ensure_mech_pre_deposit = MagicMock()
        handlers_module._PAID_CALL_FUNDING_LOCK.acquire()
        try:
            handler._maintain_paid_call_funding()
        finally:
            handlers_module._PAID_CALL_FUNDING_LOCK.release()
        handler._ensure_sufficient_funds_for_x402_payments.assert_not_called()
        handler._ensure_mech_pre_deposit.assert_not_called()

    def test_the_lock_is_released_after_an_error(self) -> None:
        """One failed cycle must not block every later one."""
        handler, _ = _make_predeposit_handler(use_mech_facilitator=False)
        handler._ensure_sufficient_funds_for_x402_payments = MagicMock(
            side_effect=RuntimeError("boom")
        )
        with pytest.raises(RuntimeError):
            handler._maintain_paid_call_funding()
        assert handlers_module._PAID_CALL_FUNDING_LOCK.acquire(blocking=False)
        handlers_module._PAID_CALL_FUNDING_LOCK.release()


class TestPaidCallFundingOffTheTradingChain:
    """Cover an agent that trades on one chain and pays for calls on another."""

    @staticmethod
    def _handler() -> Any:
        """Return a handler trading on base with the mech on the default chain."""
        handler, ctx = _make_predeposit_handler(use_mech_facilitator=True)
        ctx.params.target_investment_chains = ["base", "optimism", "mode"]
        ctx.agent_address = _PD_EOA
        return handler, ctx

    def test_the_deficit_is_reported_where_the_swap_runs(self) -> None:
        """ETH sent to the trading chain never reaches a swap on the mech chain."""
        handler, _ = self._handler()
        result = handler._inject_x402_eth_deficit({}, 1000)
        assert result == {_PD_CHAIN: {_PD_EOA: {ZERO_ADDRESS: {"deficit": "1000"}}}}

    def test_the_trading_chain_balance_does_not_offset_the_deficit(self) -> None:
        """A funded trading chain used to hide a shortfall on the mech chain."""
        handler, _ = self._handler()
        existing = {"base": {_PD_EOA: {ZERO_ADDRESS: {"balance": "999999"}}}}
        result = handler._inject_x402_eth_deficit(existing, 1000)
        assert result[_PD_CHAIN][_PD_EOA][ZERO_ADDRESS]["deficit"] == "1000"
        assert "deficit" not in result["base"][_PD_EOA][ZERO_ADDRESS]

    def test_the_plain_route_still_reports_on_the_trading_chain(self) -> None:
        """Off the facilitator the swap runs on the trading chain, as before."""
        handler, ctx = _make_predeposit_handler(use_mech_facilitator=False)
        ctx.params.target_investment_chains = ["base"]
        ctx.agent_address = _PD_EOA
        result = handler._inject_x402_eth_deficit({}, 1000)
        assert list(result) == ["base"]

    def test_differing_chains_are_warned_about(self) -> None:
        """A funder watching only the trading chain will not fund the mech chain."""
        handler, ctx = self._handler()
        handler._warn_if_paid_calls_are_funded_off_the_trading_chain()
        message = ctx.logger.warning.call_args[0][0]
        assert f"funded on {_PD_CHAIN}" in message
        assert "trades on base" in message

    @pytest.mark.parametrize(
        "use_mech_facilitator,trading_chains",
        [
            pytest.param(True, [_PD_CHAIN], id="facilitator on the trading chain"),
            pytest.param(False, ["base"], id="plain x402 route"),
        ],
    )
    def test_matching_chains_are_not_warned_about(
        self, use_mech_facilitator: bool, trading_chains: Any
    ) -> None:
        """The warning is for a misconfiguration, not for every start-up."""
        handler, ctx = _make_predeposit_handler(
            use_mech_facilitator=use_mech_facilitator
        )
        ctx.params.target_investment_chains = trading_chains
        handler._warn_if_paid_calls_are_funded_off_the_trading_chain()
        ctx.logger.warning.assert_not_called()


class TestPaidCallUnavailableMessage:
    """Cover what a refused prompt is told."""

    @staticmethod
    def _refused_prompt(handler: Any, state: Any) -> str:
        handler.context.params.use_x402 = True
        handler.context.state.request_queue = []
        handler._send_ok_response = MagicMock()
        dialogue = MagicMock()
        dialogue.dialogue_label.dialogue_reference = ("req1", "")
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock, return_value=state
        ):
            handler._handle_post_process_prompt(MagicMock(), dialogue)
        # A refused prompt must give its queue slot back, or every later
        # strategy write from chat is dropped until the agent restarts.
        assert handler.context.state.request_queue == []
        return handler._send_ok_response.call_args[0][2]["error"]

    def test_before_any_check_has_reported_it_is_initializing(self) -> None:
        """Nothing is known yet, so "initializing" is the truth."""
        handler, _ = _make_predeposit_handler()
        state = SimpleNamespace(
            sufficient_funds_for_x402_payments=False,
            x402_funding_checked=False,
            x402_eth_deficit=0,
        )
        assert self._refused_prompt(handler, state).startswith("System initializing")

    def test_a_recorded_deficit_asks_for_funds(self) -> None:
        """Waiting will not fix this, so the user is told what will."""
        handler, _ = _make_predeposit_handler()
        state = SimpleNamespace(
            sufficient_funds_for_x402_payments=False,
            x402_funding_checked=True,
            x402_eth_deficit=300000000000000,
        )
        message = self._refused_prompt(handler, state)
        assert "Add ETH to your agent" in message
        assert "initializing" not in message

    @pytest.mark.parametrize("deficit", [0, None])
    def test_a_check_without_a_deficit_is_temporarily_unavailable(
        self, deficit: Any
    ) -> None:
        """Infrastructure failures must not ask the user for money."""
        handler, _ = _make_predeposit_handler()
        state = SimpleNamespace(
            sufficient_funds_for_x402_payments=False,
            x402_funding_checked=True,
            x402_eth_deficit=deficit,
        )
        message = self._refused_prompt(handler, state)
        assert "temporarily unavailable" in message
        assert "Add ETH" not in message
        assert "initializing" not in message

    def test_a_check_that_reports_marks_funding_as_checked(self) -> None:
        """Every exit of the swap and the deposit routes through the recorder."""
        handler, _ = _make_predeposit_handler()
        state = SimpleNamespace(
            sufficient_funds_for_x402_payments=False,
            x402_eth_deficit=0,
            x402_funding_checked=False,
        )
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock, return_value=state
        ):
            handler._record_x402_topup_outcome(False, None, "LiFi quote unavailable")
        assert state.x402_funding_checked is True


class TestX402SwapSlippage:
    """Cover the slippage the top-up swap is quoted with."""

    def test_the_top_up_uses_its_own_slippage_not_the_trading_one(self) -> None:
        """The top-up is quoted with its own slippage, not the trading one."""
        handler, ctx = _make_http_handler()
        ctx.params.chain_to_chain_id_mapping = {"optimism": 10}
        ctx.params.slippage_for_swap = 0.005
        ctx.params.x402_swap_slippage = 0.05
        ctx.params.lifi_quote_to_amount_url = "https://api.example.com/quote"
        response = MagicMock(status_code=200)
        response.json.return_value = {"quote": "data"}
        with patch(
            "packages.valory.skills.optimus_abci.handlers.requests.get",
            return_value=response,
        ) as get:
            handler._get_lifi_quote_sync("0xaddr", "optimism", "0xusdc", "250000")
        assert get.call_args.kwargs["params"]["slippage"] == 0.05


class TestWhoOwnsTheChatFlag:
    """Cover which step decides the chat flag on each route."""

    @staticmethod
    def _state() -> Any:
        return SimpleNamespace(
            sufficient_funds_for_x402_payments=True,
            x402_funding_checked=False,
            x402_eth_deficit=0,
        )

    def test_the_swap_does_not_touch_the_flag_when_told_not_to(self) -> None:
        """It still records the deficit and that a check has reported."""
        handler, _ = _make_predeposit_handler()
        state = self._state()
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock, return_value=state
        ):
            handler._record_x402_topup_outcome(
                False, 777, "quote refused", gate_chat=False
            )
        assert state.sufficient_funds_for_x402_payments is True
        assert state.x402_eth_deficit == 777
        assert state.x402_funding_checked is True

    def test_the_swap_owns_the_flag_by_default(self) -> None:
        """Off the facilitator route the EOA pays, so the swap's verdict stands."""
        handler, _ = _make_predeposit_handler()
        state = self._state()
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock, return_value=state
        ):
            handler._record_x402_topup_outcome(False, 777, "quote refused")
        assert state.sufficient_funds_for_x402_payments is False

    @staticmethod
    def _run_funding(handler: Any, state: Any, deposit_outcome: Any) -> None:
        """Run the real maintenance step with a refused quote and a given pot state."""
        ctx = handler.context
        ctx.params.x402_payment_requirements = {"threshold": 200000, "topup": 250000}
        ctx.params.target_investment_chains = [_PD_CHAIN]
        handler._get_eoa_account = MagicMock(
            return_value=SimpleNamespace(address=_PD_EOA)
        )
        handler._check_usdc_balance = MagicMock(return_value=0)
        handler._get_lifi_quote_sync = MagicMock(return_value=None)
        handler._x402_floor_deficit_if_unfunded = MagicMock(return_value=777)
        handler._ensure_mech_pre_deposit = MagicMock(return_value=deposit_outcome)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock, return_value=state
        ):
            handler._maintain_paid_call_funding()

    def test_a_refused_quote_then_a_funded_pot_leaves_chat_on(self) -> None:
        """The real ordering through the real recorders: swap fails, pot is full.

        The flag must never read False, not even between the two steps, since
        prompts arriving in that window were refused with a full pot.
        """
        handler, _ = _make_predeposit_handler(use_mech_facilitator=True)
        state = self._state()
        seen = []
        original = type(state).__setattr__

        class Watching(SimpleNamespace):
            def __setattr__(self, name: str, value: Any) -> None:
                if name == "sufficient_funds_for_x402_payments":
                    seen.append(value)
                original(self, name, value)

        state = Watching(**vars(state))
        self._run_funding(handler, state, handlers_module.MechDepositOutcome.SUFFICIENT)
        assert state.sufficient_funds_for_x402_payments is True
        assert False not in seen
        assert state.x402_eth_deficit == 777

    def test_a_refused_quote_then_an_empty_pot_turns_chat_off(self) -> None:
        """A funded verdict from an earlier cycle must not survive an empty pot."""
        handler, _ = _make_predeposit_handler(use_mech_facilitator=True)
        state = self._state()
        handler._get_native_balance = MagicMock(return_value=0)
        self._run_funding(
            handler, state, handlers_module.MechDepositOutcome.UNDERFUNDED
        )
        assert state.sufficient_funds_for_x402_payments is False
        assert state.x402_eth_deficit > 0

    def test_off_the_facilitator_route_the_swap_still_gates_chat(self) -> None:
        """Nothing changes for the plain x402 route."""
        handler, _ = _make_predeposit_handler(use_mech_facilitator=False)
        state = self._state()
        self._run_funding(handler, state, handlers_module.MechDepositOutcome.SUFFICIENT)
        assert state.sufficient_funds_for_x402_payments is False
        handler._ensure_mech_pre_deposit.assert_not_called()

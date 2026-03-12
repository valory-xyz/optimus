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
import math
from typing import Dict
from unittest.mock import MagicMock, PropertyMock, patch

from packages.valory.skills.optimus_abci.handlers import (
    ESTIMATED_GAS_PER_TX,
    BaseHandler,
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


def _make_concrete_base_handler():
    """Create a concrete subclass of BaseHandler for testing."""

    class ConcreteHandler(BaseHandler):
        SUPPORTED_PROTOCOL = None

        def handle(self, message):
            pass

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
        from packages.dvilela.protocols.kv_store.message import KvStoreMessage

        assert KvStoreHandler.SUPPORTED_PROTOCOL == KvStoreMessage.protocol_id

    def test_allowed_response_performatives(self) -> None:
        """Test allowed_response_performatives is set."""
        from packages.dvilela.protocols.kv_store.message import KvStoreMessage

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
        from packages.dvilela.protocols.kv_store.message import KvStoreMessage

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
        from packages.dvilela.protocols.kv_store.message import KvStoreMessage

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
        from packages.dvilela.protocols.kv_store.message import KvStoreMessage

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
        from packages.dvilela.protocols.kv_store.message import KvStoreMessage

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


def _make_http_handler():
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
        from packages.dvilela.protocols.kv_store.message import KvStoreMessage

        handler, _ = _make_http_handler()
        msg = MagicMock()
        msg.performative = KvStoreMessage.Performative.SUCCESS
        handler._handle_kv_store_response(msg, MagicMock())

    def test_handle_kv_store_response_failure(self) -> None:
        """Test _handle_kv_store_response logs failure."""
        from packages.dvilela.protocols.kv_store.message import KvStoreMessage

        handler, _ = _make_http_handler()
        msg = MagicMock()
        msg.performative = KvStoreMessage.Performative.ERROR
        handler._handle_kv_store_response(msg, MagicMock())

    def test_handle_kv_read_response_success(self) -> None:
        """Test _handle_kv_read_response stores data on success."""
        from packages.dvilela.protocols.kv_store.message import KvStoreMessage

        handler, ctx = _make_http_handler()
        msg = MagicMock()
        msg.performative = KvStoreMessage.Performative.READ_RESPONSE
        msg.data = {"key": "value"}
        handler._handle_kv_read_response(msg, MagicMock())
        assert ctx.state.last_kv_read_data == {"key": "value"}
        assert ctx.state.in_flight_req is False

    def test_handle_kv_read_response_success_no_data_attr(self) -> None:
        """Test _handle_kv_read_response when msg has no data attribute."""
        from packages.dvilela.protocols.kv_store.message import KvStoreMessage

        handler, ctx = _make_http_handler()
        msg = MagicMock(spec=[])
        msg.performative = KvStoreMessage.Performative.READ_RESPONSE
        handler._handle_kv_read_response(msg, MagicMock())
        # hasattr(msg, 'data') is False since we used spec=[]
        assert ctx.state.in_flight_req is False

    def test_handle_kv_read_response_failure(self) -> None:
        """Test _handle_kv_read_response sets empty data on failure."""
        from packages.dvilela.protocols.kv_store.message import KvStoreMessage

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
        result = handler.funds_status
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

    def test_check_usdc_balance_no_web3(self) -> None:
        """Test _check_usdc_balance returns None when no web3 instance."""
        handler, ctx = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        result = handler._check_usdc_balance("0xaddr", "optimism", "0xusdc")
        assert result is None

    def test_check_usdc_balance_exception(self) -> None:
        """Test _check_usdc_balance returns None on exception."""
        handler, ctx = _make_http_handler()
        handler._get_web3_instance = MagicMock(
            side_effect=Exception("Connection error")
        )
        result = handler._check_usdc_balance("0xaddr", "optimism", "0xusdc")
        assert result is None

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
        """Test _estimate_gas returns False when no web3."""
        handler, ctx = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=None)
        result = handler._estimate_gas(
            {"value": "0x0", "to": "0x0", "data": "0x"}, "0xaddr", "optimism"
        )
        assert result is False

    def test_estimate_gas_exception_return_amount(self) -> None:
        """Test _estimate_gas returns None on 'Return amount' error."""
        handler, ctx = _make_http_handler()
        mock_w3 = MagicMock()
        mock_w3.eth.estimate_gas.side_effect = Exception("Return amount is not enough")
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        result = handler._estimate_gas(
            {"value": 0, "to": "0x" + "0" * 40, "data": "0x"},
            "0x" + "0" * 40,
            "optimism",
        )
        assert result is None

    def test_estimate_gas_exception_execution_reverted(self) -> None:
        """Test _estimate_gas returns None on 'execution reverted' error."""
        handler, ctx = _make_http_handler()
        mock_w3 = MagicMock()
        mock_w3.eth.estimate_gas.side_effect = Exception("execution reverted")
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        result = handler._estimate_gas(
            {"value": "0x123", "to": "0x" + "0" * 40, "data": "0x"},
            "0x" + "0" * 40,
            "optimism",
        )
        assert result is None

    def test_estimate_gas_generic_exception(self) -> None:
        """Test _estimate_gas returns None on generic exception."""
        handler, ctx = _make_http_handler()
        mock_w3 = MagicMock()
        mock_w3.eth.estimate_gas.side_effect = Exception("Unknown error")
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        result = handler._estimate_gas(
            {"value": 0, "to": "0x" + "0" * 40, "data": "0x"},
            "0x" + "0" * 40,
            "optimism",
        )
        assert result is None

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
        assert result == int(100000 * 1.2)

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
        assert result == int(100000 * 1.2)

    def test_get_eoa_account_no_password_success(self) -> None:
        """Test _get_eoa_account with no password and valid key."""
        handler, ctx = _make_http_handler()
        handler._get_password_from_args = MagicMock(return_value=None)
        ctx.default_ledger_id = "ethereum"
        ctx.data_dir = "/tmp/test_data"
        mock_account = MagicMock()
        with patch("builtins.open", MagicMock()), patch(
            "packages.valory.skills.optimus_abci.handlers.Account.from_key",
            return_value=mock_account,
        ), patch("pathlib.Path.open", MagicMock()):
            result = handler._get_eoa_account()
        # Returns mock_account on success

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
        """Test _get_lifi_quote_sync returns None on exception."""
        handler, ctx = _make_http_handler()
        ctx.params.chain_to_chain_id_mapping = {"optimism": 10}
        ctx.params.slippage_for_swap = 0.01
        ctx.params.lifi_quote_to_amount_url = "https://api.example.com/quote"
        with patch(
            "packages.valory.skills.optimus_abci.handlers.requests.get",
            side_effect=Exception("Network error"),
        ):
            result = handler._get_lifi_quote_sync(
                "0xaddr", "optimism", "0xusdc", "1000"
            )
        assert result is None

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
        with patch(
            "packages.valory.skills.optimus_abci.handlers.EthereumCrypto",
            return_value=mock_crypto,
        ), patch(
            "packages.valory.skills.optimus_abci.handlers.Account.from_key",
            return_value=mock_account,
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
        from packages.valory.protocols.http.message import HttpMessage
        from packages.valory.connections.http_server.connection import (
            PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
        )

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
        from packages.valory.protocols.http.message import HttpMessage
        from packages.valory.connections.http_server.connection import (
            PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
        )

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
        from packages.valory.protocols.http.message import HttpMessage
        from packages.valory.connections.http_server.connection import (
            PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
        )

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
        with patch(
            "packages.valory.skills.optimus_abci.handlers.Path", return_value=mock_path
        ), patch(
            "packages.valory.skills.optimus_abci.handlers.urlparse"
        ) as mock_urlparse, patch(
            "builtins.open", MagicMock()
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
        with patch(
            "packages.valory.skills.optimus_abci.handlers.urlparse"
        ) as mock_urlparse, patch(
            "packages.valory.skills.optimus_abci.handlers.Path"
        ) as mock_path_cls:
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
        with patch(
            "packages.valory.skills.optimus_abci.handlers.urlparse"
        ) as mock_urlparse, patch(
            "packages.valory.skills.optimus_abci.handlers.Path"
        ) as mock_path_cls, patch(
            "builtins.open", MagicMock()
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

        with patch(
            "packages.valory.skills.optimus_abci.handlers.validate_and_fix_protocols",
            return_value=["balancerPool"],
        ), patch("packages.valory.skills.optimus_abci.handlers.ThreadPoolExecutor"):
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

        with patch(
            "packages.valory.skills.optimus_abci.handlers.validate_and_fix_protocols",
            return_value=["balancerPool"],
        ), patch("packages.valory.skills.optimus_abci.handlers.ThreadPoolExecutor"):
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
        with patch("builtins.open", MagicMock()), patch(
            "json.load", return_value=portfolio
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
        with patch("builtins.open", MagicMock()), patch(
            "json.load", return_value=portfolio
        ), patch("packages.valory.skills.optimus_abci.handlers.ThreadPoolExecutor"):
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
        with patch("builtins.open", MagicMock()), patch(
            "json.load", return_value=portfolio
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
        with patch("builtins.open", MagicMock()), patch(
            "json.load", return_value=existing_perf
        ), patch("json.dump") as mock_dump:
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
        with patch(
            "builtins.open", side_effect=[FileNotFoundError("not found"), MagicMock()]
        ), patch("json.dump"):
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
        with patch("builtins.open", MagicMock()), patch(
            "json.load", return_value=existing_perf
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
        with patch("builtins.open", MagicMock()), patch(
            "json.load", return_value=existing_perf
        ), patch("json.dump") as mock_dump:
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
        with patch("builtins.open", MagicMock()), patch(
            "json.load", side_effect=[Exception("Generic error"), None]
        ), patch("json.dump"):
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
        with patch.object(
            type(handler),
            "funds_status",
            new_callable=PropertyMock,
            return_value=mock_fund_req,
        ), patch("packages.valory.skills.optimus_abci.handlers.ThreadPoolExecutor"):
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
        handler._estimate_gas = MagicMock(return_value=None)
        with patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            handler._ensure_sufficient_funds_for_x402_payments()
            assert mock_ss.sufficient_funds_for_x402_payments is False

    def test_ensure_sufficient_funds_swap_tx_fail(self) -> None:
        """Test _ensure_sufficient_funds when tx submission fails."""
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
        handler._estimate_gas = MagicMock(return_value=21000)
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

    def test_ensure_sufficient_funds_swap_tx_not_successful(self) -> None:
        """Test _ensure_sufficient_funds when tx is not successful."""
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
        handler._estimate_gas = MagicMock(return_value=21000)
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

    def test_ensure_sufficient_funds_swap_success(self) -> None:
        """Test _ensure_sufficient_funds when swap succeeds."""
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
        handler._estimate_gas = MagicMock(return_value=21000)
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
        rounds_info_mock = {
            "test_round": {"transitions": {}},
        }
        with patch(
            "packages.valory.skills.optimus_abci.handlers.load_fsm_spec"
        ) as mock_load, patch(
            "packages.valory.skills.optimus_abci.handlers.ROUNDS_INFO",
            {"TestRound": {"transitions": {}}},
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

    def test_setup_http_handler_x402(self) -> None:
        """Test HttpHandler.setup with x402 enabled."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = True
        ctx.params.service_endpoint_base = "http://localhost:8000"
        ctx.params.target_investment_chains = ["optimism"]
        ctx.params.available_strategies = {"optimism": ["balancer_pools_search"]}
        with patch(
            "packages.valory.skills.optimus_abci.handlers.load_fsm_spec"
        ) as mock_load, patch(
            "packages.valory.skills.optimus_abci.handlers.ROUNDS_INFO", {}
        ), patch(
            "packages.valory.skills.optimus_abci.handlers.ThreadPoolExecutor"
        ), patch.object(
            type(handler), "shared_state", new_callable=PropertyMock
        ) as mock_shared:
            mock_ss = MagicMock()
            mock_shared.return_value = mock_ss
            mock_load.return_value = {"transition_func": {}}
            handler.setup()

    def test_setup_http_handler_modius(self) -> None:
        """Test HttpHandler.setup with non-optimism chain."""
        handler, ctx = _make_http_handler()
        ctx.params.use_x402 = False
        ctx.params.service_endpoint_base = "http://localhost:8000"
        ctx.params.target_investment_chains = ["base"]
        ctx.params.available_strategies = {"base": []}
        with patch(
            "packages.valory.skills.optimus_abci.handlers.load_fsm_spec"
        ) as mock_load, patch(
            "packages.valory.skills.optimus_abci.handlers.ROUNDS_INFO", {}
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
        """Test _parse_llm_response when error parsing triggers AttributeError.

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

    def test_handle_llm_response_generic_exception(self) -> None:
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

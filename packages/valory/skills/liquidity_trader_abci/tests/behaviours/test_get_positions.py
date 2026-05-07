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

"""Tests for behaviours/get_positions.py."""

# pylint: skip-file

import json
from unittest.mock import MagicMock

from packages.valory.skills.liquidity_trader_abci.behaviours.get_positions import (
    GetPositionsBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.states.get_positions import (
    GetPositionsRound,
)


def _gen_return_false(*args, **kwargs):
    """Generator function that yields once and returns False."""
    yield
    return False


def _make_behaviour():
    """Create a GetPositionsBehaviour without __init__."""
    obj = object.__new__(GetPositionsBehaviour)
    ctx = MagicMock()
    obj.__dict__["_context"] = ctx
    obj._read_investing_paused = _gen_return_false
    return obj


def _drive(gen, sends=None):
    """Drive a generator to completion, sending values from *sends*."""
    sends = sends or []
    idx = 0
    val = None
    while True:
        try:
            val = gen.send(val)
            if idx < len(sends):
                val = sends[idx]
                idx += 1
        except StopIteration as exc:
            return exc.value


class TestGetPositionsBehaviour:
    """Tests for GetPositionsBehaviour."""

    def test_async_act_with_positions(self) -> None:
        """Test async_act when positions are returned."""
        obj = _make_behaviour()
        obj.current_positions = [{"pool": "0x123"}]

        benchmark_mock = MagicMock()
        obj.context.benchmark_tool.measure.return_value = benchmark_mock
        obj.context.agent_address = "0xagent"

        positions_data = [{"pool": "0x123", "value": 100}]

        # Stub generator methods
        def fake_get_positions():
            """Fake get positions."""
            yield
            return positions_data

        def fake_adjust(*args, **kwargs):
            """Fake adjust."""
            yield
            return None

        def fake_send(*args, **kwargs):
            """Fake send."""
            yield

        def fake_wait(*args, **kwargs):
            """Fake wait."""
            yield

        obj.get_positions = fake_get_positions
        obj._adjust_current_positions_for_backward_compatibility = fake_adjust
        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()

        gen = obj.async_act()
        _drive(gen)

        obj.set_done.assert_called_once()

    def test_async_act_with_none_positions(self) -> None:
        """Test async_act when positions are None (uses ERROR_PAYLOAD)."""
        obj = _make_behaviour()
        obj.current_positions = None

        benchmark_mock = MagicMock()
        obj.context.benchmark_tool.measure.return_value = benchmark_mock
        obj.context.agent_address = "0xagent"

        def fake_get_positions():
            """Fake get positions."""
            yield
            return None

        def fake_adjust(*args, **kwargs):
            """Fake adjust."""
            yield
            return None

        captured_payloads = []

        def fake_send(payload):
            """Fake send."""
            captured_payloads.append(payload)
            yield

        def fake_wait(*args, **kwargs):
            """Fake wait."""
            yield

        obj.get_positions = fake_get_positions
        obj._adjust_current_positions_for_backward_compatibility = fake_adjust
        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()

        gen = obj.async_act()
        _drive(gen)

        # The payload should contain the ERROR_PAYLOAD
        assert len(captured_payloads) == 1
        payload = captured_payloads[0]
        expected = json.dumps(
            GetPositionsRound.ERROR_PAYLOAD, sort_keys=True, ensure_ascii=True
        )
        assert payload.positions == expected
        obj.set_done.assert_called_once()


class TestGetPositionsWithdrawalGate:
    """Verify the gate at the top of async_act emits a withdrawal payload."""

    def test_gate_emits_withdrawal_payload_when_paused(self) -> None:
        """investing_paused=True short-circuits to a WITHDRAWAL_INITIATED payload."""
        obj = _make_behaviour()
        obj.context.benchmark_tool.measure.return_value = MagicMock()
        obj.context.agent_address = "0xagent"

        captured = {}

        def fake_read_investing_paused():
            """Fake read investing paused."""
            yield
            return True

        def fake_send(payload):
            """Fake send."""
            captured["payload"] = payload
            yield

        def fake_wait():
            """Fake wait."""
            yield

        obj.get_positions = MagicMock(
            side_effect=AssertionError(
                "get_positions must not run when investing is paused"
            )
        )
        obj._read_investing_paused = fake_read_investing_paused
        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()

        _drive(obj.async_act())

        assert captured["payload"].event == "withdrawal_initiated"
        assert captured["payload"].positions is None
        obj.set_done.assert_called_once()

    def test_gate_falls_through_when_not_paused(self) -> None:
        """investing_paused=False lets the normal positions path emit a non-withdrawal payload."""
        obj = _make_behaviour()
        obj.current_positions = []
        obj.context.benchmark_tool.measure.return_value = MagicMock()
        obj.context.agent_address = "0xagent"

        captured = {}

        def fake_read_investing_paused():
            """Fake read investing paused."""
            yield
            return False

        def fake_get_positions():
            """Fake get positions."""
            yield
            return None

        def fake_adjust(*args, **kwargs):
            """Fake adjust."""
            yield

        def fake_send(payload):
            """Fake send."""
            captured["payload"] = payload
            yield

        def fake_wait():
            """Fake wait."""
            yield

        obj._read_investing_paused = fake_read_investing_paused
        obj.get_positions = fake_get_positions
        obj._adjust_current_positions_for_backward_compatibility = fake_adjust
        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()

        _drive(obj.async_act())

        assert captured["payload"].event is None
        assert captured["payload"].positions == json.dumps(
            GetPositionsRound.ERROR_PAYLOAD, sort_keys=True, ensure_ascii=True
        )
        obj.set_done.assert_called_once()

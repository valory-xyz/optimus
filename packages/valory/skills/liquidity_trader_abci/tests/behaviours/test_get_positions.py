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
from contextlib import contextmanager
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.skills.liquidity_trader_abci.behaviours.get_positions import (
    GetPositionsBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.states.get_positions import (
    GetPositionsRound,
)


def _make_behaviour():
    """Create a GetPositionsBehaviour without __init__."""
    obj = object.__new__(GetPositionsBehaviour)
    ctx = MagicMock()
    obj.__dict__["_context"] = ctx
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

    def test_matching_round(self) -> None:
        """Test matching_round is GetPositionsRound."""
        assert GetPositionsBehaviour.matching_round is GetPositionsRound

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
            yield
            return positions_data

        def fake_adjust(*args, **kwargs):
            yield
            return None

        def fake_send(*args, **kwargs):
            yield

        def fake_wait(*args, **kwargs):
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
            yield
            return None

        def fake_adjust(*args, **kwargs):
            yield
            return None

        captured_payloads = []

        def fake_send(payload):
            captured_payloads.append(payload)
            yield

        def fake_wait(*args, **kwargs):
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

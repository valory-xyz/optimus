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

"""Test the states/check_mech_pre_deposit.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from dataclasses import fields
from typing import Any, Optional, Tuple
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.skills.abstract_round_abci.base import (
    BaseTxPayload,
    CollectSameUntilThresholdRound,
)
from packages.valory.skills.liquidity_trader_abci.payloads import (
    CheckMechPreDepositPayload,
)
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)
from packages.valory.skills.liquidity_trader_abci.states.check_mech_pre_deposit import (
    CheckMechPreDepositRound,
)

_TX_HASH = "0x" + "ab" * 32


def test_import() -> None:
    """Test that the check_mech_pre_deposit module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.states.check_mech_pre_deposit  # noqa


def _payload_values_with_event(event_value: Optional[str]) -> Tuple[Any, ...]:
    """Build a payload-values tuple in declaration order with ``event`` set.

    Mirrors ``BaseTxPayload.values`` (drops the base fields) and resolves
    ``event``'s position from ``fields()``, so the test stays correct if the
    payload's field order changes.

    :param event_value: the value to write into the ``event`` slot.
    :return: payload-values tuple with ``event_value`` at the ``event`` slot.
    """
    base = {field.name for field in fields(BaseTxPayload)}
    non_base = [
        field.name
        for field in fields(CheckMechPreDepositPayload)
        if field.name not in base
    ]
    values: list = [None] * len(non_base)
    values[non_base.index("event")] = event_value
    return tuple(values)


def _stub_round(
    threshold: bool = False, payload_values: Tuple[Any, ...] = ()
) -> CheckMechPreDepositRound:
    """Build a minimally-stubbed round bypassing ``__init__``.

    :param threshold: whether consensus has been reached.
    :param payload_values: the consensus payload's non-base values.
    :return: the round under test.
    """
    round_obj = object.__new__(CheckMechPreDepositRound)
    type(round_obj).threshold_reached = PropertyMock(return_value=threshold)
    type(round_obj).most_voted_payload_values = PropertyMock(
        return_value=payload_values
    )
    return round_obj


class TestCheckMechPreDepositRound:
    """Test CheckMechPreDepositRound.end_block."""

    def test_event_field_is_last_so_selection_key_zip_truncates(self) -> None:
        """``event`` is not in selection_key; zip only drops it while it sits last."""
        base = {field.name for field in fields(BaseTxPayload)}
        non_base = [
            field.name
            for field in fields(CheckMechPreDepositPayload)
            if field.name not in base
        ]
        assert non_base[-1] == "event"
        assert len(CheckMechPreDepositRound.selection_key) == len(non_base) - 1

    def test_withdrawal_initiated_short_circuits_before_super(self) -> None:
        """A tagged withdrawal leaves the cycle without consulting super()."""
        round_obj = _stub_round(
            threshold=True,
            payload_values=_payload_values_with_event(Event.WITHDRAWAL_INITIATED.value),
        )
        synced = MagicMock(spec=SynchronizedData)
        type(round_obj).synchronized_data = PropertyMock(return_value=synced)

        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            side_effect=AssertionError("super should not be reached on withdrawal"),
        ):
            result = round_obj.end_block()

        assert result == (synced, Event.WITHDRAWAL_INITIATED)

    def test_an_unknown_event_string_falls_through_to_super(self) -> None:
        """Only the exact withdrawal event short-circuits; anything else does not."""
        round_obj = _stub_round(
            threshold=True, payload_values=_payload_values_with_event("withdrawal_init")
        )
        synced = MagicMock(spec=SynchronizedData)
        synced.most_voted_tx_hash = None

        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(synced, Event.DONE),
        ):
            result = round_obj.end_block()

        assert result == (synced, Event.DONE)

    def test_no_consensus_yet_returns_none(self) -> None:
        """Without a result from super() the round is not over."""
        round_obj = _stub_round()
        with patch.object(
            CollectSameUntilThresholdRound, "end_block", return_value=None
        ):
            assert round_obj.end_block() is None

    def test_a_tx_hash_routes_to_settlement(self) -> None:
        """A top-up was built, so it has to be settled before the cycle carries on."""
        round_obj = _stub_round()
        synced = MagicMock(spec=SynchronizedData)
        synced.most_voted_tx_hash = _TX_HASH

        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(synced, Event.DONE),
        ):
            result = round_obj.end_block()

        assert result == (synced, Event.SETTLE)

    def test_no_tx_hash_carries_the_period_on(self) -> None:
        """The pre-deposit already covers the floor, so there is nothing to settle."""
        round_obj = _stub_round()
        synced = MagicMock(spec=SynchronizedData)
        synced.most_voted_tx_hash = None

        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(synced, Event.DONE),
        ):
            result = round_obj.end_block()

        assert result == (synced, Event.DONE)

    @pytest.mark.parametrize(
        "event", [Event.NO_MAJORITY, Event.NONE, Event.ROUND_TIMEOUT]
    )
    def test_a_non_done_event_is_passed_through_unchanged(self, event: Event) -> None:
        """Only DONE is reinterpreted; a failure event keeps its own meaning.

        :param event: the event super() reports.
        """
        round_obj = _stub_round()
        synced = MagicMock(spec=SynchronizedData)
        synced.most_voted_tx_hash = _TX_HASH

        with patch.object(
            CollectSameUntilThresholdRound, "end_block", return_value=(synced, event)
        ):
            result = round_obj.end_block()

        assert result == (synced, event)

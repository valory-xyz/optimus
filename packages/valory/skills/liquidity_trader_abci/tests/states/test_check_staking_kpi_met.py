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

"""Test the states/check_staking_kpi_met.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from dataclasses import fields
from unittest.mock import MagicMock, PropertyMock, patch

from packages.valory.skills.abstract_round_abci.base import (
    BaseTxPayload,
    CollectSameUntilThresholdRound,
)
from packages.valory.skills.liquidity_trader_abci.payloads import (
    CheckStakingKPIMetPayload,
)
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)
from packages.valory.skills.liquidity_trader_abci.states.check_staking_kpi_met import (
    CheckStakingKPIMetRound,
)


def _payload_values_with_event(event_value):
    """Build a payload-values tuple in declaration order with ``event`` set.

    Mirrors ``BaseTxPayload.values`` (drops the 3 base fields). Resolves
    ``event``'s position from ``fields()`` so the test stays correct if the
    payload's field order is reordered again later.

    :param event_value: the value to write into the ``event`` slot.
    :return: payload-values tuple with ``event_value`` at the ``event`` slot
        and ``None`` everywhere else.
    """
    non_base = [
        f.name
        for f in fields(CheckStakingKPIMetPayload)
        if f.name not in {f.name for f in fields(BaseTxPayload)}
    ]
    values = [None] * len(non_base)
    values[non_base.index("event")] = event_value
    return tuple(values)


def test_import() -> None:
    """Test that the check_staking_kpi_met module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.states.check_staking_kpi_met  # noqa


class TestCheckStakingKPIMetRound:
    """Test CheckStakingKPIMetRound class."""

    def _stub_round(self, threshold: bool, payload_values=()):
        """Build a minimally-stubbed round bypassing __init__."""
        round_obj = object.__new__(CheckStakingKPIMetRound)
        type(round_obj).threshold_reached = PropertyMock(return_value=threshold)
        type(round_obj).most_voted_payload_values = PropertyMock(
            return_value=payload_values
        )
        return round_obj

    def test_end_block_withdrawal_initiated_short_circuits(self) -> None:
        """Pre-super peek returns WITHDRAWAL_INITIATED when consensus payload tags it."""
        round_obj = self._stub_round(
            threshold=True,
            payload_values=_payload_values_with_event(Event.WITHDRAWAL_INITIATED.value),
        )
        mock_synced = MagicMock(spec=SynchronizedData)
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)

        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            side_effect=AssertionError("super should not be reached on withdrawal"),
        ):
            result = round_obj.end_block()

        assert result == (mock_synced, Event.WITHDRAWAL_INITIATED)

    def test_end_block_threshold_reached_event_none_falls_through(self) -> None:
        """Threshold reached with no withdrawal event must delegate to super()."""
        round_obj = self._stub_round(
            threshold=True,
            payload_values=(None, None, None, None, None, None),
        )
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.is_staking_kpi_met = True
        mock_synced.most_voted_tx_hash = None
        mock_synced.mech_requests = "[]"
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.STAKING_KPI_MET)

    def test_end_block_none_from_super(self) -> None:
        """Test end_block returns None when super returns None."""
        round_obj = self._stub_round(threshold=False)
        with patch.object(
            CollectSameUntilThresholdRound, "end_block", return_value=None
        ):
            result = round_obj.end_block()
        assert result is None

    def test_end_block_non_done(self) -> None:
        """Test end_block returns super result when event is not DONE."""
        round_obj = self._stub_round(threshold=False)
        mock_synced = MagicMock(spec=SynchronizedData)
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.NO_MAJORITY),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.NO_MAJORITY)

    def test_end_block_kpi_none_returns_error(self) -> None:
        """Test end_block returns ERROR when is_staking_kpi_met is None."""
        round_obj = self._stub_round(threshold=False)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.is_staking_kpi_met = None
        mock_synced.most_voted_tx_hash = None
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.ERROR)

    def test_end_block_with_tx_hash_returns_settle(self) -> None:
        """Test end_block returns SETTLE when most_voted_tx_hash is set."""
        round_obj = self._stub_round(threshold=False)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.is_staking_kpi_met = True
        mock_synced.most_voted_tx_hash = "0xhash"
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.SETTLE)

    def test_end_block_kpi_met(self) -> None:
        """Test end_block returns STAKING_KPI_MET when kpi is met."""
        round_obj = self._stub_round(threshold=False)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.is_staking_kpi_met = True
        mock_synced.most_voted_tx_hash = None
        mock_synced.mech_requests = "[]"
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.STAKING_KPI_MET)

    def test_end_block_mech_request_needed(self) -> None:
        """A pending mech request routes to MECH_REQUEST_NEEDED (new regime)."""
        round_obj = self._stub_round(threshold=False)
        mock_synced = MagicMock(spec=SynchronizedData)
        # KPI not met, no vanity tx, but the producer injected one mech request.
        mock_synced.is_staking_kpi_met = False
        mock_synced.most_voted_tx_hash = None
        mock_synced.mech_requests = '[{"prompt": "p", "tool": "t", "nonce": "n"}]'
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.MECH_REQUEST_NEEDED)

    def test_end_block_kpi_not_met(self) -> None:
        """Test end_block returns STAKING_KPI_NOT_MET when kpi is not met."""
        round_obj = self._stub_round(threshold=False)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.is_staking_kpi_met = False
        mock_synced.most_voted_tx_hash = None
        mock_synced.mech_requests = "[]"
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.STAKING_KPI_NOT_MET)

    def test_end_block_empty_payload_values_falls_through(self) -> None:
        """Empty consensus tuple does not crash; falls through to super()."""
        round_obj = self._stub_round(threshold=True, payload_values=())
        mock_synced = MagicMock(spec=SynchronizedData)
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.NO_MAJORITY),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.NO_MAJORITY)

    def test_end_block_unknown_event_string_falls_through(self) -> None:
        """An unknown event string (e.g. typo) does not match and falls through."""
        round_obj = self._stub_round(
            threshold=True,
            payload_values=(None, None, None, None, True, "withdrawal_init"),
        )
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.is_staking_kpi_met = True
        mock_synced.most_voted_tx_hash = None
        mock_synced.mech_requests = "[]"
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.STAKING_KPI_MET)


def test_payload_values_align_with_selection_key() -> None:
    """Projected payload values must land under their same-named selection_key.

    CollectSameUntilThresholdRound projects consensus into the db via
    ``dict(zip(selection_key, payload.values))``. ``payload.values`` walks the
    dataclass fields in declaration order. Any field in the payload that is not
    in selection_key (notably ``event``) MUST sit at the end of the declaration
    so the trailing zip-truncation drops it cleanly; otherwise every later
    selection_key entry inherits the wrong value, and the
    ``MECH_REQUEST_NEEDED`` branch silently sees ``mech_requests=None`` even
    when the producer built a real request.
    """
    mech_json = '[{"prompt": "p", "tool": "t", "nonce": "n"}]'
    payload = CheckStakingKPIMetPayload(
        sender="0xagent",
        tx_submitter="kpi_round",
        tx_hash=None,
        safe_contract_address="0xsafe",
        chain_id="base",
        is_staking_kpi_met=False,
        mech_requests=mech_json,
        is_activity_target_met=False,
        activity_target=2,
        activity_completed=0,
    )

    projected = dict(zip(CheckStakingKPIMetRound.selection_key, payload.values))

    assert projected["mech_requests"] == mech_json
    assert projected["is_activity_target_met"] is False
    assert projected["activity_target"] == 2
    assert projected["activity_completed"] == 0


def test_end_block_emits_mech_request_needed_via_real_payload_projection() -> None:
    """A real payload projected via the round's selection_key emits MECH_REQUEST_NEEDED.

    The earlier tests in this module set ``mock_synced.mech_requests`` directly,
    which skips the payload->db projection where the production bug actually
    lived. This test populates the SynchronizedData stand-in from the same
    ``dict(zip(selection_key, payload.values))`` the parent class uses, so a
    future field-order drift between payload and selection_key resurfaces here
    as a failure rather than as a silent ``STAKING_KPI_NOT_MET`` regression.
    """
    mech_json = '[{"prompt": "p", "tool": "t", "nonce": "n"}]'
    payload = CheckStakingKPIMetPayload(
        sender="0xagent",
        tx_submitter="kpi_round",
        tx_hash=None,
        safe_contract_address="0xsafe",
        chain_id="base",
        is_staking_kpi_met=False,
        mech_requests=mech_json,
        is_activity_target_met=False,
        activity_target=2,
        activity_completed=0,
    )
    projected = dict(zip(CheckStakingKPIMetRound.selection_key, payload.values))

    synced = MagicMock(spec=SynchronizedData)
    for name, value in projected.items():
        setattr(synced, name, value)

    round_obj = object.__new__(CheckStakingKPIMetRound)
    type(round_obj).threshold_reached = PropertyMock(return_value=False)
    type(round_obj).synchronized_data = PropertyMock(return_value=synced)

    with patch.object(
        CollectSameUntilThresholdRound,
        "end_block",
        return_value=(synced, Event.DONE),
    ):
        result = round_obj.end_block()

    assert result is not None
    _, event = result
    assert event == Event.MECH_REQUEST_NEEDED

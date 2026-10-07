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

"""Tests for behaviours/check_staking_kpi_met.py."""

# pylint: skip-file

import json
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.skills.funds_manager.behaviours import (
    GET_FUNDS_STATUS_METHOD_NAME,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.base import ZERO_ADDRESS
from packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met import (
    CheckStakingKPIMetBehaviour,
)

_AGENT_EOA = "0xagent"
_OPTIMISM_CHAIN_ID = 10


def _funds_status_hook_returning(response_body):
    """Build a fake shared_state hook that returns a fund-status response body."""

    class _FakeFundRequirements:
        def get_response_body(self):
            return response_body

    return lambda: _FakeFundRequirements()


def _gas_records(*costs_wei):
    """Build gas-cost tracker records where each entry has product == cost_wei."""
    return [
        {"gas_used": cost, "gas_price": 1, "timestamp": idx, "tx_hash": f"0x{idx:x}"}
        for idx, cost in enumerate(costs_wei)
    ]


def _balance_response(chain, balance_wei):
    """Build a flattened funds-status response with the EOA balance set."""
    return {
        chain: {
            _AGENT_EOA: {
                ZERO_ADDRESS: {
                    "balance": str(balance_wei),
                    "deficit": "0",
                    "decimals": 18,
                }
            }
        }
    }


def _gen_return_false(*args, **kwargs):
    """Generator function that yields once and returns False."""
    yield
    return False


def _make_behaviour():
    """Create a CheckStakingKPIMetBehaviour without __init__.

    Mirrors production state where the base behaviour __init__ always
    sets ``gas_cost_tracker``. Default is an empty tracker so the
    vanity-tx funding gate falls open (signal unknown).

    :return: a ``CheckStakingKPIMetBehaviour`` instance with a mocked context.
    """
    obj = object.__new__(CheckStakingKPIMetBehaviour)
    ctx = MagicMock()
    obj.__dict__["_context"] = ctx
    obj._read_investing_paused = _gen_return_false
    obj.gas_cost_tracker = MagicMock()
    obj.gas_cost_tracker.data = {}
    # Goal tracking is covered by TestActivityGoalTracking; keep it out of the
    # staking-path tests, which do not stub its contract and KV reads.
    obj._track_activity_goal = _gen_value(None)
    return obj


def _drive(gen):
    """Drive a generator to completion."""
    val = None
    while True:
        try:
            val = gen.send(val)
        except StopIteration as exc:
            return exc.value


def _gen_value(value):
    """Build a generator-function stub that yields once then returns ``value``."""

    def _gen(*args, **kwargs):
        yield
        return value

    return _gen


def _stub_staking_reads(obj, *, regime=False):
    """Stub the regime read the staked path performs.

    Whenever the service is staked, ``async_act`` resolves the staking regime
    (``_is_new_staking_regime``) to compute the activity-target status before the
    vanity/mech branch. Tests that only stub ``_is_staking_kpi_met`` would
    otherwise hit a real read or a truthy ``MagicMock`` regime; stub it to a
    deterministic value. (The nonce delta now comes from ``_is_staking_kpi_met``'s
    tuple, so no separate nonce stub is needed.)

    :param obj: the behaviour under test.
    :param regime: value returned by ``_is_new_staking_regime``.
    """
    obj._is_new_staking_regime = _gen_value(regime)


def _call_signal(obj, params_mock, chain="optimism"):
    """Call ``_real_tx_cost_vs_balance`` directly with ``params`` patched.

    Lets the funding-signal branches be unit-tested without driving the whole
    round, which is what the 100%-coverage gate needs for the ~8 return paths.

    :param obj: the behaviour under test (mocked context already attached).
    :param params_mock: object returned by the patched ``params`` property.
    :param chain: chain name passed to ``_real_tx_cost_vs_balance``.
    :return: the ``_FundingSignal`` (or ``None``) returned by the method.
    """
    with patch.object(
        type(obj), "params", new_callable=PropertyMock, return_value=params_mock
    ):
        return obj._real_tx_cost_vs_balance(chain)


def _run_async_act(obj, params_mock, synced_mock):
    """Drive ``async_act`` to completion with params/synchronized_data patched.

    Shared by every class that exercises the full round so a future change to
    one test class's fixtures cannot silently break another that borrowed it.

    :param obj: the behaviour under test (mocked context already attached).
    :param params_mock: object returned by the patched ``params`` property.
    :param synced_mock: object returned by the patched ``synchronized_data``.
    """
    with (
        patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ),
        patch.object(
            type(obj),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ),
    ):
        benchmark_mock = MagicMock()
        obj.context.benchmark_tool.measure.return_value = benchmark_mock
        obj.context.agent_address = "0xagent"

        def fake_send(*args, **kwargs):
            """Fake send."""
            yield

        def fake_wait(*args, **kwargs):
            """Fake wait."""
            yield

        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()

        gen = obj.async_act()
        _drive(gen)


class TestCheckStakingKPIMetBehaviour:
    """Tests for CheckStakingKPIMetBehaviour."""

    def test_async_act_kpi_undetermined(self) -> None:
        """An undetermined verdict (None, None) skips quietly at INFO, no tx."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        synced_mock = MagicMock()

        def fake_is_kpi_met():
            """Fake is kpi met."""
            yield
            return None, None

        obj._is_staking_kpi_met = fake_is_kpi_met
        _run_async_act(obj, params_mock, synced_mock)
        obj.context.logger.info.assert_called()
        obj.set_done.assert_called_once()

    def test_async_act_kpi_already_met(self) -> None:
        """Test async_act when KPI is already met (True)."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        synced_mock = MagicMock()

        def fake_is_kpi_met():
            """Fake is kpi met."""
            yield
            return True, 10

        obj._is_staking_kpi_met = fake_is_kpi_met
        _stub_staking_reads(obj)
        _run_async_act(obj, params_mock, synced_mock)
        obj.set_done.assert_called_once()

    def test_async_act_kpi_not_met_threshold_not_exceeded(self) -> None:
        """Test async_act when KPI is not met and threshold not exceeded."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        params_mock.staking_threshold_period = 10

        synced_mock = MagicMock()
        synced_mock.period_count = 5
        synced_mock.period_number_at_last_cp = 0

        def fake_is_kpi_met():
            """Fake is kpi met."""
            yield
            return False, 0

        obj._is_staking_kpi_met = fake_is_kpi_met
        _stub_staking_reads(obj)
        _run_async_act(obj, params_mock, synced_mock)
        obj.set_done.assert_called_once()

    def test_async_act_kpi_not_met_vanity_tx(self) -> None:
        """Test async_act when KPI not met, threshold exceeded, vanity tx needed."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        params_mock.staking_threshold_period = 10

        synced_mock = MagicMock()
        synced_mock.period_count = 20
        synced_mock.period_number_at_last_cp = 0
        synced_mock.min_num_of_safe_tx_required = 5

        def fake_is_kpi_met():
            """Fake is kpi met (verdict + nonce delta from one read)."""
            yield
            return False, 2

        def fake_prepare_vanity_tx(chain):
            """Fake prepare vanity tx."""
            yield
            return "0xvanity_hash"

        obj._is_staking_kpi_met = fake_is_kpi_met
        obj._is_new_staking_regime = _gen_value(False)  # old regime -> vanity path
        obj._prepare_vanity_tx = fake_prepare_vanity_tx
        _run_async_act(obj, params_mock, synced_mock)
        obj.set_done.assert_called_once()

    def test_new_regime_shortfall_is_reported_and_sends_nothing(self) -> None:
        """Nothing makes up the difference here, so it has to be visible.

        With the facilitator on, the agent's paid API calls are what move the
        counter. A day with investing paused or the deposit spent makes none,
        and settlement is batched so recent ones may not be counted yet, so a
        shortfall can be transient. Logging it at info loses the staking
        reward quietly, but it is not a misconfiguration.
        """
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        params_mock.staking_threshold_period = 10
        params_mock.activity_target = 5
        params_mock.use_mech_facilitator = True

        synced_mock = MagicMock()
        synced_mock.period_count = 20
        synced_mock.period_number_at_last_cp = 0
        synced_mock.min_num_of_safe_tx_required = 5

        def fake_is_kpi_met():
            """Short by three."""
            yield
            return False, 2

        obj._is_staking_kpi_met = fake_is_kpi_met
        obj._is_new_staking_regime = _gen_value(True)
        obj._prepare_vanity_tx = MagicMock(
            side_effect=AssertionError("sent a transaction on the new regime")
        )

        _run_async_act(obj, params_mock, synced_mock)

        obj.set_done.assert_called_once()
        warned = " ".join(
            str(call) for call in obj.context.logger.warning.call_args_list
        )
        assert "short by 3" in warned, warned
        obj.context.logger.error.assert_not_called()

    def test_new_regime_without_the_facilitator_cannot_meet_the_kpi(self) -> None:
        """This combination can never tick the counter, so it is an error.

        On this regime the counter is marketplace requests, and the agent only
        makes those on the facilitator route. With the route off the shortfall
        never recovers however well the agent runs, and the service is evicted
        at the liveness window, so it has to be louder than a quiet day.
        """
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        params_mock.staking_threshold_period = 10
        params_mock.activity_target = 5
        params_mock.use_mech_facilitator = False

        synced_mock = MagicMock()
        synced_mock.period_count = 20
        synced_mock.period_number_at_last_cp = 0
        synced_mock.min_num_of_safe_tx_required = 5

        def fake_is_kpi_met():
            """Short by three."""
            yield
            return False, 2

        obj._is_staking_kpi_met = fake_is_kpi_met
        obj._is_new_staking_regime = _gen_value(True)
        obj._prepare_vanity_tx = MagicMock(
            side_effect=AssertionError("sent a transaction on the new regime")
        )

        _run_async_act(obj, params_mock, synced_mock)

        obj.set_done.assert_called_once()
        logged = " ".join(str(call) for call in obj.context.logger.error.call_args_list)
        assert "short by 3" in logged, logged
        assert "use_mech_facilitator" in logged, logged

    def test_async_act_kpi_not_met_no_tx_needed(self) -> None:
        """Test async_act when tx left is 0."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        params_mock.staking_threshold_period = 10

        synced_mock = MagicMock()
        synced_mock.period_count = 20
        synced_mock.period_number_at_last_cp = 0
        synced_mock.min_num_of_safe_tx_required = 5

        def fake_is_kpi_met():
            """Fake is kpi met (delta already exceeds the requirement)."""
            yield
            return False, 10  # 5 - 10 = -5

        obj._is_staking_kpi_met = fake_is_kpi_met
        obj._is_new_staking_regime = _gen_value(False)
        _run_async_act(obj, params_mock, synced_mock)
        obj.set_done.assert_called_once()


class TestCheckStakingKPIMetWithdrawalGate:
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

        obj._is_staking_kpi_met = MagicMock(
            side_effect=AssertionError(
                "KPI lookup must not run when investing is paused"
            )
        )
        obj._track_activity_goal = MagicMock(
            side_effect=AssertionError("a withdrawal period must not be counted")
        )
        obj._read_investing_paused = fake_read_investing_paused
        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()

        _drive(obj.async_act())

        assert captured["payload"].event == "withdrawal_initiated"
        assert captured["payload"].tx_hash is None
        obj.set_done.assert_called_once()

    def test_gate_falls_through_when_not_paused(self) -> None:
        """investing_paused=False lets the normal KPI path emit a non-withdrawal payload."""
        obj = _make_behaviour()
        obj.context.benchmark_tool.measure.return_value = MagicMock()
        obj.context.agent_address = "0xagent"

        captured = {}

        def fake_read_investing_paused():
            """Fake read investing paused."""
            yield
            return False

        def fake_is_kpi_met():
            """Fake is kpi met."""
            yield
            return True, 10

        def fake_send(payload):
            """Fake send."""
            captured["payload"] = payload
            yield

        def fake_wait():
            """Fake wait."""
            yield

        obj._read_investing_paused = fake_read_investing_paused
        obj._is_staking_kpi_met = fake_is_kpi_met
        _stub_staking_reads(obj)
        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()

        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            _drive(obj.async_act())

        assert captured["payload"].event is None
        assert captured["payload"].tx_hash is None
        obj.set_done.assert_called_once()


class TestPrepareVanityTx:
    """Tests for _prepare_vanity_tx."""

    def test_success(self) -> None:
        """Test _prepare_vanity_tx success path."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_contract_interact(**kwargs):
            """Fake contract interact."""
            yield
            return "0x" + "ab" * 32

        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._prepare_vanity_tx("optimism")
            result = _drive(gen)
            assert result is not None

    def test_contract_interact_returns_none(self) -> None:
        """Test _prepare_vanity_tx when contract_interact returns None."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_contract_interact(**kwargs):
            """Fake contract interact."""
            yield
            return None

        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._prepare_vanity_tx("optimism")
            result = _drive(gen)
            assert result is None

    def test_contract_interact_exception(self) -> None:
        """Test _prepare_vanity_tx when contract_interact raises."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_contract_interact(**kwargs):
            """Fake contract interact."""
            raise ValueError("boom")
            yield  # noqa: unreachable

        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._prepare_vanity_tx("optimism")
            result = _drive(gen)
            assert result is None

    def test_hash_payload_exception(self) -> None:
        """Test _prepare_vanity_tx when hash_payload_to_hex raises."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_contract_interact(**kwargs):
            """Fake contract interact."""
            yield
            return "0x" + "ab" * 32

        obj.contract_interact = fake_contract_interact

        import packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met as mod

        original_hash = mod.hash_payload_to_hex

        def bad_hash(**kwargs):
            """Bad hash."""
            raise ValueError("bad hash")

        mod.hash_payload_to_hex = bad_hash
        try:
            with patch.object(
                type(obj), "params", new_callable=PropertyMock, return_value=params_mock
            ):
                gen = obj._prepare_vanity_tx("optimism")
                result = _drive(gen)
                assert result is None
        finally:
            mod.hash_payload_to_hex = original_hash


class TestVanityTxFundingGate:
    """Verify the funding gate at the vanity-tx decision point.

    These exercise async_act end-to-end with the KPI-not-met,
    threshold-exceeded, vanity-tx-needed shape (same setup as
    test_async_act_kpi_not_met_vanity_tx) and assert whether the
    suppression branch fires.
    """

    @staticmethod
    def _base_obj_with_kpi_unmet():
        """Build a behaviour mid-flow with the prerequisites for vanity tx in place."""
        obj = _make_behaviour()
        obj.context.agent_address = _AGENT_EOA

        def fake_is_kpi_met():
            yield
            return False, 2

        obj._is_staking_kpi_met = fake_is_kpi_met
        # Old regime: the funding gate guards the vanity-tx path.
        obj._is_new_staking_regime = _gen_value(False)
        obj.gas_cost_tracker = MagicMock()
        obj.gas_cost_tracker.data = {}
        return obj

    @staticmethod
    def _params_with_optimism_mapping():
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        params_mock.staking_threshold_period = 10
        params_mock.chain_to_chain_id_mapping = {"optimism": _OPTIMISM_CHAIN_ID}
        return params_mock

    @staticmethod
    def _synced_with_kpi_unmet():
        synced_mock = MagicMock()
        synced_mock.period_count = 20
        synced_mock.period_number_at_last_cp = 0
        synced_mock.min_num_of_safe_tx_required = 5
        return synced_mock

    def test_vanity_tx_suppressed_when_balance_below_recent_real_tx_cost(
        self,
    ) -> None:
        """Suppress the vanity tx when the EOA balance is below the real-tx cost.

        Representative values from a production incident: balance ~6.92e13 wei,
        real-tx cost ~1.55e14 wei. Guard must skip _prepare_vanity_tx and emit a
        WARNING that carries the concrete balance and cost numbers.
        """
        obj = self._base_obj_with_kpi_unmet()
        obj.gas_cost_tracker.data = {
            str(_OPTIMISM_CHAIN_ID): _gas_records(
                155_000_000_000_000,
                155_000_000_000_000,
                155_000_000_000_000,
            ),
        }
        obj.context.shared_state = {
            GET_FUNDS_STATUS_METHOD_NAME: _funds_status_hook_returning(
                _balance_response("optimism", 69_207_718_314_629)
            ),
        }

        vanity_called = {"flag": False}

        def fake_prepare_vanity_tx(chain):
            vanity_called["flag"] = True
            yield
            return "0xvanity"

        obj._prepare_vanity_tx = fake_prepare_vanity_tx

        _run_async_act(
            obj, self._params_with_optimism_mapping(), self._synced_with_kpi_unmet()
        )

        assert vanity_called["flag"] is False
        # Assert the WARNING carries the real numbers, so a regression that
        # logged zeros or swapped balance/cost would not slip through.
        warning_msg = obj.context.logger.warning.call_args.args[0]
        assert "69207718314629" in warning_msg
        assert "155000000000000" in warning_msg

    def test_vanity_tx_runs_when_balance_above_recent_real_tx_cost(self) -> None:
        """Healthy EOA: gate is silent and the existing vanity path runs."""
        obj = self._base_obj_with_kpi_unmet()
        obj.gas_cost_tracker.data = {
            str(_OPTIMISM_CHAIN_ID): _gas_records(
                100_000_000_000_000,
                100_000_000_000_000,
            ),
        }
        obj.context.shared_state = {
            GET_FUNDS_STATUS_METHOD_NAME: _funds_status_hook_returning(
                _balance_response("optimism", 500_000_000_000_000)
            ),
        }

        vanity_called = {"flag": False}

        def fake_prepare_vanity_tx(chain):
            vanity_called["flag"] = True
            yield
            return "0xvanity"

        obj._prepare_vanity_tx = fake_prepare_vanity_tx

        _run_async_act(
            obj, self._params_with_optimism_mapping(), self._synced_with_kpi_unmet()
        )

        assert vanity_called["flag"] is True

    def test_vanity_tx_runs_when_balance_equals_cost(self) -> None:
        """Fence-post: balance == cost → strict ``<`` is False → vanity runs.

        Pins the ``<`` vs ``<=`` boundary end-to-end: a regression to ``<=``
        would suppress here and fail this test.
        """
        obj = self._base_obj_with_kpi_unmet()
        obj.gas_cost_tracker.data = {
            str(_OPTIMISM_CHAIN_ID): _gas_records(200_000_000_000_000),
        }
        obj.context.shared_state = {
            GET_FUNDS_STATUS_METHOD_NAME: _funds_status_hook_returning(
                _balance_response("optimism", 200_000_000_000_000)
            ),
        }

        vanity_called = {"flag": False}

        def fake_prepare_vanity_tx(chain):
            vanity_called["flag"] = True
            yield
            return "0xvanity"

        obj._prepare_vanity_tx = fake_prepare_vanity_tx

        _run_async_act(
            obj, self._params_with_optimism_mapping(), self._synced_with_kpi_unmet()
        )

        assert vanity_called["flag"] is True
        obj.context.logger.warning.assert_not_called()

    def test_vanity_tx_runs_when_no_gas_records_yet(self) -> None:
        """Fresh boot, no real-tx history → cost signal unknown → fail open."""
        obj = self._base_obj_with_kpi_unmet()
        obj.gas_cost_tracker.data = {}
        obj.context.shared_state = {
            GET_FUNDS_STATUS_METHOD_NAME: _funds_status_hook_returning(
                _balance_response("optimism", 1)
            ),
        }

        vanity_called = {"flag": False}

        def fake_prepare_vanity_tx(chain):
            vanity_called["flag"] = True
            yield
            return "0xvanity"

        obj._prepare_vanity_tx = fake_prepare_vanity_tx

        _run_async_act(
            obj, self._params_with_optimism_mapping(), self._synced_with_kpi_unmet()
        )

        assert vanity_called["flag"] is True

    def test_vanity_tx_runs_when_funds_status_hook_missing(self) -> None:
        """funds_manager not loaded / hook absent → balance unknown → fail open."""
        obj = self._base_obj_with_kpi_unmet()
        obj.gas_cost_tracker.data = {
            str(_OPTIMISM_CHAIN_ID): _gas_records(155_000_000_000_000),
        }
        obj.context.shared_state = {}

        vanity_called = {"flag": False}

        def fake_prepare_vanity_tx(chain):
            vanity_called["flag"] = True
            yield
            return "0xvanity"

        obj._prepare_vanity_tx = fake_prepare_vanity_tx

        _run_async_act(
            obj, self._params_with_optimism_mapping(), self._synced_with_kpi_unmet()
        )

        assert vanity_called["flag"] is True


class TestRealTxCostVsBalance:
    """Direct unit tests for ``_real_tx_cost_vs_balance``.

    Covers every ``return None`` branch (the coverage gate needs each one) plus
    the success path and the ``balance == cost`` fence-post.
    """

    @staticmethod
    def _params(mapping=None):
        params_mock = MagicMock()
        params_mock.chain_to_chain_id_mapping = (
            {"optimism": _OPTIMISM_CHAIN_ID} if mapping is None else mapping
        )
        return params_mock

    @staticmethod
    def _obj_with(records=None, shared_state=None):
        obj = _make_behaviour()
        obj.context.agent_address = _AGENT_EOA
        obj.gas_cost_tracker = MagicMock()
        obj.gas_cost_tracker.data = (
            {} if records is None else {str(_OPTIMISM_CHAIN_ID): records}
        )
        obj.context.shared_state = {} if shared_state is None else shared_state
        return obj

    def _healthy_shared_state(self, balance_wei):
        return {
            GET_FUNDS_STATUS_METHOD_NAME: _funds_status_hook_returning(
                _balance_response("optimism", balance_wei)
            ),
        }

    def test_returns_none_when_chain_id_missing(self) -> None:
        """No chain_id mapping -> signal unknown."""
        obj = self._obj_with(records=_gas_records(1))
        assert _call_signal(obj, self._params(mapping={})) is None

    def test_returns_none_when_no_records(self) -> None:
        """No gas records yet -> cost unknown."""
        obj = self._obj_with(records=None)
        assert _call_signal(obj, self._params()) is None

    def test_returns_none_and_warns_on_malformed_gas_record(self) -> None:
        """A record missing keys -> warn and fail open."""
        obj = self._obj_with(records=[{"timestamp": 0, "tx_hash": "0x0"}])
        assert _call_signal(obj, self._params()) is None
        obj.context.logger.warning.assert_called_once()

    def test_returns_none_when_hook_missing(self) -> None:
        """funds-status hook not registered -> balance unknown."""
        obj = self._obj_with(records=_gas_records(1), shared_state={})
        assert _call_signal(obj, self._params()) is None

    def test_returns_none_and_warns_when_hook_raises(self) -> None:
        """Hook raising (e.g. API drift) -> warn and fail open."""

        class _Boom:
            def get_response_body(self):
                raise RuntimeError("api drift")

        obj = self._obj_with(
            records=_gas_records(1),
            shared_state={GET_FUNDS_STATUS_METHOD_NAME: lambda: _Boom()},
        )
        assert _call_signal(obj, self._params()) is None
        obj.context.logger.warning.assert_called_once()

    def test_returns_none_when_balance_absent(self) -> None:
        """Hook responds but the EOA/native balance is missing -> unknown."""
        obj = self._obj_with(
            records=_gas_records(1),
            shared_state={
                GET_FUNDS_STATUS_METHOD_NAME: _funds_status_hook_returning({})
            },
        )
        assert _call_signal(obj, self._params()) is None

    def test_returns_none_when_balance_non_numeric(self) -> None:
        """Balance present but not an int string -> unknown."""
        response = _balance_response("optimism", 0)
        response["optimism"][_AGENT_EOA][ZERO_ADDRESS]["balance"] = "not-a-number"
        obj = self._obj_with(
            records=_gas_records(1),
            shared_state={
                GET_FUNDS_STATUS_METHOD_NAME: _funds_status_hook_returning(response)
            },
        )
        assert _call_signal(obj, self._params()) is None

    def test_returns_signal_on_success(self) -> None:
        """Both signals present -> named fields carry balance and median cost."""
        obj = self._obj_with(
            records=_gas_records(100, 200, 300),
            shared_state=self._healthy_shared_state(500),
        )
        signal = _call_signal(obj, self._params())
        assert signal is not None
        assert signal.eoa_balance == 500
        assert signal.recent_real_tx_cost == 200  # median(100, 200, 300)

    def test_signal_carries_equal_values_at_boundary(self) -> None:
        """At balance == cost the method reports both as the same wei value.

        The gate's strict ``<`` policy at this boundary is pinned end-to-end by
        ``TestVanityTxFundingGate.test_vanity_tx_runs_when_balance_equals_cost``.
        """
        obj = self._obj_with(
            records=_gas_records(200),
            shared_state=self._healthy_shared_state(200),
        )
        signal = _call_signal(obj, self._params())
        assert signal is not None
        assert signal.eoa_balance == 200
        assert signal.recent_real_tx_cost == 200


class TestTheNewRegimeSendsNoActivityTx:
    """On the new regime the counter is mech-marketplace requests.

    The agent's paid CoinGecko and chat calls already are marketplace
    requests, so nothing is owed here. What must not happen is falling
    through to the vanity Safe tx, which ticks Safe nonces: the old
    regime's counter, not this one, so it would cost gas for nothing.
    """

    @staticmethod
    def _behaviour(regime: Any):
        obj = _make_behaviour()
        obj.context.agent_address = _AGENT_EOA

        def fake_is_kpi_met():
            yield
            return False, 2

        obj._is_staking_kpi_met = fake_is_kpi_met
        obj._is_new_staking_regime = _gen_value(regime)
        obj.gas_cost_tracker = MagicMock()
        obj.gas_cost_tracker.data = {}
        obj.context.shared_state = {}
        return obj

    @staticmethod
    def _params():
        params = MagicMock()
        params.staking_chain = "optimism"
        params.safe_contract_addresses = {"optimism": "0xsafe"}
        params.staking_threshold_period = 10
        params.chain_to_chain_id_mapping = {"optimism": _OPTIMISM_CHAIN_ID}
        # The new regime reads this to report progress on /healthcheck.
        params.activity_target = 5
        return params

    @staticmethod
    def _synced():
        synced = MagicMock()
        synced.period_count = 20
        synced.period_number_at_last_cp = 0
        synced.min_num_of_safe_tx_required = 5
        return synced

    def _drive(self, regime: Any) -> dict:
        obj = self._behaviour(regime)
        called = {"vanity": False}

        def fake_prepare_vanity_tx(chain: Any) -> Any:
            called["vanity"] = True
            yield
            return "0xvanity"

        obj._prepare_vanity_tx = fake_prepare_vanity_tx
        _run_async_act(obj, self._params(), self._synced())
        return called

    def test_the_new_regime_does_not_fall_through_to_the_vanity_safe_tx(self) -> None:
        """That tx ticks Safe nonces, which this regime does not count."""
        assert self._drive(True)["vanity"] is False

    def test_an_undetermined_regime_sends_nothing_either(self) -> None:
        """Guessing wrong here means gas spent on the counter nobody reads."""
        assert self._drive(None)["vanity"] is False

    def test_the_old_regime_still_sends_its_vanity_safe_tx(self) -> None:
        """Its counter is Safe nonces, which the facilitator never moves."""
        assert self._drive(False)["vanity"] is True


EPOCH = 1_000
NOW = 1_500


class _GoalRun:
    """One call of ``_track_activity_goal`` against stubbed reads and writes."""

    def __init__(
        self,
        tmp_path: Path,
        kv: Optional[Dict[str, Optional[str]]],
        ts_checkpoint: Any = EPOCH,
        period: int = 5,
        write_ok: bool = True,
    ) -> None:
        self.obj = _make_behaviour()
        del self.obj.__dict__["_track_activity_goal"]
        self.perf_path = tmp_path / "perf.json"
        self.perf_path.write_text(json.dumps({"metrics": ["kept"]}))
        self.obj.agent_performance_filepath = str(self.perf_path)
        self.obj._get_ts_checkpoint = _gen_value(ts_checkpoint)
        self.obj._read_kv = _gen_value(kv)
        self.obj._get_current_timestamp = lambda: NOW
        self.written: Dict[str, str] = {}

        def fake_write_kv(data):
            self.written.update(data)
            yield
            return write_ok

        self.obj._write_kv = fake_write_kv
        self.params = MagicMock()
        self.params.staking_chain = "optimism"
        self.params.activity_goal_target = 10
        self.synced = MagicMock()
        self.synced.period_count = period

    def __call__(self, kpi=True, target_met=None, settling=False):
        with (
            patch.object(
                type(self.obj),
                "params",
                new_callable=PropertyMock,
                return_value=self.params,
            ),
            patch.object(
                type(self.obj),
                "synchronized_data",
                new_callable=PropertyMock,
                return_value=self.synced,
            ),
        ):
            return _drive(self.obj._track_activity_goal(kpi, target_met, settling))

    @property
    def block(self):
        return json.loads(self.perf_path.read_text()).get("activity_goal")

    @property
    def progress(self):
        return int(self.written["activity_goal_progress"])


def _kv(progress, period_start=EPOCH, last_counted=4, target=None, last_met_at=None):
    return {
        "activity_goal_target": None if target is None else str(target),
        "activity_goal_progress": str(progress),
        "activity_goal_period_start": str(period_start),
        "activity_goal_last_met_at": None if last_met_at is None else str(last_met_at),
        "activity_goal_last_counted_period": str(last_counted),
    }


class TestActivityGoalTracking:
    """Counting rounds per epoch and publishing the block."""

    def test_a_working_period_adds_exactly_one(self, tmp_path: Path) -> None:
        """A period not standing by counts once and is recorded as counted."""
        run = _GoalRun(tmp_path, _kv(3))
        assert run() is False
        assert run.written == {
            "activity_goal_progress": "4",
            "activity_goal_period_start": str(EPOCH),
            "activity_goal_last_counted_period": "5",
        }
        assert run.block == {
            "unit": "rounds",
            "target": 10,
            "progress": 4,
            "is_met": False,
            "period_start": EPOCH,
            "last_met_at": None,
            "updated_at": NOW,
        }
        assert json.loads(run.perf_path.read_text())["metrics"] == ["kept"]

    def test_a_second_run_in_the_same_period_adds_nothing(self, tmp_path: Path) -> None:
        """After a checkpoint or vanity tx the round runs again; no double count."""
        run = _GoalRun(tmp_path, _kv(3, last_counted=5))
        assert run() is False
        assert run.progress == 3

    def test_epoch_rollover_resets_progress(self, tmp_path: Path) -> None:
        """A new tsCheckpoint starts the count again from this period."""
        run = _GoalRun(tmp_path, _kv(12, period_start=EPOCH - 86400, last_counted=5))
        assert run() is False
        assert run.progress == 1
        assert run.written["activity_goal_period_start"] == str(EPOCH)

    def test_the_period_reaching_the_goal_works_and_the_next_stands_by(
        self, tmp_path: Path
    ) -> None:
        """The tenth round is counted and reported met; the eleventh stands by."""
        run = _GoalRun(tmp_path, _kv(9))
        assert run() is False
        assert run.progress == 10
        assert run.block["is_met"] is True
        assert run.block["last_met_at"] == NOW

        following = _GoalRun(
            tmp_path, _kv(10, last_counted=5, last_met_at=NOW), period=6
        )
        assert following() is True
        assert following.progress == 10
        assert "standing by" in str(following.obj.context.logger.info.call_args)

    def test_a_standby_period_adds_nothing(self, tmp_path: Path) -> None:
        """Standby periods are never counted."""
        run = _GoalRun(tmp_path, _kv(10))
        assert run(kpi=True) is True
        assert run.progress == 10

    def test_met_goal_with_unmet_staking_side_keeps_counting(
        self, tmp_path: Path
    ) -> None:
        """Without the staking side the agent works, so the period counts."""
        run = _GoalRun(tmp_path, _kv(10))
        assert run(kpi=True, target_met=False) is True
        assert run.progress == 11

    def test_a_settling_period_is_not_standby(self, tmp_path: Path) -> None:
        """A vanity tx to settle keeps priority, so the period counts."""
        run = _GoalRun(tmp_path, _kv(10))
        assert run(kpi=True, settling=True) is True
        assert run.progress == 11

    def test_a_restart_keeps_progress_in_the_same_epoch(self, tmp_path: Path) -> None:
        """period_count restarts from 0, but the stored progress carries on."""
        run = _GoalRun(tmp_path, _kv(6, last_counted=40), period=0)
        assert run() is False
        assert run.progress == 7

    def test_the_users_target_overrides_the_default(self, tmp_path: Path) -> None:
        """A goal set from the chat wins over the param."""
        run = _GoalRun(tmp_path, _kv(3, target=3))
        assert run() is True
        assert run.block["target"] == 3
        assert "activity_goal_target" not in run.written

    def test_last_met_at_is_stamped_once_per_epoch(self, tmp_path: Path) -> None:
        """A later met period keeps the first stamp of the epoch."""
        run = _GoalRun(tmp_path, _kv(11, target=10, last_met_at=EPOCH + 5))
        run(kpi=False)
        assert run.written["activity_goal_last_met_at"] == str(EPOCH + 5)

    @pytest.mark.parametrize("ts_checkpoint", [None, -1, "x"])
    def test_unreadable_epoch_changes_nothing(
        self, tmp_path: Path, ts_checkpoint
    ) -> None:
        """Without tsCheckpoint the flag is None and nothing is written."""
        run = _GoalRun(tmp_path, _kv(3), ts_checkpoint=ts_checkpoint)
        assert run() is None
        assert run.written == {}
        assert run.block is None

    def test_unreadable_kv_changes_nothing(self, tmp_path: Path) -> None:
        """An unreachable KV store leaves the state and the block alone."""
        run = _GoalRun(tmp_path, None)
        assert run() is None
        assert run.written == {}
        assert run.block is None

    def test_failed_kv_write_does_not_publish(self, tmp_path: Path) -> None:
        """Progress the KV store does not hold is not shown to Pearl."""
        run = _GoalRun(tmp_path, _kv(10), write_ok=False)
        assert run() is True
        assert run.block is None
        run.obj.context.logger.error.assert_called_once()

    def test_failed_block_write_is_logged(self, tmp_path: Path) -> None:
        """A file error is logged and the verdict still returned."""
        run = _GoalRun(tmp_path, _kv(3))
        run.obj.agent_performance_filepath = str(tmp_path / "missing" / "perf.json")
        assert run() is False
        run.obj.context.logger.error.assert_called_once()

    def test_a_fresh_install_starts_from_the_default(self, tmp_path: Path) -> None:
        """No stored state means progress 0 against the param goal."""
        empty = {key: None for key in _kv(0)}
        run = _GoalRun(tmp_path, empty)
        assert run() is False
        assert run.progress == 1
        assert run.block["target"] == 10


class TestActivityGoalInThePayload:
    """The round receives the goal verdict through the payload."""

    def test_payload_carries_the_goal_verdict(self) -> None:
        """A staked period sends what the tracker computed."""
        obj = _make_behaviour()
        obj._is_staking_kpi_met = _gen_value((True, 10))
        _stub_staking_reads(obj)
        obj._track_activity_goal = _gen_value(True)
        captured = {}

        def fake_send(payload):
            captured["payload"] = payload
            yield

        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            obj.context.benchmark_tool.measure.return_value = MagicMock()
            obj.send_a2a_transaction = fake_send
            obj.wait_until_round_end = _gen_value(None)
            obj.set_done = MagicMock()
            _drive(obj.async_act())

        assert captured["payload"].is_activity_goal_met is True

    def test_undetermined_kpi_skips_the_goal(self) -> None:
        """Without a KPI verdict the period is not counted and the flag is None."""
        obj = _make_behaviour()
        obj._is_staking_kpi_met = _gen_value((None, None))
        obj._track_activity_goal = MagicMock(
            side_effect=AssertionError("must not count an undetermined period")
        )
        captured = {}

        def fake_send(payload):
            captured["payload"] = payload
            yield

        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            obj.context.benchmark_tool.measure.return_value = MagicMock()
            obj.send_a2a_transaction = fake_send
            obj.wait_until_round_end = _gen_value(None)
            obj.set_done = MagicMock()
            _drive(obj.async_act())

        assert captured["payload"].is_activity_goal_met is None

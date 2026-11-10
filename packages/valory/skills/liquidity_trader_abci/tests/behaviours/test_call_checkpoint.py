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

"""Tests for behaviours/call_checkpoint.py."""

# pylint: skip-file

from unittest.mock import MagicMock, PropertyMock, patch

from packages.valory.skills.liquidity_trader_abci.behaviours.call_checkpoint import (
    CallCheckpointBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.states.call_checkpoint import (
    StakingState,
)


def _gen_return_false(*args, **kwargs):
    """Generator function that yields once and returns False."""
    yield
    return False


TS_CHECKPOINT = 1700000000
LIVENESS_PERIOD = 86400
SCHEDULED_NEXT = TS_CHECKPOINT + LIVENESS_PERIOD


def _gen_return(value):
    """Build a generator function that yields once and returns ``value``."""

    def method(*args, **kwargs):
        """Yield once and return ``value``."""
        yield
        return value

    return method


def _make_behaviour():
    """Create a CallCheckpointBehaviour without __init__."""
    obj = object.__new__(CallCheckpointBehaviour)
    ctx = MagicMock()
    obj.__dict__["_context"] = ctx
    obj.service_staking_state = StakingState.UNSTAKED
    obj._read_investing_paused = _gen_return_false
    return obj


def _drive(gen):
    """Drive a generator to completion."""
    val = None
    while True:
        try:
            val = gen.send(val)
        except StopIteration as exc:
            return exc.value


class TestCallCheckpointBehaviour:
    """Tests for CallCheckpointBehaviour."""

    def _run_async_act(self, obj, params_mock):
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
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
            return obj

    def test_async_act_no_staking_chain(self) -> None:
        """Test async_act when staking_chain is not set."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = None
        params_mock.safe_contract_addresses = {}
        self._run_async_act(obj, params_mock)
        assert obj.service_staking_state == StakingState.UNSTAKED
        obj.set_done.assert_called_once()

    def test_async_act_staked_min_tx_none(self) -> None:
        """Test async_act when staked but min_tx calc returns None."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        params_mock.staking_token_contract_address = "0xstaking"

        def fake_get_staking_state(chain):
            """Fake get staking state."""
            obj.service_staking_state = StakingState.STAKED
            yield

        def fake_calc_min_tx(chain, ts_checkpoint, liveness_period):
            """Fake calc min tx."""
            yield
            return None

        def fake_check_checkpoint(chain, ts_checkpoint, liveness_period):
            """Fake check checkpoint."""
            yield
            return False

        obj._get_service_staking_state = fake_get_staking_state
        obj._get_ts_checkpoint = _gen_return(TS_CHECKPOINT)
        obj._get_liveness_period = _gen_return(LIVENESS_PERIOD)
        obj._calculate_min_num_of_safe_tx_required = fake_calc_min_tx
        obj._check_if_checkpoint_reached = fake_check_checkpoint
        self._run_async_act(obj, params_mock)
        obj.set_done.assert_called_once()

    def test_async_act_staked_checkpoint_reached(self) -> None:
        """Test async_act when staked and checkpoint is reached."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        params_mock.staking_token_contract_address = "0xstaking"

        def fake_get_staking_state(chain):
            """Fake get staking state."""
            obj.service_staking_state = StakingState.STAKED
            yield

        def fake_calc_min_tx(chain, ts_checkpoint, liveness_period):
            """Fake calc min tx."""
            yield
            return 5

        def fake_check_checkpoint(chain, ts_checkpoint, liveness_period):
            """Fake check checkpoint."""
            yield
            return True

        def fake_prepare_checkpoint_tx(chain):
            """Fake prepare checkpoint tx."""
            yield
            return "0xcheckpoint_tx"

        obj._get_service_staking_state = fake_get_staking_state
        obj._get_ts_checkpoint = _gen_return(TS_CHECKPOINT)
        obj._get_liveness_period = _gen_return(LIVENESS_PERIOD)
        obj._calculate_min_num_of_safe_tx_required = fake_calc_min_tx
        obj._check_if_checkpoint_reached = fake_check_checkpoint
        obj._prepare_checkpoint_tx = fake_prepare_checkpoint_tx
        self._run_async_act(obj, params_mock)
        obj.set_done.assert_called_once()

    def test_async_act_staked_checkpoint_not_reached(self) -> None:
        """Test async_act when staked but checkpoint not reached."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_get_staking_state(chain):
            """Fake get staking state."""
            obj.service_staking_state = StakingState.STAKED
            yield

        def fake_calc_min_tx(chain, ts_checkpoint, liveness_period):
            """Fake calc min tx."""
            yield
            return 5

        def fake_check_checkpoint(chain, ts_checkpoint, liveness_period):
            """Fake check checkpoint."""
            yield
            return False

        obj._get_service_staking_state = fake_get_staking_state
        obj._get_ts_checkpoint = _gen_return(TS_CHECKPOINT)
        obj._get_liveness_period = _gen_return(LIVENESS_PERIOD)
        obj._calculate_min_num_of_safe_tx_required = fake_calc_min_tx
        obj._check_if_checkpoint_reached = fake_check_checkpoint
        self._run_async_act(obj, params_mock)
        obj.set_done.assert_called_once()

    def test_async_act_reads_schedule_once_for_both_uses(self) -> None:
        """tsCheckpoint and livenessPeriod are read once and shared."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        received = []

        def fake_get_staking_state(chain):
            """Fake get staking state."""
            obj.service_staking_state = StakingState.STAKED
            yield

        def fake_calc_min_tx(chain, ts_checkpoint, liveness_period):
            """Fake calc min tx."""
            received.append((ts_checkpoint, liveness_period))
            yield
            return 5

        def fake_check_checkpoint(chain, ts_checkpoint, liveness_period):
            """Fake check checkpoint."""
            received.append((ts_checkpoint, liveness_period))
            yield
            return False

        obj._get_service_staking_state = fake_get_staking_state
        obj._get_ts_checkpoint = MagicMock(side_effect=_gen_return(TS_CHECKPOINT))
        obj._get_liveness_period = MagicMock(side_effect=_gen_return(LIVENESS_PERIOD))
        obj._calculate_min_num_of_safe_tx_required = fake_calc_min_tx
        obj._check_if_checkpoint_reached = fake_check_checkpoint
        self._run_async_act(obj, params_mock)
        obj._get_ts_checkpoint.assert_called_once_with("optimism")
        obj._get_liveness_period.assert_called_once_with("optimism")
        assert received == [(TS_CHECKPOINT, LIVENESS_PERIOD)] * 2

    def test_async_act_evicted(self) -> None:
        """Test async_act when service is evicted."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_get_staking_state(chain):
            """Fake get staking state."""
            obj.service_staking_state = StakingState.EVICTED
            yield

        obj._get_service_staking_state = fake_get_staking_state
        self._run_async_act(obj, params_mock)
        obj.set_done.assert_called_once()

    def test_async_act_unstaked_with_chain(self) -> None:
        """Test async_act when service staking state is UNSTAKED but chain is set."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_get_staking_state(chain):
            """Fake get staking state."""
            obj.service_staking_state = StakingState.UNSTAKED
            yield

        obj._get_service_staking_state = fake_get_staking_state
        self._run_async_act(obj, params_mock)
        obj.set_done.assert_called_once()


class TestCallCheckpointWithdrawalGate:
    """Verify the gate at the top of async_act emits a withdrawal payload."""

    def _drive_with_gate(self, obj, params_mock, paused: bool):
        """Drive async_act with _read_investing_paused stubbed to ``paused``."""
        captured = {}

        def fake_read_investing_paused():
            """Fake read investing paused."""
            yield
            return paused

        def fake_send(payload):
            """Fake send."""
            captured["payload"] = payload
            yield

        def fake_wait():
            """Fake wait."""
            yield

        obj._read_investing_paused = fake_read_investing_paused
        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            obj.context.benchmark_tool.measure.return_value = MagicMock()
            obj.context.agent_address = "0xagent"
            _drive(obj.async_act())

        return captured

    def test_gate_emits_withdrawal_payload_when_paused(self) -> None:
        """investing_paused=True short-circuits to a WITHDRAWAL_INITIATED payload."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        obj._get_service_staking_state = MagicMock(
            side_effect=AssertionError(
                "staking-state lookup must not run when investing is paused"
            )
        )

        captured = self._drive_with_gate(obj, params_mock, paused=True)

        assert captured["payload"].event == "withdrawal_initiated"
        assert captured["payload"].tx_hash is None
        obj.set_done.assert_called_once()

    def test_gate_falls_through_when_not_paused(self) -> None:
        """investing_paused=False lets the normal staking-state path run."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = None
        params_mock.safe_contract_addresses = {}

        captured = self._drive_with_gate(obj, params_mock, paused=False)

        assert captured["payload"].event is None
        obj.set_done.assert_called_once()


class TestCheckIfCheckpointReached:
    """Tests for _check_if_checkpoint_reached."""

    def _check(
        self,
        next_checkpoint,
        synced_ts=SCHEDULED_NEXT,
        ts_checkpoint=TS_CHECKPOINT,
        liveness_period=LIVENESS_PERIOD,
        available_rewards=10**18,
    ):
        """Run _check_if_checkpoint_reached against faked contract reads."""
        obj = _make_behaviour()
        obj._get_next_checkpoint = _gen_return(next_checkpoint)
        obj._get_available_rewards = MagicMock(
            side_effect=_gen_return(available_rewards)
        )
        rs = MagicMock()
        rs.last_round_transition_timestamp.timestamp.return_value = float(synced_ts)
        with patch.object(
            type(obj), "round_sequence", new_callable=PropertyMock, return_value=rs
        ):
            result = _drive(
                obj._check_if_checkpoint_reached(
                    "optimism", ts_checkpoint, liveness_period
                )
            )
        return obj, result

    def test_next_checkpoint_none(self) -> None:
        """Test returns False when next_checkpoint is None."""
        _, result = self._check(None)
        assert result is False

    def test_next_checkpoint_zero(self) -> None:
        """Test returns True when next_checkpoint is 0."""
        _, result = self._check(0, available_rewards=0)
        assert result is True

    def test_next_checkpoint_in_future(self) -> None:
        """Test returns False when next_checkpoint is on schedule and in the future."""
        _, result = self._check(SCHEDULED_NEXT, synced_ts=SCHEDULED_NEXT - 1)
        assert result is False

    def test_next_checkpoint_in_past(self) -> None:
        """Test returns True when next_checkpoint is in the past."""
        obj, result = self._check(SCHEDULED_NEXT, synced_ts=SCHEDULED_NEXT + 1)
        assert result is True
        obj._get_available_rewards.assert_called_once_with("optimism")

    def test_pending_rebaseline_ahead_of_synced_ts(self) -> None:
        """A v0.4 pending rebaseline is due even when its timestamp is ahead."""
        obj, result = self._check(
            TS_CHECKPOINT + 10, synced_ts=TS_CHECKPOINT + 5, available_rewards=0
        )
        assert result is True
        obj._get_available_rewards.assert_not_called()

    def test_due_by_time_without_rewards_is_skipped(self) -> None:
        """A checkpoint due only by time is not sent with no available rewards."""
        obj, result = self._check(
            SCHEDULED_NEXT, synced_ts=SCHEDULED_NEXT, available_rewards=0
        )
        assert result is False
        obj.context.logger.info.assert_called()

    def test_due_by_time_rewards_unreadable_is_sent(self) -> None:
        """An unreadable availableRewards does not suppress the checkpoint."""
        obj, result = self._check(
            SCHEDULED_NEXT, synced_ts=SCHEDULED_NEXT, available_rewards=None
        )
        assert result is True
        obj.context.logger.warning.assert_called()

    def test_schedule_unreadable_falls_back_to_timestamp(self) -> None:
        """Without tsCheckpoint, only the timestamp rule applies and nothing is skipped."""
        obj, result = self._check(
            SCHEDULED_NEXT,
            synced_ts=SCHEDULED_NEXT,
            ts_checkpoint=None,
            available_rewards=0,
        )
        assert result is True
        obj._get_available_rewards.assert_not_called()

    def test_zero_liveness_period_falls_back_to_timestamp(self) -> None:
        """A zero livenessPeriod is treated as unreadable, not as a rebaseline."""
        obj, result = self._check(
            TS_CHECKPOINT + 10, synced_ts=TS_CHECKPOINT + 5, liveness_period=0
        )
        assert result is False
        obj._get_available_rewards.assert_not_called()

    def test_schedule_unreadable_not_due(self) -> None:
        """Without livenessPeriod, a future timestamp is not due."""
        _, result = self._check(
            TS_CHECKPOINT + 10, synced_ts=TS_CHECKPOINT + 5, liveness_period=None
        )
        assert result is False


class TestPrepareCheckpointTx:
    """Tests for _prepare_checkpoint_tx."""

    def test_happy_path(self) -> None:
        """Test _prepare_checkpoint_tx returns a hash."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_token_contract_address = "0xstaking"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_contract_interact(**kwargs):
            """Fake contract interact."""
            yield
            return b"checkpoint_data"

        def fake_prepare_safe_tx(chain, data):
            """Fake prepare safe tx."""
            yield
            return "0xsafehash"

        obj.contract_interact = fake_contract_interact
        obj._prepare_safe_tx = fake_prepare_safe_tx

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._prepare_checkpoint_tx("optimism")
            result = _drive(gen)
            assert result == "0xsafehash"


class TestPrepareSafeTx:
    """Tests for _prepare_safe_tx."""

    def test_safe_tx_hash_none(self) -> None:
        """Test returns None when safe_tx_hash is None."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        params_mock.staking_token_contract_address = "0xstaking"

        def fake_contract_interact(**kwargs):
            """Fake contract interact."""
            yield
            return None

        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._prepare_safe_tx("optimism", data=b"data")
            result = _drive(gen)
            assert result is None

    def test_safe_tx_hash_success(self) -> None:
        """Test returns hashed payload when safe_tx_hash succeeds."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.safe_contract_addresses = {"optimism": "0x" + "aa" * 20}
        params_mock.staking_token_contract_address = "0x" + "bb" * 20

        def fake_contract_interact(**kwargs):
            """Fake contract interact."""
            yield
            return "0x" + "ab" * 32

        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._prepare_safe_tx("optimism", data=b"data")
            result = _drive(gen)
            assert result is not None
            assert isinstance(result, str)

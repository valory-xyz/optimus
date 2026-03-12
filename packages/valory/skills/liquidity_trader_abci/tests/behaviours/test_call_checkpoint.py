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

import pytest

from packages.valory.skills.liquidity_trader_abci.behaviours.call_checkpoint import (
    CallCheckpointBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.states.call_checkpoint import (
    StakingState,
)


def _make_behaviour():
    """Create a CallCheckpointBehaviour without __init__."""
    obj = object.__new__(CallCheckpointBehaviour)
    ctx = MagicMock()
    obj.__dict__["_context"] = ctx
    obj.service_staking_state = StakingState.UNSTAKED
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
                yield

            def fake_wait(*args, **kwargs):
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
            obj.service_staking_state = StakingState.STAKED
            yield

        def fake_calc_min_tx(chain):
            yield
            return None

        def fake_check_checkpoint(chain):
            yield
            return False

        obj._get_service_staking_state = fake_get_staking_state
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
            obj.service_staking_state = StakingState.STAKED
            yield

        def fake_calc_min_tx(chain):
            yield
            return 5

        def fake_check_checkpoint(chain):
            yield
            return True

        def fake_prepare_checkpoint_tx(chain):
            yield
            return "0xcheckpoint_tx"

        obj._get_service_staking_state = fake_get_staking_state
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
            obj.service_staking_state = StakingState.STAKED
            yield

        def fake_calc_min_tx(chain):
            yield
            return 5

        def fake_check_checkpoint(chain):
            yield
            return False

        obj._get_service_staking_state = fake_get_staking_state
        obj._calculate_min_num_of_safe_tx_required = fake_calc_min_tx
        obj._check_if_checkpoint_reached = fake_check_checkpoint
        self._run_async_act(obj, params_mock)
        obj.set_done.assert_called_once()

    def test_async_act_evicted(self) -> None:
        """Test async_act when service is evicted."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_get_staking_state(chain):
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
            obj.service_staking_state = StakingState.UNSTAKED
            yield

        obj._get_service_staking_state = fake_get_staking_state
        self._run_async_act(obj, params_mock)
        obj.set_done.assert_called_once()


class TestCheckIfCheckpointReached:
    """Tests for _check_if_checkpoint_reached."""

    def test_next_checkpoint_none(self) -> None:
        """Test returns False when next_checkpoint is None."""
        obj = _make_behaviour()

        def fake_get_next_checkpoint(chain):
            yield
            return None

        obj._get_next_checkpoint = fake_get_next_checkpoint
        gen = obj._check_if_checkpoint_reached("optimism")
        result = _drive(gen)
        assert result is False

    def test_next_checkpoint_zero(self) -> None:
        """Test returns True when next_checkpoint is 0."""
        obj = _make_behaviour()

        def fake_get_next_checkpoint(chain):
            yield
            return 0

        obj._get_next_checkpoint = fake_get_next_checkpoint
        gen = obj._check_if_checkpoint_reached("optimism")
        result = _drive(gen)
        assert result is True

    def test_next_checkpoint_in_future(self) -> None:
        """Test returns False when next_checkpoint is in the future."""
        obj = _make_behaviour()

        def fake_get_next_checkpoint(chain):
            yield
            return 9999999999

        obj._get_next_checkpoint = fake_get_next_checkpoint

        rs = MagicMock()
        rs.last_round_transition_timestamp.timestamp.return_value = 1700000000.0
        with patch.object(
            type(obj), "round_sequence", new_callable=PropertyMock, return_value=rs
        ):
            gen = obj._check_if_checkpoint_reached("optimism")
            result = _drive(gen)
            assert result is False

    def test_next_checkpoint_in_past(self) -> None:
        """Test returns True when next_checkpoint is in the past."""
        obj = _make_behaviour()

        def fake_get_next_checkpoint(chain):
            yield
            return 1600000000

        obj._get_next_checkpoint = fake_get_next_checkpoint

        rs = MagicMock()
        rs.last_round_transition_timestamp.timestamp.return_value = 1700000000.0
        with patch.object(
            type(obj), "round_sequence", new_callable=PropertyMock, return_value=rs
        ):
            gen = obj._check_if_checkpoint_reached("optimism")
            result = _drive(gen)
            assert result is True


class TestPrepareCheckpointTx:
    """Tests for _prepare_checkpoint_tx."""

    def test_happy_path(self) -> None:
        """Test _prepare_checkpoint_tx returns a hash."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_token_contract_address = "0xstaking"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_contract_interact(**kwargs):
            yield
            return b"checkpoint_data"

        def fake_prepare_safe_tx(chain, data):
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

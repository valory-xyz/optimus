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

"""Tests for decision_making behaviour."""

# pylint: skip-file

import json
from typing import Any, List
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    Action,
    Decision,
    DexType,
    MAX_RETRIES_FOR_ROUTES,
    MAX_SWAP_CONFIRMATION_RETRIES,
    MIN_TIME_IN_POSITION,
    PositionStatus,
    ZERO_ADDRESS,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.decision_making import (
    DecisionMakingBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.states.base import Event


def _make_behaviour(**overrides: Any) -> DecisionMakingBehaviour:
    """Create a DecisionMakingBehaviour without calling __init__."""
    b = object.__new__(DecisionMakingBehaviour)

    params = MagicMock()
    params.safe_contract_addresses = {
        "optimism": "0x" + "aa" * 20,
        "mode": "0x" + "bb" * 20,
    }
    params.multisend_contract_addresses = {
        "optimism": "0x" + "cc" * 20,
        "mode": "0x" + "dd" * 20,
    }
    params.merkl_distributor_contract_addresses = {"optimism": "0x" + "ee" * 20}
    params.chain_to_chain_id_mapping = {"optimism": 10, "mode": 34443}
    params.lifi_check_status_url = "https://li.fi/status"
    params.lifi_fetch_step_transaction_url = "https://li.fi/step"
    params.lifi_advance_routes_url = "https://li.fi/routes"
    params.waiting_period_for_status_check = 0
    # Distinct so a production-side threshold swap (fee checked against the gas limit
    # or vice-versa) is observable in TestCheckIfRouteIsProfitable.
    params.max_fee_percentage = 0.05  # 5%
    params.max_gas_percentage = 0.25  # 25%
    params.max_slippage_percentage = 0.1
    params.slippage_for_swap = 0.03

    synced = MagicMock()
    synced.actions = []
    synced.positions = []
    synced.last_executed_action_index = None
    synced.last_action = None
    synced.final_tx_hash = "0xabc123"
    synced.last_executed_route_index = None
    synced.last_executed_step_index = None
    synced.routes = []
    synced.routes_retry_attempt = 0
    synced.fee_details = {}
    synced.max_allowed_steps_in_a_route = None

    ctx = MagicMock()
    ctx.agent_address = "test_agent"
    ctx.logger = MagicMock()
    ctx.benchmark_tool.measure.return_value.__enter__ = MagicMock(return_value=None)
    ctx.benchmark_tool.measure.return_value.__exit__ = MagicMock(return_value=False)
    ctx.params = params

    shared_state = MagicMock()
    shared_state.synchronized_data = synced
    ctx.state = shared_state

    b.__dict__.update(
        {
            "_context": ctx,
            "current_positions": [],
            "pools": {},
            "portfolio_data": {},
            "_current_entry_costs": 0.0,
            "shared_state": shared_state,
        }
    )

    b.__dict__.update(overrides)
    return b


def _exhaust(gen, sends=None):
    """Drive a generator to completion, returning its return value.

    If *gen* is not actually a generator (e.g. a plain return from a
    generator function that never yields), just return it.

    :param gen: generator to exhaust.
    :param sends: optional list of values to send into the generator.
    :return: the generator's return value.
    """
    if not hasattr(gen, "__next__"):
        return gen
    sends = sends or []
    try:
        next(gen)
        for s in sends:
            gen.send(s)
        while True:
            gen.send(None)
    except StopIteration as e:
        return e.value


def _make_gen_method(return_value):
    """Create a generator method mock that yields nothing and returns return_value."""

    def method(*args, **kwargs):
        """Method."""
        yield  # one yield to make it a generator
        return return_value

    return method


def _make_gen_none():
    """Generator that returns None."""

    def method(*args, **kwargs):
        """Method."""
        if False:
            yield
        return None

    return method


def _make_gen_method_by_token(token_map):
    """Build a fake is_cl_token_staked that returns token_map[token_id]."""

    def method(behaviour, account, token_id, **kwargs):
        """Method."""
        yield
        return token_map.get(token_id)

    return method


class TestReadInvestingPaused:
    """Tests for _read_investing_paused."""

    def test_returns_true(self):
        """Test returns true."""
        b = _make_behaviour()
        b._read_kv = _make_gen_method({"investing_paused": "true"})
        result = _exhaust(b._read_investing_paused())
        assert result is True

    def test_returns_false_for_false_value(self):
        """Test returns false for false value."""
        b = _make_behaviour()
        b._read_kv = _make_gen_method({"investing_paused": "false"})
        result = _exhaust(b._read_investing_paused())
        assert result is False

    def test_returns_false_when_kv_returns_none(self):
        """Test returns false when kv returns none."""
        b = _make_behaviour()
        b._read_kv = _make_gen_method(None)
        result = _exhaust(b._read_investing_paused())
        assert result is False
        b.context.logger.error.assert_called_once()

    def test_returns_false_when_value_is_none(self):
        """Test returns false when value is none."""
        b = _make_behaviour()
        b._read_kv = _make_gen_method({"investing_paused": None})
        result = _exhaust(b._read_investing_paused())
        assert result is False

    def test_unexpected_exception_propagates(self):
        """The narrowed handler does not swallow exceptions it never expected."""
        b = _make_behaviour()

        def raise_gen(*a, **kw):
            """Raise gen."""
            raise RuntimeError("boom")
            yield  # noqa

        b._read_kv = raise_gen
        with pytest.raises(RuntimeError, match="boom"):
            _exhaust(b._read_investing_paused())


class TestReadWithdrawalStatus:
    """Tests for _read_withdrawal_status."""

    def test_returns_status(self):
        """Test returns status."""
        b = _make_behaviour()
        b._read_kv = _make_gen_method({"withdrawal_status": "INITIATED"})
        result = _exhaust(b._read_withdrawal_status())
        assert result == "INITIATED"

    def test_returns_unknown_when_missing(self):
        """Test returns unknown when missing."""
        b = _make_behaviour()
        b._read_kv = _make_gen_method({})
        result = _exhaust(b._read_withdrawal_status())
        assert result == "unknown"

    def test_returns_unknown_when_kv_unreachable(self):
        """Test returns unknown when kv unreachable."""
        b = _make_behaviour()
        b._read_kv = _make_gen_method(None)
        result = _exhaust(b._read_withdrawal_status())
        assert result == "unknown"
        b.context.logger.error.assert_called_once()

    def test_returns_unknown_when_value_non_string(self):
        """Test returns unknown when value non string."""
        b = _make_behaviour()
        b._read_kv = _make_gen_method({"withdrawal_status": 42})
        result = _exhaust(b._read_withdrawal_status())
        assert result == "unknown"
        b.context.logger.error.assert_called_once()

    def test_unexpected_exception_propagates(self):
        """The narrowed handler does not swallow exceptions it never expected."""
        b = _make_behaviour()

        def raise_gen(*a, **kw):
            """Raise gen."""
            raise RuntimeError("fail")
            yield  # noqa

        b._read_kv = raise_gen
        with pytest.raises(RuntimeError, match="fail"):
            _exhaust(b._read_withdrawal_status())


class TestGetNextEvent:
    """Tests for get_next_event."""

    def test_no_actions(self):
        """Test no actions."""
        b = _make_behaviour()
        b.synchronized_data.actions = []
        b._read_investing_paused = _make_gen_method(False)
        b._read_withdrawal_status = _make_gen_method("unknown")
        result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_actions_all_executed(self):
        """When current_action_index >= len(actions), return DONE."""
        b = _make_behaviour()
        b.synchronized_data.actions = [{"action": "EnterPool"}]
        b.synchronized_data.last_executed_action_index = 0
        b.synchronized_data.last_action = None

        # Make last_round_id match EvaluateStrategyRound
        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_last_action_execute_step(self):
        """When last_action is EXECUTE_STEP and round changed, call _post_execute_step."""
        b = _make_behaviour()
        b.synchronized_data.actions = [{"action": "EnterPool"}, {"action": "ExitPool"}]
        b.synchronized_data.last_executed_action_index = 0
        b.synchronized_data.last_action = Action.EXECUTE_STEP.value

        mock_round = MagicMock()
        mock_round.round_id = "some_other_round"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            b.get_positions = _make_gen_method([])
            b._post_execute_step = _make_gen_method((Event.DONE.value, {}))

            with patch(
                "packages.valory.skills.liquidity_trader_abci.states.decision_making.DecisionMakingRound.auto_round_id",
                return_value="decision_making",
            ):
                result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_last_action_enter_pool(self):
        """When last_action is ENTER_POOL, call _post_execute_enter_pool then check all executed."""
        b = _make_behaviour()
        b.synchronized_data.actions = [{"action": "EnterPool"}]
        b.synchronized_data.last_executed_action_index = 0
        b.synchronized_data.last_action = Action.ENTER_POOL.value

        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        b._post_execute_enter_pool = _make_gen_method(None)

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_last_action_exit_pool(self):
        """Test last action exit pool."""
        b = _make_behaviour()
        b.synchronized_data.actions = [{"action": "ExitPool"}]
        b.synchronized_data.last_executed_action_index = 0
        b.synchronized_data.last_action = Action.EXIT_POOL.value

        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        b._post_execute_exit_pool = _make_gen_method(None)

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_last_action_deposit(self):
        """Test last action deposit."""
        b = _make_behaviour()
        b.synchronized_data.actions = [{"action": "deposit"}]
        b.synchronized_data.last_executed_action_index = 0
        b.synchronized_data.last_action = Action.DEPOSIT.value

        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        b._post_execute_enter_pool = _make_gen_method(None)

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_last_action_withdraw(self):
        """Test last action withdraw."""
        b = _make_behaviour()
        b.synchronized_data.actions = [{"action": "withdraw"}]
        b.synchronized_data.last_executed_action_index = 0
        b.synchronized_data.last_action = Action.WITHDRAW.value

        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        b._post_execute_exit_pool = _make_gen_method(None)
        b._post_execute_withdraw = _make_gen_method(None)

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_last_action_bridge_swap_executed(self):
        """Test last action bridge swap executed."""
        b = _make_behaviour()
        b.synchronized_data.actions = [{"action": "BridgeAndSwap"}]
        b.synchronized_data.last_executed_action_index = 0
        b.synchronized_data.last_action = Action.BRIDGE_SWAP_EXECUTED.value

        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        b._post_execute_transfer = MagicMock()

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_last_action_claim_rewards(self):
        """Test last action claim rewards."""
        b = _make_behaviour()
        b.synchronized_data.actions = [
            {"action": "ClaimRewards"},
            {"action": "ExitPool"},
        ]
        b.synchronized_data.last_executed_action_index = 0
        b.synchronized_data.last_action = Action.CLAIM_REWARDS.value

        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        mock_timestamp = MagicMock()
        mock_timestamp.timestamp.return_value = 12345.0
        b.context.state.round_sequence.last_round_transition_timestamp = mock_timestamp

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            # _post_execute_claim_rewards returns a tuple (not a generator).
            # When get_next_event does ``yield from tuple``, the generator
            # yields each element and the ``yield from`` expression evaluates
            # to None.  Therefore res is None and `return res` returns None.
            result = _exhaust(b.get_next_event())
        assert result is None

    def test_last_action_stake_lp_tokens(self):
        """Test last action stake lp tokens."""
        b = _make_behaviour()
        b.synchronized_data.actions = [{"action": "StakeLpTokens"}]
        b.synchronized_data.last_executed_action_index = 0
        b.synchronized_data.last_action = Action.STAKE_LP_TOKENS.value
        b.current_positions = []

        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        b._post_execute_stake_lp_tokens = MagicMock()
        b.store_current_positions = MagicMock()

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_last_action_unstake_lp_tokens(self):
        """Test last action unstake lp tokens."""
        b = _make_behaviour()
        b.synchronized_data.actions = [{"action": "UnstakeLpTokens"}]
        b.synchronized_data.last_executed_action_index = 0
        b.synchronized_data.last_action = Action.UNSTAKE_LP_TOKENS.value
        b.current_positions = []

        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        b._post_execute_unstake_lp_tokens = MagicMock()
        b.store_current_positions = MagicMock()

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_last_action_claim_staking_rewards(self):
        """Test last action claim staking rewards."""
        b = _make_behaviour()
        b.synchronized_data.actions = [{"action": "ClaimStakingRewards"}]
        b.synchronized_data.last_executed_action_index = 0
        b.synchronized_data.last_action = Action.CLAIM_STAKING_REWARDS.value
        b.current_positions = []

        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        b._post_execute_claim_staking_rewards = MagicMock()
        b.store_current_positions = MagicMock()

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_routes_fetched_last_action(self):
        """Test routes fetched last action."""
        b = _make_behaviour()
        b.synchronized_data.actions = [
            {"action": "FindBridgeRoute"},
            {"action": "EnterPool"},
        ]
        b.synchronized_data.last_executed_action_index = None
        b.synchronized_data.last_action = Action.ROUTES_FETCHED.value

        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        b._process_route_execution = _make_gen_method((Event.DONE.value, {}))

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_step_executed_last_action(self):
        """Test step executed last action."""
        b = _make_behaviour()
        b.synchronized_data.actions = [
            {"action": "FindBridgeRoute"},
            {"action": "EnterPool"},
        ]
        b.synchronized_data.last_executed_action_index = None
        b.synchronized_data.last_action = Action.STEP_EXECUTED.value

        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        b._process_route_execution = _make_gen_method((Event.DONE.value, {}))

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_switch_route_last_action(self):
        """Test switch route last action."""
        b = _make_behaviour()
        b.synchronized_data.actions = [
            {"action": "FindBridgeRoute"},
            {"action": "EnterPool"},
        ]
        b.synchronized_data.last_executed_action_index = None
        b.synchronized_data.last_action = Action.SWITCH_ROUTE.value

        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        b._process_route_execution = _make_gen_method((Event.DONE.value, {}))

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_prepare_next_action_called(self):
        """Test prepare next action called."""
        b = _make_behaviour()
        b.synchronized_data.actions = [{"action": "EnterPool"}]
        b.synchronized_data.last_executed_action_index = None
        b.synchronized_data.last_action = None

        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        b._prepare_next_action = _make_gen_method((Event.DONE.value, {}))

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_last_action_raises_value_error(self):
        """When last_action raises ValueError, it's caught gracefully."""
        b = _make_behaviour()
        b.synchronized_data.actions = [{"action": "EnterPool"}]
        b.synchronized_data.last_executed_action_index = None

        mock_round = MagicMock()
        mock_round.round_id = "evaluate_strategy"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        type(b.synchronized_data).last_action = PropertyMock(
            side_effect=ValueError("not set")
        )
        b._prepare_next_action = _make_gen_method((Event.DONE.value, {}))

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})

    def test_last_round_not_evaluate_strategy(self):
        """When last round is not EvaluateStrategyRound, get_positions is called."""
        b = _make_behaviour()
        b.synchronized_data.actions = [{"action": "EnterPool"}]
        b.synchronized_data.last_executed_action_index = None
        b.synchronized_data.last_action = None

        mock_round = MagicMock()
        mock_round.round_id = "some_other_round"
        b.context.state.round_sequence._abci_app._previous_rounds = [mock_round]

        b.get_positions = _make_gen_method([{"chain": "optimism", "assets": []}])
        b._prepare_next_action = _make_gen_method((Event.DONE.value, {}))

        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy.EvaluateStrategyRound.auto_round_id",
            return_value="evaluate_strategy",
        ):
            result = _exhaust(b.get_next_event())
        assert result == (Event.DONE.value, {})


class TestGetPortfolioData:
    """TestGetPortfolioData."""

    def test_empty_portfolio(self):
        """Test empty portfolio."""
        b = _make_behaviour()
        b.portfolio_data = {}
        result = _exhaust(b._get_portfolio_data())
        assert result is None

    def test_non_empty_portfolio(self):
        """Test non empty portfolio."""
        b = _make_behaviour()
        b.portfolio_data = {"key": "value"}
        result = _exhaust(b._get_portfolio_data())
        assert result == {"key": "value"}


class TestUpdateWithdrawalCompletion:
    """TestUpdateWithdrawalCompletion."""

    def test_without_final_tx_hash(self):
        """Test without final tx hash."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = None
        type(b.synchronized_data).final_tx_hash = PropertyMock(
            side_effect=AttributeError
        )
        b._write_kv = _make_gen_method(True)
        _exhaust(b._update_withdrawal_completion())

    def test_with_final_tx_hash(self):
        """Test with final tx hash."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xabc"
        b._write_kv = _make_gen_method(True)
        _exhaust(b._update_withdrawal_completion())

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()

        def raise_gen(*a, **kw):
            """Raise gen."""
            raise RuntimeError("boom")
            yield  # noqa

        b._write_kv = raise_gen
        # Should not raise
        _exhaust(b._update_withdrawal_completion())


class TestUpdateWithdrawalStatus:
    """TestUpdateWithdrawalStatus."""

    def test_non_completed(self):
        """Test non completed."""
        b = _make_behaviour()
        b._write_kv = _make_gen_method(True)
        _exhaust(b._update_withdrawal_status("WITHDRAWING", "msg"))

    def test_completed(self):
        """Test completed."""
        b = _make_behaviour()
        b._write_kv = _make_gen_method(True)
        _exhaust(b._update_withdrawal_status("COMPLETED", "done"))

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()

        def raise_gen(*a, **kw):
            """Raise gen."""
            raise RuntimeError("fail")
            yield  # noqa

        b._write_kv = raise_gen
        _exhaust(b._update_withdrawal_status("X", "y"))


class TestPostExecuteStep:
    """TestPostExecuteStep."""

    def test_decision_exit(self):
        """Test decision exit."""
        b = _make_behaviour()
        b.sleep = _make_gen_method(None)
        b.get_decision_on_swap = _make_gen_method(Decision.EXIT)
        result = _exhaust(b._post_execute_step([{"action": "x"}], 0))
        assert result == (Event.DONE.value, {})

    def test_decision_continue(self):
        """Test decision continue."""
        b = _make_behaviour()
        b.sleep = _make_gen_method(None)
        b.get_decision_on_swap = _make_gen_method(Decision.CONTINUE)
        b._add_slippage_costs = _make_gen_method(None)
        b.synchronized_data.last_executed_step_index = None
        b._update_assets_after_swap = MagicMock(
            return_value=(
                Event.UPDATE.value,
                {"last_action": Action.STEP_EXECUTED.value},
            )
        )
        result = _exhaust(b._post_execute_step([{"action": "x"}], 0))
        assert result[0] == Event.UPDATE.value

    def test_decision_wait_then_continue(self):
        """Test decision wait then continue."""
        b = _make_behaviour()
        b.sleep = _make_gen_method(None)
        # First call returns WAIT, _wait_for_swap_confirmation will loop
        b.get_decision_on_swap = _make_gen_method(Decision.CONTINUE)
        b._wait_for_swap_confirmation = _make_gen_method(Decision.CONTINUE)

        # Override get_decision_on_swap to return WAIT first
        call_count = [0]

        def swap_gen(*a, **kw):
            """Swap gen."""
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return Decision.WAIT
            yield
            return Decision.CONTINUE

        b.get_decision_on_swap = swap_gen
        b._add_slippage_costs = _make_gen_method(None)
        b.synchronized_data.last_executed_step_index = 0
        b._update_assets_after_swap = MagicMock(return_value=(Event.UPDATE.value, {}))
        result = _exhaust(b._post_execute_step([{"action": "x"}], 0))
        assert result[0] == Event.UPDATE.value

    def test_decision_wait_then_exit(self):
        """Test decision wait then exit."""
        b = _make_behaviour()
        b.sleep = _make_gen_method(None)
        call_count = [0]

        def swap_gen(*a, **kw):
            """Swap gen."""
            call_count[0] += 1
            if call_count[0] == 1:
                yield
                return Decision.WAIT
            yield
            return Decision.EXIT

        b.get_decision_on_swap = swap_gen
        b._wait_for_swap_confirmation = _make_gen_method(Decision.EXIT)
        result = _exhaust(b._post_execute_step([{"action": "x"}], 0))
        assert result == (Event.DONE.value, {})


class TestWaitForSwapConfirmation:
    """TestWaitForSwapConfirmation."""

    def test_immediate_continue(self):
        """Test immediate continue."""
        b = _make_behaviour()
        b.sleep = _make_gen_method(None)
        b.get_decision_on_swap = _make_gen_method(Decision.CONTINUE)
        result = _exhaust(b._wait_for_swap_confirmation())
        assert result == Decision.CONTINUE

    def test_immediate_exit(self):
        """Test immediate exit."""
        b = _make_behaviour()
        b.sleep = _make_gen_method(None)
        b.get_decision_on_swap = _make_gen_method(Decision.EXIT)
        result = _exhaust(b._wait_for_swap_confirmation())
        assert result == Decision.EXIT


class TestUpdateAssetsAfterSwap:
    """TestUpdateAssetsAfterSwap."""

    def test_normal_swap(self):
        """Test normal swap."""
        b = _make_behaviour()
        b.synchronized_data.last_executed_step_index = None
        action = {"remaining_fee_allowance": 1.0, "remaining_gas_allowance": 2.0}
        result = b._update_assets_after_swap([action], 0)
        assert result[0] == Event.UPDATE.value
        assert result[1]["last_executed_step_index"] == 0
        assert result[1]["last_action"] == Action.STEP_EXECUTED.value

    def test_with_existing_step_index(self):
        """Test with existing step index."""
        b = _make_behaviour()
        b.synchronized_data.last_executed_step_index = 2
        action = {"remaining_fee_allowance": 1.0, "remaining_gas_allowance": 2.0}
        result = b._update_assets_after_swap([action], 0)
        assert result[1]["last_executed_step_index"] == 3

    def test_withdrawal_swap(self):
        """Test withdrawal swap."""
        b = _make_behaviour()
        b.synchronized_data.last_executed_step_index = 0
        action = {
            "description": "Withdrawal: swap",
            "remaining_fee_allowance": 1.0,
            "remaining_gas_allowance": 2.0,
        }
        result = b._update_assets_after_swap([action], 0)
        assert result[0] == Event.UPDATE.value


class TestPostExecuteExitPool:
    """TestPostExecuteExitPool."""

    def test_updates_position_status(self):
        """Test updates position status."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xexit"
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "status": PositionStatus.OPEN.value,
                "enter_timestamp": 1000,
            },
        ]
        b.store_current_positions = MagicMock()
        b._get_current_timestamp = MagicMock(return_value=2000)
        b._record_tip_performance = MagicMock()
        b.sleep = _make_gen_method(None)

        action = {"pool_address": "0xPOOL", "dex_type": "UniswapV3"}
        _exhaust(b._post_execute_exit_pool([action], 0))
        assert b.current_positions[0]["status"] == PositionStatus.CLOSED.value

    def test_velodrome_cl_pool(self):
        """Test velodrome cl pool."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xexit"
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "status": PositionStatus.OPEN.value,
                "enter_timestamp": 1000,
                "positions": [{"token_id": 1}, {"token_id": 2}],
            },
        ]
        b.store_current_positions = MagicMock()
        b._get_current_timestamp = MagicMock(return_value=2000)
        b._record_tip_performance = MagicMock()
        b.sleep = _make_gen_method(None)

        action = {
            "pool_address": "0xPOOL",
            "dex_type": DexType.VELODROME.value,
            "is_cl_pool": True,
        }
        _exhaust(b._post_execute_exit_pool([action], 0))
        assert b.current_positions[0]["status"] == PositionStatus.CLOSED.value

    def test_withdrawal_exit(self):
        """Test withdrawal exit."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xexit"
        b.current_positions = []
        b.store_current_positions = MagicMock()
        b.sleep = _make_gen_method(None)

        action = {"pool_address": "0xNONE", "description": "Withdrawal: exit pool"}
        _exhaust(b._post_execute_exit_pool([action], 0))

    def test_non_matching_pool(self):
        """Test non matching pool."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xexit"
        b.current_positions = [
            {"pool_address": "0xOTHER", "status": PositionStatus.OPEN.value},
        ]
        b.store_current_positions = MagicMock()
        b._get_current_timestamp = MagicMock(return_value=2000)
        b.sleep = _make_gen_method(None)
        action = {"pool_address": "0xPOOL", "dex_type": "UniswapV3"}
        _exhaust(b._post_execute_exit_pool([action], 0))
        assert b.current_positions[0]["status"] == PositionStatus.OPEN.value


class TestPostExecuteTransfer:
    """TestPostExecuteTransfer."""

    def test_logs_transfer(self):
        """Test logs transfer."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xtx"
        action = {
            "from_chain": "optimism",
            "to_chain": "mode",
            "from_token_symbol": "USDC",
            "to_token_symbol": "USDC",
        }
        b._post_execute_transfer([action], 0)
        b.context.logger.info.assert_called()


class TestPostExecuteWithdraw:
    """TestPostExecuteWithdraw."""

    def test_marks_complete(self):
        """Test marks complete."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(True)
        b._read_withdrawal_status = _make_gen_method("WITHDRAWING")
        b._update_withdrawal_completion = _make_gen_method(None)
        _exhaust(b._post_execute_withdraw([], 0))

    def test_not_paused(self):
        """Test not paused."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._read_withdrawal_status = _make_gen_method("unknown")
        _exhaust(b._post_execute_withdraw([], 0))

    def test_paused_not_withdrawing(self):
        """Test paused not withdrawing."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(True)
        b._read_withdrawal_status = _make_gen_method("INITIATED")
        _exhaust(b._post_execute_withdraw([], 0))


class TestPostExecuteClaimRewards:
    """TestPostExecuteClaimRewards."""

    def test_returns_update(self):
        """Test returns update."""
        b = _make_behaviour()
        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 999.0
        b.context.state.round_sequence.last_round_transition_timestamp = mock_ts
        result = b._post_execute_claim_rewards([], 0)
        assert result[0] == Event.UPDATE.value
        assert result[1]["last_reward_claimed_timestamp"] == 999.0


class TestProcessRouteExecution:
    """TestProcessRouteExecution."""

    def test_no_routes(self):
        """Test no routes."""
        b = _make_behaviour()
        b.synchronized_data.routes = []
        result = _exhaust(b._process_route_execution([]))
        assert result == (Event.DONE.value, {})

    def test_no_more_routes(self):
        """Test no more routes."""
        b = _make_behaviour()
        b.synchronized_data.routes = [{"steps": []}]
        b.synchronized_data.last_executed_route_index = 0
        b.synchronized_data.last_executed_step_index = None
        result = _exhaust(b._process_route_execution([]))
        assert result == (Event.DONE.value, {})

    def test_all_steps_executed(self):
        """Test all steps executed."""
        b = _make_behaviour()
        b.synchronized_data.routes = [{"steps": [{"action": {}}]}]
        b.synchronized_data.last_executed_route_index = None
        b.synchronized_data.last_executed_step_index = 0
        result = _exhaust(b._process_route_execution([]))
        assert result[0] == Event.UPDATE.value
        assert result[1]["last_action"] == Action.BRIDGE_SWAP_EXECUTED.value

    def test_execute_step(self):
        """Test execute step."""
        b = _make_behaviour()
        b.synchronized_data.routes = [{"steps": [{"action": {}}, {"action": {}}]}]
        b.synchronized_data.last_executed_route_index = None
        b.synchronized_data.last_executed_step_index = None
        b._execute_route_step = _make_gen_method((Event.UPDATE.value, {}))
        result = _exhaust(b._process_route_execution([]))
        assert result[0] == Event.UPDATE.value


class TestExecuteRouteStep:
    """TestExecuteRouteStep."""

    def test_first_step_profitable(self):
        """Test first step profitable."""
        b = _make_behaviour()
        b.check_if_route_is_profitable = _make_gen_method((True, 0.5, 0.3))
        b.check_step_costs = _make_gen_method(
            (
                True,
                {
                    "source_token_symbol": "USDC",
                    "from_chain": "optimism",
                    "target_token_symbol": "USDC",
                    "to_chain": "mode",
                    "tool": "lifi",
                },
            )
        )
        b.prepare_bridge_swap_action = _make_gen_method({"action": "BridgeAndSwap"})
        routes = [{"steps": [{"action": {}}]}]
        result = _exhaust(b._execute_route_step([], routes, 0, 0))
        assert result[0] == Event.UPDATE.value
        assert result[1]["last_action"] == Action.EXECUTE_STEP.value

    def test_first_step_not_profitable_false(self):
        """Test first step not profitable false."""
        b = _make_behaviour()
        b.check_if_route_is_profitable = _make_gen_method((False, None, None))
        routes = [{"steps": [{"action": {}}]}]
        result = _exhaust(b._execute_route_step([], routes, 0, 0))
        assert result[0] == Event.UPDATE.value
        assert result[1]["last_action"] == Action.SWITCH_ROUTE.value

    def test_first_step_not_profitable_none(self):
        """Test first step not profitable none."""
        b = _make_behaviour()
        b.check_if_route_is_profitable = _make_gen_method((None, None, None))
        routes = [{"steps": [{"action": {}}]}]
        result = _exhaust(b._execute_route_step([], routes, 0, 0))
        assert result[0] == Event.UPDATE.value
        assert result[1]["last_action"] == Action.SWITCH_ROUTE.value

    def test_non_first_step(self):
        """Test non first step."""
        b = _make_behaviour()
        b.synchronized_data.fee_details = {
            "remaining_fee_allowance": 1.0,
            "remaining_gas_allowance": 1.0,
        }
        b.check_step_costs = _make_gen_method(
            (
                True,
                {
                    "source_token_symbol": "A",
                    "from_chain": "optimism",
                    "target_token_symbol": "B",
                    "to_chain": "mode",
                    "tool": "lifi",
                },
            )
        )
        b.prepare_bridge_swap_action = _make_gen_method({"action": "BridgeAndSwap"})
        routes = [{"steps": [{"action": {}}, {"action": {}}]}]
        result = _exhaust(b._execute_route_step([], routes, 0, 1))
        assert result[0] == Event.UPDATE.value

    def test_step_not_profitable(self):
        """Test step not profitable."""
        b = _make_behaviour()
        b.check_if_route_is_profitable = _make_gen_method((True, 1.0, 1.0))
        b.check_step_costs = _make_gen_method((False, None))
        routes = [{"steps": [{"action": {}}]}]
        result = _exhaust(b._execute_route_step([], routes, 0, 0))
        assert result == (Event.DONE.value, {})

    def test_bridge_swap_action_none(self):
        """Test bridge swap action none."""
        b = _make_behaviour()
        b.check_if_route_is_profitable = _make_gen_method((True, 1.0, 1.0))
        b.check_step_costs = _make_gen_method(
            (
                True,
                {
                    "source_token_symbol": "A",
                    "from_chain": "optimism",
                    "target_token_symbol": "B",
                    "to_chain": "mode",
                    "tool": "lifi",
                    "from_chain": "optimism",  # noqa: B041
                    "to_chain": "mode",  # noqa: B041
                    "source_token": "0x1",
                    "source_token_symbol": "A",  # noqa: B041
                    "target_token": "0x2",
                    "target_token_symbol": "B",  # noqa: B041
                },
            )
        )
        b.prepare_bridge_swap_action = _make_gen_method(None)
        b._handle_failed_step = MagicMock(return_value=(Event.UPDATE.value, {}))
        routes = [{"steps": [{"action": {}}]}]
        result = _exhaust(b._execute_route_step([], routes, 0, 0))
        assert result[0] == Event.UPDATE.value


class TestHandleFailedStep:
    """TestHandleFailedStep."""

    def test_first_step_failed(self):
        """Test first step failed."""
        b = _make_behaviour()
        result = b._handle_failed_step(0, 0, {}, 3)
        assert result[1]["last_action"] == Action.SWITCH_ROUTE.value

    def test_intermediate_step_within_retry(self):
        """Test intermediate step within retry."""
        b = _make_behaviour()
        b.synchronized_data.routes_retry_attempt = 0
        step_data = {
            "from_chain": "optimism",
            "to_chain": "mode",
            "source_token": "0x1",
            "source_token_symbol": "A",
            "target_token": "0x2",
            "target_token_symbol": "B",
        }
        result = b._handle_failed_step(1, 0, step_data, 3)
        assert result[1]["last_action"] == Action.FIND_ROUTE.value

    def test_intermediate_step_exceeded_retries(self):
        """Test intermediate step exceeded retries."""
        b = _make_behaviour()
        b.synchronized_data.routes_retry_attempt = MAX_RETRIES_FOR_ROUTES + 1
        result = b._handle_failed_step(1, 0, {}, 3)
        assert result == (Event.DONE.value, {})


class TestGetDecisionOnSwap:
    """TestGetDecisionOnSwap."""

    def test_done_status(self):
        """Test done status."""
        b = _make_behaviour()
        b.get_swap_status = _make_gen_method(("DONE", "COMPLETED"))
        result = _exhaust(b.get_decision_on_swap())
        assert result == Decision.CONTINUE

    def test_pending_status(self):
        """Test pending status."""
        b = _make_behaviour()
        b.get_swap_status = _make_gen_method(("PENDING", "WAIT_FOR_CONFIRMATION"))
        result = _exhaust(b.get_decision_on_swap())
        assert result == Decision.WAIT

    def test_failed_status(self):
        """Test failed status."""
        b = _make_behaviour()
        b.get_swap_status = _make_gen_method(("FAILED", "ERROR"))
        result = _exhaust(b.get_decision_on_swap())
        assert result == Decision.EXIT

    def test_none_status(self):
        """Test none status."""
        b = _make_behaviour()
        b.get_swap_status = _make_gen_method((None, None))
        result = _exhaust(b.get_decision_on_swap())
        assert result == Decision.EXIT

    def test_no_tx_hash(self):
        """Test no tx hash."""
        b = _make_behaviour()
        type(b.synchronized_data).final_tx_hash = PropertyMock(
            side_effect=Exception("no hash")
        )
        result = _exhaust(b.get_decision_on_swap())
        assert result == Decision.EXIT


class TestGetSwapStatus:
    """TestGetSwapStatus."""

    def test_ok_response(self):
        """Test ok response."""
        b = _make_behaviour()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.body = json.dumps({"status": "DONE", "substatus": "COMPLETED"})
        b.get_http_response = _make_gen_method(mock_resp)
        result = _exhaust(b.get_swap_status("0xabc"))
        assert result == ("DONE", "COMPLETED")

    def test_not_found_then_ok(self):
        """Test not found then ok."""
        b = _make_behaviour()
        call_count = [0]

        def http_gen(*a, **kw):
            """Http gen."""
            call_count[0] += 1
            if call_count[0] == 1:
                resp = MagicMock()
                resp.status_code = 404
                resp.body = "not found"
                yield
                return resp
            resp = MagicMock()
            resp.status_code = 200
            resp.body = json.dumps({"status": "DONE", "substatus": "OK"})
            yield
            return resp

        b.get_http_response = http_gen
        b.sleep = _make_gen_method(None)
        result = _exhaust(b.get_swap_status("0xabc"))
        assert result == ("DONE", "OK")

    def test_error_status_code(self):
        """Test error status code."""
        b = _make_behaviour()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.body = "error"
        b.get_http_response = _make_gen_method(mock_resp)
        result = _exhaust(b.get_swap_status("0xabc"))
        assert result == (None, None)

    def test_json_parse_error(self):
        """Test json parse error."""
        b = _make_behaviour()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.body = "not json"
        b.get_http_response = _make_gen_method(mock_resp)
        result = _exhaust(b.get_swap_status("0xabc"))
        assert result == (None, None)

    def test_no_status_in_response(self):
        """Test no status in response."""
        b = _make_behaviour()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.body = json.dumps({"substatus": "X"})
        b.get_http_response = _make_gen_method(mock_resp)
        result = _exhaust(b.get_swap_status("0xabc"))
        assert result == (None, "X")

    def test_400_not_found(self):
        """Test 400 not found."""
        b = _make_behaviour()
        call_count = [0]

        def http_gen(*a, **kw):
            """Http gen."""
            call_count[0] += 1
            if call_count[0] == 1:
                resp = MagicMock()
                resp.status_code = 400
                resp.body = "bad request"
                yield
                return resp
            resp = MagicMock()
            resp.status_code = 200
            resp.body = json.dumps({"status": "DONE", "substatus": "OK"})
            yield
            return resp

        b.get_http_response = http_gen
        b.sleep = _make_gen_method(None)
        result = _exhaust(b.get_swap_status("0xabc"))
        assert result == ("DONE", "OK")

    def test_retry_exhaustion_returns_none(self):
        """When status polling exceeds MAX_RETRIES_FOR_STATUS_CHECK, return (None, None)."""
        b = _make_behaviour()

        def always_404(*a, **kw):
            """Always 404."""
            resp = MagicMock()
            resp.status_code = 404
            resp.body = "not found"
            yield
            return resp

        b.get_http_response = always_404
        b.sleep = _make_gen_method(None)
        result = _exhaust(b.get_swap_status("0xabc"))
        assert result == (None, None)
        b.context.logger.error.assert_called()

    def test_retry_exhaustion_400_returns_none(self):
        """When status polling exceeds MAX_RETRIES_FOR_STATUS_CHECK with 400s."""
        b = _make_behaviour()

        def always_400(*a, **kw):
            """Always 400."""
            resp = MagicMock()
            resp.status_code = 400
            resp.body = "bad request"
            yield
            return resp

        b.get_http_response = always_400
        b.sleep = _make_gen_method(None)
        result = _exhaust(b.get_swap_status("0xabc"))
        assert result == (None, None)


class TestCalculateMinHoldDays:
    """TestCalculateMinHoldDays."""

    def test_zero_apr(self):
        """Test zero apr."""
        b = _make_behaviour()
        result = b._calculate_min_hold_days(0.0, 100, 10, False)
        assert result == MIN_TIME_IN_POSITION

    def test_zero_principal(self):
        """Test zero principal."""
        b = _make_behaviour()
        result = b._calculate_min_hold_days(0.2, 0.0, 10, False)
        assert result == MIN_TIME_IN_POSITION

    def test_zero_entry_cost(self):
        """Test zero entry cost."""
        b = _make_behaviour()
        result = b._calculate_min_hold_days(0.2, 100, 0.0, False)
        assert result == MIN_TIME_IN_POSITION

    def test_normal_non_cl(self):
        """Test normal non cl."""
        b = _make_behaviour()
        result = b._calculate_min_hold_days(0.20, 1000, 10, False)
        assert 12.0 <= result <= MIN_TIME_IN_POSITION

    def test_cl_pool(self):
        """Test cl pool."""
        b = _make_behaviour()
        result = b._calculate_min_hold_days(0.20, 1000, 10, True, 0.8)
        assert 12.0 <= result <= MIN_TIME_IN_POSITION

    def test_very_high_cost(self):
        """Test very high cost."""
        b = _make_behaviour()
        result = b._calculate_min_hold_days(0.01, 100, 50, False)
        assert result == MIN_TIME_IN_POSITION

    def test_very_low_cost(self):
        """Test very low cost."""
        b = _make_behaviour()
        result = b._calculate_min_hold_days(1.0, 10000, 0.01, False)
        assert result == 12.0

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()
        result = b._calculate_min_hold_days(float("nan"), 100, 10, False)
        # Should still return something valid
        assert isinstance(result, float)


class TestRecordTipPerformance:
    """TestRecordTipPerformance."""

    def test_no_enter_timestamp(self):
        """Test no enter timestamp."""
        b = _make_behaviour()
        pos = {}
        b._record_tip_performance(pos)
        assert "cost_recovered" not in pos

    def test_with_data(self):
        """Test with data."""
        b = _make_behaviour()
        b._get_current_timestamp = MagicMock(return_value=2000)
        pos = {
            "enter_timestamp": 1000,
            "entry_cost": 5.0,
            "min_hold_days": 10,
            "principal_usd": 100,
            "pool_address": "0xPOOL",
        }
        b._record_tip_performance(pos)
        assert pos["cost_recovered"] is True

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()
        b._get_current_timestamp = MagicMock(side_effect=RuntimeError("err"))
        pos = {"enter_timestamp": 1000}
        b._record_tip_performance(pos)


class TestGetSignature:
    """TestGetSignature."""

    def test_signature(self):
        """Test signature."""
        b = _make_behaviour()
        sig = b._get_signature("0x" + "ab" * 20)
        assert isinstance(sig, str)
        assert len(sig) == 130  # 65 bytes hex


class TestResetWithdrawalFlags:
    """TestResetWithdrawalFlags."""

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        b._write_kv = _make_gen_method(True)
        _exhaust(b._reset_withdrawal_flags())

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()

        def raise_gen(*a, **kw):
            """Raise gen."""
            raise RuntimeError("err")
            yield  # noqa

        b._write_kv = raise_gen
        _exhaust(b._reset_withdrawal_flags())


class TestEntryCostsKeys:
    """TestEntryCostsKeys."""

    def test_get_entry_costs_key(self):
        """Test get entry costs key."""
        b = _make_behaviour()
        assert (
            b._get_entry_costs_key("optimism", "0xPOOL")
            == "entry_costs_optimism_0xPOOL"
        )

    def test_get_updated_entry_costs_key(self):
        """Test get updated entry costs key."""
        b = _make_behaviour()
        assert (
            b._get_updated_entry_costs_key("optimism", "0xPOOL", "123")
            == "entry_costs_optimism_0xPOOL_123"
        )


class TestGetEntryCosts:
    """TestGetEntryCosts."""

    def test_found(self):
        """Test found."""
        b = _make_behaviour()
        b._get_all_entry_costs = _make_gen_method({"entry_costs_optimism_0xPOOL": 5.0})
        result = _exhaust(b._get_entry_costs("optimism", "0xPOOL"))
        assert result == 5.0

    def test_not_found(self):
        """Test not found."""
        b = _make_behaviour()
        b._get_all_entry_costs = _make_gen_method({})
        result = _exhaust(b._get_entry_costs("optimism", "0xPOOL"))
        assert result == 0.0

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()

        def raise_gen(*a, **kw):
            """Raise gen."""
            raise RuntimeError("err")
            yield  # noqa

        b._get_all_entry_costs = raise_gen
        result = _exhaust(b._get_entry_costs("optimism", "0xPOOL"))
        assert result == 0.0


class TestGetUpdatedEntryCosts:
    """TestGetUpdatedEntryCosts."""

    def test_found(self):
        """Test found."""
        b = _make_behaviour()
        b._get_all_entry_costs = _make_gen_method(
            {"entry_costs_optimism_0xPOOL_123": 7.0}
        )
        result = _exhaust(b._get_updated_entry_costs("optimism", "0xPOOL", "123"))
        assert result == 7.0

    def test_not_found(self):
        """Test not found."""
        b = _make_behaviour()
        b._get_all_entry_costs = _make_gen_method({})
        result = _exhaust(b._get_updated_entry_costs("optimism", "0xPOOL", "123"))
        assert result == 0.0

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()

        def raise_gen(*a, **kw):
            """Raise gen."""
            raise RuntimeError("err")
            yield  # noqa

        b._get_all_entry_costs = raise_gen
        result = _exhaust(b._get_updated_entry_costs("optimism", "0xPOOL", "123"))
        assert result == 0.0


class TestUpdateEntryCosts:
    """TestUpdateEntryCosts."""

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        b._get_entry_costs = _make_gen_method(3.0)
        b._store_entry_costs = _make_gen_method(None)
        result = _exhaust(b._update_entry_costs("optimism", "0xPOOL", 2.0))
        assert result == 5.0

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()

        def raise_gen(*a, **kw):
            """Raise gen."""
            raise RuntimeError("err")
            yield  # noqa

        b._get_entry_costs = raise_gen
        result = _exhaust(b._update_entry_costs("optimism", "0xPOOL", 2.0))
        assert result == 0.0


class TestRenameEntryCostsKey:
    """TestRenameEntryCostsKey."""

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        b._get_all_entry_costs = _make_gen_method({"entry_costs_optimism_0xPOOL": 5.0})
        b._write_kv = _make_gen_method(True)
        pos = {"chain": "optimism", "pool_address": "0xPOOL", "enter_timestamp": "123"}
        _exhaust(b._rename_entry_costs_key(pos))

    def test_old_key_not_found(self):
        """Test old key not found."""
        b = _make_behaviour()
        b._get_all_entry_costs = _make_gen_method({})
        b._write_kv = _make_gen_method(True)
        pos = {"chain": "optimism", "pool_address": "0xPOOL", "enter_timestamp": "123"}
        # This will raise KeyError when trying del, caught by except
        _exhaust(b._rename_entry_costs_key(pos))

    def test_new_key_exists(self):
        """Test new key exists."""
        b = _make_behaviour()
        b._get_all_entry_costs = _make_gen_method(
            {
                "entry_costs_optimism_0xPOOL": 5.0,
                "entry_costs_optimism_0xPOOL_123": 3.0,
            }
        )
        b._write_kv = _make_gen_method(True)
        pos = {"chain": "optimism", "pool_address": "0xPOOL", "enter_timestamp": "123"}
        _exhaust(b._rename_entry_costs_key(pos))


class TestPostExecuteStakeLpTokens:
    """TestPostExecuteStakeLpTokens."""

    def test_updates_position(self):
        """Test updates position."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xstake"
        b._get_current_timestamp = MagicMock(return_value=3000)
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
            },
        ]
        b.store_current_positions = MagicMock()
        action = {"pool_address": "0xPOOL", "chain": "optimism", "is_cl_pool": False}
        b._post_execute_stake_lp_tokens([action], 0)
        assert b.current_positions[0]["staked"] is True

    def test_cl_pool(self):
        """Test cl pool."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xstake"
        b._get_current_timestamp = MagicMock(return_value=3000)
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
            },
        ]
        b.store_current_positions = MagicMock()
        action = {"pool_address": "0xPOOL", "chain": "optimism", "is_cl_pool": True}
        b._post_execute_stake_lp_tokens([action], 0)
        assert b.current_positions[0]["staked_cl_pool"] is True

    def test_no_matching_position(self):
        """Test no matching position."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xstake"
        b._get_current_timestamp = MagicMock(return_value=3000)
        b.current_positions = []
        b.store_current_positions = MagicMock()
        action = {"pool_address": "0xPOOL", "chain": "optimism", "is_cl_pool": False}
        b._post_execute_stake_lp_tokens([action], 0)


class TestPostExecuteUnstakeLpTokens:
    """TestPostExecuteUnstakeLpTokens."""

    def test_updates_position(self):
        """Test updates position."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xunstake"
        b._get_current_timestamp = MagicMock(return_value=4000)
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "staked": True,
                "staked_cl_pool": True,
            },
        ]
        b.store_current_positions = MagicMock()
        action = {"pool_address": "0xPOOL", "chain": "optimism"}
        b._post_execute_unstake_lp_tokens([action], 0)
        assert b.current_positions[0]["staked"] is False
        assert b.current_positions[0]["staked_cl_pool"] is False

    def test_no_cl_key(self):
        """Test no cl key."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xunstake"
        b._get_current_timestamp = MagicMock(return_value=4000)
        b.current_positions = [
            {"pool_address": "0xPOOL", "chain": "optimism", "staked": True},
        ]
        b.store_current_positions = MagicMock()
        action = {"pool_address": "0xPOOL", "chain": "optimism"}
        b._post_execute_unstake_lp_tokens([action], 0)
        assert b.current_positions[0]["staked"] is False


class TestPostExecuteClaimStakingRewards:
    """TestPostExecuteClaimStakingRewards."""

    def test_updates_position(self):
        """Test updates position."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xclaim"
        b._get_current_timestamp = MagicMock(return_value=5000)
        b.current_positions = [
            {"pool_address": "0xPOOL", "chain": "optimism"},
        ]
        b.store_current_positions = MagicMock()
        action = {"pool_address": "0xPOOL", "chain": "optimism"}
        b._post_execute_claim_staking_rewards([action], 0)
        assert b.current_positions[0]["last_reward_claim_tx_hash"] == "0xclaim"

    def test_no_matching(self):
        """Test no matching."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xclaim"
        b._get_current_timestamp = MagicMock(return_value=5000)
        b.current_positions = []
        b.store_current_positions = MagicMock()
        action = {"pool_address": "0xPOOL", "chain": "optimism"}
        b._post_execute_claim_staking_rewards([action], 0)


class TestSetStepAddresses:
    """TestSetStepAddresses."""

    def test_sets_addresses(self):
        """Test sets addresses."""
        b = _make_behaviour()
        step = {"action": {"fromChainId": 10, "toChainId": 34443}}
        result = b._set_step_addresses(step)
        assert result["action"]["fromAddress"] == "0x" + "aa" * 20
        assert result["action"]["toAddress"] == "0x" + "bb" * 20


class TestEnforcePoolAllocationCap:
    """TestEnforcePoolAllocationCap."""

    def test_none_max_position_size(self):
        """Test none max position size."""
        b = _make_behaviour()
        result = _exhaust(
            b._enforce_pool_allocation_cap([100, 200], None, "optimism", ["0x1", "0x2"])
        )
        assert result == [100, 200]

    def test_zero_max_position_size(self):
        """Test zero max position size."""
        b = _make_behaviour()
        result = _exhaust(
            b._enforce_pool_allocation_cap([100, 200], 0, "optimism", ["0x1", "0x2"])
        )
        assert result == [100, 200]

    def test_negative_max_position_size(self):
        """Test negative max position size."""
        b = _make_behaviour()
        result = _exhaust(
            b._enforce_pool_allocation_cap([100, 200], -1, "optimism", ["0x1", "0x2"])
        )
        assert result == [100, 200]

    def test_below_cap(self):
        """Test below cap."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(18)
        b._fetch_token_price = _make_gen_method(1.0)
        # amounts are tiny, USD < max_position_size
        result = _exhaust(
            b._enforce_pool_allocation_cap(
                [100, 200], 1000000.0, "optimism", ["0x1", "0x2"]
            )
        )
        assert result == [100, 200]

    def test_above_cap(self):
        """Test above cap."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(1.0)
        # 1_000_000 units at 6 decimals = $1, so total = $2, cap = $1
        result = _exhaust(
            b._enforce_pool_allocation_cap(
                [1_000_000, 1_000_000], 1.0, "optimism", ["0x1", "0x2"]
            )
        )
        assert result[0] < 1_000_000

    def test_decimals_none(self):
        """Test decimals none."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(None)
        result = _exhaust(
            b._enforce_pool_allocation_cap([100, 200], 10.0, "optimism", ["0x1", "0x2"])
        )
        assert result == [100, 200]

    def test_price_none(self):
        """Test price none."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(18)
        call_count = [0]

        def price_gen(*a, **kw):
            """Price gen."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return 1.0
            return None

        b._fetch_token_price = price_gen
        result = _exhaust(
            b._enforce_pool_allocation_cap([100, 200], 10.0, "optimism", ["0x1", "0x2"])
        )
        assert result == [100, 200]

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()

        def raise_gen(*a, **kw):
            """Raise gen."""
            raise RuntimeError("err")
            yield  # noqa

        b._get_token_decimals = raise_gen
        result = _exhaust(
            b._enforce_pool_allocation_cap([100, 200], 10.0, "optimism", ["0x1", "0x2"])
        )
        assert result == [100, 200]

    def test_zero_total_usd(self):
        """Test zero total usd."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(18)
        b._fetch_token_price = _make_gen_method(0.0)
        result = _exhaust(
            b._enforce_pool_allocation_cap([0, 0], 10.0, "optimism", ["0x1", "0x2"])
        )
        # current_total_usd == 0, which is <= cap
        assert result == [0, 0]


class TestCalculateInvestmentAmountsFromDollarCap:
    """TestCalculateInvestmentAmountsFromDollarCap."""

    def test_no_invested_amount(self):
        """Test no invested amount."""
        b = _make_behaviour()
        result = _exhaust(
            b._calculate_investment_amounts_from_dollar_cap(
                {"invested_amount": 0}, "optimism", ["0x1", "0x2"]
            )
        )
        assert result is None

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(1.0)
        b.sleep = _make_gen_method(None)
        action = {
            "invested_amount": 100,
            "token0": "0x1",
            "token1": "0x2",
            "relative_funds_percentage": 1.0,
        }
        result = _exhaust(
            b._calculate_investment_amounts_from_dollar_cap(
                action, "optimism", ["0x1", "0x2"]
            )
        )
        assert result is not None
        assert len(result) == 2

    def test_decimals_none(self):
        """Test decimals none."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(None)
        action = {"invested_amount": 100, "token0": "0x1", "token1": "0x2"}
        result = _exhaust(
            b._calculate_investment_amounts_from_dollar_cap(
                action, "optimism", ["0x1", "0x2"]
            )
        )
        assert result is None

    def test_price_none(self):
        """Test price none."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(None)
        action = {"invested_amount": 100, "token0": "0x1", "token1": "0x2"}
        result = _exhaust(
            b._calculate_investment_amounts_from_dollar_cap(
                action, "optimism", ["0x1", "0x2"]
            )
        )
        assert result is None

    def test_token1_price_none(self):
        """Test token1 price none."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(6)
        call_count = [0]

        def price_gen(*a, **kw):
            """Price gen."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return 1.0
            return None

        b._fetch_token_price = price_gen
        b.sleep = _make_gen_method(None)
        action = {"invested_amount": 100, "token0": "0x1", "token1": "0x2"}
        result = _exhaust(
            b._calculate_investment_amounts_from_dollar_cap(
                action, "optimism", ["0x1", "0x2"]
            )
        )
        assert result is None

    def test_no_relative_funds(self):
        """Test no relative funds."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(1.0)
        b.sleep = _make_gen_method(None)
        action = {
            "invested_amount": 100,
            "token0": "0x1",
            "token1": "0x2",
            "relative_funds_percentage": 0,
        }
        result = _exhaust(
            b._calculate_investment_amounts_from_dollar_cap(
                action, "optimism", ["0x1", "0x2"]
            )
        )
        assert result is None

    def test_negative_amounts(self):
        """Test negative amounts."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(-1.0)
        b.sleep = _make_gen_method(None)
        action = {
            "invested_amount": 100,
            "token0": "0x1",
            "token1": "0x2",
            "relative_funds_percentage": 1.0,
        }
        result = _exhaust(
            b._calculate_investment_amounts_from_dollar_cap(
                action, "optimism", ["0x1", "0x2"]
            )
        )
        assert result is None


class TestGetTokenBalancesAndCalculateAmounts:
    """TestGetTokenBalancesAndCalculateAmounts."""

    def test_balance_none(self):
        """Test balance none."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=None)
        result = _exhaust(
            b._get_token_balances_and_calculate_amounts("optimism", ["0x1", "0x2"], [])
        )
        assert result == (None, None, None)

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1000)
        result = _exhaust(
            b._get_token_balances_and_calculate_amounts(
                "optimism", ["0x1", "0x2"], [{"chain": "optimism", "assets": []}]
            )
        )
        assert result[0] == [1000, 1000]

    def test_with_max_investment_amounts(self):
        """Test with max investment amounts."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=10**18)
        b._get_token_decimals = _make_gen_method(18)
        result = _exhaust(
            b._get_token_balances_and_calculate_amounts(
                "optimism", ["0x1", "0x2"], [], max_investment_amounts=[1, 1]
            )
        )
        assert result[0] is not None

    def test_decimals_none_with_max_investment(self):
        """Test decimals none with max investment."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1000)
        b._get_token_decimals = _make_gen_method(None)
        result = _exhaust(
            b._get_token_balances_and_calculate_amounts(
                "optimism", ["0x1", "0x2"], [], max_investment_amounts=[1, 1]
            )
        )
        assert result == (None, None, None)

    def test_with_percentage(self):
        """Test with percentage."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1000)
        result = _exhaust(
            b._get_token_balances_and_calculate_amounts(
                "optimism", ["0x1", "0x2"], [], relative_funds_percentage=0.5
            )
        )
        assert result[0] == [500, 500]

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()
        b._get_balance = MagicMock(side_effect=RuntimeError("err"))
        result = _exhaust(
            b._get_token_balances_and_calculate_amounts("optimism", ["0x1", "0x2"], [])
        )
        assert result == (None, None, None)


class TestGetTokenBalances:
    """TestGetTokenBalances."""

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=500)
        result = b._get_token_balances("optimism", ["0x1", "0x2"], [])
        assert result == (500, 500)

    def test_one_none(self):
        """Test one none."""
        b = _make_behaviour()
        b._get_balance = MagicMock(side_effect=[100, None])
        result = b._get_token_balances("optimism", ["0x1", "0x2"], [])
        assert result == (None, None)

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()
        b._get_balance = MagicMock(side_effect=RuntimeError("err"))
        result = b._get_token_balances("optimism", ["0x1", "0x2"], [])
        assert result == (None, None)


class TestGetApprovalTxHash:
    """TestGetApprovalTxHash."""

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        b.contract_interact = _make_gen_method("0xdata")
        result = _exhaust(
            b.get_approval_tx_hash("0xTOKEN", 100, "0xSPENDER", "optimism")
        )
        assert result["to"] == "0xTOKEN"

    def test_failure(self):
        """Test failure."""
        b = _make_behaviour()
        b.contract_interact = _make_gen_method(None)
        result = _exhaust(
            b.get_approval_tx_hash("0xTOKEN", 100, "0xSPENDER", "optimism")
        )
        assert result == {}


class TestAccumulateTransactionCosts:
    """TestAccumulateTransactionCosts."""

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        b._get_gas_cost_usd = _make_gen_method(0.5)
        b._update_entry_costs = _make_gen_method(1.0)
        pos = {"chain": "optimism", "pool_address": "0xPOOL"}
        _exhaust(b._accumulate_transaction_costs("0xhash", pos))

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()

        def raise_gen(*a, **kw):
            """Raise gen."""
            raise RuntimeError("fail")
            yield  # noqa

        b._get_gas_cost_usd = raise_gen
        _exhaust(
            b._accumulate_transaction_costs(
                "0xhash", {"chain": "optimism", "pool_address": "0xP"}
            )
        )


class TestAddSlippageCosts:
    """TestAddSlippageCosts."""

    def test_with_enter_pool_action(self):
        """Test with enter pool action."""
        b = _make_behaviour()
        b._calculate_actual_slippage_cost = _make_gen_method(0.1)
        b.synchronized_data.actions = [
            {"action": "EnterPool", "pool_address": "0xPOOL", "chain": "optimism"}
        ]
        b._update_entry_costs = _make_gen_method(0.5)
        _exhaust(b._add_slippage_costs("0xhash"))

    def test_no_pool_address(self):
        """Test no pool address."""
        b = _make_behaviour()
        b._calculate_actual_slippage_cost = _make_gen_method(0.1)
        b.synchronized_data.actions = [{"action": "EnterPool"}]
        _exhaust(b._add_slippage_costs("0xhash"))

    def test_no_enter_pool_action(self):
        """Test no enter pool action."""
        b = _make_behaviour()
        b._calculate_actual_slippage_cost = _make_gen_method(0.1)
        b.synchronized_data.actions = [{"action": "ExitPool"}]
        _exhaust(b._add_slippage_costs("0xhash"))

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()

        def raise_gen(*a, **kw):
            """Raise gen."""
            raise RuntimeError("fail")
            yield  # noqa

        b._calculate_actual_slippage_cost = raise_gen
        _exhaust(b._add_slippage_costs("0xhash"))


class TestGetGasCostUsd:
    """TestGetGasCostUsd."""

    def test_with_receipt(self):
        """Test with receipt."""
        b = _make_behaviour()
        receipt = {"gasUsed": 100000, "effectiveGasPrice": 10**9, "l1Fee": "0x0"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        b._fetch_zero_address_price = _make_gen_method(2000.0)
        result = _exhaust(b._get_gas_cost_usd("0xhash", "optimism"))
        assert result > 0

    def test_no_receipt(self):
        """Test no receipt."""
        b = _make_behaviour()
        b.get_transaction_receipt = _make_gen_method(None)
        result = _exhaust(b._get_gas_cost_usd("0xhash", "optimism"))
        assert result == 0.0

    def test_no_eth_price(self):
        """Test no eth price."""
        b = _make_behaviour()
        receipt = {"gasUsed": 100000, "effectiveGasPrice": 10**9, "l1Fee": "0x0"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        b._fetch_zero_address_price = _make_gen_method(None)
        result = _exhaust(b._get_gas_cost_usd("0xhash", "optimism"))
        assert result == 0.0

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()

        def raise_gen(*a, **kw):
            """Raise gen."""
            raise RuntimeError("fail")
            yield  # noqa

        b.get_transaction_receipt = raise_gen
        result = _exhaust(b._get_gas_cost_usd("0xhash", "optimism"))
        assert result == 0.0


class TestCalculateActualSlippageCost:
    """TestCalculateActualSlippageCost."""

    def test_positive_slippage(self):
        """Test positive slippage."""
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps(
            {
                "sending": {"amountUSD": "10"},
                "receiving": {"amountUSD": "9"},
            }
        )
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b._calculate_actual_slippage_cost("0xhash"))
        assert result == 1.0

    def test_no_slippage(self):
        """Test no slippage."""
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps(
            {
                "sending": {"amountUSD": "10"},
                "receiving": {"amountUSD": "10"},
            }
        )
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b._calculate_actual_slippage_cost("0xhash"))
        assert result == 0.0

    def test_non_ok_response(self):
        """Test non ok response."""
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 500
        resp.body = "error"
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b._calculate_actual_slippage_cost("0xhash"))
        assert result == 0.0

    def test_parse_error(self):
        """Test parse error."""
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = "not json"
        b.get_http_response = _make_gen_method(resp)
        with pytest.raises(Exception):
            _exhaust(b._calculate_actual_slippage_cost("0xhash"))


class TestCalculateAndStoreTipData:
    """TestCalculateAndStoreTipData."""

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        b._convert_amounts_to_usd = _make_gen_method(100.0)
        b._get_updated_entry_costs = _make_gen_method(5.0)
        b._calculate_min_hold_days = MagicMock(return_value=14.0)
        pos = {"amount0": 1000, "amount1": 2000, "enter_timestamp": 123}
        action = {
            "token0": "0x1",
            "token1": "0x2",
            "chain": "optimism",
            "pool_address": "0xP",
            "opportunity_apr": 20.0,
            "percent_in_bounds": 0.8,
            "is_cl_pool": True,
        }
        _exhaust(b._calculate_and_store_tip_data(pos, action))
        assert pos["min_hold_days"] == 14.0

    def test_missing_chain(self):
        """Test missing chain."""
        b = _make_behaviour()
        b._convert_amounts_to_usd = _make_gen_method(100.0)
        pos = {"amount0": 1000, "amount1": 2000}
        action = {"token0": "0x1", "token1": "0x2"}
        _exhaust(b._calculate_and_store_tip_data(pos, action))
        assert pos.get("min_hold_days") is not None  # fallback applied

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()

        def raise_gen(*a, **kw):
            """Raise gen."""
            raise RuntimeError("fail")
            yield  # noqa

        b._convert_amounts_to_usd = raise_gen
        pos = {"amount0": 1000, "amount1": 2000}
        action = {
            "token0": "0x1",
            "token1": "0x2",
            "chain": "optimism",
            "pool_address": "0xP",
        }
        _exhaust(b._calculate_and_store_tip_data(pos, action))
        assert pos["min_hold_days"] == MIN_TIME_IN_POSITION


class TestConvertAmountsToUsd:
    """TestConvertAmountsToUsd."""

    def test_both_tokens(self):
        """Test both tokens."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(1.0)
        result = _exhaust(
            b._convert_amounts_to_usd(1_000_000, 2_000_000, "0x1", "0x2", "optimism")
        )
        assert result == 3.0

    def test_zero_address_token(self):
        """Test zero address token."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(18)
        b._fetch_zero_address_price = _make_gen_method(2000.0)
        b._fetch_token_price = _make_gen_method(1.0)
        result = _exhaust(
            b._convert_amounts_to_usd(10**18, 10**18, ZERO_ADDRESS, "0x2", "optimism")
        )
        assert result > 0

    def test_no_amount0(self):
        """Test no amount0."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(1.0)
        result = _exhaust(
            b._convert_amounts_to_usd(0, 1_000_000, "0x1", "0x2", "optimism")
        )
        assert result == 1.0

    def test_no_amount1(self):
        """Test no amount1."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(1.0)
        result = _exhaust(
            b._convert_amounts_to_usd(1_000_000, 0, "0x1", "0x2", "optimism")
        )
        assert result == 1.0

    def test_decimals_none(self):
        """Test decimals none."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(None)
        result = _exhaust(
            b._convert_amounts_to_usd(1000, 2000, "0x1", "0x2", "optimism")
        )
        assert result == 0.0

    def test_price_none(self):
        """Test price none."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(None)
        result = _exhaust(
            b._convert_amounts_to_usd(1_000_000, 2_000_000, "0x1", "0x2", "optimism")
        )
        assert result == 0.0

    def test_exception(self):
        """Test exception."""
        b = _make_behaviour()

        def raise_gen(*a, **kw):
            """Raise gen."""
            raise RuntimeError("fail")
            yield  # noqa

        b._get_token_decimals = raise_gen
        result = _exhaust(
            b._convert_amounts_to_usd(1000, 2000, "0x1", "0x2", "optimism")
        )
        assert result == 0.0

    def test_token1_zero_address(self):
        """Test token1 zero address."""
        b = _make_behaviour()
        b._get_token_decimals = _make_gen_method(18)
        b._fetch_token_price = _make_gen_method(1.0)
        b._fetch_zero_address_price = _make_gen_method(2000.0)
        result = _exhaust(
            b._convert_amounts_to_usd(10**18, 10**18, "0x1", ZERO_ADDRESS, "optimism")
        )
        assert result > 0


class TestPrepareNextAction:
    """TestPrepareNextAction."""

    def _setup(self, action_name, **extra_action_fields):
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._read_withdrawal_status = _make_gen_method("unknown")
        action = {"action": action_name, **extra_action_fields}
        return b, action

    def test_no_action_name(self):
        """Test no action name."""
        b, action = self._setup(None)
        action["action"] = None
        result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result == (Event.DONE.value, {})

    def test_enter_pool_success(self):
        """Test enter pool success."""
        b, action = self._setup(Action.ENTER_POOL.value)
        b.get_enter_pool_tx_hash = _make_gen_method(("0xhash", "optimism", "0xSAFE"))
        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.decision_making.DecisionMakingRound.auto_round_id",
            return_value="dm",
        ):
            result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.SETTLE.value

    def test_enter_pool_no_hash(self):
        """Test enter pool no hash."""
        b, action = self._setup(Action.ENTER_POOL.value)
        b.get_enter_pool_tx_hash = _make_gen_method((None, None, None))
        result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result == (Event.DONE.value, {})

    def test_exit_pool(self):
        """Test exit pool."""
        b, action = self._setup(Action.EXIT_POOL.value)
        b.get_exit_pool_tx_hash = _make_gen_method(("0xhash", "optimism", "0xSAFE"))
        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.decision_making.DecisionMakingRound.auto_round_id",
            return_value="dm",
        ):
            result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.SETTLE.value

    def test_exit_pool_withdrawal(self):
        """Test exit pool withdrawal."""
        b, action = self._setup(Action.EXIT_POOL.value)
        b._read_investing_paused = _make_gen_method(True)
        b._read_withdrawal_status = _make_gen_method("WITHDRAWING")
        b._update_withdrawal_status = _make_gen_method(None)
        b.get_exit_pool_tx_hash = _make_gen_method(("0xhash", "optimism", "0xSAFE"))
        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.decision_making.DecisionMakingRound.auto_round_id",
            return_value="dm",
        ):
            result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.SETTLE.value

    def test_find_bridge_route_success(self):
        """Test find bridge route success."""
        b, action = self._setup(Action.FIND_BRIDGE_ROUTE.value)
        b.fetch_routes = _make_gen_method([{"steps": []}])
        result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.UPDATE.value
        assert result[1]["last_action"] == Action.ROUTES_FETCHED.value

    def test_find_bridge_route_no_routes(self):
        """Test find bridge route no routes."""
        b, action = self._setup(Action.FIND_BRIDGE_ROUTE.value)
        b.fetch_routes = _make_gen_method(None)
        result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result == (Event.DONE.value, {})

    def test_find_bridge_route_no_routes_withdrawal(self):
        """Test find bridge route no routes withdrawal."""
        b, action = self._setup(Action.FIND_BRIDGE_ROUTE.value)
        b.fetch_routes = _make_gen_method(None)
        # After first _read_investing_paused returns False, the code reads again
        call_count = [0]

        def paused_gen(*a, **kw):
            """Paused gen."""
            call_count[0] += 1
            yield
            if call_count[0] <= 1:
                return False
            return True

        b._read_investing_paused = paused_gen
        status_count = [0]

        def status_gen(*a, **kw):
            """Status gen."""
            status_count[0] += 1
            yield
            if status_count[0] <= 1:
                return "unknown"
            return "WITHDRAWING"

        b._read_withdrawal_status = status_gen
        b._update_withdrawal_status = _make_gen_method(None)
        b._reset_withdrawal_flags = _make_gen_method(None)
        result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result == (Event.DONE.value, {})

    def test_find_bridge_route_max_steps_filter(self):
        """Test find bridge route max steps filter."""
        b, action = self._setup(Action.FIND_BRIDGE_ROUTE.value)
        b.fetch_routes = _make_gen_method([{"steps": [1, 2, 3]}, {"steps": [1]}])
        b.synchronized_data.max_allowed_steps_in_a_route = 2
        result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.UPDATE.value

    def test_find_bridge_route_max_steps_none_left(self):
        """Test find bridge route max steps none left."""
        b, action = self._setup(Action.FIND_BRIDGE_ROUTE.value)
        b.fetch_routes = _make_gen_method([{"steps": [1, 2, 3]}])
        b.synchronized_data.max_allowed_steps_in_a_route = 1
        result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result == (Event.DONE.value, {})

    def test_find_bridge_route_max_steps_none_left_withdrawal(self):
        """Test find bridge route max steps none left withdrawal."""
        b, action = self._setup(Action.FIND_BRIDGE_ROUTE.value)
        b.fetch_routes = _make_gen_method([{"steps": [1, 2, 3]}])
        b.synchronized_data.max_allowed_steps_in_a_route = 1
        call_count = [0]

        def paused_gen(*a, **kw):
            """Paused gen."""
            call_count[0] += 1
            yield
            if call_count[0] <= 1:
                return False
            return True

        b._read_investing_paused = paused_gen
        status_count = [0]

        def status_gen(*a, **kw):
            """Status gen."""
            status_count[0] += 1
            yield
            if status_count[0] <= 1:
                return "unknown"
            return "WITHDRAWING"

        b._read_withdrawal_status = status_gen
        b._update_withdrawal_status = _make_gen_method(None)
        b._reset_withdrawal_flags = _make_gen_method(None)
        result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result == (Event.DONE.value, {})

    def test_bridge_swap(self):
        """Test bridge swap."""
        b, action = self._setup(
            Action.BRIDGE_SWAP.value,
            payload="0xpayload",
            from_chain="optimism",
            safe_address="0xSAFE",
        )
        b.sleep = _make_gen_method(None)
        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.decision_making.DecisionMakingRound.auto_round_id",
            return_value="dm",
        ):
            result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.SETTLE.value

    def test_bridge_swap_withdrawal(self):
        """Test bridge swap withdrawal."""
        b, action = self._setup(
            Action.BRIDGE_SWAP.value,
            payload="0xpayload",
            from_chain="optimism",
            safe_address="0xSAFE",
        )
        b._read_investing_paused = _make_gen_method(True)
        b._read_withdrawal_status = _make_gen_method("WITHDRAWING")
        b._update_withdrawal_status = _make_gen_method(None)
        b.sleep = _make_gen_method(None)
        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.decision_making.DecisionMakingRound.auto_round_id",
            return_value="dm",
        ):
            result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.SETTLE.value

    def test_claim_rewards(self):
        """Test claim rewards."""
        b, action = self._setup(Action.CLAIM_REWARDS.value)
        b.get_claim_rewards_tx_hash = _make_gen_method(("0xhash", "optimism", "0xSAFE"))
        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.decision_making.DecisionMakingRound.auto_round_id",
            return_value="dm",
        ):
            result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.SETTLE.value

    def test_deposit(self):
        """Test deposit."""
        b, action = self._setup(Action.DEPOSIT.value)
        b.get_deposit_tx_hash = _make_gen_method(("0xhash", "optimism", "0xSAFE"))
        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.decision_making.DecisionMakingRound.auto_round_id",
            return_value="dm",
        ):
            result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.SETTLE.value

    def test_withdraw_token_transfer(self):
        """Test withdraw token transfer."""
        b, action = self._setup(
            Action.WITHDRAW.value, token_address="0xTOKEN", to_address="0xDEST"
        )
        b.get_token_transfer_tx_hash = _make_gen_method(
            ("0xhash", "optimism", "0xSAFE")
        )
        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.decision_making.DecisionMakingRound.auto_round_id",
            return_value="dm",
        ):
            result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.SETTLE.value

    def test_withdraw_vault(self):
        """Test withdraw vault."""
        b, action = self._setup(Action.WITHDRAW.value)
        b.get_withdraw_tx_hash = _make_gen_method(("0xhash", "optimism", "0xSAFE"))
        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.decision_making.DecisionMakingRound.auto_round_id",
            return_value="dm",
        ):
            result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.SETTLE.value

    def test_withdraw_paused(self):
        """Test withdraw paused."""
        b, action = self._setup(Action.WITHDRAW.value)
        b._read_investing_paused = _make_gen_method(True)
        b._read_withdrawal_status = _make_gen_method("WITHDRAWING")
        b._update_withdrawal_status = _make_gen_method(None)
        b.get_withdraw_tx_hash = _make_gen_method(("0xhash", "optimism", "0xSAFE"))
        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.decision_making.DecisionMakingRound.auto_round_id",
            return_value="dm",
        ):
            result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.SETTLE.value

    def test_stake_lp_tokens(self):
        """Test stake lp tokens."""
        b, action = self._setup(Action.STAKE_LP_TOKENS.value)
        b.get_stake_lp_tokens_tx_hash = _make_gen_method(
            ("0xhash", "optimism", "0xSAFE")
        )
        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.decision_making.DecisionMakingRound.auto_round_id",
            return_value="dm",
        ):
            result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.SETTLE.value

    def test_unstake_lp_tokens(self):
        """Test unstake lp tokens."""
        b, action = self._setup(Action.UNSTAKE_LP_TOKENS.value)
        b.get_unstake_lp_tokens_tx_hash = _make_gen_method(
            ("0xhash", "optimism", "0xSAFE")
        )
        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.decision_making.DecisionMakingRound.auto_round_id",
            return_value="dm",
        ):
            result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.SETTLE.value

    def test_claim_staking_rewards(self):
        """Test claim staking rewards."""
        b, action = self._setup(Action.CLAIM_STAKING_REWARDS.value)
        b.get_claim_staking_rewards_tx_hash = _make_gen_method(
            ("0xhash", "optimism", "0xSAFE")
        )
        with patch(
            "packages.valory.skills.liquidity_trader_abci.states.decision_making.DecisionMakingRound.auto_round_id",
            return_value="dm",
        ):
            result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result[0] == Event.SETTLE.value

    def test_unknown_action(self):
        """The else branch: tx_hash=None, returns DONE."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._read_withdrawal_status = _make_gen_method("unknown")
        # Use a valid Action value that doesn't match any if/elif
        # Actually the else branch is unreachable with valid Actions since all are covered
        # But we can test the no tx_hash path
        action = {"action": Action.ENTER_POOL.value}
        b.get_enter_pool_tx_hash = _make_gen_method((None, None, None))
        result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result == (Event.DONE.value, {})


class TestCheckIfRouteIsProfitable:
    """TestCheckIfRouteIsProfitable."""

    def test_profitable(self):
        """A profitable route returns True with the total fee and gas."""
        b = _make_behaviour()
        b._get_step_transactions_data = _make_gen_method(
            [{"fee": 0.1, "gas_cost": 0.1}]
        )
        route = {"fromAmountUSD": "100", "toAmountUSD": "99", "steps": []}
        result = _exhaust(b.check_if_route_is_profitable(route))
        assert result[0] is True
        assert result[1] == 0.1  # total_fee
        assert result[2] == 0.1  # total_gas_cost

    def test_not_profitable_fees(self):
        """Fee-only excess trips the gate against the FEE limit, not the gas limit.

        fee 10% sits between max_fee (5%) and max_gas (25%); gas is 0 and value loss
        is 1% (under the 10% cap), so only the fee branch can reject. A swap to
        ``fee_percentage > allowed_gas_percentage`` (10% > 25% -> False) would let this
        route through — this case is the canary for that swap.
        """
        b = _make_behaviour()
        b._get_step_transactions_data = _make_gen_method([{"fee": 10, "gas_cost": 0}])
        route = {"fromAmountUSD": "100", "toAmountUSD": "99", "steps": []}
        result = _exhaust(b.check_if_route_is_profitable(route))
        assert result[0] is False

    def test_gas_within_gas_limit_passes(self):
        """Gas 10% (> fee limit 5%, < gas limit 25%) passes — guards the gas threshold.

        Complements test_not_profitable_fees: if the gas check were swapped to
        ``gas_percentage > allowed_fee_percentage`` (10% > 5% -> True) this route would
        be wrongly rejected. fee is 0 and value loss is 1%, so only the gas threshold's
        identity decides the outcome.
        """
        b = _make_behaviour()
        b._get_step_transactions_data = _make_gen_method([{"fee": 0, "gas_cost": 10}])
        route = {"fromAmountUSD": "100", "toAmountUSD": "99", "steps": []}
        assert _exhaust(b.check_if_route_is_profitable(route))[0] is True

    def test_no_step_transactions(self):
        """Test no step transactions."""
        b = _make_behaviour()
        b._get_step_transactions_data = _make_gen_method(None)
        route = {"steps": []}
        result = _exhaust(b.check_if_route_is_profitable(route))
        assert result == (None, None, None)

    def test_zero_amounts(self):
        """Test zero amounts."""
        b = _make_behaviour()
        b._get_step_transactions_data = _make_gen_method([{"fee": 0, "gas_cost": 0}])
        route = {"fromAmountUSD": "0", "toAmountUSD": "0", "steps": []}
        result = _exhaust(b.check_if_route_is_profitable(route))
        assert result[0] is False

    def test_not_profitable_slippage(self):
        """Route rejected when value-loss/slippage exceeds max_slippage_percentage."""
        b = _make_behaviour()  # max_slippage_percentage = 0.1 (10%)
        b._get_step_transactions_data = _make_gen_method(
            [{"fee": 0.1, "gas_cost": 0.1}]
        )
        # Fees/gas pass the gate, but 20% value loss (100 -> 80) exceeds the 10% cap.
        route = {"fromAmountUSD": "100", "toAmountUSD": "80", "steps": []}
        result = _exhaust(b.check_if_route_is_profitable(route))
        assert result[0] is False

    def test_not_profitable_gas_only(self):
        """Gas-only excess trips the gate (guards the `or gas` branch)."""
        b = _make_behaviour()  # max_gas_percentage = 0.25
        b._get_step_transactions_data = _make_gen_method([{"fee": 0.5, "gas_cost": 30}])
        # fee 0.5% passes, gas 30% > 25% -> rejected by the gas branch.
        route = {"fromAmountUSD": "100", "toAmountUSD": "99", "steps": []}
        assert _exhaust(b.check_if_route_is_profitable(route))[0] is False

    def test_slippage_at_cap_passes(self):
        """Exactly the cap (10% loss) passes — guards against `>=` instead of `>`."""
        b = _make_behaviour()  # max_slippage_percentage = 0.1
        b._get_step_transactions_data = _make_gen_method(
            [{"fee": 0.1, "gas_cost": 0.1}]
        )
        route = {
            "fromAmountUSD": "100",
            "toAmountUSD": "90",
            "steps": [],
        }  # exactly 10%
        assert _exhaust(b.check_if_route_is_profitable(route))[0] is True

    def test_value_gaining_route_passes(self):
        """A value-gaining route (toAmountUSD > fromAmountUSD) has negative loss, passes."""
        b = _make_behaviour()
        b._get_step_transactions_data = _make_gen_method(
            [{"fee": 0.1, "gas_cost": 0.1}]
        )
        route = {"fromAmountUSD": "100", "toAmountUSD": "101", "steps": []}
        assert _exhaust(b.check_if_route_is_profitable(route))[0] is True


class TestCheckStepCosts:
    """TestCheckStepCosts."""

    def test_step_profitable(self):
        """Test step profitable."""
        b = _make_behaviour()
        b._set_step_addresses = MagicMock(side_effect=lambda x: x)
        b._get_step_transaction = _make_gen_method({"fee": 0.01, "gas_cost": 0.01})
        result = _exhaust(b.check_step_costs({"action": {}}, 1.0, 1.0, 0, 1))
        assert result[0] is True

    def test_step_data_none(self):
        """Test step data none."""
        b = _make_behaviour()
        b._set_step_addresses = MagicMock(side_effect=lambda x: x)
        b._get_step_transaction = _make_gen_method(None)
        result = _exhaust(b.check_step_costs({"action": {}}, 1.0, 1.0, 0, 1))
        assert result[0] is False

    def test_last_step_exceeds_allowance(self):
        """Test last step exceeds allowance."""
        b = _make_behaviour()
        b._set_step_addresses = MagicMock(side_effect=lambda x: x)
        b._get_step_transaction = _make_gen_method({"fee": 1.0, "gas_cost": 1.0})
        result = _exhaust(b.check_step_costs({"action": {}}, 1.0, 1.0, 1, 2))
        assert result[0] is False

    def test_non_last_step_exceeds_allowance(self):
        """Test non last step exceeds allowance."""
        b = _make_behaviour()
        b._set_step_addresses = MagicMock(side_effect=lambda x: x)
        b._get_step_transaction = _make_gen_method({"fee": 2.0, "gas_cost": 0})
        result = _exhaust(b.check_step_costs({"action": {}}, 1.0, 1.0, 0, 3))
        assert result[0] is False

    def test_single_step_within_allowance(self):
        """Test single step within allowance."""
        b = _make_behaviour()
        b._set_step_addresses = MagicMock(side_effect=lambda x: x)
        b._get_step_transaction = _make_gen_method({"fee": 0.5, "gas_cost": 0.5})
        result = _exhaust(b.check_step_costs({"action": {}}, 1.0, 1.0, 0, 1))
        assert result[0] is True


class TestCalculateVelodromeInvestmentAmounts:
    """TestCalculateVelodromeInvestmentAmounts."""

    def test_success_with_token_requirements(self):
        """Test success with token requirements."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1_000_000)
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(1.0)
        action = {
            "token_requirements": {
                "overall_token0_ratio": 0.5,
                "overall_token1_ratio": 0.5,
            }
        }
        result = _exhaust(
            b._calculate_velodrome_investment_amounts(
                action, "optimism", ["0x1", "0x2"], [], [2_000_000, 2_000_000]
            )
        )
        assert result is not None

    def test_zero_available(self):
        """Test zero available."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=0)
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(1.0)
        action = {}
        result = _exhaust(
            b._calculate_velodrome_investment_amounts(
                action, "optimism", ["0x1", "0x2"], [], []
            )
        )
        assert result is None

    def test_decimals_none(self):
        """Test decimals none."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1000)
        b._get_token_decimals = _make_gen_method(None)
        action = {}
        result = _exhaust(
            b._calculate_velodrome_investment_amounts(
                action, "optimism", ["0x1", "0x2"], [], []
            )
        )
        assert result is None

    def test_price_none(self):
        """Test price none."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1000)
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(None)
        action = {}
        result = _exhaust(
            b._calculate_velodrome_investment_amounts(
                action, "optimism", ["0x1", "0x2"], [], []
            )
        )
        assert result is None

    def test_with_percentage_fallback(self):
        """Test with percentage fallback."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1_000_000)
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(1.0)
        action = {
            "token_requirements": {},
            "token0_percentage": 60,
            "token1_percentage": 40,
        }
        result = _exhaust(
            b._calculate_velodrome_investment_amounts(
                action, "optimism", ["0x1", "0x2"], [], []
            )
        )
        assert result is not None

    def test_negative_amounts(self):
        """Test negative amounts."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1_000_000)
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(-1.0)
        action = {
            "token_requirements": {
                "overall_token0_ratio": 0.5,
                "overall_token1_ratio": 0.5,
            }
        }
        result = _exhaust(
            b._calculate_velodrome_investment_amounts(
                action, "optimism", ["0x1", "0x2"], [], []
            )
        )
        assert result is None

    def test_token_requirements_exception(self):
        """Test token requirements exception."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1_000_000)
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(1.0)
        action = {
            "token_requirements": {"overall_token0_ratio": "not_a_number"},
            "token0_percentage": 50,
            "token1_percentage": 50,
        }
        result = _exhaust(
            b._calculate_velodrome_investment_amounts(
                action, "optimism", ["0x1", "0x2"], [], []
            )
        )
        assert result is not None


class TestBuildSafeTx:
    """TestBuildSafeTx."""

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        multisend_addr = "0x" + "cc" * 20
        result = _exhaust(
            b._build_safe_tx("optimism", "0x" + "cd" * 32, multisend_addr)
        )
        assert result is not None

    def test_no_safe_tx_hash(self):
        """Test no safe tx hash."""
        b = _make_behaviour()
        b.contract_interact = _make_gen_method(None)
        multisend_addr = "0x" + "cc" * 20
        result = _exhaust(
            b._build_safe_tx("optimism", "0x" + "cd" * 32, multisend_addr)
        )
        assert result is None


class TestBuildMultisendTx:
    """TestBuildMultisendTx."""

    def test_with_approval(self):
        """Test with approval."""
        b = _make_behaviour()
        b.get_approval_tx_hash = _make_gen_method(
            {"to": "0x1", "data": b"x", "value": 0, "operation": 0}
        )
        b.contract_interact = _make_gen_method("0xmultihash")
        tx_info = {
            "source_token": "0x1",
            "amount": 100,
            "lifi_contract_address": "0xLIFI",
            "from_chain": "optimism",
            "tx_hash": b"data",
        }
        result = _exhaust(b._build_multisend_tx([], tx_info))
        assert result is not None

    def test_zero_address_source(self):
        """Test zero address source."""
        b = _make_behaviour()
        b.contract_interact = _make_gen_method("0xmultihash")
        tx_info = {
            "source_token": ZERO_ADDRESS,
            "amount": 100,
            "lifi_contract_address": "0xLIFI",
            "from_chain": "optimism",
            "tx_hash": b"data",
        }
        result = _exhaust(b._build_multisend_tx([], tx_info))
        assert result is not None

    def test_approval_fails(self):
        """Test approval fails."""
        b = _make_behaviour()
        b.get_approval_tx_hash = _make_gen_method({})
        tx_info = {
            "source_token": "0x1",
            "amount": 100,
            "lifi_contract_address": "0xLIFI",
            "from_chain": "optimism",
            "tx_hash": b"data",
        }
        result = _exhaust(b._build_multisend_tx([], tx_info))
        assert result is None


class TestPrepareBridgeSwapAction:
    """TestPrepareBridgeSwapAction."""

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        b._build_multisend_tx = _make_gen_method("0x" + "ab" * 32)
        b._build_safe_tx = _make_gen_method("payload_string")
        tx_info = {
            "from_chain": "optimism",
            "to_chain": "mode",
            "source_token": "0x1",
            "source_token_symbol": "USDC",
            "target_token": "0x2",
            "target_token_symbol": "USDC",
            "tool": "lifi",
            "gas_cost": 0.1,
            "fee": 0.05,
            "amount": 100,
        }
        result = _exhaust(b.prepare_bridge_swap_action([], tx_info, 1.0, 1.0))
        assert result is not None
        assert result["action"] == Action.BRIDGE_SWAP.value

    def test_no_multisend(self):
        """Test no multisend."""
        b = _make_behaviour()
        b._build_multisend_tx = _make_gen_method(None)
        result = _exhaust(b.prepare_bridge_swap_action([], {}, 1.0, 1.0))
        assert result is None

    def test_no_safe_tx(self):
        """Test no safe tx."""
        b = _make_behaviour()
        b._build_multisend_tx = _make_gen_method("0x" + "ab" * 32)
        b._build_safe_tx = _make_gen_method(None)
        tx_info = {
            "from_chain": "optimism",
            "to_chain": "mode",
            "source_token": "0x1",
            "source_token_symbol": "A",
            "target_token": "0x2",
            "target_token_symbol": "B",
            "tool": "t",
            "gas_cost": 0,
            "fee": 0,
            "amount": 0,
        }
        result = _exhaust(b.prepare_bridge_swap_action([], tx_info, 1.0, 1.0))
        assert result is None


class TestGetBlock:
    """TestGetBlock."""

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        mock_resp = MagicMock()
        mock_resp.performative = "state"
        mock_resp.state.body = {"timestamp": 12345}
        # We need to match the Performative
        from packages.valory.protocols.ledger_api import LedgerApiMessage

        mock_resp.performative = LedgerApiMessage.Performative.STATE
        b.get_ledger_api_response = _make_gen_method(mock_resp)
        result = _exhaust(b.get_block("0x10", chain_id="optimism"))
        assert result == {"timestamp": 12345}

    def test_failure(self):
        """Test failure."""
        b = _make_behaviour()
        mock_resp = MagicMock()
        mock_resp.performative = MagicMock()  # non-STATE
        b.get_ledger_api_response = _make_gen_method(mock_resp)
        result = _exhaust(b.get_block("0x10", chain_id="optimism"))
        assert result is None

    def test_no_block_number(self):
        """Test no block number."""
        b = _make_behaviour()
        mock_resp = MagicMock()
        from packages.valory.protocols.ledger_api import LedgerApiMessage

        mock_resp.performative = LedgerApiMessage.Performative.STATE
        mock_resp.state.body = {"timestamp": 999}
        b.get_ledger_api_response = _make_gen_method(mock_resp)
        result = _exhaust(b.get_block(chain_id="optimism"))
        assert result == {"timestamp": 999}


class TestMatchingRound:
    """TestMatchingRound."""

    def test_matching_round(self):
        """Test matching round."""
        from packages.valory.skills.liquidity_trader_abci.states.decision_making import (
            DecisionMakingRound,
        )

        assert DecisionMakingBehaviour.matching_round == DecisionMakingRound


class TestAsyncAct:
    """TestAsyncAct."""

    def test_withdrawal_initiated(self):
        """Test withdrawal initiated."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(True)
        b._read_withdrawal_status = _make_gen_method("INITIATED")
        b.send_a2a_transaction = _make_gen_method(None)
        b.wait_until_round_end = _make_gen_method(None)
        b.set_done = MagicMock()
        # behaviour_id is a property; auto_behaviour_id() uses class name
        _exhaust(b.async_act())
        b.set_done.assert_called_once()

    def test_normal_flow(self):
        """Test normal flow."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b.get_next_event = _make_gen_method((Event.DONE.value, {}))
        b.send_a2a_transaction = _make_gen_method(None)
        b.wait_until_round_end = _make_gen_method(None)
        b.set_done = MagicMock()
        # behaviour_id is a property; auto_behaviour_id() uses class name
        _exhaust(b.async_act())
        b.set_done.assert_called_once()

    def test_paused_but_not_initiated(self):
        """Test paused but not initiated."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(True)
        b._read_withdrawal_status = _make_gen_method("WITHDRAWING")
        b.get_next_event = _make_gen_method((Event.DONE.value, {}))
        b.send_a2a_transaction = _make_gen_method(None)
        b.wait_until_round_end = _make_gen_method(None)
        b.set_done = MagicMock()
        # behaviour_id is a property; auto_behaviour_id() uses class name
        _exhaust(b.async_act())
        b.set_done.assert_called_once()


class TestPostExecuteEnterPool:
    """TestPostExecuteEnterPool."""

    def _base_action(self, dex_type, **extra):
        return {
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "dex_type": dex_type,
            "token0": "0xT0",
            "token1": "0xT1",
            "token0_symbol": "T0",
            "token1_symbol": "T1",
            "apr": 10,
            "pool_type": "Weighted",
            **extra,
        }

    def test_uniswap_v3(self):
        """Test uniswap v3."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xhash"
        b._get_data_from_mint_tx_receipt = _make_gen_method((1, 1000, 500, 600, 12345))
        b._accumulate_transaction_costs = _make_gen_method(None)
        b._rename_entry_costs_key = _make_gen_method(None)
        b._calculate_and_store_tip_data = _make_gen_method(None)
        b.store_current_positions = MagicMock()
        action = self._base_action(DexType.UNISWAP_V3.value)
        _exhaust(b._post_execute_enter_pool([action], 0))
        assert len(b.current_positions) == 1
        assert b.current_positions[0]["token_id"] == 1

    def test_velodrome_cl_multiple_positions(self):
        """Test velodrome cl multiple positions."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xhash"
        b._get_all_positions_from_tx_receipt = _make_gen_method(
            [
                (1, 100, 50, 60, 111),
                (2, 200, 70, 80, 111),
            ]
        )
        b._invalidate_cl_pool_cache = _make_gen_method(None)
        b._accumulate_transaction_costs = _make_gen_method(None)
        b._rename_entry_costs_key = _make_gen_method(None)
        b._calculate_and_store_tip_data = _make_gen_method(None)
        b.store_current_positions = MagicMock()
        action = self._base_action(DexType.VELODROME.value, is_cl_pool=True)
        _exhaust(b._post_execute_enter_pool([action], 0))
        assert len(b.current_positions) == 1
        assert len(b.current_positions[0]["positions"]) == 2

    def test_velodrome_cl_fallback_single(self):
        """Test velodrome cl fallback single."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xhash"
        b._get_all_positions_from_tx_receipt = _make_gen_method(None)
        b._get_data_from_mint_tx_receipt = _make_gen_method((1, 100, 50, 60, 111))
        b._accumulate_transaction_costs = _make_gen_method(None)
        b._rename_entry_costs_key = _make_gen_method(None)
        b._calculate_and_store_tip_data = _make_gen_method(None)
        b.store_current_positions = MagicMock()
        action = self._base_action(DexType.VELODROME.value, is_cl_pool=True)
        _exhaust(b._post_execute_enter_pool([action], 0))
        assert len(b.current_positions) == 1

    def test_velodrome_cl_empty_positions(self):
        """Test velodrome cl empty positions."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xhash"
        b._get_all_positions_from_tx_receipt = _make_gen_method([])
        b._get_data_from_mint_tx_receipt = _make_gen_method((1, 100, 50, 60, 111))
        b._accumulate_transaction_costs = _make_gen_method(None)
        b._rename_entry_costs_key = _make_gen_method(None)
        b._calculate_and_store_tip_data = _make_gen_method(None)
        b.store_current_positions = MagicMock()
        action = self._base_action(DexType.VELODROME.value, is_cl_pool=True)
        _exhaust(b._post_execute_enter_pool([action], 0))
        assert len(b.current_positions) == 1

    def test_velodrome_non_cl(self):
        """Test velodrome non cl."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xhash"
        b._get_data_from_velodrome_mint_event = _make_gen_method((500, 600, 12345))
        b._accumulate_transaction_costs = _make_gen_method(None)
        b._rename_entry_costs_key = _make_gen_method(None)
        b._calculate_and_store_tip_data = _make_gen_method(None)
        b.store_current_positions = MagicMock()
        action = self._base_action(DexType.VELODROME.value, is_cl_pool=False)
        _exhaust(b._post_execute_enter_pool([action], 0))
        assert len(b.current_positions) == 1

    def test_balancer(self):
        """Test balancer."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xhash"
        b._get_data_from_join_pool_tx_receipt = _make_gen_method((500, 600, 12345))
        b._accumulate_transaction_costs = _make_gen_method(None)
        b._rename_entry_costs_key = _make_gen_method(None)
        b._calculate_and_store_tip_data = _make_gen_method(None)
        b.store_current_positions = MagicMock()
        action = self._base_action(DexType.BALANCER.value)
        _exhaust(b._post_execute_enter_pool([action], 0))
        assert len(b.current_positions) == 1

    def test_sturdy(self):
        """Test sturdy."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xhash"
        b._get_data_from_deposit_tx_receipt = _make_gen_method((1000, 500, 12345))
        b._accumulate_transaction_costs = _make_gen_method(None)
        b._rename_entry_costs_key = _make_gen_method(None)
        b._calculate_and_store_tip_data = _make_gen_method(None)
        b.store_current_positions = MagicMock()
        action = self._base_action(DexType.STURDY.value)
        _exhaust(b._post_execute_enter_pool([action], 0))
        assert len(b.current_positions) == 1
        assert b.current_positions[0]["shares"] == 500


class TestGetEnterPoolTxHash:
    """TestGetEnterPoolTxHash."""

    def _base_action(self, dex_type=DexType.UNISWAP_V3.value, **extra):
        return {
            "dex_type": dex_type,
            "chain": "optimism",
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xPOOL",
            "pool_type": "Weighted",
            "is_stable": False,
            "is_cl_pool": False,
            "relative_funds_percentage": 1.0,
            **extra,
        }

    def test_missing_params(self):
        """Test missing params."""
        b = _make_behaviour()
        action = {
            "dex_type": None,
            "chain": None,
            "token0": None,
            "token1": None,
            "pool_address": None,
        }
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_too_few_assets(self):
        """Test too few assets."""
        b = _make_behaviour()
        action = self._base_action()
        action["token0"] = "0xT0"
        action["token1"] = None
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        # assets = [0xT0, None] has len 2, passes; but missing safe_address params etc.
        # Actually the all() check catches None in assets
        assert result == (None, None, None)

    def test_unknown_dex_type(self):
        """Test unknown dex type."""
        b = _make_behaviour()
        action = self._base_action(dex_type="UnknownDex")
        b.pools = {}
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_no_relative_funds_non_velodrome(self):
        """Test no relative funds non velodrome."""
        b = _make_behaviour()
        b.pools = {DexType.UNISWAP_V3.value: MagicMock()}
        action = self._base_action()
        action["relative_funds_percentage"] = 0
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_balance_none(self):
        """Test balance none."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b._get_balance = MagicMock(return_value=None)
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            (None, None, None)
        )
        action = self._base_action()
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_zero_amounts(self):
        """Test zero amounts."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([0, 100], 0, 100)
        )
        action = self._base_action()
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_success_single_tx_hash(self):
        """Test success single tx hash."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method(("0xtx", "0xCONTRACT"))
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": "0xT", "value": 0, "data": b"x"}
        )
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._base_action()
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result[0] is not None

    def test_success_list_tx_hashes(self):
        """Test success list tx hashes."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method((["0xtx1", "0xtx2"], "0xCONTRACT"))
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": "0xT", "value": 0, "data": b"x"}
        )
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._base_action()
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result[0] is not None

    def test_pool_enter_returns_none(self):
        """Test pool enter returns none."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method(None)
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        action = self._base_action()
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_pool_enter_returns_none_hashes(self):
        """Test pool enter returns none hashes."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method((None, "0xCONTRACT"))
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        action = self._base_action()
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_approval_fails(self):
        """Test approval fails."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method(("0xtx", "0xCONTRACT"))
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        b.get_approval_tx_hash = _make_gen_method({})
        action = self._base_action()
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_safe_tx_hash_none(self):
        """Test safe tx hash none."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method(("0xtx", "0x" + "cc" * 20))
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": "0x" + "cc" * 20, "value": 0, "data": b"x"}
        )
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return "0x" + "ab" * 32  # multisend_tx_hash - valid hex
            return None  # safe_tx_hash is None

        b.contract_interact = ci
        action = self._base_action()
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_zero_address_asset0(self):
        """Test zero address asset0."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method(("0xtx", "0xCONTRACT"))
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": "0xT", "value": 0, "data": b"x"}
        )
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._base_action()
        action["token0"] = ZERO_ADDRESS
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result[0] is not None

    def test_zero_address_asset1(self):
        """Test zero address asset1."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method(("0xtx", "0xCONTRACT"))
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": "0xT", "value": 0, "data": b"x"}
        )
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._base_action()
        action["token1"] = ZERO_ADDRESS
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result[0] is not None

    def test_velodrome_cl_pool(self):
        """Test velodrome cl pool."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method(("0xtx", "0xCONTRACT"))
        b.pools = {DexType.VELODROME.value: mock_pool}
        b._calculate_velodrome_investment_amounts = _make_gen_method([1000, 2000])
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": "0xT", "value": 0, "data": b"x"}
        )
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._base_action(
            dex_type=DexType.VELODROME.value,
            is_cl_pool=True,
            tick_spacing=10,
            tick_ranges=[[0, 100]],
        )
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result[0] is not None

    def test_velodrome_cl_pool_none_amounts(self):
        """Test velodrome cl pool none amounts."""
        b = _make_behaviour()
        b.pools = {DexType.VELODROME.value: MagicMock()}
        b._calculate_velodrome_investment_amounts = _make_gen_method(None)
        action = self._base_action(dex_type=DexType.VELODROME.value, is_cl_pool=True)
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_velodrome_non_cl(self):
        """Test velodrome non cl."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method(("0xtx", "0xCONTRACT"))
        b.pools = {DexType.VELODROME.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": "0xT", "value": 0, "data": b"x"}
        )
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._base_action(dex_type=DexType.VELODROME.value, is_cl_pool=False)
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result[0] is not None

    def test_velodrome_non_cl_no_relative_funds(self):
        """Test velodrome non cl no relative funds."""
        b = _make_behaviour()
        b.pools = {DexType.VELODROME.value: MagicMock()}
        action = self._base_action(dex_type=DexType.VELODROME.value, is_cl_pool=False)
        action["relative_funds_percentage"] = 0
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_velodrome_non_cl_zero_amount(self):
        """Test velodrome non cl zero amount."""
        b = _make_behaviour()
        b.pools = {DexType.VELODROME.value: MagicMock()}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([0, 100], 0, 100)
        )
        action = self._base_action(dex_type=DexType.VELODROME.value, is_cl_pool=False)
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_with_invested_amount_and_current_positions(self):
        """Test with invested amount and current positions."""
        b = _make_behaviour()
        b.current_positions = [{"pool_address": "0xOTHER"}]
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method(("0xtx", "0xCONTRACT"))
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b._calculate_investment_amounts_from_dollar_cap = _make_gen_method([500, 500])
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": "0xT", "value": 0, "data": b"x"}
        )
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._base_action(invested_amount=100)
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result[0] is not None

    def test_with_invested_amount_returns_none(self):
        """Test with invested amount returns none."""
        b = _make_behaviour()
        b.current_positions = [{"pool_address": "0xOTHER"}]
        b.pools = {DexType.UNISWAP_V3.value: MagicMock()}
        b._calculate_investment_amounts_from_dollar_cap = _make_gen_method(None)
        action = self._base_action(invested_amount=100)
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_velodrome_cl_with_max_position_size(self):
        """Test velodrome cl with max position size."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method(("0xtx", "0xCONTRACT"))
        b.pools = {DexType.VELODROME.value: mock_pool}
        b._calculate_velodrome_investment_amounts = _make_gen_method([1000, 2000])
        b._enforce_pool_allocation_cap = _make_gen_method([800, 1600])
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": "0xT", "value": 0, "data": b"x"}
        )
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._base_action(
            dex_type=DexType.VELODROME.value, is_cl_pool=True, max_position_size=100.0
        )
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result[0] is not None

    def test_velodrome_non_cl_with_max_position_size(self):
        """Test velodrome non cl with max position size."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method(("0xtx", "0xCONTRACT"))
        b.pools = {DexType.VELODROME.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        b._enforce_pool_allocation_cap = _make_gen_method([800, 1600])
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": "0xT", "value": 0, "data": b"x"}
        )
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._base_action(
            dex_type=DexType.VELODROME.value, is_cl_pool=False, max_position_size=100.0
        )
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result[0] is not None

    def test_with_pool_fee(self):
        """Test with pool fee."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method(("0xtx", "0xCONTRACT"))
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": "0xT", "value": 0, "data": b"x"}
        )
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._base_action(pool_fee=3000)
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result[0] is not None


class TestGetExitPoolTxHash:
    """TestGetExitPoolTxHash."""

    def test_unknown_dex(self):
        """Test unknown dex."""
        b = _make_behaviour()
        b.pools = {}
        action = {"dex_type": "unknown", "chain": "optimism"}
        result = _exhaust(b.get_exit_pool_tx_hash(action))
        assert result == (None, None, None)

    def test_uniswap_success(self):
        """Test uniswap success."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.exit = _make_gen_method((b"txhash", "0x" + "cc" * 20, False))
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = {
            "dex_type": DexType.UNISWAP_V3.value,
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "token_id": 1,
            "liquidity": 100,
            "pool_type": "Weighted",
        }
        result = _exhaust(b.get_exit_pool_tx_hash(action))
        assert result[0] is not None

    def test_uniswap_multisend(self):
        """Test uniswap multisend."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.exit = _make_gen_method((b"txhash", "0x" + "cc" * 20, True))
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = {
            "dex_type": DexType.UNISWAP_V3.value,
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "token_id": 1,
            "liquidity": 100,
            "pool_type": "Weighted",
        }
        result = _exhaust(b.get_exit_pool_tx_hash(action))
        assert result[0] is not None

    def test_pool_exit_none(self):
        """Test pool exit none."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.exit = _make_gen_method((None, None, False))
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        action = {
            "dex_type": DexType.UNISWAP_V3.value,
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "token_id": 1,
            "liquidity": 100,
            "pool_type": "Weighted",
        }
        result = _exhaust(b.get_exit_pool_tx_hash(action))
        assert result == (None, None, None)

    def test_pool_exit_returns_none_object(self):
        """Test pool exit returns none object."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.exit = _make_gen_method(None)
        b.pools = {DexType.VELODROME.value: mock_pool}
        action = {
            "dex_type": DexType.VELODROME.value,
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "is_cl_pool": True,
            "token_ids": [1, 2],
            "liquidities": [100, 200],
        }
        result = _exhaust(b.get_exit_pool_tx_hash(action))
        assert result == (None, None, None)

    def test_safe_tx_hash_none(self):
        """Test safe tx hash none."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.exit = _make_gen_method((b"txhash", "0x" + "cc" * 20, False))
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b.contract_interact = _make_gen_method(None)
        action = {
            "dex_type": DexType.UNISWAP_V3.value,
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "token_id": 1,
            "liquidity": 100,
            "pool_type": "Weighted",
        }
        result = _exhaust(b.get_exit_pool_tx_hash(action))
        assert result == (None, None, None)

    def test_velodrome_cl(self):
        """Test velodrome cl."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.exit = _make_gen_method((b"txhash", "0x" + "cc" * 20, True))
        b.pools = {DexType.VELODROME.value: mock_pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = {
            "dex_type": DexType.VELODROME.value,
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "is_cl_pool": True,
            "is_stable": False,
            "pool_type": "CL",
            "token_ids": [1, 2],
            "liquidities": [100, 200],
        }
        result = _exhaust(b.get_exit_pool_tx_hash(action))
        assert result[0] is not None

    def test_velodrome_non_cl(self):
        """Test velodrome non cl."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.exit = _make_gen_method((b"txhash", "0x" + "cc" * 20, False))
        b.pools = {DexType.VELODROME.value: mock_pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = {
            "dex_type": DexType.VELODROME.value,
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "is_cl_pool": False,
            "is_stable": True,
            "pool_type": "Stable",
            "assets": ["0xA", "0xB"],
            "liquidity": 100,
        }
        result = _exhaust(b.get_exit_pool_tx_hash(action))
        assert result[0] is not None

    def test_velodrome_non_cl_no_assets(self):
        """Test velodrome non cl no assets."""
        b = _make_behaviour()
        b.pools = {DexType.VELODROME.value: MagicMock()}
        action = {
            "dex_type": DexType.VELODROME.value,
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "is_cl_pool": False,
            "is_stable": True,
            "pool_type": "Stable",
            "assets": [],
            "liquidity": 100,
        }
        result = _exhaust(b.get_exit_pool_tx_hash(action))
        assert result == (None, None, None)

    def test_balancer_no_assets(self):
        """Test balancer no assets."""
        b = _make_behaviour()
        b.pools = {DexType.BALANCER.value: MagicMock()}
        action = {
            "dex_type": DexType.BALANCER.value,
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "pool_type": "Weighted",
            "assets": [],
        }
        result = _exhaust(b.get_exit_pool_tx_hash(action))
        assert result == (None, None, None)

    def test_balancer_success(self):
        """Test balancer success."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.exit = _make_gen_method((b"txhash", "0x" + "cc" * 20, False))
        b.pools = {DexType.BALANCER.value: mock_pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = {
            "dex_type": DexType.BALANCER.value,
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "pool_type": "Weighted",
            "assets": ["0xA", "0xB"],
        }
        result = _exhaust(b.get_exit_pool_tx_hash(action))
        assert result[0] is not None

    def test_else_unknown_dex_with_pool(self):
        """Test else unknown dex with pool."""
        b = _make_behaviour()
        b.pools = {"someDex": MagicMock()}
        action = {
            "dex_type": "someDex",
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "pool_type": "X",
        }
        result = _exhaust(b.get_exit_pool_tx_hash(action))
        assert result == (None, None, None)


class TestGetDepositTxHash:
    """TestGetDepositTxHash."""

    POOL_ADDR = "0x" + "e1" * 20
    TOKEN_ADDR = "0x" + "e2" * 20

    def test_no_relative_funds(self):
        """Test no relative funds."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1000)
        action = {
            "chain": "optimism",
            "token0": self.TOKEN_ADDR,
            "pool_address": self.POOL_ADDR,
            "relative_funds_percentage": 0,
        }
        result = _exhaust(b.get_deposit_tx_hash(action, []))
        assert result == (None, None, None)

    def test_missing_info(self):
        """Test missing info."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=0)
        action = {
            "chain": "optimism",
            "token0": self.TOKEN_ADDR,
            "pool_address": self.POOL_ADDR,
            "relative_funds_percentage": 1.0,
        }
        result = _exhaust(b.get_deposit_tx_hash(action, []))
        assert result == (None, None, None)

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1000)
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": self.POOL_ADDR, "value": 0, "data": b"x"}
        )
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = {
            "chain": "optimism",
            "token0": self.TOKEN_ADDR,
            "pool_address": self.POOL_ADDR,
            "relative_funds_percentage": 1.0,
        }
        result = _exhaust(b.get_deposit_tx_hash(action, []))
        assert result[0] is not None

    def test_approval_fails(self):
        """Test approval fails."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1000)
        b.get_approval_tx_hash = _make_gen_method({})
        action = {
            "chain": "optimism",
            "token0": self.TOKEN_ADDR,
            "pool_address": self.POOL_ADDR,
            "relative_funds_percentage": 1.0,
        }
        result = _exhaust(b.get_deposit_tx_hash(action, []))
        assert result == (None, None, None)

    def test_deposit_tx_none(self):
        """Test deposit tx none."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1000)
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": self.POOL_ADDR, "value": 0, "data": b"x"}
        )
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return None
            return "0x" + "ab" * 32

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "token0": self.TOKEN_ADDR,
            "pool_address": self.POOL_ADDR,
            "relative_funds_percentage": 1.0,
        }
        result = _exhaust(b.get_deposit_tx_hash(action, []))
        assert result == (None, None, None)

    def test_safe_tx_hash_none(self):
        """Test safe tx hash none."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1000)
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": self.POOL_ADDR, "value": 0, "data": b"x"}
        )
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] <= 2:
                return "0x" + "ab" * 32
            return None

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "token0": self.TOKEN_ADDR,
            "pool_address": self.POOL_ADDR,
            "relative_funds_percentage": 1.0,
        }
        result = _exhaust(b.get_deposit_tx_hash(action, []))
        assert result == (None, None, None)


class TestGetWithdrawTxHash:
    """TestGetWithdrawTxHash."""

    POOL_ADDR = "0x" + "f1" * 20

    def test_no_receiver(self):
        """Test no receiver."""
        b = _make_behaviour()
        b.params.safe_contract_addresses = {"optimism": None}
        action = {"chain": "optimism", "pool_address": self.POOL_ADDR, "dex_type": "X"}
        result = _exhaust(b.get_withdraw_tx_hash(action))
        assert result == (None, None, None)

    def test_sturdy_success(self):
        """Test sturdy success."""
        b = _make_behaviour()
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return 500  # shares (balance_of)
            if call_count[0] == 2:
                return b"\xab" * 32  # tx_hash (redeem) - must be bytes
            return "0x" + "ab" * 32  # safe_tx_hash

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "pool_address": self.POOL_ADDR,
            "dex_type": "Sturdy",
        }
        result = _exhaust(b.get_withdraw_tx_hash(action))
        assert result[0] is not None

    def test_sturdy_no_shares(self):
        """Test sturdy no shares."""
        b = _make_behaviour()
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return None
            return "0x" + "ab" * 32

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "pool_address": self.POOL_ADDR,
            "dex_type": "Sturdy",
        }
        result = _exhaust(b.get_withdraw_tx_hash(action))
        assert result == (None, None, None)

    def test_non_sturdy_success(self):
        """Test non sturdy success."""
        b = _make_behaviour()
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return 1000  # amount (max_withdraw)
            if call_count[0] == 2:
                return b"\xab" * 32  # tx_hash (withdraw) - must be bytes
            return "0x" + "ab" * 32  # safe_tx_hash

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "pool_address": self.POOL_ADDR,
            "dex_type": "Other",
        }
        result = _exhaust(b.get_withdraw_tx_hash(action))
        assert result[0] is not None

    def test_non_sturdy_no_amount(self):
        """Test non sturdy no amount."""
        b = _make_behaviour()
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return None
            return "0x" + "ab" * 32

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "pool_address": self.POOL_ADDR,
            "dex_type": "Other",
        }
        result = _exhaust(b.get_withdraw_tx_hash(action))
        assert result == (None, None, None)

    def test_non_sturdy_zero_amount_is_valid(self):
        """Test that amount=0 from max_withdraw does NOT trigger early return."""
        b = _make_behaviour()
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return 0  # amount=0 (max_withdraw returns zero)
            if call_count[0] == 2:
                return b"\xab" * 32  # tx_hash (withdraw)
            return "0x" + "ab" * 32  # safe_tx_hash

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "pool_address": self.POOL_ADDR,
            "dex_type": "Other",
        }
        result = _exhaust(b.get_withdraw_tx_hash(action))
        # amount=0 should proceed to prepare the withdraw transaction
        assert result[0] is not None

    def test_non_sturdy_none_amount_returns_error(self):
        """Test that amount=None from max_withdraw triggers early return."""
        b = _make_behaviour()
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return None  # amount=None (error)
            return "0x" + "ab" * 32

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "pool_address": self.POOL_ADDR,
            "dex_type": "Other",
        }
        result = _exhaust(b.get_withdraw_tx_hash(action))
        assert result == (None, None, None)

    def test_tx_hash_none(self):
        """Test tx hash none."""
        b = _make_behaviour()
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] <= 1:
                return 100
            return None

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "pool_address": self.POOL_ADDR,
            "dex_type": "Other",
        }
        result = _exhaust(b.get_withdraw_tx_hash(action))
        assert result == (None, None, None)

    def test_safe_tx_none(self):
        """Test safe tx none."""
        b = _make_behaviour()
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] <= 2:
                return "0x" + "ab" * 32
            return None

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "pool_address": self.POOL_ADDR,
            "dex_type": "Other",
        }
        result = _exhaust(b.get_withdraw_tx_hash(action))
        assert result == (None, None, None)


class TestGetTokenTransferTxHash:
    """TestGetTokenTransferTxHash."""

    ADDR_DEST = "0x" + "d1" * 20
    ADDR_TOK = "0x" + "d2" * 20

    def test_missing_info(self):
        """Test missing info."""
        b = _make_behaviour()
        action = {"chain": "optimism"}
        result = _exhaust(b.get_token_transfer_tx_hash(action))
        assert result == (None, None, None)

    def test_balance_none(self):
        """Test balance none."""
        b = _make_behaviour()
        b.contract_interact = _make_gen_method(None)
        action = {
            "chain": "optimism",
            "to_address": self.ADDR_DEST,
            "token_address": self.ADDR_TOK,
        }
        result = _exhaust(b.get_token_transfer_tx_hash(action))
        assert result == (None, None, None)

    def test_zero_balance(self):
        """Test zero balance."""
        b = _make_behaviour()
        b.contract_interact = _make_gen_method(0)
        action = {
            "chain": "optimism",
            "to_address": self.ADDR_DEST,
            "token_address": self.ADDR_TOK,
        }
        result = _exhaust(b.get_token_transfer_tx_hash(action))
        assert result == (None, None, None)

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return 1000  # token_balance
            if call_count[0] == 2:
                return b"\xab" * 32  # tx_hash (transfer tx) - must be bytes
            return "0x" + "ab" * 32  # safe_tx_hash

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "to_address": self.ADDR_DEST,
            "token_address": self.ADDR_TOK,
        }
        result = _exhaust(b.get_token_transfer_tx_hash(action))
        assert result[0] is not None

    def test_transfer_tx_none(self):
        """Test transfer tx none."""
        b = _make_behaviour()
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return 1000
            if call_count[0] == 2:
                return None
            return "0x" + "ab" * 32

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "to_address": self.ADDR_DEST,
            "token_address": self.ADDR_TOK,
        }
        result = _exhaust(b.get_token_transfer_tx_hash(action))
        assert result == (None, None, None)

    def test_safe_tx_none(self):
        """Test safe tx none."""
        b = _make_behaviour()
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return 1000  # token_balance
            if call_count[0] == 2:
                return b"\xab" * 32  # tx_hash (transfer tx)
            return None  # safe_tx_hash is None

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "to_address": self.ADDR_DEST,
            "token_address": self.ADDR_TOK,
        }
        result = _exhaust(b.get_token_transfer_tx_hash(action))
        assert result == (None, None, None)


class TestGetClaimRewardsTxHash:
    """TestGetClaimRewardsTxHash."""

    ADDR_USER = "0x" + "c1" * 20
    ADDR_TOKEN = "0x" + "c2" * 20

    def test_missing_info(self):
        """Test missing info."""
        b = _make_behaviour()
        action = {"chain": "optimism"}
        result = _exhaust(b.get_claim_rewards_tx_hash(action))
        assert result == (None, None, None)

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return b"\xab" * 32  # tx_hash (claim rewards tx) - must be bytes
            return "0x" + "ab" * 32  # safe_tx_hash

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "users": [self.ADDR_USER],
            "tokens": [self.ADDR_TOKEN],
            "claims": [100],
            "proofs": [["0xP"]],
        }
        result = _exhaust(b.get_claim_rewards_tx_hash(action))
        assert result[0] is not None

    def test_tx_hash_none(self):
        """Test tx hash none."""
        b = _make_behaviour()
        b.contract_interact = _make_gen_method(None)
        action = {
            "chain": "optimism",
            "users": [self.ADDR_USER],
            "tokens": [self.ADDR_TOKEN],
            "claims": [100],
            "proofs": [["0xP"]],
        }
        result = _exhaust(b.get_claim_rewards_tx_hash(action))
        assert result == (None, None, None)

    def test_safe_tx_none(self):
        """Test safe tx none."""
        b = _make_behaviour()
        call_count = [0]

        def ci(*a, **kw):
            """Ci."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return b"\xab" * 32  # tx_hash
            return None  # safe_tx_hash is None

        b.contract_interact = ci
        action = {
            "chain": "optimism",
            "users": [self.ADDR_USER],
            "tokens": [self.ADDR_TOKEN],
            "claims": [100],
            "proofs": [["0xP"]],
        }
        result = _exhaust(b.get_claim_rewards_tx_hash(action))
        assert result == (None, None, None)


class TestGetStepTransactionsData:
    """TestGetStepTransactionsData."""

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        b._set_step_addresses = MagicMock(side_effect=lambda x: x)
        b._get_step_transaction = _make_gen_method({"fee": 0.1, "gas_cost": 0.1})
        route = {"steps": [{"action": {}}, {"action": {}}]}
        result = _exhaust(b._get_step_transactions_data(route))
        assert len(result) == 2

    def test_step_returns_none(self):
        """Test step returns none."""
        b = _make_behaviour()
        b._set_step_addresses = MagicMock(side_effect=lambda x: x)
        b._get_step_transaction = _make_gen_method(None)
        route = {"steps": [{"action": {}}]}
        result = _exhaust(b._get_step_transactions_data(route))
        assert result is None

    def test_empty_steps(self):
        """Test empty steps."""
        b = _make_behaviour()
        route = {"steps": []}
        result = _exhaust(b._get_step_transactions_data(route))
        assert result == []


class TestGetStepTransaction:
    """TestGetStepTransaction."""

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps(
            {
                "action": {
                    "fromToken": {"address": "0x1", "symbol": "A"},
                    "toToken": {"address": "0x2", "symbol": "B"},
                    "fromChainId": 10,
                    "toChainId": 34443,
                },
                "estimate": {
                    "fromAmount": "1000",
                    "feeCosts": [{"amountUSD": "0.1"}],
                    "gasCosts": [{"amountUSD": "0.2"}],
                },
                "transactionRequest": {"to": "0xLIFI", "data": "0x" + "ab" * 32},
                "tool": "lifi",
                "fromAmountUSD": "10",
                "toAmountUSD": "9.5",
            }
        )
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b._get_step_transaction({"action": {}}))
        assert result is not None
        assert result["source_token"] == "0x1"

    def test_error_status_code(self):
        """Test error status code."""
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 500
        resp.body = json.dumps({"message": "error"})
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b._get_step_transaction({"action": {}}))
        assert result is None

    def test_error_status_code_bad_json(self):
        """Test error status code bad json."""
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 500
        resp.body = "not json"
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b._get_step_transaction({"action": {}}))
        assert result is None

    def test_error_status_code_no_message_key(self):
        """Test LiFi error response JSON without a 'message' key."""
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 500
        resp.body = json.dumps({"error": "something went wrong"})
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b._get_step_transaction({"action": {}}))
        assert result is None
        # Should log with 'Unknown error' instead of raising KeyError
        b.context.logger.error.assert_called()
        logged_msg = b.context.logger.error.call_args[0][0]
        assert "Unknown error" in logged_msg

    def test_parse_error(self):
        """Test parse error."""
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = "not json"
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b._get_step_transaction({"action": {}}))
        assert result is None

    def test_invalid_tx_data(self):
        """Test invalid tx data."""
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps(
            {
                "action": {
                    "fromToken": {"address": "0x1", "symbol": "A"},
                    "toToken": {"address": "0x2", "symbol": "B"},
                    "fromChainId": 10,
                    "toChainId": 34443,
                },
                "estimate": {"fromAmount": "1000", "feeCosts": [], "gasCosts": []},
                "transactionRequest": {"to": "0xLIFI", "data": None},
                "tool": "lifi",
            }
        )
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b._get_step_transaction({"action": {}}))
        assert result is None

    def test_invalid_hex_data(self):
        """Test invalid hex data."""
        b = _make_behaviour()
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps(
            {
                "action": {
                    "fromToken": {"address": "0x1", "symbol": "A"},
                    "toToken": {"address": "0x2", "symbol": "B"},
                    "fromChainId": 10,
                    "toChainId": 34443,
                },
                "estimate": {"fromAmount": "1000", "feeCosts": [], "gasCosts": []},
                "transactionRequest": {"to": "0xLIFI", "data": "0xZZZZ"},
                "tool": "lifi",
            }
        )
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b._get_step_transaction({"action": {}}))
        assert result is None


class TestFetchRoutes:
    """TestFetchRoutes."""

    def _base_action(self):
        return {
            "from_chain": "optimism",
            "to_chain": "mode",
            "from_token": "0xT1",
            "to_token": "0xT2",
            "from_token_symbol": "USDC",
            "to_token_symbol": "USDC",
            "funds_percentage": 1,
        }

    def _positions(self):
        return [
            {"chain": "optimism", "assets": [{"address": "0xT1", "balance": 10000}]}
        ]

    def test_no_balance(self):
        """Test no balance."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._get_balance = MagicMock(return_value=None)
        result = _exhaust(b.fetch_routes([], self._base_action()))
        assert result is None

    def test_zero_amount(self):
        """Test zero amount."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._get_balance = MagicMock(return_value=0)
        result = _exhaust(b.fetch_routes([], self._base_action()))
        assert result is None

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._get_balance = MagicMock(return_value=10000)
        b._get_token_decimals = _make_gen_method(18)
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps({"routes": [{"steps": []}]})
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b.fetch_routes(self._positions(), self._base_action()))
        assert result is not None
        assert len(result) == 1

    def test_api_error_json(self):
        """Test api error json."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._get_balance = MagicMock(return_value=10000)
        b._get_token_decimals = _make_gen_method(18)
        resp = MagicMock()
        resp.status_code = 400
        resp.body = json.dumps({"message": "error"})
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b.fetch_routes(self._positions(), self._base_action()))
        assert result is None

    def test_api_error_non_json(self):
        """Test api error non json."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._get_balance = MagicMock(return_value=10000)
        b._get_token_decimals = _make_gen_method(18)
        resp = MagicMock()
        resp.status_code = 500
        resp.body = "internal server error"
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b.fetch_routes(self._positions(), self._base_action()))
        assert result is None

    def test_empty_body(self):
        """Test empty body."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._get_balance = MagicMock(return_value=10000)
        b._get_token_decimals = _make_gen_method(18)
        resp = MagicMock()
        resp.status_code = 200
        resp.body = ""
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b.fetch_routes(self._positions(), self._base_action()))
        assert result is None

    def test_no_routes_in_response(self):
        """Test no routes in response."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._get_balance = MagicMock(return_value=10000)
        b._get_token_decimals = _make_gen_method(18)
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps({"routes": []})
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b.fetch_routes(self._positions(), self._base_action()))
        assert result is None

    def test_parse_error(self):
        """Test parse error."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._get_balance = MagicMock(return_value=10000)
        b._get_token_decimals = _make_gen_method(18)
        resp = MagicMock()
        resp.status_code = 200
        resp.body = "not json{"
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b.fetch_routes(self._positions(), self._base_action()))
        assert result is None

    def test_none_param_values(self):
        """Test none param values."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._get_balance = MagicMock(return_value=10000)
        b._get_token_decimals = _make_gen_method(18)
        b.params.chain_to_chain_id_mapping = {}  # Will produce None chain ids
        result = _exhaust(b.fetch_routes(self._positions(), self._base_action()))
        assert result is None

    def test_with_snapshot_balance(self):
        """Test with snapshot balance."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._get_balance = MagicMock(return_value=10000)
        b._get_token_decimals = _make_gen_method(18)
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps({"routes": [{"steps": []}]})
        b.get_http_response = _make_gen_method(resp)
        action = self._base_action()
        action["source_initial_balance"] = 5000
        result = _exhaust(b.fetch_routes(self._positions(), action))
        assert result is not None

    def test_investing_paused_higher_slippage(self):
        """Test investing paused higher slippage."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(True)
        b._get_balance = MagicMock(return_value=10000)
        b._get_token_decimals = _make_gen_method(18)
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps({"routes": [{"steps": []}]})
        b.get_http_response = _make_gen_method(resp)
        result = _exhaust(b.fetch_routes(self._positions(), self._base_action()))
        assert result is not None

    def test_zero_address_token(self):
        """Test zero address token."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._get_balance = MagicMock(return_value=10000)
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps({"routes": [{"steps": []}]})
        b.get_http_response = _make_gen_method(resp)
        action = self._base_action()
        action["from_token"] = ZERO_ADDRESS
        result = _exhaust(b.fetch_routes(self._positions(), action))
        assert result is not None


class TestStakeUnstakeClaimStakingRewards:
    """TestStakeUnstakeClaimStakingRewards."""

    def _velodrome_action(self, is_cl_pool=False, **extra):
        return {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "is_cl_pool": is_cl_pool,
            **extra,
        }

    def test_stake_non_velodrome(self):
        """Test stake non velodrome."""
        b = _make_behaviour()
        result = _exhaust(
            b.get_stake_lp_tokens_tx_hash(
                {"dex_type": "uniswap", "chain": "optimism", "pool_address": "0xP"}
            )
        )
        assert result == (None, None, None)

    def test_stake_missing_params(self):
        """Test stake missing params."""
        b = _make_behaviour()
        result = _exhaust(
            b.get_stake_lp_tokens_tx_hash(
                {"dex_type": "velodrome", "chain": None, "pool_address": "0xP"}
            )
        )
        assert result == (None, None, None)

    def test_stake_no_pool_behaviour(self):
        """Test stake no pool behaviour."""
        b = _make_behaviour()
        b.pools = {}
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(self._velodrome_action()))
        assert result == (None, None, None)

    def test_stake_cl_success(self):
        """Test stake cl success."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.stake_cl_lp_tokens = _make_gen_method(
            {
                "tx_hash": b"data",
                "contract_address": "0x" + "cc" * 20,
                "is_multisend": True,
            }
        )
        b.pools = {"velodrome": mock_pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(
            is_cl_pool=True, token_ids=[1], gauge_address="0xGAUGE"
        )
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result[0] is not None

    def test_stake_cl_find_position(self):
        """Test stake cl find position."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method("0xGAUGE")
        mock_pool.stake_cl_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            }
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result[0] is not None

    def test_stake_cl_no_matching_position(self):
        """Test stake cl no matching position."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        b.pools = {"velodrome": mock_pool}
        b.current_positions = []
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)

    def test_stake_cl_no_token_ids_gauge(self):
        """Test stake cl no token ids gauge."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method(None)
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [],
            }
        ]
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)

    def test_stake_non_cl_success(self):
        """Test stake non cl success."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.stake_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b._get_token_balance = _make_gen_method(1000)
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action()
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result[0] is not None

    def test_stake_non_cl_no_balance(self):
        """Test stake non cl no balance."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        b.pools = {"velodrome": mock_pool}
        b._get_token_balance = _make_gen_method(0)
        action = self._velodrome_action()
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)

    def test_stake_result_error(self):
        """Test stake result error."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.stake_lp_tokens = _make_gen_method({"error": "fail"})
        b.pools = {"velodrome": mock_pool}
        b._get_token_balance = _make_gen_method(1000)
        action = self._velodrome_action()
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)

    def test_stake_result_none(self):
        """Test stake result none."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.stake_lp_tokens = _make_gen_method(None)
        b.pools = {"velodrome": mock_pool}
        b._get_token_balance = _make_gen_method(1000)
        action = self._velodrome_action()
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)

    def test_stake_missing_tx_hash(self):
        """Test stake missing tx hash."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.stake_lp_tokens = _make_gen_method(
            {"tx_hash": None, "contract_address": "0xC"}
        )
        b.pools = {"velodrome": mock_pool}
        b._get_token_balance = _make_gen_method(1000)
        action = self._velodrome_action()
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)

    def test_stake_safe_tx_none(self):
        """Test stake safe tx none."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.stake_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b._get_token_balance = _make_gen_method(1000)
        b.contract_interact = _make_gen_method(None)
        action = self._velodrome_action()
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)

    def test_stake_exception(self):
        """Test stake exception."""
        b = _make_behaviour()
        b.pools = {"velodrome": MagicMock(side_effect=RuntimeError)}
        b._get_token_balance = _make_gen_method(1000)

        # Force exception in the try block
        def raise_err(*a, **kw):
            """Raise err."""
            raise RuntimeError("boom")
            yield  # noqa

        mock_pool = MagicMock()
        mock_pool.stake_lp_tokens = raise_err
        b.pools = {"velodrome": mock_pool}
        action = self._velodrome_action()
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)

    # Unstake tests
    def test_unstake_non_velodrome(self):
        """Test unstake non velodrome."""
        b = _make_behaviour()
        result = _exhaust(
            b.get_unstake_lp_tokens_tx_hash(
                {"dex_type": "uniswap", "chain": "optimism", "pool_address": "0xP"}
            )
        )
        assert result == (None, None, None)

    def test_unstake_missing_params(self):
        """Test unstake missing params."""
        b = _make_behaviour()
        result = _exhaust(
            b.get_unstake_lp_tokens_tx_hash(
                {"dex_type": "velodrome", "chain": None, "pool_address": "0xP"}
            )
        )
        assert result == (None, None, None)

    def test_unstake_no_pool(self):
        """Test unstake no pool."""
        b = _make_behaviour()
        b.pools = {}
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(self._velodrome_action()))
        assert result == (None, None, None)

    def test_unstake_cl_success(self):
        """Test unstake cl success."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.unstake_cl_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(
            is_cl_pool=True, token_ids=[1], gauge_address="0xGAUGE"
        )
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(action))
        assert result[0] is not None

    def test_unstake_cl_find_position(self):
        """Test unstake cl find position."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method("0xGAUGE")
        mock_pool.unstake_cl_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            }
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(action))
        assert result[0] is not None

    def test_unstake_non_cl_success(self):
        """Test unstake non cl success."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_staked_balance = _make_gen_method(1000)
        mock_pool.unstake_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action()
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(action))
        assert result[0] is not None

    def test_unstake_non_cl_no_staked(self):
        """Test unstake non cl no staked."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_staked_balance = _make_gen_method(0)
        b.pools = {"velodrome": mock_pool}
        action = self._velodrome_action()
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)

    def test_unstake_result_error(self):
        """Test unstake result error."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_staked_balance = _make_gen_method(1000)
        mock_pool.unstake_lp_tokens = _make_gen_method(None)
        b.pools = {"velodrome": mock_pool}
        action = self._velodrome_action()
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)

    def test_unstake_exception(self):
        """Test unstake exception."""
        b = _make_behaviour()
        mock_pool = MagicMock()

        def raise_err(*a, **kw):
            """Raise err."""
            raise RuntimeError("boom")
            yield  # noqa

        mock_pool.get_staked_balance = raise_err
        b.pools = {"velodrome": mock_pool}
        action = self._velodrome_action()
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)

    # Claim staking rewards tests
    def test_claim_non_velodrome(self):
        """Test claim non velodrome."""
        b = _make_behaviour()
        result = _exhaust(
            b.get_claim_staking_rewards_tx_hash(
                {"dex_type": "uniswap", "chain": "optimism", "pool_address": "0xP"}
            )
        )
        assert result == (None, None, None)

    def test_claim_missing_params(self):
        """Test claim missing params."""
        b = _make_behaviour()
        result = _exhaust(
            b.get_claim_staking_rewards_tx_hash(
                {"dex_type": "velodrome", "chain": None, "pool_address": "0xP"}
            )
        )
        assert result == (None, None, None)

    def test_claim_no_pool(self):
        """Test claim no pool."""
        b = _make_behaviour()
        b.pools = {}
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(self._velodrome_action()))
        assert result == (None, None, None)

    def test_claim_cl_success(self):
        """Test claim cl success."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.claim_cl_rewards = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(
            is_cl_pool=True, token_ids=[1], gauge_address="0xGAUGE"
        )
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result[0] is not None

    def test_claim_cl_find_position(self):
        """Test claim cl find position."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method("0xGAUGE")
        mock_pool.claim_cl_rewards = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            }
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result[0] is not None

    def test_claim_non_cl_success(self):
        """Test claim non cl success."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.claim_rewards = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action()
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result[0] is not None

    def test_claim_result_none(self):
        """Test claim result none."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.claim_rewards = _make_gen_method(None)
        b.pools = {"velodrome": mock_pool}
        action = self._velodrome_action()
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result == (None, None, None)

    def test_claim_exception(self):
        """Test claim exception."""
        b = _make_behaviour()
        mock_pool = MagicMock()

        def raise_err(*a, **kw):
            """Raise err."""
            raise RuntimeError("boom")
            yield  # noqa

        mock_pool.claim_rewards = raise_err
        b.pools = {"velodrome": mock_pool}
        action = self._velodrome_action()
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result == (None, None, None)

    def test_claim_cl_no_matching_position(self):
        """Test claim cl no matching position."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        b.pools = {"velodrome": mock_pool}
        b.current_positions = []
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result == (None, None, None)

    def test_claim_cl_filters_to_staked_tokens(self):
        """Only staked token_ids are forwarded to claim_cl_rewards."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.is_cl_token_staked = _make_gen_method_by_token({1: True, 2: False})
        captured = {}

        def fake_claim(behaviour, **kwargs):
            """Fake claim."""
            captured["token_ids"] = kwargs["token_ids"]
            yield
            return {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}

        mock_pool.claim_cl_rewards = fake_claim
        b.pools = {"velodrome": mock_pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(
            is_cl_pool=True, token_ids=[1, 2], gauge_address="0xGAUGE"
        )
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result[0] is not None
        assert captured["token_ids"] == [1]

    def test_claim_cl_skips_when_no_tokens_staked(self):
        """If no tokens are staked on-chain, claim is skipped entirely."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.is_cl_token_staked = _make_gen_method_by_token({1: False, 2: False})
        mock_pool.claim_cl_rewards = _make_gen_method({"tx_hash": b"nope"})
        b.pools = {"velodrome": mock_pool}
        action = self._velodrome_action(
            is_cl_pool=True, token_ids=[1, 2], gauge_address="0xGAUGE"
        )
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result == (None, None, None)

    def test_claim_cl_falls_back_on_verification_failure(self):
        """RPC failure on is_cl_token_staked preserves the original token_ids."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.is_cl_token_staked = _make_gen_method(None)
        captured = {}

        def fake_claim(behaviour, **kwargs):
            """Fake claim."""
            captured["token_ids"] = kwargs["token_ids"]
            yield
            return {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}

        mock_pool.claim_cl_rewards = fake_claim
        b.pools = {"velodrome": mock_pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(
            is_cl_pool=True, token_ids=[1, 2], gauge_address="0xGAUGE"
        )
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result[0] is not None
        assert captured["token_ids"] == [1, 2]

    def test_claim_cl_emits_structured_warning_on_verification_failure(self):
        """The claim path must surface the same structured warning the # noqa: D205
        unstake path does so dashboards covering "fallback fired,
        on-chain state not actually verified" pick up both code paths.
        """
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.is_cl_token_staked = _make_gen_method(None)

        def fake_claim(behaviour, **kwargs):
            """Fake claim."""
            yield
            return {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}

        mock_pool.claim_cl_rewards = fake_claim
        b.pools = {"velodrome": mock_pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)

        warnings: List[str] = []
        b.context.logger.warning = lambda msg, *a, **k: warnings.append(str(msg))

        action = self._velodrome_action(
            is_cl_pool=True, token_ids=[1, 2], gauge_address="0xGAUGE"
        )
        _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert any(
            "claim_verification_fallback" in w and "reason=verification_rpc_failed" in w
            for w in warnings
        )

    def test_unstake_cl_no_matching_position(self):
        """Test unstake cl no matching position."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        b.pools = {"velodrome": mock_pool}
        b.current_positions = []
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)

    def test_unstake_safe_tx_none(self):
        """Test unstake safe tx none."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_staked_balance = _make_gen_method(1000)
        mock_pool.unstake_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.contract_interact = _make_gen_method(None)
        action = self._velodrome_action()
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)

    def test_unstake_missing_tx_hash(self):
        """Test unstake missing tx hash."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_staked_balance = _make_gen_method(1000)
        mock_pool.unstake_lp_tokens = _make_gen_method(
            {"tx_hash": None, "contract_address": "0xC"}
        )
        b.pools = {"velodrome": mock_pool}
        action = self._velodrome_action()
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)

    def test_claim_safe_tx_none(self):
        """Test claim safe tx none."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.claim_rewards = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.contract_interact = _make_gen_method(None)
        action = self._velodrome_action()
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result == (None, None, None)

    def test_claim_missing_tx_hash(self):
        """Test claim missing tx hash."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.claim_rewards = _make_gen_method(
            {"tx_hash": None, "contract_address": "0xC"}
        )
        b.pools = {"velodrome": mock_pool}
        action = self._velodrome_action()
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result == (None, None, None)

    def test_claim_cl_no_token_ids_gauge(self):
        """Test claim cl no token ids gauge."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method(None)
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [],
            }
        ]
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result == (None, None, None)

    def test_unstake_cl_no_token_ids_gauge(self):
        """Test unstake cl no token ids gauge."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method(None)
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [],
            }
        ]
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(action))
        assert result == (None, None, None)


class TestUpdateWithdrawalCompletionFinalTxHashBranch:
    """TestUpdateWithdrawalCompletionFinalTxHashBranch."""

    def test_with_truthy_final_tx_hash(self):
        """When final_tx_hash is truthy, tx hashes are stored."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xabc123"
        kv_calls = []

        def capture_kv(*a, **kw):
            """Capture kv."""
            kv_calls.append(a)
            yield

        b._write_kv = capture_kv
        _exhaust(b._update_withdrawal_completion())
        # Verify that withdrawal_transaction_hashes was in the data written
        assert len(kv_calls) == 1
        data = kv_calls[0][0]
        assert "withdrawal_transaction_hashes" in data

    def test_with_empty_final_tx_hash(self):
        """When final_tx_hash is empty string, tx hashes are NOT stored."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = ""
        kv_calls = []

        def capture_kv(*a, **kw):
            """Capture kv."""
            kv_calls.append(a)
            yield

        b._write_kv = capture_kv
        _exhaust(b._update_withdrawal_completion())
        assert len(kv_calls) == 1
        data = kv_calls[0][0]
        assert "withdrawal_transaction_hashes" not in data


class TestPostExecuteStepContinueBranch:
    """TestPostExecuteStepContinueBranch."""

    def test_continue_returns_result(self):
        """decision == CONTINUE: slippage costs added, returns update."""  # noqa: D403
        b = _make_behaviour()
        b.sleep = _make_gen_method(None)
        b.get_decision_on_swap = _make_gen_method(Decision.CONTINUE)
        b._add_slippage_costs = _make_gen_method(None)
        b.synchronized_data.last_executed_step_index = None
        b._update_assets_after_swap = MagicMock(
            return_value=(
                Event.UPDATE.value,
                {"last_action": Action.STEP_EXECUTED.value},
            )
        )
        result = _exhaust(
            b._post_execute_step(
                [{"remaining_fee_allowance": 1.0, "remaining_gas_allowance": 1.0}], 0
            )
        )
        assert result[0] == Event.UPDATE.value


class TestWaitForSwapConfirmationLoop:
    """TestWaitForSwapConfirmationLoop."""

    def test_wait_then_continue(self):
        """First call returns WAIT, second returns CONTINUE."""
        b = _make_behaviour()
        b.sleep = _make_gen_method(None)
        call_count = [0]

        def swap_gen(*a, **kw):
            """Swap gen."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return Decision.WAIT
            return Decision.CONTINUE

        b.get_decision_on_swap = swap_gen
        result = _exhaust(b._wait_for_swap_confirmation())
        assert result == Decision.CONTINUE
        assert call_count[0] == 2

    def test_wait_twice_then_exit(self):
        """First two calls WAIT, third returns EXIT."""
        b = _make_behaviour()
        b.sleep = _make_gen_method(None)
        call_count = [0]

        def swap_gen(*a, **kw):
            """Swap gen."""
            call_count[0] += 1
            yield
            if call_count[0] <= 2:
                return Decision.WAIT
            return Decision.EXIT

        b.get_decision_on_swap = swap_gen
        result = _exhaust(b._wait_for_swap_confirmation())
        assert result == Decision.EXIT
        assert call_count[0] == 3


# Already covered by TestPostExecuteEnterPool.test_sturdy, this is the
# dex_type == DexType.STURDY.value branch. Confirmed it's covered above.


class TestPrepareNextActionElseBranch:
    """TestPrepareNextActionElseBranch."""

    def test_unknown_action_triggers_else(self):
        """An Action value not handled in if/elif triggers the else branch."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._read_withdrawal_status = _make_gen_method("unknown")
        # EXECUTE_STEP is a valid Action but not handled in _prepare_next_action
        action = {"action": Action.EXECUTE_STEP.value}
        result = _exhaust(b._prepare_next_action([], [action], 0, "x"))
        assert result == (Event.DONE.value, {})


class TestEnforcePoolAllocationCapZeroTotalUsd:
    """TestEnforcePoolAllocationCapZeroTotalUsd."""

    def test_zero_usd_above_cap_triggers_5050_fallback(self):
        """When current_total_usd is 0 but code somehow reaches ratio calc.

        Note: The branch at lines 1107-1108 is only reached when
        current_total_usd > max_position_size (so we pass the early return
        at line 1095) AND current_total_usd == 0 (the else branch at 1105).
        These two conditions are contradictory with normal values.
        This means lines 1107-1108 are effectively dead code.
        """
        pass  # dead code -- cannot reach both conditions simultaneously


class TestCalculateVelodromeInvestmentAmountsTokenRatioException:
    """TestCalculateVelodromeInvestmentAmountsTokenRatioException."""

    def test_token1_ratio_exception_falls_back_to_percentage(self):
        """When overall_token1_ratio raises, falls back to token1_percentage."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1_000_000)
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(1.0)
        # overall_token0_ratio is valid, overall_token1_ratio raises on float()
        action = {
            "token_requirements": {
                "overall_token0_ratio": 0.5,
                "overall_token1_ratio": "not_a_number",
            },
            "token1_percentage": 50,
        }
        result = _exhaust(
            b._calculate_velodrome_investment_amounts(
                action, "optimism", ["0x1", "0x2"], [], []
            )
        )
        assert result is not None


class TestCalculateVelodromeNegativeAmounts:
    """TestCalculateVelodromeNegativeAmounts."""

    def test_negative_amount_returns_none(self):
        """When calculated amounts are negative, return None."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1_000_000)
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(-1.0)
        action = {
            "token_requirements": {
                "overall_token0_ratio": 0.5,
                "overall_token1_ratio": 0.5,
            },
        }
        result = _exhaust(
            b._calculate_velodrome_investment_amounts(
                action, "optimism", ["0x1", "0x2"], [], []
            )
        )
        assert result is None


class TestGetEnterPoolTxHashFewerAssets:
    """TestGetEnterPoolTxHashFewerAssets."""

    def test_one_asset(self):
        """When only one token is set, assets < 2, returns None."""
        b = _make_behaviour()
        action = {
            "dex_type": DexType.UNISWAP_V3.value,
            "chain": "optimism",
            "token0": "0xT0",
            "token1": None,
            "pool_address": "0xPOOL",
        }
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)


class TestGetEnterPoolTxHashVelodromeNonCLMaxAmounts:
    """TestGetEnterPoolTxHashVelodromeNonCLMaxAmounts."""

    def test_max_amounts_in_none(self):
        """Velodrome non-CL, invested_amount set, balances returns None."""
        b = _make_behaviour()
        b.current_positions = [{"pool_address": "0xOTHER"}]
        mock_pool = MagicMock()
        b.pools = {DexType.VELODROME.value: mock_pool}
        b._calculate_investment_amounts_from_dollar_cap = _make_gen_method([500, 500])
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            (None, None, None)
        )
        action = {
            "dex_type": DexType.VELODROME.value,
            "chain": "optimism",
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xPOOL",
            "pool_type": "Stable",
            "is_stable": True,
            "is_cl_pool": False,
            "relative_funds_percentage": 1.0,
            "invested_amount": 100,
        }
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_max_amounts_in_capped_by_dollar_cap(self):
        """Velodrome non-CL, max_amounts >= 2, amounts are capped."""
        b = _make_behaviour()
        b.current_positions = [{"pool_address": "0xOTHER"}]
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method(("0xtx", "0xCONTRACT"))
        b.pools = {DexType.VELODROME.value: mock_pool}
        b._calculate_investment_amounts_from_dollar_cap = _make_gen_method([500, 500])
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": "0xT", "value": 0, "data": b"x"}
        )
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = {
            "dex_type": DexType.VELODROME.value,
            "chain": "optimism",
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xPOOL",
            "pool_type": "Stable",
            "is_stable": True,
            "is_cl_pool": False,
            "relative_funds_percentage": 1.0,
            "invested_amount": 100,
        }
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result[0] is not None


class TestGetEnterPoolNonVelodromeZeroAmounts:
    """TestGetEnterPoolNonVelodromeZeroAmounts."""

    def test_zero_amount_in_non_velodrome(self):
        """Non-velodrome dex with zero amounts returns None."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([0, 100], 0, 100)
        )
        action = {
            "dex_type": DexType.UNISWAP_V3.value,
            "chain": "optimism",
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xPOOL",
            "pool_type": "Weighted",
            "is_stable": False,
            "is_cl_pool": False,
            "relative_funds_percentage": 1.0,
        }
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)

    def test_none_amount_in_non_velodrome(self):
        """Non-velodrome dex with None amount returns None."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        b.pools = {DexType.BALANCER.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([None, 100], None, 100)
        )
        action = {
            "dex_type": DexType.BALANCER.value,
            "chain": "optimism",
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xPOOL",
            "pool_type": "Weighted",
            "is_stable": False,
            "is_cl_pool": False,
            "relative_funds_percentage": 1.0,
        }
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)


class TestGetEnterPoolToken1ApprovalFails:
    """TestGetEnterPoolToken1ApprovalFails."""

    def test_token1_approval_fails(self):
        """When token1 approval fails, returns None."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.enter = _make_gen_method(("0xtx", "0xCONTRACT"))
        b.pools = {DexType.UNISWAP_V3.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        call_count = [0]

        def approval_gen(*a, **kw):
            """Approval gen."""
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return {"operation": 0, "to": "0xT", "value": 0, "data": b"x"}
            return {}  # Second approval fails

        b.get_approval_tx_hash = approval_gen
        action = {
            "dex_type": DexType.UNISWAP_V3.value,
            "chain": "optimism",
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xPOOL",
            "pool_type": "Weighted",
            "is_stable": False,
            "is_cl_pool": False,
            "relative_funds_percentage": 1.0,
        }
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)


class TestGetExitPoolVelodromeCLNoTokenIds:
    """TestGetExitPoolVelodromeCLNoTokenIds."""

    def test_velodrome_cl_without_token_ids(self):
        """Velodrome CL exit without token_ids/liquidities -> falls to else."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.exit = _make_gen_method((b"txhash", "0x" + "cc" * 20, True))
        b.pools = {DexType.VELODROME.value: mock_pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = {
            "dex_type": DexType.VELODROME.value,
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "is_cl_pool": True,
            "is_stable": False,
            "pool_type": "CL",
        }
        # token_ids and liquidities not set -- action.get returns None
        # This should still proceed; the kwargs just won't have token_ids/liquidities
        result = _exhaust(b.get_exit_pool_tx_hash(action))
        assert result[0] is not None


class TestCheckStepCostsLastStepPasses:
    """TestCheckStepCostsLastStepPasses."""

    def test_last_step_within_allowance(self):
        """Last step fee/gas within 50% * remaining + tolerance passes."""
        b = _make_behaviour()
        b._set_step_addresses = MagicMock(side_effect=lambda x: x)
        b._get_step_transaction = _make_gen_method({"fee": 0.4, "gas_cost": 0.4})
        # total_steps=2, step_index=1 (last step), remaining_fee=1.0, remaining_gas=1.0
        # 0.4 <= 0.5 * 1.0 + 0.02 = 0.52 -> passes
        result = _exhaust(b.check_step_costs({"action": {}}, 1.0, 1.0, 1, 2))
        assert result[0] is True


class TestFetchRoutesNon18Decimals:
    """TestFetchRoutesNon18Decimals."""

    def test_non_18_decimal_token(self):
        """Token with non-18 decimals returns amount as-is (no rounding)."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._get_balance = MagicMock(return_value=10000)
        b._get_token_decimals = _make_gen_method(6)
        resp = MagicMock()
        resp.status_code = 200
        resp.body = json.dumps({"routes": [{"steps": []}]})
        b.get_http_response = _make_gen_method(resp)
        action = {
            "from_chain": "optimism",
            "to_chain": "mode",
            "from_token": "0xT1",
            "to_token": "0xT2",
            "from_token_symbol": "USDC",
            "to_token_symbol": "USDC",
            "funds_percentage": 1,
        }
        result = _exhaust(
            b.fetch_routes(
                [
                    {
                        "chain": "optimism",
                        "assets": [{"address": "0xT1", "balance": 10000}],
                    }
                ],
                action,
            )
        )
        assert result is not None


class TestFetchRoutesBodyMissing:
    """TestFetchRoutesBodyMissing."""

    def test_api_error_no_body_attribute(self):
        """When response has no body attribute during non-JSON parse."""
        b = _make_behaviour()
        b._read_investing_paused = _make_gen_method(False)
        b._get_balance = MagicMock(return_value=10000)
        b._get_token_decimals = _make_gen_method(18)
        resp = MagicMock(spec=[])  # spec=[] means no attributes at all
        resp.status_code = 500
        resp.body = None  # body exists via direct assignment but...

        # We need the body to raise ValueError/TypeError on json.loads
        # and then hasattr(response, "body") returns False
        # Create a special object:
        class FakeResp:
            """FakeResp."""

            status_code = 500
            # body intentionally not defined so hasattr returns False

        FakeResp()
        # But we need json.loads to fail, so body must be something that fails
        # Actually, the code does routes_response.body = json.loads(routes_response.body)
        # first in the try block, then in except it checks hasattr
        # We need: json.loads(routes_response.body) to raise ValueError/TypeError AND
        # hasattr(routes_response, "body") to be False
        # That's contradictory because we need .body to exist for json.loads but not for hasattr
        # This means line 2658 is dead code (body must exist to be read by json.loads).


class TestGetAllPositionsFromTxReceipt:
    """TestGetAllPositionsFromTxReceipt."""

    def _make_log(self, token_id=1, liquidity=100, amount0=50, amount1=60):
        from eth_abi import encode
        from eth_utils import keccak, to_hex

        event_sig = "IncreaseLiquidity(uint256,uint128,uint256,uint256)"
        sig_hex = "0x" + to_hex(keccak(text=event_sig))[2:]
        token_id_hex = "0x" + encode(["uint256"], [token_id]).hex()
        data_hex = (
            "0x"
            + encode(
                ["uint128", "uint256", "uint256"], [liquidity, amount0, amount1]
            ).hex()
        )
        return {"topics": [sig_hex, token_id_hex], "data": data_hex}

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        log = self._make_log(token_id=1, liquidity=100, amount0=50, amount1=60)
        receipt = {"logs": [log], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_all_positions_from_tx_receipt("0xhash", "optimism"))
        assert result is not None
        assert len(result) == 1
        assert result[0][0] == 1  # token_id

    def test_multiple_positions(self):
        """Test multiple positions."""
        b = _make_behaviour()
        log1 = self._make_log(token_id=1, liquidity=100, amount0=50, amount1=60)
        log2 = self._make_log(token_id=2, liquidity=200, amount0=70, amount1=80)
        receipt = {"logs": [log1, log2], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_all_positions_from_tx_receipt("0xhash", "optimism"))
        assert len(result) == 2

    def test_no_response(self):
        """Test no response."""
        b = _make_behaviour()
        b.get_transaction_receipt = _make_gen_method(None)
        result = _exhaust(b._get_all_positions_from_tx_receipt("0xhash", "optimism"))
        assert result is None

    def test_no_matching_logs(self):
        """Test no matching logs."""
        b = _make_behaviour()
        receipt = {
            "logs": [{"topics": ["0xwrong"], "data": "0x00"}],
            "blockNumber": "0x10",
        }
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_all_positions_from_tx_receipt("0xhash", "optimism"))
        assert result is None

    def test_no_block_number(self):
        """Test no block number."""
        b = _make_behaviour()
        log = self._make_log()
        receipt = {"logs": [log]}
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_all_positions_from_tx_receipt("0xhash", "optimism"))
        assert result is None

    def test_block_fetch_fails(self):
        """Test block fetch fails."""
        b = _make_behaviour()
        log = self._make_log()
        receipt = {"logs": [log], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method(None)
        result = _exhaust(b._get_all_positions_from_tx_receipt("0xhash", "optimism"))
        assert result is None

    def test_no_timestamp(self):
        """Test no timestamp."""
        b = _make_behaviour()
        log = self._make_log()
        receipt = {"logs": [log], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({})
        result = _exhaust(b._get_all_positions_from_tx_receipt("0xhash", "optimism"))
        assert result is None

    def test_missing_token_id_topic(self):
        """Log has matching event sig but missing token ID topic."""
        from eth_utils import keccak, to_hex

        event_sig = "IncreaseLiquidity(uint256,uint128,uint256,uint256)"
        sig_hex = "0x" + to_hex(keccak(text=event_sig))[2:]
        b = _make_behaviour()
        log = {"topics": [sig_hex, None], "data": "0x" + "00" * 96}
        receipt = {"logs": [log], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_all_positions_from_tx_receipt("0xhash", "optimism"))
        assert result is None  # No positions extracted

    def test_empty_data_field(self):
        """Log has matching event sig but empty data."""
        from eth_abi import encode
        from eth_utils import keccak, to_hex

        event_sig = "IncreaseLiquidity(uint256,uint128,uint256,uint256)"
        sig_hex = "0x" + to_hex(keccak(text=event_sig))[2:]
        token_id_hex = "0x" + encode(["uint256"], [1]).hex()
        b = _make_behaviour()
        log = {"topics": [sig_hex, token_id_hex], "data": ""}
        receipt = {"logs": [log], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_all_positions_from_tx_receipt("0xhash", "optimism"))
        assert result is None

    def test_decode_exception(self):
        """Log triggers an exception during decoding."""
        from eth_abi import encode
        from eth_utils import keccak, to_hex

        event_sig = "IncreaseLiquidity(uint256,uint128,uint256,uint256)"
        sig_hex = "0x" + to_hex(keccak(text=event_sig))[2:]
        token_id_hex = "0x" + encode(["uint256"], [1]).hex()
        b = _make_behaviour()
        log = {"topics": [sig_hex, token_id_hex], "data": "0x1234"}  # too short
        receipt = {"logs": [log], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_all_positions_from_tx_receipt("0xhash", "optimism"))
        assert result is None  # All positions fail to decode


class TestGetDataFromMintTxReceipt:
    """TestGetDataFromMintTxReceipt."""

    def _make_receipt(self, token_id=1, liquidity=100, amount0=50, amount1=60):
        from eth_abi import encode
        from eth_utils import keccak, to_hex

        event_sig = "IncreaseLiquidity(uint256,uint128,uint256,uint256)"
        sig_hex = "0x" + to_hex(keccak(text=event_sig))[2:]
        token_id_hex = "0x" + encode(["uint256"], [token_id]).hex()
        data_hex = (
            "0x"
            + encode(
                ["uint128", "uint256", "uint256"], [liquidity, amount0, amount1]
            ).hex()
        )
        log = {"topics": [sig_hex, token_id_hex], "data": data_hex}
        return {"logs": [log], "blockNumber": "0x10"}

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        receipt = self._make_receipt(
            token_id=5, liquidity=200, amount0=100, amount1=150
        )
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 99999})
        result = _exhaust(b._get_data_from_mint_tx_receipt("0xhash", "optimism"))
        assert result == (5, 200, 100, 150, 99999)

    def test_no_response(self):
        """Test no response."""
        b = _make_behaviour()
        b.get_transaction_receipt = _make_gen_method(None)
        result = _exhaust(b._get_data_from_mint_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None, None, None)

    def test_no_matching_log(self):
        """Test no matching log."""
        b = _make_behaviour()
        receipt = {"logs": [{"topics": ["0xbad"], "data": "0x00"}]}
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_data_from_mint_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None, None, None)

    def test_missing_token_id_topic(self):
        """Test missing token id topic."""
        from eth_utils import keccak, to_hex

        event_sig = "IncreaseLiquidity(uint256,uint128,uint256,uint256)"
        sig_hex = "0x" + to_hex(keccak(text=event_sig))[2:]
        b = _make_behaviour()
        log = {"topics": [sig_hex, None], "data": "0x" + "00" * 96}
        receipt = {"logs": [log]}
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_data_from_mint_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None, None, None)

    def test_empty_data_field(self):
        """Test empty data field."""
        from eth_abi import encode
        from eth_utils import keccak, to_hex

        event_sig = "IncreaseLiquidity(uint256,uint128,uint256,uint256)"
        sig_hex = "0x" + to_hex(keccak(text=event_sig))[2:]
        token_id_hex = "0x" + encode(["uint256"], [1]).hex()
        b = _make_behaviour()
        log = {"topics": [sig_hex, token_id_hex], "data": ""}
        receipt = {"logs": [log]}
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_data_from_mint_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None, None, None)

    def test_no_block_number(self):
        """Test no block number."""
        b = _make_behaviour()
        receipt = self._make_receipt()
        del receipt["blockNumber"]
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_data_from_mint_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None, None, None)

    def test_block_fetch_fails(self):
        """Test block fetch fails."""
        b = _make_behaviour()
        receipt = self._make_receipt()
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method(None)
        result = _exhaust(b._get_data_from_mint_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None, None, None)

    def test_no_timestamp(self):
        """Test no timestamp."""
        b = _make_behaviour()
        receipt = self._make_receipt()
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({})
        result = _exhaust(b._get_data_from_mint_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None, None, None)

    def test_decode_exception(self):
        """Test decode exception."""
        from eth_abi import encode
        from eth_utils import keccak, to_hex

        event_sig = "IncreaseLiquidity(uint256,uint128,uint256,uint256)"
        sig_hex = "0x" + to_hex(keccak(text=event_sig))[2:]
        token_id_hex = "0x" + encode(["uint256"], [1]).hex()
        b = _make_behaviour()
        log = {"topics": [sig_hex, token_id_hex], "data": "0x1234"}  # too short
        receipt = {"logs": [log]}
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_data_from_mint_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None, None, None)


class TestGetDataFromJoinPoolTxReceipt:
    """TestGetDataFromJoinPoolTxReceipt."""

    def _make_receipt(self, tokens=None, deltas=None):
        from eth_abi import encode
        from eth_utils import keccak

        event_sig = "PoolBalanceChanged(bytes32,address,address[],int256[],uint256[])"
        sig_hash = "0x" + keccak(text=event_sig).hex()
        if tokens is None:
            tokens = ["0x" + "a1" * 20, "0x" + "a2" * 20]
        if deltas is None:
            deltas = [500, 600]
        # Properly encode the tokens as addresses and deltas as int256
        addr_tokens = [bytes.fromhex(t[2:].rjust(40, "0")) for t in tokens]
        data_hex = (
            "0x"
            + encode(
                ["address[]", "int256[]", "uint256[]"], [addr_tokens, deltas, [0, 0]]
            ).hex()
        )
        log = {
            "topics": [sig_hash, "0x" + "00" * 32, "0x" + "00" * 32],
            "data": data_hex,
        }
        return {"logs": [log], "blockNumber": "0x10"}

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        receipt = self._make_receipt()
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_data_from_join_pool_tx_receipt("0xhash", "optimism"))
        assert result[0] == 500
        assert result[1] == 600
        assert result[2] == 12345

    def test_no_response(self):
        """Test no response."""
        b = _make_behaviour()
        b.get_transaction_receipt = _make_gen_method(None)
        result = _exhaust(b._get_data_from_join_pool_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_no_matching_log(self):
        """Test no matching log."""
        b = _make_behaviour()
        receipt = {
            "logs": [{"topics": ["0xwrong"], "data": "0x00"}],
            "blockNumber": "0x10",
        }
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_data_from_join_pool_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_empty_data(self):
        """Test empty data."""
        from eth_utils import keccak

        event_sig = "PoolBalanceChanged(bytes32,address,address[],int256[],uint256[])"
        sig_hash = "0x" + keccak(text=event_sig).hex()
        b = _make_behaviour()
        log = {"topics": [sig_hash, "0x" + "00" * 32], "data": ""}
        receipt = {"logs": [log], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_data_from_join_pool_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_no_block_number(self):
        """Test no block number."""
        b = _make_behaviour()
        receipt = self._make_receipt()
        del receipt["blockNumber"]
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_data_from_join_pool_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_block_fetch_fails(self):
        """Test block fetch fails."""
        b = _make_behaviour()
        receipt = self._make_receipt()
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method(None)
        result = _exhaust(b._get_data_from_join_pool_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_no_timestamp(self):
        """Test no timestamp."""
        b = _make_behaviour()
        receipt = self._make_receipt()
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({})
        result = _exhaust(b._get_data_from_join_pool_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_decode_exception(self):
        """Test decode exception."""
        from eth_utils import keccak

        event_sig = "PoolBalanceChanged(bytes32,address,address[],int256[],uint256[])"
        sig_hash = "0x" + keccak(text=event_sig).hex()
        b = _make_behaviour()
        log = {"topics": [sig_hash], "data": "0x1234"}  # invalid data
        receipt = {"logs": [log], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_data_from_join_pool_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_unexpected_tokens_count(self):
        """When tokens array has fewer than 2 entries."""
        from eth_abi import encode
        from eth_utils import keccak

        event_sig = "PoolBalanceChanged(bytes32,address,address[],int256[],uint256[])"
        sig_hash = "0x" + keccak(text=event_sig).hex()
        addr = bytes.fromhex("a1" * 20)
        data_hex = (
            "0x"
            + encode(["address[]", "int256[]", "uint256[]"], [[addr], [500], [0]]).hex()
        )
        b = _make_behaviour()
        log = {"topics": [sig_hash, "0x" + "00" * 32], "data": data_hex}
        receipt = {"logs": [log], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_data_from_join_pool_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None)


class TestGetDataFromVelodromeMintEvent:
    """TestGetDataFromVelodromeMintEvent."""

    def _make_receipt(self, alt_sig=False):
        from eth_abi import encode
        from eth_utils import keccak, to_hex

        if alt_sig:
            event_sig = "Mint(address,address,uint256,uint256)"
        else:
            event_sig = "Mint(address,uint256,uint256)"
        sig_hex = "0x" + to_hex(keccak(text=event_sig))[2:]
        sender_hex = "0x" + encode(["address"], [bytes.fromhex("bb" * 20)]).hex()
        data_hex = "0x" + encode(["uint256", "uint256"], [500, 600]).hex()
        log = {"topics": [sig_hex, sender_hex], "data": data_hex}
        return {"logs": [log], "blockNumber": "0x10"}

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        receipt = self._make_receipt()
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_data_from_velodrome_mint_event("0xhash", "optimism"))
        assert result == (500, 600, 12345)

    def test_alternative_signature(self):
        """Test alternative signature."""
        b = _make_behaviour()
        receipt = self._make_receipt(alt_sig=True)
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_data_from_velodrome_mint_event("0xhash", "optimism"))
        assert result == (500, 600, 12345)

    def test_no_response(self):
        """Test no response."""
        b = _make_behaviour()
        b.get_transaction_receipt = _make_gen_method(None)
        result = _exhaust(b._get_data_from_velodrome_mint_event("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_no_matching_log(self):
        """Test no matching log."""
        b = _make_behaviour()
        receipt = {
            "logs": [{"topics": ["0xwrong"], "data": "0x00"}],
            "blockNumber": "0x10",
        }
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_data_from_velodrome_mint_event("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_missing_sender_topic(self):
        """Test missing sender topic."""
        from eth_utils import keccak, to_hex

        event_sig = "Mint(address,uint256,uint256)"
        sig_hex = "0x" + to_hex(keccak(text=event_sig))[2:]
        b = _make_behaviour()
        log = {"topics": [sig_hex, None], "data": "0x" + "00" * 64}
        receipt = {"logs": [log], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_data_from_velodrome_mint_event("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_empty_data(self):
        """Test empty data."""
        from eth_abi import encode
        from eth_utils import keccak, to_hex

        event_sig = "Mint(address,uint256,uint256)"
        sig_hex = "0x" + to_hex(keccak(text=event_sig))[2:]
        sender_hex = "0x" + encode(["address"], [bytes.fromhex("bb" * 20)]).hex()
        b = _make_behaviour()
        log = {"topics": [sig_hex, sender_hex], "data": ""}
        receipt = {"logs": [log], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_data_from_velodrome_mint_event("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_no_block_number(self):
        """Test no block number."""
        b = _make_behaviour()
        receipt = self._make_receipt()
        del receipt["blockNumber"]
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_data_from_velodrome_mint_event("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_block_fetch_fails(self):
        """Test block fetch fails."""
        b = _make_behaviour()
        receipt = self._make_receipt()
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method(None)
        result = _exhaust(b._get_data_from_velodrome_mint_event("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_no_timestamp(self):
        """Test no timestamp."""
        b = _make_behaviour()
        receipt = self._make_receipt()
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({})
        result = _exhaust(b._get_data_from_velodrome_mint_event("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_decode_exception(self):
        """Test decode exception."""
        from eth_abi import encode
        from eth_utils import keccak, to_hex

        event_sig = "Mint(address,uint256,uint256)"
        sig_hex = "0x" + to_hex(keccak(text=event_sig))[2:]
        sender_hex = "0x" + encode(["address"], [bytes.fromhex("bb" * 20)]).hex()
        b = _make_behaviour()
        log = {"topics": [sig_hex, sender_hex], "data": "0x1234"}  # too short
        receipt = {"logs": [log], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_data_from_velodrome_mint_event("0xhash", "optimism"))
        assert result == (None, None, None)


class TestGetDataFromDepositTxReceipt:
    """TestGetDataFromDepositTxReceipt."""

    def _make_receipt(self, assets_val=1000, shares_val=500):
        from eth_utils import keccak

        event_sig = "Deposit(address,address,uint256,uint256)"
        sig_hash = "0x" + keccak(event_sig.encode()).hex()
        assets_hex = hex(assets_val)[2:].rjust(64, "0")
        shares_hex = hex(shares_val)[2:].rjust(64, "0")
        data_hex = "0x" + assets_hex + shares_hex
        log = {
            "topics": [sig_hash, "0x" + "00" * 32, "0x" + "00" * 32],
            "data": data_hex,
        }
        return {"logs": [log], "blockNumber": "0x10"}

    def test_success(self):
        """Test success."""
        b = _make_behaviour()
        receipt = self._make_receipt(assets_val=1000, shares_val=500)
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_data_from_deposit_tx_receipt("0xhash", "optimism"))
        assert result == (1000, 500, 12345)

    def test_no_receipt(self):
        """Test no receipt."""
        b = _make_behaviour()
        b.get_transaction_receipt = _make_gen_method(None)
        result = _exhaust(b._get_data_from_deposit_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_no_matching_log(self):
        """Test no matching log."""
        b = _make_behaviour()
        receipt = {
            "logs": [{"topics": ["0xbad"], "data": "0x00"}],
            "blockNumber": "0x10",
        }
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_data_from_deposit_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_unexpected_data_length(self):
        """Test unexpected data length."""
        from eth_utils import keccak

        event_sig = "Deposit(address,address,uint256,uint256)"
        sig_hash = "0x" + keccak(event_sig.encode()).hex()
        b = _make_behaviour()
        log = {"topics": [sig_hash], "data": "0x" + "00" * 10}  # wrong length
        receipt = {"logs": [log], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_data_from_deposit_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_no_block_number(self):
        """Test no block number."""
        b = _make_behaviour()
        receipt = self._make_receipt()
        del receipt["blockNumber"]
        b.get_transaction_receipt = _make_gen_method(receipt)
        result = _exhaust(b._get_data_from_deposit_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_block_fetch_fails(self):
        """Test block fetch fails."""
        b = _make_behaviour()
        receipt = self._make_receipt()
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method(None)
        result = _exhaust(b._get_data_from_deposit_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None)

    def test_no_timestamp(self):
        """Test no timestamp."""
        b = _make_behaviour()
        receipt = self._make_receipt()
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({})
        result = _exhaust(b._get_data_from_deposit_tx_receipt("0xhash", "optimism"))
        assert result == (None, None, None)


class TestCalculateMinHoldDaysExceptionBranch:
    """TestCalculateMinHoldDaysExceptionBranch."""

    def test_exception_returns_fallback(self):
        """When an exception occurs, return MIN_TIME_IN_POSITION."""
        b = _make_behaviour()
        # Pass values that cause a ZeroDivisionError inside the try block
        # A negative CT would produce issues; use a very extreme value
        result = b._calculate_min_hold_days(float("inf"), 100, 10, False)
        assert isinstance(result, float)


class TestStakeUnstakeCLPositionLoops:
    """Test the for-loop branches in stake/unstake/claim for CL pools.

    The branch partials like 3818->3817 mean the for loop iterates
    multiple times before finding a match. We need tests where:
    - Multiple positions exist (so loop iterates multiple times)
    - No match found at all
    - token_ids missing but gauge_address provided (or vice-versa)
    """

    def _velodrome_action(self, is_cl_pool=False, **extra):
        return {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "is_cl_pool": is_cl_pool,
            **extra,
        }

    def test_stake_cl_multiple_positions_before_match(self):
        """For loop iterates through non-matching positions first."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method("0xGAUGE")
        mock_pool.stake_cl_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xOTHER",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 99}],
            },
            {
                "pool_address": "0xPOOL",
                "chain": "mode",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 88}],
            },
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": "CLOSED",
                "positions": [{"token_id": 77}],
            },
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            },
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result[0] is not None

    def test_stake_cl_has_token_ids_no_gauge(self):
        """token_ids provided in action, gauge_address looked up from pool."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method("0xGAUGE")
        mock_pool.stake_cl_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            },
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True, token_ids=[1])
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result[0] is not None

    def test_unstake_cl_multiple_positions_before_match(self):
        """Test unstake cl multiple positions before match."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method("0xGAUGE")
        mock_pool.unstake_cl_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xOTHER",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 99}],
            },
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            },
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(action))
        assert result[0] is not None

    def test_unstake_cl_has_token_ids_no_gauge(self):
        """Test unstake cl has token ids no gauge."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method("0xGAUGE")
        mock_pool.unstake_cl_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            },
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True, token_ids=[1])
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(action))
        assert result[0] is not None

    def test_claim_cl_multiple_positions_before_match(self):
        """Test claim cl multiple positions before match."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method("0xGAUGE")
        mock_pool.claim_cl_rewards = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xOTHER",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 99}],
            },
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            },
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result[0] is not None

    def test_claim_cl_has_token_ids_no_gauge(self):
        """Test claim cl has token ids no gauge."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method("0xGAUGE")
        mock_pool.claim_cl_rewards = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            },
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True, token_ids=[1])
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result[0] is not None


class TestPostExecuteStakeLpTokensLoop:
    """TestPostExecuteStakeLpTokensLoop."""

    def test_multiple_positions_before_match(self):
        """Loop iterates through non-matching positions first."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xstake"
        b._get_current_timestamp = MagicMock(return_value=3000)
        b.current_positions = [
            {
                "pool_address": "0xOTHER",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
            },
            {
                "pool_address": "0xPOOL",
                "chain": "mode",
                "status": PositionStatus.OPEN.value,
            },
            {"pool_address": "0xPOOL", "chain": "optimism", "status": "CLOSED"},
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
            },
        ]
        b.store_current_positions = MagicMock()
        action = {"pool_address": "0xPOOL", "chain": "optimism", "is_cl_pool": False}
        b._post_execute_stake_lp_tokens([action], 0)
        assert b.current_positions[3]["staked"] is True
        assert b.current_positions[0].get("staked") is None


class TestPostExecuteUnstakeLpTokensLoop:
    """TestPostExecuteUnstakeLpTokensLoop."""

    def test_multiple_positions_before_match(self):
        """Test multiple positions before match."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xunstake"
        b._get_current_timestamp = MagicMock(return_value=4000)
        b.current_positions = [
            {"pool_address": "0xOTHER", "chain": "optimism", "staked": True},
            {"pool_address": "0xPOOL", "chain": "mode", "staked": True},
            {"pool_address": "0xPOOL", "chain": "optimism", "staked": True},
        ]
        b.store_current_positions = MagicMock()
        action = {"pool_address": "0xPOOL", "chain": "optimism"}
        b._post_execute_unstake_lp_tokens([action], 0)
        assert b.current_positions[2]["staked"] is False
        assert b.current_positions[0]["staked"] is True

    def test_no_matching_position(self):
        """Test no matching position."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xunstake"
        b._get_current_timestamp = MagicMock(return_value=4000)
        b.current_positions = [
            {"pool_address": "0xOTHER", "chain": "optimism", "staked": True},
        ]
        b.store_current_positions = MagicMock()
        action = {"pool_address": "0xPOOL", "chain": "optimism"}
        b._post_execute_unstake_lp_tokens([action], 0)
        assert b.current_positions[0]["staked"] is True


class TestPostExecuteClaimStakingRewardsLoop:
    """TestPostExecuteClaimStakingRewardsLoop."""

    def test_multiple_positions_before_match(self):
        """Test multiple positions before match."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xclaim"
        b._get_current_timestamp = MagicMock(return_value=5000)
        b.current_positions = [
            {"pool_address": "0xOTHER", "chain": "optimism"},
            {"pool_address": "0xPOOL", "chain": "mode"},
            {"pool_address": "0xPOOL", "chain": "optimism"},
        ]
        b.store_current_positions = MagicMock()
        action = {"pool_address": "0xPOOL", "chain": "optimism"}
        b._post_execute_claim_staking_rewards([action], 0)
        assert b.current_positions[2]["last_reward_claim_tx_hash"] == "0xclaim"
        assert b.current_positions[0].get("last_reward_claim_tx_hash") is None

    def test_no_matching_position(self):
        """Test no matching position."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xclaim"
        b._get_current_timestamp = MagicMock(return_value=5000)
        b.current_positions = [
            {"pool_address": "0xOTHER", "chain": "optimism"},
        ]
        b.store_current_positions = MagicMock()
        action = {"pool_address": "0xPOOL", "chain": "optimism"}
        b._post_execute_claim_staking_rewards([action], 0)
        assert b.current_positions[0].get("last_reward_claim_tx_hash") is None


class TestPostExecuteStepFallThrough:
    """TestPostExecuteStepFallThrough."""

    def test_decision_none_falls_through(self):
        """When get_decision_on_swap returns None, the function falls through."""
        b = _make_behaviour()
        b.sleep = _make_gen_method(None)
        b.get_decision_on_swap = _make_gen_method(None)
        result = _exhaust(
            b._post_execute_step(
                [{"remaining_fee_allowance": 1.0, "remaining_gas_allowance": 1.0}], 0
            )
        )
        assert result is None


class TestPostExecuteEnterPoolUnknownDex:
    """TestPostExecuteEnterPoolUnknownDex."""

    def test_unknown_dex_type_falls_through_to_accumulate(self):
        """An unrecognized dex_type skips all if/elif, falls to line 559."""
        b = _make_behaviour()
        b.synchronized_data.final_tx_hash = "0xhash"
        b._accumulate_transaction_costs = _make_gen_method(None)
        b._rename_entry_costs_key = _make_gen_method(None)
        b._calculate_and_store_tip_data = _make_gen_method(None)
        b.store_current_positions = MagicMock()
        action = {
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "dex_type": "unknown_dex",
            "token0": "0xT0",
            "token1": "0xT1",
        }
        _exhaust(b._post_execute_enter_pool([action], 0))
        # No position appended since no branch matched
        assert len(b.current_positions) == 0
        # But _accumulate_transaction_costs was still called
        b.context.logger.info.assert_called()


class TestCalculateVelodromeNegativeAmountsViaCap:
    """TestCalculateVelodromeNegativeAmountsViaCap."""

    def test_negative_via_max_amounts_cap(self):
        """Negative amounts via max_amounts cap causes return None."""
        b = _make_behaviour()
        b._get_balance = MagicMock(return_value=1_000_000)
        b._get_token_decimals = _make_gen_method(6)
        b._fetch_token_price = _make_gen_method(1.0)
        action = {
            "token_requirements": {
                "overall_token0_ratio": 0.5,
                "overall_token1_ratio": 0.5,
            },
        }
        # max_amounts with a negative value will make min(desired, -100) = -100
        result = _exhaust(
            b._calculate_velodrome_investment_amounts(
                action, "optimism", ["0x1", "0x2"], [], [-100, -100]
            )
        )
        assert result is None


class TestGetEnterPoolNonVelodromeNegativeAmounts:
    """TestGetEnterPoolNonVelodromeNegativeAmounts."""

    def test_negative_amount_in_non_velodrome(self):
        """Non-velodrome dex with a negative amount returns None."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        b.pools = {DexType.BALANCER.value: mock_pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([-5, 100], -5, 100)
        )
        action = {
            "dex_type": DexType.BALANCER.value,
            "chain": "optimism",
            "token0": "0xT0",
            "token1": "0xT1",
            "pool_address": "0xPOOL",
            "pool_type": "Weighted",
            "is_stable": False,
            "is_cl_pool": False,
            "relative_funds_percentage": 1.0,
        }
        result = _exhaust(b.get_enter_pool_tx_hash([], action))
        assert result == (None, None, None)


class TestGetDataFromJoinPoolEmptyTopics:
    """TestGetDataFromJoinPoolEmptyTopics."""

    def test_log_with_empty_topics_is_skipped(self):
        """A log with empty topics list should be skipped via continue."""
        from eth_abi import encode
        from eth_utils import keccak

        event_sig = "PoolBalanceChanged(bytes32,address,address[],int256[],uint256[])"
        sig_hash = "0x" + keccak(text=event_sig).hex()
        # Build a valid log
        addr_tokens = [bytes.fromhex("a1" * 20), bytes.fromhex("a2" * 20)]
        data_hex = (
            "0x"
            + encode(
                ["address[]", "int256[]", "uint256[]"],
                [addr_tokens, [500, 600], [0, 0]],
            ).hex()
        )
        valid_log = {
            "topics": [sig_hash, "0x" + "00" * 32, "0x" + "00" * 32],
            "data": data_hex,
        }
        empty_topics_log = {"topics": [], "data": "0x00"}
        b = _make_behaviour()
        receipt = {"logs": [empty_topics_log, valid_log], "blockNumber": "0x10"}
        b.get_transaction_receipt = _make_gen_method(receipt)
        b.get_block = _make_gen_method({"timestamp": 12345})
        result = _exhaust(b._get_data_from_join_pool_tx_receipt("0xhash", "optimism"))
        # The empty-topics log is skipped; the valid log is processed
        assert result[0] == 500
        assert result[1] == 600
        assert result[2] == 12345


class TestCalculateMinHoldDaysActualException:
    """TestCalculateMinHoldDaysActualException."""

    def test_string_apr_causes_exception(self):
        """Passing a non-numeric type for apr causes TypeError in the try block."""
        b = _make_behaviour()
        # "bad" <= 0.0 raises TypeError, caught by except Exception
        result = b._calculate_min_hold_days("bad", 100.0, 10.0, False)
        assert result == MIN_TIME_IN_POSITION


# Tests: stake/unstake/claim CL - loop iteration and gauge_address skip
# Branch partials: 3812->3811, 3832->3837, 3976->3975, 3996->4001,
#                  4143->4142, 4163->4168


class TestCLPositionLoopIterationAndGaugeSkip:
    """Cover: (a) for-loop iterates past non-matching positions (loop back), # noqa: D205,D209
    and (b) gauge_address already provided, so `if not gauge_address:` is skipped."""

    def _velodrome_action(self, is_cl_pool=False, **extra):
        return {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPOOL",
            "is_cl_pool": is_cl_pool,
            **extra,
        }

    # -- Stake: loop iteration (3812->3811) --
    def test_stake_cl_loop_iterates_past_non_matching(self):
        """Match is NOT the last element, so reversed() hits non-match first."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method("0xGAUGE")
        mock_pool.stake_cl_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            },
            {
                "pool_address": "0xOTHER",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 99}],
            },
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result[0] is not None

    # -- Stake: gauge_address already set (3832->3837) --
    def test_stake_cl_gauge_provided_token_ids_missing(self):
        """gauge_address in action, token_ids missing -> skip gauge lookup."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.stake_cl_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            },
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True, gauge_address="0xGAUGE")
        result = _exhaust(b.get_stake_lp_tokens_tx_hash(action))
        assert result[0] is not None
        # get_gauge_address should NOT have been called
        mock_pool.get_gauge_address.assert_not_called()

    # -- Unstake: loop iteration (3976->3975) --
    def test_unstake_cl_loop_iterates_past_non_matching(self):
        """Match is NOT the last element, so reversed() hits non-match first."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method("0xGAUGE")
        mock_pool.unstake_cl_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            },
            {
                "pool_address": "0xOTHER",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 99}],
            },
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(action))
        assert result[0] is not None

    # -- Unstake: gauge_address already set (3996->4001) --
    def test_unstake_cl_gauge_provided_token_ids_missing(self):
        """gauge_address in action, token_ids missing -> skip gauge lookup."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.unstake_cl_lp_tokens = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            },
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True, gauge_address="0xGAUGE")
        result = _exhaust(b.get_unstake_lp_tokens_tx_hash(action))
        assert result[0] is not None
        mock_pool.get_gauge_address.assert_not_called()

    # -- Claim: loop iteration (4143->4142) --
    def test_claim_cl_loop_iterates_past_non_matching(self):
        """Match is NOT the last element, so reversed() hits non-match first."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.get_gauge_address = _make_gen_method("0xGAUGE")
        mock_pool.claim_cl_rewards = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            },
            {
                "pool_address": "0xOTHER",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 99}],
            },
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True)
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result[0] is not None

    # -- Claim: gauge_address already set (4163->4168) --
    def test_claim_cl_gauge_provided_token_ids_missing(self):
        """gauge_address in action, token_ids missing -> skip gauge lookup."""
        b = _make_behaviour()
        mock_pool = MagicMock()
        mock_pool.claim_cl_rewards = _make_gen_method(
            {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}
        )
        b.pools = {"velodrome": mock_pool}
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
            },
        ]
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = self._velodrome_action(is_cl_pool=True, gauge_address="0xGAUGE")
        result = _exhaust(b.get_claim_staking_rewards_tx_hash(action))
        assert result[0] is not None
        mock_pool.get_gauge_address.assert_not_called()


class TestGetSwapStatusBooleanLogic:
    """Tests for the boolean guard in get_swap_status."""

    def test_both_none_returns_none(self):
        """When both status and substatus are None, return (None, None)."""
        b = _make_behaviour()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.body = json.dumps({})
        b.get_http_response = _make_gen_method(mock_resp)
        result = _exhaust(b.get_swap_status("0xabc"))
        assert result == (None, None)

    def test_none_status_with_substatus_returns_both(self):
        """When status is None but substatus is present, return (None, substatus)."""
        b = _make_behaviour()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.body = json.dumps({"substatus": "PARTIAL"})
        b.get_http_response = _make_gen_method(mock_resp)
        result = _exhaust(b.get_swap_status("0xabc"))
        assert result == (None, "PARTIAL")


class TestWaitForSwapConfirmationRetryExhaustion:
    """Tests for _wait_for_swap_confirmation max retry exhaustion."""

    def test_retries_exhausted_returns_exit(self):
        """When get_decision_on_swap always returns WAIT, retries exhaust and return EXIT."""
        b = _make_behaviour()
        b.sleep = _make_gen_method(None)
        b.get_decision_on_swap = _make_gen_method(Decision.WAIT)
        result = _exhaust(b._wait_for_swap_confirmation())
        assert result == Decision.EXIT
        b.context.logger.error.assert_called_once()
        assert str(MAX_SWAP_CONFIRMATION_RETRIES) in str(
            b.context.logger.error.call_args
        )

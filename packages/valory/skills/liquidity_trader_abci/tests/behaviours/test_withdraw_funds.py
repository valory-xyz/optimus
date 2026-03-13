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

"""Tests for behaviours/withdraw_funds.py."""

# pylint: skip-file

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    Action,
    PositionStatus,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.withdraw_funds import (
    WithdrawFundsBehaviour,
)


def _make_behaviour():
    """Create a WithdrawFundsBehaviour without __init__."""
    obj = object.__new__(WithdrawFundsBehaviour)
    ctx = MagicMock()
    obj.__dict__["_context"] = ctx
    return obj


def _drive(gen):
    """Drive a generator to completion."""
    val = None
    while True:
        try:
            val = gen.send(val)
        except StopIteration as exc:
            return exc.value


class TestWithdrawFundsBehaviour:
    """Tests for WithdrawFundsBehaviour."""

    def _run_async_act(
        self,
        obj,
        investing_paused=False,
        withdrawal_data=None,
        target_address="",
        positions=None,
        portfolio_data=None,
        withdrawal_actions=None,
    ):
        """Helper to run async_act with various configurations."""
        benchmark_mock = MagicMock()
        obj.context.benchmark_tool.measure.return_value = benchmark_mock
        obj.context.agent_address = "0xagent"

        def fake_read_investing_paused():
            yield
            return investing_paused

        def fake_read_withdrawal_data():
            yield
            return withdrawal_data

        def fake_update_withdrawal_status(status, message):
            yield

        def fake_reset_withdrawal_flags():
            yield

        def fake_prepare_withdrawal_actions(positions_, target_addr_, portfolio_data_):
            yield
            return withdrawal_actions if withdrawal_actions is not None else []

        def fake_send(*args, **kwargs):
            yield

        def fake_wait(*args, **kwargs):
            yield

        obj._read_investing_paused = fake_read_investing_paused
        obj._read_withdrawal_data = fake_read_withdrawal_data
        obj._update_withdrawal_status = fake_update_withdrawal_status
        obj._reset_withdrawal_flags = fake_reset_withdrawal_flags
        obj._prepare_withdrawal_actions = fake_prepare_withdrawal_actions
        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()
        obj.current_positions = positions or []
        obj.portfolio_data = portfolio_data or {}

        gen = obj.async_act()
        _drive(gen)
        return obj

    def test_async_act_not_paused(self) -> None:
        """Test async_act when investing is not paused."""
        obj = _make_behaviour()
        self._run_async_act(obj, investing_paused=False)
        obj.set_done.assert_called_once()

    def test_async_act_paused_no_withdrawal_data(self) -> None:
        """Test async_act when paused but no withdrawal data found."""
        obj = _make_behaviour()
        self._run_async_act(
            obj,
            investing_paused=True,
            withdrawal_data=None,
            positions=[
                {"pool_address": "0x1", "dex_type": "uniswap", "status": "OPEN"}
            ],
        )
        obj.set_done.assert_called_once()

    def test_async_act_paused_no_target_address(self) -> None:
        """Test async_act when paused but target address is empty."""
        obj = _make_behaviour()
        self._run_async_act(
            obj,
            investing_paused=True,
            withdrawal_data={"withdrawal_target_address": ""},
            positions=[
                {"pool_address": "0x1", "dex_type": "uniswap", "status": "OPEN"}
            ],
        )
        obj.set_done.assert_called_once()

    def test_async_act_paused_no_actions(self) -> None:
        """Test async_act when paused but no withdrawal actions prepared."""
        obj = _make_behaviour()
        self._run_async_act(
            obj,
            investing_paused=True,
            withdrawal_data={"withdrawal_target_address": "0xtarget"},
            positions=[
                {"pool_address": "0x1", "dex_type": "uniswap", "status": "OPEN"}
            ],
            withdrawal_actions=[],
        )
        obj.set_done.assert_called_once()

    def test_async_act_paused_with_actions(self) -> None:
        """Test async_act when paused and withdrawal actions are prepared."""
        obj = _make_behaviour()
        self._run_async_act(
            obj,
            investing_paused=True,
            withdrawal_data={"withdrawal_target_address": "0xtarget"},
            positions=[
                {"pool_address": "0x1", "dex_type": "uniswap", "status": "OPEN"}
            ],
            withdrawal_actions=[{"action": "exit_pool"}],
        )
        obj.set_done.assert_called_once()

    def test_async_act_position_logging(self) -> None:
        """Test async_act logs position details when investing is paused."""
        obj = _make_behaviour()
        positions = [
            {"pool_address": "0xA", "dex_type": "velodrome", "status": "OPEN"},
            {"pool_address": "0xB", "dex_type": "balancer", "status": "closed"},
        ]
        self._run_async_act(
            obj,
            investing_paused=True,
            withdrawal_data={"withdrawal_target_address": "0xtarget"},
            positions=positions,
            withdrawal_actions=[{"action": "exit_pool"}],
        )
        obj.set_done.assert_called_once()


class TestReadWithdrawalData:
    """Tests for _read_withdrawal_data."""

    def test_success(self) -> None:
        """Test successful read of withdrawal data."""
        obj = _make_behaviour()

        def fake_read_kv(keys):
            yield
            return {"withdrawal_id": "123", "withdrawal_target_address": "0x1"}

        obj._read_kv = fake_read_kv
        gen = obj._read_withdrawal_data()
        result = _drive(gen)
        assert result == {"withdrawal_id": "123", "withdrawal_target_address": "0x1"}

    def test_none_result(self) -> None:
        """Test when _read_kv returns None."""
        obj = _make_behaviour()

        def fake_read_kv(keys):
            yield
            return None

        obj._read_kv = fake_read_kv
        gen = obj._read_withdrawal_data()
        result = _drive(gen)
        assert result is None

    def test_exception(self) -> None:
        """Test when _read_kv raises an exception."""
        obj = _make_behaviour()

        def fake_read_kv(keys):
            raise ValueError("boom")
            yield  # noqa: unreachable

        obj._read_kv = fake_read_kv
        gen = obj._read_withdrawal_data()
        result = _drive(gen)
        assert result is None


class TestPrepareWithdrawalActions:
    """Tests for _prepare_withdrawal_actions."""

    def _make_obj_with_stubs(
        self,
        unstake_actions=None,
        exit_actions=None,
        swap_actions=None,
        transfer_actions=None,
    ):
        """Create a behaviour with stubbed sub-methods."""
        obj = _make_behaviour()
        obj._prepare_unstaking_actions = MagicMock(return_value=unstake_actions or [])
        obj._prepare_exit_pool_actions = MagicMock(return_value=exit_actions or [])
        obj._prepare_swap_to_usdc_actions_standard = MagicMock(
            return_value=swap_actions or []
        )
        obj._prepare_transfer_usdc_actions_standard = MagicMock(
            return_value=transfer_actions or []
        )

        def fake_update_status(status, message):
            yield

        def fake_reset_flags():
            yield

        obj._update_withdrawal_status = fake_update_status
        obj._reset_withdrawal_flags = fake_reset_flags
        return obj

    def test_all_actions_present(self) -> None:
        """Test with all action types present."""
        obj = self._make_obj_with_stubs(
            unstake_actions=[{"action": "unstake"}],
            exit_actions=[{"action": Action.EXIT_POOL.value}],
            swap_actions=[{"action": Action.FIND_BRIDGE_ROUTE.value}],
            transfer_actions=[{"action": Action.WITHDRAW.value}],
        )
        gen = obj._prepare_withdrawal_actions([], "0xtarget", {})
        result = _drive(gen)
        assert len(result) == 4

    def test_no_unstake_actions(self) -> None:
        """Test with no unstaking actions."""
        obj = self._make_obj_with_stubs(
            exit_actions=[{"action": Action.EXIT_POOL.value}],
            transfer_actions=[{"action": Action.WITHDRAW.value}],
        )
        gen = obj._prepare_withdrawal_actions([], "0xtarget", {})
        result = _drive(gen)
        assert len(result) == 2

    def test_no_exit_actions(self) -> None:
        """Test with no exit actions."""
        obj = self._make_obj_with_stubs(
            transfer_actions=[{"action": Action.WITHDRAW.value}],
        )
        gen = obj._prepare_withdrawal_actions([], "0xtarget", {})
        result = _drive(gen)
        assert len(result) == 1

    def test_no_swap_actions(self) -> None:
        """Test with no swap actions."""
        obj = self._make_obj_with_stubs(
            exit_actions=[{"action": Action.EXIT_POOL.value}],
            transfer_actions=[{"action": Action.WITHDRAW.value}],
        )
        gen = obj._prepare_withdrawal_actions([], "0xtarget", {})
        result = _drive(gen)
        assert len(result) == 2

    def test_no_transfer_actions_triggers_completion(self) -> None:
        """Test with no transfer actions triggers COMPLETED status."""
        obj = self._make_obj_with_stubs(transfer_actions=[])
        gen = obj._prepare_withdrawal_actions([], "0xtarget", {})
        result = _drive(gen)
        assert len(result) == 0

    def test_all_empty(self) -> None:
        """Test when all action lists are empty."""
        obj = self._make_obj_with_stubs()
        gen = obj._prepare_withdrawal_actions([], "0xtarget", {})
        result = _drive(gen)
        assert len(result) == 0


class TestPrepareUnstakingActions:
    """Tests for _prepare_unstaking_actions."""

    def test_open_position_with_staking(self) -> None:
        """Test open position with staking metadata returns unstake action."""
        obj = _make_behaviour()
        obj._has_staking_metadata = MagicMock(return_value=True)
        obj._build_unstake_lp_tokens_action = MagicMock(
            return_value={"action": "unstake"}
        )

        positions = [{"status": PositionStatus.OPEN.value}]
        result = obj._prepare_unstaking_actions(positions)
        assert len(result) == 1

    def test_open_position_without_staking(self) -> None:
        """Test open position without staking metadata returns empty."""
        obj = _make_behaviour()
        obj._has_staking_metadata = MagicMock(return_value=False)

        positions = [{"status": PositionStatus.OPEN.value}]
        result = obj._prepare_unstaking_actions(positions)
        assert len(result) == 0

    def test_open_position_build_returns_none(self) -> None:
        """Test open position with staking but build returns None."""
        obj = _make_behaviour()
        obj._has_staking_metadata = MagicMock(return_value=True)
        obj._build_unstake_lp_tokens_action = MagicMock(return_value=None)

        positions = [{"status": PositionStatus.OPEN.value}]
        result = obj._prepare_unstaking_actions(positions)
        assert len(result) == 0

    def test_closed_position_skipped(self) -> None:
        """Test closed position is skipped."""
        obj = _make_behaviour()
        positions = [{"status": "closed"}]
        result = obj._prepare_unstaking_actions(positions)
        assert len(result) == 0

    def test_empty_positions(self) -> None:
        """Test empty positions list."""
        obj = _make_behaviour()
        result = obj._prepare_unstaking_actions([])
        assert len(result) == 0


class TestPrepareExitPoolActions:
    """Tests for _prepare_exit_pool_actions."""

    def test_open_position_success(self) -> None:
        """Test open position creates exit action."""
        obj = _make_behaviour()
        obj._build_exit_pool_action_base = MagicMock(
            return_value={
                "action": Action.EXIT_POOL.value,
                "chain": "optimism",
            }
        )

        positions = [
            {
                "status": "OPEN",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "token0": "0xtoken0",
                "token1": "0xtoken1",
                "pool_fee": 3000,
                "tick_spacing": 60,
                "tick_ranges": [[100, 200]],
            }
        ]
        result = obj._prepare_exit_pool_actions(positions)
        assert len(result) == 1
        assert result[0]["pool_fee"] == 3000
        assert result[0]["tick_spacing"] == 60
        assert result[0]["tick_ranges"] == [[100, 200]]

    def test_open_position_lowercase(self) -> None:
        """Test position with lowercase 'open' status."""
        obj = _make_behaviour()
        obj._build_exit_pool_action_base = MagicMock(
            return_value={
                "action": Action.EXIT_POOL.value,
            }
        )

        positions = [
            {
                "status": "open",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "token0": "0xtoken0",
                "token1": "0xtoken1",
            }
        ]
        result = obj._prepare_exit_pool_actions(positions)
        assert len(result) == 1

    def test_open_position_build_returns_none(self) -> None:
        """Test open position when build returns None."""
        obj = _make_behaviour()
        obj._build_exit_pool_action_base = MagicMock(return_value=None)

        positions = [{"status": "OPEN"}]
        result = obj._prepare_exit_pool_actions(positions)
        assert len(result) == 0

    def test_closed_position_skipped(self) -> None:
        """Test closed position is skipped."""
        obj = _make_behaviour()
        positions = [{"status": "closed"}]
        result = obj._prepare_exit_pool_actions(positions)
        assert len(result) == 0

    def test_no_optional_fields(self) -> None:
        """Test open position without optional fields (pool_fee, tick_spacing, tick_ranges)."""
        obj = _make_behaviour()
        obj._build_exit_pool_action_base = MagicMock(
            return_value={
                "action": Action.EXIT_POOL.value,
            }
        )

        positions = [
            {
                "status": "OPEN",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "token0": "0xtoken0",
                "token1": "0xtoken1",
            }
        ]
        result = obj._prepare_exit_pool_actions(positions)
        assert len(result) == 1
        assert "pool_fee" not in result[0]
        assert "tick_spacing" not in result[0]
        assert "tick_ranges" not in result[0]

    def test_pool_fee_none(self) -> None:
        """Test when pool_fee is explicitly None."""
        obj = _make_behaviour()
        obj._build_exit_pool_action_base = MagicMock(
            return_value={
                "action": Action.EXIT_POOL.value,
            }
        )

        positions = [
            {
                "status": "OPEN",
                "token0_symbol": "A",
                "token1_symbol": "B",
                "token0": "0x0",
                "token1": "0x1",
                "pool_fee": None,
                "tick_spacing": None,
                "tick_ranges": None,
            }
        ]
        result = obj._prepare_exit_pool_actions(positions)
        assert len(result) == 1
        assert "pool_fee" not in result[0]
        assert "tick_spacing" not in result[0]
        assert "tick_ranges" not in result[0]


class TestPrepareSwapToUsdcActionsStandard:
    """Tests for _prepare_swap_to_usdc_actions_standard."""

    def _make_obj(self, usdc_addr="0xusdc", olas_addr="0xolas"):
        """Create a behaviour with common stubs."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.target_investment_chains = ["optimism"]
        obj.context.params = params_mock
        obj._get_usdc_address = MagicMock(return_value=usdc_addr)
        obj._get_olas_address = MagicMock(return_value=olas_addr)
        return obj

    def test_swap_non_usdc_asset(self) -> None:
        """Test swap action created for non-USDC asset with value > 1."""
        obj = self._make_obj()
        obj._build_swap_to_usdc_action = MagicMock(return_value={"action": "swap"})

        portfolio = {
            "portfolio_breakdown": [
                {"asset": "WETH", "address": "0xweth", "value_usd": 100},
            ]
        }
        result = obj._prepare_swap_to_usdc_actions_standard(portfolio)
        assert len(result) == 1

    def test_skip_usdc_asset(self) -> None:
        """Test USDC asset is skipped."""
        obj = self._make_obj(usdc_addr="0xusdc")

        portfolio = {
            "portfolio_breakdown": [
                {"asset": "USDC", "address": "0xusdc", "value_usd": 500},
            ]
        }
        result = obj._prepare_swap_to_usdc_actions_standard(portfolio)
        assert len(result) == 0

    def test_skip_olas_asset(self) -> None:
        """Test OLAS asset is skipped."""
        obj = self._make_obj(olas_addr="0xolas")

        portfolio = {
            "portfolio_breakdown": [
                {"asset": "OLAS", "address": "0xolas", "value_usd": 500},
            ]
        }
        result = obj._prepare_swap_to_usdc_actions_standard(portfolio)
        assert len(result) == 0

    def test_skip_small_balance(self) -> None:
        """Test asset with balance <= 1 is skipped."""
        obj = self._make_obj()

        portfolio = {
            "portfolio_breakdown": [
                {"asset": "WETH", "address": "0xweth", "value_usd": 0.5},
            ]
        }
        result = obj._prepare_swap_to_usdc_actions_standard(portfolio)
        assert len(result) == 0

    def test_build_swap_returns_none(self) -> None:
        """Test when _build_swap_to_usdc_action returns None."""
        obj = self._make_obj()
        obj._build_swap_to_usdc_action = MagicMock(return_value=None)

        portfolio = {
            "portfolio_breakdown": [
                {"asset": "WETH", "address": "0xweth", "value_usd": 100},
            ]
        }
        result = obj._prepare_swap_to_usdc_actions_standard(portfolio)
        assert len(result) == 0

    def test_empty_portfolio(self) -> None:
        """Test empty portfolio breakdown."""
        obj = self._make_obj()
        portfolio = {"portfolio_breakdown": []}
        result = obj._prepare_swap_to_usdc_actions_standard(portfolio)
        assert len(result) == 0

    def test_no_token_address_not_usdc(self) -> None:
        """Test asset with no token_address (None) is not matched as USDC."""
        obj = self._make_obj()
        obj._build_swap_to_usdc_action = MagicMock(return_value={"action": "swap"})

        portfolio = {
            "portfolio_breakdown": [
                {"asset": "UNKNOWN", "address": None, "value_usd": 100},
            ]
        }
        result = obj._prepare_swap_to_usdc_actions_standard(portfolio)
        # With None address, the USDC/OLAS checks fail (token_address is falsy), so it proceeds
        assert len(result) == 1

    def test_usdc_address_none(self) -> None:
        """Test when _get_usdc_address returns None."""
        obj = self._make_obj(usdc_addr=None)
        obj._build_swap_to_usdc_action = MagicMock(return_value={"action": "swap"})

        portfolio = {
            "portfolio_breakdown": [
                {"asset": "WETH", "address": "0xweth", "value_usd": 100},
            ]
        }
        result = obj._prepare_swap_to_usdc_actions_standard(portfolio)
        assert len(result) == 1

    def test_olas_address_none(self) -> None:
        """Test when _get_olas_address returns None."""
        obj = self._make_obj(olas_addr=None)
        obj._build_swap_to_usdc_action = MagicMock(return_value={"action": "swap"})

        portfolio = {
            "portfolio_breakdown": [
                {"asset": "WETH", "address": "0xweth", "value_usd": 100},
            ]
        }
        result = obj._prepare_swap_to_usdc_actions_standard(portfolio)
        assert len(result) == 1

    def test_value_usd_exactly_one(self) -> None:
        """Test asset with value_usd exactly 1 is skipped (<=1)."""
        obj = self._make_obj()

        portfolio = {
            "portfolio_breakdown": [
                {"asset": "WETH", "address": "0xweth", "value_usd": 1},
            ]
        }
        result = obj._prepare_swap_to_usdc_actions_standard(portfolio)
        assert len(result) == 0

    def test_no_has_get_olas_address(self) -> None:
        """Test when _get_olas_address attribute doesn't exist (hasattr returns False)."""
        obj = self._make_obj()
        # Remove the _get_olas_address attribute
        del obj._get_olas_address
        obj._build_swap_to_usdc_action = MagicMock(return_value={"action": "swap"})

        portfolio = {
            "portfolio_breakdown": [
                {"asset": "WETH", "address": "0xweth", "value_usd": 100},
            ]
        }
        result = obj._prepare_swap_to_usdc_actions_standard(portfolio)
        assert len(result) == 1


class TestPrepareTransferUsdcActionsStandard:
    """Tests for _prepare_transfer_usdc_actions_standard."""

    def test_creates_transfer_action(self) -> None:
        """Test creates a transfer action."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.target_investment_chains = ["optimism"]
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        obj.context.params = params_mock
        obj._get_usdc_address = MagicMock(return_value="0xusdc")

        result = obj._prepare_transfer_usdc_actions_standard("0xtarget")
        assert len(result) == 1
        assert result[0]["action"] == Action.WITHDRAW.value
        assert result[0]["to_address"] == "0xtarget"
        assert result[0]["token_symbol"] == "USDC"
        assert result[0]["funds_percentage"] == 1.0


class TestUpdateWithdrawalStatus:
    """Tests for _update_withdrawal_status."""

    def test_status_completed(self) -> None:
        """Test COMPLETED status sets completed_at and investing_paused."""
        obj = _make_behaviour()
        written = {}

        def fake_write_kv(data):
            written.update(data)
            yield

        obj._write_kv = fake_write_kv
        gen = obj._update_withdrawal_status("COMPLETED", "Done!")
        _drive(gen)
        assert "withdrawal_completed_at" in written
        assert written["investing_paused"] == "false"

    def test_status_failed(self) -> None:
        """Test FAILED status sets investing_paused."""
        obj = _make_behaviour()
        written = {}

        def fake_write_kv(data):
            written.update(data)
            yield

        obj._write_kv = fake_write_kv
        gen = obj._update_withdrawal_status("FAILED", "Error")
        _drive(gen)
        assert written["investing_paused"] == "false"
        assert "withdrawal_completed_at" not in written

    def test_status_withdrawing(self) -> None:
        """Test WITHDRAWING status doesn't set extra fields."""
        obj = _make_behaviour()
        written = {}

        def fake_write_kv(data):
            written.update(data)
            yield

        obj._write_kv = fake_write_kv
        gen = obj._update_withdrawal_status("WITHDRAWING", "In progress")
        _drive(gen)
        assert "investing_paused" not in written
        assert "withdrawal_completed_at" not in written

    def test_exception_in_write_kv(self) -> None:
        """Test exception during _write_kv."""
        obj = _make_behaviour()

        def fake_write_kv(data):
            raise ValueError("write error")
            yield  # noqa: unreachable

        obj._write_kv = fake_write_kv
        gen = obj._update_withdrawal_status("COMPLETED", "Done!")
        _drive(gen)
        obj.context.logger.error.assert_called()


class TestResetWithdrawalFlags:
    """Tests for _reset_withdrawal_flags."""

    def test_success(self) -> None:
        """Test successful reset."""
        obj = _make_behaviour()
        written = {}

        def fake_write_kv(data):
            written.update(data)
            yield

        obj._write_kv = fake_write_kv
        gen = obj._reset_withdrawal_flags()
        _drive(gen)
        assert written["investing_paused"] == "false"

    def test_exception(self) -> None:
        """Test exception during reset."""
        obj = _make_behaviour()

        def fake_write_kv(data):
            raise ValueError("write error")
            yield  # noqa: unreachable

        obj._write_kv = fake_write_kv
        gen = obj._reset_withdrawal_flags()
        _drive(gen)
        obj.context.logger.error.assert_called()


class TestReadInvestingPaused:
    """Tests for _read_investing_paused."""

    def test_value_true(self) -> None:
        """Test returns True when investing_paused is 'true'."""
        obj = _make_behaviour()

        def fake_read_kv(keys):
            yield
            return {"investing_paused": "true"}

        obj._read_kv = fake_read_kv
        gen = obj._read_investing_paused()
        result = _drive(gen)
        assert result is True

    def test_value_false(self) -> None:
        """Test returns False when investing_paused is 'false'."""
        obj = _make_behaviour()

        def fake_read_kv(keys):
            yield
            return {"investing_paused": "false"}

        obj._read_kv = fake_read_kv
        gen = obj._read_investing_paused()
        result = _drive(gen)
        assert result is False

    def test_value_true_mixed_case(self) -> None:
        """Test returns True when investing_paused is 'True' (mixed case)."""
        obj = _make_behaviour()

        def fake_read_kv(keys):
            yield
            return {"investing_paused": "True"}

        obj._read_kv = fake_read_kv
        gen = obj._read_investing_paused()
        result = _drive(gen)
        assert result is True

    def test_result_none(self) -> None:
        """Test returns False when result is None."""
        obj = _make_behaviour()

        def fake_read_kv(keys):
            yield
            return None

        obj._read_kv = fake_read_kv
        gen = obj._read_investing_paused()
        result = _drive(gen)
        assert result is False

    def test_value_none(self) -> None:
        """Test returns False when investing_paused value is None."""
        obj = _make_behaviour()

        def fake_read_kv(keys):
            yield
            return {"investing_paused": None}

        obj._read_kv = fake_read_kv
        gen = obj._read_investing_paused()
        result = _drive(gen)
        assert result is False

    def test_exception(self) -> None:
        """Test returns False when exception occurs."""
        obj = _make_behaviour()

        def fake_read_kv(keys):
            raise ValueError("boom")
            yield  # noqa: unreachable

        obj._read_kv = fake_read_kv
        gen = obj._read_investing_paused()
        result = _drive(gen)
        assert result is False

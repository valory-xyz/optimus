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

"""Tests for behaviours/agent_top_up.py."""

# pylint: skip-file

import json
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.contracts.gnosis_safe.contract import SafeOperation
from packages.valory.skills.funds_manager.behaviours import (
    GET_AGENT_TOPUP_METHOD_NAME,
    GET_FUNDS_STATUS_METHOD_NAME,
)
from packages.valory.skills.funds_manager.models import (
    AgentTopUp,
    FundRequirements,
    TopUpAction,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.agent_top_up import (
    AGENT_TOPUP_CHECK_INTERVAL_SECONDS,
    AgentTopUpBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.states.agent_top_up import (
    AgentTopUpRound,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    skill_input_hex_to_payload,
)

AGENT = "0x1111111111111111111111111111111111111111"
SAFE = "0x2222222222222222222222222222222222222222"
NATIVE = "0x0000000000000000000000000000000000000000"
SAFE_TX_HASH = "0x" + "ab" * 32
NOW = 1_700_000_000
NO_ACTION = AgentTopUp(
    action=TopUpAction.NONE, move=0, signer_deficit=None, safe_deficit=None
)


def _funds(balance: Optional[int] = 100, deficit: Optional[int] = 300) -> Any:
    """Optimism status with the Agent Signer below its threshold."""
    native = {"topup": 400, "threshold": 200, "is_native": True}
    return FundRequirements.model_validate(
        {
            "optimism": {
                "accounts": {
                    AGENT: {
                        "tokens": {
                            NATIVE: {**native, "balance": balance, "deficit": deficit}
                        }
                    },
                    SAFE: {"tokens": {NATIVE: {**native, "balance": 10**18}}},
                }
            }
        }
    )


def _move(amount: int) -> AgentTopUp:
    """A MOVE decision."""
    return AgentTopUp(
        action=TopUpAction.MOVE, move=amount, signer_deficit=0, safe_deficit=None
    )


def _gen(value: Any) -> Any:
    """Return a generator function that yields once and returns ``value``."""

    def _f(*args: Any, **kwargs: Any) -> Generator:
        yield
        return value

    return _f


def _drive(gen: Any) -> Any:
    """Drive a generator to completion."""
    val = None
    while True:
        try:
            val = gen.send(val)
        except StopIteration as exc:
            return exc.value


class _Harness:
    """An AgentTopUpBehaviour wired to stubs."""

    def __init__(
        self,
        topups: Optional[Dict[str, AgentTopUp]] = None,
        funds: Any = None,
        investing_paused: bool = False,
        last_topup_check: Optional[int] = None,
        x402_eth_deficit: int = 0,
        withdrawal_actions: Optional[List[Dict]] = None,
        kv: Optional[Dict] = None,
        safe_tx_hash: Optional[str] = SAFE_TX_HASH,
    ) -> None:
        self.obj = object.__new__(AgentTopUpBehaviour)
        self.ctx = MagicMock()
        self.obj.__dict__["_context"] = self.ctx
        self.ctx.agent_address = AGENT
        self.get_funds_status = MagicMock(
            return_value=_funds() if funds is None else funds
        )
        self.get_agent_topup = MagicMock(return_value=topups or {})
        self.ctx.shared_state = {
            GET_FUNDS_STATUS_METHOD_NAME: self.get_funds_status,
            GET_AGENT_TOPUP_METHOD_NAME: self.get_agent_topup,
        }
        self.state = MagicMock()
        self.state.last_topup_check = last_topup_check
        self.state.x402_eth_deficit = x402_eth_deficit
        self.params = MagicMock()
        self.params.target_investment_chains = ["optimism"]
        self.params.safe_contract_addresses = {"optimism": SAFE}
        self.coingecko = MagicMock()
        self.coingecko.use_mech_facilitator = False
        self.synced = MagicMock()
        self.synced.withdrawal_actions = withdrawal_actions
        self.obj._get_current_timestamp = lambda: NOW
        self.obj._read_investing_paused = _gen(investing_paused)
        self.obj._read_kv = MagicMock(side_effect=_gen(kv))
        self.contract_interact = MagicMock(side_effect=_gen(safe_tx_hash))
        self.obj.contract_interact = self.contract_interact
        self.payloads: List[Any] = []

        def fake_send(payload: Any) -> Generator:
            self.payloads.append(payload)
            yield

        self.obj.send_a2a_transaction = fake_send
        self.obj.wait_until_round_end = _gen(None)
        self.obj.set_done = MagicMock()

    def run(self) -> Any:
        """Run async_act and return the payload sent."""
        cls = type(self.obj)
        with (
            patch.object(
                cls, "params", new_callable=PropertyMock, return_value=self.params
            ),
            patch.object(
                cls, "shared_state", new_callable=PropertyMock, return_value=self.state
            ),
            patch.object(
                cls, "coingecko", new_callable=PropertyMock, return_value=self.coingecko
            ),
            patch.object(
                cls,
                "synchronized_data",
                new_callable=PropertyMock,
                return_value=self.synced,
            ),
        ):
            _drive(self.obj.async_act())
        self.obj.set_done.assert_called_once()
        assert len(self.payloads) == 1
        return self.payloads[0]


def _assert_no_tx(payload: Any) -> None:
    """The payload carries no tx, so the round ends DONE."""
    assert payload.tx_submitter == AgentTopUpRound.auto_round_id()
    assert payload.tx_hash is None
    assert payload.safe_contract_address is None
    assert payload.chain_id is None


class TestGate:
    """The check runs at start, then at most once per interval."""

    def test_a_fresh_process_checks_and_stamps(self) -> None:
        """No previous check: the status is read and the time recorded."""
        harness = _Harness()
        harness.run()
        harness.get_funds_status.assert_called_once()
        assert harness.state.last_topup_check == NOW

    def test_closed_within_the_interval(self) -> None:
        """A check less than an interval ago skips the status read."""
        last = NOW - AGENT_TOPUP_CHECK_INTERVAL_SECONDS + 1
        harness = _Harness(topups={"optimism": _move(5)}, last_topup_check=last)
        _assert_no_tx(harness.run())
        harness.get_funds_status.assert_not_called()
        assert harness.state.last_topup_check == last

    def test_reopens_after_the_interval(self) -> None:
        """A check an interval ago or more runs again."""
        last = NOW - AGENT_TOPUP_CHECK_INTERVAL_SECONDS
        harness = _Harness(last_topup_check=last)
        harness.run()
        harness.get_funds_status.assert_called_once()
        assert harness.state.last_topup_check == NOW


class TestNoMove:
    """Every path that moves nothing sends a payload without a tx."""

    @pytest.mark.parametrize(
        "missing", [GET_FUNDS_STATUS_METHOD_NAME, GET_AGENT_TOPUP_METHOD_NAME]
    )
    def test_hooks_not_registered(self, missing: str) -> None:
        """Without the funds_manager hooks nothing is read or moved."""
        harness = _Harness(topups={"optimism": _move(5)})
        del harness.ctx.shared_state[missing]
        _assert_no_tx(harness.run())
        harness.ctx.logger.warning.assert_called_once()

    def test_status_read_fails(self) -> None:
        """A failing status read moves nothing."""
        harness = _Harness(topups={"optimism": _move(5)})
        harness.get_funds_status.side_effect = RuntimeError("rpc down")
        _assert_no_tx(harness.run())
        harness.get_agent_topup.assert_not_called()
        assert "rpc down" in harness.ctx.logger.warning.call_args[0][0]

    @pytest.mark.parametrize(
        "topup",
        [
            NO_ACTION,
            AgentTopUp(
                action=TopUpAction.SIGNER_ALERT,
                move=0,
                signer_deficit=300,
                safe_deficit=None,
            ),
            AgentTopUp(
                action=TopUpAction.SAFE_ALERT,
                move=0,
                signer_deficit=0,
                safe_deficit=700,
            ),
        ],
    )
    def test_no_move_decision(self, topup: AgentTopUp) -> None:
        """Alerts and no-action decisions are left to /funds-status."""
        harness = _Harness(topups={"optimism": topup})
        _assert_no_tx(harness.run())
        harness.contract_interact.assert_not_called()

    def test_safe_tx_hash_unavailable(self) -> None:
        """A move whose Safe tx cannot be built sends no tx."""
        harness = _Harness(topups={"optimism": _move(5)}, safe_tx_hash=None)
        _assert_no_tx(harness.run())
        harness.ctx.logger.error.assert_called_once()


class TestMove:
    """A MOVE decision becomes one plain CALL from the Agent Safe."""

    def test_builds_one_plain_call_to_the_agent_signer(self) -> None:
        """The Safe tx sends the amount to the agent EOA with no data."""
        harness = _Harness(topups={"optimism": _move(300)})
        payload = harness.run()

        assert payload.tx_submitter == AgentTopUpRound.auto_round_id()
        assert payload.safe_contract_address == SAFE
        assert payload.chain_id == "optimism"
        tx = skill_input_hex_to_payload(payload.tx_hash)
        assert tx["safe_tx_hash"] == SAFE_TX_HASH[2:]
        assert tx["to_address"] == AGENT
        assert tx["ether_value"] == 300
        assert tx["operation"] == SafeOperation.CALL.value
        assert tx["data"] == b""

        kwargs = harness.contract_interact.call_args.kwargs
        assert kwargs["contract_address"] == SAFE
        assert kwargs["to_address"] == AGENT
        assert kwargs["value"] == 300
        assert kwargs["data"] == b""
        assert kwargs["operation"] == SafeOperation.CALL.value
        assert kwargs["chain_id"] == "optimism"

    def test_first_move_chain_wins(self) -> None:
        """Only one chain is topped up per check."""
        harness = _Harness(
            topups={"base": NO_ACTION, "optimism": _move(1), "mode": _move(2)}
        )
        assert harness.run().chain_id == "optimism"
        assert harness.contract_interact.call_count == 1


class TestRuleInputs:
    """The rule gets the same inputs /funds-status gives it."""

    def test_normal_mode_passes_the_x402_extra_and_no_override(self) -> None:
        """The x402 need beyond the deficit is added on the paid-call chain."""
        harness = _Harness(x402_eth_deficit=1000)
        harness.run()
        funds = harness.get_funds_status.return_value
        harness.get_agent_topup.assert_called_once_with(
            funds, agent_need_extra={"optimism": 600}, agent_need_override=None
        )

    def test_withdrawal_mode_overrides_with_the_gas_left(self) -> None:
        """The need is the gas for the withdrawal actions not executed yet."""
        harness = _Harness(
            investing_paused=True,
            withdrawal_actions=[{"a": 1}, {"a": 2}, {"a": 3}],
            kv={"withdrawal_transaction_hashes": json.dumps(["0xdone"])},
        )
        harness.run()
        kwargs = harness.get_agent_topup.call_args.kwargs
        assert kwargs["agent_need_override"] == {"optimism": 2_400_000_000_000 - 100}

    @pytest.mark.parametrize("kv", [None, {}, {"withdrawal_transaction_hashes": "x"}])
    def test_withdrawal_mode_unreadable_executed_count(self, kv: Any) -> None:
        """An unreadable executed count counts every action as left."""
        harness = _Harness(investing_paused=True, withdrawal_actions=[{"a": 1}], kv=kv)
        harness.run()
        kwargs = harness.get_agent_topup.call_args.kwargs
        assert kwargs["agent_need_override"] == {"optimism": 1_200_000_000_000 - 100}

    @pytest.mark.parametrize(
        "funds, actions",
        [
            (None, []),
            (None, None),
            (_funds(balance=1000, deficit=0), [{"a": 1}]),
            (_funds(balance=None, deficit=None), [{"a": 1}]),
            (FundRequirements.model_validate({}), [{"a": 1}]),
        ],
    )
    def test_withdrawal_mode_nothing_due(self, funds: Any, actions: Any) -> None:
        """No action left, no Agent Signer deficit or no status: nothing is due."""
        harness = _Harness(
            funds=funds, investing_paused=True, withdrawal_actions=actions
        )
        harness.run()
        kwargs = harness.get_agent_topup.call_args.kwargs
        assert kwargs["agent_need_override"] == {"optimism": 0}
        harness.obj._read_kv.assert_not_called()

    def test_withdrawal_mode_counts_the_x402_need_as_a_deficit(self) -> None:
        """An x402 need alone opens the withdrawal gas, as /funds-status does."""
        harness = _Harness(
            funds=_funds(balance=1000, deficit=0),
            investing_paused=True,
            x402_eth_deficit=5000,
            withdrawal_actions=[{"a": 1}],
            kv={"withdrawal_transaction_hashes": "[]"},
        )
        harness.run()
        kwargs = harness.get_agent_topup.call_args.kwargs
        assert kwargs["agent_need_override"] == {"optimism": 1_200_000_000_000 - 1000}

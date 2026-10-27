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

"""Tests that DecisionMakingBehaviour hands Aero actions their protocol and gauge."""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    DexType,
    PositionStatus,
)
from packages.valory.skills.liquidity_trader_abci.tests.behaviours.test_decision_making import (
    _exhaust,
    _make_behaviour,
    _make_gen_method,
)

GAUGE = "0xGauge"
TX_RESULT = {"tx_hash": b"data", "contract_address": "0x" + "cc" * 20}


def _recording(result: Any, calls: List[Dict[str, Any]]) -> Any:
    """Return a generator method recording its kwargs and returning result."""

    def method(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        yield
        return result

    return method


def _aero_action(is_cl_pool: bool = False, **extra: Any) -> Dict[str, Any]:
    return {
        "dex_type": DexType.AERODROME.value,
        "chain": "optimism",
        "pool_address": "0xPOOL",
        "is_cl_pool": is_cl_pool,
        "protocol": "aero",
        "gauge_address": GAUGE,
        **extra,
    }


class TestEnterAndExit:
    """Entry and exit kwargs."""

    def test_enter_passes_protocol_and_pool_factory(self) -> None:
        """The Aero entry gets the protocol and the pool's creating factory."""
        b = _make_behaviour()
        calls: List[Dict[str, Any]] = []
        pool = MagicMock()
        pool.enter = _recording(("0xtx", "0xMetarouter"), calls)
        b.pools = {DexType.AERODROME.value: pool}
        b._get_token_balances_and_calculate_amounts = _make_gen_method(
            ([1000, 2000], 1000, 2000)
        )
        b.get_approval_tx_hash = _make_gen_method(
            {"operation": 0, "to": "0xT", "value": 0, "data": b"x"}
        )
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        action = _aero_action(
            token0="0xT0",
            token1="0xT1",
            relative_funds_percentage=1.0,
            pool_factory="0xFactory",
        )

        result = _exhaust(b.get_enter_pool_tx_hash([], action))

        assert result[0] is not None
        assert calls[0]["protocol"] == "aero"
        assert calls[0]["pool_factory"] == "0xFactory"

    def test_exit_passes_protocol(self) -> None:
        """The Aero exit gets the protocol of the position."""
        b = _make_behaviour()
        calls: List[Dict[str, Any]] = []
        pool = MagicMock()
        pool.exit = _recording((b"tx", "0x" + "cc" * 20, True), calls)
        b.pools = {DexType.AERODROME.value: pool}
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)

        _exhaust(b.get_exit_pool_tx_hash(_aero_action(assets=["0xA", "0xB"])))

        assert calls[0]["protocol"] == "aero"

    def test_entered_position_stores_protocol_and_gauge(self) -> None:
        """The stored position keeps what its later unstake and exit need."""
        b = _make_behaviour()
        b._get_data_from_velodrome_mint_event = _make_gen_method((500, 600, 12345))
        b._accumulate_transaction_costs = _make_gen_method(None)
        b._rename_entry_costs_key = _make_gen_method(None)
        b._calculate_and_store_tip_data = _make_gen_method(None)
        b.store_current_positions = MagicMock()

        _exhaust(b._post_execute_enter_pool([_aero_action()], 0))

        (position,) = b.current_positions
        assert position["protocol"] == "aero"
        assert position["gauge_address"] == GAUGE


class TestStakingCalls:
    """Stake, unstake and claim go to the stored gauge with the Aero protocol."""

    def _behaviour(self, **pool_methods: Any) -> Any:
        b = _make_behaviour()
        pool = MagicMock()
        for name, method in pool_methods.items():
            setattr(pool, name, method)
        b.pools = {"velodrome": pool}
        b._get_token_balance = _make_gen_method(1000)
        b.contract_interact = _make_gen_method("0x" + "ab" * 32)
        return b

    def test_v2_stake_uses_stored_gauge(self) -> None:
        """The v2 stake is built on the action's gauge."""
        calls: List[Dict[str, Any]] = []
        b = self._behaviour(stake_lp_tokens=_recording(TX_RESULT, calls))
        _exhaust(b.get_stake_lp_tokens_tx_hash(_aero_action()))
        assert calls[0]["gauge_address"] == GAUGE

    def test_cl_stake_passes_protocol(self) -> None:
        """The CL stake approves the NFT on the Aero position manager."""
        calls: List[Dict[str, Any]] = []
        b = self._behaviour(stake_cl_lp_tokens=_recording(TX_RESULT, calls))
        _exhaust(
            b.get_stake_lp_tokens_tx_hash(_aero_action(is_cl_pool=True, token_ids=[1]))
        )
        assert calls[0]["protocol"] == "aero"

    def test_cl_stake_resolves_gauge_from_position(self) -> None:
        """Without a gauge on the action the open position's gauge is used."""
        gauge_calls: List[Dict[str, Any]] = []
        b = self._behaviour(
            get_gauge_address=_recording(GAUGE, gauge_calls),
            stake_cl_lp_tokens=_make_gen_method(TX_RESULT),
        )
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
                "gauge_address": GAUGE,
            }
        ]
        _exhaust(
            b.get_stake_lp_tokens_tx_hash(
                _aero_action(is_cl_pool=True, gauge_address=None)
            )
        )
        assert gauge_calls[0]["gauge_address"] == GAUGE

    @pytest.mark.parametrize(
        "builder, pool_method",
        [
            ("get_unstake_lp_tokens_tx_hash", "unstake_cl_lp_tokens"),
            ("get_claim_staking_rewards_tx_hash", "claim_cl_rewards"),
        ],
    )
    def test_cl_unstake_and_claim_resolve_gauge_from_position(
        self, builder: str, pool_method: str
    ) -> None:
        """Without a gauge on the action the open position's stored gauge is used."""
        gauge_calls: List[Dict[str, Any]] = []
        b = self._behaviour(
            get_gauge_address=_recording(GAUGE, gauge_calls),
            **{pool_method: _make_gen_method(TX_RESULT)},
        )
        b._filter_staked_token_ids = _make_gen_method(([1], False))
        b.current_positions = [
            {
                "pool_address": "0xPOOL",
                "chain": "optimism",
                "status": PositionStatus.OPEN.value,
                "positions": [{"token_id": 1}],
                "gauge_address": GAUGE,
            }
        ]
        _exhaust(getattr(b, builder)(_aero_action(is_cl_pool=True, gauge_address=None)))
        assert gauge_calls[0]["gauge_address"] == GAUGE

    def test_v2_unstake_uses_stored_gauge_and_protocol(self) -> None:
        """The v2 unstake reads the balance on, and withdraws from, the stored gauge."""
        balance_calls: List[Dict[str, Any]] = []
        unstake_calls: List[Dict[str, Any]] = []
        b = self._behaviour(
            get_staked_balance=_recording(100, balance_calls),
            unstake_lp_tokens=_recording(TX_RESULT, unstake_calls),
        )
        _exhaust(b.get_unstake_lp_tokens_tx_hash(_aero_action()))
        assert balance_calls[0]["gauge_address"] == GAUGE
        assert unstake_calls[0]["gauge_address"] == GAUGE
        assert unstake_calls[0]["protocol"] == "aero"

    def test_cl_unstake_passes_protocol(self) -> None:
        """The CL unstake runs the Aero penalty check."""
        calls: List[Dict[str, Any]] = []
        b = self._behaviour(unstake_cl_lp_tokens=_recording(TX_RESULT, calls))
        _exhaust(
            b.get_unstake_lp_tokens_tx_hash(
                _aero_action(is_cl_pool=True, token_ids=[1])
            )
        )
        assert calls[0]["protocol"] == "aero"

    def test_v2_claim_uses_stored_gauge_and_protocol(self) -> None:
        """The v2 claim uses claimEmissions on the stored gauge."""
        calls: List[Dict[str, Any]] = []
        b = self._behaviour(claim_rewards=_recording(TX_RESULT, calls))
        _exhaust(b.get_claim_staking_rewards_tx_hash(_aero_action()))
        assert calls[0]["gauge_address"] == GAUGE
        assert calls[0]["protocol"] == "aero"

    def test_cl_claim_passes_protocol(self) -> None:
        """The CL claim uses one claimEmissions for the staked tokens."""
        calls: List[Dict[str, Any]] = []
        b = self._behaviour(claim_cl_rewards=_recording(TX_RESULT, calls))
        b._filter_staked_token_ids = _make_gen_method(([1], False))
        _exhaust(
            b.get_claim_staking_rewards_tx_hash(
                _aero_action(is_cl_pool=True, token_ids=[1])
            )
        )
        assert calls[0]["protocol"] == "aero"

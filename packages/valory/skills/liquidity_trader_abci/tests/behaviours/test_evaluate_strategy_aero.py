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

"""Tests for the legacy-pool exit and the Aero paths of EvaluateStrategyBehaviour."""

import json
import time
from unittest.mock import MagicMock

from packages.valory.skills.liquidity_trader_abci.behaviours.base import Action
from packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy import (
    STRATEGY_BACKOFF_KV_KEY,
)
from packages.valory.skills.liquidity_trader_abci.tests.behaviours.test_evaluate_strategy import (
    _drive,
    _gen_return,
    _mk,
)

SAFE = "0x" + "aa" * 20
GAUGE = "0xGauge"


def _legacy_v2(**extra):
    """Return an open legacy Velodrome v2 position."""
    return {
        "status": "open",
        "dex_type": "velodrome",
        "chain": "optimism",
        "pool_address": "0xLegacyV2",
        "token0": "0xA",
        "token1": "0xB",
        "token0_symbol": "A",
        "token1_symbol": "B",
        "is_cl_pool": False,
        "is_stable": True,
        **extra,
    }


def _legacy_cl(**extra):
    """Return an open legacy Velodrome CL position."""
    return {
        "status": "open",
        "dex_type": "velodrome",
        "chain": "optimism",
        "pool_address": "0xLegacyCL",
        "token0": "0xA",
        "token1": "0xB",
        "is_cl_pool": True,
        "staked": True,
        "positions": [{"token_id": 7, "liquidity": 10}],
        **extra,
    }


def _pool(pending, staked: int) -> MagicMock:
    """Return a velodrome pool behaviour reporting the given gauge state."""
    pool = MagicMock()
    pool.read_pending_rewards = MagicMock(side_effect=_gen_return(pending))
    pool.get_staked_balance = MagicMock(side_effect=_gen_return(staked))
    return pool


class TestPrepareLegacyVelodromeExitActions:
    """Tests for _prepare_legacy_velodrome_exit_actions."""

    def test_no_legacy_positions(self):
        """Aero, closed and non-Velodrome positions are not legacy exits."""
        b = _mk(
            current_positions=[
                _legacy_v2(protocol="aero"),
                _legacy_v2(status="closed"),
                {"status": "open", "dex_type": "balancerPool"},
            ]
        )
        assert _drive(b._prepare_legacy_velodrome_exit_actions()) == []

    def test_v2_claims_then_unstakes_then_exits(self):
        """A staked legacy v2 position claims, unstakes and exits, in that order."""
        b = _mk(current_positions=[_legacy_v2()], pools={"velodrome": _pool(5, 100)})
        actions = _drive(b._prepare_legacy_velodrome_exit_actions(), sends=[None] * 10)

        assert [a["action"] for a in actions] == [
            Action.CLAIM_STAKING_REWARDS.value,
            Action.UNSTAKE_LP_TOKENS.value,
            Action.EXIT_POOL.value,
        ]
        exit_action = actions[-1]
        assert exit_action["pool_address"] == "0xLegacyV2"
        assert exit_action["assets"] == ["0xA", "0xB"]
        assert "protocol" not in exit_action
        assert exit_action["description"] == "Exit A/B pool for Aero migration"
        pool = b.pools["velodrome"]
        pool.read_pending_rewards.assert_called_once_with(
            b, lp_token="0xLegacyV2", user_address=SAFE, chain="optimism"
        )

    def test_v2_unstaked_without_rewards_only_exits(self):
        """Nothing staked and nothing pending leaves only the exit."""
        b = _mk(current_positions=[_legacy_v2()], pools={"velodrome": _pool(0, 0)})
        actions = _drive(b._prepare_legacy_velodrome_exit_actions(), sends=[None] * 10)
        assert [a["action"] for a in actions] == [Action.EXIT_POOL.value]

    def test_v2_unreadable_rewards_waits(self):
        """When pending rewards cannot be read the position is left for the next cycle."""
        b = _mk(
            current_positions=[_legacy_v2(), _legacy_cl()],
            pools={"velodrome": _pool(None, 100)},
        )
        b._build_unstake_lp_tokens_action_verified = _gen_return(None)
        actions = _drive(b._prepare_legacy_velodrome_exit_actions(), sends=[None] * 10)
        assert [a["pool_address"] for a in actions] == ["0xLegacyCL"]
        b.pools["velodrome"].get_staked_balance.assert_not_called()

    def test_v2_without_safe_still_exits(self):
        """When the claim and unstake cannot be built the exit is still sent."""
        b = _mk(
            current_positions=[_legacy_v2()],
            pools={"velodrome": _pool(5, 100)},
            params={"safe_contract_addresses": {}},
        )
        actions = _drive(b._prepare_legacy_velodrome_exit_actions(), sends=[None] * 10)
        assert [a["action"] for a in actions] == [Action.EXIT_POOL.value]

    def test_cl_unstakes_without_separate_claim(self):
        """A legacy CL gauge's withdraw claims, so only unstake and exit are built."""
        b = _mk(current_positions=[_legacy_cl()], pools={"velodrome": _pool(5, 1)})
        b._build_unstake_lp_tokens_action_verified = _gen_return(
            {"action": Action.UNSTAKE_LP_TOKENS.value, "token_ids": [7]}
        )
        actions = _drive(b._prepare_legacy_velodrome_exit_actions(), sends=[None] * 10)

        assert [a["action"] for a in actions] == [
            Action.UNSTAKE_LP_TOKENS.value,
            Action.EXIT_POOL.value,
        ]
        assert actions[-1]["token_ids"] == [7]
        b.pools["velodrome"].read_pending_rewards.assert_not_called()

    def test_cl_not_staked_only_exits(self):
        """A CL position no longer in its gauge is only exited."""
        b = _mk(current_positions=[_legacy_cl()], pools={"velodrome": _pool(0, 0)})
        b._build_unstake_lp_tokens_action_verified = _gen_return(None)
        actions = _drive(b._prepare_legacy_velodrome_exit_actions(), sends=[None] * 10)
        assert [a["action"] for a in actions] == [Action.EXIT_POOL.value]


class TestAsyncActLegacyExit:
    """The legacy exit runs before every other gate and resets the backoff."""

    def test_legacy_exit_skips_gates_and_resets_backoff(self):
        """Legacy exit actions are sent without TiP, funds or backoff checks."""
        b = _mk()
        b.shared_state.strategy_state_restored = False
        b._read_investing_paused = _gen_return(False)
        exits = [{"action": Action.EXIT_POOL.value}]
        b._prepare_legacy_velodrome_exit_actions = _gen_return(exits)
        b._apply_tip_filters_to_exit_decisions = MagicMock()
        b.check_funds = MagicMock()
        b._read_kv = _gen_return(
            {
                STRATEGY_BACKOFF_KV_KEY: json.dumps(
                    {
                        "consecutive_no_action_count": 3,
                        "last_strategy_evaluation_time": time.time() - 60,
                    }
                )
            }
        )
        writes = {}

        def _write_kv(data):
            writes.update(data)
            yield
            return True

        b._write_kv = _write_kv
        sent = []

        def _send(actions=None):
            sent.append(actions)
            yield

        b.send_actions = _send
        _drive(b.async_act(), sends=[None] * 20)

        assert sent == [exits]
        b._apply_tip_filters_to_exit_decisions.assert_not_called()
        b.check_funds.assert_not_called()
        assert b.shared_state.consecutive_no_action_count == 0
        assert (
            json.loads(writes[STRATEGY_BACKOFF_KV_KEY])["consecutive_no_action_count"]
            == 0
        )


class TestAeroStakingActions:
    """Aero opportunities and positions carry their gauge and protocol."""

    def _opportunity(self, **extra):
        return {
            "chain": "base",
            "pool_address": "0xAeroPool",
            "dex_type": "aerodrome",
            "is_cl_pool": False,
            "protocol": "aero",
            "gauge_address": GAUGE,
            **extra,
        }

    def test_stake_action_carries_gauge_and_protocol(self):
        """The stake action keeps the opportunity's gauge for the Aero stake."""
        b = _mk()
        action = b._build_stake_lp_tokens_action(self._opportunity())
        assert action["gauge_address"] == GAUGE
        assert action["protocol"] == "aero"

    def test_claim_action_carries_protocol(self):
        """The claim action keeps the position's protocol for claimEmissions."""
        b = _mk(params={"safe_contract_addresses": {"base": SAFE}})
        action = b._build_claim_staking_rewards_action(self._opportunity())
        assert action["protocol"] == "aero"
        assert action["gauge_address"] == GAUGE

    def test_staking_gated_on_activated_gauge(self):
        """Aero staking depends on the gauge, not on a legacy voter being configured."""
        b = _mk(params={"velodrome_voter_contract_addresses": {}})
        assert b._should_add_staking_actions(self._opportunity()) is True
        assert (
            b._should_add_staking_actions(self._opportunity(gauge_address=None))
            is False
        )

    def test_gauge_lookup_uses_stored_gauge(self):
        """A stored gauge is used without asking the legacy voter."""
        b = _mk()
        assert _drive(b._get_gauge_address_for_position(self._opportunity())) == GAUGE


class TestCachedClOpportunityOnAeroChains:
    """A cached CL opportunity is reused only when it is an Aero one."""

    def _cached(self, **enter_extra):
        return {
            "enter_pool_action": {
                "action": "EnterPool",
                "chain": "base",
                "pool_address": "0xCL",
                **enter_extra,
            },
            "token0": "0xA",
            "token1": "0xB",
            "pool_address": "0xCL",
            "position_requirements": [],
        }

    def test_legacy_cache_not_reused(self):
        """A pre-Aero cache on an Aero chain is not turned back into actions."""
        b = _mk(
            params={
                "safe_contract_addresses": {"base": SAFE},
                "aero_metarouter_contract_addresses": {"base": "0xMetarouter"},
            }
        )
        result = _drive(
            b._reconstruct_actions_from_cached_cl_pool(self._cached(), "base")
        )
        assert result is None

    def test_aero_cache_restakes_on_its_gauge(self):
        """An Aero cache rebuilds the stake action on the cached gauge."""
        b = _mk(
            params={
                "safe_contract_addresses": {"base": SAFE},
                "aero_metarouter_contract_addresses": {"base": "0xMetarouter"},
            }
        )
        b._get_token_balance = _gen_return(1000)
        b._initialize_entry_costs_for_new_position = _gen_return(None)
        cached = self._cached(protocol="aero", gauge_address=GAUGE)
        actions = _drive(
            b._reconstruct_actions_from_cached_cl_pool(cached, "base"),
            sends=[None] * 15,
        )
        stake = actions[-1]
        assert stake["action"] == Action.STAKE_LP_TOKENS.value
        assert stake["gauge_address"] == GAUGE
        assert stake["protocol"] == "aero"

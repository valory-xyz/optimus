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

"""Tests for the Aero paths of the pools/velodrome.py module."""

# pylint: skip-file

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.liquidity_trader_abci.pools.velodrome import (
    AERO_PROTOCOL,
    ZERO_ADDRESS,
    VelodromePoolBehaviour,
    is_aero_position,
)
from packages.valory.skills.liquidity_trader_abci.tests.pools.test_velodrome import (
    _mock_gen,
    exhaust_generator,
    make_behaviour,
)

CHAIN = "base"
SAFE = "0xSafe"
POOL = "0xPool"
GAUGE = "0xGauge"
METAROUTER = "0xMetarouter"
FACTORY = "0xFactory"


class _Recorder:
    """A fake contract_interact answering by callable and recording each call."""

    def __init__(self, answers: Dict[str, Any]) -> None:
        """Store the answers, keyed by contract_callable."""
        self.answers = answers
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        """Answer one contract call."""
        self.calls.append(kwargs)
        yield
        answer = self.answers.get(kwargs["contract_callable"])
        return answer(kwargs) if callable(answer) else answer

    def called(self, contract_callable: str) -> List[Dict[str, Any]]:
        """Return the calls made to contract_callable."""
        return [c for c in self.calls if c["contract_callable"] == contract_callable]


def _aero_behaviour(answers: Dict[str, Any]) -> VelodromePoolBehaviour:
    """Return a behaviour with Aero params and a recording contract_interact."""
    b = make_behaviour()
    params = b.context.params
    params.aero_metarouter_contract_addresses = {CHAIN: METAROUTER}
    params.aero_non_fungible_position_manager_contract_addresses = {CHAIN: "0xAeroNPM"}
    params.velodrome_non_fungible_position_manager_contract_addresses = {
        CHAIN: "0xLegacyNPM"
    }
    params.multisend_contract_addresses = {CHAIN: "0xMulti"}
    params.slippage_tolerance = 0.01
    timestamp = MagicMock()
    timestamp.timestamp.return_value = 1_000_000.0
    b.context.state.round_sequence.last_round_transition_timestamp = timestamp
    b.contract_interact = _Recorder(answers)
    return b


def test_is_aero_position() -> None:
    """Only a position written by an Aero entry is Aero."""
    assert is_aero_position({"protocol": AERO_PROTOCOL})
    assert not is_aero_position({"dex_type": "aerodrome"})


class TestPositionManagerAddress:
    """Tests for _get_position_manager_address."""

    def test_aero_and_legacy(self) -> None:
        """Aero positions use Aero's NPM; everything else keeps the legacy one."""
        b = _aero_behaviour({})
        assert b._get_position_manager_address(CHAIN, AERO_PROTOCOL) == "0xAeroNPM"
        assert b._get_position_manager_address(CHAIN) == "0xLegacyNPM"


class TestQuoteAeroAddLiquidity:
    """Tests for _quote_aero_add_liquidity."""

    @pytest.mark.parametrize(
        "amounts, reserves, expected",
        [
            ((100, 300), (0, 10), (100, 300)),
            ((100, 300), (10, 20), (100, 200)),
            ((100, 100), (10, 20), (50, 100)),
        ],
    )
    def test_trims_larger_side(self, amounts, reserves, expected) -> None:
        """The side above the pool ratio is trimmed; an empty pool takes both."""
        assert (
            VelodromePoolBehaviour._quote_aero_add_liquidity(list(amounts), reserves)
            == expected
        )


class TestEnterAeroV2Pool:
    """Tests for the Aero stable/volatile entry."""

    def test_enter_routes_aero_v2_to_metarouter(self) -> None:
        """enter() sends a non-CL Aero pool through the Metarouter with reserve-based minimums."""
        b = _aero_behaviour(
            {"get_reserves": [1000, 2000], "get_add_liquidity_tx": b"\x22"}
        )
        result = exhaust_generator(
            b.enter(
                pool_address=POOL,
                safe_address=SAFE,
                assets=["0xA", "0xB"],
                chain=CHAIN,
                max_amounts_in=[100, 300],
                is_cl_pool=False,
                protocol=AERO_PROTOCOL,
                pool_factory=FACTORY,
            )
        )

        assert result == (b"\x22", METAROUTER)
        (call,) = b.contract_interact.called("get_add_liquidity_tx")
        assert call["contract_address"] == METAROUTER
        assert call["factory"] == FACTORY
        assert (call["amount_a"], call["amount_b"]) == (100, 300)
        # The router commits 100/200, so the minimums are 1% below that.
        assert (call["amount_a_min"], call["amount_b_min"]) == (99, 198)
        assert call["recipient"] == SAFE
        assert call["deadline"] == 1_000_000 + 20 * 60
        # Aero pools revert on stable(): the legacy is_stable lookup is not made.
        assert not b.contract_interact.called("get_stable")

    def test_missing_factory(self) -> None:
        """Without the creating factory the Metarouter cannot be called."""
        b = _aero_behaviour({})
        result = exhaust_generator(
            b._enter_aero_v2_pool(POOL, SAFE, ["0xA", "0xB"], CHAIN, [1, 1], None)
        )
        assert result == (None, None)
        assert not b.contract_interact.calls

    def test_missing_reserves(self) -> None:
        """Without reserves no minimums can be computed."""
        b = _aero_behaviour({"get_reserves": None})
        result = exhaust_generator(
            b._enter_aero_v2_pool(POOL, SAFE, ["0xA", "0xB"], CHAIN, [1, 1], FACTORY)
        )
        assert result == (None, None)


class TestEnterAeroClPool:
    """Tests for the Aero CL entry."""

    def test_enter_mints_aero_cl_through_aero_npm(self) -> None:
        """enter() mints an Aero CL position on Aero's NPM, not the legacy one."""
        b = _aero_behaviour({"mint": "0xMintTx"})
        b._get_cached_price_at_selection = _mock_gen(1.0)
        b._check_price_movement = _mock_gen(False)
        b._get_sqrt_price_x96 = _mock_gen(2**96)
        b._get_token_decimals = _mock_gen(18)
        band = {
            "tick_lower": -100,
            "tick_upper": 100,
            "allocation": 1.0,
            "token0_ratio": 0.5,
            "token1_ratio": 0.5,
        }
        result = exhaust_generator(
            b.enter(
                pool_address=POOL,
                safe_address=SAFE,
                assets=["0xA", "0xB"],
                chain=CHAIN,
                max_amounts_in=[1000, 1000],
                is_cl_pool=True,
                tick_ranges=[band],
                tick_spacing=60,
                protocol=AERO_PROTOCOL,
            )
        )

        assert result == (["0xMintTx"], "0xAeroNPM")
        (call,) = b.contract_interact.called("mint")
        assert call["contract_address"] == "0xAeroNPM"
        assert call["recipient"] == SAFE


class TestExitAeroClPool:
    """Tests for the Aero CL exit."""

    def test_exit_burns_aero_cl_through_aero_npm(self) -> None:
        """exit() reads and burns an Aero CL position on Aero's NPM, not the legacy one."""
        position = {
            "nonce": 0,
            "operator": "0x0",
            "token0": "0xA",
            "token1": "0xB",
            "tickSpacing": 60,
            "tickLower": -100,
            "tickUpper": 100,
            "liquidity": 100,
        }
        b = _aero_behaviour(
            {
                "ownerOf": SAFE,
                "get_position": position,
                "decrease_liquidity": "0xDecreaseTx",
                "collect": "0xCollectTx",
                "get_tx_data": "0xabcd",
            }
        )
        b._read_kv = _mock_gen({"withdrawal_status": "WITHDRAWING"})
        result = exhaust_generator(
            b.exit(
                pool_address=POOL,
                safe_address=SAFE,
                chain=CHAIN,
                is_cl_pool=True,
                protocol=AERO_PROTOCOL,
                token_ids=[1],
                liquidities=[100],
            )
        )

        assert result == (bytes.fromhex("abcd"), "0xMulti", True)
        npm_calls = [
            c
            for c in b.contract_interact.calls
            if c["contract_callable"] != "get_tx_data"
        ]
        assert [c["contract_callable"] for c in npm_calls] == [
            "ownerOf",
            "get_position",
            "get_position",
            "decrease_liquidity",
            "collect",
        ]
        assert {c["contract_address"] for c in npm_calls} == {"0xAeroNPM"}
        (multisend,) = b.contract_interact.called("get_tx_data")
        assert [tx["to"] for tx in multisend["multi_send_txs"]] == [
            "0xAeroNPM",
            "0xAeroNPM",
        ]


class TestExitAeroV2Pool:
    """Tests for the Aero stable/volatile exit."""

    def _answers(self, **overrides: Any) -> Dict[str, Any]:
        answers = {
            "get_balance": 500,
            "get_reserves": [1000, 2000],
            "get_total_supply": 1000,
            "build_approval_tx": b"approve",
            "get_remove_liquidity_tx": b"remove",
            "get_tx_data": "0xabcd",
        }
        answers.update(overrides)
        return answers

    def test_exit_routes_aero_v2_to_metarouter(self) -> None:
        """exit() burns the Safe's LP balance through the Metarouter in one multisend."""
        b = _aero_behaviour(self._answers())
        result = exhaust_generator(
            b.exit(
                pool_address=POOL,
                safe_address=SAFE,
                chain=CHAIN,
                is_cl_pool=False,
                protocol=AERO_PROTOCOL,
                assets=["0xA", "0xB"],
            )
        )

        assert result == (bytes.fromhex("abcd"), "0xMulti", True)
        (remove,) = b.contract_interact.called("get_remove_liquidity_tx")
        assert remove["contract_address"] == METAROUTER
        assert remove["liquidity"] == 500
        # 500/1000 of the reserves, less 1% slippage.
        assert (remove["amount0_min"], remove["amount1_min"]) == (495, 990)
        (approve,) = b.contract_interact.called("build_approval_tx")
        assert approve["spender"] == METAROUTER
        (multisend,) = b.contract_interact.called("get_tx_data")
        assert [tx["to"] for tx in multisend["multi_send_txs"]] == [POOL, METAROUTER]

    def test_uses_given_liquidity(self) -> None:
        """A known liquidity is burned without reading the Safe's balance."""
        b = _aero_behaviour(self._answers())
        exhaust_generator(b._exit_aero_v2_pool(POOL, SAFE, CHAIN, 10))
        assert not b.contract_interact.called("get_balance")

    @pytest.mark.parametrize(
        "overrides",
        [
            {"get_balance": 0},
            {"get_total_supply": None},
            {"get_reserves": None},
            {"get_remove_liquidity_tx": None},
            {"build_approval_tx": None},
            {"get_tx_data": None},
        ],
    )
    def test_failures_return_nothing(self, overrides: Dict[str, Any]) -> None:
        """Any missing read or tx stops the exit."""
        b = _aero_behaviour(self._answers(**overrides))
        result = exhaust_generator(b._exit_aero_v2_pool(POOL, SAFE, CHAIN, None))
        assert result == (None, None, None)

    def test_no_metarouter(self) -> None:
        """No Metarouter configured for the chain stops the exit."""
        b = _aero_behaviour(self._answers())
        b.context.params.aero_metarouter_contract_addresses = {}
        result = exhaust_generator(b._exit_aero_v2_pool(POOL, SAFE, CHAIN, None))
        assert result == (None, None, None)


class TestStoredGauge:
    """Gauge lookups use the gauge an Aero position stores."""

    def test_get_gauge_address_returns_stored_gauge(self) -> None:
        """A stored gauge is returned without calling the legacy voter."""
        b = _aero_behaviour({})
        result = exhaust_generator(
            b.get_gauge_address(POOL, chain=CHAIN, gauge_address=GAUGE)
        )
        assert result == GAUGE
        assert not b.contract_interact.calls


class TestAeroUnstake:
    """Early-unstake penalty checks before leaving an Aero gauge."""

    def test_v2_penalty_active_keeps_stake(self) -> None:
        """An active penalty on a v2 gauge stops the withdraw."""
        b = _aero_behaviour({"balance_of": 100, "is_penalty_active": True})
        result = exhaust_generator(
            b.unstake_lp_tokens(
                POOL,
                100,
                chain=CHAIN,
                safe_address=SAFE,
                gauge_address=GAUGE,
                protocol=AERO_PROTOCOL,
            )
        )
        assert "penalty" in result["error"]
        assert not b.contract_interact.called("withdraw")

    def test_v2_penalty_inactive_withdraws(self) -> None:
        """Without a penalty the v2 withdraw is built on the stored gauge."""
        b = _aero_behaviour(
            {"balance_of": 100, "is_penalty_active": False, "withdraw": b"w"}
        )
        result = exhaust_generator(
            b.unstake_lp_tokens(
                POOL,
                100,
                chain=CHAIN,
                safe_address=SAFE,
                gauge_address=GAUGE,
                protocol=AERO_PROTOCOL,
            )
        )
        assert result["contract_address"] == GAUGE
        (penalty,) = b.contract_interact.called("is_penalty_active")
        assert penalty["account"] == SAFE

    def test_v2_unreadable_penalty_withdraws(self) -> None:
        """An unreadable penalty never holds the stake, as for CL gauges."""
        b = _aero_behaviour(
            {"balance_of": 100, "is_penalty_active": None, "withdraw": b"w"}
        )
        result = exhaust_generator(
            b.unstake_lp_tokens(
                POOL,
                100,
                chain=CHAIN,
                safe_address=SAFE,
                gauge_address=GAUGE,
                protocol=AERO_PROTOCOL,
            )
        )
        assert result["tx_hash"] == b"w"

    def test_cl_skips_tokens_inside_penalty_window(self) -> None:
        """Only CL tokens past their minimum stake are withdrawn."""
        b = _aero_behaviour({"withdraw": b"w", "get_tx_data": "0xab"})
        b._is_cl_unstake_penalty_active = MagicMock(
            side_effect=lambda gauge, token_id, chain: _gen(token_id == 1)
        )
        result = exhaust_generator(
            b.unstake_cl_lp_tokens(
                [1, 2],
                GAUGE,
                chain=CHAIN,
                safe_address=SAFE,
                protocol=AERO_PROTOCOL,
            )
        )
        assert result["unstaked_positions"] == [{"token_id": 2}]

    def test_cl_all_tokens_penalised(self) -> None:
        """When every token is inside its window nothing is withdrawn."""
        b = _aero_behaviour({})
        b._is_cl_unstake_penalty_active = _mock_gen(True)
        result = exhaust_generator(
            b.unstake_cl_lp_tokens(
                [1], GAUGE, chain=CHAIN, safe_address=SAFE, protocol=AERO_PROTOCOL
            )
        )
        assert "early-unstake" in result["error"]


def _gen(value: Any) -> Any:
    """Return a generator that yields once and returns value."""
    yield
    return value


class TestClUnstakePenaltyActive:
    """Tests for _is_cl_unstake_penalty_active."""

    def _behaviour(self, penalty_config: Any, block: Any) -> VelodromePoolBehaviour:
        b = _aero_behaviour(
            {
                "deposit_block": 100,
                "gauge_factory": "0xCLGaugeFactory",
                "effective_penalty_config": penalty_config,
            }
        )
        b._get_latest_block_number = _mock_gen(block)
        return b

    @pytest.mark.parametrize(
        "config, block, expected",
        [
            ({"min_stake_blocks": 7, "penalty_rate": 1_000_000}, 106, True),
            ({"min_stake_blocks": 7, "penalty_rate": 1_000_000}, 107, False),
            ({"min_stake_blocks": 7, "penalty_rate": 0}, 101, False),
            ({"min_stake_blocks": 0, "penalty_rate": 1_000_000}, 100, False),
            (None, 101, False),
            ({"min_stake_blocks": 7, "penalty_rate": 1}, None, False),
        ],
    )
    def test_penalty_window(self, config, block, expected) -> None:
        """The penalty applies only inside depositBlock + minStakeBlocks at a positive rate."""
        b = self._behaviour(config, block)
        result = exhaust_generator(b._is_cl_unstake_penalty_active(GAUGE, 5, CHAIN))
        assert result is expected

    def test_reads_factory_from_gauge(self) -> None:
        """The penalty config is read from the factory the gauge reports."""
        b = self._behaviour({"min_stake_blocks": 7, "penalty_rate": 1}, 200)
        exhaust_generator(b._is_cl_unstake_penalty_active(GAUGE, 5, CHAIN))
        (call,) = b.contract_interact.called("effective_penalty_config")
        assert call["contract_address"] == "0xCLGaugeFactory"
        assert call["gauge"] == GAUGE

    def test_unreadable_deposit_block(self) -> None:
        """A failed gauge read is treated as no penalty."""
        b = _aero_behaviour({"deposit_block": None, "gauge_factory": "0xF"})
        result = exhaust_generator(b._is_cl_unstake_penalty_active(GAUGE, 5, CHAIN))
        assert result is False


class TestLatestBlockNumber:
    """Tests for _get_latest_block_number."""

    def test_returns_block_number(self) -> None:
        """The latest block's number is returned."""
        b = _aero_behaviour({})
        response = MagicMock()
        response.performative = LedgerApiMessage.Performative.STATE
        response.state.body = {"number": 42}
        b.get_ledger_api_response = _mock_gen(response)
        assert exhaust_generator(b._get_latest_block_number(CHAIN)) == 42

    def test_failure_returns_none(self) -> None:
        """A failed ledger call returns None."""
        b = _aero_behaviour({})
        response = MagicMock()
        response.performative = LedgerApiMessage.Performative.ERROR
        b.get_ledger_api_response = _mock_gen(response)
        assert exhaust_generator(b._get_latest_block_number(CHAIN)) is None


class TestAeroClaims:
    """Aero gauges claim with claimEmissions."""

    def test_v2_claim_uses_claim_emissions(self) -> None:
        """The v2 claim pays the Safe through claimEmissions(account, recipient)."""
        b = _aero_behaviour({"earned": 5, "claim_emissions": b"c"})
        result = exhaust_generator(
            b.claim_rewards(
                POOL,
                chain=CHAIN,
                safe_address=SAFE,
                gauge_address=GAUGE,
                protocol=AERO_PROTOCOL,
            )
        )
        assert result["tx_hash"] == b"c"
        (call,) = b.contract_interact.called("claim_emissions")
        assert (call["account"], call["recipient"]) == (SAFE, SAFE)
        assert not b.contract_interact.called("get_reward")

    def test_cl_claim_uses_one_claim_emissions(self) -> None:
        """All staked CL tokens are claimed in one claimEmissions call."""
        b = _aero_behaviour({"earned_by_account": 9, "claim_cl_emissions": b"c"})
        result = exhaust_generator(
            b.claim_cl_rewards(
                GAUGE, [1, 2], chain=CHAIN, safe_address=SAFE, protocol=AERO_PROTOCOL
            )
        )
        assert result["is_multisend"] is False
        assert result["pending_rewards"] == 9
        (call,) = b.contract_interact.called("claim_cl_emissions")
        assert call["token_ids"] == [1, 2]

    def test_cl_claim_nothing_pending(self) -> None:
        """No pending emissions means nothing to claim."""
        b = _aero_behaviour({"earned_by_account": 0})
        result = exhaust_generator(
            b.claim_cl_rewards(
                GAUGE, [1], chain=CHAIN, safe_address=SAFE, protocol=AERO_PROTOCOL
            )
        )
        assert "tx_hash" not in result
        assert result["success"] is True

    def test_cl_claim_tx_fails(self) -> None:
        """A failed claim tx is an error."""
        b = _aero_behaviour({"earned_by_account": 3, "claim_cl_emissions": None})
        result = exhaust_generator(
            b.claim_cl_rewards(
                GAUGE, [1], chain=CHAIN, safe_address=SAFE, protocol=AERO_PROTOCOL
            )
        )
        assert "error" in result

    @pytest.mark.parametrize("earned, expected", [(11, 11), (None, 0)])
    def test_cl_pending_rewards(self, earned, expected) -> None:
        """Aero CL pending rewards come from the account-wide earned(address)."""
        b = _aero_behaviour({"earned_by_account": earned})
        result = exhaust_generator(
            b.get_cl_pending_rewards(
                SAFE, chain=CHAIN, gauge_address=GAUGE, protocol=AERO_PROTOCOL
            )
        )
        assert result == expected
        assert not b.contract_interact.called("earned")


class TestReadPendingRewards:
    """read_pending_rewards tells an unreadable gauge from an empty one."""

    def test_unreadable_earned(self) -> None:
        """A failed earned() read is None, while get_pending_rewards still reports 0."""
        b = _aero_behaviour({"earned": None})
        assert (
            exhaust_generator(
                b.read_pending_rewards(POOL, SAFE, chain=CHAIN, gauge_address=GAUGE)
            )
            is None
        )
        assert (
            exhaust_generator(
                b.get_pending_rewards(POOL, SAFE, chain=CHAIN, gauge_address=GAUGE)
            )
            == 0
        )

    def test_no_gauge_has_nothing_pending(self) -> None:
        """A pool without a gauge has nothing to claim."""
        b = _aero_behaviour({"gauges": ZERO_ADDRESS})
        b.context.params.velodrome_voter_contract_addresses = {CHAIN: "0xVoter"}
        assert exhaust_generator(b.read_pending_rewards(POOL, SAFE, chain=CHAIN)) == 0

    def test_unreadable_gauge(self) -> None:
        """A failed voter lookup is unreadable, not nothing pending."""
        b = _aero_behaviour({"gauges": None})
        b.context.params.velodrome_voter_contract_addresses = {CHAIN: "0xVoter"}
        assert exhaust_generator(b.read_pending_rewards(POOL, SAFE, chain=CHAIN)) is None
        assert not b.contract_interact.called("earned")


class TestReadStakedBalance:
    """read_staked_balance tells an unreadable gauge from an empty one."""

    def test_unreadable_balance(self) -> None:
        """A failed balanceOf() read is None, while get_staked_balance still reports 0."""
        b = _aero_behaviour({"balance_of": None})
        assert (
            exhaust_generator(
                b.read_staked_balance(POOL, SAFE, chain=CHAIN, gauge_address=GAUGE)
            )
            is None
        )
        assert (
            exhaust_generator(
                b.get_staked_balance(POOL, SAFE, chain=CHAIN, gauge_address=GAUGE)
            )
            == 0
        )

    @pytest.mark.parametrize("gauge, expected", [(ZERO_ADDRESS, 0), (None, None)])
    def test_legacy_gauge_lookup(self, gauge: Any, expected: Any) -> None:
        """A pool without a gauge has nothing staked; a failed voter lookup is unreadable."""
        b = _aero_behaviour({"gauges": gauge})
        b.context.params.velodrome_voter_contract_addresses = {CHAIN: "0xVoter"}
        assert (
            exhaust_generator(b.read_staked_balance(POOL, SAFE, chain=CHAIN))
            == expected
        )
        assert not b.contract_interact.called("balance_of")

    def test_empty_balance(self) -> None:
        """A zero balance is read as zero, not as unreadable."""
        b = _aero_behaviour({"balance_of": 0})
        assert (
            exhaust_generator(
                b.read_staked_balance(POOL, SAFE, chain=CHAIN, gauge_address=GAUGE)
            )
            == 0
        )

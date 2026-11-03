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

"""Tests for the start-up reconciliation of CL positions in FetchStrategiesBehaviour."""

# pylint: skip-file

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from eth_abi import decode, encode
from eth_utils import keccak, to_checksum_address

from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    Action,
    MIN_TIME_IN_POSITION,
)
from packages.valory.skills.liquidity_trader_abci.tests.behaviours.test_evaluate_strategy import (
    _mk as _mk_evaluate,
)
from packages.valory.skills.liquidity_trader_abci.tests.behaviours.test_fetch_strategies import (
    _drive,
    _gen_none,
    _gen_return,
    _mk,
)
from packages.valory.skills.liquidity_trader_abci.utils.multicall import (
    MULTICALL3_ADDRESS,
)

CHAIN = "optimism"
SAFE = to_checksum_address("0x7fe908a35c335df9aaedacc83230fb8a8e944080")
VELO_NPM = to_checksum_address("0x416b433906b1b72fa758e166e239c43d68dc6f29")
AERO_NPM = to_checksum_address("0x4857db576ee0e45f10525ec59527bfd10beec60b")
VELO_FACTORY = to_checksum_address("0x" + "f1" * 20)
AERO_FACTORY = to_checksum_address("0xfeab42d90055db3a0616249f76b6002d72266778")
POOL = to_checksum_address("0xba2b011b3de41274946e6ef93337b451fe47d727")
GAUGE = to_checksum_address("0x" + "9a" * 20)
OTHER_POOL = to_checksum_address("0x" + "0b" * 20)
OTHER_GAUGE = to_checksum_address("0x" + "9b" * 20)
USDC = to_checksum_address("0x0b2c639c533813f4aa9d7837caf62653d097ff85")
USDT = to_checksum_address("0x94b008aa00579c1307b0ef2c499ad98a8ce58e58")
ZERO = "0x" + "0" * 40
MINT_TX = "0x2a3848ed96ff3d9d8af713b2372c13630917add97abe6e8d20e2e57ad8b2b2e8"
MINT_TIME = 1_793_424_883

POSITION_TYPES = [
    "uint96",
    "address",
    "address",
    "address",
    "int24",
    "int24",
    "int24",
    "uint128",
    "uint256",
    "uint256",
    "uint128",
    "uint128",
]


class FakeChain:
    """Answers Multicall3 reads from an in-memory chain state."""

    def __init__(self) -> None:
        """Start with a Safe that holds nothing."""
        self.owned: Dict[str, List[int]] = {VELO_NPM: [], AERO_NPM: []}
        self.factories = {VELO_NPM: VELO_FACTORY, AERO_NPM: AERO_FACTORY}
        self.positions: Dict[Tuple[str, int], Tuple[str, str, int, int, int, int]] = {}
        self.pool_gauges: Dict[str, str] = {POOL: GAUGE, OTHER_POOL: OTHER_GAUGE}
        self.staked: Dict[str, List[int]] = {}
        self.gauge_nfts = {GAUGE: AERO_NPM, OTHER_GAUGE: VELO_NPM}
        self.pools = {(AERO_FACTORY, USDC, USDT, 1): POOL}
        self.sqrt_prices = {POOL: 2**96}
        self.failing: set = set()
        self.batches: List[List[str]] = []

    def nft(
        self,
        manager: str,
        token_id: int,
        liquidity: int,
        ticks=(-1, 1),
        pool_tokens=(USDC, USDT, 1),
    ) -> None:
        """Mint ``token_id`` on ``manager`` (held by nobody until owned/staked)."""
        token0, token1, tick_spacing = pool_tokens
        self.positions[(manager, token_id)] = (
            token0,
            token1,
            tick_spacing,
            ticks[0],
            ticks[1],
            liquidity,
        )

    def _answer(self, target: str, signature: str, args: tuple) -> Optional[bytes]:
        target = to_checksum_address(target)
        if signature == "balanceOf(address)":
            return encode(["uint256"], [len(self.owned.get(target, []))])
        if signature == "factory()":
            return encode(["address"], [self.factories[target]])
        if signature == "gauge()":
            return encode(["address"], [self.pool_gauges.get(target, ZERO)])
        if signature == "tokenOfOwnerByIndex(address,uint256)":
            return encode(["uint256"], [self.owned[target][args[1]]])
        if signature == "stakedValues(address)":
            return encode(["uint256[]"], [self.staked.get(target, [])])
        if signature == "nft()":
            return encode(["address"], [self.gauge_nfts[target]])
        if signature == "positions(uint256)":
            token0, token1, spacing, lower, upper, liquidity = self.positions[
                (target, args[0])
            ]
            return encode(
                POSITION_TYPES,
                [0, ZERO, token0, token1, spacing, lower, upper, liquidity, 0, 0, 0, 0],
            )
        if signature == "getPool(address,address,int24)":
            token0, token1, tick_spacing = args
            key = (target, to_checksum_address(token0), to_checksum_address(token1))
            return encode(["address"], [self.pools.get(key + (tick_spacing,), ZERO)])
        if signature == "slot0()":
            return encode(
                ["uint160", "int24", "uint16", "uint16", "uint16", "bool"],
                [self.sqrt_prices[target], 0, 0, 0, 0, True],
            )
        raise AssertionError(f"unexpected read {signature}")

    SIGNATURES = [
        "balanceOf(address)",
        "factory()",
        "gauge()",
        "tokenOfOwnerByIndex(address,uint256)",
        "stakedValues(address)",
        "nft()",
        "positions(uint256)",
        "getPool(address,address,int24)",
        "slot0()",
    ]

    def aggregate3(
        self, calls: List[Tuple[str, bytes, bool]]
    ) -> List[Tuple[bool, bytes]]:
        """Answer each (target, call_data, allow_failure) call."""
        by_selector = {keccak(text=s)[:4]: s for s in self.SIGNATURES}
        results, batch = [], []
        for target, call_data, allow_failure in calls:
            assert allow_failure
            signature = by_selector[call_data[:4]]
            batch.append(signature)
            inner = signature[signature.index("(") + 1 : -1]
            types = [t for t in inner.split(",") if t]
            args = decode(types, call_data[4:]) if types else ()
            if signature in self.failing:
                results.append((False, b""))
                continue
            results.append((True, self._answer(target, signature, args)))
        self.batches.append(batch)
        return results


def _behaviour(
    chain: FakeChain,
    mints: Optional[Dict[Any, str]] = None,
    search_complete: bool = True,
    **params: Any,
):
    """Return a FetchStrategiesBehaviour reading ``chain``."""
    obj = _mk()
    obj.params.target_investment_chains = [CHAIN]
    obj.params.safe_contract_addresses = {CHAIN: SAFE}
    obj.params.velodrome_non_fungible_position_manager_contract_addresses = {
        CHAIN: VELO_NPM
    }
    obj.params.aero_non_fungible_position_manager_contract_addresses = {CHAIN: AERO_NPM}
    obj.params.reconcile_log_window_blocks = 10_000
    obj.params.reconcile_log_min_window_blocks = 500
    obj.params.reconcile_log_lookback_blocks = 302_400
    for name, value in params.items():
        setattr(obj.params, name, value)
    obj.contract_calls: List[Dict[str, Any]] = []
    mints = mints or {}

    def contract_interact(**kwargs):
        obj.contract_calls.append(kwargs)
        yield
        callable_ = kwargs["contract_callable"]
        if callable_ == "aggregate3":
            assert kwargs["contract_address"] == MULTICALL3_ADDRESS
            if "aggregate3" in chain.failing:
                return None
            return chain.aggregate3(kwargs["calls"])
        if callable_ == "get_mint_tx_hashes":
            manager = kwargs["contract_address"]
            found = {}
            for token_id in kwargs["token_ids"]:
                tx_hash = mints.get((manager, token_id), mints.get(token_id))
                if tx_hash:
                    found[token_id] = tx_hash
            return {"tx_hashes": found, "complete": search_complete}
        if callable_ == "get_token_symbol":
            return {USDC: "USDC", USDT: "USDT"}[kwargs["contract_address"]]
        raise AssertionError(f"unexpected contract call {callable_}")

    obj.contract_interact = contract_interact
    obj.get_block = MagicMock(side_effect=_gen_return({"number": 143_000_000}))
    obj._get_all_positions_from_tx_receipt = MagicMock(
        side_effect=_gen_return(
            [
                (57869, 1_000, 3_000_000, 2_000_000, MINT_TIME),
                (57870, 2_000, 4_000_000, 2_060_000, MINT_TIME),
            ]
        )
    )
    obj._get_cached_cl_pool_data = MagicMock(side_effect=_gen_return(None))
    obj._invalidate_cl_pool_cache = MagicMock(side_effect=_gen_return(True))
    obj._accumulate_transaction_costs = MagicMock(side_effect=_gen_none)
    obj._rename_entry_costs_key = MagicMock(side_effect=_gen_none)
    obj._get_updated_entry_costs = MagicMock(side_effect=_gen_return(0.01))
    obj._convert_amounts_to_usd = MagicMock(side_effect=_gen_return(11.06))
    obj._get_current_timestamp = lambda: 1_793_500_000
    obj.update_position_amounts = MagicMock(side_effect=_gen_none)
    obj.store_current_positions = MagicMock()
    obj.get_http_response = MagicMock(
        side_effect=AssertionError("reconciliation must not call HTTP services")
    )
    return obj


def _incident_chain() -> FakeChain:
    """A Safe holding two Aero NFTs minted in one tx and recorded nowhere."""
    chain = FakeChain()
    chain.nft(AERO_NPM, 57869, 1_000)
    chain.nft(AERO_NPM, 57870, 2_000)
    chain.owned[AERO_NPM] = [57869, 57870]
    return chain


def _closed_entries() -> List[Dict[str, Any]]:
    """Eleven closed Velodrome CL entries in another pool, their NFTs emptied."""
    return [
        {
            "chain": CHAIN,
            "pool_address": OTHER_POOL,
            "dex_type": "velodrome",
            "is_cl_pool": True,
            "status": "closed",
            "positions": [{"token_id": 100 + i, "liquidity": 5}],
        }
        for i in range(11)
    ]


def _open_entry(token_ids: List[int], **extra: Any) -> Dict[str, Any]:
    """An open Aero CL entry in POOL."""
    return {
        "chain": CHAIN,
        "pool_address": POOL,
        "dex_type": "velodrome",
        "protocol": "aero",
        "is_cl_pool": True,
        "status": "open",
        "staked": False,
        "positions": [{"token_id": t, "liquidity": 7} for t in token_ids],
        **extra,
    }


class TestAdoption:
    """NFTs on chain that current_positions does not hold become open positions."""

    def test_entry_lost_after_its_tx_mined_is_adopted_with_its_mint_data(self) -> None:
        """An NFT minted before _post_execute_enter_pool ran is adopted with its mint amounts and tx hash."""
        chain = _incident_chain()
        for i in range(11):
            chain.nft(VELO_NPM, 100 + i, 0, pool_tokens=(USDC, USDT, 100))
        chain.owned[VELO_NPM] = [100 + i for i in range(11)]
        obj = _behaviour(chain, mints={57869: MINT_TX, 57870: MINT_TX})
        obj.current_positions = _closed_entries()

        _drive(obj.reconcile_positions())

        (adopted,) = [p for p in obj.current_positions if p.get("adopted")]
        assert adopted == {
            "chain": CHAIN,
            "pool_address": POOL,
            "pool_id": POOL,
            "dex_type": "velodrome",
            "protocol": "aero",
            "token0": USDC,
            "token1": USDT,
            "token0_symbol": "USDC",
            "token1_symbol": "USDT",
            "is_cl_pool": True,
            "gauge_address": GAUGE,
            "staked": False,
            "status": "open",
            "enter_tx_hash": MINT_TX,
            "enter_timestamp": MINT_TIME,
            "amount0": 7_000_000,
            "amount1": 4_060_000,
            "positions": [
                {
                    "token_id": 57869,
                    "liquidity": 1_000,
                    "current_liquidity": 1_000,
                    "amount0": 3_000_000,
                    "amount1": 2_000_000,
                },
                {
                    "token_id": 57870,
                    "liquidity": 2_000,
                    "current_liquidity": 2_000,
                    "amount0": 4_000_000,
                    "amount1": 2_060_000,
                },
            ],
            "min_hold_days": MIN_TIME_IN_POSITION,
            "principal_usd": 11.06,
            "entry_cost": 0.02,
            "cost_recovered": False,
            "adopted": True,
        }
        assert all(p["status"] == "closed" for p in obj.current_positions[:11])
        obj._accumulate_transaction_costs.assert_called_once_with(MINT_TX, adopted)
        (search,) = [
            c
            for c in obj.contract_calls
            if c["contract_callable"] == "get_mint_tx_hashes"
        ]
        assert search["contract_address"] == AERO_NPM
        assert search["data_key"] == "mint_search"
        assert search["owner"] == SAFE
        assert sorted(search["token_ids"]) == [57869, 57870]
        assert search["to_block"] == 143_000_000
        assert (
            search["window_blocks"],
            search["min_window_blocks"],
            search["lookback_blocks"],
        ) == (10_000, 500, 302_400)
        assert search["chain_id"] == CHAIN
        obj._convert_amounts_to_usd.assert_called_once_with(
            7_000_000, 4_060_000, USDC, USDT, CHAIN
        )

    def test_same_token_id_on_both_managers_keeps_its_own_mint(self) -> None:
        """Token IDs are per manager: equal IDs on Velodrome and Aero do not share mint data."""
        velo_tx, aero_tx = "0x" + "11" * 32, "0x" + "22" * 32
        chain = FakeChain()
        chain.nft(VELO_NPM, 7, 10, pool_tokens=(USDC, USDT, 100))
        chain.nft(AERO_NPM, 7, 20)
        chain.owned = {VELO_NPM: [7], AERO_NPM: [7]}
        chain.pools[(VELO_FACTORY, USDC, USDT, 100)] = OTHER_POOL
        obj = _behaviour(chain, mints={(VELO_NPM, 7): velo_tx, (AERO_NPM, 7): aero_tx})
        receipts = {
            velo_tx: [(7, 10, 100, 200, 1_000)],
            aero_tx: [(7, 20, 300, 400, 2_000)],
        }
        obj._get_all_positions_from_tx_receipt = MagicMock(
            side_effect=lambda tx_hash, chain: _gen_return(receipts[tx_hash])()
        )

        _drive(obj.reconcile_positions())

        adopted = {p["pool_address"]: p for p in obj.current_positions}
        assert adopted[OTHER_POOL]["enter_tx_hash"] == velo_tx
        assert (adopted[OTHER_POOL]["amount0"], adopted[OTHER_POOL]["amount1"]) == (
            100,
            200,
        )
        assert adopted[POOL]["enter_tx_hash"] == aero_tx
        assert (adopted[POOL]["amount0"], adopted[POOL]["amount1"]) == (300, 400)

    def test_adopted_position_is_stored(self) -> None:
        """The adopted entry is in what reconciliation writes to current_positions."""
        obj = _behaviour(_incident_chain(), mints={57869: MINT_TX, 57870: MINT_TX})
        stored: List[List[Dict[str, Any]]] = []
        obj.store_current_positions = MagicMock(
            side_effect=lambda: stored.append([dict(p) for p in obj.current_positions])
        )

        _drive(obj.reconcile_positions())

        assert stored and [p.get("adopted") for p in stored[-1]] == [True]

    def test_base_adoption_is_an_aerodrome_position(self) -> None:
        """On Base an adopted NFT gets the aerodrome dex type the pool search assigns there."""
        obj = _behaviour(_incident_chain(), mints={57869: MINT_TX, 57870: MINT_TX})
        for name in (
            "target_investment_chains",
            "safe_contract_addresses",
            "velodrome_non_fungible_position_manager_contract_addresses",
            "aero_non_fungible_position_manager_contract_addresses",
        ):
            value = getattr(obj.params, name)
            setattr(
                obj.params,
                name,
                ["base"] if isinstance(value, list) else {"base": value[CHAIN]},
            )

        _drive(obj.reconcile_positions())

        (adopted,) = obj.current_positions
        assert adopted["chain"] == "base"
        assert adopted["dex_type"] == "aerodrome"

    def test_nft_without_a_mint_at_an_off_centre_price_uses_pool_ticks(self) -> None:
        """Current amounts follow the pool's tick convention: a band above the price is all token0."""
        chain = FakeChain()
        chain.nft(AERO_NPM, 57869, 10**12, ticks=(100, 200))
        chain.owned[AERO_NPM] = [57869]
        obj = _behaviour(chain)

        _drive(obj.reconcile_positions())

        (band,) = obj.current_positions[0]["positions"]
        assert band["amount0"] > 0
        assert band["amount1"] == 0

    def test_failed_mint_search_is_an_error_not_a_miss(self) -> None:
        """A search the RPC cut short is logged as a failure, not as a mint older than the lookback."""
        obj = _behaviour(_incident_chain(), search_complete=False)

        _drive(obj.reconcile_positions())

        assert [p["enter_timestamp"] for p in obj.current_positions] == [
            1_793_500_000
        ] * 2
        errors = " ".join(str(c) for c in obj.context.logger.error.call_args_list)
        warnings = " ".join(str(c) for c in obj.context.logger.warning.call_args_list)
        assert "mint search on" in errors
        assert "within the lookback" not in warnings

    def test_staked_nft_missing_from_the_file_is_adopted_as_staked(self) -> None:
        """An NFT found in a known pool's gauge is adopted with staked True."""
        chain = FakeChain()
        chain.nft(AERO_NPM, 57869, 1_000)
        chain.staked[GAUGE] = [57869]
        obj = _behaviour(chain, mints={57869: MINT_TX})
        obj._get_cached_cl_pool_data = MagicMock(
            side_effect=_gen_return({"pool_address": POOL})
        )

        _drive(obj.reconcile_positions())

        (adopted,) = obj.current_positions
        assert adopted["staked"] is True
        assert adopted["pool_address"] == POOL
        assert [b["token_id"] for b in adopted["positions"]] == [57869]

    def test_nft_without_a_found_mint_is_valued_at_current_amounts(self) -> None:
        """Without a mint tx the NFT gets its own entry at today's amounts and time."""
        chain = FakeChain()
        chain.nft(AERO_NPM, 57869, 10**12, ticks=(-10, 10))
        chain.owned[AERO_NPM] = [57869]
        obj = _behaviour(chain)

        _drive(obj.reconcile_positions())

        (adopted,) = obj.current_positions
        assert adopted["enter_timestamp"] == 1_793_500_000
        assert "enter_tx_hash" not in adopted
        assert adopted["entry_cost"] == 0.0
        band = adopted["positions"][0]
        # Price at the middle of a symmetric band: equal raw amounts of both tokens.
        assert band["amount0"] == band["amount1"] > 0
        obj._accumulate_transaction_costs.assert_not_called()

    def test_velodrome_nft_is_adopted_without_a_protocol(self) -> None:
        """An NFT of the Velodrome manager is adopted as a legacy Velodrome position."""
        chain = FakeChain()
        chain.nft(VELO_NPM, 9, 10, pool_tokens=(USDC, USDT, 100))
        chain.owned[VELO_NPM] = [9]
        chain.pools[(VELO_FACTORY, USDC, USDT, 100)] = OTHER_POOL
        chain.sqrt_prices[OTHER_POOL] = 2**96
        obj = _behaviour(chain)

        _drive(obj.reconcile_positions())

        (adopted,) = obj.current_positions
        assert "protocol" not in adopted
        assert adopted["pool_address"] == OTHER_POOL
        assert adopted["gauge_address"] == OTHER_GAUGE

    def test_empty_nfts_are_not_adopted(self) -> None:
        """Exited positions keep their NFT; zero liquidity is filtered without a skip list."""
        chain = FakeChain()
        chain.nft(AERO_NPM, 57869, 0)
        chain.owned[AERO_NPM] = [57869]
        obj = _behaviour(chain)

        _drive(obj.reconcile_positions())

        assert obj.current_positions == []

    @pytest.mark.parametrize(
        "cached_apr, expected", [(12.5, {"entry_apr": 12.5}), (0, {})]
    )
    def test_entry_apr_comes_only_from_a_cached_opportunity_for_the_pool(
        self, cached_apr: float, expected: Dict[str, float]
    ) -> None:
        """entry_apr is the cached opportunity's APR for the same pool, never 0."""
        obj = _behaviour(_incident_chain(), mints={57869: MINT_TX, 57870: MINT_TX})
        cached = {
            "pool_address": POOL.lower(),
            "enter_pool_action": {
                "opportunity_apr": cached_apr,
                "dex_type": "velodrome",
                "token0_symbol": "USDC",
                "token1_symbol": "USDT",
            },
        }
        obj._get_cached_cl_pool_data = MagicMock(side_effect=_gen_return(cached))

        _drive(obj.reconcile_positions())

        (adopted,) = obj.current_positions
        assert {k: v for k, v in adopted.items() if k == "entry_apr"} == expected

    def test_cached_opportunity_for_an_adopted_pool_is_cleared(self) -> None:
        """A cached CL opportunity for a pool that now has an open position is cleared."""
        obj = _behaviour(_incident_chain(), mints={57869: MINT_TX, 57870: MINT_TX})
        obj._get_cached_cl_pool_data = MagicMock(
            side_effect=_gen_return({"pool_address": POOL})
        )

        _drive(obj.reconcile_positions())

        obj._invalidate_cl_pool_cache.assert_called_once_with(CHAIN)

    def test_failure_to_clear_the_cache_is_logged(self) -> None:
        """A cache that could not be cleared is reported, since a stale one re-enters the pool."""
        obj = _behaviour(_incident_chain(), mints={57869: MINT_TX, 57870: MINT_TX})
        obj._get_cached_cl_pool_data = MagicMock(
            side_effect=_gen_return({"pool_address": POOL})
        )
        obj._invalidate_cl_pool_cache = MagicMock(side_effect=_gen_return(False))

        _drive(obj.reconcile_positions())

        errors = " ".join(str(c) for c in obj.context.logger.error.call_args_list)
        assert "Could not clear the cached CL opportunity" in errors

    def test_cached_opportunity_for_another_pool_is_kept(self) -> None:
        """A cache for a pool with no open position is left alone."""
        obj = _behaviour(_incident_chain(), mints={57869: MINT_TX, 57870: MINT_TX})
        obj._get_cached_cl_pool_data = MagicMock(
            side_effect=_gen_return({"pool_address": OTHER_POOL})
        )

        _drive(obj.reconcile_positions())

        obj._invalidate_cl_pool_cache.assert_not_called()


class TestFileEntries:
    """Entries in current_positions are checked against the chain."""

    def test_open_entry_with_zero_liquidity_is_closed(self) -> None:
        """An open entry whose NFTs are empty on chain is marked closed."""
        chain = FakeChain()
        chain.nft(AERO_NPM, 1, 0)
        chain.owned[AERO_NPM] = [1]
        obj = _behaviour(chain)
        obj.current_positions = [_open_entry([1])]

        _drive(obj.reconcile_positions())

        assert obj.current_positions[0]["status"] == "closed"

    def test_open_entry_held_nowhere_is_closed(self) -> None:
        """An open entry whose NFTs are in neither the Safe nor its gauge is closed."""
        obj = _behaviour(FakeChain())
        obj.current_positions = [_open_entry([1, 2])]

        _drive(obj.reconcile_positions())

        assert obj.current_positions[0]["status"] == "closed"

    def test_open_entry_found_in_its_gauge_stays_open_and_staked(self) -> None:
        """A stake that settled after a restart is recorded, not mistaken for an exit."""
        chain = FakeChain()
        chain.nft(AERO_NPM, 1, 50)
        chain.staked[GAUGE] = [1]
        obj = _behaviour(chain)
        obj.current_positions = [_open_entry([1])]

        _drive(obj.reconcile_positions())

        entry = obj.current_positions[0]
        assert entry["status"] == "open"
        assert entry["staked"] is True
        assert entry["positions"][0]["current_liquidity"] == 50
        assert len(obj.current_positions) == 1

    def test_staked_entry_back_in_the_safe_is_unstaked(self) -> None:
        """The staked flag follows where the NFT was found."""
        chain = FakeChain()
        chain.nft(AERO_NPM, 1, 50)
        chain.owned[AERO_NPM] = [1]
        obj = _behaviour(chain)
        obj.current_positions = [_open_entry([1], staked=True)]

        _drive(obj.reconcile_positions())

        assert obj.current_positions[0]["staked"] is False

    def test_closed_entry_still_holding_liquidity_is_reopened(self) -> None:
        """Liquidity on chain under a closed entry is reopened rather than duplicated."""
        chain = FakeChain()
        chain.nft(AERO_NPM, 1, 50)
        chain.owned[AERO_NPM] = [1]
        obj = _behaviour(chain)
        obj.current_positions = [_open_entry([1], status="closed")]

        _drive(obj.reconcile_positions())

        assert [p["status"] for p in obj.current_positions] == ["open"]

    def test_entry_on_an_unconfigured_manager_is_left_alone(self) -> None:
        """Without a position manager for its protocol nothing about it is known."""
        obj = _behaviour(
            FakeChain(),
            aero_non_fungible_position_manager_contract_addresses={},
        )
        obj.current_positions = [_open_entry([1])]

        _drive(obj.reconcile_positions())

        assert obj.current_positions[0]["status"] == "open"
        assert "current_liquidity" not in obj.current_positions[0]["positions"][0]

    def test_entry_without_a_pool_is_not_closed_for_a_missing_nft(self) -> None:
        """Without its pool the gauge was not read, so a missing NFT proves nothing."""
        obj = _behaviour(FakeChain())
        entry = _open_entry([1, None])
        del entry["pool_address"]
        obj.current_positions = [entry]

        _drive(obj.reconcile_positions())

        assert obj.current_positions[0]["positions"][0] == {
            "token_id": 1,
            "liquidity": 7,
        }

    def test_other_positions_are_refreshed_as_before(self) -> None:
        """Only the reconciled chains' CL positions skip the per-position refresh."""
        obj = _behaviour(FakeChain())

        _drive(obj.reconcile_positions())

        obj.update_position_amounts.assert_called_once_with(skip_cl_chains=[CHAIN])


class TestReadFailures:
    """A failed read never closes a position."""

    @pytest.mark.parametrize(
        "failing",
        [
            "aggregate3",
            "gauge()",
            "tokenOfOwnerByIndex(address,uint256)",
            "positions(uint256)",
        ],
    )
    def test_failed_read_falls_back_to_the_stored_positions(self, failing: str) -> None:
        """When any batch or read fails the chain is not reconciled and nothing is closed."""
        chain = _incident_chain()
        chain.failing = {failing}
        obj = _behaviour(chain)
        obj.current_positions = [_open_entry([1])]

        _drive(obj.reconcile_positions())

        assert obj.current_positions[0]["status"] == "open"
        assert len(obj.current_positions) == 1
        obj.update_position_amounts.assert_called_once_with(skip_cl_chains=[])

    def test_only_nfts_of_known_pools_are_adopted(self) -> None:
        """An NFT with no pool does not stop the others from being adopted."""
        chain = _incident_chain()
        chain.nft(AERO_NPM, 3, 10, pool_tokens=(USDC, USDT, 50))
        chain.owned[AERO_NPM].append(3)
        obj = _behaviour(chain, mints={57869: MINT_TX, 57870: MINT_TX})

        _drive(obj.reconcile_positions())

        (adopted,) = obj.current_positions
        assert [b["token_id"] for b in adopted["positions"]] == [57869, 57870]

    def test_nft_of_an_unknown_pool_is_not_adopted(self) -> None:
        """An NFT whose (token0, token1, tickSpacing) has no pool in the factory is skipped."""
        chain = _incident_chain()
        chain.pools = {}
        obj = _behaviour(chain)

        _drive(obj.reconcile_positions())

        assert obj.current_positions == []

    @pytest.mark.parametrize("failing", ["getPool(address,address,int24)", "slot0()"])
    def test_failed_adoption_read_adopts_nothing(self, failing: str) -> None:
        """When the pool or price of an untracked NFT cannot be read it is not adopted yet."""
        chain = _incident_chain()
        chain.failing = {failing}
        obj = _behaviour(chain)

        _drive(obj.reconcile_positions())

        assert obj.current_positions == []

    def test_unreadable_head_block_adopts_at_current_amounts(self) -> None:
        """Without the head block the mint search is skipped."""
        obj = _behaviour(_incident_chain(), mints={57869: MINT_TX, 57870: MINT_TX})
        obj.get_block = MagicMock(side_effect=_gen_return(None))

        _drive(obj.reconcile_positions())

        assert [p["enter_timestamp"] for p in obj.current_positions] == [
            1_793_500_000,
            1_793_500_000,
        ]

    def test_gauge_staking_another_managers_nfts_is_ignored(self) -> None:
        """Staked NFTs of an unconfigured manager are not adopted."""
        chain = FakeChain()
        chain.gauge_nfts[GAUGE] = to_checksum_address("0x" + "ee" * 20)
        chain.staked[GAUGE] = [5]
        obj = _behaviour(chain)
        obj._get_cached_cl_pool_data = MagicMock(
            side_effect=_gen_return({"pool_address": POOL})
        )

        _drive(obj.reconcile_positions())

        assert obj.current_positions == []

    def test_gauge_of_an_unknown_manager_does_not_close_its_staked_entry(self) -> None:
        """When a gauge's NFTs cannot be attributed, its pool's open entry is left untouched."""
        chain = FakeChain()
        chain.gauge_nfts[GAUGE] = to_checksum_address("0x" + "ee" * 20)
        chain.staked[GAUGE] = [1]
        obj = _behaviour(chain)
        obj.current_positions = [_open_entry([1], staked=True)]

        _drive(obj.reconcile_positions())

        entry = obj.current_positions[0]
        assert entry["status"] == "open"
        assert "current_liquidity" not in entry["positions"][0]

    def test_unexpected_error_falls_back_to_the_stored_positions(self) -> None:
        """An exception while reconciling a chain leaves it to the per-position refresh."""
        obj = _behaviour(_incident_chain())
        obj._read_cl_holdings = MagicMock(side_effect=RuntimeError("boom"))
        obj.current_positions = [_open_entry([1])]

        _drive(obj.reconcile_positions())

        assert obj.current_positions[0]["status"] == "open"
        obj.update_position_amounts.assert_called_once_with(skip_cl_chains=[])
        obj.context.logger.exception.assert_called_once()

    def test_chain_without_a_safe_or_manager_is_not_reconciled(self) -> None:
        """Nothing is read for a chain the agent has no Safe on."""
        obj = _behaviour(FakeChain(), safe_contract_addresses={})

        _drive(obj.reconcile_positions())

        assert obj.contract_calls == []
        obj.update_position_amounts.assert_called_once_with(skip_cl_chains=[])


class TestBoundedReads:
    """Reconciliation needs no Safe Transaction Service and a fixed number of batches."""

    def test_batches_do_not_grow_with_the_number_of_nfts(self) -> None:
        """Thirty untracked NFTs are read in five Multicall3 batches and one log search."""
        chain = FakeChain()
        for token_id in range(1, 31):
            chain.nft(AERO_NPM, token_id, 10, ticks=(-10, 10))
        chain.owned[AERO_NPM] = list(range(1, 16))
        chain.staked[GAUGE] = list(range(16, 31))
        obj = _behaviour(chain)
        obj._get_cached_cl_pool_data = MagicMock(
            side_effect=_gen_return({"pool_address": POOL})
        )

        _drive(obj.reconcile_positions())

        assert len(obj.current_positions) == 30
        callables = [c["contract_callable"] for c in obj.contract_calls]
        assert callables.count("aggregate3") == 5
        assert callables.count("get_mint_tx_hashes") == 1
        obj.get_http_response.assert_not_called()


class TestAdoptedPositionDownstream:
    """An adopted position works with the flows that read current_positions."""

    def _adopted(self) -> Dict[str, Any]:
        obj = _behaviour(_incident_chain(), mints={57869: MINT_TX, 57870: MINT_TX})
        _drive(obj.reconcile_positions())
        (adopted,) = obj.current_positions
        return adopted

    def test_withdraw_flow_exits_an_adopted_position(self) -> None:
        """The withdrawal exit action carries both NFTs on the Aero manager."""
        obj = _mk()

        (action,) = obj._prepare_exit_pool_actions([self._adopted()])

        assert action["action"] == "ExitPool"
        assert action["pool_address"] == POOL
        assert action["protocol"] == "aero"
        assert action["token_ids"] == [57869, 57870]
        assert action["liquidities"] == [1_000, 2_000]

    def test_portfolio_values_an_adopted_position(self) -> None:
        """The adopted position is valued through the Velodrome handler."""
        obj = _mk()
        obj.current_positions = [self._adopted()]
        obj.params.safe_contract_addresses = {CHAIN: SAFE}
        obj.store_current_positions = MagicMock()
        valued: List[Dict[str, Any]] = []

        def handler(position, chain):
            valued.append(position)
            yield
            return ({USDC: Decimal("7"), USDT: Decimal("4")}, "Velodrome", {})

        obj._handle_velodrome_position = handler
        obj._calculate_position_value = _gen_return(Decimal("11.06"))
        obj._calculate_safe_balances_value = _gen_return(Decimal("0"))
        obj.calculate_stakig_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_airdrop_rewards_value = _gen_return(Decimal("0"))
        obj.calculate_withdrawals_value = _gen_return(Decimal("0"))
        captured: Dict[str, Any] = {}

        def metrics(total, *args):
            captured["total"] = total
            yield

        obj._update_portfolio_metrics = metrics
        obj.calculate_initial_investment_value_from_funding_events = _gen_return(1.0)
        obj._calculate_total_volume = _gen_return(1.0)
        obj._create_portfolio_data = lambda *a, **kw: {}
        obj.store_portfolio_data = MagicMock()

        _drive(obj.calculate_user_share_values())

        assert [p["pool_address"] for p in valued] == [POOL]
        assert captured["total"] == Decimal("11.06")

    def test_adopted_aero_position_in_the_safe_is_restaked(self) -> None:
        """The next cycle stakes an adopted Aero position found unstaked."""
        adopted = self._adopted()
        obj = _mk_evaluate(
            current_positions=[adopted], pools={"velodrome": MagicMock()}
        )
        obj.params.safe_contract_addresses = {CHAIN: SAFE}
        obj._filter_staked_token_ids = MagicMock(side_effect=_gen_return(([], False)))
        obj.store_current_positions = MagicMock()

        actions = _drive(obj._prepare_aero_restake_actions())

        assert [a["action"] for a in actions] == [Action.STAKE_LP_TOKENS.value]
        assert actions[0]["gauge_address"] == GAUGE
        assert actions[0]["protocol"] == "aero"

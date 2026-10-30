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

"""Tests for the Aero discovery and pricing in velodrome_pools_search."""

# pylint: skip-file

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
from web3 import Web3

import packages.valory.customs.velodrome_pools_search.velodrome_pools_search as velo
from packages.valory.customs.velodrome_pools_search.velodrome_pools_search import (
    AERO_CONTRACT_ADDRESSES,
    AERO_HOURLY_LOOKBACKS,
    AERO_PROTOCOL,
    AERO_TOKEN_ADDRESS_BASE,
    BASE_CHAIN_ID,
    OPTIMISM_CHAIN_ID,
    SECONDS_PER_DAY,
    WAD,
    _cl_liquidity_per_token1,
    _daily_rates,
    _rate_per_second,
    _signature_types,
    aero_cl_pool_metrics,
    aero_daily_volume_usd,
    aero_depth_metrics,
    aero_multicall,
    aero_v2_pool_metrics,
    calculate_aero_metrics,
    calculate_metrics,
    discover_aero_pools,
    get_aero_pools,
    get_aero_price_usd,
    get_opportunities_for_aero,
    get_token_decimals,
    read_aero_pool_state,
    run,
    sharpe_ratio_from_daily_returns,
)

MODULE = "packages.valory.customs.velodrome_pools_search.velodrome_pools_search"
BASE = AERO_CONTRACT_ADDRESSES[BASE_CHAIN_ID]
USDC = Web3.to_checksum_address("0x833589fcd6edb6e08f4c7c32d4f71b54bda02913")
EUSD = Web3.to_checksum_address("0xcfa3ef56d303ae4faaba0592388f19d7c3399fb4")
MSUSD = Web3.to_checksum_address("0x526728dbc96689597f85ae4cd716d4f7fccbae9d")
WETH = Web3.to_checksum_address("0x4200000000000000000000000000000000000006")
WHITELIST = {"base": {USDC.lower(): "USDC", EUSD.lower(): "eUSD"}}


def _address(n: int) -> str:
    return Web3.to_checksum_address("0x" + f"{n:040x}")


class _Multicall:
    """A fake aero_multicall answering by (target, signature) or signature."""

    def __init__(self, answers: Dict[Any, Any]) -> None:
        self.answers = answers
        self.calls: List[Tuple[str, str, Tuple[Any, ...]]] = []

    def __call__(self, w3: Any, calls: List[Any]) -> List[Optional[Tuple[Any, ...]]]:
        results = []
        for target, signature, args, _ in calls:
            self.calls.append((target, signature, args))
            answer = self.answers.get((target, signature), self.answers.get(signature))
            results.append(answer(target, args) if callable(answer) else answer)
        return results

    def signatures(self) -> List[str]:
        return [signature for _, signature, _ in self.calls]


class TestSignatureTypes:
    """Tests for _signature_types."""

    def test_types(self) -> None:
        """Argument types are read from a flat signature."""
        assert _signature_types("observe(address,uint48[])") == ["address", "uint48[]"]
        assert _signature_types("allPoolsLength()") == []


class _Aggregate3:
    """A fake Multicall3 whose answers are produced by respond."""

    def __init__(self, respond: Any) -> None:
        self.respond = respond
        self.batches: List[int] = []

    def __call__(self, encoded: List[Any]) -> Any:
        self.batches.append(len(encoded))
        call = MagicMock()
        call.call.side_effect = lambda: self.respond(encoded)
        return call


def _w3_with(aggregate3: Any) -> Any:
    w3 = MagicMock()
    w3.codec = Web3().codec
    w3.eth.contract.return_value.functions.aggregate3 = aggregate3
    return w3


class TestAeroMulticall:
    """Tests for aero_multicall."""

    def test_decodes_and_marks_failures(self) -> None:
        """Successful reads are decoded; failed or undecodable ones are None."""
        codec = Web3().codec

        def respond(encoded: List[Any]) -> List[Any]:
            assert encoded[0][2][:4] == Web3.keccak(text="balanceOf(address)")[:4]
            assert encoded[0][1] is True
            return [
                (True, codec.encode(["uint256"], [7])),
                (False, b""),
                (True, b"\x01"),
            ]

        w3 = _w3_with(_Aggregate3(respond))
        result = aero_multicall(
            w3,
            [
                (USDC, "balanceOf(address)", (EUSD,), ["uint256"]),
                (USDC, "decimals()", (), ["uint8"]),
                (USDC, "decimals()", (), ["uint256"]),
            ],
        )
        assert result == [(7,), None, None]

    def test_failed_batch_and_batching(self) -> None:
        """Reads are sent in batches; a reverted batch yields None for each read."""
        aggregate3 = _Aggregate3(MagicMock(side_effect=Exception("rpc down")))
        w3 = _w3_with(aggregate3)
        with patch(f"{MODULE}.MULTICALL_BATCH_SIZE", 2):
            result = aero_multicall(w3, [(USDC, "decimals()", (), ["uint8"])] * 3)
        assert result == [None, None, None]
        assert aggregate3.batches == [2, 1]


def _discovery_answers(volatile_pools: List[str]) -> Dict[Any, Any]:
    """Answers for one stable, a list of volatile and no CL pools."""
    stable_pool = _address(1)
    pools = {
        BASE["stable_pool_factory"]: [stable_pool],
        BASE["volatile_pool_factory"]: volatile_pools,
        BASE["cl_factory"]: [],
    }
    return {
        "allPoolsLength()": lambda target, args: (len(pools[target]),),
        "allPools(uint256)": lambda target, args: (pools[target][args[0]],),
        "token0()": (USDC.lower(),),
        "token1()": (EUSD.lower(),),
        "computeGaugeAddress(address)": lambda target, args: (
            _address(1000 + int(args[0], 16)),
        ),
    }


class TestDiscoverAeroPools:
    """Tests for discover_aero_pools."""

    def test_lists_pools_and_reuses_cache(self) -> None:
        """Pools are read once; the next cycle only reads pools created since."""
        cache: Dict[str, Any] = {}
        multicall = _Multicall(_discovery_answers([_address(2)]))
        with patch(f"{MODULE}.aero_multicall", multicall):
            result = discover_aero_pools(MagicMock(), BASE_CHAIN_ID, cache)

        assert result["pools"][_address(1)] == ["stable", USDC, EUSD, _address(1001)]
        assert result["pools"][_address(2)][0] == "volatile"
        assert result["counts"][BASE["volatile_pool_factory"]] == 1
        assert cache[f"aero_pools_{BASE_CHAIN_ID}_index"]["data"] is result

        multicall = _Multicall(_discovery_answers([_address(2), _address(3)]))
        with patch(f"{MODULE}.aero_multicall", multicall):
            again = discover_aero_pools(MagicMock(), BASE_CHAIN_ID, cache)

        assert set(again["pools"]) == {_address(1), _address(2), _address(3)}
        listed = [
            args for _, sig, args in multicall.calls if sig == "allPools(uint256)"
        ]
        assert listed == [(1,)]

    @pytest.mark.parametrize(
        "failing", ["allPoolsLength()", "allPools(uint256)", "token1()"]
    )
    def test_failed_reads_are_retried_later(self, failing: str) -> None:
        """A factory whose pools cannot be read keeps its count, so they are read again."""
        answers = _discovery_answers([_address(2)])
        answers[failing] = None
        with patch(f"{MODULE}.aero_multicall", _Multicall(answers)):
            result = discover_aero_pools(MagicMock(), BASE_CHAIN_ID, None)
        assert result["pools"] == {}
        assert BASE["volatile_pool_factory"] not in result["counts"]


class TestGetTokenDecimals:
    """Tests for get_token_decimals."""

    def test_reads_only_unknown_tokens(self) -> None:
        """Known decimals are reused and unreadable tokens are left out."""
        cache = {"decimals": {USDC: 6}}
        multicall = _Multicall(
            {(EUSD, "decimals()"): (18,), (WETH, "decimals()"): None}
        )
        with patch(f"{MODULE}.aero_multicall", multicall):
            result = get_token_decimals(MagicMock(), [USDC, EUSD, WETH], cache)
        assert result == {USDC: 6, EUSD: 18}
        assert [target for target, _, _ in multicall.calls] == sorted([EUSD, WETH])


class TestGetAeroPriceUsd:
    """Tests for get_aero_price_usd."""

    TOKEN = AERO_TOKEN_ADDRESS_BASE.lower()

    def test_cached(self) -> None:
        """A cached price is reused."""
        cache: Dict[str, Any] = {}
        velo.set_cached_value(self.TOKEN, 0, {"usd": 0.5}, cache, "aero_spot")
        assert get_aero_price_usd(None, price_cache=cache) == 0.5

    def test_recent_failure_not_retried(self) -> None:
        """A recent failure is not paid for again."""
        cache: Dict[str, Any] = {}
        velo.set_failed_fetch(self.TOKEN, 0, cache, "aero_spot")
        with patch(f"{MODULE}.CoinGeckoAPI") as cg:
            assert get_aero_price_usd("key", price_cache=cache) is None
        cg.assert_not_called()

    def test_x402(self) -> None:
        """With x402 the token price is requested by Base contract."""
        session = MagicMock()
        session.adapters = {}
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {self.TOKEN: {"usd": 0.678}}
        cache: Dict[str, Any] = {}
        assert get_aero_price_usd(None, session, "https://proxy/", cache) == 0.678
        url = session.get.call_args.args[0]
        assert url == "https://proxy/api/v3/simple/token_price/base"
        assert (
            session.get.call_args.kwargs["params"]["contract_addresses"] == self.TOKEN
        )
        assert get_aero_price_usd(None, price_cache=cache) == 0.678

    def test_x402_error_status(self) -> None:
        """A non-200 answer is a failed fetch."""
        session = MagicMock()
        session.adapters = {}
        session.get.return_value.status_code = 402
        cache: Dict[str, Any] = {}
        assert get_aero_price_usd(None, session, "https://proxy", cache) is None
        assert velo.is_recent_failure(self.TOKEN, 0, cache, "aero_spot")

    @pytest.mark.parametrize(
        "key, kwarg", [("CG-demo", "demo_api_key"), ("pro", "api_key")]
    )
    def test_api_key(self, key: str, kwarg: str) -> None:
        """Without x402 the CoinGecko client prices AERO by contract."""
        with (
            patch(f"{MODULE}.CoinGeckoAPI") as cg,
            patch(f"{MODULE}.is_pro_api_key", return_value=kwarg == "api_key"),
        ):
            cg.return_value.get_token_price.return_value = {self.TOKEN: {"usd": 0.6}}
            assert get_aero_price_usd(key) == 0.6
        assert cg.call_args.kwargs == {kwarg: key}
        cg.return_value.get_token_price.assert_called_once_with(
            id="base", contract_addresses=self.TOKEN, vs_currencies="usd"
        )

    def test_no_credentials(self) -> None:
        """No key and no x402 means no price."""
        assert get_aero_price_usd(None) is None

    def test_request_error(self) -> None:
        """A raising client is a failed fetch."""
        with (
            patch(f"{MODULE}.CoinGeckoAPI", side_effect=Exception("boom")),
            patch(f"{MODULE}.is_pro_api_key", return_value=False),
        ):
            assert get_aero_price_usd("key") is None


class TestTapeRates:
    """Tests for the tape rate helpers."""

    def test_rate_uses_returned_timestamps(self) -> None:
        """The elapsed time comes from the observations, not the lookback asked for."""
        assert _rate_per_second((300, 200), (100, 100), 0, 1) == 2
        assert _rate_per_second((300, 100), (100, 100), 0, 1) is None

    def test_daily_rates_skip_refreshes(self) -> None:
        """Observations without elapsed time are skipped."""
        observations = [(0, 0), (86400, 86400), (86400, 86400), (2 * 86400, 2 * 86400)]
        assert _daily_rates(observations, 0, 1) == [1.0, 1.0]


class TestSharpeRatio:
    """Tests for sharpe_ratio_from_daily_returns."""

    def test_too_few_returns(self) -> None:
        """One return gives no ratio."""
        assert sharpe_ratio_from_daily_returns([0.01]) is None

    def test_constant_returns(self) -> None:
        """A series without variance gives 0."""
        assert sharpe_ratio_from_daily_returns([0.01, 0.01, 0.01]) == 0

    def test_capped(self) -> None:
        """Steady positive returns are capped at 10, negative excess at -10."""
        assert sharpe_ratio_from_daily_returns([0.01, 0.011, 0.012]) == 10
        assert sharpe_ratio_from_daily_returns([0.0, 0.0000001, 0.0]) == -10

    def test_value(self) -> None:
        """A noisy series is annualised."""
        ratio = sharpe_ratio_from_daily_returns([0.001, -0.002, 0.003, 0.0])
        assert -10 < ratio < 10


class TestDepthMetrics:
    """Tests for aero_depth_metrics."""

    def test_without_volume(self) -> None:
        """Unknown volume gives no depth."""
        assert aero_depth_metrics(1000.0, None) == (None, None)

    def test_values(self) -> None:
        """Depth follows the Sugar formula and the position is capped at a TVL share."""
        depth, max_size = aero_depth_metrics(100_000.0, 50_000.0, 0.02)
        assert depth == pytest.approx(
            float(velo.np.log1p(100_000) * velo.np.log1p(50_000))
        )
        assert max_size == 2000.0


class TestClLiquidity:
    """Tests for _cl_liquidity_per_token1."""

    def test_out_of_range(self) -> None:
        """A band away from the price earns nothing."""
        assert _cl_liquidity_per_token1(2**96, 10, 20) is None

    def test_in_range(self) -> None:
        """At price 1 a symmetric band of 2 ticks holds about 1e-4 token1 per unit of liquidity."""
        per_token1 = _cl_liquidity_per_token1(2**96, -1, 1)
        assert per_token1 == pytest.approx(1 / 1e-4, rel=1e-3)


def _v2_pool(**extra: Any) -> Dict[str, Any]:
    return {
        "kind": "stable",
        "decimals0": 6,
        "decimals1": 18,
        "reserves": (1_000_000 * 10**6, 1_000_000 * 10**18),
        "total_supply": 2_000_000 * WAD,
        # As many LP tokens as the reference position mints, so it would earn half.
        "gauge_total_supply": 1000 * WAD,
        # 1e-4 AERO per LP token over one day
        "gauge_observations": [(10**14 + 1, SECONDS_PER_DAY), (1, 0)],
        "gauge_daily_observations": [
            (0, 0),
            (10**14, SECONDS_PER_DAY),
            (2 * 10**14, 2 * SECONDS_PER_DAY),
        ],
        **extra,
    }


class TestV2Metrics:
    """Tests for aero_v2_pool_metrics."""

    def test_apr(self) -> None:
        """APR = reward per LP token x stake share x AERO price / LP price, annualised."""
        metrics = aero_v2_pool_metrics(_v2_pool(), aero_price=0.5)
        # Each LP token earns 1e-4 AERO a day, halved by the position's own stake.
        assert metrics["tvl"] == pytest.approx(2_000_000)
        assert metrics["apr"] == pytest.approx(0.25e-4 * 365 * 100, rel=1e-6)
        assert metrics["daily_returns"] == pytest.approx([0.25e-4, 0.25e-4])

    @pytest.mark.parametrize(
        "position_usd, share", [(1000.0, 0.5), (3000.0, 0.25), (0.0, 1.0)]
    )
    def test_position_dilutes_the_stake(self, position_usd: float, share: float) -> None:
        """The share is the gauge's stake over that stake plus the position's LP."""
        metrics = aero_v2_pool_metrics(_v2_pool(), 0.5, position_usd=position_usd)
        assert metrics["apr"] == pytest.approx(0.5e-4 * 365 * 100 * share, rel=1e-6)

    def test_empty_pool(self) -> None:
        """A pool without supply cannot be priced, so it has no APR rather than 0."""
        metrics = aero_v2_pool_metrics(_v2_pool(total_supply=0), aero_price=0.5)
        assert metrics["apr"] is None

    def test_no_gauge_tape(self) -> None:
        """A gauge tape with no observations gives no APR rather than 0."""
        metrics = aero_v2_pool_metrics(_v2_pool(gauge_observations=[]), 0.5)
        assert metrics["apr"] is None

    def test_no_emissions(self) -> None:
        """A tape that has not grown gives 0 APR."""
        metrics = aero_v2_pool_metrics(
            _v2_pool(gauge_observations=[(1, 10), (1, 0)]), aero_price=0.5
        )
        assert metrics["apr"] == 0.0

    def test_quiet_gauge_falls_back_to_the_hourly_ring(self) -> None:
        """A daily ring with one return is merged with the hourly ring and the live reading."""
        pool = _v2_pool(
            gauge_daily_observations=[(0, 0), (10**14, SECONDS_PER_DAY)],
            gauge_hourly_observations=[
                (10**14, SECONDS_PER_DAY),
                (3 * 10**14, 3 * SECONDS_PER_DAY),
            ],
        )
        metrics = aero_v2_pool_metrics(pool, aero_price=0.5)
        assert metrics["daily_returns"] == pytest.approx([0.25e-4, 0.25e-4])

    def test_gauge_with_one_observation_has_no_series(self) -> None:
        """Without two returns in either ring no Sharpe series exists, and it is logged."""
        pool = _v2_pool(
            id=_address(1),
            gauge_daily_observations=[(0, 0)],
            gauge_hourly_observations=[(0, 0), (10**14, SECONDS_PER_DAY)],
        )
        with patch(f"{MODULE}.logger") as logger:
            metrics = aero_v2_pool_metrics(pool, aero_price=0.5)
        assert metrics["daily_returns"] == pytest.approx([0.25e-4])
        assert "too few gauge observations" in logger.warning.call_args.args[0]


def _cl_pool(**extra: Any) -> Dict[str, Any]:
    return {
        "kind": "cl",
        "decimals0": 6,
        "decimals1": 6,
        "balance0": 500_000 * 10**6,
        "balance1": 500_000 * 10**6,
        "sqrt_price_x96": 2**96,
        "staked_liquidity": 0,
        # 10 AERO credited over a day
        "gauge_observations": [(0, 10 * WAD, SECONDS_PER_DAY), (0, 0, 0)],
        "gauge_daily_observations": [
            (0, 0, 0),
            (0, 10 * WAD, SECONDS_PER_DAY),
            (0, 30 * WAD, 2 * SECONDS_PER_DAY),
        ],
        **extra,
    }


class TestClMetrics:
    """Tests for aero_cl_pool_metrics."""

    BANDS = [{"tick_lower": -1, "tick_upper": 1, "allocation": 1.0}]

    def test_sole_staker_earns_all_emissions(self) -> None:
        """With no other stake the position takes every credited AERO."""
        metrics = aero_cl_pool_metrics(_cl_pool(), 1.0, self.BANDS, position_usd=1000.0)
        assert metrics["tvl"] == pytest.approx(1_000_000)
        assert metrics["apr"] == pytest.approx(10 * 365 / 1000 * 100)
        assert metrics["daily_returns"] == pytest.approx([0.01, 0.02])

    def test_stake_dilutes_the_pool(self) -> None:
        """The share is the position's liquidity over staked liquidity plus its own."""
        position_liquidity = 1000.0 * 10**6 * _cl_liquidity_per_token1(2**96, -1, 1)
        metrics = aero_cl_pool_metrics(
            _cl_pool(staked_liquidity=position_liquidity), 1.0, self.BANDS, 1000.0
        )
        assert metrics["apr"] == pytest.approx(10 * 365 / 1000 * 100 / 2)

    def test_out_of_range_bands(self) -> None:
        """Bands away from the price cannot be priced, so they give no APR rather than 0."""
        bands = [{"tick_lower": 5, "tick_upper": 10, "allocation": 1.0}]
        assert aero_cl_pool_metrics(_cl_pool(), 1.0, bands)["apr"] is None

    def test_no_emissions(self) -> None:
        """A tape that has not grown gives 0 APR."""
        pool = _cl_pool(gauge_observations=[(0, 5, 10), (0, 5, 0)])
        assert aero_cl_pool_metrics(pool, 1.0, self.BANDS)["apr"] == 0.0

    def test_quiet_gauge_falls_back_to_the_hourly_ring(self) -> None:
        """CL gauges merge their rings by the observation timestamp too."""
        pool = _cl_pool(
            gauge_daily_observations=[(0, 0, 0)],
            gauge_hourly_observations=[
                (0, 10 * WAD, SECONDS_PER_DAY),
                (0, 30 * WAD, 2 * SECONDS_PER_DAY),
            ],
        )
        metrics = aero_cl_pool_metrics(pool, 1.0, self.BANDS, position_usd=1000.0)
        assert metrics["daily_returns"] == pytest.approx([0.01, 0.02])


class TestDailyVolume:
    """Tests for aero_daily_volume_usd."""

    def test_no_tape(self) -> None:
        """No pool tape answer gives no volume."""
        assert aero_daily_volume_usd({"pool_observations": None}) is None

    def test_v2(self) -> None:
        """Each swap's input and output are averaged."""
        newer = (0, 0, 4 * 10**6, 2 * 10**18, 0, 0, 0, 0, 2 * SECONDS_PER_DAY, 0)
        older = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        pool = {
            "kind": "stable",
            "decimals0": 6,
            "decimals1": 18,
            "pool_observations": [newer, older],
        }
        assert aero_daily_volume_usd(pool) == pytest.approx(1.5)

    def test_cl(self) -> None:
        """CL volumes are read from their own slots."""
        newer = (0, SECONDS_PER_DAY, 0, 0, 0, 0, 2 * 10**6, 2 * 10**6, 0, 0, 0, 0, 0, 0)
        older = (0,) * 14
        pool = {
            "kind": "cl",
            "decimals0": 6,
            "decimals1": 6,
            "pool_observations": [newer, older],
        }
        assert aero_daily_volume_usd(pool) == pytest.approx(2.0)

    def test_no_elapsed_time(self) -> None:
        """A single observation returned twice gives no volume."""
        observation = (0,) * 10
        pool = {
            "kind": "stable",
            "decimals0": 6,
            "decimals1": 6,
            "pool_observations": [observation, observation],
        }
        assert aero_daily_volume_usd(pool) is None


def _state_answers() -> Dict[Any, Any]:
    return {
        "isActivated(address)": lambda target, args: (args[0] != _address(9),),
        "getReserves()": (10, 20, 0),
        "totalSupply()": (30,),
        "slot0()": (2**96, 0, 0, 0, 0, True),
        "stakedLiquidity()": (40,),
        "balanceOf(address)": (50,),
        "observe(address,uint48[])": ([(1, 2), (0, 1)],),
        "dailyObservations(address)": ([(0, 0)],),
    }


class TestReadAeroPoolState:
    """Tests for read_aero_pool_state."""

    def _pools(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": _address(1),
                "kind": "stable",
                "gauge": _address(8),
                "token0": USDC,
                "token1": EUSD,
            },
            {
                "id": _address(2),
                "kind": "cl",
                "gauge": _address(7),
                "token0": USDC,
                "token1": EUSD,
            },
            {
                "id": _address(3),
                "kind": "stable",
                "gauge": _address(9),
                "token0": USDC,
                "token1": EUSD,
            },
        ]

    def test_reads_activated_pools(self) -> None:
        """Pools whose gauge is not activated are dropped before the live reads."""
        multicall = _Multicall(_state_answers())
        requested = self._pools()
        with patch(f"{MODULE}.aero_multicall", multicall):
            pools = read_aero_pool_state(MagicMock(), BASE_CHAIN_ID, requested)

        assert [pool["id"] for pool in pools] == [_address(1), _address(2)]
        assert [pool["gauge_activated"] for pool in requested] == [True, True, False]
        v2, cl = pools
        assert v2["reserves"] == (10, 20)
        assert v2["total_supply"] == 30
        assert v2["gauge_total_supply"] == 30
        assert (_address(8), "totalSupply()", ()) in multicall.calls
        assert cl["staked_liquidity"] == 40
        assert (cl["balance0"], cl["sqrt_price_x96"]) == (50, 2**96)
        assert v2["gauge_hourly_observations"] == [(1, 2), (0, 1)]
        assert (
            BASE["gauge_tape"],
            "observe(address,uint48[])",
            (_address(8), AERO_HOURLY_LOOKBACKS),
        ) in multicall.calls
        targets = {(target, sig) for target, sig, _ in multicall.calls}
        assert (BASE["gauge_tape"], "observe(address,uint48[])") in targets
        assert (BASE["cl_pool_tape"], "observe(address,uint48[])") in targets
        assert not any(
            _address(3) == args[0]
            for _, sig, args in multicall.calls
            if sig == "observe(address,uint48[])"
        )

    def test_unreadable_activation(self) -> None:
        """A failed activation read drops the pool and records the activation as unknown."""
        answers = _state_answers()
        answers["isActivated(address)"] = None
        requested = self._pools()
        with patch(f"{MODULE}.aero_multicall", _Multicall(answers)):
            assert read_aero_pool_state(MagicMock(), BASE_CHAIN_ID, requested) == []
        assert [pool["gauge_activated"] for pool in requested] == [None, None, None]

    def test_unreadable_pools_dropped(self) -> None:
        """A pool whose state cannot be read is left out; a missing pool tape is kept as None."""
        answers = _state_answers()
        answers["totalSupply()"] = None
        answers[(BASE["cl_pool_tape"], "observe(address,uint48[])")] = None
        with patch(f"{MODULE}.aero_multicall", _Multicall(answers)):
            pools = read_aero_pool_state(MagicMock(), BASE_CHAIN_ID, self._pools())
        assert [pool["id"] for pool in pools] == [_address(2)]
        assert pools[0]["pool_observations"] is None

    def test_unreadable_hourly_ring_is_empty(self) -> None:
        """A failed hourly read leaves the pool with only its daily ring."""
        answers = _state_answers()
        answers["observe(address,uint48[])"] = lambda target, args: (
            None if args[1] == AERO_HOURLY_LOOKBACKS else ([(1, 2), (0, 1)],)
        )
        with patch(f"{MODULE}.aero_multicall", _Multicall(answers)):
            pools = read_aero_pool_state(MagicMock(), BASE_CHAIN_ID, self._pools())
        assert [pool["gauge_hourly_observations"] for pool in pools] == [[], []]

    def test_unreadable_cl_pool_dropped(self) -> None:
        """A CL pool whose staked liquidity cannot be read is left out."""
        answers = _state_answers()
        answers["stakedLiquidity()"] = None
        with patch(f"{MODULE}.aero_multicall", _Multicall(answers)):
            pools = read_aero_pool_state(MagicMock(), BASE_CHAIN_ID, self._pools())
        assert [pool["id"] for pool in pools] == [_address(1)]


class TestGetAeroPools:
    """Tests for get_aero_pools."""

    def _cache(self) -> Dict[str, Any]:
        return {
            "counts": {},
            "pools": {
                _address(1): ["stable", USDC, EUSD, _address(8)],
                _address(2): ["volatile", USDC, WETH, _address(7)],
                _address(3): ["cl", USDC, EUSD, _address(6)],
            },
            "decimals": {},
        }

    def test_not_connected(self) -> None:
        """An unreachable RPC is an error."""
        with patch(f"{MODULE}.get_web3_connection") as conn:
            conn.return_value.is_connected.return_value = False
            assert "error" in get_aero_pools(BASE_CHAIN_ID, [], WHITELIST)

    def test_whitelisted_pools_read(self) -> None:
        """Only whitelisted pools with readable decimals reach the live reads."""
        read = MagicMock(side_effect=lambda w3, chain_id, pools: pools)
        decimals = {USDC: 6, EUSD: 18}
        with (
            patch(f"{MODULE}.get_web3_connection"),
            patch(f"{MODULE}.discover_aero_pools", return_value=self._cache()),
            patch(f"{MODULE}.read_aero_pool_state", read),
            patch(
                f"{MODULE}.get_token_decimals",
                side_effect=lambda w3, tokens, cache: decimals,
            ),
        ):
            pools = get_aero_pools(BASE_CHAIN_ID, [_address(3)], WHITELIST, {})

        assert [pool["id"] for pool in pools] == [_address(1)]
        pool = pools[0]
        assert pool["pool_factory"] == BASE["stable_pool_factory"]
        assert (pool["decimals0"], pool["decimals1"]) == (6, 18)
        assert [token["symbol"] for token in pool["inputTokens"]] == ["USDC", "eUSD"]

    def test_unknown_decimals_dropped(self) -> None:
        """A pool whose token decimals cannot be read is not priced."""
        with (
            patch(f"{MODULE}.get_web3_connection"),
            patch(f"{MODULE}.discover_aero_pools", return_value=self._cache()),
            patch(
                f"{MODULE}.read_aero_pool_state", side_effect=lambda w3, c, pools: pools
            ),
            patch(f"{MODULE}.get_token_decimals", return_value={USDC: 6}),
        ):
            assert get_aero_pools(BASE_CHAIN_ID, [], WHITELIST, {}) == []

    def test_nothing_whitelisted(self) -> None:
        """No whitelisted pool means no live reads."""
        with (
            patch(f"{MODULE}.get_web3_connection"),
            patch(f"{MODULE}.discover_aero_pools", return_value=self._cache()),
            patch(f"{MODULE}.read_aero_pool_state") as read,
        ):
            assert get_aero_pools(BASE_CHAIN_ID, [], {"base": {"0xdead": "X"}}) == []
        read.assert_not_called()


def _live_pool(kind: str = "stable", **extra: Any) -> Dict[str, Any]:
    return {
        "id": _address(1),
        "chain": "base",
        "kind": kind,
        "pool_factory": BASE["stable_pool_factory"],
        "token0": USDC,
        "token1": EUSD,
        "gauge": _address(8),
        "inputTokens": [{"id": USDC, "symbol": "USDC"}, {"id": EUSD, "symbol": "eUSD"}],
        "pool_observations": None,
        **extra,
    }


class TestGetOpportunitiesForAero:
    """Tests for get_opportunities_for_aero."""

    def _run(
        self,
        pools: Any,
        price: Optional[float] = 0.7,
        coingecko_prices: Optional[Dict[str, float]] = None,
        cl_metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run the search; given CoinGecko prices, the real IL score fetches the histories."""
        metrics = {"apr": 12.0, "tvl": 50_000.0, "daily_returns": [0.001, 0.002]}
        if coingecko_prices is None:
            il_score = patch(
                f"{MODULE}.calculate_velodrome_il_risk_score_multi", return_value=0.01
            )
        else:
            client = MagicMock()
            client.get_coin_market_chart_range_by_id.side_effect = (
                lambda id, **_: {  # pylint: disable=redefined-builtin
                    "prices": [[0, 0.9], [1, coingecko_prices[id]]]
                }
            )
            il_score = patch(f"{MODULE}.CoinGeckoAPI", return_value=client)
        with (
            patch(f"{MODULE}.get_aero_pools", return_value=pools),
            patch(f"{MODULE}.get_aero_price_usd", return_value=price),
            patch(f"{MODULE}.aero_v2_pool_metrics", return_value=metrics),
            patch(f"{MODULE}.aero_cl_pool_metrics", return_value=cl_metrics or metrics),
            patch(
                f"{MODULE}.calculate_tick_lower_and_upper_velodrome",
                return_value=[{"tick_lower": -1, "tick_upper": 1, "allocation": 1.0}],
            ),
            patch(f"{MODULE}.is_pro_api_key", return_value=False),
            patch(f"{MODULE}.time.sleep"),
            il_score,
        ):
            return get_opportunities_for_aero(
                [],
                "key",
                BASE_CHAIN_ID,
                coin_id_mapping={
                    "base": {
                        "usdc": "usd-coin",
                        "eusd": "electronic-usd",
                        "msusd": "metronome-synth-usd",
                    }
                },
                **kwargs,
            )

    def test_errors(self) -> None:
        """Discovery errors, no pools, no AERO price and no pool past the filter are errors."""
        assert self._run({"error": "rpc"}) == {"error": "rpc"}
        assert "error" in self._run([])
        assert "AERO" in self._run([_live_pool()], price=None)["error"]
        assert "error" in self._run([_live_pool()], min_tvl_threshold=10**9)

    def test_opportunities(self) -> None:
        """Opportunities carry the protocol, gauge and factory the skill acts on."""
        v2, cl = self._run([_live_pool(), _live_pool("cl", id=_address(2))])

        assert v2["protocol"] == AERO_PROTOCOL
        assert v2["dex_type"] == "aerodrome"
        assert v2["gauge_address"] == _address(8)
        assert v2["pool_factory"] == BASE["stable_pool_factory"]
        assert (v2["is_stable"], v2["is_cl_pool"]) == (True, False)
        assert v2["apr"] == v2["advertised_apr"] == 12.0
        assert v2["sharpe_ratio"] is not None
        assert v2["il_risk_score"] == 0.01
        assert v2["depth_score"] is None
        assert "tick_bands" not in v2
        assert cl["is_cl_pool"] is True
        assert cl["tick_bands"][0]["tick_lower"] == -1

    def test_unpriced_pool_is_not_offered(self) -> None:
        """A pool whose emissions cannot be priced has no APR to rank it by."""
        unpriced = {"apr": None, "tvl": 50_000.0, "daily_returns": []}
        (opportunity,) = self._run(
            [_live_pool(), _live_pool("cl", id=_address(2))], cl_metrics=unpriced
        )
        assert opportunity["pool_address"] == _address(1)

    def _msusd_and_eusd_pools(self, msusd_price: float, **kwargs: Any) -> List[str]:
        msusd_pool = _live_pool(
            id=_address(3),
            token1=MSUSD,
            inputTokens=[
                {"id": USDC, "symbol": "USDC"},
                {"id": MSUSD, "symbol": "msUSD"},
            ],
        )
        opportunities = self._run(
            [_live_pool(), msusd_pool],
            coingecko_prices={
                "usd-coin": 1.0,
                "electronic-usd": 0.999,
                "metronome-synth-usd": msusd_price,
            },
            price_cache={},
            **kwargs,
        )
        return [opportunity["pool_address"] for opportunity in opportunities]

    def test_depegged_token_drops_its_pools(self) -> None:
        """A pool holding a token priced more than 2% below $1 is not offered."""
        assert self._msusd_and_eusd_pools(0.88) == [_address(1)]

    def test_token_near_peg_is_kept(self) -> None:
        """A token within 2% of $1 keeps its pool."""
        assert self._msusd_and_eusd_pools(0.995) == [_address(1), _address(3)]

    @pytest.mark.parametrize(
        "msusd_price, max_stable_depeg, offered",
        [(0.88, 0.15, [_address(1), _address(3)]), (0.995, 0.001, [_address(1)])],
    )
    def test_threshold_read_from_parameter(
        self, msusd_price: float, max_stable_depeg: float, offered: List[str]
    ) -> None:
        """The depeg threshold comes from the max_stable_depeg strategy parameter."""
        pools = self._msusd_and_eusd_pools(
            msusd_price, max_stable_depeg=max_stable_depeg
        )
        assert pools == offered

    def test_token_without_price_is_kept(self) -> None:
        """A token with no fetched price keeps the $1 assumption."""
        (opportunity,) = self._run([_live_pool()], price_cache={})
        assert opportunity["pool_address"] == _address(1)

    def test_optimism_dex_type(self) -> None:
        """Optimism opportunities keep the velodrome dex type."""
        with (
            patch(
                f"{MODULE}.get_aero_pools", return_value=[_live_pool(chain="optimism")]
            ),
            patch(f"{MODULE}.get_aero_price_usd", return_value=0.7),
            patch(
                f"{MODULE}.aero_v2_pool_metrics",
                return_value={"apr": 1.0, "tvl": 20_000.0, "daily_returns": []},
            ),
        ):
            (opportunity,) = get_opportunities_for_aero([], None, OPTIMISM_CHAIN_ID)
        assert opportunity["dex_type"] == "velodrome"
        assert opportunity["il_risk_score"] is None


class TestCalculateAeroMetrics:
    """Tests for calculate_aero_metrics and its routing."""

    def _position(self, **extra: Any) -> Dict[str, Any]:
        return {
            "protocol": AERO_PROTOCOL,
            "chain": "base",
            "pool_address": _address(1).lower(),
            "token0": USDC,
            "token1": EUSD,
            "token0_symbol": "USDC",
            "token1_symbol": "eUSD",
            "gauge_address": _address(8),
            "is_stable": True,
            **extra,
        }

    def _metrics(
        self,
        position: Dict[str, Any],
        readable: bool = True,
        decimals: Any = None,
        tick_bands: Any = None,
        activated: Optional[bool] = True,
        live_state: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Read a position's metrics; given its pool's live state, the pool metrics run for real."""

        def read_state(w3: Any, chain_id: int, pools: Any) -> Any:
            for pool in pools:
                pool["gauge_activated"] = activated
                pool.update(live_state or {})
            return pools if readable and activated else []

        def pool_metrics(real: Any) -> Dict[str, Any]:
            if live_state is not None:
                return {"wraps": real}
            return {
                "return_value": {"apr": 1.0, "tvl": 100.0, "daily_returns": [0.1, 0.2]}
            }

        with (
            patch(f"{MODULE}.get_web3_connection"),
            patch(
                f"{MODULE}.get_token_decimals",
                return_value=decimals if decimals is not None else {USDC: 6, EUSD: 18},
            ),
            patch(f"{MODULE}.read_aero_pool_state", side_effect=read_state),
            patch(f"{MODULE}.get_aero_price_usd", return_value=0.7),
            patch(
                f"{MODULE}.aero_v2_pool_metrics", **pool_metrics(aero_v2_pool_metrics)
            ) as v2,
            patch(
                f"{MODULE}.aero_cl_pool_metrics", **pool_metrics(aero_cl_pool_metrics)
            ) as cl,
            patch(
                f"{MODULE}.calculate_tick_lower_and_upper_velodrome",
                return_value=tick_bands,
            ),
            patch(
                f"{MODULE}.calculate_velodrome_il_risk_score_multi", return_value=0.02
            ),
        ):
            result = calculate_aero_metrics(
                position,
                "key",
                {"base": {"usdc": "usd-coin", "eusd": "electronic-usd"}},
            )
        return result, v2, cl

    def test_v2(self) -> None:
        """A held stable position is valued from its pool's tapes."""
        result, v2, _ = self._metrics(self._position())
        assert result["apr"] == 1.0
        assert result["il_risk_score"] == 0.02
        assert result["sharpe_ratio"] == 10
        assert result["depth_score"] is None
        pool = v2.call_args.args[0]
        assert pool["kind"] == "stable"
        assert pool["gauge"] == _address(8)

    def test_cl(self) -> None:
        """A held CL position is valued from its gauge's credited emissions."""
        bands = [{"tick_lower": -10, "tick_upper": 10, "allocation": 1.0}]
        result, _, cl = self._metrics(self._position(is_cl_pool=True), tick_bands=bands)
        assert result["apr"] == 1.0
        assert cl.call_args.args[2] == bands

    def test_cl_without_tick_bands_has_no_apr(self) -> None:
        """Unpriced tick bands give no live APR rather than an APR of 0."""
        result, _, cl = self._metrics(
            self._position(is_cl_pool=True), live_state=_cl_pool()
        )
        assert cl.call_args.args[2] == []
        assert "apr" not in result
        assert result["il_risk_score"] == 0.02

    def test_cl_with_out_of_range_tick_bands_has_no_apr(self) -> None:
        """Tick bands the price is outside of give no live APR rather than an APR of 0."""
        bands = [{"tick_lower": 5, "tick_upper": 10, "allocation": 1.0}]
        result, _, cl = self._metrics(
            self._position(is_cl_pool=True), tick_bands=bands, live_state=_cl_pool()
        )
        assert cl.call_args.args[2] == bands
        assert "apr" not in result
        assert result["il_risk_score"] == 0.02

    def test_deactivated_gauge_earns_nothing(self) -> None:
        """A held pool whose gauge is deactivated earns no emissions: its live APR is 0."""
        result, v2, _ = self._metrics(self._position(), activated=False)
        assert result == {"apr": 0.0}
        v2.assert_not_called()

    def test_unreadable(self) -> None:
        """Missing decimals, state, activation or position data give no metrics."""
        assert self._metrics(self._position(), decimals={USDC: 6})[0] is None
        assert self._metrics(self._position(), readable=False)[0] is None
        assert self._metrics(self._position(), activated=None)[0] is None
        assert self._metrics(self._position(gauge_address=None))[0] is None
        assert self._metrics(self._position(chain="mode"))[0] is None

    def test_calculate_metrics_routes_aero(self) -> None:
        """calculate_metrics hands Aero positions to calculate_aero_metrics."""
        with patch(f"{MODULE}.calculate_aero_metrics", return_value={"x": 1}) as aero:
            assert calculate_metrics(self._position(), "key", {}) == {"x": 1}
        aero.assert_called_once()


class TestRunRoutesAero:
    """run() searches Aero on Base and Optimism and keeps Sugar on Mode."""

    def test_base_uses_aero(self) -> None:
        """Base goes to get_opportunities_for_aero, not to Sugar."""
        with (
            patch(f"{MODULE}.get_web3_connection"),
            patch(
                f"{MODULE}.get_opportunities_for_aero", return_value=[{"pool": 1}]
            ) as aero,
            patch(f"{MODULE}.get_opportunities_for_velodrome") as sugar,
        ):
            result = run(
                chains=["base"],
                current_positions=[],
                whitelisted_assets={},
                coingecko_api_key="key",
            )
        assert result["result"] == [{"pool": 1}]
        assert aero.call_args.args[2] == BASE_CHAIN_ID
        sugar.assert_not_called()

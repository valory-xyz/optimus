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

"""Tests for the Aero additions to LiquidityTraderBaseBehaviour."""

from unittest.mock import MagicMock

from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    AERO_TOKEN_ADDRESS_BASE,
    REWARD_TOKEN_ADDRESSES,
    XAERO_ADDRESSES,
)
from packages.valory.skills.liquidity_trader_abci.tests.behaviours.test_base import (
    _exhaust,
    _make_behaviour,
    _make_gen,
)

XAERO = XAERO_ADDRESSES["optimism"]


def test_aero_reward_tokens_are_excluded_from_investment() -> None:
    """Aero's AERO and xAERO sit with the other reward tokens."""
    assert AERO_TOKEN_ADDRESS_BASE in REWARD_TOKEN_ADDRESSES["base"]
    assert XAERO in REWARD_TOKEN_ADDRESSES["optimism"]


class TestXAeroPrice:
    """xAERO has no CoinGecko listing and is priced as Base AERO."""

    def test_xaero_priced_as_base_aero(self) -> None:
        """The price request goes to Base for Aero's AERO contract."""
        b = _make_behaviour()
        b.coingecko.chain_to_platform_id_mapping["base"] = "base"
        b._get_cached_price = _make_gen(None)  # type: ignore[method-assign]
        b._cache_price = _make_gen(None)  # type: ignore[method-assign]
        b.coingecko.request = MagicMock(  # type: ignore[method-assign]
            return_value=(True, {AERO_TOKEN_ADDRESS_BASE.lower(): {"usd": 0.678}})
        )

        price = _exhaust(b._fetch_token_price(XAERO.lower(), "optimism"))

        assert price == 0.678
        endpoint = b.coingecko.request.call_args.kwargs["endpoint"]
        assert AERO_TOKEN_ADDRESS_BASE in endpoint
        assert "simple/token_price/base" in endpoint


class TestXAeroNeverSwapped:
    """xAERO is held, never swapped to USDC."""

    def test_swap_to_usdc_skips_xaero(self) -> None:
        """No swap action is built for xAERO."""
        b = _make_behaviour()
        assert b._build_swap_to_usdc_action("optimism", XAERO.lower(), "xAERO") is None


class TestAeroPositionActions:
    """Exit and unstake actions keep the position's protocol."""

    def _position(self, **extra):
        return {
            "dex_type": "aerodrome",
            "chain": "optimism",
            "pool_address": "0xPool",
            "is_cl_pool": False,
            "protocol": "aero",
            "gauge_address": "0xGauge",
            **extra,
        }

    def test_exit_action_carries_protocol(self) -> None:
        """The exit goes through the Aero contracts the position was entered with."""
        b = _make_behaviour()
        assert b._build_exit_pool_action_base(self._position())["protocol"] == "aero"

    def test_unstake_action_carries_protocol_and_gauge(self) -> None:
        """The unstake action targets the stored gauge with the Aero penalty check."""
        b = _make_behaviour()
        action = b._build_unstake_lp_tokens_action(self._position())
        assert action["protocol"] == "aero"
        assert action["gauge_address"] == "0xGauge"

    def test_staking_metadata_ignores_stored_gauge(self) -> None:
        """An Aero position stores its gauge even when unstaked."""
        b = _make_behaviour()
        assert b._has_staking_metadata(self._position()) is False
        assert b._has_staking_metadata(self._position(staked=True)) is True

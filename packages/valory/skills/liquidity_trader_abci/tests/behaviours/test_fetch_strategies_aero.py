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

"""Tests for how FetchStrategiesBehaviour values Aero positions."""

from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
    TRANSFER_EVENT_SIGNATURE,
    ZERO_ADDRESS_PADDED,
)
from packages.valory.skills.liquidity_trader_abci.tests.behaviours.test_fetch_strategies import (
    _drive,
    _gen_return,
    _mk,
)

GAUGE = "0xGauge"
AERO_POSITION = {"protocol": "aero", "gauge_address": GAUGE, "pool_address": "0xP"}


class TestRewardToken:
    """The reward token follows the protocol a position was entered with."""

    def _behaviour(self) -> Any:
        b = _mk()
        b.params.velo_token_contract_addresses = {"optimism": "0xVELO"}
        b.params.aero_reward_token_contract_addresses = {"optimism": "0xXAERO"}
        return b

    def test_address(self) -> None:
        """Aero positions earn the Aero reward token; legacy ones VELO."""
        b = self._behaviour()
        assert b._get_velo_token_address("optimism", AERO_POSITION) == "0xXAERO"
        assert b._get_velo_token_address("optimism", {}) == "0xVELO"
        assert b._get_velo_token_address("optimism") == "0xVELO"

    @pytest.mark.parametrize(
        "chain, position, symbol",
        [
            ("base", AERO_POSITION, "AERO"),
            ("optimism", AERO_POSITION, "xAERO"),
            ("optimism", {}, "VELO"),
            ("base", None, "AERO"),
        ],
    )
    def test_symbol(self, chain, position, symbol) -> None:
        """Aero pays AERO on Base and xAERO on Optimism."""
        assert (
            self._behaviour()._get_velo_family_reward_symbol(chain, position) == symbol
        )

    def test_aero_reward_priced_by_contract(self) -> None:
        """Aero rewards are priced as Base AERO by contract, not by coin id."""
        b = self._behaviour()
        calls: List[Any] = []

        def _price(token_address: str, chain: str) -> Any:
            calls.append((token_address, chain))
            yield
            return 0.678

        b._fetch_token_price = _price
        b._fetch_coin_price = MagicMock()
        b.params.aero_reward_token_contract_addresses["base"] = "0xAERO"
        assert (
            _drive(b._fetch_velo_family_reward_price("optimism", AERO_POSITION))
            == 0.678
        )
        assert calls == [("0xAERO", "base")]
        b._fetch_coin_price.assert_not_called()

    def test_legacy_reward_priced_by_coin_id(self) -> None:
        """Legacy rewards keep the CoinGecko coin id lookup."""
        b = self._behaviour()
        b.get_coin_id_from_symbol = MagicMock(return_value="velodrome-finance")
        b._fetch_coin_price = _gen_return(0.05)
        assert _drive(b._fetch_velo_family_reward_price("optimism", {})) == 0.05
        b.get_coin_id_from_symbol.assert_called_once_with("VELO", "optimism")


class TestStoredGauge:
    """Aero positions resolve their gauge without the legacy voter."""

    def test_resolve_returns_stored_gauge(self) -> None:
        """A stored gauge is returned as resolved."""
        b = _mk()
        b.contract_interact = MagicMock()
        result = _drive(b._resolve_velodrome_gauge_address("0xP", "base", GAUGE))
        assert result == (True, GAUGE)
        b.contract_interact.assert_not_called()

    def test_aero_cl_pending_rewards_read_once(self) -> None:
        """Aero CL pending rewards are one account-wide read on the stored gauge."""
        b = _mk()
        calls: List[Dict[str, Any]] = []

        def _pending(*args: Any, **kwargs: Any) -> Any:
            calls.append(kwargs)
            yield
            return 3 * 10**18

        pool = MagicMock()
        pool.get_gauge_address = _gen_return(GAUGE)
        pool.get_cl_pending_rewards = _pending
        b.pools = {"velodrome": pool}
        position = {
            **AERO_POSITION,
            "is_cl_pool": True,
            "positions": [{"token_id": 1}, {"token_id": 2}],
        }

        rewards = _drive(b._get_velodrome_pending_rewards(position, "base", "0xSafe"))

        assert rewards == Decimal(3)
        assert len(calls) == 1
        assert calls[0]["protocol"] == "aero"
        assert calls[0]["gauge_address"] == GAUGE


def _transfer(token: str, sender: str, recipient: str) -> Dict[str, Any]:
    """Return an ERC20 Transfer log."""
    return {
        "address": token,
        "topics": [TRANSFER_EVENT_SIGNATURE, sender, recipient],
        "data": "0x01",
    }


class TestMetarouterEntryValidation:
    """The v2 pool address check reads Metarouter entries correctly (D12)."""

    def test_pool_found_from_lp_mint(self) -> None:
        """The pool is the LP mint, not one of the token pulls through the router."""
        b = _mk()
        safe = "0x" + "00" * 12 + "aa" * 20
        router = "0x" + "00" * 12 + "bb" * 20
        pool_topic = "0x" + "00" * 12 + "cc" * 20
        receipt = {
            "logs": [
                _transfer("0xTokenA", safe, router),
                _transfer("0xTokenB", safe, router),
                _transfer("0xTokenA", router, pool_topic),
                _transfer("0xTokenB", router, pool_topic),
                _transfer("0xPOOL", ZERO_ADDRESS_PADDED, safe),
            ]
        }
        b.get_transaction_receipt = _gen_return(receipt)
        position = {
            "enter_tx_hash": "0xhash",
            "chain": "base",
            "pool_address": "0xpool",
        }

        assert _drive(b._validate_velodrome_v2_pool_address(position)) is True
        assert position["pool_address"] == "0xpool"
        assert position["is_updated"] is True

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

"""Test the pools/balancer.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import pytest
from enum import Enum
from unittest.mock import MagicMock

from packages.valory.skills.liquidity_trader_abci.pools.balancer import (
    ZERO_ADDRESS,
    BalancerPoolBehaviour,
    ExitKind,
    JoinKind,
    PoolType,
)


def test_import() -> None:
    """Test that the balancer module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.pools.balancer  # noqa


def test_zero_address() -> None:
    """Test ZERO_ADDRESS constant."""
    assert ZERO_ADDRESS == "0x0000000000000000000000000000000000000000"


class TestPoolType:
    """Test PoolType enum."""

    def test_is_enum(self) -> None:
        """Test PoolType is an Enum."""
        assert issubclass(PoolType, Enum)

    def test_values(self) -> None:
        """Test PoolType values."""
        assert PoolType.WEIGHTED.value == "Weighted"
        assert PoolType.COMPOSABLE_STABLE.value == "ComposableStable"
        assert PoolType.LIQUIDITY_BOOTSTRAPING.value == "LiquidityBootstrapping"
        assert PoolType.META_STABLE.value == "MetaStable"
        assert PoolType.STABLE.value == "Stable"
        assert PoolType.INVESTMENT.value == "Investment"


class TestJoinKind:
    """Test JoinKind classes."""

    def test_weighted_pool(self) -> None:
        """Test WeightedPool join kinds."""
        assert JoinKind.WeightedPool.INIT.value == 0
        assert JoinKind.WeightedPool.EXACT_TOKENS_IN_FOR_BPT_OUT.value == 1
        assert JoinKind.WeightedPool.TOKEN_IN_FOR_EXACT_BPT_OUT.value == 2
        assert JoinKind.WeightedPool.ALL_TOKENS_IN_FOR_EXACT_BPT_OUT.value == 3

    def test_stable_and_meta_stable_pool(self) -> None:
        """Test StableAndMetaStablePool join kinds."""
        assert JoinKind.StableAndMetaStablePool.INIT.value == 0
        assert JoinKind.StableAndMetaStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT.value == 1
        assert JoinKind.StableAndMetaStablePool.TOKEN_IN_FOR_EXACT_BPT_OUT.value == 2

    def test_composable_stable_pool(self) -> None:
        """Test ComposableStablePool join kinds."""
        assert JoinKind.ComposableStablePool.INIT.value == 0
        assert JoinKind.ComposableStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT.value == 1
        assert JoinKind.ComposableStablePool.TOKEN_IN_FOR_EXACT_BPT_OUT.value == 2
        assert JoinKind.ComposableStablePool.ALL_TOKENS_IN_FOR_EXACT_BPT_OUT.value == 3


class TestExitKind:
    """Test ExitKind classes."""

    def test_weighted_pool(self) -> None:
        """Test WeightedPool exit kinds."""
        assert ExitKind.WeightedPool.EXACT_BPT_IN_FOR_ONE_TOKEN_OUT.value == 0
        assert ExitKind.WeightedPool.EXACT_BPT_IN_FOR_TOKENS_OUT.value == 1
        assert ExitKind.WeightedPool.BPT_IN_FOR_EXACT_TOKENS_OUT.value == 2
        assert ExitKind.WeightedPool.MANAGEMENT_FEE_TOKENS_OUT.value == 3

    def test_stable_and_meta_stable_pool(self) -> None:
        """Test StableAndMetaStablePool exit kinds."""
        assert ExitKind.StableAndMetaStablePool.EXACT_BPT_IN_FOR_ONE_TOKEN_OUT.value == 0
        assert ExitKind.StableAndMetaStablePool.EXACT_BPT_IN_FOR_TOKENS_OUT.value == 1
        assert ExitKind.StableAndMetaStablePool.BPT_IN_FOR_EXACT_TOKENS_OUT.value == 2

    def test_composable_stable_pool(self) -> None:
        """Test ComposableStablePool exit kinds."""
        assert ExitKind.ComposableStablePool.EXACT_BPT_IN_FOR_ONE_TOKEN_OUT.value == 0
        assert ExitKind.ComposableStablePool.BPT_IN_FOR_EXACT_TOKENS_OUT.value == 1
        assert ExitKind.ComposableStablePool.EXACT_BPT_IN_FOR_ALL_TOKENS_OUT.value == 2


class TestBalancerPoolBehaviourAdjustAmounts:
    """Test BalancerPoolBehaviour.adjust_amounts method."""

    def _make_behaviour(self) -> BalancerPoolBehaviour:
        """Create a BalancerPoolBehaviour without __init__."""
        obj = object.__new__(BalancerPoolBehaviour)
        obj.__dict__["_context"] = MagicMock()
        return obj

    def test_adjust_amounts_basic(self) -> None:
        """Test adjust_amounts with basic input."""
        obj = self._make_behaviour()
        result = obj.adjust_amounts(
            assets=["0xA", "0xB"],
            assets_new=["0xa", "0xb"],
            max_amounts_in=[100, 200],
        )
        assert result == [100, 200]

    def test_adjust_amounts_reordered(self) -> None:
        """Test adjust_amounts reorders amounts correctly."""
        obj = self._make_behaviour()
        result = obj.adjust_amounts(
            assets=["0xA", "0xB"],
            assets_new=["0xb", "0xa"],
            max_amounts_in=[100, 200],
        )
        assert result == [200, 100]

    def test_adjust_amounts_new_asset(self) -> None:
        """Test adjust_amounts with unknown asset defaults to 0."""
        obj = self._make_behaviour()
        result = obj.adjust_amounts(
            assets=["0xA"],
            assets_new=["0xa", "0xc"],
            max_amounts_in=[100],
        )
        assert result == [100, 0]

    def test_adjust_amounts_invalid_assets_type(self) -> None:
        """Test adjust_amounts raises ValueError for invalid asset types."""
        obj = self._make_behaviour()
        with pytest.raises(ValueError, match="All assets must be strings or bytes"):
            obj.adjust_amounts(
                assets=[123],
                assets_new=["0xa"],
                max_amounts_in=[100],
            )

    def test_adjust_amounts_length_mismatch(self) -> None:
        """Test adjust_amounts raises ValueError for mismatched lengths."""
        obj = self._make_behaviour()
        with pytest.raises(ValueError, match="Length of assets and max_amounts_in must match"):
            obj.adjust_amounts(
                assets=["0xA", "0xB"],
                assets_new=["0xa"],
                max_amounts_in=[100],
            )

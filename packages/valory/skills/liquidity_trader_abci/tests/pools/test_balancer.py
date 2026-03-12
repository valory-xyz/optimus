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

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.skills.liquidity_trader_abci.pools.balancer import (
    BalancerPoolBehaviour,
    ExitKind,
    JoinKind,
    PoolType,
    ZERO_ADDRESS,
)


def test_import() -> None:
    """Test that the balancer module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.pools.balancer  # noqa


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
        with pytest.raises(
            ValueError, match="Length of assets and max_amounts_in must match"
        ):
            obj.adjust_amounts(
                assets=["0xA", "0xB"],
                assets_new=["0xa"],
                max_amounts_in=[100],
            )


class TestBalancerPoolBehaviourInit:
    """Test BalancerPoolBehaviour __init__."""

    def test_init_calls_super(self) -> None:
        """Test that __init__ calls super().__init__."""
        with patch.object(
            BalancerPoolBehaviour.__bases__[0], "__init__", return_value=None
        ):
            obj = BalancerPoolBehaviour.__new__(BalancerPoolBehaviour)
            BalancerPoolBehaviour.__init__(obj, some_kwarg="test")


def _make_behaviour():
    """Create a BalancerPoolBehaviour without __init__."""
    obj = object.__new__(BalancerPoolBehaviour)
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


class TestDetermineJoinKind:
    """Tests for _determine_join_kind."""

    def test_weighted(self) -> None:
        """Test join kind for Weighted pool."""
        obj = _make_behaviour()
        result = obj._determine_join_kind(PoolType.WEIGHTED.value)
        assert result == JoinKind.WeightedPool.EXACT_TOKENS_IN_FOR_BPT_OUT.value

    def test_liquidity_bootstrapping(self) -> None:
        """Test join kind for LiquidityBootstrapping pool."""
        obj = _make_behaviour()
        result = obj._determine_join_kind(PoolType.LIQUIDITY_BOOTSTRAPING.value)
        assert result == JoinKind.WeightedPool.EXACT_TOKENS_IN_FOR_BPT_OUT.value

    def test_investment(self) -> None:
        """Test join kind for Investment pool."""
        obj = _make_behaviour()
        result = obj._determine_join_kind(PoolType.INVESTMENT.value)
        assert result == JoinKind.WeightedPool.EXACT_TOKENS_IN_FOR_BPT_OUT.value

    def test_stable(self) -> None:
        """Test join kind for Stable pool."""
        obj = _make_behaviour()
        result = obj._determine_join_kind(PoolType.STABLE.value)
        assert (
            result == JoinKind.StableAndMetaStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT.value
        )

    def test_meta_stable(self) -> None:
        """Test join kind for MetaStable pool."""
        obj = _make_behaviour()
        result = obj._determine_join_kind(PoolType.META_STABLE.value)
        assert (
            result == JoinKind.StableAndMetaStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT.value
        )

    def test_composable_stable(self) -> None:
        """Test join kind for ComposableStable pool."""
        obj = _make_behaviour()
        result = obj._determine_join_kind(PoolType.COMPOSABLE_STABLE.value)
        assert result == JoinKind.ComposableStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT.value

    def test_unknown(self) -> None:
        """Test join kind for unknown pool type."""
        obj = _make_behaviour()
        result = obj._determine_join_kind("UnknownType")
        assert result is None


class TestDetermineExitKind:
    """Tests for _determine_exit_kind."""

    def test_weighted(self) -> None:
        """Test exit kind for Weighted pool."""
        obj = _make_behaviour()
        result = obj._determine_exit_kind(PoolType.WEIGHTED.value)
        assert result == ExitKind.WeightedPool.EXACT_BPT_IN_FOR_TOKENS_OUT.value

    def test_liquidity_bootstrapping(self) -> None:
        """Test exit kind for LiquidityBootstrapping pool."""
        obj = _make_behaviour()
        result = obj._determine_exit_kind(PoolType.LIQUIDITY_BOOTSTRAPING.value)
        assert result == ExitKind.WeightedPool.EXACT_BPT_IN_FOR_TOKENS_OUT.value

    def test_investment(self) -> None:
        """Test exit kind for Investment pool."""
        obj = _make_behaviour()
        result = obj._determine_exit_kind(PoolType.INVESTMENT.value)
        assert result == ExitKind.WeightedPool.EXACT_BPT_IN_FOR_TOKENS_OUT.value

    def test_stable(self) -> None:
        """Test exit kind for Stable pool."""
        obj = _make_behaviour()
        result = obj._determine_exit_kind(PoolType.STABLE.value)
        assert (
            result == ExitKind.StableAndMetaStablePool.EXACT_BPT_IN_FOR_TOKENS_OUT.value
        )

    def test_meta_stable(self) -> None:
        """Test exit kind for MetaStable pool."""
        obj = _make_behaviour()
        result = obj._determine_exit_kind(PoolType.META_STABLE.value)
        assert (
            result == ExitKind.StableAndMetaStablePool.EXACT_BPT_IN_FOR_TOKENS_OUT.value
        )

    def test_composable_stable(self) -> None:
        """Test exit kind for ComposableStable pool."""
        obj = _make_behaviour()
        result = obj._determine_exit_kind(PoolType.COMPOSABLE_STABLE.value)
        assert (
            result
            == ExitKind.ComposableStablePool.EXACT_BPT_IN_FOR_ALL_TOKENS_OUT.value
        )

    def test_unknown(self) -> None:
        """Test exit kind for unknown pool type."""
        obj = _make_behaviour()
        result = obj._determine_exit_kind("UnknownType")
        assert result is None


class TestGetPoolId:
    """Tests for _get_pool_id."""

    def test_success(self) -> None:
        """Test successful pool id retrieval."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return "0xpoolid123"

        obj.contract_interact = fake_contract_interact
        gen = obj._get_pool_id("0xpool", "optimism")
        result = _drive(gen)
        assert result == "0xpoolid123"

    def test_returns_none(self) -> None:
        """Test pool id retrieval returns None."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return None

        obj.contract_interact = fake_contract_interact
        gen = obj._get_pool_id("0xpool", "optimism")
        result = _drive(gen)
        assert result is None


class TestGetTokens:
    """Tests for _get_tokens."""

    def test_no_vault_address(self) -> None:
        """Test when no vault address is configured."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.balancer_vault_contract_addresses = {}

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._get_tokens("0xpool", "optimism")
            result = _drive(gen)
            assert result is None

    def test_no_pool_id(self) -> None:
        """Test when pool_id lookup returns None."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.balancer_vault_contract_addresses = {"optimism": "0xvault"}

        def fake_get_pool_id(addr, chain):
            yield
            return None

        obj._get_pool_id = fake_get_pool_id

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._get_tokens("0xpool", "optimism")
            result = _drive(gen)
            assert result is None

    def test_no_pool_tokens(self) -> None:
        """Test when contract_interact returns None for pool tokens."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.balancer_vault_contract_addresses = {"optimism": "0xvault"}

        call_count = [0]

        def fake_get_pool_id(addr, chain):
            yield
            return "0xpoolid"

        def fake_contract_interact(**kwargs):
            yield
            return None

        obj._get_pool_id = fake_get_pool_id
        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._get_tokens("0xpool", "optimism")
            result = _drive(gen)
            assert result is None

    def test_success(self) -> None:
        """Test successful token retrieval."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.balancer_vault_contract_addresses = {"optimism": "0xvault"}

        def fake_get_pool_id(addr, chain):
            yield
            return "0xpoolid"

        def fake_contract_interact(**kwargs):
            yield
            return [["0xtoken0", "0xtoken1"], [1000, 2000]]

        obj._get_pool_id = fake_get_pool_id
        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._get_tokens("0xpool", "optimism")
            result = _drive(gen)
            assert result == {"token0": "0xtoken0", "token1": "0xtoken1"}


class TestUpdateValue:
    """Tests for update_value."""

    def test_missing_pool_id(self) -> None:
        """Test with missing pool_id."""
        obj = _make_behaviour()
        gen = obj.update_value(pool_id=None, chain="optimism", vault_address="0xvault")
        result = _drive(gen)
        assert result == (None, None)

    def test_missing_chain(self) -> None:
        """Test with missing chain."""
        obj = _make_behaviour()
        gen = obj.update_value(pool_id="0xpoolid", chain=None, vault_address="0xvault")
        result = _drive(gen)
        assert result == (None, None)

    def test_invalid_pool_info(self) -> None:
        """Test with invalid pool_info from contract."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return None

        obj.contract_interact = fake_contract_interact
        gen = obj.update_value(
            pool_id="0xpoolid",
            chain="optimism",
            vault_address="0xvault",
            assets=["0xA"],
            max_amounts_in=[100],
        )
        result = _drive(gen)
        assert result == (None, None)

    def test_pool_info_empty_first_element(self) -> None:
        """Test with pool_info where first element is empty."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return [[], [1000]]

        obj.contract_interact = fake_contract_interact
        gen = obj.update_value(
            pool_id="0xpoolid",
            chain="optimism",
            vault_address="0xvault",
            assets=["0xA"],
            max_amounts_in=[100],
        )
        result = _drive(gen)
        assert result == (None, None)

    def test_pool_info_not_list(self) -> None:
        """Test with pool_info that is not a list."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return "not_a_list"

        obj.contract_interact = fake_contract_interact
        gen = obj.update_value(
            pool_id="0xpoolid",
            chain="optimism",
            vault_address="0xvault",
            assets=["0xA"],
            max_amounts_in=[100],
        )
        result = _drive(gen)
        assert result == (None, None)

    def test_success(self) -> None:
        """Test successful update_value."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return [["0xa", "0xb"], [1000, 2000]]

        obj.contract_interact = fake_contract_interact
        gen = obj.update_value(
            pool_id="0xpoolid",
            chain="optimism",
            vault_address="0xvault",
            assets=["0xA", "0xB"],
            max_amounts_in=[100, 200],
        )
        result = _drive(gen)
        assert result[0] == ["0xa", "0xb"]
        assert result[1] == [100, 200]

    def test_exception(self) -> None:
        """Test when contract_interact raises an exception."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            raise ValueError("contract error")
            yield  # noqa

        obj.contract_interact = fake_contract_interact
        gen = obj.update_value(
            pool_id="0xpoolid",
            chain="optimism",
            vault_address="0xvault",
            assets=["0xA"],
            max_amounts_in=[100],
        )
        result = _drive(gen)
        assert result == (None, None)


class TestEnter:
    """Tests for enter."""

    def _make_enter_obj(
        self,
        vault_address="0xvault",
        pool_id="0xpoolid",
        new_assets=None,
        new_max_amounts=None,
        expected_bpt=None,
        tx_hash="0xtxhash",
    ):
        """Create a behaviour with stubs for enter method."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.balancer_vault_contract_addresses = (
            {"optimism": vault_address} if vault_address else {}
        )
        params_mock.slippage_tolerance = 0.01

        def fake_get_pool_id(addr, chain):
            yield
            return pool_id

        def fake_update_value(**kwargs):
            yield
            return (new_assets, new_max_amounts)

        def fake_query_proportional_join(**kwargs):
            yield
            return expected_bpt

        call_idx = [0]

        def fake_contract_interact(**kwargs):
            call_idx[0] += 1
            yield
            return tx_hash

        obj._get_pool_id = fake_get_pool_id
        obj.update_value = fake_update_value
        obj.query_proportional_join = fake_query_proportional_join
        obj.contract_interact = fake_contract_interact

        return obj, params_mock

    def test_missing_params(self) -> None:
        """Test enter with missing required parameters."""
        obj = _make_behaviour()
        gen = obj.enter(
            pool_address=None,
            safe_address="0xsafe",
            assets=["0xa"],
            chain="optimism",
            max_amounts_in=[100],
            pool_type="Weighted",
        )
        result = _drive(gen)
        assert result == (None, None)

    def test_unknown_join_kind(self) -> None:
        """Test enter with unknown pool type."""
        obj = _make_behaviour()
        gen = obj.enter(
            pool_address="0xpool",
            safe_address="0xsafe",
            assets=["0xa"],
            chain="optimism",
            max_amounts_in=[100],
            pool_type="UnknownType",
        )
        result = _drive(gen)
        assert result == (None, None)

    def test_no_vault_address(self) -> None:
        """Test enter when no vault address configured."""
        obj, params_mock = self._make_enter_obj(vault_address=None)
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj.enter(
                pool_address="0xpool",
                safe_address="0xsafe",
                assets=["0xa"],
                chain="optimism",
                max_amounts_in=[100],
                pool_type="Weighted",
            )
            result = _drive(gen)
            assert result == (None, None)

    def test_no_pool_id(self) -> None:
        """Test enter when pool id is None."""
        obj, params_mock = self._make_enter_obj(pool_id=None)
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj.enter(
                pool_address="0xpool",
                safe_address="0xsafe",
                assets=["0xa"],
                chain="optimism",
                max_amounts_in=[100],
                pool_type="Weighted",
            )
            result = _drive(gen)
            assert result == (None, None)

    def test_no_expected_bpt(self) -> None:
        """Test enter when expected BPT is None."""
        obj, params_mock = self._make_enter_obj(
            new_assets=["0xa"], new_max_amounts=[100], expected_bpt=None
        )
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj.enter(
                pool_address="0xpool",
                safe_address="0xsafe",
                assets=["0xa"],
                chain="optimism",
                max_amounts_in=[100],
                pool_type="Weighted",
            )
            result = _drive(gen)
            assert result == (None, None)

    def test_success(self) -> None:
        """Test successful enter."""
        obj, params_mock = self._make_enter_obj(
            new_assets=["0xa", "0xb"],
            new_max_amounts=[100, 200],
            expected_bpt=1000,
            tx_hash="0xjoin_hash",
        )
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj.enter(
                pool_address="0xpool",
                safe_address="0xsafe",
                assets=["0xa", "0xb"],
                chain="optimism",
                max_amounts_in=[100, 200],
                pool_type="Weighted",
            )
            result = _drive(gen)
            assert result == ("0xjoin_hash", "0xvault")

    def test_success_with_zero_address(self) -> None:
        """Test successful enter with ZERO_ADDRESS in assets."""
        obj, params_mock = self._make_enter_obj(
            new_assets=["0xa", ZERO_ADDRESS],
            new_max_amounts=[100, 0],
            expected_bpt=500,
            tx_hash="0xjoin_hash",
        )
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj.enter(
                pool_address="0xpool",
                safe_address="0xsafe",
                assets=["0xa", ZERO_ADDRESS],
                chain="optimism",
                max_amounts_in=[100, 0],
                pool_type="Weighted",
            )
            result = _drive(gen)
            assert result == ("0xjoin_hash", "0xvault")


class TestExit:
    """Tests for exit."""

    def _make_exit_obj(
        self,
        vault_address="0xvault",
        pool_id="0xpoolid",
        bpt_amount=1000,
        expected_amounts=None,
        tx_hash="0xtxhash",
    ):
        """Create a behaviour with stubs for exit method."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.balancer_vault_contract_addresses = (
            {"optimism": vault_address} if vault_address else {}
        )
        params_mock.slippage_tolerance = 0.01

        def fake_get_pool_id(addr, chain):
            yield
            return pool_id

        def fake_query_proportional_exit(**kwargs):
            yield
            return expected_amounts

        interact_calls = [0]

        def fake_contract_interact(**kwargs):
            interact_calls[0] += 1
            yield
            if interact_calls[0] == 1:
                return bpt_amount
            return tx_hash

        obj._get_pool_id = fake_get_pool_id
        obj.query_proportional_exit = fake_query_proportional_exit
        obj.contract_interact = fake_contract_interact

        return obj, params_mock

    def test_missing_params(self) -> None:
        """Test exit with missing required parameters."""
        obj = _make_behaviour()
        gen = obj.exit(
            pool_address=None,
            safe_address="0xsafe",
            assets=["0xa"],
            chain="optimism",
            pool_type="Weighted",
        )
        result = _drive(gen)
        assert result == (None, None, None)

    def test_unknown_exit_kind(self) -> None:
        """Test exit with unknown pool type."""
        obj = _make_behaviour()
        gen = obj.exit(
            pool_address="0xpool",
            safe_address="0xsafe",
            assets=["0xa"],
            chain="optimism",
            pool_type="UnknownType",
        )
        result = _drive(gen)
        assert result == (None, None, None)

    def test_no_vault_address(self) -> None:
        """Test exit when no vault address configured."""
        obj, params_mock = self._make_exit_obj(vault_address=None)
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj.exit(
                pool_address="0xpool",
                safe_address="0xsafe",
                assets=["0xa"],
                chain="optimism",
                pool_type="Weighted",
            )
            result = _drive(gen)
            assert result == (None, None, None)

    def test_no_pool_id(self) -> None:
        """Test exit when pool id is None."""
        obj, params_mock = self._make_exit_obj(pool_id=None)
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj.exit(
                pool_address="0xpool",
                safe_address="0xsafe",
                assets=["0xa"],
                chain="optimism",
                pool_type="Weighted",
            )
            result = _drive(gen)
            assert result == (None, None, None)

    def test_bpt_amount_none(self) -> None:
        """Test exit when BPT balance is None."""
        obj, params_mock = self._make_exit_obj(bpt_amount=None)
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj.exit(
                pool_address="0xpool",
                safe_address="0xsafe",
                assets=["0xa"],
                chain="optimism",
                pool_type="Weighted",
            )
            result = _drive(gen)
            assert result == (None, None, None)

    def test_no_expected_amounts(self) -> None:
        """Test exit when expected amounts is None."""
        obj, params_mock = self._make_exit_obj(expected_amounts=None)
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj.exit(
                pool_address="0xpool",
                safe_address="0xsafe",
                assets=["0xa"],
                chain="optimism",
                pool_type="Weighted",
            )
            result = _drive(gen)
            assert result == (None, None, None)

    def test_success(self) -> None:
        """Test successful exit."""
        obj, params_mock = self._make_exit_obj(
            expected_amounts=[500, 600], tx_hash="0xexit_hash"
        )
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj.exit(
                pool_address="0xpool",
                safe_address="0xsafe",
                assets=["0xa", "0xb"],
                chain="optimism",
                pool_type="Weighted",
            )
            result = _drive(gen)
            assert result == ("0xexit_hash", "0xvault", False)

    def test_success_with_zero_address(self) -> None:
        """Test successful exit with ZERO_ADDRESS in assets."""
        obj, params_mock = self._make_exit_obj(
            expected_amounts=[500, 0], tx_hash="0xexit_hash"
        )
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj.exit(
                pool_address="0xpool",
                safe_address="0xsafe",
                assets=["0xa", ZERO_ADDRESS],
                chain="optimism",
                pool_type="Weighted",
            )
            result = _drive(gen)
            assert result == ("0xexit_hash", "0xvault", False)


class TestQueryProportionalExit:
    """Tests for query_proportional_exit."""

    def test_no_pool_tokens_info(self) -> None:
        """Test when pool tokens info is None."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return None

        obj.contract_interact = fake_contract_interact
        gen = obj.query_proportional_exit(
            pool_id="0xpoolid",
            pool_address="0xpool",
            vault_address="0xvault",
            chain="optimism",
            bpt_amount_in=1000,
        )
        result = _drive(gen)
        assert result is None

    def test_pool_tokens_info_too_short(self) -> None:
        """Test when pool tokens info has less than 2 elements."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return [["0xa"]]

        obj.contract_interact = fake_contract_interact
        gen = obj.query_proportional_exit(
            pool_id="0xpoolid",
            pool_address="0xpool",
            vault_address="0xvault",
            chain="optimism",
            bpt_amount_in=1000,
        )
        result = _drive(gen)
        assert result is None

    def test_no_total_bpt_supply(self) -> None:
        """Test when total BPT supply is None."""
        obj = _make_behaviour()
        call_count = [0]

        def fake_contract_interact(**kwargs):
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return [["0xa", "0xb"], [1000, 2000]]
            return None

        obj.contract_interact = fake_contract_interact
        gen = obj.query_proportional_exit(
            pool_id="0xpoolid",
            pool_address="0xpool",
            vault_address="0xvault",
            chain="optimism",
            bpt_amount_in=1000,
        )
        result = _drive(gen)
        assert result is None

    def test_success(self) -> None:
        """Test successful proportional exit query."""
        obj = _make_behaviour()
        call_count = [0]

        def fake_contract_interact(**kwargs):
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return [["0xa", "0xb"], [10000, 20000]]
            return 100000  # total supply

        obj.contract_interact = fake_contract_interact
        gen = obj.query_proportional_exit(
            pool_id="0xpoolid",
            pool_address="0xpool",
            vault_address="0xvault",
            chain="optimism",
            bpt_amount_in=10000,
        )
        result = _drive(gen)
        assert result is not None
        assert isinstance(result, list)

    def test_exception(self) -> None:
        """Test when an exception occurs during proportional exit."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            raise ValueError("error")
            yield  # noqa

        obj.contract_interact = fake_contract_interact
        gen = obj.query_proportional_exit(
            pool_id="0xpoolid",
            pool_address="0xpool",
            vault_address="0xvault",
            chain="optimism",
            bpt_amount_in=1000,
        )
        result = _drive(gen)
        assert result is None


class TestQueryProportionalJoin:
    """Tests for query_proportional_join."""

    def test_no_pool_tokens_info(self) -> None:
        """Test when pool tokens info is None."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return None

        obj.contract_interact = fake_contract_interact
        gen = obj.query_proportional_join(
            pool_id="0xpoolid",
            pool_address="0xpool",
            vault_address="0xvault",
            chain="optimism",
            amounts_in=[100, 200],
        )
        result = _drive(gen)
        assert result is None

    def test_pool_tokens_info_too_short(self) -> None:
        """Test when pool tokens info has less than 2 elements."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return [["0xa"]]

        obj.contract_interact = fake_contract_interact
        gen = obj.query_proportional_join(
            pool_id="0xpoolid",
            pool_address="0xpool",
            vault_address="0xvault",
            chain="optimism",
            amounts_in=[100],
        )
        result = _drive(gen)
        assert result is None

    def test_no_total_bpt_supply(self) -> None:
        """Test when total BPT supply is None."""
        obj = _make_behaviour()
        call_count = [0]

        def fake_contract_interact(**kwargs):
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return [["0xa", "0xb"], [1000, 2000]]
            return None

        obj.contract_interact = fake_contract_interact
        gen = obj.query_proportional_join(
            pool_id="0xpoolid",
            pool_address="0xpool",
            vault_address="0xvault",
            chain="optimism",
            amounts_in=[100, 200],
        )
        result = _drive(gen)
        assert result is None

    def test_success(self) -> None:
        """Test successful proportional join query."""
        obj = _make_behaviour()
        call_count = [0]

        def fake_contract_interact(**kwargs):
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return [["0xa", "0xb"], [10000, 20000]]
            return 100000

        obj.contract_interact = fake_contract_interact
        gen = obj.query_proportional_join(
            pool_id="0xpoolid",
            pool_address="0xpool",
            vault_address="0xvault",
            chain="optimism",
            amounts_in=[100, 200],
        )
        result = _drive(gen)
        assert result is not None

    def test_exception(self) -> None:
        """Test when an exception occurs during proportional join."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            raise ValueError("error")
            yield  # noqa

        obj.contract_interact = fake_contract_interact
        gen = obj.query_proportional_join(
            pool_id="0xpoolid",
            pool_address="0xpool",
            vault_address="0xvault",
            chain="optimism",
            amounts_in=[100, 200],
        )
        result = _drive(gen)
        assert result is None


class TestQueryJoinBpt:
    """Tests for _query_join_bpt."""

    def test_no_queries_address(self) -> None:
        """Test when no queries address is configured."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.balancer_queries_contract_addresses = {}

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._query_join_bpt(
                pool_id="0xpoolid",
                sender="0xsafe",
                assets=["0x" + "aa" * 20],
                max_amounts_in=[100],
                join_kind=1,
                from_internal_balance=False,
                chain="optimism",
            )
            result = _drive(gen)
            assert result is None

    def test_success(self) -> None:
        """Test successful BPT query."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.balancer_queries_contract_addresses = {"optimism": "0xqueries"}

        def fake_contract_interact(**kwargs):
            yield
            return {"bpt_out": 5000}

        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._query_join_bpt(
                pool_id="0xpoolid",
                sender="0xsafe",
                assets=["0x" + "aa" * 20],
                max_amounts_in=[100],
                join_kind=1,
                from_internal_balance=False,
                chain="optimism",
            )
            result = _drive(gen)
            assert result == 5000

    def test_exception(self) -> None:
        """Test when an exception occurs during join BPT query."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.balancer_queries_contract_addresses = {"optimism": "0xqueries"}

        def fake_contract_interact(**kwargs):
            raise ValueError("bad query")
            yield  # noqa

        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._query_join_bpt(
                pool_id="0xpoolid",
                sender="0xsafe",
                assets=["0x" + "aa" * 20],
                max_amounts_in=[100],
                join_kind=1,
                from_internal_balance=False,
                chain="optimism",
            )
            result = _drive(gen)
            assert result is None


class TestQueryExitAmounts:
    """Tests for _query_exit_amounts."""

    def test_no_queries_address(self) -> None:
        """Test when no queries address is configured."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.balancer_queries_contract_addresses = {}

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._query_exit_amounts(
                pool_id="0xpoolid",
                sender="0xsafe",
                assets=["0x" + "aa" * 20],
                bpt_amount_in=1000,
                exit_kind=1,
                to_internal_balance=False,
                chain="optimism",
            )
            result = _drive(gen)
            assert result is None

    def test_success(self) -> None:
        """Test successful exit amounts query."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.balancer_queries_contract_addresses = {"optimism": "0xqueries"}

        def fake_contract_interact(**kwargs):
            yield
            return {"amounts_out": [500, 600]}

        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._query_exit_amounts(
                pool_id="0xpoolid",
                sender="0xsafe",
                assets=["0x" + "aa" * 20],
                bpt_amount_in=1000,
                exit_kind=1,
                to_internal_balance=False,
                chain="optimism",
            )
            result = _drive(gen)
            assert result == [500, 600]

    def test_invalid_result(self) -> None:
        """Test when result doesn't contain amounts_out."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.balancer_queries_contract_addresses = {"optimism": "0xqueries"}

        def fake_contract_interact(**kwargs):
            yield
            return {"other_key": "value"}

        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._query_exit_amounts(
                pool_id="0xpoolid",
                sender="0xsafe",
                assets=["0x" + "aa" * 20],
                bpt_amount_in=1000,
                exit_kind=1,
                to_internal_balance=False,
                chain="optimism",
            )
            result = _drive(gen)
            assert result is None

    def test_none_result(self) -> None:
        """Test when contract returns None."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.balancer_queries_contract_addresses = {"optimism": "0xqueries"}

        def fake_contract_interact(**kwargs):
            yield
            return None

        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._query_exit_amounts(
                pool_id="0xpoolid",
                sender="0xsafe",
                assets=["0x" + "aa" * 20],
                bpt_amount_in=1000,
                exit_kind=1,
                to_internal_balance=False,
                chain="optimism",
            )
            result = _drive(gen)
            assert result is None

    def test_exception(self) -> None:
        """Test when an exception occurs during exit amounts query."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.balancer_queries_contract_addresses = {"optimism": "0xqueries"}

        def fake_contract_interact(**kwargs):
            raise ValueError("bad query")
            yield  # noqa

        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._query_exit_amounts(
                pool_id="0xpoolid",
                sender="0xsafe",
                assets=["0x" + "aa" * 20],
                bpt_amount_in=1000,
                exit_kind=1,
                to_internal_balance=False,
                chain="optimism",
            )
            result = _drive(gen)
            assert result is None

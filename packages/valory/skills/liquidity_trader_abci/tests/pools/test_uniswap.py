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

"""Test the pools/uniswap.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import sys
from unittest.mock import MagicMock, PropertyMock, patch

from packages.valory.skills.liquidity_trader_abci.pools.uniswap import (
    INT_MAX,
    MAX_TICK,
    MIN_TICK,
    ZERO_ADDRESS,
    MintParams,
    UniswapPoolBehaviour,
)


def test_import() -> None:
    """Test that the uniswap module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.pools.uniswap  # noqa


def test_zero_address() -> None:
    """Test ZERO_ADDRESS constant."""
    assert ZERO_ADDRESS == "0x0000000000000000000000000000000000000000"


def test_min_tick() -> None:
    """Test MIN_TICK constant."""
    assert MIN_TICK == -887272


def test_max_tick() -> None:
    """Test MAX_TICK constant."""
    assert MAX_TICK == 887272


def test_int_max() -> None:
    """Test INT_MAX constant."""
    assert INT_MAX == sys.maxsize


class TestMintParams:
    """Test MintParams class."""

    def test_init(self) -> None:
        """Test MintParams initialization."""
        params = MintParams(
            token0="0xtoken0",
            token1="0xtoken1",
            fee=3000,
            tickLower=-100,
            tickUpper=100,
            amount0Desired=1000,
            amount1Desired=2000,
            amount0Min=900,
            amount1Min=1800,
            recipient="0xrecipient",
            deadline=999999,
        )
        assert params.token0 == "0xtoken0"
        assert params.token1 == "0xtoken1"
        assert params.fee == 3000
        assert params.tickLower == -100
        assert params.tickUpper == 100
        assert params.amount0Desired == 1000
        assert params.amount1Desired == 2000
        assert params.amount0Min == 900
        assert params.amount1Min == 1800
        assert params.recipient == "0xrecipient"
        assert params.deadline == 999999


def _make_behaviour():
    """Create a UniswapPoolBehaviour without __init__."""
    obj = object.__new__(UniswapPoolBehaviour)
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


class TestUniswapInit:
    """Test UniswapPoolBehaviour __init__."""

    def test_init_calls_super(self) -> None:
        """Test that __init__ calls super().__init__."""
        with patch.object(UniswapPoolBehaviour.__bases__[0], "__init__", return_value=None):
            obj = UniswapPoolBehaviour.__new__(UniswapPoolBehaviour)
            UniswapPoolBehaviour.__init__(obj, some_kwarg="test")


class TestGetTokens:
    """Tests for _get_tokens."""

    def test_success(self) -> None:
        """Test successful token retrieval."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return ["0xtoken0", "0xtoken1"]

        obj.contract_interact = fake_contract_interact
        gen = obj._get_tokens("0xpool", "optimism")
        result = _drive(gen)
        assert result == {"token0": "0xtoken0", "token1": "0xtoken1"}

    def test_returns_none(self) -> None:
        """Test when contract returns None."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return None

        obj.contract_interact = fake_contract_interact
        gen = obj._get_tokens("0xpool", "optimism")
        result = _drive(gen)
        assert result is None


class TestGetPoolFee:
    """Tests for _get_pool_fee."""

    def test_success(self) -> None:
        """Test successful fee retrieval."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return 3000

        obj.contract_interact = fake_contract_interact
        gen = obj._get_pool_fee("0xpool", "optimism")
        result = _drive(gen)
        assert result == 3000

    def test_returns_none(self) -> None:
        """Test when contract returns None."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return None

        obj.contract_interact = fake_contract_interact
        gen = obj._get_pool_fee("0xpool", "optimism")
        result = _drive(gen)
        assert result is None


class TestGetTickSpacing:
    """Tests for _get_tick_spacing."""

    def test_success(self) -> None:
        """Test successful tick spacing retrieval."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return 60

        obj.contract_interact = fake_contract_interact
        gen = obj._get_tick_spacing("0xpool", "optimism")
        result = _drive(gen)
        assert result == 60

    def test_returns_none(self) -> None:
        """Test when contract returns None."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return None

        obj.contract_interact = fake_contract_interact
        gen = obj._get_tick_spacing("0xpool", "optimism")
        result = _drive(gen)
        assert result is None


class TestCalculateTickLowerAndUpper:
    """Tests for _calculate_tick_lower_and_upper."""

    def test_no_tick_spacing(self) -> None:
        """Test when tick spacing is None."""
        obj = _make_behaviour()

        def fake_get_tick_spacing(addr, chain):
            yield
            return None

        obj._get_tick_spacing = fake_get_tick_spacing
        gen = obj._calculate_tick_lower_and_upper("0xpool", "optimism")
        result = _drive(gen)
        assert result == (None, None)

    def test_tick_spacing_60(self) -> None:
        """Test with tick spacing of 60."""
        obj = _make_behaviour()

        def fake_get_tick_spacing(addr, chain):
            yield
            return 60

        obj._get_tick_spacing = fake_get_tick_spacing
        gen = obj._calculate_tick_lower_and_upper("0xpool", "optimism")
        result = _drive(gen)
        lower, upper = result
        assert lower is not None
        assert upper is not None
        assert lower % 60 == 0
        assert upper % 60 == 0
        assert lower < 0
        assert upper > 0

    def test_tick_spacing_1(self) -> None:
        """Test with tick spacing of 1."""
        obj = _make_behaviour()

        def fake_get_tick_spacing(addr, chain):
            yield
            return 1

        obj._get_tick_spacing = fake_get_tick_spacing
        gen = obj._calculate_tick_lower_and_upper("0xpool", "optimism")
        result = _drive(gen)
        lower, upper = result
        assert lower == MIN_TICK
        assert upper == MAX_TICK

    def test_tick_spacing_200(self) -> None:
        """Test with tick spacing of 200."""
        obj = _make_behaviour()

        def fake_get_tick_spacing(addr, chain):
            yield
            return 200

        obj._get_tick_spacing = fake_get_tick_spacing
        gen = obj._calculate_tick_lower_and_upper("0xpool", "optimism")
        result = _drive(gen)
        lower, upper = result
        assert lower % 200 == 0
        assert upper % 200 == 0


class TestBurnToken:
    """Tests for burn_token."""

    def test_no_position_manager(self) -> None:
        """Test when no position manager address."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {}

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.burn_token(123, "optimism")
            result = _drive(gen)
            assert result is None

    def test_success(self) -> None:
        """Test successful burn."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_contract_interact(**kwargs):
            yield
            return "0xburn_hash"

        obj.contract_interact = fake_contract_interact

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.burn_token(123, "optimism")
            result = _drive(gen)
            assert result == "0xburn_hash"


class TestCollectTokens:
    """Tests for collect_tokens."""

    def test_no_position_manager(self) -> None:
        """Test when no position manager address."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {}

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.collect_tokens(123, "0xrecip", 1000, 2000, "optimism")
            result = _drive(gen)
            assert result is None

    def test_success(self) -> None:
        """Test successful collect."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_contract_interact(**kwargs):
            yield
            return "0xcollect_hash"

        obj.contract_interact = fake_contract_interact

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.collect_tokens(123, "0xrecip", 1000, 2000, "optimism")
            result = _drive(gen)
            assert result == "0xcollect_hash"


class TestDecreaseLiquidity:
    """Tests for decrease_liquidity."""

    def test_no_position_manager(self) -> None:
        """Test when no position manager address."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {}

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.decrease_liquidity(123, 1000, 0, 0, 99999, "optimism")
            result = _drive(gen)
            assert result is None

    def test_success(self) -> None:
        """Test successful decrease."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_contract_interact(**kwargs):
            yield
            return "0xdecrease_hash"

        obj.contract_interact = fake_contract_interact

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.decrease_liquidity(123, 1000, 0, 0, 99999, "optimism")
            result = _drive(gen)
            assert result == "0xdecrease_hash"


class TestGetLiquidityForToken:
    """Tests for get_liquidity_for_token."""

    def test_no_position_manager(self) -> None:
        """Test when no position manager address."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {}

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.get_liquidity_for_token(123, "optimism")
            result = _drive(gen)
            assert result is None

    def test_no_position(self) -> None:
        """Test when position data is None."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_contract_interact(**kwargs):
            yield
            return None

        obj.contract_interact = fake_contract_interact

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.get_liquidity_for_token(123, "optimism")
            result = _drive(gen)
            assert result is None

    def test_success(self) -> None:
        """Test successful liquidity retrieval (index 7)."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        position_data = [0, 0, 0, 0, 0, 0, 0, 5000, 0, 0, 0, 0]

        def fake_contract_interact(**kwargs):
            yield
            return position_data

        obj.contract_interact = fake_contract_interact

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.get_liquidity_for_token(123, "optimism")
            result = _drive(gen)
            assert result == 5000


class TestEnter:
    """Tests for enter."""

    def test_missing_params(self) -> None:
        """Test enter with missing required parameters."""
        obj = _make_behaviour()
        gen = obj.enter(pool_address=None, safe_address="0xsafe", assets=["0xa"],
                        chain="optimism", max_amounts_in=[100])
        result = _drive(gen)
        assert result == (None, None)

    def test_no_position_manager(self) -> None:
        """Test enter when no position manager address."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {}

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.enter(pool_address="0xpool", safe_address="0xsafe",
                            assets=["0xa", "0xb"], chain="optimism",
                            max_amounts_in=[100, 200])
            result = _drive(gen)
            assert result == (None, None)

    def test_no_pool_fee_and_fetch_fails(self) -> None:
        """Test enter when pool_fee is not provided and fetch fails."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_get_pool_fee(addr, chain):
            yield
            return None

        obj._get_pool_fee = fake_get_pool_fee

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.enter(pool_address="0xpool", safe_address="0xsafe",
                            assets=["0xa", "0xb"], chain="optimism",
                            max_amounts_in=[100, 200])
            result = _drive(gen)
            assert result == (None, None)

    def test_no_tick_spacing(self) -> None:
        """Test enter when tick calculation fails."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_calc_ticks(addr, chain):
            yield
            return None, None

        obj._calculate_tick_lower_and_upper = fake_calc_ticks

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.enter(pool_address="0xpool", safe_address="0xsafe",
                            assets=["0xa", "0xb"], chain="optimism",
                            max_amounts_in=[100, 200], pool_fee=3000)
            result = _drive(gen)
            assert result == (None, None)

    def test_slippage_protection_fails(self) -> None:
        """Test enter when slippage protection returns None."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_calc_ticks(addr, chain):
            yield
            return -887220, 887220

        def fake_slippage_for_mint(addr, tl, tu, amounts, chain):
            yield
            return None, None

        obj._calculate_tick_lower_and_upper = fake_calc_ticks
        obj._calculate_slippage_protection_for_mint = fake_slippage_for_mint

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.enter(pool_address="0xpool", safe_address="0xsafe",
                            assets=["0xa", "0xb"], chain="optimism",
                            max_amounts_in=[100, 200], pool_fee=3000)
            result = _drive(gen)
            assert result == (None, None)

    def test_success(self) -> None:
        """Test successful enter."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_calc_ticks(addr, chain):
            yield
            return -887220, 887220

        def fake_slippage_for_mint(addr, tl, tu, amounts, chain):
            yield
            return 90, 180

        def fake_contract_interact(**kwargs):
            yield
            return "0xmint_hash"

        rs = MagicMock()
        rs.last_round_transition_timestamp.timestamp.return_value = 1700000000.0

        obj._calculate_tick_lower_and_upper = fake_calc_ticks
        obj._calculate_slippage_protection_for_mint = fake_slippage_for_mint
        obj.contract_interact = fake_contract_interact
        obj.context.state.round_sequence = rs

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.enter(pool_address="0xpool", safe_address="0xsafe",
                            assets=["0xa", "0xb"], chain="optimism",
                            max_amounts_in=[100, 200], pool_fee=3000)
            result = _drive(gen)
            assert result == ("0xmint_hash", "0xpm")

    def test_success_fetch_pool_fee(self) -> None:
        """Test successful enter when pool_fee is fetched."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_get_pool_fee(addr, chain):
            yield
            return 3000

        def fake_calc_ticks(addr, chain):
            yield
            return -887220, 887220

        def fake_slippage_for_mint(addr, tl, tu, amounts, chain):
            yield
            return 90, 180

        def fake_contract_interact(**kwargs):
            yield
            return "0xmint_hash"

        rs = MagicMock()
        rs.last_round_transition_timestamp.timestamp.return_value = 1700000000.0

        obj._get_pool_fee = fake_get_pool_fee
        obj._calculate_tick_lower_and_upper = fake_calc_ticks
        obj._calculate_slippage_protection_for_mint = fake_slippage_for_mint
        obj.contract_interact = fake_contract_interact
        obj.context.state.round_sequence = rs

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.enter(pool_address="0xpool", safe_address="0xsafe",
                            assets=["0xa", "0xb"], chain="optimism",
                            max_amounts_in=[100, 200])
            result = _drive(gen)
            assert result == ("0xmint_hash", "0xpm")


class TestExit:
    """Tests for exit."""

    def test_missing_params(self) -> None:
        """Test exit with missing required parameters."""
        obj = _make_behaviour()
        gen = obj.exit(token_id=None, safe_address="0xsafe", chain="optimism")
        result = _drive(gen)
        assert result == (None, None, None)

    def test_no_position_manager(self) -> None:
        """Test exit when no position manager address."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {}

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.exit(token_id=123, safe_address="0xsafe", chain="optimism")
            result = _drive(gen)
            assert result == (None, None, None)

    def test_no_liquidity_and_fetch_fails(self) -> None:
        """Test exit when liquidity is not provided and fetch fails."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_get_liquidity(tid, chain):
            yield
            return None

        obj.get_liquidity_for_token = fake_get_liquidity

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.exit(token_id=123, safe_address="0xsafe", chain="optimism")
            result = _drive(gen)
            assert result == (None, None, None)

    def test_slippage_protection_fails(self) -> None:
        """Test exit when slippage protection returns None."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_slippage_for_decrease(tid, liq, chain, pool):
            yield
            return None, None

        obj._calculate_slippage_protection_for_decrease = fake_slippage_for_decrease

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.exit(token_id=123, safe_address="0xsafe", chain="optimism",
                           liquidity=5000, pool_address="0xpool")
            result = _drive(gen)
            assert result == (None, None, None)

    def test_decrease_liquidity_fails(self) -> None:
        """Test exit when decrease_liquidity fails."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_slippage_for_decrease(tid, liq, chain, pool):
            yield
            return 0, 0

        def fake_decrease(tid, liq, a0, a1, dl, chain):
            yield
            return None

        rs = MagicMock()
        rs.last_round_transition_timestamp.timestamp.return_value = 1700000000.0

        obj._calculate_slippage_protection_for_decrease = fake_slippage_for_decrease
        obj.decrease_liquidity = fake_decrease
        obj.context.state.round_sequence = rs

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.exit(token_id=123, safe_address="0xsafe", chain="optimism",
                           liquidity=5000, pool_address="0xpool")
            result = _drive(gen)
            assert result == (None, None, None)

    def test_collect_tokens_fails(self) -> None:
        """Test exit when collect_tokens fails."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_slippage_for_decrease(tid, liq, chain, pool):
            yield
            return 0, 0

        def fake_decrease(tid, liq, a0, a1, dl, chain):
            yield
            return "0xdecrease"

        def fake_collect(token_id, recipient, amount0_max, amount1_max, chain):
            yield
            return None

        rs = MagicMock()
        rs.last_round_transition_timestamp.timestamp.return_value = 1700000000.0

        obj._calculate_slippage_protection_for_decrease = fake_slippage_for_decrease
        obj.decrease_liquidity = fake_decrease
        obj.collect_tokens = fake_collect
        obj.context.state.round_sequence = rs

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.exit(token_id=123, safe_address="0xsafe", chain="optimism",
                           liquidity=5000, pool_address="0xpool")
            result = _drive(gen)
            assert result == (None, None, None)

    def test_no_multisend_address(self) -> None:
        """Test exit when multisend address is empty."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}
        params_mock.multisend_contract_addresses = {"optimism": ""}

        def fake_slippage_for_decrease(tid, liq, chain, pool):
            yield
            return 0, 0

        def fake_decrease(tid, liq, a0, a1, dl, chain):
            yield
            return "0xdecrease"

        def fake_collect(token_id, recipient, amount0_max, amount1_max, chain):
            yield
            return "0xcollect"

        rs = MagicMock()
        rs.last_round_transition_timestamp.timestamp.return_value = 1700000000.0

        obj._calculate_slippage_protection_for_decrease = fake_slippage_for_decrease
        obj.decrease_liquidity = fake_decrease
        obj.collect_tokens = fake_collect
        obj.context.state.round_sequence = rs

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.exit(token_id=123, safe_address="0xsafe", chain="optimism",
                           liquidity=5000, pool_address="0xpool")
            result = _drive(gen)
            assert result == (None, None, None)

    def test_multisend_tx_hash_none(self) -> None:
        """Test exit when multisend tx hash is None."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}
        params_mock.multisend_contract_addresses = {"optimism": "0xmultisend"}

        def fake_slippage_for_decrease(tid, liq, chain, pool):
            yield
            return 0, 0

        def fake_decrease(tid, liq, a0, a1, dl, chain):
            yield
            return "0xdecrease"

        def fake_collect(token_id, recipient, amount0_max, amount1_max, chain):
            yield
            return "0xcollect"

        def fake_contract_interact(**kwargs):
            yield
            return None

        rs = MagicMock()
        rs.last_round_transition_timestamp.timestamp.return_value = 1700000000.0

        obj._calculate_slippage_protection_for_decrease = fake_slippage_for_decrease
        obj.decrease_liquidity = fake_decrease
        obj.collect_tokens = fake_collect
        obj.contract_interact = fake_contract_interact
        obj.context.state.round_sequence = rs

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.exit(token_id=123, safe_address="0xsafe", chain="optimism",
                           liquidity=5000, pool_address="0xpool")
            result = _drive(gen)
            assert result == (None, None, None)

    def test_success(self) -> None:
        """Test successful exit."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}
        params_mock.multisend_contract_addresses = {"optimism": "0xmultisend"}

        def fake_slippage_for_decrease(tid, liq, chain, pool):
            yield
            return 0, 0

        def fake_decrease(tid, liq, a0, a1, dl, chain):
            yield
            return "0xdecrease"

        def fake_collect(token_id, recipient, amount0_max, amount1_max, chain):
            yield
            return "0xcollect"

        def fake_contract_interact(**kwargs):
            yield
            return "0xabcdef1234567890"

        rs = MagicMock()
        rs.last_round_transition_timestamp.timestamp.return_value = 1700000000.0

        obj._calculate_slippage_protection_for_decrease = fake_slippage_for_decrease
        obj.decrease_liquidity = fake_decrease
        obj.collect_tokens = fake_collect
        obj.contract_interact = fake_contract_interact
        obj.context.state.round_sequence = rs

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.exit(token_id=123, safe_address="0xsafe", chain="optimism",
                           liquidity=5000, pool_address="0xpool")
            result = _drive(gen)
            assert result[1] == "0xmultisend"
            assert result[2] is True
            assert isinstance(result[0], bytes)

    def test_fetch_liquidity_when_not_provided(self) -> None:
        """Test exit when liquidity is fetched from contract."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}
        params_mock.multisend_contract_addresses = {"optimism": "0xmultisend"}

        def fake_get_liquidity(tid, chain):
            yield
            return 5000

        def fake_slippage_for_decrease(tid, liq, chain, pool):
            yield
            return 0, 0

        def fake_decrease(tid, liq, a0, a1, dl, chain):
            yield
            return "0xdecrease"

        def fake_collect(token_id, recipient, amount0_max, amount1_max, chain):
            yield
            return "0xcollect"

        def fake_contract_interact(**kwargs):
            yield
            return "0xabcdef1234567890"

        rs = MagicMock()
        rs.last_round_transition_timestamp.timestamp.return_value = 1700000000.0

        obj.get_liquidity_for_token = fake_get_liquidity
        obj._calculate_slippage_protection_for_decrease = fake_slippage_for_decrease
        obj.decrease_liquidity = fake_decrease
        obj.collect_tokens = fake_collect
        obj.contract_interact = fake_contract_interact
        obj.context.state.round_sequence = rs

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj.exit(token_id=123, safe_address="0xsafe", chain="optimism",
                           pool_address="0xpool")
            result = _drive(gen)
            assert result[2] is True


class TestCalculateSlippageProtectionForMint:
    """Tests for _calculate_slippage_protection_for_mint."""

    def test_no_slot0_data(self) -> None:
        """Test when slot0 data is None."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return None

        obj.contract_interact = fake_contract_interact
        gen = obj._calculate_slippage_protection_for_mint(
            "0xpool", -887220, 887220, [100, 200], "optimism"
        )
        result = _drive(gen)
        assert result == (None, None)

    def test_slot0_missing_sqrt_price(self) -> None:
        """Test when slot0 data is missing sqrt_price_x96."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            yield
            return {"other_key": 123}

        obj.contract_interact = fake_contract_interact
        gen = obj._calculate_slippage_protection_for_mint(
            "0xpool", -887220, 887220, [100, 200], "optimism"
        )
        result = _drive(gen)
        assert result == (None, None)

    def test_success(self) -> None:
        """Test successful slippage calculation."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.slippage_tolerance = 0.01

        def fake_contract_interact(**kwargs):
            yield
            # A reasonable sqrt_price_x96 value for 1:1 ratio
            return {"sqrt_price_x96": 79228162514264337593543950336}

        obj.contract_interact = fake_contract_interact

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj._calculate_slippage_protection_for_mint(
                "0xpool", -887220, 887220, [1000000, 1000000], "optimism"
            )
            result = _drive(gen)
            a0, a1 = result
            assert a0 is not None
            assert a1 is not None

    def test_exception(self) -> None:
        """Test when exception occurs."""
        obj = _make_behaviour()

        def fake_contract_interact(**kwargs):
            raise ValueError("error")
            yield  # noqa

        obj.contract_interact = fake_contract_interact
        gen = obj._calculate_slippage_protection_for_mint(
            "0xpool", -887220, 887220, [100, 200], "optimism"
        )
        result = _drive(gen)
        assert result == (None, None)


class TestCalculateSlippageProtectionForDecrease:
    """Tests for _calculate_slippage_protection_for_decrease."""

    def test_no_position_manager(self) -> None:
        """Test when no position manager address."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {}

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj._calculate_slippage_protection_for_decrease(
                123, 5000, "optimism", "0xpool"
            )
            result = _drive(gen)
            assert result == (0, 0)

    def test_no_position(self) -> None:
        """Test when position data is None."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_contract_interact(**kwargs):
            yield
            return None

        obj.contract_interact = fake_contract_interact

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj._calculate_slippage_protection_for_decrease(
                123, 5000, "optimism", "0xpool"
            )
            result = _drive(gen)
            assert result == (0, 0)

    def test_no_slot0_data(self) -> None:
        """Test when slot0 data is None."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        call_count = [0]

        def fake_contract_interact(**kwargs):
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return {"tickLower": -887220, "tickUpper": 887220}
            return None

        obj.contract_interact = fake_contract_interact

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj._calculate_slippage_protection_for_decrease(
                123, 5000, "optimism", "0xpool"
            )
            result = _drive(gen)
            assert result == (0, 0)

    def test_slot0_missing_sqrt_price(self) -> None:
        """Test when slot0 data is missing sqrt_price_x96."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        call_count = [0]

        def fake_contract_interact(**kwargs):
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return {"tickLower": -887220, "tickUpper": 887220}
            return {"no_sqrt": True}

        obj.contract_interact = fake_contract_interact

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj._calculate_slippage_protection_for_decrease(
                123, 5000, "optimism", "0xpool"
            )
            result = _drive(gen)
            assert result == (0, 0)

    def test_success(self) -> None:
        """Test successful slippage calculation for decrease."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}
        params_mock.slippage_tolerance = 0.01

        call_count = [0]

        def fake_contract_interact(**kwargs):
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return {"tickLower": -887220, "tickUpper": 887220}
            return {"sqrt_price_x96": 79228162514264337593543950336}

        obj.contract_interact = fake_contract_interact

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj._calculate_slippage_protection_for_decrease(
                123, 5000, "optimism", "0xpool"
            )
            result = _drive(gen)
            a0, a1 = result
            assert a0 is not None
            assert a1 is not None

    def test_exception(self) -> None:
        """Test when exception occurs."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.uniswap_position_manager_contract_addresses = {"optimism": "0xpm"}

        def fake_contract_interact(**kwargs):
            raise ValueError("error")
            yield  # noqa

        obj.contract_interact = fake_contract_interact

        with patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock):
            gen = obj._calculate_slippage_protection_for_decrease(
                123, 5000, "optimism", "0xpool"
            )
            result = _drive(gen)
            assert result == (None, None)

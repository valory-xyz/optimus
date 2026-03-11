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

"""Test the behaviours/base.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from enum import Enum

from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    AGENT_TYPE,
    APR_UPDATE_INTERVAL,
    ASSETS_FILENAME,
    CAMPAIGN_TYPES,
    COIN_ID_MAPPING,
    DECISION_MAKING_TIMEOUT,
    ERC20_DECIMALS,
    ETH_INITIAL_AMOUNT,
    ETH_REMAINING_KEY,
    ETHER_VALUE,
    EVALUATE_STRATEGY_TIMEOUT,
    FETCH_STRATEGIES_TIMEOUT,
    HTTP_NOT_FOUND,
    HTTP_OK,
    INTEGRATOR,
    LIVENESS_RATIO_SCALE_FACTOR,
    MAX_RETRIES_FOR_API_CALL,
    MAX_RETRIES_FOR_ROUTES,
    MAX_STEP_COST_RATIO,
    METRICS_NAME,
    METRICS_TYPE,
    METRICS_UPDATE_INTERVAL,
    MIN_TIME_IN_POSITION,
    POOL_FILENAME,
    PORTFOLIO_UPDATE_INTERVAL,
    READ_MODE,
    REQUIRED_REQUESTS_SAFETY_MARGIN,
    RETRIES,
    REWARD_TOKEN_ADDRESSES,
    SAFE_TX_GAS,
    SLEEP_TIME,
    THRESHOLDS,
    UTF8,
    WAITING_PERIOD_FOR_BALANCE_TO_REFLECT,
    WHITELISTED_ASSETS,
    WRITE_MODE,
    ZERO_ADDRESS,
    Action,
    Chain,
    Decision,
    DexType,
    GasCostTracker,
    LiquidityTraderBaseBehaviour,
    PositionStatus,
    SwapStatus,
    TradingType,
)


def test_import() -> None:
    """Test that the behaviours base module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.behaviours.base  # noqa


class TestDexType:
    """Test DexType enum."""

    def test_is_enum(self) -> None:
        """Test DexType is an Enum."""
        assert issubclass(DexType, Enum)

    def test_values(self) -> None:
        """Test DexType values."""
        assert DexType.BALANCER.value == "balancerPool"
        assert DexType.UNISWAP_V3.value == "UniswapV3"
        assert DexType.STURDY.value == "Sturdy"
        assert DexType.VELODROME.value == "velodrome"

    def test_member_count(self) -> None:
        """Test DexType has expected number of members."""
        assert len(DexType) == 4


class TestAction:
    """Test Action enum."""

    def test_is_enum(self) -> None:
        """Test Action is an Enum."""
        assert issubclass(Action, Enum)

    def test_key_values(self) -> None:
        """Test Action key values."""
        assert Action.CLAIM_REWARDS.value == "ClaimRewards"
        assert Action.EXIT_POOL.value == "ExitPool"
        assert Action.ENTER_POOL.value == "EnterPool"
        assert Action.BRIDGE_SWAP.value == "BridgeAndSwap"
        assert Action.FIND_BRIDGE_ROUTE.value == "FindBridgeRoute"
        assert Action.EXECUTE_STEP.value == "execute_step"
        assert Action.ROUTES_FETCHED.value == "routes_fetched"
        assert Action.FIND_ROUTE.value == "find_route"
        assert Action.BRIDGE_SWAP_EXECUTED.value == "bridge_swap_executed"
        assert Action.STEP_EXECUTED.value == "step_executed"
        assert Action.SWITCH_ROUTE.value == "switch_route"
        assert Action.WITHDRAW.value == "withdraw"
        assert Action.DEPOSIT.value == "deposit"
        assert Action.STAKE_LP_TOKENS.value == "StakeLpTokens"
        assert Action.UNSTAKE_LP_TOKENS.value == "UnstakeLpTokens"
        assert Action.CLAIM_STAKING_REWARDS.value == "ClaimStakingRewards"

    def test_member_count(self) -> None:
        """Test Action has expected number of members."""
        assert len(Action) == 16


class TestSwapStatus:
    """Test SwapStatus enum."""

    def test_is_enum(self) -> None:
        """Test SwapStatus is an Enum."""
        assert issubclass(SwapStatus, Enum)

    def test_values(self) -> None:
        """Test SwapStatus values."""
        assert SwapStatus.DONE.value == "DONE"
        assert SwapStatus.PENDING.value == "PENDING"
        assert SwapStatus.INVALID.value == "INVALID"
        assert SwapStatus.NOT_FOUND.value == "NOT_FOUND"
        assert SwapStatus.FAILED.value == "FAILED"


class TestDecision:
    """Test Decision enum."""

    def test_is_enum(self) -> None:
        """Test Decision is an Enum."""
        assert issubclass(Decision, Enum)

    def test_values(self) -> None:
        """Test Decision values."""
        assert Decision.CONTINUE.value == "continue"
        assert Decision.WAIT.value == "wait"
        assert Decision.EXIT.value == "exit"


class TestPositionStatus:
    """Test PositionStatus enum."""

    def test_is_enum(self) -> None:
        """Test PositionStatus is an Enum."""
        assert issubclass(PositionStatus, Enum)

    def test_values(self) -> None:
        """Test PositionStatus values."""
        assert PositionStatus.OPEN.value == "open"
        assert PositionStatus.CLOSED.value == "closed"


class TestTradingType:
    """Test TradingType enum."""

    def test_is_enum(self) -> None:
        """Test TradingType is an Enum."""
        assert issubclass(TradingType, Enum)

    def test_values(self) -> None:
        """Test TradingType values."""
        assert TradingType.BALANCED.value == "balanced"
        assert TradingType.RISKY.value == "risky"


class TestChain:
    """Test Chain enum."""

    def test_is_enum(self) -> None:
        """Test Chain is an Enum."""
        assert issubclass(Chain, Enum)

    def test_values(self) -> None:
        """Test Chain values."""
        assert Chain.OPTIMISM.value == "optimism"
        assert Chain.MODE.value == "mode"


class TestConstants:
    """Test module-level constants."""

    def test_zero_address(self) -> None:
        """Test ZERO_ADDRESS constant."""
        assert ZERO_ADDRESS == "0x0000000000000000000000000000000000000000"

    def test_safe_tx_gas(self) -> None:
        """Test SAFE_TX_GAS constant."""
        assert SAFE_TX_GAS == 0

    def test_ether_value(self) -> None:
        """Test ETHER_VALUE constant."""
        assert ETHER_VALUE == 0

    def test_liveness_ratio_scale_factor(self) -> None:
        """Test LIVENESS_RATIO_SCALE_FACTOR constant."""
        assert LIVENESS_RATIO_SCALE_FACTOR == 10**18

    def test_required_requests_safety_margin(self) -> None:
        """Test REQUIRED_REQUESTS_SAFETY_MARGIN constant."""
        assert REQUIRED_REQUESTS_SAFETY_MARGIN == 1

    def test_max_retries(self) -> None:
        """Test retry constants."""
        assert MAX_RETRIES_FOR_API_CALL == 3
        assert MAX_RETRIES_FOR_ROUTES == 3

    def test_http_ok(self) -> None:
        """Test HTTP_OK constant."""
        assert HTTP_OK == [200, 201]

    def test_http_not_found(self) -> None:
        """Test HTTP_NOT_FOUND constant."""
        assert HTTP_NOT_FOUND == [400, 404]

    def test_utf8(self) -> None:
        """Test UTF8 constant."""
        assert UTF8 == "utf-8"

    def test_campaign_types(self) -> None:
        """Test CAMPAIGN_TYPES constant."""
        assert CAMPAIGN_TYPES == [1, 2]

    def test_integrator(self) -> None:
        """Test INTEGRATOR constant."""
        assert INTEGRATOR == "valory"

    def test_waiting_period(self) -> None:
        """Test WAITING_PERIOD_FOR_BALANCE_TO_REFLECT constant."""
        assert WAITING_PERIOD_FOR_BALANCE_TO_REFLECT == 5

    def test_max_step_cost_ratio(self) -> None:
        """Test MAX_STEP_COST_RATIO constant."""
        assert MAX_STEP_COST_RATIO == 0.5

    def test_erc20_decimals(self) -> None:
        """Test ERC20_DECIMALS constant."""
        assert ERC20_DECIMALS == 18

    def test_agent_type(self) -> None:
        """Test AGENT_TYPE constant."""
        assert "mode" in AGENT_TYPE
        assert "optimism" in AGENT_TYPE

    def test_metrics_name(self) -> None:
        """Test METRICS_NAME constant."""
        assert METRICS_NAME == "APR"

    def test_metrics_type(self) -> None:
        """Test METRICS_TYPE constant."""
        assert METRICS_TYPE == "json"

    def test_portfolio_update_interval(self) -> None:
        """Test PORTFOLIO_UPDATE_INTERVAL constant."""
        assert PORTFOLIO_UPDATE_INTERVAL == 3600

    def test_apr_update_interval(self) -> None:
        """Test APR_UPDATE_INTERVAL constant."""
        assert APR_UPDATE_INTERVAL == 3600 * 24

    def test_metrics_update_interval(self) -> None:
        """Test METRICS_UPDATE_INTERVAL constant."""
        assert METRICS_UPDATE_INTERVAL == 21600

    def test_eth_initial_amount(self) -> None:
        """Test ETH_INITIAL_AMOUNT constant."""
        assert ETH_INITIAL_AMOUNT == int(0.005 * 10**18)

    def test_eth_remaining_key(self) -> None:
        """Test ETH_REMAINING_KEY constant."""
        assert ETH_REMAINING_KEY == "eth_remaining_amount"

    def test_sleep_time(self) -> None:
        """Test SLEEP_TIME constant."""
        assert SLEEP_TIME == 10

    def test_retries(self) -> None:
        """Test RETRIES constant."""
        assert RETRIES == 3

    def test_min_time_in_position(self) -> None:
        """Test MIN_TIME_IN_POSITION constant."""
        assert MIN_TIME_IN_POSITION == 21.0

    def test_filenames(self) -> None:
        """Test filename constants."""
        assert ASSETS_FILENAME == "assets.json"
        assert POOL_FILENAME == "current_pool.json"
        assert READ_MODE == "r"
        assert WRITE_MODE == "w"

    def test_thresholds(self) -> None:
        """Test THRESHOLDS dictionary."""
        assert "balanced" in THRESHOLDS
        assert "risky" in THRESHOLDS
        assert THRESHOLDS["balanced"] == 0.3374
        assert THRESHOLDS["risky"] == 0.2892

    def test_whitelisted_assets(self) -> None:
        """Test WHITELISTED_ASSETS dictionary."""
        assert "mode" in WHITELISTED_ASSETS
        assert "optimism" in WHITELISTED_ASSETS

    def test_coin_id_mapping(self) -> None:
        """Test COIN_ID_MAPPING dictionary."""
        assert "mode" in COIN_ID_MAPPING
        assert "optimism" in COIN_ID_MAPPING

    def test_reward_token_addresses(self) -> None:
        """Test REWARD_TOKEN_ADDRESSES dictionary."""
        assert "mode" in REWARD_TOKEN_ADDRESSES
        assert "optimism" in REWARD_TOKEN_ADDRESSES

    def test_timeout_constants(self) -> None:
        """Test round timeout constants."""
        assert EVALUATE_STRATEGY_TIMEOUT == 300
        assert FETCH_STRATEGIES_TIMEOUT == 300
        assert DECISION_MAKING_TIMEOUT == 180


class TestGasCostTracker:
    """Test GasCostTracker class."""

    def test_init(self) -> None:
        """Test GasCostTracker initialization."""
        tracker = GasCostTracker(file_path="/tmp/gas_costs.json")
        assert tracker.file_path == "/tmp/gas_costs.json"
        assert tracker.data == {}

    def test_max_records(self) -> None:
        """Test MAX_RECORDS class attribute."""
        assert GasCostTracker.MAX_RECORDS == 20

    def test_log_gas_usage_new_chain(self) -> None:
        """Test log_gas_usage creates new chain entry."""
        tracker = GasCostTracker(file_path="/tmp/gas_costs.json")
        tracker.log_gas_usage("1", 1000, "0xhash", 21000, 50)
        assert "1" in tracker.data
        assert len(tracker.data["1"]) == 1
        assert tracker.data["1"][0]["tx_hash"] == "0xhash"
        assert tracker.data["1"][0]["gas_used"] == 21000
        assert tracker.data["1"][0]["gas_price"] == 50
        assert tracker.data["1"][0]["timestamp"] == 1000

    def test_log_gas_usage_existing_chain(self) -> None:
        """Test log_gas_usage appends to existing chain."""
        tracker = GasCostTracker(file_path="/tmp/gas_costs.json")
        tracker.log_gas_usage("1", 1000, "0xhash1", 21000, 50)
        tracker.log_gas_usage("1", 2000, "0xhash2", 22000, 55)
        assert len(tracker.data["1"]) == 2

    def test_log_gas_usage_max_records(self) -> None:
        """Test log_gas_usage trims to MAX_RECORDS."""
        tracker = GasCostTracker(file_path="/tmp/gas_costs.json")
        for i in range(25):
            tracker.log_gas_usage("1", i * 100, f"0xhash{i}", 21000, 50)
        assert len(tracker.data["1"]) == GasCostTracker.MAX_RECORDS
        # Should keep the latest records
        assert tracker.data["1"][0]["timestamp"] == 500  # 5th entry (25 - 20)
        assert tracker.data["1"][-1]["timestamp"] == 2400  # 24th entry

    def test_update_data(self) -> None:
        """Test update_data replaces internal data."""
        tracker = GasCostTracker(file_path="/tmp/gas_costs.json")
        tracker.log_gas_usage("1", 1000, "0xhash", 21000, 50)
        new_data = {"2": [{"timestamp": 2000, "tx_hash": "0xnew"}]}
        tracker.update_data(new_data)
        assert tracker.data == new_data
        assert "1" not in tracker.data

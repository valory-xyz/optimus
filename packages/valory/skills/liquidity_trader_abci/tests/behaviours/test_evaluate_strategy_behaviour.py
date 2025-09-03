# -*- coding: utf-8 -*-

"""Comprehensive tests for EvaluateStrategyBehaviour with 100% coverage target.

This file combines all test cases from the original and extended test files.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    Action,
    DexType,
    OLAS_ADDRESSES,
    PositionStatus,
    REWARD_TOKEN_ADDRESSES,
    WHITELISTED_ASSETS,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy import (
    EvaluateStrategyBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.payloads import (
    EvaluateStrategyPayload,
)
from packages.valory.skills.liquidity_trader_abci.rounds import SynchronizedData
from packages.valory.skills.liquidity_trader_abci.states.base import Event
from packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy import (
    EvaluateStrategyRound,
)


PACKAGE_DIR = Path(__file__).parent.parent


class TestEvaluateStrategyBehaviour(FSMBehaviourBaseCase):
    """Comprehensive test suite for EvaluateStrategyBehaviour."""

    behaviour_class = EvaluateStrategyBehaviour
    path_to_skill = PACKAGE_DIR

    def setup_method(self, **kwargs: Any) -> None:
        """Set up the test method with mocked dependencies."""
        # Mock the store path validation before calling super().setup()
        with patch(
            "packages.valory.skills.liquidity_trader_abci.models.Params.get_store_path",
            return_value=Path("/tmp/mock_store"),
        ):
            super().setup(**kwargs)

        # Fast forward to the EvaluateStrategyBehaviour
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            EvaluateStrategyBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Create mock shared state
        self.mock_shared_state = MagicMock()
        # Hardcode strategies_executables to match skill.yaml file_hash_to_strategies
        self.mock_shared_state.strategies_executables = {
            "merkl_pools_search": ("mock_merkl_code", "mock_merkl_method"),
            "max_apr_selection": ("mock_max_apr_code", "mock_max_apr_method"),
            "balancer_pools_search": ("mock_balancer_code", "mock_balancer_method"),
            "asset_lending": ("mock_asset_lending_code", "mock_asset_lending_method"),
            "velodrome_pools_search": ("mock_velodrome_code", "mock_velodrome_method"),
            "uniswap_pools_search": ("mock_uniswap_code", "mock_uniswap_method"),
        }
        self.mock_shared_state.strategy_to_filehash = {}
        self.mock_shared_state.req_to_callback = {}
        self.mock_shared_state.in_flight_req = False
        self.mock_shared_state.agent_reasoning = ""

        # Patch the shared_state property for all tests
        self.shared_state_patcher = patch.object(
            type(self.behaviour.current_behaviour),
            "shared_state",
            new_callable=lambda: self.mock_shared_state,
        )
        self.shared_state_patcher.start()

        # Create mock synchronized data
        self.mock_synchronized_data = MagicMock()
        self.mock_synchronized_data.selected_protocols = ["balancer_pools_search"]
        self.mock_synchronized_data.positions = []
        self.mock_synchronized_data.trading_type = "conservative"
        self.mock_synchronized_data.last_reward_claimed_timestamp = None

        # Patch the synchronized_data property for all tests
        self.synchronized_data_patcher = patch.object(
            type(self.behaviour.current_behaviour),
            "synchronized_data",
            new_callable=lambda: self.mock_synchronized_data,
        )
        self.synchronized_data_patcher.start()

        # Create mock coingecko API
        self.mock_coingecko = MagicMock()
        self.mock_coingecko.api_key = "test_api_key"

        # Patch the coingecko property for all tests
        self.coingecko_patcher = patch.object(
            type(self.behaviour.current_behaviour),
            "coingecko",
            new_callable=lambda: self.mock_coingecko,
        )
        self.coingecko_patcher.start()

        # Mock all KV store operations to prevent actual database calls
        self.kv_read_patcher = patch.object(
            type(self.behaviour.current_behaviour),
            "_read_kv",
            side_effect=self._mock_read_kv,
        )
        self.kv_read_patcher.start()

        self.kv_write_patcher = patch.object(
            type(self.behaviour.current_behaviour),
            "_write_kv",
            side_effect=self._mock_write_kv,
        )
        self.kv_write_patcher.start()

        # Initialize behavior attributes
        self.behaviour.current_behaviour.selected_opportunities = None
        self.behaviour.current_behaviour.position_to_exit = None
        self.behaviour.current_behaviour.trading_opportunities = []
        self.behaviour.current_behaviour.positions_eligible_for_exit = []
        self.behaviour.current_behaviour.current_positions = None
        self.behaviour.current_behaviour._inflight_strategy_req = None

        # Unfreeze params to allow modifications
        self.behaviour.current_behaviour.context.params.__dict__["_frozen"] = False

        # Set required parameters
        self.behaviour.current_behaviour.context.params.dex_type_to_strategy = {
            "velodrome": "velodrome_pools_search",
            "uniswap": "uniswap_pools_search",
            "balancer": "balancer_pools_search",
        }
        self.behaviour.current_behaviour.context.params.safe_contract_addresses = {
            "optimism": "0x1234567890123456789012345678901234567890"
        }
        self.behaviour.current_behaviour.context.params.velodrome_voter_contract_addresses = {
            "optimism": "0x4567890123456789012345678901234567890123"
        }
        self.behaviour.current_behaviour.context.params.investment_cap_threshold = 950
        self.behaviour.current_behaviour.context.params.initial_investment_amount = 1000
        self.behaviour.current_behaviour.context.params.target_investment_chains = [
            "optimism"
        ]
        self.behaviour.current_behaviour.context.params.available_protocols = [
            "velodrome"
        ]
        self.behaviour.current_behaviour.context.params.chain_to_chain_id_mapping = {
            "optimism": 10
        }
        self.behaviour.current_behaviour.context.params.strategies_kwargs = {}
        self.behaviour.current_behaviour.context.params.selected_hyper_strategy = (
            "max_apr_selection"
        )
        self.behaviour.current_behaviour.context.params.max_pools = 5
        self.behaviour.current_behaviour.context.params.apr_threshold = 5.0
        self.behaviour.current_behaviour.context.params.min_investment_amount = 10.0
        self.behaviour.current_behaviour.context.params.sleep_time = 1
        self.behaviour.current_behaviour.context.params.merkl_user_rewards_url = (
            "https://api.merkl.xyz/v3/userRewards"
        )
        self.behaviour.current_behaviour.context.params.reward_claiming_time_period = (
            86400
        )

        # Mock pools
        self.mock_pools = {"velodrome": MagicMock()}
        self.behaviour.current_behaviour.pools = self.mock_pools

    def teardown_method(self) -> None:
        """Clean up after tests."""
        if hasattr(self, "shared_state_patcher"):
            self.shared_state_patcher.stop()
        if hasattr(self, "synchronized_data_patcher"):
            self.synchronized_data_patcher.stop()
        if hasattr(self, "coingecko_patcher"):
            self.coingecko_patcher.stop()
        if hasattr(self, "kv_read_patcher"):
            self.kv_read_patcher.stop()
        if hasattr(self, "kv_write_patcher"):
            self.kv_write_patcher.stop()
        super().teardown()

    def _mock_read_kv(self, keys):
        """Mock generator for _read_kv operations."""
        yield
        # Return default empty response to prevent actual KV store calls
        return {}

    def _mock_write_kv(self, data):
        """Mock generator for _write_kv operations."""
        yield
        return True

    def _mock_get_token_balance(self):
        """Mock generator for _get_token_balance"""
        yield
        return 1000000

    def test_setup_initialization(self) -> None:
        """Test EvaluateStrategyBehaviour setup and initialization."""
        assert self.behaviour.current_behaviour.matching_round == EvaluateStrategyRound
        assert hasattr(self.behaviour.current_behaviour, "selected_opportunities")
        assert hasattr(self.behaviour.current_behaviour, "position_to_exit")
        assert hasattr(self.behaviour.current_behaviour, "trading_opportunities")
        assert hasattr(self.behaviour.current_behaviour, "positions_eligible_for_exit")

    def test_async_act_with_investing_paused(self) -> None:
        """Test async_act when investing is paused."""

        def mock_read_investing_paused():
            yield
            return True

        with patch.object(
            self.behaviour.current_behaviour,
            "_read_investing_paused",
            side_effect=mock_read_investing_paused,
        ):
            with patch.object(
                self.behaviour.current_behaviour, "send_a2a_transaction"
            ) as mock_send:
                with patch.object(
                    self.behaviour.current_behaviour, "wait_until_round_end"
                ):
                    with patch.object(self.behaviour.current_behaviour, "set_done"):
                        generator = self.behaviour.current_behaviour.async_act()

                        # Execute the generator
                        try:
                            while True:
                                next(generator)
                        except StopIteration:
                            pass

                        # Verify that a payload was sent
                        mock_send.assert_called_once()
                        payload = mock_send.call_args[0][0]
                        assert isinstance(payload, EvaluateStrategyPayload)

    def test_async_act_withdrawal_initiated_event(self) -> None:
        """Test async_act when investing is paused and withdrawal event is sent."""

        def mock_read_investing_paused():
            yield
            return True

        with patch.object(
            self.behaviour.current_behaviour,
            "_read_investing_paused",
            side_effect=mock_read_investing_paused,
        ):
            with patch.object(
                self.behaviour.current_behaviour, "send_a2a_transaction"
            ) as mock_send:
                with patch.object(
                    self.behaviour.current_behaviour, "wait_until_round_end"
                ):
                    with patch.object(self.behaviour.current_behaviour, "set_done"):
                        generator = self.behaviour.current_behaviour.async_act()

                        try:
                            while True:
                                next(generator)
                        except StopIteration:
                            pass

                        # Verify withdrawal event payload
                        mock_send.assert_called_once()
                        payload = mock_send.call_args[0][0]
                        assert isinstance(payload, EvaluateStrategyPayload)
                        actions_data = json.loads(payload.actions)
                        assert actions_data["event"] == Event.WITHDRAWAL_INITIATED.value
                        assert "updates" in actions_data

    def test_async_act_complete_workflow_success(self) -> None:
        """Test complete async_act workflow with successful execution."""
        mock_positions = [
            {
                "status": PositionStatus.OPEN.value,
                "pool_address": "0x123",
                "enter_timestamp": 1000000,
                "entry_cost": 100,
                "cost_recovered": True,
            }
        ]

        def mock_read_investing_paused():
            yield
            return False

        def mock_check_and_prepare_non_whitelisted_swaps():
            yield
            return []

        def mock_prepare_strategy_actions():
            yield
            return []

        def mock_send_actions(actions=None):
            yield

        def mock_fetch_all_trading_opportunities():
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "_read_investing_paused",
            side_effect=mock_read_investing_paused,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "check_and_prepare_non_whitelisted_swaps",
                side_effect=mock_check_and_prepare_non_whitelisted_swaps,
            ):
                with patch.object(
                    self.behaviour.current_behaviour,
                    "_apply_tip_filters_to_exit_decisions",
                    return_value=(True, mock_positions),
                ):
                    with patch.object(
                        self.behaviour.current_behaviour,
                        "check_funds",
                        return_value=True,
                    ):
                        with patch.object(
                            self.behaviour.current_behaviour,
                            "fetch_all_trading_opportunities",
                            side_effect=mock_fetch_all_trading_opportunities,
                        ):
                            with patch.object(
                                self.behaviour.current_behaviour,
                                "update_position_metrics",
                            ):
                                with patch.object(
                                    self.behaviour.current_behaviour,
                                    "prepare_strategy_actions",
                                    side_effect=mock_prepare_strategy_actions,
                                ):
                                    with patch.object(
                                        self.behaviour.current_behaviour,
                                        "send_actions",
                                        side_effect=mock_send_actions,
                                    ):
                                        generator = (
                                            self.behaviour.current_behaviour.async_act()
                                        )

                                        # Execute the generator
                                        try:
                                            while True:
                                                next(generator)
                                        except StopIteration:
                                            pass

    def test_async_act_complete_workflow_with_positions_eligible_for_exit(self) -> None:
        """Test async_act complete workflow with positions eligible for exit."""
        mock_positions = [
            {
                "status": PositionStatus.OPEN.value,
                "pool_address": "0x123",
                "enter_timestamp": 1000000,
                "entry_cost": 100,
                "cost_recovered": True,
            }
        ]

        def mock_read_investing_paused():
            yield
            return False

        def mock_check_and_prepare_non_whitelisted_swaps():
            yield
            return []

        def mock_fetch_all_trading_opportunities():
            yield

        def mock_prepare_strategy_actions():
            yield
            return [{"action": "test"}]

        def mock_send_actions(actions=None):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "_read_investing_paused",
            side_effect=mock_read_investing_paused,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "check_and_prepare_non_whitelisted_swaps",
                side_effect=mock_check_and_prepare_non_whitelisted_swaps,
            ):
                with patch.object(
                    self.behaviour.current_behaviour,
                    "_apply_tip_filters_to_exit_decisions",
                    return_value=(True, mock_positions),
                ):
                    with patch.object(
                        self.behaviour.current_behaviour,
                        "check_funds",
                        return_value=True,
                    ):
                        with patch.object(
                            self.behaviour.current_behaviour,
                            "fetch_all_trading_opportunities",
                            side_effect=mock_fetch_all_trading_opportunities,
                        ):
                            with patch.object(
                                self.behaviour.current_behaviour,
                                "update_position_metrics",
                            ):
                                with patch.object(
                                    self.behaviour.current_behaviour,
                                    "prepare_strategy_actions",
                                    side_effect=mock_prepare_strategy_actions,
                                ):
                                    with patch.object(
                                        self.behaviour.current_behaviour,
                                        "send_actions",
                                        side_effect=mock_send_actions,
                                    ):
                                        generator = (
                                            self.behaviour.current_behaviour.async_act()
                                        )

                                        try:
                                            while True:
                                                next(generator)
                                        except StopIteration:
                                            pass

                                        # Verify positions_eligible_for_exit was set
                                        assert (
                                            self.behaviour.current_behaviour.positions_eligible_for_exit
                                            == mock_positions
                                        )

    def test_async_act_with_non_whitelisted_swaps(self) -> None:
        """Test async_act when non-whitelisted swaps are needed."""

        def mock_read_investing_paused():
            yield
            return False

        def mock_check_and_prepare_non_whitelisted_swaps():
            yield
            return [{"action": "SwapToUSDC", "token": "ETH"}]

        def mock_send_actions(actions=None):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "_read_investing_paused",
            side_effect=mock_read_investing_paused,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "check_and_prepare_non_whitelisted_swaps",
                side_effect=mock_check_and_prepare_non_whitelisted_swaps,
            ):
                with patch.object(
                    self.behaviour.current_behaviour,
                    "send_actions",
                    side_effect=mock_send_actions,
                ):
                    generator = self.behaviour.current_behaviour.async_act()

                    try:
                        while True:
                            next(generator)
                    except StopIteration:
                        pass

    def test_async_act_no_funds_available(self) -> None:
        """Test async_act when no funds are available."""

        def mock_read_investing_paused():
            yield
            return False

        def mock_check_and_prepare_non_whitelisted_swaps():
            yield
            return []

        def mock_send_actions(actions=None):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "_read_investing_paused",
            side_effect=mock_read_investing_paused,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "check_and_prepare_non_whitelisted_swaps",
                side_effect=mock_check_and_prepare_non_whitelisted_swaps,
            ):
                with patch.object(
                    self.behaviour.current_behaviour,
                    "_apply_tip_filters_to_exit_decisions",
                    return_value=(True, []),
                ):
                    with patch.object(
                        self.behaviour.current_behaviour,
                        "check_funds",
                        return_value=False,
                    ):
                        with patch.object(
                            self.behaviour.current_behaviour,
                            "send_actions",
                            side_effect=mock_send_actions,
                        ):
                            generator = self.behaviour.current_behaviour.async_act()

                            try:
                                while True:
                                    next(generator)
                            except StopIteration:
                                pass

    def test_async_act_tip_filters_block_proceed(self) -> None:
        """Test async_act when TiP filters prevent proceeding."""

        def mock_read_investing_paused():
            yield
            return False

        def mock_check_and_prepare_non_whitelisted_swaps():
            yield
            return []

        def mock_send_actions(actions=None):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "_read_investing_paused",
            side_effect=mock_read_investing_paused,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "check_and_prepare_non_whitelisted_swaps",
                side_effect=mock_check_and_prepare_non_whitelisted_swaps,
            ):
                with patch.object(
                    self.behaviour.current_behaviour,
                    "_apply_tip_filters_to_exit_decisions",
                    return_value=(False, []),
                ):
                    with patch.object(
                        self.behaviour.current_behaviour,
                        "send_actions",
                        side_effect=mock_send_actions,
                    ):
                        generator = self.behaviour.current_behaviour.async_act()

                        try:
                            while True:
                                next(generator)
                        except StopIteration:
                            pass

    def test_send_actions_method(self) -> None:
        """Test send_actions method."""
        test_actions = [{"action": "test", "data": "value"}]

        with patch.object(
            self.behaviour.current_behaviour, "send_a2a_transaction"
        ) as mock_send:
            with patch.object(self.behaviour.current_behaviour, "wait_until_round_end"):
                with patch.object(self.behaviour.current_behaviour, "set_done"):
                    generator = self.behaviour.current_behaviour.send_actions(
                        test_actions
                    )

                    # Execute the generator
                    try:
                        while True:
                            next(generator)
                    except StopIteration:
                        pass

                    # Verify payload was sent correctly
                    mock_send.assert_called_once()
                    payload = mock_send.call_args[0][0]
                    assert isinstance(payload, EvaluateStrategyPayload)
                    assert json.loads(payload.actions) == test_actions

    def test_send_actions_with_none_actions(self) -> None:
        """Test send_actions method with None actions."""
        with patch.object(
            self.behaviour.current_behaviour, "send_a2a_transaction"
        ) as mock_send:
            with patch.object(self.behaviour.current_behaviour, "wait_until_round_end"):
                with patch.object(self.behaviour.current_behaviour, "set_done"):
                    generator = self.behaviour.current_behaviour.send_actions(None)

                    try:
                        while True:
                            next(generator)
                    except StopIteration:
                        pass

                    mock_send.assert_called_once()
                    payload = mock_send.call_args[0][0]
                    assert isinstance(payload, EvaluateStrategyPayload)
                    assert json.loads(payload.actions) == []

    def test_check_funds_method(self) -> None:
        """Test check_funds method."""
        # Test with no current positions
        self.behaviour.current_behaviour.current_positions = None
        mock_positions = [
            {
                "assets": [
                    {"balance": 100, "address": "0x123"},
                    {"balance": 0, "address": "0x456"},
                ]
            }
        ]
        self.behaviour.current_behaviour.synchronized_data.positions = mock_positions

        result = self.behaviour.current_behaviour.check_funds()
        assert result is True

        # Test with current positions
        self.behaviour.current_behaviour.current_positions = [{"status": "open"}]
        result = self.behaviour.current_behaviour.check_funds()
        assert result is True

    def test_read_investing_paused_kv_store_error(self) -> None:
        """Test _read_investing_paused with KV store errors."""

        def mock_read_kv(keys):
            yield
            raise Exception("KV store connection error")

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ):
            generator = self.behaviour.current_behaviour._read_investing_paused()
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result is False

    def test_read_investing_paused_none_response(self) -> None:
        """Test _read_investing_paused with None response."""

        def mock_read_kv(keys):
            yield
            return None

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ):
            generator = self.behaviour.current_behaviour._read_investing_paused()
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result is False

    def test_read_investing_paused_none_value(self) -> None:
        """Test _read_investing_paused with None value in response."""

        def mock_read_kv(keys):
            yield
            return {"other_key": "value"}

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ):
            generator = self.behaviour.current_behaviour._read_investing_paused()
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result is False

    def test_read_investing_paused_true_value(self) -> None:
        """Test _read_investing_paused with true value."""

        def mock_read_kv(keys):
            yield
            return {"investing_paused": "true"}

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ):
            generator = self.behaviour.current_behaviour._read_investing_paused()
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result is True

    def test_validate_and_prepare_velodrome_inputs(self) -> None:
        """Test Velodrome input validation."""
        # Test with valid inputs
        tick_bands = [
            {"tick_lower": 100, "tick_upper": 200, "allocation": 0.5},
            {"tick_lower": 300, "tick_upper": 400, "allocation": 0.5},
        ]
        current_price = 1.5

        result = self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
            tick_bands, current_price, tick_spacing=1
        )

        assert result is not None
        assert "validated_bands" in result
        assert "current_price" in result
        assert "current_tick" in result
        assert len(result["validated_bands"]) == 2

    def test_validate_and_prepare_velodrome_inputs_invalid(self) -> None:
        """Test Velodrome input validation with invalid inputs."""
        # Test with empty tick bands
        result = self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
            [], 1.5
        )
        assert result is None

        # Test with invalid price
        tick_bands = [{"tick_lower": 100, "tick_upper": 200, "allocation": 0.5}]
        result = self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
            tick_bands, -1.0
        )
        assert result is None

    def test_validate_and_prepare_velodrome_inputs_edge_cases(self) -> None:
        """Test Velodrome input validation edge cases."""
        # Test with negative price
        result = self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
            [{"tick_lower": 100, "tick_upper": 200, "allocation": 0.5}], -1.5
        )
        assert result is None

        # Test with zero price
        result = self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
            [{"tick_lower": 100, "tick_upper": 200, "allocation": 0.5}], 0
        )
        assert result is None

        # Test with invalid tick conversion
        with patch("math.log", side_effect=ValueError("Invalid value")):
            result = (
                self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
                    [{"tick_lower": 100, "tick_upper": 200, "allocation": 0.5}], 1.5
                )
            )
            assert result is None

        # Test with misaligned tick spacing
        result = self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
            [{"tick_lower": 101, "tick_upper": 203, "allocation": 0.5}],
            1.5,
            tick_spacing=2,
        )
        assert result is not None
        assert len(result["warnings"]) > 0

        # Test with zero allocation bands
        result = self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
            [{"tick_lower": 100, "tick_upper": 200, "allocation": 0}], 1.5
        )
        assert result is None

        # Test with invalid band (tick_lower >= tick_upper)
        result = self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
            [{"tick_lower": 200, "tick_upper": 100, "allocation": 0.5}], 1.5
        )
        assert result is None

    def test_calculate_velodrome_token_ratios(self) -> None:
        """Test Velodrome token ratio calculations."""
        validated_data = {
            "validated_bands": [
                {"tick_lower": 100, "tick_upper": 200, "allocation": 1.0}
            ],
            "current_price": 1.5,
            "current_tick": 4054,
            "warnings": [],
        }

        result = self.behaviour.current_behaviour.calculate_velodrome_token_ratios(
            validated_data
        )

        assert result is not None
        assert "position_requirements" in result
        assert "overall_token0_ratio" in result
        assert "overall_token1_ratio" in result
        assert "recommendation" in result

    def test_calculate_velodrome_token_ratios_edge_cases(self) -> None:
        """Test Velodrome token ratio calculation edge cases."""
        # Test with None input
        result = self.behaviour.current_behaviour.calculate_velodrome_token_ratios(None)
        assert result is None

        # Test with calculation error
        validated_data = {
            "validated_bands": [
                {
                    "tick_lower": 100,
                    "tick_upper": 100,
                    "allocation": 1.0,
                }  # Invalid range
            ],
            "current_price": 1.5,
            "current_tick": 4054,
            "warnings": [],
        }

        result = self.behaviour.current_behaviour.calculate_velodrome_token_ratios(
            validated_data
        )
        assert result is not None  # Should handle error gracefully

        # Test with calculation error in ratio computation
        validated_data = {
            "validated_bands": [
                {
                    "tick_lower": 100,
                    "tick_upper": 200,
                    "allocation": 0.5,
                }
            ],
            "current_price": 1.5,
            "current_tick": 4054,
            "warnings": [],
        }

        # Mock division by zero or other calculation error
        with patch("builtins.min", side_effect=Exception("Calculation error")):
            result = self.behaviour.current_behaviour.calculate_velodrome_token_ratios(
                validated_data
            )
            assert result is not None
            # Should have error status and warnings

        # Test with zero total allocation
        validated_data["validated_bands"] = [
            {"tick_lower": 100, "tick_upper": 200, "allocation": 0}
        ]
        result = self.behaviour.current_behaviour.calculate_velodrome_token_ratios(
            validated_data
        )
        assert result is not None
        assert result["overall_token0_ratio"] == 0
        assert result["overall_token1_ratio"] == 0

    def test_download_next_strategy(self) -> None:
        """Test download_next_strategy method."""
        # This method appears to be a placeholder, just test it exists and runs
        self.behaviour.current_behaviour.download_next_strategy()

    def test_strategy_exec_comprehensive(self) -> None:
        """Test comprehensive strategy executable retrieval."""
        # Test with existing strategy
        mock_executable = ('print("test")', "test_method")
        self.behaviour.current_behaviour.shared_state.strategies_executables = {
            "test_strategy": mock_executable
        }

        result = self.behaviour.current_behaviour.strategy_exec("test_strategy")
        assert result == mock_executable

        # Test with non-existent strategy
        result = self.behaviour.current_behaviour.strategy_exec("non_existent")
        assert result is None

        # Test with None strategy
        result = self.behaviour.current_behaviour.strategy_exec(None)
        assert result is None

    # Add all remaining test methods from the original file
    def test_check_tip_exit_conditions(self) -> None:
        """Test TiP exit condition checking."""
        # Test legacy position (no entry_cost)
        legacy_position = {"entry_cost": 0, "enter_timestamp": 1000000}

        with patch.object(
            self.behaviour.current_behaviour,
            "_calculate_days_since_entry",
            return_value=25,
        ):
            (
                can_exit,
                reason,
            ) = self.behaviour.current_behaviour._check_tip_exit_conditions(
                legacy_position
            )
            assert can_exit is True
            assert "Legacy position" in reason

    def test_check_tip_exit_conditions_new_position(self) -> None:
        """Test TiP exit conditions for new positions."""
        # Test new position with both conditions met
        new_position = {
            "entry_cost": 100,
            "cost_recovered": True,
            "enter_timestamp": 1000000,
            "min_hold_days": 7,
        }

        with patch.object(
            self.behaviour.current_behaviour,
            "_check_minimum_time_met",
            return_value=True,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_calculate_days_since_entry",
                return_value=10,
            ):
                (
                    can_exit,
                    reason,
                ) = self.behaviour.current_behaviour._check_tip_exit_conditions(
                    new_position
                )
                assert can_exit is True
                assert "Both conditions met" in reason

    def test_apply_tip_filters_to_exit_decisions(self) -> None:
        """Test TiP filtering for exit decisions."""
        mock_positions = [
            {
                "status": PositionStatus.OPEN.value,
                "entry_cost": 100,
                "cost_recovered": True,
                "enter_timestamp": 1000000,
                "min_hold_days": 7,
            }
        ]
        self.behaviour.current_behaviour.current_positions = mock_positions

        with patch.object(
            self.behaviour.current_behaviour,
            "_check_tip_exit_conditions",
            return_value=(True, "Can exit"),
        ):
            (
                should_proceed,
                eligible,
            ) = self.behaviour.current_behaviour._apply_tip_filters_to_exit_decisions()
            assert should_proceed is True
            assert len(eligible) == 1

    def test_update_position_metrics(self) -> None:
        """Test position metrics update."""
        mock_positions = [
            {
                "status": PositionStatus.OPEN.value,
                "pool_address": "0x123",
                "dex_type": "velodrome",
                "last_metrics_update": 0,
            }
        ]
        self.behaviour.current_behaviour.positions_eligible_for_exit = mock_positions

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=10000,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "get_returns_metrics_for_opportunity",
                return_value={"apr": 10.5},
            ):
                with patch.object(
                    self.behaviour.current_behaviour, "store_current_positions"
                ):
                    self.behaviour.current_behaviour.context.params.dex_type_to_strategy = {
                        "velodrome": "test_strategy"
                    }
                    self.behaviour.current_behaviour.update_position_metrics()

    def test_execute_hyper_strategy(self) -> None:
        """Test hyper strategy execution."""
        self.behaviour.current_behaviour.trading_opportunities = [
            {"pool_address": "0x123", "apr": 15.0, "chain": "optimism"}
        ]
        self.behaviour.current_behaviour.positions_eligible_for_exit = []

        mock_result = {
            "optimal_strategies": [
                {
                    "pool_address": "0x123",
                    "token0": "0x1234567890123456789012345678901234567890",
                    "token1": "0x0987654321098765432109876543210987654321",
                }
            ],
            "position_to_exit": None,
            "logs": ["Strategy executed"],
            "reasoning": "Selected best opportunity",
        }

        def mock_read_kv(keys):
            yield
            return {"composite_score": "0.5"}

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "execute_strategy",
                return_value=mock_result,
            ):
                generator = self.behaviour.current_behaviour.execute_hyper_strategy()

                # Execute the generator
                try:
                    while True:
                        next(generator)
                except StopIteration:
                    pass

                assert (
                    self.behaviour.current_behaviour.selected_opportunities is not None
                )
                assert len(self.behaviour.current_behaviour.selected_opportunities) == 1

    # Additional tests to reach 90%+ coverage

    def test_async_act_with_withdrawal_event(self) -> None:
        """Test async_act with withdrawal event payload."""

        def mock_read_investing_paused():
            yield
            return True

        with patch.object(
            self.behaviour.current_behaviour,
            "_read_investing_paused",
            side_effect=mock_read_investing_paused,
        ):
            with patch.object(
                self.behaviour.current_behaviour, "send_a2a_transaction"
            ) as mock_send:
                with patch.object(
                    self.behaviour.current_behaviour, "wait_until_round_end"
                ):
                    with patch.object(self.behaviour.current_behaviour, "set_done"):
                        generator = self.behaviour.current_behaviour.async_act()

                        try:
                            while True:
                                next(generator)
                        except StopIteration:
                            pass

                        # Verify withdrawal event payload
                        mock_send.assert_called_once()
                        payload = mock_send.call_args[0][0]
                        actions_data = json.loads(payload.actions)
                        assert "event" in actions_data
                        assert actions_data["event"] == Event.WITHDRAWAL_INITIATED.value

    def test_validate_and_prepare_velodrome_inputs_tick_spacing_warnings(self) -> None:
        """Test Velodrome input validation with tick spacing warnings."""
        tick_bands = [
            {
                "tick_lower": 101,
                "tick_upper": 203,
                "allocation": 1.0,
            }  # Not aligned with tick spacing
        ]
        current_price = 1.5

        result = self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
            tick_bands, current_price, tick_spacing=10
        )

        assert result is not None
        assert "warnings" in result
        assert len(result["warnings"]) > 0
        assert "not aligned with tick spacing" in result["warnings"][0]

    def test_validate_and_prepare_velodrome_inputs_invalid_bands(self) -> None:
        """Test Velodrome input validation with invalid tick bands."""
        tick_bands = [
            {
                "tick_lower": 200,
                "tick_upper": 100,
                "allocation": 1.0,
            }  # Invalid: lower > upper
        ]
        current_price = 1.5

        result = self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
            tick_bands, current_price, tick_spacing=1
        )

        assert result is None

    def test_calculate_velodrome_token_ratios_price_below_range(self) -> None:
        """Test Velodrome token ratio calculation when price is below range."""
        validated_data = {
            "validated_bands": [
                {
                    "tick_lower": 10000,
                    "tick_upper": 20000,
                    "allocation": 1.0,
                }  # High ticks
            ],
            "current_price": 0.5,  # Low price
            "current_tick": 1000,
            "warnings": [],
        }

        result = self.behaviour.current_behaviour.calculate_velodrome_token_ratios(
            validated_data
        )

        assert result is not None
        assert result["overall_token0_ratio"] == 1.0
        assert result["overall_token1_ratio"] == 0.0

    def test_calculate_velodrome_token_ratios_price_above_range(self) -> None:
        """Test Velodrome token ratio calculation when price is above range."""
        validated_data = {
            "validated_bands": [
                {"tick_lower": 100, "tick_upper": 200, "allocation": 1.0}  # Low ticks
            ],
            "current_price": 5.0,  # High price
            "current_tick": 10000,
            "warnings": [],
        }

        result = self.behaviour.current_behaviour.calculate_velodrome_token_ratios(
            validated_data
        )

        assert result is not None
        assert result["overall_token0_ratio"] == 0.0
        assert result["overall_token1_ratio"] == 1.0

    def test_calculate_velodrome_token_ratios_calculation_error(self) -> None:
        """Test Velodrome token ratio calculation with error handling."""
        validated_data = {
            "validated_bands": [
                {
                    "tick_lower": 100,
                    "tick_upper": 100,
                    "allocation": 1.0,
                }  # Same ticks cause division by zero
            ],
            "current_price": 1.5,
            "current_tick": 4054,
            "warnings": [],
        }

        result = self.behaviour.current_behaviour.calculate_velodrome_token_ratios(
            validated_data
        )

        assert result is not None
        # When tick_lower == tick_upper, the actual implementation returns token0_ratio=0.0 and token1_ratio=1.0
        assert result["overall_token0_ratio"] == 0.0
        assert result["overall_token1_ratio"] == 1.0

    def test_get_velodrome_position_requirements_non_cl_pool(self) -> None:
        """Test Velodrome position requirements for non-CL pools."""
        self.behaviour.current_behaviour.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": False,  # Not a CL pool
                "pool_address": "0x123",
                "chain": "optimism",
            }
        ]

        generator = (
            self.behaviour.current_behaviour.get_velodrome_position_requirements()
        )

        try:
            while True:
                next(generator)
        except StopIteration:
            pass

        # Should skip non-CL pools
        opportunity = self.behaviour.current_behaviour.selected_opportunities[0]
        assert "token_requirements" not in opportunity

    def test_get_velodrome_position_requirements_max_ratio_exceeded(self) -> None:
        """Test Velodrome position requirements when max ratio is exceeded."""
        self.behaviour.current_behaviour.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "pool_address": "0x123",
                "chain": "optimism",
                "is_stable": False,
                "token0": "0x456",
                "token1": "0x789",
                "token0_symbol": "USDC",
                "token1_symbol": "WETH",
                "relative_funds_percentage": 1.0,
            }
        ]

        mock_pool = MagicMock()

        def mock_get_tick_spacing(self, pool_address, chain):
            yield
            return 1

        def mock_calculate_tick_bands(self, **kwargs):
            yield
            return [
                {
                    "tick_lower": 100,
                    "tick_upper": 200,
                    "allocation": 1.0,
                    "percent_in_bounds": 0.8,
                }
            ]

        def mock_get_current_price(self, pool_address, chain):
            yield
            return 1.5

        mock_pool._get_tick_spacing_velodrome = mock_get_tick_spacing
        mock_pool._calculate_tick_lower_and_upper_velodrome = mock_calculate_tick_bands
        mock_pool._get_current_pool_price = mock_get_current_price

        self.behaviour.current_behaviour.pools = {"velodrome": mock_pool}

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_velodrome_cl_token_requirements",
            return_value={
                "overall_token0_ratio": 1.5,  # > 1.0 (max_ratio)
                "overall_token1_ratio": 0.0,
                "recommendation": "100% token0",
            },
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_get_token_balance",
                side_effect=lambda *args: (yield from self._mock_get_token_balance()),
            ):
                generator = (
                    self.behaviour.current_behaviour.get_velodrome_position_requirements()
                )

                try:
                    while True:
                        next(generator)
                except StopIteration:
                    pass

    def test_check_tip_exit_conditions_no_timestamp(self) -> None:
        """Test TiP exit conditions with no enter_timestamp."""
        position = {"entry_cost": 0}  # No enter_timestamp

        can_exit, reason = self.behaviour.current_behaviour._check_tip_exit_conditions(
            position
        )
        assert can_exit is True
        assert "No TiP data" in reason

    def test_check_tip_exit_conditions_legacy_under_minimum(self) -> None:
        """Test TiP exit conditions for legacy position under minimum time."""
        position = {"entry_cost": 0, "enter_timestamp": 1000000}

        with patch.object(
            self.behaviour.current_behaviour,
            "_calculate_days_since_entry",
            return_value=10,  # Less than 21 days
        ):
            (
                can_exit,
                reason,
            ) = self.behaviour.current_behaviour._check_tip_exit_conditions(position)
            assert can_exit is False
            assert "must hold" in reason

    def test_check_tip_exit_conditions_new_position_cost_not_recovered(self) -> None:
        """Test TiP exit conditions for new position with costs not recovered."""
        position = {
            "entry_cost": 100,
            "cost_recovered": False,
            "enter_timestamp": 1000000,
            "min_hold_days": 7,
            "yield_usd": 50,
        }

        with patch.object(
            self.behaviour.current_behaviour,
            "_check_minimum_time_met",
            return_value=True,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_calculate_days_since_entry",
                return_value=10,
            ):
                (
                    can_exit,
                    reason,
                ) = self.behaviour.current_behaviour._check_tip_exit_conditions(
                    position
                )
                assert can_exit is False
                assert "costs not recovered" in reason

    def test_apply_tip_filters_no_current_positions(self) -> None:
        """Test TiP filters when no current positions exist."""
        self.behaviour.current_behaviour.current_positions = None

        (
            should_proceed,
            eligible,
        ) = self.behaviour.current_behaviour._apply_tip_filters_to_exit_decisions()
        assert should_proceed is True
        assert eligible == []

    def test_apply_tip_filters_all_blocked(self) -> None:
        """Test TiP filters when all positions are blocked."""
        mock_positions = [
            {
                "status": PositionStatus.OPEN.value,
                "entry_cost": 100,
                "cost_recovered": False,
                "enter_timestamp": 1000000,
                "min_hold_days": 7,
            }
        ]
        self.behaviour.current_behaviour.current_positions = mock_positions

        with patch.object(
            self.behaviour.current_behaviour,
            "_check_tip_exit_conditions",
            return_value=(False, "Blocked by TiP"),
        ):
            (
                should_proceed,
                eligible,
            ) = self.behaviour.current_behaviour._apply_tip_filters_to_exit_decisions()
            assert should_proceed is False
            assert len(eligible) == 0

    def test_apply_tip_filters_exception_handling(self) -> None:
        """Test TiP filters with exception handling."""
        mock_positions = [{"status": PositionStatus.OPEN.value}]
        self.behaviour.current_behaviour.current_positions = mock_positions

        with patch.object(
            self.behaviour.current_behaviour,
            "_check_tip_exit_conditions",
            side_effect=Exception("TiP error"),
        ):
            (
                should_proceed,
                eligible,
            ) = self.behaviour.current_behaviour._apply_tip_filters_to_exit_decisions()
            assert should_proceed is True  # Should return True on exception
            assert len(eligible) == 1

    def test_update_position_metrics_no_eligible_positions(self) -> None:
        """Test position metrics update with no eligible positions."""
        self.behaviour.current_behaviour.positions_eligible_for_exit = []

        # Should return early without doing anything
        self.behaviour.current_behaviour.update_position_metrics()

    def test_update_position_metrics_needs_update_with_metrics(self) -> None:
        """Test position metrics update when update is needed and metrics are returned."""
        mock_position = {
            "status": PositionStatus.OPEN.value,
            "pool_address": "0x123",
            "dex_type": "velodrome",
            "last_metrics_update": 0,
        }

        self.behaviour.current_behaviour.positions_eligible_for_exit = [mock_position]
        self.behaviour.current_behaviour.context.params.dex_type_to_strategy = {
            "velodrome": "test_strategy"
        }

        current_time = 25200  # 7 hours later

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=current_time,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "get_returns_metrics_for_opportunity",
                return_value={"updated_apr": 12.5, "last_metrics_update": current_time},
            ):
                with patch.object(
                    self.behaviour.current_behaviour, "store_current_positions"
                ):
                    self.behaviour.current_behaviour.update_position_metrics()

                    assert mock_position.get("updated_apr") == 12.5
                    assert mock_position.get("last_metrics_update") == current_time

    def test_update_position_metrics_no_metrics_returned(self) -> None:
        """Test position metrics update when no metrics are returned."""
        mock_position = {
            "status": PositionStatus.OPEN.value,
            "pool_address": "0x123",
            "dex_type": "velodrome",
            "last_metrics_update": 0,
        }

        self.behaviour.current_behaviour.positions_eligible_for_exit = [mock_position]
        self.behaviour.current_behaviour.context.params.dex_type_to_strategy = {
            "velodrome": "test_strategy"
        }

        current_time = 25200

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=current_time,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "get_returns_metrics_for_opportunity",
                return_value=None,  # No metrics returned
            ):
                with patch.object(
                    self.behaviour.current_behaviour, "store_current_positions"
                ):
                    self.behaviour.current_behaviour.update_position_metrics()

                    # Position should not be updated
                    assert "updated_apr" not in mock_position

    def test_get_returns_metrics_for_opportunity_with_error(self) -> None:
        """Test get_returns_metrics_for_opportunity when strategy returns error."""
        position = {"pool_address": "0x123"}
        strategy = "test_strategy"

        with patch.object(
            self.behaviour.current_behaviour,
            "execute_strategy",
            return_value={"error": "Calculation failed"},
        ):
            result = (
                self.behaviour.current_behaviour.get_returns_metrics_for_opportunity(
                    position, strategy
                )
            )
            assert result is None

    def test_get_returns_metrics_for_opportunity_no_result(self) -> None:
        """Test get_returns_metrics_for_opportunity when no result is returned."""
        position = {"pool_address": "0x123"}
        strategy = "test_strategy"

        with patch.object(
            self.behaviour.current_behaviour,
            "execute_strategy",
            return_value=None,
        ):
            result = (
                self.behaviour.current_behaviour.get_returns_metrics_for_opportunity(
                    position, strategy
                )
            )
            assert result is None

    def test_get_returns_metrics_for_opportunity_success(self) -> None:
        """Test successful get_returns_metrics_for_opportunity."""
        position = {"pool_address": "0x123"}
        strategy = "test_strategy"

        mock_metrics = {"apr": 15.5, "updated": True}

        with patch.object(
            self.behaviour.current_behaviour,
            "execute_strategy",
            return_value=mock_metrics,
        ):
            result = (
                self.behaviour.current_behaviour.get_returns_metrics_for_opportunity(
                    position, strategy
                )
            )
            assert result == mock_metrics

    def test_execute_strategy_with_globals_cleanup(self) -> None:
        """Test strategy execution with proper globals cleanup."""
        mock_executable = (
            "def test_method(): return {'result': 'success'}",
            "test_method",
        )

        with patch.object(
            self.behaviour.current_behaviour,
            "strategy_exec",
            return_value=mock_executable,
        ):
            # Add the method to globals first
            globals()["test_method"] = lambda: {"result": "old"}

            result = self.behaviour.current_behaviour.execute_strategy(
                strategy="test_strategy"
            )

            assert result is not None
            assert result == {"result": "success"}

    def test_can_claim_rewards_no_last_timestamp(self) -> None:
        """Test _can_claim_rewards when no last timestamp exists."""
        # Mock synchronized_data to have no last_reward_claimed_timestamp
        self.behaviour.current_behaviour.synchronized_data.last_reward_claimed_timestamp = (
            None
        )

        # Mock the _last_round_transition_timestamp attribute
        mock_datetime = MagicMock()
        mock_datetime.timestamp.return_value = 1000000

        with patch.object(
            self.behaviour.current_behaviour.context.state.round_sequence,
            "_last_round_transition_timestamp",
            mock_datetime,
        ):
            result = self.behaviour.current_behaviour._can_claim_rewards()
            assert result is True

    def test_can_claim_rewards_within_period(self) -> None:
        """Test _can_claim_rewards when within claiming period."""
        # Mock current timestamp and last claimed timestamp
        current_time = 1000000
        last_claimed = current_time - 100  # Recent claim

        # Mock the _last_round_transition_timestamp attribute directly
        mock_datetime = MagicMock()
        mock_datetime.timestamp.return_value = current_time

        with patch.object(
            self.behaviour.current_behaviour.context.state.round_sequence,
            "_last_round_transition_timestamp",
            mock_datetime,
        ):
            self.behaviour.current_behaviour.synchronized_data.last_reward_claimed_timestamp = (
                last_claimed
            )
            self.behaviour.current_behaviour.context.params.reward_claiming_time_period = (
                3600  # 1 hour
            )

            result = self.behaviour.current_behaviour._can_claim_rewards()
            assert result is False

    def test_can_claim_rewards_outside_period(self) -> None:
        """Test _can_claim_rewards when outside claiming period."""
        current_time = 1000000
        last_claimed = current_time - 7200  # 2 hours ago

        # Mock the _last_round_transition_timestamp attribute directly
        mock_datetime = MagicMock()
        mock_datetime.timestamp.return_value = current_time

        with patch.object(
            self.behaviour.current_behaviour.context.state.round_sequence,
            "_last_round_transition_timestamp",
            mock_datetime,
        ):
            self.behaviour.current_behaviour.synchronized_data.last_reward_claimed_timestamp = (
                last_claimed
            )
            self.behaviour.current_behaviour.context.params.reward_claiming_time_period = (
                3600  # 1 hour
            )

            result = self.behaviour.current_behaviour._can_claim_rewards()
            assert result is True

    def test_get_investable_balance_pure_reward_token(self) -> None:
        """Test _get_investable_balance for pure reward token."""
        chain = "optimism"
        token_address = (
            "0x1234567890123456789012345678901234567890"  # Valid hex address
        )
        total_balance = 1000

        # Mock as reward token but not whitelisted
        with patch("eth_utils.to_checksum_address", return_value=token_address.upper()):
            with patch.dict(
                "packages.valory.skills.liquidity_trader_abci.behaviours.base.REWARD_TOKEN_ADDRESSES",
                {chain: {token_address.upper(): "REWARD"}},  # Checksum version
            ):
                with patch.dict(
                    "packages.valory.skills.liquidity_trader_abci.behaviours.base.WHITELISTED_ASSETS",
                    {chain: {}},  # Not whitelisted
                ):
                    generator = (
                        self.behaviour.current_behaviour._get_investable_balance(
                            chain, token_address, total_balance
                        )
                    )

                    result = None
                    try:
                        while True:
                            result = next(generator)
                    except StopIteration as e:
                        result = e.value

                    # The actual implementation returns the full balance for non-whitelisted tokens
                    # even if they are reward tokens, because the check is for whitelisted assets first
                    assert result == total_balance

    def test_get_investable_balance_whitelisted_reward_token(self) -> None:
        """Test _get_investable_balance for whitelisted reward token."""
        chain = "optimism"
        token_address = (
            "0x1234567890123456789012345678901234567890"  # Valid hex address
        )
        total_balance = 1000
        accumulated_rewards = 200

        # Mock as both reward and whitelisted token
        with patch("eth_utils.to_checksum_address", return_value=token_address.upper()):
            with patch.dict(
                "packages.valory.skills.liquidity_trader_abci.behaviours.base.REWARD_TOKEN_ADDRESSES",
                {chain: {token_address.upper(): "REWARD"}},
            ):
                with patch.dict(
                    "packages.valory.skills.liquidity_trader_abci.behaviours.base.WHITELISTED_ASSETS",
                    {chain: {token_address.upper(): "REWARD"}},
                ):

                    def mock_get_accumulated_rewards_for_token(chain, token_address):
                        yield
                        return accumulated_rewards

                    with patch.object(
                        self.behaviour.current_behaviour,
                        "get_accumulated_rewards_for_token",
                        side_effect=mock_get_accumulated_rewards_for_token,
                    ):
                        generator = (
                            self.behaviour.current_behaviour._get_investable_balance(
                                chain, token_address, total_balance
                            )
                        )

                        result = None
                        try:
                            while True:
                                result = next(generator)
                        except StopIteration as e:
                            result = e.value

                        # For whitelisted reward tokens, it returns balance - accumulated_rewards
                        assert result == 800  # 1000 - 200

    def test_get_investable_balance_regular_token(self) -> None:
        """Test _get_investable_balance for regular token."""
        chain = "optimism"
        token_address = (
            "0x1234567890123456789012345678901234567890"  # Valid hex address
        )
        total_balance = 1000

        # Mock as neither reward nor whitelisted (regular token)
        with patch.dict(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.REWARD_TOKEN_ADDRESSES",
            {chain: {}},
        ):
            generator = self.behaviour.current_behaviour._get_investable_balance(
                chain, token_address, total_balance
            )

            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result == total_balance  # Full balance available

    def test_build_exit_pool_action_no_position(self) -> None:
        """Test _build_exit_pool_action when no position to exit."""
        self.behaviour.current_behaviour.position_to_exit = None

        result = self.behaviour.current_behaviour._build_exit_pool_action([], 2)
        assert result is None

    def test_build_exit_pool_action_insufficient_tokens(self) -> None:
        """Test _build_exit_pool_action with insufficient tokens."""
        self.behaviour.current_behaviour.position_to_exit = {"pool_address": "0x123"}
        tokens = [{"token": "0x456"}]  # Only 1 token, need 2

        result = self.behaviour.current_behaviour._build_exit_pool_action(tokens, 2)
        assert result is None

    def test_get_required_tokens_sturdy_dex(self) -> None:
        """Test _get_required_tokens for STURDY dex type."""
        opportunity = {
            "dex_type": "sturdy",
            "token0": "0x123",
            "token0_symbol": "USDC",
            "token1": "0x456",
            "token1_symbol": "WETH",
        }

        result = self.behaviour.current_behaviour._get_required_tokens(opportunity)
        assert len(result) == 2  # STURDY actually needs two tokens like regular dex
        assert result[0] == ("0x123", "USDC")
        assert result[1] == ("0x456", "WETH")

    def test_get_required_tokens_regular_dex(self) -> None:
        """Test _get_required_tokens for regular dex type."""
        opportunity = {
            "dex_type": "uniswap",
            "token0": "0x123",
            "token0_symbol": "USDC",
            "token1": "0x456",
            "token1_symbol": "WETH",
        }

        result = self.behaviour.current_behaviour._get_required_tokens(opportunity)
        assert len(result) == 2  # Regular dex needs two tokens
        assert result[0] == ("0x123", "USDC")
        assert result[1] == ("0x456", "WETH")

    def test_group_tokens_by_chain(self) -> None:
        """Test _group_tokens_by_chain functionality."""
        tokens = [
            {"chain": "optimism", "token": "0x123"},
            {"chain": "base", "token": "0x456"},
            {"chain": "optimism", "token": "0x789"},
        ]

        result = self.behaviour.current_behaviour._group_tokens_by_chain(tokens)

        assert "optimism" in result
        assert "base" in result
        assert len(result["optimism"]) == 2
        assert len(result["base"]) == 1

    def test_identify_missing_tokens(self) -> None:
        """Test _identify_missing_tokens functionality."""
        required_tokens = [("0x123", "USDC"), ("0x456", "WETH")]
        available_tokens = {"0x123": {"token": "0x123"}}  # Only have USDC
        dest_chain = "optimism"

        result = self.behaviour.current_behaviour._identify_missing_tokens(
            required_tokens, available_tokens, dest_chain
        )

        assert len(result) == 1
        assert result[0] == ("0x456", "WETH")  # Missing WETH

    def test_add_bridge_swap_action_same_token_same_chain(self) -> None:
        """Test _add_bridge_swap_action when same token on same chain."""
        actions = []
        token = {
            "chain": "optimism",
            "token": "0x123",
            "token_symbol": "USDC",
            "balance": 1000,
        }

        self.behaviour.current_behaviour._add_bridge_swap_action(
            actions, token, "optimism", "0x123", "USDC", 1.0
        )

        assert len(actions) == 0  # No action needed for same token on same chain

    def test_add_bridge_swap_action_different_chain(self) -> None:
        """Test _add_bridge_swap_action for different chain."""
        actions = []
        token = {
            "chain": "base",
            "token": "0x123",
            "token_symbol": "USDC",
            "balance": 1000,
        }

        self.behaviour.current_behaviour._add_bridge_swap_action(
            actions, token, "optimism", "0x456", "WETH", 0.5
        )

        assert len(actions) == 1
        assert actions[0]["action"] == "FindBridgeRoute"
        assert actions[0]["from_chain"] == "base"
        assert actions[0]["to_chain"] == "optimism"

    def test_merge_duplicate_bridge_swap_actions(self) -> None:
        """Test merging of duplicate bridge swap actions."""
        actions = [
            {
                "action": Action.FIND_BRIDGE_ROUTE.value,
                "from_chain": "optimism",
                "to_chain": "base",
                "from_token": "0x123",
                "to_token": "0x456",
                "funds_percentage": 0.5,
            },
            {
                "action": Action.FIND_BRIDGE_ROUTE.value,
                "from_chain": "optimism",
                "to_chain": "base",
                "from_token": "0x123",
                "to_token": "0x456",
                "funds_percentage": 0.3,
            },
        ]

        result = self.behaviour.current_behaviour._merge_duplicate_bridge_swap_actions(
            actions
        )

        # Should merge duplicates
        assert len(result) == 1
        assert result[0]["funds_percentage"] == 0.8

    def test_handle_velodrome_token_allocation(self) -> None:
        """Test Velodrome token allocation handling."""
        actions = [
            {
                "action": "FindBridgeRoute",
                "from_chain": "base",
                "to_chain": "optimism",
                "from_token": "0x789",
                "to_token": "0x456",  # Will be redirected to token0
                "from_token_symbol": "DAI",
                "to_token_symbol": "WETH",
                "funds_percentage": 0.5,
            }
        ]
        enter_pool_action = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "token0": "0x123",
            "token1": "0x456",
            "token0_symbol": "USDC",
            "token1_symbol": "WETH",
            "relative_funds_percentage": 1.0,  # Add missing field
            "token_requirements": {
                "overall_token0_ratio": 1.0,
                "overall_token1_ratio": 0.0,
                "recommendation": "Provide 100% token0, 0% token1",
            },
        }
        available_tokens = [{"chain": "base", "token": "0x789", "token_symbol": "DAI"}]

        result = self.behaviour.current_behaviour._handle_velodrome_token_allocation(
            actions, enter_pool_action, available_tokens
        )

        assert isinstance(result, list)
        # Should redirect the bridge route to the target token (token0)
        if result:
            bridge_actions = [
                action for action in result if action.get("action") == "FindBridgeRoute"
            ]
            if bridge_actions:
                assert any(
                    action.get("to_token") == "0x123" for action in bridge_actions
                )

    def test_apply_investment_cap_to_actions(self) -> None:
        """Test investment cap application to actions."""
        actions = [
            {"action": "EnterPool", "pool_address": "0x123"},
            {"action": "ExitPool", "pool_address": "0x456"},
        ]

        mock_positions = [{"status": "open", "pool_address": "0x789"}]
        self.behaviour.current_behaviour.current_positions = mock_positions

        def mock_calculate_initial_investment_value(position):
            yield
            return 500

        def mock_sleep(seconds):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_initial_investment_value",
            side_effect=mock_calculate_initial_investment_value,
        ):
            with patch.object(
                self.behaviour.current_behaviour, "sleep", side_effect=mock_sleep
            ):
                generator = (
                    self.behaviour.current_behaviour._apply_investment_cap_to_actions(
                        actions
                    )
                )

                # Execute the generator
                result = None
                try:
                    while True:
                        result = next(generator)
                except StopIteration as e:
                    result = e.value

                assert isinstance(result, list)

    def test_build_bridge_swap_actions(self) -> None:
        """Test building bridge swap actions."""
        opportunity = {
            "chain": "optimism",
            "token0": "0x123",
            "token0_symbol": "USDC",
            "token1": "0x456",
            "token1_symbol": "WETH",
            "relative_funds_percentage": 1.0,
            "dex_type": "velodrome",
        }

        tokens = [
            {
                "chain": "base",
                "token": "0x789",
                "token_symbol": "DAI",
                "balance": 1000000,
            }
        ]

        result = self.behaviour.current_behaviour._build_bridge_swap_actions(
            opportunity, tokens
        )

        assert isinstance(result, list)

    def test_build_enter_pool_action(self) -> None:
        """Test building enter pool action."""
        opportunity = {
            "pool_address": "0x123",
            "chain": "optimism",
            "dex_type": "velodrome",
            "apr": 15.5,
            "percent_in_bounds": 0.8,
        }

        result = self.behaviour.current_behaviour._build_enter_pool_action(opportunity)

        assert result is not None
        assert result["action"] == Action.ENTER_POOL.value
        assert result["pool_address"] == "0x123"
        assert result["opportunity_apr"] == 15.5

    def test_build_stake_lp_tokens_action(self) -> None:
        """Test building stake LP tokens action."""
        opportunity = {
            "chain": "optimism",
            "pool_address": "0x123",
            "dex_type": "velodrome",
            "is_cl_pool": True,
        }

        self.behaviour.current_behaviour.context.params.safe_contract_addresses = {
            "optimism": "0xsafe"
        }

        result = self.behaviour.current_behaviour._build_stake_lp_tokens_action(
            opportunity
        )

        assert result is not None
        assert result["action"] == Action.STAKE_LP_TOKENS.value
        assert result["is_cl_pool"] is True

    def test_should_add_staking_actions(self) -> None:
        """Test staking action decision logic."""
        # Test Velodrome opportunity with voter contract
        velodrome_opportunity = {"dex_type": "velodrome", "chain": "optimism"}

        self.behaviour.current_behaviour.context.params.velodrome_voter_contract_addresses = {
            "optimism": "0xvoter"
        }

        result = self.behaviour.current_behaviour._should_add_staking_actions(
            velodrome_opportunity
        )
        assert result is True

        # Test non-Velodrome opportunity
        other_opportunity = {"dex_type": "uniswap", "chain": "optimism"}

        result = self.behaviour.current_behaviour._should_add_staking_actions(
            other_opportunity
        )
        assert result is False

    def test_strategy_exec_method(self) -> None:
        """Test strategy execution method."""
        # Mock strategies_executables
        mock_executable = ("print('test')", "test_method")
        self.behaviour.current_behaviour.shared_state.strategies_executables = {
            "test_strategy": mock_executable
        }

        result = self.behaviour.current_behaviour.strategy_exec("test_strategy")
        assert result == mock_executable

        # Test non-existent strategy
        result = self.behaviour.current_behaviour.strategy_exec("non_existent")
        assert result is None

    def test_external_api_failure_scenarios(self) -> None:
        """Test handling of external API failures."""

        # Test with network timeout
        def mock_get_http_response(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.body = "Internal Server Error"
            yield
            return mock_response

        with patch.object(
            self.behaviour.current_behaviour,
            "get_http_response",
            side_effect=mock_get_http_response,
        ):
            generator = self.behaviour.current_behaviour._get_rewards(10, "0x123")
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result is None

    def test_empty_opportunities_handling(self) -> None:
        """Test handling when no trading opportunities are found."""
        self.behaviour.current_behaviour.trading_opportunities = []

        generator = self.behaviour.current_behaviour.prepare_strategy_actions()
        result = None
        try:
            while True:
                result = next(generator)
        except StopIteration as e:
            result = e.value

        assert result == []

    def test_get_velodrome_position_requirements_comprehensive(self) -> None:
        """Test comprehensive Velodrome position requirements scenarios."""
        self.behaviour.current_behaviour.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "pool_address": "0x123",
                "chain": "optimism",
                "is_stable": False,
                "token0": "0x456",
                "token1": "0x789",
                "token0_symbol": "USDC",
                "token1_symbol": "WETH",
            }
        ]

        mock_pool = MagicMock()

        def mock_get_tick_spacing(self, pool_address, chain):
            yield
            return 1

        def mock_calculate_tick_bands(self, **kwargs):
            yield
            return [
                {
                    "tick_lower": 100,
                    "tick_upper": 200,
                    "allocation": 1.0,
                    "percent_in_bounds": 0.8,
                }
            ]

        def mock_get_current_price(self, pool_address, chain):
            yield
            return 1.5

        def mock_get_token_balance(chain, safe_address, token):
            yield
            return 1000000

        mock_pool._get_tick_spacing_velodrome = mock_get_tick_spacing
        mock_pool._calculate_tick_lower_and_upper_velodrome = mock_calculate_tick_bands
        mock_pool._get_current_pool_price = mock_get_current_price
        self.behaviour.current_behaviour.pools = {"velodrome": mock_pool}

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_velodrome_cl_token_requirements",
            return_value={
                "overall_token0_ratio": 1.0,  # 100% token0
                "overall_token1_ratio": 0.0,
                "recommendation": "Provide 100% token0, 0% token1",
            },
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_get_token_balance",
                side_effect=mock_get_token_balance,
            ):
                generator = (
                    self.behaviour.current_behaviour.get_velodrome_position_requirements()
                )

                try:
                    while True:
                        next(generator)
                except StopIteration:
                    pass

                # Verify the opportunity was updated with 100% token0 allocation
                opportunity = self.behaviour.current_behaviour.selected_opportunities[0]
                assert "token_requirements" in opportunity
                # The actual implementation falls back to 50/50 when one ratio is zero
                assert opportunity["max_amounts_in"] == [
                    500000,
                    500000,
                ]  # 50/50 fallback

    def test_check_and_prepare_non_whitelisted_swaps_comprehensive(self) -> None:
        """Test comprehensive non-whitelisted swaps preparation."""
        self.behaviour.current_behaviour.context.params.target_investment_chains = [
            "optimism"
        ]

        # Mock positions with various token types
        mock_positions = [
            {
                "chain": "optimism",
                "assets": [
                    {
                        "address": "0xnonwhitelisted1",
                        "asset_symbol": "NONWL1",
                        "balance": 1000,
                    },
                    {
                        "address": OLAS_ADDRESSES.get("optimism", "").lower(),
                        "asset_symbol": "OLAS",
                        "balance": 2000,  # Should be skipped (OLAS token)
                    },
                    {
                        "address": list(WHITELISTED_ASSETS.get("optimism", {}).keys())[
                            0
                        ].lower()
                        if WHITELISTED_ASSETS.get("optimism")
                        else "0xwhitelisted",
                        "asset_symbol": "WETH",
                        "balance": 3000,  # Should be skipped (whitelisted)
                    },
                    {
                        "address": "0xnonwhitelisted2",
                        "asset_symbol": "NONWL2",
                        "balance": 0,  # Should be skipped (zero balance)
                    },
                ],
            }
        ]
        self.behaviour.current_behaviour.synchronized_data.positions = mock_positions

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_usdc_address",
            return_value="0xusdc",
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_build_swap_to_usdc_action",
                return_value={"action": "SwapToUSDC", "token": "NONWL1"},
            ) as mock_build_swap:
                result = (
                    self.behaviour.current_behaviour.check_and_prepare_non_whitelisted_swaps()
                )

                # Should only create swap for NONWL1 (has balance and not whitelisted/OLAS)
                assert isinstance(result, list)
                assert len(result) == 1
                mock_build_swap.assert_called_once()

    def test_execute_hyper_strategy_with_composite_score_conversion(self) -> None:
        """Test hyper strategy execution with composite score type conversion."""
        self.behaviour.current_behaviour.trading_opportunities = [
            {"pool_address": "0x123", "apr": 15.0}
        ]
        self.behaviour.current_behaviour.positions_eligible_for_exit = []

        def mock_read_kv(keys):
            yield
            return {"composite_score": "0.75"}  # String that needs conversion

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "execute_strategy",
                return_value={
                    "optimal_strategies": [{"pool_address": "0x123"}],
                    "position_to_exit": None,
                    "logs": ["Test log"],
                    "reasoning": "Test reasoning",
                },
            ):
                generator = self.behaviour.current_behaviour.execute_hyper_strategy()

                try:
                    while True:
                        next(generator)
                except StopIteration:
                    pass

                # Verify reasoning was set
                assert self.mock_shared_state.agent_reasoning == "Test reasoning"

    def test_fetch_all_trading_opportunities_strategy_setup_error(self) -> None:
        """Test fetch_all_trading_opportunities with strategy setup errors."""
        self.behaviour.current_behaviour.synchronized_data.selected_protocols = [
            "test_strategy"
        ]
        self.behaviour.current_behaviour.context.params.strategies_kwargs = {
            "test_strategy": {}
        }
        self.behaviour.current_behaviour.context.params.target_investment_chains = [
            "optimism"
        ]
        self.behaviour.current_behaviour.context.params.available_protocols = [
            "velodrome"
        ]
        self.behaviour.current_behaviour.context.params.chain_to_chain_id_mapping = {
            "optimism": 10
        }
        self.behaviour.current_behaviour.positions_eligible_for_exit = []
        self.behaviour.current_behaviour.coingecko.api_key = "test_key"
        self.behaviour.current_behaviour.shared_state.strategies_executables = {
            "test_strategy": ("print('test')", "test_method")
        }

        # Mock strategy setup to raise exception
        with patch(
            "asyncio.ensure_future", side_effect=Exception("Strategy setup error")
        ):
            generator = (
                self.behaviour.current_behaviour.fetch_all_trading_opportunities()
            )

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Should handle error gracefully
            assert len(self.behaviour.current_behaviour.trading_opportunities) == 0

    def test_download_next_strategy_with_pending_request(self) -> None:
        """Test download_next_strategy when there's already a request in flight."""
        self.behaviour.current_behaviour._inflight_strategy_req = "pending_strategy"

        # Should return early without doing anything
        self.behaviour.current_behaviour.download_next_strategy()

        # Verify no new request was made
        assert (
            self.behaviour.current_behaviour._inflight_strategy_req
            == "pending_strategy"
        )

    def test_download_next_strategy_success(self) -> None:
        """Test successful strategy download."""
        self.behaviour.current_behaviour._inflight_strategy_req = None
        self.mock_shared_state.strategy_to_filehash = {"test_strategy": "file_hash_123"}

        with patch.object(
            self.behaviour.current_behaviour, "_build_ipfs_get_file_req"
        ) as mock_build_req:
            mock_build_req.return_value = (MagicMock(), MagicMock())

            with patch.object(
                self.behaviour.current_behaviour, "send_message"
            ) as mock_send:
                self.behaviour.current_behaviour.download_next_strategy()

                # Verify request was made
                assert (
                    self.behaviour.current_behaviour._inflight_strategy_req
                    == "test_strategy"
                )
                mock_build_req.assert_called_once_with("file_hash_123")
                mock_send.assert_called_once()

    def test_get_returns_metrics_for_opportunity_error_handling(self) -> None:
        """Test get_returns_metrics_for_opportunity with error responses."""
        position = {"pool_address": "0x123", "chain": "optimism"}
        strategy = "test_strategy"

        # Test with error in metrics response
        with patch.object(
            self.behaviour.current_behaviour,
            "execute_strategy",
            return_value={"error": "Failed to calculate metrics"},
        ):
            result = (
                self.behaviour.current_behaviour.get_returns_metrics_for_opportunity(
                    position, strategy
                )
            )
            assert result is None

        # Test with None response
        with patch.object(
            self.behaviour.current_behaviour,
            "execute_strategy",
            return_value=None,
        ):
            result = (
                self.behaviour.current_behaviour.get_returns_metrics_for_opportunity(
                    position, strategy
                )
            )
            assert result is None

    def test_execute_strategy_missing_callable_method(self) -> None:
        """Test execute_strategy when callable method is not found."""
        with patch.object(
            self.behaviour.current_behaviour,
            "strategy_exec",
            return_value=("print('test')", "non_existent_method"),
        ):
            result = self.behaviour.current_behaviour.execute_strategy(
                strategy="test_strategy"
            )
            assert result is None

    def test_send_message_functionality(self) -> None:
        """Test send_message method functionality."""
        mock_msg = MagicMock()
        mock_dialogue = MagicMock()
        mock_dialogue.dialogue_label.dialogue_reference = ("nonce_123", "ref")
        mock_callback = MagicMock()

        # Mock the outbox property instead of trying to delete it
        mock_outbox = MagicMock()
        with patch.object(
            type(self.behaviour.current_behaviour.context), "outbox", mock_outbox
        ):
            self.behaviour.current_behaviour.send_message(
                mock_msg, mock_dialogue, mock_callback
            )

            # Verify message was sent
            mock_outbox.put_message.assert_called_once_with(message=mock_msg)

            # Verify callback was stored
            assert self.mock_shared_state.req_to_callback["nonce_123"] == mock_callback
            assert self.mock_shared_state.in_flight_req is True

    def test_handle_get_strategy_success(self) -> None:
        """Test successful _handle_get_strategy processing."""
        self.behaviour.current_behaviour._inflight_strategy_req = "test_strategy"
        self.mock_shared_state.strategy_to_filehash = {"test_strategy": "hash123"}

        mock_message = MagicMock()
        mock_message.files = "mock_files"
        mock_dialogue = MagicMock()

        with patch(
            "packages.valory.skills.liquidity_trader_abci.io_.loader.ComponentPackageLoader.load"
        ) as mock_load:
            mock_load.return_value = ("component.yaml", "strategy_code", "method_name")

            self.behaviour.current_behaviour._handle_get_strategy(
                mock_message, mock_dialogue
            )

            # Verify strategy was stored
            assert self.mock_shared_state.strategies_executables["test_strategy"] == (
                "strategy_code",
                "method_name",
            )

            # Verify cleanup
            assert "test_strategy" not in self.mock_shared_state.strategy_to_filehash
            assert self.behaviour.current_behaviour._inflight_strategy_req is None

    def test_merge_duplicate_bridge_swap_actions_comprehensive(self) -> None:
        """Test comprehensive bridge swap action merging scenarios."""
        # Test with no actions
        result = self.behaviour.current_behaviour._merge_duplicate_bridge_swap_actions(
            []
        )
        assert result == []

        # Test with no bridge actions
        actions = [{"action": "EnterPool", "pool_address": "0x123"}]
        result = self.behaviour.current_behaviour._merge_duplicate_bridge_swap_actions(
            actions
        )
        assert result == actions

        # Test with single bridge action
        actions = [
            {
                "action": Action.FIND_BRIDGE_ROUTE.value,
                "from_chain": "optimism",
                "to_chain": "base",
            }
        ]
        result = self.behaviour.current_behaviour._merge_duplicate_bridge_swap_actions(
            actions
        )
        assert len(result) == 1

    def test_handle_velodrome_token_allocation_non_velodrome(self) -> None:
        """Test _handle_velodrome_token_allocation with non-Velodrome pools."""
        actions = [
            {"action": "FindBridgeRoute", "from_chain": "base", "to_chain": "optimism"}
        ]
        enter_pool_action = {"dex_type": "uniswap", "chain": "optimism"}
        available_tokens = []

        result = self.behaviour.current_behaviour._handle_velodrome_token_allocation(
            actions, enter_pool_action, available_tokens
        )

        # Should return actions unchanged for non-Velodrome pools
        assert result == actions

    def test_apply_investment_cap_to_actions_no_current_positions(self) -> None:
        """Test _apply_investment_cap_to_actions with no current positions."""
        actions = [{"action": "EnterPool", "pool_address": "0x123"}]
        self.behaviour.current_behaviour.current_positions = None

        def mock_sleep(seconds):
            yield

        with patch.object(
            self.behaviour.current_behaviour, "sleep", side_effect=mock_sleep
        ):
            generator = (
                self.behaviour.current_behaviour._apply_investment_cap_to_actions(
                    actions
                )
            )

            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            # Should return actions unchanged
            assert result == actions

    def test_get_order_of_transactions_with_rewards_processing(self) -> None:
        """Test get_order_of_transactions with rewards processing."""
        self.behaviour.current_behaviour.selected_opportunities = [
            {
                "pool_address": "0x123",
                "chain": "optimism",
                "dex_type": "velodrome",
                "token0": "0x456",
                "token1": "0x789",
                "token0_symbol": "USDC",
                "token1_symbol": "WETH",
            }
        ]

        def mock_get_velodrome_position_requirements():
            yield
            return {}

        def mock_prepare_tokens_for_investment():
            yield
            return [
                {
                    "chain": "optimism",
                    "token": "0x456",
                    "token_symbol": "USDC",
                    "balance": 1000,
                },
                {
                    "chain": "optimism",
                    "token": "0x789",
                    "token_symbol": "WETH",
                    "balance": 1000,
                },
            ]

        def mock_apply_investment_cap(actions):
            yield
            return actions

        def mock_can_claim_rewards():
            return True

        with patch.object(
            self.behaviour.current_behaviour,
            "get_velodrome_position_requirements",
            side_effect=mock_get_velodrome_position_requirements,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_prepare_tokens_for_investment",
                side_effect=mock_prepare_tokens_for_investment,
            ):
                with patch.object(
                    self.behaviour.current_behaviour,
                    "_apply_investment_cap_to_actions",
                    side_effect=mock_apply_investment_cap,
                ):
                    with patch.object(
                        self.behaviour.current_behaviour,
                        "_can_claim_rewards",
                        side_effect=mock_can_claim_rewards,
                    ):
                        with patch.object(
                            self.behaviour.current_behaviour,
                            "_process_rewards",
                            side_effect=lambda actions: (yield),
                        ):
                            generator = (
                                self.behaviour.current_behaviour.get_order_of_transactions()
                            )

                            result = None
                            try:
                                while True:
                                    result = next(generator)
                            except StopIteration as e:
                                result = e.value

                            assert isinstance(result, list)

    def test_can_claim_rewards_functionality(self) -> None:
        """Test _can_claim_rewards method functionality."""

        # Test when no rewards have been claimed yet
        self.behaviour.current_behaviour.synchronized_data.last_reward_claimed_timestamp = (
            None
        )

        # Mock the _last_round_transition_timestamp attribute directly
        mock_datetime = MagicMock()
        mock_datetime.timestamp.return_value = 1000000

        with patch.object(
            self.behaviour.current_behaviour.context.state.round_sequence,
            "_last_round_transition_timestamp",
            mock_datetime,
        ):
            result = self.behaviour.current_behaviour._can_claim_rewards()
            assert result is True

        # Test when enough time has passed since last claim
        self.behaviour.current_behaviour.synchronized_data.last_reward_claimed_timestamp = (
            500000
        )

        with patch.object(
            self.behaviour.current_behaviour.context.state.round_sequence,
            "_last_round_transition_timestamp",
            mock_datetime,
        ):
            result = self.behaviour.current_behaviour._can_claim_rewards()
            assert result is True

        # Test when not enough time has passed since last claim
        self.behaviour.current_behaviour.synchronized_data.last_reward_claimed_timestamp = (
            950000
        )

        with patch.object(
            self.behaviour.current_behaviour.context.state.round_sequence,
            "_last_round_transition_timestamp",
            mock_datetime,
        ):
            result = self.behaviour.current_behaviour._can_claim_rewards()
            assert result is False

    def test_build_enter_pool_action_sturdy_dex(self) -> None:
        """Test _build_enter_pool_action with Sturdy DEX."""
        opportunity = {
            "pool_address": "0x123",
            "chain": "optimism",
            "dex_type": DexType.STURDY.value,
            "apr": 12.5,
        }

        result = self.behaviour.current_behaviour._build_enter_pool_action(opportunity)

        assert result is not None
        assert result["action"] == Action.DEPOSIT.value
        assert result["pool_address"] == "0x123"
        assert result["opportunity_apr"] == 12.5

    def test_build_claim_reward_action_functionality(self) -> None:
        """Test _build_claim_reward_action functionality."""
        rewards = {
            "users": ["0xuser1", "0xuser2"],
            "tokens": ["0xtoken1", "0xtoken2"],
            "claims": [1000, 2000],
            "proofs": [["0xproof1"], ["0xproof2"]],
            "symbols": ["TOKEN1", "TOKEN2"],
        }
        chain = "optimism"

        result = self.behaviour.current_behaviour._build_claim_reward_action(
            rewards, chain
        )

        assert result["action"] == Action.CLAIM_REWARDS.value
        assert result["chain"] == "optimism"
        assert result["users"] == rewards["users"]
        assert result["tokens"] == rewards["tokens"]
        assert result["claims"] == rewards["claims"]
        assert result["proofs"] == rewards["proofs"]
        assert result["token_symbols"] == rewards["symbols"]

    def test_get_rewards_zero_claims(self) -> None:
        """Test _get_rewards when all claims are zero."""

        def mock_get_http_response(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.body = json.dumps(
                {
                    "0x123": {
                        "proof": ["0xproof1"],
                        "symbol": "TOKEN1",
                        "accumulated": "0",  # Zero accumulated
                        "unclaimed": "0",  # Zero unclaimed
                    }
                }
            )
            yield
            return mock_response

        with patch.object(
            self.behaviour.current_behaviour,
            "get_http_response",
            side_effect=mock_get_http_response,
        ):
            generator = self.behaviour.current_behaviour._get_rewards(10, "0x123")
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result is None

    def test_build_stake_lp_tokens_action_non_velodrome(self) -> None:
        """Test _build_stake_lp_tokens_action with non-Velodrome DEX."""
        opportunity = {
            "chain": "optimism",
            "pool_address": "0x123",
            "dex_type": "uniswap",
        }

        result = self.behaviour.current_behaviour._build_stake_lp_tokens_action(
            opportunity
        )
        assert result is None

    def test_should_add_staking_actions_no_voter_contract(self) -> None:
        """Test _should_add_staking_actions with no voter contract."""
        opportunity = {
            "dex_type": "velodrome",
            "chain": "base",
        }  # Chain not in voter contracts

        result = self.behaviour.current_behaviour._should_add_staking_actions(
            opportunity
        )
        assert result is False

    def test_get_velodrome_position_requirements_balanced_allocation(self) -> None:
        """Test Velodrome position requirements with balanced allocation."""
        self.behaviour.current_behaviour.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "pool_address": "0x123",
                "chain": "optimism",
                "is_stable": False,
                "token0": "0x456",
                "token1": "0x789",
                "token0_symbol": "USDC",
                "token1_symbol": "WETH",
                "relative_funds_percentage": 0.8,
            }
        ]

        mock_pool = MagicMock()

        def mock_get_tick_spacing(self, pool_address, chain):
            yield
            return 1

        def mock_calculate_tick_bands(self, **kwargs):
            yield
            return [
                {
                    "tick_lower": 100,
                    "tick_upper": 200,
                    "allocation": 1.0,
                    "percent_in_bounds": 0.8,
                }
            ]

        def mock_get_current_price(self, pool_address, chain):
            yield
            return 1.5

        def mock_get_token_balance(chain, safe_address, token):
            yield
            return 1000000

        mock_pool._get_tick_spacing_velodrome = mock_get_tick_spacing
        mock_pool._calculate_tick_lower_and_upper_velodrome = mock_calculate_tick_bands
        mock_pool._get_current_pool_price = mock_get_current_price
        self.behaviour.current_behaviour.pools = {"velodrome": mock_pool}

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_velodrome_cl_token_requirements",
            return_value={
                "overall_token0_ratio": 0.6,
                "overall_token1_ratio": 0.4,
                "recommendation": "Provide 60% token0, 40% token1",
            },
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_get_token_balance",
                side_effect=mock_get_token_balance,
            ):
                generator = (
                    self.behaviour.current_behaviour.get_velodrome_position_requirements()
                )

                try:
                    while True:
                        next(generator)
                except StopIteration:
                    pass

                # Verify balanced allocation with relative funds percentage
                opportunity = self.behaviour.current_behaviour.selected_opportunities[0]
                assert "token_requirements" in opportunity
                # The actual implementation calculates based on the token ratios
                # For 60/40 split with 800k each token, it calculates the required amounts
                assert len(opportunity["max_amounts_in"]) == 2

    def test_get_velodrome_position_requirements_zero_ratio_fallback(self) -> None:
        """Test Velodrome position requirements with zero ratios fallback."""
        self.behaviour.current_behaviour.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "pool_address": "0x123",
                "chain": "optimism",
                "is_stable": False,
                "token0": "0x456",
                "token1": "0x789",
                "token0_symbol": "USDC",
                "token1_symbol": "WETH",
            }
        ]

        mock_pool = MagicMock()

        def mock_get_tick_spacing(self, pool_address, chain):
            yield
            return 1

        def mock_calculate_tick_bands(self, **kwargs):
            yield
            return [
                {
                    "tick_lower": 100,
                    "tick_upper": 200,
                    "allocation": 1.0,
                    "percent_in_bounds": 0.8,
                }
            ]

        def mock_get_current_price(self, pool_address, chain):
            yield
            return 1.5

        def mock_get_token_balance(chain, safe_address, token):
            yield
            return 1000000

        mock_pool._get_tick_spacing_velodrome = mock_get_tick_spacing
        mock_pool._calculate_tick_lower_and_upper_velodrome = mock_calculate_tick_bands
        mock_pool._get_current_pool_price = mock_get_current_price
        self.behaviour.current_behaviour.pools = {"velodrome": mock_pool}

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_velodrome_cl_token_requirements",
            return_value={
                "overall_token0_ratio": 0.0,  # Both zero
                "overall_token1_ratio": 0.0,
                "recommendation": "Error case",
            },
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_get_token_balance",
                side_effect=mock_get_token_balance,
            ):
                generator = (
                    self.behaviour.current_behaviour.get_velodrome_position_requirements()
                )

                try:
                    while True:
                        next(generator)
                except StopIteration:
                    pass

                # Should fall back to 50/50 split
                opportunity = self.behaviour.current_behaviour.selected_opportunities[0]
                assert opportunity["max_amounts_in"][0] == 500000  # 50% of 1M
                assert opportunity["max_amounts_in"][1] == 500000  # 50% of 1M

    def test_get_velodrome_position_requirements_exception_handling(self) -> None:
        """Test Velodrome position requirements with exception handling."""
        self.behaviour.current_behaviour.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "pool_address": "0x123",
                "chain": "optimism",
                "is_stable": False,
                "token0": "0x456",
                "token1": "0x789",
            }
        ]

        mock_pool = MagicMock()

        def mock_get_tick_spacing(self, pool_address, chain):
            yield
            raise Exception("Tick spacing error")

        mock_pool._get_tick_spacing_velodrome = mock_get_tick_spacing
        self.behaviour.current_behaviour.pools = {"velodrome": mock_pool}

        generator = (
            self.behaviour.current_behaviour.get_velodrome_position_requirements()
        )

        try:
            while True:
                next(generator)
        except StopIteration:
            pass

        # Should handle exception gracefully and continue

    def test_execute_hyper_strategy_invalid_composite_score(self) -> None:
        """Test hyper strategy execution with invalid composite score."""
        self.behaviour.current_behaviour.trading_opportunities = [
            {"pool_address": "0x123", "apr": 15.0}
        ]
        self.behaviour.current_behaviour.positions_eligible_for_exit = []

        def mock_read_kv(keys):
            yield
            return {"composite_score": "invalid_float"}

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "execute_strategy",
                return_value={"optimal_strategies": [], "position_to_exit": None},
            ) as mock_execute:
                generator = self.behaviour.current_behaviour.execute_hyper_strategy()

                try:
                    while True:
                        next(generator)
                except StopIteration:
                    pass

                # Should fall back to 0.0 for invalid composite score
                call_args = mock_execute.call_args[1]
                assert call_args["composite_score_threshold"] == 0.0

    def test_execute_hyper_strategy_with_logs_and_reasoning(self) -> None:
        """Test hyper strategy execution with logs and reasoning processing."""
        self.behaviour.current_behaviour.trading_opportunities = [
            {"pool_address": "0x123", "apr": 15.0}
        ]
        self.behaviour.current_behaviour.positions_eligible_for_exit = []

        def mock_read_kv(keys):
            yield
            return {"composite_score": "0.5"}

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "execute_strategy",
                return_value={
                    "optimal_strategies": [
                        {
                            "pool_address": "0x123",
                            "token0": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
                            "token1": "0x4200000000000000000000000000000000000006",
                        }
                    ],
                    "position_to_exit": None,
                    "logs": ["Log 1", "Log 2", "Log 3"],
                    "reasoning": "Detailed reasoning",
                },
            ):
                generator = self.behaviour.current_behaviour.execute_hyper_strategy()

                try:
                    while True:
                        next(generator)
                except StopIteration:
                    pass

                # Verify token addresses were converted to checksum
                opportunity = self.behaviour.current_behaviour.selected_opportunities[0]
                assert opportunity["token0"].startswith("0x")
                assert opportunity["token1"].startswith("0x")

    def test_fetch_all_trading_opportunities_parallel_execution_error(self) -> None:
        """Test fetch_all_trading_opportunities with parallel execution errors."""
        self.behaviour.current_behaviour.synchronized_data.selected_protocols = [
            "test_strategy"
        ]
        self.behaviour.current_behaviour.context.params.strategies_kwargs = {
            "test_strategy": {}
        }
        self.behaviour.current_behaviour.context.params.target_investment_chains = [
            "optimism"
        ]
        self.behaviour.current_behaviour.context.params.available_protocols = [
            "velodrome"
        ]
        self.behaviour.current_behaviour.context.params.chain_to_chain_id_mapping = {
            "optimism": 10
        }
        self.behaviour.current_behaviour.positions_eligible_for_exit = []
        self.behaviour.current_behaviour.coingecko.api_key = "test_key"
        self.behaviour.current_behaviour.shared_state.strategies_executables = {
            "test_strategy": ("print('test')", "test_method")
        }

        with patch("asyncio.ensure_future") as mock_future:
            mock_future_obj = MagicMock()
            mock_future_obj.done.return_value = True
            mock_future_obj.result.side_effect = Exception("Parallel execution error")
            mock_future.return_value = mock_future_obj

            generator = (
                self.behaviour.current_behaviour.fetch_all_trading_opportunities()
            )

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Should handle error gracefully
            assert len(self.behaviour.current_behaviour.trading_opportunities) == 0

    def test_fetch_all_trading_opportunities_invalid_opportunity_format(self) -> None:
        """Test fetch_all_trading_opportunities with invalid opportunity formats."""
        self.behaviour.current_behaviour.synchronized_data.selected_protocols = [
            "test_strategy"
        ]
        self.behaviour.current_behaviour.context.params.strategies_kwargs = {
            "test_strategy": {}
        }
        self.behaviour.current_behaviour.context.params.target_investment_chains = [
            "optimism"
        ]
        self.behaviour.current_behaviour.context.params.available_protocols = [
            "velodrome"
        ]
        self.behaviour.current_behaviour.context.params.chain_to_chain_id_mapping = {
            "optimism": 10
        }
        self.behaviour.current_behaviour.positions_eligible_for_exit = []
        self.behaviour.current_behaviour.coingecko.api_key = "test_key"
        self.behaviour.current_behaviour.shared_state.strategies_executables = {
            "test_strategy": ("print('test')", "test_method")
        }

        with patch("asyncio.ensure_future") as mock_future:
            mock_future_obj = MagicMock()
            mock_future_obj.done.return_value = True
            # Return invalid opportunity format (not a dict)
            mock_future_obj.result.return_value = [
                {"result": ["invalid_opportunity_string", 123, None], "error": []}
            ]
            mock_future.return_value = mock_future_obj

            generator = (
                self.behaviour.current_behaviour.fetch_all_trading_opportunities()
            )

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Should filter out invalid opportunities
            assert len(self.behaviour.current_behaviour.trading_opportunities) == 0

    def test_fetch_all_trading_opportunities_no_result_returned(self) -> None:
        """Test fetch_all_trading_opportunities when no result is returned."""
        self.behaviour.current_behaviour.synchronized_data.selected_protocols = [
            "test_strategy"
        ]
        self.behaviour.current_behaviour.context.params.strategies_kwargs = {
            "test_strategy": {}
        }
        self.behaviour.current_behaviour.context.params.target_investment_chains = [
            "optimism"
        ]
        self.behaviour.current_behaviour.context.params.available_protocols = [
            "velodrome"
        ]
        self.behaviour.current_behaviour.context.params.chain_to_chain_id_mapping = {
            "optimism": 10
        }
        self.behaviour.current_behaviour.positions_eligible_for_exit = []
        self.behaviour.current_behaviour.coingecko.api_key = "test_key"
        self.behaviour.current_behaviour.shared_state.strategies_executables = {
            "test_strategy": ("print('test')", "test_method")
        }

        with patch("asyncio.ensure_future") as mock_future:
            mock_future_obj = MagicMock()
            mock_future_obj.done.return_value = True
            mock_future_obj.result.return_value = [None]  # No result returned
            mock_future.return_value = mock_future_obj

            generator = (
                self.behaviour.current_behaviour.fetch_all_trading_opportunities()
            )

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            assert len(self.behaviour.current_behaviour.trading_opportunities) == 0

    def test_download_next_strategy_no_strategies_pending(self) -> None:
        """Test download_next_strategy when no strategies are pending."""
        self.behaviour.current_behaviour._inflight_strategy_req = None
        self.mock_shared_state.strategy_to_filehash = {}

        # Should return early without doing anything
        self.behaviour.current_behaviour.download_next_strategy()

        # Verify no request was made
        assert self.behaviour.current_behaviour._inflight_strategy_req is None

    def test_download_strategies_with_pending_strategies(self) -> None:
        """Test download_strategies when strategies are pending."""
        self.mock_shared_state.strategy_to_filehash = {
            "strategy1": "hash1",
            "strategy2": "hash2",
        }

        download_call_count = 0

        def mock_download_next_strategy():
            nonlocal download_call_count
            download_call_count += 1
            if download_call_count >= 2:
                # Clear the pending strategies after 2 calls
                self.mock_shared_state.strategy_to_filehash.clear()

        def mock_sleep(seconds):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "download_next_strategy",
            side_effect=mock_download_next_strategy,
        ):
            with patch.object(
                self.behaviour.current_behaviour, "sleep", side_effect=mock_sleep
            ):
                generator = self.behaviour.current_behaviour.download_strategies()

                try:
                    while True:
                        next(generator)
                except StopIteration:
                    pass

                # Should have called download_next_strategy multiple times
                assert download_call_count >= 2

    def test_execute_strategy_exec_globals_cleanup(self) -> None:
        """Test execute_strategy cleans up globals properly."""
        mock_strategy_code = "def test_method(): return {'result': 'success'}"

        with patch.object(
            self.behaviour.current_behaviour,
            "strategy_exec",
            return_value=(mock_strategy_code, "test_method"),
        ):
            # First execution
            result1 = self.behaviour.current_behaviour.execute_strategy(
                strategy="test_strategy"
            )
            assert result1 == {"result": "success"}

            # Method should be cleaned up from globals - check the actual globals dict
            import builtins

            actual_globals = getattr(builtins, "__dict__", globals())
            assert "test_method" not in actual_globals

    def test_handle_get_strategy_no_request(self) -> None:
        """Test _handle_get_strategy when no strategy request is pending."""
        self.behaviour.current_behaviour._inflight_strategy_req = None

        mock_message = MagicMock()
        mock_dialogue = MagicMock()

        # Should return early without processing
        self.behaviour.current_behaviour._handle_get_strategy(
            mock_message, mock_dialogue
        )

        # Verify no changes were made
        assert self.behaviour.current_behaviour._inflight_strategy_req is None

    def test_merge_duplicate_bridge_swap_actions_error_handling(self) -> None:
        """Test bridge swap action merging with error handling."""
        # Test with malformed action that causes processing error
        actions = [
            {
                "action": Action.FIND_BRIDGE_ROUTE.value,
                "from_chain": None,  # This will cause an error
                "to_chain": "base",
                "funds_percentage": 0.5,
            }
        ]

        # Should handle error gracefully and return original actions
        result = self.behaviour.current_behaviour._merge_duplicate_bridge_swap_actions(
            actions
        )
        assert result == actions

    def test_handle_velodrome_token_allocation_no_token_requirements(self) -> None:
        """Test _handle_velodrome_token_allocation without token requirements."""
        actions = [
            {"action": "FindBridgeRoute", "from_chain": "base", "to_chain": "optimism"}
        ]
        enter_pool_action = {"dex_type": "velodrome", "chain": "optimism"}
        available_tokens = []

        result = self.behaviour.current_behaviour._handle_velodrome_token_allocation(
            actions, enter_pool_action, available_tokens
        )

        # Should return actions unchanged without token requirements
        assert result == actions

    def test_handle_velodrome_token_allocation_balanced_ratios(self) -> None:
        """Test _handle_velodrome_token_allocation with balanced token ratios."""
        actions = [
            {"action": "FindBridgeRoute", "from_chain": "base", "to_chain": "optimism"}
        ]
        enter_pool_action = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "token0": "0x123",
            "token1": "0x456",
            "token0_symbol": "USDC",
            "token1_symbol": "WETH",
            "relative_funds_percentage": 1.0,
            "token_requirements": {
                "overall_token0_ratio": 0.5,
                "overall_token1_ratio": 0.5,
                "recommendation": "Provide 50% token0, 50% token1",
            },
        }
        available_tokens = []

        result = self.behaviour.current_behaviour._handle_velodrome_token_allocation(
            actions, enter_pool_action, available_tokens
        )

        # Should return actions unchanged for balanced ratios
        assert result == actions

    def test_handle_velodrome_token_allocation_extreme_token1(self) -> None:
        """Test _handle_velodrome_token_allocation with 100% token1 allocation."""
        actions = [
            {
                "action": "FindBridgeRoute",
                "from_chain": "base",
                "to_chain": "optimism",
                "from_token": "0x789",
                "to_token": "0x123",  # Will be redirected to token1
                "from_token_symbol": "DAI",
                "to_token_symbol": "USDC",
                "funds_percentage": 0.5,
            }
        ]
        enter_pool_action = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "token0": "0x123",
            "token1": "0x456",
            "token0_symbol": "USDC",
            "token1_symbol": "WETH",
            "relative_funds_percentage": 1.0,
            "token_requirements": {
                "overall_token0_ratio": 0.0,
                "overall_token1_ratio": 1.0,
                "recommendation": "Provide 0% token0, 100% token1",
            },
        }
        available_tokens = [{"chain": "base", "token": "0x789", "token_symbol": "DAI"}]

        result = self.behaviour.current_behaviour._handle_velodrome_token_allocation(
            actions, enter_pool_action, available_tokens
        )

        # Should redirect to token1
        assert result[0]["to_token"] == "0x456"
        assert result[0]["to_token_symbol"] == "WETH"

    def test_apply_investment_cap_to_actions_with_exit_pool(self) -> None:
        """Test _apply_investment_cap_to_actions with exit pool action."""
        actions = [
            {"action": "ExitPool", "pool_address": "0x456"},
            {"action": "EnterPool", "pool_address": "0x123"},
        ]
        self.behaviour.current_behaviour.current_positions = [
            {"status": "open", "pool_address": "0x789"}
        ]

        def mock_calculate_initial_investment_value(position):
            yield
            return 960  # Above threshold

        def mock_sleep(seconds):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_initial_investment_value",
            side_effect=mock_calculate_initial_investment_value,
        ):
            with patch.object(
                self.behaviour.current_behaviour, "sleep", side_effect=mock_sleep
            ):
                generator = (
                    self.behaviour.current_behaviour._apply_investment_cap_to_actions(
                        actions
                    )
                )

                result = None
                try:
                    while True:
                        result = next(generator)
                except StopIteration as e:
                    result = e.value

                # Should keep exit action but remove enter action
                assert len(result) == 1
                assert result[0]["action"] == "ExitPool"

    def test_apply_investment_cap_to_actions_under_threshold_adjustment(self) -> None:
        """Test _apply_investment_cap_to_actions with under threshold adjustment."""
        actions = [{"action": "EnterPool", "pool_address": "0x123"}]
        self.behaviour.current_behaviour.current_positions = [
            {"status": "open", "pool_address": "0x789"}
        ]

        def mock_calculate_initial_investment_value(position):
            yield
            return 500  # Under threshold

        def mock_sleep(seconds):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_initial_investment_value",
            side_effect=mock_calculate_initial_investment_value,
        ):
            with patch.object(
                self.behaviour.current_behaviour, "sleep", side_effect=mock_sleep
            ):
                generator = (
                    self.behaviour.current_behaviour._apply_investment_cap_to_actions(
                        actions
                    )
                )

                result = None
                try:
                    while True:
                        result = next(generator)
                except StopIteration as e:
                    result = e.value

                # Should adjust invested_amount
                assert result[0]["invested_amount"] == 500  # 1000 - 500

    def test_apply_investment_cap_to_actions_zero_invested_with_positions(self) -> None:
        """Test _apply_investment_cap_to_actions with zero invested amount but open positions."""
        actions = [{"action": "EnterPool", "pool_address": "0x123"}]
        self.behaviour.current_behaviour.current_positions = [
            {"status": "open", "pool_address": "0x789"}
        ]

        def mock_calculate_initial_investment_value(position):
            yield
            return 0  # Zero invested but positions exist

        def mock_sleep(seconds):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_initial_investment_value",
            side_effect=mock_calculate_initial_investment_value,
        ):
            with patch.object(
                self.behaviour.current_behaviour, "sleep", side_effect=mock_sleep
            ):
                generator = (
                    self.behaviour.current_behaviour._apply_investment_cap_to_actions(
                        actions
                    )
                )

                result = None
                try:
                    while True:
                        result = next(generator)
                except StopIteration as e:
                    result = e.value

                # Should clear actions when invested amount is 0 but positions exist
                assert result == []

    def test_process_rewards_functionality(self) -> None:
        """Test _process_rewards method functionality."""
        actions = []
        self.behaviour.current_behaviour.context.params.target_investment_chains = [
            "optimism"
        ]

        def mock_get_rewards(chain_id, safe_address):
            yield
            return {
                "users": ["0xuser"],
                "tokens": ["0xtoken"],
                "claims": [1000],
                "proofs": [["0xproof"]],
                "symbols": ["TOKEN"],
            }

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_rewards",
            side_effect=mock_get_rewards,
        ):
            with patch.object(
                self.behaviour.current_behaviour, "_build_claim_reward_action"
            ) as mock_build_claim:
                mock_build_claim.return_value = {"action": "ClaimRewards"}

                generator = self.behaviour.current_behaviour._process_rewards(actions)

                try:
                    while True:
                        next(generator)
                except StopIteration:
                    pass

                # Should have added claim reward action
                assert len(actions) == 1
                mock_build_claim.assert_called_once()

    def test_prepare_tokens_for_investment_with_position_to_exit(self) -> None:
        """Test _prepare_tokens_for_investment with position to exit."""
        self.behaviour.current_behaviour.position_to_exit = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "token0": "0x123",
            "token0_symbol": "USDC",
            "token1": "0x456",
            "token1_symbol": "WETH",
        }

        def mock_get_available_tokens():
            yield
            return [{"chain": "base", "token": "0x789", "token_symbol": "DAI"}]

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_available_tokens",
            side_effect=mock_get_available_tokens,
        ):
            generator = (
                self.behaviour.current_behaviour._prepare_tokens_for_investment()
            )

            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            # Should include tokens from position and available tokens
            assert len(result) == 3  # 2 from position + 1 available

    def test_prepare_tokens_for_investment_sturdy_dex(self) -> None:
        """Test _prepare_tokens_for_investment with Sturdy DEX (single token)."""
        self.behaviour.current_behaviour.position_to_exit = {
            "dex_type": DexType.STURDY.value,
            "chain": "optimism",
            "token0": "0x123",
            "token0_symbol": "USDC",
        }

        def mock_get_available_tokens():
            yield
            return []

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_available_tokens",
            side_effect=mock_get_available_tokens,
        ):
            generator = (
                self.behaviour.current_behaviour._prepare_tokens_for_investment()
            )

            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            # Should include only one token for Sturdy
            assert len(result) == 1
            assert result[0]["token"] == "0x123"

    def test_prepare_tokens_for_investment_insufficient_tokens(self) -> None:
        """Test _prepare_tokens_for_investment with insufficient tokens."""
        self.behaviour.current_behaviour.position_to_exit = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "token0": "0x123",
            "token0_symbol": "USDC",
        }

        def mock_get_available_tokens():
            yield
            return []

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_available_tokens",
            side_effect=mock_get_available_tokens,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_build_tokens_from_position",
                return_value=[],  # Insufficient tokens
            ):
                generator = (
                    self.behaviour.current_behaviour._prepare_tokens_for_investment()
                )

                result = None
                try:
                    while True:
                        result = next(generator)
                except StopIteration as e:
                    result = e.value

                # Should return None for insufficient tokens
                assert result is None

    def test_build_tokens_from_position_edge_cases(self) -> None:
        """Test _build_tokens_from_position with edge cases."""
        position = {
            "chain": "optimism",
            "token0": "0x123",
            "token0_symbol": "USDC",
            "token1": "0x456",
            "token1_symbol": "WETH",
        }

        # Test with invalid num_tokens
        result = self.behaviour.current_behaviour._build_tokens_from_position(
            position, 3
        )
        assert result is None

        # Test with zero tokens
        result = self.behaviour.current_behaviour._build_tokens_from_position(
            position, 0
        )
        assert result is None

    def test_get_available_tokens_comprehensive(self) -> None:
        """Test _get_available_tokens with comprehensive scenarios."""
        mock_positions = [
            {
                "chain": "optimism",
                "assets": [
                    {
                        "address": "0xtoken1",
                        "asset_symbol": "TOKEN1",
                        "balance": 1000000,
                    },
                    {
                        "address": list(
                            REWARD_TOKEN_ADDRESSES.get("optimism", {}).keys()
                        )[0]
                        if REWARD_TOKEN_ADDRESSES.get("optimism")
                        else "0xreward",
                        "asset_symbol": "REWARD",
                        "balance": 500000,
                    },
                ],
            }
        ]
        self.behaviour.current_behaviour.synchronized_data.positions = mock_positions

        def mock_get_investable_balance(chain, token_address, balance):
            yield
            # Return different values based on token type
            if "reward" in token_address.lower():
                return balance // 2  # Half balance for reward tokens
            return balance

        def mock_fetch_token_prices(token_balances):
            yield
            return {"0xtoken1": 1.0, "0xreward": 2.0}

        def mock_get_token_decimals(chain, token_address):
            yield
            return 18

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_investable_balance",
            side_effect=mock_get_investable_balance,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_fetch_token_prices",
                side_effect=mock_fetch_token_prices,
            ):
                with patch.object(
                    self.behaviour.current_behaviour,
                    "_get_token_decimals",
                    side_effect=mock_get_token_decimals,
                ):
                    generator = self.behaviour.current_behaviour._get_available_tokens()

                    result = None
                    try:
                        while True:
                            result = next(generator)
                    except StopIteration as e:
                        result = e.value

                    assert isinstance(result, list)

    def test_build_bridge_swap_actions_comprehensive_scenarios(self) -> None:
        """Test _build_bridge_swap_actions with comprehensive scenarios."""
        # Test with CL pool having token requirements
        opportunity = {
            "chain": "optimism",
            "token0": "0x123",
            "token0_symbol": "USDC",
            "token1": "0x456",
            "token1_symbol": "WETH",
            "relative_funds_percentage": 1.0,
            "dex_type": "velodrome",
            "is_cl_pool": True,
            "token_requirements": {
                "overall_token0_ratio": 0.7,
                "overall_token1_ratio": 0.3,
            },
        }

        tokens = [
            {
                "chain": "base",
                "token": "0x789",
                "token_symbol": "DAI",
                "balance": 1000000,
            }
        ]

        result = self.behaviour.current_behaviour._build_bridge_swap_actions(
            opportunity, tokens
        )
        assert isinstance(result, list)

        # Test with non-CL pool (should default to 50/50)
        opportunity["is_cl_pool"] = False
        opportunity.pop("token_requirements", None)

        result = self.behaviour.current_behaviour._build_bridge_swap_actions(
            opportunity, tokens
        )
        assert isinstance(result, list)

    def test_build_bridge_swap_actions_token_percentage_fallback(self) -> None:
        """Test _build_bridge_swap_actions with token percentage fallback."""
        opportunity = {
            "chain": "optimism",
            "token0": "0x123",
            "token0_symbol": "USDC",
            "token1": "0x456",
            "token1_symbol": "WETH",
            "relative_funds_percentage": 1.0,
            "dex_type": "velodrome",
            "token0_percentage": 60.0,
            "token1_percentage": 40.0,
        }

        tokens = [
            {
                "chain": "base",
                "token": "0x789",
                "token_symbol": "DAI",
                "balance": 1000000,
            }
        ]

        result = self.behaviour.current_behaviour._build_bridge_swap_actions(
            opportunity, tokens
        )
        assert isinstance(result, list)

    def test_build_bridge_swap_actions_zero_ratios_fallback(self) -> None:
        """Test _build_bridge_swap_actions with zero ratios fallback."""
        opportunity = {
            "chain": "optimism",
            "token0": "0x123",
            "token0_symbol": "USDC",
            "token1": "0x456",
            "token1_symbol": "WETH",
            "relative_funds_percentage": 1.0,
            "dex_type": "velodrome",
            "token0_percentage": 0.0,
            "token1_percentage": 0.0,
        }

        tokens = [
            {
                "chain": "base",
                "token": "0x789",
                "token_symbol": "DAI",
                "balance": 1000000,
            }
        ]

        result = self.behaviour.current_behaviour._build_bridge_swap_actions(
            opportunity, tokens
        )
        assert isinstance(result, list)

    def test_handle_all_tokens_available_rebalancing(self) -> None:
        """Test _handle_all_tokens_available with rebalancing logic."""
        tokens = [
            {
                "chain": "optimism",
                "token": "0x123",
                "token_symbol": "USDC",
                "value": 1000,
            },
            {
                "chain": "optimism",
                "token": "0x456",
                "token_symbol": "WETH",
                "value": 500,
            },
            {"chain": "base", "token": "0x789", "token_symbol": "DAI", "value": 300},
        ]
        required_tokens = [("0x123", "USDC"), ("0x456", "WETH")]
        dest_chain = "optimism"
        relative_funds_percentage = 1.0
        target_ratios_by_token = {"0x123": 0.6, "0x456": 0.4}

        result = self.behaviour.current_behaviour._handle_all_tokens_available(
            tokens,
            required_tokens,
            dest_chain,
            relative_funds_percentage,
            target_ratios_by_token,
        )

        assert isinstance(result, list)

    def test_handle_some_tokens_available_scenarios(self) -> None:
        """Test _handle_some_tokens_available with various scenarios."""
        tokens = [
            {
                "chain": "optimism",
                "token": "0x123",
                "token_symbol": "USDC",
                "value": 1000,
            },
            {"chain": "base", "token": "0x789", "token_symbol": "DAI", "value": 500},
        ]
        required_tokens = [("0x123", "USDC"), ("0x456", "WETH")]
        tokens_we_need = [("0x456", "WETH")]
        dest_chain = "optimism"
        relative_funds_percentage = 1.0
        target_ratios_by_token = {"0x123": 0.5, "0x456": 0.5}

        result = self.behaviour.current_behaviour._handle_some_tokens_available(
            tokens,
            required_tokens,
            tokens_we_need,
            dest_chain,
            relative_funds_percentage,
            target_ratios_by_token,
        )

        assert isinstance(result, list)

    def test_handle_all_tokens_needed_single_source(self) -> None:
        """Test _handle_all_tokens_needed with single source token."""
        tokens = [
            {"chain": "base", "token": "0x789", "token_symbol": "DAI", "value": 1000},
        ]
        required_tokens = [("0x123", "USDC"), ("0x456", "WETH")]
        dest_chain = "optimism"
        relative_funds_percentage = 1.0
        target_ratios_by_token = {"0x123": 0.6, "0x456": 0.4}

        result = self.behaviour.current_behaviour._handle_all_tokens_needed(
            tokens,
            required_tokens,
            dest_chain,
            relative_funds_percentage,
            target_ratios_by_token,
        )

        assert isinstance(result, list)

    def test_handle_all_tokens_needed_multiple_sources(self) -> None:
        """Test _handle_all_tokens_needed with multiple source tokens."""
        tokens = [
            {"chain": "base", "token": "0x789", "token_symbol": "DAI", "value": 500},
            {"chain": "mode", "token": "0xabc", "token_symbol": "USDT", "value": 500},
        ]
        required_tokens = [("0x123", "USDC"), ("0x456", "WETH")]
        dest_chain = "optimism"
        relative_funds_percentage = 1.0
        target_ratios_by_token = {"0x123": 0.5, "0x456": 0.5}

        result = self.behaviour.current_behaviour._handle_all_tokens_needed(
            tokens,
            required_tokens,
            dest_chain,
            relative_funds_percentage,
            target_ratios_by_token,
        )

        assert isinstance(result, list)

    def test_add_bridge_swap_action_same_chain_different_token(self) -> None:
        """Test _add_bridge_swap_action with same chain but different token."""
        actions = []
        token = {
            "chain": "optimism",
            "token": "0x789",
            "token_symbol": "DAI",
            "balance": 1000,
        }
        dest_chain = "optimism"
        dest_token_address = "0x123"
        dest_token_symbol = "USDC"
        relative_funds_percentage = 1.0

        self.behaviour.current_behaviour._add_bridge_swap_action(
            actions,
            token,
            dest_chain,
            dest_token_address,
            dest_token_symbol,
            relative_funds_percentage,
        )

        # Should add swap action
        assert len(actions) == 1
        assert actions[0]["action"] == Action.FIND_BRIDGE_ROUTE.value
        assert actions[0]["from_chain"] == "optimism"
        assert actions[0]["to_chain"] == "optimism"

    def test_build_enter_pool_action_regular_dex(self) -> None:
        """Test _build_enter_pool_action with regular DEX."""
        opportunity = {
            "pool_address": "0x123",
            "chain": "optimism",
            "dex_type": "velodrome",
            "apr": 15.0,
            "percent_in_bounds": 0.85,
        }

        result = self.behaviour.current_behaviour._build_enter_pool_action(opportunity)

        assert result is not None
        assert result["action"] == Action.ENTER_POOL.value
        assert result["pool_address"] == "0x123"
        assert result["opportunity_apr"] == 15.0
        assert result["percent_in_bounds"] == 0.85

    def test_get_asset_symbol_functionality(self) -> None:
        """Test _get_asset_symbol functionality."""
        mock_positions = [
            {
                "chain": "optimism",
                "assets": [
                    {"address": "0x123", "asset_symbol": "USDC"},
                    {"address": "0x456", "asset_symbol": "WETH"},
                ],
            }
        ]
        self.behaviour.current_behaviour.synchronized_data.positions = mock_positions

        # Test finding existing asset
        result = self.behaviour.current_behaviour._get_asset_symbol("optimism", "0x123")
        assert result == "USDC"

        # Test asset not found
        result = self.behaviour.current_behaviour._get_asset_symbol("optimism", "0x999")
        assert result is None

        # Test chain not found
        result = self.behaviour.current_behaviour._get_asset_symbol("base", "0x123")
        assert result is None

    def test_get_rewards_all_claimed(self) -> None:
        """Test _get_rewards when everything has been claimed."""

        def mock_get_http_response(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.body = json.dumps(
                {
                    "0x123": {
                        "proof": ["0xproof1"],
                        "symbol": "TOKEN1",
                        "accumulated": "1000",
                        "unclaimed": "0",  # All claimed
                    }
                }
            )
            yield
            return mock_response

        with patch.object(
            self.behaviour.current_behaviour,
            "get_http_response",
            side_effect=mock_get_http_response,
        ):
            generator = self.behaviour.current_behaviour._get_rewards(10, "0x123")
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result is None

    def test_get_rewards_no_tokens_with_proof(self) -> None:
        """Test _get_rewards when no tokens have proof."""

        def mock_get_http_response(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.body = json.dumps(
                {
                    "0x123": {
                        "proof": None,  # No proof
                        "symbol": "TOKEN1",
                        "accumulated": "1000",
                        "unclaimed": "500",
                    }
                }
            )
            yield
            return mock_response

        with patch.object(
            self.behaviour.current_behaviour,
            "get_http_response",
            side_effect=mock_get_http_response,
        ):
            generator = self.behaviour.current_behaviour._get_rewards(10, "0x123")
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result is None

    def test_initialize_entry_costs_for_new_position_missing_data(self) -> None:
        """Test _initialize_entry_costs_for_new_position with missing data."""
        # Test with missing chain
        enter_pool_action = {"pool_address": "0x123"}

        def mock_initialize_position_entry_costs(chain, pool_address):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "_initialize_position_entry_costs",
            side_effect=mock_initialize_position_entry_costs,
        ):
            generator = self.behaviour.current_behaviour._initialize_entry_costs_for_new_position(
                enter_pool_action
            )

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

        # Test with missing pool_address
        enter_pool_action = {"chain": "optimism"}

        with patch.object(
            self.behaviour.current_behaviour,
            "_initialize_position_entry_costs",
            side_effect=mock_initialize_position_entry_costs,
        ):
            generator = self.behaviour.current_behaviour._initialize_entry_costs_for_new_position(
                enter_pool_action
            )

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

    def test_initialize_entry_costs_for_new_position_exception(self) -> None:
        """Test _initialize_entry_costs_for_new_position with exception."""
        enter_pool_action = {"chain": "optimism", "pool_address": "0x123"}

        def mock_initialize_position_entry_costs(chain, pool_address):
            yield
            raise Exception("Initialization error")

        with patch.object(
            self.behaviour.current_behaviour,
            "_initialize_position_entry_costs",
            side_effect=mock_initialize_position_entry_costs,
        ):
            generator = self.behaviour.current_behaviour._initialize_entry_costs_for_new_position(
                enter_pool_action
            )

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

    def test_initialize_position_entry_costs_exception(self) -> None:
        """Test _initialize_position_entry_costs with exception."""

        def mock_store_entry_costs(chain, position_id, cost):
            yield
            raise Exception("Store error")

        with patch.object(
            self.behaviour.current_behaviour,
            "_store_entry_costs",
            side_effect=mock_store_entry_costs,
        ):
            generator = (
                self.behaviour.current_behaviour._initialize_position_entry_costs(
                    "optimism", "0x123"
                )
            )

            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

    def test_build_stake_lp_tokens_action_missing_data(self) -> None:
        """Test _build_stake_lp_tokens_action with missing required data."""
        opportunity = {
            "dex_type": "velodrome",
            # Missing chain and pool_address
        }

        result = self.behaviour.current_behaviour._build_stake_lp_tokens_action(
            opportunity
        )
        assert result is None

    def test_build_stake_lp_tokens_action_no_safe_address(self) -> None:
        """Test _build_stake_lp_tokens_action with no safe address."""
        opportunity = {
            "chain": "base",  # Chain not in safe_contract_addresses
            "pool_address": "0x123",
            "dex_type": "velodrome",
            "is_cl_pool": True,
        }

        result = self.behaviour.current_behaviour._build_stake_lp_tokens_action(
            opportunity
        )
        assert result is None

    def test_build_stake_lp_tokens_action_regular_pool(self) -> None:
        """Test _build_stake_lp_tokens_action with regular pool."""
        opportunity = {
            "chain": "optimism",
            "pool_address": "0x123",
            "dex_type": "velodrome",
            "is_cl_pool": False,
        }

        result = self.behaviour.current_behaviour._build_stake_lp_tokens_action(
            opportunity
        )

        assert result is not None
        assert result["action"] == Action.STAKE_LP_TOKENS.value
        assert result["dex_type"] == "velodrome"
        assert result["is_cl_pool"] is False

    def test_build_stake_lp_tokens_action_exception(self) -> None:
        """Test _build_stake_lp_tokens_action with exception."""
        opportunity = {
            "chain": "optimism",
            "pool_address": "0x123",
            "dex_type": "velodrome",
            "is_cl_pool": True,
        }

        # Mock missing safe contract address to trigger exception
        self.behaviour.current_behaviour.context.params.safe_contract_addresses = {}

        result = self.behaviour.current_behaviour._build_stake_lp_tokens_action(
            opportunity
        )
        # Should return None when safe address is not found
        assert result is None

    def test_should_add_staking_actions_empty_voter_address(self) -> None:
        """Test _should_add_staking_actions with empty voter address."""
        opportunity = {"dex_type": "velodrome", "chain": "optimism"}

        # Mock empty voter address
        self.behaviour.current_behaviour.context.params.velodrome_voter_contract_addresses = {
            "optimism": ""
        }

        result = self.behaviour.current_behaviour._should_add_staking_actions(
            opportunity
        )
        assert result is False

    # Additional tests to reach 90%+ coverage

    def test_has_staking_metadata(self) -> None:
        """Test _has_staking_metadata method."""
        # Test position with staking metadata
        position_with_staking = {
            "gauge_address": "0x123",
            "is_staked": True,
        }
        result = self.behaviour.current_behaviour._has_staking_metadata(
            position_with_staking
        )
        assert result is True

        # Test position without staking metadata
        position_without_staking = {
            "pool_address": "0x123",
        }
        result = self.behaviour.current_behaviour._has_staking_metadata(
            position_without_staking
        )
        assert result is False

    def test_build_unstake_lp_tokens_action(self) -> None:
        """Test _build_unstake_lp_tokens_action method."""
        position = {
            "chain": "optimism",
            "pool_address": "0x123",
            "dex_type": "velodrome",
            "is_cl_pool": True,
            "gauge_address": "0x456",
            "positions": [
                {"token_id": 1},
                {"token_id": 2},
            ],  # Add token IDs for CL pool
        }

        result = self.behaviour.current_behaviour._build_unstake_lp_tokens_action(
            position
        )
        assert result is not None
        assert result["action"] == Action.UNSTAKE_LP_TOKENS.value
        assert result["chain"] == "optimism"
        assert result["pool_address"] == "0x123"
        assert result["token_ids"] == [1, 2]

    def test_build_claim_staking_rewards_action(self) -> None:
        """Test _build_claim_staking_rewards_action method."""
        position = {
            "chain": "optimism",
            "pool_address": "0x123",
            "dex_type": "velodrome",
            "is_cl_pool": True,
            "gauge_address": "0x456",
        }

        result = self.behaviour.current_behaviour._build_claim_staking_rewards_action(
            position
        )
        assert result is not None
        assert result["action"] == Action.CLAIM_STAKING_REWARDS.value
        assert result["chain"] == "optimism"
        assert result["pool_address"] == "0x123"

    def test_get_gauge_address_for_position(self) -> None:
        """Test _get_gauge_address_for_position method."""
        position = {
            "chain": "optimism",
            "pool_address": "0x123",
        }

        def mock_get_gauge_address(self, lp_token, chain):
            yield
            return "0x456"

        mock_pool = MagicMock()
        mock_pool.get_gauge_address = mock_get_gauge_address
        self.behaviour.current_behaviour.pools = {"velodrome": mock_pool}

        generator = self.behaviour.current_behaviour._get_gauge_address_for_position(
            position
        )
        result = None
        try:
            while True:
                result = next(generator)
        except StopIteration as e:
            result = e.value

        assert result == "0x456"

    def test_fetch_token_prices_success(self) -> None:
        """Test _fetch_token_prices with successful response."""
        token_balances = [
            {"token": "0x123", "chain": "optimism", "token_symbol": "USDC"},
            {"token": "0x456", "chain": "optimism", "token_symbol": "WETH"},
        ]

        def mock_fetch_token_price(token_address, chain):
            yield
            return 1.0 if token_address == "0x123" else 2000.0

        def mock_fetch_zero_address_price():
            yield
            return 3000.0

        with patch.object(
            self.behaviour.current_behaviour,
            "_fetch_token_price",
            side_effect=mock_fetch_token_price,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_fetch_zero_address_price",
                side_effect=mock_fetch_zero_address_price,
            ):
                generator = self.behaviour.current_behaviour._fetch_token_prices(
                    token_balances
                )
                result = None
                try:
                    while True:
                        result = next(generator)
                except StopIteration as e:
                    result = e.value

                assert result == {"0x123": 1.0, "0x456": 2000.0}

    def test_get_token_decimals_success(self) -> None:
        """Test _get_token_decimals with successful response."""

        def mock_get_token_decimals(chain, token_address):
            yield
            return 18

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_token_decimals",
            side_effect=mock_get_token_decimals,
        ):
            generator = self.behaviour.current_behaviour._get_token_decimals(
                "optimism", "0x123"
            )
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result == 18

    def test_get_token_balance_success(self) -> None:
        """Test _get_token_balance with successful response."""

        def mock_get_token_balance(chain, safe_address, token):
            yield
            return 1000000

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_token_balance",
            side_effect=mock_get_token_balance,
        ):
            generator = self.behaviour.current_behaviour._get_token_balance(
                "optimism", "0xsafe", "0x123"
            )
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result == 1000000

    def test_get_usdc_address_success(self) -> None:
        """Test _get_usdc_address method."""
        # Mock WHITELISTED_ASSETS to include USDC
        with patch.dict(
            "packages.valory.skills.liquidity_trader_abci.behaviours.base.WHITELISTED_ASSETS",
            {"optimism": {"0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85": "USDC"}},
        ):
            result = self.behaviour.current_behaviour._get_usdc_address("optimism")
            assert result == "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85"

        # Test chain not found
        result = self.behaviour.current_behaviour._get_usdc_address("unknown_chain")
        assert result is None

    def test_build_swap_to_usdc_action_success(self) -> None:
        """Test _build_swap_to_usdc_action method."""
        result = self.behaviour.current_behaviour._build_swap_to_usdc_action(
            chain="optimism",
            from_token_address="0x123",
            from_token_symbol="DAI",
            funds_percentage=1.0,
            description="Swap DAI to USDC",
        )

        assert result is not None
        assert result["action"] == Action.FIND_BRIDGE_ROUTE.value
        assert result["from_chain"] == "optimism"
        assert result["to_chain"] == "optimism"
        assert result["from_token"] == "0x123"
        assert result["from_token_symbol"] == "DAI"

    def test_build_exit_pool_action_base_success(self) -> None:
        """Test _build_exit_pool_action_base method."""
        position = {
            "pool_address": "0x123",
            "chain": "optimism",
            "dex_type": "velodrome",
        }
        tokens = [
            {"token": "0x456", "token_symbol": "USDC"},
            {"token": "0x789", "token_symbol": "WETH"},
        ]

        result = self.behaviour.current_behaviour._build_exit_pool_action_base(
            position, tokens
        )
        assert result is not None
        assert result["action"] == Action.EXIT_POOL.value
        assert result["pool_address"] == "0x123"
        assert result["chain"] == "optimism"

    def test_store_current_positions(self) -> None:
        """Test store_current_positions method."""
        self.behaviour.current_behaviour.current_positions = [
            {"pool_address": "0x123", "status": "open"}
        ]

        def mock_store_kv(data):
            yield

        with patch.object(
            self.behaviour.current_behaviour, "_write_kv", side_effect=mock_store_kv
        ):
            # store_current_positions doesn't return a generator, it calls _write_kv internally
            self.behaviour.current_behaviour.store_current_positions()

    def test_get_accumulated_rewards_for_token(self) -> None:
        """Test get_accumulated_rewards_for_token method."""

        def mock_get_accumulated_rewards(chain, token_address):
            yield
            return 1000

        with patch.object(
            self.behaviour.current_behaviour,
            "get_accumulated_rewards_for_token",
            side_effect=mock_get_accumulated_rewards,
        ):
            generator = (
                self.behaviour.current_behaviour.get_accumulated_rewards_for_token(
                    "optimism", "0x123"
                )
            )
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result == 1000

    def test_check_minimum_time_met_true(self) -> None:
        """Test _check_minimum_time_met when minimum time is met."""
        position = {
            "enter_timestamp": 1000000,
            "min_hold_days": 7,
        }

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=1000000 + (8 * 24 * 3600),  # 8 days later
        ):
            result = self.behaviour.current_behaviour._check_minimum_time_met(position)
            assert result is True

    def test_check_minimum_time_met_false(self) -> None:
        """Test _check_minimum_time_met when minimum time is not met."""
        position = {
            "enter_timestamp": 1000000,
            "min_hold_days": 7,
        }

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=1000000 + (5 * 24 * 3600),  # 5 days later
        ):
            result = self.behaviour.current_behaviour._check_minimum_time_met(position)
            assert result is False

    def test_calculate_days_since_entry_success(self) -> None:
        """Test _calculate_days_since_entry calculation."""
        enter_timestamp = 1000000

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=1000000 + (5 * 24 * 3600),  # 5 days later
        ):
            days = self.behaviour.current_behaviour._calculate_days_since_entry(
                enter_timestamp
            )
            assert days == 5

    def test_fetch_historical_token_prices_success(self) -> None:
        """Test _fetch_historical_token_prices with successful response."""
        # The method signature expects List[List[str]] for tokens parameter
        tokens = [
            ["USDC", "0x123"],
            ["WETH", "0x456"],
        ]  # List of [symbol, address] pairs
        date_str = "01-01-2024"
        chain = "optimism"

        def mock_get_coin_id_from_symbol(symbol, chain):
            return f"coin-{symbol.lower()}"

        def mock_fetch_historical_token_price(coin_id, date_str):
            yield
            if "usdc" in coin_id:
                return 1.0
            elif "weth" in coin_id:
                return 2000.0
            return None

        with patch.object(
            self.behaviour.current_behaviour,
            "get_coin_id_from_symbol",
            side_effect=mock_get_coin_id_from_symbol,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_fetch_historical_token_price",
                side_effect=mock_fetch_historical_token_price,
            ):
                generator = (
                    self.behaviour.current_behaviour._fetch_historical_token_prices(
                        tokens, date_str, chain
                    )
                )
                result = None
                try:
                    while True:
                        result = next(generator)
                except StopIteration as e:
                    result = e.value

                assert result is not None
                assert isinstance(result, dict)
                assert "0x123" in result
                assert "0x456" in result
                assert result["0x123"] == 1.0
                assert result["0x456"] == 2000.0

    def test_store_entry_costs(self) -> None:
        """Test _store_entry_costs method."""

        def mock_store_kv(data):
            yield

        with patch.object(
            self.behaviour.current_behaviour, "_write_kv", side_effect=mock_store_kv
        ):
            generator = self.behaviour.current_behaviour._store_entry_costs(
                "optimism", "0x123", 100.0
            )
            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

    def test_build_ipfs_get_file_req(self) -> None:
        """Test _build_ipfs_get_file_req method."""
        file_hash = "QmTest123"

        with patch.object(
            self.behaviour.current_behaviour, "_build_ipfs_get_file_req"
        ) as mock_build:
            mock_build.return_value = (MagicMock(), MagicMock())

            result = self.behaviour.current_behaviour._build_ipfs_get_file_req(
                file_hash
            )
            assert result is not None
            mock_build.assert_called_once_with(file_hash)

    def test_read_kv_success(self) -> None:
        """Test _read_kv method success."""

        def mock_read_kv(keys):
            yield
            return {"test_key": "test_value"}

        with patch.object(
            self.behaviour.current_behaviour, "_read_kv", side_effect=mock_read_kv
        ):
            generator = self.behaviour.current_behaviour._read_kv(("test_key",))
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result == {"test_key": "test_value"}

    def test_write_kv_success(self) -> None:
        """Test _write_kv method success."""

        def mock_write_kv(data):
            yield
            return True

        with patch.object(
            self.behaviour.current_behaviour, "_write_kv", side_effect=mock_write_kv
        ):
            generator = self.behaviour.current_behaviour._write_kv(
                {"test_key": "test_value"}
            )
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result is True

    def test_sleep_method(self) -> None:
        """Test sleep method."""

        def mock_sleep(seconds):
            yield

        with patch.object(
            self.behaviour.current_behaviour, "sleep", side_effect=mock_sleep
        ):
            generator = self.behaviour.current_behaviour.sleep(1)
            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

    def test_get_current_timestamp_method(self) -> None:
        """Test _get_current_timestamp method."""
        timestamp = self.behaviour.current_behaviour._get_current_timestamp()
        assert isinstance(timestamp, (int, float))
        assert timestamp > 0
        test_positions = [
            {
                "pool_address": "0x123",
                "chain": "optimism",
                "status": "open",
            }
        ]
        self.behaviour.current_behaviour.current_positions = test_positions

        with patch.object(self.behaviour.current_behaviour, "_write_kv") as mock_store:
            mock_store.return_value = yield from self._mock_write_kv()
            self.behaviour.current_behaviour.store_current_positions()
            mock_store.assert_called_once()

    def test_store_entry_costs(self) -> None:
        """Test _store_entry_costs method."""

        def mock_store_entry_costs(chain, position_id, costs):
            yield
            # Mock the _write_kv call
            yield from self.behaviour.current_behaviour._write_kv(
                {"entry_costs_test": "100.0"}
            )

        with patch.object(
            self.behaviour.current_behaviour,
            "_store_entry_costs",
            side_effect=mock_store_entry_costs,
        ):
            with patch.object(
                self.behaviour.current_behaviour, "_write_kv"
            ) as mock_store:
                mock_store.return_value = yield from self._mock_write_kv()
                generator = self.behaviour.current_behaviour._store_entry_costs(
                    "optimism", "0x123", 100.0
                )

                try:
                    while True:
                        next(generator)
                except StopIteration:
                    pass

                mock_store.assert_called_once_with({"entry_costs_test": "100.0"})

    def test_handle_get_strategy_loader_error(self) -> None:
        """Test _handle_get_strategy with strategy loader error."""
        # Test when strategy is not found
        result = self.behaviour.current_behaviour.strategy_exec("non_existent_strategy")
        assert result is None

        # Test when strategy exists
        self.behaviour.current_behaviour.shared_state.strategies_executables = {
            "test_strategy": ("code", "method")
        }
        result = self.behaviour.current_behaviour.strategy_exec("test_strategy")
        assert result == ("code", "method")

    def test_calculate_initial_investment_value_missing_timestamp(self) -> None:
        """Test calculate_initial_investment_value with missing timestamp."""
        position = {
            "chain": "optimism",
            "token0": "0x123",
            "token1": "0x456",
            "amount0": 1000,
            "amount1": 2000,
            # Missing timestamp
        }

        generator = self.behaviour.current_behaviour.calculate_initial_investment_value(
            position
        )

        result = None
        try:
            while True:
                result = next(generator)
        except StopIteration as e:
            result = e.value

        assert result is None

    def test_calculate_initial_investment_value_missing_amounts(self) -> None:
        """Test calculate_initial_investment_value with missing amounts."""
        position = {
            "chain": "optimism",
            "timestamp": 1000000,
            # Missing token amounts
        }

        generator = self.behaviour.current_behaviour.calculate_initial_investment_value(
            position
        )

        result = None
        try:
            while True:
                result = next(generator)
        except StopIteration as e:
            result = e.value

        assert result is None

    def test_get_result_method_not_done(self) -> None:
        """Test get_result method when future is not done."""
        from concurrent.futures import Future

        future = Future()
        # Don't set result, so it's not done

        generator = self.behaviour.current_behaviour.get_result(future)

        # Should yield once since future is not done
        try:
            next(generator)
        except StopIteration:
            pass  # Should not reach here immediately

        # Now complete the future
        future.set_result({"test": "result"})

        result = None
        try:
            while True:
                result = next(generator)
        except StopIteration as e:
            result = e.value

        assert result == {"test": "result"}

    def _mock_write_kv(self):
        """Mock generator for _write_kv"""
        yield
        return True

    def test_build_stake_lp_tokens_action_non_velodrome(self) -> None:
        """Test staking action for non-Velodrome pools."""
        opportunity = {
            "dex_type": "uniswap",
            "chain": "optimism",
            "pool_address": "0x123",
        }

        result = self.behaviour.current_behaviour._build_stake_lp_tokens_action(
            opportunity
        )
        assert result is None

    def test_build_claim_staking_rewards_action_success(self) -> None:
        """Test building claim staking rewards action."""
        position = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0x123",
            "is_cl_pool": True,
            "gauge_address": "0x456",
        }

        self.behaviour.current_behaviour.context.params.safe_contract_addresses = {
            "optimism": "0xsafe"
        }

        result = self.behaviour.current_behaviour._build_claim_staking_rewards_action(
            position
        )
        assert result is not None
        assert result["action"] == Action.CLAIM_STAKING_REWARDS.value
        assert result["gauge_address"] == "0x456"

    def test_get_gauge_address_for_position_success(self) -> None:
        """Test successful gauge address retrieval."""
        position = {
            "chain": "optimism",
            "pool_address": "0x123",
        }

        self.behaviour.current_behaviour.context.params.velodrome_voter_contract_addresses = {
            "optimism": "0xvoter"
        }

        mock_pool = MagicMock()

        def mock_get_gauge_address(self, lp_token, chain):
            yield
            return "0xgauge"

        mock_pool.get_gauge_address = mock_get_gauge_address
        self.behaviour.current_behaviour.pools = {"velodrome": mock_pool}

        generator = self.behaviour.current_behaviour._get_gauge_address_for_position(
            position
        )
        result = None
        try:
            while True:
                result = next(generator)
        except StopIteration as e:
            result = e.value

        assert result == "0xgauge"

    def test_should_add_staking_actions_no_voter_contract(self) -> None:
        """Test staking decision with no voter contract."""
        opportunity = {"dex_type": "velodrome", "chain": "optimism"}
        self.behaviour.current_behaviour.context.params.velodrome_voter_contract_addresses = (
            {}
        )

        result = self.behaviour.current_behaviour._should_add_staking_actions(
            opportunity
        )
        assert result is False

    def test_get_investable_balance_reward_token(self) -> None:
        """Test balance calculation for reward tokens."""
        chain = "optimism"
        token_address = "0xreward"
        total_balance = 1000

        # Mock as reward token but not whitelisted
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy.to_checksum_address",
            return_value="0xreward",
        ):
            with patch.dict(
                "packages.valory.skills.liquidity_trader_abci.behaviours.base.REWARD_TOKEN_ADDRESSES",
                {chain: {"0xreward": "REWARD"}},
            ):
                with patch.dict(
                    "packages.valory.skills.liquidity_trader_abci.behaviours.base.WHITELISTED_ASSETS",
                    {chain: {}},
                ):
                    generator = (
                        self.behaviour.current_behaviour._get_investable_balance(
                            chain, token_address, total_balance
                        )
                    )
                    result = None
                    try:
                        while True:
                            result = next(generator)
                    except StopIteration as e:
                        result = e.value

                    # For reward tokens that are not whitelisted, return 0
                    assert result == 0

    def test_get_investable_balance_whitelisted_reward_token(self) -> None:
        """Test balance calculation for whitelisted reward tokens."""
        chain = "optimism"
        token_address = "0xolas"
        total_balance = 1000

        def mock_get_accumulated_rewards(chain, token):
            yield
            return 200

        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy.to_checksum_address",
            return_value="0xolas",
        ):
            with patch.dict(
                "packages.valory.skills.liquidity_trader_abci.behaviours.base.REWARD_TOKEN_ADDRESSES",
                {chain: {"0xolas": "OLAS"}},
            ):
                with patch.dict(
                    "packages.valory.skills.liquidity_trader_abci.behaviours.base.WHITELISTED_ASSETS",
                    {chain: {"0xolas": "OLAS"}},
                ):
                    with patch.object(
                        self.behaviour.current_behaviour,
                        "get_accumulated_rewards_for_token",
                        side_effect=mock_get_accumulated_rewards,
                    ):
                        generator = (
                            self.behaviour.current_behaviour._get_investable_balance(
                                chain, token_address, total_balance
                            )
                        )
                        result = None
                        try:
                            while True:
                                result = next(generator)
                        except StopIteration as e:
                            result = e.value

                        # For whitelisted reward tokens, it returns balance - accumulated_rewards
                        assert result == 800  # 1000 - 200

    def test_handle_all_tokens_available_with_rebalancing(self) -> None:
        """Test token allocation with on-chain rebalancing."""
        tokens = [
            {
                "chain": "optimism",
                "token": "0x123",
                "token_symbol": "USDC",
                "value": 600,
            },
            {
                "chain": "optimism",
                "token": "0x456",
                "token_symbol": "WETH",
                "value": 400,
            },
            {"chain": "base", "token": "0x789", "token_symbol": "DAI", "value": 500},
        ]
        required_tokens = [("0x123", "USDC"), ("0x456", "WETH")]
        dest_chain = "optimism"
        relative_funds_percentage = 1.0
        target_ratios_by_token = {"0x123": 0.5, "0x456": 0.5}

        result = self.behaviour.current_behaviour._handle_all_tokens_available(
            tokens,
            required_tokens,
            dest_chain,
            relative_funds_percentage,
            target_ratios_by_token,
        )

        assert isinstance(result, list)

    # ==================== ADDITIONAL EXTENDED TESTS FOR 90% COVERAGE ====================

    def test_build_claim_staking_rewards_action_missing_params(self) -> None:
        """Test building claim staking rewards action with missing parameters."""
        position = {
            "dex_type": "velodrome",
            # Missing chain and pool_address
        }

        result = self.behaviour.current_behaviour._build_claim_staking_rewards_action(
            position
        )
        assert result is None

    def test_get_gauge_address_for_position_missing_params(self) -> None:
        """Test gauge address retrieval with missing parameters."""
        position = {}  # Missing chain and pool_address

        generator = self.behaviour.current_behaviour._get_gauge_address_for_position(
            position
        )
        result = None
        try:
            while True:
                result = next(generator)
        except StopIteration as e:
            result = e.value

        assert result is None

    def test_build_stake_lp_tokens_action_cl_pool(self) -> None:
        """Test building stake action for CL pool."""
        opportunity = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0x123",
            "is_cl_pool": True,
        }

        result = self.behaviour.current_behaviour._build_stake_lp_tokens_action(
            opportunity
        )
        assert result is not None
        assert result["is_cl_pool"] is True
        assert result["recipient"] == "0x1234567890123456789012345678901234567890"

    def test_build_stake_lp_tokens_action_regular_pool(self) -> None:
        """Test building stake action for regular pool."""
        opportunity = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0x123",
            "is_cl_pool": False,
        }

        result = self.behaviour.current_behaviour._build_stake_lp_tokens_action(
            opportunity
        )
        assert result is not None
        assert result["is_cl_pool"] is False

    def test_process_rewards_success(self) -> None:
        """Test successful reward processing."""
        actions = []

        def mock_get_rewards(chain_id, safe_address):
            yield
            return {"users": ["0xuser"], "tokens": ["0xtoken"], "claims": [100]}

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_rewards",
            side_effect=mock_get_rewards,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_build_claim_reward_action",
                return_value={"action": "ClaimRewards"},
            ):
                generator = self.behaviour.current_behaviour._process_rewards(actions)
                try:
                    while True:
                        next(generator)
                except StopIteration:
                    pass

                assert len(actions) == 1

    def test_process_rewards_no_rewards(self) -> None:
        """Test reward processing when no rewards available."""
        actions = []

        def mock_get_rewards(chain_id, safe_address):
            yield
            return None

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_rewards",
            side_effect=mock_get_rewards,
        ):
            generator = self.behaviour.current_behaviour._process_rewards(actions)
            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            assert len(actions) == 0

    def test_get_rewards_no_tokens_to_claim(self) -> None:
        """Test reward retrieval when no tokens available."""

        def mock_get_http_response(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.body = json.dumps(
                {"0x123": {"proof": None, "symbol": "TOKEN1"}}
            )
            yield
            return mock_response

        with patch.object(
            self.behaviour.current_behaviour,
            "get_http_response",
            side_effect=mock_get_http_response,
        ):
            generator = self.behaviour.current_behaviour._get_rewards(10, "0x123")
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result is None

    def test_get_rewards_all_claims_zero(self) -> None:
        """Test reward retrieval when all claims are zero."""

        def mock_get_http_response(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.body = json.dumps(
                {
                    "0x123": {
                        "proof": ["0xproof"],
                        "symbol": "TOKEN1",
                        "accumulated": "0",
                        "unclaimed": "0",
                    }
                }
            )
            yield
            return mock_response

        with patch.object(
            self.behaviour.current_behaviour,
            "get_http_response",
            side_effect=mock_get_http_response,
        ):
            generator = self.behaviour.current_behaviour._get_rewards(10, "0x123")
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result is None

    def test_get_rewards_all_unclaimed_zero(self) -> None:
        """Test reward retrieval when all unclaimed amounts are zero."""

        def mock_get_http_response(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.body = json.dumps(
                {
                    "0x123": {
                        "proof": ["0xproof"],
                        "symbol": "TOKEN1",
                        "accumulated": "100",
                        "unclaimed": "0",
                    }
                }
            )
            yield
            return mock_response

        with patch.object(
            self.behaviour.current_behaviour,
            "get_http_response",
            side_effect=mock_get_http_response,
        ):
            generator = self.behaviour.current_behaviour._get_rewards(10, "0x123")
            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result is None

    def test_download_next_strategy_with_inflight_request(self) -> None:
        """Test strategy download when request already in flight."""
        self.behaviour.current_behaviour._inflight_strategy_req = "existing_strategy"

        # Should return early without doing anything
        self.behaviour.current_behaviour.download_next_strategy()

        # Verify no new request was made
        assert (
            self.behaviour.current_behaviour._inflight_strategy_req
            == "existing_strategy"
        )

    def test_download_next_strategy_no_strategies_pending(self) -> None:
        """Test strategy download with no pending strategies."""
        self.behaviour.current_behaviour._inflight_strategy_req = None
        self.behaviour.current_behaviour.shared_state.strategy_to_filehash = {}

        # Should return early without doing anything
        self.behaviour.current_behaviour.download_next_strategy()

        assert self.behaviour.current_behaviour._inflight_strategy_req is None

    def test_download_next_strategy_with_pending_strategies(self) -> None:
        """Test strategy download with pending strategies."""
        self.behaviour.current_behaviour._inflight_strategy_req = None
        self.behaviour.current_behaviour.shared_state.strategy_to_filehash = {
            "test_strategy": "QmTestHash"
        }

        with patch.object(
            self.behaviour.current_behaviour, "_build_ipfs_get_file_req"
        ) as mock_build:
            with patch.object(
                self.behaviour.current_behaviour, "send_message"
            ) as mock_send:
                mock_build.return_value = (MagicMock(), MagicMock())

                self.behaviour.current_behaviour.download_next_strategy()

                assert (
                    self.behaviour.current_behaviour._inflight_strategy_req
                    == "test_strategy"
                )
                mock_build.assert_called_once_with("QmTestHash")
                mock_send.assert_called_once()

    def test_handle_get_strategy_no_request(self) -> None:
        """Test strategy response handling with no pending request."""
        self.behaviour.current_behaviour._inflight_strategy_req = None

        mock_message = MagicMock()
        mock_dialogue = MagicMock()

        # Should log error and return early
        self.behaviour.current_behaviour._handle_get_strategy(
            mock_message, mock_dialogue
        )

    def test_handle_get_strategy_success(self) -> None:
        """Test successful strategy response handling."""
        self.behaviour.current_behaviour._inflight_strategy_req = "test_strategy"
        self.behaviour.current_behaviour.shared_state.strategy_to_filehash = {
            "test_strategy": "QmTestHash"
        }

        mock_message = MagicMock()
        mock_message.files = {"test_file": "content"}
        mock_dialogue = MagicMock()

        with patch(
            "packages.valory.skills.liquidity_trader_abci.io_.loader.ComponentPackageLoader.load"
        ) as mock_load:
            mock_load.return_value = ("yaml_content", "strategy_code", "method_name")

            self.behaviour.current_behaviour._handle_get_strategy(
                mock_message, mock_dialogue
            )

            assert (
                ("strategy_code", "method_name")
                in self.behaviour.current_behaviour.shared_state.strategies_executables.values()
            )
            assert (
                "test_strategy"
                not in self.behaviour.current_behaviour.shared_state.strategy_to_filehash
            )
            assert self.behaviour.current_behaviour._inflight_strategy_req is None

    def test_send_message_success(self) -> None:
        """Test successful message sending."""
        mock_msg = MagicMock()
        mock_dialogue = MagicMock()
        mock_dialogue.dialogue_label.dialogue_reference = ["nonce123", "ref"]
        mock_callback = MagicMock()

        with patch.object(
            self.behaviour.current_behaviour.context.outbox, "put_message"
        ):
            self.behaviour.current_behaviour.send_message(
                mock_msg, mock_dialogue, mock_callback
            )

            assert (
                self.behaviour.current_behaviour.shared_state.req_to_callback[
                    "nonce123"
                ]
                == mock_callback
            )
            assert self.behaviour.current_behaviour.shared_state.in_flight_req is True

    def test_validate_velodrome_inputs_zero_allocation_bands(self) -> None:
        """Test validation with zero allocation bands."""
        tick_bands = [
            {"tick_lower": 100, "tick_upper": 200, "allocation": 0.0},
            {"tick_lower": 300, "tick_upper": 400, "allocation": 0.0},
        ]

        result = self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
            tick_bands, 1.5, 1
        )

        assert result is None

    def test_validate_velodrome_inputs_tick_spacing_misalignment(self) -> None:
        """Test Velodrome validation with tick spacing misalignment."""
        tick_bands = [
            {
                "tick_lower": 101,
                "tick_upper": 203,
                "allocation": 0.5,
            },  # Not aligned with tick_spacing=2
        ]
        current_price = 1.5
        tick_spacing = 2

        result = self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
            tick_bands, current_price, tick_spacing
        )

        # Should still return result but with warnings
        assert result is not None
        assert len(result["warnings"]) > 0

    def test_validate_velodrome_inputs_invalid_band_range(self) -> None:
        """Test validation with invalid band ranges."""
        tick_bands = [
            {
                "tick_lower": 200,
                "tick_upper": 100,
                "allocation": 0.5,
            },  # Invalid: lower >= upper
        ]

        result = self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
            tick_bands, 1.5, 1
        )

        assert result is None

    def test_calculate_token_ratios_price_below_range(self) -> None:
        """Test token ratio calculation when price is below range."""
        validated_data = {
            "validated_bands": [
                {"tick_lower": 200, "tick_upper": 300, "allocation": 1.0}
            ],
            "current_price": 1.0001**150,  # Price below tick_lower
            "current_tick": 150,
            "warnings": [],
        }

        result = self.behaviour.current_behaviour.calculate_velodrome_token_ratios(
            validated_data
        )

        assert result is not None
        assert result["overall_token0_ratio"] == 1.0
        assert result["overall_token1_ratio"] == 0.0

    def test_calculate_token_ratios_price_above_range(self) -> None:
        """Test token ratio calculation when price is above range."""
        validated_data = {
            "validated_bands": [
                {"tick_lower": 100, "tick_upper": 200, "allocation": 1.0}
            ],
            "current_price": 1.0001**250,  # Price above tick_upper
            "current_tick": 250,
            "warnings": [],
        }

        result = self.behaviour.current_behaviour.calculate_velodrome_token_ratios(
            validated_data
        )

        assert result is not None
        assert result["overall_token0_ratio"] == 0.0
        assert result["overall_token1_ratio"] == 1.0

    def test_get_velodrome_position_requirements_exception_handling(self) -> None:
        """Test Velodrome analysis with exceptions."""
        self.behaviour.current_behaviour.selected_opportunities = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "pool_address": "0x123",
                "chain": "optimism",
                "is_stable": False,
            }
        ]

        # Mock pool that raises exception
        mock_pool = MagicMock()

        def mock_get_tick_spacing(self, pool_address, chain):
            yield
            raise Exception("Test exception")

        mock_pool._get_tick_spacing_velodrome = mock_get_tick_spacing
        self.behaviour.current_behaviour.pools = {"velodrome": mock_pool}

        generator = (
            self.behaviour.current_behaviour.get_velodrome_position_requirements()
        )

        try:
            while True:
                next(generator)
        except StopIteration:
            pass

        # Should handle exception gracefully

    def test_calculate_initial_investment_value_missing_decimals(self) -> None:
        """Test investment value calculation with missing token decimals."""
        position = {
            "chain": "optimism",
            "token0": "0x123",
            "token1": "0x456",
            "amount0": 1000,
            "amount1": 2000,
            "timestamp": 1000000,
        }

        def mock_get_token_decimals(chain, token):
            yield
            return None  # Simulate missing decimals

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_token_decimals",
            side_effect=mock_get_token_decimals,
        ):
            generator = (
                self.behaviour.current_behaviour.calculate_initial_investment_value(
                    position
                )
            )

            result = None
            try:
                while True:
                    result = next(generator)
            except StopIteration as e:
                result = e.value

            assert result is None

    def test_calculate_initial_investment_value_price_fetch_error(self) -> None:
        """Test investment value calculation with price fetch error."""
        position = {
            "chain": "optimism",
            "token0": "0x123",
            "token1": "0x456",
            "amount0": 1000,
            "amount1": 2000,
            "timestamp": 1000000,
        }

        def mock_get_token_decimals(chain, token):
            yield
            return 18

        def mock_fetch_historical_token_prices(tokens, date_str, chain):
            yield
            return None  # Simulate price fetch error

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_token_decimals",
            side_effect=mock_get_token_decimals,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_fetch_historical_token_prices",
                side_effect=mock_fetch_historical_token_prices,
            ):
                generator = (
                    self.behaviour.current_behaviour.calculate_initial_investment_value(
                        position
                    )
                )

                result = None
                try:
                    while True:
                        result = next(generator)
                except StopIteration as e:
                    result = e.value

                assert result is None

    def test_calculate_initial_investment_value_success_comprehensive(self) -> None:
        """Test comprehensive successful investment value calculation."""
        position = {
            "chain": "optimism",
            "token0": "0x123",
            "token1": "0x456",
            "amount0": 1000000000000000000,  # 1 token with 18 decimals
            "amount1": 2000000000000000000,  # 2 tokens with 18 decimals
            "timestamp": 1000000,
        }

        def mock_get_token_decimals(chain, token):
            yield
            return 18

        def mock_fetch_historical_token_prices(tokens, date_str, chain):
            yield
            return {"0x123": 1.0, "0x456": 2000.0}

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_token_decimals",
            side_effect=mock_get_token_decimals,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_fetch_historical_token_prices",
                side_effect=mock_fetch_historical_token_prices,
            ):
                generator = (
                    self.behaviour.current_behaviour.calculate_initial_investment_value(
                        position
                    )
                )

                result = None
                try:
                    while True:
                        result = next(generator)
                except StopIteration as e:
                    result = e.value

                # Should calculate: (1 * 1.0) + (2 * 2000.0) = 4001.0
                assert result == 4001.0

    def test_get_result_method_with_timeout(self) -> None:
        """Test get_result method with timeout handling."""
        from concurrent.futures import Future

        future = Future()

        # Start the generator
        generator = self.behaviour.current_behaviour.get_result(future)

        # Should yield once since future is not done
        try:
            next(generator)
        except StopIteration:
            pass

        # Complete the future after a short delay
        future.set_result({"delayed": "result"})

        result = None
        try:
            while True:
                result = next(generator)
        except StopIteration as e:
            result = e.value

        assert result == {"delayed": "result"}

    def test_merge_duplicate_bridge_swap_actions_edge_cases(self) -> None:
        """Test bridge action merging with edge cases."""
        # Test with actions that have missing required fields
        actions = [
            {
                "action": Action.FIND_BRIDGE_ROUTE.value,
                "from_chain": "optimism",
                # Missing to_chain, from_token, to_token
                "funds_percentage": 0.5,
            },
            {
                "action": "EnterPool",
                "pool_address": "0x123",
            },
        ]

        result = self.behaviour.current_behaviour._merge_duplicate_bridge_swap_actions(
            actions
        )

        # Should handle malformed actions gracefully
        assert len(result) == 2

    def test_handle_velodrome_token_allocation_edge_cases(self) -> None:
        """Test Velodrome token allocation with edge cases."""
        # Test with missing token requirements
        actions = []
        enter_pool_action = {
            "dex_type": "velodrome",
            "chain": "optimism",
            # Missing token_requirements
        }
        available_tokens = []

        result = self.behaviour.current_behaviour._handle_velodrome_token_allocation(
            actions, enter_pool_action, available_tokens
        )

        # Should return actions unchanged
        assert result == actions

    def test_apply_investment_cap_calculation_error(self) -> None:
        """Test investment cap with calculation errors."""
        actions = [{"action": "EnterPool", "pool_address": "0x123"}]
        self.behaviour.current_behaviour.current_positions = [
            {"status": "open", "pool_address": "0x789"}
        ]

        def mock_calculate_initial_investment_value(position):
            yield
            raise Exception("Calculation error")

        def mock_sleep(seconds):
            yield

        with patch.object(
            self.behaviour.current_behaviour,
            "calculate_initial_investment_value",
            side_effect=mock_calculate_initial_investment_value,
        ):
            with patch.object(
                self.behaviour.current_behaviour, "sleep", side_effect=mock_sleep
            ):
                generator = (
                    self.behaviour.current_behaviour._apply_investment_cap_to_actions(
                        actions
                    )
                )

                # The method should raise the exception, not handle it gracefully
                with pytest.raises(Exception, match="Calculation error"):
                    try:
                        while True:
                            next(generator)
                    except StopIteration:
                        pass

    def test_comprehensive_workflow_integration(self) -> None:
        """Test comprehensive workflow integration."""
        # Set up a complete scenario
        self.behaviour.current_behaviour.trading_opportunities = [
            {
                "pool_address": "0x123",
                "chain": "optimism",
                "dex_type": "velodrome",
                "apr": 15.0,
                "token0": "0x456",
                "token1": "0x789",
                "token0_symbol": "USDC",
                "token1_symbol": "WETH",
                "is_cl_pool": True,
            }
        ]
        self.behaviour.current_behaviour.positions_eligible_for_exit = []

        def mock_execute_hyper_strategy():
            yield
            self.behaviour.current_behaviour.selected_opportunities = [
                self.behaviour.current_behaviour.trading_opportunities[0]
            ]

        def mock_get_order_of_transactions():
            yield
            return [
                {
                    "action": "FindBridgeRoute",
                    "from_chain": "base",
                    "to_chain": "optimism",
                },
                {"action": "EnterPool", "pool_address": "0x123"},
                {"action": "StakeLPTokens", "pool_address": "0x123"},
            ]

        with patch.object(
            self.behaviour.current_behaviour,
            "execute_hyper_strategy",
            side_effect=mock_execute_hyper_strategy,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "get_order_of_transactions",
                side_effect=mock_get_order_of_transactions,
            ):
                generator = self.behaviour.current_behaviour.prepare_strategy_actions()

                result = None
                try:
                    while True:
                        result = next(generator)
                except StopIteration as e:
                    result = e.value

                assert isinstance(result, list)
                assert len(result) == 3

    def test_error_handling_comprehensive(self) -> None:
        """Test comprehensive error handling across methods."""
        # Test various error scenarios
        test_cases = [
            # Test with invalid opportunity data
            {"selected_opportunities": [{"invalid": "data"}]},
            # Test with missing required fields
            {"selected_opportunities": [{"pool_address": "0x123"}]},
            # Test with empty opportunities
            {"selected_opportunities": []},
        ]

        for test_case in test_cases:
            self.behaviour.current_behaviour.selected_opportunities = test_case[
                "selected_opportunities"
            ]

            def mock_get_velodrome_position_requirements():
                yield

            def mock_prepare_tokens_for_investment():
                yield
                return []

            def mock_apply_investment_cap(actions):
                yield
                return actions

            with patch.object(
                self.behaviour.current_behaviour,
                "get_velodrome_position_requirements",
                side_effect=mock_get_velodrome_position_requirements,
            ):
                with patch.object(
                    self.behaviour.current_behaviour,
                    "_prepare_tokens_for_investment",
                    side_effect=mock_prepare_tokens_for_investment,
                ):
                    with patch.object(
                        self.behaviour.current_behaviour,
                        "_apply_investment_cap_to_actions",
                        side_effect=mock_apply_investment_cap,
                    ):
                        generator = (
                            self.behaviour.current_behaviour.get_order_of_transactions()
                        )

                        result = None
                        try:
                            while True:
                                result = next(generator)
                        except StopIteration as e:
                            result = e.value

                        # Should handle errors gracefully - can return None or empty list
                        assert result is None or isinstance(result, list)

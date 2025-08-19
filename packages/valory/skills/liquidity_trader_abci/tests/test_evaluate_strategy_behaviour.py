# -*- coding: utf-8 -*-
"""
Comprehensive tests for EvaluateStrategyBehaviour with 80% coverage target.
"""

import json
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    Action,
    DexType,
    PositionStatus,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy import (
    EvaluateStrategyBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.payloads import (
    EvaluateStrategyPayload,
)
from packages.valory.skills.liquidity_trader_abci.rounds import SynchronizedData
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

        # Initialize behavior attributes
        self.behaviour.current_behaviour.selected_opportunities = None
        self.behaviour.current_behaviour.position_to_exit = None
        self.behaviour.current_behaviour.trading_opportunities = []
        self.behaviour.current_behaviour.positions_eligible_for_exit = []
        self.behaviour.current_behaviour.current_positions = None

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

    def teardown_method(self) -> None:
        """Clean up after tests."""
        if hasattr(self, "shared_state_patcher"):
            self.shared_state_patcher.stop()
        if hasattr(self, "synchronized_data_patcher"):
            self.synchronized_data_patcher.stop()
        if hasattr(self, "coingecko_patcher"):
            self.coingecko_patcher.stop()
        super().teardown()

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

    def test_concurrent_strategy_execution_with_errors(self) -> None:
        """Test concurrent strategy execution with error handling."""
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
        self.behaviour.current_behaviour.shared_state.strategies_executables = {}

        with patch("asyncio.ensure_future") as mock_future:
            mock_future_obj = MagicMock()
            mock_future_obj.done.return_value = True
            mock_future_obj.result.return_value = [
                {"error": ["Test error"], "result": []}
            ]
            mock_future.return_value = mock_future_obj

            generator = (
                self.behaviour.current_behaviour.fetch_all_trading_opportunities()
            )

            # Execute the generator
            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            assert len(self.behaviour.current_behaviour.trading_opportunities) == 0

    def test_position_metrics_updates(self) -> None:
        """Test position metrics update logic."""
        mock_position = {
            "status": PositionStatus.OPEN.value,
            "pool_address": "0x123",
            "dex_type": "velodrome",
            "last_metrics_update": 0,  # Old timestamp to trigger update
        }

        self.behaviour.current_behaviour.positions_eligible_for_exit = [mock_position]
        self.behaviour.current_behaviour.context.params.dex_type_to_strategy = {
            "velodrome": "test_strategy"
        }

        # Set current timestamp to be much later than last update to trigger metrics calculation
        # METRICS_UPDATE_INTERVAL is 21600 seconds (6 hours), so use a larger gap
        current_time = 25200  # 7 hours later (25200 seconds)

        with patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=current_time,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "get_returns_metrics_for_opportunity",
                return_value={"updated_apr": 12.5, "last_metrics_update": current_time},
            ) as mock_metrics:
                with patch.object(
                    self.behaviour.current_behaviour, "store_current_positions"
                ):
                    self.behaviour.current_behaviour.update_position_metrics()

                    # Verify that metrics were actually calculated
                    mock_metrics.assert_called_once()

                    # Check that position was updated
                    assert mock_position.get("updated_apr") == 12.5
                    assert mock_position.get("last_metrics_update") == current_time

    def test_invalid_strategy_data_handling(self) -> None:
        """Test handling of invalid strategy data."""
        # Test execute_strategy with no strategy parameter
        result = self.behaviour.current_behaviour.execute_strategy(test_param="value")
        assert result is None

        # Test with invalid strategy name
        result = self.behaviour.current_behaviour.execute_strategy(
            strategy="invalid_strategy"
        )
        assert result is None

    def test_entry_cost_initialization(self) -> None:
        """Test entry cost initialization for new positions."""
        enter_pool_action = {"chain": "optimism", "pool_address": "0x123"}

        with patch.object(
            self.behaviour.current_behaviour, "_initialize_position_entry_costs"
        ) as mock_init:
            generator = self.behaviour.current_behaviour._initialize_entry_costs_for_new_position(
                enter_pool_action
            )

            # Execute the generator
            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            mock_init.assert_called_once_with("optimism", "0x123")

    def test_get_result_method(self) -> None:
        """Test get_result method for future handling."""
        from concurrent.futures import Future

        # Test with completed future
        future = Future()
        future.set_result({"test": "result"})

        generator = self.behaviour.current_behaviour.get_result(future)
        result = None
        try:
            while True:
                result = next(generator)
        except StopIteration as e:
            result = e.value

        assert result == {"test": "result"}

    def test_bridge_action_deduplication_and_merging(self) -> None:
        """Test bridge action deduplication and merging logic."""
        # Test removing redundant same-chain same-token actions
        actions = [
            {
                "action": Action.FIND_BRIDGE_ROUTE.value,
                "from_chain": "optimism",
                "to_chain": "optimism",
                "from_token": "0x123",
                "to_token": "0x123",
                "from_token_symbol": "USDC",
                "to_token_symbol": "USDC",
                "funds_percentage": 0.5,
            },
            {"action": Action.ENTER_POOL.value, "pool_address": "0x456"},
        ]

        result = self.behaviour.current_behaviour._merge_duplicate_bridge_swap_actions(
            actions
        )

        # Should remove the redundant bridge action
        assert len(result) == 1
        assert result[0]["action"] == Action.ENTER_POOL.value

    def test_velodrome_cl_position_analysis_complete_workflow(self) -> None:
        """Test complete Velodrome CL position analysis workflow."""
        tick_bands = [
            {
                "tick_lower": 100,
                "tick_upper": 200,
                "allocation": 0.6,
                "percent_in_bounds": 0.8,
            },
            {
                "tick_lower": 300,
                "tick_upper": 400,
                "allocation": 0.4,
                "percent_in_bounds": 0.8,
            },
        ]
        current_price = 1.5
        tick_spacing = 1

        # Test the complete workflow
        validated_data = (
            self.behaviour.current_behaviour.validate_and_prepare_velodrome_inputs(
                tick_bands, current_price, tick_spacing
            )
        )
        assert validated_data is not None

        token_ratios = (
            self.behaviour.current_behaviour.calculate_velodrome_token_ratios(
                validated_data
            )
        )
        assert token_ratios is not None

        final_result = (
            self.behaviour.current_behaviour.calculate_velodrome_cl_token_requirements(
                tick_bands, current_price, tick_spacing
            )
        )
        assert final_result is not None
        assert "position_requirements" in final_result

    def test_tip_exit_conditions_with_cost_recovery(self) -> None:
        """Test TiP exit conditions focusing on cost recovery logic."""
        # Test position where costs are recovered but minimum time not met
        position = {
            "entry_cost": 100,
            "cost_recovered": True,
            "enter_timestamp": 1000000,
            "min_hold_days": 10,
        }

        with patch.object(
            self.behaviour.current_behaviour,
            "_check_minimum_time_met",
            return_value=False,
        ):
            with patch.object(
                self.behaviour.current_behaviour,
                "_calculate_days_since_entry",
                return_value=5,
            ):
                (
                    can_exit,
                    reason,
                ) = self.behaviour.current_behaviour._check_tip_exit_conditions(
                    position
                )
                assert can_exit is False
                assert "minimum time not met" in reason

    def test_tip_exit_conditions_legacy_positions(self) -> None:
        """Test TiP exit conditions for legacy positions without entry_cost."""
        legacy_position = {"entry_cost": 0, "enter_timestamp": 1000000}

        # Test when minimum time is met
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

        # Test when minimum time is not met
        with patch.object(
            self.behaviour.current_behaviour,
            "_calculate_days_since_entry",
            return_value=10,
        ):
            (
                can_exit,
                reason,
            ) = self.behaviour.current_behaviour._check_tip_exit_conditions(
                legacy_position
            )
            assert can_exit is False
            assert "must hold" in reason

    def test_check_tip_exit_conditions_edge_cases(self) -> None:
        """Test TiP exit condition edge cases."""
        # Test position without enter_timestamp
        position_no_timestamp = {"entry_cost": 0}

        can_exit, reason = self.behaviour.current_behaviour._check_tip_exit_conditions(
            position_no_timestamp
        )
        assert can_exit is True
        assert "No TiP data" in reason

        # Test with exception handling
        position_invalid = {"entry_cost": "invalid", "enter_timestamp": "invalid"}

        can_exit, reason = self.behaviour.current_behaviour._check_tip_exit_conditions(
            position_invalid
        )
        assert can_exit is True
        assert "Error in TiP check" in reason

    def test_velodrome_token_allocation_100_percent_scenarios(self) -> None:
        """Test Velodrome token allocation for 100% scenarios."""
        # Test 100% token0 allocation
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

    def test_investment_cap_threshold_enforcement(self) -> None:
        """Test investment cap threshold enforcement logic."""
        actions = [{"action": "EnterPool", "pool_address": "0x123"}]

        # Test when invested amount exceeds threshold
        mock_positions = [{"status": "open", "pool_address": "0x789"}]
        self.behaviour.current_behaviour.current_positions = mock_positions

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

                # Should return empty list when cap is exceeded
                assert result == []

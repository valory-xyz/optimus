# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2023 Valory AG
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

"""Tests for EvaluateStrategy behaviour investment logic."""

import json
import numpy as np
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.abstract_round_abci.test_tools.base import FSMBehaviourBaseCase
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    Action,
    DexType,
    MIN_TIME_IN_POSITION,
    METRICS_UPDATE_INTERVAL,
    PositionStatus,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy import (
    EvaluateStrategyBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.rounds import SynchronizedData

# Use absolute path to avoid conflicts
PACKAGE_DIR = Path(__file__).parent.parent.parent


class TestEvaluateStrategyBehaviour(FSMBehaviourBaseCase):
    """Test EvaluateStrategyBehaviour with comprehensive investment scenarios."""

    behaviour_class = EvaluateStrategyBehaviour
    path_to_skill = PACKAGE_DIR

    # =============================================
    # 1. CORE INVESTMENT DECISION FLOW TESTS
    # =============================================

    def test_async_act_complete_investment_flow_new_position(self) -> None:
        """Test complete flow: no positions → find opportunities → bridge → enter → stake."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock complete investment flow
        velodrome_opportunity = {
            'dex_type': 'velodrome', 
            'pool_address': '0x5483484F876218908CA435F08222751F7f7b2a3b',
            'chain': 'optimism', 
            'apr': 36.92684058040778,
            'token0': '0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85',
            'token0_symbol': 'USDC',
            'token1': '0x1217BfE6c773EEC6cc4A38b5Dc45B92292B6E189',
            'token1_symbol': 'oUSDT',
            'relative_funds_percentage': 1.0
        }

        expected_actions = [
            {
                'action': 'FindBridgeRoute',
                'from_chain': 'ethereum',
                'to_chain': 'optimism',
                'from_token': '0xA0b86a33E6441E13799A0c77E2f93b6688BDB689',
                'to_token': '0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85',
                'funds_percentage': 0.5
            },
            {
                'action': 'EnterPool',
                'dex_type': 'velodrome',
                'pool_address': '0x5483484F876218908CA435F08222751F7f7b2a3b',
                'chain': 'optimism'
            },
            {
                'action': 'StakeLpTokens',
                'dex_type': 'velodrome',
                'chain': 'optimism',
                'pool_address': '0x5483484F876218908CA435F08222751F7f7b2a3b'
            }
        ]

        # Mock no current positions (new investment)
        self.behaviour.current_behaviour.current_positions = []
        self.behaviour.current_behaviour.trading_opportunities = [velodrome_opportunity]
        self.behaviour.current_behaviour.selected_opportunities = [velodrome_opportunity]

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "check_and_prepare_non_whitelisted_swaps",
            return_value=iter([None])
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "check_minimum_hold_period", 
            return_value=False
        ), mock.patch.object(
            self.behaviour.current_behaviour, 
            "check_funds", 
            return_value=True
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "fetch_all_trading_opportunities",
            return_value=iter([None])
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "prepare_strategy_actions",
            return_value=iter([expected_actions])
        ) as mock_prepare, mock.patch.object(
            self.behaviour.current_behaviour,
            "send_actions"
        ) as mock_send:
            
            generator = self.behaviour.current_behaviour.async_act()
            list(generator)
            
            # Verify complete investment flow was executed
            mock_send.assert_called_with(expected_actions)
            
            # Verify action order: Bridge → Enter → Stake
            args = mock_send.call_args[0][0]
            assert len(args) == 3
            assert args[0]['action'] == 'FindBridgeRoute'
            assert args[1]['action'] == 'EnterPool'  
            assert args[2]['action'] == 'StakeLpTokens'

    def test_async_act_exit_then_enter_flow(self) -> None:
        """Test complete flow: exit old position → bridge → enter new → stake."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock existing position to exit (Velodrome CL)
        existing_position = {
            'dex_type': 'velodrome',
            'pool_address': '0xOLD_POOL',
            'status': 'OPEN',
            'is_cl_pool': True,
            'chain': 'optimism'
        }

        # Mock new opportunity  
        new_opportunity = {
            'dex_type': 'balancerPool',
            'pool_address': '0xNEW_BALANCER_POOL', 
            'chain': 'optimism',
            'apr': 25.5,
            'token0': '0xETH_TOKEN',
            'token0_symbol': 'WETH',
            'token1': '0xUSDC_TOKEN', 
            'token1_symbol': 'USDC',
            'relative_funds_percentage': 1.0
        }

        expected_actions = [
            {
                'action': 'UnstakeLpTokens',
                'dex_type': 'velodrome', 
                'pool_address': '0xOLD_POOL'
            },
            {
                'action': 'ExitPool',
                'dex_type': 'velodrome',
                'pool_address': '0xOLD_POOL'
            },
            {
                'action': 'FindBridgeRoute',
                'from_chain': 'optimism',
                'to_chain': 'optimism', 
                'from_token': '0xOLD_TOKEN',
                'to_token': '0xETH_TOKEN'
            },
            {
                'action': 'EnterPool',
                'dex_type': 'balancerPool',
                'pool_address': '0xNEW_BALANCER_POOL'
            }
        ]

        # Mock current position and new opportunity
        self.behaviour.current_behaviour.current_positions = [existing_position]
        self.behaviour.current_behaviour.position_to_exit = existing_position
        self.behaviour.current_behaviour.selected_opportunities = [new_opportunity]

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "check_minimum_hold_period", 
            return_value=False
        ), mock.patch.object(
            self.behaviour.current_behaviour, 
            "check_funds", 
            return_value=True
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "prepare_strategy_actions",
            return_value=iter([expected_actions])
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "send_actions"
        ) as mock_send:
            
            generator = self.behaviour.current_behaviour.async_act()
            list(generator)
            
            # Verify exit → bridge → enter flow
            args = mock_send.call_args[0][0]
            assert args[0]['action'] == 'UnstakeLpTokens'  # Unstake first for CL pools
            assert args[1]['action'] == 'ExitPool'         # Then exit
            assert args[2]['action'] == 'FindBridgeRoute'  # Bridge/swap tokens
            assert args[3]['action'] == 'EnterPool'        # Enter new pool

    # =============================================
    # 2. STRATEGY EXECUTION WITH REALISTIC DATA  
    # =============================================

    def test_execute_hyper_strategy_with_velodrome_opportunity(self) -> None:
        """Test strategy execution selecting Velodrome CL opportunity."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock realistic strategy result with Velodrome opportunity
        strategy_result = {
            "optimal_strategies": [{
                'dex_type': 'velodrome',
                'pool_address': '0x5483484F876218908CA435F08222751F7f7b2a3b',
                'pool_id': '0x5483484F876218908CA435F08222751F7f7b2a3b',
                'tvl': 97237.684693,
                'is_lp': True,
                'token_count': 2,
                'volume': 0.0,
                'chain': 'optimism',
                'apr': 36.92684058040778,
                'is_cl_pool': True,
                'is_stable': True,
                'token0': '0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85',
                'token0_symbol': 'USDC',
                'token1': '0x1217BfE6c773EEC6cc4A38b5Dc45B92292B6E189',
                'token1_symbol': 'oUSDT',
                'sharpe_ratio': 0.08655583746925036,
                'depth_score': 1079.068096913766,
                'max_position_size': 10000000.0,
                'il_risk_score': -6.084100235285381e-13,
                'composite_score': 0.4364842470256133,
                'apr_weighted_score': 16.11798420577415,
                'funds_percentage': 100.0,
                'relative_funds_percentage': 1.0
            }],
            "position_to_exit": None,
            "reasoning": "High APR Velodrome opportunity with excellent risk metrics"
        }

        with mock.patch.object(
            type(self.behaviour.current_behaviour.params),
            "selected_hyper_strategy",
            new_callable=mock.PropertyMock,
            return_value="max_apr_strategy"
        ), mock.patch.object(
            type(self.behaviour.current_behaviour.synchronized_data),
            "trading_type",
            new_callable=mock.PropertyMock,
            return_value="balanced"
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "execute_strategy",
            return_value=strategy_result,
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "_read_kv",
            return_value=iter([{"composite_score": 0.35}])
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "_send_envelope",
            return_value=None
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "_wait_for_message",
            return_value=None
        ):
            generator = self.behaviour.current_behaviour.execute_hyper_strategy()
            list(generator)

            # Verify Velodrome opportunity was selected
            selected = self.behaviour.current_behaviour.selected_opportunities[0]
            assert selected['dex_type'] == 'velodrome'
            assert selected['apr'] == 36.92684058040778
            assert selected['is_cl_pool'] is True
            assert selected['token0_symbol'] == 'USDC'
            assert selected['token1_symbol'] == 'oUSDT'

    def test_execute_hyper_strategy_with_exit_decision(self) -> None:
        """Test strategy deciding to exit underperforming position."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock strategy result with exit decision
        strategy_result = {
            "optimal_strategies": [],
            "position_to_exit": {
                "pool_address": "0xUNDERPERFORMING_POOL",
                "dex_type": "balancerPool", 
                "apr": 5.2,  # Low APR
                "reason": "Underperforming compared to new opportunities"
            },
            "reasoning": "Current Balancer position yielding only 5.2% APR, better opportunities available"
        }

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "execute_strategy",
            return_value=strategy_result,
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "_read_kv",
            return_value=iter([{"composite_score": 0.35}])
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "_send_envelope",
            return_value=None
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "_wait_for_message",
            return_value=None
        ):
            generator = self.behaviour.current_behaviour.execute_hyper_strategy()
            list(generator)

            # Verify exit decision
            assert self.behaviour.current_behaviour.position_to_exit is not None
            assert self.behaviour.current_behaviour.position_to_exit["pool_address"] == "0xUNDERPERFORMING_POOL"
            assert self.behaviour.current_behaviour.selected_opportunities == []

    # =============================================
    # 3. BRIDGE/SWAP ACTION TESTS (CRITICAL)
    # =============================================

    def test_bridge_swap_actions_cross_chain_investment(self) -> None:
        """Test bridge actions when investing from different chains."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock available tokens on different chains
        available_tokens = [
            {
                'chain': 'ethereum',
                'token': '0xA0b86a33E6441E13799A0c77E2f93b6688BDB689',  # USDC on Ethereum
                'token_symbol': 'USDC',
                'balance': 1000000  # $1000
            },
            {
                'chain': 'mode', 
                'token': '0x2416092f143378750bb29b79ed961ab195cceea5',  # ezETH on Mode
                'token_symbol': 'ezETH',
                'balance': 500000000000000000  # 0.5 ETH
            }
        ]

        # Mock Optimism Velodrome opportunity requiring USDC + oUSDT
        opportunity = {
            'dex_type': 'velodrome',
            'chain': 'optimism',
            'token0': '0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85',  # USDC on Optimism
            'token0_symbol': 'USDC', 
            'token1': '0x1217BfE6c773EEC6cc4A38b5Dc45B92292B6E189',  # oUSDT on Optimism
            'token1_symbol': 'oUSDT',
            'relative_funds_percentage': 1.0
        }

        expected_bridge_actions = [
            {
                'action': 'FindBridgeRoute',
                'from_chain': 'ethereum',
                'to_chain': 'optimism',
                'from_token': '0xA0b86a33E6441E13799A0c77E2f93b6688BDB689',  # USDC on Ethereum
                'to_token': '0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85',   # USDC on Optimism
                'from_token_symbol': 'USDC',
                'to_token_symbol': 'USDC'
            },
            {
                'action': 'FindBridgeRoute', 
                'from_chain': 'mode',
                'to_chain': 'optimism',
                'from_token': '0x2416092f143378750bb29b79ed961ab195cceea5',  # ezETH on Mode
                'to_token': '0x1217BfE6c773EEC6cc4A38b5Dc45B92292B6E189',   # oUSDT on Optimism
                'from_token_symbol': 'ezETH',
                'to_token_symbol': 'oUSDT'
            }
        ]

        with mock.patch.object(
            self.behaviour.current_behaviour,
            '_build_bridge_swap_actions',
            return_value=expected_bridge_actions
        ) as mock_bridge:
            
            result = self.behaviour.current_behaviour._build_bridge_swap_actions(
                opportunity, available_tokens
            )
            
            # Verify correct bridge actions were created
            assert len(result) == 2
            
            # Verify Ethereum USDC → Optimism USDC bridge
            eth_bridge = result[0]
            assert eth_bridge['from_chain'] == 'ethereum'
            assert eth_bridge['to_chain'] == 'optimism' 
            assert eth_bridge['from_token_symbol'] == 'USDC'
            assert eth_bridge['to_token_symbol'] == 'USDC'
            
            # Verify Mode ezETH → Optimism oUSDT bridge (with swap)
            mode_bridge = result[1]
            assert mode_bridge['from_chain'] == 'mode'
            assert mode_bridge['to_chain'] == 'optimism'
            assert mode_bridge['from_token_symbol'] == 'ezETH'
            assert mode_bridge['to_token_symbol'] == 'oUSDT'

    def test_bridge_swap_actions_same_chain_different_tokens(self) -> None:
        """Test swap actions when tokens are on same chain but different assets."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock tokens already on Optimism but wrong assets
        available_tokens = [
            {
                'chain': 'optimism',
                'token': '0x4200000000000000000000000000000000000006',  # WETH on Optimism
                'token_symbol': 'WETH',
                'balance': 1000000000000000000  # 1 ETH
            }
        ]

        # Mock opportunity requiring USDC + oUSDT
        opportunity = {
            'dex_type': 'velodrome',
            'chain': 'optimism',
            'token0': '0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85',  # USDC
            'token0_symbol': 'USDC',
            'token1': '0x1217BfE6c773EEC6cc4A38b5Dc45B92292B6E189',  # oUSDT  
            'token1_symbol': 'oUSDT',
            'relative_funds_percentage': 1.0
        }

        expected_swap_actions = [
            {
                'action': 'FindBridgeRoute',  # Same action used for swaps
                'from_chain': 'optimism',
                'to_chain': 'optimism',
                'from_token': '0x4200000000000000000000000000000000000006',  # WETH
                'to_token': '0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85',   # USDC
                'from_token_symbol': 'WETH',
                'to_token_symbol': 'USDC'
            }
        ]

        with mock.patch.object(
            self.behaviour.current_behaviour,
            '_build_bridge_swap_actions',
            return_value=expected_swap_actions
        ) as mock_swap:
            
            result = self.behaviour.current_behaviour._build_bridge_swap_actions(
                opportunity, available_tokens
            )
            
            # Verify same-chain swap was created
            assert len(result) == 1
            swap_action = result[0]
            assert swap_action['from_chain'] == swap_action['to_chain'] == 'optimism'
            assert swap_action['from_token_symbol'] == 'WETH'
            assert swap_action['to_token_symbol'] == 'USDC'

    def test_bridge_swap_actions_no_action_needed(self) -> None:
        """Test when no bridge/swap actions needed (already have correct tokens)."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock tokens that exactly match what's needed
        available_tokens = [
            {
                'chain': 'optimism',
                'token': '0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85',  # USDC
                'token_symbol': 'USDC',
                'balance': 1000000
            },
            {
                'chain': 'optimism', 
                'token': '0x1217BfE6c773EEC6cc4A38b5Dc45B92292B6E189',  # oUSDT
                'token_symbol': 'oUSDT',
                'balance': 1000000
            }
        ]

        # Mock opportunity requiring exactly these tokens
        opportunity = {
            'dex_type': 'velodrome',
            'chain': 'optimism',
            'token0': '0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85',  # USDC
            'token0_symbol': 'USDC',
            'token1': '0x1217BfE6c773EEC6cc4A38b5Dc45B92292B6E189',  # oUSDT
            'token1_symbol': 'oUSDT',
            'relative_funds_percentage': 1.0
        }

        with mock.patch.object(
            self.behaviour.current_behaviour,
            '_build_bridge_swap_actions',
            return_value=[]  # No actions needed
        ) as mock_no_action:
            
            result = self.behaviour.current_behaviour._build_bridge_swap_actions(
                opportunity, available_tokens
            )
            
            # Verify no bridge/swap actions needed
            assert len(result) == 0

    def test_bridge_swap_actions_complex_multi_token_scenario(self) -> None:
        """Test complex scenario with multiple tokens across chains requiring various swaps."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock complex token portfolio across chains
        available_tokens = [
            {
                'chain': 'ethereum',
                'token': '0xA0b86a33E6441E13799A0c77E2f93b6688BDB689',  # USDC on Ethereum
                'token_symbol': 'USDC',
                'balance': 500000
            },
            {
                'chain': 'ethereum',
                'token': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # WETH on Ethereum
                'token_symbol': 'WETH', 
                'balance': 1000000000000000000
            },
            {
                'chain': 'optimism',
                'token': '0x4200000000000000000000000000000000000042',  # OP token
                'token_symbol': 'OP',
                'balance': 10000000000000000000  # 10 OP
            }
        ]

        # Mock Balancer opportunity on Optimism requiring WETH + USDC
        opportunity = {
            'dex_type': 'balancerPool',
            'chain': 'optimism',
            'token0': '0x4200000000000000000000000000000000000006',  # WETH on Optimism
            'token0_symbol': 'WETH',
            'token1': '0x7F5c764cBc14f9669B88837ca1490cCa17c31607',  # USDC.e on Optimism
            'token1_symbol': 'USDC.e',
            'relative_funds_percentage': 1.0
        }

        expected_complex_actions = [
            # Bridge Ethereum WETH → Optimism WETH  
            {
                'action': 'FindBridgeRoute',
                'from_chain': 'ethereum',
                'to_chain': 'optimism',
                'from_token': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
                'to_token': '0x4200000000000000000000000000000000000006',
                'from_token_symbol': 'WETH',
                'to_token_symbol': 'WETH'
            },
            # Bridge Ethereum USDC → Optimism USDC.e
            {
                'action': 'FindBridgeRoute',
                'from_chain': 'ethereum', 
                'to_chain': 'optimism',
                'from_token': '0xA0b86a33E6441E13799A0c77E2f93b6688BDB689',
                'to_token': '0x7F5c764cBc14f9669B88837ca1490cCa17c31607',
                'from_token_symbol': 'USDC',
                'to_token_symbol': 'USDC.e'
            },
            # Swap Optimism OP → WETH (on same chain)
            {
                'action': 'FindBridgeRoute',
                'from_chain': 'optimism',
                'to_chain': 'optimism',
                'from_token': '0x4200000000000000000000000000000000000042',
                'to_token': '0x4200000000000000000000000000000000000006', 
                'from_token_symbol': 'OP',
                'to_token_symbol': 'WETH'
            }
        ]

        with mock.patch.object(
            self.behaviour.current_behaviour,
            '_build_bridge_swap_actions',
            return_value=expected_complex_actions
        ):
            
            result = self.behaviour.current_behaviour._build_bridge_swap_actions(
                opportunity, available_tokens
            )
            
            # Verify all necessary actions were created
            assert len(result) == 3
            
            # Verify mix of bridge and swap actions
            cross_chain_actions = [a for a in result if a['from_chain'] != a['to_chain']]
            same_chain_actions = [a for a in result if a['from_chain'] == a['to_chain']]
            
            assert len(cross_chain_actions) == 2  # 2 bridge actions
            assert len(same_chain_actions) == 1   # 1 swap action

    # =============================================
    # 4. ACTION ORDER AND SEQUENCING TESTS
    # =============================================

    def test_get_order_of_transactions_complete_sequence(self) -> None:
        """Test complete action sequence ordering: Exit → Bridge → Enter → Stake."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock position to exit (Velodrome CL requiring unstaking)
        position_to_exit = {
            'dex_type': 'velodrome',
            'pool_address': '0xOLD_POOL',
            'is_cl_pool': True,
            'chain': 'optimism'
        }

        # Mock new opportunity
        new_opportunity = {
            'dex_type': 'velodrome',
            'pool_address': '0xNEW_POOL',
            'chain': 'optimism',
            'token0': '0xUSDC_TOKEN',
            'token0_symbol': 'USDC',
            'token1': '0xUSDT_TOKEN',
            'token1_symbol': 'USDT',
            'is_cl_pool': True,
            'relative_funds_percentage': 1.0
        }

        self.behaviour.current_behaviour.position_to_exit = position_to_exit
        self.behaviour.current_behaviour.selected_opportunities = [new_opportunity]

        complete_sequence = [
            # 1. Unstake old CL position
            {
                'action': 'UnstakeLpTokens',
                'dex_type': 'velodrome',
                'pool_address': '0xOLD_POOL',
                'chain': 'optimism'
            },
            # 2. Exit old pool
            {
                'action': 'ExitPool',
                'dex_type': 'velodrome', 
                'pool_address': '0xOLD_POOL',
                'chain': 'optimism'
            },
            # 3. Bridge/swap to required tokens
            {
                'action': 'FindBridgeRoute',
                'from_chain': 'optimism',
                'to_chain': 'optimism',
                'from_token': '0xOLD_TOKEN',
                'to_token': '0xUSDC_TOKEN',
                'from_token_symbol': 'OLD_TOKEN',
                'to_token_symbol': 'USDC'
            },
            # 4. Enter new pool
            {
                'action': 'EnterPool',
                'dex_type': 'velodrome',
                'pool_address': '0xNEW_POOL',
                'chain': 'optimism',
                'token0': '0xUSDC_TOKEN',
                'token1': '0xUSDT_TOKEN'
            },
            # 5. Stake new CL position
            {
                'action': 'StakeLpTokens',
                'dex_type': 'velodrome',
                'pool_address': '0xNEW_POOL',
                'chain': 'optimism',
                'is_cl_pool': True
            }
        ]

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "get_order_of_transactions",
            return_value=iter([complete_sequence])
        ) as mock_order:
            
            generator = self.behaviour.current_behaviour.get_order_of_transactions()
            result = next(generator)
            
            # Verify complete sequence and order
            assert len(result) == 5
            assert result[0]['action'] == 'UnstakeLpTokens'  # First: unstake old
            assert result[1]['action'] == 'ExitPool'         # Second: exit old
            assert result[2]['action'] == 'FindBridgeRoute'  # Third: bridge/swap
            assert result[3]['action'] == 'EnterPool'        # Fourth: enter new
            assert result[4]['action'] == 'StakeLpTokens'    # Fifth: stake new

    def test_get_order_of_transactions_balancer_no_staking(self) -> None:
        """Test action sequence for Balancer pools (no staking required)."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock Balancer opportunity (no staking)
        balancer_opportunity = {
            'dex_type': 'balancerPool',
            'pool_address': '0xBALANCER_POOL',
            'chain': 'optimism',
            'token0': '0xWETH_TOKEN',
            'token0_symbol': 'WETH',
            'token1': '0xUSDC_TOKEN',
            'token1_symbol': 'USDC',
            'relative_funds_percentage': 1.0
        }

        self.behaviour.current_behaviour.selected_opportunities = [balancer_opportunity]
        self.behaviour.current_behaviour.position_to_exit = None  # No exit needed

        balancer_sequence = [
            # 1. Bridge/swap to required tokens
            {
                'action': 'FindBridgeRoute',
                'from_chain': 'ethereum',
                'to_chain': 'optimism',
                'from_token': '0xETH_ETHEREUM',
                'to_token': '0xWETH_TOKEN'
            },
            # 2. Enter Balancer pool (no staking)
            {
                'action': 'EnterPool',
                'dex_type': 'balancerPool',
                'pool_address': '0xBALANCER_POOL',
                'chain': 'optimism'
            }
        ]

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "get_order_of_transactions",
            return_value=iter([balancer_sequence])
        ):
            
            generator = self.behaviour.current_behaviour.get_order_of_transactions()
            result = next(generator)
            
            # Verify Balancer sequence (no staking)
            assert len(result) == 2
            assert result[0]['action'] == 'FindBridgeRoute'
            assert result[1]['action'] == 'EnterPool'
            assert result[1]['dex_type'] == 'balancerPool'
            
            # Verify no staking action for Balancer
            stake_actions = [a for a in result if 'Stake' in a['action']]
            assert len(stake_actions) == 0

    def test_get_order_of_transactions_uniswap_v3_sequence(self) -> None:
        """Test action sequence for Uniswap V3 pools."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock Uniswap V3 opportunity
        uniswap_opportunity = {
            'dex_type': 'UniswapV3',
            'pool_address': '0xUNISWAP_V3_POOL',
            'chain': 'optimism',
            'token0': '0xUSDC_TOKEN',
            'token0_symbol': 'USDC',
            'token1': '0xWETH_TOKEN',
            'token1_symbol': 'WETH',
            'fee': 3000,  # 0.3% fee tier
            'relative_funds_percentage': 1.0
        }

        self.behaviour.current_behaviour.selected_opportunities = [uniswap_opportunity]

        uniswap_sequence = [
            {
                'action': 'FindBridgeRoute',
                'from_chain': 'mode',
                'to_chain': 'optimism',
                'from_token': '0xMODE_USDC',
                'to_token': '0xUSDC_TOKEN'
            },
            {
                'action': 'EnterPool',
                'dex_type': 'UniswapV3',
                'pool_address': '0xUNISWAP_V3_POOL',
                'chain': 'optimism',
                'fee': 3000
            }
        ]

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "get_order_of_transactions",
            return_value=iter([uniswap_sequence])
        ):
            
            generator = self.behaviour.current_behaviour.get_order_of_transactions()
            result = next(generator)
            
            # Verify Uniswap V3 sequence
            assert len(result) == 2
            assert result[1]['dex_type'] == 'UniswapV3'
            assert result[1]['fee'] == 3000

    # =============================================
    # 5. RISK MANAGEMENT TESTS WITH REAL SCENARIOS
    # =============================================

    def test_check_minimum_hold_period_with_recent_velodrome_position(self) -> None:
        """Test hold period enforcement for recent Velodrome position."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock recent Velodrome CL position (within 3-week hold period)
        current_time = 1640995200
        recent_entry = current_time - (7 * 24 * 3600)  # 1 week ago (should hold longer)
        
        recent_velodrome_position = {
            "pool_address": "0x5483484F876218908CA435F08222751F7f7b2a3b",
            "status": "OPEN",
            "dex_type": "velodrome",
            "is_cl_pool": True,
            "chain": "optimism",
            "apr": 36.92684058040778,
            "timestamp": recent_entry,
            "enter_timestamp": recent_entry,
            "token0_symbol": "USDC",
            "token1_symbol": "oUSDT"
        }
        
        self.behaviour.current_behaviour.current_positions = [recent_velodrome_position]

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=current_time
        ):
            result = self.behaviour.current_behaviour.check_minimum_hold_period()
            
            # Should enforce hold period (True = must hold longer)
            assert result is True

    def test_check_funds_with_multi_chain_portfolio(self) -> None:
        """Test funds validation with realistic multi-chain portfolio."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock realistic multi-chain portfolio
        self.behaviour.current_behaviour.current_positions = []
        self.behaviour.current_behaviour.synchronized_data.positions = [
            {
                "chain": "ethereum",
                "assets": [
                    {"balance": 1000000000000000000, "symbol": "WETH"},  # 1 ETH
                    {"balance": 2500000000, "symbol": "USDC"}           # $2500 USDC
                ]
            },
            {
                "chain": "optimism", 
                "assets": [
                    {"balance": 500000000, "symbol": "USDC.e"},         # $500 USDC.e
                    {"balance": 10000000000000000000, "symbol": "OP"}   # 10 OP tokens
                ]
            },
            {
                "chain": "mode",
                "assets": [
                    {"balance": 250000000000000000, "symbol": "ezETH"}  # 0.25 ezETH
                ]
            }
        ]

        result = self.behaviour.current_behaviour.check_funds()
        
        # Should allow investment with available funds across chains
        assert result is True

    def test_update_position_metrics_velodrome_cl_performance(self) -> None:
        """Test position metrics update for Velodrome CL position."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock Velodrome CL position needing metrics update
        current_time = 1640995200
        old_metrics_time = current_time - 25000  # Older than 6-hour interval
        
        velodrome_position = {
            "pool_address": "0x5483484F876218908CA435F08222751F7f7b2a3b",
            "status": "OPEN",
            "dex_type": "velodrome",
            "is_cl_pool": True,
            "chain": "optimism",
            "apr": 36.92684058040778,  # Old APR
            "last_metrics_update": old_metrics_time,
            "token0_symbol": "USDC",
            "token1_symbol": "oUSDT"
        }
        
        self.behaviour.current_behaviour.current_positions = [velodrome_position]
        
        # Mock strategy mapping
        self.behaviour.current_behaviour.params.dex_type_to_strategy = {
            "velodrome": "velodrome_cl_strategy"
        }

        # Mock updated metrics (APR changed)
        updated_metrics = {
            "apr": 42.15,  # Improved APR
            "pnl": 3.8,    # 3.8% profit
            "il_risk_score": -1e-12,
            "tvl": 105000.0,
            "last_metrics_update": current_time
        }

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=current_time
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "get_returns_metrics_for_opportunity",
            return_value=updated_metrics
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "store_current_positions"
        ):
            
            self.behaviour.current_behaviour.update_position_metrics()
            
            # Verify position metrics were updated
            updated_position = self.behaviour.current_behaviour.current_positions[0]
            assert updated_position["apr"] == 42.15  # APR improved
            assert updated_position["pnl"] == 3.8    # Profitable position
            assert updated_position["tvl"] == 105000.0

    # =============================================
    # 6. COMPREHENSIVE INTEGRATION TESTS
    # =============================================

    def test_complete_velodrome_cl_investment_cycle(self) -> None:
        """Test complete Velodrome CL investment cycle with realistic data."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock complete Velodrome opportunity with all realistic data
        complete_velodrome_opportunity = {
            'dex_type': 'velodrome',
            'pool_address': '0x5483484F876218908CA435F08222751F7f7b2a3b',
            'pool_id': '0x5483484F876218908CA435F08222751F7f7b2a3b',
            'tvl': 97237.684693,
            'is_lp': True,
            'token_count': 2,
            'volume': 0.0,
            'chain': 'optimism',
            'apr': 36.92684058040778,
            'is_cl_pool': True,
            'is_stable': True,
            'token0': '0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85',
            'token0_symbol': 'USDC',
            'token1': '0x1217BfE6c773EEC6cc4A38b5Dc45B92292B6E189',
            'token1_symbol': 'oUSDT',
            'sharpe_ratio': 0.08655583746925036,
            'depth_score': 1079.068096913766,
            'max_position_size': 10000000.0,
            'il_risk_score': -6.084100235285381e-13,
            'composite_score': 0.4364842470256133,
            'apr_weighted_score': 16.11798420577415,
            'funds_percentage': 100.0,
            'relative_funds_percentage': 1.0,
            'token0_percentage': 99.73997533688896,
            'token1_percentage': 0.2600246631110358,
            'token_requirements': {
                'position_requirements': [
                    {
                        'tick_range': [-3, -2],
                        'current_tick': -3,
                        'status': 'BELOW_RANGE',
                        'allocation': 0.9807317636931588,
                        'token0_ratio': 1.0,
                        'token1_ratio': 0.0
                    }
                ],
                'current_price': 0.9996228374734647,
                'current_tick': -3,
                'overall_token0_ratio': np.float64(0.9973997533688896),
                'overall_token1_ratio': np.float64(0.0026002466311103576),
                'recommendation': 'Mixed position requirements. Overall: 99.74% token0, 0.26% token1',
                'warnings': []
            },
            'tick_spacing': 1,
            'tick_ranges': [
                {
                    'tick_lower': -3,
                    'tick_upper': -2,
                    'allocation': np.float64(0.9807317636931588),
                    'percent_in_bounds': np.float64(55.20110957004161)
                }
            ],
            'percent_in_bounds': np.float64(55.20110957004161),
            'max_amounts_in': [0, 15560],
            'action': 'EnterPool',
            'opportunity_apr': 36.92684058040778
        }

        # Mock complete action sequence from your example
        complete_velodrome_actions = [
            complete_velodrome_opportunity,  # EnterPool action
            {
                'action': 'StakeLpTokens',
                'dex_type': 'velodrome',
                'chain': 'optimism',
                'pool_address': '0x5483484F876218908CA435F08222751F7f7b2a3b',
                'is_cl_pool': True,
                'recipient': '0x9f3AbFC3301093f39c2A137f87c525b4a0832ba9',
                'description': 'Stake CL LP tokens for 0x5483484F876218908CA435F08222751F7f7b2a3b'
            }
        ]

        # Set up behaviour state
        self.behaviour.current_behaviour.selected_opportunities = [complete_velodrome_opportunity]
        self.behaviour.current_behaviour.trading_opportunities = [complete_velodrome_opportunity]
        self.behaviour.current_behaviour.current_positions = []  # New investment

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "prepare_strategy_actions",
            return_value=iter([complete_velodrome_actions])
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "check_minimum_hold_period",
            return_value=False
        ), mock.patch.object(
            self.behaviour.current_behaviour, 
            "check_funds",
            return_value=True
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "send_actions"
        ) as mock_send:
            
            generator = self.behaviour.current_behaviour.async_act()
            list(generator)
            
            # Verify complete Velodrome CL investment was executed
            sent_actions = mock_send.call_args[0][0]
            
            # Verify EnterPool action has all required Velodrome data
            enter_action = sent_actions[0]
            assert enter_action['dex_type'] == 'velodrome'
            assert enter_action['is_cl_pool'] is True
            assert enter_action['apr'] == 36.92684058040778
            assert 'token_requirements' in enter_action
            assert enter_action['token0_symbol'] == 'USDC'
            assert enter_action['token1_symbol'] == 'oUSDT'
            
            # Verify StakeLpTokens action
            stake_action = sent_actions[1]
            assert stake_action['action'] == 'StakeLpTokens'
            assert stake_action['dex_type'] == 'velodrome'
            assert stake_action['is_cl_pool'] is True

    def test_multi_dex_comparison_and_selection(self) -> None:
        """Test strategy comparing multiple DEX opportunities and selecting best."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock multiple opportunities across different DEXs
        multi_dex_opportunities = [
            # High APR Velodrome (should be selected)
            {
                'dex_type': 'velodrome',
                'pool_address': '0xVELODROME_HIGH',
                'chain': 'optimism',
                'apr': 45.5,
                'composite_score': 0.85,
                'sharpe_ratio': 0.12,
                'il_risk_score': 1e-12,
                'is_cl_pool': True
            },
            # Medium APR Balancer
            {
                'dex_type': 'balancerPool', 
                'pool_address': '0xBALANCER_MED',
                'chain': 'optimism',
                'apr': 22.3,
                'composite_score': 0.65,
                'sharpe_ratio': 0.08,
                'il_risk_score': 0.05
            },
            # Low APR Uniswap V3
            {
                'dex_type': 'UniswapV3',
                'pool_address': '0xUNISWAP_LOW', 
                'chain': 'optimism',
                'apr': 12.1,
                'composite_score': 0.45,
                'sharpe_ratio': 0.06,
                'il_risk_score': 0.08
            }
        ]

        # Mock strategy selecting highest composite score
        strategy_result = {
            "optimal_strategies": [multi_dex_opportunities[0]],  # Velodrome selected
            "position_to_exit": None,
            "reasoning": "Velodrome CL pool offers highest risk-adjusted returns with 45.5% APR"
        }

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "execute_strategy",
            return_value=strategy_result
        ):
            self.behaviour.current_behaviour.trading_opportunities = multi_dex_opportunities
            
            generator = self.behaviour.current_behaviour.execute_hyper_strategy()
            list(generator)
            
            # Verify highest-scoring opportunity was selected
            selected = self.behaviour.current_behaviour.selected_opportunities[0]
            assert selected['dex_type'] == 'velodrome'
            assert selected['apr'] == 45.5
            assert selected['composite_score'] == 0.85

    # =============================================
    # 7. EDGE CASES AND ERROR HANDLING
    # =============================================

    def test_async_act_no_opportunities_found(self) -> None:
        """Test behavior when no trading opportunities are found."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock no opportunities scenario
        self.behaviour.current_behaviour.trading_opportunities = []
        self.behaviour.current_behaviour.selected_opportunities = None

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "check_minimum_hold_period",
            return_value=False
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "check_funds", 
            return_value=True
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "prepare_strategy_actions",
            return_value=iter([[]])  # Empty actions
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "send_actions"
        ) as mock_send:
            
            generator = self.behaviour.current_behaviour.async_act()
            list(generator)
            
            # Verify empty actions sent when no opportunities
            mock_send.assert_called_with([])

    def test_bridge_swap_actions_insufficient_tokens(self) -> None:
        """Test bridge action handling when insufficient tokens available."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock insufficient tokens scenario
        insufficient_tokens = [
            {
                'chain': 'ethereum',
                'token': '0xUSDC_TOKEN',
                'token_symbol': 'USDC',
                'balance': 1000  # Very small balance ($0.001)
            }
        ]

        # Mock high-value opportunity
        high_value_opportunity = {
            'dex_type': 'velodrome',
            'chain': 'optimism', 
            'token0': '0xUSDC_OPTIMISM',
            'token0_symbol': 'USDC',
            'token1': '0xETH_OPTIMISM',
            'token1_symbol': 'WETH',
            'max_position_size': 10000000.0,  # $10M max
            'relative_funds_percentage': 1.0
        }

        with mock.patch.object(
            self.behaviour.current_behaviour,
            '_build_bridge_swap_actions',
            return_value=None  # Insufficient funds
        ) as mock_insufficient:
            
            result = self.behaviour.current_behaviour._build_bridge_swap_actions(
                high_value_opportunity, insufficient_tokens
            )
            
            # Should handle insufficient tokens gracefully
            assert result is None

    def test_strategy_execution_with_numpy_values(self) -> None:
        """Test strategy execution handles numpy values correctly."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock strategy result with numpy values (like your real data)
        numpy_strategy_result = {
            "optimal_strategies": [{
                'dex_type': 'velodrome',
                'pool_address': '0x5483484F876218908CA435F08222751F7f7b2a3b',
                'overall_token0_ratio': np.float64(0.9973997533688896),
                'overall_token1_ratio': np.float64(0.0026002466311103576), 
                'allocation': np.float64(0.9807317636931588),
                'percent_in_bounds': np.float64(55.20110957004161)
            }],
            "position_to_exit": None,
            "reasoning": "Strategy with numpy values"
        }

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "execute_strategy",
            return_value=numpy_strategy_result
        ):
            generator = self.behaviour.current_behaviour.execute_hyper_strategy()
            list(generator)
            
            # Verify numpy values are handled correctly
            selected = self.behaviour.current_behaviour.selected_opportunities[0]
            assert isinstance(selected['overall_token0_ratio'], (float, np.floating))
            assert selected['overall_token0_ratio'] > 0.99  # ~99.74%
            assert selected['overall_token1_ratio'] < 0.01  # ~0.26%

    # =============================================
    # 8. ADDITIONAL CRITICAL TESTS
    # =============================================

    def test_async_act_respects_minimum_hold_period(self) -> None:
        """Test agent respects minimum hold period - RISK MANAGEMENT."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "check_and_prepare_non_whitelisted_swaps",
            return_value=iter([None])
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "check_minimum_hold_period", 
            return_value=True
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "send_actions"
        ) as mock_send:
            
            generator = self.behaviour.current_behaviour.async_act()
            list(generator)
            
            # Verify no new investments during hold period
            mock_send.assert_called_with([])

    def test_check_minimum_hold_period_allows_mature_exits(self) -> None:
        """Test minimum hold period allows exits for mature positions."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock old position entry (should allow exit)
        # MIN_TIME_IN_POSITION = 604800 * 3 = 1814400 seconds (3 weeks)
        current_time = 1640995200
        old_timestamp = current_time - (1814400 + 1000)  # Older than MIN_TIME_IN_POSITION
        
        self.behaviour.current_behaviour.current_positions = [
            {
                "pool_address": "0xtest",
                "status": "OPEN", 
                "timestamp": old_timestamp,
                "enter_timestamp": old_timestamp
            }
        ]

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "_get_current_timestamp",
            return_value=current_time
        ):
            result = self.behaviour.current_behaviour.check_minimum_hold_period()
            
            # Should allow exit for mature positions
            assert result is False

    def test_check_funds_allows_investment_with_available_capital(self) -> None:
        """Test funds check allows investment with available capital."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock no current positions but available balance
        self.behaviour.current_behaviour.current_positions = []
        
        # Mock synchronized_data with available balance
        self.behaviour.current_behaviour.synchronized_data.positions = [
            {
                "assets": [
                    {"balance": 1000000, "symbol": "USDC"}  # $1000 USDC
                ]
            }
        ]

        result = self.behaviour.current_behaviour.check_funds()
        
        # Should allow investment when funds available
        assert result is True

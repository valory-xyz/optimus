# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   You may not use this file except in compliance with the License.
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

"""End-to-end integration tests for Velodrome protocol workflows."""

import pytest
from unittest.mock import MagicMock, patch

from packages.valory.skills.liquidity_trader_abci.behaviours.base import DexType
from packages.valory.skills.liquidity_trader_abci.pools.velodrome import VelodromePoolBehaviour

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions, TestDataGenerator
from tests.integration.fixtures.pool_data_fixtures import (
    velodrome_stable_pool_data,
    velodrome_volatile_pool_data,
    velodrome_cl_pool_data,
    user_assets_2tokens,
)


class TestVelodromeE2EWorkflows(ProtocolIntegrationTestBase):
    """Test complete Velodrome protocol workflows."""

    def test_complete_velodrome_liquidity_provision_workflow(self, velodrome_volatile_pool_data, user_assets_2tokens):
        """Test complete workflow from pool selection to liquidity provision and staking."""
        # Setup test data
        pool_data = velodrome_volatile_pool_data
        user_assets = user_assets_2tokens
        
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"tokens": [pool_data["token0"], pool_data["token1"]]},  # pool tokens
            {"reserve0": pool_data["reserve0"], "reserve1": pool_data["reserve1"]},  # pool reserves
            {"tx_hash": "0xadd_liquidity_tx_hash"},  # add liquidity transaction
            {"balance": 0},  # initial gauge balance
            {"tx_hash": "0xdeposit_tx_hash"},  # deposit to gauge transaction
        ]
        
        behaviour.contract_interact = MagicMock(side_effect=mock_contract_responses)
        
        with patch.multiple(
            behaviour,
            get_http_response=self._mock_http_responses(),
            _read_kv=self._mock_kv_read(),
            _write_kv=self._mock_kv_write()
        ):
            # Step 1: Pool selection and validation
            selected_pool = behaviour._select_optimal_pool([pool_data])
            assert selected_pool == pool_data
            
            # Step 2: Calculate optimal amounts
            optimal_amounts = behaviour._calculate_optimal_amounts(
                selected_pool, user_assets
            )
            assert len(optimal_amounts) == 2
            assert optimal_amounts[pool_data["token0"]] > 0
            assert optimal_amounts[pool_data["token1"]] > 0
            
            # Step 3: Generate add liquidity transaction
            add_liquidity_tx = list(behaviour.enter(
                pool_address=selected_pool["pool_address"],
                safe_address=self.test_addresses["safe"],
                assets=[selected_pool["token0"], selected_pool["token1"]],
                chain="optimism",
                max_amounts_in=[optimal_amounts[selected_pool["token0"]], optimal_amounts[selected_pool["token1"]]],
                pool_fee=selected_pool.get("fee", 0),
                is_stable=selected_pool["is_stable"]
            ))
            
            # Verify transaction was generated
            assert len(add_liquidity_tx) > 0
            assert add_liquidity_tx[0] is not None
            
            # Step 4: Stake LP tokens in gauge
            if selected_pool.get("gauge_address"):
                stake_tx = list(behaviour._stake_in_gauge(
                    selected_pool["gauge_address"],
                    optimal_amounts.get("lp_tokens", 100000000000000000000)  # Mock LP tokens received
                ))
                
                # Verify staking transaction was generated
                assert len(stake_tx) > 0
                assert stake_tx[0] is not None
            
            # Step 5: Track new position
            new_position = {
                "pool_address": selected_pool["pool_address"],
                "lp_balance": 100000000000000000000,  # Mock LP balance
                "staked_amount": 100000000000000000000,  # Mock staked amount
                "gauge_address": selected_pool.get("gauge_address"),
                "dex_type": DexType.VELODROME.value
            }
            
            # Step 6: Test position value calculation
            position_value = list(behaviour.get_user_share_value_velodrome(
                selected_pool["pool_address"],
                self.test_addresses["safe"],
                "optimism",
                new_position
            ))
            
            assert len(position_value) > 0
            assert "amount0" in position_value[0]
            assert "amount1" in position_value[0]

    def test_complete_velodrome_reward_claiming_workflow(self, velodrome_volatile_pool_data):
        """Test complete workflow for reward claiming and compounding."""
        # Setup existing position with accumulated rewards
        existing_position = {
            "pool_address": velodrome_volatile_pool_data["pool_address"],
            "lp_balance": 100000000000000000000,  # 100 LP tokens
            "staked_amount": 100000000000000000000,  # 100 staked LP tokens
            "gauge_address": velodrome_volatile_pool_data["gauge_address"],
            "dex_type": DexType.VELODROME.value
        }
        
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"earned": 50000000000000000000},  # earned rewards
            {"tx_hash": "0xget_reward_tx_hash"},  # get reward transaction
        ]
        
        behaviour.contract_interact = MagicMock(side_effect=mock_contract_responses)
        
        with patch.multiple(
            behaviour,
            get_http_response=self._mock_http_responses(),
            _read_kv=self._mock_kv_read(),
            _write_kv=self._mock_kv_write()
        ):
            # Step 1: Check for claimable rewards
            claimable_rewards = behaviour._check_claimable_rewards(existing_position)
            assert claimable_rewards["earned"] > 0
            
            # Step 2: Generate get reward transaction
            claim_tx = list(behaviour._claim_rewards(
                existing_position["gauge_address"]
            ))
            
            # Verify transaction was generated
            assert len(claim_tx) > 0
            assert claim_tx[0] is not None
            
            # Step 3: Test reward compounding (if enabled)
            if behaviour._should_compound_rewards(claimable_rewards):
                compound_tx = list(behaviour._compound_rewards(
                    existing_position, claimable_rewards
                ))
                assert len(compound_tx) > 0

    def test_complete_velodrome_position_management_workflow(self, velodrome_volatile_pool_data):
        """Test complete position management workflow."""
        # Setup existing position
        existing_position = {
            "pool_address": velodrome_volatile_pool_data["pool_address"],
            "lp_balance": 100000000000000000000,  # 100 LP tokens
            "staked_amount": 100000000000000000000,  # 100 staked LP tokens
            "gauge_address": velodrome_volatile_pool_data["gauge_address"],
            "dex_type": DexType.VELODROME.value
        }
        
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"earned": 0},  # no pending rewards
            {"tx_hash": "0xwithdraw_tx_hash"},  # withdraw from gauge
            {"tx_hash": "0xremove_liquidity_tx_hash"},  # remove liquidity
        ]
        
        behaviour.contract_interact = MagicMock(side_effect=mock_contract_responses)
        
        with patch.multiple(
            behaviour,
            get_http_response=self._mock_http_responses(),
            _read_kv=self._mock_kv_read(),
            _write_kv=self._mock_kv_write()
        ):
            # Step 1: Withdraw from gauge
            withdraw_tx = list(behaviour._withdraw_from_gauge(
                existing_position["gauge_address"],
                existing_position["staked_amount"]
            ))
            
            # Verify transaction was generated
            assert len(withdraw_tx) > 0
            assert withdraw_tx[0] is not None
            
            # Step 2: Remove liquidity
            remove_tx = list(behaviour.exit(
                pool_address=existing_position["pool_address"],
                safe_address=self.test_addresses["safe"],
                assets=[velodrome_volatile_pool_data["token0"], velodrome_volatile_pool_data["token1"]],
                chain="optimism",
                lp_amount=existing_position["lp_balance"]
            ))
            
            # Verify transaction was generated
            assert len(remove_tx) > 0
            assert remove_tx[0] is not None

    def test_complete_velodrome_rebalancing_workflow(self, velodrome_volatile_pool_data, user_assets_2tokens):
        """Test complete position rebalancing workflow."""
        # Setup existing position
        existing_position = {
            "pool_address": velodrome_volatile_pool_data["pool_address"],
            "lp_balance": 100000000000000000000,  # 100 LP tokens
            "staked_amount": 100000000000000000000,  # 100 staked LP tokens
            "gauge_address": velodrome_volatile_pool_data["gauge_address"],
            "dex_type": DexType.VELODROME.value
        }
        
        # New pool with better APR
        new_pool_data = velodrome_volatile_pool_data.copy()
        new_pool_data["pool_address"] = "0xNewPoolAddress"
        new_pool_data["gauge_address"] = "0xNewGaugeAddress"
        new_pool_data["apr"] = 25.0  # Higher APR
        
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Mock contract interactions for withdrawal and entry
        mock_contract_responses = [
            # Withdrawal from old position
            {"earned": 0},  # no pending rewards
            {"tx_hash": "0xwithdraw_tx_hash"},  # withdraw from gauge
            {"tx_hash": "0xremove_liquidity_tx_hash"},  # remove liquidity
            # Entry to new pool
            {"tokens": [new_pool_data["token0"], new_pool_data["token1"]]},  # pool tokens
            {"reserve0": new_pool_data["reserve0"], "reserve1": new_pool_data["reserve1"]},  # pool reserves
            {"tx_hash": "0xadd_liquidity_tx_hash"},  # add liquidity
            {"balance": 0},  # initial gauge balance
            {"tx_hash": "0xdeposit_tx_hash"},  # deposit to gauge
        ]
        
        behaviour.contract_interact = MagicMock(side_effect=mock_contract_responses)
        
        with patch.multiple(
            behaviour,
            get_http_response=self._mock_http_responses(),
            _read_kv=self._mock_kv_read(),
            _write_kv=self._mock_kv_write()
        ):
            # Step 1: Identify rebalancing opportunity
            rebalancing_opportunity = behaviour._identify_rebalancing_opportunity(
                [existing_position], [new_pool_data]
            )
            assert rebalancing_opportunity is not None
            assert rebalancing_opportunity["old_pool"] == existing_position
            assert rebalancing_opportunity["new_pool"] == new_pool_data
            
            # Step 2: Execute rebalancing
            rebalancing_result = behaviour._execute_rebalancing(rebalancing_opportunity)
            
            # Verify rebalancing was executed
            assert rebalancing_result["success"] is True
            assert "withdrawal_tx" in rebalancing_result
            assert "entry_tx" in rebalancing_result
            
            # Step 3: Update position tracking
            updated_positions = behaviour._update_positions_after_rebalancing(
                rebalancing_result
            )
            
            # Verify positions were updated
            assert len(updated_positions) == 1
            assert updated_positions[0]["pool_address"] == new_pool_data["pool_address"]

    def test_complete_velodrome_multi_pool_diversification_workflow(self, user_assets_2tokens):
        """Test multi-pool diversification workflow."""
        # Setup multiple pools
        pool1_data = TestDataGenerator.generate_velodrome_pool_data(
            is_stable=True,
            pool_address="0xStablePoolAddress",
            gauge_address="0xStableGaugeAddress"
        )
        pool1_data["apr"] = 8.0
        
        pool2_data = TestDataGenerator.generate_velodrome_pool_data(
            is_stable=False,
            pool_address="0xVolatilePoolAddress",
            gauge_address="0xVolatileGaugeAddress"
        )
        pool2_data["apr"] = 15.0
        
        pool3_data = TestDataGenerator.generate_velodrome_pool_data(
            is_cl_pool=True,
            pool_address="0xCLPoolAddress",
            gauge_address="0xCLGaugeAddress"
        )
        pool3_data["apr"] = 20.0
        
        available_pools = [pool1_data, pool2_data, pool3_data]
        
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Mock contract interactions for multiple pools
        mock_contract_responses = []
        for pool in available_pools:
            mock_contract_responses.extend([
                {"tokens": [pool["token0"], pool["token1"]]},  # pool tokens
                {"reserve0": pool["reserve0"], "reserve1": pool["reserve1"]},  # pool reserves
                {"tx_hash": f"add_liquidity_{pool['pool_address'][:8]}_tx_hash"},  # add liquidity
                {"balance": 0},  # initial gauge balance
                {"tx_hash": f"deposit_{pool['gauge_address'][:8]}_tx_hash"},  # deposit to gauge
            ])
        
        behaviour.contract_interact = MagicMock(side_effect=mock_contract_responses)
        
        with patch.multiple(
            behaviour,
            get_http_response=self._mock_http_responses(),
            _read_kv=self._mock_kv_read(),
            _write_kv=self._mock_kv_write()
        ):
            # Step 1: Select optimal pool allocation
            allocation = behaviour._calculate_optimal_allocation(
                available_pools, user_assets_2tokens
            )
            
            # Verify allocation
            assert len(allocation) > 0
            total_allocation = sum(allocation.values())
            assert total_allocation <= 1.0  # Should not exceed 100%
            
            # Step 2: Execute diversification
            diversification_result = behaviour._execute_diversification(
                allocation, available_pools, user_assets_2tokens
            )
            
            # Verify diversification was executed
            assert diversification_result["success"] is True
            assert len(diversification_result["transactions"]) > 0
            assert len(diversification_result["positions"]) > 0
            
            # Step 3: Verify position tracking
            for position in diversification_result["positions"]:
                TestAssertions.assert_position_data_structure(position, DexType.VELODROME.value)

    def test_complete_velodrome_emergency_withdrawal_workflow(self, velodrome_volatile_pool_data):
        """Test emergency withdrawal workflow."""
        # Setup position with high impermanent loss
        existing_position = {
            "pool_address": velodrome_volatile_pool_data["pool_address"],
            "lp_balance": 100000000000000000000,  # 100 LP tokens
            "staked_amount": 100000000000000000000,  # 100 staked LP tokens
            "gauge_address": velodrome_volatile_pool_data["gauge_address"],
            "impermanent_loss": 15.0,  # 15% IL
            "dex_type": DexType.VELODROME.value
        }
        
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"earned": 0},  # no pending rewards
            {"tx_hash": "0xwithdraw_tx_hash"},  # withdraw from gauge
            {"tx_hash": "0xremove_liquidity_tx_hash"},  # remove liquidity
        ]
        
        behaviour.contract_interact = MagicMock(side_effect=mock_contract_responses)
        
        with patch.multiple(
            behaviour,
            get_http_response=self._mock_http_responses(),
            _read_kv=self._mock_kv_read(),
            _write_kv=self._mock_kv_write()
        ):
            # Step 1: Check for emergency conditions
            emergency_conditions = behaviour._check_emergency_conditions([existing_position])
            assert len(emergency_conditions) > 0
            assert existing_position in emergency_conditions
            
            # Step 2: Execute emergency withdrawal
            emergency_result = behaviour._execute_emergency_withdrawal(existing_position)
            
            # Verify emergency withdrawal was executed
            assert emergency_result["success"] is True
            assert "withdrawal_tx" in emergency_result
            assert "remove_liquidity_tx" in emergency_result
            
            # Step 3: Update position tracking
            updated_positions = behaviour._update_positions_after_emergency_withdrawal(
                [existing_position], emergency_result
            )
            
            # Verify position was removed
            assert len(updated_positions) == 0

    def test_complete_velodrome_fee_harvesting_workflow(self, velodrome_volatile_pool_data):
        """Test fee harvesting workflow."""
        # Setup position with accumulated fees
        existing_position = {
            "pool_address": velodrome_volatile_pool_data["pool_address"],
            "lp_balance": 100000000000000000000,  # 100 LP tokens
            "staked_amount": 100000000000000000000,  # 100 staked LP tokens
            "gauge_address": velodrome_volatile_pool_data["gauge_address"],
            "accumulated_fees": 5000000000000000000,  # 5 tokens in fees
            "dex_type": DexType.VELODROME.value
        }
        
        behaviour = self.create_mock_behaviour(VelodromePoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"earned": existing_position["accumulated_fees"]},  # earned rewards
            {"tx_hash": "0xget_reward_tx_hash"},  # get reward transaction
        ]
        
        behaviour.contract_interact = MagicMock(side_effect=mock_contract_responses)
        
        with patch.multiple(
            behaviour,
            get_http_response=self._mock_http_responses(),
            _read_kv=self._mock_kv_read(),
            _write_kv=self._mock_kv_write()
        ):
            # Step 1: Check for harvestable fees
            harvestable_fees = behaviour._check_harvestable_fees([existing_position])
            assert len(harvestable_fees) > 0
            assert existing_position in harvestable_fees
            
            # Step 2: Execute fee harvesting
            harvest_result = behaviour._execute_fee_harvesting(existing_position)
            
            # Verify fee harvesting was executed
            assert harvest_result["success"] is True
            assert "harvest_tx" in harvest_result
            assert harvest_result["harvested_amount"] > 0
            
            # Step 3: Update position tracking
            updated_position = behaviour._update_position_after_fee_harvest(
                existing_position, harvest_result
            )
            
            # Verify fees were reset
            assert updated_position["accumulated_fees"] == 0

    def _mock_http_responses(self):
        """Mock HTTP responses for external API calls."""
        def mock_get_http_response(*args, **kwargs):
            url = args[1] if len(args) > 1 else kwargs.get("url", "")
            if "velodrome" in url or "thegraph" in url:
                return MagicMock(
                    status_code=200,
                    body='{"data": {"pairs": []}}'
                )
            else:
                return MagicMock(
                    status_code=200,
                    body="{}"
                )
        return mock_get_http_response

    def _mock_kv_read(self):
        """Mock KV store read operations."""
        def mock_read_kv(keys):
            yield
            return {}
        return mock_read_kv

    def _mock_kv_write(self):
        """Mock KV store write operations."""
        def mock_write_kv(key, value):
            yield
            return None
        return mock_write_kv

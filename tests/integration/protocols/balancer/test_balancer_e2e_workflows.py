# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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

"""End-to-end integration tests for Balancer protocol workflows."""

import pytest
from unittest.mock import MagicMock, patch

from packages.valory.skills.liquidity_trader_abci.behaviours.base import DexType
from packages.valory.skills.liquidity_trader_abci.pools.balancer import BalancerPoolBehaviour

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions, TestDataGenerator
from tests.integration.fixtures.pool_data_fixtures import (
    balancer_weighted_pool_data,
    balancer_stable_pool_data,
    user_assets_2tokens,
    user_assets_3tokens,
)


class TestBalancerE2EWorkflows(ProtocolIntegrationTestBase):
    """Test complete Balancer protocol workflows."""

    def test_complete_balancer_liquidity_provision_workflow(self, balancer_weighted_pool_data, user_assets_2tokens):
        """Test complete workflow from pool selection to liquidity provision."""
        # Setup test data
        pool_data = balancer_weighted_pool_data
        user_assets = user_assets_2tokens
        
        # Mock the complete workflow
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"balance": 0},  # Initial user BPT balance
            {"data": pool_data["total_supply"]},  # Total supply
            {"tokens": pool_data["tokens"]},  # Pool tokens
            {"tx_hash": b"join_pool_tx_hash"},  # Join transaction
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
            assert optimal_amounts[pool_data["tokens"][0]] > 0
            assert optimal_amounts[pool_data["tokens"][1]] > 0
            
            # Step 3: Generate join transaction
            join_tx = list(behaviour.enter(
                pool_address=selected_pool["pool_address"],
                safe_address=self.test_addresses["safe"],
                assets=selected_pool["tokens"],
                chain="optimism",
                max_amounts_in=list(optimal_amounts.values()),
                pool_id=selected_pool["pool_id"]
            ))
            
            # Verify transaction was generated
            assert len(join_tx) > 0
            assert join_tx[0] is not None
            
            # Step 4: Verify position tracking
            position_data = {
                "pool_id": selected_pool["pool_id"],
                "pool_address": selected_pool["pool_address"],
                "dex_type": DexType.BALANCER.value,
                "amounts": optimal_amounts,
                "timestamp": 1234567890
            }
            
            # Step 5: Test position value calculation
            position_value = list(behaviour.get_user_share_value_balancer(
                self.test_addresses["safe"],
                selected_pool["pool_id"],
                selected_pool["pool_address"],
                "optimism"
            ))
            
            assert len(position_value) > 0
            assert "token0" in position_value[0]
            assert "token1" in position_value[0]

    def test_complete_balancer_liquidity_withdrawal_workflow(self, balancer_weighted_pool_data):
        """Test complete workflow from position identification to withdrawal."""
        # Setup existing position
        existing_position = {
            "pool_id": balancer_weighted_pool_data["pool_id"],
            "pool_address": balancer_weighted_pool_data["pool_address"],
            "dex_type": DexType.BALANCER.value,
            "user_bpt_balance": 100000000000000000000,  # 100 BPT
            "total_supply": 1000000000000000000000,     # 1000 BPT
            "tokens": balancer_weighted_pool_data["tokens"]
        }
        
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"balance": existing_position["user_bpt_balance"]},  # User BPT balance
            {"data": existing_position["total_supply"]},  # Total supply
            {"tokens": existing_position["tokens"]},  # Pool tokens
            {"tx_hash": b"exit_pool_tx_hash"},  # Exit transaction
        ]
        
        behaviour.contract_interact = MagicMock(side_effect=mock_contract_responses)
        
        with patch.multiple(
            behaviour,
            get_http_response=self._mock_http_responses(),
            _read_kv=self._mock_kv_read(),
            _write_kv=self._mock_kv_write()
        ):
            # Step 1: Identify position for withdrawal
            withdrawal_candidate = behaviour._identify_withdrawal_candidate([existing_position])
            assert withdrawal_candidate == existing_position
            
            # Step 2: Calculate withdrawal amounts
            withdrawal_amounts = behaviour._calculate_withdrawal_amounts(
                withdrawal_candidate, 0.5  # 50% withdrawal
            )
            
            # Step 3: Generate exit transaction
            exit_tx = list(behaviour.exit(
                pool_address=withdrawal_candidate["pool_address"],
                safe_address=self.test_addresses["safe"],
                assets=withdrawal_candidate["tokens"],
                chain="optimism",
                bpt_amount_in=withdrawal_amounts["bpt_amount"],
                pool_id=withdrawal_candidate["pool_id"]
            ))
            
            # Verify transaction was generated
            assert len(exit_tx) > 0
            assert exit_tx[0] is not None
            
            # Step 4: Update position tracking
            updated_position = behaviour._update_position_after_withdrawal(
                withdrawal_candidate, withdrawal_amounts
            )
            
            assert updated_position["user_bpt_balance"] < existing_position["user_bpt_balance"]

    def test_balancer_pool_rebalancing_workflow(self, balancer_weighted_pool_data, user_assets_2tokens):
        """Test complete pool rebalancing workflow."""
        # Setup existing position
        existing_position = {
            "pool_id": balancer_weighted_pool_data["pool_id"],
            "pool_address": balancer_weighted_pool_data["pool_address"],
            "dex_type": DexType.BALANCER.value,
            "user_bpt_balance": 100000000000000000000,  # 100 BPT
            "total_supply": 1000000000000000000000,     # 1000 BPT
            "tokens": balancer_weighted_pool_data["tokens"]
        }
        
        # New pool with better APR
        new_pool_data = balancer_weighted_pool_data.copy()
        new_pool_data["pool_id"] = "0x2345678901234567890123456789012345678901234567890123456789012345"
        new_pool_data["pool_address"] = "0xNewPoolAddress"
        new_pool_data["apr"] = 20.0  # Higher APR
        
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Mock contract interactions for withdrawal and entry
        mock_contract_responses = [
            # Withdrawal from old pool
            {"balance": existing_position["user_bpt_balance"]},
            {"data": existing_position["total_supply"]},
            {"tokens": existing_position["tokens"]},
            {"tx_hash": b"exit_pool_tx_hash"},
            # Entry to new pool
            {"balance": 0},  # Initial BPT balance in new pool
            {"data": new_pool_data["total_supply"]},
            {"tokens": new_pool_data["tokens"]},
            {"tx_hash": b"join_pool_tx_hash"},
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
            assert updated_positions[0]["pool_id"] == new_pool_data["pool_id"]

    def test_balancer_multi_pool_diversification_workflow(self, user_assets_3tokens):
        """Test multi-pool diversification workflow."""
        # Setup multiple pools
        pool1_data = TestDataGenerator.generate_balancer_pool_data(
            pool_id="0x1111111111111111111111111111111111111111111111111111111111111111",
            pool_type="Weighted",
            num_tokens=2,
            weights=[600000000000000000, 400000000000000000],  # 60/40 weights
        )
        pool1_data["apr"] = 12.0
        
        pool2_data = TestDataGenerator.generate_balancer_pool_data(
            pool_id="0x2222222222222222222222222222222222222222222222222222222222222222",
            pool_type="ComposableStable",
            num_tokens=3,
            weights=[333333333333333333, 333333333333333333, 333333333333333334],
        )
        pool2_data["apr"] = 8.0
        
        pool3_data = TestDataGenerator.generate_balancer_pool_data(
            pool_id="0x3333333333333333333333333333333333333333333333333333333333333333",
            pool_type="Weighted",
            num_tokens=2,
            weights=[500000000000000000, 500000000000000000],  # 50/50 weights
        )
        pool3_data["apr"] = 15.0
        
        available_pools = [pool1_data, pool2_data, pool3_data]
        
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Mock contract interactions for multiple pools
        mock_contract_responses = []
        for pool in available_pools:
            mock_contract_responses.extend([
                {"balance": 0},  # Initial BPT balance
                {"data": pool["total_supply"]},  # Total supply
                {"tokens": pool["tokens"]},  # Pool tokens
                {"tx_hash": f"join_pool_{pool['pool_id'][:8]}_tx_hash".encode()},  # Join transaction
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
                available_pools, user_assets_3tokens
            )
            
            # Verify allocation
            assert len(allocation) > 0
            total_allocation = sum(allocation.values())
            assert total_allocation <= 1.0  # Should not exceed 100%
            
            # Step 2: Execute diversification
            diversification_result = behaviour._execute_diversification(
                allocation, available_pools, user_assets_3tokens
            )
            
            # Verify diversification was executed
            assert diversification_result["success"] is True
            assert len(diversification_result["transactions"]) > 0
            assert len(diversification_result["positions"]) > 0
            
            # Step 3: Verify position tracking
            for position in diversification_result["positions"]:
                TestAssertions.assert_position_data_structure(position, DexType.BALANCER.value)

    def test_balancer_emergency_withdrawal_workflow(self, balancer_weighted_pool_data):
        """Test emergency withdrawal workflow."""
        # Setup position with high impermanent loss
        existing_position = {
            "pool_id": balancer_weighted_pool_data["pool_id"],
            "pool_address": balancer_weighted_pool_data["pool_address"],
            "dex_type": DexType.BALANCER.value,
            "user_bpt_balance": 100000000000000000000,  # 100 BPT
            "total_supply": 1000000000000000000000,     # 1000 BPT
            "tokens": balancer_weighted_pool_data["tokens"],
            "impermanent_loss": 15.0,  # 15% IL
        }
        
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"balance": existing_position["user_bpt_balance"]},
            {"data": existing_position["total_supply"]},
            {"tokens": existing_position["tokens"]},
            {"tx_hash": b"emergency_exit_tx_hash"},
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
            
            # Step 3: Update position tracking
            updated_positions = behaviour._update_positions_after_emergency_withdrawal(
                emergency_result
            )
            
            # Verify position was removed
            assert len(updated_positions) == 0

    def test_balancer_fee_harvesting_workflow(self, balancer_weighted_pool_data):
        """Test fee harvesting workflow."""
        # Setup position with accumulated fees
        existing_position = {
            "pool_id": balancer_weighted_pool_data["pool_id"],
            "pool_address": balancer_weighted_pool_data["pool_address"],
            "dex_type": DexType.BALANCER.value,
            "user_bpt_balance": 100000000000000000000,  # 100 BPT
            "total_supply": 1000000000000000000000,     # 1000 BPT
            "tokens": balancer_weighted_pool_data["tokens"],
            "accumulated_fees": 5000000000000000000,  # 5 tokens in fees
        }
        
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"balance": existing_position["user_bpt_balance"]},
            {"data": existing_position["total_supply"]},
            {"tokens": existing_position["tokens"]},
            {"tx_hash": b"harvest_fees_tx_hash"},
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
            if "balancer" in url:
                return MagicMock(
                    status_code=200,
                    body='{"pools": []}'
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

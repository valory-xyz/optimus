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

"""End-to-end integration tests for Uniswap V3 protocol workflows."""

import pytest
from unittest.mock import MagicMock, patch

from packages.valory.skills.liquidity_trader_abci.behaviours.base import DexType
from packages.valory.skills.liquidity_trader_abci.pools.uniswap import UniswapPoolBehaviour

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase
from tests.integration.protocols.base.test_helpers import TestAssertions, TestDataGenerator
from tests.integration.fixtures.pool_data_fixtures import (
    uniswap_v3_pool_data,
    uniswap_v3_high_fee_pool_data,
    uniswap_v3_low_fee_pool_data,
    user_assets_2tokens,
)


class TestUniswapV3E2EWorkflows(ProtocolIntegrationTestBase):
    """Test complete Uniswap V3 protocol workflows."""

    def test_complete_uniswap_v3_position_creation_workflow(self, uniswap_v3_pool_data, user_assets_2tokens):
        """Test complete workflow from pool analysis to position creation."""
        # Setup test data
        pool_data = uniswap_v3_pool_data
        user_assets = user_assets_2tokens
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"slot0": {"sqrt_price_x96": 79228162514264337593543950336, "tick": -276310, "unlocked": True}},  # slot0
            {"tokens": [pool_data["token0"], pool_data["token1"]]},  # pool tokens
            {"tx_hash": "0xmint_tx_hash"},  # mint transaction
        ]
        
        behaviour.contract_interact = MagicMock(side_effect=mock_contract_responses)
        
        with patch.multiple(
            behaviour,
            get_http_response=self._mock_http_responses(),
            _read_kv=self._mock_kv_read(),
            _write_kv=self._mock_kv_write()
        ):
            # Step 1: Pool analysis and selection
            selected_pool = behaviour._select_optimal_pool([pool_data])
            assert selected_pool == pool_data
            
            # Step 2: Calculate optimal tick range
            tick_range = behaviour._calculate_optimal_tick_range(
                selected_pool, user_assets
            )
            assert tick_range["tick_lower"] < tick_range["tick_upper"]
            assert tick_range["tick_lower"] % selected_pool["tick_spacing"] == 0
            assert tick_range["tick_upper"] % selected_pool["tick_spacing"] == 0
            
            # Step 3: Calculate liquidity amounts
            liquidity_amounts = behaviour._calculate_liquidity_amounts(
                selected_pool, tick_range, user_assets
            )
            
            # Step 4: Generate mint transaction
            mint_tx = list(behaviour.enter(
                pool_address=selected_pool["pool_address"],
                safe_address=self.test_addresses["safe"],
                assets=[selected_pool["token0"], selected_pool["token1"]],
                chain="optimism",
                max_amounts_in=[liquidity_amounts["amount0"], liquidity_amounts["amount1"]],
                pool_fee=selected_pool["fee"],
                tick_lower=tick_range["tick_lower"],
                tick_upper=tick_range["tick_upper"]
            ))
            
            # Verify transaction was generated
            assert len(mint_tx) > 0
            assert mint_tx[0] is not None
            
            # Step 5: Track new position
            new_position = {
                "token_id": 12345,  # Mock NFT token ID
                "pool_address": selected_pool["pool_address"],
                "token0": selected_pool["token0"],
                "token1": selected_pool["token1"],
                "fee": selected_pool["fee"],
                "tick_lower": tick_range["tick_lower"],
                "tick_upper": tick_range["tick_upper"],
                "liquidity": liquidity_amounts["liquidity"],
                "dex_type": DexType.UNISWAP_V3.value
            }
            
            # Step 6: Test position value calculation
            position_value = list(behaviour.get_user_share_value_uniswap(
                selected_pool["pool_address"],
                new_position["token_id"],
                "optimism",
                new_position
            ))
            
            assert len(position_value) > 0
            assert "amount0" in position_value[0]
            assert "amount1" in position_value[0]

    def test_complete_uniswap_v3_fee_collection_workflow(self, uniswap_v3_pool_data):
        """Test complete workflow for fee collection and compounding."""
        # Setup existing position with accumulated fees
        existing_position = {
            "token_id": 12345,
            "pool_address": uniswap_v3_pool_data["pool_address"],
            "token0": uniswap_v3_pool_data["token0"],
            "token1": uniswap_v3_pool_data["token1"],
            "fee": uniswap_v3_pool_data["fee"],
            "tick_lower": -276320,
            "tick_upper": -276300,
            "liquidity": 1000000000000000000,
            "tokens_owed0": 100000000000000000,  # 0.1 tokens
            "tokens_owed1": 200000000000000000,  # 0.2 tokens
            "dex_type": DexType.UNISWAP_V3.value
        }
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"liquidity": 1000000000000000000, "tokensOwed0": 100000000000000000, "tokensOwed1": 200000000000000000},  # position data
            {"tx_hash": "0xcollect_tx_hash"},  # collect transaction
        ]
        
        behaviour.contract_interact = MagicMock(side_effect=mock_contract_responses)
        
        with patch.multiple(
            behaviour,
            get_http_response=self._mock_http_responses(),
            _read_kv=self._mock_kv_read(),
            _write_kv=self._mock_kv_write()
        ):
            # Step 1: Check for collectable fees
            collectable_fees = behaviour._check_collectable_fees(existing_position)
            assert collectable_fees["amount0"] > 0
            assert collectable_fees["amount1"] > 0
            
            # Step 2: Generate collect transaction
            collect_tx = list(behaviour._collect_fees(
                existing_position["token_id"],
                collectable_fees["amount0"],
                collectable_fees["amount1"]
            ))
            
            # Verify transaction was generated
            assert len(collect_tx) > 0
            assert collect_tx[0] is not None
            
            # Step 3: Test fee compounding (if enabled)
            if behaviour._should_compound_fees(collectable_fees):
                compound_tx = list(behaviour._compound_fees(
                    existing_position, collectable_fees
                ))
                assert len(compound_tx) > 0

    def test_complete_uniswap_v3_position_management_workflow(self, uniswap_v3_pool_data):
        """Test complete position management workflow."""
        # Setup existing position
        existing_position = {
            "token_id": 12345,
            "pool_address": uniswap_v3_pool_data["pool_address"],
            "token0": uniswap_v3_pool_data["token0"],
            "token1": uniswap_v3_pool_data["token1"],
            "fee": uniswap_v3_pool_data["fee"],
            "tick_lower": -276320,
            "tick_upper": -276300,
            "liquidity": 1000000000000000000,
            "tokens_owed0": 0,
            "tokens_owed1": 0,
            "dex_type": DexType.UNISWAP_V3.value
        }
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"liquidity": 1000000000000000000, "tokensOwed0": 0, "tokensOwed1": 0},  # position data
            {"tx_hash": "0xdecrease_liquidity_tx_hash"},  # decrease liquidity transaction
            {"tx_hash": "0xcollect_tx_hash"},  # collect transaction
            {"tx_hash": "0xburn_tx_hash"},  # burn transaction
        ]
        
        behaviour.contract_interact = MagicMock(side_effect=mock_contract_responses)
        
        with patch.multiple(
            behaviour,
            get_http_response=self._mock_http_responses(),
            _read_kv=self._mock_kv_read(),
            _write_kv=self._mock_kv_write()
        ):
            # Step 1: Decrease liquidity
            decrease_tx = list(behaviour._decrease_liquidity(
                existing_position["token_id"],
                500000000000000000  # 50% of liquidity
            ))
            
            # Verify transaction was generated
            assert len(decrease_tx) > 0
            assert decrease_tx[0] is not None
            
            # Step 2: Collect remaining fees
            collect_tx = list(behaviour._collect_fees(
                existing_position["token_id"],
                0,  # amount0_max
                0   # amount1_max
            ))
            
            # Verify transaction was generated
            assert len(collect_tx) > 0
            assert collect_tx[0] is not None
            
            # Step 3: Burn position
            burn_tx = list(behaviour._burn_position(
                existing_position["token_id"]
            ))
            
            # Verify transaction was generated
            assert len(burn_tx) > 0
            assert burn_tx[0] is not None

    def test_complete_uniswap_v3_rebalancing_workflow(self, uniswap_v3_pool_data, user_assets_2tokens):
        """Test complete position rebalancing workflow."""
        # Setup existing position
        existing_position = {
            "token_id": 12345,
            "pool_address": uniswap_v3_pool_data["pool_address"],
            "token0": uniswap_v3_pool_data["token0"],
            "token1": uniswap_v3_pool_data["token1"],
            "fee": uniswap_v3_pool_data["fee"],
            "tick_lower": -276320,
            "tick_upper": -276300,
            "liquidity": 1000000000000000000,
            "tokens_owed0": 0,
            "tokens_owed1": 0,
            "dex_type": DexType.UNISWAP_V3.value
        }
        
        # New optimal tick range
        new_tick_range = {
            "tick_lower": -276340,
            "tick_upper": -276280,
        }
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"liquidity": 1000000000000000000, "tokensOwed0": 0, "tokens_owed1": 0},  # position data
            {"tx_hash": "0xdecrease_liquidity_tx_hash"},  # decrease liquidity
            {"tx_hash": "0xcollect_tx_hash"},  # collect fees
            {"tx_hash": "0xincrease_liquidity_tx_hash"},  # increase liquidity with new range
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
                existing_position, new_tick_range
            )
            assert rebalancing_opportunity is not None
            
            # Step 2: Execute rebalancing
            rebalancing_result = behaviour._execute_rebalancing(
                existing_position, new_tick_range, user_assets_2tokens
            )
            
            # Verify rebalancing was executed
            assert rebalancing_result["success"] is True
            assert "decrease_tx" in rebalancing_result
            assert "collect_tx" in rebalancing_result
            assert "increase_tx" in rebalancing_result
            
            # Step 3: Update position tracking
            updated_position = behaviour._update_position_after_rebalancing(
                existing_position, rebalancing_result
            )
            
            # Verify position was updated
            assert updated_position["tick_lower"] == new_tick_range["tick_lower"]
            assert updated_position["tick_upper"] == new_tick_range["tick_upper"]

    def test_complete_uniswap_v3_multi_position_management(self, uniswap_v3_pool_data):
        """Test management of multiple positions."""
        # Setup multiple positions
        positions = [
            {
                "token_id": 12345,
                "pool_address": uniswap_v3_pool_data["pool_address"],
                "token0": uniswap_v3_pool_data["token0"],
                "token1": uniswap_v3_pool_data["token1"],
                "fee": uniswap_v3_pool_data["fee"],
                "tick_lower": -276320,
                "tick_upper": -276300,
                "liquidity": 1000000000000000000,
                "tokens_owed0": 100000000000000000,
                "tokens_owed1": 200000000000000000,
                "dex_type": DexType.UNISWAP_V3.value
            },
            {
                "token_id": 12346,
                "pool_address": uniswap_v3_pool_data["pool_address"],
                "token0": uniswap_v3_pool_data["token0"],
                "token1": uniswap_v3_pool_data["token1"],
                "fee": uniswap_v3_pool_data["fee"],
                "tick_lower": -276340,
                "tick_upper": -276280,
                "liquidity": 2000000000000000000,
                "tokens_owed0": 50000000000000000,
                "tokens_owed1": 100000000000000000,
                "dex_type": DexType.UNISWAP_V3.value
            }
        ]
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions for multiple positions
        mock_contract_responses = []
        for position in positions:
            mock_contract_responses.extend([
                {"liquidity": position["liquidity"], "tokensOwed0": position["tokens_owed0"], "tokensOwed1": position["tokens_owed1"]},  # position data
                {"tx_hash": f"0xcollect_{position['token_id']}_tx_hash"},  # collect transaction
            ])
        
        behaviour.contract_interact = MagicMock(side_effect=mock_contract_responses)
        
        with patch.multiple(
            behaviour,
            get_http_response=self._mock_http_responses(),
            _read_kv=self._mock_kv_read(),
            _write_kv=self._mock_kv_write()
        ):
            # Step 1: Identify positions with collectable fees
            collectable_positions = behaviour._identify_collectable_positions(positions)
            assert len(collectable_positions) == 2
            
            # Step 2: Execute batch fee collection
            collection_result = behaviour._execute_batch_fee_collection(collectable_positions)
            
            # Verify batch collection was executed
            assert collection_result["success"] is True
            assert len(collection_result["transactions"]) == 2
            
            # Step 3: Update position tracking
            updated_positions = behaviour._update_positions_after_batch_collection(
                positions, collection_result
            )
            
            # Verify positions were updated
            assert len(updated_positions) == 2
            for position in updated_positions:
                assert position["tokens_owed0"] == 0
                assert position["tokens_owed1"] == 0

    def test_complete_uniswap_v3_emergency_withdrawal_workflow(self, uniswap_v3_pool_data):
        """Test emergency withdrawal workflow."""
        # Setup position with high impermanent loss
        existing_position = {
            "token_id": 12345,
            "pool_address": uniswap_v3_pool_data["pool_address"],
            "token0": uniswap_v3_pool_data["token0"],
            "token1": uniswap_v3_pool_data["token1"],
            "fee": uniswap_v3_pool_data["fee"],
            "tick_lower": -276320,
            "tick_upper": -276300,
            "liquidity": 1000000000000000000,
            "tokens_owed0": 0,
            "tokens_owed1": 0,
            "impermanent_loss": 15.0,  # 15% IL
            "dex_type": DexType.UNISWAP_V3.value
        }
        
        behaviour = self.create_mock_behaviour(UniswapPoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"liquidity": 1000000000000000000, "tokensOwed0": 0, "tokensOwed1": 0},  # position data
            {"tx_hash": "0xdecrease_liquidity_tx_hash"},  # decrease liquidity
            {"tx_hash": "0xcollect_tx_hash"},  # collect fees
            {"tx_hash": "0xburn_tx_hash"},  # burn position
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
            assert "decrease_tx" in emergency_result
            assert "collect_tx" in emergency_result
            assert "burn_tx" in emergency_result
            
            # Step 3: Update position tracking
            updated_positions = behaviour._update_positions_after_emergency_withdrawal(
                [existing_position], emergency_result
            )
            
            # Verify position was removed
            assert len(updated_positions) == 0

    def _mock_http_responses(self):
        """Mock HTTP responses for external API calls."""
        def mock_get_http_response(*args, **kwargs):
            url = args[1] if len(args) > 1 else kwargs.get("url", "")
            if "uniswap" in url or "thegraph" in url:
                return MagicMock(
                    status_code=200,
                    body='{"data": {"pools": []}}'
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

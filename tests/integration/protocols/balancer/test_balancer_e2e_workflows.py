# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
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
)


class TestBalancerE2EWorkflows(ProtocolIntegrationTestBase):
    """Test complete Balancer protocol workflows using real methods."""

    def _consume_generator(self, gen):
        """Helper to run generator and return final value."""
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    def test_balancer_enter_workflow(self, balancer_weighted_pool_data):
        """Test complete enter workflow using real methods."""
        # Setup test data
        pool_data = balancer_weighted_pool_data
        user_assets = {
            "tokens": ["0x4200000000000000000000000000000000000006", "0x7F5c764cBc14f9669B88837ca1490cCa17c31607"],
            "balances": [1000000000000000000000, 2000000000000000000000]
        }
        
        # Mock the behaviour
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Mock contract interactions for enter workflow
        def mock_contract_interact(*args, **kwargs):
            yield  # Yield first
            if kwargs.get("contract_callable") == "get_pool_id":
                return {"pool_id": pool_data["pool_id"]}
            elif kwargs.get("contract_callable") == "get_pool_tokens":
                return (pool_data["tokens"], [1000000000000000000000, 2000000000000000000000], 12345)
            elif kwargs.get("contract_callable") == "get_total_supply":
                return 1000000000000000000000  # Return total supply as integer
            elif kwargs.get("contract_callable") == "query_join":
                return {"bpt_out": 500000000000000000000}
            elif kwargs.get("contract_callable") == "join_pool":
                return "0x1234567890abcdef"  # Return tx_hash as string
            return None
        
        behaviour.contract_interact = mock_contract_interact
        
        with patch.multiple(
            behaviour,
            get_http_response=self.mock_http_response_generator([])
        ):
            # Test the complete enter workflow
            enter_generator = behaviour.enter(
                pool_address=pool_data["pool_address"],
                safe_address=self.test_addresses["safe"],
                assets=pool_data["tokens"],
                chain="optimism",
                max_amounts_in=[1000000000000000000000, 2000000000000000000000],
                pool_id=pool_data["pool_id"],
                pool_type="Weighted"  # Add required pool_type parameter
            )
            
            # Consume the generator to get the final result
            enter_result = self._consume_generator(enter_generator)
            
            # Verify enter transaction was generated
            assert enter_result is not None
            assert len(enter_result) == 2  # Should return (tx_hash, vault_address)
            assert enter_result[0] is not None  # tx_hash
            assert enter_result[1] is not None  # vault_address

    def test_balancer_exit_workflow(self, balancer_weighted_pool_data):
        """Test complete exit workflow using real methods."""
        # Setup existing position
        existing_position = {
            "pool_id": balancer_weighted_pool_data["pool_id"],
            "pool_address": balancer_weighted_pool_data["pool_address"],
            "dex_type": DexType.BALANCER.value,
            "user_bpt_balance": 100000000000000000000,  # 100 BPT
            "tokens": balancer_weighted_pool_data["tokens"],
        }
        
        # Mock the behaviour
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Mock contract interactions for exit workflow
        def mock_contract_interact(*args, **kwargs):
            yield  # Yield first
            if kwargs.get("contract_callable") == "get_pool_id":
                return {"pool_id": existing_position["pool_id"]}
            elif kwargs.get("contract_callable") == "get_balance":
                return existing_position["user_bpt_balance"]
            elif kwargs.get("contract_callable") == "get_pool_tokens":
                return (existing_position["tokens"], [1000000000000000000000, 2000000000000000000000], 12345)
            elif kwargs.get("contract_callable") == "get_total_supply":
                return 1000000000000000000000
            elif kwargs.get("contract_callable") == "query_exit":
                return {"amounts_out": [50000000000000000000, 100000000000000000000]}
            elif kwargs.get("contract_callable") == "exit_pool":
                return "0x1234567890abcdef"  # Return tx_hash as string
            return None
        
        behaviour.contract_interact = mock_contract_interact
        
        with patch.multiple(
            behaviour,
            get_http_response=self.mock_http_response_generator([])
        ):
            # Test the complete exit workflow
            exit_generator = behaviour.exit(
                pool_address=existing_position["pool_address"],
                safe_address=self.test_addresses["safe"],
                assets=existing_position["tokens"],
                chain="optimism",
                bpt_amount_in=existing_position["user_bpt_balance"],
                pool_id=existing_position["pool_id"],
                pool_type="Weighted"
            )
            
            # Consume the generator to get the final result
            exit_result = self._consume_generator(exit_generator)
            
            # Verify exit transaction was generated
            assert exit_result is not None
            assert len(exit_result) == 3  # Should return (tx_hash, vault_address, bool)
            assert exit_result[0] is not None  # tx_hash
            assert exit_result[1] is not None  # vault_address
            assert isinstance(exit_result[2], bool)  # bool flag

    def test_balancer_proportional_join_workflow(self, balancer_weighted_pool_data):
        """Test proportional join workflow using real methods."""
        # Setup test data
        pool_data = balancer_weighted_pool_data
        user_assets = {
            "tokens": ["0x4200000000000000000000000000000000000006", "0x7F5c764cBc14f9669B88837ca1490cCa17c31607"],
            "balances": [1000000000000000000000, 2000000000000000000000]
        }
        
        # Mock the behaviour
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            yield  # Yield first
            if kwargs.get("contract_callable") == "get_pool_id":
                return {"pool_id": pool_data["pool_id"]}
            elif kwargs.get("contract_callable") == "get_pool_tokens":
                return (pool_data["tokens"], [1000000000000000000000, 2000000000000000000000], 12345)
            elif kwargs.get("contract_callable") == "get_total_supply":
                return 1000000000000000000000
            elif kwargs.get("contract_callable") == "query_join":
                return {"bpt_out": 500000000000000000000}
            return None
        
        behaviour.contract_interact = mock_contract_interact
        
        with patch.multiple(
            behaviour,
            get_http_response=self.mock_http_response_generator([])
        ):
            # Test proportional join query
            join_generator = behaviour.query_proportional_join(
                pool_id=pool_data["pool_id"],
                pool_address=pool_data["pool_address"],
                vault_address="0xVaultAddress",
                chain="optimism",
                amounts_in=[1000000000000000000000, 2000000000000000000000]
            )
            
            # Consume the generator to get the final result
            join_result = self._consume_generator(join_generator)
            
            # Verify proportional join query worked
            assert join_result is not None
            assert join_result > 0

    def test_balancer_proportional_exit_workflow(self, balancer_weighted_pool_data):
        """Test proportional exit workflow using real methods."""
        # Setup existing position
        existing_position = {
            "pool_id": balancer_weighted_pool_data["pool_id"],
            "pool_address": balancer_weighted_pool_data["pool_address"],
            "dex_type": DexType.BALANCER.value,
            "user_bpt_balance": 100000000000000000000,  # 100 BPT
            "tokens": balancer_weighted_pool_data["tokens"],
        }
        
        # Mock the behaviour
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Mock contract interactions
        def mock_contract_interact(*args, **kwargs):
            yield  # Yield first
            if kwargs.get("contract_callable") == "get_pool_id":
                return {"pool_id": existing_position["pool_id"]}
            elif kwargs.get("contract_callable") == "get_pool_tokens":
                return (existing_position["tokens"], [1000000000000000000000, 2000000000000000000000], 12345)
            elif kwargs.get("contract_callable") == "get_total_supply":
                return 1000000000000000000000
            elif kwargs.get("contract_callable") == "query_exit":
                return {"amounts_out": [50000000000000000000, 100000000000000000000]}
            return None
        
        behaviour.contract_interact = mock_contract_interact
        
        with patch.multiple(
            behaviour,
            get_http_response=self.mock_http_response_generator([])
        ):
            # Test proportional exit query
            exit_generator = behaviour.query_proportional_exit(
                pool_id=existing_position["pool_id"],
                pool_address=existing_position["pool_address"],
                vault_address="0xVaultAddress",
                chain="optimism",
                bpt_amount_in=existing_position["user_bpt_balance"]
            )
            
            # Consume the generator to get the final result
            exit_result = self._consume_generator(exit_generator)
            
            # Verify proportional exit query worked
            assert exit_result is not None
            assert isinstance(exit_result, list)
            assert len(exit_result) > 0
            assert all(amount > 0 for amount in exit_result)

    def test_balancer_amount_adjustment_workflow(self):
        """Test amount adjustment workflow using real methods."""
        # Setup test data
        user_assets = {
            "tokens": ["0x4200000000000000000000000000000000000006", "0x7F5c764cBc14f9669B88837ca1490cCa17c31607"],
            "balances": [1000000000000000000000, 2000000000000000000000]
        }
        target_amounts = [1000000000000000000000, 2000000000000000000000]  # 1000, 2000 tokens
        
        # Mock the behaviour
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        with patch.multiple(
            behaviour,
            get_http_response=self.mock_http_response_generator([])
        ):
            # Test amount adjustment
            adjusted_amounts = behaviour.adjust_amounts(
                assets=user_assets["tokens"],
                assets_new=user_assets["tokens"],
                max_amounts_in=target_amounts
            )
            
            # Verify amount adjustment worked
            assert adjusted_amounts is not None
            assert len(adjusted_amounts) == len(target_amounts)
            assert all(amount >= 0 for amount in adjusted_amounts)

    def test_balancer_value_update_workflow(self, balancer_weighted_pool_data):
        """Test value update workflow using real methods."""
        # Setup test data
        pool_data = balancer_weighted_pool_data
        position_value = 1000000000000000000000  # 1000 tokens
        
        # Mock the behaviour
        behaviour = self.create_mock_behaviour(BalancerPoolBehaviour)
        
        # Mock contract interactions
        mock_contract_responses = [
            {"balance": 100000000000000000000},  # User BPT balance
            {"data": 1000000000000000000000},  # Total supply
            {"tokens": pool_data["tokens"]},  # Pool tokens
        ]
        
        behaviour.contract_interact = MagicMock(side_effect=mock_contract_responses)
        
        with patch.multiple(
            behaviour,
            get_http_response=self.mock_http_response_generator([])
        ):
            # Test value update
            updated_value = list(behaviour.update_value(
                pool_address=pool_data["pool_address"],
                safe_address=self.test_addresses["safe"],
                assets=pool_data["tokens"],
                chain="optimism",
                pool_id=pool_data["pool_id"]
            ))
            
            # Verify value update worked
            assert len(updated_value) > 0
            assert updated_value[0] is not None
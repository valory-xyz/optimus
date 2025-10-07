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

"""Fixtures for staking and compliance tests."""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock

from packages.valory.skills.liquidity_trader_abci.states.base import StakingState


class MockStakingContract:
    """Mock staking contract that simulates OLAS staking mechanics."""
    
    def __init__(self):
        self.staking_states = {}  # service_id -> staking_state
        self.checkpoints = {}     # service_id -> last_checkpoint_timestamp
        self.rewards = {}         # service_id -> accumulated_rewards
        self.kpi_requirements = {}  # service_id -> min_transactions_required
        self.transaction_counts = {}  # service_id -> transaction_count_since_last_cp
        self.liveness_periods = {}  # service_id -> liveness_period
        self.liveness_ratios = {}  # service_id -> liveness_ratio
        
    def get_service_staking_state(self, service_id: int) -> int:
        """Get staking state for a service."""
        return self.staking_states.get(service_id, StakingState.UNSTAKED.value)
    
    def set_service_staking_state(self, service_id: int, state: int) -> None:
        """Set staking state for a service."""
        self.staking_states[service_id] = state
    
    def get_last_checkpoint(self, service_id: int) -> Optional[int]:
        """Get last checkpoint timestamp for a service."""
        return self.checkpoints.get(service_id)
    
    def set_last_checkpoint(self, service_id: int, timestamp: int) -> None:
        """Set last checkpoint timestamp for a service."""
        self.checkpoints[service_id] = timestamp
    
    def get_accumulated_rewards(self, service_id: int) -> int:
        """Get accumulated rewards for a service."""
        return self.rewards.get(service_id, 0)
    
    def set_accumulated_rewards(self, service_id: int, amount: int) -> None:
        """Set accumulated rewards for a service."""
        self.rewards[service_id] = amount
    
    def get_min_transactions_required(self, service_id: int) -> int:
        """Get minimum transactions required for a service."""
        return self.kpi_requirements.get(service_id, 5)
    
    def set_min_transactions_required(self, service_id: int, count: int) -> None:
        """Set minimum transactions required for a service."""
        self.kpi_requirements[service_id] = count
    
    def get_transaction_count_since_checkpoint(self, service_id: int) -> int:
        """Get transaction count since last checkpoint for a service."""
        return self.transaction_counts.get(service_id, 0)
    
    def set_transaction_count_since_checkpoint(self, service_id: int, count: int) -> None:
        """Set transaction count since last checkpoint for a service."""
        self.transaction_counts[service_id] = count
    
    def increment_transaction_count(self, service_id: int) -> None:
        """Increment transaction count for a service."""
        current_count = self.transaction_counts.get(service_id, 0)
        self.transaction_counts[service_id] = current_count + 1
    
    def get_liveness_period(self, service_id: int) -> int:
        """Get liveness period for a service."""
        return self.liveness_periods.get(service_id, 86400)  # Default 24 hours
    
    def set_liveness_period(self, service_id: int, period: int) -> None:
        """Set liveness period for a service."""
        self.liveness_periods[service_id] = period
    
    def get_liveness_ratio(self, service_id: int) -> int:
        """Get liveness ratio for a service."""
        return self.liveness_ratios.get(service_id, 100)  # Default 100%
    
    def set_liveness_ratio(self, service_id: int, ratio: int) -> None:
        """Set liveness ratio for a service."""
        self.liveness_ratios[service_id] = ratio
    
    def build_checkpoint_tx(self) -> bytes:
        """Build checkpoint transaction data."""
        return b"0xcheckpoint_data"
    
    def build_vanity_tx(self) -> bytes:
        """Build vanity transaction data."""
        return b"0xvanity_data"
    
    def build_stake_tx(self, service_id: int) -> bytes:
        """Build stake transaction data."""
        return f"0xstake_data_{service_id}".encode()
    
    def build_unstake_tx(self, service_id: int) -> bytes:
        """Build unstake transaction data."""
        return f"0xunstake_data_{service_id}".encode()


@pytest.fixture
def mock_staking_contract():
    """Mock staking contract with realistic behavior."""
    return MockStakingContract()


@pytest.fixture
def staking_compliance_scenarios():
    """Various compliance test scenarios."""
    return {
        "normal_compliance": {
            "service_id": 1,
            "staking_state": StakingState.STAKED.value,
            "min_transactions_required": 5,
            "transactions_since_checkpoint": 8,
            "expected_kpi_met": True,
            "expected_vanity_tx": False,
        },
        "kpi_not_met": {
            "service_id": 2,
            "staking_state": StakingState.STAKED.value,
            "min_transactions_required": 10,
            "transactions_since_checkpoint": 3,
            "expected_kpi_met": False,
            "expected_vanity_tx": True,
        },
        "service_unstaked": {
            "service_id": 3,
            "staking_state": StakingState.UNSTAKED.value,
            "min_transactions_required": 5,
            "transactions_since_checkpoint": 0,
            "expected_kpi_met": None,
            "expected_vanity_tx": False,
        },
        "service_evicted": {
            "service_id": 4,
            "staking_state": StakingState.EVICTED.value,
            "min_transactions_required": 5,
            "transactions_since_checkpoint": 0,
            "expected_kpi_met": None,
            "expected_vanity_tx": False,
        },
        "threshold_exceeded": {
            "service_id": 5,
            "staking_state": StakingState.STAKED.value,
            "min_transactions_required": 5,
            "transactions_since_checkpoint": 2,
            "period_count": 10,
            "period_number_at_last_cp": 3,
            "staking_threshold_period": 5,
            "expected_kpi_met": False,
            "expected_vanity_tx": True,
        },
    }


@pytest.fixture
def kpi_test_data():
    """KPI requirement test data."""
    return {
        "valid_kpi_met": {
            "min_num_of_safe_tx_required": 5,
            "multisig_nonces_since_last_cp": 8,
            "expected_result": True,
        },
        "valid_kpi_not_met": {
            "min_num_of_safe_tx_required": 10,
            "multisig_nonces_since_last_cp": 3,
            "expected_result": False,
        },
        "invalid_min_tx_required": {
            "min_num_of_safe_tx_required": None,
            "multisig_nonces_since_last_cp": 5,
            "expected_result": None,
        },
        "invalid_nonces": {
            "min_num_of_safe_tx_required": 5,
            "multisig_nonces_since_last_cp": None,
            "expected_result": None,
        },
    }


@pytest.fixture
def checkpoint_test_data():
    """Checkpoint test data."""
    return {
        "checkpoint_reached": {
            "next_checkpoint": 1600000000,
            "current_timestamp": 1700000000,
            "expected_reached": True,
        },
        "checkpoint_not_reached": {
            "next_checkpoint": 1800000000,
            "current_timestamp": 1700000000,
            "expected_reached": False,
        },
        "checkpoint_exactly_at_time": {
            "next_checkpoint": 1700000000,
            "current_timestamp": 1700000000,
            "expected_reached": True,
        },
        "no_next_checkpoint": {
            "next_checkpoint": None,
            "current_timestamp": 1700000000,
            "expected_reached": False,
        },
        "zero_checkpoint": {
            "next_checkpoint": 0,
            "current_timestamp": 1700000000,
            "expected_reached": True,
        },
    }


@pytest.fixture
def vanity_transaction_test_data():
    """Vanity transaction test data."""
    return {
        "successful_vanity_tx": {
            "safe_tx_hash": "0xSafeTxHash1234567890123456789012345678901234567890123456789012345678901234",
            "expected_final_hash": "0xFinalHash1234567890123456789012345678901234567890123456789012345678901234",
            "should_succeed": True,
        },
        "failed_contract_interaction": {
            "safe_tx_hash": None,
            "expected_final_hash": None,
            "should_succeed": False,
        },
        "empty_hash": {
            "safe_tx_hash": "",
            "expected_final_hash": None,
            "should_succeed": False,
        },
        "hash_payload_failure": {
            "safe_tx_hash": "0xValidHash",
            "expected_final_hash": None,
            "should_succeed": False,
            "hash_payload_exception": True,
        },
    }


@pytest.fixture
def staking_state_test_data():
    """Staking state test data."""
    return {
        "staked_service": {
            "service_id": 1,
            "staking_state": StakingState.STAKED.value,
            "expected_state": StakingState.STAKED.value,
        },
        "unstaked_service": {
            "service_id": 2,
            "staking_state": StakingState.UNSTAKED.value,
            "expected_state": StakingState.UNSTAKED.value,
        },
        "evicted_service": {
            "service_id": 3,
            "staking_state": StakingState.EVICTED.value,
            "expected_state": StakingState.EVICTED.value,
        },
    }


@pytest.fixture
def mock_contract_responses():
    """Mock contract interaction responses."""
    return {
        "get_service_staking_state": {
            "staked": StakingState.STAKED.value,
            "unstaked": StakingState.UNSTAKED.value,
            "evicted": StakingState.EVICTED.value,
        },
        "build_checkpoint_tx": b"0xcheckpoint_data",
        "build_vanity_tx": b"0xvanity_data",
        "get_liveness_period": 86400,  # 24 hours
        "get_liveness_ratio": 100,  # 100%
        "get_multisig_nonces": [10, 15, 20],  # Mock nonce list
        "get_multisig_nonces_since_last_cp": 5,
        "calculate_min_num_of_safe_tx_required": 5,
    }


@pytest.fixture
def test_addresses():
    """Test addresses for staking tests."""
    return {
        "safe_address": "0xSafeAddress123456789012345678901234567890123456",
        "staking_token_address": "0xStakingToken123456789012345678901234567890123456",
        "service_address": "0xServiceAddress123456789012345678901234567890123456",
        "agent_address": "0xAgentAddress123456789012345678901234567890123456",
    }


@pytest.fixture
def test_chains():
    """Test chains for staking tests."""
    return {
        "optimism": {
            "chain_id": 10,
            "safe_address": "0xSafeAddress123456789012345678901234567890123456",
            "staking_token_address": "0xStakingToken123456789012345678901234567890123456",
        },
        "mode": {
            "chain_id": 34443,
            "safe_address": "0xSafeAddress456789012345678901234567890123456789",
            "staking_token_address": "0xStakingToken456789012345678901234567890123456789",
        },
    }

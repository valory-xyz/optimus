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

"""Test the states/base.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import json
from unittest.mock import MagicMock

from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    StakingState,
    SynchronizedData,
)


def test_import() -> None:
    """Test that the base states module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.states.base  # noqa


class TestStakingState:
    """Test StakingState enum."""

    def test_unstaked(self) -> None:
        """Test UNSTAKED value."""
        assert StakingState.UNSTAKED.value == 0

    def test_staked(self) -> None:
        """Test STAKED value."""
        assert StakingState.STAKED.value == 1

    def test_evicted(self) -> None:
        """Test EVICTED value."""
        assert StakingState.EVICTED.value == 2


class TestEvent:
    """Test Event enum."""

    def test_done(self) -> None:
        """Test DONE value."""
        assert Event.DONE.value == "done"

    def test_round_timeout(self) -> None:
        """Test ROUND_TIMEOUT value."""
        assert Event.ROUND_TIMEOUT.value == "round_timeout"

    def test_all_events_exist(self) -> None:
        """Test all expected events exist."""
        expected_events = [
            "DONE", "ROUND_TIMEOUT", "NO_MAJORITY", "NONE", "WAIT",
            "SETTLE", "UPDATE", "ERROR", "ACTION_EXECUTED",
            "CHECKPOINT_TX_EXECUTED", "VANITY_TX_EXECUTED",
            "TRANSFER_COMPLETED", "WITHDRAWAL_COMPLETED",
            "WITHDRAWAL_INITIATED", "UNRECOGNIZED",
            "NEXT_CHECKPOINT_NOT_REACHED_YET", "SERVICE_NOT_STAKED",
            "SERVICE_EVICTED", "STAKING_KPI_MET", "STAKING_KPI_NOT_MET",
        ]
        for event_name in expected_events:
            assert hasattr(Event, event_name)


def _make_synced_data(**kwargs) -> SynchronizedData:
    """Create SynchronizedData with given db values."""
    setup_data = {k: [v] for k, v in kwargs.items()}
    db = AbciAppDB(setup_data=setup_data)
    return SynchronizedData(db=db)


class TestSynchronizedData:
    """Test SynchronizedData class."""

    def test_most_voted_tx_hash_none(self) -> None:
        """Test most_voted_tx_hash returns None by default."""
        data = _make_synced_data()
        assert data.most_voted_tx_hash is None

    def test_most_voted_tx_hash_set(self) -> None:
        """Test most_voted_tx_hash returns set value."""
        data = _make_synced_data(most_voted_tx_hash="0xhash")
        assert data.most_voted_tx_hash == "0xhash"

    def test_positions_default(self) -> None:
        """Test positions returns empty list by default."""
        data = _make_synced_data()
        assert data.positions == []

    def test_positions_set(self) -> None:
        """Test positions returns parsed JSON."""
        data = _make_synced_data(positions=json.dumps([{"pool": "test"}]))
        assert data.positions == [{"pool": "test"}]

    def test_positions_none_value(self) -> None:
        """Test positions returns empty list when value is None."""
        data = _make_synced_data()
        # Set None directly
        data.db._data[0]["positions"] = [None]
        assert data.positions == []

    def test_context_default(self) -> None:
        """Test context returns empty list by default."""
        data = _make_synced_data()
        assert data.context == []

    def test_context_set(self) -> None:
        """Test context returns parsed JSON."""
        data = _make_synced_data(context=json.dumps([{"key": "value"}]))
        assert data.context == [{"key": "value"}]

    def test_context_none_value(self) -> None:
        """Test context returns empty list when value is None."""
        data = _make_synced_data()
        data.db._data[0]["context"] = [None]
        assert data.context == []

    def test_selected_protocols_default(self) -> None:
        """Test selected_protocols returns empty list by default."""
        data = _make_synced_data()
        assert data.selected_protocols == []

    def test_selected_protocols_set(self) -> None:
        """Test selected_protocols returns parsed JSON."""
        data = _make_synced_data(selected_protocols=json.dumps(["balancerPool"]))
        assert data.selected_protocols == ["balancerPool"]

    def test_selected_protocols_none_value(self) -> None:
        """Test selected_protocols returns empty list when value is None."""
        data = _make_synced_data()
        data.db._data[0]["selected_protocols"] = [None]
        assert data.selected_protocols == []

    def test_trading_type_default(self) -> None:
        """Test trading_type returns empty string by default."""
        data = _make_synced_data()
        assert data.trading_type == ""

    def test_trading_type_set(self) -> None:
        """Test trading_type returns set value."""
        data = _make_synced_data(trading_type="pool_manager")
        assert data.trading_type == "pool_manager"

    def test_actions_default(self) -> None:
        """Test actions returns empty list by default."""
        data = _make_synced_data()
        assert data.actions == []

    def test_actions_set(self) -> None:
        """Test actions returns parsed JSON."""
        data = _make_synced_data(actions=json.dumps([{"action": "enter"}]))
        assert data.actions == [{"action": "enter"}]

    def test_actions_none_value(self) -> None:
        """Test actions returns empty list when value is None."""
        data = _make_synced_data()
        data.db._data[0]["actions"] = [None]
        assert data.actions == []

    def test_last_executed_action_index_default(self) -> None:
        """Test last_executed_action_index returns None by default."""
        data = _make_synced_data()
        assert data.last_executed_action_index is None

    def test_last_executed_action_index_set(self) -> None:
        """Test last_executed_action_index returns set value."""
        data = _make_synced_data(last_executed_action_index=3)
        assert data.last_executed_action_index == 3

    def test_last_reward_claimed_timestamp_default(self) -> None:
        """Test last_reward_claimed_timestamp returns None by default."""
        data = _make_synced_data()
        assert data.last_reward_claimed_timestamp is None

    def test_last_reward_claimed_timestamp_set(self) -> None:
        """Test last_reward_claimed_timestamp returns set value."""
        data = _make_synced_data(last_reward_claimed_timestamp=1000)
        assert data.last_reward_claimed_timestamp == 1000

    def test_service_staking_state(self) -> None:
        """Test service_staking_state."""
        data = _make_synced_data(service_staking_state=1)
        assert data.service_staking_state == 1

    def test_min_num_of_safe_tx_required(self) -> None:
        """Test min_num_of_safe_tx_required."""
        data = _make_synced_data(min_num_of_safe_tx_required=5)
        assert data.min_num_of_safe_tx_required == 5

    def test_is_staking_kpi_met_default(self) -> None:
        """Test is_staking_kpi_met returns False by default."""
        data = _make_synced_data()
        assert data.is_staking_kpi_met is False

    def test_is_staking_kpi_met_set(self) -> None:
        """Test is_staking_kpi_met returns set value."""
        data = _make_synced_data(is_staking_kpi_met=True)
        assert data.is_staking_kpi_met is True

    def test_chain_id_default(self) -> None:
        """Test chain_id returns None by default."""
        data = _make_synced_data()
        assert data.chain_id is None

    def test_chain_id_set(self) -> None:
        """Test chain_id returns set value."""
        data = _make_synced_data(chain_id="1")
        assert data.chain_id == "1"

    def test_period_number_at_last_cp_default(self) -> None:
        """Test period_number_at_last_cp returns 0 by default."""
        data = _make_synced_data()
        assert data.period_number_at_last_cp == 0

    def test_period_number_at_last_cp_set(self) -> None:
        """Test period_number_at_last_cp returns set value."""
        data = _make_synced_data(period_number_at_last_cp=5)
        assert data.period_number_at_last_cp == 5

    def test_last_executed_route_index_default(self) -> None:
        """Test last_executed_route_index returns None by default."""
        data = _make_synced_data()
        assert data.last_executed_route_index is None

    def test_last_executed_step_index_default(self) -> None:
        """Test last_executed_step_index returns None by default."""
        data = _make_synced_data()
        assert data.last_executed_step_index is None

    def test_routes_retry_attempt_default(self) -> None:
        """Test routes_retry_attempt returns 0 by default."""
        data = _make_synced_data()
        assert data.routes_retry_attempt == 0

    def test_routes_default(self) -> None:
        """Test routes returns empty list by default."""
        data = _make_synced_data()
        assert data.routes == []

    def test_routes_set(self) -> None:
        """Test routes returns parsed JSON."""
        data = _make_synced_data(routes=json.dumps([{"route": "test"}]))
        assert data.routes == [{"route": "test"}]

    def test_routes_none_value(self) -> None:
        """Test routes returns empty list when value is None."""
        data = _make_synced_data()
        data.db._data[0]["routes"] = [None]
        assert data.routes == []

    def test_fee_details_default(self) -> None:
        """Test fee_details returns empty dict by default."""
        data = _make_synced_data()
        assert data.fee_details == {}

    def test_max_allowed_steps_in_a_route_default(self) -> None:
        """Test max_allowed_steps_in_a_route returns None by default."""
        data = _make_synced_data()
        assert data.max_allowed_steps_in_a_route is None

    def test_last_action_default(self) -> None:
        """Test last_action returns None by default."""
        data = _make_synced_data()
        assert data.last_action is None

    def test_last_action_set(self) -> None:
        """Test last_action returns set value."""
        data = _make_synced_data(last_action="ENTER_POOL")
        assert data.last_action == "ENTER_POOL"

    def test_withdrawal_actions_default(self) -> None:
        """Test withdrawal_actions returns empty list by default."""
        data = _make_synced_data()
        assert data.withdrawal_actions == []

    def test_withdrawal_actions_set(self) -> None:
        """Test withdrawal_actions returns parsed JSON."""
        data = _make_synced_data(withdrawal_actions=json.dumps([{"action": "withdraw"}]))
        assert data.withdrawal_actions == [{"action": "withdraw"}]

    def test_withdrawal_actions_none_value(self) -> None:
        """Test withdrawal_actions returns empty list when value is None."""
        data = _make_synced_data()
        data.db._data[0]["withdrawal_actions"] = [None]
        assert data.withdrawal_actions == []

    def test_investing_paused_default(self) -> None:
        """Test investing_paused returns False by default."""
        data = _make_synced_data()
        assert data.investing_paused is False

    def test_investing_paused_set(self) -> None:
        """Test investing_paused returns set value."""
        data = _make_synced_data(investing_paused=True)
        assert data.investing_paused is True

    def test_tx_submitter(self) -> None:
        """Test tx_submitter returns set value."""
        data = _make_synced_data(tx_submitter="test_submitter")
        assert data.tx_submitter == "test_submitter"

    def test_final_tx_hash(self) -> None:
        """Test final_tx_hash returns set value."""
        data = _make_synced_data(final_tx_hash="0xfinalhash")
        assert data.final_tx_hash == "0xfinalhash"

    def _make_collection_data(self, key: str) -> SynchronizedData:
        """Create synced data with a proper serialized collection for deserialization."""
        from packages.valory.skills.liquidity_trader_abci.payloads import GetPositionsPayload
        # Use a real payload to create proper serialization
        payload = GetPositionsPayload(sender="agent_0", positions="[]")
        collection = {"agent_0": payload.json}
        db = AbciAppDB(setup_data={key: [collection]})
        return SynchronizedData(db=db)

    def test_participant_to_tx_round(self) -> None:
        """Test participant_to_tx_round property."""
        data = self._make_collection_data("participant_to_tx_round")
        result = data.participant_to_tx_round
        assert isinstance(result, dict)
        assert "agent_0" in result

    def test_participant_to_positions_round(self) -> None:
        """Test participant_to_positions_round property."""
        data = self._make_collection_data("participant_to_positions_round")
        result = data.participant_to_positions_round
        assert isinstance(result, dict)

    def test_participant_to_strategies_round(self) -> None:
        """Test participant_to_strategies_round property."""
        data = self._make_collection_data("participant_to_strategies_round")
        result = data.participant_to_strategies_round
        assert isinstance(result, dict)

    def test_participant_to_context_round(self) -> None:
        """Test participant_to_context_round property."""
        data = self._make_collection_data("participant_to_context_round")
        result = data.participant_to_context_round
        assert isinstance(result, dict)

    def test_participant_to_actions_round(self) -> None:
        """Test participant_to_actions_round property."""
        data = self._make_collection_data("participant_to_actions_round")
        result = data.participant_to_actions_round
        assert isinstance(result, dict)

    def test_participant_to_checkpoint(self) -> None:
        """Test participant_to_checkpoint property."""
        data = self._make_collection_data("participant_to_checkpoint")
        result = data.participant_to_checkpoint
        assert isinstance(result, dict)

    def test_participant_to_staking_kpi(self) -> None:
        """Test participant_to_staking_kpi property."""
        data = self._make_collection_data("participant_to_staking_kpi")
        result = data.participant_to_staking_kpi
        assert isinstance(result, dict)

    def test_participant_to_decision_making(self) -> None:
        """Test participant_to_decision_making property."""
        data = self._make_collection_data("participant_to_decision_making")
        result = data.participant_to_decision_making
        assert isinstance(result, dict)

    def test_participant_to_post_tx_settlement(self) -> None:
        """Test participant_to_post_tx_settlement property."""
        data = self._make_collection_data("participant_to_post_tx_settlement")
        result = data.participant_to_post_tx_settlement
        assert isinstance(result, dict)

    def test_participant_to_withdraw_funds(self) -> None:
        """Test participant_to_withdraw_funds property."""
        data = self._make_collection_data("participant_to_withdraw_funds")
        result = data.participant_to_withdraw_funds
        assert isinstance(result, dict)

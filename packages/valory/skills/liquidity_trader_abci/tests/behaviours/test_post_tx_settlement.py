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

"""Tests for behaviours/post_tx_settlement.py."""

# pylint: skip-file

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.skills.liquidity_trader_abci.behaviours.post_tx_settlement import (
    PostTxSettlementBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.states.post_tx_settlement import (
    CheckStakingKPIMetRound,
    PostTxSettlementRound,
)


def _make_behaviour():
    """Create a PostTxSettlementBehaviour without __init__."""
    obj = object.__new__(PostTxSettlementBehaviour)
    ctx = MagicMock()
    obj.__dict__["_context"] = ctx
    return obj


def _drive(gen):
    """Drive a generator to completion."""
    val = None
    while True:
        try:
            val = gen.send(val)
        except StopIteration as exc:
            return exc.value


class TestPostTxSettlementBehaviour:
    """Tests for PostTxSettlementBehaviour."""

    def test_matching_round(self) -> None:
        """Test matching_round is PostTxSettlementRound."""
        assert PostTxSettlementBehaviour.matching_round is PostTxSettlementRound

    def test_async_act_not_checkpoint_submitter(self) -> None:
        """Test async_act when tx_submitter is NOT the CheckStakingKPIMetRound."""
        obj = _make_behaviour()
        benchmark_mock = MagicMock()
        obj.context.benchmark_tool.measure.return_value = benchmark_mock
        obj.context.agent_address = "0xagent"
        obj.context.params = MagicMock()

        synced = MagicMock()
        synced.tx_submitter = "some_other_round"
        synced.final_tx_hash = "0xhash123"
        synced.chain_id = "optimism"

        with patch.object(type(obj), "synchronized_data", new_callable=PropertyMock, return_value=synced):
            def fake_fetch_gas():
                yield
                return None

            def fake_send(*args, **kwargs):
                yield

            def fake_wait(*args, **kwargs):
                yield

            obj.fetch_and_log_gas_details = fake_fetch_gas
            obj.send_a2a_transaction = fake_send
            obj.wait_until_round_end = fake_wait
            obj.set_done = MagicMock()

            gen = obj.async_act()
            _drive(gen)

            obj.set_done.assert_called_once()

    def test_async_act_checkpoint_submitter_skips_gas(self) -> None:
        """Test async_act when tx_submitter IS the CheckStakingKPIMetRound (skips gas)."""
        obj = _make_behaviour()
        benchmark_mock = MagicMock()
        obj.context.benchmark_tool.measure.return_value = benchmark_mock
        obj.context.agent_address = "0xagent"

        synced = MagicMock()
        synced.tx_submitter = CheckStakingKPIMetRound.auto_round_id()

        with patch.object(type(obj), "synchronized_data", new_callable=PropertyMock, return_value=synced):
            gas_called = []

            def fake_fetch_gas():
                gas_called.append(True)
                yield

            def fake_send(*args, **kwargs):
                yield

            def fake_wait(*args, **kwargs):
                yield

            obj.fetch_and_log_gas_details = fake_fetch_gas
            obj.send_a2a_transaction = fake_send
            obj.wait_until_round_end = fake_wait
            obj.set_done = MagicMock()

            gen = obj.async_act()
            _drive(gen)

            assert len(gas_called) == 0
            obj.set_done.assert_called_once()


class TestFetchAndLogGasDetails:
    """Tests for fetch_and_log_gas_details."""

    def _setup(self):
        obj = _make_behaviour()
        synced = MagicMock()
        synced.final_tx_hash = "0xhash"
        synced.chain_id = "optimism"

        params_mock = MagicMock()
        params_mock.chain_to_chain_id_mapping = {"optimism": "10"}

        rs = MagicMock()
        rs.last_round_transition_timestamp.timestamp.return_value = 1700000000.0

        obj.gas_cost_tracker = MagicMock()
        obj.store_gas_costs = MagicMock()

        return obj, synced, params_mock, rs

    def test_response_none(self) -> None:
        """Test fetch_and_log_gas_details when response is None."""
        obj, synced, params_mock, rs = self._setup()

        with patch.object(type(obj), "synchronized_data", new_callable=PropertyMock, return_value=synced), \
             patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock), \
             patch.object(type(obj), "round_sequence", new_callable=PropertyMock, return_value=rs):

            def fake_get_receipt(**kwargs):
                yield
                return None

            obj.get_transaction_receipt = fake_get_receipt
            gen = obj.fetch_and_log_gas_details()
            _drive(gen)
            obj.context.logger.error.assert_called()

    def test_response_missing_gas_fields(self) -> None:
        """Test fetch_and_log_gas_details when gas fields are missing."""
        obj, synced, params_mock, rs = self._setup()

        with patch.object(type(obj), "synchronized_data", new_callable=PropertyMock, return_value=synced), \
             patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock), \
             patch.object(type(obj), "round_sequence", new_callable=PropertyMock, return_value=rs):

            def fake_get_receipt(**kwargs):
                yield
                return {"effectiveGasPrice": None, "gasUsed": None}

            obj.get_transaction_receipt = fake_get_receipt
            gen = obj.fetch_and_log_gas_details()
            _drive(gen)
            obj.context.logger.warning.assert_called()

    def test_response_no_chain_id(self) -> None:
        """Test fetch_and_log_gas_details when chain_id mapping is missing."""
        obj, synced, params_mock, rs = self._setup()
        params_mock.chain_to_chain_id_mapping = {}

        with patch.object(type(obj), "synchronized_data", new_callable=PropertyMock, return_value=synced), \
             patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock), \
             patch.object(type(obj), "round_sequence", new_callable=PropertyMock, return_value=rs):

            def fake_get_receipt(**kwargs):
                yield
                return {"effectiveGasPrice": 50, "gasUsed": 21000}

            obj.get_transaction_receipt = fake_get_receipt
            gen = obj.fetch_and_log_gas_details()
            _drive(gen)
            obj.context.logger.error.assert_called()

    def test_response_success(self) -> None:
        """Test fetch_and_log_gas_details with full success path."""
        obj, synced, params_mock, rs = self._setup()

        with patch.object(type(obj), "synchronized_data", new_callable=PropertyMock, return_value=synced), \
             patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock), \
             patch.object(type(obj), "round_sequence", new_callable=PropertyMock, return_value=rs):

            def fake_get_receipt(**kwargs):
                yield
                return {"effectiveGasPrice": 50, "gasUsed": 21000}

            obj.get_transaction_receipt = fake_get_receipt
            gen = obj.fetch_and_log_gas_details()
            _drive(gen)
            obj.gas_cost_tracker.log_gas_usage.assert_called_once()
            obj.store_gas_costs.assert_called_once()

    def test_response_has_gas_used_but_no_effective_gas_price(self) -> None:
        """Test when gasUsed present but effectiveGasPrice missing."""
        obj, synced, params_mock, rs = self._setup()

        with patch.object(type(obj), "synchronized_data", new_callable=PropertyMock, return_value=synced), \
             patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock), \
             patch.object(type(obj), "round_sequence", new_callable=PropertyMock, return_value=rs):

            def fake_get_receipt(**kwargs):
                yield
                return {"effectiveGasPrice": None, "gasUsed": 21000}

            obj.get_transaction_receipt = fake_get_receipt
            gen = obj.fetch_and_log_gas_details()
            _drive(gen)
            obj.context.logger.warning.assert_called()

    def test_response_has_effective_gas_price_but_no_gas_used(self) -> None:
        """Test when effectiveGasPrice present but gasUsed missing."""
        obj, synced, params_mock, rs = self._setup()

        with patch.object(type(obj), "synchronized_data", new_callable=PropertyMock, return_value=synced), \
             patch.object(type(obj), "params", new_callable=PropertyMock, return_value=params_mock), \
             patch.object(type(obj), "round_sequence", new_callable=PropertyMock, return_value=rs):

            def fake_get_receipt(**kwargs):
                yield
                return {"effectiveGasPrice": 50, "gasUsed": None}

            obj.get_transaction_receipt = fake_get_receipt
            gen = obj.fetch_and_log_gas_details()
            _drive(gen)
            obj.context.logger.warning.assert_called()

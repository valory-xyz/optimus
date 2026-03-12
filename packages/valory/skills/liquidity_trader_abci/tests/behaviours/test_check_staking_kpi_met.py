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

"""Tests for behaviours/check_staking_kpi_met.py."""

# pylint: skip-file

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met import (
    CheckStakingKPIMetBehaviour,
)


def _make_behaviour():
    """Create a CheckStakingKPIMetBehaviour without __init__."""
    obj = object.__new__(CheckStakingKPIMetBehaviour)
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


class TestCheckStakingKPIMetBehaviour:
    """Tests for CheckStakingKPIMetBehaviour."""

    def _run_async_act(self, obj, params_mock, synced_mock):
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ), patch.object(
            type(obj),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=synced_mock,
        ):
            benchmark_mock = MagicMock()
            obj.context.benchmark_tool.measure.return_value = benchmark_mock
            obj.context.agent_address = "0xagent"

            def fake_send(*args, **kwargs):
                yield

            def fake_wait(*args, **kwargs):
                yield

            obj.send_a2a_transaction = fake_send
            obj.wait_until_round_end = fake_wait
            obj.set_done = MagicMock()

            gen = obj.async_act()
            _drive(gen)

    def test_async_act_kpi_check_error(self) -> None:
        """Test async_act when _is_staking_kpi_met returns None."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        synced_mock = MagicMock()

        def fake_is_kpi_met():
            yield
            return None

        obj._is_staking_kpi_met = fake_is_kpi_met
        self._run_async_act(obj, params_mock, synced_mock)
        obj.context.logger.error.assert_called()
        obj.set_done.assert_called_once()

    def test_async_act_kpi_already_met(self) -> None:
        """Test async_act when KPI is already met (True)."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        synced_mock = MagicMock()

        def fake_is_kpi_met():
            yield
            return True

        obj._is_staking_kpi_met = fake_is_kpi_met
        self._run_async_act(obj, params_mock, synced_mock)
        obj.set_done.assert_called_once()

    def test_async_act_kpi_not_met_threshold_not_exceeded(self) -> None:
        """Test async_act when KPI is not met and threshold not exceeded."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        params_mock.staking_threshold_period = 10

        synced_mock = MagicMock()
        synced_mock.period_count = 5
        synced_mock.period_number_at_last_cp = 0

        def fake_is_kpi_met():
            yield
            return False

        obj._is_staking_kpi_met = fake_is_kpi_met
        self._run_async_act(obj, params_mock, synced_mock)
        obj.set_done.assert_called_once()

    def test_async_act_kpi_not_met_vanity_tx(self) -> None:
        """Test async_act when KPI not met, threshold exceeded, vanity tx needed."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        params_mock.staking_threshold_period = 10

        synced_mock = MagicMock()
        synced_mock.period_count = 20
        synced_mock.period_number_at_last_cp = 0
        synced_mock.min_num_of_safe_tx_required = 5

        def fake_is_kpi_met():
            yield
            return False

        def fake_get_nonces(chain, multisig):
            yield
            return 2

        def fake_prepare_vanity_tx(chain):
            yield
            return "0xvanity_hash"

        obj._is_staking_kpi_met = fake_is_kpi_met
        obj._get_multisig_nonces_since_last_cp = fake_get_nonces
        obj._prepare_vanity_tx = fake_prepare_vanity_tx
        self._run_async_act(obj, params_mock, synced_mock)
        obj.set_done.assert_called_once()

    def test_async_act_kpi_not_met_nonces_none(self) -> None:
        """Test async_act when nonces or min_num is None."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        params_mock.staking_threshold_period = 10

        synced_mock = MagicMock()
        synced_mock.period_count = 20
        synced_mock.period_number_at_last_cp = 0
        synced_mock.min_num_of_safe_tx_required = None

        def fake_is_kpi_met():
            yield
            return False

        def fake_get_nonces(chain, multisig):
            yield
            return None

        obj._is_staking_kpi_met = fake_is_kpi_met
        obj._get_multisig_nonces_since_last_cp = fake_get_nonces
        self._run_async_act(obj, params_mock, synced_mock)
        obj.set_done.assert_called_once()

    def test_async_act_kpi_not_met_no_tx_needed(self) -> None:
        """Test async_act when tx left is 0."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        params_mock.staking_threshold_period = 10

        synced_mock = MagicMock()
        synced_mock.period_count = 20
        synced_mock.period_number_at_last_cp = 0
        synced_mock.min_num_of_safe_tx_required = 5

        def fake_is_kpi_met():
            yield
            return False

        def fake_get_nonces(chain, multisig):
            yield
            return 10  # 5 - 10 = -5

        obj._is_staking_kpi_met = fake_is_kpi_met
        obj._get_multisig_nonces_since_last_cp = fake_get_nonces
        self._run_async_act(obj, params_mock, synced_mock)
        obj.set_done.assert_called_once()


class TestPrepareVanityTx:
    """Tests for _prepare_vanity_tx."""

    def test_success(self) -> None:
        """Test _prepare_vanity_tx success path."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_contract_interact(**kwargs):
            yield
            return "0x" + "ab" * 32

        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._prepare_vanity_tx("optimism")
            result = _drive(gen)
            assert result is not None

    def test_contract_interact_returns_none(self) -> None:
        """Test _prepare_vanity_tx when contract_interact returns None."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_contract_interact(**kwargs):
            yield
            return None

        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._prepare_vanity_tx("optimism")
            result = _drive(gen)
            assert result is None

    def test_contract_interact_exception(self) -> None:
        """Test _prepare_vanity_tx when contract_interact raises."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_contract_interact(**kwargs):
            raise ValueError("boom")
            yield  # noqa: unreachable

        obj.contract_interact = fake_contract_interact

        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            gen = obj._prepare_vanity_tx("optimism")
            result = _drive(gen)
            assert result is None

    def test_hash_payload_exception(self) -> None:
        """Test _prepare_vanity_tx when hash_payload_to_hex raises."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_contract_interact(**kwargs):
            yield
            return "0x" + "ab" * 32

        obj.contract_interact = fake_contract_interact

        import packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met as mod

        original_hash = mod.hash_payload_to_hex

        def bad_hash(**kwargs):
            raise ValueError("bad hash")

        mod.hash_payload_to_hex = bad_hash
        try:
            with patch.object(
                type(obj), "params", new_callable=PropertyMock, return_value=params_mock
            ):
                gen = obj._prepare_vanity_tx("optimism")
                result = _drive(gen)
                assert result is None
        finally:
            mod.hash_payload_to_hex = original_hash

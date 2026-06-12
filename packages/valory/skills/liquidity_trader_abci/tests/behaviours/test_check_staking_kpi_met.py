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

from packages.valory.skills.funds_manager.behaviours import (
    GET_FUNDS_STATUS_METHOD_NAME,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.base import ZERO_ADDRESS
from packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met import (
    CheckStakingKPIMetBehaviour,
)


_AGENT_EOA = "0xagent"
_OPTIMISM_CHAIN_ID = 10


def _funds_status_hook_returning(response_body):
    """Build a fake shared_state hook that returns a fund-status response body."""

    class _FakeFundRequirements:
        def get_response_body(self):
            return response_body

    return lambda: _FakeFundRequirements()


def _gas_records(*costs_wei):
    """Build gas-cost tracker records where each entry has product == cost_wei."""
    return [
        {"gas_used": cost, "gas_price": 1, "timestamp": idx, "tx_hash": f"0x{idx:x}"}
        for idx, cost in enumerate(costs_wei)
    ]


def _balance_response(chain, balance_wei):
    """Build a flattened funds-status response with the EOA balance set."""
    return {
        chain: {
            _AGENT_EOA: {
                ZERO_ADDRESS: {
                    "balance": str(balance_wei),
                    "deficit": "0",
                    "decimals": 18,
                }
            }
        }
    }


def _gen_return_false(*args, **kwargs):
    """Generator function that yields once and returns False."""
    yield
    return False


def _make_behaviour():
    """Create a CheckStakingKPIMetBehaviour without __init__.

    Mirrors production state where the base behaviour __init__ always
    sets ``gas_cost_tracker``. Default is an empty tracker so the
    vanity-tx funding gate falls open (signal unknown).
    """
    obj = object.__new__(CheckStakingKPIMetBehaviour)
    ctx = MagicMock()
    obj.__dict__["_context"] = ctx
    obj._read_investing_paused = _gen_return_false
    obj.gas_cost_tracker = MagicMock()
    obj.gas_cost_tracker.data = {}
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
        with (
            patch.object(
                type(obj), "params", new_callable=PropertyMock, return_value=params_mock
            ),
            patch.object(
                type(obj),
                "synchronized_data",
                new_callable=PropertyMock,
                return_value=synced_mock,
            ),
        ):
            benchmark_mock = MagicMock()
            obj.context.benchmark_tool.measure.return_value = benchmark_mock
            obj.context.agent_address = "0xagent"

            def fake_send(*args, **kwargs):
                """Fake send."""
                yield

            def fake_wait(*args, **kwargs):
                """Fake wait."""
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
            """Fake is kpi met."""
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
            """Fake is kpi met."""
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
            """Fake is kpi met."""
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
            """Fake is kpi met."""
            yield
            return False

        def fake_get_nonces(chain, multisig):
            """Fake get nonces."""
            yield
            return 2

        def fake_prepare_vanity_tx(chain):
            """Fake prepare vanity tx."""
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
            """Fake is kpi met."""
            yield
            return False

        def fake_get_nonces(chain, multisig):
            """Fake get nonces."""
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
            """Fake is kpi met."""
            yield
            return False

        def fake_get_nonces(chain, multisig):
            """Fake get nonces."""
            yield
            return 10  # 5 - 10 = -5

        obj._is_staking_kpi_met = fake_is_kpi_met
        obj._get_multisig_nonces_since_last_cp = fake_get_nonces
        self._run_async_act(obj, params_mock, synced_mock)
        obj.set_done.assert_called_once()


class TestCheckStakingKPIMetWithdrawalGate:
    """Verify the gate at the top of async_act emits a withdrawal payload."""

    def test_gate_emits_withdrawal_payload_when_paused(self) -> None:
        """investing_paused=True short-circuits to a WITHDRAWAL_INITIATED payload."""
        obj = _make_behaviour()
        obj.context.benchmark_tool.measure.return_value = MagicMock()
        obj.context.agent_address = "0xagent"

        captured = {}

        def fake_read_investing_paused():
            """Fake read investing paused."""
            yield
            return True

        def fake_send(payload):
            """Fake send."""
            captured["payload"] = payload
            yield

        def fake_wait():
            """Fake wait."""
            yield

        obj._is_staking_kpi_met = MagicMock(
            side_effect=AssertionError(
                "KPI lookup must not run when investing is paused"
            )
        )
        obj._read_investing_paused = fake_read_investing_paused
        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()

        _drive(obj.async_act())

        assert captured["payload"].event == "withdrawal_initiated"
        assert captured["payload"].tx_hash is None
        obj.set_done.assert_called_once()

    def test_gate_falls_through_when_not_paused(self) -> None:
        """investing_paused=False lets the normal KPI path emit a non-withdrawal payload."""
        obj = _make_behaviour()
        obj.context.benchmark_tool.measure.return_value = MagicMock()
        obj.context.agent_address = "0xagent"

        captured = {}

        def fake_read_investing_paused():
            """Fake read investing paused."""
            yield
            return False

        def fake_is_kpi_met():
            """Fake is kpi met."""
            yield
            return True

        def fake_send(payload):
            """Fake send."""
            captured["payload"] = payload
            yield

        def fake_wait():
            """Fake wait."""
            yield

        obj._read_investing_paused = fake_read_investing_paused
        obj._is_staking_kpi_met = fake_is_kpi_met
        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()

        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        with patch.object(
            type(obj), "params", new_callable=PropertyMock, return_value=params_mock
        ):
            _drive(obj.async_act())

        assert captured["payload"].event is None
        assert captured["payload"].tx_hash is None
        obj.set_done.assert_called_once()


class TestPrepareVanityTx:
    """Tests for _prepare_vanity_tx."""

    def test_success(self) -> None:
        """Test _prepare_vanity_tx success path."""
        obj = _make_behaviour()
        params_mock = MagicMock()
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}

        def fake_contract_interact(**kwargs):
            """Fake contract interact."""
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
            """Fake contract interact."""
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
            """Fake contract interact."""
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
            """Fake contract interact."""
            yield
            return "0x" + "ab" * 32

        obj.contract_interact = fake_contract_interact

        import packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met as mod

        original_hash = mod.hash_payload_to_hex

        def bad_hash(**kwargs):
            """Bad hash."""
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


class TestVanityTxFundingGate:
    """Verify the funding gate at the vanity-tx decision point.

    These exercise async_act end-to-end with the KPI-not-met,
    threshold-exceeded, vanity-tx-needed shape (same setup as
    test_async_act_kpi_not_met_vanity_tx) and assert whether the
    suppression branch fires.
    """

    @staticmethod
    def _base_obj_with_kpi_unmet():
        """Build a behaviour mid-flow with the prerequisites for vanity tx in place."""
        obj = _make_behaviour()
        obj.context.agent_address = _AGENT_EOA

        def fake_is_kpi_met():
            yield
            return False

        def fake_get_nonces(chain, multisig):
            yield
            return 2

        obj._is_staking_kpi_met = fake_is_kpi_met
        obj._get_multisig_nonces_since_last_cp = fake_get_nonces
        obj.gas_cost_tracker = MagicMock()
        obj.gas_cost_tracker.data = {}
        return obj

    @staticmethod
    def _params_with_optimism_mapping():
        params_mock = MagicMock()
        params_mock.staking_chain = "optimism"
        params_mock.safe_contract_addresses = {"optimism": "0xsafe"}
        params_mock.staking_threshold_period = 10
        params_mock.chain_to_chain_id_mapping = {"optimism": _OPTIMISM_CHAIN_ID}
        return params_mock

    @staticmethod
    def _synced_with_kpi_unmet():
        synced_mock = MagicMock()
        synced_mock.period_count = 20
        synced_mock.period_number_at_last_cp = 0
        synced_mock.min_num_of_safe_tx_required = 5
        return synced_mock

    def test_vanity_tx_suppressed_when_balance_below_recent_real_tx_cost(
        self,
    ) -> None:
        """David's edge case: EOA below real-tx cost but vanity tx still cheap.

        Balance ~6.92e13 wei, real-tx cost ~1.55e14 wei from his Pearl
        logs. Guard must skip _prepare_vanity_tx and emit a WARNING.
        """
        obj = self._base_obj_with_kpi_unmet()
        obj.gas_cost_tracker.data = {
            str(_OPTIMISM_CHAIN_ID): _gas_records(
                155_000_000_000_000,
                155_000_000_000_000,
                155_000_000_000_000,
            ),
        }
        obj.context.shared_state = {
            GET_FUNDS_STATUS_METHOD_NAME: _funds_status_hook_returning(
                _balance_response("optimism", 69_207_718_314_629)
            ),
        }

        vanity_called = {"flag": False}

        def fake_prepare_vanity_tx(chain):
            vanity_called["flag"] = True
            yield
            return "0xvanity"

        obj._prepare_vanity_tx = fake_prepare_vanity_tx

        TestCheckStakingKPIMetBehaviour()._run_async_act(
            obj, self._params_with_optimism_mapping(), self._synced_with_kpi_unmet()
        )

        assert vanity_called["flag"] is False
        obj.context.logger.warning.assert_called()

    def test_vanity_tx_runs_when_balance_above_recent_real_tx_cost(self) -> None:
        """Healthy EOA: gate is silent and the existing vanity path runs."""
        obj = self._base_obj_with_kpi_unmet()
        obj.gas_cost_tracker.data = {
            str(_OPTIMISM_CHAIN_ID): _gas_records(
                100_000_000_000_000,
                100_000_000_000_000,
            ),
        }
        obj.context.shared_state = {
            GET_FUNDS_STATUS_METHOD_NAME: _funds_status_hook_returning(
                _balance_response("optimism", 500_000_000_000_000)
            ),
        }

        vanity_called = {"flag": False}

        def fake_prepare_vanity_tx(chain):
            vanity_called["flag"] = True
            yield
            return "0xvanity"

        obj._prepare_vanity_tx = fake_prepare_vanity_tx

        TestCheckStakingKPIMetBehaviour()._run_async_act(
            obj, self._params_with_optimism_mapping(), self._synced_with_kpi_unmet()
        )

        assert vanity_called["flag"] is True

    def test_vanity_tx_runs_when_no_gas_records_yet(self) -> None:
        """Fresh boot, no real-tx history → cost signal unknown → fail open."""
        obj = self._base_obj_with_kpi_unmet()
        obj.gas_cost_tracker.data = {}
        obj.context.shared_state = {
            GET_FUNDS_STATUS_METHOD_NAME: _funds_status_hook_returning(
                _balance_response("optimism", 1)
            ),
        }

        vanity_called = {"flag": False}

        def fake_prepare_vanity_tx(chain):
            vanity_called["flag"] = True
            yield
            return "0xvanity"

        obj._prepare_vanity_tx = fake_prepare_vanity_tx

        TestCheckStakingKPIMetBehaviour()._run_async_act(
            obj, self._params_with_optimism_mapping(), self._synced_with_kpi_unmet()
        )

        assert vanity_called["flag"] is True

    def test_vanity_tx_runs_when_funds_status_hook_missing(self) -> None:
        """funds_manager not loaded / hook absent → balance unknown → fail open."""
        obj = self._base_obj_with_kpi_unmet()
        obj.gas_cost_tracker.data = {
            str(_OPTIMISM_CHAIN_ID): _gas_records(155_000_000_000_000),
        }
        obj.context.shared_state = {}

        vanity_called = {"flag": False}

        def fake_prepare_vanity_tx(chain):
            vanity_called["flag"] = True
            yield
            return "0xvanity"

        obj._prepare_vanity_tx = fake_prepare_vanity_tx

        TestCheckStakingKPIMetBehaviour()._run_async_act(
            obj, self._params_with_optimism_mapping(), self._synced_with_kpi_unmet()
        )

        assert vanity_called["flag"] is True

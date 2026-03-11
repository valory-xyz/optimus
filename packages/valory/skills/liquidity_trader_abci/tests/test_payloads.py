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

"""Test the payloads.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from packages.valory.skills.liquidity_trader_abci.payloads import (
    APRPopulationPayload,
    CallCheckpointPayload,
    CheckStakingKPIMetPayload,
    DecisionMakingPayload,
    EvaluateStrategyPayload,
    FetchStrategiesPayload,
    GetPositionsPayload,
    MultisigTxPayload,
    PostTxSettlementPayload,
    WithdrawFundsPayload,
)


SENDER = "sender_address"


def test_import() -> None:
    """Test that the payloads module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.payloads  # noqa


class TestMultisigTxPayload:
    """Test MultisigTxPayload."""

    def test_creation(self) -> None:
        """Test MultisigTxPayload creation."""
        payload = MultisigTxPayload(
            sender=SENDER,
            tx_submitter="submitter",
            tx_hash="0xhash",
            safe_contract_address="0xsafe",
            chain_id="1",
        )
        assert payload.tx_submitter == "submitter"
        assert payload.tx_hash == "0xhash"
        assert payload.safe_contract_address == "0xsafe"
        assert payload.chain_id == "1"

    def test_creation_with_none_values(self) -> None:
        """Test MultisigTxPayload with None values."""
        payload = MultisigTxPayload(
            sender=SENDER,
            tx_submitter="submitter",
            tx_hash=None,
            safe_contract_address=None,
            chain_id=None,
        )
        assert payload.tx_hash is None
        assert payload.safe_contract_address is None
        assert payload.chain_id is None


class TestCallCheckpointPayload:
    """Test CallCheckpointPayload."""

    def test_creation(self) -> None:
        """Test CallCheckpointPayload creation."""
        payload = CallCheckpointPayload(
            sender=SENDER,
            tx_submitter="submitter",
            tx_hash="0xhash",
            safe_contract_address="0xsafe",
            chain_id="1",
            service_staking_state=1,
            min_num_of_safe_tx_required=5,
        )
        assert payload.service_staking_state == 1
        assert payload.min_num_of_safe_tx_required == 5

    def test_creation_with_none(self) -> None:
        """Test CallCheckpointPayload with None min_num_of_safe_tx_required."""
        payload = CallCheckpointPayload(
            sender=SENDER,
            tx_submitter="submitter",
            tx_hash=None,
            safe_contract_address=None,
            chain_id=None,
            service_staking_state=0,
            min_num_of_safe_tx_required=None,
        )
        assert payload.min_num_of_safe_tx_required is None


class TestCheckStakingKPIMetPayload:
    """Test CheckStakingKPIMetPayload."""

    def test_creation(self) -> None:
        """Test CheckStakingKPIMetPayload creation."""
        payload = CheckStakingKPIMetPayload(
            sender=SENDER,
            tx_submitter="submitter",
            tx_hash="0xhash",
            safe_contract_address="0xsafe",
            chain_id="1",
            is_staking_kpi_met=True,
        )
        assert payload.is_staking_kpi_met is True

    def test_creation_with_none(self) -> None:
        """Test CheckStakingKPIMetPayload with None."""
        payload = CheckStakingKPIMetPayload(
            sender=SENDER,
            tx_submitter="submitter",
            tx_hash=None,
            safe_contract_address=None,
            chain_id=None,
            is_staking_kpi_met=None,
        )
        assert payload.is_staking_kpi_met is None


class TestGetPositionsPayload:
    """Test GetPositionsPayload."""

    def test_creation(self) -> None:
        """Test GetPositionsPayload creation."""
        payload = GetPositionsPayload(
            sender=SENDER,
            positions='[{"pool": "test"}]',
        )
        assert payload.positions == '[{"pool": "test"}]'

    def test_creation_with_none(self) -> None:
        """Test GetPositionsPayload with None."""
        payload = GetPositionsPayload(sender=SENDER, positions=None)
        assert payload.positions is None


class TestAPRPopulationPayload:
    """Test APRPopulationPayload."""

    def test_creation(self) -> None:
        """Test APRPopulationPayload creation."""
        payload = APRPopulationPayload(
            sender=SENDER,
            context="context_data",
            content="content_data",
        )
        assert payload.context == "context_data"
        assert payload.content == "content_data"

    def test_creation_default_content(self) -> None:
        """Test APRPopulationPayload with default content."""
        payload = APRPopulationPayload(
            sender=SENDER,
            context="context_data",
        )
        assert payload.content is None


class TestEvaluateStrategyPayload:
    """Test EvaluateStrategyPayload."""

    def test_creation(self) -> None:
        """Test EvaluateStrategyPayload creation."""
        payload = EvaluateStrategyPayload(
            sender=SENDER,
            actions='[{"action": "enter"}]',
        )
        assert payload.actions == '[{"action": "enter"}]'

    def test_creation_with_none(self) -> None:
        """Test EvaluateStrategyPayload with None."""
        payload = EvaluateStrategyPayload(sender=SENDER, actions=None)
        assert payload.actions is None


class TestDecisionMakingPayload:
    """Test DecisionMakingPayload."""

    def test_creation(self) -> None:
        """Test DecisionMakingPayload creation."""
        payload = DecisionMakingPayload(
            sender=SENDER,
            content='{"event": "done"}',
        )
        assert payload.content == '{"event": "done"}'


class TestPostTxSettlementPayload:
    """Test PostTxSettlementPayload."""

    def test_creation(self) -> None:
        """Test PostTxSettlementPayload creation."""
        payload = PostTxSettlementPayload(
            sender=SENDER,
            content='{"event": "action_executed"}',
        )
        assert payload.content == '{"event": "action_executed"}'


class TestFetchStrategiesPayload:
    """Test FetchStrategiesPayload."""

    def test_creation(self) -> None:
        """Test FetchStrategiesPayload creation."""
        payload = FetchStrategiesPayload(
            sender=SENDER,
            content='{"selected_protocols": ["balancerPool"]}',
        )
        assert payload.content == '{"selected_protocols": ["balancerPool"]}'


class TestWithdrawFundsPayload:
    """Test WithdrawFundsPayload."""

    def test_creation(self) -> None:
        """Test WithdrawFundsPayload creation."""
        payload = WithdrawFundsPayload(
            sender=SENDER,
            withdrawal_actions='[{"action": "withdraw"}]',
        )
        assert payload.withdrawal_actions == '[{"action": "withdraw"}]'

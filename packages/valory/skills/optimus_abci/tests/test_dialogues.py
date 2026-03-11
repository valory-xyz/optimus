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

"""Test the dialogues.py module of the optimus_abci skill."""

# pylint: skip-file

from unittest.mock import MagicMock

from packages.dvilela.protocols.kv_store.dialogues import (
    KvStoreDialogue as BaseKvStoreDialogue,
)
from packages.dvilela.protocols.kv_store.dialogues import (
    KvStoreDialogues as BaseKvStoreDialogues,
)
from packages.valory.protocols.srr.dialogues import SrrDialogue as BaseSrrDialogue
from packages.valory.protocols.srr.dialogues import SrrDialogues as BaseSrrDialogues
from packages.valory.skills.abstract_round_abci.dialogues import (
    AbciDialogue as BaseAbciDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    AbciDialogues as BaseAbciDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    ContractApiDialogue as BaseContractApiDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    ContractApiDialogues as BaseContractApiDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    HttpDialogue as BaseHttpDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    HttpDialogues as BaseHttpDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    IpfsDialogue as BaseIpfsDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    IpfsDialogues as BaseIpfsDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    LedgerApiDialogue as BaseLedgerApiDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    LedgerApiDialogues as BaseLedgerApiDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    SigningDialogue as BaseSigningDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    SigningDialogues as BaseSigningDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    TendermintDialogue as BaseTendermintDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    TendermintDialogues as BaseTendermintDialogues,
)
from packages.valory.skills.optimus_abci.dialogues import (
    AbciDialogue,
    AbciDialogues,
    ContractApiDialogue,
    ContractApiDialogues,
    HttpDialogue,
    HttpDialogues,
    IpfsDialogue,
    IpfsDialogues,
    KvStoreDialogue,
    KvStoreDialogues,
    LedgerApiDialogue,
    LedgerApiDialogues,
    SigningDialogue,
    SigningDialogues,
    SrrDialogue,
    SrrDialogues,
    TendermintDialogue,
    TendermintDialogues,
)


def test_import() -> None:
    """Test that the dialogues module can be imported."""
    import packages.valory.skills.optimus_abci.dialogues  # noqa


def test_abci_dialogue_alias() -> None:
    """Test AbciDialogue is an alias."""
    assert AbciDialogue is BaseAbciDialogue


def test_abci_dialogues_alias() -> None:
    """Test AbciDialogues is an alias."""
    assert AbciDialogues is BaseAbciDialogues


def test_signing_dialogue_alias() -> None:
    """Test SigningDialogue is an alias."""
    assert SigningDialogue is BaseSigningDialogue


def test_signing_dialogues_alias() -> None:
    """Test SigningDialogues is an alias."""
    assert SigningDialogues is BaseSigningDialogues


def test_ledger_api_dialogue_alias() -> None:
    """Test LedgerApiDialogue is an alias."""
    assert LedgerApiDialogue is BaseLedgerApiDialogue


def test_ledger_api_dialogues_alias() -> None:
    """Test LedgerApiDialogues is an alias."""
    assert LedgerApiDialogues is BaseLedgerApiDialogues


def test_contract_api_dialogue_alias() -> None:
    """Test ContractApiDialogue is an alias."""
    assert ContractApiDialogue is BaseContractApiDialogue


def test_contract_api_dialogues_alias() -> None:
    """Test ContractApiDialogues is an alias."""
    assert ContractApiDialogues is BaseContractApiDialogues


def test_tendermint_dialogue_alias() -> None:
    """Test TendermintDialogue is an alias."""
    assert TendermintDialogue is BaseTendermintDialogue


def test_tendermint_dialogues_alias() -> None:
    """Test TendermintDialogues is an alias."""
    assert TendermintDialogues is BaseTendermintDialogues


def test_ipfs_dialogue_alias() -> None:
    """Test IpfsDialogue is an alias."""
    assert IpfsDialogue is BaseIpfsDialogue


def test_ipfs_dialogues_alias() -> None:
    """Test IpfsDialogues is an alias."""
    assert IpfsDialogues is BaseIpfsDialogues


def test_srr_dialogue_alias() -> None:
    """Test SrrDialogue is an alias."""
    assert SrrDialogue is BaseSrrDialogue


def test_http_dialogue_alias() -> None:
    """Test HttpDialogue is an alias."""
    assert HttpDialogue is BaseHttpDialogue


def test_http_dialogues_alias() -> None:
    """Test HttpDialogues is an alias."""
    assert HttpDialogues is BaseHttpDialogues


def test_kv_store_dialogue_alias() -> None:
    """Test KvStoreDialogue is an alias."""
    assert KvStoreDialogue is BaseKvStoreDialogue


class TestSrrDialogues:
    """Test SrrDialogues class."""

    def test_initialization(self) -> None:
        """Test SrrDialogues initialization."""
        mock_context = MagicMock()
        mock_context.skill_id = "test_skill/test:0.1.0"
        dialogues = SrrDialogues(
            name="srr_dialogues", skill_context=mock_context
        )
        assert dialogues is not None

    def test_role_from_first_message(self) -> None:
        """Test SrrDialogues role_from_first_message returns SKILL."""
        mock_context = MagicMock()
        mock_context.skill_id = "test_skill/test:0.1.0"
        dialogues = SrrDialogues(
            name="srr_dialogues", skill_context=mock_context
        )
        assert isinstance(dialogues, SrrDialogues)
        assert isinstance(dialogues, BaseSrrDialogues)
        # Invoke the role_from_first_message function to cover line 129
        role = dialogues._role_from_first_message(MagicMock(), "receiver_addr")
        from packages.valory.protocols.srr.dialogues import SrrDialogue as BaseSrrDialogue
        assert role == BaseSrrDialogue.Role.SKILL


class TestKvStoreDialogues:
    """Test KvStoreDialogues class."""

    def test_initialization(self) -> None:
        """Test KvStoreDialogues initialization."""
        mock_context = MagicMock()
        mock_context.skill_id = "test_skill/test:0.1.0"
        dialogues = KvStoreDialogues(
            name="kv_store_dialogues", skill_context=mock_context
        )
        assert dialogues is not None

    def test_role_from_first_message(self) -> None:
        """Test KvStoreDialogues role_from_first_message returns SKILL."""
        mock_context = MagicMock()
        mock_context.skill_id = "test_skill/test:0.1.0"
        dialogues = KvStoreDialogues(
            name="kv_store_dialogues", skill_context=mock_context
        )
        assert isinstance(dialogues, KvStoreDialogues)
        assert isinstance(dialogues, BaseKvStoreDialogues)
        # Invoke the role_from_first_message function to cover line 158
        role = dialogues._role_from_first_message(MagicMock(), "receiver_addr")
        from packages.dvilela.protocols.kv_store.dialogues import (
            KvStoreDialogue as BaseKvStoreDialogue,
        )
        assert role == BaseKvStoreDialogue.Role.SKILL

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
from packages.valory.skills.optimus_abci.dialogues import (
    KvStoreDialogues,
    SrrDialogues,
)


def test_import() -> None:
    """Test that the dialogues module can be imported."""
    import packages.valory.skills.optimus_abci.dialogues  # noqa


class TestSrrDialogues:
    """Test SrrDialogues class."""

    def test_role_from_first_message(self) -> None:
        """Test SrrDialogues role_from_first_message returns SKILL."""
        mock_context = MagicMock()
        mock_context.skill_id = "test_skill/test:0.1.0"
        dialogues = SrrDialogues(name="srr_dialogues", skill_context=mock_context)
        role = dialogues._role_from_first_message(MagicMock(), "receiver_addr")
        assert role == BaseSrrDialogue.Role.SKILL


class TestKvStoreDialogues:
    """Test KvStoreDialogues class."""

    def test_role_from_first_message(self) -> None:
        """Test KvStoreDialogues role_from_first_message returns SKILL."""
        mock_context = MagicMock()
        mock_context.skill_id = "test_skill/test:0.1.0"
        dialogues = KvStoreDialogues(
            name="kv_store_dialogues", skill_context=mock_context
        )
        role = dialogues._role_from_first_message(MagicMock(), "receiver_addr")
        assert role == BaseKvStoreDialogue.Role.SKILL

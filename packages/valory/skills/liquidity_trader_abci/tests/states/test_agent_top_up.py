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

"""Test the states/agent_top_up.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.skills.abstract_round_abci.base import (
    CollectSameUntilThresholdRound,
)
from packages.valory.skills.liquidity_trader_abci.states.agent_top_up import (
    AgentTopUpRound,
)
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)


def _end_block(super_result: Any) -> Optional[Any]:
    """Run AgentTopUpRound.end_block over a stubbed parent result."""
    round_obj = object.__new__(AgentTopUpRound)
    with patch.object(
        CollectSameUntilThresholdRound, "end_block", return_value=super_result
    ):
        return round_obj.end_block()


def _synced(tx_hash: Optional[str]) -> MagicMock:
    """Synchronized data carrying the voted tx hash."""
    synced = MagicMock(spec=SynchronizedData)
    synced.most_voted_tx_hash = tx_hash
    return synced


class TestAgentTopUpRound:
    """Test AgentTopUpRound."""

    def test_still_collecting(self) -> None:
        """No result while the votes are still coming in."""
        assert _end_block(None) is None

    def test_a_voted_tx_settles(self) -> None:
        """A voted top-up tx goes to settlement."""
        synced = _synced("0xhash")
        assert _end_block((synced, Event.DONE)) == (synced, Event.SETTLE)

    def test_no_tx_is_done(self) -> None:
        """Nothing to move continues the period."""
        synced = _synced(None)
        assert _end_block((synced, Event.DONE)) == (synced, Event.DONE)

    @pytest.mark.parametrize("event", [Event.NO_MAJORITY, Event.ROUND_TIMEOUT])
    def test_other_events_pass_through(self, event: Event) -> None:
        """Events other than DONE are returned as the parent emits them."""
        synced = _synced("0xhash")
        assert _end_block((synced, event)) == (synced, event)

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

"""Test the chat.py module of the optimus_abci skill."""

# pylint: skip-file

from unittest.mock import patch

import pytest

from packages.valory.skills.optimus_abci import chat
from packages.valory.skills.optimus_abci.chat import (
    ANSWERING_TIMEOUT_SECONDS,
    ChatSession,
    ConfigChange,
    ConfigProposal,
    changes_text,
    describe_feed_entry,
)


def _proposal() -> ConfigProposal:
    return ConfigProposal(
        changes=[ConfigChange("Trading type", "Balanced", "Risky")],
        strategy_changed=True,
        selected_protocols=["balancer_pools_search"],
        trading_type="risky",
        max_loss_percentage=20.0,
        composite_score=0.3,
        activity_goal=None,
        reasoning="ok",
    )


class TestChatSession:
    """What the chat keeps in memory."""

    def test_answering_until_finished(self) -> None:
        """A message is answering from start to finish."""
        session = ChatSession()
        assert not session.is_answering("c1")
        session.start_answering("c1")
        assert session.is_answering("c1")
        assert not session.is_answering("c2")
        session.finish_answering("c1")
        assert not session.is_answering("c1")

    def test_finishing_an_unknown_message_is_harmless(self) -> None:
        """Finishing a message never started does nothing."""
        ChatSession().finish_answering("c1")

    def test_a_stale_answer_stops_blocking(self) -> None:
        """An answer that never came back stops blocking after the timeout."""
        session = ChatSession()
        with patch.object(chat.time, "time", return_value=1000.0):
            session.start_answering("c1")
        with patch.object(
            chat.time, "time", return_value=1000.0 + ANSWERING_TIMEOUT_SECONDS - 1
        ):
            assert session.is_answering("c1")
        with patch.object(
            chat.time, "time", return_value=1000.0 + ANSWERING_TIMEOUT_SECONDS
        ):
            assert not session.is_answering("c1")

    def test_a_proposal_is_taken_once(self) -> None:
        """The pending change gets one reply, then is gone."""
        session = ChatSession()
        assert session.take_proposal() is None
        proposal = _proposal()
        session.proposal = proposal
        assert session.take_proposal() is proposal
        assert session.take_proposal() is None


def test_changes_text() -> None:
    """Changes read as one fragment."""
    assert (
        changes_text(
            [
                ConfigChange("Trading type", "Balanced", "Risky"),
                ConfigChange("Max loss", "10%", "20%"),
            ]
        )
        == "Trading type from Balanced to Risky; Max loss from 10% to 20%"
    )


class TestDescribeFeedEntry:
    """Feed entries as the chat prompt shows them."""

    def test_event(self) -> None:
        """An event lists its fields."""
        entry = {
            "id": 1,
            "ts": 1760227200,
            "kind": "event",
            "title": "Entered a pool",
            "fields": [
                {"label": "Pool", "value": "WETH/USDC on Velodrome"},
                {"label": "Amount", "value": "$10.00"},
                "junk",
            ],
        }
        assert describe_feed_entry(entry) == (
            "- 2025-10-12 00:00 UTC Entered a pool: "
            "Pool: WETH/USDC on Velodrome; Amount: $10.00"
        )

    def test_event_without_fields(self) -> None:
        """Malformed fields are left out."""
        entry = {"ts": 1760227200, "kind": "event", "title": "T", "fields": "x"}
        assert describe_feed_entry(entry) == "- 2025-10-12 00:00 UTC T: "

    @pytest.mark.parametrize("role, speaker", [("user", "User"), ("agent", "Agent")])
    def test_message(self, role: str, speaker: str) -> None:
        """A message names who said it and is cut to length."""
        entry = {"ts": 1760227200, "kind": "message", "role": role, "text": "x" * 400}
        assert describe_feed_entry(entry) == (
            f"- 2025-10-12 00:00 UTC {speaker}: " + "x" * 300
        )

    @pytest.mark.parametrize("ts", [None, "soon", 10**20])
    def test_unreadable_time(self, ts) -> None:
        """A missing or bad timestamp does not break the line."""
        entry = {"kind": "message", "role": "user", "text": "hi"}
        if ts is not None:
            entry["ts"] = ts
        assert describe_feed_entry(entry) == "- unknown time User: hi"

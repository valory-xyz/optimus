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

"""The chat on the agent's main page: the messages being answered and the pending change."""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from packages.valory.skills.liquidity_trader_abci.utils.feed import (
    FeedEntry,
    KIND_EVENT,
    ROLE_USER,
)

# A message whose answer never came back stops blocking its resend after this.
ANSWERING_TIMEOUT_SECONDS = 300
# Keeps a long earlier reply from crowding the prompt.
RECENT_TEXT_LIMIT = 300
FEED_TIME_FORMAT = "%Y-%m-%d %H:%M UTC"


@dataclass(frozen=True)
class ChatTurn:
    """A chat message being answered: Pearl's id for it and its line as written."""

    client_id: str
    message: FeedEntry


@dataclass(frozen=True)
class ConfigChange:
    """One setting a chat change sets, as the user reads it."""

    setting: str
    before: str
    after: str


@dataclass(frozen=True)
class ConfigProposal:  # pylint: disable=too-many-instance-attributes
    """A config change from the chat, kept unapplied until the user confirms it.

    ``strategy_changed`` says whether the protocols, the trading type or the
    max loss change, which are stored together; ``activity_goal`` is ``None``
    when the goal does not change.
    """

    changes: List[ConfigChange]
    strategy_changed: bool
    selected_protocols: List[str]
    trading_type: str
    max_loss_percentage: float
    composite_score: float
    activity_goal: Optional[int]
    reasoning: str


class ChatSession:
    """What the chat keeps in memory only, so a restart drops it."""

    def __init__(self) -> None:
        """Initialize the session."""
        self._answering_since: Dict[str, float] = {}
        self.proposal: Optional[ConfigProposal] = None

    def is_answering(self, client_id: str) -> bool:
        """Return whether the message with this client id is still being answered.

        :param client_id: Pearl's id of the message.
        :return: whether an answer is in flight.
        """
        since = self._answering_since.get(client_id)
        return since is not None and time.time() - since < ANSWERING_TIMEOUT_SECONDS

    def start_answering(self, client_id: str) -> None:
        """Mark the message with this client id as being answered.

        :param client_id: Pearl's id of the message.
        """
        self._answering_since[client_id] = time.time()

    def finish_answering(self, client_id: str) -> None:
        """Mark the message with this client id as no longer being answered.

        :param client_id: Pearl's id of the message.
        """
        self._answering_since.pop(client_id, None)

    def take_proposal(self) -> Optional[ConfigProposal]:
        """Return the change waiting for confirmation and forget it: it gets one reply.

        :return: the change, if any.
        """
        proposal, self.proposal = self.proposal, None
        return proposal


def changes_text(changes: List[ConfigChange]) -> str:
    """Describe the changes in one sentence fragment.

    :param changes: the changes.
    :return: the text.
    """
    return "; ".join(
        f"{change.setting} from {change.before} to {change.after}" for change in changes
    )


def describe_feed_entry(entry: FeedEntry) -> str:
    """Describe one feed entry on one line, for the chat prompt.

    :param entry: the entry.
    :return: the line.
    """
    try:
        when = datetime.fromtimestamp(int(entry["ts"]), tz=timezone.utc).strftime(
            FEED_TIME_FORMAT
        )
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        when = "unknown time"
    if entry.get("kind") == KIND_EVENT:
        fields = entry.get("fields")
        details = "; ".join(
            f"{item.get('label')}: {item.get('value')}"
            for item in (fields if isinstance(fields, list) else [])
            if isinstance(item, dict)
        )
        return f"- {when} {entry.get('title')}: {details}"
    speaker = "User" if entry.get("role") == ROLE_USER else "Agent"
    text = str(entry.get("text", ""))[:RECENT_TEXT_LIMIT]
    return f"- {when} {speaker}: {text}"

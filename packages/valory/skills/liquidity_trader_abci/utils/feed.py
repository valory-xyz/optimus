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

"""The agent's feed: one append-only JSON-lines file of events and chat messages.

Pearl reads this file through the middleware, whether the agent runs or not,
so the line format is a contract: every line is a whole JSON object ending in
a newline, and a line that does not parse (a write cut short) is skipped.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

FEED_FILE = "feed.jsonl"

KIND_EVENT = "event"
KIND_MESSAGE = "message"
ROLE_USER = "user"
ROLE_AGENT = "agent"

NOT_AVAILABLE = "Not available"
NONE_TEXT = "None"

_READ_BLOCK_SIZE = 64 * 1024
# Searches read back this far, so their cost does not grow with the feed's age.
SCAN_WINDOW_SECONDS = 30 * 24 * 60 * 60
_NEWLINE = b"\n"
# The FSM and the HTTP handler both append; the lock keeps ids unique and lines whole.
_APPEND_LOCK = threading.Lock()

# How each DEX is named to the user, by the position's ``dex_type``.
_DEX_NAMES = {
    "balancerPool": "Balancer",
    "UniswapV3": "Uniswap V3",
    "Sturdy": "Sturdy",
    "velodrome": "Velodrome",
    "aerodrome": "Aerodrome",
}

Field = Dict[str, str]
FeedEntry = Dict[str, Any]


class EventType(Enum):
    """The events the agent writes, with the title Pearl shows for each."""

    POOL_ENTERED = "pool_entered"
    POOL_EXITED = "pool_exited"
    POSITIONS_EXITED = "positions_exited"
    CONFIG_CHANGED = "config_changed"

    @property
    def title(self) -> str:
        """Return the title shown on the event's card."""
        return _EVENT_TITLES[self]


_EVENT_TITLES = {
    EventType.POOL_ENTERED: "Entered a pool",
    EventType.POOL_EXITED: "Exited a pool",
    EventType.POSITIONS_EXITED: "Exited all positions",
    EventType.CONFIG_CHANGED: "Changed the configuration",
}


def field(label: str, value: str) -> Field:
    """Build one display-ready field of an event card.

    :param label: the field's label.
    :param value: the field's value, as shown.
    :return: the field.
    """
    return {"label": label, "value": value}


def format_usd(amount: float) -> str:
    """Format a USD amount, e.g. ``$1,234.56``.

    :param amount: the amount in USD.
    :return: the display text.
    """
    return f"${amount:,.2f}"


def format_result(exit_usd: float, principal_usd: float) -> str:
    """Format what a position gained or lost, e.g. ``+$1.20 (+2.40%)``.

    :param exit_usd: the value received on exit, in USD.
    :param principal_usd: the value put in on entry, in USD.
    :return: the display text.
    """
    change = exit_usd - principal_usd
    sign = "-" if change < 0 else "+"
    return f"{sign}${abs(change):,.2f} ({sign}{abs(change) / principal_usd:.2%})"


def format_date(timestamp: int) -> str:
    """Format a unix timestamp as a UTC day, e.g. ``12 Oct 2026``.

    :param timestamp: unix seconds.
    :return: the display text.
    """
    day = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return f"{day.day} {day.strftime('%b %Y')}"


def pool_name(position: Mapping[str, Any]) -> str:
    """Name a pool position for the user, e.g. ``WETH/USDC on Velodrome``.

    :param position: the position or the action that opened it.
    :return: the display text.
    """
    symbols = [
        str(symbol)
        for symbol in (position.get("token0_symbol"), position.get("token1_symbol"))
        if symbol
    ]
    tokens = "/".join(symbols) or str(position.get("pool_address") or "Unknown pool")
    dex_type = position.get("dex_type")
    dex = _DEX_NAMES.get(str(dex_type), dex_type)
    return f"{tokens} on {dex}" if dex else tokens


def _parse(line: bytes) -> Optional[FeedEntry]:
    """Return the entry on a line, or ``None`` if the line is not a whole entry.

    :param line: the raw line, without its newline.
    :return: the entry, or ``None``.
    """
    try:
        entry = json.loads(line)
    except (ValueError, RecursionError):
        return None
    if not isinstance(entry, dict):
        return None
    entry_id = entry.get("id")
    if isinstance(entry_id, bool) or not isinstance(entry_id, int):
        return None
    return entry


def _lines_newest_first(path: Path) -> Iterator[bytes]:
    """Yield the file's non-empty lines from the last to the first.

    Reads blocks from the end, so finding recent entries does not read the
    whole history.

    :param path: the feed file.
    :yield: each line, without its newline.
    """
    try:
        feed_file = open(path, "rb")  # pylint: disable=consider-using-with
    except FileNotFoundError:
        return
    with feed_file:
        position = feed_file.seek(0, os.SEEK_END)
        partial = b""
        while position > 0:
            size = min(_READ_BLOCK_SIZE, position)
            position -= size
            feed_file.seek(position)
            lines = (feed_file.read(size) + partial).split(_NEWLINE)
            # The first piece may be the end of a line that starts in the
            # previous block, so it waits for that block.
            partial = lines.pop(0)
            yield from (line for line in reversed(lines) if line)
        if partial:
            yield partial


def _entries_newest_first(path: Path) -> Iterator[FeedEntry]:
    """Yield the parseable entries from the last to the first.

    :param path: the feed file.
    :yield: each entry.
    """
    for line in _lines_newest_first(path):
        entry = _parse(line)
        if entry is not None:
            yield entry


def _timestamp(entry: FeedEntry) -> Optional[int]:
    """Return the entry's unix time, or ``None`` if it has none.

    :param entry: the entry.
    :return: the time.
    """
    ts = entry.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, int):
        return None
    return ts


def _scan_window(path: Path) -> Iterator[FeedEntry]:
    """Yield the entries newest first, stopping at the first older than the window.

    :param path: the feed file.
    :yield: each entry.
    """
    cutoff = int(time.time()) - SCAN_WINDOW_SECONDS
    for entry in _entries_newest_first(path):
        ts = _timestamp(entry)
        if ts is not None and ts < cutoff:
            return
        yield entry


def _append(store_path: Path, body: Dict[str, Any]) -> FeedEntry:
    """Append one entry with the next id and return it as written.

    :param store_path: the directory holding the feed.
    :param body: the entry without ``id`` and ``ts``.
    :return: the entry written.
    """
    path = Path(store_path) / FEED_FILE
    with _APPEND_LOCK:
        last = next(_entries_newest_first(path), None)
        entry: FeedEntry = {
            "id": 1 if last is None else last["id"] + 1,
            "ts": int(time.time()),
            **body,
        }
        line = json.dumps(entry, separators=(",", ":")).encode("utf-8") + _NEWLINE
        with open(path, "a+b") as feed_file:
            if feed_file.seek(0, os.SEEK_END) > 0:
                feed_file.seek(-1, os.SEEK_END)
                # A cut last line stays its own (skipped) line, not the start of this one.
                if feed_file.read(1) != _NEWLINE:
                    line = _NEWLINE + line
            feed_file.write(line)
            feed_file.flush()
            os.fsync(feed_file.fileno())
    return entry


def append_event(
    store_path: Path, event_type: EventType, fields: List[Field]
) -> FeedEntry:
    """Append an event line.

    :param store_path: the directory holding the feed.
    :param event_type: the event.
    :param fields: the card's fields, in display order.
    :return: the entry written.
    """
    body = {
        "kind": KIND_EVENT,
        "type": event_type.value,
        "title": event_type.title,
        "fields": fields,
    }
    return _append(store_path, body)


def record_event(
    store_path: Path,
    event_type: EventType,
    fields: List[Field],
    logger: Logger,
) -> None:
    """Append an event line; a failed write is logged, never raised.

    The feed only reports what the agent did, so it must not stop it.

    :param store_path: the directory holding the feed.
    :param event_type: the event.
    :param fields: the card's fields, in display order.
    :param logger: where to log a failed write.
    """
    try:
        append_event(store_path, event_type, fields)
    except OSError as e:
        logger.error(f"Could not write the {event_type.value} event to the feed: {e}")


def append_message(
    store_path: Path, role: str, text: str, client_id: Optional[str] = None
) -> FeedEntry:
    """Append a chat message line.

    :param store_path: the directory holding the feed.
    :param role: ``user`` or ``agent``.
    :param text: the message.
    :param client_id: Pearl's id of the user message this line is or answers.
    :return: the entry written.
    """
    body: Dict[str, Any] = {"kind": KIND_MESSAGE, "role": role, "text": text}
    if client_id is not None:
        body["client_id"] = client_id
    return _append(store_path, body)


def read_recent(store_path: Path, limit: int) -> List[FeedEntry]:
    """Return up to ``limit`` of the latest entries, oldest first.

    :param store_path: the directory holding the feed.
    :param limit: the most entries to return.
    :return: the entries.
    """
    entries: List[FeedEntry] = []
    for entry in _entries_newest_first(Path(store_path) / FEED_FILE):
        if len(entries) >= limit:
            break
        entries.append(entry)
    return entries[::-1]


def has_event_since(store_path: Path, event_type: EventType, since: int) -> bool:
    """Return whether an event of this type was written at or after a time.

    Only the scan window is read; a clock set back within it writes newer
    lines with older times, so the search does not stop at ``since``.

    :param store_path: the directory holding the feed.
    :param event_type: the event.
    :param since: unix seconds.
    :return: whether such an event is in the feed.
    """
    for entry in _scan_window(Path(store_path) / FEED_FILE):
        ts = _timestamp(entry)
        if (
            entry.get("kind") == KIND_EVENT
            and entry.get("type") == event_type.value
            and ts is not None
            and ts >= since
        ):
            return True
    return False


def find_exchange(
    store_path: Path, client_id: str
) -> Tuple[Optional[FeedEntry], Optional[FeedEntry]]:
    """Return the user line with this ``client_id`` and the agent line answering it.

    Only the scan window is read, so a message older than it reads as new.

    :param store_path: the directory holding the feed.
    :param client_id: Pearl's id of the user message.
    :return: the user line and the reply, each ``None`` when not written.
    """
    message: Optional[FeedEntry] = None
    reply: Optional[FeedEntry] = None
    for entry in _scan_window(Path(store_path) / FEED_FILE):
        if entry.get("kind") != KIND_MESSAGE or entry.get("client_id") != client_id:
            continue
        if entry.get("role") == ROLE_AGENT and reply is None:
            reply = entry
        elif entry.get("role") == ROLE_USER:
            message = entry
            # The reply is always written after the message it answers.
            break
    return message, reply

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

"""Tests for utils/feed.py."""

# pylint: skip-file

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.skills.liquidity_trader_abci.utils import feed
from packages.valory.skills.liquidity_trader_abci.utils.feed import (
    EventType,
    FEED_FILE,
    append_event,
    append_message,
    field,
    find_exchange,
    format_date,
    format_result,
    format_usd,
    pool_name,
    read_recent,
    record_event,
)


def _lines(store: Path):
    return (store / FEED_FILE).read_bytes().split(b"\n")


def _entries(store: Path):
    return [json.loads(line) for line in _lines(store) if line]


class TestAppend:
    """Appending whole lines with unique, increasing ids."""

    def test_first_entry_creates_the_file_with_id_one(self, tmp_path: Path) -> None:
        """The first line gets id 1 and ends in a newline."""
        with patch.object(feed.time, "time", return_value=1700000000.9):
            entry = append_message(tmp_path, "user", "hi", "c1")
        assert entry == {
            "id": 1,
            "ts": 1700000000,
            "kind": "message",
            "role": "user",
            "text": "hi",
            "client_id": "c1",
        }
        assert (tmp_path / FEED_FILE).read_bytes().endswith(b"\n")
        assert _entries(tmp_path) == [entry]

    def test_ids_increase_with_each_line(self, tmp_path: Path) -> None:
        """Each line gets the next id."""
        append_message(tmp_path, "user", "a")
        append_event(tmp_path, EventType.POOL_ENTERED, [field("Pool", "X")])
        append_message(tmp_path, "agent", "b")
        assert [e["id"] for e in _entries(tmp_path)] == [1, 2, 3]

    def test_message_without_client_id_has_no_client_id(self, tmp_path: Path) -> None:
        """A message with no client id leaves the key out."""
        entry = append_message(tmp_path, "agent", "hello")
        assert "client_id" not in entry

    def test_event_line_is_display_ready(self, tmp_path: Path) -> None:
        """An event carries its type, title and fields in order."""
        fields = [field("Pool", "WETH/USDC on Velodrome"), field("Amount", "$10.00")]
        entry = append_event(tmp_path, EventType.POOL_ENTERED, fields)
        assert entry["kind"] == "event"
        assert entry["type"] == "pool_entered"
        assert entry["title"] == "Entered a pool"
        assert entry["fields"] == fields

    def test_append_never_rewrites_earlier_lines(self, tmp_path: Path) -> None:
        """Earlier bytes stay as written."""
        append_message(tmp_path, "user", "a")
        before = (tmp_path / FEED_FILE).read_bytes()
        append_message(tmp_path, "user", "b")
        assert (tmp_path / FEED_FILE).read_bytes().startswith(before)

    def test_cut_last_line_stays_its_own_line(self, tmp_path: Path) -> None:
        """A write cut short is not glued to the next entry."""
        append_message(tmp_path, "user", "a")
        with open(tmp_path / FEED_FILE, "ab") as f:
            f.write(b'{"id": 2, "ts": 1, "ki')
        entry = append_message(tmp_path, "user", "b")
        assert entry["id"] == 2
        lines = [line for line in _lines(tmp_path) if line]
        assert lines[1] == b'{"id": 2, "ts": 1, "ki'
        assert json.loads(lines[2])["text"] == "b"

    def test_concurrent_appends_get_unique_ids(self, tmp_path: Path) -> None:
        """Appends from several threads never share an id."""
        threads = [
            threading.Thread(target=append_message, args=(tmp_path, "user", str(i)))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(e["id"] for e in _entries(tmp_path)) == list(range(1, 21))

    def test_titles_match_the_event_names(self) -> None:
        """Each event type has the title Pearl shows."""
        assert {t.value: t.title for t in EventType} == {
            "pool_entered": "Entered a pool",
            "pool_exited": "Exited a pool",
            "positions_exited": "Exited all positions",
            "config_changed": "Changed the configuration",
        }


class TestRecordEvent:
    """Recording an event never stops the agent."""

    def test_writes_the_event(self, tmp_path: Path) -> None:
        """A writable store gets the line."""
        logger = MagicMock()
        record_event(tmp_path, EventType.CONFIG_CHANGED, [], logger)
        assert _entries(tmp_path)[0]["type"] == "config_changed"
        logger.error.assert_not_called()

    def test_a_failed_write_is_logged(self, tmp_path: Path) -> None:
        """A missing store directory is logged, not raised."""
        logger = MagicMock()
        record_event(tmp_path / "missing", EventType.CONFIG_CHANGED, [], logger)
        logger.error.assert_called_once()


class TestRead:
    """Reading recent entries and finding an answered message."""

    def test_missing_file_reads_empty(self, tmp_path: Path) -> None:
        """No feed yet means no entries."""
        assert read_recent(tmp_path, 5) == []
        assert find_exchange(tmp_path, "c1") == (None, None)

    def test_recent_entries_are_the_latest_oldest_first(self, tmp_path: Path) -> None:
        """The latest entries come back in file order."""
        for i in range(5):
            append_message(tmp_path, "user", str(i))
        assert [e["text"] for e in read_recent(tmp_path, 3)] == ["2", "3", "4"]

    def test_unparseable_lines_are_skipped(self, tmp_path: Path) -> None:
        """A cut line, a non-object and a line without an int id are skipped."""
        append_message(tmp_path, "user", "a")
        with open(tmp_path / FEED_FILE, "ab") as f:
            f.write(b"[1, 2]\n")
            f.write(b'{"id": true}\n')
            f.write(b'{"id": "3"}\n')
            f.write(b"\n")
            f.write(b'{"id": 9, "kind"')
        assert [e["text"] for e in read_recent(tmp_path, 10)] == ["a"]

    def test_lines_across_read_blocks(self, tmp_path: Path) -> None:
        """Lines that straddle the read block boundary are read whole."""
        with patch.object(feed, "_READ_BLOCK_SIZE", 16):
            for i in range(10):
                append_message(tmp_path, "user", "x" * 20 + str(i))
            entries = read_recent(tmp_path, 10)
        assert [e["id"] for e in entries] == list(range(1, 11))

    def test_finds_the_message_and_its_reply(self, tmp_path: Path) -> None:
        """The user line and the agent line with the client id are found."""
        message = append_message(tmp_path, "user", "q", "c1")
        append_message(tmp_path, "user", "other", "c2")
        reply = append_message(tmp_path, "agent", "a", "c1")
        assert find_exchange(tmp_path, "c1") == (message, reply)

    def test_message_without_reply(self, tmp_path: Path) -> None:
        """An unanswered message is found without a reply."""
        message = append_message(tmp_path, "user", "q", "c1")
        assert find_exchange(tmp_path, "c1") == (message, None)

    def test_text_mentioning_the_client_id_is_not_a_match(self, tmp_path: Path) -> None:
        """Only the client_id key matches, not a text that contains it."""
        append_message(tmp_path, "user", '"c1"', "c2")
        append_event(tmp_path, EventType.CONFIG_CHANGED, [field("Setting", '"c1"')])
        assert find_exchange(tmp_path, "c1") == (None, None)

    def test_only_message_lines_with_the_client_id_match(self, tmp_path: Path) -> None:
        """A line that carries the id elsewhere, or an event with it, is not a match."""
        with open(tmp_path / FEED_FILE, "ab") as f:
            f.write(b'{"id": 1, "kind": "event", "client_id": "c1"}\n')
            f.write(b'{"id": 2, "kind": "message", "note": "c1", "client_id": "c2"}\n')
        assert find_exchange(tmp_path, "c1") == (None, None)

    def test_the_latest_reply_is_kept(self, tmp_path: Path) -> None:
        """Of two replies to one message, the latest is returned."""
        message = append_message(tmp_path, "user", "q", "c1")
        append_message(tmp_path, "agent", "first", "c1")
        latest = append_message(tmp_path, "agent", "second", "c1")
        assert find_exchange(tmp_path, "c1") == (message, latest)

    def test_file_starting_with_a_blank_line(self, tmp_path: Path) -> None:
        """Blank lines, even the first, are skipped."""
        (tmp_path / FEED_FILE).write_bytes(b'\n{"id": 1, "kind": "message"}\n')
        assert [e["id"] for e in read_recent(tmp_path, 5)] == [1]


class TestFormatting:
    """Display-ready values."""

    def test_format_usd(self) -> None:
        """USD amounts have two decimals and thousands separators."""
        assert format_usd(1234.5) == "$1,234.50"

    @pytest.mark.parametrize(
        "exit_usd, principal, expected",
        [
            (102.0, 100.0, "+$2.00 (+2.00%)"),
            (99.5, 100.0, "-$0.50 (-0.50%)"),
            (100.0, 100.0, "+$0.00 (+0.00%)"),
        ],
    )
    def test_format_result(self, exit_usd: float, principal: float, expected: str):
        """The result is signed in dollars and percent of the principal."""
        assert format_result(exit_usd, principal) == expected

    def test_format_date(self) -> None:
        """Dates are UTC days."""
        assert format_date(1760227200) == "12 Oct 2025"

    @pytest.mark.parametrize(
        "position, expected",
        [
            (
                {
                    "token0_symbol": "WETH",
                    "token1_symbol": "USDC",
                    "dex_type": "velodrome",
                },
                "WETH/USDC on Velodrome",
            ),
            (
                {"token0_symbol": "USDC", "dex_type": "aerodrome"},
                "USDC on Aerodrome",
            ),
            ({"pool_address": "0xP", "dex_type": "newDex"}, "0xP on newDex"),
            ({"token0_symbol": "A", "token1_symbol": "B"}, "A/B"),
            ({}, "Unknown pool"),
        ],
    )
    def test_pool_name(self, position, expected: str) -> None:
        """Pools are named by their tokens and DEX."""
        assert pool_name(position) == expected

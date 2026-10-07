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

"""Tests for utils/activity_goal.py."""

# pylint: skip-file

import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from packages.valory.skills.liquidity_trader_abci.utils import activity_goal
from packages.valory.skills.liquidity_trader_abci.utils.activity_goal import (
    build_activity_goal_block,
    is_valid_activity_goal,
    merge_agent_performance,
    parse_stored_int,
    read_activity_goal_block,
    retarget_activity_goal,
    should_stand_by,
    stamp_last_met_at,
    staking_side_met,
)

EPOCH = 1_000
NOW = 1_500


def _block(**overrides):
    block = build_activity_goal_block(10, 4, EPOCH, None, NOW)
    block.update(overrides)
    return block


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data))


def _read(path: Path):
    return json.loads(path.read_text())


@pytest.mark.parametrize(
    "value,expected",
    [(0, True), (500, True), (-1, False), (2.0, False), ("3", False), (True, False)],
)
def test_is_valid_activity_goal(value, expected) -> None:
    """Only non-negative, non-bool integers are goals."""
    assert is_valid_activity_goal(value) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [("7", 7), ("0", 0), (None, None), ("x", None), ("-3", None), (5, 5)],
)
def test_parse_stored_int(raw, expected) -> None:
    """KV strings parse to non-negative ints; anything else is absent."""
    assert parse_stored_int(raw) == expected


class TestStandbyGate:
    """The staking side is regime-aware and both halves must be met."""

    def test_new_regime_uses_the_activity_target(self) -> None:
        """When the activity target is computed, it decides over the KPI."""
        assert staking_side_met(False, True) is True
        assert staking_side_met(True, False) is False

    def test_old_regime_uses_the_kpi(self) -> None:
        """Without an activity target, the on-chain KPI decides."""
        assert staking_side_met(True, None) is True
        assert staking_side_met(None, None) is None

    @pytest.mark.parametrize(
        "kpi,target,goal,expected",
        [
            (True, None, True, True),
            (True, None, False, False),
            (True, None, None, False),
            (None, None, True, False),
            (False, True, True, True),
            (True, False, True, False),
        ],
    )
    def test_should_stand_by(self, kpi, target, goal, expected) -> None:
        """Standby needs both the staking side and the goal."""
        assert should_stand_by(kpi, target, goal) is expected


class TestBlock:
    """The block shape and the met check."""

    def test_shape(self) -> None:
        """The block carries exactly the shared fields, in rounds."""
        assert build_activity_goal_block(10, 4, EPOCH, None, NOW) == {
            "unit": "rounds",
            "target": 10,
            "progress": 4,
            "is_met": False,
            "period_start": EPOCH,
            "last_met_at": None,
            "updated_at": NOW,
        }

    def test_met_at_the_boundary(self) -> None:
        """Progress equal to the target meets it."""
        assert build_activity_goal_block(10, 10, EPOCH, None, NOW)["is_met"] is True
        assert build_activity_goal_block(10, 9, EPOCH, None, NOW)["is_met"] is False

    def test_zero_target_is_met_immediately(self) -> None:
        """A goal of zero is met with no rounds counted."""
        assert build_activity_goal_block(0, 0, EPOCH, None, NOW)["is_met"] is True


class TestStampLastMetAt:
    """``last_met_at`` is stamped once per epoch."""

    def test_stamped_when_first_met(self) -> None:
        """The first time progress reaches the target, now is stamped."""
        assert stamp_last_met_at(10, 10, EPOCH, None, NOW) == NOW

    def test_kept_once_stamped_this_epoch(self) -> None:
        """A later period in the same epoch keeps the first stamp."""
        assert stamp_last_met_at(11, 10, EPOCH, EPOCH + 1, NOW) == EPOCH + 1

    def test_restamped_in_a_new_epoch(self) -> None:
        """A stamp from an earlier epoch is replaced when met again."""
        assert stamp_last_met_at(10, 10, EPOCH, EPOCH - 1, NOW) == NOW

    def test_not_stamped_when_unmet(self) -> None:
        """An unmet goal keeps the previous stamp."""
        assert stamp_last_met_at(3, 10, EPOCH, EPOCH - 1, NOW) == EPOCH - 1


class TestMergeAgentPerformance:
    """The merge writer keeps every key it was not asked to change."""

    def test_keeps_unrelated_keys(self, tmp_path: Path) -> None:
        """Only the given keys are replaced."""
        path = tmp_path / "perf.json"
        _write(path, {"metrics": [1], "agent_behavior": "hi", "timestamp": 1})
        merge_agent_performance(path, {"activity_goal": _block()})
        assert _read(path) == {
            "metrics": [1],
            "agent_behavior": "hi",
            "timestamp": 1,
            "activity_goal": _block(),
        }

    def test_defaults_only_fill_missing_keys(self, tmp_path: Path) -> None:
        """Defaults never overwrite what is already on disk."""
        path = tmp_path / "perf.json"
        _write(path, {"agent_behavior": "hi"})
        merge_agent_performance(
            path, {"metrics": [2]}, defaults={"agent_behavior": None, "x": 0}
        )
        assert _read(path) == {"agent_behavior": "hi", "metrics": [2], "x": 0}

    @pytest.mark.parametrize("content", [None, "{not json", "[1, 2]"])
    def test_missing_or_corrupt_file_starts_empty(self, tmp_path: Path, content) -> None:
        """A missing, corrupt or non-object file is replaced by the merged keys."""
        path = tmp_path / "perf.json"
        if content is not None:
            path.write_text(content)
        merge_agent_performance(path, {"metrics": []})
        assert _read(path) == {"metrics": []}

    def test_failed_write_leaves_the_file_and_no_temp(self, tmp_path: Path) -> None:
        """A failure mid-write keeps the old content and cleans up the temp file."""
        path = tmp_path / "perf.json"
        _write(path, {"metrics": [1]})
        with patch.object(activity_goal.os, "replace", side_effect=OSError("boom")):
            with pytest.raises(OSError):
                merge_agent_performance(path, {"metrics": [2]})
        assert _read(path) == {"metrics": [1]}
        assert [p.name for p in tmp_path.iterdir()] == ["perf.json"]

    def test_failed_cleanup_still_raises_the_write_error(self, tmp_path: Path) -> None:
        """A temp file that cannot be removed does not hide the original error."""
        path = tmp_path / "perf.json"
        with (
            patch.object(activity_goal.os, "replace", side_effect=OSError("boom")),
            patch.object(activity_goal.os, "unlink", side_effect=OSError("gone")),
        ):
            with pytest.raises(OSError, match="boom"):
                merge_agent_performance(path, {"metrics": [2]})

    def test_concurrent_writers_lose_no_keys(self, tmp_path: Path) -> None:
        """Two threads merging different keys both land."""
        path = tmp_path / "perf.json"
        _write(path, {})

        def writer(prefix: str) -> None:
            for i in range(50):
                merge_agent_performance(path, {f"{prefix}{i}": i})

        threads = [threading.Thread(target=writer, args=(p,)) for p in "ab"]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        data = _read(path)
        assert all(data[f"{p}{i}"] == i for p in "ab" for i in range(50))


class TestReadActivityGoalBlock:
    """Reading the block back for the chat."""

    def test_returns_the_block(self, tmp_path: Path) -> None:
        """A valid block is returned as written."""
        path = tmp_path / "perf.json"
        _write(path, {"activity_goal": _block()})
        assert read_activity_goal_block(path) == _block()

    @pytest.mark.parametrize(
        "data",
        [{}, {"activity_goal": "x"}, {"activity_goal": _block(progress=-1)}],
    )
    def test_missing_or_invalid_is_none(self, tmp_path: Path, data) -> None:
        """No usable block reads as ``None``."""
        path = tmp_path / "perf.json"
        _write(path, data)
        assert read_activity_goal_block(path) is None

    def test_missing_file_is_none(self, tmp_path: Path) -> None:
        """No file reads as ``None``."""
        assert read_activity_goal_block(tmp_path / "perf.json") is None


class TestRetargetActivityGoal:
    """A goal change from the chat applies to the current epoch at once."""

    def test_lowering_to_progress_meets_the_goal(self, tmp_path: Path) -> None:
        """Lowering the target to the progress made flips ``is_met`` and stamps it."""
        path = tmp_path / "perf.json"
        _write(path, {"metrics": [1], "activity_goal": _block(progress=4)})
        block = retarget_activity_goal(path, 4, NOW + 1)
        expected = build_activity_goal_block(4, 4, EPOCH, NOW + 1, NOW + 1)
        assert block == expected
        assert _read(path) == {"metrics": [1], "activity_goal": expected}

    def test_raising_keeps_progress_and_unmeets(self, tmp_path: Path) -> None:
        """Raising the target keeps the progress and an earlier stamp."""
        path = tmp_path / "perf.json"
        _write(path, {"activity_goal": _block(target=4, is_met=True, last_met_at=NOW)})
        block = retarget_activity_goal(path, 500, NOW + 1)
        assert block == build_activity_goal_block(500, 4, EPOCH, NOW, NOW + 1)

    def test_invalid_stamp_is_dropped(self, tmp_path: Path) -> None:
        """A malformed ``last_met_at`` is not carried forward."""
        path = tmp_path / "perf.json"
        _write(path, {"activity_goal": _block(last_met_at="yesterday")})
        block = retarget_activity_goal(path, 20, NOW + 1)
        assert block is not None and block["last_met_at"] is None

    def test_no_block_yet_writes_nothing(self, tmp_path: Path) -> None:
        """Without a block there is no progress to keep; the FSM writes it later."""
        path = tmp_path / "perf.json"
        _write(path, {"metrics": []})
        assert retarget_activity_goal(path, 5, NOW) is None
        assert _read(path) == {"metrics": []}

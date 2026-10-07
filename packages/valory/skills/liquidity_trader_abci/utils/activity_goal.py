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

"""Per-epoch activity goal: the standby gate and the block Pearl reads.

The goal is counted in rounds, one per FSM period in which the agent works.
Both the FSM (main loop) and the HTTP handler (background threads) write
``agent_performance.json``, so every write goes through the locked merge here.
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

ACTIVITY_GOAL_KEY = "activity_goal"
ACTIVITY_GOAL_UNIT = "rounds"

# KV store keys holding the goal state across restarts.
KV_ACTIVITY_GOAL_TARGET = "activity_goal_target"
KV_ACTIVITY_GOAL_PROGRESS = "activity_goal_progress"
KV_ACTIVITY_GOAL_PERIOD_START = "activity_goal_period_start"
KV_ACTIVITY_GOAL_LAST_MET_AT = "activity_goal_last_met_at"
KV_ACTIVITY_GOAL_LAST_COUNTED_PERIOD = "activity_goal_last_counted_period"
KV_ACTIVITY_GOAL_KEYS = (
    KV_ACTIVITY_GOAL_TARGET,
    KV_ACTIVITY_GOAL_PROGRESS,
    KV_ACTIVITY_GOAL_PERIOD_START,
    KV_ACTIVITY_GOAL_LAST_MET_AT,
    KV_ACTIVITY_GOAL_LAST_COUNTED_PERIOD,
)

PathLike = Union[str, Path]

_AGENT_PERFORMANCE_LOCK = threading.Lock()


def is_non_negative_int(value: Any) -> bool:
    """Return whether ``value`` is a non-negative, non-bool int.

    Goals, counts and timestamps in the block must all be of this kind.

    :param value: the candidate value.
    :return: whether it is valid.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def parse_stored_int(raw: Any) -> Optional[int]:
    """Parse a non-negative integer stored as a KV string.

    :param raw: the stored value, or ``None`` when the key is absent.
    :return: the integer, or ``None`` when absent or malformed.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def staking_side_met(
    is_staking_kpi_met: Optional[bool], is_activity_target_met: Optional[bool]
) -> Optional[bool]:
    """Return the staking half of the standby gate.

    ``is_activity_target_met`` is only computed on the decoupled-activity
    regime; on the old regime it is ``None`` and the on-chain KPI decides. This
    matches how Pearl derives the staking side for its status header.

    :param is_staking_kpi_met: the on-chain liveness KPI verdict.
    :param is_activity_target_met: the new-regime activity-target verdict.
    :return: whether the staking side is met, or ``None`` when undetermined.
    """
    if is_activity_target_met is not None:
        return is_activity_target_met
    return is_staking_kpi_met


def should_stand_by(
    is_staking_kpi_met: Optional[bool],
    is_activity_target_met: Optional[bool],
    is_activity_goal_met: Optional[bool],
) -> bool:
    """Return whether the period should skip its work and stand by.

    :param is_staking_kpi_met: the on-chain liveness KPI verdict.
    :param is_activity_target_met: the new-regime activity-target verdict.
    :param is_activity_goal_met: whether the rounds goal was met before this period.
    :return: ``True`` only when both the staking side and the goal are met.
    """
    return (
        is_activity_goal_met is True
        and staking_side_met(is_staking_kpi_met, is_activity_target_met) is True
    )


def stamp_last_met_at(
    progress: int,
    target: int,
    period_start: int,
    last_met_at: Optional[int],
    now: int,
) -> Optional[int]:
    """Stamp ``last_met_at`` the first time the goal is met in an epoch.

    :param progress: rounds counted in the current epoch.
    :param target: the goal in effect.
    :param period_start: the start of the current epoch.
    :param last_met_at: the previous stamp, if any.
    :param now: the current timestamp.
    :return: the stamp to keep.
    """
    already_met_this_epoch = last_met_at is not None and last_met_at >= period_start
    if progress >= target and not already_met_this_epoch:
        return now
    return last_met_at


def build_activity_goal_block(
    target: int,
    progress: int,
    period_start: int,
    last_met_at: Optional[int],
    now: int,
) -> Dict[str, Any]:
    """Build the ``activity_goal`` block of ``agent_performance.json``.

    :param target: the goal in effect.
    :param progress: rounds counted in the current epoch.
    :param period_start: the start of the current epoch (``tsCheckpoint``).
    :param last_met_at: when the goal was last met, if ever.
    :param now: the current timestamp.
    :return: the block.
    """
    return {
        "unit": ACTIVITY_GOAL_UNIT,
        "target": target,
        "progress": progress,
        "is_met": progress >= target,
        "period_start": period_start,
        "last_met_at": last_met_at,
        "updated_at": now,
    }


def _read_json_object(file_path: PathLike) -> Dict[str, Any]:
    """Read a JSON object, treating a missing, corrupt or non-object file as empty.

    :param file_path: the file to read.
    :return: the object.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json_atomically(file_path: PathLike, data: Dict[str, Any]) -> None:
    """Write JSON so that a crash mid-write never leaves a truncated file.

    :param file_path: the file to (over)write.
    :param data: the JSON-serialisable content.
    """
    path = Path(file_path)
    # Same directory, so os.replace stays on one filesystem and is atomic.
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def merge_agent_performance(
    file_path: PathLike,
    updates: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None,
) -> None:
    """Merge top-level keys into the agent performance file, keeping the rest.

    :param file_path: the agent performance file.
    :param updates: keys to overwrite.
    :param defaults: keys to set only when the file does not have them yet.
    """
    with _AGENT_PERFORMANCE_LOCK:
        data = _read_json_object(file_path)
        for key, value in (defaults or {}).items():
            data.setdefault(key, value)
        data.update(updates)
        _write_json_atomically(file_path, data)


def _valid_block(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the ``activity_goal`` block of ``data`` if its counts are usable.

    :param data: the agent performance content.
    :return: the block, or ``None`` when it is missing or invalid.
    """
    block = data.get(ACTIVITY_GOAL_KEY)
    if not isinstance(block, dict):
        return None
    for key in ("target", "progress", "period_start"):
        if not is_non_negative_int(block.get(key)):
            return None
    return block


def read_activity_goal_block(file_path: PathLike) -> Optional[Dict[str, Any]]:
    """Return the ``activity_goal`` block on disk.

    :param file_path: the agent performance file.
    :return: the block, or ``None`` when the file or the block is missing or invalid.
    """
    with _AGENT_PERFORMANCE_LOCK:
        return _valid_block(_read_json_object(file_path))


def retarget_activity_goal(
    file_path: PathLike, target: int, now: int
) -> Optional[Dict[str, Any]]:
    """Apply a new goal to the block on disk, keeping its progress and epoch.

    Lets Pearl see a goal change at once; the next staking check recomputes
    the block from the KV store.

    :param file_path: the agent performance file.
    :param target: the new goal.
    :param now: the current timestamp.
    :return: the block written, or ``None`` when no valid block exists yet.
    """
    with _AGENT_PERFORMANCE_LOCK:
        data = _read_json_object(file_path)
        previous = _valid_block(data)
        if previous is None:
            return None
        progress = previous["progress"]
        period_start = previous["period_start"]
        last_met_at = previous.get("last_met_at")
        if not is_non_negative_int(last_met_at):
            last_met_at = None
        last_met_at = stamp_last_met_at(
            progress, target, period_start, last_met_at, now
        )
        block = build_activity_goal_block(
            target, progress, period_start, last_met_at, now
        )
        data[ACTIVITY_GOAL_KEY] = block
        _write_json_atomically(file_path, data)
    return block

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

"""Tests for utils/checkpoint.py."""

# pylint: skip-file

import pytest

from packages.valory.skills.liquidity_trader_abci.utils.checkpoint import (
    is_checkpoint_due,
    is_rebaseline_signal,
    should_send_checkpoint,
)

TS_CHECKPOINT = 1_700_000_000
LIVENESS_PERIOD = 86_400
SCHEDULED_NEXT = TS_CHECKPOINT + LIVENESS_PERIOD


@pytest.mark.parametrize(
    "next_ts, expected",
    [
        (SCHEDULED_NEXT, False),
        (SCHEDULED_NEXT - 1, True),
        (SCHEDULED_NEXT + 1, True),
        (0, True),
    ],
)
def test_is_rebaseline_signal(next_ts: int, expected: bool) -> None:
    """The signal fires whenever the getter departs from the schedule."""
    assert is_rebaseline_signal(next_ts, TS_CHECKPOINT, LIVENESS_PERIOD) is expected


@pytest.mark.parametrize(
    "next_ts, synced_ts, expected",
    [
        # on schedule, period not over yet
        (SCHEDULED_NEXT, SCHEDULED_NEXT - 1, False),
        # on schedule, period just over
        (SCHEDULED_NEXT, SCHEDULED_NEXT, True),
        (SCHEDULED_NEXT, SCHEDULED_NEXT + 1, True),
        # rebaseline pending, getter ahead of synced_ts
        (TS_CHECKPOINT + 10, TS_CHECKPOINT + 5, True),
        # off schedule and already in the past
        (TS_CHECKPOINT + 10, TS_CHECKPOINT + 20, True),
    ],
)
def test_is_checkpoint_due(next_ts: int, synced_ts: int, expected: bool) -> None:
    """Due when the getter is off schedule or the period is over."""
    assert (
        is_checkpoint_due(next_ts, TS_CHECKPOINT, LIVENESS_PERIOD, synced_ts)
        is expected
    )


@pytest.mark.parametrize(
    "due, rebaseline_signal, available_rewards, expected",
    [
        (False, False, 0, False),
        (False, False, 10, False),
        (False, True, 10, False),
        (True, False, 0, False),
        (True, False, 1, True),
        (True, True, 0, True),
        (True, True, 1, True),
    ],
)
def test_should_send_checkpoint(
    due: bool, rebaseline_signal: bool, available_rewards: int, expected: bool
) -> None:
    """Send only when due and the checkpoint would change something."""
    assert should_send_checkpoint(due, rebaseline_signal, available_rewards) is expected

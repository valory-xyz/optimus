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

"""When the staking checkpoint is due, and when sending it changes anything.

``next_ts`` is ``getNextRewardCheckpointTimestamp()``. StakingToken v0.4 returns
``block.timestamp`` from it while a rebaseline is pending; older versions always
return ``tsCheckpoint + livenessPeriod``.
"""


def is_rebaseline_signal(
    next_ts: int, ts_checkpoint: int, liveness_period: int
) -> bool:
    """Return whether the next checkpoint departs from the regular schedule.

    :param next_ts: the next reward checkpoint timestamp.
    :param ts_checkpoint: the last checkpoint timestamp.
    :param liveness_period: the staking liveness period.
    :return: whether a rebaseline is signalled.
    """
    return next_ts != ts_checkpoint + liveness_period


def is_checkpoint_due(
    next_ts: int, ts_checkpoint: int, liveness_period: int, synced_ts: int
) -> bool:
    """Return whether the checkpoint is due.

    :param next_ts: the next reward checkpoint timestamp.
    :param ts_checkpoint: the last checkpoint timestamp.
    :param liveness_period: the staking liveness period.
    :param synced_ts: the agents' synced timestamp.
    :return: whether the checkpoint is due.
    """
    return (
        is_rebaseline_signal(next_ts, ts_checkpoint, liveness_period)
        or next_ts <= synced_ts
    )


def should_send_checkpoint(
    due: bool, rebaseline_signal: bool, available_rewards: int
) -> bool:
    """Return whether the checkpoint transaction is worth sending.

    :param due: whether the checkpoint is due.
    :param rebaseline_signal: whether a rebaseline is signalled.
    :param available_rewards: the staking contract's available rewards.
    :return: whether to send the checkpoint transaction.
    """
    return due and (available_rewards > 0 or rebaseline_signal)

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

"""Test the states/check_staking_kpi_met.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from unittest.mock import MagicMock, patch

from packages.valory.skills.abstract_round_abci.base import (
    CollectSameUntilThresholdRound,
)
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)
from packages.valory.skills.liquidity_trader_abci.states.check_staking_kpi_met import (
    CheckStakingKPIMetRound,
)


def test_import() -> None:
    """Test that the check_staking_kpi_met module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.states.check_staking_kpi_met  # noqa


class TestCheckStakingKPIMetRound:
    """Test CheckStakingKPIMetRound class."""

    def test_end_block_none_from_super(self) -> None:
        """Test end_block returns None when super returns None."""
        round_obj = object.__new__(CheckStakingKPIMetRound)
        with patch.object(
            CollectSameUntilThresholdRound, "end_block", return_value=None
        ):
            result = round_obj.end_block()
        assert result is None

    def test_end_block_withdrawal_initiated(self) -> None:
        """Test end_block returns WITHDRAWAL_INITIATED event."""
        round_obj = object.__new__(CheckStakingKPIMetRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.WITHDRAWAL_INITIATED),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.WITHDRAWAL_INITIATED)

    def test_end_block_non_done(self) -> None:
        """Test end_block returns super result when event is not DONE."""
        round_obj = object.__new__(CheckStakingKPIMetRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.NO_MAJORITY),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.NO_MAJORITY)

    def test_end_block_kpi_none_returns_error(self) -> None:
        """Test end_block returns ERROR when is_staking_kpi_met is None."""
        round_obj = object.__new__(CheckStakingKPIMetRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.is_staking_kpi_met = None
        mock_synced.most_voted_tx_hash = None
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.ERROR)

    def test_end_block_with_tx_hash_returns_settle(self) -> None:
        """Test end_block returns SETTLE when most_voted_tx_hash is set."""
        round_obj = object.__new__(CheckStakingKPIMetRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.is_staking_kpi_met = True
        mock_synced.most_voted_tx_hash = "0xhash"
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.SETTLE)

    def test_end_block_kpi_met(self) -> None:
        """Test end_block returns STAKING_KPI_MET when kpi is met."""
        round_obj = object.__new__(CheckStakingKPIMetRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.is_staking_kpi_met = True
        mock_synced.most_voted_tx_hash = None
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.STAKING_KPI_MET)

    def test_end_block_kpi_not_met(self) -> None:
        """Test end_block returns STAKING_KPI_NOT_MET when kpi is not met."""
        round_obj = object.__new__(CheckStakingKPIMetRound)
        mock_synced = MagicMock(spec=SynchronizedData)
        mock_synced.is_staking_kpi_met = False
        mock_synced.most_voted_tx_hash = None
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.STAKING_KPI_NOT_MET)

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

"""Test the states/apr_population.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from unittest.mock import MagicMock, PropertyMock, patch

from packages.valory.skills.abstract_round_abci.base import (
    CollectSameUntilThresholdRound,
)
from packages.valory.skills.liquidity_trader_abci.states.apr_population import (
    APRPopulationRound,
)
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)


def test_import() -> None:
    """Test that the apr_population module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.states.apr_population  # noqa


class TestAPRPopulationRound:
    """Test APRPopulationRound class."""

    def _stub_round(self, threshold: bool, payload_values=()):
        """Build a minimally-stubbed round bypassing __init__."""
        round_obj = object.__new__(APRPopulationRound)
        type(round_obj).threshold_reached = PropertyMock(return_value=threshold)
        type(round_obj).most_voted_payload_values = PropertyMock(
            return_value=payload_values
        )
        return round_obj

    def test_end_block_withdrawal_initiated_short_circuits(self) -> None:
        """Pre-super peek returns WITHDRAWAL_INITIATED when consensus payload tags it."""
        round_obj = self._stub_round(
            threshold=True,
            payload_values=(
                "APR Population (withdrawal initiated)",
                None,
                Event.WITHDRAWAL_INITIATED.value,
            ),
        )
        mock_synced = MagicMock(spec=SynchronizedData)
        type(round_obj).synchronized_data = PropertyMock(return_value=mock_synced)

        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            side_effect=AssertionError("super should not be reached on withdrawal"),
        ):
            result = round_obj.end_block()

        assert result == (mock_synced, Event.WITHDRAWAL_INITIATED)

    def test_end_block_normal_path_delegates_to_super(self) -> None:
        """Normal payload (no event tag) delegates to super().end_block()."""
        round_obj = self._stub_round(
            threshold=True,
            payload_values=("APR Population", None, None),
        )
        mock_synced = MagicMock(spec=SynchronizedData)
        with patch.object(
            CollectSameUntilThresholdRound,
            "end_block",
            return_value=(mock_synced, Event.DONE),
        ):
            result = round_obj.end_block()
        assert result == (mock_synced, Event.DONE)

    def test_end_block_below_threshold_returns_super_result(self) -> None:
        """Below threshold: peek is skipped, super's no_majority/None propagates."""
        round_obj = self._stub_round(threshold=False)
        with patch.object(
            CollectSameUntilThresholdRound, "end_block", return_value=None
        ):
            result = round_obj.end_block()
        assert result is None

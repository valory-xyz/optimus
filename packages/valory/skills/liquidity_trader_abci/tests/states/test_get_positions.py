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

"""Test the states/get_positions.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from packages.valory.skills.abstract_round_abci.base import (
    CollectSameUntilThresholdRound,
    get_name,
)
from packages.valory.skills.liquidity_trader_abci.payloads import GetPositionsPayload
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)
from packages.valory.skills.liquidity_trader_abci.states.get_positions import (
    GetPositionsRound,
)


def test_import() -> None:
    """Test that the get_positions module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.states.get_positions  # noqa


class TestGetPositionsRound:
    """Test GetPositionsRound class."""

    def test_is_collect_same(self) -> None:
        """Test inherits from CollectSameUntilThresholdRound."""
        assert issubclass(GetPositionsRound, CollectSameUntilThresholdRound)

    def test_payload_class(self) -> None:
        """Test payload_class attribute."""
        assert GetPositionsRound.payload_class is GetPositionsPayload

    def test_done_event(self) -> None:
        """Test done_event attribute."""
        assert GetPositionsRound.done_event == Event.DONE

    def test_no_majority_event(self) -> None:
        """Test no_majority_event attribute."""
        assert GetPositionsRound.no_majority_event == Event.NO_MAJORITY

    def test_none_event(self) -> None:
        """Test none_event attribute."""
        assert GetPositionsRound.none_event == Event.NONE

    def test_collection_key(self) -> None:
        """Test collection_key attribute."""
        assert GetPositionsRound.collection_key == get_name(
            SynchronizedData.participant_to_positions_round
        )

    def test_selection_key(self) -> None:
        """Test selection_key attribute."""
        assert GetPositionsRound.selection_key == get_name(SynchronizedData.positions)

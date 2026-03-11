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

from packages.valory.skills.abstract_round_abci.base import (
    CollectSameUntilThresholdRound,
    get_name,
)
from packages.valory.skills.liquidity_trader_abci.payloads import APRPopulationPayload
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

    def test_is_collect_same(self) -> None:
        """Test inherits from CollectSameUntilThresholdRound."""
        assert issubclass(APRPopulationRound, CollectSameUntilThresholdRound)

    def test_payload_class(self) -> None:
        """Test payload_class attribute."""
        assert APRPopulationRound.payload_class is APRPopulationPayload

    def test_synchronized_data_class(self) -> None:
        """Test synchronized_data_class attribute."""
        assert APRPopulationRound.synchronized_data_class is SynchronizedData

    def test_done_event(self) -> None:
        """Test done_event attribute."""
        assert APRPopulationRound.done_event == Event.DONE

    def test_no_majority_event(self) -> None:
        """Test no_majority_event attribute."""
        assert APRPopulationRound.no_majority_event == Event.NO_MAJORITY

    def test_none_event(self) -> None:
        """Test none_event attribute."""
        assert APRPopulationRound.none_event == Event.NONE

    def test_withdrawal_initiated(self) -> None:
        """Test withdrawal_initiated attribute."""
        assert APRPopulationRound.withdrawal_initiated == Event.WITHDRAWAL_INITIATED

    def test_collection_key(self) -> None:
        """Test collection_key attribute."""
        assert APRPopulationRound.collection_key == get_name(
            SynchronizedData.participant_to_context_round
        )

    def test_selection_key(self) -> None:
        """Test selection_key attribute."""
        assert APRPopulationRound.selection_key == get_name(SynchronizedData.context)

    def test_error_payload(self) -> None:
        """Test ERROR_PAYLOAD attribute."""
        assert APRPopulationRound.ERROR_PAYLOAD == {}

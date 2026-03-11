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

"""Test the behaviours.py module of the optimus_abci skill."""

# pylint: skip-file

from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
)
from packages.valory.skills.optimus_abci.behaviours import OptimusConsensusBehaviour
from packages.valory.skills.optimus_abci.composition import OptimusAbciApp
from packages.valory.skills.registration_abci.behaviours import (
    RegistrationStartupBehaviour,
)
from packages.valory.skills.termination_abci.behaviours import BackgroundBehaviour


def test_import() -> None:
    """Test that the behaviours module can be imported."""
    import packages.valory.skills.optimus_abci.behaviours  # noqa


def test_optimus_consensus_behaviour_class() -> None:
    """Test that OptimusConsensusBehaviour is an AbstractRoundBehaviour subclass."""
    assert issubclass(OptimusConsensusBehaviour, AbstractRoundBehaviour)


def test_initial_behaviour_cls() -> None:
    """Test that the initial behaviour class is set correctly."""
    assert (
        OptimusConsensusBehaviour.initial_behaviour_cls
        == RegistrationStartupBehaviour
    )


def test_abci_app_cls() -> None:
    """Test that the abci_app_cls is set correctly."""
    assert OptimusConsensusBehaviour.abci_app_cls == OptimusAbciApp


def test_behaviours_set_is_non_empty() -> None:
    """Test that the behaviours set is non-empty."""
    assert len(OptimusConsensusBehaviour.behaviours) > 0


def test_background_behaviours_cls() -> None:
    """Test that the background behaviours cls is set correctly."""
    assert OptimusConsensusBehaviour.background_behaviours_cls == {BackgroundBehaviour}

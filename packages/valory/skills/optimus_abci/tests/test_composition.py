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

"""Test the composition.py module of the optimus_abci skill."""

# pylint: skip-file

import packages.valory.skills.liquidity_trader_abci.rounds as LiquidityTraderAbci
import packages.valory.skills.mech_interact_abci.states.final_states as MechFinalStates
import packages.valory.skills.mech_interact_abci.states.mech_version as MechVersionStates
import packages.valory.skills.mech_interact_abci.states.request as MechRequestStates
import packages.valory.skills.mech_interact_abci.states.response as MechResponseStates
import packages.valory.skills.registration_abci.rounds as RegistrationAbci
import packages.valory.skills.reset_pause_abci.rounds as ResetAndPauseAbci
import packages.valory.skills.transaction_settlement_abci.rounds as TxSettlementAbci
from packages.valory.skills.abstract_round_abci.base import BackgroundAppConfig
from packages.valory.skills.optimus_abci.composition import (
    OptimusAbciApp,
    abci_app_transition_mapping,
    termination_config,
)
from packages.valory.skills.termination_abci.rounds import (
    BackgroundRound,
    Event,
    TerminationAbciApp,
)


def test_import() -> None:
    """Test that the composition module can be imported."""
    import packages.valory.skills.optimus_abci.composition  # noqa


def test_abci_app_transition_mapping_keys() -> None:
    """Test that the transition mapping has the expected source rounds."""
    expected_keys = {
        RegistrationAbci.FinishedRegistrationRound,
        LiquidityTraderAbci.FinishedCallCheckpointRound,
        LiquidityTraderAbci.FinishedCheckStakingKPIMetRound,
        LiquidityTraderAbci.FinishedDecisionMakingRound,
        LiquidityTraderAbci.FinishedEvaluateStrategyRound,
        LiquidityTraderAbci.FinishedTxPreparationRound,
        LiquidityTraderAbci.FailedMultiplexerRound,
        TxSettlementAbci.FinishedTransactionSubmissionRound,
        TxSettlementAbci.FailedRound,
        ResetAndPauseAbci.FinishedResetAndPauseRound,
        ResetAndPauseAbci.FinishedResetAndPauseErrorRound,
        # mech_interact_abci legs (new staking regime)
        LiquidityTraderAbci.FinishedWithMechRequestRound,
        LiquidityTraderAbci.FinishedWithMechResponsePollRound,
        LiquidityTraderAbci.FinishedWithOffchainMechDepositSettledRound,
        MechFinalStates.FinishedMarketplaceLegacyDetectedRound,
        MechFinalStates.FinishedMechLegacyDetectedRound,
        MechFinalStates.FinishedMechInformationRound,
        MechFinalStates.FailedMechInformationRound,
        MechFinalStates.FinishedMechRequestRound,
        MechFinalStates.FinishedMechPurchaseSubscriptionRound,
        MechFinalStates.FinishedMechResponseRound,
        MechFinalStates.FinishedMechResponseTimeoutRound,
        MechFinalStates.FinishedMechRequestSkipRound,
        MechFinalStates.FinishedOffchainMechRequestRound,
        MechFinalStates.FinishedOffchainMechDepositNeededRound,
        MechFinalStates.FailedOffchainMechRequestRound,
    }
    assert set(abci_app_transition_mapping.keys()) == expected_keys


def test_abci_app_transition_mapping_values() -> None:
    """Test that the transition mapping has the expected destination rounds."""
    expected_values = {
        LiquidityTraderAbci.FetchStrategiesRound,
        TxSettlementAbci.RandomnessTransactionSubmissionRound,
        ResetAndPauseAbci.ResetAndPauseRound,
        LiquidityTraderAbci.PostTxSettlementRound,
        RegistrationAbci.RegistrationRound,
        # mech_interact_abci legs (new staking regime)
        MechVersionStates.MechVersionDetectionRound,
        MechRequestStates.MechRequestRound,
        MechResponseStates.MechResponseRound,
        LiquidityTraderAbci.CheckStakingKPIMetRound,
        LiquidityTraderAbci.GetPositionsRound,
    }
    assert set(abci_app_transition_mapping.values()) == expected_values


def test_abci_app_transition_mapping_specific_transitions() -> None:
    """Test specific transitions in the mapping."""
    assert (
        abci_app_transition_mapping[RegistrationAbci.FinishedRegistrationRound]
        == LiquidityTraderAbci.FetchStrategiesRound
    )
    assert (
        abci_app_transition_mapping[LiquidityTraderAbci.FinishedDecisionMakingRound]
        == ResetAndPauseAbci.ResetAndPauseRound
    )
    assert (
        abci_app_transition_mapping[TxSettlementAbci.FinishedTransactionSubmissionRound]
        == LiquidityTraderAbci.PostTxSettlementRound
    )
    assert (
        abci_app_transition_mapping[TxSettlementAbci.FailedRound]
        == ResetAndPauseAbci.ResetAndPauseRound
    )
    assert (
        abci_app_transition_mapping[ResetAndPauseAbci.FinishedResetAndPauseRound]
        == LiquidityTraderAbci.FetchStrategiesRound
    )
    assert (
        abci_app_transition_mapping[ResetAndPauseAbci.FinishedResetAndPauseErrorRound]
        == RegistrationAbci.RegistrationRound
    )
    # Off-chain mech edges. Pin each target individually — a flat
    # value-set assertion would let a silent swap (e.g. routing
    # FinishedWithOffchainMechDepositSettledRound to MechResponseRound
    # instead of MechRequestRound) pass, which would break the
    # _retry_pending leg the executor depends on after deposit settlement.
    assert (
        abci_app_transition_mapping[
            LiquidityTraderAbci.FinishedWithOffchainMechDepositSettledRound
        ]
        == MechRequestStates.MechRequestRound
    )
    assert (
        abci_app_transition_mapping[MechFinalStates.FinishedOffchainMechRequestRound]
        == MechResponseStates.MechResponseRound
    )
    assert (
        abci_app_transition_mapping[
            MechFinalStates.FinishedOffchainMechDepositNeededRound
        ]
        == TxSettlementAbci.RandomnessTransactionSubmissionRound
    )
    assert (
        abci_app_transition_mapping[MechFinalStates.FailedOffchainMechRequestRound]
        == LiquidityTraderAbci.CheckStakingKPIMetRound
    )


def test_termination_config() -> None:
    """Test that the termination config is correctly defined."""
    assert isinstance(termination_config, BackgroundAppConfig)
    assert termination_config.round_cls == BackgroundRound
    assert termination_config.start_event == Event.TERMINATE
    assert termination_config.abci_app == TerminationAbciApp


def test_optimus_abci_app_is_chained() -> None:
    """Test that OptimusAbciApp is composed from the expected apps."""
    # The OptimusAbciApp should have transition_function defined
    assert hasattr(OptimusAbciApp, "transition_function")
    assert len(OptimusAbciApp.transition_function) > 0


def test_optimus_abci_app_has_background_app() -> None:
    """Test that OptimusAbciApp has background apps configured."""
    assert hasattr(OptimusAbciApp, "background_apps")
    assert len(OptimusAbciApp.background_apps) > 0

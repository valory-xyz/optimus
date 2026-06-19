# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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

"""This package contains round behaviours of OptimusAbciApp."""

import packages.valory.skills.liquidity_trader_abci.rounds as LiquidityTraderAbci
import packages.valory.skills.mech_interact_abci.rounds as MechInteractAbci
import packages.valory.skills.mech_interact_abci.states.final_states as MechFinalStates
import packages.valory.skills.mech_interact_abci.states.mech_version as MechVersionStates
import packages.valory.skills.mech_interact_abci.states.request as MechRequestStates
import packages.valory.skills.mech_interact_abci.states.response as MechResponseStates
import packages.valory.skills.registration_abci.rounds as RegistrationAbci
import packages.valory.skills.reset_pause_abci.rounds as ResetAndPauseAbci
import packages.valory.skills.transaction_settlement_abci.rounds as TxSettlementAbci
from packages.valory.skills.abstract_round_abci.abci_app_chain import (
    AbciAppTransitionMapping,
    chain,
)
from packages.valory.skills.abstract_round_abci.base import BackgroundAppConfig
from packages.valory.skills.termination_abci.rounds import (
    BackgroundRound,
    Event,
    TerminationAbciApp,
)

abci_app_transition_mapping: AbciAppTransitionMapping = {
    RegistrationAbci.FinishedRegistrationRound: LiquidityTraderAbci.FetchStrategiesRound,
    LiquidityTraderAbci.FinishedCallCheckpointRound: TxSettlementAbci.RandomnessTransactionSubmissionRound,
    LiquidityTraderAbci.FinishedCheckStakingKPIMetRound: TxSettlementAbci.RandomnessTransactionSubmissionRound,
    LiquidityTraderAbci.FinishedDecisionMakingRound: ResetAndPauseAbci.ResetAndPauseRound,
    LiquidityTraderAbci.FinishedEvaluateStrategyRound: ResetAndPauseAbci.ResetAndPauseRound,
    LiquidityTraderAbci.FinishedTxPreparationRound: TxSettlementAbci.RandomnessTransactionSubmissionRound,
    LiquidityTraderAbci.FailedMultiplexerRound: ResetAndPauseAbci.ResetAndPauseRound,
    TxSettlementAbci.FinishedTransactionSubmissionRound: LiquidityTraderAbci.PostTxSettlementRound,
    TxSettlementAbci.FailedRound: ResetAndPauseAbci.ResetAndPauseRound,
    ResetAndPauseAbci.FinishedResetAndPauseRound: LiquidityTraderAbci.FetchStrategiesRound,
    ResetAndPauseAbci.FinishedResetAndPauseErrorRound: RegistrationAbci.RegistrationRound,
    # --- mech_interact_abci: VersionDetection + Request legs (no Response leg) ---
    # New staking regime: the producer (CheckStakingKPIMetRound) hands off here to
    # build and settle one MechMarketplace.request, which ticks mapRequestCounts.
    LiquidityTraderAbci.FinishedWithMechRequestRound: MechVersionStates.MechVersionDetectionRound,
    # Version detection resolves into the request leg (V1, no-marketplace, or the
    # V2 mech-information path), or retries on a failed information fetch.
    MechFinalStates.FinishedMarketplaceLegacyDetectedRound: MechRequestStates.MechRequestRound,
    MechFinalStates.FinishedMechLegacyDetectedRound: MechRequestStates.MechRequestRound,
    MechFinalStates.FinishedMechInformationRound: MechRequestStates.MechRequestRound,
    MechFinalStates.FailedMechInformationRound: MechVersionStates.MechVersionDetectionRound,
    # The built request tx settles through the normal settlement FSM; afterwards
    # PostTxSettlementRound routes mech_request_round -> FinishedWithMechResponsePollRound.
    MechFinalStates.FinishedMechRequestRound: TxSettlementAbci.RandomnessTransactionSubmissionRound,
    MechFinalStates.FinishedMechPurchaseSubscriptionRound: TxSettlementAbci.RandomnessTransactionSubmissionRound,
    # After the request settles, poll the mech response leg purely so the FSM is
    # valid (every composed round must be reachable) and to confirm the liveness
    # request was serviced; the answer is discarded (no answer-handling round).
    LiquidityTraderAbci.FinishedWithMechResponsePollRound: MechResponseStates.MechResponseRound,
    MechFinalStates.FinishedMechResponseRound: LiquidityTraderAbci.CheckStakingKPIMetRound,
    MechFinalStates.FinishedMechResponseTimeoutRound: LiquidityTraderAbci.CheckStakingKPIMetRound,
    # No request built (should not happen — the producer always injects one):
    # fall back to normal trading.
    MechFinalStates.FinishedMechRequestSkipRound: LiquidityTraderAbci.GetPositionsRound,
}

termination_config = BackgroundAppConfig(
    round_cls=BackgroundRound,
    start_event=Event.TERMINATE,
    abci_app=TerminationAbciApp,
)

OptimusAbciApp = chain(
    (
        RegistrationAbci.AgentRegistrationAbciApp,
        LiquidityTraderAbci.LiquidityTraderAbciApp,
        TxSettlementAbci.TransactionSubmissionAbciApp,
        MechInteractAbci.MechInteractAbciApp,
        ResetAndPauseAbci.ResetPauseAbciApp,
    ),
    abci_app_transition_mapping,
).add_background_app(termination_config)

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

"""This module contains the DegenarateRounds of LiquidityTraderAbciApp."""

from packages.valory.skills.abstract_round_abci.base import DegenerateRound


class FailedMultiplexerRound(DegenerateRound):
    """FailedMultiplexerRound"""


class FinishedCallCheckpointRound(DegenerateRound):
    """FinishedCallCheckpointRound"""


class FinishedCheckStakingKPIMetRound(DegenerateRound):
    """FinishedCheckStakingKPIMetRound"""


class FinishedWithMechRequestRound(DegenerateRound):
    """Producer hand-off to the composed mech_interact_abci legs (new regime).

    Reached from ``CheckStakingKPIMetRound`` on ``MECH_REQUEST_NEEDED``. The
    ``optimus_abci`` composition remaps this degenerate round to
    ``MechVersionDetectionRound`` so the mech request is built and settled.
    """


class FinishedWithMechResponsePollRound(DegenerateRound):
    """Hand-off to mech_interact_abci's response leg after the request settles.

    Reached from ``PostTxSettlementRound`` once the mech-request tx settles
    (``MECH_REQUEST_TX_EXECUTED``). The ``optimus_abci`` composition remaps this
    to ``MechResponseRound``. The response is polled only to keep the FSM valid
    (every composed round must be reachable) and to confirm the liveness request
    was serviced — its content is discarded (no answer-handling round is
    composed). The response round then routes back to ``CheckStakingKPIMetRound``.
    """


class FinishedDecisionMakingRound(DegenerateRound):
    """FinishedDecisionMakingRound"""


class FinishedEvaluateStrategyRound(DegenerateRound):
    """FinishedEvaluateStrategyRound"""


class FinishedTxPreparationRound(DegenerateRound):
    """FinishedTxPreparationRound"""

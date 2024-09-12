# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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

"""This module contains the transaction payloads of the LiquidityTraderAbciApp."""

from dataclasses import dataclass
from typing import Optional

from packages.valory.skills.abstract_round_abci.base import BaseTxPayload


@dataclass(frozen=True)
class CallCheckpointPayload(BaseTxPayload):
    """A transaction payload for the CallCheckpointRound."""

    tx_submitter: str
    service_staking_state: int
    min_num_of_safe_tx_required: Optional[int]
    is_staking_kpi_met: Optional[bool]
    tx_hash: Optional[str]
    safe_contract_address: Optional[str]
    chain_id: Optional[str]


@dataclass(frozen=True)
class CheckStakingKPIMetPayload(BaseTxPayload):
    """A transaction payload for the CheckStakingKPIMetRound."""

    tx_submitter: str
    is_staking_kpi_met: Optional[bool]
    tx_hash: Optional[str]
    safe_contract_address: Optional[str]
    chain_id: Optional[str]


@dataclass(frozen=True)
class GetPositionsPayload(BaseTxPayload):
    """Represent a transaction payload for the GetPositionsRound."""

    positions: Optional[str]


@dataclass(frozen=True)
class EvaluateStrategyPayload(BaseTxPayload):
    """Represent a transaction payload for the EvaluateStrategyRound."""

    actions: Optional[str]


@dataclass(frozen=True)
class DecisionMakingPayload(BaseTxPayload):
    """Represent a transaction payload for the DecisionMakingRound."""

    content: str

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
class MultisigTxPayload(BaseTxPayload):
    """Represents a transaction payload for preparing an on-chain transaction to be sent via the agents' multisig."""

    tx_submitter: str
    tx_hash: Optional[str]
    safe_contract_address: Optional[str]
    chain_id: Optional[str]


@dataclass(frozen=True)
class CallCheckpointPayload(MultisigTxPayload):
    """A transaction payload for the CallCheckpointRound."""

    service_staking_state: int
    min_num_of_safe_tx_required: Optional[int]


@dataclass(frozen=True)
class CheckStakingKPIMetPayload(MultisigTxPayload):
    """A transaction payload for the CheckStakingKPIMetRound."""

    is_staking_kpi_met: Optional[bool]


@dataclass(frozen=True)
class GetPositionsPayload(BaseTxPayload):
    """Represent a transaction payload for the GetPositionsRound."""

    positions: Optional[str]


@dataclass(frozen=True)
class APRPopulationPayload(BaseTxPayload):
    """Represent a transaction payload for the APRPopulationPayload."""

    context: str
    content: Optional[str] = None


@dataclass(frozen=True)
class EvaluateStrategyPayload(BaseTxPayload):
    """Represent a transaction payload for the EvaluateStrategyRound."""

    actions: Optional[str]


@dataclass(frozen=True)
class DecisionMakingPayload(BaseTxPayload):
    """Represent a transaction payload for the DecisionMakingRound."""

    content: str


@dataclass(frozen=True)
class PostTxSettlementPayload(BaseTxPayload):
    """Represent a transaction payload for the PostTxSettlementRound."""

    content: str


@dataclass(frozen=True)
class FetchStrategiesPayload(BaseTxPayload):
    """Represent a transaction payload for the FetchStrategiesRound."""

    content: str


@dataclass(frozen=True)
class WithdrawFundsPayload(BaseTxPayload):
    """Represent a transaction payload for the WithdrawFundsRound."""

    withdrawal_actions: str

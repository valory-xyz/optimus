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

"""This module contains a wrapper for the Aero CL gauge factory contract interface."""

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi

PUBLIC_ID = PublicId.from_str("valory/aero_cl_gauge_factory:0.1.0")


class AeroCLGaugeFactoryContract(Contract):
    """The Aero CL gauge factory contract."""

    contract_id = PUBLIC_ID

    @classmethod
    def effective_penalty_config(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        gauge: str,
    ) -> JSONLike:
        """Get the early-unstake penalty that applies to a CL gauge."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        min_stake_blocks, penalty_rate = contract_instance.functions.effectivePenaltyConfig(
            ledger_api.api.to_checksum_address(gauge)
        ).call()
        return dict(min_stake_blocks=min_stake_blocks, penalty_rate=penalty_rate)

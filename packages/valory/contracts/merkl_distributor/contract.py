# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2022-2023 Valory AG
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

"""This class contains a wrapper for distributor contract interface."""
import logging
from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi
from eth_abi import encode

PUBLIC_ID = PublicId.from_str("valory/merkl_distributor:0.1.0")
_logger = logging.getLogger(
    f"aea.packages.{PUBLIC_ID.author}.contracts.{PUBLIC_ID.name}.contract"
)
class DistributorContract(Contract):
    """The Distributor contract."""

    contract_id = PUBLIC_ID

    @classmethod
    def claim_rewards(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        users: list,
        tokens: list,
        amounts: list,
        proofs: list
    ) -> JSONLike:
        """Prepare a claim rewards transaction."""

        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI(
            "claim",
            args=(
                users,
                tokens,
                amounts,
                proofs
            )
        )
        return {"tx_hash": bytes.fromhex(data[2:])}

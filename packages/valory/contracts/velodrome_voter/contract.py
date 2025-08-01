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

"""This module contains the class to interact with the Velodrome Voter contract."""

from typing import Any, List

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea.crypto.base import LedgerApi


class VelodromeVoterContract(Contract):
    """The Velodrome Voter contract."""

    contract_id = PublicId.from_str("valory/velodrome_voter:0.1.0")

    @classmethod
    def claim_rewards(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        gauges: List[str],
        tokens: List[List[str]],
        **kwargs: Any,
    ) -> JSONLike:
        """
        Batch claim rewards from multiple gauges.

        :param ledger_api: the ledger API object
        :param contract_address: the contract address
        :param gauges: list of gauge addresses
        :param tokens: list of token arrays for each gauge
        :param kwargs: additional keyword arguments
        :return: the raw transaction
        """
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI(
            fn_name="claimRewards",
            args=[gauges, tokens],
        )
        return {"data": data}

    @classmethod
    def claim_bribes(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        bribes: List[str],
        tokens: List[List[str]],
        token_id: int,
        **kwargs: Any,
    ) -> JSONLike:
        """
        Claim bribes for a specific token ID.

        :param ledger_api: the ledger API object
        :param contract_address: the contract address
        :param bribes: list of bribe addresses
        :param tokens: list of token arrays for each bribe
        :param token_id: the NFT token ID
        :param kwargs: additional keyword arguments
        :return: the raw transaction
        """
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI(
            fn_name="claimBribes",
            args=[bribes, tokens, token_id],
        )
        return {"data": data}

    @classmethod
    def claim_fees(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        fees: List[str],
        tokens: List[List[str]],
        token_id: int,
        **kwargs: Any,
    ) -> JSONLike:
        """
        Claim fees for a specific token ID.

        :param ledger_api: the ledger API object
        :param contract_address: the contract address
        :param fees: list of fee addresses
        :param tokens: list of token arrays for each fee
        :param token_id: the NFT token ID
        :param kwargs: additional keyword arguments
        :return: the raw transaction
        """
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI(
            fn_name="claimFees",
            args=[fees, tokens, token_id],
        )
        return {"data": data}

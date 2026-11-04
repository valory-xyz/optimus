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

"""This module contains a wrapper for the Multicall3 contract."""

from typing import List, Tuple

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi

PUBLIC_ID = PublicId.from_str("valory/multicall3:0.1.0")

Call3 = Tuple[str, bytes, bool]


class Multicall3Contract(Contract):
    """The Multicall3 contract."""

    contract_id = PUBLIC_ID

    @classmethod
    def aggregate3(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        calls: List[Call3],
    ) -> JSONLike:
        """Run ``(target, call_data, allow_failure)`` reads in one call; results keep their order."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        encoded_calls = [
            (ledger_api.api.to_checksum_address(target), allow_failure, call_data)
            for target, call_data, allow_failure in calls
        ]
        results = contract_instance.functions.aggregate3(encoded_calls).call()
        return dict(results=[(bool(success), bytes(data)) for success, data in results])

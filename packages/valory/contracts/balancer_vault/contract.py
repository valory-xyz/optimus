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

"""This class contains a wrapper for vault contract interface."""

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi
from eth_abi import encode

PUBLIC_ID = PublicId.from_str("valory/balancer_vault:0.1.0")



class VaultContract(Contract):
    """The Weighted Stable Pool contract."""

    contract_id = PUBLIC_ID

    @classmethod
    def get_pool_tokens(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        pool_id: str,
    ) -> JSONLike:
        """get the balance of the given account."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        pool_id_bytes = bytes.fromhex(pool_id)
        data, _, _ = contract_instance.functions.getPoolTokens(pool_id_bytes).call()
        return dict(tokens=data)


    @classmethod
    def join_pool(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        pool_id: str,
        sender: str,
        recipient: str,
        assets: list,
        max_amounts_in: list,
        join_kind: int,
        from_internal_balance: bool = False,

    ) -> JSONLike:
        """Prepare a join pool transaction."""

        minimum_BPT = 0

        encoded_user_data = encode(
            ['uint256', 'uint256[]', 'uint256'],
            [join_kind, max_amounts_in, minimum_BPT]
        )

        request = (
            assets,
            max_amounts_in,
            encoded_user_data,
            from_internal_balance,
        )

        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI(
            "joinPool",
            args=(
                bytes.fromhex(pool_id[2:]),
                sender,
                recipient,
                request
            )
        )
        return {"tx_hash": bytes.fromhex(data[2:])}
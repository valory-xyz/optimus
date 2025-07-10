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

"""This module contains a wrapper for Balancer Queries contract interface."""

from typing import List

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi
from eth_abi import encode

PUBLIC_ID = PublicId.from_str("valory/balancer_queries:0.1.0")


class BalancerQueriesContract(Contract):
    """Balancer Queries contract wrapper."""

    contract_id = PUBLIC_ID


    @classmethod
    def query_join(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        pool_id: str,
        sender: str,
        recipient: str,
        assets: List[str],
        max_amounts_in: List[int],
        join_kind: int,
        minimum_bpt: int,
        from_internal_balance: bool,
    ) -> JSONLike:
        """Query join operation to get expected BPT output - view function, no gas cost."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        
        encoded_user_data = encode(
            ["uint256", "uint256[]", "uint256"],
            [join_kind, max_amounts_in, minimum_bpt],
        )
        
        # Prepare join request struct
        join_request = (
            assets,
            max_amounts_in,
            encoded_user_data,
            from_internal_balance,
        )
        
        # Call queryJoin - this is a view function
        result = contract_instance.functions.queryJoin(
            bytes.fromhex(pool_id[2:]),
            sender,
            recipient,
            join_request
        ).call()
        
        # Result contains (bptOut, amountsIn)
        return {"result":{"bpt_out":result[0], "amounts_in":result[1]}}

    @classmethod
    def query_exit(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        pool_id: str,
        sender: str,
        recipient: str,
        assets: List[str],
        min_amounts_out: List[int],
        exit_kind: int,
        bpt_amount_in: int,
        to_internal_balance: bool,
    ) -> JSONLike:
        """Query exit operation to get expected token amounts - view function, no gas cost."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        
        encoded_user_data = encode(["uint256", "uint256"], [exit_kind, bpt_amount_in])
        
        # Prepare exit request struct
        exit_request = (
            assets,
            min_amounts_out,
            encoded_user_data,
            to_internal_balance,
        )
        
        # Call queryExit - this is a view function
        result = contract_instance.functions.queryExit(
            bytes.fromhex(pool_id[2:]),
            sender,
            recipient,
            exit_request
        ).call()
        

        return {"result":{"bpt_in":result[0], "amounts_out":result[1]}}

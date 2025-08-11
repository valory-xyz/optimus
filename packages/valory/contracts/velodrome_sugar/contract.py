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

"""This module contains the class to interact with the Velodrome Sugar contract."""

from typing import Any, Dict, List, Tuple

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea.crypto.base import LedgerApi


class VelodromeSugarContract(Contract):
    """The Velodrome Sugar contract."""

    contract_id = PublicId.from_str("valory/velodrome_sugar:0.1.0")


    @classmethod
    def positions(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        limit: int,
        offset: int,
        account: str,
    ) -> JSONLike:
        """Get user positions and rewards data using Velodrome Sugar contract."""
        checksumed_account = ledger_api.api.to_checksum_address(account)
        contract_instance = cls.get_instance(ledger_api, contract_address)
        result = contract_instance.functions.positions(
            limit, offset, checksumed_account
        ).call()
        
        # Convert the result to a more readable format
        positions = []
        for position in result:
            position_data = {
                "id": position[0],
                "lp": position[1],
                "liquidity": position[2],
                "staked": position[3],
                "amount0": position[4],
                "amount1": position[5],
                "staked0": position[6],
                "staked1": position[7],
                "unstaked_earned0": position[8],
                "unstaked_earned1": position[9],
                "emissions_earned": position[10],
                "tick_lower": position[11],
                "tick_upper": position[12],
                "sqrt_ratio_lower": position[13],
                "sqrt_ratio_upper": position[14],
                "alm": position[15],
            }
            positions.append(position_data)
        
        return {"positions": positions}

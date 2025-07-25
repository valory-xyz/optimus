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
    def principal(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        position_manager: str,
        token_id: int,
        sqrt_price_x96: int,
    ) -> JSONLike:
        """Get position principal using Velodrome Sugar contract."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        result = contract_instance.functions.principal(
            position_manager, token_id, sqrt_price_x96
        ).call()
        return {"amounts": result}

    @classmethod
    def get_amounts_for_liquidity(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        sqrt_price_x96: int,
        sqrt_ratio_a_x96: int,
        sqrt_ratio_b_x96: int,
        liquidity: int,
    ) -> JSONLike:
        """Get amounts for liquidity using Velodrome Sugar contract."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        result = contract_instance.functions.getAmountsForLiquidity(
            sqrt_price_x96, sqrt_ratio_a_x96, sqrt_ratio_b_x96, liquidity
        ).call()
        return {"amounts": result}

    @classmethod
    def get_sqrt_ratio_at_tick(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        tick: int,
    ) -> JSONLike:
        """Get sqrt ratio at tick using Velodrome Sugar contract."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        result = contract_instance.functions.getSqrtRatioAtTick(tick).call()
        return {"sqrt_ratio": result}

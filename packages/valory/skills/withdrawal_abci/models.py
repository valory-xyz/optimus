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

"""This module contains the models for the 'withdrawal_abci' skill."""

from typing import Any, Dict, Optional

from aea.skills.base import Model


class WithdrawalParams(Model):
    """Model for withdrawal parameters."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the withdrawal parameters."""
        super().__init__(**kwargs)
        self.target_investment_chains = kwargs.get("target_investment_chains", ["optimism"])
        self.safe_contract_addresses = kwargs.get("safe_contract_addresses", {})
        self.usdc_address = kwargs.get("usdc_address", "0xA0b86a33E6441b8c4C8C8C8C8C8C8C8C8C8C8C8C")


class WithdrawalSharedState(Model):
    """Model for withdrawal shared state."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the withdrawal shared state."""
        super().__init__(**kwargs)
        self.withdrawal_id: Optional[str] = None
        self.withdrawal_status: Optional[str] = None
        self.withdrawal_message: Optional[str] = None
        self.withdrawal_target_address: Optional[str] = None
        self.withdrawal_chain: Optional[str] = None
        self.withdrawal_safe_address: Optional[str] = None
        self.withdrawal_tx_link: Optional[str] = None
        self.portfolio_data: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Set a value in the shared state."""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the shared state."""
        return getattr(self, key, default) 
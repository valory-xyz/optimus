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
"""This module contains the transaction payloads for the liquidity trader abci."""

from datetime import datetime
from typing import Dict, Type

import pytest

from packages.valory.skills.abstract_round_abci.base import BaseTxPayload
from packages.valory.skills.liquidity_trader_abci.payloads import (
    CallCheckpointPayload,
    CheckStakingKPIMetPayload,
    DecisionMakingPayload,
    EvaluateStrategyPayload,
    GetPositionsPayload,
    PostTxSettlementPayload,
)


@pytest.mark.parametrize(
    "payload_class, payload_kwargs",
    [
        (
            CallCheckpointPayload,
            {
               service_staking_state: 1
               min_num_of_safe_tx_required: 1
            },
        ),
        (
            CheckStakingKPIMetPayload,
            {
                is_staking_kpi_met: True 
            },
        ),
        (
            DecisionMakingPayload,
            {
               content: "content"
            },
        ),
        (
            EvaluateStrategyPayload,
            {
                actions: "dummy actions"
            },
        ),
        (
            GetPositionsPayload,
            {
                positions: "dummy position"
            },
        ),
        (
            PostTxSettlementPayload,
            {
                content: "content"
            },
        ),
    ],
)

def test_payload(payload_class: Type[BaseTxPayload], payload_kwargs: Dict) -> None:
    """Test payloads."""
    payload = payload_class(sender="sender", **payload_kwargs)

    for key, value in payload_kwargs.items():
        assert getattr(payload, key) == value

    assert payload.sender == "sender"
    assert payload.data == payload_kwargs
    assert payload_class.from_json(payload.json) == payload

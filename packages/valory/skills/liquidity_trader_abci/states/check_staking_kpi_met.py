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

"""This module contains the CheckStakingKPIMetRound of LiquidityTraderAbciApp."""

from typing import Optional, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    get_name,
)
from packages.valory.skills.liquidity_trader_abci.payloads import (
    CheckStakingKPIMetPayload,
)
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)


class CheckStakingKPIMetRound(CollectSameUntilThresholdRound):
    """CheckStakingKPIMetRound"""

    payload_class = CheckStakingKPIMetPayload
    synchronized_data_class = SynchronizedData
    done_event: Event = Event.DONE
    no_majority_event: Event = Event.NO_MAJORITY
    none_event: Event = Event.NONE
    collection_key = get_name(SynchronizedData.participant_to_staking_kpi)
    selection_key = (
        get_name(SynchronizedData.tx_submitter),
        get_name(SynchronizedData.most_voted_tx_hash),
        get_name(SynchronizedData.safe_contract_address),
        get_name(SynchronizedData.chain_id),
        get_name(SynchronizedData.is_staking_kpi_met),
    )

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        res = super().end_block()
        if res is None:
            return None

        synced_data, event = cast(Tuple[SynchronizedData, Event], res)

        # Check if this is a withdrawal initiation event
        if event == Event.WITHDRAWAL_INITIATED:
            return synced_data, Event.WITHDRAWAL_INITIATED

        if event != Event.DONE:
            return res

        if synced_data.is_staking_kpi_met is None:
            return synced_data, Event.ERROR
        if synced_data.most_voted_tx_hash is not None:
            return synced_data, Event.SETTLE
        if synced_data.is_staking_kpi_met is True:
            return synced_data, Event.STAKING_KPI_MET
        if synced_data.is_staking_kpi_met is False:
            return synced_data, Event.STAKING_KPI_NOT_MET

        return res

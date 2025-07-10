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

"""This module contains the PostTxSettlementRound of LiquidityTraderAbciApp."""

import json
from typing import Dict, Optional, Tuple

from packages.valory.skills.abstract_round_abci.base import (
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    get_name,
)
from packages.valory.skills.liquidity_trader_abci.payloads import (
    PostTxSettlementPayload,
)
from packages.valory.skills.liquidity_trader_abci.states.base import (
    Event,
    SynchronizedData,
)
from packages.valory.skills.liquidity_trader_abci.states.call_checkpoint import (
    CallCheckpointRound,
)
from packages.valory.skills.liquidity_trader_abci.states.check_staking_kpi_met import (
    CheckStakingKPIMetRound,
)
from packages.valory.skills.liquidity_trader_abci.states.decision_making import (
    DecisionMakingRound,
)
from packages.valory.skills.liquidity_trader_abci.states.fetch_strategies import (
    FetchStrategiesRound,
)


class PostTxSettlementRound(CollectSameUntilThresholdRound):
    """A round that will be called after tx settlement is done."""

    payload_class = PostTxSettlementPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    none_event: Event = Event.NONE
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_post_tx_settlement)
    selection_key = (get_name(SynchronizedData.chain_id),)

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            # Parse event from payload content
            event = self._parse_event_from_payload()

            # Handle regular events
            submitter_to_event: Dict[str, Event] = {
                CallCheckpointRound.auto_round_id(): Event.CHECKPOINT_TX_EXECUTED,
                CheckStakingKPIMetRound.auto_round_id(): Event.VANITY_TX_EXECUTED,
                DecisionMakingRound.auto_round_id(): Event.ACTION_EXECUTED,
                FetchStrategiesRound.auto_round_id(): Event.TRANSFER_COMPLETED,
            }

            synced_data = SynchronizedData(self.synchronized_data.db)
            event = submitter_to_event.get(synced_data.tx_submitter, Event.UNRECOGNIZED)

            if event == Event.CHECKPOINT_TX_EXECUTED:
                synced_data = synced_data.update(
                    synchronized_data_class=SynchronizedData,
                    period_number_at_last_cp=synced_data.period_count,
                )

            return synced_data, event

    def _parse_event_from_payload(self) -> str:
        """Parse event from payload content."""
        try:
            # Get the first payload to extract event
            for payload in self.collection.values():
                if payload.content:
                    content_data = json.loads(payload.content)
                    return content_data.get("event", "TRANSFER_COMPLETED")
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass

        # Fallback to default event
        return "TRANSFER_COMPLETED"

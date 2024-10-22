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

"""This module contains the base functionality for the rounds of the decision-making abci app."""

from enum import Enum
from typing import Optional, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    CollectionRound,
    DeserializedCollection,
)
from packages.valory.skills.market_data_fetcher_abci.rounds import (
    SynchronizedData as MarketFetcherSyncedData,
)
from packages.valory.skills.portfolio_tracker_abci.rounds import (
    SynchronizedData as PortfolioTrackerSyncedData,
)
from packages.valory.skills.strategy_evaluator_abci.payloads import IPFSHashPayload
from packages.valory.skills.trader_decision_maker_abci.rounds import (
    SynchronizedData as DecisionMakerSyncedData,
)
from packages.valory.skills.transaction_settlement_abci.rounds import (
    SynchronizedData as TxSettlementSyncedData,
)


class Event(Enum):
    """Event enumeration for the price estimation demo."""

    NO_ORDERS = "no_orders"
    PREPARE_SWAP = "prepare_swap"
    BACKTEST_POSITIVE_PROXY_SERVER = "prepare_swap_proxy_server"
    BACKTEST_POSITIVE_EVM = "prepare_swap_evm"
    PREPARE_INCOMPLETE_SWAP = "prepare_incomplete_swap"
    ERROR_PREPARING_SWAPS = "error_preparing_swaps"
    NO_INSTRUCTIONS = "no_instructions"
    INSTRUCTIONS_PREPARED = "instructions_prepared"
    TRANSACTION_PREPARED = "transaction_prepared"
    INCOMPLETE_INSTRUCTIONS_PREPARED = "incomplete_instructions_prepared"
    ERROR_PREPARING_INSTRUCTIONS = "error_preparing_instructions"
    SWAP_TX_PREPARED = "swap_tx_prepared"
    SWAPS_QUEUE_EMPTY = "swaps_queue_empty"
    TX_PREPARATION_FAILED = "none"
    PROXY_SWAPPED = "proxy_swapped"
    PROXY_SWAP_FAILED = "proxy_swap_failed"
    BACKTEST_POSITIVE = "backtest_succeeded"
    BACKTEST_NEGATIVE = "backtest_negative"
    BACKTEST_FAILED = "backtest_failed"
    ERROR_BACKTESTING = "error_backtesting"
    ROUND_TIMEOUT = "round_timeout"
    PROXY_SWAP_TIMEOUT = "proxy_swap_timeout"
    NO_MAJORITY = "no_majority"


class SynchronizedData(
    DecisionMakerSyncedData,
    MarketFetcherSyncedData,
    PortfolioTrackerSyncedData,
    TxSettlementSyncedData,
):
    """Class to represent the synchronized data.

    This data is replicated by the tendermint application.
    """

    def _optional_str(self, db_key: str) -> Optional[str]:
        """Get an optional string from the db."""
        val = self.db.get_strict(db_key)
        if val is None:
            return None
        return str(val)

    def _get_deserialized(self, key: str) -> DeserializedCollection:
        """Strictly get a collection and return it deserialized."""
        serialized = self.db.get_strict(key)
        return CollectionRound.deserialize_collection(serialized)

    @property
    def orders_hash(self) -> Optional[str]:
        """Get the hash of the orders' data."""
        return self._optional_str("orders_hash")

    @property
    def backtested_orders_hash(self) -> Optional[str]:
        """Get the hash of the backtested orders' data."""
        return self._optional_str("backtested_orders_hash")

    @property
    def incomplete_exec(self) -> bool:
        """Get whether the strategies did not complete successfully."""
        return bool(self.db.get_strict("incomplete_exec"))

    @property
    def tx_id(self) -> str:
        """Get the transaction's id."""
        return str(self.db.get_strict("tx_id"))

    @property
    def instructions_hash(self) -> Optional[str]:
        """Get the hash of the instructions' data."""
        return self._optional_str("instructions_hash")

    @property
    def incomplete_instructions(self) -> bool:
        """Get whether the instructions were not built for all the swaps."""
        return bool(self.db.get_strict("incomplete_instructions"))

    @property
    def participant_to_orders(self) -> DeserializedCollection:
        """Get the participants to orders."""
        return self._get_deserialized("participant_to_orders")

    @property
    def participant_to_instructions(self) -> DeserializedCollection:
        """Get the participants to swap(s) instructions."""
        return self._get_deserialized("participant_to_instructions")

    @property
    def participant_to_tx_preparation(self) -> DeserializedCollection:
        """Get the participants to the next swap's tx preparation."""
        return self._get_deserialized("participant_to_tx_preparation")

    @property
    def participant_to_backtesting(self) -> DeserializedCollection:
        """Get the participants to the backtesting."""
        return self._get_deserialized("participant_to_backtesting")


class IPFSRound(CollectSameUntilThresholdRound):
    """A round for sending data to IPFS and storing the returned hash."""

    payload_class = IPFSHashPayload
    synchronized_data_class = SynchronizedData
    incomplete_event: Event
    no_hash_event: Event
    no_majority_event = Event.NO_MAJORITY

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        res = super().end_block()
        if res is None:
            return None

        synced_data, event = cast(Tuple[SynchronizedData, Enum], res)
        if event == self.done_event:
            return synced_data, self.get_swap_event(synced_data)
        return synced_data, event

    def get_swap_event(self, synced_data: SynchronizedData) -> Enum:
        """Get the swap event based on the synchronized data."""
        if not isinstance(self.selection_key, tuple) or len(self.selection_key) != 2:
            raise ValueError(
                f"The default implementation of `get_swap_event` for {self.__class__!r} "
                "only supports two selection keys. "
                "Please override the method to match the intended logic."
            )

        hash_db_key, incomplete_db_key = self.selection_key
        if getattr(synced_data, hash_db_key) is None:
            return self.no_hash_event
        if getattr(synced_data, incomplete_db_key):
            return self.incomplete_event
        return self.done_event

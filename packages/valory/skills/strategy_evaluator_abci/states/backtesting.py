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

"""This module contains the backtesting state of the swap(s)."""

from typing import Any

from packages.valory.skills.abstract_round_abci.base import get_name
from packages.valory.skills.strategy_evaluator_abci.states.base import (
    Event,
    IPFSRound,
    SynchronizedData,
)


class BacktestRound(IPFSRound):
    """A round in which the agents prepare swap(s) instructions."""

    done_event = Event.BACKTEST_POSITIVE
    incomplete_event = Event.BACKTEST_FAILED
    no_hash_event = Event.ERROR_BACKTESTING
    none_event = Event.BACKTEST_NEGATIVE
    selection_key = (
        get_name(SynchronizedData.backtested_orders_hash),
        get_name(SynchronizedData.incomplete_exec),
    )
    collection_key = get_name(SynchronizedData.participant_to_backtesting)

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the strategy execution round."""
        super().__init__(*args, **kwargs)
        if self.context.params.use_proxy_server:
            self.done_event = Event.BACKTEST_POSITIVE_PROXY_SERVER
        # Note, using evm takes precedence over proxy server
        if not self.context.params.use_solana:
            self.done_event = Event.BACKTEST_POSITIVE_EVM

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

"""This module contains the final states of the strategy evaluator abci app."""

from packages.valory.skills.abstract_round_abci.base import DegenerateRound


class SwapTxPreparedRound(DegenerateRound):
    """A round representing that the strategy evaluator has prepared swap(s) transaction."""


class NoMoreSwapsRound(DegenerateRound):
    """A round representing that the strategy evaluator has no more swap transactions to prepare."""


class HodlRound(DegenerateRound):
    """A round representing that the strategy evaluator has not prepared any swap transactions."""


class StrategyExecutionFailedRound(DegenerateRound):
    """A round representing that the strategy evaluator has failed to execute the strategy."""


class InstructionPreparationFailedRound(DegenerateRound):
    """A round representing that the strategy evaluator has failed to prepare the instructions for the swaps."""


class BacktestingNegativeRound(DegenerateRound):
    """A round representing that the backtesting has returned with a negative result."""


class BacktestingFailedRound(DegenerateRound):
    """A round representing that the backtesting has failed to run."""


# TODO use this in portfolio tracker
# class RefillRequiredRound(DegenerateRound):
#     """A round representing that a refill is required for swapping."""

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

"""This module contains the behaviour for backtesting the swap(s)."""

from typing import Any, Dict, Generator, List, Optional, Tuple, cast

from packages.valory.skills.abstract_round_abci.io_.store import SupportedFiletype
from packages.valory.skills.strategy_evaluator_abci.behaviours.base import (
    CALLABLE_KEY,
    STRATEGY_KEY,
    StrategyEvaluatorBaseBehaviour,
)
from packages.valory.skills.strategy_evaluator_abci.behaviours.strategy_exec import (
    OUTPUT_MINT,
    TRANSFORMED_PRICE_DATA_KEY,
)
from packages.valory.skills.strategy_evaluator_abci.states.backtesting import (
    BacktestRound,
)


EVALUATE_CALLABLE_KEY = "evaluate_callable"
ASSET_KEY = "asset"
BACKTEST_RESULT_KEY = "sharpe_ratio"


class BacktestBehaviour(StrategyEvaluatorBaseBehaviour):
    """A behaviour in which the agents backtest the swap(s)."""

    matching_round = BacktestRound

    def backtest(self, transformed_data: Dict[str, Any], output_mint: str) -> bool:
        """Backtest the given token and return whether the sharpe ratio is greater than the threshold."""
        token_data = transformed_data.get(output_mint, None)
        if token_data is None:
            self.context.logger.error(
                f"No data were found in the fetched transformed data for token {output_mint!r}."
            )
            return False

        # the following are always passed to a strategy script, which may choose to ignore any
        kwargs: Dict[str, Any] = self.params.strategies_kwargs
        kwargs.update(
            {
                STRATEGY_KEY: self.synchronized_data.selected_strategy,
                CALLABLE_KEY: EVALUATE_CALLABLE_KEY,
                # TODO it is not clear which asset's data we should pass here
                #  shouldn't the evaluate method take both input and output token's data into account?
                TRANSFORMED_PRICE_DATA_KEY: token_data,
                ASSET_KEY: output_mint,
            }
        )
        results = self.execute_strategy_callable(**kwargs)
        if results is None:
            self.context.logger.error(
                f"Something went wrong while backtesting token {output_mint!r}."
            )
            return False
        self.log_from_strategy_results(results)
        sharpe: Optional[float] = results.get(BACKTEST_RESULT_KEY, None)
        if sharpe is None or not isinstance(sharpe, float):
            self.context.logger.error(
                f"No float sharpe value can be extracted using key {BACKTEST_RESULT_KEY!r} in strategy's {results=}."
            )
            return False

        self.context.logger.info(f"{sharpe=}.")
        return sharpe >= self.params.sharpe_threshold

    def filter_orders(
        self, orders: List[Dict[str, str]]
    ) -> Generator[None, None, Tuple[List[Dict[str, str]], bool]]:
        """Backtest the swap(s) and decide whether we should proceed to perform them or not."""
        transformed_data = yield from self.get_from_ipfs(
            self.synchronized_data.transformed_data_hash, SupportedFiletype.JSON
        )
        transformed_data = cast(Optional[Dict[str, Any]], transformed_data)
        if transformed_data is None:
            self.context.logger.error("Could not get the transformed data from IPFS.")
            # return empty orders and incomplete status, because the transformed data are necessary for the backtesting
            return [], True

        self.context.logger.info(
            f"Using trading strategy {self.synchronized_data.selected_strategy!r} for backtesting..."
        )

        success_orders = []
        incomplete = False
        for order in orders:
            token = order.get(OUTPUT_MINT, None)
            if token is None:
                err = f"{OUTPUT_MINT!r} key was not found in {order=}."
                self.context.logger.error(err)
                incomplete = True
                continue

            backtest_passed = self.backtest(transformed_data, token)
            if backtest_passed:
                success_orders.append(order)
                continue

            incomplete = True

        if len(success_orders) == 0:
            incomplete = True

        return success_orders, incomplete

    def async_act(self) -> Generator:
        """Do the action."""
        yield from self.get_process_store_act(
            self.synchronized_data.orders_hash,
            self.filter_orders,
            str(self.swap_decision_filepath),
        )

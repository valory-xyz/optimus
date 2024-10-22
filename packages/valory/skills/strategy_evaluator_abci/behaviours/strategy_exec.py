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

"""This module contains the behaviour for executing a strategy."""

from typing import Any, Dict, Generator, List, Optional, Tuple, cast

from packages.valory.skills.abstract_round_abci.io_.store import SupportedFiletype
from packages.valory.skills.portfolio_tracker_abci.behaviours import SOL_ADDRESS
from packages.valory.skills.strategy_evaluator_abci.behaviours.base import (
    CALLABLE_KEY,
    StrategyEvaluatorBaseBehaviour,
)
from packages.valory.skills.strategy_evaluator_abci.models import AMOUNT_PARAM
from packages.valory.skills.strategy_evaluator_abci.states.strategy_exec import (
    StrategyExecRound,
)


STRATEGY_KEY = "trading_strategy"
PRICE_DATA_KEY = "price_data"
TRANSFORMED_PRICE_DATA_KEY = "transformed_data"
TOKEN_ID_KEY = "token_id"  # nosec B105:hardcoded_password_string
PORTFOLIO_DATA_KEY = "portfolio_data"
SWAP_DECISION_FIELD = "signal"
BUY_DECISION = "buy"
SELL_DECISION = "sell"
HODL_DECISION = "hold"
AVAILABLE_DECISIONS = (BUY_DECISION, SELL_DECISION, HODL_DECISION)
NO_SWAP_DECISION = {SWAP_DECISION_FIELD: HODL_DECISION}
SOL = "SOL"
RUN_CALLABLE_KEY = "run_callable"
INPUT_MINT = "inputMint"
OUTPUT_MINT = "outputMint"


class StrategyExecBehaviour(StrategyEvaluatorBaseBehaviour):
    """A behaviour in which the agents execute the selected strategy and decide on the swap(s)."""

    matching_round = StrategyExecRound

    def __init__(self, **kwargs: Any):
        """Initialize the behaviour."""
        super().__init__(**kwargs)
        self.sol_balance: int = 0
        self.sol_balance_after_swaps: int = 0

    def get_swap_amount(self) -> int:
        """Get the swap amount."""
        if self.params.use_proxy_server:
            api = self.context.tx_settlement_proxy
        else:
            api = self.context.swap_quotes

        return api.parameters.get(AMOUNT_PARAM, 0)

    def is_balance_sufficient(
        self,
        token: str,
        token_balance: int,
    ) -> Optional[bool]:
        """Check whether the balance of the given token is enough to perform the swap transaction."""
        if token == SOL_ADDRESS and self.sol_balance_after_swaps <= 0:
            warning = "Preceding trades are expected to use up all the SOL. Not taking any action."
            self.context.logger.warning(warning)
            return False

        swap_cost = self.params.expected_swap_tx_cost
        # we set it to `None` if no swaps have been prepared yet
        sol_before_swap = (
            None
            if self.sol_balance_after_swaps == self.sol_balance
            else self.sol_balance_after_swaps
        )
        if swap_cost > self.sol_balance_after_swaps:
            self.context.logger.warning(
                "There is not enough SOL to cover the expected swap tx's cost. "
                f"SOL balance after preceding swaps ({self.sol_balance_after_swaps}) < swap cost ({swap_cost}). "
                f"Not taking any actions."
            )
            return False
        self.sol_balance_after_swaps -= swap_cost

        swap_amount = self.get_swap_amount()
        if token == SOL_ADDRESS:
            # do not use the SOL's address to simplify the log messages
            token = SOL
            compared_balance = self.sol_balance_after_swaps
            self.sol_balance_after_swaps -= swap_amount
        else:
            compared_balance = token_balance

        self.context.logger.info(f"Balance ({token}): {token_balance}.")
        if swap_amount > compared_balance:
            warning = (
                f"There is not enough balance to cover the swap amount ({swap_amount}) "
            )
            if token == SOL:
                # subtract the SOL we'd have before this swap or the token's balance if there are no preceding swaps
                preceding_swaps_amount = token_balance - (
                    sol_before_swap or token_balance
                )

                warning += (
                    f"plus the expected swap tx's cost ({swap_cost}) ["
                    f"also taking into account preceding swaps' amount ({preceding_swaps_amount})] "
                )
                # the swap's cost which was subtracted during the first `get_native_balance` call
                # should be included in the swap amount
                self.sol_balance_after_swaps += swap_amount
                swap_amount += swap_cost
                token_balance -= preceding_swaps_amount
            self.sol_balance_after_swaps += swap_cost
            warning += f"({token_balance} < {swap_amount}) for {token!r}. Not taking any actions."
            self.context.logger.warning(warning)
            return False

        self.context.logger.info("Balance is sufficient.")
        return True

    def get_swap_decision(
        self,
        token_data: Any,
        portfolio_data: Dict[str, int],
        token: str,
    ) -> Optional[str]:
        """Get the swap decision given a token's data."""
        strategy = self.synchronized_data.selected_strategy
        self.context.logger.info(f"Using trading strategy {strategy!r}.")
        # the following are always passed to a strategy script, which may choose to ignore any
        kwargs: Dict[str, Any] = self.params.strategies_kwargs
        kwargs.update(
            {
                STRATEGY_KEY: strategy,
                CALLABLE_KEY: RUN_CALLABLE_KEY,
                TRANSFORMED_PRICE_DATA_KEY: token_data,
                PORTFOLIO_DATA_KEY: portfolio_data,
                TOKEN_ID_KEY: token,
            }
        )
        results = self.execute_strategy_callable(**kwargs)
        if results is None:
            results = NO_SWAP_DECISION

        self.log_from_strategy_results(results)
        decision = results.get(SWAP_DECISION_FIELD, None)
        if decision is None:
            self.context.logger.error(
                f"Required field {SWAP_DECISION_FIELD!r} was not returned by {strategy} strategy."
                "Not taking any actions."
            )
        if decision not in AVAILABLE_DECISIONS:
            self.context.logger.error(
                f"Invalid decision {decision!r} was detected! Expected one of {AVAILABLE_DECISIONS}."
                "Not taking any actions."
            )
            decision = None

        return decision

    def get_token_swap_position(self, decision: str) -> Optional[str]:
        """Get the position of the non-native token in the swap operation."""
        token_swap_position = None

        if decision == BUY_DECISION:
            token_swap_position = OUTPUT_MINT
        elif decision == SELL_DECISION:
            token_swap_position = INPUT_MINT
        elif decision != HODL_DECISION:
            self.context.logger.error(
                f"Unrecognised decision {decision!r} found! Expected one of {AVAILABLE_DECISIONS}."
            )

        return token_swap_position

    def get_solana_orders(
        self, token_data: Dict[str, Any]
    ) -> Generator[None, None, Tuple[List[Dict[str, str]], bool]]:
        """Get a mapping from a string indicating whether to buy or sell, to a list of tokens."""
        portfolio = yield from self.get_from_ipfs(
            self.synchronized_data.portfolio_hash, SupportedFiletype.JSON
        )
        portfolio = cast(Optional[Dict[str, int]], portfolio)
        if portfolio is None:
            self.context.logger.error("Could not get the portfolio from IPFS.")
            # return empty orders and incomplete status, because the portfolio is necessary for all the swaps
            return [], True

        sol_balance = portfolio.get(SOL_ADDRESS, None)
        if sol_balance is None:
            err = "The portfolio data do not contain any information for SOL."
            self.context.logger.error(err)
            # return empty orders and incomplete status, because SOL are necessary for all the swaps
            return [], True
        self.sol_balance = self.sol_balance_after_swaps = sol_balance

        orders: List[Dict[str, str]] = []
        incomplete = False
        for token, data in token_data.items():
            if token == SOL_ADDRESS:
                continue

            decision = self.get_swap_decision(data, portfolio, token)
            if decision is None:
                incomplete = True
                continue

            msg = f"Decided to {decision} token with address {token!r}."
            self.context.logger.info(msg)
            quote_data = {INPUT_MINT: SOL_ADDRESS, OUTPUT_MINT: SOL_ADDRESS}
            token_swap_position = self.get_token_swap_position(decision)
            if token_swap_position is None:
                # holding token, no tx to perform
                continue

            quote_data[token_swap_position] = token
            input_token = quote_data[INPUT_MINT]
            if input_token is not SOL:
                token_balance = portfolio.get(input_token, None)
                if token_balance is None:
                    err = f"The portfolio data do not contain any information for {token!r}."
                    self.context.logger.error(err)
                    # return, because a swap for another token might be performed
                    continue
            else:
                token_balance = self.sol_balance

            enough_tokens = self.is_balance_sufficient(input_token, token_balance)
            if not enough_tokens:
                incomplete = True
                continue
            orders.append(quote_data)

        # we only yield here to convert this method to a generator, so that it can be used by `get_process_store_act`
        yield
        return orders, incomplete

    def get_evm_orders(
        self,
        token_data: Dict[str, Any],
        required_amount: int = 0.00001,
    ) -> Generator[None, None, Tuple[List[Dict[str, str]], bool]]:
        """Get a mapping from a string indicating whether to buy or sell, to a list of tokens."""
        # We need to check if the portfolio contains any information for the NATIVE_TOKEN which is yet to be defined
        # We will temporarily skip this check
        # TODO: Define NATIVE_TOKEN, BASE_TOKEN, and LEDGER_ID
        # TODO: Check if the portfolio contains any information for the NATIVE_TOKEN
        # TODO: Mapping of ledger id to base token
        # TODO: update evm balance checker to include native token balance in the portfolio.

        portfolio: Optional[Dict[str, int]] = yield from self.get_from_ipfs(  # type: ignore
            self.synchronized_data.portfolio_hash, SupportedFiletype.JSON
        )
        ledger_id: str = self.params.ledger_ids[0]
        base_token = self.params.base_tokens.get(ledger_id, None)
        if base_token is None:
            self.context.logger.error(
                f"Could not get the base token for ledger {ledger_id!r} from the configuration."
            )
            return [], True
        native_token = self.params.native_currencies.get(ledger_id, None)
        if native_token is None:
            self.context.logger.error(
                f"Could not get the native token for ledger {ledger_id!r} from the configuration."
            )
            return [], True

        if portfolio is None:
            self.context.logger.error("Could not get the portfolio from IPFS.")
            # return empty orders and incomplete status, because the portfolio is necessary for all the swaps
            return [], True

        native_balance = portfolio.get(native_token, None)
        if native_balance is None:
            err = f"The portfolio data do not contain any information for the native {native_token!r} on ledger {ledger_id!r}."
            self.context.logger.error(err)
            # return empty orders and incomplete status, because SOL are necessary for all the swaps
            return [], True

        orders: List[Dict[str, str]] = []
        incomplete = False
        for token, data in token_data.items():
            if token == base_token or token == native_token:
                continue

            decision = self.get_swap_decision(data, portfolio, token)
            if decision is None:
                incomplete = True
                continue

            msg = f"Decided to {decision} token with address {token!r}."
            self.context.logger.info(msg)
            token_swap_position = self.get_token_swap_position(decision)
            if token_swap_position is None:
                # holding token, no tx to perform
                continue

            input_token = base_token
            output_token = token

            input_balance = portfolio.get(input_token, None)
            if input_balance is None:
                err = f"The portfolio does not contain information for the base token {base_token!r}. The base token is required for all swaps."
                self.context.logger.error(err)
                # return, because a swap for another token might be performed
                continue

            if input_token == native_token:
                incomplete = True
                continue

            if input_balance < int(required_amount):
                err = f"The portfolio does not contain enough balance for the base token {base_token!r}. Current balance: {input_balance}. Required amount: {required_amount}."
                self.context.logger.error(err)
                breakpoint()
                incomplete = True
                continue
            quote_data = {INPUT_MINT: input_token, OUTPUT_MINT: output_token}
            orders.append(quote_data)

        # we only yield here to convert this method to a generator, so that it can be used by `get_process_store_act`
        yield
        return orders, incomplete

    def async_act(self) -> Generator:
        """Do the action."""
        # We check if we are processing the solana or the ethereum chain
        processing_function = (
            self.get_solana_orders if self.params.use_solana else self.get_evm_orders
        )
        yield from self.get_process_store_act(
            self.synchronized_data.transformed_data_hash,
            processing_function,  # type: ignore
            str(self.swap_decision_filepath),
        )

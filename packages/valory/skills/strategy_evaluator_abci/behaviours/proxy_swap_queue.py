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

"""This module contains the behaviour for sending a transaction for the next swap in the queue of orders."""

from typing import Dict, Generator, List, Optional, cast

from packages.valory.skills.abstract_round_abci.io_.store import SupportedFiletype
from packages.valory.skills.strategy_evaluator_abci.behaviours.base import (
    StrategyEvaluatorBaseBehaviour,
)
from packages.valory.skills.strategy_evaluator_abci.models import TxSettlementProxy
from packages.valory.skills.strategy_evaluator_abci.payloads import SendSwapProxyPayload
from packages.valory.skills.strategy_evaluator_abci.states.proxy_swap_queue import (
    ProxySwapQueueRound,
)


OrdersType = Optional[List[Dict[str, str]]]


PROXY_STATUS_FIELD = "status"
PROXY_TX_ID_FIELD = "txId"
PROXY_TX_URL_FIELD = "url"
PROXY_ERROR_MESSAGE_FIELD = "message"
PROXY_SUCCESS_RESPONSE = "ok"
PROXY_ERROR_RESPONSE = "error"


class ProxySwapQueueBehaviour(StrategyEvaluatorBaseBehaviour):
    """A behaviour in which the agent utilizes the proxy server to perform the next swap transaction in priority.

    Warning: This can only work with a single agent service.
    """

    matching_round = ProxySwapQueueRound

    def setup(self) -> None:
        """Initialize the behaviour."""
        self.context.tx_settlement_proxy.reset_retries()

    @property
    def orders(self) -> OrdersType:
        """Get the orders from the shared state."""
        return self.shared_state.orders

    @orders.setter
    def orders(self, orders: OrdersType) -> None:
        """Set the orders to the shared state."""
        self.shared_state.orders = orders

    def get_orders(self) -> Generator:
        """Get the orders from IPFS."""
        if self.orders is None:
            # only fetch once per new batch and store in the shared state for future reference
            hash_ = self.synchronized_data.backtested_orders_hash
            orders = yield from self.get_from_ipfs(hash_, SupportedFiletype.JSON)
            self.orders = cast(OrdersType, orders)

    def handle_success(self, tx_id: Optional[str], url: Optional[str]) -> None:
        """Handle a successful response."""
        if not tx_id:
            err = "The proxy server returned no transaction id for successful transaction!"
            self.context.logger.error(err)

        swap_msg = f"Successfully performed swap transaction with id {tx_id}"
        swap_msg += f": {url}" if url is not None else "."
        self.context.logger.info(swap_msg)

    def handle_error(self, err: str) -> None:
        """Handle an error response."""
        err = f"Proxy server failed to settle transaction with message: {err}"
        self.context.logger.error(err)

    def handle_unknown_status(self, status: Optional[str]) -> None:
        """Handle a response with an unknown status."""
        err = f"Unknown {status=} was received from the  transaction settlement proxy server!"
        self.context.logger.error(err)

    def handle_response(self, response: Optional[Dict[str, str]]) -> Optional[str]:
        """Handle the response from the proxy server."""
        self.context.logger.debug(f"Proxy server {response=}.")
        if response is None:
            return None

        status = response.get(PROXY_STATUS_FIELD, None)
        tx_id = response.get(PROXY_TX_ID_FIELD, None)
        if status == PROXY_SUCCESS_RESPONSE:
            url = response.get(PROXY_TX_URL_FIELD, None)
            self.handle_success(tx_id, url)
        elif status == PROXY_ERROR_RESPONSE:
            err = response.get(PROXY_ERROR_MESSAGE_FIELD, "")
            self.handle_error(err)
        else:
            self.handle_unknown_status(status)

        return tx_id

    def perform_next_order(self) -> Generator[None, None, Optional[str]]:
        """Perform the next order in priority and return the tx id or `None` if not sent."""
        if self.orders is None:
            err = "Orders were expected to be set."
            self.context.logger.error(err)
            return None

        if len(self.orders) == 0:
            self.context.logger.info("No more orders to process.")
            self.orders = None
            return ""

        quote_data = self.orders.pop(0)
        msg = f"Attempting to swap {quote_data['inputMint']} -> {quote_data['outputMint']}..."
        self.context.logger.info(msg)

        proxy_api = cast(TxSettlementProxy, self.context.tx_settlement_proxy)
        # hacky solution
        params = proxy_api.get_spec()["parameters"]
        quote_data.update(params)

        response = yield from self._get_response(proxy_api, {}, content=quote_data)
        return self.handle_response(response)

    def async_act(self) -> Generator:
        """Do the action."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            yield from self.get_orders()
            sender = self.context.agent_address
            tx_id = yield from self.perform_next_order()
            payload = SendSwapProxyPayload(sender, tx_id)

        yield from self.finish_behaviour(payload)

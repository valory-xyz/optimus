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

"""This module contains the behaviour for deciding the next round post transaction settlement for the 'liquidity_trader_abci' skill."""

from typing import (
    Generator,
)
from packages.valory.skills.liquidity_trader_abci.rounds import (
    CheckStakingKPIMetRound,
    PostTxSettlementPayload,
    PostTxSettlementRound,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    LiquidityTraderBaseBehaviour,
)

class PostTxSettlementBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that is executed after a tx is settled via the transaction_settlement_abci."""

    matching_round = PostTxSettlementRound

    def async_act(self) -> Generator:
        """Simply log that a tx is settled and wait for the round end."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            msg = f"The transaction submitted by {self.synchronized_data.tx_submitter} was successfully settled."
            self.context.logger.info(msg)
            # we do not want to track the gas costs for vanity tx
            if (
                not self.synchronized_data.tx_submitter
                == CheckStakingKPIMetRound.auto_round_id()
            ):
                yield from self.fetch_and_log_gas_details()

            payload = PostTxSettlementPayload(
                sender=self.context.agent_address, content="Transaction settled"
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()

    def fetch_and_log_gas_details(self):
        """Fetch the transaction receipt and log the gas price and cost."""
        tx_hash = self.synchronized_data.final_tx_hash
        chain = self.synchronized_data.chain_id

        response = yield from self.get_transaction_receipt(
            tx_digest=tx_hash,
            chain_id=chain,
        )
        if not response:
            self.context.logger.error(
                f"Error fetching tx receipt! Response: {response}"
            )
            return

        effective_gas_price = response.get("effectiveGasPrice")
        gas_used = response.get("gasUsed")
        if gas_used and effective_gas_price:
            self.context.logger.info(
                f"Gas Details - Effective Gas Price: {effective_gas_price}, Gas Used: {gas_used}"
            )
            timestamp = int(
                self.round_sequence.last_round_transition_timestamp.timestamp()
            )
            chain_id = self.params.chain_to_chain_id_mapping.get(chain)
            if not chain_id:
                self.context.logger.error(f"No chain id found for chain {chain}")
                return
            self.gas_cost_tracker.log_gas_usage(
                str(chain_id), timestamp, tx_hash, gas_used, effective_gas_price
            )
            self.store_gas_costs()
            return
        else:
            self.context.logger.warning(
                "Gas used or effective gas price not found in the response."
            )

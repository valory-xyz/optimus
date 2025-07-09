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

"""This module contains the behaviour for checking is staking kpi is met for the 'liquidity_trader_abci' skill."""


from typing import Generator, Optional, Type

from packages.valory.contracts.gnosis_safe.contract import (
    GnosisSafeContract,
    SafeOperation,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    ETHER_VALUE,
    LiquidityTraderBaseBehaviour,
    SAFE_TX_GAS,
    ZERO_ADDRESS,
)
from packages.valory.skills.liquidity_trader_abci.payloads import (
    CheckStakingKPIMetPayload,
)
from packages.valory.skills.liquidity_trader_abci.states.check_staking_kpi_met import (
    CheckStakingKPIMetRound,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)


class CheckStakingKPIMetBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that checks if the staking KPI has been met and makes vanity transactions if necessary."""

    # pylint-disable too-many-ancestors
    matching_round: Type[AbstractRound] = CheckStakingKPIMetRound

    def async_act(self) -> Generator:
        """Do the action."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            # PRIORITY: Check if investing is paused due to withdrawal
            investing_paused = yield from self._read_kv(keys=("investing_paused",))
            if investing_paused and investing_paused.get("investing_paused") == "true":
                self.context.logger.info("Investing paused due to withdrawal - skipping KPI check")
                # Skip KPI check during withdrawal
                vanity_tx_hex = None
                is_staking_kpi_met = False
            else:
                vanity_tx_hex = None
                is_staking_kpi_met = yield from self._is_staking_kpi_met()
                if is_staking_kpi_met is None:
                    self.context.logger.error("Error checking if staking KPI is met.")
                elif is_staking_kpi_met is True:
                    self.context.logger.info("KPI already met for the day!")
                else:
                    is_period_threshold_exceeded = (
                        self.synchronized_data.period_count
                        - self.synchronized_data.period_number_at_last_cp
                        >= self.params.staking_threshold_period
                    )

                    if is_period_threshold_exceeded:
                        min_num_of_safe_tx_required = (
                            self.synchronized_data.min_num_of_safe_tx_required
                        )
                        multisig_nonces_since_last_cp = (
                            yield from self._get_multisig_nonces_since_last_cp(
                                chain=self.params.staking_chain,
                                multisig=self.params.safe_contract_addresses.get(
                                    self.params.staking_chain
                                ),
                            )
                        )
                        if (
                            multisig_nonces_since_last_cp is not None
                            and min_num_of_safe_tx_required is not None
                        ):
                            num_of_tx_left_to_meet_kpi = (
                                min_num_of_safe_tx_required - multisig_nonces_since_last_cp
                            )
                            if num_of_tx_left_to_meet_kpi > 0:
                                self.context.logger.info(
                                    f"Number of tx left to meet KPI: {num_of_tx_left_to_meet_kpi}"
                                )
                                self.context.logger.info("Preparing vanity tx..")
                                vanity_tx_hex = yield from self._prepare_vanity_tx(
                                    chain=self.params.staking_chain
                                )
                                self.context.logger.info(f"tx hash: {vanity_tx_hex}")

            tx_submitter = self.matching_round.auto_round_id()
            payload = CheckStakingKPIMetPayload(
                self.context.agent_address,
                tx_submitter,
                vanity_tx_hex,
                self.params.safe_contract_addresses.get(self.params.staking_chain),
                self.params.staking_chain,
                is_staking_kpi_met,
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()

    def _prepare_vanity_tx(self, chain: str) -> Generator[None, None, Optional[str]]:
        self.context.logger.info(f"Preparing vanity transaction for chain: {chain}")

        safe_address = self.params.safe_contract_addresses.get(chain)
        self.context.logger.debug(f"Safe address for chain {chain}: {safe_address}")

        tx_data = b"0x"
        self.context.logger.debug(f"Transaction data: {tx_data}")

        try:
            safe_tx_hash = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=safe_address,
                contract_public_id=GnosisSafeContract.contract_id,
                contract_callable="get_raw_safe_transaction_hash",
                data_key="tx_hash",
                to_address=ZERO_ADDRESS,
                value=ETHER_VALUE,
                data=tx_data,
                operation=SafeOperation.CALL.value,
                safe_tx_gas=SAFE_TX_GAS,
                chain_id=chain,
            )
        except Exception as e:
            self.context.logger.error(f"Exception during contract interaction: {e}")
            return None

        if safe_tx_hash is None:
            self.context.logger.error("Error preparing vanity tx: safe_tx_hash is None")
            return None

        self.context.logger.debug(f"Safe transaction hash: {safe_tx_hash}")

        try:
            tx_hash = hash_payload_to_hex(
                safe_tx_hash=safe_tx_hash[2:],
                ether_value=ETHER_VALUE,
                safe_tx_gas=SAFE_TX_GAS,
                operation=SafeOperation.CALL.value,
                to_address=ZERO_ADDRESS,
                data=tx_data,
            )
        except Exception as e:
            self.context.logger.error(f"Exception during hash payload conversion: {e}")
            return None

        self.context.logger.info(f"Vanity transaction hash: {tx_hash}")

        return tx_hash

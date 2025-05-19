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

"""This module contains the behaviour for calling the checkpoint on staking contract for the 'liquidity_trader_abci' skill."""

from typing import Generator, Optional

from packages.valory.contracts.gnosis_safe.contract import GnosisSafeContract
from packages.valory.contracts.staking_token.contract import StakingTokenContract
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    ETHER_VALUE,
    LiquidityTraderBaseBehaviour,
    SAFE_TX_GAS,
)
from packages.valory.skills.liquidity_trader_abci.states.call_checkpoint import (
    CallCheckpointPayload,
    CallCheckpointRound,
    StakingState,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)


class CallCheckpointBehaviour(
    LiquidityTraderBaseBehaviour
):  # pylint-disable too-many-ancestors
    """Behaviour that calls the checkpoint contract function if the service is staked and if it is necessary."""

    matching_round = CallCheckpointRound

    def async_act(self) -> Generator:
        """Do the action."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            checkpoint_tx_hex = None
            min_num_of_safe_tx_required = None

            if not self.params.staking_chain:
                self.context.logger.warning("Service has not been staked on any chain!")
                self.service_staking_state = StakingState.UNSTAKED
            else:
                yield from self._get_service_staking_state(
                    chain=self.params.staking_chain
                )
                if self.service_staking_state == StakingState.STAKED:
                    min_num_of_safe_tx_required = (
                        yield from self._calculate_min_num_of_safe_tx_required(
                            chain=self.params.staking_chain
                        )
                    )
                    if min_num_of_safe_tx_required is None:
                        self.context.logger.error(
                            "Error calculating min number of safe tx required."
                        )
                    else:
                        self.context.logger.info(
                            f"The minimum number of safe tx required to unlock rewards are {min_num_of_safe_tx_required}"
                        )
                    is_checkpoint_reached = (
                        yield from self._check_if_checkpoint_reached(
                            chain=self.params.staking_chain
                        )
                    )
                    if is_checkpoint_reached:
                        self.context.logger.info(
                            "Checkpoint reached! Preparing checkpoint tx.."
                        )
                        checkpoint_tx_hex = yield from self._prepare_checkpoint_tx(
                            chain=self.params.staking_chain
                        )
                elif self.service_staking_state == StakingState.EVICTED:
                    self.context.logger.error("Service has been evicted!")

                else:
                    self.context.logger.error("Service has not been staked")

            tx_submitter = self.matching_round.auto_round_id()
            payload = CallCheckpointPayload(
                self.context.agent_address,
                tx_submitter,
                checkpoint_tx_hex,
                self.params.safe_contract_addresses.get(self.params.staking_chain),
                self.params.staking_chain,
                self.service_staking_state.value,
                min_num_of_safe_tx_required,
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()

    def _check_if_checkpoint_reached(
        self, chain: str
    ) -> Generator[None, None, Optional[bool]]:
        next_checkpoint = yield from self._get_next_checkpoint(chain)
        if next_checkpoint is None:
            return False

        if next_checkpoint == 0:
            return True

        synced_timestamp = int(
            self.round_sequence.last_round_transition_timestamp.timestamp()
        )
        return next_checkpoint <= synced_timestamp

    def _prepare_checkpoint_tx(
        self, chain: str
    ) -> Generator[None, None, Optional[str]]:
        checkpoint_data = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=self.params.staking_token_contract_address,
            contract_public_id=StakingTokenContract.contract_id,
            contract_callable="build_checkpoint_tx",
            data_key="data",
            chain_id=chain,
        )

        safe_tx_hash = yield from self._prepare_safe_tx(chain, data=checkpoint_data)

        return safe_tx_hash

    def _prepare_safe_tx(
        self, chain: str, data: bytes
    ) -> Generator[None, None, Optional[str]]:
        safe_address = self.params.safe_contract_addresses.get(chain)
        safe_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=safe_address,
            contract_public_id=GnosisSafeContract.contract_id,
            contract_callable="get_raw_safe_transaction_hash",
            data_key="tx_hash",
            data=data,
            to_address=self.params.staking_token_contract_address,
            value=ETHER_VALUE,
            safe_tx_gas=SAFE_TX_GAS,
            chain_id=chain,
        )

        if safe_tx_hash is None:
            return None

        safe_tx_hash = safe_tx_hash[2:]
        return hash_payload_to_hex(
            safe_tx_hash=safe_tx_hash,
            ether_value=ETHER_VALUE,
            safe_tx_gas=SAFE_TX_GAS,
            to_address=self.params.staking_token_contract_address,
            data=data,
        )

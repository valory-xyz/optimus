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

"""This module contains the behaviour for fetching the positions for the 'liquidity_trader_abci' skill."""

import json
from typing import Any, Dict, Generator, List, Optional, Type

from packages.valory.contracts.balancer_weighted_pool.contract import (
    WeightedPoolContract,
)
from packages.valory.contracts.sturdy_yearn_v3_vault.contract import (
    YearnV3VaultContract,
)
from packages.valory.contracts.uniswap_v3_non_fungible_position_manager.contract import (
    UniswapV3NonfungiblePositionManagerContract,
)
from packages.valory.contracts.velodrome_non_fungible_position_manager.contract import (
    VelodromeNonFungiblePositionManagerContract,
)
from packages.valory.contracts.velodrome_pool.contract import VelodromePoolContract
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    DexType,
    LiquidityTraderBaseBehaviour,
    PositionStatus,
)
from packages.valory.skills.liquidity_trader_abci.states.get_positions import (
    GetPositionsPayload,
    GetPositionsRound,
)


class GetPositionsBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that gets the balances of the assets of agent safes."""

    matching_round: Type[AbstractRound] = GetPositionsRound
    current_positions = None

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            if not self.assets:
                self.assets = self.params.initial_assets
                self.store_assets()

            positions = yield from self.get_positions()
            yield from self._adjust_current_positions_for_backward_compatibility(
                self.current_positions
            )

            # Update the amounts of all open positions
            yield from self.update_position_amounts()

            self.check_and_update_zero_liquidity_positions()
            self.context.logger.info(f"Current Positions: {self.current_positions}")

            self.context.logger.info(f"POSITIONS: {positions}")
            sender = self.context.agent_address

            if positions is None:
                positions = GetPositionsRound.ERROR_PAYLOAD

            serialized_positions = json.dumps(positions, sort_keys=True)
            payload = GetPositionsPayload(sender=sender, positions=serialized_positions)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def check_and_update_zero_liquidity_positions(self) -> None:
        """Check for positions with zero liquidity and mark them as closed."""
        if not self.current_positions:
            return

        for position in self.current_positions:
            if position.get("status") != PositionStatus.OPEN:
                continue

            dex_type = position.get("dex_type")

            if dex_type == DexType.VELODROME.value and position.get("is_cl_pool"):
                # For Velodrome CL pools, check all sub-positions
                all_positions_zero = True
                for pos in position.get("positions", []):
                    if (
                        pos.get("current_liquidity", 1) != 0
                    ):  # Default to 1 if not found to avoid false closures
                        all_positions_zero = False
                        break

                if all_positions_zero and position.get(
                    "positions"
                ):  # Only update if there are positions
                    position["status"] = PositionStatus.CLOSED
                    self.context.logger.info(
                        f"Marked Velodrome CL position as closed due to zero liquidity in all positions: {position}"
                    )
            else:
                # For all other position types
                if (
                    position.get("current_liquidity", 1) == 0
                ):  # Default to 1 if not found to avoid false closures
                    position["status"] = PositionStatus.CLOSED
                    self.context.logger.info(
                        f"Marked {dex_type} position as closed due to zero liquidity: {position}"
                    )

        # Store the updated positions
        self.store_current_positions()

    def update_position_amounts(self) -> Generator[None, None, None]:
        """Update the amounts of all open positions."""
        if not self.current_positions:
            self.context.logger.info("No positions to update.")
            return

        for position in self.current_positions:
            # Only update open positions
            if position.get("status") != PositionStatus.OPEN:
                continue

            dex_type = position.get("dex_type")
            chain = position.get("chain")

            if not dex_type or not chain:
                self.context.logger.warning(
                    f"Position missing dex_type or chain: {position}"
                )
                continue

            self.context.logger.info(
                f"Updating position of type {dex_type} on chain {chain}"
            )

            # Update based on the type of position
            if dex_type == DexType.BALANCER.value:
                yield from self._update_balancer_position(position)
            elif dex_type == DexType.UNISWAP_V3.value:
                yield from self._update_uniswap_position(position)
            elif dex_type == DexType.VELODROME.value:
                yield from self._update_velodrome_position(position)
            elif dex_type == DexType.STURDY.value:
                yield from self._update_sturdy_position(position)
            else:
                self.context.logger.warning(f"Unknown position type: {dex_type}")

        # Store the updated positions
        self.store_current_positions()

    def _update_balancer_position(
        self, position: Dict[str, Any]
    ) -> Generator[None, None, None]:
        """Update a Balancer position."""
        pool_address = position.get("pool_address")
        safe_address = self.params.safe_contract_addresses.get(position.get("chain"))
        chain = position.get("chain")

        if not all([pool_address, safe_address, chain]):
            self.context.logger.warning(
                f"Missing required parameters for Balancer position: {position}"
            )
            return

        # Get the current balance of LP tokens
        balance = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=WeightedPoolContract.contract_id,
            contract_callable="get_balance",
            data_key="balance",
            account=safe_address,
            chain_id=chain,
        )

        if balance is not None:
            # Update the position with the current liquidity
            position["current_liquidity"] = balance
            self.context.logger.info(f"Updated Balancer position amount: {balance}")
        else:
            self.context.logger.warning(
                f"Failed to get balance for Balancer position: {position}"
            )

    def _update_uniswap_position(
        self, position: Dict[str, Any]
    ) -> Generator[None, None, None]:
        """Update a Uniswap V3 position."""
        token_id = position.get("token_id")
        chain = position.get("chain")

        if not all([token_id, chain]):
            self.context.logger.warning(
                f"Missing required parameters for Uniswap position: {position}"
            )
            return

        position_manager_address = (
            self.params.uniswap_position_manager_contract_addresses.get(chain)
        )
        if not position_manager_address:
            self.context.logger.warning(
                f"No position manager address found for chain {chain}"
            )
            return

        # Get the current liquidity
        position_data = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=position_manager_address,
            contract_public_id=UniswapV3NonfungiblePositionManagerContract.contract_id,
            contract_callable="get_position",
            data_key="data",
            token_id=token_id,
            chain_id=chain,
        )

        if position_data and position_data.get("liquidity"):
            # Update the position with the current liquidity
            position["current_liquidity"] = position_data.get("liquidity")
            self.context.logger.info(
                f"Updated Uniswap position liquidity: {position['current_liquidity']}"
            )
        else:
            self.context.logger.warning(
                f"Failed to get liquidity for Uniswap position: {position}"
            )

    def _update_velodrome_position(
        self, position: Dict[str, Any]
    ) -> Generator[None, None, None]:
        """Update a Velodrome position."""
        chain = position.get("chain")
        is_cl_pool = position.get("is_cl_pool", False)

        if not chain:
            self.context.logger.warning(
                f"Missing required parameters for Velodrome position: {position}"
            )
            return

        if is_cl_pool:
            # Handle Velodrome concentrated liquidity pool
            for pos in position.get("positions"):
                token_id = pos.get("token_id")
                if not token_id:
                    self.context.logger.warning(
                        f"Missing token_id for Velodrome CL position: {pos}"
                    )
                    return

                position_manager_address = self.params.velodrome_non_fungible_position_manager_contract_addresses.get(
                    chain
                )
                if not position_manager_address:
                    self.context.logger.warning(
                        f"No position manager address found for chain {chain}"
                    )
                    return

                # Get the current liquidity
                position_data = yield from self.contract_interact(
                    performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                    contract_address=position_manager_address,
                    contract_public_id=VelodromeNonFungiblePositionManagerContract.contract_id,
                    contract_callable="get_position",
                    data_key="data",
                    token_id=token_id,
                    chain_id=chain,
                )

                if position_data and position_data.get("liquidity"):
                    # Update the position with the current liquidity
                    pos["current_liquidity"] = position_data.get("liquidity")
                    self.context.logger.info(
                        f"Updated Uniswap position liquidity: {pos['current_liquidity']}"
                    )
                else:
                    self.context.logger.warning(
                        f"Failed to get liquidity for Uniswap position: {pos}"
                    )
        else:
            # Handle Velodrome stable/volatile pool
            pool_address = position.get("pool_address")
            safe_address = self.params.safe_contract_addresses.get(
                position.get("chain")
            )

            if not all([pool_address, safe_address]):
                self.context.logger.warning(
                    f"Missing required parameters for Velodrome pool position: {position}"
                )
                return

            # Get the current balance of LP tokens
            balance = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=pool_address,
                contract_public_id=VelodromePoolContract.contract_id,
                contract_callable="get_balance",
                data_key="balance",
                account=safe_address,
                chain_id=chain,
            )

            if balance is not None:
                # Update the position with the current amount
                position["current_liquidity"] = balance
                self.context.logger.info(
                    f"Updated Velodrome pool position amount: {balance}"
                )
            else:
                self.context.logger.warning(
                    f"Failed to get balance for Velodrome pool position: {position}"
                )

    def _update_sturdy_position(
        self, position: Dict[str, Any]
    ) -> Generator[None, None, None]:
        """Update a Sturdy position."""
        # For Sturdy positions, we need to call balance_of on the relevant contract
        pool_address = position.get("pool_address")
        safe_address = self.params.safe_contract_addresses.get(position.get("chain"))
        chain = position.get("chain")

        if not all([pool_address, safe_address, chain]):
            self.context.logger.warning(
                f"Missing required parameters for Sturdy position: {position}"
            )
            return

        # Get the current balance
        balance = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=YearnV3VaultContract.contract_id,
            contract_callable="balance_of",
            data_key="amount",
            account=safe_address,
            chain_id=chain,
        )

        if balance is not None:
            # Update the position with the current amount
            position["current_liquidity"] = balance
            self.context.logger.info(f"Updated Sturdy position amount: {balance}")
        else:
            self.context.logger.warning(
                f"Failed to get balance for Sturdy position: {position}"
            )

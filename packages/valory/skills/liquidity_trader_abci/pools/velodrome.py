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

"""This package contains the implementation of the Velodrome Pool Behaviour."""

import sys
from abc import ABC
from enum import Enum
from typing import Any, Dict, Generator, Optional, Tuple, cast

from web3 import Web3

from packages.valory.contracts.multisend.contract import (
    MultiSendContract,
    MultiSendOperation,
)
from packages.valory.contracts.velodrome_router.contract import VelodromeRouterContract
from packages.valory.contracts.velodrome_cl_pool.contract import VelodromeCLPoolContract
from packages.valory.contracts.velodrome_cl_manager.contract import VelodromeCLPoolManagerContract
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.liquidity_trader_abci.models import SharedState
from packages.valory.skills.liquidity_trader_abci.pool_behaviour import PoolBehaviour


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
MIN_TICK = -887272
MAX_TICK = 887272
INT_MAX = sys.maxsize


class PoolType(Enum):
    """Velodrome Pool Types"""
    STABLE = "Stable"
    VOLATILE = "Volatile"
    CONCENTRATED_LIQUIDITY = "ConcentratedLiquidity"


class VelodromePoolBehaviour(PoolBehaviour, ABC):
    """Velodrome Pool Behaviour that handles both Stable/Volatile and Concentrated Liquidity pools"""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Velodrome pool behaviour."""
        super().__init__(**kwargs)

    def enter(self, **kwargs: Any) -> Generator[None, None, Optional[Tuple[str, str]]]:
        """Add liquidity to a Velodrome pool."""
        pool_address = kwargs.get("pool_address")
        safe_address = kwargs.get("safe_address")
        assets = kwargs.get("assets")
        chain = kwargs.get("chain")
        max_amounts_in = kwargs.get("max_amounts_in")
        pool_type = kwargs.get("pool_type")

        if not all([pool_address, safe_address, assets, chain, max_amounts_in, pool_type]):
            self.context.logger.error(
                f"Missing required parameters for entering the pool. Here are the kwargs: {kwargs}"
            )
            return None, None

        # Determine the pool type and call the appropriate method
        if pool_type == PoolType.CONCENTRATED_LIQUIDITY.value:
            return (yield from self._enter_cl_pool(
                pool_address=pool_address,
                safe_address=safe_address,
                assets=assets,
                chain=chain,
                max_amounts_in=max_amounts_in,
                tick_lower=kwargs.get("tick_lower"),
                tick_upper=kwargs.get("tick_upper"),
            ))
        else:
            # Handle Stable or Volatile pools
            return (yield from self._enter_stable_volatile_pool(
                pool_address=pool_address,
                safe_address=safe_address,
                assets=assets,
                chain=chain,
                max_amounts_in=max_amounts_in,
                is_stable=(pool_type == PoolType.STABLE.value),
            ))

    def exit(self, **kwargs: Any) -> Generator[None, None, Optional[Tuple[str, str, bool]]]:
        """Remove liquidity from a Velodrome pool."""
        pool_address = kwargs.get("pool_address")
        safe_address = kwargs.get("safe_address")
        chain = kwargs.get("chain")
        liquidity = kwargs.get("liquidity")
        pool_type = kwargs.get("pool_type")

        if not all([pool_address, safe_address, chain, liquidity, pool_type]):
            self.context.logger.error(
                f"Missing required parameters for exiting the pool. Here are the kwargs: {kwargs}"
            )
            return None, None, None

        # Determine the pool type and call the appropriate method
        if pool_type == PoolType.CONCENTRATED_LIQUIDITY.value:
            return (yield from self._exit_cl_pool(
                pool_address=pool_address,
                safe_address=safe_address,
                chain=chain,
                liquidity=liquidity,
                tick_lower=kwargs.get("tick_lower"),
                tick_upper=kwargs.get("tick_upper"),
            ))
        else:
            # Handle Stable or Volatile pools
            return (yield from self._exit_stable_volatile_pool(
                pool_address=pool_address,
                safe_address=safe_address,
                assets=kwargs.get("assets"),
                chain=chain,
                liquidity=liquidity,
                is_stable=(pool_type == PoolType.STABLE.value),
            ))

    def _enter_stable_volatile_pool(
        self,
        pool_address: str,
        safe_address: str,
        assets: list,
        chain: str,
        max_amounts_in: list,
        is_stable: bool,
    ) -> Generator[None, None, Optional[Tuple[str, str]]]:
        """Add liquidity to a Velodrome Stable/Volatile pool."""
        # Get router contract address
        router_address = self.params.velodrome_router_contract_addresses.get(chain, "")
        if not router_address:
            self.context.logger.error(
                f"No router contract address found for chain {chain}"
            )
            return None, None

        # Set minimum amounts (with slippage protection)
        # TO-DO: Implement proper slippage protection
        amount_a_min = int(max_amounts_in[0] * 0.99)  # 1% slippage
        amount_b_min = int(max_amounts_in[1] * 0.99)  # 1% slippage

        # Set deadline (20 minutes from now)
        last_update_time = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()
        deadline = int(last_update_time) + (20 * 60)

        # Call addLiquidity on the router contract
        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=router_address,
            contract_public_id=VelodromeRouterContract.contract_id,
            contract_callable="add_liquidity",
            data_key="tx_bytes",
            token_a=assets[0],
            token_b=assets[1],
            stable=is_stable,
            amount_a_desired=max_amounts_in[0],
            amount_b_desired=max_amounts_in[1],
            amount_a_min=amount_a_min,
            amount_b_min=amount_b_min,
            to=safe_address,
            deadline=deadline,
            chain_id=chain,
        )

        return tx_hash, router_address

    def _exit_stable_volatile_pool(
        self,
        pool_address: str,
        safe_address: str,
        assets: list,
        chain: str,
        liquidity: int,
        is_stable: bool,
    ) -> Generator[None, None, Optional[Tuple[str, str, bool]]]:
        """Remove liquidity from a Velodrome Stable/Volatile pool."""
        # Get router contract address
        router_address = self.params.velodrome_router_contract_addresses.get(chain, "")
        if not router_address:
            self.context.logger.error(
                f"No router contract address found for chain {chain}"
            )
            return None, None, None

        # Set minimum amounts (with slippage protection)
        # TO-DO: Implement proper slippage protection
        amount_a_min = 0
        amount_b_min = 0

        # Set deadline (20 minutes from now)
        last_update_time = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()
        deadline = int(last_update_time) + (20 * 60)

        # Call removeLiquidity on the router contract
        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=router_address,
            contract_public_id=VelodromeRouterContract.contract_id,
            contract_callable="remove_liquidity",
            data_key="tx_bytes",
            token_a=assets[0],
            token_b=assets[1],
            stable=is_stable,
            liquidity=liquidity,
            amount_a_min=amount_a_min,
            amount_b_min=amount_b_min,
            to=safe_address,
            deadline=deadline,
            chain_id=chain,
        )

        return tx_hash, router_address, False

    def _enter_cl_pool(
        self,
        pool_address: str,
        safe_address: str,
        assets: list,
        chain: str,
        max_amounts_in: list,
        tick_lower: Optional[int] = None,
        tick_upper: Optional[int] = None,
    ) -> Generator[None, None, Optional[Tuple[str, str]]]:
        """Add liquidity to a Velodrome Concentrated Liquidity pool."""
        # Get CLPoolManager contract address
        manager_address = self.params.velodrome_cl_manager_contract_addresses.get(chain, "")
        if not manager_address:
            self.context.logger.error(
                f"No CLPoolManager contract address found for chain {chain}"
            )
            return None, None

        # If tick range not provided, use full range
        if not tick_lower or not tick_upper:
            tick_lower, tick_upper = yield from self._calculate_tick_range(pool_address, chain)
            if not tick_lower or not tick_upper:
                return None, None

        # Set minimum amounts (with slippage protection)
        # TO-DO: Implement proper slippage protection
        amount0_min = int(max_amounts_in[0] * 0.99)  # 1% slippage
        amount1_min = int(max_amounts_in[1] * 0.99)  # 1% slippage

        # Call createPosition on the CLPoolManager contract
        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=manager_address,
            contract_public_id=VelodromeCLPoolManagerContract.contract_id,
            contract_callable="create_position",
            data_key="tx_bytes",
            pool=pool_address,
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            amount0_desired=max_amounts_in[0],
            amount1_desired=max_amounts_in[1],
            amount0_min=amount0_min,
            amount1_min=amount1_min,
            chain_id=chain,
        )

        return tx_hash, manager_address

    def _exit_cl_pool(
        self,
        pool_address: str,
        safe_address: str,
        chain: str,
        liquidity: int,
        tick_lower: int,
        tick_upper: int,
    ) -> Generator[None, None, Optional[Tuple[str, str, bool]]]:
        """Remove liquidity from a Velodrome Concentrated Liquidity pool."""
        # Get CLPool contract address
        cl_pool_address = pool_address

        # Get multisend contract address
        multisend_address = self.params.multisend_contract_addresses.get(chain, "")
        if not multisend_address:
            self.context.logger.error(
                f"No multisend contract address found for chain {chain}"
            )
            return None, None, None

        multi_send_txs = []

        # First, burn liquidity
        burn_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=cl_pool_address,
            contract_public_id=VelodromeCLPoolContract.contract_id,
            contract_callable="burn",
            data_key="tx_bytes",
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            amount=liquidity,
            chain_id=chain,
        )
        if not burn_tx_hash:
            return None, None, None

        multi_send_txs.append(
            {
                "operation": MultiSendOperation.CALL,
                "to": cl_pool_address,
                "value": 0,
                "data": burn_tx_hash,
            }
        )

        # Then, collect tokens
        # Use maximum uint128 value for amount0_requested and amount1_requested
        max_uint128 = 2**128 - 1

        collect_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=cl_pool_address,
            contract_public_id=VelodromeCLPoolContract.contract_id,
            contract_callable="collect",
            data_key="tx_bytes",
            recipient=safe_address,
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            amount0_requested=max_uint128,
            amount1_requested=max_uint128,
            chain_id=chain,
        )
        if not collect_tx_hash:
            return None, None, None

        multi_send_txs.append(
            {
                "operation": MultiSendOperation.CALL,
                "to": cl_pool_address,
                "value": 0,
                "data": collect_tx_hash,
            }
        )

        # Prepare multisend transaction
        multisend_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=multisend_address,
            contract_public_id=MultiSendContract.contract_id,
            contract_callable="get_tx_data",
            data_key="data",
            multi_send_txs=multi_send_txs,
            chain_id=chain,
        )
        if not multisend_tx_hash:
            return None, None, None

        self.context.logger.info(f"multisend_tx_hash = {multisend_tx_hash}")
        return bytes.fromhex(multisend_tx_hash[2:]), multisend_address, True

    def _calculate_tick_range(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[Tuple[int, int]]]:
        """Calculate tick range for the position."""
        # For simplicity, use full range
        # In a real implementation, this would calculate a more optimal range
        # based on current price and expected price movement
        return MIN_TICK, MAX_TICK

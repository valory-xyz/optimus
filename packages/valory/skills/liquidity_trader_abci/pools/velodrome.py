#!/usr/bin/env python3
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
import json
import time
from abc import ABC
from enum import Enum
import numpy as np
import time
import json
import email.utils
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, cast, Callable

from web3 import Web3

from packages.valory.contracts.multisend.contract import (
    MultiSendContract,
    MultiSendOperation,
)
from packages.valory.contracts.velodrome_router.contract import VelodromeRouterContract
from packages.valory.contracts.velodrome_cl_pool.contract import VelodromeCLPoolContract
from packages.valory.contracts.velodrome_pool.contract import VelodromePoolContract
from packages.valory.contracts.velodrome_non_fungible_position_manager.contract import VelodromeNonFungiblePositionManagerContract
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.liquidity_trader_abci.models import SharedState
from packages.valory.skills.liquidity_trader_abci.pool_behaviour import PoolBehaviour
from collections import defaultdict

# Constants for price history functions
PRICE_VOLATILITY_THRESHOLD = 0.02  # 2% threshold for stablecoin detection
DEFAULT_DAYS = 30
API_CACHE_SIZE = 128


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
MIN_TICK = -887272
MAX_TICK = 887272
INT_MAX = sys.maxsize



class VelodromePoolBehaviour(PoolBehaviour, ABC):
    """Velodrome Pool Behaviour that handles both Stable/Volatile and Concentrated Liquidity pools"""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Velodrome pool behaviour."""
        super().__init__(**kwargs)

    def enter(self, **kwargs: Any) -> Generator[None, None, Optional[Tuple[Union[str, List[str]], str]]]:
        """Add liquidity to a Velodrome pool."""
        pool_address = kwargs.get("pool_address")
        safe_address = kwargs.get("safe_address")
        assets = kwargs.get("assets")
        chain = kwargs.get("chain")
        max_amounts_in = kwargs.get("max_amounts_in")
        is_cl_pool = kwargs.get("is_cl_pool", False)
        is_stable = kwargs.get("is_stable")

        if not all([pool_address, safe_address, assets, chain, max_amounts_in]):
            self.context.logger.error(
                f"Missing required parameters for entering the pool. Here are the kwargs: {kwargs}"
            )
            return None, None

        # Determine the pool type and call the appropriate method
        if is_cl_pool:
            return (yield from self._enter_cl_pool(
                pool_address=pool_address,
                safe_address=safe_address,
                assets=assets,
                chain=chain,
                max_amounts_in=max_amounts_in,
                is_stable=is_stable,
                pool_fee=kwargs.get("pool_fee"),
            ))
        else:
            # Handle Stable or Volatile pools
            return (yield from self._enter_stable_volatile_pool(
                pool_address=pool_address,
                safe_address=safe_address,
                assets=assets,
                chain=chain,
                max_amounts_in=max_amounts_in,
                is_stable= is_stable,
            ))

    def exit(self, **kwargs: Any) -> Generator[None, None, Optional[Tuple[str, str, bool]]]:
        """Remove liquidity from a Velodrome pool."""
        pool_address = kwargs.get("pool_address")
        safe_address = kwargs.get("safe_address")
        chain = kwargs.get("chain")
        is_cl_pool = kwargs.get("is_cl_pool", False)
        is_stable = kwargs.get("is_stable")
        if not all([pool_address, safe_address, chain]):
            self.context.logger.error(
                f"Missing required parameters for exiting the pool. Here are the kwargs: {kwargs}"
            )
            return None, None, None

        # Determine the pool type and call the appropriate method
        if is_cl_pool:
            # For CL pools, we can handle either a single token_id or multiple token_ids
            token_ids = kwargs.get("token_ids")
            liquidities = kwargs.get("liquidities")
            
            # If neither token_id nor token_ids is provided, log an error
            if not token_ids or not liquidities:
                self.context.logger.error(
                    f"Missing token_ids or liquidities for exiting CL pool"
                )
                return None, None, None
                
            return (yield from self._exit_cl_pool(
                token_ids=token_ids,
                safe_address=safe_address,
                chain=chain,
                liquidities=liquidities,
            ))
        else:
            # Handle Stable or Volatile pools
            if not kwargs.get("assets"):
                self.context.logger.error(
                    f"Missing assets for exiting stable/volatile pool"
                )
                return None, None, None
                
            return (yield from self._exit_stable_volatile_pool(
                pool_address=pool_address,
                safe_address=safe_address,
                assets=kwargs.get("assets"),
                chain=chain,
                liquidity=kwargs.get("liquidity"),
                is_stable=is_stable,
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
        amount_a_min = 0
        amount_b_min = 0  

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
            data_key="tx_hash",
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

        if not liquidity:
            liquidity = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=pool_address,
                contract_public_id=VelodromePoolContract.contract_id,
                contract_callable="get_balance",
                data_key="balance",
                account=safe_address,
                chain_id=chain,
            )

        if not liquidity:
            self.context.logger.error(f"No liquidity found for account ({safe_address}) in pool ({pool_address})")
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

        # Use multisend to batch approve and remove liquidity transactions
        multi_send_txs = []
        
        # First, approve the liquidity tokens to the router
        approve_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=VelodromePoolContract.contract_id,
            contract_callable="build_approval_tx",
            data_key="tx_hash",
            spender=router_address,
            amount=liquidity,
            chain_id=chain,
        )
        
        if not approve_tx_hash:
            self.context.logger.error(f"Failed to approve liquidity tokens to router")
            return None, None, None
            
        multi_send_txs.append(
            {
                "operation": MultiSendOperation.CALL,
                "to": pool_address,
                "value": 0,
                "data": approve_tx_hash,
            }
        )

        # Then, call removeLiquidity on the router contract
        remove_liquidity_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=router_address,
            contract_public_id=VelodromeRouterContract.contract_id,
            contract_callable="remove_liquidity",
            data_key="tx_hash",
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
        
        if not remove_liquidity_tx_hash:
            self.context.logger.error(f"Failed to create remove_liquidity transaction")
            return None, None, None
            
        multi_send_txs.append(
            {
                "operation": MultiSendOperation.CALL,
                "to": router_address,
                "value": 0,
                "data": remove_liquidity_tx_hash,
            }
        )
        
        # Prepare multisend transaction
        multisend_address = self.params.multisend_contract_addresses.get(chain, "")
        if not multisend_address:
            self.context.logger.error(
                f"Could not find multisend address for chain {chain}"
            )
            return None, None, None

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

    def _enter_cl_pool(
        self,
        pool_address: str,
        safe_address: str,
        assets: list,
        chain: str,
        max_amounts_in: list,
        is_stable: bool,
        pool_fee: Optional[int] = None,
    ) -> Generator[None, None, Optional[Tuple[Union[str, List[str]], str]]]:
        """Add liquidity to a Velodrome Concentrated Liquidity pool."""
        # Get NonFungiblePositionManager contract address
        position_manager_address = self.params.velodrome_non_fungible_position_manager_contract_addresses.get(chain, "")
        if not position_manager_address:
            self.context.logger.error(
                f"No NonFungiblePositionManager contract address found for chain {chain}"
            )
            return None, None

        #we provide in 50/50 ratio
        #works only for stablecoins for now (because the decimals are same, for non-stablecoins this won't work)
        if max_amounts_in and len(max_amounts_in) == 2:
            min_amount = min(max_amounts_in)
            max_amounts_in = [min_amount, min_amount]
        # Calculate tick ranges based on pool's tick spacing
        tick_ranges = yield from self._calculate_tick_lower_and_upper_velodrome(
            pool_address, chain, is_stable

        )
        if not tick_ranges:
            self.context.logger.error(f"Failed to calculate tick ranges for pool {pool_address}")
            return None, None

        # Get tick spacing for the pool
        tick_spacing = yield from self._get_tick_spacing_velodrome(pool_address, chain)
        if not tick_spacing:
            self.context.logger.error(f"Could not get tick spacing for pool {pool_address}")
            return None, None

        # Get or calculate sqrt_price_x96
        sqrt_price_x96 = yield from self._get_sqrt_price_x96(pool_address, chain)
        if sqrt_price_x96 is None:
            self.context.logger.error(f"Could not determine sqrt_price_x96 for pool {pool_address}")
            sqrt_price_x96 = 0  # Default value

        # TO-DO: add slippage protection
        amount0_min = 0
        amount1_min = 0

        # deadline is set to be 20 minutes from current time
        last_update_time = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()
        deadline = int(last_update_time) + (20 * 60)

        # If no positions, return error
        if not tick_ranges:
            self.context.logger.error(f"No valid positions calculated for pool {pool_address}")
            return None, None
            
        self.context.logger.info(
            f"Using max amounts: {max_amounts_in[0]} token0, {max_amounts_in[1]} token1"
        )
            
        # Calculate the amount to allocate to each position
        # Track total allocated to ensure we don't exceed max_amounts_in
        total_allocated_0 = 0
        total_allocated_1 = 0
        
        # First pass: calculate desired amounts
        for position in tick_ranges:
            allocation = position.get("allocation", 0)
            if allocation <= 0:
                position["amount0_desired"] = 0
                position["amount1_desired"] = 0
                continue
                
            # Calculate amounts based on allocation directly
            # We allocate the same percentage of each token based on the band allocation
            amount0_desired = int(max_amounts_in[0] * allocation)
            amount1_desired = int(max_amounts_in[1] * allocation)
            
            # Store in position for second pass
            position["amount0_desired"] = amount0_desired
            position["amount1_desired"] = amount1_desired
            
            # Track totals
            total_allocated_0 += amount0_desired
            total_allocated_1 += amount1_desired
        
        # Second pass: adjust if we're over the max
        if total_allocated_0 > max_amounts_in[0] or total_allocated_1 > max_amounts_in[1]:
            # Scale down proportionally
            scale_factor_0 = max_amounts_in[0] / max(1, total_allocated_0)
            scale_factor_1 = max_amounts_in[1] / max(1, total_allocated_1)
            scale_factor = min(scale_factor_0, scale_factor_1)
            
            self.context.logger.info(f"Scaling down allocations by factor {scale_factor:.4f}")
            
            # Apply scaling
            for position in tick_ranges:
                position["amount0_desired"] = int(position["amount0_desired"] * scale_factor)
                position["amount1_desired"] = int(position["amount1_desired"] * scale_factor)
        
        # Process each position and collect individual transaction hashes
        tx_hashes = []
        
        # Process each position
        for position in tick_ranges:
            amount0_desired = position.get("amount0_desired", 0)
            amount1_desired = position.get("amount1_desired", 0)
            
            self.context.logger.info(
                f"Position allocation: {position.get('allocation', 0):.1%}, "
                f"Amounts: {amount0_desired}/{amount1_desired}"
            )
            
            if amount0_desired <= 0 or amount1_desired <= 0:
                self.context.logger.warning(
                    f"Skipping position with too small allocation: {position.get('allocation', 0):.1%}"
                )
                continue
                
            # Call mint on the CLPoolManager contract
            mint_tx_hash = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=position_manager_address,
                contract_public_id=VelodromeNonFungiblePositionManagerContract.contract_id,
                contract_callable="mint",
                data_key="tx_hash",
                token0=assets[0],
                token1=assets[1],
                tick_spacing=tick_spacing,
                tick_lower=position["tick_lower"],
                tick_upper=position["tick_upper"],
                amount0_desired=amount0_desired,
                amount1_desired=amount1_desired,
                amount0_min=amount0_min,
                amount1_min=amount1_min,
                recipient=safe_address,
                deadline=deadline,
                sqrt_price_x96=0,
                chain_id=chain,
            )
            
            if not mint_tx_hash:
                self.context.logger.error(f"Failed to create mint transaction for position {position}")
                continue
                
            tx_hashes.append(mint_tx_hash)
        
        # If no transactions were created, return error
        if not tx_hashes:
            self.context.logger.error("No valid mint transactions created")
            return None, None
        
            
        # Return the list of transaction hashes
        return tx_hashes, position_manager_address

    def _get_sqrt_price_x96(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[int]]:
        """Get the current sqrt_price_x96 from the pool.
        
        If the pool exists and has liquidity, fetches the current price.
        If not, returns a default value for a 1:1 price ratio.
        """
        # Use the slot0 function to get pool state including sqrt_price_x96
        if not pool_address:
            self.context.logger.error("No pool address provided")
            return None
            
        slot0_data = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=VelodromeCLPoolContract.contract_id,
            contract_callable="slot0",
            data_key='slot0',
            chain_id=chain,
        )
        
        if slot0_data is None:
            self.context.logger.error(f"Could not get slot0 data for pool {pool_address}")
            return None
            
        sqrt_price_x96 = slot0_data.get("sqrt_price_x96")
        self.context.logger.info(f"Using sqrt_price_x96 value {sqrt_price_x96} for pool {pool_address}")
        return sqrt_price_x96

    def _exit_cl_pool(
        self,
        token_ids: Optional[List[int]] = None,
        safe_address: str = None,
        chain: str = None,
        liquidities: Optional[int] = None,
    ) -> Generator[None, None, Optional[Tuple[str, str, bool]]]:
        """
        Remove liquidity from a Velodrome Concentrated Liquidity pool.
        
        This method can handle exiting a single position or multiple positions
        (using token_ids). If token_ids is provided, it will exit all positions in the list.
        """
        position_manager_address = self.params.velodrome_non_fungible_position_manager_contract_addresses.get(chain, "")
        if not position_manager_address:
            self.context.logger.error(
                f"No CLPoolManager contract address found for chain {chain}"
            )
            return None, None, None
            
        multi_send_txs = []
        
        # TO-DO: Calculate min amounts accounting for slippage
        amount0_min = 0
        amount1_min = 0
        
        # deadline is set to be 20 minutes from current time
        last_update_time = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()
        deadline = int(last_update_time) + (20 * 60)
        
        # Use a high max value similar to Uniswap's approach for collecting tokens
        amount_max = Web3.to_wei(2**100 - 1, "wei")
            
        # Process each position
        for index, position_token_id in enumerate(token_ids):
            # Get liquidity for this position from the corresponding index
            position_liquidity = liquidities[index]
            if not position_liquidity:
                position_liquidity = yield from self.get_liquidity_for_token_velodrome(position_token_id, chain)
                if not position_liquidity:
                    self.context.logger.warning(f"Could not get liquidity for token ID {position_token_id}, skipping")
                    continue
                    
            # Decrease liquidity for this position
            decrease_liquidity_tx_hash = yield from self.decrease_liquidity_velodrome(
                position_token_id, position_liquidity, amount0_min, amount1_min, deadline, chain
            )
            if not decrease_liquidity_tx_hash:
                self.context.logger.warning(f"Failed to decrease liquidity for token ID {position_token_id}, skipping")
                continue
                
            multi_send_txs.append({
                "operation": MultiSendOperation.CALL,
                "to": position_manager_address,
                "value": 0,
                "data": decrease_liquidity_tx_hash,
            })
            
            # Collect tokens for this position
            collect_tokens_tx_hash = yield from self.collect_tokens_velodrome(
                token_id=position_token_id,
                recipient=safe_address,
                amount0_max=amount_max,
                amount1_max=amount_max,
                chain=chain,
            )
            if not collect_tokens_tx_hash:
                self.context.logger.warning(f"Failed to collect tokens for token ID {position_token_id}, skipping")
                continue
                
            multi_send_txs.append({
                "operation": MultiSendOperation.CALL,
                "to": position_manager_address,
                "value": 0,
                "data": collect_tokens_tx_hash,
            })

        # prepare multisend
        multisend_address = self.params.multisend_contract_addresses.get(chain, "")
        if not multisend_address:
            self.context.logger.error(
                f"Could not find multisend address for chain {chain}"
            )
            return None, None, None

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

    def get_liquidity_for_token_velodrome(
        self, token_id: int, chain: str
    ) -> Generator[None, None, Optional[int]]:
        """Get liquidity for token"""
        position_manager_address = self.params.velodrome_non_fungible_position_manager_contract_addresses.get(chain, "")
        if not position_manager_address:
            self.context.logger.error(
                f"No CLPoolManager contract address found for chain {chain}"
            )
            return None

        position = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=position_manager_address,
            contract_public_id=VelodromeNonFungiblePositionManagerContract.contract_id,
            contract_callable="get_position",
            data_key="data",
            token_id=token_id,
            chain_id=chain,
        )

        if not position:
            return None

        # liquidity is returned from positions mapping at the appropriate index
        # Typically liquidity is at a specific index in the struct
        liquidity = position[2]  # This should match the index where liquidity is stored in the positions struct
        return liquidity

    def decrease_liquidity_velodrome(
        self,
        token_id: int,
        liquidity: int,
        amount0_min: int,
        amount1_min: int,
        deadline: int,
        chain: str,
    ) -> Generator[None, None, Optional[str]]:
        """Decrease liquidity"""
        position_manager_address = self.params.velodrome_non_fungible_position_manager_contract_addresses.get(chain, "")
        if not position_manager_address:
            self.context.logger.error(
                f"No CLPoolManager contract address found for chain {chain}"
            )
            return None

        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=position_manager_address,
            contract_public_id=VelodromeNonFungiblePositionManagerContract.contract_id,
            contract_callable="decrease_liquidity",
            data_key="tx_hash",
            token_id=token_id,
            liquidity=liquidity,
            amount0_min=amount0_min,
            amount1_min=amount1_min,
            deadline=deadline,
            chain_id=chain,
        )

        return tx_hash

    def collect_tokens_velodrome(
        self,
        token_id: int,
        recipient: str,
        amount0_max: int,
        amount1_max: int,
        chain: str,
    ) -> Generator[None, None, Optional[str]]:
        """Collect tokens"""
        position_manager_address = self.params.velodrome_non_fungible_position_manager_contract_addresses.get(chain, "")
        if not position_manager_address:
            self.context.logger.error(
                f"No CLPoolManager contract address found for chain {chain}"
            )
            return None

        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=position_manager_address,
            contract_public_id=VelodromeNonFungiblePositionManagerContract.contract_id,
            contract_callable="collect",
            data_key="tx_hash",
            token_id=token_id,
            recipient=recipient,
            amount0_max=amount0_max,
            amount1_max=amount1_max,
            chain_id=chain,
        )

        return tx_hash


    def _get_tick_spacing_velodrome(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[int]]:
        """Get velodrome pool tick spacing"""
        tick_spacing = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=VelodromeCLPoolContract.contract_id,
            contract_callable="get_tick_spacing",
            data_key="data",
            chain_id=chain,
        )

        if not tick_spacing:
            self.context.logger.error(
                f"Could not fetch tick spacing for velodrome pool {pool_address}"
            )
            return None

        self.context.logger.info(
            f"Tick spacing for velodrome pool {pool_address} : {tick_spacing}"
        )
        return tick_spacing

    def _calculate_tick_lower_and_upper_velodrome(
        self, pool_address: str, chain: str, is_stable: bool
    ) -> Generator[None, None, Optional[List[Any]]]:
        """
        Calculate tick range for Velodrome concentrated liquidity pool.
        
        Uses historical price data and the stablecoin model to determine optimal tick bounds.
        
        Args:
            pool_address: The address of the Velodrome concentrated liquidity pool
            chain: The blockchain chain ID
            
        Returns:
            A tuple of (lower_tick, upper_tick) or None if calculation fails
        """
        self.context.logger.info(f"Calculating tick ranges using stablecoin model for pool {pool_address}")
        
        # 1. Fetch tick spacing from velodrome cl pool
        tick_spacing = yield from self._get_tick_spacing_velodrome(pool_address, chain)
        if not tick_spacing:
            self.context.logger.error(f"Failed to get tick spacing for pool {pool_address}")
            return None
            
        # 2. Get the token addresses from the pool
        token0, token1 = yield from self._get_pool_tokens(pool_address, chain)
        if not token0 or not token1:
            self.context.logger.error(f"Failed to get tokens for pool {pool_address}")
            return None
        
        # 3. Get current price
        current_price = yield from self._get_current_pool_price(pool_address, chain)
        if current_price is None:
            self.context.logger.error(f"Failed to get current price for pool {pool_address}")
            return None
            
        try:
            # 4. Get historical price data for both tokens and calculate price ratio history
            self.context.logger.info(f"Fetching historical price data for tokens: {token0} and {token1}")
            try:
                pool_data = yield from self.get_pool_token_history(
                    chain=chain, 
                    token0_address=token0, 
                    token1_address=token1
                )
                
                # Check if we have valid pool data
                if pool_data is None:
                    self.context.logger.error(
                        f"Failed to get pool token history for {token0} and {token1}. "
                        f"Aborting operation."
                    )
                    return None
                
                # Check if we have price data
                ratio_prices = pool_data.get("ratio_prices", [])
                
                if not ratio_prices:
                    self.context.logger.error(
                        f"Could not get price ratio history for pool {pool_address}. "
                        f"Aborting operation."
                    )
                    return None
            except Exception as e:
                self.context.logger.error(f"Error fetching historical price data: {str(e)}")
                return None
                
            
            # 5. Use stablecoin model to optimize bands
            # For stablecoin pools, we want narrow ranges
            # For volatile pools, we might adjust parameters
            model_params = {
                "ema_period": 50,  # Default from the model
                "std_dev_window": 100,  # Default from the model
                "verbose": True
            }
            
            # For stablecoin pools, we can use more aggressive settings
            if is_stable:
                model_params["min_width_pct"] = 0.0001  # Narrower bands for stablecoins
            
            # Run the optimization
            result = self.optimize_stablecoin_bands(
                prices=ratio_prices,
                **model_params
            )
            
            if not result:
                self.context.logger.error(f"Error in stablecoin model calculation")
                return None
            
            # 7. Calculate standard deviation for current window
            ema = self.calculate_ema(ratio_prices[-100:], model_params["ema_period"])
            std_dev = self.calculate_std_dev(ratio_prices[-100:], ema, model_params["std_dev_window"])
            current_std_dev = std_dev[-1]  # Use the most recent standard deviation
            
            # 8. Define a price to tick conversion function
            def price_to_tick(price: float) -> int:
                """Convert price to tick using the base 1.0001 formula."""
                # log base 1.0001 of the price
                return int(np.log(price) / np.log(1.0001))
            
            # 9. Calculate tick range using model band multipliers
            band_multipliers = result["band_multipliers"]
            
            # Get the most recent EMA value
            current_ema = ema[-1]
                
            # Calculate tick range using the exact formula: Upper bound = EMA + (sigma*multiplier)
            tick_range_results = self.calculate_tick_range_from_bands_wrapper(
                band_multipliers=band_multipliers,
                standard_deviation=current_std_dev,
                ema=current_ema,  # Use EMA instead of current_price
                tick_spacing=tick_spacing,
                price_to_tick_function=price_to_tick,
                min_tick=MIN_TICK,
                max_tick=MAX_TICK
            )
            
            
            # Prepare positions data for all three bands
            positions = []
            band_allocations = result["band_allocations"]
            
            for i, band_name in enumerate(["inner", "middle", "outer"]):
                band_data = tick_range_results[f"band{i+1}"]
                tick_lower = band_data["tick_lower"]
                tick_upper = band_data["tick_upper"]
                
                positions.append({
                    "tick_lower": tick_lower,
                    "tick_upper": tick_upper,
                    "allocation": band_allocations[i],
                })
            
            for p in positions:
                if p["tick_lower"] == p["tick_upper"]:
                    self.context.logger.info(
                        f"Adjusting position with equal ticks: tick_lower={p['tick_lower']}, tick_upper={p['tick_upper']}. "
                        f"Setting tick_upper to {p['tick_lower'] + tick_spacing}."
                    )
                    p["tick_upper"] = p["tick_lower"] + tick_spacing
                    self.context.logger.info(
                        f"Adjusted position: tick_lower={p['tick_lower']}, tick_upper={p['tick_upper']}."
                    )
                    
            tick_to_band = defaultdict(lambda: {"tick_lower": None, "tick_upper": None, "allocation": 0.0})

            for p in positions:
                key = (p["tick_lower"], p["tick_upper"])
                if tick_to_band[key]["tick_lower"] is None:
                    tick_to_band[key]["tick_lower"] = p["tick_lower"]
                    tick_to_band[key]["tick_upper"] = p["tick_upper"]
                tick_to_band[key]["allocation"] += p["allocation"]

            collapsed_positions = [
                {"tick_lower": v["tick_lower"], "tick_upper": v["tick_upper"], "allocation": v["allocation"]}
                for v in tick_to_band.values()
            ]

            self.context.logger.info(f"Collapsed positions before normalization: {collapsed_positions}")
            total_alloc = sum(p["allocation"] for p in collapsed_positions)
            self.context.logger.info(f"Total allocation before normalization: {total_alloc}")

            if total_alloc > 0:
                for p in collapsed_positions:
                    p["allocation"] /= total_alloc
                    self.context.logger.info(
                        f"Normalized allocation for position with ticks ({p['tick_lower']}, {p['tick_upper']}): {p['allocation']:.1%}"
                    )

            positions = collapsed_positions
            self.context.logger.info(f"Final positions after normalization: {positions}")
            self.context.logger.info("Band details (inner, middle, outer):")

            for i, position in enumerate(positions):
                band_name = ["inner", "middle", "outer"][i]
                self.context.logger.info(
                    f"  {band_name.upper()}: ticks=({position['tick_lower']}, {position['tick_upper']}), "
                    f"allocation={position['allocation']:.1%}"
                )
            
            self.context.logger.info(
                f"Model band multipliers: {band_multipliers[0]:.4f}, "
                f"{band_multipliers[1]:.4f}, {band_multipliers[2]:.4f}"
            )
            
            return positions
            
        except Exception as e:
            self.context.logger.error(f"Error in stablecoin model calculation: {str(e)}")
            return None
            
            
    def _get_pool_tokens(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str]]]:
        """Get the token addresses from a Velodrome pool."""
        try:
            # Call the token0 and token1 functions on the pool contract
            tokens = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=pool_address,
                contract_public_id=VelodromeCLPoolContract.contract_id,
                contract_callable="get_pool_tokens",
                data_key="tokens",
                chain_id=chain,
            )
            
            
            if not tokens:
                self.context.logger.error(f"Could not get token addresses for pool {pool_address}")
                return None, None
                
            return tokens[0], tokens[1]
            
        except Exception as e:
            self.context.logger.error(f"Error getting pool tokens: {str(e)}")
            return None, None

    def tick_to_price(self, tick: int, token0_decimals: int = 0, token1_decimals: int = 0) -> float:
        """
        Convert a tick to a price.
        
        Args:
            tick: The tick to convert
            token0_decimals: Number of decimals for token0
            token1_decimals: Number of decimals for token1
            
        Returns:
            The price corresponding to the tick
        """
        # The formula is: price = 1.0001^tick
        price = 1.0001 ** tick
        
        # Adjust for token decimals if provided
        if token0_decimals != 0 or token1_decimals != 0:
            price = price * (10 ** (token1_decimals - token0_decimals))
            
        return price
          
    def _calculate_token_ratios(
        self, price_lower: float, price_upper: float, current_price: float
    ) -> Tuple[float, float]:
        """
        Calculate the optimal ratio of token0 and token1 for a given price range.
        
        This function determines how to allocate tokens within a specific price band
        based on the current price and the band's boundaries.
        
        Args:
            price_lower: Lower price bound of the band
            price_upper: Upper price bound of the band
            current_price: Current price
            
        Returns:
            Tuple of (token0_ratio, token1_ratio) representing the optimal allocation
        """
        # If current price is outside the range, allocate 100% to one token
        if current_price <= price_lower:
            return 1.0, 0.0  # All token0
        elif current_price >= price_upper:
            return 0.0, 1.0  # All token1
        
        # Calculate L (virtual liquidity) for the price range
        # L = sqrt(P_upper) * sqrt(P_lower) / (sqrt(P_upper) - sqrt(P_lower))
        sqrt_price_upper = np.sqrt(price_upper)
        sqrt_price_lower = np.sqrt(price_lower)
        sqrt_price_current = np.sqrt(current_price)
        
        # Calculate token ratios based on the current price position within the range
        # These formulas are derived from Uniswap v3 whitepaper
        token0_ratio = (sqrt_price_upper - sqrt_price_current) / (sqrt_price_upper - sqrt_price_lower)
        token1_ratio = (sqrt_price_current - sqrt_price_lower) / (sqrt_price_upper - sqrt_price_lower)
        
        return token0_ratio, token1_ratio
        
    def optimize_stablecoin_bands(
        self,
        prices: List[float],
        ema_period: int = 50,
        std_dev_window: int = 100,
        min_width_pct: float = 0.00001,
        verbose: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Optimize multi-tier bands using Monte Carlo approach.
        
        This is a Python implementation of the R model in production-stablecoin-model.r
        
        Args:
            prices: Vector of price data
            ema_period: EMA period length
            std_dev_window: Standard deviation window size
            min_width_pct: Minimum band width as percentage
            verbose: Whether to print progress
            
        Returns:
            Dict with optimized band parameters
        """
        # Calculate EMA and standard deviation
        prices_array = np.array(prices)
        ema = self.calculate_ema(prices_array, ema_period)
        std_dev = self.calculate_std_dev(prices_array, ema, std_dev_window)
        
        # Calculate Z-scores
        z_scores = np.abs(prices_array - ema) / np.maximum(std_dev, 1e-6)  # Avoid division by zero
        
        # Define the three recursion levels similar to the R model
        recursion_levels = [
            {
                "name": "Level 1 (Initial)",
                "min_multiplier": 0.1,
                "max_multiplier": 1.5,
                "num_simulations": 250,
                "trigger_threshold": 0.95,  # >95% in inner band
                "trigger_multiplier": 0.15   # if multiplier < 0.15
            },
            {
                "name": "Level 2 (Narrow)",
                "min_multiplier": 0.01,
                "max_multiplier": 0.2,
                "num_simulations": 300,
                "trigger_threshold": 0.95,  # >95% in inner band
                "trigger_multiplier": 0.02   # if multiplier < 0.02
            },
            {
                "name": "Level 3 (Ultra-Narrow)",
                "min_multiplier": 0.001,
                "max_multiplier": 0.03,
                "num_simulations": 350,
                "trigger_threshold": None,  # Final level, no trigger
                "trigger_multiplier": None
            }
        ]
        
        # Initialize results storage
        results_by_level = []
        best_overall = None
        current_level = 0
        
        # Run through recursion levels
        while current_level < len(recursion_levels):
            level_config = recursion_levels[current_level]
            
            if verbose:
                self.context.logger.info(f"Running {level_config['name']} optimization "
                       f"({level_config['min_multiplier']}σ to {level_config['max_multiplier']}σ)")
            
            # Run Monte Carlo optimization for this level
            level_results = self._run_monte_carlo_level(
                prices_array, ema, std_dev, z_scores,
                level_config["min_multiplier"],
                level_config["max_multiplier"],
                level_config["num_simulations"],
                min_width_pct,
                verbose
            )
            
            # Store this level's results
            results_by_level.append(level_results)
            
            # Update best overall if needed
            if best_overall is None or level_results["best_config"]["zscore_economic_score"] > best_overall["best_config"]["zscore_economic_score"]:
                best_overall = {
                    "best_config": level_results["best_config"],
                    "level": current_level,
                    "level_name": level_config["name"]
                }
            
            # Check if we should move to the next recursion level
            if level_config["trigger_threshold"] is not None and level_config["trigger_multiplier"] is not None:
                best_config = level_results["best_config"]
                
                # Check if inner band allocation exceeds threshold and multiplier is below trigger
                if (best_config["band_allocations"][0] > level_config["trigger_threshold"] and 
                    best_config["band_multipliers"][0] < level_config["trigger_multiplier"]):
                    if verbose:
                        self.context.logger.info(f">> Triggering next recursion level <<")
                        self.context.logger.info(f"Inner band allocation: {best_config['band_allocations'][0]*100:.1f}% "
                               f"(threshold: {level_config['trigger_threshold']*100:.1f}%)")
                        self.context.logger.info(f"Inner band multiplier: {best_config['band_multipliers'][0]:.4f}σ "
                               f"(trigger: {level_config['trigger_multiplier']:.4f}σ)")
                    current_level += 1
                else:
                    if verbose:
                        self.context.logger.info("Optimal configuration found, no further recursion needed")
                    break  # Exit the recursion if trigger conditions not met
            else:
                # This is the final level, so we're done
                break
        
        # After all recursion levels, return the best overall result
        if verbose:
            self.context.logger.info(f"BEST OVERALL CONFIGURATION:")
            self.context.logger.info(f"From {best_overall['level_name']}")
            self.context.logger.info(f"Band Multipliers: {best_overall['best_config']['band_multipliers'][0]:.4f}, "
                   f"{best_overall['best_config']['band_multipliers'][1]:.4f}, "
                   f"{best_overall['best_config']['band_multipliers'][2]:.4f}")
            self.context.logger.info(f"Band Allocations: {best_overall['best_config']['band_allocations'][0]*100:.1f}%, "
                   f"{best_overall['best_config']['band_allocations'][1]*100:.1f}%, "
                   f"{best_overall['best_config']['band_allocations'][2]*100:.1f}%")
            self.context.logger.info(f"Z-Score Economic Score: {best_overall['best_config']['zscore_economic_score']:.4f}")
        
        # Return the best configuration
        return {
            "band_multipliers": best_overall["best_config"]["band_multipliers"],
            "band_allocations": best_overall["best_config"]["band_allocations"],
            "zscore_economic_score": best_overall["best_config"]["zscore_economic_score"],
            "percent_in_bounds": best_overall["best_config"]["percent_in_bounds"],
            "avg_weighted_width": best_overall["best_config"]["avg_weighted_width"],
            "from_level": best_overall["level"],
            "from_level_name": best_overall["level_name"]
        }

    def _run_monte_carlo_level(
        self,
        prices: np.ndarray,
        ema: np.ndarray,
        std_dev: np.ndarray,
        z_scores: np.ndarray,
        min_multiplier: float,
        max_multiplier: float,
        num_simulations: int,
        min_width_pct: float = 0.00001,
        verbose: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Run Monte Carlo optimization for a specific level.
        
        Args:
            prices: Array of price data
            ema: Array of EMA values
            std_dev: Array of standard deviation values
            z_scores: Array of Z-score values
            min_multiplier: Minimum multiplier for inner band
            max_multiplier: Maximum multiplier for inner band
            num_simulations: Number of Monte Carlo simulations
            min_width_pct: Minimum band width as percentage
            verbose: Whether to print progress
            
        Returns:
            Dict with best configuration and all results
        """
        # Prepare arrays to hold simulation results
        sim_nums = np.arange(num_simulations)
        m1_values = np.zeros(num_simulations)
        m2_values = np.zeros(num_simulations)
        m3_values = np.zeros(num_simulations)
        a1_values = np.zeros(num_simulations)
        a2_values = np.zeros(num_simulations)
        a3_values = np.zeros(num_simulations)
        percent_in_bounds_values = np.zeros(num_simulations)
        avg_weighted_width_values = np.zeros(num_simulations)
        zscore_economic_score_values = np.zeros(num_simulations)
        
        # Run simulations
        for sim_num in range(num_simulations):
            if verbose and sim_num % 20 == 0:
                self.context.logger.info(f"Level {min_multiplier:.4f}-{max_multiplier:.4f}: "
                       f"Simulation {sim_num+1}/{num_simulations}")
            
            # Generate random inner band multiplier within range
            m1 = np.random.uniform(min_multiplier, max_multiplier)
            
            # Apply proportional scaling for middle and outer bands
            m2_min = m1 * 1.5
            m2_max = m1 * 2.5
            m2 = np.random.uniform(m2_min, m2_max)
            
            m3_min = m2 * 1.3
            m3_max = m2 * 2.0
            m3 = np.random.uniform(m3_min, m3_max)
            
            # Random allocation with minimum 0.0001 (0.01%) per band
            # Higher probability of selecting high inner band allocation
            if np.random.random() < 0.7:  # 70% chance of selecting from high allocation range
                a1 = np.random.uniform(0.95, 0.998)  # Focus on high inner band allocations
            else:
                a1 = np.random.uniform(0.5, 0.95)    # Also test lower allocations
            
            # Distribute remaining allocation
            remaining = 1.0 - a1
            a2_proportion = np.random.uniform(0.6, 0.8)  # Middle gets 60-80% of remainder
            a2 = max(0.0001, remaining * a2_proportion)
            a3 = 1.0 - a1 - a2
            
            # Ensure minimum allocation
            if a3 < 0.0001:
                a3 = 0.0001
                a2 = 1.0 - a1 - a3
            
            # Double-check allocation sum (floating point errors)
            total = a1 + a2 + a3
            if abs(total - 1.0) > 1e-10:
                a1 = a1 / total  # Normalize to ensure exact sum of 1.0
                a2 = a2 / total
                a3 = a3 / total
            
            # Combine parameters
            band_multipliers = np.array([m1, m2, m3])
            band_allocations = np.array([a1, a2, a3])
            
            # Evaluate band configuration
            result = self._evaluate_band_configuration(
                z_scores, prices, ema, std_dev,
                band_multipliers, band_allocations,
                min_width_pct
            )
            
            # Store results
            m1_values[sim_num] = m1
            m2_values[sim_num] = m2
            m3_values[sim_num] = m3
            a1_values[sim_num] = a1
            a2_values[sim_num] = a2
            a3_values[sim_num] = a3
            percent_in_bounds_values[sim_num] = result["percent_in_bounds"]
            avg_weighted_width_values[sim_num] = result["avg_weighted_width"]
            zscore_economic_score_values[sim_num] = result["zscore_economic_score"]
        
        # Find the best configuration (highest Z-score economic score)
        best_idx = np.argmax(zscore_economic_score_values)
        
        # Extract the top configuration
        top_config = {
            "band_multipliers": np.array([m1_values[best_idx], m2_values[best_idx], m3_values[best_idx]]),
            "band_allocations": np.array([a1_values[best_idx], a2_values[best_idx], a3_values[best_idx]]),
            "zscore_economic_score": zscore_economic_score_values[best_idx],
            "percent_in_bounds": percent_in_bounds_values[best_idx],
            "avg_weighted_width": avg_weighted_width_values[best_idx]
        }
        
        if verbose:
            self.context.logger.info(f"Best configuration for level {min_multiplier:.4f}-{max_multiplier:.4f}:")
            self.context.logger.info(f"  Inner Band: {top_config['band_multipliers'][0]:.4f}σ "
                   f"({top_config['band_allocations'][0]*100:.1f}% allocation)")
            self.context.logger.info(f"  Middle Band: {top_config['band_multipliers'][1]:.4f}σ "
                   f"({top_config['band_allocations'][1]*100:.1f}% allocation)")
            self.context.logger.info(f"  Outer Band: {top_config['band_multipliers'][2]:.4f}σ "
                   f"({top_config['band_allocations'][2]*100:.1f}% allocation)")
            self.context.logger.info(f"  Performance: {top_config['percent_in_bounds']:.2f}% in bounds, "
                   f"{top_config['avg_weighted_width']:.4f}% weighted width")
            self.context.logger.info(f"  Z-Score Economic Score: {top_config['zscore_economic_score']:.4f}")
        
        # Return the best configuration and all results
        return {
            "best_config": top_config,
            "all_results": {
                "sim_nums": sim_nums,
                "m1_values": m1_values,
                "m2_values": m2_values,
                "m3_values": m3_values,
                "a1_values": a1_values,
                "a2_values": a2_values,
                "a3_values": a3_values,
                "percent_in_bounds_values": percent_in_bounds_values,
                "avg_weighted_width_values": avg_weighted_width_values,
                "zscore_economic_score_values": zscore_economic_score_values
            }
        }

    def _evaluate_band_configuration(
        self,
        z_scores: np.ndarray,
        prices: np.ndarray,
        ema: np.ndarray,
        std_dev: np.ndarray,
        band_multipliers: np.ndarray,
        band_allocations: np.ndarray,
        min_width_pct: float = 0.00001
    ) -> Dict[str, Any]:
        """
        Evaluate a specific band configuration.
        
        Args:
            z_scores: Array of Z-score values
            prices: Array of price data
            ema: Array of EMA values
            std_dev: Array of standard deviation values
            band_multipliers: Array of 3 band multipliers
            band_allocations: Array of 3 band allocations
            min_width_pct: Minimum band width as percentage
            
        Returns:
            Dict with evaluation metrics
        """
        # Calculate band regions using Z-scores
        # Note: R uses 1-indexed arrays, but Python uses 0-indexed arrays
        # So band_multipliers[0] in Python corresponds to band_multipliers[1] in R
        band1_mask = z_scores <= band_multipliers[0]
        band2_mask = (z_scores > band_multipliers[0]) & (z_scores <= band_multipliers[1])
        band3_mask = (z_scores > band_multipliers[1]) & (z_scores <= band_multipliers[2])
        
        # Calculate price coverage by band
        band1_count = np.sum(band1_mask)
        band2_count = np.sum(band2_mask)
        band3_count = np.sum(band3_mask)
        total_count = len(z_scores)
        
        # Calculate percentage in bounds (all bands combined)
        percent_in_bounds = (band1_count + band2_count + band3_count) / total_count * 100
        
        # Calculate average weighted width based on allocations and band multipliers
        avg_width_pct = np.mean(std_dev / ema * 100)  # Base width as percentage of price
        
        # Calculate weighted width based on band allocation and multipliers
        # This matches the R implementation exactly, accounting for 0-indexing in Python
        b1_width = 2 * band_multipliers[0] * avg_width_pct * band_allocations[0]
        b2_width = 2 * (band_multipliers[1] - band_multipliers[0]) * avg_width_pct * band_allocations[1]
        b3_width = 2 * (band_multipliers[2] - band_multipliers[1]) * avg_width_pct * band_allocations[2]
        
        # Ensure minimum width
        b1_width = max(b1_width, min_width_pct * band_allocations[0])
        b2_width = max(b2_width, min_width_pct * band_allocations[1])
        b3_width = max(b3_width, min_width_pct * band_allocations[2])
        
        # Total weighted width
        avg_weighted_width = b1_width + b2_width + b3_width
        
        # Calculate Z-score weighted coverage
        band1_coverage = band1_count / total_count
        band2_coverage = band2_count / total_count
        band3_coverage = band3_count / total_count
        
        weighted_coverage = (
            band1_coverage * band_allocations[0] + 
            band2_coverage * band_allocations[1] + 
            band3_coverage * band_allocations[2]
        )
        
        # Calculate Z-score economic score
        zscore_economic_score = weighted_coverage * (1 / avg_weighted_width) * 100
        
        # Return results
        return {
            "percent_in_bounds": percent_in_bounds,
            "avg_weighted_width": avg_weighted_width,
            "zscore_economic_score": zscore_economic_score,
            "band_coverage": [band1_coverage, band2_coverage, band3_coverage]
        }

    def get_pool_token_history(
        self,
        chain: str, 
        token0_address: str, 
        token1_address: str, 
        days: int = DEFAULT_DAYS,
        api_key: Optional[str] = None
    ) -> Generator[None, None, Dict[str, Any]]:
        """
        Get historical price data for both tokens in a pool.
        
        Args:
            chain: The blockchain name (e.g., "optimism")
            token0_address: Token0 contract address
            token1_address: Token1 contract address
            days: Number of days of history
            api_key: Optional CoinGecko API key
            
        Returns:
            Dictionary with price history for both tokens and ratio prices
        """
        self.context.logger.info(f"Fetching historical price data for tokens: {token0_address} and {token1_address}")
        
        try:
            # Initialize headers for API requests
            headers = {
                "Accept": "application/json",
            }
            if api_key:
                headers["x-cg-api-key"] = api_key
            
            # Convert chain name to CoinGecko platform ID
            platform_map = {
                "ethereum": "ethereum",
                "optimism": "optimistic-ethereum",
                "arbitrum": "arbitrum-one",
                "polygon": "polygon-pos",
                "base": "base",
                "mode": "mode",
            }
            
            platform = platform_map.get(chain.lower())
            if not platform:
                self.context.logger.warning(f"Unsupported chain: {chain}")
                return None
            
            # Get coin IDs for both tokens
            token0_id = yield from self._get_coin_id_from_address(chain, token0_address, platform, headers)
            token1_id = yield from self._get_coin_id_from_address(chain, token1_address, platform, headers)
            
            if not token0_id or not token1_id:
                self.context.logger.warning(f"Could not find coin IDs for tokens: {token0_address}, {token1_address}")
                return None
            
            # Get price history for both tokens
            token0_data = yield from self._get_historical_market_data(token0_id, days, headers)
            token1_data = yield from self._get_historical_market_data(token1_id, days, headers)
            
            # Check if we have valid price data
            if not token0_data or not token1_data or not token0_data.get("prices") or not token1_data.get("prices"):
                self.context.logger.warning("Missing price data for one or both tokens")
                return None
            
            # Combine data and calculate price ratios
            token0_prices = token0_data.get("prices", [])
            token1_prices = token1_data.get("prices", [])
            token0_timestamps = token0_data.get("timestamps", [])
            token1_timestamps = token1_data.get("timestamps", [])
            
            # Ensure price lists are the same length by taking the shorter one
            min_length = min(len(token0_prices), len(token1_prices))
            token0_prices = token0_prices[:min_length]
            token1_prices = token1_prices[:min_length]
            token0_timestamps = token0_timestamps[:min_length]
            token1_timestamps = token1_timestamps[:min_length]
            
            # Calculate ratio prices (token0/token1)
            ratio_prices = []
            for i in range(min_length):
                if token1_prices[i] > 0:
                    ratio_prices.append(token0_prices[i] / token1_prices[i])
                else:
                    ratio_prices.append(token0_prices[i])  # Avoid division by zero
            
            # Get current price (latest)
            current_price = ratio_prices[-1] if ratio_prices else 1.0
            
            
            return {
                "ratio_prices": ratio_prices,
                "current_price": current_price,
                "days": days
            }
        except Exception as e:
            self.context.logger.error(f"Error getting pool token history: {str(e)}")
            return None
            
    def _get_coin_id_from_address(
        self,
        chain: str, 
        address: str, 
        platform: str,
        headers: Dict[str, str]
    ) -> Generator[None, None, Optional[str]]:
        """
        Get CoinGecko coin ID from token address.
        
        Args:
            chain: The blockchain name
            address: Token contract address
            platform: CoinGecko platform ID
            headers: Request headers
            
        Returns:
            CoinGecko coin ID or None if not found
        """
        try:
            # Rate limiting to avoid CoinGecko API restrictions
            yield from self.sleep(1)
            
            # Make the API request
            endpoint = f"https://api.coingecko.com/api/v3/coins/{platform}/contract/{address}"
            response = yield from self.get_http_response(
                method="GET",
                url=endpoint,
                headers=headers,
                content=None
            )
            
            # Check if response has status_code attribute (HTTP response object)
            if hasattr(response, 'status_code'):
                if response.status_code != 200:
                    self.context.logger.warning(f"Error getting coin ID for {chain}/{address}: {response.status_code}")
                    return None
                
                try:
                    coin_data = json.loads(response.body)
                    return coin_data.get("id")
                except json.JSONDecodeError:
                    self.context.logger.warning(f"Error parsing response for {chain}/{address}")
                    return None
            else:
                # Response is already parsed as a dictionary
                if isinstance(response, dict) and "id" in response:
                    return response.get("id")
                else:
                    self.context.logger.warning(f"Unexpected response format for {chain}/{address}")
                    return None
                
        except Exception as e:
            self.context.logger.warning(f"Error getting coin ID for {chain}/{address}: {str(e)}")
            return None

    
    def _get_historical_market_data(
        self,
        coin_id: str, 
        days: int,
        headers: Dict[str, str]
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """
        Get historical market data for a coin from CoinGecko, with robust rate limit handling.
        """

        endpoint = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
        yield from self.sleep(2)
        try:
            response = yield from self.get_http_response(
                method="GET",
                url=endpoint,
                headers=headers,
                content=None
            )
            # Try to parse the response body as JSON
            try:
                response_json = json.loads(response.body)
            except Exception:
                response_json = {}

            # Check for HTTP 429 or JSON error code 429
            is_rate_limited = False
            if hasattr(response_json, 'status_code') and response_json.status_code == 429:
                is_rate_limited = True
            elif isinstance(response_json, dict):
                status = response_json.get("status", {})
                if status.get("error_code") == 429:
                    is_rate_limited = True

            if is_rate_limited:
                self.context.logger.error(
                    f"Rate limit reached on CoinGecko API. Waiting for 10 seconds before retrying... (Attempt {attempt + 1} of {retries})"
                )
                yield from self.sleep(10)
                return None

            # Handle other non-200 responses
            if hasattr(response_json, 'status_code') and response_json.status_code != 200:
                self.context.logger.warning(
                    f"Error getting market data for {coin_id}: {response_json.status_code}"
                )
                return None

            # Parse response body
            try:
                prices_data = response_json.get("prices", [])
                timestamps = [entry[0]/1000 for entry in prices_data]  # ms to seconds
                prices = [entry[1] for entry in prices_data]
                return {
                    "coin_id": coin_id,
                    "timestamps": timestamps,
                    "prices": prices,
                    "days": days,
                    "last_updated": time.time()
                }
            except json.JSONDecodeError:
                self.context.logger.warning(f"Error parsing response for {coin_id}")
                return None

        except Exception as e:
            self.context.logger.warning(f"Error getting market data for {coin_id}: {str(e)}")
            return None

        
    def _get_current_pool_price(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[float]]:
        """Get the current price from a Velodrome concentrated liquidity pool."""
        try:
            # Get the sqrt_price_x96 from the pool
            sqrt_price_x96 = yield from self._get_sqrt_price_x96(pool_address, chain)
            if sqrt_price_x96 is None:
                return None
                
            # Convert sqrt_price_x96 to price
            # The formula is: price = (sqrt_price_x96 / 2^96)^2
            price = (sqrt_price_x96 / (2**96))**2
            self.context.logger.info(f"Current pool price: {price}")
            return price
        except Exception as e:
            self.context.logger.error(f"Error getting current pool price: {str(e)}")
            return None
        
    
    def calculate_tick_range_from_bands_wrapper(
        self,
        band_multipliers: List[float],
        standard_deviation: float,
        ema: float,
        tick_spacing: int,
        price_to_tick_function: Callable,
        min_tick: int = MIN_TICK,
        max_tick: int = MAX_TICK
    ) -> Dict[str, Any]:
        """
        Bridge function to calculate tick ranges from band multipliers.
        
        Args:
            band_multipliers: List of band multipliers [inner, middle, outer]
            standard_deviation: Current standard deviation (sigma)
            ema: Exponential Moving Average value
            tick_spacing: Pool tick spacing
            price_to_tick_function: Function to convert price to tick
            min_tick: Minimum allowed tick
            max_tick: Maximum allowed tick
            
        Returns:
            Dictionary with tick ranges for each band
        """
        # Convert band multipliers to price ranges using the formula:
        # Upper bound = EMA + (sigma*multiplier)
        # Lower bound = EMA - (sigma*multiplier)
        
        # Calculate band price ranges
        band1_lower = ema - (band_multipliers[0] * standard_deviation)
        band1_upper = ema + (band_multipliers[0] * standard_deviation)
        self.context.logger.info(f"Band 1 price range: lower={band1_lower}, upper={band1_upper}")
        
        band2_lower = ema - (band_multipliers[1] * standard_deviation)
        band2_upper = ema + (band_multipliers[1] * standard_deviation)
        self.context.logger.info(f"Band 2 price range: lower={band2_lower}, upper={band2_upper}")
        
        band3_lower = ema - (band_multipliers[2] * standard_deviation)
        band3_upper = ema + (band_multipliers[2] * standard_deviation)
        self.context.logger.info(f"Band 3 price range: lower={band3_lower}, upper={band3_upper}")

        # Convert to ticks and round to tick spacing
        def round_to_spacing(tick):
            return int(tick // tick_spacing) * tick_spacing
        
        # Convert prices to ticks
        band1_tick_lower = round_to_spacing(price_to_tick_function(band1_lower))
        band1_tick_upper = round_to_spacing(price_to_tick_function(band1_upper))
        self.context.logger.info(f"Band 1 ticks: lower={band1_tick_lower}, upper={band1_tick_upper}")
        
        band2_tick_lower = round_to_spacing(price_to_tick_function(band2_lower))
        band2_tick_upper = round_to_spacing(price_to_tick_function(band2_upper))
        self.context.logger.info(f"Band 2 ticks: lower={band2_tick_lower}, upper={band2_tick_upper}")
        
        band3_tick_lower = round_to_spacing(price_to_tick_function(band3_lower))
        band3_tick_upper = round_to_spacing(price_to_tick_function(band3_upper))
        self.context.logger.info(f"Band 3 ticks: lower={band3_tick_lower}, upper={band3_tick_upper}")
        
        # Ensure ticks are within allowed range
        band1_tick_lower = max(min_tick, min(max_tick, band1_tick_lower))
        band1_tick_upper = max(min_tick, min(max_tick, band1_tick_upper))
        self.context.logger.info(f"Band 1 ticks adjusted: lower={band1_tick_lower}, upper={band1_tick_upper}")
        
        band2_tick_lower = max(min_tick, min(max_tick, band2_tick_lower))
        band2_tick_upper = max(min_tick, min(max_tick, band2_tick_upper))
        self.context.logger.info(f"Band 2 ticks adjusted: lower={band2_tick_lower}, upper={band2_tick_upper}")
        
        band3_tick_lower = max(min_tick, min(max_tick, band3_tick_lower))
        band3_tick_upper = max(min_tick, min(max_tick, band3_tick_upper))
        self.context.logger.info(f"Band 3 ticks adjusted: lower={band3_tick_lower}, upper={band3_tick_upper}")
        
        
        # Build result dictionary
        return {
            "band1": {
                "tick_lower": band1_tick_lower,
                "tick_upper": band1_tick_upper,
            },
            "band2": {
                "tick_lower": band2_tick_lower,
                "tick_upper": band2_tick_upper,
            },
            "band3": {
                "tick_lower": band3_tick_lower,
                "tick_upper": band3_tick_upper,
            },
            "inner_ticks": (band1_tick_lower, band1_tick_upper),
            "middle_ticks": (band2_tick_lower, band2_tick_upper),
            "outer_ticks": (band3_tick_lower, band3_tick_upper)
        } 
    
    def calculate_ema(self, prices: List[float], period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average - compatibility with pools implementation.
        
        Args:
            prices: Vector of price data
            period: EMA period length
        
        Returns:
            Vector of EMA values
        """
        prices_array = np.array(prices)
        ema = np.zeros_like(prices_array)
        
        # Initialize with first price
        ema[0] = prices_array[0]
        
        # Calculate EMA
        alpha = 2 / (period + 1)
        for i in range(1, len(prices_array)):
            ema[i] = prices_array[i] * alpha + ema[i-1] * (1 - alpha)
        
        return ema


    def calculate_std_dev(self, prices: List[float], ema: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate Rolling Standard Deviation - compatibility with pools implementation.
        
        Args:
            prices: Vector of price data
            ema: Vector of EMA values
            window: Rolling window size
        
        Returns:
            Vector of standard deviation values
        """
        prices_array = np.array(prices)
        length = len(prices_array)
        std_dev = np.zeros(length)
        
        # Calculate rolling standard deviation
        for i in range(window - 1, length):
            window_prices = prices_array[i-window+1:i+1]
            window_ema = ema[i-window+1:i+1]
            deviations = window_prices - window_ema
            std_dev[i] = np.std(deviations)
        
        # Fill initial values
        for i in range(window - 1):
            if i > 0:
                window_prices = prices_array[:i+1]
                window_ema = ema[:i+1]
                deviations = window_prices - window_ema
                std_dev[i] = np.std(deviations)
            else:
                std_dev[i] = 0.001 * prices_array[i]  # Small default value
        
        return std_dev

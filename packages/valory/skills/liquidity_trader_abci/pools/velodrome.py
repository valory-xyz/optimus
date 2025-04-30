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
from abc import ABC
from enum import Enum
import numpy as np
from typing import Any, Dict, Generator, List, Optional, Tuple, cast

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
from packages.valory.skills.liquidity_trader_abci.pools import (
    calculate_ema,
    calculate_std_dev,
    optimize_stablecoin_bands,
    calculate_tick_range_from_bands,
)
from packages.valory.skills.liquidity_trader_abci.utils.price_history import (
    get_pool_token_history,
    get_stablecoin_pair_history,
    check_is_stablecoin_pool,
)


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
MIN_TICK = -887272
MAX_TICK = 887272
INT_MAX = sys.maxsize



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
            return (yield from self._exit_cl_pool(
                token_id=kwargs.get("token_id"),
                safe_address=safe_address,
                chain=chain,
                liquidity=kwargs.get("liquidity"),
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
        pool_fee: Optional[int] = None,
    ) -> Generator[None, None, Optional[Tuple[str, str]]]:
        """Add liquidity to a Velodrome Concentrated Liquidity pool."""
        # Get NonFungiblePositionManager contract address
        position_manager_address = self.params.velodrome_non_fungible_position_manager_contract_addresses.get(chain, "")
        if not position_manager_address:
            self.context.logger.error(
                f"No NonFungiblePositionManager contract address found for chain {chain}"
            )
            return None, None

        # Calculate tick range based on pool's tick spacing
        # tick_lower, tick_upper = yield from self._calculate_tick_lower_and_upper_velodrome(
        #     pool_address, chain
        # )
        tick_lower = MIN_TICK
        tick_upper = MAX_TICK
        if not tick_lower or not tick_upper:
            return None, None

        # Get tick spacing for the pool
        tick_spacing = yield from self._get_tick_spacing_velodrome(pool_address, chain)
        if not tick_spacing:
            self.context.logger.error(f"Could not get tick spacing for pool {pool_address}")
            return None, None

        # Get or calculate sqrt_price_x96
        sqrt_price_x96 = 0
        if sqrt_price_x96 is None:
            self.context.logger.error(f"Could not determine sqrt_price_x96 for pool {pool_address}")
            return None, None

        # TO-DO: add slippage protection
        amount0_min = 0
        amount1_min = 0

        # deadline is set to be 20 minutes from current time
        last_update_time = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()
        deadline = int(last_update_time) + (20 * 60)

        # Call mint on the CLPoolManager contract
        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=position_manager_address,
            contract_public_id=VelodromeNonFungiblePositionManagerContract.contract_id,
            contract_callable="mint",
            data_key="tx_hash",
            token0=assets[0],
            token1=assets[1],
            tick_spacing=tick_spacing,
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            amount0_desired=max_amounts_in[0],
            amount1_desired=max_amounts_in[1],
            amount0_min=amount0_min,
            amount1_min=amount1_min,
            recipient=safe_address,
            deadline=deadline,
            sqrt_price_x96=sqrt_price_x96,
            chain_id=chain,
        )

        return tx_hash, position_manager_address

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
        token_id: int,
        safe_address: str,
        chain: str,
        liquidity: Optional[int] = None,
    ) -> Generator[None, None, Optional[Tuple[str, str, bool]]]:
        """Remove liquidity from a Velodrome Concentrated Liquidity pool."""

        position_manager_address = self.params.velodrome_non_fungible_position_manager_contract_addresses.get(chain, "")
        if not position_manager_address:
            self.context.logger.error(
                f"No CLPoolManager contract address found for chain {chain}"
            )
            return None, None, None
            
        multi_send_txs = []

        # decrease liquidity
        # TO-DO: Calculate min amounts accounting for slippage
        amount0_min = 0
        amount1_min = 0

        # fetch liquidity from contract if not provided
        if not liquidity:
            liquidity = yield from self.get_liquidity_for_token_velodrome(token_id, chain)
            if not liquidity:
                return None, None, None

        # deadline is set to be 20 minutes from current time
        last_update_time = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()
        deadline = int(last_update_time) + (20 * 60)

        decrease_liquidity_tx_hash = yield from self.decrease_liquidity_velodrome(
            token_id, liquidity, amount0_min, amount1_min, deadline, chain
        )
        if not decrease_liquidity_tx_hash:
            return None, None, None
        multi_send_txs.append(
            {
                "operation": MultiSendOperation.CALL,
                "to": position_manager_address,
                "value": 0,
                "data": decrease_liquidity_tx_hash,
            }
        )

        # collect the tokens
        # Use a high max value similar to Uniswap's approach
        amount_max = Web3.to_wei(2**100 - 1, "wei")
        collect_tokens_tx_hash = yield from self.collect_tokens_velodrome(
            token_id=token_id,
            recipient=safe_address,
            amount0_max=amount_max,
            amount1_max=amount_max,
            chain=chain,
        )
        if not collect_tokens_tx_hash:
            return None, None, None
        multi_send_txs.append(
            {
                "operation": MultiSendOperation.CALL,
                "to": position_manager_address,
                "value": 0,
                "data": collect_tokens_tx_hash,
            }
        )

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
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[Tuple[int, int]]]:
        """
        Calculate tick range for Velodrome concentrated liquidity pool.
        
        Uses historical price data and the stablecoin model to determine optimal tick bounds.
        
        Args:
            pool_address: The address of the Velodrome concentrated liquidity pool
            chain: The blockchain chain ID
            
        Returns:
            A tuple of (lower_tick, upper_tick) or None if calculation fails
        """
        self.context.logger.info(f"Calculating tick range using stablecoin model for pool {pool_address}")
        
        # 1. Fetch tick spacing from velodrome cl pool
        tick_spacing = yield from self._get_tick_spacing_velodrome(pool_address, chain)
        if not tick_spacing:
            self.context.logger.error(f"Failed to get tick spacing for pool {pool_address}")
            return None, None
            
        # 2. Get the token addresses from the pool
        token0, token1 = yield from self._get_pool_tokens(pool_address, chain)
        if not token0 or not token1:
            self.context.logger.error(f"Failed to get tokens for pool {pool_address}")
            return (yield from self._calculate_basic_tick_range(tick_spacing))
        
        # 3. Get current price
        current_price = yield from self._get_current_pool_price(pool_address, chain)
        if current_price is None:
            self.context.logger.error(f"Failed to get current price for pool {pool_address}")
            return (yield from self._calculate_basic_tick_range(tick_spacing))
            
        try:
            # 4. Get historical price data for both tokens and calculate price ratio history
            self.context.logger.info(f"Fetching historical price data for tokens: {token0} and {token1}")
            pool_data = get_pool_token_history(
                chain=chain, 
                token0_address=token0, 
                token1_address=token1
            )
            
            # Check if we have price data
            token0_prices = pool_data["token0"]["prices"]
            token1_prices = pool_data["token1"]["prices"]
            ratio_prices = pool_data["ratio_prices"]
            
            if not ratio_prices:
                self.context.logger.warning(
                    f"Could not get price ratio history for pool {pool_address}. "
                    f"Falling back to basic tick range."
                )
                return (yield from self._calculate_basic_tick_range(tick_spacing))
                
            # 5. Check if this is a stablecoin pool (both tokens are stablecoins)
            # This is optional but helps tailor the approach
            is_stablecoin_pool = False
            if token0_prices and token1_prices:
                is_stablecoin_pool = check_is_stablecoin_pool(token0_prices, token1_prices)
                
            self.context.logger.info(
                f"Pool {pool_address} identified as {'stablecoin' if is_stablecoin_pool else 'regular'} pool"
            )
            
            # 6. Use stablecoin model to optimize bands
            # For stablecoin pools, we want narrow ranges
            # For volatile pools, we might adjust parameters
            model_params = {
                "ema_period": 50,  # Default from the model
                "std_dev_window": 100,  # Default from the model
                "verbose": False
            }
            
            # For stablecoin pools, we can use more aggressive settings
            if is_stablecoin_pool:
                model_params["min_width_pct"] = 0.0001  # Narrower bands for stablecoins
            
            # Run the optimization
            result = optimize_stablecoin_bands(
                prices=ratio_prices,
                **model_params
            )
            
            # 7. Calculate standard deviation for current window
            ema = calculate_ema(ratio_prices[-100:], model_params["ema_period"])
            std_dev = calculate_std_dev(ratio_prices[-100:], ema, model_params["std_dev_window"])
            current_std_dev = std_dev[-1]  # Use the most recent standard deviation
            
            # 8. Define a price to tick conversion function
            def price_to_tick(price: float) -> int:
                """Convert price to tick using the base 1.0001 formula."""
                # log base 1.0001 of the price
                return int(np.log(price) / np.log(1.0001))
            
            # 9. Calculate tick range using model band multipliers
            band_multipliers = result["band_multipliers"]
            
            # For stablecoin pools, use a more conservative range but still use the 
            # calculate_tick_range_from_bands function for consistent processing
            if is_stablecoin_pool:
                # Use a more conservative multiplier for stablecoins
                sigma_range = (band_multipliers[0] + band_multipliers[1]) / 2
                self.context.logger.info(f"Using stablecoin sigma range: {sigma_range:.4f}")
                
                # Create modified band_multipliers with all three bands set to our chosen sigma_range
                # This ensures the tick calculation will use our preferred range
                band_multipliers = [sigma_range, sigma_range * 1.5, sigma_range * 2]
                
            # Calculate tick range 
            tick_range_results = calculate_tick_range_from_bands(
                band_multipliers=band_multipliers,
                standard_deviation=current_std_dev,
                current_price=current_price,
                tick_spacing=tick_spacing,
                price_to_tick_function=price_to_tick,
                min_tick=MIN_TICK,
                max_tick=MAX_TICK
            )
            
            # For backward compatibility, use the outer band ticks
            tick_lower, tick_upper = tick_range_results["outer_ticks"]
                
            # Ensure we have a meaningful range (at least 10 ticks)
            min_range = 10 * tick_spacing
            if tick_upper - tick_lower < min_range:
                # Expand the range
                midpoint = (tick_upper + tick_lower) // 2
                tick_lower = midpoint - (min_range // 2)
                tick_upper = midpoint + (min_range // 2)
                # Re-adjust to be within bounds
                tick_lower = max(tick_lower, MIN_TICK)
                tick_upper = min(tick_upper, MAX_TICK)
            
            self.context.logger.info(
                f"Stablecoin model calculated tick range: LOWER={tick_lower}, UPPER={tick_upper}"
            )
            
            # Log information about all three bands
            self.context.logger.info("Band details (inner, middle, outer):")
            for i, band_name in enumerate(["inner", "middle", "outer"]):
                band_data = tick_range_results[f"band{i+1}"]
                self.context.logger.info(
                    f"  {band_name.upper()}: ticks=({band_data['tick_lower']}, {band_data['tick_upper']}), "
                    f"ratio={band_data['price_ratio']:.4f}"
                )
            
            self.context.logger.info(
                f"Model band multipliers: {band_multipliers[0]:.4f}, "
                f"{band_multipliers[1]:.4f}, {band_multipliers[2]:.4f}"
            )
            
            # Log recommended allocation ratios
            self.context.logger.info(
                f"Recommended band allocations: {result['band_allocations'][0]:.1%}, "
                f"{result['band_allocations'][1]:.1%}, {result['band_allocations'][2]:.1%}"
            )
            
            return tick_lower, tick_upper
            
        except Exception as e:
            self.context.logger.error(f"Error in stablecoin model calculation: {str(e)}")
            self.context.logger.info("Falling back to basic tick range calculation")
            return (yield from self._calculate_basic_tick_range(tick_spacing))
            
    def _get_pool_tokens(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str]]]:
        """Get the token addresses from a Velodrome pool."""
        try:
            # Call the token0 and token1 functions on the pool contract
            token0 = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=pool_address,
                contract_public_id=VelodromeCLPoolContract.contract_id,
                contract_callable="token0",
                data_key="token0",
                chain_id=chain,
            )
            
            token1 = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=pool_address,
                contract_public_id=VelodromeCLPoolContract.contract_id,
                contract_callable="token1",
                data_key="token1",
                chain_id=chain,
            )
            
            if not token0 or not token1:
                self.context.logger.error(f"Could not get token addresses for pool {pool_address}")
                return None, None
                
            return token0, token1
            
        except Exception as e:
            self.context.logger.error(f"Error getting pool tokens: {str(e)}")
            return None, None

    def _calculate_basic_tick_range(
        self, tick_spacing: int
    ) -> Generator[None, None, Tuple[int, int]]:
        """Calculate basic tick range using min/max ticks adjusted to tick spacing."""
        # Adjust MIN_TICK to the nearest higher multiple of tick_spacing
        adjusted_tick_lower = abs(MIN_TICK) // tick_spacing * tick_spacing
        if adjusted_tick_lower > abs(MIN_TICK):
            adjusted_tick_lower = adjusted_tick_lower - tick_spacing
        adjusted_tick_lower = -adjusted_tick_lower

        # Adjust MAX_TICK to the nearest lower multiple of tick_spacing
        adjusted_tick_upper = MAX_TICK // tick_spacing * tick_spacing
        if adjusted_tick_upper > MAX_TICK:
            adjusted_tick_upper = adjusted_tick_upper - tick_spacing

        self.context.logger.info(
            f"Basic tick range: LOWER={adjusted_tick_lower}, UPPER={adjusted_tick_upper}"
        )
        return adjusted_tick_lower, adjusted_tick_upper
        
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

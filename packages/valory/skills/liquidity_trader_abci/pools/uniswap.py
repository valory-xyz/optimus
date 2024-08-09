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

"""This package contains the implemenatation of the BalancerPoolBehaviour class."""

import sys
from abc import ABC
from typing import Any, Dict, Generator, List, Optional, Tuple, cast

from packages.valory.contracts.uniswap_v3_non_fungible_position_manager.contract import (
    UniswapV3NonfungiblePositionManagerContract,
)
from packages.valory.contracts.uniswap_v3_pool.contract import UniswapV3PoolContract
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.liquidity_trader_abci.models import SharedState
from packages.valory.skills.liquidity_trader_abci.pool_behaviour import PoolBehaviour


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
MIN_TICK = -887272
MAX_TICK = 887272
INT_MAX = sys.maxsize


class MintParams:
    def __init__(
        self,
        token0,
        token1,
        fee,
        tickLower,
        tickUpper,
        amount0Desired,
        amount1Desired,
        amount0Min,
        amount1Min,
        recipient,
        deadline,
    ):
        self.token0 = token0
        self.token1 = token1
        self.fee = fee
        self.tickLower = tickLower
        self.tickUpper = tickUpper
        self.amount0Desired = amount0Desired
        self.amount1Desired = amount1Desired
        self.amount0Min = amount0Min
        self.amount1Min = amount1Min
        self.recipient = recipient
        self.deadline = deadline


class UniswapPoolBehaviour(PoolBehaviour, ABC):
    """BalancerPoolBehaviour"""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the balancer pool behaviour."""
        super().__init__(**kwargs)
        # https://docs.balancer.fi/reference/joins-and-exits/pool-exits.html#userdata
        self.exit_kind: int = 1  # EXACT_BPT_IN_FOR_TOKENS_OUT
        # https://docs.balancer.fi/reference/joins-and-exits/pool-joins.html#userdata
        self.join_kind: int = 1  # EXACT_TOKENS_IN_FOR_BPT_OUT

    def enter(self, **kwargs: Any) -> Generator[None, None, Optional[Tuple[str, str]]]:
        """Add liquidity in a uniswap pool."""
        pool_address = kwargs.get("pool_address")
        safe_address = kwargs.get("safe_address")
        assets = kwargs.get("assets")
        chain = kwargs.get("chain")
        max_amounts_in = kwargs.get("max_amounts_in")

        if not all([pool_address, safe_address, assets, chain, max_amounts_in]):
            self.context.logger.error(
                "Missing required parameters for entering the pool"
            )
            return None, None

        position_manager_address = (
            self.params.uniswap_position_manager_contract_addresses.get(chain, "")
        )
        if not position_manager_address:
            self.context.logger.error(
                f"No position_manager contract address found for chain {chain}"
            )
            return None, None

        # Fetch fee from uniswap v3 pool
        pool_fee = yield from self._get_pool_fee(pool_address, chain)
        if pool_fee is None:
            return None, None

        # TO-DO: specify a more concentrated position
        tick_lower, tick_upper = yield from self._calculate_tick_lower_and_upper(
            pool_address, chain
        )
        if not tick_lower or not tick_upper:
            return None, None

        # TO-DO: add slippage protection
        amount0_min = 0
        amount1_min = 0

        # deadline is set to be 20 minutes from current time
        last_update_time = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()
        deadline = int(last_update_time) + (20 * 60)

        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=position_manager_address,
            contract_public_id=UniswapV3NonfungiblePositionManagerContract.contract_id,
            contract_callable="add_liquidity",
            data_key="tx_hash",
            token0=assets[0],
            token1=assets[1],
            fee=pool_fee,
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            amount0_desired=max_amounts_in[0],
            amount1_desired=max_amounts_in[1],
            amount0_min=amount0_min,
            amount1_min=amount1_min,
            recipient=safe_address,
            deadline=deadline,
            chain_id=chain,
        )
        return tx_hash, position_manager_address

    def exit(self, **kwargs: Any) -> Generator[None, None, Optional[str]]:
        """Remove liquidity in a uniswap pool."""
        token_id = kwargs.get("token_id")
        safe_address = kwargs.get("safe_address")
        chain = kwargs.get("chain")

        if not all([token_id, safe_address, chain]):
            self.context.logger.error(
                "Missing required parameters for entering the pool"
            )
            return None

        # burn the NFT
        burn_tokens_tx_hash = yield from self.burn_position(token_id, chain)

        # collect the tokens
        collect_tokens_tx_hash = yield from self.collect_tokens(
            token_id, safe_address, chain
        )

        # prepare multisend

    def burn_position(
        self, token_id: int, chain: str
    ) -> Generator[None, None, Optional[str]]:
        """Burn position"""
        pass

    def collect_tokens(
        self, token_id: int, safe_address: str, chain: str
    ) -> Generator[None, None, Optional[str]]:
        """Collect tokens"""
        position_manager_address = (
            self.params.uniswap_position_manager_contract_addresses.get(chain, "")
        )
        if not position_manager_address:
            self.context.logger.error(
                f"No position_manager contract address found for chain {chain}"
            )
            return None

        # set amount0_max and amount1_max to int.max to collect all fees
        amount_max = INT_MAX

        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=position_manager_address,
            contract_public_id=UniswapV3NonfungiblePositionManagerContract.contract_id,
            contract_callable="collect_tokens",
            data_key="tx_hash",
            token_id=token_id,
            recipient=safe_address,
            amount0_max=amount_max,
            amount1_max=amount_max,
            chain_id=chain,
        )

        return tx_hash

    def _get_tokens(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[List[str]]]:
        """Get uniswap pool tokens"""
        pool_tokens = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=UniswapV3PoolContract.contract_id,
            contract_callable="get_pool_tokens",
            data_key="tokens",
            chain_id=chain,
        )

        if not pool_tokens:
            self.context.logger.error(
                f"Could not fetch tokens for uniswap pool {pool_address}"
            )
            return []

        self.context.logger.info(
            f"Tokens for uniswap pool {pool_address} : {pool_tokens}"
        )
        return pool_tokens

    def _get_pool_fee(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[int]]:
        """Get uniswap pool fee"""
        pool_fee = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=UniswapV3PoolContract.contract_id,
            contract_callable="get_pool_fee",
            data_key="data",
            chain_id=chain,
        )

        if pool_fee is None:
            self.context.logger.error(
                f"Could not fetch pool fee for uniswap pool {pool_address}"
            )
            return None

        self.context.logger.info(f"Fee for uniswap pool {pool_address} : {pool_fee}")
        return pool_fee

    def _get_tick_spacing(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[int]]:
        """Get uniswap pool fee"""
        tick_spacing = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=UniswapV3PoolContract.contract_id,
            contract_callable="get_tick_spacing",
            data_key="data",
            chain_id=chain,
        )

        if tick_spacing is None:
            self.context.logger.error(
                f"Could not fetch tick spacing for uniswap pool {pool_address}"
            )
            return None

        self.context.logger.info(
            f"Tick spacing for uniswap pool {pool_address} : {tick_spacing}"
        )
        return tick_spacing

    def _calculate_tick_lower_and_upper(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[Tuple[int, int]]]:
        self.context.logger.info(f"inside function {pool_address} {chain}")
        # Fetch tick spacing from uniswap v3 pool
        tick_spacing = yield from self._get_tick_spacing(pool_address, chain)
        if tick_spacing is None:
            return None, None
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
            f"TICK LOWER: {adjusted_tick_lower} TICK UPPER: {adjusted_tick_upper}"
        )
        return adjusted_tick_lower, adjusted_tick_upper

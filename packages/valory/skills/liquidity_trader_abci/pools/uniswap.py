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
from typing import Any, Dict, Generator, Optional, Tuple, cast

from web3 import Web3

from packages.valory.contracts.multisend.contract import (
    MultiSendContract,
    MultiSendOperation,
)
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
    """Mint parameters for uniswap v3"""

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
        """Initialize mint parameters"""
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

    def enter(self, **kwargs: Any) -> Generator[None, None, Optional[Tuple[str, str]]]:
        """Add liquidity in a uniswap pool."""
        pool_address = kwargs.get("pool_address")
        safe_address = kwargs.get("safe_address")
        assets = kwargs.get("assets")
        chain = kwargs.get("chain")
        max_amounts_in = kwargs.get("max_amounts_in")
        pool_fee = kwargs.get("pool_fee")

        if not all([pool_address, safe_address, assets, chain, max_amounts_in]):
            self.context.logger.error(
                f"Missing required parameters for entering the pool. Here are the kwargs: {kwargs}"
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

        # Fetch fee from uniswap v3 pool if pool fee is not provided
        if not pool_fee:
            pool_fee = yield from self._get_pool_fee(pool_address, chain)
            if not pool_fee:
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
            contract_callable="mint",
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

    def exit(
        self, **kwargs: Any
    ) -> Generator[None, None, Optional[Tuple[str, str, bool]]]:
        """Remove liquidity in a uniswap pool."""
        token_id = kwargs.get("token_id")
        safe_address = kwargs.get("safe_address")
        chain = kwargs.get("chain")
        liquidity = kwargs.get("liquidity")

        if not all([token_id, safe_address, chain]):
            self.context.logger.error(
                f"Missing required parameters for exiting the pool. Here are the kwargs: {kwargs}"
            )
            return None, None, None

        position_manager_address = (
            self.params.uniswap_position_manager_contract_addresses.get(chain, "")
        )
        if not position_manager_address:
            self.context.logger.error(
                f"No position_manager contract address found for chain {chain}"
            )
            return None, None, None

        multi_send_txs = []

        # decrease liquidity
        # TO-DO: Calculate min amounts accouting for slippage
        amount0_min = 0
        amount1_min = 0

        # fetch liquidity from contract
        if not liquidity:
            liquidity = yield from self.get_liquidity_for_token(token_id, chain)
            if not liquidity:
                return None, None, None

        # deadline is set to be 20 minutes from current time
        last_update_time = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()
        deadline = int(last_update_time) + (20 * 60)

        decrease_liquidity_tx_hash = yield from self.decrease_liquidity(
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
        # Note: We initially set `amount_max` to the maximum value of uint256 since the collect function sends the lesser of `amount_max` or `tokensOwed`.
        # However, that value was too large, so we adjusted it to 2**100 - 1 wei.
        amount_max = Web3.to_wei(2**100 - 1, "wei")
        collect_tokens_tx_hash = yield from self.collect_tokens(
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
        multisend_address = self.params.multisend_contract_addresses[chain]
        if not multisend_address:
            self.context.logger.error(
                f"Could not find multisend address for chain {chain}"
            )
            return None, None, None

        multisend_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
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

    def burn_token(
        self, token_id: int, chain: str
    ) -> Generator[None, None, Optional[str]]:
        """Burn position"""
        position_manager_address = (
            self.params.uniswap_position_manager_contract_addresses.get(chain, "")
        )
        if not position_manager_address:
            self.context.logger.error(
                f"No position_manager contract address found for chain {chain}"
            )
            return None

        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=position_manager_address,
            contract_public_id=UniswapV3NonfungiblePositionManagerContract.contract_id,
            contract_callable="burn_token",
            data_key="tx_hash",
            token_id=token_id,
            chain_id=chain,
        )

        return tx_hash

    def collect_tokens(
        self,
        token_id: int,
        recipient: str,
        amount0_max: int,
        amount1_max: int,
        chain: str,
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

        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=position_manager_address,
            contract_public_id=UniswapV3NonfungiblePositionManagerContract.contract_id,
            contract_callable="collect_tokens",
            data_key="tx_hash",
            token_id=token_id,
            recipient=recipient,
            amount0_max=amount0_max,
            amount1_max=amount1_max,
            chain_id=chain,
        )

        return tx_hash

    def decrease_liquidity(
        self,
        token_id: int,
        liquidity: int,
        amount0_min: int,
        amount1_min: int,
        deadline: int,
        chain: str,
    ) -> Generator[None, None, Optional[str]]:
        """Decrease liquidity"""
        position_manager_address = (
            self.params.uniswap_position_manager_contract_addresses.get(chain, "")
        )
        if not position_manager_address:
            self.context.logger.error(
                f"No position_manager contract address found for chain {chain}"
            )
            return None

        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=position_manager_address,
            contract_public_id=UniswapV3NonfungiblePositionManagerContract.contract_id,
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

    def get_liquidity_for_token(
        self, token_id: int, chain: str
    ) -> Generator[None, None, Optional[str]]:
        """Get liquidity for token"""
        position_manager_address = (
            self.params.uniswap_position_manager_contract_addresses.get(chain, "")
        )
        if not position_manager_address:
            self.context.logger.error(
                f"No position_manager contract address found for chain {chain}"
            )
            return None

        position = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=position_manager_address,
            contract_public_id=UniswapV3NonfungiblePositionManagerContract.contract_id,
            contract_callable="get_position",
            data_key="data",
            token_id=token_id,
            chain_id=chain,
        )

        if not position:
            return None

        # liquidity is returned at the 7th index from contract
        liquidity = position[7]
        return liquidity

    def _get_tokens(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get uniswap pool tokens"""
        """Return a dict of tokens {"token0": 0x00, "token1": 0x00}"""

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
            return None

        tokens = {"token0": pool_tokens[0], "token1": pool_tokens[1]}
        self.context.logger.info(f"Tokens for uniswap pool {pool_address} : {tokens}")
        return tokens

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

        if not pool_fee:
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

        if not tick_spacing:
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
        if not tick_spacing:
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

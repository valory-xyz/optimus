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

from abc import ABC
from typing import Any, Dict, Generator, List, Optional, Tuple
from enum import Enum

from packages.valory.contracts.balancer_vault.contract import VaultContract
from packages.valory.contracts.balancer_weighted_pool.contract import (
    WeightedPoolContract,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.liquidity_trader_abci.pool_behaviour import PoolBehaviour


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

class PoolType(Enum):
    """PoolType"""

    WEIGHTED = "Weighted"
    COMPOSABLE_STABLE = "ComposableStable"
    LIQUIDITY_BOOTSTRAPING = "LiquidityBootstrapping"
    META_STABLE = "MetaStable"
    STABLE = "Stable"
    INVESTMENT = "Investment"

class JoinKind:
    """JoinKind Enums for different pool types."""
    # https://docs.balancer.fi/reference/joins-and-exits/pool-joins.html#userdata
    class WeightedPool(Enum):
        INIT = 0
        EXACT_TOKENS_IN_FOR_BPT_OUT = 1
        TOKEN_IN_FOR_EXACT_BPT_OUT = 2
        ALL_TOKENS_IN_FOR_EXACT_BPT_OUT = 3

    class StableAndMetaStablePool(Enum):
        INIT = 0
        EXACT_TOKENS_IN_FOR_BPT_OUT = 1
        TOKEN_IN_FOR_EXACT_BPT_OUT = 2

    class ComposableStablePool(Enum):
        INIT = 0
        EXACT_TOKENS_IN_FOR_BPT_OUT = 1
        TOKEN_IN_FOR_EXACT_BPT_OUT = 2
        ALL_TOKENS_IN_FOR_EXACT_BPT_OUT = 3

class ExitKind:
    """ExitKind Enums for different pool types."""
    # https://docs.balancer.fi/reference/joins-and-exits/pool-exits.html#userdata
    class WeightedPool(Enum):
        EXACT_BPT_IN_FOR_ONE_TOKEN_OUT = 0
        EXACT_BPT_IN_FOR_TOKENS_OUT = 1
        BPT_IN_FOR_EXACT_TOKENS_OUT = 2
        MANAGEMENT_FEE_TOKENS_OUT = 3  # for InvestmentPool only

    class StableAndMetaStablePool(Enum):
        EXACT_BPT_IN_FOR_ONE_TOKEN_OUT = 0
        EXACT_BPT_IN_FOR_TOKENS_OUT = 1
        BPT_IN_FOR_EXACT_TOKENS_OUT = 2

    class ComposableStablePool(Enum):
        EXACT_BPT_IN_FOR_ONE_TOKEN_OUT = 0
        BPT_IN_FOR_EXACT_TOKENS_OUT = 1
        EXACT_BPT_IN_FOR_ALL_TOKENS_OUT = 2


class BalancerPoolBehaviour(PoolBehaviour, ABC):
    """BalancerPoolBehaviour"""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the balancer pool behaviour."""
        super().__init__(**kwargs)

    def enter(self, **kwargs: Any) -> Generator[None, None, Optional[Tuple[str, str]]]:
        """Enter a Balancer pool."""
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

        join_kind = self._determine_join_kind(pool_type)
        if not join_kind:
            self.context.logger.error(f"Could not determine join kind for pool type {pool_type}")
            return None, None
        # Get vault contract address from balancer weighted pool contract
        vault_address = self.params.balancer_vault_contract_addresses.get(chain)
        if not vault_address:
            self.context.logger.error(f"No vault address found for chain {chain}")
            return None, None

        # Fetch the pool id
        pool_id = yield from self._get_pool_id(pool_address, chain)  # getPoolId()
        if not pool_id:
            return None, None

        # TO-DO: calculate minimum_bpt
        minimum_bpt = 0

        # fromInternalBalance - True if sending from internal token balances. False if sending ERC20.
        from_internal_balance = ZERO_ADDRESS in assets

        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=vault_address,
            contract_public_id=VaultContract.contract_id,
            contract_callable="join_pool",
            data_key="tx_hash",
            pool_id=pool_id,
            sender=safe_address,
            recipient=safe_address,
            assets=assets,
            max_amounts_in=max_amounts_in,
            join_kind=join_kind,
            minimum_bpt=minimum_bpt,
            from_internal_balance=from_internal_balance,
            chain_id=chain,
        )

        return tx_hash, vault_address

    def exit(
        self, **kwargs: Any
    ) -> Generator[None, None, Optional[Tuple[str, str, bool]]]:
        """Exit pool"""
        pool_address = kwargs.get("pool_address")
        safe_address = kwargs.get("safe_address")
        assets = kwargs.get("assets")
        chain = kwargs.get("chain")
        pool_type = kwargs.get("pool_type")
        if not all([pool_address, safe_address, assets, chain, pool_type]):
            self.context.logger.error(
                f"Missing required parameters for exiting the pool. Here are the kwargs: {kwargs}"
            )
            return None, None, None

        exit_kind = self._determine_exit_kind(pool_type)
        if not exit_kind:
            self.context.logger.error(f"Could not determine exit kind for pool type {pool_type}")
            return None, None, None

        # Get vault contract address from balancer weighted pool contract
        vault_address = self.params.balancer_vault_contract_addresses.get(chain)
        if not vault_address:
            self.context.logger.error(f"No vault address found for chain {chain}")
            return None, None, None

        # Fetch the pool id
        pool_id = yield from self._get_pool_id(pool_address, chain)  # getPoolId()
        if not pool_id:
            return None, None, None

        # TO-DO: queryExit in BalancerQueries to find the current amounts of tokens we would get for our BPT, and then account for some possible slippage.
        min_amounts_out = [0, 0]

        # fetch the amount of LP tokens
        bpt_amount_in = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=str(WeightedPoolContract.contract_id),
            contract_callable="get_balance",
            data_key="balance",
            account=safe_address,
            chain_id=chain,
        )
        if bpt_amount_in is None:
            self.context.logger.error(
                f"Error fetching BPT Amount for safe {safe_address} for pool {pool_address}"
            )
            return None, None, None

        # toInternalBalance - True if receiving internal token balances. False if receiving ERC20.
        to_internal_balance = ZERO_ADDRESS in assets

        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=vault_address,
            contract_public_id=VaultContract.contract_id,
            contract_callable="exit_pool",
            data_key="tx_hash",
            pool_id=pool_id,
            sender=safe_address,
            recipient=safe_address,
            assets=assets,
            min_amounts_out=min_amounts_out,
            exit_kind=exit_kind,
            bpt_amount_in=bpt_amount_in,
            to_internal_balance=to_internal_balance,
            chain_id=chain,
        )

        return tx_hash, vault_address, False

    def _get_tokens(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get balancer pool tokens"""
        """Return a dict of tokens {"token0": 0x00, "token1": 0x00}"""

        # Get vault contract address from balancer weighted pool contract
        vault_address = self.params.balancer_vault_contract_addresses.get(chain)
        if not vault_address:
            self.context.logger.error(f"No vault address found for chain {chain}")
            return None

        # Fetch the pool id
        pool_id = yield from self._get_pool_id(pool_address, chain)  # getPoolId()
        if not pool_id:
            return None

        pool_tokens = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=vault_address,
            contract_public_id=VaultContract.contract_id,
            contract_callable="get_pool_tokens",
            data_key="tokens",
            pool_id=pool_id,
            chain_id=chain,
        )

        if not pool_tokens:
            self.context.logger.error(
                f"Could not fetch tokens for balancer pool id {pool_id}"
            )
            return None

        tokens = {"token0": pool_tokens[0][0], "token1": pool_tokens[0][1]}
        self.context.logger.info(
            f"Tokens for balancer poolId {pool_id} : {pool_tokens}"
        )
        return tokens

    def _get_pool_id(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[str]]:
        """Get pool id"""

        pool_id = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=WeightedPoolContract.contract_id,
            contract_callable="get_pool_id",
            data_key="pool_id",
            chain_id=chain,
        )

        if not pool_id:
            self.context.logger.error(
                f"Could not fetch the Pool ID for pool {pool_address}"
            )
            return None

        self.context.logger.info(f"PoolId for balancer pool {pool_address}: {pool_id}")
        return pool_id

    def _determine_join_kind(self, pool_type: PoolType) -> Optional[int]:
        """Determine the join kind based on the pool type."""
        if pool_type in [PoolType.WEIGHTED.value, PoolType.LIQUIDITY_BOOTSTRAPING.value, PoolType.INVESTMENT.value]:
            return JoinKind.WeightedPool.EXACT_TOKENS_IN_FOR_BPT_OUT.value
        elif pool_type in [PoolType.STABLE.value, PoolType.META_STABLE.value]:
            return JoinKind.StableAndMetaStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT.value
        elif pool_type == PoolType.COMPOSABLE_STABLE.value:
            return JoinKind.ComposableStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT.value
        else:
            self.context.logger.error(f"Unknown pool type: {pool_type}")
            return None

    def _determine_exit_kind(self, pool_type: PoolType) -> Optional[int]:
        """Determine the exit kind based on the pool type."""
        if pool_type in [PoolType.WEIGHTED.value, PoolType.LIQUIDITY_BOOTSTRAPING.value, PoolType.INVESTMENT.value]:
            return ExitKind.WeightedPool.EXACT_BPT_IN_FOR_TOKENS_OUT.value
        elif pool_type in [PoolType.STABLE.value, PoolType.META_STABLE.value]:
            return ExitKind.StableAndMetaStablePool.EXACT_BPT_IN_FOR_TOKENS_OUT.value
        elif pool_type == PoolType.COMPOSABLE_STABLE.value:
            return ExitKind.ComposableStablePool.EXACT_BPT_IN_FOR_ALL_TOKENS_OUT.value
        else:
            self.context.logger.error(f"Unknown pool type: {pool_type}")
            return None



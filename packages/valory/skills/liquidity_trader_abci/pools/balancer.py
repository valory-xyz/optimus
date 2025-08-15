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
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple

from eth_utils import to_checksum_address

from packages.valory.contracts.balancer_queries.contract import BalancerQueriesContract
from packages.valory.contracts.balancer_vault.contract import VaultContract
from packages.valory.contracts.balancer_weighted_pool.contract import (
    WeightedPoolContract,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.liquidity_trader_abci.pool_behaviour import PoolBehaviour
from packages.valory.skills.liquidity_trader_abci.utils.balancer_math import (
    BalancerProportionalMath,
)


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
        """Weighted pool join kind"""

        INIT = 0
        EXACT_TOKENS_IN_FOR_BPT_OUT = 1
        TOKEN_IN_FOR_EXACT_BPT_OUT = 2
        ALL_TOKENS_IN_FOR_EXACT_BPT_OUT = 3

    class StableAndMetaStablePool(Enum):
        """Stable and meta stable pool join kind"""

        INIT = 0
        EXACT_TOKENS_IN_FOR_BPT_OUT = 1
        TOKEN_IN_FOR_EXACT_BPT_OUT = 2

    class ComposableStablePool(Enum):
        """Composable stable pool join kind"""

        INIT = 0
        EXACT_TOKENS_IN_FOR_BPT_OUT = 1
        TOKEN_IN_FOR_EXACT_BPT_OUT = 2
        ALL_TOKENS_IN_FOR_EXACT_BPT_OUT = 3


class ExitKind:
    """ExitKind Enums for different pool types."""

    # https://docs.balancer.fi/reference/joins-and-exits/pool-exits.html#userdata
    class WeightedPool(Enum):
        """Weighted pool exit kind"""

        EXACT_BPT_IN_FOR_ONE_TOKEN_OUT = 0
        EXACT_BPT_IN_FOR_TOKENS_OUT = 1
        BPT_IN_FOR_EXACT_TOKENS_OUT = 2
        MANAGEMENT_FEE_TOKENS_OUT = 3  # for InvestmentPool only

    class StableAndMetaStablePool(Enum):
        """Stable and meta stable pool exit kind"""

        EXACT_BPT_IN_FOR_ONE_TOKEN_OUT = 0
        EXACT_BPT_IN_FOR_TOKENS_OUT = 1
        BPT_IN_FOR_EXACT_TOKENS_OUT = 2

    class ComposableStablePool(Enum):
        """Composable stable pool exit kind"""

        EXACT_BPT_IN_FOR_ONE_TOKEN_OUT = 0
        BPT_IN_FOR_EXACT_TOKENS_OUT = 1
        EXACT_BPT_IN_FOR_ALL_TOKENS_OUT = 2


class BalancerPoolBehaviour(PoolBehaviour, ABC):
    """BalancerPoolBehaviour"""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the balancer pool behaviour."""
        super().__init__(**kwargs)

    def update_value(
        self, **kwargs: Any
    ) -> Generator[None, None, Tuple[Optional[list], Optional[list]]]:
        """Fetch and flatten pool token addresses."""

        pool_id = kwargs.get("pool_id")
        chain = kwargs.get("chain")
        vault_address = kwargs.get("vault_address")
        max_amounts_in = kwargs.get("max_amounts_in")
        self.context.logger.info(f"Into update value max_amounts_in:{max_amounts_in}")
        assets = kwargs.get("assets")
        if not pool_id or not chain:
            self.context.logger.error(
                "Missing required parameters: 'pool_id' or 'chain'"
            )
            return None, None
        try:
            pool_info = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=vault_address,
                contract_public_id=VaultContract.contract_id,
                contract_callable="get_pool_tokens",
                pool_id=pool_id,
                data_key="tokens",
                chain_id=chain,
            )
            if not pool_info or not isinstance(pool_info, list) or not pool_info[0]:
                self.context.logger.error(
                    "Invalid pool_info data received from contract interaction."
                )
                return None, None
            # Safely extract and flatten token addresses
            tokens_nested = pool_info[0]
            new_max_amounts_in = self.adjust_amounts(
                assets, tokens_nested, max_amounts_in
            )

            return tokens_nested, new_max_amounts_in
        except Exception as e:
            self.context.logger.error(f"Error fetching pool tokens: {str(e)}")
            return None, None

    def adjust_amounts(
        self, assets: List[str], assets_new: List[str], max_amounts_in: List[int]
    ) -> List[int]:
        """Adjust token amounts to match the order of tokens in a Balancer pool."""
        # Log the inputs for debugging
        self.context.logger.info(
            f"assets, assets_new, max_amounts_in:{assets, assets_new, max_amounts_in}"
        )

        # Input validation
        if not all(isinstance(x, (str, bytes)) for x in assets):
            raise ValueError("All assets must be strings or bytes")
        if len(assets) != len(max_amounts_in):
            raise ValueError("Length of assets and max_amounts_in must match")

        # Normalize all addresses to lowercase for consistent comparison
        normalized_assets = [asset.lower() for asset in assets]
        normalized_assets_new = [asset.lower() for asset in assets_new]

        # Create a mapping from normalized asset to amount
        asset_to_amount = dict(zip(normalized_assets, max_amounts_in))

        # Map amounts for new assets based on the normalized keys
        new_max_amounts_in = [
            asset_to_amount.get(asset, 0) for asset in normalized_assets_new
        ]

        # Log the outcome for debugging
        self.context.logger.info(
            f"Into adjust amounts new_max_amounts_in: {new_max_amounts_in}"
        )

        return new_max_amounts_in

    def enter(self, **kwargs: Any) -> Generator[None, None, Optional[Tuple[str, str]]]:
        """Enter a Balancer pool."""
        pool_address = kwargs.get("pool_address")
        safe_address = kwargs.get("safe_address")
        assets = kwargs.get("assets")
        chain = kwargs.get("chain")
        max_amounts_in = kwargs.get("max_amounts_in")
        pool_type = kwargs.get("pool_type")
        if not all(
            [pool_address, safe_address, assets, chain, max_amounts_in, pool_type]
        ):
            self.context.logger.error(
                f"Missing required parameters for entering the pool. Here are the kwargs: {kwargs}"
            )
            return None, None

        self.context.logger.info("enter into the pool")

        join_kind = self._determine_join_kind(pool_type)
        self.context.logger.info(f"Determined join kind: {join_kind}")

        if not join_kind:
            self.context.logger.error(
                f"Could not determine join kind for pool type {pool_type}"
            )
            return None, None
        # Get vault contract address from balancer weighted pool contract
        vault_address = self.params.balancer_vault_contract_addresses.get(chain)
        self.context.logger.info(f"Vault address retrieved: {vault_address}")

        if not vault_address:
            self.context.logger.error(f"No vault address found for chain {chain}")
            return None, None

        # Fetch the pool id
        pool_id = yield from self._get_pool_id(pool_address, chain)  # getPoolId()
        self.context.logger.info(f"Pool ID retrieved: {pool_id}")
        if not pool_id:
            return None, None

        self.context.logger.info(f"assets, max_amounts_in:{assets, max_amounts_in}")

        new_assets, new_max_amounts_in = yield from self.update_value(
            assets=assets,
            max_amounts_in=max_amounts_in,
            vault_address=vault_address,
            pool_id=pool_id,
            chain=chain,
        )
        # fromInternalBalance - True if sending from internal token balances. False if sending ERC20.
        from_internal_balance = ZERO_ADDRESS in assets
        self.context.logger.info(
            f"new_assets, new_max_amounts_in:{new_assets, new_max_amounts_in}"
        )

        # Query expected BPT output and apply slippage protection
        expected_bpt_out = yield from self.query_proportional_join(
            pool_id=pool_id,
            pool_address=pool_address,
            vault_address=vault_address,
            chain=chain,
            amounts_in=max_amounts_in,
        )
        if expected_bpt_out:
            slippage_tolerance = self.params.slippage_tolerance
            minimum_bpt = int(expected_bpt_out * (1 - slippage_tolerance))
            self.context.logger.info(
                f"Expected BPT: {expected_bpt_out}, Minimum BPT with slippage: {minimum_bpt}"
            )
        else:
            self.context.logger.warning(
                "Unable to estimate minimum BPT, discarding transaction"
            )
            return None, None

        self.context.logger.info("Preparing transaction for pool join.")

        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=vault_address,
            contract_public_id=VaultContract.contract_id,
            contract_callable="join_pool",
            data_key="tx_hash",
            pool_id=pool_id,
            sender=safe_address,
            recipient=safe_address,
            assets=new_assets,
            max_amounts_in=new_max_amounts_in,
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
            self.context.logger.error(
                f"Could not determine exit kind for pool type {pool_type}"
            )
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

        # Query expected amounts out and apply slippage protection
        expected_amounts = yield from self.query_proportional_exit(
            pool_id=pool_id,
            pool_address=pool_address,
            vault_address=vault_address,
            chain=chain,
            bpt_amount_in=bpt_amount_in,
        )

        if expected_amounts:
            slippage_tolerance = self.params.slippage_tolerance
            min_amounts_out = [
                int(amt * (1 - slippage_tolerance)) for amt in expected_amounts
            ]
            self.context.logger.info(
                f"Expected amounts: {expected_amounts}, Min amounts with slippage: {min_amounts_out}"
            )
        else:
            self.context.logger.warning(
                "Unable to estimate minimum amounts out, discarding transaction"
            )
            return None, None, None

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
        if pool_type in [
            PoolType.WEIGHTED.value,
            PoolType.LIQUIDITY_BOOTSTRAPING.value,
            PoolType.INVESTMENT.value,
        ]:
            return JoinKind.WeightedPool.EXACT_TOKENS_IN_FOR_BPT_OUT.value
        elif pool_type in [PoolType.STABLE.value, PoolType.META_STABLE.value]:
            return JoinKind.StableAndMetaStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT.value
        elif pool_type == PoolType.COMPOSABLE_STABLE.value:
            return JoinKind.ComposableStablePool.EXACT_TOKENS_IN_FOR_BPT_OUT.value
        else:
            self.context.logger.error(f"Unknown pool type: {pool_type}")
            return None

    def _query_join_bpt(
        self,
        pool_id: str,
        sender: str,
        assets: List[str],
        max_amounts_in: List[int],
        join_kind: int,
        from_internal_balance: bool,
        chain: str,
    ) -> Generator[None, None, Optional[int]]:
        """Query expected BPT output for join operation."""
        queries_address = self.params.balancer_queries_contract_addresses.get(chain)
        if not queries_address:
            self.context.logger.warning(
                f"No BalancerQueries address found for chain {chain}"
            )
            return None

        try:
            # Ensure all asset addresses are checksummed
            checksummed_assets = [to_checksum_address(asset) for asset in assets]

            result = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=queries_address,
                contract_public_id=BalancerQueriesContract.contract_id,
                contract_callable="query_join",
                data_key="result",
                pool_id=pool_id,
                sender=sender,
                recipient=sender,
                assets=checksummed_assets,
                max_amounts_in=max_amounts_in,
                join_kind=join_kind,
                minimum_bpt=0,
                from_internal_balance=from_internal_balance,
                chain_id=chain,
            )
            self.context.logger.info(f"Result from query_join: {result}")
            return result.get("bpt_out")

        except Exception as e:
            self.context.logger.error(f"Error querying join BPT: {str(e)}")
            return None

    def _query_exit_amounts(
        self,
        pool_id: str,
        sender: str,
        assets: List[str],
        bpt_amount_in: int,
        exit_kind: int,
        to_internal_balance: bool,
        chain: str,
    ) -> Generator[None, None, Optional[List[int]]]:
        """Query expected token amounts for exit operation."""
        queries_address = self.params.balancer_queries_contract_addresses.get(chain)
        if not queries_address:
            self.context.logger.warning(
                f"No BalancerQueries address found for chain {chain}"
            )
            return None

        try:
            # Ensure all asset addresses are checksummed
            checksummed_assets = [to_checksum_address(asset) for asset in assets]

            result = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=queries_address,
                contract_public_id=BalancerQueriesContract.contract_id,
                contract_callable="query_exit",
                data_key="result",
                pool_id=pool_id,
                sender=sender,
                recipient=sender,
                assets=checksummed_assets,
                min_amounts_out=[0] * len(checksummed_assets),
                exit_kind=exit_kind,
                bpt_amount_in=bpt_amount_in,
                to_internal_balance=to_internal_balance,
                chain_id=chain,
            )

            if result and "amounts_out" in result:
                return result["amounts_out"]
            else:
                self.context.logger.warning("Invalid result from queryExit")
                return None

        except Exception as e:
            self.context.logger.error(f"Error querying exit amounts: {str(e)}")
            return None

    def _determine_exit_kind(self, pool_type: PoolType) -> Optional[int]:
        """Determine the exit kind based on the pool type."""
        if pool_type in [
            PoolType.WEIGHTED.value,
            PoolType.LIQUIDITY_BOOTSTRAPING.value,
            PoolType.INVESTMENT.value,
        ]:
            return ExitKind.WeightedPool.EXACT_BPT_IN_FOR_TOKENS_OUT.value
        elif pool_type in [PoolType.STABLE.value, PoolType.META_STABLE.value]:
            return ExitKind.StableAndMetaStablePool.EXACT_BPT_IN_FOR_TOKENS_OUT.value
        elif pool_type == PoolType.COMPOSABLE_STABLE.value:
            return ExitKind.ComposableStablePool.EXACT_BPT_IN_FOR_ALL_TOKENS_OUT.value
        else:
            self.context.logger.error(f"Unknown pool type: {pool_type}")
            return None

    def query_proportional_exit(
        self,
        pool_id: str,
        pool_address: str,
        vault_address: str,
        chain: str,
        bpt_amount_in: int,
    ) -> Generator[None, None, Optional[List[int]]]:
        """Simulate proportional exit for any pool type"""
        try:
            # Fetch pool token information
            pool_tokens_info = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=vault_address,
                contract_public_id=VaultContract.contract_id,
                contract_callable="get_pool_tokens",
                pool_id=pool_id,
                data_key="tokens",
                chain_id=chain,
            )

            if not pool_tokens_info or len(pool_tokens_info) < 2:
                self.context.logger.error("Failed to fetch pool token information")
                return None

            token_balances = pool_tokens_info[1]  # Token balances in the pool

            # Fetch total BPT supply
            total_bpt_supply = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=pool_address,
                contract_public_id=WeightedPoolContract.contract_id,
                contract_callable="get_total_supply",
                data_key="data",
                chain_id=chain,
            )

            if not total_bpt_supply:
                self.context.logger.error("Failed to fetch total BPT supply")
                return None

            # Simulate proportional exit
            amounts_out = BalancerProportionalMath.query_proportional_exit(
                token_balances=token_balances,
                total_bpt_supply=total_bpt_supply,
                bpt_amount_in=bpt_amount_in,
            )

            self.context.logger.info(
                f"Proportional exit simulation - BPT in: {bpt_amount_in}, "
                f"Amounts out: {amounts_out}"
            )

            return amounts_out

        except Exception as e:
            self.context.logger.error(
                f"Error in proportional exit simulation: {str(e)}"
            )
            return None

    def query_proportional_join(
        self,
        pool_id: str,
        pool_address: str,
        vault_address: str,
        chain: str,
        amounts_in: List[int],
    ) -> Generator[None, None, Optional[int]]:
        """Simulate proportional join for any pool type"""
        try:
            # Fetch pool token information
            pool_tokens_info = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=vault_address,
                contract_public_id=VaultContract.contract_id,
                contract_callable="get_pool_tokens",
                pool_id=pool_id,
                data_key="tokens",
                chain_id=chain,
            )

            if not pool_tokens_info or len(pool_tokens_info) < 2:
                self.context.logger.error("Failed to fetch pool token information")
                return None

            token_balances = pool_tokens_info[1]  # Token balances in the pool

            # Fetch total BPT supply
            total_bpt_supply = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=pool_address,
                contract_public_id=WeightedPoolContract.contract_id,
                contract_callable="get_total_supply",
                data_key="data",
                chain_id=chain,
            )

            if not total_bpt_supply:
                self.context.logger.error("Failed to fetch total BPT supply")
                return None

            # Simulate proportional join
            bpt_out = BalancerProportionalMath.query_proportional_join(
                token_balances=token_balances,
                total_bpt_supply=total_bpt_supply,
                amounts_in=amounts_in,
            )

            self.context.logger.info(
                f"Proportional join simulation - BPT out: {bpt_out}, "
                f"Input amounts: {amounts_in}"
            )

            return bpt_out

        except Exception as e:
            self.context.logger.error(
                f"Error in proportional join simulation: {str(e)}"
            )
            return None

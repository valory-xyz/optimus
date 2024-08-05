from abc import ABC
from typing import Any, Dict, Generator, List, Optional

from packages.valory.contracts.balancer_vault.contract import VaultContract
from packages.valory.contracts.balancer_weighted_pool.contract import (
    WeightedPoolContract,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.liquidity_trader_abci.pool_behaviour import PoolBehaviour


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


class BalancerPoolBehaviour(PoolBehaviour, ABC):
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the balancer pool behaviour."""
        super().__init__(**kwargs)
        self.vault_address: str = ""
        self.exit_kind: int = 1
        self.join_kind: int = 1
        self.max_amounts_in: List[int] = []
        self.min_amounts_in: List[int] = []
        self.pool_id: str = ""

    def enter(self, **kwargs: Any) -> Generator[None, None, Optional[str]]:
        """Enter a Balancer pool."""

        pool_address = kwargs.get("pool_address")
        safe_address = kwargs.get("safe_address")
        assets = kwargs.get("assets")
        chain = kwargs.get("chain")
        positions = kwargs.get("positions")

        if not all([pool_address, safe_address, assets, chain, positions]):
            self.context.logger.error(
                "Missing required parameters for entering the pool"
            )
            return None

        # fromInternalBalance - True if sending from internal token balances. False if sending ERC20.
        from_internal_balance = ZERO_ADDRESS in assets

        # get pool id
        self.pool_id = yield from self._get_pool_id(pool_address, chain)  # getPoolId()
        if self.pool_id is None:
            return None

        # Get vault contract address from balancer weighted pool contract
        self.vault_address = yield from self._get_vault_for_pool(pool_address, chain)
        if not self.vault_address:
            return None, None, None

        self.max_amounts_in = [
            self._get_balance(chain, assets[0], positions),
            self._get_balance(chain, assets[1], positions),
        ]
        if any(amount == 0 for amount in self.max_amounts_in):
            self.context.logger.error("Insufficient balance for entering pool")
            return None, None, None

        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=self.vault_address,
            contract_public_id=VaultContract.contract_id,
            contract_callable="join_pool",
            data_key="tx_hash",
            pool_id=self.pool_id,
            sender=safe_address,
            recipient=safe_address,
            assets=assets,
            max_amounts_in=self.max_amounts_in,
            join_kind=self.join_kind,
            from_internal_balance=from_internal_balance,
            chain_id=chain,
        )

        return tx_hash

    def exit(self, **kwargs: Any) -> Generator[None, None, Optional[str]]:
        """Exit a pool with dynamic parameters."""
        # Extract parameters from kwargs
        vault_address = kwargs.get("vault_address")
        pool_id = kwargs.get("pool_id")
        safe_address = kwargs.get("safe_address")
        assets = kwargs.get("assets")
        min_amounts_out = kwargs.get("min_amounts_out")
        exit_kind = kwargs.get("exit_kind")
        bpt_amount_in = kwargs.get("bpt_amount_in")
        chain = kwargs.get("chain")

        if not all(
            [
                vault_address,
                pool_id,
                safe_address,
                assets,
                min_amounts_out,
                exit_kind,
                bpt_amount_in,
                chain,
            ]
        ):
            self.context.logger.error(
                "Missing required parameters for exiting the pool"
            )
            return None

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
            min_amounts_out=self.min_amounts_out,
            exit_kind=self.exit_kind,
            bpt_amount_in=bpt_amount_in,
            to_internal_balance=to_internal_balance,
            chain_id=chain,
        )

        return tx_hash

    def _get_tokens(
        self, pool_id: str, vault_address: str, chain: str
    ) -> Generator[None, None, Optional[List[str]]]:
        """Get balancer pool tokens"""
        pool_tokens = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=vault_address,
            contract_public_id=VaultContract.contract_id,
            contract_callable="get_pool_tokens",
            data_key="tokens",
            pool_id=pool_id,
            chain_id=chain if chain != "base" else "bnb",
        )

        if not pool_tokens:
            self.context.logger.error(
                f"Could not fetch tokens for balancer pool id {pool_id}"
            )
            return None

        self.context.logger.info(
            f"Tokens for balancer poolId {pool_id} : {pool_tokens}"
        )
        return pool_tokens

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
            chain_id=chain if chain != "base" else "bnb",
        )

        if not pool_id:
            self.context.logger.error(
                f"Could not fetch the pool id for pool {pool_address}"
            )
            return None

        self.context.logger.info(f"PoolId for balancer pool {pool_address}: {pool_id}")
        return pool_id

    def _get_vault_for_pool(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[str]]:
        """Get vault for pool"""
        vault_address = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=WeightedPoolContract.contract_id,
            contract_callable="get_vault_address",
            data_key="vault",
            chain_id=chain if chain != "base" else "bnb",
        )

        if not vault_address:
            self.context.logger.error(
                f"Could not fetch the vault address for pool {pool_address}"
            )
            return None

        self.context.logger.info(
            f"Vault contract address for balancer pool {pool_address}: {vault_address}"
        )
        return vault_address

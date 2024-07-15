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

"""This package contains round behaviours of LiquidityTraderAbciApp."""

import json
from abc import ABC
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Type, cast

from aea.configurations.data_types import PublicId

from packages.valory.contracts.balancer_vault.contract import VaultContract
from packages.valory.contracts.balancer_weighted_pool.contract import (
    WeightedPoolContract,
)
from packages.valory.contracts.erc20.contract import ERC20
from packages.valory.contracts.gnosis_safe.contract import GnosisSafeContract
from packages.valory.contracts.velodrome_pool.contract import PoolContract
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.models import Params
from packages.valory.skills.liquidity_trader_abci.rounds import (
    DecisionMakingPayload,
    DecisionMakingRound,
    EvaluateStrategyPayload,
    EvaluateStrategyRound,
    Event,
    GetPositionsPayload,
    GetPositionsRound,
    LiquidityTraderAbciApp,
    SynchronizedData,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
SAFE_TX_GAS = 0
ETHER_VALUE = 0

WaitableConditionType = Generator[None, None, Any]


class Action(Enum):
    """Action"""

    # Kept the values as Round name, so that in DecisionMaking we can match tx_submitter with action name(which is round name) and decide the next action
    EXIT_POOL = "PrepareExitPoolTxRound"
    ENTER_POOL = "PrepareEnterPoolTxRound"
    BRIDGE_SWAP = "PrepareSwapTxRound"


class LiquidityTraderBaseBehaviour(BaseBehaviour, ABC):
    """Base behaviour for the liquidity_trader_abci skill."""

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return cast(SynchronizedData, super().synchronized_data)

    @property
    def params(self) -> Params:
        """Return the params."""
        return cast(Params, super().params)

    def default_error(
        self, contract_id: str, contract_callable: str, response_msg: ContractApiMessage
    ) -> None:
        """Return a default contract interaction error message."""
        self.context.logger.error(
            f"Could not successfully interact with the {contract_id} contract "
            f"using {contract_callable!r}: {response_msg}"
        )

    def contract_interaction_error(
        self, contract_id: str, contract_callable: str, response_msg: ContractApiMessage
    ) -> None:
        """Return a contract interaction error message."""
        # contracts can only return one message, i.e., multiple levels cannot exist.
        for level in ("info", "warning", "error"):
            msg = response_msg.raw_transaction.body.get(level, None)
            logger = getattr(self.context.logger, level)
            if msg is not None:
                logger(msg)
                return

        self.default_error(contract_id, contract_callable, response_msg)

    def contract_interact(
        self,
        performative: ContractApiMessage.Performative,
        contract_address: str,
        contract_public_id: PublicId,
        contract_callable: str,
        data_key: str,
        **kwargs: Any,
    ) -> WaitableConditionType:
        """Interact with a contract."""
        contract_id = str(contract_public_id)

        self.context.logger.info(
            f"Interacting with contract {contract_id} at address {contract_address}\n"
            f"Calling method {contract_callable} with parameters: {kwargs}"
        )

        response_msg = yield from self.get_contract_api_response(
            performative,
            contract_address,
            contract_id,
            contract_callable,
            **kwargs,
        )

        self.context.logger.info(f"Contract response: {response_msg}")

        if response_msg.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.default_error(contract_id, contract_callable, response_msg)
            return None

        data = response_msg.raw_transaction.body.get(data_key, None)
        if data is None:
            self.contract_interaction_error(
                contract_id, contract_callable, response_msg
            )
            return None

        return data

    def _get_asset_balances(self) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get asset balances"""
        asset_balances_dict: Dict[str, list] = defaultdict(list)

        if not self.params.allowed_assets:
            self.context.logger.error("No assets provided.")
            return None

        allowed_assets = self.params.allowed_assets

        for chain, assets in allowed_assets.items():
            account = self.params.safe_contract_addresses[chain]

            if account == ZERO_ADDRESS:
                self.context.logger.error(f"No safe address set for chain {chain}")
                continue

            # Temp hack: for now we use bnb in the place of optimism
            if chain == "optimism":
                chain = "bnb"

            for asset_name, asset_address in assets.items():
                # Native balance
                if asset_address == ZERO_ADDRESS:
                    ledger_api_response = yield from self.get_ledger_api_response(
                        performative=LedgerApiMessage.Performative.GET_STATE,  # type: ignore
                        ledger_callable="get_balance",
                        block_identifier="latest",
                        account=account,
                        chain_id=chain,
                    )

                    if (
                        ledger_api_response.performative
                        != LedgerApiMessage.Performative.STATE
                    ):
                        self.context.logger.error(
                            f"Could not calculate the balance of the safe: {ledger_api_response}"
                        )
                        balance = None
                    else:
                        balance = int(
                            ledger_api_response.state.body["get_balance_result"]
                        )

                    asset_balances_dict[chain].append(
                        {
                            "asset_type": "native",
                            "address": asset_address,
                            "balance": balance,
                        }
                    )

                # Token balance
                else:
                    balance = yield from self.contract_interact(
                        performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                        contract_address=asset_address,
                        contract_public_id=ERC20.contract_id,
                        contract_callable="check_balance",
                        data_key="token",
                        account=account,
                        chain_id=chain,
                    )

                    asset_balances_dict[chain].append(
                        {
                            "asset_type": "erc_20",
                            "address": asset_address,
                            "balance": balance,
                        }
                    )

                self.context.logger.info(
                    f"Balance of account {account} on {chain} chain for {asset_name}: {balance}"
                )

        return asset_balances_dict

    def _get_lp_pool_balances(self) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get LP pool balances"""
        pool_balances_dict: Dict[str, list] = defaultdict(list)
        if not self.params.allowed_lp_pool_addresses:
            self.context.logger.error("No LP Pool addresses provided.")
            return None

        lp_pool_addresses = self.params.allowed_lp_pool_addresses

        for dex_type, lp_pools in lp_pool_addresses.items():
            for chain, pools in lp_pools.items():
                for pool_address in pools:
                    account = self.params.safe_contract_addresses[chain]
                    if account == ZERO_ADDRESS:
                        self.context.logger.error(
                            f"No safe address set for chain {chain}"
                        )
                        continue
                    if account is None:
                        self.context.logger.error(
                            f"No account found for chain: {chain}"
                        )
                        return None

                    if dex_type == "balancerPool":
                        contract_callable = "get_balance"
                        contract_id = WeightedPoolContract.contract_id
                    elif dex_type == "velodrome":
                        contract_callable = "get_balance"
                        contract_id = PoolContract.contract_id
                    else:
                        self.context.logger.error(f"{dex_type} not supported")
                        return None

                    # OPTIMISM NOT SUPPORTED YET
                    if chain == "optimism":
                        chain = "bnb"

                    balance = yield from self.contract_interact(
                        performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                        contract_address=pool_address,
                        contract_public_id=contract_id,
                        contract_callable=contract_callable,
                        data_key="balance",
                        account=account,
                        chain_id=chain,
                    )

                    if balance is not None and int(balance) > 0:
                        self.current_pool = {
                            "chain": chain,
                            "address": pool_address,
                            "dex_type": dex_type,
                            "balance": balance,
                        }
                        self.context.logger.info(
                            f"Current pool updated to: {self.current_pool}"
                        )

                    # OPTIMISM NOT SUPPORTED YET
                    if chain == "bnb":
                        chain = "optimism"

                    self.context.logger.info(
                        f"Balance of account {account} on {chain} chain for pool address {pool_address} in {dex_type} DEX: {balance}"
                    )
                    pool_balances_dict[chain].append(
                        {
                            "asset_type": "pool",
                            "address": pool_address,
                            "balance": balance,
                        }
                    )

        return pool_balances_dict

    def get_positions(self) -> Generator[None, None, List[Dict[str, Any]]]:
        """Get positions"""
        asset_balances = yield from self._get_asset_balances()
        pool_balances = yield from self._get_lp_pool_balances()

        all_balances = defaultdict(list)
        if asset_balances:
            for chain, assets in asset_balances.items():
                all_balances[chain].extend(assets)

        if pool_balances:
            for chain, assets in pool_balances.items():
                all_balances[chain].extend(assets)

        positions = [
            {"chain": chain, "assets": assets} for chain, assets in all_balances.items()
        ]

        return positions

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
            chain_id=chain,
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

    def _get_balance(self, chain: str, token: str) -> Optional[int]:
        """Get balance"""
        positions = self.synchronized_data.positions
        for position in positions:
            if position["chain"] == chain:
                for asset in position["assets"]:
                    if asset["address"] == token:
                        return asset["balance"]
        return 0


class GetPositionsBehaviour(LiquidityTraderBaseBehaviour):
    """GetPositionsBehaviour"""

    matching_round: Type[AbstractRound] = GetPositionsRound
    current_pool = None

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            positions = yield from self.get_positions()
            self.context.logger.info(f"POSITIONS: {positions}")
            sender = self.context.agent_address

            if positions is None:
                positions = GetPositionsRound.ERROR_PAYLOAD

            if self.current_pool is None:
                self.current_pool = GetPositionsRound.ERROR_PAYLOAD

            serialized_positions = json.dumps(positions)
            serialized_current_pool = json.dumps(self.current_pool)
            payload = GetPositionsPayload(
                sender=sender,
                positions=serialized_positions,
                current_pool=serialized_current_pool,
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()


class EvaluateStrategyBehaviour(LiquidityTraderBaseBehaviour):
    """EvaluateStrategyBehaviour"""

    matching_round: Type[AbstractRound] = EvaluateStrategyRound
    highest_apr_pool = None

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            yield from self.get_highest_apr_pool()
            invest_in_pool = self.get_decision()
            actions = []
            if invest_in_pool:
                actions = yield from self.get_order_of_transactions()
                self.context.logger.info(f"Actions: {actions}")

            serialized_actions = json.dumps(actions)
            sender = self.context.agent_address
            payload = EvaluateStrategyPayload(sender=sender, actions=serialized_actions)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_highest_apr_pool(self) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get highest APR pool"""
        filtered_pools = yield from self._get_filtered_pools()

        if not filtered_pools:
            self.context.logger.error("No pool data retrieved.")
            return None

        highest_apr = -float("inf")

        for dex_type, chains in filtered_pools.items():
            for chain, campaigns in chains.items():
                for campaign in campaigns:
                    apr = campaign.get("apr", 0)
                    self.context.logger.info(f"{apr} APR")
                    if apr is None:
                        apr = 0
                    if apr > highest_apr:
                        highest_apr = apr
                        try:
                            pool_tokens = list(
                                campaign["typeInfo"]["poolTokens"].keys()
                            )
                        except Exception:
                            self.context.logger.error(
                                f"No underlying token addresses present in the pool {campaign}"
                            )
                            continue
                        self.highest_apr_pool = {
                            "dex_type": dex_type,
                            "chain": chain,
                            "apr": apr,
                            "token0": pool_tokens[0],
                            "token1": pool_tokens[1],
                        }

        if self.highest_apr_pool:
            self.context.logger.info(f"Highest APR pool found: {self.highest_apr_pool}")
        else:
            self.context.logger.warning("No pools with APR found.")

    def _get_filtered_pools(self) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get filtered pools"""
        allowed_lp_pools = self.params.allowed_lp_pool_addresses
        allowed_dexs = list(allowed_lp_pools.keys())
        allowed_assets = self.params.allowed_assets

        filtered_pools: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for chain, chain_id in self.params.allowed_chains.items():
            # api_url = self.params.pool_data_api_url.format(chain_id=chain_id, type=1)  # noqa: E800
            api_url = self.params.pool_data_api_url
            self.context.logger.info(f"{api_url}")

            response = yield from self.get_http_response(
                method="GET",
                url=api_url,
                headers={"accept": "application/json"},
            )

            if response.status_code != 200:
                self.context.logger.error(
                    f"Could not retrieve data from url {api_url} "
                    f"Received status code {response.status_code}."
                )
                return None

            try:
                data = json.loads(response.body)
                self.context.logger.info("Pool Info retrieved from API")
            except (ValueError, TypeError) as e:
                self.context.logger.error(
                    f"Could not parse response from api, "
                    f"the following error was encountered {type(e).__name__}: {e}"
                )
                return None

            try:
                campaigns = data[str(chain_id)]
            except Exception:
                self.context.logger.error(
                    f"No info available for chainId {chain_id} in response"
                )
                continue

            for campaign_list in campaigns.values():
                for campaign in campaign_list.values():
                    dex_type = campaign.get("type", None)
                    if dex_type in allowed_dexs:
                        if dex_type == "balancerPool":
                            pool_tokens = list(
                                campaign["typeInfo"]["poolTokens"].keys()
                            )
                            token0 = pool_tokens[0]
                            token1 = pool_tokens[1]
                        if dex_type == "velodrome":
                            token0 = campaign["typeInfo"].get("token0", None)
                            token1 = campaign["typeInfo"].get("token1", None)
                        if (
                            token0 in allowed_assets[chain].values()
                            and token1 in allowed_assets[chain].values()
                        ):
                            if (
                                campaign.get("mainParameter", None)
                                in allowed_lp_pools[dex_type][chain]
                            ):
                                filtered_pools[dex_type][chain].append(campaign)
                                self.context.logger.info(
                                    f"Added campaign for {chain} on {dex_type}: {campaign}"
                                )

        self.context.logger.info(f"Filtered pools: {filtered_pools}")
        return filtered_pools

    def get_decision(self) -> bool:
        """Get decision"""
        # Step 1: Check highest APR exceeds threshold
        if self.highest_apr_pool:
            exceeds_apr_threshold = (
                self.highest_apr_pool["apr"] > self.params.apr_threshold
            )
            if not exceeds_apr_threshold:
                self.context.logger.info(
                    f"apr of pool {self.highest_apr_pool['apr']} does not exceed apr_threshold {self.params.apr_threshold}"
                )
                return False

        # Step 2: Check if the highest APR pool is better than the current pool
        current_pool = self.synchronized_data.current_pool
        if (
            current_pool
            and self.highest_apr_pool
            and current_pool.get("apr", 0) > self.highest_apr_pool["apr"]
        ):
            self.context.logger.info(
                f"apr of pool {self.highest_apr_pool['apr']} does not exceed current pool apr [{current_pool['apr']}]"
            )
            return False

        # Step 2: Check round interval
        is_round_threshold_exceeded = self._check_round_threshold_exceeded()
        if not is_round_threshold_exceeded:
            return False

        return True

    def _check_round_threshold_exceeded(self) -> bool:
        """Check round threshold exceeded"""
        transaction_history = self.synchronized_data.transaction_history

        if not transaction_history:
            return True

        latest_transaction = transaction_history[-1]
        return (
            latest_transaction["round"] + self.params.round_threshold
            >= self.round_sequence
        )

    def get_order_of_transactions(
        self,
    ) -> Generator[None, None, Optional[List[Dict[str, Any]]]]:
        """Get order of transactions"""
        # Step 1- check if any liquidity exists, otherwise check for funds in safe
        actions = []
        tokens = {}

        if not self.synchronized_data.current_pool:
            # If there is no current pool, then check for which tokens we have balance
            tokens = self._get_tokens_over_min_balance()
        else:
            # If there is current pool, then get the lp pool token addresses
            tokens = yield from self._get_exit_pool_tokens()
            action = {
                "action": Action.EXIT_POOL.value,
                "dex_type": self.current_pool["dex_type"],
                "chain": self.synchronized_data.current_pool["chain"],
                "assets": [tokens[0], tokens[1]],
            }

        if not tokens:
            return []

        # Step 2- build bridge and swap tokens action
        bridge_swap_tokens_pairs = yield from self._get_bridge_and_swap_info(tokens)

        if bridge_swap_tokens_pairs:
            for token_pair in bridge_swap_tokens_pairs:
                action = {
                    "action": Action.BRIDGE_SWAP.value,
                    "source_chain": self.synchronized_data.current_pool["chain"],
                    "destination_chain": self.highest_apr_pool["chain"],
                    "assets": [token_pair[0], token_pair[1]],
                }
            actions.append(action)

        # Step 3: get the info on which pool to enter
        action = {
            "action": Action.ENTER_POOL.value,
            "dex_type": self.highest_apr_pool["dex_type"],
            "chain": self.highest_apr_pool["chain"],
            "assets": [
                self.highest_apr_pool["token0"],
                self.highest_apr_pool["token1"],
            ],
        }
        actions.append(action)

        self.context.logger.info(f"Actions {actions}")
        return actions

    def _get_tokens_over_min_balance(self) -> Optional[List[Any]]:
        """Get tokens over min balance"""
        # check if safe has funds for token0 and token1
        token0 = self.highest_apr_pool["token0"]
        token1 = self.highest_apr_pool["token1"]
        chain = self.highest_apr_pool["chain"]

        if chain == "optimism":
            chain = "bnb"

        token0_balance = self._get_balance(chain, token0)
        token1_balance = self._get_balance(chain, token1)

        if chain == "bnb":
            chain = "optimism"

        tokens = []

        # we need at-most 2 tokens for which we have balance above min_threshold to be able to move forward
        if (
            token0_balance
            > self.params.min_balance_multiplier * self.params.gas_reserve[chain]
        ):
            self.context.logger.info(
                f"SUFFICIENT BALANCE :- {token0} balance {token0_balance}"
            )
            tokens.append([chain, token0])

        if (
            token1_balance
            > self.params.min_balance_multiplier * self.params.gas_reserve[chain]
        ):
            self.context.logger.info(
                f"SUFFICIENT BALANCE :- {token1} balance {token1_balance}"
            )
            tokens.append([chain, token1])

        for position in self.synchronized_data.positions:
            for asset in position["assets"]:
                if asset["asset_type"] in ["erc20", "native"]:
                    min_balance = (
                        self.params.min_balance_multiplier
                        * self.params.gas_reserve[position["chain"]]
                    )
                    if asset["balance"] > min_balance:
                        tokens.append([chain, asset["address"]])

        return tokens

    def _get_exit_pool_tokens(self) -> Generator[None, None, Optional[List[str]]]:
        """Get exit pool tokens"""
        dex_type = self.synchronized_data.current_pool["dex_type"]
        pool_address = self.synchronized_data.current_pool["address"]
        chain = self.synchronized_data.current_pool["chain"]
        if dex_type == "balancerPool":
            # Get poolId from balancer weighted pool contract
            pool_id = yield from self._get_pool_id(pool_address)
            if not pool_id:
                return None

            # Get vault contract address from balancer weighted pool contract
            vault_address = yield from self._get_vault_for_pool(pool_address, chain)
            if not vault_address:
                return None

            # Get pool tokens from balancer vault contract
            tokens = yield from self._get_balancer_pool_tokens(pool_id, vault_address)

        if dex_type == "velodrome":
            # Get pool tokens from velodrome pool contract
            tokens = yield from self._get_velodrome_pool_tokens()

        return [[chain, tokens[0]], [chain, tokens[1]]]

    def _get_pool_id(self, pool_address: str) -> Generator[None, None, Optional[str]]:
        """Get pool ids"""
        pool_id = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=WeightedPoolContract.contract_id,
            contract_callable="get_pool_id",
            data_key="pool_id",
            chain_id=self.synchronized_data.current_pool["chain"],
        )

        if not pool_id:
            self.context.logger.error(
                f"Could not fetch the pool id for pool {pool_address}"
            )
            return None

        self.context.logger.info(f"PoolId for balancer pool {pool_address}: {pool_id}")
        return pool_id

    def _get_balancer_pool_tokens(
        self, pool_id: str, vault_address: str
    ) -> Generator[None, None, Optional[List[str]]]:
        """Get balancer pool tokens"""
        pool_tokens = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=vault_address,
            contract_public_id=VaultContract.contract_id,
            contract_callable="get_pool_tokens",
            data_key="tokens",
        )

        if not pool_tokens:
            self.context.logger.error(
                f"Could not fetch tokens for balancer pool id {pool_id}"
            )
            return None

        self.context.logger.error(
            f"Tokens for balancer poolId {pool_id} : {pool_tokens}"
        )
        return pool_tokens

    def _get_velodrome_pool_tokens(
        self, pool_address: str
    ) -> Generator[None, None, Optional[List[str]]]:
        """Get velodrome pool tokens"""
        pool_tokens = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=PoolContract.contract_id,
            contract_callable="get_pool_tokens",
            data_key="tokens",
        )

        if not pool_tokens:
            self.context.logger.error(
                f"Could not fetch tokens for velodrome pool {pool_address}"
            )
            return None

        self.context.logger.error(
            f"Tokens for velodrome pool {pool_address} : {pool_tokens}"
        )
        return pool_tokens

    def _get_bridge_and_swap_info(
        self, exisiting_tokens: List[str]
    ) -> Generator[None, None, Optional[List[Any]]]:
        """Get bridge and swap info"""
        if not self.highest_apr_pool:
            return None

        destination_token_0 = self.highest_apr_pool["token0"]
        destination_token_1 = self.highest_apr_pool["token1"]
        source_token_0 = exisiting_tokens[0]
        source_token_1 = exisiting_tokens[1]
        destination_chain = self.highest_apr_pool["chain"]
        source_token0_chain = exisiting_tokens[0][0]
        source_token1_chain = exisiting_tokens[1][0]

        bridge_swap_tokens_pairs = []

        if source_token0_chain == source_token1_chain:
            if destination_chain == source_token0_chain:
                # decide if a swap is needed
                if (
                    source_token_0 != destination_token_0
                    and source_token_0 != destination_token_1
                ):
                    from_token = source_token_0
                    to_token = (
                        destination_token_1
                        if source_token_1 == destination_token_0
                        else destination_token_0
                    )
                    bridge_swap_tokens_pairs.append([from_token, to_token])

                if (
                    source_token_1 != destination_token_0
                    and source_token_1 != destination_token_1
                ):
                    from_token = source_token_1
                    to_token = (
                        destination_token_1
                        if source_token_0 == destination_token_0
                        else destination_token_0
                    )
                    bridge_swap_tokens_pairs.append([from_token, to_token])
            else:
                # TO-IMPLEMENT
                # Decide which tokens to swap based and bridge based considering other factors as well (like what will be more profitable)
                bridge_swap_tokens_pairs.append([source_token_0, destination_token_0])
                bridge_swap_tokens_pairs.append([source_token_1, destination_token_1])
        else:
            # ASSUMPTION
            # if we are not in any pool initially, we have funds(i.e. erc20 or native tokens) on the same chain
            return []

        return bridge_swap_tokens_pairs

    def _get_pool_address(
        self, dex_type: str, chain: str, token0: str, token1: str
    ) -> Generator[None, None, Optional[str]]:
        """Get pool address"""
        return None


class DecisionMakingBehaviour(LiquidityTraderBaseBehaviour):
    """DecisionMakingBehaviour"""

    matching_round: Type[AbstractRound] = DecisionMakingRound

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            next_event, updates = yield from self.get_next_event()
            payload = DecisionMakingPayload(
                sender=sender,
                content=json.dumps(
                    {"event": next_event, "updates": updates}, sort_keys=True
                ),
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_next_event(self) -> Generator[None, None, Tuple[str, Dict]]:
        """Get next event"""
        actions = self.synchronized_data.actions

        # If there are no actions, we return
        if not actions:
            self.context.logger.info("No actions to prepare")
            return Event.DONE.value, {}

        positions = self.synchronized_data.positions

        # If the previous round was not EvaluateStrategyRound, we need to update the balances after a transaction
        last_round_id = self.context.state.round_sequence._abci_app._previous_rounds[
            -1
        ].round_id

        if last_round_id != EvaluateStrategyRound.auto_round_id():
            positions = yield from self.get_positions()

        # Prepare the next action
        next_action = Action(actions[0]["action"])

        if next_action == Action.ENTER_POOL:
            tx_hash, chain_id, safe_address = yield from self.get_enter_pool_tx_hash(
                positions
            )

        elif next_action == Action.EXIT_POOL:
            tx_hash, chain_id, safe_address = yield from self.get_exit_pool_tx_hash(
                positions
            )

        elif next_action == Action.ENTER_POOL:
            tx_hash, chain_id, safe_address = yield from self.get_swap_tx_hash(
                positions
            )

        else:
            tx_hash = None
            chain_id = None
            safe_address = None

        if not tx_hash:
            self.context.logger.error("There was an error preparing the next action")
            return Event.DONE.value, {}

        return Event.SETTLE.value, {
            "most_voted_tx_hash": tx_hash,
            "chain_id": chain_id,
            "safe_contract_address": safe_address,
            "positions": positions,
        }

    def get_enter_pool_tx_hash(
        self, positions
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Get enter pool tx hash"""
        if not self.synchronized_data.actions:
            return None, None, None

        dex_type = self.synchronized_data.actions[0]["dex_type"]

        if dex_type == "balancerPool":
            (
                tx_hash,
                chain_id,
                safe_address,
            ) = yield from self.get_enter_pool_balancer_tx_hash(positions)
            return tx_hash, chain_id, safe_address

        if dex_type == "velodrome":
            (
                tx_hash,
                chain_id,
                safe_address,
            ) = yield from self.get_enter_pool_velodrome_tx_hash(positions)
            return tx_hash, chain_id, safe_address

        self.context.logger.error(f"Unknown type of dex: {dex_type}")
        return None, None, None

    def get_enter_pool_balancer_tx_hash(
        self, positions
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Get enter pool tx hash for Balancer"""

        if not self.synchronized_data.actions:
            return None, None, None

        action = self.synchronized_data.actions[0]
        chain = action["chain"]
        if chain == "optimism":
            chain = "bnb"

        # Hardcoded 50WETH_50OLAS pool
        pool_address = "0x5BB3E58887264B667f915130fD04bbB56116C278"
        pool_id = "0x5bb3e58887264b667f915130fd04bbb56116c27800020000000000000000012a"  # getPoolId()

        # Get vault contract address from balancer weighted pool contract
        vault_address = yield from self._get_vault_for_pool(pool_address, chain)
        if not vault_address:
            return None, None, None

        max_amounts_in = [
            self._get_balance(chain, action["assets"][0]),
            self._get_balance(chain, action["assets"][1]),
        ]

        # https://docs.balancer.fi/reference/joins-and-exits/pool-joins.html#userdata
        user_data = 1  # EXACT_TOKENS_IN_FOR_BPT_OUT

        # fromInternalBalance - True if sending from internal token balances. False if sending ERC20.
        from_internal_balance = ZERO_ADDRESS in action["assets"]

        # Get assets balances from positions
        safe_address = self.params.safe_contract_addresses[action["chain"]]

        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=vault_address,
            contract_public_id=VaultContract.contract_id,
            contract_callable="join_pool",
            data_key="tx_hash",
            pool_id=pool_id,
            sender=safe_address,
            recipient=safe_address,
            assets=action["assets"],
            max_amounts_in=max_amounts_in,
            user_data=user_data,
            from_internal_balance=from_internal_balance,
            chain_id=chain,
        )

        if not tx_hash:
            return None, None, None

        safe_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=safe_address,
            contract_public_id=GnosisSafeContract.contract_id,
            contract_callable="get_raw_safe_transaction_hash",
            data_key="tx_hash",
            to_address=vault_address,
            value=ETHER_VALUE,
            data=tx_hash,
            safe_tx_gas=SAFE_TX_GAS,
            chain_id=chain,
        )

        if not safe_tx_hash:
            return None, None, None

        safe_tx_hash = safe_tx_hash[2:]
        self.context.logger.info(f"Hash of the Safe transaction: {safe_tx_hash}")

        payload_string = hash_payload_to_hex(
            safe_tx_hash, ETHER_VALUE, SAFE_TX_GAS, vault_address, tx_hash
        )

        self.context.logger.info(f"Tx hash payload string is {payload_string}")

        return payload_string, chain, safe_address

    def get_enter_pool_velodrome_tx_hash(
        self, positions
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Get enter pool tx hash for Balancer"""
        pass

    def get_exit_pool_tx_hash(
        self, positions
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Get exit pool tx hash"""
        pass

    def get_swap_tx_hash(
        self, positions
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Get swap tx hash"""
        # Call li.fi API
        pass


class LiquidityTraderRoundBehaviour(AbstractRoundBehaviour):
    """LiquidityTraderRoundBehaviour"""

    initial_behaviour_cls = GetPositionsBehaviour
    abci_app_cls = LiquidityTraderAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = [
        GetPositionsBehaviour,
        EvaluateStrategyBehaviour,
        DecisionMakingBehaviour,
    ]

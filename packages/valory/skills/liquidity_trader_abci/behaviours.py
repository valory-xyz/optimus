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
from typing import Any, Dict, Generator, List, Optional, Set, Type, cast

from packages.valory.contracts.balancer_vault.contract import VaultContract
from packages.valory.contracts.balancer_weighted_pool.contract import (
    WeightedPoolContract,
)
from packages.valory.contracts.erc20.contract import ERC20
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
    ClaimOPPayload,
    ClaimOPRound,
    DecisionMakingPayload,
    DecisionMakingRound,
    EvaluateStrategyPayload,
    EvaluateStrategyRound,
    Event,
    GetPositionsPayload,
    GetPositionsRound,
    LiquidityTraderAbciApp,
    PrepareExitPoolTxPayload,
    PrepareExitPoolTxRound,
    PrepareSwapTxPayload,
    PrepareSwapTxRound,
    SynchronizedData,
    TxPreparationPayload,
    TxPreparationRound,
)


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


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


class ClaimOPBehaviour(LiquidityTraderBaseBehaviour):
    """ClaimOPBehaviour"""

    matching_round: Type[AbstractRound] = ClaimOPRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            payload = ClaimOPPayload(sender=sender, content=...)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()


class DecisionMakingBehaviour(LiquidityTraderBaseBehaviour):
    """DecisionMakingBehaviour"""

    matching_round: Type[AbstractRound] = DecisionMakingRound

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            next_action = self.get_next_round()
            try:
                event = Event(next_action)
            except ValueError:
                self.context.logger.error(f"Invalid EVENT: {next_action}")
                event = Event.ERROR

            payload = DecisionMakingPayload(sender=sender, event=event)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_next_round(self) -> str:
        """Get next round"""
        tx_submitter = self.synchronized_data.tx_submitter
        actions = self.synchronized_data.actions

        # Case 1: tx_submitter is not set, execute the first action
        if not tx_submitter:
            next_round = actions[0]["action"]
        else:
            # Case 2: tx_submitter is set, match it and execute the next action
            current_action_index = None
            for index, action in enumerate(actions):
                if action["action"] == tx_submitter:
                    current_action_index = index
                    break

            if current_action_index is None or current_action_index + 1 >= len(actions):
                return ""
            else:
                next_round = actions[current_action_index + 1]["action"]

        return next_round


class EvaluateStrategyBehaviour(LiquidityTraderBaseBehaviour):
    """EvaluateStrategyBehaviour"""

    matching_round: Type[AbstractRound] = EvaluateStrategyRound
    highest_apr_pool = None

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            yield from self.get_highest_apr_pool()
            invest_in_pool = self.get_decision()
            if not invest_in_pool:
                actions = []
            else:
                actions = yield from self.get_order_of_transactions()

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
                self.context.logger.info(f"Pool Info retrieved from API {data}")
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
                    f"No info available for chainId {chain_id} in response {data}"
                )
                continue

            for campaign_list in campaigns.values():
                for campaign in campaign_list.values():
                    dex_type = campaign["type"]
                    # For testing remove the condition since balancer and velodrome are not present in pool data for now
                    if dex_type in allowed_dexs:
                        # if campaign["token0"] in allowed_assets[chain] and campaign["token1"] in allowed_assets[chain]:
                        # pool_address = yield from self._get_pool_address(dex_type, chain, campaign["token0"], campaign["token1"])
                        # if pool_address in allowed_lp_pools[dex_type][chain]:
                        filtered_pools[dex_type][chain].append(campaign)
                        self.context.logger.info(
                            f"Added campaign for {chain} on {dex_type}: {campaign}"
                        )

        self.context.logger.info(f"Filtered pools: {filtered_pools}")
        return filtered_pools

    def get_decision(self) -> bool:
        """Get decision"""
        # Step 1: Check highest APR exceeds threshold
        exceeds_apr_threshold = self.highest_apr_pool["apr"] > self.params.apr_threshold
        if not exceeds_apr_threshold:
            self.context.logger.info(
                f"apr of pool {self.highest_apr_pool['apr']} does not exceed apr_threshold {self.params.apr_threshold}"
            )
            return False

        # Step 2: Check round interval
        is_round_threshold_exceeded = self._check_round_threshold_exceeded()
        if not is_round_threshold_exceeded:
            return False

        return True

    def _check_round_threshold_exceeded(self) -> bool:
        """Check round threshold exceeded"""
        transacation_history = self.synchronized_data.transaction_history

        if not transacation_history:
            return True

        latest_transaction = transacation_history[-1]
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
                "chain": self.synchronized_data.current_pool["chain"],
                "assets": [tokens[0], tokens[1]],
            }

        if not tokens:
            return {}

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

        token0_balance = self._get_balance(chain, token0)
        token1_balance = self._get_balance(chain, token1)

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

    def _get_balance(self, chain: str, token: str) -> Optional[int]:
        """Get balance"""
        positions = self.synchronized_data.positions
        for position in positions:
            if position["chain"] == chain:
                for asset in position["assets"]:
                    if asset["address"] == token:
                        return asset["balance"]
        return 0

    def _get_exit_pool_tokens(self) -> Generator[None, None, List[str]]:
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
            vault_address = yield from self._get_vault_for_pool(pool_address)
            if not vault_address:
                return None

            # Get pool tokens from balancer vault contract
            tokens = yield from self._get_balancer_pool_tokens(pool_id, vault_address)

        if dex_type == "velodrome":
            # Get pool tokens from velodrome pool contract
            tokens = yield from self._get_velodrome_pool_tokens()

        return [[chain, tokens[0]], [chain, tokens[1]]]

    def _get_pool_id(self, pool_address: str) -> Generator[None, None, str]:
        """Get pool ids"""
        response_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=pool_address,
            contract_id=str(WeightedPoolContract.contract_id),
            contract_callable="get_pool_id",
            chain_id=self.synchronized_data.current_pool["chain"],
        )

        if response_msg.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.context.logger.error(
                f"Could not fetch the pool id for pool {pool_address}: {response_msg}"
            )
            return None

        else:
            pool_id = response_msg.raw_transaction.body.get("pool_id", None)
            self.context.logger.info(
                f"PoolId for balancer pool {pool_address}: {pool_id}"
            )
            return pool_id

    def _get_vault_for_pool(self, pool_address: str) -> Generator[None, None, str]:
        """Get vault for pool"""
        response_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=pool_address,
            contract_id=str(WeightedPoolContract.contract_id),
            contract_callable="get_vault_address",
            chain_id=self.synchronized_data.current_pool["chain"],
        )

        if response_msg.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.context.logger.error(
                f"Could not fetch the vault address for pool {pool_address}: {response_msg}"
            )
            return None

        else:
            vault_address = response_msg.raw_transaction.body.get("vault", None)
            self.context.logger.info(
                f"Vault contract address for balancer pool {pool_address}: {vault_address}"
            )
            return vault_address

    def _get_balancer_pool_tokens(
        self, pool_id: str, vault_address: str
    ) -> Generator[None, None, List[str]]:
        """Get balancer pool tokens"""
        response_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=vault_address,
            contract_id=str(VaultContract.contract_id),
            contract_callable="get_pool_tokens",
            pool_id=pool_id,
        )

        if response_msg.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.context.logger.error(
                f"Could not fetch tokens for balancer pool id {pool_id}: {response_msg}"
            )
            tokens = None

        else:
            tokens = response_msg.raw_transaction.body.get("tokens", None)
            self.context.logger.error(
                f"Tokens for balancer poolId {pool_id} : {tokens}"
            )
            return tokens

    def _get_velodrome_pool_tokens(
        self, pool_address: str
    ) -> Generator[None, None, List[str]]:
        """Get velodrome pool tokens"""
        response_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=pool_address,
            contract_id=str(PoolContract.contract_id),
            contract_callable="get_pool_tokens",
        )

        if response_msg.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.context.logger.error(
                f"Could not fetch tokens for velodrome pool {pool_address}: {response_msg}"
            )
            tokens = None

        else:
            tokens = response_msg.raw_transaction.body.get("tokens", None)
            self.context.logger.error(
                f"Tokens for velodrome pool {pool_address} : {tokens}"
            )
            return tokens

    def _get_bridge_and_swap_info(
        self, exisiting_tokens: List[str]
    ) -> Generator[None, None, Optional[List[Any]]]:
        """Get bridge and swap info"""
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
        pass


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

    def get_positions(self) -> Generator[None, None, List[Dict[str, Any]]]:
        """Get positions"""
        asset_balances = yield from self._get_asset_balances()
        pool_balances = yield from self._get_lp_pool_balances()

        all_balances = defaultdict(list)
        for chain, assets in asset_balances.items():
            all_balances[chain].extend(assets)
        for chain, assets in pool_balances.items():
            all_balances[chain].extend(assets)

        positions = [
            {"chain": chain, "assets": assets} for chain, assets in all_balances.items()
        ]

        return positions

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
            if chain == "optimism":
                chain = "bnb"
            for asset_name, asset_address in assets.items():
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

                else:
                    response_msg = yield from self.get_contract_api_response(
                        performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
                        contract_address=asset_address,
                        contract_id=str(ERC20.contract_id),
                        contract_callable="check_balance",
                        account=account,
                        chain_id=chain,
                    )

                    if (
                        response_msg.performative
                        != ContractApiMessage.Performative.RAW_TRANSACTION
                    ):
                        self.context.logger.error(
                            f"Could not calculate the balance of the safe: {response_msg}"
                        )
                        balance = None

                    else:
                        balance = response_msg.raw_transaction.body.get("token", None)

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
                        contract_id = str(WeightedPoolContract.contract_id)
                    elif dex_type == "velodrome":
                        contract_callable = "get_balance"
                        contract_id = str(PoolContract.contract_id)
                    else:
                        self.context.logger.error(f"{dex_type} not supported")
                        return None

                    # OPTIMISM NOT SUPPORTED YET
                    if chain == "optimism":
                        chain = "bnb"

                    response_msg = yield from self.get_contract_api_response(
                        performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
                        contract_address=pool_address,
                        contract_id=contract_id,
                        contract_callable=contract_callable,
                        account=account,
                        chain_id=chain,
                    )

                    if (
                        response_msg.performative
                        != ContractApiMessage.Performative.RAW_TRANSACTION
                    ):
                        self.context.logger.error(
                            f"Could not calculate the balance of the safe: {response_msg}"
                        )
                        balance = None

                    else:
                        balance = response_msg.raw_transaction.body.get("balance", None)

                    if balance > 0:
                        self.current_pool = {
                            "chain": chain,
                            "address": pool_address,
                            "dex_type": dex_type,
                            "balance": balance,
                        }
                        self.context.logger.info(f"Current pool : {self.current_pool}")

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


class PrepareExitPoolTxBehaviour(LiquidityTraderBaseBehaviour):
    """PrepareExitPoolTxBehaviour"""

    matching_round: Type[AbstractRound] = PrepareExitPoolTxRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            payload = PrepareExitPoolTxPayload(sender=sender, content=...)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()


class PrepareSwapTxBehaviour(LiquidityTraderBaseBehaviour):
    """PrepareSwapTxBehaviour"""

    matching_round: Type[AbstractRound] = PrepareSwapTxRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            payload = PrepareSwapTxPayload(sender=sender, content=...)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()


class TxPreparationBehaviour(LiquidityTraderBaseBehaviour):
    """TxPreparationBehaviour"""

    matching_round: Type[AbstractRound] = TxPreparationRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            payload = TxPreparationPayload(sender=sender, content=...)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()


class LiquidityTraderRoundBehaviour(AbstractRoundBehaviour):
    """LiquidityTraderRoundBehaviour"""

    initial_behaviour_cls = GetPositionsBehaviour
    abci_app_cls = LiquidityTraderAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = [
        ClaimOPBehaviour,
        DecisionMakingBehaviour,
        EvaluateStrategyBehaviour,
        GetPositionsBehaviour,
        PrepareExitPoolTxBehaviour,
        PrepareSwapTxBehaviour,
        TxPreparationBehaviour,
    ]

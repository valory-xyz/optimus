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

from abc import ABC
from typing import Generator, Set, Type, cast, Optional, List, Dict, Any
from collections import defaultdict
import json

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)

from packages.valory.skills.liquidity_trader_abci.models import Params
from packages.valory.skills.liquidity_trader_abci.rounds import (
    SynchronizedData,
    LiquidityTraderAbciApp,
    ClaimOPRound,
    DecisionMakingRound,
    EvaluateStrategyRound,
    GetPositionsRound,
    PrepareExitPoolTxRound,
    PrepareSwapTxRound,
    TxPreparationRound,
    Event
)
from packages.valory.skills.liquidity_trader_abci.rounds import (
    ClaimOPPayload,
    DecisionMakingPayload,
    EvaluateStrategyPayload,
    GetPositionsPayload,
    PrepareExitPoolTxPayload,
    PrepareSwapTxPayload,
    TxPreparationPayload,
)
from packages.valory.protocols.contract_api import ContractApiMessage

from packages.valory.contracts.balancer_weighted_pool.contract import WeightedPoolContract
from packages.valory.contracts.velodrome_pool.contract import PoolContract
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

    def async_act(self) -> Generator:
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            # apr_data = yield from self.get_apr_for_pools()
            # actions = [
            #             {
            #                 "action": "exit_pool",
            #                 "chain": "ethereum",
            #                 "pool_address": "0x00...",
            #                 "assets": [
            #                     {"asset": "ETH", "amount": 10},
            #                     {"asset": "DAI", "amount": 200}
            #                 ],
            #                 "additional_params": {},
            #             },
            #             {
            #                 "action": "bridge_and_swap",
            #                 "source": {
            #                     "chain": "ethereum",
            #                     "token": "DAI"
            #                 },
            #                 "destination": {
            #                     "chain": "optimism",
            #                     "token": "USDC"
            #                 },
            #                 "amount": 300,
            #                 "additional_params": {},
            #             },
            #             {
            #                 "action": "bridge_and_swap",
            #                 "source": {
            #                     "chain": "ethereum",
            #                     "token": "USDC"
            #                 },
            #                 "destination": {
            #                     "chain": "optimism",
            #                     "token": "USDC"
            #                 },
            #                 "amount": 300,
            #                 "additional_params": {},
            #             },
            #             {
            #                 "action": "enter_pool",
            #                 "chain": "optimism",
            #                 "pool_address": "0x00...",
            #                 "assets": [
            #                     {"asset": "ETH", "amount": 20},
            #                     {"asset": "DAI", "amount": 100}
            #                 ],
            #                 "additional_params": {},
            #             }
            #         ]
            serialized_actions = json.dumps(actions)
            sender = self.context.agent_address
            payload = EvaluateStrategyPayload(sender=sender, actions=serialized_actions)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

class GetPositionsBehaviour(LiquidityTraderBaseBehaviour):
    """GetPositionsBehaviour"""

    matching_round: Type[AbstractRound] = GetPositionsRound

    def async_act(self) -> Generator:
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            positions = yield from self._get_positions()
            self.context.logger.info(f"POSITIONS: {positions}")
            sender = self.context.agent_address

            if positions is None:
                payload = GetPositionsPayload(sender=sender, positions=GetPositionsRound.ERROR_PAYLOAD)
            else:
                serialized_positions = json.dumps(positions)
                payload = GetPositionsPayload(sender=sender, positions=serialized_positions)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def _get_positions(self) -> Generator[None, None, Optional[List[Dict[str,str]]]]:
        positions_dict: Dict[str, list] = defaultdict(list)

        if not self.params.assets:
            self.context.logger.error("No assets provided.")
            return None
        
        assets = json.loads(json.loads(self.params.assets)) 

        for dex_type, asset_list in assets.items():
            for asset in asset_list:
                chain = asset[0]
                pool_address = asset[1]
                account = self._get_safe_contract_address(chain)

                if account is None:
                    self.context.logger.error(f"No account found for chain: {chain}")
                    return None

                if dex_type == "balancer":
                    contract_callable = "get_balance"
                    contract_id = str(WeightedPoolContract.contract_id)
                elif dex_type == "velodrome":
                    contract_callable = "get_balance"
                    contract_id = str(PoolContract.contract_id)
                else:
                    self.context.logger.error(f"{dex_type} not supported")
                    return None
                
                #OPTIMISM NOT SUPPORTED YET
                if chain == "optimism":
                    chain = "bnb"

                response_msg = yield from self.get_contract_api_response(
                    performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
                    contract_address=pool_address,
                    contract_id=contract_id,
                    contract_callable=contract_callable,
                    account=account,
                    chain_id=chain
                )

                if response_msg.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
                    self.context.logger.error(
                        f"Could not calculate the balance of the safe: {response_msg}"
                    )
                    return None

                balance = response_msg.raw_transaction.body.get("balance", None)

                #OPTIMISM NOT SUPPORTED YET
                if chain == "bnb":
                    chain = "optimism"
                self.context.logger.info(f"Balance of account {account} on {chain} chain for pool address {pool_address} in {dex_type} DEX: {balance}")
                positions_dict[chain].append({"pool_address": pool_address, "dex_type": dex_type,"balance": balance})

        positions = [{"chain": chain, "assets": assets} for chain, assets in positions_dict.items()]
        return positions
    
    def _get_safe_contract_address(self, chain:str) -> str:
        safe_contract_addresses_mapping = self.params.safe_contract_addresses

        for (chain_id,safe_contract_address) in safe_contract_addresses_mapping:
            if chain_id == chain:
                return safe_contract_address
            
        return None


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
        TxPreparationBehaviour
    ]

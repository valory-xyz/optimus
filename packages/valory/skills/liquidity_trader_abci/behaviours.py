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
import os.path
from abc import ABC
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Type, cast
from urllib.parse import urlencode

from aea.configurations.data_types import PublicId

from packages.valory.contracts.erc20.contract import ERC20
from packages.valory.contracts.gnosis_safe.contract import (
    GnosisSafeContract,
    SafeOperation,
)
from packages.valory.contracts.multisend.contract import (
    MultiSendContract,
    MultiSendOperation,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.models import Params
from packages.valory.skills.liquidity_trader_abci.pools.balancer import (
    BalancerPoolBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.pools.uniswap import (
    UniswapPoolBehaviour,
)
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

class DexTypes(Enum):
    BALANCER = "balancerPool"
    UNISWAP_V3 = "UniswapV3"

class Action(Enum):
    """Action"""

    # Kept the values as Round name, so that in DecisionMaking we can match tx_submitter with action name(which is round name) and decide the next action
    EXIT_POOL = "ExitPool"
    ENTER_POOL = "EnterPool"
    BRIDGE_SWAP = "BridgeAndSwap"


class SwapStatus(Enum):
    """SwapStatus"""

    DONE = "DONE"
    PENDING = "PENDING"
    INVALID = "INVALID"
    NOT_FOUND = "NOT_FOUND"
    FAILED = "FAILED"


class SwapPendingSubStatus(Enum):
    """SwapPendingSubStatus"""

    WAIT_SOURCE_CONFIRMATIONS = "WAIT_SOURCE_CONFIRMATIONS"
    WAIT_DESTINATION_TRANSACTION = "WAIT_DESTINATION_TRANSACTION"


class SwapDoneSubStaus(Enum):
    """SwapDoneSubStaus"""

    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    REFUNDED = "REFUNDED"


class Decision(Enum):
    """Decision"""

    CONTINUE = "continue"
    WAIT = "wait"
    EXIT = "exit"


ASSETS_FILENAME = "assets.json"
POOL_FILENAME = "current_pool.json"
READ_MODE = "r"
WRITE_MODE = "w"


class LiquidityTraderBaseBehaviour(BalancerPoolBehaviour, UniswapPoolBehaviour, ABC):
    """Base behaviour for the liquidity_trader_abci skill."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize `LiquidityTraderBaseBehaviour`."""
        super().__init__(**kwargs)
        parent_dir = os.path.dirname(self.context.data_dir)
        self.assets: Dict[str, Any] = {}
        # TO-DO: this will not work if we run it as a service
        self.assets_filepath = os.path.join(parent_dir, ASSETS_FILENAME)
        self.current_pool: Dict[str, Any] = {}
        self.current_pool_filepath: str = os.path.join(parent_dir, POOL_FILENAME)
        self.pools: Dict[str, Any] = {}
        self.pools[DexTypes.BALANCER.value] = BalancerPoolBehaviour
        self.pools[DexTypes.UNISWAP_V3.value] = UniswapPoolBehaviour
        # Read the assets and current pool
        self.read_current_pool()
        self.read_assets()

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

    def get_positions(self) -> Generator[None, None, List[Dict[str, Any]]]:
        """Get positions"""
        asset_balances = yield from self._get_asset_balances()
        all_balances = defaultdict(list)
        if asset_balances:
            for chain, assets in asset_balances.items():
                all_balances[chain].extend(assets)

        positions = [
            {"chain": chain, "assets": assets} for chain, assets in all_balances.items()
        ]

        return positions

    def _get_asset_balances(self) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get asset balances"""
        asset_balances_dict: Dict[str, list] = defaultdict(list)

        for chain, assets in self.assets.items():
            account = self.params.safe_contract_addresses.get(chain, ZERO_ADDRESS)
            if account == ZERO_ADDRESS:
                self.context.logger.error(f"No safe address set for chain {chain}")
                continue

            for asset_symbol, asset_address in assets.items():
                if asset_address == ZERO_ADDRESS:
                    balance = yield from self._get_native_balance(chain, account)
                else:
                    balance = yield from self._get_token_balance(
                        chain, account, asset_address
                    )

                asset_balances_dict[chain].append(
                    {
                        "asset_symbol": asset_symbol,
                        "asset_type": "native"
                        if asset_address == ZERO_ADDRESS
                        else "erc_20",
                        "address": asset_address,
                        "balance": balance,
                    }
                )

                self.context.logger.info(
                    f"Balance of account {account} on {chain} chain for {asset_symbol}: {balance}"
                )

        return asset_balances_dict

    def _get_native_balance(
        self, chain: str, account: str
    ) -> Generator[None, None, Optional[int]]:
        """Get native balance"""
        ledger_api_response = yield from self.get_ledger_api_response(
            performative=LedgerApiMessage.Performative.GET_STATE,
            ledger_callable="get_balance",
            block_identifier="latest",
            account=account,
            chain_id=chain,
        )

        if ledger_api_response.performative != LedgerApiMessage.Performative.STATE:
            self.context.logger.error(
                f"Could not calculate the balance of the safe: {ledger_api_response}"
            )
            return None

        return int(ledger_api_response.state.body["get_balance_result"])

    def _get_token_balance(
        self, chain: str, account: str, asset_address: str
    ) -> Generator[None, None, Optional[int]]:
        """Get token balance"""
        balance = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=asset_address,
            contract_public_id=ERC20.contract_id,
            contract_callable="check_balance",
            data_key="token",
            account=account,
            chain_id=chain,
        )
        return balance

    def _get_balance(
        self, chain: str, token: str, positions: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[int]:
        """Get balance"""
        if not positions:
            positions = self.synchronized_data.positions

        for position in positions:
            if position["chain"] == chain:
                for asset in position["assets"]:
                    if asset["address"] == token:
                        return asset["balance"]
        return None

    def _store_data(self, data: Any, attribute: str, filepath: str) -> None:
        """Generic method to store data as JSON."""
        if data is None:
            self.context.logger.warning(f"No {attribute} to store.")
            return

        try:
            with open(filepath, WRITE_MODE) as file:
                try:
                    json.dump(data, file)
                    return
                except (IOError, OSError):
                    err = f"Error writing to file {filepath!r}!"
        except (FileNotFoundError, PermissionError, OSError) as e:
            err = f"Error writing to file {filepath!r}: {str(e)}"

        self.context.logger.error(err)

    def _read_data(self, attribute: str, filepath: str) -> None:
        """Generic method to read data from a JSON file"""
        try:
            with open(filepath, READ_MODE) as file:
                try:
                    data = json.load(file)
                    setattr(self, attribute, data)
                    return
                except (json.JSONDecodeError, TypeError) as e:
                    err = f"Error decoding {attribute} from {filepath!r}: {str(e)}"
                    setattr(self, attribute, {})
        except (FileNotFoundError, PermissionError, OSError) as e:
            err = f"Error reading from file {filepath!r}: {str(e)}"
            setattr(self, attribute, {})

        self.context.logger.error(err)

    def store_assets(self) -> None:
        """Store the list of assets as JSON."""
        self._store_data(self.assets, "assets", self.assets_filepath)

    def read_assets(self) -> None:
        """Read the list of assets as JSON."""
        self._read_data("assets", self.assets_filepath)

    def store_current_pool(self) -> None:
        """Store the current pool as JSON."""
        self._store_data(self.current_pool, "current_pool", self.current_pool_filepath)

    def read_current_pool(self) -> None:
        """Read the current pool as JSON."""
        self._read_data("current_pool", self.current_pool_filepath)


class GetPositionsBehaviour(LiquidityTraderBaseBehaviour):
    """GetPositionsBehaviour"""

    matching_round: Type[AbstractRound] = GetPositionsRound
    current_pool = None

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            if not self.assets:
                self.assets = self.params.initial_assets
                self.store_assets()

            positions = yield from self.get_positions()
            self.context.logger.info(f"POSITIONS: {positions}")
            sender = self.context.agent_address

            if positions is None:
                positions = GetPositionsRound.ERROR_PAYLOAD

            serialized_positions = json.dumps(positions, sort_keys=True)
            payload = GetPositionsPayload(sender=sender, positions=serialized_positions)

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
            actions = []
            if self.highest_apr_pool is not None:
                invest_in_pool = self.get_decision()
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
            self.context.logger.error("No pool data retrieved")
            return None

        highest_apr = -float("inf")
        self.highest_apr_pool = None

        for dex_type, chains in filtered_pools.items():
            for chain, campaigns in chains.items():
                for campaign in campaigns:
                    apr = campaign.get("apr", 0)
                    if apr is None:
                        apr = 0
                    if apr > highest_apr:
                        highest_apr = apr
                        self.highest_apr_pool = self._extract_pool_info(
                            dex_type, chain, apr, campaign
                        )

        if self.highest_apr_pool:
            self.context.logger.info(f"Highest APR pool found: {self.highest_apr_pool}")
        else:
            self.context.logger.warning("No pools with APR found.")

        return self.highest_apr_pool

    def _extract_pool_info(
        self, dex_type, chain, apr, campaign
    ) -> Optional[Dict[str, Any]]:
        """Extract pool info from campaign data"""
        # TO-DO: Add support for pools with more than two tokens.

        pool_token_addresses = []
        pool_token_symbols = []

        if dex_type == DexTypes.BALANCER.value:
            type_info = campaign.get("typeInfo", {})
            pool_tokens = type_info.get("poolTokens", {})    
            if not pool_tokens:
                self.context.logger.error(
                    f"No pool tokens info present in campaign {campaign}"
                )
                return None           
            pool_token_addresses = list(pool_tokens.keys())
            pool_token_symbols = [token.get("symbol", "Unknown") for token in pool_tokens.values()]

        if dex_type == DexTypes.UNISWAP_V3.value:
            pool_info = campaign.get("campaignParameters", {})
            token0 = pool_info.get("token0", "")
            token1 = pool_info.get("token1", "")
            token0_symbol = pool_info.get("symbolToken0", "")
            token1_symbol = pool_info.get("symbolToken1", "")
            pool_token_addresses.extend([token0, token1])
            pool_token_symbols.extend([token0_symbol, token1_symbol])

        if not pool_token_addresses or not pool_token_symbols or len(pool_token_addresses) < 2 or len(pool_token_symbols) < 2:
            self.context.logger.error(
                f"Incomplete data for pool addresses or pool symbols: {pool_token_addresses} {pool_token_symbols}"
            )
            return None

        pool_address = campaign.get("mainParameter")
        if not pool_address:
            self.context.logger.error(f"Missing pool address in campaign {campaign}")
            return None

        return {
            "dex_type": dex_type,
            "chain": chain,
            "apr": apr,
            "token0": pool_token_addresses[0],
            "token0_symbol": pool_token_symbols[0],
            "token1": pool_token_addresses[1],
            "token1_symbol": pool_token_symbols[1],
            "pool_address": pool_address,
        }

    def _get_filtered_pools(self) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get filtered pools"""

        filtered_pools = defaultdict(lambda: defaultdict(list))

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
                    f"Could not retrieve data from url {api_url}. Status code {response.status_code}."
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

            campaigns = data.get(str(chain_id))
            if not campaigns:
                self.context.logger.error(
                    f"No info available for chainId {chain_id} in response"
                )
                continue

            self._filter_campaigns(chain, campaigns, filtered_pools)

        return filtered_pools

    def _filter_campaigns(self, chain, campaigns, filtered_pools):
        """Filter campaigns based on allowed assets and LP pools"""
        allowed_dexs = self.params.allowed_dexs

        for campaign_list in campaigns.values():
            for campaign in campaign_list.values():
                dex_type = campaign.get("type", "") if campaign.get("type", "") else campaign.get("ammName", "")
                if not dex_type:
                    self.context.logger.warning("Dex type not specified in campaign")
                    continue
                campaign_apr = campaign.get("apr")
                if campaign_apr is None:
                    self.context.logger.warning("APR not specified for campaign")
                    continue
                # The pool apr should be greater than the current pool apr
                if campaign_apr > self.current_pool.get("apr", 0.0):
                    if dex_type in allowed_dexs:
                        campaign_pool_address = campaign.get("mainParameter", "")
                        current_pool_address = self.current_pool.get("address", "")
                        if not campaign_pool_address:
                            self.context.logger.warning(
                                "No pool address found for campaign"
                            )
                            continue
                        # The pool should not be the current pool
                        if campaign_pool_address != current_pool_address:
                            filtered_pools[dex_type][chain].append(campaign)

    def get_decision(self) -> bool:
        """Get decision"""
        if not self._is_apr_threshold_exceeded():
            return False

        # TO-DO: Decide on the correct method/logic for maintaining the period number for the last transaction.
        # if not self._is_round_threshold_exceeded():  # noqa: E800
        #     self.context.logger.info("Round threshold not exceeded")  # noqa: E800
        #     return False  # noqa: E800

        return True

    def _is_apr_threshold_exceeded(self) -> bool:
        """Check if the highest APR exceeds the threshold"""
        if not self.highest_apr_pool:
            self.context.logger.info("No highest APR pool found.")
            return False

        highest_apr = self.highest_apr_pool.get("apr", 0)
        if highest_apr > self.params.apr_threshold:
            return True
        else:
            self.context.logger.info(
                f"APR of selected pool {highest_apr} does not exceed APR threshold ({self.params.apr_threshold})."
            )
            return False

    def get_order_of_transactions(
        self,
    ) -> Generator[None, None, Optional[List[Dict[str, Any]]]]:
        """Get the order of transactions to perform based on the current pool status and token balances."""
        actions = []

        if not self.current_pool:
            tokens = self._get_tokens_over_min_balance()
            if not tokens:
                self.context.logger.error(
                    "Minimun 2 tokens required in safe with over minimum balance to enter a pool"
                )
                return None
        else:
            # If there is current pool, then get the lp pool token addresses
            tokens = yield from self._get_exit_pool_tokens()  # noqa: E800
            if not tokens:
                return None

            exit_pool_action = self._build_exit_pool_action(tokens)
            if not exit_pool_action:
                self.context.logger.error("Error building exit pool action")
                return None

            actions.append(exit_pool_action)

        bridge_swap_actions = self._build_bridge_swap_actions(tokens)
        if bridge_swap_actions:
            actions.extend(bridge_swap_actions)

        enter_pool_action = self._build_enter_pool_action()
        if not enter_pool_action:
            self.context.logger.error("Error building enter pool action")
            return None
        actions.append(enter_pool_action)

        self.context.logger.info(f"Actions: {actions}")
        return actions

    def _get_tokens_over_min_balance(self) -> Optional[List[Any]]:
        # ASSUMPTION : WE HAVE FUNDS FOR ATLEAST 2 TOKENS
        """Get tokens over min balance"""
        tokens = []
        highest_apr_chain = self.highest_apr_pool.get("chain", "")
        token0 = self.highest_apr_pool.get("token0", "")
        token1 = self.highest_apr_pool.get("token1", "")
        token0_symbol = self.highest_apr_pool.get("token0", "")
        token1_symbol = self.highest_apr_pool.get("token1", "")

        # Ensure we have valid data before proceeding
        if (
            not highest_apr_chain
            or not token0
            or not token1
            or not token0_symbol
            or not token1_symbol
        ):
            self.context.logger.error(
                f"Missing data in highest_apr_pool {self.highest_apr_pool}"
            )
            return None

        # minimum balance threshold based on the current chain
        min_balance_threshold = (
            self.params.min_balance_multiplier
            * self.params.gas_reserve.get(highest_apr_chain, 0)
        )

        # Check balances for token0 and token1 on the highest APR pool chain
        for token, symbol in [(token0, token0_symbol), (token1, token1_symbol)]:
            balance = self._get_balance(highest_apr_chain, token)

            if balance > min_balance_threshold:
                tokens.append(
                    {"chain": highest_apr_chain, "token": token, "token_symbol": symbol}
                )

        # We needs funds for atleast 2 tokens
        if len(tokens) == 2:
            return tokens

        seen_tokens = set((token["chain"], token["token"]) for token in tokens)

        # If we still need more tokens, check all positions
        if len(tokens) < 2:
            for position in self.synchronized_data.positions:
                chain = position.get("chain", "")
                for asset in position.get("assets", []):
                    asset_address = asset.get("address", "")
                    if not chain or not asset_address:
                        continue
                    if (chain, asset_address) not in seen_tokens:
                        if asset.get("asset_type", "") in ["erc_20", "native"]:
                            min_balance = (
                                self.params.min_balance_multiplier
                                * self.params.gas_reserve.get(chain, 0)
                            )
                            if asset.get("balance", 0) > min_balance:
                                tokens.append(
                                    {
                                        "chain": position["chain"],
                                        "token": asset["address"],
                                        "token_symbol": asset["asset_symbol"],
                                    }
                                )
                                if len(tokens) == 2:
                                    return tokens

        return tokens

    def _get_exit_pool_tokens(self) -> Generator[None, None, Optional[List[Any]]]:
        """Get exit pool tokens"""

        if not self.current_pool:
            self.context.logger.error("No pool present")
            return None

        dex_type = self.current_pool.get("dex_type")
        pool_address = self.current_pool.get("address")
        chain = self.current_pool.get("chain")

        pool = self.pools.get("balancerPool", None)
        if pool is None:
            self.context.logger.error(f"Unknown dex type: {dex_type}")
            return None

        if dex_type == "balancerPool":
            # Get tokens from balancer weighted pool contract
            tokens = yield from pool._get_tokens(self, pool_address, chain)
            if not tokens:
                return None

        # Assuming tokens list contains tuples with token addresses
        if len(tokens) < 2:
            self.context.logger.error("Not enough tokens found in the pool.")
            return None

        return [
            {
                "chain": chain,
                "token": tokens[0][0],
                "token_symbol": self._get_asset_symbol(chain, tokens[0][0]),
            },
            {
                "chain": chain,
                "token": tokens[0][1],
                "token_symbol": self._get_asset_symbol(chain, tokens[0][1]),
            },
        ]

    def _build_exit_pool_action(self, tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build action for exiting the current pool."""
        if not self.current_pool:
            self.context.logger.error("No pool present")
            return None

        if len(tokens) < 2:
            self.context.logger.error("Insufficient tokens provided for exit action.")
            return None

        return {
            "action": Action.EXIT_POOL.value,
            "dex_type": self.current_pool.get("dex_type"),
            "chain": self.current_pool.get("chain"),
            "assets": [tokens[0].get("token"), tokens[1].get("token")],
            "pool_address": self.current_pool.get("address"),
        }

    def _build_bridge_swap_actions(
        self, tokens: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build bridge and swap actions for the given tokens."""
        # TO-DO: Add logic to handle swaps when there is a balance for only one token.
        if not self.highest_apr_pool:
            self.context.logger.error("No pool present.")
            return None

        bridge_swap_actions = []

        # Get the highest APR pool's tokens
        dest_token0_address = self.highest_apr_pool.get("token0", "")
        dest_token1_address = self.highest_apr_pool.get("token1", "")
        dest_token0_symbol = self.highest_apr_pool.get("token0_symbol", "")
        dest_token1_symbol = self.highest_apr_pool.get("token1_symbol", "")
        dest_chain = self.highest_apr_pool.get("chain", "")

        if (
            not dest_token0_address
            or not dest_token1_address
            or not dest_token0_symbol
            or not dest_token1_symbol
            or not dest_chain
        ):
            self.context.logger.error(
                f"Incomplete data in highest APR pool {self.highest_apr_pool}"
            )
            return []

        source_token0_chain = tokens[0].get("chain", "")
        source_token0_address = tokens[0].get("token", "")
        source_token0_symbol = tokens[0].get("token_symbol", "")
        source_token1_chain = tokens[1].get("chain", "")
        source_token1_address = tokens[1].get("token", "")
        source_token1_symbol = tokens[1].get("token_symbol", "")

        if (
            not source_token0_chain
            or not source_token0_address
            or not source_token0_symbol
            or not source_token1_chain
            or not source_token1_address
            or not source_token1_symbol
        ):
            self.context.logger.error(f"Incomplete data in tokens {tokens}")
            return []

        # If either of the token to swap and destination token match, we need to check which token don't match and build the bridge swap action based on this assessment.
        if source_token0_chain == dest_chain or source_token1_chain == dest_chain:
            if source_token0_address not in [dest_token0_address, dest_token1_address]:
                # for example :- from_tokens = [usdc, xdai], to_tokens = [xdai, weth], then the pair to be created is [usdc, weth]
                to_token = (
                    dest_token1_address
                    if source_token1_address == dest_token0_address
                    else dest_token0_address
                )
                to_token_symbol = (
                    dest_token0_symbol
                    if to_token == dest_token0_address
                    else dest_token1_symbol
                )

                bridge_swap_action = {
                    "action": Action.BRIDGE_SWAP.value,
                    "from_chain": source_token0_chain,
                    "to_chain": dest_chain,
                    "from_token": source_token0_address,
                    "from_token_symbol": source_token0_symbol,
                    "to_token": to_token,
                    "to_token_symbol": to_token_symbol,
                }
                bridge_swap_actions.append(bridge_swap_action)

            if source_token1_address not in [dest_token0_address, dest_token1_address]:
                to_token = (
                    dest_token0_address
                    if source_token0_address == dest_token1_address
                    else dest_token1_address
                )
                to_token_symbol = (
                    dest_token0_symbol
                    if to_token == dest_token0_address
                    else dest_token1_symbol
                )

                bridge_swap_action = {
                    "action": Action.BRIDGE_SWAP.value,
                    "from_chain": source_token1_chain,
                    "to_chain": dest_chain,
                    "from_token": source_token1_address,
                    "from_token_symbol": source_token1_symbol,
                    "to_token": to_token,
                    "to_token_symbol": to_token_symbol,
                }
                bridge_swap_actions.append(bridge_swap_action)
        else:
            bridge_swap_action = {
                "action": Action.BRIDGE_SWAP.value,
                "from_chain": source_token0_chain,
                "to_chain": dest_chain,
                "from_token": source_token0_address,
                "from_token_symbol": source_token0_symbol,
                "to_token": dest_token0_address,
                "to_token_symbol": dest_token0_symbol,
            }
            bridge_swap_actions.append(bridge_swap_action)

            bridge_swap_action = {
                "action": Action.BRIDGE_SWAP.value,
                "from_chain": source_token1_chain,
                "to_chain": dest_chain,
                "from_token": source_token1_address,
                "from_token_symbol": source_token1_symbol,
                "to_token": dest_token1_address,
                "to_token_symbol": dest_token1_symbol,
            }
            bridge_swap_actions.append(bridge_swap_action)

        return bridge_swap_actions

    def _build_enter_pool_action(self) -> Dict[str, Any]:
        """Build action for entering the pool with the highest APR."""
        if not self.highest_apr_pool:
            self.context.logger.error("No pool present.")
            return None

        return {
            "action": Action.ENTER_POOL.value,
            "dex_type": self.highest_apr_pool.get("dex_type", ""),
            "chain": self.highest_apr_pool.get("chain", ""),
            "assets": [
                self.highest_apr_pool.get("token0", ""),
                self.highest_apr_pool.get("token1", ""),
            ],
            "pool_address": self.highest_apr_pool.get("pool_address", ""),
            "apr": self.highest_apr_pool.get("apr", 0),
        }

    def _get_asset_symbol(self, chain: str, address: str) -> Optional[str]:
        positions = self.synchronized_data.positions
        for position in positions:
            if position.get("chain", "") == chain:
                for asset in position["assets"]:
                    if asset.get("address", "") == address:
                        return asset.get("asset_symbol", "")

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

        last_executed_action_index = self.synchronized_data.last_executed_action_index
        current_action_index = (
            0 if last_executed_action_index is None else last_executed_action_index + 1
        )

        last_round_id = self.context.state.round_sequence._abci_app._previous_rounds[
            -1
        ].round_id

        # check tx status if last action was bridge and swap and the last round was not DecisionMaking or EvaluateStrategy
        if (
            last_round_id != DecisionMakingRound.auto_round_id()
            and last_round_id != EvaluateStrategyRound.auto_round_id()
            and Action(actions[last_executed_action_index].get("action", ""))
            == Action.BRIDGE_SWAP
        ):
            self.context.logger.info("Checking the status of swap tx")
            decision = yield from self.get_decision_on_swap()
            self.context.logger.info(f"Action to take {decision}")

            # If tx is pending then we wait until it gets confirmed or refunded
            if decision == Decision.WAIT:
                self.context.logger.info("Waiting for tx to get executed")
                while decision == Decision.WAIT:
                    # Wait for 5 seconds between each status check
                    yield from self.sleep(5)
                    self.context.logger.info("Checking the status of swap tx again")
                    decision = (
                        yield from self.get_decision_on_swap()
                    )  # Check the status again
                    self.context.logger.info(f"Action to take {decision}")

                if decision == Decision.EXIT:
                    self.context.logger.error("Swap failed")
                    return Event.DONE.value, {}

            elif decision == Decision.EXIT:
                self.context.logger.error("Swap failed")
                return Event.DONE.value, {}

            # if swap was successful we update the list of assets
            elif decision == Decision.CONTINUE:
                action = actions[last_executed_action_index]
                self.read_assets()
                current_assets = self.assets
                # add asset if it doesn't exist
                if not current_assets:
                    current_assets = self.params.initial_assets

                to_chain = action.get("to_chain", "")
                if to_chain not in current_assets:
                    current_assets[to_chain] = {}

                # Add the 'to_token' if the 'to_token_symbol' key doesn't exist
                to_token_symbol = action.get("to_token_symbol", "")
                if to_token_symbol and to_token_symbol not in current_assets[to_chain]:
                    current_assets[to_chain][to_token_symbol] = action.get(
                        "to_token", ""
                    )

                from_chain = action.get("from_chain", "")
                if from_chain not in current_assets:
                    current_assets[from_chain] = {}

                # Add the 'from_token' if the 'from_token_symbol' key doesn't exist
                from_token_symbol = action.get("from_token_symbol", "")
                if (
                    from_token_symbol
                    and from_token_symbol not in current_assets[from_chain]
                ):
                    current_assets[from_chain][from_token_symbol] = action.get(
                        "from_token", ""
                    )

                self.assets = current_assets
                self.store_assets()
                self.context.logger.info(
                    f"Bridge and swap was sucessful! Updating list of assets to {self.assets}"
                )

        # If last action was Enter Pool and it was successful we update the current pool
        if (
            last_executed_action_index is not None
            and Action(actions[last_executed_action_index].get("action", ""))
            == Action.ENTER_POOL
        ):
            action = actions[last_executed_action_index]                
            current_pool = {
                "chain": action["chain"],
                "address": action["pool_address"],
                "dex_type": action["dex_type"],
                "assets": action["assets"],
                "apr": action["apr"],
            }
            if action.get("dex_type","") == DexTypes.UNISWAP_V3.value:
                self.context.logger.info(f"{self.synchronized_data}")
                #TO-DO: Fetch token_id from transaction receipt
                current_pool["token_id"] = 1
            self.current_pool = current_pool
            self.store_current_pool()
            self.context.logger.info(
                f"Enter pool was successful! Updating current pool to {current_pool}"
            )

        # If last action was Exit Pool and it was successful we remove the current pool
        if (
            last_executed_action_index is not None
            and Action(actions[last_executed_action_index]["action"])
            == Action.EXIT_POOL
        ):
            current_pool = {}
            self.current_pool = current_pool
            self.store_current_pool()
            self.context.logger.info("Exit was successful! Removing current pool")

        # if all actions have been executed we exit DecisionMaking
        if current_action_index >= len(self.synchronized_data.actions):
            self.context.logger.info("All actions have been executed")
            return Event.DONE.value, {}

        positions = self.synchronized_data.positions

        # If the previous round was not EvaluateStrategyRound, we need to update the balances after a transaction
        if last_round_id != EvaluateStrategyRound.auto_round_id():
            positions = yield from self.get_positions()

        # Prepare the next action
        next_action = Action(actions[current_action_index]["action"])
        self.context.logger.info(f"ACTION TO BE PERFORMED: {next_action}")
        next_action_details = self.synchronized_data.actions[current_action_index]

        if next_action == Action.ENTER_POOL:
            tx_hash, chain_id, safe_address = yield from self.get_enter_pool_tx_hash(
                positions, next_action_details
            )

        elif next_action == Action.EXIT_POOL:
            tx_hash, chain_id, safe_address = yield from self.get_exit_pool_tx_hash(
                next_action_details
            )

        elif next_action == Action.BRIDGE_SWAP:
            tx_hash, chain_id, safe_address = yield from self.get_swap_tx_hash(
                positions, next_action_details
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
            # TO-DO: Decide on the correct method/logic for maintaining the period number for the last transaction.
            "last_executed_action_index": current_action_index,
        }

    def get_decision_on_swap(self) -> Generator[None, None, str]:
        """Get decision on swap"""
        # TO-DO: Add logic to handle other statuses as well. Specifically:
        # If a transaction fails, wait for it to be refunded.
        # If the transaction is still not confirmed and the round times out, implement logic to continue checking the status.

        try:
            tx_hash = self.synchronized_data.final_tx_hash
            self.context.logger.error(f"final tx hash {tx_hash}")
        except Exception:
            self.context.logger.error("No tx-hash found")
            return Decision.EXIT

        status, sub_status = yield from self.get_swap_status(tx_hash)
        if status is None or sub_status is None:
            return Decision.EXIT

        self.context.logger.info(
            f"SWAP STATUS - {status}, SWAP SUBSTATUS - {sub_status}"
        )

        if status == SwapStatus.DONE.value:
            # only continue if tx is fully completed
            if (
                sub_status == SwapDoneSubStaus.COMPLETED.value
                or sub_status == SwapDoneSubStaus.PARTIAL.value
            ):
                return Decision.CONTINUE
            # exit if it is refunded
            if sub_status == SwapDoneSubStaus.REFUNDED.value:
                return Decision.EXIT
        # wait if it is pending
        elif status == SwapStatus.PENDING.value:
            if (
                sub_status == SwapPendingSubStatus.WAIT_DESTINATION_TRANSACTION.value
                or sub_status == SwapPendingSubStatus.WAIT_SOURCE_CONFIRMATIONS.value
            ):
                return Decision.WAIT
        # exit if it fails
        else:
            return Decision.EXIT

    def get_swap_status(
        self, tx_hash: str
    ) -> Generator[None, None, Optional[Tuple[str, str]]]:
        """Fetch the status of tx"""

        url = f"{self.params.lifi_check_status_url}?txHash={tx_hash}"
        self.context.logger.info(f"checking status from endpoint {url}")

        response = yield from self.get_http_response(
            method="GET",
            url=url,
            headers={"accept": "application/json"},
        )

        if response.status_code != 200:
            self.context.logger.error(
                f"Received status code {response.status_code} from url {url}."
                f"Message {response.body}"
            )
            return None, None

        try:
            tx_status = json.loads(response.body)
        except (ValueError, TypeError) as e:
            self.context.logger.error(
                f"Could not parse response from api, "
                f"the following error was encountered {type(e).__name__}: {e}"
            )
            return None, None

        status = tx_status.get("status", "")
        sub_status = tx_status.get("substatus", "")

        if not status and sub_status:
            self.context.logger.error("No status or sub_status found in response")
            return None, None

        return status, sub_status

    def get_enter_pool_tx_hash(
        self, positions, action
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Get enter pool tx hash"""
        dex_type = action.get("dex_type", "")
        chain = action.get("chain", "")
        assets = action.get("assets", [])
        pool_address = action.get("pool_address", "")
        safe_address = self.params.safe_contract_addresses.get(
            action.get("chain", ""), ""
        )

        pool = self.pools.get(dex_type, None)
        if pool is None:
            self.context.logger.error(f"Unknown dex type: {dex_type}")
            return None, None, None

        # Fetch the amount of tokens to send
        max_amounts_in = [
            self._get_balance(chain, assets[0], positions),
            self._get_balance(chain, assets[1], positions),
        ]
        if any(amount == 0 or amount is None for amount in max_amounts_in):
            self.context.logger.error("Insufficient balance for entering pool")
            return None, None, None

        tx_hash, contract_address = yield from pool.enter(
            self,
            pool_address=pool_address,
            safe_address=safe_address,
            assets=assets,
            chain=chain,
            max_amounts_in=max_amounts_in,
        )
        if not tx_hash or not contract_address:
            return None, None, None

        multi_send_txs = []
        if not action["assets"][0] == ZERO_ADDRESS:
            # Approve asset 0
            token0_approval_tx_payload = yield from self.get_approval_tx_hash(
                token_address=action["assets"][0],
                amount=max_amounts_in[0],
                spender=contract_address,
                chain=chain,
            )

            if not token0_approval_tx_payload:
                self.context.logger.error("Error preparing approval tx payload")
                return None, None, None

            multi_send_txs.append(token0_approval_tx_payload)

        if not action["assets"][1] == ZERO_ADDRESS:
            # Approve asset 1
            token1_approval_tx_payload = yield from self.get_approval_tx_hash(
                token_address=action["assets"][1],
                amount=max_amounts_in[1],
                spender=contract_address,
                chain=chain,
            )
            if not token1_approval_tx_payload:
                self.context.logger.error("Error preparing approval tx payload")
                return None, None, None

            multi_send_txs.append(token1_approval_tx_payload)

        multi_send_txs.append(
            {
                "operation": MultiSendOperation.CALL,
                "to": contract_address,
                "value": 0,
                "data": tx_hash,
            }
        )

        # Get the transaction from the multisend contract
        multisend_address = self.params.multisend_contract_addresses[chain]

        multisend_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=multisend_address,
            contract_public_id=MultiSendContract.contract_id,
            contract_callable="get_tx_data",
            data_key="data",
            multi_send_txs=multi_send_txs,
            chain_id=chain,
        )

        self.context.logger.info(f"multisend_tx_hash = {multisend_tx_hash}")

        safe_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=safe_address,
            contract_public_id=GnosisSafeContract.contract_id,
            contract_callable="get_raw_safe_transaction_hash",
            data_key="tx_hash",
            to_address=multisend_address,
            value=ETHER_VALUE,
            data=bytes.fromhex(multisend_tx_hash[2:]),
            operation=SafeOperation.DELEGATE_CALL.value,
            safe_tx_gas=SAFE_TX_GAS,
            chain_id=chain,
        )

        if not safe_tx_hash:
            return None, None, None

        safe_tx_hash = safe_tx_hash[2:]
        self.context.logger.info(f"Hash of the Safe transaction: {safe_tx_hash}")

        payload_string = hash_payload_to_hex(
            safe_tx_hash=safe_tx_hash,
            ether_value=ETHER_VALUE,
            safe_tx_gas=SAFE_TX_GAS,
            operation=SafeOperation.DELEGATE_CALL.value,
            to_address=multisend_address,
            data=bytes.fromhex(multisend_tx_hash[2:]),
            gas_limit=self.params.manual_gas_limit,
        )

        self.context.logger.info(f"Tx hash payload string is {payload_string}")

        return payload_string, chain, safe_address

    def get_approval_tx_hash(
        self, token_address, amount: int, spender: str, chain: str
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get approve token tx hashes"""

        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=token_address,
            contract_public_id=ERC20.contract_id,
            contract_callable="build_approval_tx",
            data_key="data",
            spender=spender,
            amount=amount,
            chain_id=chain,
        )

        if not tx_hash:
            return {}

        return {
            "operation": MultiSendOperation.CALL,
            "to": token_address,
            "value": 0,
            "data": tx_hash,
        }

    def get_exit_pool_tx_hash(
        self, action
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Get enter pool tx hash for Balancer"""
        dex_type = action.get("dex_type", "")
        chain = action.get("chain", "")
        assets = action.get("assets", [])
        pool_address = action.get("pool_address", "")
        safe_address = self.params.safe_contract_addresses.get(
            action.get("chain", ""), ""
        )

        pool = self.pools.get(dex_type, None)
        if pool is None:
            self.context.logger.error(f"Unknown dex type: {dex_type}")
            return None, None, None

        # Get assets balances from positions
        safe_address = self.params.safe_contract_addresses[action["chain"]]

        tx_hash = yield from pool.exit(
            self,
            safe_address=safe_address,
            assets=assets,
            pool_address=pool_address,
            chain=chain,
        )

        if not tx_hash:
            return None, None, None

        vault_address = self.params.balancer_vault_contract_addresses.get(chain, "")
        if not vault_address:
            self.context.logger.error(f"No vault address found for chain {chain}")

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

    def get_swap_tx_hash(
        self, positions, action
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Get swap tx hash"""
        multi_send_txs = []
        chain = action["from_chain"]

        # Get swap tx
        (
            swap_tx_hash,
            lifi_contract_address,
            token_to_swap,
            amount,
        ) = yield from self.get_swap_tx_info(positions, action)

        if (
            not swap_tx_hash
            or not lifi_contract_address
            or not token_to_swap
            or not amount
        ):
            self.context.logger.error("Error fetching the swap related info")
            return None, None, None

        if not token_to_swap == ZERO_ADDRESS:
            approval_tx_payload = yield from self.get_approval_tx_hash(
                token_address=token_to_swap,
                amount=amount,
                spender=lifi_contract_address,
                chain=chain,
            )
            if not approval_tx_payload:
                self.context.logger.error("Error preparing approval tx payload")
                return None, None, None

            multi_send_txs.append(approval_tx_payload)

        multi_send_txs.append(
            {
                "operation": MultiSendOperation.CALL,
                "to": lifi_contract_address,
                "value": 0 if token_to_swap != ZERO_ADDRESS else amount,
                "data": swap_tx_hash,
            }
        )

        # Get the transaction from the multisend contract
        multisend_address = self.params.multisend_contract_addresses[chain]

        multisend_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=multisend_address,
            contract_public_id=MultiSendContract.contract_id,
            contract_callable="get_tx_data",
            data_key="data",
            multi_send_txs=multi_send_txs,
            chain_id=chain,
        )

        self.context.logger.info(f"multisend_tx_hash = {multisend_tx_hash}")

        safe_address = self.params.safe_contract_addresses[chain]

        safe_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=safe_address,
            contract_public_id=GnosisSafeContract.contract_id,
            contract_callable="get_raw_safe_transaction_hash",
            data_key="tx_hash",
            to_address=multisend_address,
            value=ETHER_VALUE,
            data=bytes.fromhex(multisend_tx_hash[2:]),
            operation=SafeOperation.DELEGATE_CALL.value,
            safe_tx_gas=SAFE_TX_GAS,
            chain_id=chain,
        )

        if not safe_tx_hash:
            return None, None, None

        safe_tx_hash = safe_tx_hash[2:]
        self.context.logger.info(f"Hash of the Safe transaction: {safe_tx_hash}")

        payload_string = hash_payload_to_hex(
            safe_tx_hash=safe_tx_hash,
            ether_value=ETHER_VALUE,
            safe_tx_gas=SAFE_TX_GAS,
            operation=SafeOperation.DELEGATE_CALL.value,
            to_address=multisend_address,
            data=bytes.fromhex(multisend_tx_hash[2:]),
            gas_limit=self.params.manual_gas_limit,
        )

        self.context.logger.info(f"Tx hash payload string is {payload_string}")

        return payload_string, chain, safe_address

    def get_swap_tx_info(
        self, positions, action
    ) -> Generator[None, None, Optional[Tuple[str, str, str, int]]]:
        """Get the quote for asset transfer from API"""
        chain_keys = self.params.chain_to_chain_key_mapping

        from_chain = chain_keys.get(action.get("from_chain", ""), "")
        to_chain = chain_keys.get(action.get("to_chain", ""), "")
        from_token = action.get("from_token", "")
        to_token = action.get("to_token", "")

        from_address = self.params.safe_contract_addresses.get(
            action.get("from_chain", ""), ""
        )
        if not from_address:
            self.context.logger.error(
                f"Could not find safe address for chain {from_chain}"
            )
            return None, None, None, None

        to_address = self.params.safe_contract_addresses.get(
            action.get("to_chain", ""), ""
        )
        if not to_address:
            self.context.logger.error(
                f"Could not find safe address for chain {to_chain}"
            )
            return None, None, None, None

        # TO-DO: Add logic to check if the amount is greater than the minimum required amount for a swap. Currently, we haven't set any such limit.
        amount = self._get_balance(
            action.get("from_chain", ""), action.get("from_token", ""), positions
        )
        if not amount or amount == 0:
            self.context.logger.error("Insufficient amount to swap")
            return None, None, None, None

        # TO-DO: add logic to dynamically adjust the value of slippage
        slippage = self.params.slippage_for_swap

        params = {
            "fromChain": from_chain,
            "toChain": to_chain,
            "fromToken": from_token,
            "toToken": to_token,
            "fromAddress": from_address,
            "toAddress": to_address,
            "fromAmount": amount,
            "slippage": slippage,
        }

        base_url = self.params.lifi_request_quote_url
        url = f"{base_url}?{urlencode(params)}"
        self.context.logger.info(f"URL :- {url}")

        # TO-DO: Add logic to handle scenarios when a route is not available for a swap.
        response = yield from self.get_http_response(
            method="GET",
            url=url,
            headers={"accept": "application/json"},
        )

        if response.status_code != 200:
            self.context.logger.error(
                f"Received status code {response.status_code} from url {url}."
                f"Message {response.body}"
            )
            return None, None, None, None

        try:
            quote = json.loads(response.body)
        except (ValueError, TypeError) as e:
            self.context.logger.error(
                f"Could not parse response from api, "
                f"the following error was encountered {type(e).__name__}: {e}"
            )
            return None, None, None, None

        tx_request = quote.get("transactionRequest", "")
        self.context.logger.info(f"transaction data from api {tx_request}")

        data = tx_request.get("data", "")
        if not tx_request or not data:
            self.context.logger.error(f"No tx data found in response {quote}")
            return None, None, None, None

        return bytes.fromhex(data[2:]), tx_request["to"], from_token, amount


class LiquidityTraderRoundBehaviour(AbstractRoundBehaviour):
    """LiquidityTraderRoundBehaviour"""

    initial_behaviour_cls = GetPositionsBehaviour
    abci_app_cls = LiquidityTraderAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = [
        GetPositionsBehaviour,
        EvaluateStrategyBehaviour,
        DecisionMakingBehaviour,
    ]

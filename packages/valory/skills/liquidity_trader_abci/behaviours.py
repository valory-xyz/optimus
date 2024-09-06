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
import math
import os.path
from abc import ABC
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Type, Union, cast
from urllib.parse import urlencode

from aea.configurations.data_types import PublicId
from eth_abi import decode
from eth_utils import keccak, to_bytes, to_hex
from web3 import Web3

from packages.valory.contracts.erc20.contract import ERC20
from packages.valory.contracts.gnosis_safe.contract import (
    GnosisSafeContract,
    SafeOperation,
)
from packages.valory.contracts.merkl_distributor.contract import DistributorContract
from packages.valory.contracts.multisend.contract import (
    MultiSendContract,
    MultiSendOperation,
)
from packages.valory.contracts.staking_activity_checker.contract import (
    StakingActivityCheckerContract,
)
from packages.valory.contracts.staking_token.contract import StakingTokenContract
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.models import Params, SharedState
from packages.valory.skills.liquidity_trader_abci.pools.balancer import (
    BalancerPoolBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.pools.uniswap import (
    UniswapPoolBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.rounds import (
    CallCheckpointPayload,
    CallCheckpointRound,
    CheckStakingKPIMetPayload,
    CheckStakingKPIMetRound,
    DecisionMakingPayload,
    DecisionMakingRound,
    EvaluateStrategyPayload,
    EvaluateStrategyRound,
    Event,
    GetPositionsPayload,
    GetPositionsRound,
    LiquidityTraderAbciApp,
    PostTxSettlementRound,
    StakingState,
    SynchronizedData,
)
from packages.valory.skills.liquidity_trader_abci.strategies.simple_strategy import (
    SimpleStrategyBehaviour,
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

    CLAIM_REWARDS = "ClaimRewards"
    EXIT_POOL = "ExitPool"
    ENTER_POOL = "EnterPool"
    BRIDGE_SWAP = "BridgeAndSwap"
    FIND_BRIDGE_ROUTE = "FindBridgeRoute"


class SwapStatus(Enum):
    """SwapStatus"""

    DONE = "DONE"
    PENDING = "PENDING"
    INVALID = "INVALID"
    NOT_FOUND = "NOT_FOUND"
    FAILED = "FAILED"


class Decision(Enum):
    """Decision"""

    CONTINUE = "continue"
    WAIT = "wait"
    EXIT = "exit"


ASSETS_FILENAME = "assets.json"
POOL_FILENAME = "current_pool.json"
READ_MODE = "r"
WRITE_MODE = "w"


class LiquidityTraderBaseBehaviour(
    BalancerPoolBehaviour, UniswapPoolBehaviour, SimpleStrategyBehaviour, ABC
):
    """Base behaviour for the liquidity_trader_abci skill."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize `LiquidityTraderBaseBehaviour`."""
        super().__init__(**kwargs)
        self.assets: Dict[str, Any] = {}
        # TO-DO: this will not work if we run it as a service
        self.assets_filepath = self.params.store_path / self.params.assets_info_filename
        self.current_pool: Dict[str, Any] = {}
        self.current_pool_filepath: str =  self.params.store_path / self.params.pool_info_filename
        self.pools: Dict[str, Any] = {}
        self.pools[DexTypes.BALANCER.value] = BalancerPoolBehaviour
        self.pools[DexTypes.UNISWAP_V3.value] = UniswapPoolBehaviour
        self.strategy = SimpleStrategyBehaviour
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

    @property
    def shared_state(self) -> SharedState:
        """Get the parameters."""
        return cast(SharedState, self.context.state)

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
            account = self.params.safe_contract_addresses.get(chain)
            if not account:
                self.context.logger.error(f"No safe address set for chain {chain}")
                continue

            for asset_address, asset_symbol in assets.items():
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
            if position.get("chain") == chain:
                for asset in position.get("assets", {}):
                    if asset.get("address") == token:
                        return asset.get("balance")

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
        except FileNotFoundError:
            # Create the file if it doesn't exist
            with open(filepath, WRITE_MODE) as file:
                json.dump({}, file)
            setattr(self, attribute, {})
            return
        except (PermissionError, OSError) as e:
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


class CallCheckpointBehaviour(
    LiquidityTraderBaseBehaviour
):  # pylint-disable too-many-ancestors
    matching_round = CallCheckpointRound

    def async_act(self) -> Generator:
        """Do the action."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            service_staking_state = yield from self._get_service_staking_state(chain="optimism")
            checkpoint_tx_hex = None
            min_num_of_safe_tx_required = None

            if service_staking_state == StakingState.STAKED:
                is_checkpoint_reached = yield from self._check_if_checkpoint_reached(
                    chain="optimism"
                )
                if is_checkpoint_reached:
                    self.context.logger.info(
                        "Checkpoint reached! Preparing checkpoint tx.."
                    )
                    checkpoint_tx_hex = yield from self._prepare_checkpoint_tx(
                        chain="optimism"
                    )

                    min_num_of_safe_tx_required = (
                        yield from self._calculate_min_num_of_safe_tx_required(
                            chain="optimism"
                        )
                    )
                    self.context.logger.info(
                        f"The minimum number of safe tx required to unlock rewards are {min_num_of_safe_tx_required}"
                    )

                    self.shared_state.last_checkpoint_executed_period_number = (
                        self.synchronized_data.period_count
                    )

            elif service_staking_state == StakingState.EVICTED:
                self.context.logger.error("Service has been evicted!")

            else:
                self.context.logger.error("Service has not been staked")

            tx_submitter = self.matching_round.auto_round_id()
            payload = CallCheckpointPayload(
                self.context.agent_address,
                tx_submitter,
                service_staking_state.value,
                min_num_of_safe_tx_required,
                checkpoint_tx_hex,
                safe_contract_address=self.params.safe_contract_addresses.get(
                    "optimism"
                ),
                chain_id="optimism",
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()

    def _calculate_min_num_of_safe_tx_required(
        self, chain: str
    ) -> Generator[None, None, Optional[int]]:
        """Calculates the minimun number of tx to hit to unlock the staking rewards"""
        liveness_ratio = yield from self._get_liveness_ratio(chain)
        liveness_period = yield from self._get_liveness_period(chain)

        if not liveness_ratio or not liveness_period:
            return None

        # Calculate the minimum number of transactions
        min_num_of_safe_tx_required = math.ceil(
            liveness_ratio * liveness_period // 10**18
        )

        return min_num_of_safe_tx_required

    def _get_liveness_ratio(self, chain: str) -> Generator[None, None, Optional[int]]:
        liveness_ratio = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=self.params.staking_activity_checker_contract_address,
            contract_public_id=StakingActivityCheckerContract.contract_id,
            contract_callable="liveness_ratio",
            data_key="data",
            chain_id=chain,
        )

        if liveness_ratio is None or liveness_ratio == 0:
            self.context.logger.error(
                f"Invalid value for liveness ratio: {liveness_ratio}"
            )

        return liveness_ratio

    def _get_liveness_period(self, chain: str) -> Generator[None, None, Optional[int]]:
        liveness_period = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=self.params.staking_token_contract_address,
            contract_public_id=StakingTokenContract.contract_id,
            contract_callable="get_liveness_period",
            data_key="data",
            chain_id=chain,
        )

        if liveness_period is None or liveness_period == 0:
            self.context.logger.error(
                f"Invalid value for liveness period: {liveness_period}"
            )

        return liveness_period

    def _get_service_staking_state(
        self, chain: str
    ) -> Generator[None, None, StakingState]:
        service_id = self.params.on_chain_service_id
        if service_id is None:
            self.context.logger.warning(
                "Cannot perform any staking-related operations without a configured on-chain service id. "
                "Assuming service status 'UNSTAKED'."
            )
            return StakingState.UNSTAKED

        service_staking_state = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=self.params.staking_token_contract_address,
            contract_public_id=StakingTokenContract.contract_id,
            contract_callable="get_service_staking_state",
            data_key="data",
            service_id=service_id,
            chain_id=chain,
        )
        if service_staking_state is None:
            self.context.logger.warning(
                "Error fetching staking state for service."
                "Assuming service status 'UNSTAKED'."
            )
            return StakingState.UNSTAKED

        return StakingState(service_staking_state)

    def _check_if_checkpoint_reached(
        self, chain: str
    ) -> Generator[None, None, Optional[bool]]:
        next_checkpoint = yield from self._get_next_checkpoint(chain)
        if next_checkpoint is None:
            return None

        if next_checkpoint == 0:
            return True

        synced_timestamp = int(
            self.round_sequence.last_round_transition_timestamp.timestamp()
        )
        return next_checkpoint <= synced_timestamp

    def _get_next_checkpoint(self, chain: str) -> Generator[None, None, Optional[int]]:
        """Get the timestamp in which the next checkpoint is reached."""
        next_checkpoint = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=self.params.staking_token_contract_address,
            contract_public_id=StakingTokenContract.contract_id,
            contract_callable="get_next_checkpoint_ts",
            data_key="data",
            chain_id=chain,
        )
        return next_checkpoint

    def _prepare_checkpoint_tx(
        self, chain: str
    ) -> Generator[None, None, Optional[str]]:
        checkpoint_data = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=self.params.staking_token_contract_address,
            contract_public_id=StakingTokenContract.contract_id,
            contract_callable="build_checkpoint_tx",
            data_key="data",
            chain_id=chain,
        )

        safe_tx_hash = yield from self._prepare_safe_tx(chain, data=checkpoint_data)

        return safe_tx_hash

    def _prepare_safe_tx(
        self, chain: str, data: bytes
    ) -> Generator[None, None, Optional[str]]:
        safe_address = self.params.safe_contract_addresses.get(chain)
        safe_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=safe_address,
            contract_public_id=GnosisSafeContract.contract_id,
            contract_callable="get_raw_safe_transaction_hash",
            data_key="tx_hash",
            data=data,
            to_address=self.params.staking_token_contract_address,
            value=ETHER_VALUE,
            safe_tx_gas=SAFE_TX_GAS,
            chain_id=chain,
        )

        if safe_tx_hash is None:
            return None

        safe_tx_hash = safe_tx_hash[2:]
        return hash_payload_to_hex(
            safe_tx_hash=safe_tx_hash,
            ether_value=ETHER_VALUE,
            safe_tx_gas=SAFE_TX_GAS,
            operation=SafeOperation.CALL.value,
            to_address=self.params.staking_token_contract_address,
            data=data,
        )


class CheckStakingKPIMetBehaviour(LiquidityTraderBaseBehaviour):
    # pylint-disable too-many-ancestors
    matching_round: Type[AbstractRound] = CheckStakingKPIMetRound

    def async_act(self) -> Generator:
        """Do the action."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            vanity_tx_hex = None
            kpi_met_for_the_day = self.synchronized_data.kpi_met_for_the_day

            if not kpi_met_for_the_day:
                last_round_id = (
                    self.context.state.round_sequence._abci_app._previous_rounds[
                        -1
                    ].round_id
                )
                
                is_post_tx_settlement_round = (
                    last_round_id == PostTxSettlementRound.auto_round_id()
                    and self.synchronized_data.tx_submitter
                    != CallCheckpointRound.auto_round_id()
                )
                is_period_threshold_exceeded = self._check_period_threshold_exceeded()

                if is_post_tx_settlement_round or is_period_threshold_exceeded:
                    min_num_of_safe_tx_required = (
                        self.synchronized_data.min_num_of_safe_tx_required
                    )
                    if min_num_of_safe_tx_required is None:
                        self.context.logger.info(
                            f"Invalid value for minimum number of safe tx: {min_num_of_safe_tx_required}"
                        )
                    else:
                        num_of_tx_left_to_meet_kpi = (
                            min_num_of_safe_tx_required
                            - self.synchronized_data.curr_num_of_safe_tx
                        )
                        if num_of_tx_left_to_meet_kpi > 0:
                            self.context.logger.info(
                                f"KPI not hit for the day, preparing vanity tx.."
                            )
                            vanity_tx_hex = yield from self._prepare_vanity_tx(
                                chain="optimism"
                            )
                            self.context.logger.info(f"tx hash: {vanity_tx_hex}")
                        else:
                            kpi_met_for_the_day = True
                            self.context.logger.info(f"KPI met for the day!")

            tx_submitter = self.matching_round.auto_round_id()
            payload = CheckStakingKPIMetPayload(
                self.context.agent_address,
                tx_submitter,
                kpi_met_for_the_day,
                vanity_tx_hex,
                safe_contract_address=self.params.safe_contract_addresses.get(
                    "optimism"
                ),
                chain_id="optimism",
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()

    def _check_period_threshold_exceeded(self) -> Generator[None, None, bool]:
        if not self.shared_state.last_checkpoint_executed_period_number:
            return False

        elapsed_periods = (
            self.synchronized_data.period_count
            - self.shared_state.last_checkpoint_executed_period_number
        )
        return elapsed_periods >= self.params.staking_threshold_period

    def _prepare_vanity_tx(self, chain: str) -> Generator[None, None, Optional[str]]:
        safe_address = self.params.safe_contract_addresses.get(chain)
        tx_data = b"0x"
        safe_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=safe_address,
            contract_public_id=GnosisSafeContract.contract_id,
            contract_callable="get_raw_safe_transaction_hash",
            data_key="tx_hash",
            to_address=ZERO_ADDRESS,
            value=ETHER_VALUE,
            data=tx_data,
            operation=SafeOperation.CALL.value,
            safe_tx_gas=SAFE_TX_GAS,
            chain_id=chain,
        )

        if safe_tx_hash is None:
            self.context.logger.info("Error preparing vanity tx")
            return None

        tx_hash = hash_payload_to_hex(
            safe_tx_hash=safe_tx_hash[2:],
            ether_value=ETHER_VALUE,
            safe_tx_gas=SAFE_TX_GAS,
            operation=SafeOperation.CALL.value,
            to_address=ZERO_ADDRESS,
            data=tx_data,
        )

        return tx_hash


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
                invest_in_pool = self.strategy.get_decision(
                    self, pool_apr=self.highest_apr_pool.get("apr")
                )
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

    def get_highest_apr_pool(self) -> Generator[None, None, None]:
        """Get highest APR pool"""
        filtered_pools = yield from self._get_filtered_pools()

        if not filtered_pools:
            self.context.logger.info("Could not find any eligible pool")
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

    def _extract_pool_info(
        self, dex_type, chain, apr, campaign
    ) -> Optional[Dict[str, Any]]:
        """Extract pool info from campaign data"""
        # TO-DO: Add support for pools with more than two tokens.
        pool_token_dict = {}
        pool_address = campaign.get("mainParameter")
        if not pool_address:
            self.context.logger.error(f"Missing pool address in campaign {campaign}")
            return None

        if dex_type == DexTypes.BALANCER.value:
            type_info = campaign.get("typeInfo", {})
            pool_tokens = type_info.get("poolTokens", {})
            # Extracting token0 and token1 with their symbols and addresses
            pool_token_items = list(pool_tokens.items())
            if len(pool_token_items) < 2 or any(
                token.get("symbol") is None or address is None
                for address, token in pool_token_items
            ):
                self.context.logger.error(
                    f"Invalid pool tokens found in campaign {pool_token_items}"
                )
                return None

            pool_token_dict = {
                "token0": pool_token_items[0][0],
                "token1": pool_token_items[1][0],
                "token0_symbol": pool_token_items[0][1].get("symbol"),
                "token1_symbol": pool_token_items[1][1].get("symbol"),
            }

        if dex_type == DexTypes.UNISWAP_V3.value:
            pool_info = campaign.get("campaignParameters", {})
            if not pool_info:
                self.context.logger.error(
                    f"No pool tokens info present in campaign {campaign}"
                )
                return None
            # Construct the dict for Uniswap V3 tokens with their symbols and addresses
            pool_token_dict = {
                "token0": pool_info.get("token0"),
                "token1": pool_info.get("token1"),
                "token0_symbol": pool_info.get("symbolToken0"),
                "token1_symbol": pool_info.get("symbolToken1"),
                "pool_fee": pool_info.get("poolFee"),
            }

        if any(v is None for v in pool_token_dict.values()):
            self.context.logger.error(
                f"Invalid pool tokens found in campaign {pool_token_dict}"
            )
            return None

        pool_data = {
            "dex_type": dex_type,
            "chain": chain,
            "apr": apr,
            "pool_address": pool_address,
        }
        pool_data.update(pool_token_dict)
        return pool_data

    def _get_filtered_pools(self) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get filtered pools"""

        filtered_pools = defaultdict(lambda: defaultdict(list))

        for chain in self.params.allowed_chains:
            chain_id = self.params.chain_to_chain_id_mapping.get(chain)
            base_url = self.params.merkl_fetch_campaigns_args.get("url")
            creator = self.params.merkl_fetch_campaigns_args.get("creator")
            live = self.params.merkl_fetch_campaigns_args.get("live", "true")

            params = {"chainIds": chain_id, "creatorTag": creator, "live": live}
            api_url = f"{base_url}?{urlencode(params)}"
            self.context.logger.info(f"Fetching campaigns from {api_url}")

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
                dex_type = (
                    campaign.get("type")
                    if campaign.get("type")
                    else campaign.get("ammName")
                )
                if not dex_type:
                    continue

                campaign_apr = campaign.get("apr")
                if not campaign_apr:
                    continue

                campaign_type = campaign.get("campaignType")
                if not campaign_type:
                    continue

                # The pool apr should be greater than the current pool apr
                if dex_type in allowed_dexs:
                    # type 1 and 2 stand for ERC20 and Concentrated liquidity campaigns respectively
                    # https://docs.merkl.xyz/integrate-merkl/integrate-merkl-to-your-app#merkl-api
                    if campaign_type in [1, 2]:
                        if not campaign_apr > self.current_pool.get("apr", 0.0):
                            self.context.logger.info(
                                "APR does not exceed the current pool APR"
                            )
                            continue
                        campaign_pool_address = campaign.get("mainParameter")
                        if not campaign_pool_address:
                            continue
                        current_pool_address = self.current_pool.get("address")
                        # The pool should not be the current pool
                        if campaign_pool_address != current_pool_address:
                            filtered_pools[dex_type][chain].append(campaign)

    def get_order_of_transactions(
        self,
    ) -> Generator[None, None, Optional[List[Dict[str, Any]]]]:
        """Get the order of transactions to perform based on the current pool status and token balances."""
        actions = []

        if self._can_claim_rewards():
            # check current reward
            allowed_chains = self.params.allowed_chains
            # we can claim all our token rewards at once
            # hence we build only one action per chain
            for chain in allowed_chains:
                chain_id = self.params.chain_to_chain_id_mapping.get(chain)
                safe_address = self.params.safe_contract_addresses.get(chain)
                rewards = yield from self._get_rewards(chain_id, safe_address)
                if not rewards:
                    continue
                action = self._build_claim_reward_action(rewards, chain)
                actions.append(action)

        if not self.current_pool:
            tokens = self._get_tokens_over_min_balance()
            if not tokens or len(tokens) < 2:
                self.context.logger.error(
                    f"Minimun 2 tokens required in safe with over minimum balance to enter a pool, provided: {tokens}"
                )
                return None
        else:
            # If there is current pool, then get the lp pool token addresses
            tokens = yield from self._get_exit_pool_tokens()  # noqa: E800
            if not tokens or len(tokens) < 2:
                self.context.logger.error(
                    f"2 tokens required to exit pool, provided: {tokens}"
                )

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

        return actions

    def _get_tokens_over_min_balance(self) -> Optional[List[Any]]:
        # ASSUMPTION : WE HAVE FUNDS FOR ATLEAST 2 TOKENS
        """Get tokens over min balance"""
        tokens = []
        highest_apr_chain = self.highest_apr_pool.get("chain")
        token0 = self.highest_apr_pool.get("token0")
        token1 = self.highest_apr_pool.get("token1")
        token0_symbol = self.highest_apr_pool.get("token0")
        token1_symbol = self.highest_apr_pool.get("token1")

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

        # TO-DO: set the value for gas_reserve for each chain
        # min_balance_threshold = (
        #     self.params.min_balance_multiplier
        #     * self.params.gas_reserve.get(highest_apr_chain, 0)
        # )
        min_balance_threshold = 0

        # Check balances for token0 and token1 on the highest APR pool chain
        for token, symbol in [(token0, token0_symbol), (token1, token1_symbol)]:
            balance = self._get_balance(highest_apr_chain, token)
            if balance and balance > min_balance_threshold:
                tokens.append(
                    {"chain": highest_apr_chain, "token": token, "token_symbol": symbol}
                )

        # We needs funds for atleast 2 tokens
        if len(tokens) == 2:
            return tokens

        seen_tokens = set((token.get("chain"), token.get("token")) for token in tokens)

        # If we still need more tokens, check all positions
        if len(tokens) < 2:
            token_balances = []
            for position in self.synchronized_data.positions:
                chain = position.get("chain")
                for asset in position.get("assets", {}):
                    asset_address = asset.get("address")
                    if not chain or not asset_address:
                        continue
                    if (chain, asset_address) not in seen_tokens:
                        if asset.get("asset_type") in ["erc_20", "native"]:
                            balance = asset.get("balance", 0)
                            # TO-DO: set the value for gas_reserve for each chain
                            min_balance = 0
                            if balance and balance > min_balance:
                                token_balances.append(
                                    {
                                        "chain": chain,
                                        "token": asset_address,
                                        "token_symbol": asset.get("asset_symbol"),
                                        "balance": balance,
                                    }
                                )

            # Sort tokens by balance in descending order and add the highest one
            token_balances.sort(key=lambda x: x["balance"], reverse=True)

            # TO:DO - Add another way to choose tokens because we can't rely on balance alone
            # (a.some tokens have 6 decimals  b.even though tokens have higher amount they might be less valuable)
            for token_data in token_balances:
                tokens.append(token_data)
                if len(tokens) == 2:
                    break

        return tokens

    def _get_exit_pool_tokens(self) -> Generator[None, None, Optional[List[Any]]]:
        """Get exit pool tokens"""

        if not self.current_pool:
            self.context.logger.error("No pool present")
            return None

        dex_type = self.current_pool.get("dex_type")
        pool_address = self.current_pool.get("address")
        chain = self.current_pool.get("chain")

        pool = self.pools.get(dex_type)
        if not pool:
            self.context.logger.error(f"Unknown dex type: {dex_type}")
            return None

        # Get tokens from balancer weighted pool contract
        tokens = yield from pool._get_tokens(self, pool_address, chain)
        if not tokens or any(v is None for v in tokens.items()):
            self.context.logger.error(f"Missing information in tokens: {tokens}")
            return None

        return [
            {
                "chain": chain,
                "token": tokens.get("token0"),
                "token_symbol": self._get_asset_symbol(chain, tokens.get("token0")),
            },
            {
                "chain": chain,
                "token": tokens.get("token1"),
                "token_symbol": self._get_asset_symbol(chain, tokens.get("token1")),
            },
        ]

    def _build_exit_pool_action(
        self, tokens: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Build action for exiting the current pool."""
        if not self.current_pool:
            self.context.logger.error("No pool present")
            return None

        if len(tokens) < 2:
            self.context.logger.error(
                f"Insufficient tokens provided for exit action. Required atleast 2, provided: {tokens}"
            )
            return None

        exit_pool_action = {
            "action": Action.EXIT_POOL.value,
            "dex_type": self.current_pool.get("dex_type"),
            "chain": self.current_pool.get("chain"),
            "assets": [tokens[0].get("token"), tokens[1].get("token")],
            "pool_address": self.current_pool.get("address"),
        }

        if exit_pool_action["dex_type"] == DexTypes.UNISWAP_V3.value:
            exit_pool_action["token_id"] = self.current_pool.get("token_id")
            exit_pool_action["liquidity"] = self.current_pool.get("liquidity")

        return exit_pool_action

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
        dest_token0_address = self.highest_apr_pool.get("token0")
        dest_token1_address = self.highest_apr_pool.get("token1")
        dest_token0_symbol = self.highest_apr_pool.get("token0_symbol")
        dest_token1_symbol = self.highest_apr_pool.get("token1_symbol")
        dest_chain = self.highest_apr_pool.get("chain")

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
            return None

        source_token0_chain = tokens[0].get("chain")
        source_token0_address = tokens[0].get("token")
        source_token0_symbol = tokens[0].get("token_symbol")
        source_token1_chain = tokens[1].get("chain")
        source_token1_address = tokens[1].get("token")
        source_token1_symbol = tokens[1].get("token_symbol")

        if (
            not source_token0_chain
            or not source_token0_address
            or not source_token0_symbol
            or not source_token1_chain
            or not source_token1_address
            or not source_token1_symbol
        ):
            self.context.logger.error(f"Incomplete data in tokens {tokens}")
            return None

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
                    "action": Action.FIND_BRIDGE_ROUTE.value,
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
                    "action": Action.FIND_BRIDGE_ROUTE.value,
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
                "action": Action.FIND_BRIDGE_ROUTE.value,
                "from_chain": source_token0_chain,
                "to_chain": dest_chain,
                "from_token": source_token0_address,
                "from_token_symbol": source_token0_symbol,
                "to_token": dest_token0_address,
                "to_token_symbol": dest_token0_symbol,
            }
            bridge_swap_actions.append(bridge_swap_action)

            bridge_swap_action = {
                "action": Action.FIND_BRIDGE_ROUTE.value,
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
            "dex_type": self.highest_apr_pool.get("dex_type"),
            "chain": self.highest_apr_pool.get("chain"),
            "assets": [
                self.highest_apr_pool.get("token0"),
                self.highest_apr_pool.get("token1"),
            ],
            "pool_address": self.highest_apr_pool.get("pool_address"),
            "apr": self.highest_apr_pool.get("apr"),
        }

    def _build_claim_reward_action(
        self, rewards: Dict[str, Any], chain: str
    ) -> Dict[str, Any]:
        return {
            "action": Action.CLAIM_REWARDS.value,
            "chain": chain,
            "users": rewards.get("users"),
            "tokens": rewards.get("tokens"),
            "claims": rewards.get("claims"),
            "proofs": rewards.get("proofs"),
            "token_symbols": rewards.get("symbols"),
        }

    def _get_asset_symbol(self, chain: str, address: str) -> Optional[str]:
        positions = self.synchronized_data.positions
        for position in positions:
            if position.get("chain") == chain:
                for asset in position.get("assets", {}):
                    if asset.get("address") == address:
                        return asset.get("asset_symbol")

        return None

    def _get_rewards(
        self, chain_id: int, user_address: str
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        base_url = self.params.merkl_user_rewards_url
        params = {"user": user_address, "chainId": chain_id, "proof": True}
        api_url = f"{base_url}?{urlencode(params)}"
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
            self.context.logger.info(f"User rewards: {data}")
            tokens = [k for k, v in data.items() if v.get("proof")]
            if not tokens:
                self.context.logger.info("No tokens to claim!")
                return None
            symbols = [data[t].get("symbol") for t in tokens]
            claims = [int(data[t].get("accumulated", 0)) for t in tokens]

            # Check if all claims are zero
            if all(claim == 0 for claim in claims):
                self.context.logger.info("All claims are zero, nothing to claim")
                return None

            unclaimed = [int(data[t].get("unclaimed", 0)) for t in tokens]
            # Check if everything has been already claimed are zero
            if all(claim == 0 for claim in unclaimed):
                self.context.logger.info(
                    "All accumulated claims already made. Nothing left to claim."
                )
                return None

            proofs = [data[t].get("proof") for t in tokens]
            return {
                "users": [user_address] * len(tokens),
                "tokens": tokens,
                "symbols": symbols,
                "claims": claims,
                "proofs": proofs,
            }
        except (ValueError, TypeError) as e:
            self.context.logger.error(
                f"Could not parse response from api, "
                f"the following error was encountered {type(e).__name__}: {e}"
            )
            return None

    def _can_claim_rewards(self) -> bool:
        # Check if rewards can be claimed. Rewards can be claimed if either:
        # 1. No rewards have been claimed yet (last_reward_claimed_timestamp is None), or
        # 2. The current timestamp exceeds the allowed reward claiming time period since the last claim.

        current_timestamp = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()

        last_claimed_timestamp = self.synchronized_data.last_reward_claimed_timestamp
        if last_claimed_timestamp is None:
            return True
        return (
            current_timestamp
            >= last_claimed_timestamp + self.params.reward_claiming_time_period
        )


class DecisionMakingBehaviour(LiquidityTraderBaseBehaviour):
    """DecisionMakingBehaviour"""

    matching_round: Type[AbstractRound] = DecisionMakingRound

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            (
                next_event,
                updates,
                bridge_and_swap_actions,
            ) = yield from self.get_next_event()

            payload = DecisionMakingPayload(
                sender=sender,
                content=json.dumps(
                    {
                        "event": next_event,
                        "updates": updates,
                        "bridge_and_swap_actions": bridge_and_swap_actions,
                    },
                    sort_keys=True,
                ),
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_next_event(self) -> Generator[None, None, Tuple[str, Dict, Optional[Dict]]]:
        """Get next event"""

        actions = self.synchronized_data.actions
        # If there are no actions, we return
        if not actions:
            self.context.logger.info("No actions to prepare")
            return Event.DONE.value, {}, {}

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
            and Action(actions[last_executed_action_index].get("action"))
            == Action.BRIDGE_SWAP
        ):
            self.context.logger.info("Checking the status of swap tx")
            decision = yield from self.get_decision_on_swap()
            self.context.logger.info(f"Action to take {decision}")

            # If tx is pending then we wait until it gets confirmed or refunded
            if decision == Decision.WAIT:
                self.context.logger.info("Waiting for tx to get executed")
                while decision == Decision.WAIT:
                    # Wait for given time between each status check
                    yield from self.sleep(self.params.waiting_period_for_retry)
                    self.context.logger.info("Checking the status of swap tx again")
                    decision = (
                        yield from self.get_decision_on_swap()
                    )  # Check the status again
                    self.context.logger.info(f"Action to take {decision}")

            if decision == Decision.EXIT:
                self.context.logger.error("Swap failed")
                return Event.DONE.value, {}, {}

            # if swap was successful we update the list of assets
            if decision == Decision.CONTINUE:
                action = actions[last_executed_action_index]
                self._add_token_to_assets(
                    action.get("from_chain"),
                    action.get("from_token"),
                    action.get("from_token_symbol"),
                )
                self._add_token_to_assets(
                    action.get("to_chain"),
                    action.get("to_token"),
                    action.get("to_token_symbol"),
                )

        # If last action was Enter Pool and it was successful we update the current pool
        if (
            last_executed_action_index is not None
            and Action(actions[last_executed_action_index].get("action"))
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
            if action.get("dex_type") == DexTypes.UNISWAP_V3.value:
                token_id, liquidity = yield from self._get_data_from_mint_tx_receipt(
                    self.synchronized_data.final_tx_hash, action.get("chain")
                )
                current_pool["token_id"] = token_id
                current_pool["liquidity"] = liquidity
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

        # If last action was Claim Rewards and it was successful we update the list of assets and the last_reward_claimed_timestamp
        if (
            last_executed_action_index is not None
            and last_round_id != DecisionMakingRound.auto_round_id()
            and Action(actions[last_executed_action_index]["action"])
            == Action.CLAIM_REWARDS
        ):
            action = actions[last_executed_action_index]
            chain = action.get("chain")
            for token, token_symbol in zip(
                action.get("tokens"), action.get("token_symbols")
            ):
                self._add_token_to_assets(chain, token, token_symbol)

            current_timestamp = cast(
                SharedState, self.context.state
            ).round_sequence.last_round_transition_timestamp.timestamp()

            return (
                Event.UPDATE.value,
                {"last_reward_claimed_timestamp": current_timestamp},
                {},
            )

        # if all actions have been executed we exit DecisionMaking
        if current_action_index >= len(self.synchronized_data.actions):
            self.context.logger.info("All actions have been executed")
            return Event.DONE.value, {}, {}

        positions = self.synchronized_data.positions

        # If the previous round was not EvaluateStrategyRound, we need to update the balances after a transaction
        if last_round_id != EvaluateStrategyRound.auto_round_id():
            positions = yield from self.get_positions()

        # Prepare the next action
        next_action = Action(actions[current_action_index].get("action"))
        next_action_details = self.synchronized_data.actions[current_action_index]
        self.context.logger.info(f"ACTION DETAILS: {next_action_details}")

        bridge_and_swap_actions = {}
        if next_action == Action.ENTER_POOL:
            tx_hash, chain_id, safe_address = yield from self.get_enter_pool_tx_hash(
                positions, next_action_details
            )

        elif next_action == Action.EXIT_POOL:
            tx_hash, chain_id, safe_address = yield from self.get_exit_pool_tx_hash(
                next_action_details
            )

        elif next_action == Action.FIND_BRIDGE_ROUTE:
            bridge_and_swap_actions = yield from self.get_transaction_data_for_route(
                positions, next_action_details
            )
            if not bridge_and_swap_actions or not bridge_and_swap_actions.get(
                "actions"
            ):
                return Event.DONE.value, {}, {}

            return (
                Event.UPDATE.value,
                {"last_executed_action_index": current_action_index},
                bridge_and_swap_actions,
            )

        elif next_action == Action.BRIDGE_SWAP:
            # wait for sometime to get the balances reflected
            yield from self.sleep(5)
            tx_hash = next_action_details.get("payload")
            chain_id = next_action_details.get("from_chain")
            safe_address = next_action_details.get("safe_address")

        elif next_action == Action.CLAIM_REWARDS:
            tx_hash, chain_id, safe_address = yield from self.get_claim_rewards_tx_hash(
                next_action_details
            )

        else:
            tx_hash = None
            chain_id = None
            safe_address = None

        if not tx_hash:
            self.context.logger.error("There was an error preparing the next action")
            return Event.DONE.value, {}, {}

        return (
            Event.SETTLE.value,
            {
                "tx_submitter": DecisionMakingRound.auto_round_id(),
                "most_voted_tx_hash": tx_hash,
                "chain_id": chain_id,
                "safe_contract_address": safe_address,
                "positions": positions,
                # TO-DO: Decide on the correct method/logic for maintaining the period number for the last transaction.
                "last_executed_action_index": current_action_index,
            },
            {},
        )

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
            return Decision.CONTINUE
        # wait if it is pending
        elif status == SwapStatus.PENDING.value:
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

        status = tx_status.get("status")
        sub_status = tx_status.get("substatus")

        if not status and sub_status:
            self.context.logger.error("No status or sub_status found in response")
            return None, None

        return status, sub_status

    def get_enter_pool_tx_hash(
        self, positions, action
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Get enter pool tx hash"""
        dex_type = action.get("dex_type")
        chain = action.get("chain")
        assets = action.get("assets", {})
        if not assets or len(assets) < 2:
            self.context.logger.error(f"2 assets required, provided: {assets}")
            return None, None, None
        pool_address = action.get("pool_address")
        pool_fee = action.get("pool_fee")
        safe_address = self.params.safe_contract_addresses.get(action.get("chain"))

        pool = self.pools.get(dex_type)
        if not pool:
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
            pool_fee=pool_fee,
        )
        if not tx_hash or not contract_address:
            return None, None, None

        multi_send_txs = []
        value = 0
        if not assets[0] == ZERO_ADDRESS:
            # Approve asset 0
            token0_approval_tx_payload = yield from self.get_approval_tx_hash(
                token_address=assets[0],
                amount=max_amounts_in[0],
                spender=contract_address,
                chain=chain,
            )

            if not token0_approval_tx_payload:
                self.context.logger.error("Error preparing approval tx payload")
                return None, None, None

            multi_send_txs.append(token0_approval_tx_payload)
        else:
            value = max_amounts_in[0]

        if not assets[1] == ZERO_ADDRESS:
            # Approve asset 1
            token1_approval_tx_payload = yield from self.get_approval_tx_hash(
                token_address=assets[1],
                amount=max_amounts_in[1],
                spender=contract_address,
                chain=chain,
            )
            if not token1_approval_tx_payload:
                self.context.logger.error("Error preparing approval tx payload")
                return None, None, None

            multi_send_txs.append(token1_approval_tx_payload)
        else:
            value = max_amounts_in[1]

        multi_send_txs.append(
            {
                "operation": MultiSendOperation.CALL,
                "to": contract_address,
                "value": value,
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
        """Get exit pool tx hash"""
        dex_type = action.get("dex_type")
        chain = action.get("chain")
        assets = action.get("assets", {})
        if not assets or len(assets) < 2:
            self.context.logger.error(f"2 assets required, provided: {assets}")
            return None, None, None
        pool_address = action.get("pool_address")
        token_id = action.get("token_id")
        liquidity = action.get("liquidity")
        safe_address = self.params.safe_contract_addresses.get(action.get("chain"))

        pool = self.pools.get(dex_type)
        if not pool:
            self.context.logger.error(f"Unknown dex type: {dex_type}")
            return None, None, None

        exit_pool_kwargs = {}

        if dex_type == DexTypes.BALANCER.value:
            exit_pool_kwargs.update(
                {
                    "safe_address": safe_address,
                    "assets": assets,
                    "pool_address": pool_address,
                    "chain": chain,
                }
            )

        if dex_type == DexTypes.UNISWAP_V3.value:
            exit_pool_kwargs.update(
                {
                    "token_id": token_id,
                    "safe_address": safe_address,
                    "chain": chain,
                    "liquidity": liquidity,
                }
            )

        if not exit_pool_kwargs:
            self.context.logger.error("Could not find kwargs for exit pool")
            return None, None, None

        tx_hash, contract_address, is_multisend = yield from pool.exit(
            self, **exit_pool_kwargs
        )
        if not tx_hash or not contract_address:
            return None, None, None

        safe_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=safe_address,
            contract_public_id=GnosisSafeContract.contract_id,
            contract_callable="get_raw_safe_transaction_hash",
            data_key="tx_hash",
            to_address=contract_address,
            value=ETHER_VALUE,
            data=tx_hash,
            operation=SafeOperation.DELEGATE_CALL.value
            if is_multisend
            else SafeOperation.CALL.value,
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
            operation=SafeOperation.DELEGATE_CALL.value
            if is_multisend
            else SafeOperation.CALL.value,
            to_address=contract_address,
            data=tx_hash,
            gas_limit=self.params.manual_gas_limit,
        )

        self.context.logger.info(f"Tx hash payload string is {payload_string}")

        return payload_string, chain, safe_address

    def get_transaction_data_for_route(
        self, positions, action
    ) -> Generator[None, None, Dict]:
        bridge_and_swap_actions = {"actions": []}
        from_chain = action.get("from_chain")
        to_chain = action.get("to_chain")
        from_chain_id = self.params.chain_to_chain_id_mapping.get(from_chain)
        to_chain_id = self.params.chain_to_chain_id_mapping.get(to_chain)
        from_token_address = action.get("from_token")
        to_token_address = action.get("to_token")
        from_token_symbol = action.get("from_token_symbol")
        to_token_symbol = action.get("to_token_symbol")
        allow_switch_chain = True
        slippage = self.params.slippage_for_swap
        amount = self._get_balance(from_chain, from_token_address, positions)
        from_address = self.params.safe_contract_addresses.get(from_chain)
        to_address = self.params.safe_contract_addresses.get(to_chain)

        # TO:DO - Add logic to maintain a list of blacklisted bridges
        params = {
            "fromAddress": from_address,
            "toAddress": to_address,
            "fromChainId": from_chain_id,
            "fromAmount": amount,
            "fromTokenAddress": from_token_address,
            "toChainId": to_chain_id,
            "toTokenAddress": to_token_address,
            "options": {
                "slippage": slippage,
                "allowSwitchChain": allow_switch_chain,
                "integrator": "valory",
                "bridges": {"deny": ["stargateV2Bus"]},
            },
        }

        if any(value is None for key, value in params.items()):
            self.context.logger.error(f"Missing value in params: {params}")
            return {}

        self.context.logger.info(
            f"Finding route: {from_token_symbol}({from_chain}) --> {to_token_symbol}({to_chain})"
        )

        url = self.params.lifi_advance_routes_url
        routes_response = yield from self.get_http_response(
            "POST",
            url,
            json.dumps(params).encode(),
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",  # Ensure the correct content type
            },
        )

        if routes_response.status_code != 200:
            response = json.loads(routes_response.body)
            self.context.logger.error(f"Error encountered: {response['message']}")
            return {}

        try:
            routes_response = json.loads(routes_response.body)
        except (ValueError, TypeError) as e:
            self.context.logger.error(
                f"Could not parse response from api, "
                f"the following error was encountered {type(e).__name__}: {e}"
            )
            return {}

        routes = routes_response.get("routes", [])
        for route in routes:
            steps = route.get("steps", {})
            all_steps_successful = True
            for step in steps:
                from_chain_id = step.get("action", {}).get("fromChainId")
                from_chain = next(
                    (
                        k
                        for k, v in self.params.chain_to_chain_id_mapping.items()
                        if v == from_chain_id
                    ),
                    None,
                )
                to_chain_id = step.get("action", {}).get("toChainId")
                to_chain = next(
                    (
                        k
                        for k, v in self.params.chain_to_chain_id_mapping.items()
                        if v == to_chain_id
                    ),
                    None,
                )
                step["action"]["fromAddress"] = self.params.safe_contract_addresses.get(
                    from_chain
                )
                step["action"]["toAddress"] = self.params.safe_contract_addresses.get(
                    to_chain
                )
                from_token_symbol = (
                    step.get("action", {}).get("fromToken", {}).get("symbol")
                )
                to_token_symbol = (
                    step.get("action", {}).get("toToken", {}).get("symbol")
                )
                tool = step.get("tool")

                self.context.logger.info(
                    f"TX: {from_token_symbol}({from_chain}) --> {to_token_symbol}({to_chain}). Tool being used: {tool}"
                )

                tx_info = yield from self.get_step_transaction(step)

                if not tx_info or any(value is None for value in params.values()):
                    self.context.logger.error(f"Missing value in params: {params}")
                    all_steps_successful = False
                    break

                multisend_txs = []

                if tx_info.get("source_token") != ZERO_ADDRESS:
                    approval_tx_payload = yield from self.get_approval_tx_hash(
                        token_address=tx_info.get("source_token"),
                        amount=tx_info.get("amount"),
                        spender=tx_info.get("lifi_contract_address"),
                        chain=tx_info.get("from_chain"),
                    )
                    if not approval_tx_payload:
                        self.context.logger.error("Error preparing approval tx payload")
                        all_steps_successful = False
                        break

                    multisend_txs.append(approval_tx_payload)

                multisend_txs.append(
                    {
                        "operation": MultiSendOperation.CALL,
                        "to": tx_info.get("lifi_contract_address"),
                        "value": 0
                        if tx_info.get("source_token") != ZERO_ADDRESS
                        else tx_info.get("amount"),
                        "data": tx_info.get("tx_hash"),
                    }
                )

                multisend_address = self.params.multisend_contract_addresses[
                    tx_info.get("from_chain")
                ]

                multisend_tx_hash = yield from self.contract_interact(
                    performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
                    contract_address=multisend_address,
                    contract_public_id=MultiSendContract.contract_id,
                    contract_callable="get_tx_data",
                    data_key="data",
                    multi_send_txs=multisend_txs,
                    chain_id=tx_info.get("from_chain"),
                )

                data = bytes.fromhex(multisend_tx_hash[2:])
                liquidity_provider = (
                    self.params.intermediate_tokens.get(tx_info.get("from_chain"), {})
                    .get(tx_info.get("source_token"), {})
                    .get("liquidity_provider")
                )
                # if we do not have an address from where we can perform a mock transfer we do not perform the simulation and proceed to the next steps
                if liquidity_provider:
                    is_ok = yield from self._simulate_execution_bundle(
                        to_address=multisend_address,
                        data=data,
                        token=tx_info.get("source_token"),
                        mock_transfer_from=liquidity_provider,
                        amount=tx_info.get("amount"),
                        chain=tx_info.get("from_chain"),
                    )
                    if not is_ok:
                        self.context.logger.info(
                            f"Simulation failed for bridge/swap tx: {tx_info.get('source_token_symbol')}({tx_info.get('from_chain')}) --> {tx_info.get('target_token_symbol')}({tx_info.get('to_chain')}). Tool used: {tx_info.get('tool')}"
                        )
                        all_steps_successful = False
                        break
                    self.context.logger.info(
                        f"Simulation successful for bridge/swap tx: {tx_info.get('source_token_symbol')}({tx_info.get('from_chain')}) --> {tx_info.get('target_token_symbol')}({tx_info.get('to_chain')}). Tool used: {tx_info.get('tool')}"
                    )

                safe_address = self.params.safe_contract_addresses.get(
                    tx_info.get("from_chain")
                )
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
                    chain_id=tx_info.get("from_chain"),
                )

                if not safe_tx_hash:
                    self.context.logger.error(f"Error preparing safe tx!")
                    all_steps_successful = False
                    break

                safe_tx_hash = safe_tx_hash[2:]
                tx_params = dict(
                    ether_value=ETHER_VALUE,
                    safe_tx_gas=SAFE_TX_GAS,
                    operation=SafeOperation.DELEGATE_CALL.value,
                    to_address=multisend_address,
                    data=bytes.fromhex(multisend_tx_hash[2:]),
                    safe_tx_hash=safe_tx_hash,
                    gas_limit=self.params.manual_gas_limit,
                )
                payload_string = hash_payload_to_hex(**tx_params)

                bridge_and_swap_actions["actions"].append(
                    {
                        "action": Action.BRIDGE_SWAP.value,
                        "from_chain": tx_info.get("from_chain"),
                        "to_chain": tx_info.get("to_chain"),
                        "from_token": tx_info.get("source_token"),
                        "from_token_symbol": tx_info.get("source_token_symbol"),
                        "to_token": tx_info.get("target_token"),
                        "to_token_symbol": tx_info.get("target_token_symbol"),
                        "payload": payload_string,
                        "safe_address": safe_address,
                    }
                )

            if all_steps_successful:
                self.context.logger.info(
                    f"BRIDGE SWAP ACTIONS: {bridge_and_swap_actions}"
                )
                return bridge_and_swap_actions

        self.context.logger.error("NONE OF THE ROUTES WERE SUCCESFUL!")
        return {}

    def get_step_transaction(
        self, step: Dict[str, Any]
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        base_url = "https://li.quest/v1/advanced/stepTransaction"
        response = yield from self.get_http_response(
            "POST",
            base_url,
            json.dumps(step).encode(),
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",  # Ensure the correct content type
            },
        )

        if response.status_code != 200:
            response = json.loads(response.body)
            self.context.logger.error(f"Error encountered: {response['message']}")
            return None

        try:
            response = json.loads(response.body)
        except (ValueError, TypeError) as e:
            self.context.logger.error(
                f"Could not parse response from api, "
                f"the following error was encountered {type(e).__name__}: {e}"
            )
            return None

        source_token = response.get("action", {}).get("fromToken", {}).get("address")
        source_token_symbol = (
            response.get("action", {}).get("fromToken", {}).get("symbol")
        )
        target_token = response.get("action", {}).get("toToken", {}).get("address")
        target_token_symbol = (
            response.get("action", {}).get("toToken", {}).get("symbol")
        )
        amount = int(response.get("estimate", {}).get("fromAmount", {}))
        lifi_contract_address = response.get("transactionRequest", {}).get("to")
        from_chain_id = response.get("action", {}).get("fromChainId")
        from_chain = next(
            (
                k
                for k, v in self.params.chain_to_chain_id_mapping.items()
                if v == from_chain_id
            ),
            None,
        )
        to_chain_id = response.get("action", {}).get("toChainId")
        to_chain = next(
            (
                k
                for k, v in self.params.chain_to_chain_id_mapping.items()
                if v == to_chain_id
            ),
            None,
        )
        tool = response.get("tool")
        data = response.get("transactionRequest", {}).get("data")
        tx_hash = bytes.fromhex(data[2:])

        return {
            "source_token": source_token,
            "source_token_symbol": source_token_symbol,
            "target_token": target_token,
            "target_token_symbol": target_token_symbol,
            "amount": amount,
            "lifi_contract_address": lifi_contract_address,
            "from_chain": from_chain,
            "to_chain": to_chain,
            "tool": tool,
            "data": data,
            "tx_hash": tx_hash,
        }

    def _simulate_execution_bundle(
        self,
        to_address: str,
        data: bytes,
        token: str,
        mock_transfer_from: str,
        amount: int,
        chain: str,
        **kwargs: Any,
    ) -> Generator[None, None, bool]:
        safe_address = self.params.safe_contract_addresses.get(chain)
        agent_address = self.context.agent_address
        safe_tx = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=safe_address,
            contract_id=str(GnosisSafeContract.contract_id),
            contract_callable="get_raw_safe_transaction",
            sender_address=agent_address,
            owners=(agent_address,),
            to_address=to_address,
            value=ETHER_VALUE,
            data=data,
            safe_tx_gas=SAFE_TX_GAS,
            signatures_by_owner={agent_address: self._get_signature(agent_address)},
            operation=SafeOperation.DELEGATE_CALL.value,
            chain_id=chain,
        )

        tx_data = safe_tx.raw_transaction.body["data"]

        url_template = self.params.tenderly_bundle_simulation_url
        values = {
            "tenderly_account_slug": self.params.tenderly_account_slug,
            "tenderly_project_slug": self.params.tenderly_project_slug,
        }
        api_url = url_template.format(**values)

        transfer_data = self._encode_transfer_data(token, safe_address, amount)
        body = {
            "simulations": [
                {
                    "network_id": self.params.chain_to_chain_id_mapping.get(chain),
                    "save": True,
                    "save_if_fails": True,
                    "simulation_type": "quick",
                    "from": mock_transfer_from,
                    "to": token,
                    "input": transfer_data,
                },
                {
                    "network_id": self.params.chain_to_chain_id_mapping.get(chain),
                    "save": True,
                    "save_if_fails": True,
                    "simulation_type": "quick",
                    "from": self.context.agent_address,
                    "to": safe_address,
                    "input": tx_data,
                },
            ]
        }

        response = yield from self.get_http_response(
            "POST",
            api_url,
            json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Access-Key": self.params.tenderly_access_key,
            },
        )
        if response.status_code != 200:
            self.context.logger.error(
                f"Could not retrieve data from url {api_url}. Status code {response.status_code}."
            )
            return False

        try:
            data = json.loads(response.body)
            if data:
                simulation_results = data.get("simulation_results", [])
                status = False
                if simulation_results:
                    simulation_results = simulation_results[0]
                    for simulation in simulation_results.values():
                        if isinstance(simulation, Dict):
                            status = simulation.get("status", False)
                return status

        except (ValueError, TypeError) as e:
            self.context.logger.error(
                f"Could not parse response from api, "
                f"the following error was encountered {type(e).__name__}: {e}"
            )
            return False

    def get_claim_rewards_tx_hash(
        self, action
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Get claim rewards tx hash"""
        chain = action.get("chain")
        users = action.get("users")
        tokens = action.get("tokens")
        amounts = action.get("claims")
        proofs = action.get("proofs")

        if not tokens or not amounts or not proofs:
            self.context.logger.error(f"Missing information in action : {action}")
            return None, None, None

        safe_address = self.params.safe_contract_addresses.get(action.get("chain"))
        contract_address = self.params.merkl_distributor_contract_addresses.get(chain)
        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=contract_address,
            contract_public_id=DistributorContract.contract_id,
            contract_callable="claim_rewards",
            data_key="tx_hash",
            users=users,
            tokens=tokens,
            amounts=amounts,
            proofs=proofs,
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
            value=ETHER_VALUE,
            data=tx_hash,
            to_address=contract_address,
            operation=SafeOperation.CALL.value,
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
            operation=SafeOperation.CALL.value,
            to_address=contract_address,
            data=tx_hash,
            gas_limit=self.params.manual_gas_limit,
        )

        self.context.logger.info(f"Tx hash payload string is {payload_string}")

        return payload_string, chain, safe_address

    def _encode_transfer_data(self, token: str, to_address: str, amount: int) -> str:
        transfer_data = (
            Web3()
            .eth.contract(
                address=token,
                abi=[
                    {
                        "constant": False,
                        "inputs": [
                            {"name": "to", "type": "address"},
                            {"name": "value", "type": "uint256"},
                        ],
                        "name": "transfer",
                        "outputs": [{"name": "", "type": "bool"}],
                        "payable": False,
                        "stateMutability": "nonpayable",
                        "type": "function",
                    }
                ],
            )
            .encodeABI(fn_name="transfer", args=[to_address, amount])
        )

        return transfer_data

    def _get_data_from_mint_tx_receipt(
        self, tx_hash: str, chain: str
    ) -> Generator[None, None, Optional[Tuple[int, int]]]:
        response = yield from self.get_transaction_receipt(
            tx_hash,
            chain_id=chain,
        )
        if not response:
            self.context.logger.error(
                f"Error fetching tx receipt! Response: {response}"
            )
            return None, None

        # Define the event signature and calculate its hash
        event_signature = "IncreaseLiquidity(uint256,uint128,uint256,uint256)"
        event_signature_hash = keccak(text=event_signature)
        event_signature_hex = to_hex(event_signature_hash)[2:]

        # Extract logs from the response
        logs = response.get("logs", [])

        # Find the log that matches the IncreaseLiquidity event
        log = next(
            (
                log
                for log in logs
                if log.get("topics", [])[0][2:] == event_signature_hex
            ),
            None,
        )

        if not log:
            self.context.logger.error("No logs found for IncreaseLiquidity event")
            return None, None

        # Decode indexed parameter (tokenId)
        try:
            # Decode indexed parameter (tokenId)
            token_id_topic = log.get("topics", [])[1]
            if not token_id_topic:
                self.context.logger.error(f"Token ID topic is missing from log {log}")
                return None, None
            # Convert hex to bytes and decode
            token_id_bytes = bytes.fromhex(token_id_topic[2:])
            token_id = decode(["uint256"], token_id_bytes)[0]

            # Decode non-indexed parameters (liquidity, amount0, amount1) from the data field
            data_hex = log.get("data")
            if not data_hex:
                self.context.logger.error(f"Data field is empty in log {log}")
                return None, None

            data_bytes = bytes.fromhex(data_hex[2:])
            decoded_data = decode(["uint128", "uint256", "uint256"], data_bytes)
            liquidity = decoded_data[0]

            self.context.logger.info(f"tokenId returned from mint function: {token_id}")
            self.context.logger.info(
                f"liquditiy returned from mint function: {liquidity}"
            )

            return token_id, liquidity

        except Exception as e:
            self.context.logger.error(f"Error decoding token ID: {e}")
            return None, None

    def _add_token_to_assets(self, chain, token, symbol):
        # Read current assets
        self.read_assets()
        current_assets = self.assets

        # Initialize assets if empty
        if not current_assets:
            current_assets = self.params.initial_assets

        # Ensure the chain key exists in assets
        if chain not in current_assets:
            current_assets[chain] = {}

        # Add token to the specified chain if it doesn't exist
        if token not in current_assets[chain]:
            current_assets[chain][token] = symbol

        # Store updated assets
        self.assets = current_assets
        self.store_assets()

        self.context.logger.info(f"Updated assets: {self.assets}")

    def _get_signature(self, owner: str) -> str:
        signatures = b""
        # Convert address to bytes and ensure it is 32 bytes long (left-padded with zeros)
        r_bytes = to_bytes(hexstr=owner[2:].rjust(64, "0"))

        # `s` as 32 zero bytes
        s_bytes = b"\x00" * 32

        # `v` as a single byte
        v_bytes = to_bytes(1)

        # Concatenate r, s, and v to form the packed signature
        packed_signature = r_bytes + s_bytes + v_bytes
        signatures += packed_signature

        return signatures.hex()


class PostTxSettlementBehaviour(LiquidityTraderBaseBehaviour):
    """
    This behaviour should be executed after a tx is settled via the transaction_settlement_abci.
    """

    matching_round = PostTxSettlementRound

    def async_act(self) -> Generator:
        """Simply log that a tx is settled and wait for the round end."""
        msg = f"The transaction submitted by {self.synchronized_data.tx_submitter} was successfully settled."
        self.context.logger.info(msg)
        yield from self.wait_until_round_end()
        self.set_done()


class LiquidityTraderRoundBehaviour(AbstractRoundBehaviour):
    """LiquidityTraderRoundBehaviour"""

    initial_behaviour_cls = CallCheckpointBehaviour
    abci_app_cls = LiquidityTraderAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = [
        CallCheckpointBehaviour,
        CheckStakingKPIMetBehaviour,
        GetPositionsBehaviour,
        EvaluateStrategyBehaviour,
        DecisionMakingBehaviour,
        PostTxSettlementBehaviour,
    ]

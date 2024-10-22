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
from abc import ABC
from collections import defaultdict
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    cast,
)
from urllib.parse import urlencode

from aea.configurations.data_types import PublicId
from eth_abi import decode
from eth_utils import keccak, to_bytes, to_hex

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
from packages.valory.skills.liquidity_trader_abci.models import (
    Coingecko,
    Params,
    SharedState,
)
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
    PostTxSettlementPayload,
    PostTxSettlementRound,
    StakingState,
    SynchronizedData,
    DecideAgentRound,
    DecideAgentPayload,
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

# Liveness ratio from the staking contract is expressed in calls per 10**18 seconds.
LIVENESS_RATIO_SCALE_FACTOR = 10**18

# A safety margin in case there is a delay between the moment the KPI condition is
# satisfied, and the moment where the checkpoint is called.
REQUIRED_REQUESTS_SAFETY_MARGIN = 1
MAX_RETRIES_FOR_API_CALL = 3
MAX_RETRIES_FOR_ROUTES = 3
HTTP_OK = [200, 201]
UTF8 = "utf-8"
STAKING_CHAIN = "optimism"
CAMPAIGN_TYPES = [1, 2]
INTEGRATOR = "valory"
WAITING_PERIOD_FOR_BALANCE_TO_REFLECT = 5
MAX_STEP_COST_RATIO = 0.5
WaitableConditionType = Generator[None, None, Any]
HTTP_NOT_FOUND = [400, 404]
ERC20_DECIMALS = 18


class DexTypes(Enum):
    """DexTypes"""

    BALANCER = "balancerPool"
    UNISWAP_V3 = "UniswapV3"


class Action(Enum):
    """Action"""

    CLAIM_REWARDS = "ClaimRewards"
    EXIT_POOL = "ExitPool"
    ENTER_POOL = "EnterPool"
    BRIDGE_SWAP = "BridgeAndSwap"
    FIND_BRIDGE_ROUTE = "FindBridgeRoute"
    EXECUTE_STEP = "execute_step"
    ROUTES_FETCHED = "routes_fetched"
    FIND_ROUTE = "find_route"
    BRIDGE_SWAP_EXECUTED = "bridge_swap_executed"
    STEP_EXECUTED = "step_executed"
    SWITCH_ROUTE = "switch_route"


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


class GasCostTracker:
    """Class to track and report gas costs."""

    MAX_RECORDS = 20  # Maximum number of records to keep per chain

    def __init__(self, file_path):
        """Initialize GasCostTracker"""
        self.file_path = file_path
        self.data = {}

    def log_gas_usage(self, chain, timestamp, tx_hash, gas_used, gas_price):
        """Log the gas cost for a transaction."""
        gas_cost_entry = {
            "timestamp": timestamp,
            "tx_hash": tx_hash,
            "gas_used": gas_used,
            "gas_price": gas_price,
        }
        if chain not in self.data:
            self.data[chain] = []

        # Add new record and maintain only the latest MAX_RECORDS
        self.data[chain].append(gas_cost_entry)
        if len(self.data[chain]) > self.MAX_RECORDS:
            self.data[chain] = self.data[chain][-self.MAX_RECORDS :]

    def update_data(self, new_data: dict):
        """Update the internal data with new data."""
        self.data = new_data


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
        self.current_pool_filepath: str = (
            self.params.store_path / self.params.pool_info_filename
        )
        self.pools: Dict[str, Any] = {}
        self.pools[DexTypes.BALANCER.value] = BalancerPoolBehaviour
        self.pools[DexTypes.UNISWAP_V3.value] = UniswapPoolBehaviour
        self.service_staking_state = StakingState.UNSTAKED
        self.strategy = SimpleStrategyBehaviour
        self.gas_cost_tracker = GasCostTracker(
            file_path=self.params.store_path / self.params.gas_cost_info_filename
        )
        # Read the assets and current pool
        self.read_current_pool()
        self.read_assets()
        self.read_gas_costs()

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

    @property
    def coingecko(self) -> Coingecko:
        """Return the Coingecko."""
        return cast(Coingecko, self.context.coingecko)

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
                    balance = 0 if balance is None else balance

                asset_balances_dict[chain].append(
                    {
                        "asset_symbol": asset_symbol,
                        "asset_type": (
                            "native" if asset_address == ZERO_ADDRESS else "erc_20"
                        ),
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
        for position in positions:
            if position.get("chain") == chain:
                for asset in position.get("assets", {}):
                    if asset.get("address") == token:
                        return asset.get("balance")

        return None

    def _get_token_decimals(
        self, chain: str, asset_address: str
    ) -> Generator[None, None, Optional[int]]:
        """Get token decimals"""
        decimals = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=asset_address,
            contract_public_id=ERC20.contract_id,
            contract_callable="get_token_decimals",
            data_key="data",
            chain_id=chain,
        )
        return decimals

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

    def _read_data(
        self, attribute: str, filepath: str, class_object: bool = False
    ) -> None:
        """Generic method to read data from a JSON file"""
        try:
            with open(filepath, READ_MODE) as file:
                try:
                    data = json.load(file)
                    if hasattr(self, attribute):
                        current_attr = getattr(self, attribute)
                        if class_object and hasattr(current_attr, "update_data"):
                            current_attr.update_data(data)
                        else:
                            setattr(self, attribute, data)
                    else:
                        self.context.logger.warning(
                            f"Attribute {attribute} does not exist."
                        )
                    return
                except (json.JSONDecodeError, TypeError) as e:
                    err = f"Error decoding {attribute} from {filepath!r}: {str(e)}"
        except FileNotFoundError:
            # Create the file if it doesn't exist
            with open(filepath, WRITE_MODE) as file:
                json.dump({}, file)
            return
        except (PermissionError, OSError) as e:
            err = f"Error reading from file {filepath!r}: {str(e)}"

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

    def store_gas_costs(self) -> None:
        """Store the gas costs as JSON."""
        self._store_data(
            self.gas_cost_tracker.data,
            "gas_cost_tracker",
            self.gas_cost_tracker.file_path,
        )

    def read_gas_costs(self) -> None:
        """Read the gas costs from JSON."""
        self._read_data("gas_cost_tracker", self.gas_cost_tracker.file_path, True)

    def _calculate_min_num_of_safe_tx_required(
        self, chain: str
    ) -> Generator[None, None, Optional[int]]:
        """Calculates the minimun number of tx to hit to unlock the staking rewards"""
        liveness_ratio = yield from self._get_liveness_ratio(chain)
        liveness_period = yield from self._get_liveness_period(chain)
        if not liveness_ratio or not liveness_period:
            return None

        current_timestamp = int(
            self.round_sequence.last_round_transition_timestamp.timestamp()
        )

        last_ts_checkpoint = yield from self._get_ts_checkpoint(chain=STAKING_CHAIN)
        if last_ts_checkpoint is None:
            return None

        min_num_of_safe_tx_required = (
            math.ceil(
                max(liveness_period, (current_timestamp - last_ts_checkpoint))
                * liveness_ratio
                // LIVENESS_RATIO_SCALE_FACTOR
            )
            + REQUIRED_REQUESTS_SAFETY_MARGIN
        )

        return min_num_of_safe_tx_required

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

    def _get_ts_checkpoint(self, chain: str) -> Generator[None, None, Optional[int]]:
        """Get the ts checkpoint"""
        ts_checkpoint = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=self.params.staking_token_contract_address,
            contract_public_id=StakingTokenContract.contract_id,
            contract_callable="ts_checkpoint",
            data_key="data",
            chain_id=chain,
        )
        return ts_checkpoint

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

    def _is_staking_kpi_met(self) -> Generator[None, None, Optional[bool]]:
        """Return whether the staking KPI has been met (only for staked services)."""
        if self.synchronized_data.service_staking_state != StakingState.STAKED.value:
            return None

        min_num_of_safe_tx_required = self.synchronized_data.min_num_of_safe_tx_required
        if min_num_of_safe_tx_required is None:
            self.context.logger.error(
                "Error calculating min number of safe tx required."
            )
            return None

        multisig_nonces_since_last_cp = (
            yield from self._get_multisig_nonces_since_last_cp(
                chain=STAKING_CHAIN,
                multisig=self.params.safe_contract_addresses.get(STAKING_CHAIN),
            )
        )
        if multisig_nonces_since_last_cp is None:
            return None

        if multisig_nonces_since_last_cp >= min_num_of_safe_tx_required:
            return True

        return False

    def _get_multisig_nonces(
        self, chain: str, multisig: str
    ) -> Generator[None, None, Optional[int]]:
        multisig_nonces = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=self.params.staking_activity_checker_contract_address,
            contract_public_id=StakingActivityCheckerContract.contract_id,
            contract_callable="get_multisig_nonces",
            data_key="data",
            chain_id=chain,
            multisig=multisig,
        )
        if multisig_nonces is None or len(multisig_nonces) == 0:
            return None
        return multisig_nonces[0]

    def _get_multisig_nonces_since_last_cp(
        self, chain: str, multisig: str
    ) -> Generator[None, None, Optional[int]]:
        multisig_nonces = yield from self._get_multisig_nonces(chain, multisig)
        if multisig_nonces is None:
            return None

        service_info = yield from self._get_service_info(chain)
        if service_info is None or len(service_info) == 0 or len(service_info[2]) == 0:
            self.context.logger.error(f"Error fetching service info {service_info}")
            return None

        multisig_nonces_on_last_checkpoint = service_info[2][0]

        multisig_nonces_since_last_cp = (
            multisig_nonces - multisig_nonces_on_last_checkpoint
        )
        self.context.logger.info(
            f"Number of safe transactions since last checkpoint: {multisig_nonces_since_last_cp}"
        )
        return multisig_nonces_since_last_cp

    def _get_service_info(
        self, chain: str
    ) -> Generator[None, None, Optional[Tuple[Any, Any, Tuple[Any, Any]]]]:
        """Get the service info."""
        service_id = self.params.on_chain_service_id
        if service_id is None:
            self.context.logger.warning(
                "Cannot perform any staking-related operations without a configured on-chain service id. "
                "Assuming service status 'UNSTAKED'."
            )
            return None

        service_info = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=self.params.staking_token_contract_address,
            contract_public_id=StakingTokenContract.contract_id,
            contract_callable="get_service_info",
            data_key="data",
            service_id=service_id,
            chain_id=chain,
        )
        return service_info

    def _get_service_staking_state(self, chain: str) -> Generator[None, None, None]:
        service_id = self.params.on_chain_service_id
        if service_id is None:
            self.context.logger.warning(
                "Cannot perform any staking-related operations without a configured on-chain service id. "
                "Assuming service status 'UNSTAKED'."
            )
            self.service_staking_state = StakingState.UNSTAKED
            return

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
            self.service_staking_state = StakingState.UNSTAKED
            return

        self.service_staking_state = StakingState(service_staking_state)
        return


class CallCheckpointBehaviour(
    LiquidityTraderBaseBehaviour
):  # pylint-disable too-many-ancestors
    """Behaviour that calls the checkpoint contract function if the service is staked and if it is necessary."""

    matching_round = CallCheckpointRound

    def async_act(self) -> Generator:
        """Do the action."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            yield from self._get_service_staking_state(chain="optimism")
            checkpoint_tx_hex = None
            min_num_of_safe_tx_required = None
            if self.service_staking_state == StakingState.STAKED:
                min_num_of_safe_tx_required = (
                    yield from self._calculate_min_num_of_safe_tx_required(
                        chain=STAKING_CHAIN
                    )
                )
                if min_num_of_safe_tx_required is None:
                    self.context.logger.error(
                        "Error calculating min number of safe tx required."
                    )
                else:
                    self.context.logger.info(
                        f"The minimum number of safe tx required to unlock rewards are {min_num_of_safe_tx_required}"
                    )
                is_checkpoint_reached = yield from self._check_if_checkpoint_reached(
                    chain=STAKING_CHAIN
                )
                if is_checkpoint_reached:
                    self.context.logger.info(
                        "Checkpoint reached! Preparing checkpoint tx.."
                    )
                    checkpoint_tx_hex = yield from self._prepare_checkpoint_tx(
                        chain=STAKING_CHAIN
                    )
            elif self.service_staking_state == StakingState.EVICTED:
                self.context.logger.error("Service has been evicted!")

            else:
                self.context.logger.error("Service has not been staked")

            tx_submitter = self.matching_round.auto_round_id()
            payload = CallCheckpointPayload(
                self.context.agent_address,
                tx_submitter,
                checkpoint_tx_hex,
                self.params.safe_contract_addresses.get(STAKING_CHAIN),
                STAKING_CHAIN,
                self.service_staking_state.value,
                min_num_of_safe_tx_required,
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()

    def _check_if_checkpoint_reached(
        self, chain: str
    ) -> Generator[None, None, Optional[bool]]:
        next_checkpoint = yield from self._get_next_checkpoint(chain)
        if next_checkpoint is None:
            return False

        if next_checkpoint == 0:
            return True

        synced_timestamp = int(
            self.round_sequence.last_round_transition_timestamp.timestamp()
        )
        return next_checkpoint <= synced_timestamp

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
            to_address=self.params.staking_token_contract_address,
            data=data,
        )


class CheckStakingKPIMetBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that checks if the staking KPI has been met and makes vanity transactions if necessary."""

    # pylint-disable too-many-ancestors
    matching_round: Type[AbstractRound] = CheckStakingKPIMetRound

    def async_act(self) -> Generator:
        """Do the action."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            vanity_tx_hex = None
            is_staking_kpi_met = yield from self._is_staking_kpi_met()
            if is_staking_kpi_met is None:
                self.context.logger.error("Error checking if staking KPI is met.")
            elif is_staking_kpi_met is True:
                self.context.logger.info("KPI already met for the day!")
            else:
                is_period_threshold_exceeded = (
                    self.synchronized_data.period_count
                    - self.synchronized_data.period_number_at_last_cp
                    >= self.params.staking_threshold_period
                )

                if is_period_threshold_exceeded:
                    min_num_of_safe_tx_required = (
                        self.synchronized_data.min_num_of_safe_tx_required
                    )
                    multisig_nonces_since_last_cp = (
                        yield from self._get_multisig_nonces_since_last_cp(
                            chain=STAKING_CHAIN,
                            multisig=self.params.safe_contract_addresses.get(
                                STAKING_CHAIN
                            ),
                        )
                    )
                    if (
                        multisig_nonces_since_last_cp is not None
                        and min_num_of_safe_tx_required is not None
                    ):
                        num_of_tx_left_to_meet_kpi = (
                            min_num_of_safe_tx_required - multisig_nonces_since_last_cp
                        )
                        if num_of_tx_left_to_meet_kpi > 0:
                            self.context.logger.info(
                                f"Number of tx left to meet KPI: {num_of_tx_left_to_meet_kpi}"
                            )
                            self.context.logger.info("Preparing vanity tx..")
                            vanity_tx_hex = yield from self._prepare_vanity_tx(
                                chain=STAKING_CHAIN
                            )
                            self.context.logger.info(f"tx hash: {vanity_tx_hex}")

            tx_submitter = self.matching_round.auto_round_id()
            payload = CheckStakingKPIMetPayload(
                self.context.agent_address,
                tx_submitter,
                vanity_tx_hex,
                self.params.safe_contract_addresses.get(STAKING_CHAIN),
                STAKING_CHAIN,
                is_staking_kpi_met,
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()

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
    """Behaviour that gets the balances of the assets of agent safes."""

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
    """Behaviour that finds the opportunity and builds actions."""

    matching_round: Type[AbstractRound] = EvaluateStrategyRound
    highest_apr_pool = None

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            yield from self.find_highest_apr_pool()
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

    def find_highest_apr_pool(self) -> Generator[None, None, None]:
        """Find the pool with the highest APR."""
        all_pools = yield from self._fetch_all_pools()
        if not all_pools:
            self.context.logger.info("No pools found.")
            return

        eligible_pools = self._filter_eligible_pools(all_pools)
        if not eligible_pools:
            self.context.logger.info("No eligible pools found.")
            return
        self.highest_apr_pool = yield from self._determine_highest_apr_pool(
            eligible_pools
        )

        if self.highest_apr_pool:
            self.context.logger.info(f"Highest APR pool found: {self.highest_apr_pool}")
        else:
            self.context.logger.warning("No pools with APR found.")

    def _fetch_all_pools(self) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Fetch all pools based on allowed chains."""
        chain_ids = ",".join(
            str(self.params.chain_to_chain_id_mapping[chain])
            for chain in self.params.allowed_chains
        )
        base_url = self.params.merkl_fetch_campaigns_args.get("url")
        creator = self.params.merkl_fetch_campaigns_args.get("creator")
        live = self.params.merkl_fetch_campaigns_args.get("live", "true")

        params = {
            "chainIds": chain_ids,
            "creatorTag": creator,
            "live": live,
            "types": CAMPAIGN_TYPES,
        }
        api_url = f"{base_url}?{urlencode(params, doseq=True)}"
        self.context.logger.info(f"Fetching campaigns from {api_url}")

        response = yield from self.get_http_response(
            method="GET",
            url=api_url,
            headers={"accept": "application/json"},
        )

        if response.status_code not in HTTP_OK:
            self.context.logger.error(
                f"Could not retrieve data from url {api_url}. Status code {response.status_code}. Error Message: {response.body}"
            )
            return None

        try:
            return json.loads(response.body)
        except (ValueError, TypeError) as e:
            self.context.logger.error(
                f"Could not parse response from api, the following error was encountered {type(e).__name__}: {e}"
            )
            return None

    def _filter_eligible_pools(self, all_pools: Dict[str, Any]) -> Dict[str, Any]:
        """Filter pools based on allowed assets and LP pools."""
        eligible_pools = defaultdict(lambda: defaultdict(list))
        allowed_dexs = self.params.allowed_dexs

        for chain_id, campaigns in all_pools.items():
            for campaign_list in campaigns.values():
                for campaign in campaign_list.values():
                    dex_type = campaign.get("type") or campaign.get("ammName")
                    if not dex_type or dex_type not in allowed_dexs:
                        continue

                    campaign_apr = campaign.get("apr", 0)
                    if (
                        not campaign_apr
                        or campaign_apr <= 0
                        or campaign_apr <= self.current_pool.get("apr", 0)
                    ):
                        continue

                    campaign_pool_address = campaign.get("mainParameter")
                    if (
                        not campaign_pool_address
                        or campaign_pool_address == self.current_pool.get("address")
                    ):
                        continue

                    chain = next(
                        (
                            k
                            for k, v in self.params.chain_to_chain_id_mapping.items()
                            if v == int(chain_id)
                        ),
                        None,
                    )
                    eligible_pools[dex_type][chain].append(campaign)

        return eligible_pools

    def _determine_highest_apr_pool(
        self, eligible_pools: Dict[str, Any]
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Determine the pool with the highest APR from the eligible pools."""

        highest_apr_pool = None
        highest_apr_pool_info = None
        while eligible_pools:
            highest_apr = -float("inf")
            for dex_type, chains in eligible_pools.items():
                for chain, campaigns in chains.items():
                    for campaign in campaigns:
                        apr = campaign.get("apr", 0) or 0
                        if apr > highest_apr:
                            highest_apr = apr
                            highest_apr_pool_info = (dex_type, chain, campaign)

            if highest_apr_pool_info:
                dex_type, chain, campaign = highest_apr_pool_info
                highest_apr_pool = yield from self._extract_pool_info(
                    dex_type, chain, highest_apr, campaign
                )

                # Check the number of tokens for the highest APR pool if it's a Balancer pool
                if dex_type == DexTypes.BALANCER.value:
                    pool_id = highest_apr_pool.get("pool_id")
                    if highest_apr_pool.get("pool_type") is None:
                        self.context.logger.error(
                            f"Pool type not found for pool {pool_id}"
                        )
                        continue
                    tokensList = yield from self._fetch_balancer_pool_info(
                        pool_id, chain, detail="tokensList"
                    )
                    if not tokensList or len(tokensList) != 2:
                        num_of_tokens = len(tokensList) if tokensList else None
                        self.context.logger.warning(
                            f"Balancer pool {pool_id} has {num_of_tokens} tokens, currently we support pools with only 2 tokens"
                        )
                        self.context.logger.info("Searching for another pool")
                        highest_apr_pool = None
                        highest_apr_pool_info = None

                        # Remove the invalid pool from eligible pools and continue searching
                        eligible_pools[dex_type][chain].remove(campaign)
                        if not eligible_pools[dex_type][chain]:
                            del eligible_pools[dex_type][chain]

                        if not eligible_pools[dex_type]:
                            del eligible_pools[dex_type]

                        continue

                return highest_apr_pool

        self.context.logger.warning("No eligible pools found.")
        return None

    def _extract_pool_info(
        self, dex_type, chain, apr, campaign
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Extract pool info from campaign data."""
        pool_address = campaign.get("mainParameter")
        if not pool_address:
            self.context.logger.error(f"Missing pool address in campaign {campaign}")
            return None

        pool_token_dict = {}
        pool_id = None
        pool_type = None

        if dex_type == DexTypes.BALANCER.value:
            type_info = campaign.get("typeInfo", {})
            pool_id = type_info.get("poolId")
            pool_tokens = type_info.get("poolTokens", {})
            pool_token_items = list(pool_tokens.items())
            if len(pool_token_items) < 2 or any(
                token.get("symbol") is None for _, token in pool_token_items
            ):
                self.context.logger.error(
                    f"Invalid pool tokens found in campaign {pool_token_items}"
                )
                return None

            pool_type = yield from self._fetch_balancer_pool_info(
                pool_id, chain, detail="poolType"
            )
            pool_token_dict = {
                "token0": pool_token_items[0][0],
                "token1": pool_token_items[1][0],
                "token0_symbol": pool_token_items[0][1].get("symbol"),
                "token1_symbol": pool_token_items[1][1].get("symbol"),
            }

        elif dex_type == DexTypes.UNISWAP_V3.value:
            pool_info = campaign.get("campaignParameters", {})
            if not pool_info:
                self.context.logger.error(
                    f"No pool tokens info present in campaign {campaign}"
                )
                return None

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

        return {
            "dex_type": dex_type,
            "chain": chain,
            "apr": apr,
            "pool_address": pool_address,
            "pool_id": pool_id,
            "pool_type": pool_type,
            **pool_token_dict,
        }

    def _fetch_balancer_pool_info(
        self, pool_id: str, chain: str, detail: str
    ) -> Generator[None, None, Optional[Any]]:
        """Fetch the pool type for a Balancer pool using a GraphQL query."""

        def to_content(query: str) -> bytes:
            """Convert the given query string to payload content."""
            finalized_query = {"query": query}
            encoded_query = json.dumps(finalized_query, sort_keys=True).encode("utf-8")
            return encoded_query

        query = f"""
                    query {{
                    pools(where: {{ id: "{pool_id}" }}) {{ # noqa: B028
                        id
                        {detail}
                    }}
                    }}
                """

        url = self.params.balancer_graphql_endpoints.get(chain)
        if not url:
            self.context.logger.error(f"No graphql endpoint found for chain {chain}")
            return None

        response = yield from self.get_http_response(
            content=to_content(query),
            method="POST",
            url=url,
            headers={"Content-Type": "application/json"},
        )
        if response.status_code not in HTTP_OK:
            self.context.logger.error(
                f"Received status code {response.status_code} from the API. Response: {response.body}"
            )
            return None

        try:
            res = json.loads(response.body)
            if res is None:
                self.context.logger.error(
                    f"Could not get pool type for pool ID {pool_id}"
                )
                return None

            pools = res.get("data", {}).get("pools", [])
            if pools:
                self.context.logger.info(
                    f"{detail} for pool {pool_id}: {pools[0].get(detail)}"
                )
                return pools[0].get(detail)
            return None
        except json.JSONDecodeError as e:
            self.context.logger.error(f"Error decoding JSON response: {str(e)}")
            return None

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
                    if campaign_type in CAMPAIGN_TYPES:
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
            tokens = yield from self._get_top_tokens_by_value()
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
        if bridge_swap_actions is None:
            self.context.logger.info("Error preparing bridge swap actions")
            return None
        if bridge_swap_actions:
            actions.extend(bridge_swap_actions)

        enter_pool_action = self._build_enter_pool_action()
        if not enter_pool_action:
            self.context.logger.error("Error building enter pool action")
            return None
        actions.append(enter_pool_action)

        return actions

    def _get_top_tokens_by_value(self) -> Generator[None, None, Optional[List[Any]]]:
        """Get tokens over min balance"""
        token_balances = []
        for position in self.synchronized_data.positions:
            chain = position.get("chain")
            for asset in position.get("assets", {}):
                asset_address = asset.get("address")
                if not chain or not asset_address:
                    continue
                balance = asset.get("balance", 0)
                if balance and balance > 0:
                    token_balances.append(
                        {
                            "chain": chain,
                            "token": asset_address,
                            "token_symbol": asset.get("asset_symbol"),
                            "balance": balance,
                        }
                    )

        # Fetch prices for tokens with balance greater than zero
        token_prices = yield from self._fetch_token_prices(token_balances)

        # Calculate the relative value of each token
        for token_data in token_balances:
            token_address = token_data["token"]
            chain = token_data["chain"]
            token_price = token_prices.get(token_address, 0)
            if token_address == ZERO_ADDRESS:
                decimals = 18
            else:
                decimals = yield from self._get_token_decimals(chain, token_address)
            token_data["value"] = (
                token_data["balance"] / (10**decimals)
            ) * token_price

        # Sort tokens by value in descending order and add the highest ones
        token_balances.sort(key=lambda x: x["value"], reverse=True)

        tokens = []
        for token_data in token_balances:
            if token_data["value"] >= self.params.min_swap_amount_threshold:
                tokens.append(token_data)
            if len(tokens) == 2:
                self.context.logger.info(
                    f"Tokens selected for bridging/swapping: {tokens}"
                )
                break

        if len(tokens) < 2:
            self.context.logger.error(
                f"Insufficient tokens with value over minimum threshold i.e. ${self.params.min_swap_amount_threshold}. Required at least 2, available: {token_balances}"
            )
            return None

        return tokens

    def _fetch_token_prices(
        self, token_balances: List[Dict[str, Any]]
    ) -> Generator[None, None, Dict[str, float]]:
        """Fetch token prices from Coingecko"""
        token_prices = {}
        headers = {
            "Accept": "application/json",
        }
        if self.coingecko.api_key:
            headers["x-cg-api-key"] = self.coingecko.api_key

        for token_data in token_balances:
            token_address = token_data["token"]
            chain = token_data.get("chain")
            if not chain:
                self.context.logger.error(f"Missing chain for token {token_address}")
                continue

            if token_address == ZERO_ADDRESS:
                success, response_json = yield from self._request_with_retries(
                    endpoint=self.coingecko.coin_price_endpoint.format(
                        coin_id="ethereum"
                    ),
                    headers=headers,
                    rate_limited_code=self.coingecko.rate_limited_code,
                    rate_limited_callback=self.coingecko.rate_limited_status_callback,
                    retry_wait=self.params.sleep_time,
                )

                if success:
                    # Extract the first (and only) item in the response dictionary
                    token_data = next(iter(response_json.values()), {})
                    price = token_data.get("usd", 0)
                    token_prices[token_address] = price
            else:
                platform_id = self.coingecko.chain_to_platform_id_mapping.get(chain)
                if not platform_id:
                    self.context.logger.error(f"Missing platform id for chain {chain}")

                success, response_json = yield from self._request_with_retries(
                    endpoint=self.coingecko.token_price_endpoint.format(
                        token_address=token_address, asset_platform_id=platform_id
                    ),
                    headers=headers,
                    rate_limited_code=self.coingecko.rate_limited_code,
                    rate_limited_callback=self.coingecko.rate_limited_status_callback,
                    retry_wait=self.params.sleep_time,
                )

                if success:
                    token_data = response_json.get(token_address.lower(), {})
                    price = token_data.get("usd", 0)
                    token_prices[token_address] = price

        return token_prices

    def _request_with_retries(
        self,
        endpoint: str,
        rate_limited_callback: Callable,
        method: str = "GET",
        body: Optional[Any] = None,
        headers: Optional[Dict] = None,
        rate_limited_code: int = 429,
        max_retries: int = MAX_RETRIES_FOR_API_CALL,
        retry_wait: int = 0,
    ) -> Generator[None, None, Tuple[bool, Dict]]:
        """Request wrapped around a retry mechanism"""

        self.context.logger.info(f"HTTP {method} call: {endpoint}")
        content = json.dumps(body).encode(UTF8) if body else None

        retries = 0
        while True:
            # Make the request
            response = yield from self.get_http_response(
                method, endpoint, content, headers
            )

            try:
                response_json = json.loads(response.body)
            except json.decoder.JSONDecodeError as exc:
                self.context.logger.error(f"Exception during json loading: {exc}")
                response_json = {"exception": str(exc)}

            if response.status_code == rate_limited_code:
                rate_limited_callback()
                return False, response_json

            if response.status_code not in HTTP_OK or "exception" in response_json:
                self.context.logger.error(
                    f"Request failed [{response.status_code}]: {response_json}"
                )
                retries += 1
                if retries == max_retries:
                    break
                yield from self.sleep(retry_wait)
                continue

            self.context.logger.info("Request succeeded.")
            return True, response_json

        self.context.logger.error(f"Request failed after {retries} retries.")
        return False, response_json

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
            "pool_type": self.current_pool.get("pool_type"),
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
            if (
                source_token0_address not in [dest_token0_address, dest_token1_address]
                or source_token0_chain != dest_chain
            ):
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

            if (
                source_token1_address not in [dest_token0_address, dest_token1_address]
                or source_token1_chain != dest_chain
            ):
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
            "pool_type": self.highest_apr_pool.get("pool_type"),
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

        if response.status_code not in HTTP_OK:
            self.context.logger.error(
                f"Could not retrieve data from url {api_url}. Status code {response.status_code}. Error Message: {response.body}"
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
    """Behaviour that executes all the actions."""

    matching_round: Type[AbstractRound] = DecisionMakingRound

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            (next_event, updates) = yield from self.get_next_event()

            payload = DecisionMakingPayload(
                sender=sender,
                content=json.dumps(
                    {
                        "event": next_event,
                        "updates": updates,
                    },
                    sort_keys=True,
                ),
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_next_event(self) -> Generator[None, None, Tuple[str, Dict]]:
        """Get next event"""
        actions = self.synchronized_data.actions
        if not actions:
            self.context.logger.info("No actions to prepare")
            return Event.DONE.value, {}

        positions = self.synchronized_data.positions
        last_round_id = self.context.state.round_sequence._abci_app._previous_rounds[
            -1
        ].round_id
        if last_round_id != EvaluateStrategyRound.auto_round_id():
            positions = yield from self.get_positions()

        last_executed_action_index = self.synchronized_data.last_executed_action_index
        current_action_index = (
            0 if last_executed_action_index is None else last_executed_action_index + 1
        )

        if (
            self.synchronized_data.last_action == Action.EXECUTE_STEP.value
            and last_round_id != DecisionMakingRound.auto_round_id()
        ):
            res = yield from self._post_execute_step(
                actions, last_executed_action_index
            )
            return res

        if last_executed_action_index is not None:
            if self.synchronized_data.last_action == Action.ENTER_POOL.value:
                yield from self._post_execute_enter_pool(
                    actions, last_executed_action_index
                )
            if self.synchronized_data.last_action == Action.EXIT_POOL.value:
                yield from self._post_execute_exit_pool()
            if (
                self.synchronized_data.last_action == Action.CLAIM_REWARDS.value
                and last_round_id != DecisionMakingRound.auto_round_id()
            ):
                return self._post_execute_claim_rewards(
                    actions, last_executed_action_index
                )

        if (
            last_executed_action_index is not None
            and self.synchronized_data.last_action
            in [
                Action.ROUTES_FETCHED.value,
                Action.STEP_EXECUTED.value,
                Action.SWITCH_ROUTE.value,
            ]
        ):
            res = yield from self._process_route_execution(positions)
            return res

        if current_action_index >= len(self.synchronized_data.actions):
            self.context.logger.info("All actions have been executed")
            return Event.DONE.value, {}

        res = yield from self._prepare_next_action(
            positions, actions, current_action_index, last_round_id
        )
        return res

    def _post_execute_step(
        self, actions, last_executed_action_index
    ) -> Generator[None, None, Tuple[Optional[str], Optional[Dict]]]:
        """Handle the execution of a step."""
        self.context.logger.info("Checking the status of swap tx")
        # we wait for some time before checking the status of the tx because the tx may take time to reflect on the lifi endpoint
        yield from self.sleep(self.params.waiting_period_for_status_check)
        decision = yield from self.get_decision_on_swap()
        self.context.logger.info(f"Action to take {decision}")

        # If tx is pending then we wait until it gets confirmed or refunded
        if decision == Decision.WAIT:
            decision = yield from self._wait_for_swap_confirmation()

        if decision == Decision.EXIT:
            self.context.logger.error("Swap failed")
            return Event.DONE.value, {}

        if decision == Decision.CONTINUE:
            return self._update_assets_after_swap(actions, last_executed_action_index)

    def _wait_for_swap_confirmation(self) -> Generator[None, None, Optional[Decision]]:
        """Wait for swap confirmation."""
        self.context.logger.info("Waiting for tx to get executed")
        while True:
            yield from self.sleep(self.params.waiting_period_for_status_check)
            decision = yield from self.get_decision_on_swap()
            self.context.logger.info(f"Action to take {decision}")
            if decision != Decision.WAIT:
                break
        return decision

    def _update_assets_after_swap(
        self, actions, last_executed_action_index
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Update assets after a successful swap."""
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
        fee_details = {
            "remaining_fee_allowance": action.get("remaining_fee_allowance"),
            "remaining_gas_allowance": action.get("remaining_gas_allowance"),
        }
        return Event.UPDATE.value, {
            "last_executed_step_index": (
                self.synchronized_data.last_executed_step_index + 1
                if self.synchronized_data.last_executed_step_index is not None
                else 0
            ),
            "fee_details": fee_details,
            "last_action": Action.STEP_EXECUTED.value,
        }

    def _post_execute_enter_pool(self, actions, last_executed_action_index):
        """Handle entering a pool."""
        action = actions[last_executed_action_index]
        current_pool = {
            "chain": action["chain"],
            "address": action["pool_address"],
            "dex_type": action["dex_type"],
            "assets": action["assets"],
            "apr": action["apr"],
            "pool_type": action["pool_type"],
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

    def _post_execute_exit_pool(self):
        """Handle exiting a pool."""
        self.current_pool = {}
        self.store_current_pool()
        self.context.logger.info("Exit was successful! Removing current pool")
        # when we exit the pool, it may take time to reflect the balance of our assets in safe
        yield from self.sleep(WAITING_PERIOD_FOR_BALANCE_TO_REFLECT)

    def _post_execute_claim_rewards(
        self, actions, last_executed_action_index
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Handle claiming rewards."""
        action = actions[last_executed_action_index]
        chain = action.get("chain")
        for token, token_symbol in zip(
            action.get("tokens"), action.get("token_symbols")
        ):
            self._add_token_to_assets(chain, token, token_symbol)

        current_timestamp = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()

        return Event.UPDATE.value, {
            "last_reward_claimed_timestamp": current_timestamp,
            "last_action": Action.CLAIM_REWARDS.value,
        }

    def _process_route_execution(
        self, positions
    ) -> Generator[None, None, Tuple[Optional[str], Optional[Dict]]]:
        """Handle route execution."""
        routes = self.synchronized_data.routes
        if not routes:
            self.context.logger.error("No routes found!")
            return Event.DONE.value, {}

        last_executed_route_index = (
            -1
            if self.synchronized_data.last_executed_route_index is None
            else self.synchronized_data.last_executed_route_index
        )
        to_execute_route_index = last_executed_route_index + 1

        last_executed_step_index = (
            -1
            if self.synchronized_data.last_executed_step_index is None
            else self.synchronized_data.last_executed_step_index
        )
        to_execute_step_index = last_executed_step_index + 1

        if to_execute_route_index >= len(routes):
            self.context.logger.error("No more routes left to execute")
            return Event.DONE.value, {}
        if to_execute_step_index >= len(
            routes[to_execute_route_index].get("steps", [])
        ):
            self.context.logger.info("All steps executed successfully!")
            return Event.UPDATE.value, {
                "last_executed_route_index": None,
                "last_executed_step_index": None,
                "fee_details": None,
                "routes": None,
                "max_allowed_steps_in_a_route": None,
                "routes_retry_attempt": 0,
                "last_action": Action.BRIDGE_SWAP_EXECUTED.value,
            }

        res = yield from self._execute_route_step(
            positions, routes, to_execute_route_index, to_execute_step_index
        )
        return res

    def _execute_route_step(
        self, positions, routes, to_execute_route_index, to_execute_step_index
    ) -> Generator[None, None, Tuple[Optional[str], Optional[Dict]]]:
        """Execute a step in the route."""
        steps = routes[to_execute_route_index].get("steps")
        step = steps[to_execute_step_index]

        remaining_fee_allowance = 0
        remaining_gas_allowance = 0

        if to_execute_step_index == 0:
            (
                is_profitable,
                total_fee,
                total_gas_cost,
            ) = yield from self.check_if_route_is_profitable(
                routes[to_execute_route_index]
            )
            if not is_profitable:
                self.context.logger.error(
                    "Route not profitable. Switching to next route.."
                )
                return Event.UPDATE.value, {
                    "last_executed_route_index": to_execute_route_index,
                    "last_executed_step_index": None,
                    "last_action": Action.SWITCH_ROUTE.value,
                }

            remaining_fee_allowance = total_fee
            remaining_gas_allowance = total_gas_cost

        else:
            remaining_fee_allowance = self.synchronized_data.fee_details.get(
                "remaining_fee_allowance"
            )
            remaining_gas_allowance = self.synchronized_data.fee_details.get(
                "remaining_gas_allowance"
            )

        step_profitable, step_data = yield from self.check_step_costs(
            step,
            remaining_fee_allowance,
            remaining_gas_allowance,
            to_execute_step_index,
            len(steps),
        )
        if not step_profitable:
            return Event.DONE.value, {}

        self.context.logger.info(
            f"Preparing bridge swap action for {step_data.get('source_token_symbol')}({step_data.get('from_chain')}) "
            f"to {step_data.get('target_token_symbol')}({step_data.get('to_chain')}) using tool {step_data.get('tool')}"
        )
        bridge_swap_action = yield from self.prepare_bridge_swap_action(
            positions, step_data, remaining_fee_allowance, remaining_gas_allowance
        )
        if not bridge_swap_action:
            return self._handle_failed_step(
                to_execute_step_index, to_execute_route_index, step_data, len(steps)
            )

        return Event.UPDATE.value, {
            "new_action": bridge_swap_action,
            "last_action": Action.EXECUTE_STEP.value,
        }

    def _handle_failed_step(
        self, to_execute_step_index, to_execute_route_index, step_data, total_steps
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Handle a failed step in the route."""
        if to_execute_step_index == 0:
            self.context.logger.error("First step failed. Switching to next route..")
            return Event.UPDATE.value, {
                "last_executed_route_index": to_execute_route_index,
                "last_executed_step_index": None,
                "last_action": Action.SWITCH_ROUTE.value,
            }

        self.context.logger.error("Intermediate step failed. Fetching new routes..")
        if self.synchronized_data.routes_retry_attempt > MAX_RETRIES_FOR_ROUTES:
            self.context.logger.error("Exceeded retry limit")
            return Event.DONE.value, {}

        routes_retry_attempt = self.synchronized_data.routes_retry_attempt + 1
        find_route_action = {
            "action": Action.FIND_BRIDGE_ROUTE.value,
            "from_chain": step_data["from_chain"],
            "to_chain": step_data["to_chain"],
            "from_token": step_data["source_token"],
            "from_token_symbol": step_data["source_token_symbol"],
            "to_token": step_data["target_token"],
            "to_token_symbol": step_data["target_token_symbol"],
        }

        return Event.UPDATE.value, {
            "last_executed_step_index": None,
            "last_executed_route_index": None,
            "fee_details": None,
            "routes": None,
            "new_action": find_route_action,
            "routes_retry_attempt": routes_retry_attempt,
            "max_allowed_steps_in_a_route": total_steps - to_execute_step_index,
            "last_action": Action.FIND_ROUTE.value,
        }

    def _prepare_next_action(
        self, positions, actions, current_action_index, last_round_id
    ) -> Generator[None, None, Tuple[Optional[str], Optional[Dict]]]:
        """Prepare the next action."""
        next_action = Action(actions[current_action_index].get("action"))
        next_action_details = self.synchronized_data.actions[current_action_index]
        self.context.logger.info(f"ACTION DETAILS: {next_action_details}")

        if next_action == Action.ENTER_POOL:
            tx_hash, chain_id, safe_address = yield from self.get_enter_pool_tx_hash(
                positions, next_action_details
            )
            last_action = Action.ENTER_POOL.value

        elif next_action == Action.EXIT_POOL:
            tx_hash, chain_id, safe_address = yield from self.get_exit_pool_tx_hash(
                next_action_details
            )
            last_action = Action.EXIT_POOL.value

        elif next_action == Action.FIND_BRIDGE_ROUTE:
            routes = yield from self.fetch_routes(positions, next_action_details)
            if self.synchronized_data.max_allowed_steps_in_a_route:
                routes = [
                    route
                    for route in routes
                    if len(route.get("steps", []))
                    <= self.synchronized_data.max_allowed_steps_in_a_route
                ]

            serialized_routes = json.dumps(routes)
            if not routes:
                self.context.logger.error("Error fetching routes")
                return Event.DONE.value, {}

            return Event.UPDATE.value, {
                "routes": serialized_routes,
                "last_action": Action.ROUTES_FETCHED.value,
                "last_executed_action_index": current_action_index,
            }

        elif next_action == Action.BRIDGE_SWAP:
            yield from self.sleep(WAITING_PERIOD_FOR_BALANCE_TO_REFLECT)
            tx_hash = next_action_details.get("payload")
            chain_id = next_action_details.get("from_chain")
            safe_address = next_action_details.get("safe_address")
            last_action = Action.EXECUTE_STEP.value

        elif next_action == Action.CLAIM_REWARDS:
            tx_hash, chain_id, safe_address = yield from self.get_claim_rewards_tx_hash(
                next_action_details
            )
            last_action = Action.CLAIM_REWARDS.value

        else:
            tx_hash = None
            chain_id = None
            safe_address = None
            last_action = None

        if not tx_hash:
            return Event.DONE.value, {}

        return Event.SETTLE.value, {
            "tx_submitter": DecisionMakingRound.auto_round_id(),
            "most_voted_tx_hash": tx_hash,
            "chain_id": chain_id,
            "safe_contract_address": safe_address,
            "positions": positions,
            "last_executed_action_index": current_action_index,
            "last_action": last_action,
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

        while True:
            response = yield from self.get_http_response(
                method="GET",
                url=url,
                headers={"accept": "application/json"},
            )

            if response.status_code in HTTP_NOT_FOUND:
                self.context.logger.warning(f"Message {response.body}. Retrying..")
                yield from self.sleep(self.params.waiting_period_for_status_check)
                continue

            if response.status_code not in HTTP_OK:
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
        pool_type = action.get("pool_type")

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
            self.context.logger.error(
                f"Insufficient balance for entering pool: {max_amounts_in}"
            )
            return None, None, None

        tx_hash, contract_address = yield from pool.enter(
            self,
            pool_address=pool_address,
            safe_address=safe_address,
            assets=assets,
            chain=chain,
            max_amounts_in=max_amounts_in,
            pool_fee=pool_fee,
            pool_type=pool_type,
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
        pool_type = action.get("pool_type")
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
                    "pool_type": pool_type,
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
            operation=(
                SafeOperation.DELEGATE_CALL.value
                if is_multisend
                else SafeOperation.CALL.value
            ),
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
            operation=(
                SafeOperation.DELEGATE_CALL.value
                if is_multisend
                else SafeOperation.CALL.value
            ),
            to_address=contract_address,
            data=tx_hash,
        )

        self.context.logger.info(f"Tx hash payload string is {payload_string}")

        return payload_string, chain, safe_address

    def prepare_bridge_swap_action(
        self,
        positions: List[Dict[str, Any]],
        tx_info: Dict[str, Any],
        remaining_fee_allowance: float,
        remaining_gas_allowance: float,
    ) -> Generator[None, None, Optional[Dict]]:
        """Prepares the bridge swap action"""
        multisend_tx_hash = yield from self._build_multisend_tx(positions, tx_info)
        if not multisend_tx_hash:
            return None

        multisend_tx_data = bytes.fromhex(multisend_tx_hash[2:])
        from_chain = tx_info.get("from_chain")
        multisend_address = self.params.multisend_contract_addresses[
            tx_info.get("from_chain")
        ]

        is_ok = yield from self._simulate_transaction(
            to_address=multisend_address,
            data=multisend_tx_data,
            token=tx_info.get("source_token"),
            amount=tx_info.get("amount"),
            chain=tx_info.get("from_chain"),
        )
        if not is_ok:
            self.context.logger.info(
                f"Simulation failed for bridge/swap tx: {tx_info.get('source_token_symbol')}({tx_info.get('from_chain')}) --> {tx_info.get('target_token_symbol')}({tx_info.get('to_chain')}). Tool used: {tx_info.get('tool')}"
            )
            return None

        self.context.logger.info(
            f"Simulation successful for bridge/swap tx: {tx_info.get('source_token_symbol')}({tx_info.get('from_chain')}) --> {tx_info.get('target_token_symbol')}({tx_info.get('to_chain')}). Tool used: {tx_info.get('tool')}"
        )

        payload_string = yield from self._build_safe_tx(
            from_chain, multisend_tx_hash, multisend_address
        )
        if not payload_string:
            return None

        bridge_and_swap_action = {
            "action": Action.BRIDGE_SWAP.value,
            "from_chain": tx_info.get("from_chain"),
            "to_chain": tx_info.get("to_chain"),
            "from_token": tx_info.get("source_token"),
            "from_token_symbol": tx_info.get("source_token_symbol"),
            "to_token": tx_info.get("target_token"),
            "to_token_symbol": tx_info.get("target_token_symbol"),
            "payload": payload_string,
            "safe_address": self.params.safe_contract_addresses.get(from_chain),
            "remaining_gas_allowance": remaining_gas_allowance
            - tx_info.get("gas_cost"),
            "remaining_fee_allowance": remaining_fee_allowance - tx_info.get("fee"),
        }
        return bridge_and_swap_action

    def check_if_route_is_profitable(
        self, route: Dict[str, Any]
    ) -> Generator[None, None, Tuple[Optional[bool], Optional[float], Optional[float]]]:
        """Checks if the entire route is profitable"""
        step_transactions = yield from self._get_step_transactions_data(route)
        total_gas_cost = 0
        total_fee = 0
        total_fee += sum(float(tx_info.get("fee", 0)) for tx_info in step_transactions)
        total_gas_cost += sum(
            float(tx_info.get("gas_cost", 0)) for tx_info in step_transactions
        )
        from_amount_usd = float(route.get("fromAmountUSD", 0))
        to_amount_usd = float(route.get("toAmountUSD", 0))

        if not from_amount_usd or not to_amount_usd:
            return False, None, None

        allowed_fee_percentage = self.params.max_fee_percentage * 100
        allowed_gas_percentage = self.params.max_gas_percentage * 100

        fee_percentage = (total_fee / from_amount_usd) * 100
        gas_percentage = (total_gas_cost / from_amount_usd) * 100

        self.context.logger.info(
            f"Fee is {fee_percentage:.2f}% of total amount, allowed is {allowed_fee_percentage:.2f}% and gas is {gas_percentage:.2f}% of total amount, allowed is {allowed_gas_percentage:.2f}%."
            f"Details: total_fee={total_fee}, total_gas_cost={total_gas_cost}, from_amount_usd={from_amount_usd}, to_amount_usd={to_amount_usd}"
        )

        if (
            fee_percentage > allowed_fee_percentage
            or gas_percentage > allowed_gas_percentage
        ):
            self.context.logger.error("Route is not profitable!")
            return False, None, None

        self.context.logger.info("Route is profitable!")
        return True, total_fee, total_gas_cost

    def check_step_costs(
        self,
        step,
        remaining_fee_allowance,
        remaining_gas_allowance,
        step_index,
        total_steps,
    ) -> Generator[None, None, Tuple[Optional[bool], Optional[Dict[str, Any]]]]:
        """Check if the step costs are within the allowed range."""
        step = self._set_step_addresses(step)
        step_data = yield from self._get_step_transaction(step)
        if not step_data:
            self.context.logger.error("Error fetching step transaction")
            return False, None

        step_fee = step_data.get("fee", 0)
        step_gas_cost = step_data.get("gas_cost", 0)

        TOLERANCE = 0.02

        if total_steps != 1 and step_index == total_steps - 1:
            # For the last step, ensure it is not more than 50% of the remaining fee and gas allowance
            if (
                step_fee > MAX_STEP_COST_RATIO * remaining_fee_allowance + TOLERANCE
                or step_gas_cost
                > MAX_STEP_COST_RATIO * remaining_gas_allowance + TOLERANCE
            ):
                self.context.logger.error(
                    f"Step exceeds 50% of the remaining fee or gas allowance. "
                    f"Step fee: {step_fee}, Remaining fee allowance: {remaining_fee_allowance}, "
                    f"Step gas cost: {step_gas_cost}, Remaining gas allowance: {remaining_gas_allowance}. Dropping step."
                )
                return False, None

        else:
            if (
                step_fee > remaining_fee_allowance + TOLERANCE
                or step_gas_cost > remaining_gas_allowance + TOLERANCE
            ):
                self.context.logger.error(
                    f"Step exceeds remaining fee or gas allowance. "
                    f"Step fee: {step_fee}, Remaining fee allowance: {remaining_fee_allowance}, "
                    f"Step gas cost: {step_gas_cost}, Remaining gas allowance: {remaining_gas_allowance}. Dropping step."
                )
                return False, None

        self.context.logger.info(
            f"Step is profitable! Step fee: {step_fee}, Step gas cost: {step_gas_cost}"
        )
        return True, step_data

    def _build_safe_tx(
        self, from_chain, multisend_tx_hash, multisend_address
    ) -> Generator[None, None, Optional[str]]:
        safe_address = self.params.safe_contract_addresses.get(from_chain)
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
            chain_id=from_chain,
        )

        if not safe_tx_hash:
            self.context.logger.error("Error preparing safe tx!")
            return None

        safe_tx_hash = safe_tx_hash[2:]
        tx_params = dict(
            ether_value=ETHER_VALUE,
            safe_tx_gas=SAFE_TX_GAS,
            operation=SafeOperation.DELEGATE_CALL.value,
            to_address=multisend_address,
            data=bytes.fromhex(multisend_tx_hash[2:]),
            safe_tx_hash=safe_tx_hash,
        )
        payload_string = hash_payload_to_hex(**tx_params)
        return payload_string

    def _build_multisend_tx(
        self, positions, tx_info
    ) -> Generator[None, None, Optional[str]]:
        multisend_txs = []
        amount = self._get_balance(
            tx_info.get("from_chain"), tx_info.get("source_token"), positions
        )

        if tx_info.get("source_token") != ZERO_ADDRESS:
            approval_tx_payload = yield from self.get_approval_tx_hash(
                token_address=tx_info.get("source_token"),
                amount=amount,
                spender=tx_info.get("lifi_contract_address"),
                chain=tx_info.get("from_chain"),
            )
            if not approval_tx_payload:
                self.context.logger.error("Error preparing approval tx payload")
                return None

            multisend_txs.append(approval_tx_payload)

        multisend_txs.append(
            {
                "operation": MultiSendOperation.CALL,
                "to": tx_info.get("lifi_contract_address"),
                "value": (0 if tx_info.get("source_token") != ZERO_ADDRESS else amount),
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

        return multisend_tx_hash

    def _get_step_transactions_data(
        self, route: Dict[str, Any]
    ) -> Generator[None, None, Optional[List[Any]]]:
        step_transactions = []
        steps = route.get("steps", [])
        for step in steps:
            step = self._set_step_addresses(step)
            tx_info = yield from self._get_step_transaction(step)
            step_transactions.append(tx_info)

        return step_transactions

    def _get_step_transaction(
        self, step: Dict[str, Any]
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get transaction data for a step from LiFi API"""
        base_url = self.params.lifi_fetch_step_transaction_url
        response = yield from self.get_http_response(
            "POST",
            base_url,
            json.dumps(step).encode(),
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",  # Ensure the correct content type
            },
        )

        if response.status_code not in HTTP_OK:
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

        estimate = response.get("estimate", {})
        fee_costs = estimate.get("feeCosts", [])
        gas_costs = estimate.get("gasCosts", [])
        fee = 0
        gas_cost = 0
        fee += sum(float(fee_cost.get("amountUSD", 0)) for fee_cost in fee_costs)
        gas_cost += sum(float(gas_cost.get("amountUSD", 0)) for gas_cost in gas_costs)

        from_amount_usd = float(response.get("fromAmountUSD", 0))
        to_amount_usd = float(response.get("toAmountUSD", 0))
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
            "fee": fee,
            "gas_cost": gas_cost,
            "from_amount_usd": from_amount_usd,
            "to_amount_usd": to_amount_usd,
        }

    def _set_step_addresses(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Set the fromAddress and toAddress in the step action. Lifi response had mixed up address, temporary solution to fix it"""
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
        # lifi response had mixed up address, temporary solution to fix it
        step["action"]["fromAddress"] = self.params.safe_contract_addresses.get(
            from_chain
        )
        step["action"]["toAddress"] = self.params.safe_contract_addresses.get(to_chain)

        return step

    def fetch_routes(
        self, positions, action
    ) -> Generator[None, None, Optional[List[Any]]]:
        """Get transaction data for route from LiFi API"""

        def round_down_amount(amount: int, decimals: int) -> int:
            """Round down the amount to the nearest round_factor to avoid API rounding issues."""
            if decimals == 18:
                # For tokens like ETH/WETH with 18 decimals, round to nearest 1000 wei
                round_factor = 1000
                rounded_amount = (amount // round_factor) * round_factor
                return rounded_amount
            else:
                return amount

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
        from_address = self.params.safe_contract_addresses.get(from_chain)
        to_address = self.params.safe_contract_addresses.get(to_chain)

        amount = self._get_balance(from_chain, from_token_address, positions)
        token_decimals = ERC20_DECIMALS
        if from_token_address != ZERO_ADDRESS:
            token_decimals = yield from self._get_token_decimals(
                from_chain, from_token_address
            )

        amount = round_down_amount(amount, token_decimals)
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
                "integrator": INTEGRATOR,
                "slippage": slippage,
                "allowSwitchChain": allow_switch_chain,
                "bridges": {"deny": ["stargateV2Bus"]},
            },
        }

        if any(value is None for key, value in params.items()):
            self.context.logger.error(f"Missing value in params: {params}")
            return None

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
            return None

        try:
            routes_response = json.loads(routes_response.body)
        except (ValueError, TypeError) as e:
            self.context.logger.error(
                f"Could not parse response from api, "
                f"the following error was encountered {type(e).__name__}: {e}"
            )
            return None

        routes = routes_response.get("routes", [])
        if not routes:
            self.context.logger.error("No routes available for this pair")
            return None

        return routes

    def _simulate_transaction(
        self,
        to_address: str,
        data: bytes,
        token: str,
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

        body = {
            "simulations": [
                {
                    "network_id": self.params.chain_to_chain_id_mapping.get(chain),
                    "from": self.context.agent_address,
                    "to": safe_address,
                    "simulation_type": "quick",
                    "input": tx_data,
                }
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

        if response.status_code not in HTTP_OK:
            self.context.logger.error(
                f"Could not retrieve data from url {api_url}. Status code {response.status_code}. Error Message {response.body}"
            )
            return False

        try:
            data = json.loads(response.body)
            if data:
                simulation_results = data.get("simulation_results", [])
                status = False
                if simulation_results:
                    for simulation_result in simulation_results:
                        simulation = simulation_result.get("simulation", {})
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
        )

        self.context.logger.info(f"Tx hash payload string is {payload_string}")

        return payload_string, chain, safe_address

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
    """Behaviour that is executed after a tx is settled via the transaction_settlement_abci."""

    matching_round = PostTxSettlementRound

    def async_act(self) -> Generator:
        """Simply log that a tx is settled and wait for the round end."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            msg = f"The transaction submitted by {self.synchronized_data.tx_submitter} was successfully settled."
            self.context.logger.info(msg)
            # we do not want to track the gas costs for vanity tx
            if (
                not self.synchronized_data.tx_submitter
                == CheckStakingKPIMetRound.auto_round_id()
            ):
                yield from self.fetch_and_log_gas_details()

            payload = PostTxSettlementPayload(
                sender=self.context.agent_address, content="Transaction settled"
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()

    def fetch_and_log_gas_details(self):
        """Fetch the transaction receipt and log the gas price and cost."""
        tx_hash = self.synchronized_data.final_tx_hash
        chain = self.synchronized_data.chain_id

        response = yield from self.get_transaction_receipt(
            tx_digest=tx_hash,
            chain_id=chain,
        )
        if not response:
            self.context.logger.error(
                f"Error fetching tx receipt! Response: {response}"
            )
            return

        effective_gas_price = response.get("effectiveGasPrice")
        gas_used = response.get("gasUsed")
        if gas_used and effective_gas_price:
            self.context.logger.info(
                f"Gas Details - Effective Gas Price: {effective_gas_price}, Gas Used: {gas_used}"
            )
            timestamp = int(
                self.round_sequence.last_round_transition_timestamp.timestamp()
            )
            chain_id = self.params.chain_to_chain_id_mapping.get(chain)
            if not chain_id:
                self.context.logger.error(f"No chain id found for chain {chain}")
                return
            self.gas_cost_tracker.log_gas_usage(
                str(chain_id), timestamp, tx_hash, gas_used, effective_gas_price
            )
            self.store_gas_costs()
            return
        else:
            self.context.logger.warning(
                "Gas used or effective gas price not found in the response."
            )

class DecideAgentBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that executes all the actions."""

    matching_round: Type[AbstractRound] = DecideAgentRound

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            if self.params.agent_transition is True:
                next_event = Event.MOVE_TO_NEXT_AGENT
            else:
                next_event = Event.DONT_MOVE_TO_NEXT_AGENT
                    
            payload = DecideAgentPayload(
                sender=sender,
                content=json.dumps(
                    {
                        "event": next_event
                    }
                ),
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
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
        DecideAgentBehaviour,
        PostTxSettlementBehaviour,
    ]


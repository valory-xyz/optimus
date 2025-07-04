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

"""This module contains the base behaviour for the 'liquidity_trader_abci' skill."""

import json
import logging
import math
import types
from abc import ABC
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, cast

from aea.configurations.data_types import PublicId
from aea.protocols.base import Message
from eth_utils import to_checksum_address

from packages.dvilela.connections.kv_store.connection import (
    PUBLIC_ID as KV_STORE_CONNECTION_PUBLIC_ID,
)
from packages.dvilela.protocols.kv_store.dialogues import (
    KvStoreDialogue,
    KvStoreDialogues,
)
from packages.dvilela.protocols.kv_store.message import KvStoreMessage
from packages.valory.connections.mirror_db.connection import (
    PUBLIC_ID as MIRRORDB_CONNECTION_PUBLIC_ID,
)
from packages.valory.contracts.erc20.contract import ERC20
from packages.valory.contracts.staking_activity_checker.contract import (
    StakingActivityCheckerContract,
)
from packages.valory.contracts.staking_token.contract import StakingTokenContract
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.protocols.srr.dialogues import SrrDialogue, SrrDialogues
from packages.valory.protocols.srr.message import SrrMessage
from packages.valory.skills.abstract_round_abci.models import Requests
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
from packages.valory.skills.liquidity_trader_abci.pools.velodrome import (
    VelodromePoolBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.states.base import (
    StakingState,
    SynchronizedData,
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
CAMPAIGN_TYPES = [1, 2]
INTEGRATOR = "valory"
WAITING_PERIOD_FOR_BALANCE_TO_REFLECT = 5
MAX_STEP_COST_RATIO = 0.5
WaitableConditionType = Generator[None, None, Any]
HTTP_NOT_FOUND = [400, 404]
ERC20_DECIMALS = 18
AGENT_TYPE = {"mode": "Modius", "optimism": "Optimus"}
METRICS_NAME = "APR"
METRICS_TYPE = "json"
PORTFOLIO_UPDATE_INTERVAL = 3600 * 2  # 2hr
APR_UPDATE_INTERVAL = 3600 * 24  # 24hr
METRICS_UPDATE_INTERVAL = 21600  # 6hr
# Initial available amount for ETH (0.005 ETH)
ETH_INITIAL_AMOUNT = int(0.005 * 10**18)
# Key for tracking remaining ETH in kv_store
ETH_REMAINING_KEY = "eth_remaining_amount"
SLEEP_TIME = 10  # Configurable sleep time
RETRIES = 3  # Number of API call retries
MIN_TIME_IN_POSITION = 604800 * 3  # 3 weeks
PRICE_CACHE_KEY_PREFIX = "token_price_cache_"
CACHE_TTL = 3600  # 1 hour in seconds
OLAS_ADDRESSES = {
    "mode": "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9",
    "optimism": "0xFC2E6e6BCbd49ccf3A5f029c79984372DcBFE527",
}


class DexType(Enum):
    """DexType"""

    BALANCER = "balancerPool"
    UNISWAP_V3 = "UniswapV3"
    STURDY = "Sturdy"
    VELODROME = "velodrome"


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
    WITHDRAW = "withdraw"
    DEPOSIT = "deposit"


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


class PositionStatus(Enum):
    """PositionStatus"""

    OPEN = "open"
    CLOSED = "closed"


class TradingType(Enum):
    """TradingType"""

    BALANCED = "balanced"
    RISKY = "risky"


THRESHOLDS = {TradingType.BALANCED.value: 0.3374, TradingType.RISKY.value: 0.2892}

ASSETS_FILENAME = "assets.json"
POOL_FILENAME = "current_pool.json"
READ_MODE = "r"
WRITE_MODE = "w"

# Whitelist of allowed token addresses for each chain with their symbols
WHITELISTED_ASSETS = {
    "mode": {
        # MODE tokens - stablecoins
        "0x4200000000000000000000000000000000000006": "WETH",
        "0xdfc7c877a950e49d2610114102175a06c2e3167a": "MODE",
        "0xcfd1d50ce23c46d3cf6407487b2f8934e96dc8f9": "OLAS",
        "0x2416092f143378750bb29b79ed961ab195cceea5": "ezETH",
        "0x6b2a01a5f79deb4c2f3c0eda7b01df456fbd726a": "uniBTC",
        "0x04C0599Ae5A44757c0af6F9eC3b93da8976c150A": "weETH.mode",
        "0x80137510979822322193FC997d400D5A6C747bf7": "STONE",
        "0xe7903B1F75C534Dd8159b313d92cDCfbC62cB3Cd": "wrsETH",
        "0x8b2EeA0999876AAB1E7955fe01A5D261b570452C": "wMLT",
        "0x66eEd5FF1701E6ed8470DC391F05e27B1d0657eb": "BMX",
        "0x7f9AdFbd38b669F03d1d11000Bc76b9AaEA28A81": "XVELO",
        "0xd988097fb8612cc24eec14542bc03424c656005f": "USDC",
        "0xf0f161fda2712db8b566946122a5af183995e2ed": "USDT",
        "0x1217bfe6c773eec6cc4a38b5dc45b92292b6e189": "oUSDT",
    },
    "optimism": {
        # Optimism tokens - stablecoins
        "0x0b2c639c533813f4aa9d7837caf62653d097ff85": "USDC",
        "0x01bff41798a0bcf287b996046ca68b395dbc1071": "USDT0",
        "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58": "USDT",
        "0x7f5c764cbc14f9669b88837ca1490cca17c31607": "USDC.e",
        "0x8ae125e8653821e851f12a49f7765db9a9ce7384": "DOLA",
        "0xc40f949f8a4e094d1b49a23ea9241d289b7b2819": "LUSD",
        "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1": "DAI",
        "0x087c440f251ff6cfe62b86dde1be558b95b4bb9b": "BOLD",
        "0x2e3d870790dc77a83dd1d18184acc7439a53f475": "FRAX",
        "0x2218a117083f5b482b0bb821d27056ba9c04b1d3": "sDAI",
        "0x1217bfe6c773eec6cc4a38b5dc45b92292b6e189": "oUSDT",
        "0x4f604735c1cf31399c6e711d5962b2b3e0225ad3": "USDGLO",
    },
}

COIN_ID_MAPPING = {
    "mode": {
        "usdc": "mode-bridged-usdc-mode",
        "msdai": None,
        "usdt": "mode-bridged-usdt-mode",
        "ousdt": "openusdt",
        "weth": "l2-standard-bridged-weth-modee",
        "ezeth": "renzo-restaked-eth",
        "mode": "mode",
        "olas": "autonolas",
        "unibtc": "universal-btc",
        "weeth.mode": None,
        "stone": "stakestone-ether",
        "wrseth": "wrapped-rseth",
        "wmlt": "bmx-wrapped-mode-liquidity-token",
        "bmx": "bmx",
        "xvelo": None,
        "iusdc": "ironclad-usd",
    },
    "optimism": {
        "usdc": "usd-coin",
        "alusd": "alchemix-usd",
        "usdt0": "usdt0",
        "usdt": "bridged-usdt",
        "msusd": None,
        "usdc.e": "bridged-usd-coin-optimism",
        "usx": "token-dforce-usd",
        "dola": "dola-usd",
        "lusd": "liquity-usd",
        "dai": "makerdao-optimism-bridged-dai-optimism",
        "bold": "liquity-bold",
        "frax": "frax",
        "sdai": "savings-dai",
        "usd+": "overnight-fi-usd-optimism",
        "ousdt": "openusdt",
        "usdglo": "glo-dollar",
        "iusdc": "ironclad-usd",
    },
}


class LiquidityTraderBaseBehaviour(
    BalancerPoolBehaviour, UniswapPoolBehaviour, VelodromePoolBehaviour, ABC
):
    """Base behaviour for the liquidity_trader_abci skill."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize `LiquidityTraderBaseBehaviour`."""
        super().__init__(**kwargs)
        self.assets: Dict[str, Any] = {}
        # TO-DO: this will not work if we run it as a service
        self.assets_filepath = self.params.store_path / self.params.assets_info_filename
        self.current_positions: List[Dict[str, Any]] = []
        self.current_positions_filepath: str = (
            self.params.store_path / self.params.pool_info_filename
        )
        self.portfolio_data: Dict[str, Any] = {}
        self.portfolio_data_filepath: str = (
            self.params.store_path / self.params.portfolio_info_filename
        )
        self.whitelisted_assets: Dict[str, Any] = {}
        self.whitelisted_assets_filepath: str = (
            self.params.store_path / self.params.whitelisted_assets_filename
        )
        self.funding_events: Dict[str, Any] = {}
        self.funding_events_filepath: str = (
            self.params.store_path / self.params.funding_events_filename
        )
        self.pools: Dict[str, Any] = {}
        self.pools[DexType.BALANCER.value] = BalancerPoolBehaviour
        self.pools[DexType.UNISWAP_V3.value] = UniswapPoolBehaviour
        self.pools[DexType.VELODROME.value] = VelodromePoolBehaviour
        self.service_staking_state = StakingState.UNSTAKED
        self._inflight_strategy_req: Optional[str] = None
        self.gas_cost_tracker = GasCostTracker(
            file_path=self.params.store_path / self.params.gas_cost_info_filename
        )
        self.initial_investment_values_per_pool = {}

        # Read the assets and current pool
        self.read_current_positions()
        self.read_assets()
        self.read_gas_costs()
        self.read_portfolio_data()

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
                    decimal = 18
                else:
                    balance = yield from self._get_token_balance(
                        chain, account, asset_address
                    )
                    balance = 0 if balance is None else balance
                    decimal = yield from self._get_token_decimals(chain, asset_address)

                asset_balances_dict[chain].append(
                    {
                        "asset_symbol": asset_symbol,
                        "asset_type": (
                            "native" if asset_address == ZERO_ADDRESS else "erc_20"
                        ),
                        "address": to_checksum_address(asset_address),
                        "balance": balance,
                    }
                )

                self.context.logger.info(
                    f"Balance of account {account} on {chain} for {asset_symbol}: {self._convert_to_token_units(balance, decimal)}"
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
                    if asset.get("address").lower() == token.lower():
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

    def _convert_to_token_units(self, amount: int, token_decimal: int = 18) -> str:
        """Convert smallest unit to token's base unit."""
        if token_decimal is None or amount is None:
            return None

        value = amount / 10**token_decimal
        return f"{value:.{token_decimal}f}"

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
            initial_data = [] if attribute == "current_positions" else {}
            with open(filepath, WRITE_MODE) as file:
                json.dump(initial_data, file)
            return
        except (PermissionError, OSError) as e:
            err = f"Error reading from file {filepath!r}: {str(e)}"

        self.context.logger.error(err)

    def _adjust_current_positions_for_backward_compatibility(
        self, data: Any
    ) -> Generator[None, None, None]:
        """Adjust the 'current_positions' data for backward compatibility and update self.current_positions."""
        adjusted_positions: List[Dict[str, Any]] = []

        if isinstance(data, dict):
            data = [data]

        if isinstance(data, list):
            # Backward compatibility adjustments for each position
            for position in data:
                if "address" in position:
                    position["pool_address"] = position.pop("address")
                if "assets" in position:
                    assets = position.pop("assets")
                    if isinstance(assets, list):
                        if len(assets) >= 1:
                            position["token0"] = assets[0]
                            position[
                                "token0_symbol"
                            ] = yield from self._get_token_symbol(
                                position.get("chain"), assets[0]
                            )
                        if len(assets) >= 2:
                            position["token1"] = assets[1]
                            position[
                                "token1_symbol"
                            ] = yield from self._get_token_symbol(
                                position.get("chain"), assets[1]
                            )
                if "status" not in position:
                    position["status"] = PositionStatus.OPEN.value
                adjusted_positions.append(position)

            self.current_positions = adjusted_positions
            self.store_current_positions()
        else:
            self.context.logger.warning("Unexpected data format for current_positions.")
            self.current_positions = []

    def _get_token_symbol(
        self, chain: str, address: str
    ) -> Generator[None, None, Optional[str]]:
        """Fetch the token symbol from the assets data."""
        token_symbol = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=address,
            contract_public_id=ERC20.contract_id,
            contract_callable="get_token_symbol",
            data_key="data",
            chain_id=chain,
        )
        return token_symbol

    def store_assets(self) -> None:
        """Store the list of assets as JSON."""
        self._store_data(self.assets, "assets", self.assets_filepath)

    def read_assets(self) -> None:
        """Read the list of assets as JSON."""
        self._read_data("assets", self.assets_filepath)

    def store_current_positions(self) -> None:
        """Store the current pool as JSON."""
        self._store_data(
            self.current_positions, "current_positions", self.current_positions_filepath
        )

    def read_current_positions(self) -> None:
        """Read the current pool as JSON."""
        self._read_data("current_positions", self.current_positions_filepath)

    def store_whitelisted_assets(self) -> None:
        """Store the list of assets as JSON."""
        self._store_data(
            self.whitelisted_assets,
            "whitelisted_assets",
            self.whitelisted_assets_filepath,
        )

    def read_whitelisted_assets(self) -> None:
        """Read the list of assets as JSON."""
        self._read_data("whitelisted_assets", self.whitelisted_assets_filepath)

    def store_funding_events(self) -> None:
        """Store the list of assets as JSON."""
        self._store_data(
            self.funding_events, "funding_events", self.funding_events_filepath
        )

    def read_funding_events(self) -> None:
        """Read the list of assets as JSON."""
        self._read_data("funding_events", self.funding_events_filepath)

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

    def store_portfolio_data(self) -> None:
        """Store the portfolio data as JSON."""
        self._store_data(
            self.portfolio_data, "portfolio_data", self.portfolio_data_filepath
        )

    def read_portfolio_data(self) -> None:
        """Read the portfolio data from JSON."""
        self._read_data("portfolio_data", self.portfolio_data_filepath)

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

        last_ts_checkpoint = yield from self._get_ts_checkpoint(
            chain=self.params.staking_chain
        )
        if last_ts_checkpoint is None:
            return None

        min_num_of_safe_tx_required = (
            math.ceil(
                max(liveness_period, (current_timestamp - last_ts_checkpoint))
                * liveness_ratio
                / LIVENESS_RATIO_SCALE_FACTOR
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
                chain=self.params.staking_chain,
                multisig=self.params.safe_contract_addresses.get(
                    self.params.staking_chain
                ),
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

    def _fetch_token_price(
        self, token_address: str, chain: str
    ) -> Generator[None, None, Optional[float]]:
        """Fetch the price for a specific token, with in-memory caching."""
        timestamp = int(self._get_current_timestamp())
        date_str = datetime.utcfromtimestamp(timestamp).strftime("%d-%m-%Y")

        cached_price = yield from self._get_cached_price(token_address, date_str)
        if cached_price is not None:
            return cached_price

        headers = {
            "Accept": "application/json",
        }
        if self.coingecko.api_key:
            headers["x-cg-api-key"] = self.coingecko.api_key

        platform_id = self.coingecko.chain_to_platform_id_mapping.get(chain)
        if not platform_id:
            self.context.logger.error(f"Missing platform id for chain {chain}")
            return None

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
            # Cache the price
            if price:
                yield from self._cache_price(token_address, price, date_str)
            return price

        return None

    def _get_price_cache_key(
        self, token_address: str, date: Optional[str] = None
    ) -> str:
        """Get the cache key for a token's price data."""
        key = f"{PRICE_CACHE_KEY_PREFIX}{token_address.lower()}_{date}"
        return key

    def _get_cached_price(
        self, token_address: str, date: str
    ) -> Generator[None, None, Optional[float]]:
        """Get cached price for a token."""
        cache_key = self._get_price_cache_key(token_address, date)
        result = yield from self._read_kv((cache_key,))

        if not result or not result.get(cache_key):
            return None

        try:
            price_data = json.loads(result[cache_key])
            if date:
                # For historical price
                return price_data.get(date)
            else:
                # For current price, use "current" as key
                current_data = price_data.get("current")
                if not current_data:
                    return None
                price, timestamp = current_data
                if self._get_current_timestamp() - timestamp < CACHE_TTL:
                    return price
            return None
        except (json.JSONDecodeError, TypeError):
            self.context.logger.error(f"Invalid cache data for token {token_address}")
            return None

    def _cache_price(
        self, token_address: str, price: float, date: str
    ) -> Generator[None, None, None]:
        """Cache price for a token."""
        cache_key = self._get_price_cache_key(token_address, date)

        # First read existing cache
        result = yield from self._read_kv((cache_key,))
        price_data = {}

        if result and result.get(cache_key):
            try:
                price_data = json.loads(result[cache_key])
            except json.JSONDecodeError:
                self.context.logger.error(
                    f"Invalid cache data for token {token_address}, resetting cache"
                )

        if date:
            # For historical price
            price_data[date] = price
        else:
            # For current price, store with timestamp
            price_data["current"] = (price, self._get_current_timestamp())

        yield from self._write_kv({cache_key: json.dumps(price_data)})

    def _calculate_rate_limit_wait_time(self) -> int:
        """Calculate the wait time for rate limiting based on the rate limiter state."""
        if not hasattr(self.coingecko, "rate_limiter"):
            return 0

        rate_limiter = self.coingecko.rate_limiter

        # If we have no credits left, return 0 (will be handled by caller)
        if rate_limiter.no_credits:
            return 0

        # If we're rate limited, calculate time until next minute
        if rate_limiter.rate_limited:
            from time import time

            current_time = time()
            time_since_last_request = current_time - rate_limiter.last_request_time

            # Wait for the remainder of the current minute
            wait_time = max(0, 60 - int(time_since_last_request))
            return min(wait_time, 60)  # Cap at 60 seconds

        return 0

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
        """Request wrapped around a retry mechanism, now also retries on HTTP 503 (Service Unavailable) with exponential backoff."""

        self.context.logger.info(f"HTTP {method} call: {endpoint}")
        content = json.dumps(body).encode(UTF8) if body else None

        # Add delay before CoinGecko API calls to respect rate limits (20 calls/minute = 3 seconds between calls)
        if "coingecko.com" in endpoint:
            self.context.logger.warning(
                "Adding 2-second delay for CoinGecko API rate limiting"
            )
            yield from self.sleep(2)

        retries = 0
        backoff = 2  # seconds, for exponential backoff on 503

        while True:
            # Make the request
            response = yield from self.get_http_response(
                method, endpoint, content, headers
            )

            try:
                response_json = json.loads(response.body)
            except json.decoder.JSONDecodeError as exc:
                self.context.logger.error(f"Exception during json loading: {exc}")
                self.context.logger.info(f"Received response: {response}")
                response_json = {"exception": str(exc)}

            # Handle rate limiting as a retryable error
            if response.status_code == rate_limited_code:
                self.context.logger.warning(
                    f"Rate limited (attempt {retries + 1}/{max_retries})"
                )
                rate_limited_callback()
                retries += 1
                if retries >= max_retries:
                    self.context.logger.error(
                        f"Request failed after {retries} rate limit retries."
                    )
                    return False, response_json

                # Wait 60 seconds for rate limit to reset
                self.context.logger.info(
                    "Waiting 60 seconds before retrying rate-limited request"
                )
                yield from self.sleep(60)
                continue

            # Handle HTTP 503 Service Unavailable with exponential backoff
            if response.status_code == 503:
                self.context.logger.warning(
                    f"503 Service Unavailable (attempt {retries + 1}/{max_retries}). Retrying in {backoff} seconds."
                )
                retries += 1
                if retries >= max_retries:
                    self.context.logger.error(
                        f"Request failed after {retries} retries due to repeated 503 errors."
                    )
                    return False, response_json
                yield from self.sleep(backoff)
                backoff *= 2  # Exponential backoff
                continue

            if response.status_code not in HTTP_OK or "exception" in response_json:
                self.context.logger.error(
                    f"Request failed [{response.status_code}]: {response_json}"
                )
                retries += 1
                if retries >= max_retries:
                    break
                yield from self.sleep(retry_wait)
                continue

            self.context.logger.info("Request succeeded.")
            return True, response_json

        self.context.logger.error(f"Request failed after {retries} retries.")
        return False, response_json

    def _do_connection_request(
        self,
        message: Message,
        dialogue: Message,
        timeout: Optional[float] = None,
    ) -> Generator[None, None, Message]:
        """Do a request and wait the response, asynchronously."""

        self.context.outbox.put_message(message=message)
        request_nonce = self._get_request_nonce_from_dialogue(dialogue)  # type: ignore
        cast(Requests, self.context.requests).request_id_to_callback[
            request_nonce
        ] = self.get_callback_request()
        response = yield from self.wait_for_message(timeout=timeout)
        return response

    def _call_mirrordb(self, method: str, **kwargs: Any) -> Generator[None, None, Any]:
        """Send a request message to the MirrorDB connection."""
        try:
            srr_dialogues = cast(SrrDialogues, self.context.srr_dialogues)
            srr_message, srr_dialogue = srr_dialogues.create(
                counterparty=str(MIRRORDB_CONNECTION_PUBLIC_ID),
                performative=SrrMessage.Performative.REQUEST,
                payload=json.dumps({"method": method, "kwargs": kwargs}),
            )
            srr_message = cast(SrrMessage, srr_message)
            srr_dialogue = cast(SrrDialogue, srr_dialogue)
            response = yield from self._do_connection_request(srr_message, srr_dialogue)  # type: ignore

            response_json = json.loads(response.payload)  # type: ignore

            if "error" in response_json:
                self.context.logger.error(response_json["error"])
                return None

            return response_json.get("response")  # type: ignore
        except Exception as e:  # pylint: disable=broad-except
            self.context.logger.error(f"Exception while calling MirrorDB: {e}")
            return None

    def _read_kv(
        self,
        keys: Tuple[str, ...],
    ) -> Generator[None, None, Optional[Dict]]:
        """Send a request message from the skill context."""
        self.context.logger.info(f"Reading keys from db: {keys}")
        kv_store_dialogues = cast(KvStoreDialogues, self.context.kv_store_dialogues)
        kv_store_message, srr_dialogue = kv_store_dialogues.create(
            counterparty=str(KV_STORE_CONNECTION_PUBLIC_ID),
            performative=KvStoreMessage.Performative.READ_REQUEST,
            keys=keys,
        )
        kv_store_message = cast(KvStoreMessage, kv_store_message)
        kv_store_dialogue = cast(KvStoreDialogue, srr_dialogue)
        response = yield from self._do_connection_request(
            kv_store_message, kv_store_dialogue  # type: ignore
        )
        if response.performative != KvStoreMessage.Performative.READ_RESPONSE:
            return None

        data = {key: response.data.get(key, None) for key in keys}  # type: ignore

        return data

    def _write_kv(
        self,
        data: Dict[str, str],
    ) -> Generator[None, None, bool]:
        """Send a request message from the skill context."""
        kv_store_dialogues = cast(KvStoreDialogues, self.context.kv_store_dialogues)
        kv_store_message, srr_dialogue = kv_store_dialogues.create(
            counterparty=str(KV_STORE_CONNECTION_PUBLIC_ID),
            performative=KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST,
            data=data,
        )
        kv_store_message = cast(KvStoreMessage, kv_store_message)
        kv_store_dialogue = cast(KvStoreDialogue, srr_dialogue)
        response = yield from self._do_connection_request(
            kv_store_message, kv_store_dialogue  # type: ignore
        )
        return response == KvStoreMessage.Performative.SUCCESS

    def _fetch_token_prices(
        self, token_balances: List[Dict[str, Any]]
    ) -> Generator[None, None, Dict[str, float]]:
        """Fetch token prices from Coingecko"""
        token_prices = {}

        for token_data in token_balances:
            token_address = token_data["token"]
            chain = token_data.get("chain")
            if not chain:
                self.context.logger.error(f"Missing chain for token {token_address}")
                continue

            if token_address == ZERO_ADDRESS:
                price = yield from self._fetch_zero_address_price()
            else:
                price = yield from self._fetch_token_price(token_address, chain)

            if price is not None:
                token_prices[token_address] = price

        return token_prices

    def _fetch_zero_address_price(self) -> Generator[None, None, Optional[float]]:
        """Fetch the price for the zero address (Ethereum)."""
        timestamp = int(self._get_current_timestamp())
        date_str = datetime.utcfromtimestamp(timestamp).strftime("%d-%m-%Y")

        cached_price = yield from self._get_cached_price(ZERO_ADDRESS, date_str)
        if cached_price is not None:
            return cached_price

        headers = {
            "Accept": "application/json",
        }
        if self.coingecko.api_key:
            headers["x-cg-api-key"] = self.coingecko.api_key

        success, response_json = yield from self._request_with_retries(
            endpoint=self.coingecko.coin_price_endpoint.format(coin_id="ethereum"),
            headers=headers,
            rate_limited_code=self.coingecko.rate_limited_code,
            rate_limited_callback=self.coingecko.rate_limited_status_callback,
            retry_wait=self.params.sleep_time,
        )

        if success:
            token_data = next(iter(response_json.values()), {})
            price = token_data.get("usd", 0)
            if price:
                yield from self._cache_price(ZERO_ADDRESS, price, date_str)
            return price
        return None

    def _get_current_timestamp(self) -> int:
        return cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()

    def calculate_initial_investment_value(
        self, position: Dict[str, Any]
    ) -> Generator[None, None, Optional[float]]:
        """Calculate the initial investment value based on the initial transaction."""
        chain = position.get("chain")
        token0 = position.get("token0")
        token1 = position.get("token1")
        amount0 = position.get("amount0")
        amount1 = position.get("amount1")
        timestamp = position.get("timestamp") or position.get("enter_timestamp")

        if None in (token0, amount0, timestamp):
            self.context.logger.error(
                "Missing token0, amount0, or timestamp in position data."
            )
            return None

        token0_decimals = yield from self._get_token_decimals(chain, token0)
        if not token0_decimals:
            return None

        initial_amount0 = amount0 / (10**token0_decimals)
        self.context.logger.info(f"initial_amount0 : {initial_amount0}")

        if token1 is not None and amount1 is not None:
            token1_decimals = yield from self._get_token_decimals(chain, token1)
            if not token1_decimals:
                return None
            initial_amount1 = amount1 / (10**token1_decimals)
            self.context.logger.info(f"initial_amount1 : {initial_amount1}")

        date_str = datetime.utcfromtimestamp(timestamp).strftime("%d-%m-%Y")
        tokens = []
        # Fetch historical prices
        tokens.append([position.get("token0_symbol"), position.get("token0")])
        if position.get("token1") is not None:
            tokens.append([position.get("token1_symbol"), position.get("token1")])

        historical_prices = yield from self._fetch_historical_token_prices(
            tokens, date_str, chain
        )

        if not historical_prices:
            self.context.logger.error("Failed to fetch historical token prices.")
            return None

        # Get the price for token0
        initial_price0 = historical_prices.get(position.get("token0"))
        if initial_price0 is None:
            self.context.logger.error("Historical price not found for token0.")
            return None

        # Calculate initial investment value for token0
        V_initial = initial_amount0 * initial_price0
        self.context.logger.info(f"V_initial : {V_initial}")

        # If token1 exists, include it in the calculations
        if position.get("token1") is not None and initial_amount1 is not None:
            initial_price1 = historical_prices.get(position.get("token1"))
            if initial_price1 is None:
                self.context.logger.error("Historical price not found for token1.")
                return None
            V_initial += initial_amount1 * initial_price1
            self.context.logger.info(f"V_initial : {V_initial}")

        return V_initial

    def calculate_initial_investment(self) -> Generator[None, None, Optional[float]]:
        """Calculate the initial investment value for all open positions."""
        initial_value = 0.0
        for position in self.current_positions:
            if position.get("status") == PositionStatus.OPEN.value:
                position_value = yield from self.calculate_initial_investment_value(
                    position
                )
                self.context.logger.info(f"Position value: {position_value}")
                if position_value is not None:
                    initial_value += float(position_value)
                    pool_id = position.get("pool_address", position.get("pool_id"))
                    tx_hash = position.get("tx_hash")
                    position_key = f"{pool_id}_{tx_hash}"
                    self.initial_investment_values_per_pool[
                        position_key
                    ] = position_value

                else:
                    self.context.logger.warning(
                        f"Skipping position with null value: {position.get('id', 'unknown')}"
                    )

        self.context.logger.info(f"Total initial investment value: {initial_value}")
        if initial_value <= 0:
            self.context.logger.warning("Initial value is zero or negative")
            return None

        return initial_value

    def _fetch_historical_token_prices(
        self, tokens: List[List[str]], date_str: str, chain: str
    ) -> Generator[None, None, Dict[str, float]]:
        """Fetch historical token prices for a specific date."""
        historical_prices = {}

        for token_symbol, token_address in tokens:
            # Get CoinGecko ID.
            coingecko_id = self.get_coin_id_from_symbol(token_symbol, chain)
            if not coingecko_id:
                self.context.logger.error(
                    f"CoinGecko ID not found for token {token_address} with symbol {token_symbol}."
                )
                continue

            price = yield from self._fetch_historical_token_price(
                coingecko_id, date_str
            )
            if price:
                historical_prices[token_address] = price

        return historical_prices

    def _fetch_historical_token_price(
        self, coingecko_id, date_str
    ) -> Generator[None, None, Optional[float]]:
        # First check the cache
        cached_price = yield from self._get_cached_price(coingecko_id, date_str)
        if cached_price is not None:
            return cached_price

        endpoint = self.coingecko.historical_price_endpoint.format(
            coin_id=coingecko_id,
            date=date_str,
        )

        headers = {"Accept": "application/json"}
        if self.coingecko.api_key:
            headers["x-cg-api-key"] = self.coingecko.api_key

        success, response_json = yield from self._request_with_retries(
            endpoint=endpoint,
            headers=headers,
            rate_limited_code=self.coingecko.rate_limited_code,
            rate_limited_callback=self.coingecko.rate_limited_status_callback,
            retry_wait=self.params.sleep_time,
        )

        if success:
            price = (
                response_json.get("market_data", {}).get("current_price", {}).get("usd")
            )
            if price:
                # Cache the historical price
                yield from self._cache_price(coingecko_id, price, date_str)
                return price
            else:
                self.context.logger.error(
                    f"No price in response for token {coingecko_id}"
                )
                return None
        else:
            self.context.logger.error(
                f"Failed to fetch historical price for {coingecko_id}"
            )
            return None

    def _fetch_token_name_from_contract(
        self, chain: str, token_address: str
    ) -> Generator[None, None, Optional[str]]:
        """Fetch the token name from the ERC20 contract."""

        token_name = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=token_address,
            contract_public_id=ERC20.contract_id,
            contract_callable="get_name",
            data_key="data",
            chain_id=chain,
        )
        return token_name

    def get_coin_id_from_symbol(self, symbol, chain_name) -> Optional[str]:
        """Retrieve the CoinGecko token ID using the token's address, symbol, and chain name."""
        # Check if coin_list is valid
        symbol = symbol.lower()
        if symbol in COIN_ID_MAPPING.get(chain_name, {}):
            self.context.logger.info(f"Found coin id for {symbol} in {chain_name}")
            return COIN_ID_MAPPING[chain_name][symbol]

        return None

    def get_eth_remaining_amount(self) -> Generator[None, None, int]:
        """Get the remaining ETH amount for swaps from kv_store."""
        result = yield from self._read_kv((ETH_REMAINING_KEY,))
        if not result or not result.get(ETH_REMAINING_KEY):
            # If not found in kv_store, initialize it
            amount = yield from self.reset_eth_remaining_amount()
            return int(amount)

        try:
            chain = self.params.target_investment_chains[0]
            account = self.params.safe_contract_addresses.get(chain)
            amount = yield from self._get_native_balance(chain, account)
            if amount:
                return min(int(result[ETH_REMAINING_KEY]), amount)

            return int(result[ETH_REMAINING_KEY])
        except (ValueError, TypeError):
            self.context.logger.error(
                f"Invalid ETH remaining amount in kv_store: {result[ETH_REMAINING_KEY]}"
            )
            amount = yield from self.reset_eth_remaining_amount()
            return int(amount)

    def update_eth_remaining_amount(
        self, amount_used: int
    ) -> Generator[None, None, None]:
        """Update the remaining ETH amount after a swap in kv_store."""
        current_remaining = yield from self.get_eth_remaining_amount()
        new_remaining = max(0, current_remaining - amount_used)
        self.context.logger.info(
            f"Updating ETH remaining amount in kv_store: {current_remaining} -> {new_remaining}"
        )
        yield from self._write_kv({ETH_REMAINING_KEY: str(new_remaining)})

    def reset_eth_remaining_amount(self) -> Generator[None, None, int]:
        """Reset the remaining ETH amount to the initial value in kv_store."""
        chain = self.params.target_investment_chains[0]
        account = self.params.safe_contract_addresses.get(chain)
        amount = yield from self._get_native_balance(chain, account)
        if amount:
            amount = min(int(ETH_INITIAL_AMOUNT), amount)
        else:
            amount = 0
        self.context.logger.info(
            f"Resetting ETH remaining amount in kv_store to {amount}"
        )
        yield from self._write_kv({ETH_REMAINING_KEY: str(amount)})
        return amount


def execute_strategy(
    strategy: str, strategies_executables: Dict[str, Tuple[str, str]], **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Execute the strategy and return the results."""
    # Reconstruct the logger
    logger = logging.getLogger(__name__)

    strategy_exec_tuple = strategies_executables.get(strategy, None)
    if strategy_exec_tuple is None:
        logger.error(f"No executable was found for {strategy=}!")
        return None

    strategy_exec, callable_method = strategy_exec_tuple
    if callable_method in globals():
        del globals()[callable_method]

    # Execute the strategy code
    exec(strategy_exec, globals())  # pylint: disable=W0122  # nosec
    method = globals().get(callable_method, None)
    if method is None:
        logger.error(
            f"No {callable_method!r} method was found in {strategy} executable."
        )
        return None

    # Call the method and collect results if it's a generator
    result = method(**kwargs)
    if isinstance(result, types.GeneratorType):
        result = list(result)
    return result


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

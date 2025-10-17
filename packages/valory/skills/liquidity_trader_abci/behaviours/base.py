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
import time
import types
from abc import ABC
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
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
from packages.valory.contracts.velodrome_slipstream_helper.contract import (
    VelodromeSlipstreamHelperContract,
)
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
PORTFOLIO_UPDATE_INTERVAL = 3600 * 1  # 2hr
APR_UPDATE_INTERVAL = 3600 * 24  # 24hr
METRICS_UPDATE_INTERVAL = 21600  # 6hr
# Initial available amount for ETH (0.005 ETH)
ETH_INITIAL_AMOUNT = int(0.005 * 10**18)
# Key for tracking remaining ETH in kv_store
ETH_REMAINING_KEY = "eth_remaining_amount"
SLEEP_TIME = 10  # Configurable sleep time
RETRIES = 3  # Number of API call retries
MIN_TIME_IN_POSITION = 21.0  # 3 weeks
PRICE_CACHE_KEY_PREFIX = "token_price_cache_"
CACHE_TTL = 3600  # 1 hour in seconds
REWARD_UPDATE_INTERVAL = 12 * 3600  # 12 hours
REWARD_UPDATE_KEY_PREFIX = "last_reward_update_"
LAST_EPOCH_KEY_PREFIX = "last_processed_epoch_"
OLAS_ADDRESSES = {
    "mode": "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9",
    "optimism": "0xFC2E6e6BCbd49ccf3A5f029c79984372DcBFE527",
}

# Airdrop reward tracking constants
AIRDROP_REWARDS_KEY_PREFIX = "airdrop_rewards_"
AIRDROP_TOTAL_KEY = "total_airdrop_rewards"

# Reward tokens that should be excluded from investment consideration
REWARD_TOKEN_ADDRESSES = {
    "mode": {
        "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9": "OLAS",  # OLAS on Mode
    },
    "optimism": {
        "0xFC2E6e6BCbd49ccf3A5f029c79984372DcBFE527": "OLAS",  # OLAS on Optimism
    },
}

# Round timeout constants for health check
# These timeouts define how long specific rounds can run before being considered unhealthy
# Used to prevent false restarts when the agent is performing legitimate long-running operations
EVALUATE_STRATEGY_TIMEOUT = (
    300  # 5 minutes - time for executing all strategies in parallel
)
FETCH_STRATEGIES_TIMEOUT = (
    300  # 5 minutes - time for downloading strategy files from IPFS
)
DECISION_MAKING_TIMEOUT = 180  # 3 minutes - time for executing on-chain transactions


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
    STAKE_LP_TOKENS = "StakeLpTokens"
    UNSTAKE_LP_TOKENS = "UnstakeLpTokens"
    CLAIM_STAKING_REWARDS = "ClaimStakingRewards"


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


class Chain(Enum):
    """Chain Names"""

    OPTIMISM = "optimism"
    MODE = "mode"


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
        "0x7f9AdFbd38b669F03d1d11000Bc76b9AaEA28A81": "VELO",
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
        "velo": "velodrome-finance",  # VELO can be bridged back 1:1 for native VELO on OP
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
        "velo": "velodrome-finance",
    },
}

# Reward tokens that should be excluded from investment consideration
REWARD_TOKEN_ADDRESSES = {
    "mode": {
        "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9": "OLAS",  # OLAS on Mode
        "0x7f9AdFbd38b669F03d1d11000Bc76b9AaEA28A81": "VELO",  # VELO on Mode
        "0xd988097fb8612cc24eeC14542bC03424c656005f": "USDC",  # USDC on Mode (for airdrop exclusion)
    },
    "optimism": {
        "0xFC2E6e6BCbd49ccf3A5f029c79984372DcBFE527": "OLAS",  # OLAS on Optimism
        "0x9560e827aF36c94D2Ac33a39bCE1Fe78631088Db": "XVELO",  # VELO on Optimism
    },
}


class LiquidityTraderBaseBehaviour(
    BalancerPoolBehaviour, UniswapPoolBehaviour, VelodromePoolBehaviour, ABC
):
    """Base behaviour for the liquidity_trader_abci skill."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize `LiquidityTraderBaseBehaviour`."""
        super().__init__(**kwargs)
        self.current_positions: List[Dict[str, Any]] = []
        # Convert store_path to Path object for file operations
        store_path = Path(self.params.store_path)

        self.current_positions_filepath: str = (
            store_path / self.params.pool_info_filename
        )
        self.portfolio_data: Dict[str, Any] = {}
        self.portfolio_data_filepath: str = (
            store_path / self.params.portfolio_info_filename
        )
        self.assets: Dict[str, Any] = {}
        self.whitelisted_assets: Dict[str, Any] = {}
        self.whitelisted_assets_filepath: str = (
            store_path / self.params.whitelisted_assets_filename
        )
        self.funding_events: Dict[str, Any] = {}
        self.funding_events_filepath: str = (
            store_path / self.params.funding_events_filename
        )
        self.agent_performance: Dict[str, Any] = {}
        self.agent_performance_filepath: str = (
            store_path / self.params.agent_performance_filename
        )
        self.pools: Dict[str, Any] = {}
        self.pools[DexType.BALANCER.value] = BalancerPoolBehaviour
        self.pools[DexType.UNISWAP_V3.value] = UniswapPoolBehaviour
        self.pools[DexType.VELODROME.value] = VelodromePoolBehaviour
        self.service_staking_state = StakingState.UNSTAKED
        self._inflight_strategy_req: Optional[str] = None
        self.gas_cost_tracker = GasCostTracker(
            file_path=store_path / self.params.gas_cost_info_filename
        )
        self.initial_investment_values_per_pool = {}
        self._current_entry_costs = 0.0

        # Read the current pool and other data
        self.read_current_positions()
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
        """Get positions using SafeApi for Optimism, Mode Explorer for Mode"""
        all_balances = defaultdict(list)

        for chain in self.params.target_investment_chains:
            if chain == Chain.OPTIMISM.value:
                # Use SafeApi for Optimism
                balances = yield from self._get_optimism_balances_from_safe_api()
            elif chain == Chain.MODE.value:
                # Use Mode Explorer API for Mode
                balances = yield from self._get_mode_balances_from_explorer_api()

            if balances:
                all_balances[chain].extend(balances)

        positions = [
            {"chain": chain, "assets": assets} for chain, assets in all_balances.items()
        ]

        return positions

    def _get_optimism_balances_from_safe_api(
        self,
    ) -> Generator[None, None, List[Dict[str, Any]]]:
        """Get Optimism balances using SafeApi with pagination"""
        safe_address = self.params.safe_contract_addresses.get("optimism")
        if not safe_address:
            self.context.logger.error("No safe address set for Optimism chain")
            return []

        self.context.logger.info(
            f"Fetching Optimism balances from SafeApi for safe: {safe_address}"
        )

        # Fetch all balances with pagination
        all_balances = yield from self._fetch_safe_balances_with_pagination(
            safe_address
        )

        balances = []
        for balance_data in all_balances:
            token_address = balance_data.get("tokenAddress")
            token_info = balance_data.get("token")
            balance = balance_data.get("balance", "0")

            if token_address is None:
                # Native ETH
                balances.append(
                    {
                        "asset_symbol": "ETH",
                        "asset_type": "native",
                        "address": to_checksum_address(ZERO_ADDRESS),
                        "balance": int(balance),
                    }
                )
            else:
                # ERC-20 token
                if token_info:
                    if (
                        token_address
                        == "0xfAf87e196A29969094bE35DfB0Ab9d0b8518dB84"  # nosec B105
                    ):
                        continue

                    balances.append(
                        {
                            "asset_symbol": token_info.get("symbol", "UNKNOWN"),
                            "asset_type": "erc_20",
                            "address": to_checksum_address(token_address),
                            "balance": int(balance),
                        }
                    )

        self.context.logger.info(
            f"Retrieved {len(balances)} token balances from SafeApi"
        )
        return balances

    def _fetch_safe_balances_with_pagination(
        self, safe_address: str
    ) -> Generator[None, None, List[Dict]]:
        """Fetch all balances from SafeApi with pagination support"""
        all_balances = []
        offset = 0
        limit = 100  # Default page size

        while True:
            url = f"{self.params.safe_api_base_url}/{safe_address}/balances/"
            params = f"?exclude_spam=true&limit={limit}&offset={offset}"
            endpoint = url + params

            self.context.logger.info(
                f"Fetching SafeApi page: offset={offset}, limit={limit}"
            )

            success, response_data = yield from self._request_with_retries(
                endpoint=endpoint,
                method="GET",
                headers={
                    "Accept": "application/json",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
                rate_limited_callback=lambda: None,  # No specific rate limit callback for SafeApi
                max_retries=MAX_RETRIES_FOR_API_CALL,
                retry_wait=2,
            )

            if not success:
                self.context.logger.error(
                    f"Failed to fetch SafeApi data: {response_data}"
                )
                break

            results = response_data.get("results", [])
            if not results:
                self.context.logger.info("No more results from SafeApi")
                break

            all_balances.extend(results)

            # Check if there's a next page
            next_url = response_data.get("next")
            if not next_url:
                self.context.logger.info("Reached last page of SafeApi results")
                break

            offset += limit

        self.context.logger.info(
            f"Total balances fetched from SafeApi: {len(all_balances)}"
        )
        return all_balances

    def _get_mode_balances_from_explorer_api(
        self,
    ) -> Generator[None, None, List[Dict[str, Any]]]:
        """Get Mode balances using Mode Explorer API for ERC-20 tokens and RPC for ETH"""
        safe_address = self.params.safe_contract_addresses.get(Chain.MODE.value)
        if not safe_address:
            self.context.logger.error("No safe address set for Mode chain")
            return []

        self.context.logger.info(
            f"Fetching Mode balances from Explorer API for safe: {safe_address}"
        )

        balances = []

        # Get native ETH balance using RPC
        eth_balance = yield from self._get_native_balance("mode", safe_address)
        if eth_balance and eth_balance > 0:
            balances.append(
                {
                    "asset_symbol": "ETH",
                    "asset_type": "native",
                    "address": to_checksum_address(ZERO_ADDRESS),
                    "balance": eth_balance,
                }
            )

        # Get ERC-20 token balances from Mode Explorer API
        token_balances = yield from self._fetch_mode_token_balances(safe_address)
        balances.extend(token_balances)

        self.context.logger.info(
            f"Retrieved {len(balances)} token balances from Mode Explorer API"
        )
        return balances

    def _fetch_mode_token_balances(
        self, safe_address: str
    ) -> Generator[None, None, List[Dict[str, Any]]]:
        """Fetch ERC-20 token balances from Mode Explorer API with pagination, filtering out LP tokens"""
        balances = []

        # Get active LP token addresses to filter out
        active_lp_addresses = self._get_active_lp_addresses()

        # Fetch all token balances with pagination
        all_tokens = yield from self._fetch_mode_tokens_with_pagination(safe_address)

        # Convert to the expected format and filter out zero balances and LP tokens
        # because the API also returns LP token balances which are also ERC20 tokens and there is no direct filter available
        for token_data in all_tokens:
            token_info = token_data.get("token", {})
            token_address = token_info.get("address")
            token_symbol = token_info.get("symbol", "UNKNOWN")

            # Rename XVELO to VELO for Mode chain
            if token_symbol == "XVELO":  # nosec B105
                token_symbol = "VELO"  # nosec B105
                self.context.logger.info(
                    f"Renamed XVELO to VELO for Mode chain token at {token_address}"
                )

            balance_value = token_data.get("value", "0")

            if not token_address or not balance_value or balance_value == "0":
                continue

            # Filter out LP tokens that correspond to active positions
            if token_address.lower() in active_lp_addresses:
                self.context.logger.info(
                    f"Filtering out LP token {token_symbol} ({token_address}) - active position"
                )
                continue

            try:
                balance = int(balance_value)
                if balance > 0:
                    balances.append(
                        {
                            "asset_symbol": token_symbol,
                            "asset_type": "erc_20",
                            "address": to_checksum_address(token_address),
                            "balance": balance,
                        }
                    )
            except (ValueError, TypeError):
                self.context.logger.warning(
                    f"Invalid balance value for token {token_address}: {balance_value}"
                )
                continue

        return balances

    def _get_active_lp_addresses(self) -> set:
        """Get set of active LP token addresses from current positions"""
        active_lp_addresses = set()

        for position in self.current_positions:
            if position.get("status") == PositionStatus.OPEN.value:
                pool_address = position.get("pool_address")
                if pool_address:
                    active_lp_addresses.add(pool_address.lower())
                    self.context.logger.debug(
                        f"Added active LP address: {pool_address}"
                    )

        self.context.logger.info(
            f"Found {len(active_lp_addresses)} active LP addresses to filter"
        )
        return active_lp_addresses

    def _get_reward_token_addresses(self, chain: str) -> set:
        """Get set of reward token addresses to filter out from investment consideration"""
        reward_addresses = set()

        chain_rewards = REWARD_TOKEN_ADDRESSES.get(chain, {})
        for address, symbol in chain_rewards.items():
            reward_addresses.add(address.lower())
            self.context.logger.debug(
                f"Added reward token address for {chain}: {symbol} ({address})"
            )

        self.context.logger.info(
            f"Found {len(reward_addresses)} reward token addresses to filter for {chain}"
        )
        return reward_addresses

    def _fetch_mode_tokens_with_pagination(
        self, safe_address: str
    ) -> Generator[None, None, List[Dict]]:
        """Fetch all token balances from Mode Explorer API with pagination support"""
        all_tokens = []
        next_page_params = None
        while True:
            # Build endpoint URL
            base_url = f"{self.params.mode_explorer_api_base_url}/api/v2/addresses/{safe_address}/tokens"
            params = "?type=ERC-20"

            if next_page_params:
                # Add pagination parameters if available
                for key, value in next_page_params.items():
                    params += f"&{key}={value}"

            endpoint = base_url + params

            self.context.logger.info(f"Fetching Mode tokens page: {endpoint}")

            success, response_data = yield from self._request_with_retries(
                endpoint=endpoint,
                method="GET",
                headers={
                    "Accept": "application/json",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
                rate_limited_callback=lambda: None,
                max_retries=MAX_RETRIES_FOR_API_CALL,
                retry_wait=2,
            )

            if not success:
                self.context.logger.error(
                    f"Failed to fetch Mode token data: {response_data}"
                )
                break

            items = response_data.get("items", [])
            if not items:
                self.context.logger.info("No more token results from Mode Explorer API")
                break

            all_tokens.extend(items)

            # Check if there's a next page
            next_page_params = response_data.get("next_page_params")
            if not next_page_params:
                self.context.logger.info(
                    "Reached last page of Mode Explorer API results"
                )
                break

        self.context.logger.info(
            f"Total tokens fetched from Mode Explorer API: {len(all_tokens)}"
        )
        return all_tokens

    def _get_balance(
        self, chain: str, token: str, positions: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[int]:
        """Get balance"""
        if not positions:
            return None

        for position in positions:
            if position.get("chain") == chain:
                assets = position.get("assets", {})
                for asset in assets:
                    asset_address = asset.get("address")
                    if asset_address and asset_address.lower() == token.lower():
                        balance = asset.get("balance")
                        return balance

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

                # Add TiP fields for backward compatibility
                if "entry_cost" not in position:
                    position["entry_cost"] = 0.0  # Total USD cost (gas + slippage)
                if "min_hold_days" not in position:
                    # Legacy positions get 21 days (3 weeks) hardcoded if they have enter_timestamp
                    position["min_hold_days"] = (
                        MIN_TIME_IN_POSITION if position.get("enter_timestamp") else 0.0
                    )
                if "cost_recovered" not in position:
                    position["cost_recovered"] = False
                if "principal_usd" not in position:
                    position["principal_usd"] = 0.0

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

    def store_agent_performance(self) -> None:
        """Store the agent performance as JSON."""
        self._store_data(
            self.agent_performance, "agent_performance", self.agent_performance_filepath
        )

    def read_agent_performance(self) -> None:
        """Read the agent performance as JSON."""
        try:
            self._read_data("agent_performance", self.agent_performance_filepath)
            if not self.agent_performance:
                self.initialize_agent_performance()
        except Exception as e:
            self.context.logger.warning(
                f"Failed to read agent performance data: {str(e)}"
            )
            self.initialize_agent_performance()

    def initialize_agent_performance(self) -> None:
        """Initialize agent performance with empty state."""
        self.agent_performance = {
            "timestamp": None,
            "metrics": [],
            "agent_behavior": None,
        }

    def update_agent_performance_timestamp(self) -> None:
        """Update the timestamp in agent performance data."""
        try:
            self.agent_performance["timestamp"] = int(
                datetime.now(timezone.utc).timestamp()
            )
        except Exception as e:
            self.context.logger.error(
                f"Failed to update agent performance timestamp: {str(e)}"
            )

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

        self.context.logger.info(
            f"Fetching current price for token {token_address} on {chain} chain"
        )

        cached_price = yield from self._get_cached_price(token_address, date_str)
        if cached_price is not None:
            self.context.logger.info(
                f"Using cached price for token {token_address}: ${cached_price}"
            )
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

        self.context.logger.info(
            f"Fetching price from CoinGecko API for token {token_address} on platform {platform_id}"
        )

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
                self.context.logger.info(
                    f"Successfully fetched price for token {token_address}: ${price}"
                )
                yield from self._cache_price(token_address, price, date_str)
            else:
                self.context.logger.warning(
                    f"No price data returned for token {token_address}"
                )
            return price
        else:
            self.context.logger.error(
                f"Failed to fetch price for token {token_address} from CoinGecko"
            )

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
        self, coin_id: str, price: float, date: str
    ) -> Generator[None, None, None]:
        """Cache price for a token."""
        cache_key = self._get_price_cache_key(coin_id, date)

        # First read existing cache
        result = yield from self._read_kv((cache_key,))
        price_data = {}

        if result and result.get(cache_key):
            try:
                price_data = json.loads(result[cache_key])
            except json.JSONDecodeError:
                self.context.logger.error(
                    f"Invalid cache data for token {coin_id}, resetting cache"
                )

        if date:
            # For historical price
            price_data[date] = price
        else:
            # For current price, store with timestamp
            price_data["current"] = (price, self._get_current_timestamp())

        yield from self._write_kv(
            {cache_key: json.dumps(price_data, ensure_ascii=True)}
        )

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
        content = json.dumps(body, ensure_ascii=True).encode(UTF8) if body else None

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
                payload=json.dumps(
                    {"method": method, "kwargs": kwargs}, ensure_ascii=True
                ),
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

    def _invalidate_cl_pool_cache(self, chain: str) -> Generator[None, None, None]:
        """Invalidate (delete) the cached CL pool data for a chain.

        Called when enter pool succeeds so we don't keep retrying the same pool.

        :param chain: the chain to invalidate the cache for
        :yield: None: Generator yield for async operations
        """
        try:
            kv_key = f"velodrome_cl_pool_{chain}"

            # Delete the cache by writing an invalidation marker
            yield from self._write_kv(
                {kv_key: json.dumps({"invalidated": True}, ensure_ascii=True)}
            )

            self.context.logger.info(
                f"Invalidated CL pool cache for chain {chain} after successful pool entry"
            )

        except Exception as e:
            self.context.logger.error(f"Error invalidating CL pool cache: {str(e)}")

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
        price = yield from self._fetch_coin_price("ethereum")
        return price

    def _fetch_coin_price(self, coin_id: str) -> Generator[None, None, Optional[float]]:
        """Fetch the price for any coin using CoinGecko coin price endpoint."""
        timestamp = int(self._get_current_timestamp())
        date_str = datetime.utcfromtimestamp(timestamp).strftime("%d-%m-%Y")

        self.context.logger.info(f"Fetching current price for coin {coin_id}")

        cached_price = yield from self._get_cached_price(coin_id, date_str)
        if cached_price is not None:
            self.context.logger.info(
                f"Using cached price for {coin_id}: ${cached_price}"
            )
            return cached_price

        headers = {
            "Accept": "application/json",
        }
        if self.coingecko.api_key:
            headers["x-cg-api-key"] = self.coingecko.api_key

        self.context.logger.info(
            f"Fetching {coin_id} price from CoinGecko coin price endpoint"
        )

        success, response_json = yield from self._request_with_retries(
            endpoint=self.coingecko.coin_price_endpoint.format(coin_id=coin_id),
            headers=headers,
            rate_limited_code=self.coingecko.rate_limited_code,
            rate_limited_callback=self.coingecko.rate_limited_status_callback,
            retry_wait=self.params.sleep_time,
        )

        if success:
            token_data = next(iter(response_json.values()), {})
            price = token_data.get("usd", 0)
            if price:
                self.context.logger.info(
                    f"Successfully fetched {coin_id} price: ${price}"
                )
                yield from self._cache_price(coin_id, price, date_str)
            else:
                self.context.logger.warning(f"No price data returned for {coin_id}")
            return price
        else:
            self.context.logger.error(f"Failed to fetch {coin_id} price from CoinGecko")
        return None

    def _get_current_timestamp(self) -> int:
        return int(time.time())

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
        self.context.logger.info(
            f"Fetching historical price for token {coingecko_id} on date {date_str}"
        )

        # First check the cache
        cached_price = yield from self._get_cached_price(coingecko_id, date_str)
        if cached_price is not None:
            self.context.logger.info(
                f"Using cached historical price for {coingecko_id} on {date_str}: ${cached_price}"
            )
            return cached_price

        endpoint = self.coingecko.historical_price_endpoint.format(
            coin_id=coingecko_id,
            date=date_str,
        )

        headers = {"Accept": "application/json"}
        if self.coingecko.api_key:
            headers["x-cg-api-key"] = self.coingecko.api_key

        self.context.logger.info(
            f"Fetching historical price from CoinGecko API for {coingecko_id} on {date_str}"
        )

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
                self.context.logger.info(
                    f"Successfully fetched historical price for {coingecko_id} on {date_str}: ${price}"
                )
                # Cache the historical price
                yield from self._cache_price(coingecko_id, price, date_str)
                return price
            else:
                self.context.logger.warning(
                    f"No price data in response for token {coingecko_id} on {date_str}"
                )
                return None
        else:
            self.context.logger.error(
                f"Failed to fetch historical price for {coingecko_id} on {date_str} from CoinGecko"
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
            on_chain_amount = yield from self._get_native_balance(chain, account)
            cached_amount = int(result[ETH_REMAINING_KEY])

            if on_chain_amount is not None:
                # If there's a mismatch, sync the cached value with on-chain balance
                if cached_amount != on_chain_amount:
                    self.context.logger.info(
                        f"Syncing ETH remaining amount from cached {cached_amount} to on-chain {on_chain_amount}"
                    )
                    yield from self._write_kv({ETH_REMAINING_KEY: str(on_chain_amount)})
                    return on_chain_amount

                return cached_amount

            return cached_amount
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
        """Reset the remaining ETH amount to the on-chain balance in kv_store."""
        chain = self.params.target_investment_chains[0]
        account = self.params.safe_contract_addresses.get(chain)
        amount = yield from self._get_native_balance(chain, account)
        if amount is None:
            amount = 0
        self.context.logger.info(
            f"Resetting ETH remaining amount in kv_store to {amount}"
        )
        yield from self._write_kv({ETH_REMAINING_KEY: str(amount)})
        return amount

    def should_update_rewards_from_subgraph(
        self, chain: str, period: int = None
    ) -> Generator[None, None, bool]:
        """Check if rewards should be updated from subgraph (period == 0 or 24-hour interval)"""
        # Update immediately if period == 0
        if self.synchronized_data.period_count == 0:
            self.context.logger.info(
                f"Reward update for {chain}: period == 0, forcing immediate update"
            )
            return True

        update_key = f"{REWARD_UPDATE_KEY_PREFIX}{chain}"
        result = yield from self._read_kv((update_key,))

        if not result or not result.get(update_key):
            return True

        try:
            last_update = int(float(result[update_key]))
        except (ValueError, TypeError):
            self.context.logger.error(
                f"Invalid timestamp format in kv_store: {result[update_key]}"
            )
            return True  # Force update if timestamp is invalid
        current_time = self._get_current_timestamp()

        time_since_update = current_time - last_update
        should_update = time_since_update >= REWARD_UPDATE_INTERVAL

        self.context.logger.info(
            f"Reward update check for {chain}: Last update {time_since_update/3600:.1f}h ago, "
            f"Should update: {should_update}"
        )

        return should_update

    def query_service_rewards(
        self, chain: str
    ) -> Generator[None, None, Optional[Dict]]:
        """Query subgraph for service rewards using service query"""
        endpoint = self.params.staking_subgraph_endpoints.get(chain)
        service_id = self.params.on_chain_service_id

        query = {
            "query": f"""
            {{
            service(id: {service_id!r}) {{
                blockNumber
                blockTimestamp
                currentOlasStaked
                id
                olasRewardsEarned
            }}
            }}
            """
        }

        self.context.logger.info(
            f"Querying subgraph for service {service_id} on {chain}"
        )

        response = yield from self.get_http_response(
            method="POST",
            url=endpoint,
            content=json.dumps(query, ensure_ascii=True).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        if response.status_code in HTTP_OK:
            data = json.loads(response.body)
            service_data = data.get("data", {}).get("service")
            if service_data:
                self.context.logger.info(
                    f"Retrieved service data from subgraph: olasRewardsEarned={service_data.get('olasRewardsEarned', 0)}"
                )
                return service_data
            else:
                self.context.logger.warning(
                    f"No service data found for service ID {service_id}"
                )
                return None

        self.context.logger.error(
            f"Failed to query subgraph: {response.status_code} - {response.body}"
        )
        return None

    def update_accumulated_rewards_for_chain(
        self, chain: str
    ) -> Generator[None, None, None]:
        """Update accumulated rewards for a chain using subgraph data"""
        should_update_rewards = yield from self.should_update_rewards_from_subgraph(
            chain
        )
        if not should_update_rewards:
            return

        self.context.logger.info(f"Starting reward update for {chain}")

        # Query service data directly using simple GraphQL query
        service_data = yield from self.query_service_rewards(chain)

        if not service_data:
            self.context.logger.info(f"No service data found for {chain}")
            return

        # Get olasRewardsEarned directly from service data
        olas_rewards_earned = service_data.get("olasRewardsEarned", "0")
        try:
            total_rewards = int(olas_rewards_earned)
        except (ValueError, TypeError):
            self.context.logger.error(
                f"Invalid olasRewardsEarned value: {olas_rewards_earned}"
            )
            total_rewards = 0

        # Update stored accumulated rewards for OLAS token
        olas_address = OLAS_ADDRESSES.get(chain)
        if olas_address:
            rewards_key = f"accumulated_rewards_{chain}_{olas_address.lower()}"

            # Store the total rewards directly (not incremental)
            yield from self._write_kv({rewards_key: str(total_rewards)})

            self.context.logger.info(
                f"Updated rewards for {chain}: Total accumulated: {total_rewards} OLAS "
                f"(Block: {service_data.get('blockNumber', 'N/A')}, "
                f"Timestamp: {service_data.get('blockTimestamp', 'N/A')})"
            )

        # Mark as updated
        update_key = f"{REWARD_UPDATE_KEY_PREFIX}{chain}"
        yield from self._write_kv({update_key: str(int(self._get_current_timestamp()))})

        self.context.logger.info(f"Completed reward update for {chain}")

    def get_accumulated_rewards_for_token(
        self, chain: str, token_address: str
    ) -> Generator[None, None, int]:
        """Get stored accumulated rewards for a token"""
        rewards_key = f"accumulated_rewards_{chain}_{token_address.lower()}"
        result = yield from self._read_kv((rewards_key,))
        if not result:
            return 0

        rewards_value = result.get(rewards_key)
        if rewards_value is None:
            return 0

        try:
            return int(rewards_value)
        except (ValueError, TypeError):
            self.context.logger.warning(
                f"Invalid rewards value for {rewards_key}: {rewards_value}, returning 0"
            )
            return 0

    def _calculate_days_since_entry(self, enter_timestamp: int) -> float:
        """Calculate days elapsed since position entry."""
        current_time = int(self._get_current_timestamp())
        return (current_time - enter_timestamp) / (24 * 3600)

    def _check_minimum_time_met(self, position: Dict) -> bool:
        """Check if minimum time requirement is met for a position."""
        if not position.get("enter_timestamp") or position.get("min_hold_days", 0) == 0:
            return True  # No time requirements

        days_elapsed = self._calculate_days_since_entry(position["enter_timestamp"])
        return days_elapsed >= position["min_hold_days"]

    def _store_entry_costs(
        self, chain: str, position_id: str, costs: float
    ) -> Generator[None, None, None]:
        """Store entry costs in KV store with unique key"""
        try:
            key = self._get_entry_costs_key(chain, position_id)

            # Get existing entry costs dictionary
            entry_costs_dict = yield from self._get_all_entry_costs()

            # Update the dictionary
            entry_costs_dict[key] = costs

            # Store back to KV store
            yield from self._write_kv(
                {"entry_costs_dict": json.dumps(entry_costs_dict, ensure_ascii=True)}
            )

            self.context.logger.info(f"Stored entry costs: {key} = ${costs:.6f}")
        except Exception as e:
            self.context.logger.error(f"Error storing entry costs: {e}")

    def _get_entry_costs_key(self, chain: str, position_id: str) -> str:
        """Generate unique key for entry costs storage"""
        return f"entry_costs_{chain}_{position_id}"

    def _get_all_entry_costs(self) -> Generator[None, None, Dict[str, float]]:
        """Get all entry costs from KV store"""
        try:
            result = yield from self._read_kv(("entry_costs_dict",))
            if result and result.get("entry_costs_dict"):
                entry_costs_dict = json.loads(result["entry_costs_dict"])
                # Convert string values back to float
                return {k: float(v) for k, v in entry_costs_dict.items()}
            return {}
        except Exception as e:
            self.context.logger.error(f"Error getting all entry costs: {e}")
            return {}

    def _build_exit_pool_action_base(
        self, position: Dict[str, Any], tokens: List[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Build action for exiting a pool position.

        :param position: position data containing pool information
        :param tokens: optional list of tokens for the exit action
        :return: exit pool action dictionary
        """
        if not position:
            self.context.logger.error("No position provided for exit action")
            return None

        dex_type = position.get("dex_type")
        chain = position.get("chain")
        pool_address = position.get("pool_address")
        pool_type = position.get("pool_type")
        is_stable = position.get("is_stable")
        is_cl_pool = position.get("is_cl_pool")

        # Determine action type based on DEX
        action_type = (
            Action.WITHDRAW.value
            if dex_type == DexType.STURDY.value
            else Action.EXIT_POOL.value
        )

        # Build base action
        exit_pool_action = {
            "action": action_type,
            "dex_type": dex_type,
            "chain": chain,
            "pool_address": pool_address,
            "pool_type": pool_type,
            "is_stable": is_stable,
            "is_cl_pool": is_cl_pool,
        }

        # Add assets if provided
        if tokens:
            exit_pool_action["assets"] = [token.get("token") for token in tokens]

        # Handle Velodrome CL pools with multiple positions
        if (
            dex_type == DexType.VELODROME.value
            and is_cl_pool
            and "positions" in position
        ):
            # Extract token IDs from all positions
            token_ids = [pos["token_id"] for pos in position.get("positions", [])]
            liquidities = [pos["liquidity"] for pos in position.get("positions", [])]
            if token_ids and liquidities:
                self.context.logger.info(
                    f"Exiting Velodrome CL pool with {len(token_ids)} positions. "
                    f"Token IDs: {token_ids}"
                )
                exit_pool_action["token_ids"] = token_ids
                exit_pool_action["liquidities"] = liquidities
        # For single position case (backward compatibility)
        elif "token_id" in position:
            exit_pool_action["token_id"] = position.get("token_id")
            exit_pool_action["liquidity"] = position.get("liquidity")

        return exit_pool_action

    def _build_swap_to_usdc_action(
        self,
        chain: str,
        from_token_address: str,
        from_token_symbol: str,
        funds_percentage: float = 1.0,
        description: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build action for swapping a token to USDC.

        :param chain: blockchain chain
        :param from_token_address: source token address
        :param from_token_symbol: source token symbol
        :param funds_percentage: percentage of funds to use (default: 1.0 = 100%)
        :param description: optional description for the action
        :return: swap action dictionary
        """
        try:
            usdc_address = self._get_usdc_address(chain)
            olas_address = (
                self._get_olas_address(chain)
                if hasattr(self, "_get_olas_address")
                else None
            )
            if not usdc_address:
                self.context.logger.error(f"Could not get USDC address for {chain}")
                return None

            # Skip if it's already USDC (by address)
            if (
                from_token_address
                and from_token_address.lower() == usdc_address.lower()
            ):
                self.context.logger.info(
                    "Skipping USDC - it's already USDC (by address)"
                )
                return None

            # Skip if it's OLAS (by address)
            if (
                olas_address
                and from_token_address
                and from_token_address.lower() == olas_address.lower()
            ):
                self.context.logger.info(
                    "Skipping OLAS - do not swap OLAS during withdrawal (by address)"
                )
                return None

            swap_action = {
                "action": Action.FIND_BRIDGE_ROUTE.value,
                "chain": chain,
                "from_chain": chain,
                "to_chain": chain,
                "from_token": from_token_address,
                "from_token_symbol": from_token_symbol,
                "to_token": usdc_address,
                "to_token_symbol": "USDC",
                "funds_percentage": funds_percentage,
            }

            if description:
                swap_action["description"] = description

            self.context.logger.info(f"Created swap action: {swap_action}")
            return swap_action

        except Exception as e:
            self.context.logger.error(f"Error building swap to USDC action: {str(e)}")
            return None

    def _get_usdc_address(self, chain: str) -> Optional[str]:
        """
        Get USDC token address for the specified chain.

        :param chain: blockchain chain
        :return: USDC contract address
        """
        try:
            # Common USDC addresses for different chains
            usdc_addresses = {
                "mode": "0xd988097fb8612cc24eeC14542bC03424c656005f",
                "optimism": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
            }

            usdc_address = usdc_addresses.get(chain.lower())
            if usdc_address:
                usdc_address = to_checksum_address(usdc_address)
                self.context.logger.info(
                    f"Found USDC address for {chain}: {usdc_address}"
                )
                return usdc_address
            else:
                self.context.logger.warning(
                    f"No USDC address configured for chain: {chain}"
                )
                return None

        except Exception as e:
            self.context.logger.error(
                f"Error getting USDC address for {chain}: {str(e)}"
            )
            return None

    def _get_olas_address(self, chain: str) -> Optional[str]:
        """Get the OLAS token address for the given chain."""
        olas_addresses = {
            "optimism": "0xfc2e6e6bcbd49ccf3a5f029c79984372dcbfe527",
            "mode": "0xcfd1d50ce23c46d3cf6407487b2f8934e96dc8f9",
        }
        return olas_addresses.get(chain)

    def get_velodrome_position_principal(
        self,
        chain: str,
        position_manager_address: str,
        token_id: int,
        sqrt_price_x96: int,
    ) -> Generator[None, None, Tuple[int, int]]:
        """Get Velodrome position principal using Sugar contract."""
        sugar_address = self.params.velodrome_slipstream_helper_contract_addresses.get(
            chain
        )
        if not sugar_address:
            self.context.logger.error(
                f"No Velodrome Sugar contract address for chain {chain}"
            )
            return 0, 0

        # Use Velodrome's official principal function for maximum accuracy
        amounts = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=sugar_address,
            contract_public_id=VelodromeSlipstreamHelperContract.contract_id,
            contract_callable="principal",
            data_key="amounts",
            position_manager=position_manager_address,
            token_id=token_id,
            sqrt_price_x96=sqrt_price_x96,
            chain_id=chain,
        )

        if amounts:
            return amounts[0], amounts[1]
        return 0, 0

    def get_velodrome_amounts_for_liquidity(
        self,
        chain: str,
        sqrt_price_x96: int,
        sqrt_ratio_a_x96: int,
        sqrt_ratio_b_x96: int,
        liquidity: int,
    ) -> Generator[None, None, Tuple[int, int]]:
        """Get amounts for liquidity using Velodrome Sugar contract."""
        sugar_address = self.params.velodrome_slipstream_helper_contract_addresses.get(
            chain
        )
        if not sugar_address:
            self.context.logger.error(
                f"No Velodrome Sugar contract address for chain {chain}"
            )
            return 0, 0

        # Use Velodrome's official getAmountsForLiquidity function
        amounts = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=sugar_address,
            contract_public_id=VelodromeSlipstreamHelperContract.contract_id,
            contract_callable="get_amounts_for_liquidity",
            data_key="amounts",
            sqrt_price_x96=sqrt_price_x96,
            sqrt_ratio_a_x96=sqrt_ratio_a_x96,
            sqrt_ratio_b_x96=sqrt_ratio_b_x96,
            liquidity=liquidity,
            chain_id=chain,
        )

        if amounts:
            return amounts[0], amounts[1]
        return 0, 0

    def get_velodrome_sqrt_ratio_at_tick(
        self, chain: str, tick: int
    ) -> Generator[None, None, int]:
        """Get sqrt ratio at tick using Velodrome Sugar contract."""
        sugar_address = self.params.velodrome_slipstream_helper_contract_addresses.get(
            chain
        )
        if not sugar_address:
            self.context.logger.error(
                f"No Velodrome Sugar contract address for chain {chain}"
            )
            return 0

        # Use Velodrome's official getSqrtRatioAtTick function
        sqrt_ratio = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=sugar_address,
            contract_public_id=VelodromeSlipstreamHelperContract.contract_id,
            contract_callable="get_sqrt_ratio_at_tick",
            data_key="sqrt_ratio",
            tick=tick,
            chain_id=chain,
        )

        return sqrt_ratio if sqrt_ratio else 0

    def _has_staking_metadata(self, position: Dict[str, Any]) -> bool:
        """Check if position has staking metadata."""
        return bool(
            position.get("gauge_address")
            or position.get("staked")
            or position.get("staked_amount", 0) > 0
        )

    def _build_unstake_lp_tokens_action(
        self, position: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build action for unstaking LP tokens before position exit."""
        try:
            chain = position.get("chain")
            pool_address = position.get("pool_address")
            dex_type = position.get("dex_type")
            is_cl_pool = position.get("is_cl_pool", False)
            gauge_address = position.get("gauge_address")

            # Only create unstaking actions for Velodrome pools
            if dex_type != "velodrome":
                self.context.logger.info(
                    f"Skipping unstaking for non-Velodrome pool: {dex_type}"
                )
                return None

            if not all([chain, pool_address]):
                self.context.logger.error(
                    "Missing required parameters for unstake action"
                )
                return None

            safe_address = self.params.safe_contract_addresses.get(chain)
            if not safe_address:
                self.context.logger.error(f"No safe address found for chain {chain}")
                return None

            if is_cl_pool:
                # For CL pools, we need token_id information
                positions_data = position.get("positions", [])
                token_ids = [pos["token_id"] for pos in positions_data]
                if not token_ids:
                    # Try to get from single position format
                    token_id = position.get("token_id")
                    if token_id is not None:
                        token_ids = [token_id]

                if not token_ids:
                    self.context.logger.error(
                        "No token IDs found for CL pool unstaking"
                    )
                    return None

                unstake_action = {
                    "action": Action.UNSTAKE_LP_TOKENS.value,
                    "dex_type": dex_type,
                    "chain": chain,
                    "pool_address": pool_address,
                    "is_cl_pool": True,
                    "token_ids": token_ids,
                    "description": f"Unstake CL LP tokens for {pool_address}",
                }
            else:
                # For regular pools, we need the staked amount
                unstake_action = {
                    "action": Action.UNSTAKE_LP_TOKENS.value,
                    "dex_type": dex_type,
                    "chain": chain,
                    "pool_address": pool_address,
                    "is_cl_pool": False,
                    "description": f"Unstake LP tokens for {pool_address}",
                }

            # Add gauge address if available
            if gauge_address:
                unstake_action["gauge_address"] = gauge_address

            self.context.logger.info(
                f"Created unstake LP tokens action: {unstake_action}"
            )
            return unstake_action

        except Exception as e:
            self.context.logger.error(
                f"Error building unstake LP tokens action: {str(e)}"
            )
            return None

    def get_agent_type_by_name(
        self, type_name
    ) -> Generator[None, None, Optional[Dict]]:
        """Get agent type by name."""
        response = yield from self._call_mirrordb(
            method="read_",
            method_name="get_agent_type_by_name",
            endpoint=f"api/agent-types/name/{type_name}",
        )
        return response

    def create_agent_type(self, type_name, description) -> Generator[None, None, Dict]:
        """Create a new agent type."""
        # Prepare agent type data
        agent_type_data = {"type_name": type_name, "description": description}

        endpoint = "api/agent-types/"

        # Call API
        response = yield from self._call_mirrordb(
            method="create_",
            method_name="create_agent_type",
            endpoint=endpoint,
            data=agent_type_data,
        )

        return response

    def get_attr_def_by_name(self, attr_name) -> Generator[None, None, Optional[Dict]]:
        """Get agent type by name."""
        response = yield from self._call_mirrordb(
            method="read_",
            method_name="get_attr_def_by_name",
            endpoint=f"api/attributes/name/{attr_name}",
        )
        return response

    def create_attribute_definition(
        self, type_id, attr_name, data_type, is_required, default_value, agent_id
    ) -> Generator[None, None, Dict]:
        """Create a new attribute definition for a specific agent type."""
        # Prepare attribute definition data
        attr_def_data = {
            "type_id": type_id,
            "attr_name": attr_name,
            "data_type": data_type,
            "is_required": is_required,
            "default_value": default_value,
        }

        # Generate timestamp and prepare signature
        timestamp = int(self.round_sequence.last_round_transition_timestamp.timestamp())
        endpoint = f"api/agent-types/{type_id}/attributes/"
        message = f"timestamp:{timestamp},endpoint:{endpoint}"
        signature = yield from self.sign_message(message)
        if not signature:
            return None

        # Prepare authentication data
        auth_data = {"agent_id": agent_id, "signature": signature, "message": message}

        # Call API
        response = yield from self._call_mirrordb(
            method="create_",
            method_name="create_attribute_definition",
            endpoint=endpoint,
            data={"attr_def": attr_def_data, "auth": auth_data},
        )

        return response

    def get_agent_registry_by_address(
        self, eth_address
    ) -> Generator[None, None, Optional[Dict]]:
        """Get agent registry by Ethereum address."""
        response = yield from self._call_mirrordb(
            method="read_",
            method_name="get_agent_registry_by_address",
            endpoint=f"api/agent-registry/address/{eth_address}",
        )
        return response

    def create_agent_registry(
        self, agent_name, type_id, eth_address
    ) -> Generator[None, None, Dict]:
        """Create a new agent registry."""
        # Prepare agent registry data
        agent_registry_data = {
            "agent_name": agent_name,
            "type_id": type_id,
            "eth_address": eth_address,
        }

        # Call API
        response = yield from self._call_mirrordb(
            method="create_",
            method_name="create_agent_registry",
            endpoint="api/agent-registry/",
            data=agent_registry_data,
        )

        return response

    def create_agent_attribute(
        self,
        agent_id,
        attr_def_id,
        json_value=None,
    ) -> Generator[None, None, Dict]:
        """Create a new attribute value for a specific agent."""
        # Prepare the agent attribute data with all values set to None initially
        agent_attr_data = {
            "agent_id": agent_id,
            "attr_def_id": attr_def_id,
            "string_value": None,
            "integer_value": None,
            "float_value": None,
            "boolean_value": None,
            "date_value": None,
            "json_value": json_value,
        }

        # Generate timestamp and prepare signature
        timestamp = int(self.round_sequence.last_round_transition_timestamp.timestamp())
        endpoint = f"api/agents/{agent_id}/attributes/"
        message = f"timestamp:{timestamp},endpoint:{endpoint}"
        signature = yield from self.sign_message(message)
        if not signature:
            return None

        # Prepare authentication data
        auth_data = {"agent_id": agent_id, "signature": signature, "message": message}

        # Call API
        response = yield from self._call_mirrordb(
            method="create_",
            method_name="create_agent_attribute",
            endpoint=endpoint,
            data={"agent_attr": agent_attr_data, "auth": auth_data},
        )

        return response

    def _get_or_create_agent_type(
        self, eth_address: str
    ) -> Generator[Dict[str, Any], None, None]:
        """Get or create agent type."""
        data = yield from self._read_kv(keys=("agent_type",))
        if not data or not data.get("agent_type"):
            type_name = AGENT_TYPE.get(self.params.target_investment_chains[0])
            agent_type = yield from self.get_agent_type_by_name(type_name)
            if not agent_type:
                agent_type = yield from self.create_agent_type(
                    type_name,
                    "An agent for DeFi liquidity management and APR tracking",
                )
                if not agent_type:
                    raise Exception("Failed to create agent type.")
                yield from self._write_kv({"agent_type": json.dumps(agent_type)})
            return agent_type

        return json.loads(data["agent_type"])

    def _get_or_create_attr_def(
        self, type_id: str, agent_id: str
    ) -> Generator[Dict[str, Any], None, None]:
        """Get or create APR attribute definition."""
        data = yield from self._read_kv(keys=("attr_def",))
        if not data or not data.get("attr_def"):
            attr_def = yield from self.get_attr_def_by_name(METRICS_NAME)
            if not attr_def:
                attr_def = yield from self.create_attribute_definition(
                    type_id,
                    METRICS_NAME,
                    METRICS_TYPE,
                    True,
                    "{}",
                    agent_id,
                )
                if not attr_def:
                    raise Exception("Failed to create attribute definition.")
                yield from self._write_kv({"attr_def": json.dumps(attr_def)})
            return attr_def

        return json.loads(data["attr_def"])

    def _get_or_create_agent_registry(
        self,
    ) -> Generator[Optional[Dict[str, Any]], None, None]:
        """Get or create agent registry entry (copied from APRPopulationBehaviour)."""
        try:
            data = yield from self._read_kv(keys=("agent_registry",))
            if not data or not data.get("agent_registry"):
                # Get or create agent type first
                eth_address = self.context.agent_address
                agent_type = yield from self._get_or_create_agent_type(eth_address)
                if not agent_type:
                    self.context.logger.error("Failed to get or create agent type")
                    return None

                type_id = agent_type["type_id"]

                agent_registry = yield from self.get_agent_registry_by_address(
                    eth_address
                )
                if not agent_registry:
                    agent_name = self.generate_name(eth_address)
                    self.context.logger.info(
                        f"Creating agent registry with name: {agent_name}"
                    )
                    agent_registry = yield from self.create_agent_registry(
                        agent_name, type_id, eth_address
                    )
                    if not agent_registry:
                        self.context.logger.error("Failed to create agent registry")
                        return None
                    yield from self._write_kv(
                        {"agent_registry": json.dumps(agent_registry)}
                    )
                return agent_registry

            return json.loads(data["agent_registry"])
        except Exception as e:
            self.context.logger.error(
                f"Error in _get_or_create_agent_registry: {str(e)}"
            )
            return None

    def generate_phonetic_syllable(self, seed):
        """Generates phonetic syllable"""
        phonetic_syllables = [
            "ba",
            "bi",
            "bu",
            "ka",
            "ke",
            "ki",
            "ko",
            "ku",
            "da",
            "de",
            "di",
            "do",
            "du",
            "fa",
            "fe",
            "fi",
            "fo",
            "fu",
            "ga",
            "ge",
            "gi",
            "go",
            "gu",
            "ha",
            "he",
            "hi",
            "ho",
            "hu",
            "ja",
            "je",
            "ji",
            "jo",
            "ju",
            "ka",
            "ke",
            "ki",
            "ko",
            "ku",
            "la",
            "le",
            "li",
            "lo",
            "lu",
            "ma",
            "me",
            "mi",
            "mo",
            "mu",
            "na",
            "ne",
            "ni",
            "no",
            "nu",
            "pa",
            "pe",
            "pi",
            "po",
            "pu",
            "ra",
            "re",
            "ri",
            "ro",
            "ru",
            "sa",
            "se",
            "si",
            "so",
            "su",
            "ta",
            "te",
            "ti",
            "to",
            "tu",
            "va",
            "ve",
            "vi",
            "vo",
            "vu",
            "wa",
            "we",
            "wi",
            "wo",
            "wu",
            "ya",
            "ye",
            "yi",
            "yo",
            "yu",
            "za",
            "ze",
            "zi",
            "zo",
            "zu",
            "bal",
            "ben",
            "bir",
            "bom",
            "bun",
            "cam",
            "cen",
            "cil",
            "cor",
            "cus",
            "dan",
            "del",
            "dim",
            "dor",
            "dun",
            "fam",
            "fen",
            "fil",
            "fon",
            "fur",
            "gar",
            "gen",
            "gil",
            "gon",
            "gus",
            "han",
            "hel",
            "him",
            "hon",
            "hus",
            "jan",
            "jel",
            "jim",
            "jon",
            "jus",
            "kan",
            "kel",
            "kim",
            "kon",
            "kus",
            "lan",
            "lel",
            "lim",
            "lon",
            "lus",
            "mar",
            "mel",
            "min",
            "mon",
            "mus",
            "nar",
            "nel",
            "nim",
            "nor",
            "nus",
            "par",
            "pel",
            "pim",
            "pon",
            "pus",
            "rar",
            "rel",
            "rim",
            "ron",
            "rus",
            "sar",
            "sel",
            "sim",
            "son",
            "sus",
            "tar",
            "tel",
            "tim",
            "ton",
            "tus",
            "var",
            "vel",
            "vim",
            "von",
            "vus",
            "war",
            "wel",
            "wim",
            "won",
            "wus",
            "yar",
            "yel",
            "yim",
            "yon",
            "yus",
            "zar",
            "zel",
            "zim",
            "zon",
            "zus",
            "zez",
            "zzt",
            "bzt",
            "vzt",
            "kzt",
            "mek",
            "tek",
            "nek",
            "lek",
            "tron",
            "dron",
            "kron",
            "pron",
            "bot",
            "rot",
            "not",
            "lot",
            "zap",
            "blip",
            "bleep",
            "beep",
            "wire",
            "byte",
            "bit",
            "chip",
        ]
        return phonetic_syllables[seed % len(phonetic_syllables)]

    def generate_phonetic_name(self, address, start_index, syllables):
        """Generates phonetic name"""
        return "".join(
            self.generate_phonetic_syllable(
                int(address[start_index + i * 8 : start_index + (i + 1) * 8], 16)
            )
            for i in range(syllables)
        ).lower()

    def generate_name(self, address):
        """Generates name from address"""
        first_name = self.generate_phonetic_name(address, 2, 2)
        last_name_prefix = self.generate_phonetic_name(address, 18, 2)
        last_name_number = int(address[-4:], 16) % 100
        return f"{first_name}-{last_name_prefix}{str(last_name_number).zfill(2)}"

    def sign_message(self, message) -> Generator[None, None, Optional[str]]:
        """Sign a message."""
        message_bytes = message.encode("utf-8")
        signature = yield from self.get_signature(message_bytes)
        if signature:
            signature_hex = signature[2:]
            return signature_hex
        return None

    def _update_airdrop_rewards(
        self, new_amount: int, chain: str, tx_hash: str = None
    ) -> Generator[None, None, None]:
        """Update total airdrop rewards in KV store with deduplication."""
        try:
            # If tx_hash is provided, check for deduplication
            if tx_hash:
                processed_key = f"airdrop_processed_{chain}_{tx_hash}"
                result = yield from self._read_kv((processed_key,))

                if result and result.get(processed_key):
                    self.context.logger.info(
                        f"Airdrop transaction {tx_hash} already processed for {chain}, skipping"
                    )
                    return

                # Mark transaction as processed
                yield from self._write_kv({processed_key: "true"})

            total_key = f"{AIRDROP_TOTAL_KEY}_{chain}"
            current_total = yield from self._get_total_airdrop_rewards(chain)
            new_total = current_total + new_amount
            yield from self._write_kv({total_key: str(new_total)})
            self.context.logger.info(
                f"Updated airdrop rewards for {chain}: +{new_amount}, total: {new_total}"
            )
        except Exception as e:
            self.context.logger.error(f"Error updating airdrop rewards: {e}")

    def _get_total_airdrop_rewards(self, chain: str) -> Generator[None, None, int]:
        """Get total accumulated airdrop rewards for a chain."""
        try:
            total_key = f"{AIRDROP_TOTAL_KEY}_{chain}"
            result = yield from self._read_kv((total_key,))
            if result and result.get(total_key):
                return int(result[total_key])
            return 0
        except (ValueError, TypeError):
            self.context.logger.warning(
                f"Invalid airdrop total for {chain}, returning 0"
            )
            return 0

    def _is_airdrop_transfer(self, transfer: Dict, airdrop_contract: str) -> bool:
        """Check if a transfer is from the airdrop contract."""
        if not airdrop_contract:
            return False

        from_address = transfer.get("from_address", "")
        token_address = transfer.get("token_address", "")
        symbol = transfer.get("symbol", "")

        # Check if transfer is from airdrop contract and is OLAS token
        return (
            from_address.lower() == airdrop_contract.lower()
            and symbol.upper() == "OLAS"
            and token_address.lower() == OLAS_ADDRESSES.get("mode", "").lower()
        )


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

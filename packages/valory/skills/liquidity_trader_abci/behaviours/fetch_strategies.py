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

"""This module contains the behaviour for fetching the strategies to execute for 'liquidity_trader_abci' skill."""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Context, Decimal, getcontext
from typing import Any, Dict, Generator, List, Optional, Tuple, Type

import requests
from eth_utils import to_checksum_address

from packages.valory.contracts.balancer_vault.contract import VaultContract
from packages.valory.contracts.balancer_weighted_pool.contract import (
    WeightedPoolContract,
)
from packages.valory.contracts.erc20.contract import ERC20
from packages.valory.contracts.gnosis_safe.contract import (
    GnosisSafeContract,
    SafeOperation,
)
from packages.valory.contracts.service_registry.contract import ServiceRegistryContract
from packages.valory.contracts.sturdy_yearn_v3_vault.contract import (
    YearnV3VaultContract,
)
from packages.valory.contracts.uniswap_v3_non_fungible_position_manager.contract import (
    UniswapV3NonfungiblePositionManagerContract,
)
from packages.valory.contracts.uniswap_v3_pool.contract import UniswapV3PoolContract
from packages.valory.contracts.velodrome_cl_pool.contract import VelodromeCLPoolContract
from packages.valory.contracts.velodrome_non_fungible_position_manager.contract import (
    VelodromeNonFungiblePositionManagerContract,
)
from packages.valory.contracts.velodrome_pool.contract import VelodromePoolContract
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    Chain,
    DexType,
    LiquidityTraderBaseBehaviour,
    OLAS_ADDRESSES,
    PORTFOLIO_UPDATE_INTERVAL,
    PositionStatus,
    SAFE_TX_GAS,
    TradingType,
    WHITELISTED_ASSETS,
    ZERO_ADDRESS,
)
from packages.valory.skills.liquidity_trader_abci.states.base import StakingState
from packages.valory.skills.liquidity_trader_abci.states.fetch_strategies import (
    FetchStrategiesPayload,
    FetchStrategiesRound,
)
from packages.valory.skills.liquidity_trader_abci.states.post_tx_settlement import (
    PostTxSettlementRound,
)
from packages.valory.skills.liquidity_trader_abci.utils.tick_math import (
    get_amounts_for_liquidity,
    get_sqrt_ratio_at_tick,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)


# Add these constants to the class or base file
CONTRACT_CHECK_CACHE_PREFIX = "contract_check_"
TRANSFER_EVENT_SIGNATURE = (
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
)
ZERO_ADDRESS_PADDED = (
    "0x0000000000000000000000000000000000000000000000000000000000000000"
)


class FetchStrategiesBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that gets the balances of the assets of agent safes."""

    matching_round: Type[AbstractRound] = FetchStrategiesRound
    strategies = None

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            # Normal fetch strategies logic
            sender = self.context.agent_address

            agent_config = os.environ.get("AEA_AGENT", "")
            agent_hash = agent_config.split(":")[-1] if agent_config else "Not found"
            self.context.logger.info(f"Agent hash: {agent_hash}")

            if self.current_positions:
                self.context.logger.info(
                    f"Current Positions - {self.current_positions}"
                )

            # Update the amounts of all open positions
            if self.synchronized_data.period_count == 0:
                # Validate Velodrome v2 pool addresses before updating position amounts
                self.context.logger.info("Validating Velodrome v2 pool addresses")
                yield from self._validate_velodrome_v2_pool_addresses()

                self.context.logger.info("Updating position amounts for period 0")
                yield from self.update_position_amounts()
                self.context.logger.info(
                    "Checking and updating zero liquidity positions"
                )
                self.check_and_update_zero_liquidity_positions()

            self.context.logger.info(f"Current Positions: {self.current_positions}")

            sender = self.context.agent_address

            chain = self.params.target_investment_chains[0]
            safe_address = self.params.safe_contract_addresses.get(chain)

            # update locally stored eth balance in-case it's incorrect
            eth_balance = yield from self._get_native_balance(chain, safe_address)
            self.context.logger.info(f"Current ETH balance: {eth_balance}")

            if self.synchronized_data.period_count == 0:
                yield from self._check_and_create_eth_revert_transactions(
                    chain, safe_address, sender
                )

            db_data = yield from self._read_kv(
                keys=("selected_protocols", "trading_type")
            )

            selected_protocols = db_data.get("selected_protocols", None)
            if selected_protocols is None:
                serialized_protocols = []
            else:
                serialized_protocols = json.loads(selected_protocols)

            trading_type = db_data.get("trading_type", None)

            if not serialized_protocols:
                serialized_protocols = []
                for chain in self.params.target_investment_chains:
                    chain_strategies = self.params.available_strategies.get(chain, [])
                    serialized_protocols.extend(chain_strategies)

            # Update KV store if protocols were fixed
            if (
                serialized_protocols != json.loads(selected_protocols)
                if selected_protocols
                else []
            ):
                self.context.logger.info("Updating KV store with validated protocols")
                yield from self._write_kv(
                    {
                        "selected_protocols": json.dumps(
                            serialized_protocols, ensure_ascii=True
                        )
                    }
                )

            if not trading_type:
                trading_type = TradingType.BALANCED.value

            self.context.logger.info(
                f"Reading values from kv store... Selected protocols: {serialized_protocols}, Trading type: {trading_type}"
            )
            self.shared_state.trading_type = trading_type
            self.shared_state.selected_protocols = selected_protocols

            # Initialize assets from initial_assets if empty
            if not self.assets:
                self.assets = self.params.initial_assets

            # Filter whitelisted assets based on price changes
            if not self.whitelisted_assets:
                self.whitelisted_assets = WHITELISTED_ASSETS
                self.store_whitelisted_assets()
            else:
                self.read_whitelisted_assets()

            # Check if one day has passed since last whitelist update
            db_data = yield from self._read_kv(keys=("last_whitelisted_updated",))
            last_updated = db_data.get("last_whitelisted_updated", "0")
            if not last_updated:
                last_updated = 0

            current_time = int(self._get_current_timestamp())
            one_day_in_seconds = 24 * 60 * 60

            try:
                last_updated_int = int(last_updated)
            except (ValueError, TypeError):
                self.context.logger.warning(
                    "Invalid last updated timestamp, defaulting to 0"
                )
                last_updated_int = 0

            time_since_update = int(current_time) - last_updated_int
            self.context.logger.info(
                f"Time since last update: {time_since_update} seconds"
            )

            if not (time_since_update >= one_day_in_seconds):
                self.context.logger.info("Tracking whitelisted assets")
                yield from self._track_whitelisted_assets()
                # Store current timestamp as last updated
                self.context.logger.info("Updating last whitelist update timestamp")
                yield from self._write_kv({"last_whitelisted_updated": current_time})

            # Check if we need to recalculate the portfolio
            self.context.logger.info("Recalculating user share values")
            yield from self.calculate_user_share_values()
            # Store the updated portfolio data
            self.context.logger.info("Storing updated portfolio data")
            self.store_portfolio_data()

            # Update agent performance metrics
            self._update_agent_performance_metrics()

            payload = FetchStrategiesPayload(
                sender=sender,
                content=json.dumps(
                    {
                        "selected_protocols": serialized_protocols,
                        "trading_type": trading_type,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                ),
            )
            self.context.logger.info(f"Created payload with content: {payload.content}")

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def _track_whitelisted_assets(self) -> Generator[None, None, None]:
        """Track whitelisted assets based on price changes over the last day and remove if the price has dropped"""
        self.context.logger.info("Starting price-based filtering of whitelisted assets")

        # Get current timestamp and calculate yesterday's timestamp
        current_time = datetime.now()
        yesterday = current_time - timedelta(days=1)

        # Format dates for historical price API
        today_str = current_time.strftime("%d-%m-%Y")
        yesterday_str = yesterday.strftime("%d-%m-%Y")

        # Track assets to remove
        assets_to_remove = {}

        for chain, assets in WHITELISTED_ASSETS.items():
            if chain not in self.params.target_investment_chains:
                continue

            self.context.logger.info(f"Checking price changes for {chain} assets")
            assets_to_remove[chain] = []

            for token_address, token_symbol in assets.items():
                try:
                    # Get historical prices for yesterday and today
                    yesterday_price = yield from self._get_historical_price_for_date(
                        token_address, token_symbol, yesterday_str, chain
                    )
                    today_price = yield from self._get_historical_price_for_date(
                        token_address, token_symbol, today_str, chain
                    )

                    if yesterday_price is None or today_price is None:
                        self.context.logger.warning(
                            f"Could not fetch prices for {token_symbol} ({token_address}) on {chain}"
                        )
                        continue

                    # Calculate price change percentage
                    price_change_percent = (
                        (today_price - yesterday_price) / yesterday_price
                    ) * 100

                    self.context.logger.info(
                        f"{token_symbol} price change: {price_change_percent:.2f}% "
                        f"(Yesterday: ${yesterday_price:.6f}, Today: ${today_price:.6f})"
                    )

                    # Check if price has dropped more than 5%
                    if price_change_percent < -5.0:
                        self.context.logger.warning(
                            f"Removing {token_symbol} from whitelist due to {price_change_percent:.2f}% price drop"
                        )
                        assets_to_remove[chain].append(token_address)

                    yield from self.sleep(5)

                except Exception as e:
                    self.context.logger.error(
                        f"Error checking price for {token_symbol} ({token_address}): {str(e)}"
                    )
                    continue

        # Update the assets by removing the filtered ones
        for chain, addresses_to_remove in assets_to_remove.items():
            if addresses_to_remove:
                self.context.logger.info(
                    f"Removing {len(addresses_to_remove)} assets from {chain} whitelist"
                )
                # Update the assets in memory
                if chain in self.whitelisted_assets:
                    for address in addresses_to_remove:
                        if address in self.whitelisted_assets[chain]:
                            removed_symbol = self.whitelisted_assets[chain].pop(address)
                            self.context.logger.info(
                                f"Removed {removed_symbol} ({address}) from {chain} assets"
                            )

                # Store the updated assets
                self.store_whitelisted_assets()

        self.context.logger.info(
            "Completed price-based filtering of whitelisted assets"
        )

    def _check_and_create_eth_revert_transactions(
        self, chain, safe_address, sender
    ) -> Generator[None, None, None]:
        """Check if there are any ETH revert transactions and create them if there are."""

        if not safe_address:
            self.context.logger.error(f"No safe address found for chain {chain}")
            return

        res = yield from self._track_eth_transfers_and_reversions(safe_address, chain)

        to_address = res.get("master_safe_address")
        eth_amount = int(res.get("reversion_amount", 0) * 10**18)

        if eth_amount > 0:
            if not to_address:
                self.context.logger.error(
                    f"No master safe address found for chain {chain}"
                )
                # Continue with normal flow
            else:
                self.context.logger.info(
                    f"Creating ETH transfer transaction: {eth_amount} wei to {to_address}"
                )

                # Create ETH transfer transaction
                safe_tx_hash = yield from self.contract_interact(
                    performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                    contract_address=safe_address,
                    contract_public_id=GnosisSafeContract.contract_id,
                    contract_callable="get_raw_safe_transaction_hash",
                    data_key="tx_hash",
                    to_address=to_address,
                    value=eth_amount,
                    data=b"",
                    operation=SafeOperation.CALL.value,
                    safe_tx_gas=SAFE_TX_GAS,
                    chain_id=chain,
                )

                if safe_tx_hash:
                    safe_tx_hash = safe_tx_hash[2:]
                    payload_string = hash_payload_to_hex(
                        safe_tx_hash=safe_tx_hash,
                        ether_value=eth_amount,
                        safe_tx_gas=SAFE_TX_GAS,
                        operation=SafeOperation.CALL.value,
                        to_address=to_address,
                        data=b"",
                    )

                    # Create settlement payload
                    payload = FetchStrategiesPayload(
                        sender=sender,
                        content=json.dumps(
                            {
                                "event": "settle",
                                "updates": {
                                    "tx_submitter": FetchStrategiesRound.auto_round_id(),
                                    "most_voted_tx_hash": payload_string,
                                    "chain_id": chain,
                                    "safe_contract_address": safe_address,
                                },
                            },
                            sort_keys=True,
                        ),
                    )

                    with self.context.benchmark_tool.measure(
                        self.behaviour_id
                    ).consensus():
                        yield from self.send_a2a_transaction(payload)
                        yield from self.wait_until_round_end()

                    self.set_done()
                    return

    def _get_historical_price_for_date(
        self, token_address: str, token_symbol: str, date_str: str, chain: str
    ) -> Generator[None, None, Optional[float]]:
        """Get historical price for a specific token on a specific date."""
        try:
            # For zero address (ETH), use a different approach
            if token_address == ZERO_ADDRESS:
                return (yield from self._fetch_historical_eth_price(date_str))

            # Get CoinGecko ID for the token
            coingecko_id = self.get_coin_id_from_symbol(token_symbol, chain)

            if not coingecko_id:
                self.context.logger.error(
                    f"Could not find CoinGecko ID for {token_symbol} ({token_address})"
                )
                return None

            # Fetch historical price
            price = yield from self._fetch_historical_token_price(
                coingecko_id, date_str
            )
            return price

        except Exception as e:
            self.context.logger.error(
                f"Error fetching historical price for {token_symbol}: {str(e)}"
            )
            return None

    def _fetch_historical_eth_price(
        self, date_str: str
    ) -> Generator[None, None, Optional[float]]:
        """Fetch historical ETH price for a specific date."""
        endpoint = self.coingecko.historical_price_endpoint.format(
            coin_id="ethereum",
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
                return price
            else:
                self.context.logger.error("No ETH price in response")
                return None
        else:
            self.context.logger.error("Failed to fetch historical ETH price")
            return None

    def should_recalculate_portfolio(
        self, last_portfolio_data: Dict
    ) -> Generator[None, None, bool]:
        """Determine if the portfolio should be recalculated."""
        chain = self.params.target_investment_chains[0]
        initial_investment = yield from self._load_chain_total_investment(chain)
        final_value = last_portfolio_data.get("portfolio_value", None)

        if initial_investment is None or final_value is None:
            return True

        last_round_id = self.context.state.round_sequence._abci_app._previous_rounds[
            -1
        ].round_id
        if last_round_id == PostTxSettlementRound.auto_round_id():
            return True

        return self._is_time_update_due() or self._have_positions_changed(
            last_portfolio_data
        )

    def _is_time_update_due(self) -> bool:
        """Check if enough time has passed since last update."""
        current_time = self._get_current_timestamp()
        last_update_time = self.portfolio_data.get("last_updated", 0)

        return (current_time - last_update_time) >= PORTFOLIO_UPDATE_INTERVAL

    def _have_positions_changed(self, last_portfolio_data: Dict) -> bool:
        """Check if positions have changed by comparing current and last positions."""
        current_positions = self.current_positions
        last_positions = last_portfolio_data.get("allocations", [])

        # If there's no last portfolio data or no allocations key, consider positions as changed
        if not last_portfolio_data or "allocations" not in last_portfolio_data:
            self.context.logger.info(
                "Portfolio update needed: No last portfolio data or allocations key available"
            )
            return True

        # Early return if the number of positions changed
        if len(current_positions) != len(last_positions):
            self.context.logger.info(
                f"Portfolio update needed: Position count changed. Current: {len(current_positions)}, Previous: {len(last_positions)}"
            )
            return True

        # Create sets of position identifiers with their status
        current_position_set = {
            (
                position.get(
                    "pool_address", position.get("pool_id")
                ),  # Handle both Balancer and other DEXes
                position.get("dex_type"),
                position.get("status"),
            )
            for position in current_positions
        }

        last_position_set = {
            (
                position.get("id"),  # pool_id or pool_address from allocations
                position.get("type"),  # dex_type in allocations
                PositionStatus.OPEN.value,  # allocations only contain open positions
            )
            for position in last_positions
        }

        # Check for any differences
        new_positions = current_position_set - last_position_set
        closed_positions = last_position_set - current_position_set

        if new_positions:
            self.context.logger.info(
                f"Portfolio update needed: New positions opened: {new_positions}"
            )
            return True

        if closed_positions:
            self.context.logger.info(
                f"Portfolio update needed: Positions closed: {closed_positions}"
            )
            return True

        return False

    def calculate_user_share_values(self) -> Generator[None, None, None]:
        """Calculate the value of shares for the user using Optimus subgraph data."""

        # Get chain
        chain = self.params.target_investment_chains[0]

        # Fetch portfolio data from subgraph
        portfolio_data = yield from self._fetch_portfolio_from_subgraph(chain)

        if not portfolio_data:
            self.context.logger.error(
                "Subgraph unavailable - cannot calculate portfolio without subgraph data"
            )
            return

        # Subgraph data available - use it as base
        self.context.logger.info("Using portfolio data from Optimus subgraph")

        # Add local calculations for staking and airdrop rewards
        staking_rewards_value = yield from self.calculate_stakig_rewards_value()
        airdrop_rewards_value = yield from self.calculate_airdrop_rewards_value()

        # Update portfolio data with local reward calculations
        portfolio_data["airdropped_rewards"] = float(airdrop_rewards_value)

        # Recalculate ROI including staking rewards
        initial_investment = portfolio_data.get("initial_investment", 0)
        if initial_investment and initial_investment > 0:
            portfolio_value = portfolio_data.get("portfolio_value", 0)
            withdrawals = portfolio_data.get("value_in_withdrawals", 0)

            # Total ROI includes staking rewards
            total_roi_decimal = (
                (portfolio_value + float(staking_rewards_value) + withdrawals)
                / initial_investment
            ) - 1
            portfolio_data["total_roi"] = round(total_roi_decimal * 100, 2)

            # Partial ROI is just trading (portfolio + withdrawals)
            partial_roi_decimal = (
                (portfolio_value + withdrawals) / initial_investment
            ) - 1
            portfolio_data["partial_roi"] = round(partial_roi_decimal * 100, 2)

        self.portfolio_data = portfolio_data

    def _get_tick_ranges(
        self, position: Dict, chain: str
    ) -> Generator[List[Dict[str, int]], None, None]:
        """Get tick ranges for a position."""
        if position.get("dex_type") not in [
            DexType.UNISWAP_V3.value,
            DexType.VELODROME.value,
        ]:
            return []

        # For Velodrome positions, we only get tick ranges if it's a CL pool
        if position.get("dex_type") == DexType.VELODROME.value and not position.get(
            "is_cl_pool"
        ):
            return []

        pool_address = position.get("pool_address")
        if not pool_address:
            return []

        # Get current tick from pool
        contract_id = (
            VelodromeCLPoolContract.contract_id
            if position.get("dex_type") == DexType.VELODROME.value
            else UniswapV3PoolContract.contract_id
        )

        slot0_data = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=contract_id,
            contract_callable="slot0",
            data_key="slot0",
            chain_id=chain,
        )

        if not slot0_data or "tick" not in slot0_data:
            self.context.logger.error(
                f"Failed to get current tick for pool {pool_address}"
            )
            return []

        current_tick = slot0_data["tick"]

        # Get position manager address
        position_manager_address = (
            self.params.velodrome_non_fungible_position_manager_contract_addresses.get(
                chain
            )
            if position.get("dex_type") == DexType.VELODROME.value
            else self.params.uniswap_position_manager_contract_addresses.get(chain)
        )

        if not position_manager_address:
            self.context.logger.error(
                f"No position manager address found for chain {chain}"
            )
            return []

        # Get contract ID for position manager
        contract_id = (
            VelodromeNonFungiblePositionManagerContract.contract_id
            if position.get("dex_type") == DexType.VELODROME.value
            else UniswapV3NonfungiblePositionManagerContract.contract_id
        )

        tick_ranges = []

        # Handle both single positions and multiple positions (Velodrome CL)
        positions_to_process = (
            position["positions"] if position.get("positions") else [position]
        )

        for pos in positions_to_process:
            token_id = pos.get("token_id")
            if not token_id:
                continue

            position_data = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=position_manager_address,
                contract_public_id=contract_id,
                contract_callable="get_position",
                data_key="data",
                token_id=token_id,
                chain_id=chain,
            )

            if not position_data:
                continue

            tick_lower = position_data.get("tickLower")
            tick_upper = position_data.get("tickUpper")

            if tick_lower is not None and tick_upper is not None:
                tick_ranges.append(
                    {
                        "current_tick": current_tick,
                        "tick_lower": tick_lower,
                        "tick_upper": tick_upper,
                        "token_id": token_id,
                        "in_range": tick_lower <= current_tick <= tick_upper,
                    }
                )

        return tick_ranges

    def _update_position_with_current_value(
        self,
        position: Dict,
        current_value_usd: Decimal,
        chain: str,
        user_balances: Dict = None,
        token_info: Dict = None,
        token_prices: Dict = None,
    ) -> Generator[None, None, None]:
        """Update position with current value and corrected yield calculation"""
        try:
            # Store current value in position
            position["current_value_usd"] = float(current_value_usd)
            position["last_updated"] = int(self._get_current_timestamp())

            # Get initial token amounts from position data
            initial_amount0 = position.get("amount0", 0)  # Raw amount (with decimals)
            initial_amount1 = position.get("amount1", 0)  # Raw amount (with decimals)

            # Use provided user_balances if available, otherwise get them
            if user_balances is None:
                user_balances = yield from self._get_current_token_balances(
                    position, chain
                )

            if (
                user_balances
                and initial_amount0 is not None
                and initial_amount1 is not None
            ):
                # Calculate yield as token quantity increases priced at current prices
                yield_usd = yield from self._calculate_corrected_yield(
                    position,
                    initial_amount0,
                    initial_amount1,
                    user_balances,
                    chain,
                    token_prices,
                )

                # Check cost recovery using actual yield
                entry_cost = position.get("entry_cost", 0)
                if yield_usd >= entry_cost:
                    position["cost_recovered"] = True
                    position["yield_usd"] = float(yield_usd)
                    self.context.logger.info(
                        f"Position {position.get('pool_address')} has recovered costs: "
                        f"yield=${yield_usd:.2f} >= entry_cost=${entry_cost:.2f}"
                    )
                else:
                    position["cost_recovered"] = False
                    position["yield_usd"] = float(yield_usd)
                    recovery_percentage = (
                        (yield_usd / entry_cost) * 100 if entry_cost > 0 else 0
                    )
                    remaining_yield_needed = entry_cost - yield_usd
                    self.context.logger.info(
                        f"Position {position.get('pool_address')} cost recovery progress: "
                        f"{recovery_percentage:.1f}% (${yield_usd:.2f}/${entry_cost:.2f}), "
                        f"need ${remaining_yield_needed:.2f} more"
                    )
            else:
                # Fallback for positions without complete initial data
                position["cost_recovered"] = False
                position["yield_usd"] = 0.0

        except Exception as e:
            self.context.logger.error(
                f"Error updating position with current value: {e}"
            )
            position["cost_recovered"] = False
            position["yield_usd"] = 0.0

    def _calculate_corrected_yield(
        self,
        position: Dict,
        initial_amount0: int,
        initial_amount1: int,
        current_balances: Dict,
        chain: str,
        token_prices: Dict = None,
    ) -> Generator[Decimal, None, None]:
        """Calculate yield as token quantity increases priced at current prices"""

        token0_address = position.get("token0")
        token1_address = position.get("token1")

        # Get token decimals
        token0_decimals = yield from self._get_token_decimals(chain, token0_address)
        token1_decimals = yield from self._get_token_decimals(chain, token1_address)

        if token0_decimals is None or token1_decimals is None:
            self.context.logger.error(
                "Could not get token decimals for yield calculation"
            )
            return Decimal(0)

        # Convert initial amounts to decimal-adjusted values
        initial_token0_decimal = Decimal(initial_amount0) / Decimal(
            10**token0_decimals
        )
        initial_token1_decimal = Decimal(initial_amount1) / Decimal(
            10**token1_decimals
        )

        # Get current balances (already decimal-adjusted)
        current_token0_balance = current_balances.get(token0_address, Decimal(0))
        current_token1_balance = current_balances.get(token1_address, Decimal(0))

        # Calculate token quantity increases (the actual yield from fees)
        token0_increase = max(
            Decimal(0), current_token0_balance - initial_token0_decimal
        )
        token1_increase = max(
            Decimal(0), current_token1_balance - initial_token1_decimal
        )

        # Use provided token prices if available, otherwise fetch them
        if (
            token_prices
            and token0_address in token_prices
            and token1_address in token_prices
        ):
            token0_price = token_prices[token0_address]
            token1_price = token_prices[token1_address]
        else:
            # Price the increases at current prices
            token0_price = yield from self._fetch_token_price(token0_address, chain)
            token1_price = yield from self._fetch_token_price(token1_address, chain)

            if token0_price is None or token1_price is None:
                self.context.logger.error(
                    "Could not fetch current token prices for yield calculation"
                )
                return Decimal(0)

            token0_price = Decimal(str(token0_price))
            token1_price = Decimal(str(token1_price))

        # Calculate base yield from token increases (fees)
        base_yield_usd = (token0_increase * token0_price) + (
            token1_increase * token1_price
        )

        # Add VELO rewards if position is staked
        velo_rewards_usd = Decimal(0)
        if position.get("staked", False) and position.get("dex_type") == "velodrome":
            # Get VELO token address and check if it's in current_balances
            velo_token_address = self._get_velo_token_address(chain)
            if velo_token_address and velo_token_address in current_balances:
                velo_balance = current_balances[velo_token_address]
                if velo_balance > 0:
                    # Get VELO price
                    if token_prices and velo_token_address in token_prices:
                        velo_price = token_prices[velo_token_address]
                    else:
                        velo_coin_id = self.get_coin_id_from_symbol("VELO", chain)
                        velo_price = yield from self._fetch_coin_price(velo_coin_id)
                        if velo_price is None:
                            self.context.logger.warning(
                                "Could not fetch VELO price for yield calculation"
                            )
                            velo_price = Decimal(0)
                        else:
                            velo_price = Decimal(str(velo_price))

                    velo_rewards_usd = velo_balance * velo_price

        # Total yield = base yield + VELO rewards
        total_yield_usd = base_yield_usd + velo_rewards_usd

        # Enhanced logging
        log_message = (
            f"Corrected yield calculation for {position.get('pool_address', 'unknown')}: "
            f"Initial amounts: {initial_token0_decimal:.6f} {position.get('token0_symbol', 'TOKEN0')}, "
            f"{initial_token1_decimal:.6f} {position.get('token1_symbol', 'TOKEN1')} | "
            f"Current amounts: {current_token0_balance:.6f} {position.get('token0_symbol', 'TOKEN0')}, "
            f"{current_token1_balance:.6f} {position.get('token1_symbol', 'TOKEN1')} | "
            f"Token increases: {token0_increase:.6f} @ ${token0_price:.2f} = ${token0_increase * token0_price:.2f}, "
            f"{token1_increase:.6f} @ ${token1_price:.2f} = ${token1_increase * token1_price:.2f} | "
            f"Base yield: ${base_yield_usd:.2f}"
        )

        if velo_rewards_usd > 0:
            velo_balance = current_balances.get(
                self._get_velo_token_address(chain), Decimal(0)
            )
            velo_price = (
                velo_rewards_usd / velo_balance if velo_balance > 0 else Decimal(0)
            )
            log_message += f" | VELO rewards: {velo_balance:.6f} @ ${velo_price:.2f} = ${velo_rewards_usd:.2f}"

        log_message += f" | Total yield: ${total_yield_usd:.2f}"

        self.context.logger.info(log_message)

        return total_yield_usd

    def _get_current_token_balances(
        self, position: Dict, chain: str
    ) -> Generator[Any, Any, Dict[str, Decimal]]:
        """Get current token balances for a position (reuse existing calculation logic)"""
        dex_type = position.get("dex_type")
        current_user_shares = {}
        if dex_type == DexType.BALANCER.value:
            user_address = self.params.safe_contract_addresses.get(chain)
            pool_id = position.get("pool_id")
            pool_address = position.get("pool_address")
            current_user_shares = yield from self.get_user_share_value_balancer(
                user_address, pool_id, pool_address, chain
            )

        elif dex_type == DexType.UNISWAP_V3.value:
            pool_address = position.get("pool_address")
            token_id = position.get("token_id")
            current_user_shares = yield from self.get_user_share_value_uniswap(
                pool_address, token_id, chain, position
            )

        elif dex_type == DexType.VELODROME.value:
            user_address = self.params.safe_contract_addresses.get(chain)
            pool_address = position.get("pool_address")
            token_id = position.get("token_id")
            current_user_shares = yield from self.get_user_share_value_velodrome(
                user_address, pool_address, token_id, chain, position
            )

        elif dex_type == DexType.STURDY.value:
            user_address = self.params.safe_contract_addresses.get(chain)
            aggregator_address = position.get("pool_address")
            asset_address = position.get("token0")
            current_user_shares = yield from self.get_user_share_value_sturdy(
                user_address, aggregator_address, asset_address, chain
            )

        return current_user_shares

    def get_user_share_value_velodrome(
        self, user_address: str, pool_address: str, token_id: int, chain: str, position
    ) -> Generator[None, None, Optional[Dict[str, Decimal]]]:
        """Calculate the user's share value and token balances in a Velodrome pool."""
        token0_address = position.get("token0")
        token1_address = position.get("token1")
        is_cl_pool = position.get("is_cl_pool", False)

        if not token0_address or not token1_address:
            self.context.logger.error("Token addresses not found")
            return {}

        if is_cl_pool:
            result = yield from self._get_user_share_value_velodrome_cl(
                pool_address,
                chain,
                position,
                token0_address,
                token1_address,
            )
        else:
            result = yield from self._get_user_share_value_velodrome_non_cl(
                user_address,
                pool_address,
                chain,
                position,
                token0_address,
                token1_address,
            )
        return result

    def _get_token_decimals_pair(self, chain, token0_address, token1_address):
        token0_decimals = yield from self._get_token_decimals(chain, token0_address)
        token1_decimals = yield from self._get_token_decimals(chain, token1_address)
        if token0_decimals is None or token1_decimals is None:
            self.context.logger.error("Failed to get token decimals")
            return None, None
        return token0_decimals, token1_decimals

    def _adjust_for_decimals(self, amount, decimals):
        return Decimal(str(amount)) / Decimal(10**decimals)

    def _calculate_cl_position_value(
        self,
        pool_address: str,
        chain: str,
        position: Dict[str, Any],
        token0_address: str,
        token1_address: str,
        position_manager_address: str,
        contract_id: Any,
        dex_type: str,
        get_position_callable: str = "get_position",
        position_data_key: str = "data",
        slot0_contract_id: Any = None,
    ) -> Generator[None, None, Dict[str, Decimal]]:
        """Calculate concentrated liquidity position value."""
        # Early validation of required parameters
        if not all(
            [
                pool_address,
                chain,
                position,
                token0_address,
                token1_address,
                position_manager_address,
            ]
        ):
            self.context.logger.error("Missing required parameters")
            return {}

        # Get slot0 data in a single call
        slot0_data = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=slot0_contract_id or contract_id,
            contract_callable="slot0",
            data_key="slot0",
            chain_id=chain,
        )

        # Early return if slot0 data is invalid
        if not slot0_data:
            self.context.logger.error(
                f"Failed to get slot0 data for pool {pool_address}"
            )
            return {}

        # Extract and validate slot0 data
        sqrt_price_x96 = slot0_data.get("sqrt_price_x96")
        current_tick = slot0_data.get("tick")
        if not sqrt_price_x96 or current_tick is None:
            self.context.logger.error(f"Invalid slot0 data: {slot0_data}")
            return {}

        # Get token decimals in parallel using a list comprehension
        token_decimals = yield from self._get_token_decimals_pair(
            chain, token0_address, token1_address
        )
        if None in token_decimals:
            return {}
        token0_decimals, token1_decimals = token_decimals

        # Initialize quantities
        total_token0_qty = Decimal(0)
        total_token1_qty = Decimal(0)

        # Handle position(s)
        positions_to_process = (
            position["positions"] if position.get("positions") else [position]
        )

        # Process all positions
        for pos in positions_to_process:
            token_id = pos.get("token_id")
            if not token_id:
                continue

            position_details = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=position_manager_address,
                contract_public_id=contract_id,
                contract_callable=get_position_callable,
                data_key=position_data_key,
                token_id=token_id,
                chain_id=chain,
            )

            if not position_details:
                self.context.logger.error(
                    f"Failed to get position details for token ID {token_id}"
                )
                continue

            # Calculate amounts for this position
            amount0, amount1 = yield from self._calculate_position_amounts(
                position_details, current_tick, sqrt_price_x96, pos, dex_type, chain
            )

            total_token0_qty += Decimal(amount0)
            total_token1_qty += Decimal(amount1)

        # Calculate final adjusted quantities
        result = {
            token0_address: self._adjust_for_decimals(
                total_token0_qty, token0_decimals
            ),
            token1_address: self._adjust_for_decimals(
                total_token1_qty, token1_decimals
            ),
        }

        # Log results
        self.context.logger.info(
            f"CL Pool Total Position Balances - "
            f"Token0: {result[token0_address]} {position.get('token0_symbol')}, "
            f"Token1: {result[token1_address]} {position.get('token1_symbol')}"
        )

        return result

    def _calculate_position_amounts(
        self,
        position_details: Dict[str, Any],
        current_tick: int,
        sqrt_price_x96: int,
        position: Dict[str, Any],
        dex_type: str,
        chain: str,
    ) -> Generator[None, None, Optional[Tuple[int, int]]]:
        """Calculate token amounts with DEX-specific logic."""
        # Extract position details
        tick_lower_val = position_details.get("tickLower")
        tick_upper_val = position_details.get("tickUpper")
        liquidity_val = position_details.get("liquidity")

        if tick_lower_val is None or tick_upper_val is None or liquidity_val is None:
            self.context.logger.error("Missing required position details")
            return 0, 0

        tick_lower = int(tick_lower_val)
        tick_upper = int(tick_upper_val)
        liquidity = int(liquidity_val)
        tokens_owed0 = int(position_details.get("tokensOwed0", 0))
        tokens_owed1 = int(position_details.get("tokensOwed1", 0))
        token_id = position.get("token_id")

        # Log position details
        self.context.logger.info(
            f"For {dex_type} position, liquidity range is [{tick_lower}, {tick_upper}] "
            f"and current tick is {current_tick}"
        )

        # Use DEX-specific calculation methods
        if dex_type == DexType.VELODROME.value and token_id:
            # For Velodrome: Use Sugar contract's principal() function for maximum accuracy
            position_manager_address = self.params.velodrome_non_fungible_position_manager_contract_addresses.get(
                chain
            )
            if position_manager_address:
                self.context.logger.info(
                    "Using Velodrome Sugar contract for position calculation"
                )
                amount0, amount1 = yield from self.get_velodrome_position_principal(
                    chain, position_manager_address, token_id, sqrt_price_x96
                )
            else:
                self.context.logger.warning(
                    "No Velodrome position manager found, falling back to Sugar getAmountsForLiquidity"
                )
                sqrt_a = yield from self.get_velodrome_sqrt_ratio_at_tick(
                    chain, tick_lower
                )
                sqrt_b = yield from self.get_velodrome_sqrt_ratio_at_tick(
                    chain, tick_upper
                )
                amount0, amount1 = yield from self.get_velodrome_amounts_for_liquidity(
                    chain, sqrt_price_x96, sqrt_a, sqrt_b, liquidity
                )
        else:
            # For Uniswap and others: Use existing tick math implementation
            self.context.logger.info(
                "Using custom tick math for non-Velodrome position"
            )
            sqrtA = get_sqrt_ratio_at_tick(tick_lower)
            sqrtB = get_sqrt_ratio_at_tick(tick_upper)
            amount0, amount1 = get_amounts_for_liquidity(
                sqrt_price_x96, sqrtA, sqrtB, liquidity
            )

        # Add uncollected fees
        amount0 += tokens_owed0
        amount1 += tokens_owed1

        # Log the calculation results
        if tick_lower <= current_tick <= tick_upper:
            self.context.logger.info(
                f"Position is in range. Current tick: {current_tick}, "
                f"Range: [{tick_lower}, {tick_upper}], "
                f"Calculated amounts: {amount0}/{amount1} (including fees: {tokens_owed0}/{tokens_owed1})"
            )
        else:
            self.context.logger.info(
                f"Position is out of range. Current tick: {current_tick}, "
                f"Range: [{tick_lower}, {tick_upper}], "
                f"Calculated amounts: {amount0}/{amount1} (including fees: {tokens_owed0}/{tokens_owed1})"
            )

        return amount0, amount1

    def _get_user_share_value_velodrome_cl(
        self,
        pool_address,
        chain,
        position,
        token0_address,
        token1_address,
    ):
        """Calculate the user's share value and token balances in a Velodrome CL pool."""
        position_manager_address = (
            self.params.velodrome_non_fungible_position_manager_contract_addresses.get(
                chain
            )
        )
        if not position_manager_address:
            self.context.logger.error(
                f"No position manager address found for chain {chain}"
            )
            return {}

        return (
            yield from self._calculate_cl_position_value(
                pool_address=pool_address,
                chain=chain,
                position=position,
                token0_address=token0_address,
                token1_address=token1_address,
                position_manager_address=position_manager_address,
                contract_id=VelodromeNonFungiblePositionManagerContract.contract_id,
                dex_type=DexType.VELODROME.value,
                get_position_callable="get_position",
                position_data_key="data",
                slot0_contract_id=VelodromeCLPoolContract.contract_id,
            )
        )

    def _get_user_share_value_velodrome_non_cl(
        self,
        user_address,
        pool_address,
        chain,
        position,
        token0_address,
        token1_address,
    ):
        """Calculate the user's share value and token balances in a Velodrome non-CL pool."""
        user_balance = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=ERC20.contract_id,
            contract_callable="check_balance",
            data_key="token",
            account=user_address,
            chain_id=chain,
        )
        if user_balance is None:
            self.context.logger.error(
                f"Failed to get user balance for pool: {pool_address}"
            )
            return {}

        total_supply = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=ERC20.contract_id,
            contract_callable="get_total_supply",
            data_key="data",
            chain_id=chain,
        )
        if not total_supply:
            self.context.logger.error(
                f"Failed to get total supply for pool: {pool_address}"
            )
            return {}

        reserves = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=VelodromePoolContract.contract_id,
            contract_callable="get_reserves",
            data_key="data",
            chain_id=chain,
        )
        if not reserves:
            self.context.logger.error(
                f"Failed to get reserves for pool: {pool_address}"
            )
            return {}

        getcontext().prec = 50
        user_share = Decimal(str(user_balance)) / Decimal(str(total_supply))
        token0_decimals, token1_decimals = yield from self._get_token_decimals_pair(
            chain, token0_address, token1_address
        )
        if token0_decimals is None or token1_decimals is None:
            return {}

        token0_balance = self._adjust_for_decimals(reserves[0], token0_decimals)
        token1_balance = self._adjust_for_decimals(reserves[1], token1_decimals)
        user_token0_balance = user_share * token0_balance
        user_token1_balance = user_share * token1_balance

        self.context.logger.info(
            f"Velodrome Non-CL Pool Balances - "
            f"User share: {user_share}, "
            f"Token0: {user_token0_balance} {position.get('token0_symbol')}, "
            f"Token1: {user_token1_balance} {position.get('token1_symbol')}"
        )
        return {
            token0_address: user_token0_balance,
            token1_address: user_token1_balance,
        }

    def get_user_share_value_uniswap(
        self, pool_address: str, token_id: int, chain: str, position
    ) -> Generator[None, None, Optional[Dict[str, Decimal]]]:
        """Calculate the user's share value and token balances in a Uniswap V3 position."""
        token0_address = position.get("token0")
        token1_address = position.get("token1")

        if not token0_address or not token1_address or not token_id:
            self.context.logger.error("Token addresses or token_id not found")
            return {}

        # Get the position manager address for the chain
        position_manager_address = (
            self.params.uniswap_position_manager_contract_addresses.get(chain)
        )
        if not position_manager_address:
            self.context.logger.error(
                f"No position manager address found for chain {chain}"
            )
            return {}

        # Use the common helper function for CL position calculation
        return (
            yield from self._calculate_cl_position_value(
                pool_address=pool_address,
                chain=chain,
                position=position,
                token0_address=token0_address,
                token1_address=token1_address,
                position_manager_address=position_manager_address,
                contract_id=UniswapV3NonfungiblePositionManagerContract.contract_id,
                dex_type=DexType.UNISWAP_V3.value,
                get_position_callable="get_position",
                position_data_key="data",
                slot0_contract_id=UniswapV3PoolContract.contract_id,
            )
        )

    def get_user_share_value_balancer(
        self, user_address: str, pool_id: str, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[Dict[str, Decimal]]]:
        """Calculate the user's share value and token balances in a Balancer pool using direct contract calls."""

        # Step 1: Get the pool tokens and balances from the Vault contract
        vault_address = self.params.balancer_vault_contract_addresses.get(chain)
        if not vault_address:
            self.context.logger.error(f"Vault address not found for chain: {chain}")
            return {}

        pool_tokens_data = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=vault_address,
            contract_public_id=VaultContract.contract_id,
            contract_callable="get_pool_tokens",
            data_key="tokens",
            pool_id=pool_id,
            chain_id=chain,
        )

        if not pool_tokens_data:
            self.context.logger.error(
                f"Failed to get pool tokens for pool ID: {pool_id}"
            )
            return {}

        tokens = pool_tokens_data[0]  # Array of token addresses
        balances = pool_tokens_data[1]  # Array of token balances

        # Step 2: Get the user's balance of pool tokens (BPT)
        user_balance = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=ERC20.contract_id,
            contract_callable="check_balance",
            data_key="token",
            account=user_address,
            chain_id=chain,
        )

        if user_balance is None:
            self.context.logger.error(
                f"Failed to get user balance for pool: {pool_address}"
            )
            return {}

        # Step 3: Get the total supply of pool tokens
        total_supply = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=ERC20.contract_id,
            contract_callable="get_total_supply",
            data_key="data",
            chain_id=chain,
        )

        if total_supply is None or total_supply == 0:
            self.context.logger.error(
                f"Failed to get total supply for pool: {pool_address}"
            )
            return {}

        # Step 4: Calculate the user's share of the pool
        getcontext().prec = 50  # Increase decimal precision
        ctx = Context(prec=50)  # Use higher-precision data type

        user_balance_decimal = Decimal(str(user_balance))
        total_supply_decimal = Decimal(str(total_supply))

        if total_supply_decimal == 0:
            self.context.logger.error(f"Total supply is zero for pool: {pool_address}")
            return {}

        user_share = ctx.divide(user_balance_decimal, total_supply_decimal)

        # Step 5: Calculate user's token balances
        user_token_balances = {}
        for i, token_address in enumerate(tokens):
            token_address = to_checksum_address(token_address)
            token_balance = Decimal(str(balances[i]))

            # Get token decimals
            token_decimals = yield from self._get_token_decimals(chain, token_address)
            if token_decimals is None:
                self.context.logger.error(
                    f"Failed to get decimals for token: {token_address}"
                )
                continue

            # Adjust token balance based on decimals
            adjusted_token_balance = token_balance / Decimal(10**token_decimals)

            # Calculate user's token balance
            user_token_balance = user_share * adjusted_token_balance
            user_token_balances[token_address] = user_token_balance

        return user_token_balances

    def get_user_share_value_sturdy(
        self, user_address: str, aggregator_address: str, asset_address: str, chain: str
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Calculate the user's share value and token balance in a Sturdy vault."""
        # Get user's underlying asset balance in the vault
        user_asset_balance = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=aggregator_address,
            contract_public_id=YearnV3VaultContract.contract_id,
            contract_callable="balance_of",
            data_key="amount",
            owner=user_address,
            chain_id=chain,
        )
        if user_asset_balance is None:
            self.context.logger.error("Failed to get user's asset balance.")
            return {}

        user_asset_balance = Decimal(user_asset_balance)

        # Get decimals for proper scaling
        decimals = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=aggregator_address,
            contract_public_id=YearnV3VaultContract.contract_id,
            contract_callable="decimals",
            data_key="decimals",
            chain_id=chain,
        )
        if decimals is None:
            self.context.logger.error("Failed to get decimals.")
            return {}

        scaling_factor = Decimal(10 ** int(decimals))

        # Adjust decimals for assets
        user_asset_balance /= scaling_factor

        return {asset_address: user_asset_balance}

    def calculate_airdrop_rewards_value(self) -> Generator[None, None, Decimal]:
        """Calculate airdrop rewards equivalent in USD (MODE chain only)"""
        chain = self.params.target_investment_chains[0]
        if chain != "mode":
            return Decimal(0)

        airdrop_rewards_wei = yield from self._get_total_airdrop_rewards(chain)
        if airdrop_rewards_wei > 0:
            # Convert from wei to USDC (6 decimals for USDC)
            airdrop_usdc_balance = Decimal(str(airdrop_rewards_wei)) / Decimal(10**6)

            # Fetch actual USDC price
            usdc_address = self._get_usdc_address(chain)
            usdc_price = yield from self._fetch_token_price(usdc_address, chain)

            if usdc_price is not None:
                usdc_price_decimal = Decimal(str(usdc_price))
                airdrop_value_usd = airdrop_usdc_balance * usdc_price_decimal

                self.context.logger.info(
                    f"USDC airdrop rewards - USDC: {airdrop_usdc_balance} @ ${usdc_price} = ${airdrop_value_usd}"
                )
            else:
                # Fallback to $1 if price fetch fails
                airdrop_value_usd = airdrop_usdc_balance
                self.context.logger.warning(
                    f"Could not fetch USDC price, using $1 fallback - USDC: {airdrop_usdc_balance} (${airdrop_value_usd})"
                )

            return airdrop_value_usd

        return Decimal(0)

    def calculate_stakig_rewards_value(self) -> Generator[None, None, Decimal]:
        """Calculates staking rewards equivalent in USD"""
        chain = self.params.target_investment_chains[0]
        yield from self.update_accumulated_rewards_for_chain(chain)
        # After processing all balances, add OLAS rewards separately
        olas_address = OLAS_ADDRESSES.get(chain)
        if olas_address:
            accumulated_olas_rewards = (
                yield from self.get_accumulated_rewards_for_token(chain, olas_address)
            )
            if accumulated_olas_rewards > 0:
                # Convert from wei to OLAS (18 decimals)
                olas_balance = Decimal(str(accumulated_olas_rewards)) / Decimal(
                    10**18
                )

                # Get OLAS price
                olas_price = yield from self._fetch_token_price(olas_address, chain)
                if olas_price is not None:
                    olas_price = Decimal(str(olas_price))
                    olas_value_usd = olas_balance * olas_price

                    self.context.logger.info(
                        f"OLAS accumulated rewards - OLAS: {olas_balance} (${olas_value_usd})"
                    )
                    return olas_value_usd

                else:
                    self.context.logger.warning(
                        "Could not fetch price for OLAS rewards"
                    )
                    return Decimal(0)

        return Decimal(0)

    def calculate_withdrawals_value(self) -> Generator[None, None, Decimal]:
        """Calculate the value of withdrawals."""
        chain = self.params.target_investment_chains[0]

        if chain == "mode":
            all_erc20_transfers_mode = self._track_erc20_transfers_mode(
                self.params.safe_contract_addresses.get(chain),
                datetime.now().timestamp(),
            )
            if all_erc20_transfers_mode is None:
                self.context.logger.warning(
                    "Failed to fetch ERC20 transfers, returning zero withdrawal value"
                )
                return Decimal(0)
            outgoing_erc20_transfers_mode = all_erc20_transfers_mode["outgoing"]
            self.context.logger.info(
                f"Outgoing ERC20 transfers: {outgoing_erc20_transfers_mode}"
            )
            withdrawal_value = (
                yield from self._track_and_calculate_withdrawal_value_mode(
                    outgoing_erc20_transfers_mode,
                )
            )
            self.context.logger.info(f"Withdrawal value: ${withdrawal_value}")
            return withdrawal_value
        else:
            return Decimal(0)

    def _track_and_calculate_withdrawal_value_mode(
        self,
        outgoing_erc20_transfers: Dict,
    ) -> Generator[None, None, Decimal]:
        """Track USDC transfers from safe address and handle withdrawal logic."""
        try:
            if not outgoing_erc20_transfers:
                self.context.logger.warning(
                    "No outgoing transfers found for Mode chain"
                )
                return Decimal(0)

            # Track USDC transfers
            usdc_transfers = []
            withdrawal_transfers = []
            withdrawal_value = Decimal(0)

            # Sort transfers by timestamp
            sorted_outgoing_transfers = []
            for _, transfers in outgoing_erc20_transfers.items():
                for transfer in transfers:
                    if isinstance(transfer, dict) and "timestamp" in transfer:
                        sorted_outgoing_transfers.append(transfer)

            sorted_outgoing_transfers.sort(key=lambda x: x["timestamp"])

            for transfer in sorted_outgoing_transfers:
                if transfer.get("symbol") == "USDC":
                    usdc_transfers.append(transfer)
                    withdrawal_transfers.append(transfer)
            self.context.logger.info(f"USDC transfers: {usdc_transfers}")
            self.context.logger.info(f"Withdrawal transfers: {withdrawal_transfers}")
            withdrawal_value = yield from self._calculate_total_withdrawal_value(
                withdrawal_transfers
            )
            return withdrawal_value
        except Exception as e:
            self.context.logger.error(
                f"Error calculating total withdrawal value: {str(e)}"
            )
            return Decimal(0)

    def _calculate_total_withdrawal_value(
        self,
        withdrawal_transfers: List[Dict],
    ) -> Generator[None, None, Decimal]:
        """Calculate the total withdrawal value."""
        withdrawal_value = Decimal(0)

        for _, transfer in enumerate(withdrawal_transfers):
            # Parse ISO timestamp format
            timestamp_str = transfer.get("timestamp", "")
            if timestamp_str:
                try:
                    # Parse ISO format timestamp
                    transfer_datetime = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    transfer_date = transfer_datetime.strftime("%d-%m-%Y")
                except (ValueError, TypeError) as e:
                    self.context.logger.error(
                        f"Error parsing timestamp {timestamp_str}: {e}"
                    )
                    continue
            else:
                self.context.logger.warning("No timestamp found in transfer")
                continue
            # Get USDC coin ID for Mode chain
            usdc_coin_id = self.get_coin_id_from_symbol("USDC", "mode")
            if not usdc_coin_id:
                self.context.logger.warning("No coin ID found for USDC on Mode chain")
                continue

            usdc_price = yield from self._fetch_historical_token_price(
                usdc_coin_id, transfer_date
            )
            if usdc_price:
                withdrawal_amount = Decimal(str(transfer.get("amount", 0)))
                usdc_price_decimal = Decimal(str(usdc_price))
                withdrawal_value += withdrawal_amount * usdc_price_decimal
            else:
                self.context.logger.warning(f"No USDC price found for {transfer_date}")

        return withdrawal_value

    def check_and_update_zero_liquidity_positions(self) -> None:
        """Check for positions with zero liquidity and mark them as closed, and reopen closed positions that now have liquidity."""
        if not self.current_positions:
            return

        # First pass: Mark OPEN positions as CLOSED if liquidity = 0
        for position in self.current_positions:
            if position.get("status") != PositionStatus.OPEN.value:
                continue

            dex_type = position.get("dex_type")

            if dex_type == DexType.VELODROME.value and position.get("is_cl_pool"):
                # For Velodrome CL pools, check all sub-positions
                all_positions_zero = True
                for pos in position.get("positions", []):
                    if (
                        pos.get("current_liquidity", 1) != 0
                    ):  # Default to 1 if not found to avoid false closures
                        all_positions_zero = False
                        break

                if all_positions_zero and position.get(
                    "positions"
                ):  # Only update if there are positions
                    position["status"] = PositionStatus.CLOSED.value
                    self.context.logger.info(
                        f"Marked Velodrome CL position as closed due to zero liquidity in all positions: {position}"
                    )
            else:
                # For all other position types
                if (
                    position.get("current_liquidity", 1) == 0
                ):  # Default to 1 if not found to avoid false closures
                    position["status"] = PositionStatus.CLOSED.value
                    self.context.logger.info(
                        f"Marked {dex_type} position as closed due to zero liquidity: {position}"
                    )

        # Store the updated positions
        self.store_current_positions()

    def update_position_amounts(self) -> Generator[None, None, None]:
        """Update the amounts of all open positions."""
        if not self.current_positions:
            self.context.logger.info("No positions to update.")
            return

        for position in self.current_positions:
            # Only update open positions
            if position.get("status") != PositionStatus.OPEN.value:
                continue

            dex_type = position.get("dex_type")
            chain = position.get("chain")

            if not dex_type or not chain:
                self.context.logger.warning(
                    f"Position missing dex_type or chain: {position}"
                )
                continue

            self.context.logger.info(
                f"Updating position of type {dex_type} on chain {chain}"
            )

            # Update based on the type of position
            if dex_type == DexType.BALANCER.value:
                yield from self._update_balancer_position(position)
            elif dex_type == DexType.UNISWAP_V3.value:
                yield from self._update_uniswap_position(position)
            elif dex_type == DexType.VELODROME.value:
                yield from self._update_velodrome_position(position)
            elif dex_type == DexType.STURDY.value:
                yield from self._update_sturdy_position(position)
            else:
                self.context.logger.warning(f"Unknown position type: {dex_type}")

        # Store the updated positions
        self.store_current_positions()

    def _update_balancer_position(
        self, position: Dict[str, Any]
    ) -> Generator[None, None, None]:
        """Update a Balancer position."""
        pool_address = position.get("pool_address")
        safe_address = self.params.safe_contract_addresses.get(position.get("chain"))
        chain = position.get("chain")

        if not all([pool_address, safe_address, chain]):
            self.context.logger.warning(
                f"Missing required parameters for Balancer position: {position}"
            )
            return

        # Get the current balance of LP tokens
        balance = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=WeightedPoolContract.contract_id,
            contract_callable="get_balance",
            data_key="balance",
            account=safe_address,
            chain_id=chain,
        )

        if balance is not None:
            # Update the position with the current liquidity
            position["current_liquidity"] = balance
            self.context.logger.info(f"Updated Balancer position amount: {balance}")
        else:
            self.context.logger.warning(
                f"Failed to get balance for Balancer position: {position}"
            )

    def _update_uniswap_position(
        self, position: Dict[str, Any]
    ) -> Generator[None, None, None]:
        """Update a Uniswap V3 position."""
        token_id = position.get("token_id")
        chain = position.get("chain")

        if not all([token_id, chain]):
            self.context.logger.warning(
                f"Missing required parameters for Uniswap position: {position}"
            )
            return

        position_manager_address = (
            self.params.uniswap_position_manager_contract_addresses.get(chain)
        )
        if not position_manager_address:
            self.context.logger.warning(
                f"No position manager address found for chain {chain}"
            )
            return

        # Get the current liquidity
        position_data = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=position_manager_address,
            contract_public_id=UniswapV3NonfungiblePositionManagerContract.contract_id,
            contract_callable="get_position",
            data_key="data",
            token_id=token_id,
            chain_id=chain,
        )

        if position_data and position_data.get("liquidity"):
            # Update the position with the current liquidity
            position["current_liquidity"] = position_data.get("liquidity")
            self.context.logger.info(
                f"Updated Uniswap position liquidity: {position['current_liquidity']}"
            )
        else:
            self.context.logger.warning(
                f"Failed to get liquidity for Uniswap position: {position}"
            )

    def _update_velodrome_position(
        self, position: Dict[str, Any]
    ) -> Generator[None, None, None]:
        """Update a Velodrome position."""
        chain = position.get("chain")
        is_cl_pool = position.get("is_cl_pool", False)

        if not chain:
            self.context.logger.warning(
                f"Missing required parameters for Velodrome position: {position}"
            )
            return

        if is_cl_pool:
            # Handle Velodrome concentrated liquidity pool
            for pos in position.get("positions"):
                token_id = pos.get("token_id")
                if not token_id:
                    self.context.logger.warning(
                        f"Missing token_id for Velodrome CL position: {pos}"
                    )
                    return

                position_manager_address = self.params.velodrome_non_fungible_position_manager_contract_addresses.get(
                    chain
                )
                if not position_manager_address:
                    self.context.logger.warning(
                        f"No position manager address found for chain {chain}"
                    )
                    return

                # Get the current liquidity
                position_data = yield from self.contract_interact(
                    performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                    contract_address=position_manager_address,
                    contract_public_id=VelodromeNonFungiblePositionManagerContract.contract_id,
                    contract_callable="get_position",
                    data_key="data",
                    token_id=token_id,
                    chain_id=chain,
                )

                if position_data and position_data.get("liquidity"):
                    # Update the position with the current liquidity
                    pos["current_liquidity"] = position_data.get("liquidity")
                    self.context.logger.info(
                        f"Updated Uniswap position liquidity: {pos['current_liquidity']}"
                    )
                else:
                    self.context.logger.warning(
                        f"Failed to get liquidity for Uniswap position: {pos}"
                    )
        else:
            # Handle Velodrome stable/volatile pool
            pool_address = position.get("pool_address")
            safe_address = self.params.safe_contract_addresses.get(
                position.get("chain")
            )

            if not all([pool_address, safe_address]):
                self.context.logger.warning(
                    f"Missing required parameters for Velodrome pool position: {position}"
                )
                return

            # Get the current balance of LP tokens
            balance = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=pool_address,
                contract_public_id=VelodromePoolContract.contract_id,
                contract_callable="get_balance",
                data_key="balance",
                account=safe_address,
                chain_id=chain,
            )

            if balance is not None:
                # Update the position with the current amount
                position["current_liquidity"] = balance
                self.context.logger.info(
                    f"Updated Velodrome pool position amount: {balance}"
                )
            else:
                self.context.logger.warning(
                    f"Failed to get balance for Velodrome pool position: {position}"
                )

    def _update_sturdy_position(
        self, position: Dict[str, Any]
    ) -> Generator[None, None, None]:
        """Update a Sturdy position."""
        # For Sturdy positions, we need to call balance_of on the relevant contract
        pool_address = position.get("pool_address")
        safe_address = self.params.safe_contract_addresses.get(position.get("chain"))
        chain = position.get("chain")

        if not all([pool_address, safe_address, chain]):
            self.context.logger.warning(
                f"Missing required parameters for Sturdy position: {position}"
            )
            return

        # Get the current balance
        balance = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=YearnV3VaultContract.contract_id,
            contract_callable="balance_of",
            data_key="amount",
            owner=safe_address,
            chain_id=chain,
        )

        if balance is not None:
            # Update the position with the current amount
            position["current_liquidity"] = balance
            self.context.logger.info(f"Updated Sturdy position amount: {balance}")
        else:
            self.context.logger.warning(
                f"Failed to get balance for Sturdy position: {position}"
            )

    def calculate_initial_investment_value_from_funding_events(
        self,
    ) -> Generator[None, None, Optional[float]]:
        """Calculate initial investment value using transfers."""
        total_investment = 0.0

        for chain in self.params.target_investment_chains:
            safe_address = self.params.safe_contract_addresses.get(chain)
            if not safe_address:
                self.context.logger.warning(f"No safe address found for {chain} chain")
                continue

            self.context.logger.info(
                f"Calculating initial investment from {chain} transfers for address: {safe_address}"
            )

            current_date = datetime.now().strftime("%Y-%m-%d")

            # Default to not fetching full history
            fetch_till_date = False

            # Check if airdrop detection is enabled and if we need a full historical scan
            if chain == "mode" and self.params.airdrop_started:
                airdrop_scan_completed = yield from self._read_kv(
                    ("airdrop_full_scan_completed",)
                )

                if not airdrop_scan_completed or not airdrop_scan_completed.get(
                    "airdrop_full_scan_completed"
                ):
                    # First time airdrop is enabled - do full historical scan
                    self.context.logger.info(
                        "Airdrop detection enabled for first time - performing full historical scan"
                    )
                    fetch_till_date = True
                    # Mark scan as completed after this run
                    yield from self._write_kv({"airdrop_full_scan_completed": "true"})
                else:
                    self.context.logger.info(
                        "Airdrop detection enabled - using incremental scan (full scan already completed)"
                    )
                    fetch_till_date = False
            else:
                # Normal logic when airdrop is not started
                # Check when we last calculated initial value
                last_calculated_timestamp = yield from self._read_kv(
                    keys=("last_initial_value_calculated_timestamp",)
                )

                if (
                    last_calculated_timestamp
                    and (
                        timestamp := last_calculated_timestamp.get(
                            "last_initial_value_calculated_timestamp"
                        )
                    )
                    and timestamp is not None
                ):
                    self.context.logger.info(
                        f"Found last calculation timestamp: {timestamp}"
                    )
                    try:
                        last_date = datetime.utcfromtimestamp(int(timestamp)).strftime(
                            "%Y-%m-%d"
                        )
                        self.context.logger.info(f"Last calculation date: {last_date}")
                    except (ValueError, TypeError):
                        self.context.logger.warning(
                            "Invalid timestamp format, defaulting to 1970-01-01"
                        )
                        last_date = "1970-01-01"

                    # If last calculation was today, return cached value
                    if last_date == current_date:
                        self.context.logger.info(
                            "Last calculation was today, using cached value"
                        )
                        investment = yield from self._load_chain_total_investment(chain)
                        if investment:
                            return investment
                        else:
                            fetch_till_date = True

                    # Otherwise need to calculate new value but not full history
                    self.context.logger.info(
                        "Last calculation was not today, calculating new value without full history"
                    )
                    fetch_till_date = False

                # No previous calculation, need to fetch full history
                else:
                    self.context.logger.info(
                        "No previous calculation found, fetching full transfer history"
                    )
                    fetch_till_date = True

            # Fetch all transfers until current date based on chain
            self.context.logger.info(f"Fetching transfers for chain: {chain}")
            if chain == Chain.MODE.value:
                self.context.logger.info("Using Mode-specific transfer fetching")
                all_transfers = yield from self._fetch_all_transfers_until_date_mode(
                    safe_address, current_date, fetch_till_date
                )
            elif chain == Chain.OPTIMISM.value:
                self.context.logger.info("Using Optimism-specific transfer fetching")
                all_transfers = (
                    yield from self._fetch_all_transfers_until_date_optimism(
                        safe_address, current_date
                    )
                )
            else:
                self.context.logger.warning(f"Unsupported chain: {chain}, skipping")
                continue

            if not all_transfers:
                self.context.logger.warning(f"No transfers found for {chain} chain")
                continue

            # Calculate investment value for this chain
            chain_investment = yield from self._calculate_chain_investment_value(
                all_transfers, chain, safe_address
            )
            total_investment += chain_investment

        yield from self._save_chain_total_investment(chain, total_investment)

        timestamp = int(self._get_current_timestamp())
        yield from self._write_kv(
            {"last_initial_value_calculated_timestamp": str(timestamp)}
        )

        self.context.logger.info(
            f"Total initial investment from all chains: ${total_investment}"
        )

        return total_investment if total_investment > 0 else None

    def _calculate_chain_investment_value(
        self, all_transfers: Dict, chain: str, safe_address: str
    ) -> Generator[None, None, float]:
        """Calculate investment value for a specific chain and update stored total."""
        new_investment = 0.0
        total_reversion = 0.0

        # Track ETH transfers and reversions first
        reversion_info = yield from self._track_eth_transfers_and_reversions(
            safe_address, chain
        )
        reversion_amount = reversion_info.get("reversion_amount", 0)
        historical_reversion_value = reversion_info.get(
            "historical_reversion_value", 0.0
        )
        reversion_date = reversion_info.get("reversion_date")

        if historical_reversion_value > 0:
            total_reversion += historical_reversion_value
            self.context.logger.info(
                f"{chain.upper()} REVERSION: {historical_reversion_value} ETH (from {reversion_date})"
            )

        if reversion_amount > 0:
            date_str = (
                reversion_date
                if reversion_date
                else datetime.now().strftime("%d-%m-%Y")
            )
            eth_price = yield from self._fetch_historical_eth_price(date_str)
            if eth_price:
                reversion_value = reversion_amount * eth_price
                total_reversion += reversion_value
                self.context.logger.info(
                    f"{chain.upper()} REVERSION: {reversion_amount} ETH @ ${eth_price} = ${reversion_value} (from {reversion_date})"
                )

        for date, transfers in all_transfers.items():
            for transfer in transfers:
                try:
                    # Get token price for the transfer date
                    token_symbol = transfer.get("symbol", "Unknown")
                    amount = transfer.get("delta", transfer.get("amount", 0))

                    if amount <= 0:
                        continue

                    # Check if this is an airdropped USDC transfer and exclude it from initial investment
                    if (
                        chain == "mode"
                        and token_symbol.upper() == "USDC"
                        and self.params.airdrop_started
                        and self.params.airdrop_contract_address
                    ):
                        from_address = transfer.get("from_address", "")
                        if (
                            from_address.lower()
                            == self.params.airdrop_contract_address.lower()
                        ):
                            self.context.logger.info(
                                f"Excluding airdropped USDC transfer from initial investment: {amount} USDC from {from_address}"
                            )
                            continue

                    # Get historical price for the transfer date
                    date_str = datetime.strptime(date, "%Y-%m-%d").strftime("%d-%m-%Y")

                    if token_symbol == "ETH":  # nosec B105
                        price = yield from self._fetch_historical_eth_price(date_str)
                    else:
                        coingecko_id = self.get_coin_id_from_symbol(token_symbol, chain)
                        if coingecko_id:
                            price = yield from self._fetch_historical_token_price(
                                coingecko_id, date_str
                            )
                        else:
                            price = None

                    if price:
                        transfer_value = amount * price
                        new_investment += transfer_value
                        self.context.logger.info(
                            f"{chain.upper()} NEW transfer on {date}: {amount} {token_symbol} @ ${price} = ${transfer_value}"
                        )

                except Exception as e:
                    self.context.logger.error(
                        f"Error processing {chain} transfer: {str(e)}"
                    )
                    continue

        # Update total investment for this chain
        updated_total = new_investment - total_reversion
        yield from self._save_chain_total_investment(chain, updated_total)

        self.context.logger.info(
            f"Total {chain} investment (updated): ${updated_total}"
        )

        return updated_total

    def _fetch_all_transfers_until_date_mode(
        self, address: str, end_date: str, fetch_till_date: bool
    ) -> Generator[None, None, Dict]:
        """Fetch all Mode transfers from the beginning until a specific date, organized by date."""
        # Load existing unified data from kv_store
        self.funding_events = self.read_funding_events()
        if self.funding_events:
            existing_mode_data = self.funding_events.get("mode", {})
        else:
            self.funding_events = {}
            existing_mode_data = {}

        # Check for backward compatibility - if any ETH transfers don't have "delta" field
        # then we need to refetch everything with the new format
        needs_refetch = False
        if existing_mode_data:
            for date, transfers in existing_mode_data.items():
                for transfer in transfers:
                    if transfer.get("type") == "eth" and "delta" not in transfer:
                        self.context.logger.info(
                            f"Found ETH transfer without delta field on {date}. "
                            "Setting fetch_till_date=True and clearing funding_events for refetch."
                        )
                        needs_refetch = True
                        break
                if needs_refetch:
                    break

        if needs_refetch:
            # Clear existing data and force full refetch
            existing_mode_data = {}
            self.funding_events = {}
            fetch_till_date = True
            self.context.logger.info(
                "Cleared funding_events for backward compatibility refetch"
            )

        all_transfers_by_date = defaultdict(list)
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        end_datetime = end_datetime.replace(tzinfo=timezone.utc)

        try:
            self.context.logger.info(f"Fetching all Mode transfers until {end_date}...")

            # Fetch token transfers
            self.context.logger.info("Fetching Mode token transfers...")

            success = yield from self._fetch_token_transfers_mode(
                address, end_datetime, all_transfers_by_date, fetch_till_date
            )
            if not success:
                self.context.logger.info("No token transfers found for Mode")
                all_transfers_by_date = self.funding_events["mode"]

            # Fetch ETH transfers
            self.context.logger.info("Fetching Mode ETH transfers...")
            self._fetch_eth_transfers_mode(
                address, end_datetime, all_transfers_by_date, fetch_till_date
            )
            if not success:
                self.context.logger.info("No ETH transfers found for Mode")
                all_transfers_by_date = self.funding_events["mode"]

            # Merge with existing data and save
            for date, transfers in all_transfers_by_date.items():
                if date not in existing_mode_data:  # Only store new dates
                    existing_mode_data[date] = transfers

            # Update unified data structure
            if not self.funding_events:
                self.funding_events = {}
            self.funding_events["mode"] = existing_mode_data
            self.store_funding_events()

            # Print summary
            total_dates = len(all_transfers_by_date)
            total_transfers = sum(
                len(transfers) for transfers in all_transfers_by_date.values()
            )

            self.context.logger.info(
                f"Mode Summary: {total_dates} dates with transfers, {total_transfers} total transfers"
            )

            return dict(all_transfers_by_date)

        except Exception as e:
            self.context.logger.error(f"Error fetching Mode transfers: {e}")
            self.read_funding_events()
            all_transfers_by_date = self.funding_events.get("mode", {})
            return all_transfers_by_date

    def _fetch_all_transfers_until_date_optimism(
        self, address: str, end_date: str
    ) -> Generator[None, None, Dict]:
        """Fetch all Optimism transfers from the beginning until a specific date, organized by date."""
        # Load existing unified data from kv_store
        self.funding_events = self.read_funding_events()
        if self.funding_events:
            existing_optimism_data = self.funding_events.get("optimism", {})
        else:
            existing_optimism_data = {}

        all_transfers_by_date = defaultdict(list)

        try:
            self.context.logger.info(
                f"Fetching all Optimism transfers until {end_date}..."
            )

            # Use SafeGlobal API for Optimism
            yield from self._fetch_optimism_transfers_safeglobal(
                address, end_date, all_transfers_by_date, existing_optimism_data
            )

            # Merge with existing data and save
            for date, transfers in all_transfers_by_date.items():
                if date not in existing_optimism_data:  # Only store new dates
                    existing_optimism_data[date] = transfers

            # Update unified data structure
            if not self.funding_events:
                self.funding_events = {}
            self.funding_events["optimism"] = existing_optimism_data
            self.store_funding_events()

            # Print summary
            total_dates = len(all_transfers_by_date)
            total_transfers = sum(
                len(transfers) for transfers in all_transfers_by_date.values()
            )

            self.context.logger.info(
                f"Optimism Summary: {total_dates} dates with transfers, {total_transfers} total transfers"
            )

            return dict(all_transfers_by_date)

        except Exception as e:
            self.context.logger.error(f"Error fetching Optimism transfers: {e}")
            return {}

    def _fetch_token_transfers(
        self,
        address: str,
        end_datetime: datetime,
        all_transfers_by_date: dict,
        existing_data: dict,
    ) -> Generator[None, None, None]:
        """Fetch token transfers from Mode blockchain explorer."""
        base_url = "https://explorer-mode-mainnet-0.t.conduit.xyz/api/v2"
        processed_count = 0

        endpoint = f"{base_url}/addresses/{address}/token-transfers"
        success, response_json = yield from self._request_with_retries(
            endpoint=endpoint,
            headers={"Accept": "application/json"},
            rate_limited_code=429,
            rate_limited_callback=self.coingecko.rate_limited_status_callback,
            retry_wait=self.params.sleep_time,
        )

        if not success:
            self.context.logger.error("Failed to fetch token transfers")
            return None

        transfers = response_json.get("items", [])
        if not transfers:
            return None

        for tx in transfers:
            tx_datetime = self._get_datetime_from_timestamp(tx.get("timestamp"))
            tx_date = tx_datetime.strftime("%Y-%m-%d") if tx_datetime else None

            # Stop if we've gone past our end date
            if tx_datetime and tx_datetime > end_datetime:
                continue

            # Skip if date already exists in stored data
            if tx_date and tx_date in existing_data:
                continue

            if tx_date and tx_date <= end_datetime.strftime("%Y-%m-%d"):
                from_address = tx.get("from", {})
                if self._should_include_transfer(
                    from_address, tx, is_eth_transfer=False
                ):
                    token = tx.get("token", {})
                    symbol = token.get("symbol", "Unknown")
                    total = tx.get("total", {})
                    value_raw = int(total.get("value", "0"))
                    decimals = int(token.get("decimals", 18))
                    amount = value_raw / (10**decimals)

                    transfer_data = {
                        "from_address": from_address.get("hash", ""),
                        "amount": amount,
                        "token_address": token.get("address", ""),
                        "symbol": symbol,
                        "timestamp": tx.get("timestamp", ""),
                        "tx_hash": tx.get("transaction_hash", ""),
                        "type": "token",
                    }

                    all_transfers_by_date[tx_date].append(transfer_data)
                    processed_count += 1

        self.context.logger.info(f"Completed token transfers: {processed_count} found")

    def _is_gnosis_safe(self, address_info: dict) -> bool:
        """Check if an address is a Gnosis Safe."""
        if not address_info or not address_info.get("is_contract"):
            return False
        name = address_info.get("name", "")
        return name == "GnosisSafeProxy"

    def _should_include_transfer(
        self, from_address: dict, tx_data: dict = None, is_eth_transfer: bool = False
    ) -> bool:
        """Determine if a transfer should be included based on filtering criteria."""
        if not from_address:
            return False

        from_hash = from_address.get("hash", "")
        if from_hash.lower() in [
            "0x0000000000000000000000000000000000000000",
            "0x0",
            "",
        ]:
            return False

        if tx_data and is_eth_transfer:
            if tx_data.get("status") != "ok" or int(tx_data.get("value", "0")) <= 0:
                return False

        return not from_address.get("is_contract") or self._is_gnosis_safe(from_address)

    def _get_datetime_from_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Convert timestamp string to datetime object."""
        try:
            # Handle different timestamp formats and ensure timezone awareness
            if timestamp_str.endswith("Z"):
                # ISO format with Z suffix
                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            elif "+" in timestamp_str or timestamp_str.endswith("UTC"):
                # Already has timezone info
                dt = datetime.fromisoformat(timestamp_str.replace("UTC", "+00:00"))
            else:
                # Assume UTC if no timezone info
                dt = datetime.fromisoformat(timestamp_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)

            return dt
        except (ValueError, TypeError) as e:
            self.context.logger.error(f"Error parsing timestamp {timestamp_str}: {e}")
            return None

    def _fetch_token_transfers_mode(
        self,
        address: str,
        target_date: str,
        all_transfers_by_date: dict,
        fetch_all_till_date: bool = False,
    ) -> Generator[None, None, bool]:
        """Fetch token transfers from Mode blockchain explorer for a specific date or all transfers till that date."""
        base_url = "https://explorer-mode-mainnet-0.t.conduit.xyz/api/v2"
        processed_count = 0
        endpoint = f"{base_url}/addresses/{address}/token-transfers"

        has_more_pages = True
        params = {"filter": "to"}  # Only fetch incoming transfers

        # Check if we have existing mode events and get latest date
        mode_events = self.funding_events.get("mode", {})
        if mode_events:
            try:
                latest_date = list(mode_events.keys())[0]
                latest_datetime = datetime.strptime(latest_date, "%Y-%m-%d")
                latest_datetime = latest_datetime.replace(tzinfo=timezone.utc)
            except (IndexError, ValueError):
                latest_datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
        else:
            latest_datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)

        while has_more_pages:
            response = requests.get(
                endpoint,
                params=params,
                headers={"Accept": "application/json"},
                timeout=30,
                verify=False,  # nosec B501
            )

            if not response.status_code == 200:
                self.context.logger.error("Failed to fetch Mode token transfers")
                return False

            response_data = response.json()
            transfers = response_data.get("items", [])
            if not transfers:
                break

            passed_target_date = False

            for tx in transfers:
                tx_datetime = self._get_datetime_from_timestamp(tx.get("timestamp"))
                if not tx_datetime:
                    continue

                tx_date = tx_datetime.strftime("%Y-%m-%d")

                if not fetch_all_till_date:
                    if tx_datetime < latest_datetime:
                        # We've gone past our latest stored date, stop processing
                        has_more_pages = False
                        passed_target_date = True
                        break

                from_address = tx.get("from", {})
                if from_address.get("hash", address).lower() == address.lower():
                    continue

                # Check for airdrop transfers first
                if self._is_airdrop_transfer(tx):
                    total = tx.get("total", {})
                    value_raw = int(total.get("value", "0"))
                    tx_hash = tx.get("transaction_hash", "")
                    yield from self._update_airdrop_rewards(value_raw, "mode", tx_hash)

                    token = tx.get("token", {})
                    decimals = int(token.get("decimals", 18))
                    amount = value_raw / (10**decimals)
                    self.context.logger.info(
                        f"Detected USDC airdrop transfer: {amount} USDC from {from_address.get('hash', '')} "
                        f"tx_hash: {tx_hash}"
                    )

                if self._should_include_transfer_mode(
                    from_address, tx, is_eth_transfer=False
                ):
                    token = tx.get("token", {})
                    symbol = token.get("symbol", "Unknown")
                    total = tx.get("total", {})
                    value_raw = int(total.get("value", "0"))
                    decimals = int(token.get("decimals", 18))
                    amount = value_raw / (10**decimals)

                    if amount == 0:
                        continue

                    if symbol.lower() != "usdc":
                        continue
                    transfer_data = {
                        "from_address": from_address.get("hash", ""),
                        "amount": amount,
                        "token_address": token.get("address", ""),
                        "symbol": symbol,
                        "timestamp": tx.get("timestamp", ""),
                        "tx_hash": tx.get("transaction_hash", ""),
                        "type": "token",
                    }

                    if tx_date not in all_transfers_by_date:
                        all_transfers_by_date[tx_date] = []
                    all_transfers_by_date[tx_date].append(transfer_data)
                    processed_count += 1

            # Stop pagination based on fetch mode
            if passed_target_date:
                break

            # Handle pagination
            next_page_params = response_data.get("next_page_params")
            if next_page_params:
                # Update params for next page
                params.update(
                    {
                        "block_number": next_page_params.get("block_number"),
                        "index": next_page_params.get("index"),
                    }
                )
                has_more_pages = True
            else:
                has_more_pages = False

        date_range = (
            f"for {target_date}" if not fetch_all_till_date else f"till {target_date}"
        )
        self.context.logger.info(
            f"Completed Mode token transfers {date_range}: {processed_count} found"
        )
        return True

    def _fetch_eth_transfers_mode(
        self,
        address: str,
        target_date: str,
        all_transfers_by_date: dict,
        fetch_till_date: bool,
    ) -> bool:
        """Fetch ETH balance history from Mode blockchain explorer."""
        base_url = "https://explorer-mode-mainnet-0.t.conduit.xyz/api/v2"
        endpoint = f"{base_url}/addresses/{address}/coin-balance-history"

        has_more_pages = True
        params = {}
        processed_count = 0

        # Check if we have existing mode events and get latest date
        mode_events = self.funding_events.get("mode", {})
        if mode_events:
            try:
                latest_date = list(mode_events.keys())[0]
                latest_datetime = datetime.strptime(latest_date, "%Y-%m-%d")
                latest_datetime = latest_datetime.replace(tzinfo=timezone.utc)
            except (IndexError, ValueError):
                latest_datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
        else:
            latest_datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)

        while has_more_pages:
            response = requests.get(
                endpoint,
                params=params,
                headers={"Accept": "application/json"},
                timeout=30,
                verify=False,  # nosec B501
            )

            if not response.status_code == 200:
                self.context.logger.error("Failed to fetch Mode coin balance history")
                return False

            response_data = response.json()
            balance_history = response_data.get("items", [])

            if not balance_history:
                break

            passed_target_date = False
            for entry in balance_history:
                current_value = int(entry.get("value", "0")) / 10**18
                if current_value <= 0:
                    continue

                # Calculate delta value
                delta_value = int(entry.get("delta", "0")) / 10**18

                # Get tx_hash
                tx_hash = entry.get("transaction_hash")

                # Filter: only include if delta > 0 and tx_hash is null/empty
                if delta_value <= 0:
                    continue

                if tx_hash is not None and tx_hash != "":
                    continue

                tx_datetime = datetime.strptime(
                    entry.get("block_timestamp", ""), "%Y-%m-%dT%H:%M:%SZ"
                )
                # Ensure timezone awareness for comparison
                tx_datetime = tx_datetime.replace(tzinfo=timezone.utc)
                # Convert from wei to ETH
                tx_date = tx_datetime.strftime("%Y-%m-%d")

                # Store balance data
                transfer_data = {
                    "amount": current_value,
                    "delta": delta_value,
                    "timestamp": entry.get("block_timestamp"),
                    "tx_hash": tx_hash,
                    "type": "eth",
                    "block_number": entry.get("block_number"),
                    "symbol": "ETH",
                }

                if not fetch_till_date:
                    if tx_datetime < latest_datetime:
                        # We've gone past our latest stored date, stop processing
                        passed_target_date = True
                        has_more_pages = False
                        break

                if tx_date not in all_transfers_by_date:
                    all_transfers_by_date[tx_date] = []
                all_transfers_by_date[tx_date].append(transfer_data)
                processed_count += 1

            # Stop pagination based on fetch mode
            if passed_target_date:
                break

            # Handle pagination
            next_page_params = response_data.get("next_page_params")
            if next_page_params and not passed_target_date:
                # Update params for next page
                params.update(
                    {
                        "block_number": next_page_params.get("block_number"),
                        "index": next_page_params.get("index"),
                    }
                )
                has_more_pages = True
            else:
                has_more_pages = False

        date_range = (
            f"for {target_date}" if not fetch_till_date else f"till {target_date}"
        )
        self.context.logger.info(
            f"Completed Mode coin balance history {date_range}: {processed_count} filtered transfers found"
        )
        return True

    def _should_include_transfer_mode(
        self, from_address: dict, tx_data: dict = None, is_eth_transfer: bool = False
    ) -> bool:
        """Determine if a Mode transfer should be included based on filtering criteria."""
        if not from_address:
            return False

        from_hash = from_address.get("hash", "")
        if from_hash.lower() in [
            "0x0000000000000000000000000000000000000000",
            "0x0",
            "",
        ]:
            return False

        if tx_data and is_eth_transfer:
            if tx_data.get("status") != "ok" or int(tx_data.get("value", "0")) <= 0:
                return False

        return not from_address.get("is_contract") or self._is_gnosis_safe(from_address)

    def _save_transfer_data_mode(self, data: Dict) -> Generator[None, None, None]:
        """Save Mode transfer data to kv_store."""
        yield from self._write_kv({"mode_transfer_data": json.dumps(data)})

    # Optimism-specific transfer methods
    def _fetch_optimism_transfers_safeglobal(
        self,
        address: str,
        end_date: str,
        all_transfers_by_date: dict,
        existing_data: dict,
    ) -> Generator[None, None, None]:
        """Fetch Optimism transfers using SafeGlobal API."""
        base_url = "https://safe-transaction-optimism.safe.global/api/v1"

        try:
            self.context.logger.info(
                "Fetching Optimism transfers using SafeGlobal API..."
            )

            # Fetch incoming transfers
            transfers_url = f"{base_url}/safes/{address}/incoming-transfers/"

            processed_count = 0
            seen_transfer_ids = set()
            while True:
                success, response_json = yield from self._request_with_retries(
                    endpoint=transfers_url,
                    headers={"Accept": "application/json"},
                    rate_limited_code=429,
                    rate_limited_callback=self.coingecko.rate_limited_status_callback,
                    retry_wait=self.params.sleep_time,
                )

                if not success:
                    self.context.logger.error("Failed to fetch Optimism transfers")
                    break

                transfers = response_json.get("results", [])
                if not transfers:
                    break

                for transfer in transfers:
                    # Parse timestamp
                    timestamp = transfer.get("executionDate")
                    if not timestamp:
                        continue

                    tx_datetime = self._get_datetime_from_timestamp(timestamp)
                    tx_date = tx_datetime.strftime("%Y-%m-%d") if tx_datetime else None

                    if not tx_date:
                        continue

                    # Skip if date already exists in stored data
                    if tx_date in existing_data:
                        continue

                    # Only process transfers until end_date
                    if tx_date > end_date:
                        continue

                    # Deduplicate by transferId if present, otherwise transactionHash
                    unique_id = transfer.get("transferId") or transfer.get(
                        "transactionHash", ""
                    )
                    if unique_id in seen_transfer_ids:
                        continue
                    seen_transfer_ids.add(unique_id)

                    # Process the transfer
                    from_address = transfer.get("from", address)
                    transfer_type = transfer.get("type", "")

                    if from_address.lower() == address.lower():
                        continue

                    # Filter from address - only include EOAs and GnosisSafe contracts
                    should_include = yield from self._should_include_transfer_optimism(
                        from_address
                    )
                    if not should_include:
                        continue

                    # Check transfer type
                    if transfer_type == "ERC20_TRANSFER":
                        # Token transfer
                        token_info = transfer.get("tokenInfo", {})
                        token_address = transfer.get("tokenAddress", "")
                        if not token_info:
                            if token_address:
                                decimals = yield from self._get_token_decimals(
                                    "optimism", token_address
                                )
                                symbol = yield from self._get_token_symbol(
                                    "optimism", token_address
                                )
                            else:
                                continue
                        else:
                            symbol = token_info.get("symbol", "Unknown")
                            decimals = int(token_info.get("decimals", 18) or 18)

                        if symbol.lower() != "usdc":
                            continue

                        value_raw = int(transfer.get("value", "0") or "0")
                        amount = value_raw / (10**decimals)

                        transfer_data = {
                            "from_address": from_address,
                            "amount": amount,
                            "token_address": token_address,
                            "symbol": symbol,
                            "timestamp": timestamp,
                            "tx_hash": transfer.get("transactionHash", ""),
                            "type": "token",
                        }

                    elif transfer_type == "ETHER_TRANSFER":
                        # ETH transfer
                        try:
                            value_wei = int(transfer.get("value", "0") or "0")
                            amount_eth = value_wei / 10**18

                            # Skip zero-value ETH transfers
                            if amount_eth <= 0:
                                continue
                        except (ValueError, TypeError):
                            self.context.logger.warning(
                                f"Skipping transfer with invalid value: {transfer.get('value')}"
                            )
                            continue

                        transfer_data = {
                            "from_address": from_address,
                            "amount": amount_eth,
                            "token_address": "",
                            "symbol": "ETH",
                            "timestamp": timestamp,
                            "tx_hash": transfer.get("transactionHash", ""),
                            "type": "eth",
                        }

                    elif transfer_type == "ERC721_TRANSFER":
                        # NFT transfer - skip for now
                        continue

                    else:
                        # Unknown transfer type
                        continue

                    all_transfers_by_date[tx_date].append(transfer_data)
                    processed_count += 1

                # Show progress
                if processed_count % 100 == 0:
                    self.context.logger.info(
                        f"Processed {processed_count} Optimism transfers..."
                    )

                # Check for next page and advance the URL
                cursor = response_json.get("next")
                if not cursor:
                    break
                transfers_url = cursor

            self.context.logger.info(
                f"Completed Optimism transfers: {processed_count} found"
            )

        except Exception as e:
            self.context.logger.error(f"Error fetching Optimism transfers: {e}")

    def _should_include_transfer_optimism(
        self, from_address: str
    ) -> Generator[None, None, bool]:
        """Determine if an Optimism transfer should be included based on from address type."""
        if not from_address:
            return False

        # Exclude zero address
        if from_address.lower() in [
            "0x0000000000000000000000000000000000000000",
            "0x0",
            "",
        ]:
            return False

        # Check cache first
        cache_key = f"{CONTRACT_CHECK_CACHE_PREFIX}optimism_{from_address.lower()}"
        cached_result = yield from self._read_kv((cache_key,))

        if cached_result and cached_result.get(cache_key):
            try:
                cached_data = json.loads(cached_result[cache_key])
                return cached_data.get("is_eoa", False)
            except (json.JSONDecodeError, KeyError):
                pass

        try:
            # Use Optimism RPC to check if address is a contract
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_getCode",
                "params": [from_address, "latest"],
                "id": 1,
            }

            success, result = yield from self._request_with_retries(
                endpoint="https://mainnet.optimism.io",
                method="POST",
                body=payload,
                rate_limited_code=429,
                rate_limited_callback=self.coingecko.rate_limited_status_callback,
                retry_wait=self.params.sleep_time,
            )

            if not success:
                self.context.logger.error("Failed to check contract code")
                return False

            code = result.get("result", "0x")

            # If code is '0x', it's an EOA
            if code == "0x":
                is_eoa = True
            else:
                # If it has code, check if it's a GnosisSafe
                safe_check_url = f"https://safe-transaction-optimism.safe.global/api/v1/safes/{from_address}/"
                success, _ = yield from self._request_with_retries(
                    endpoint=safe_check_url,
                    headers={"Accept": "application/json"},
                    rate_limited_code=429,
                    rate_limited_callback=self.coingecko.rate_limited_status_callback,
                    retry_wait=self.params.sleep_time,
                )
                is_eoa = success

            # Cache the result permanently (no TTL)
            cache_data = {"is_eoa": is_eoa}
            yield from self._write_kv({cache_key: json.dumps(cache_data)})

            if not is_eoa:
                self.context.logger.info(
                    f"Excluding transfer from contract: {from_address}"
                )
            return is_eoa

        except Exception as e:
            self.context.logger.error(f"Error checking address {from_address}: {e}")
            return False

    def _save_transfer_data_optimism(self, data: Dict) -> Generator[None, None, None]:
        """Save Optimism transfer data to kv_store."""
        yield from self._write_kv({"optimism_transfer_data": json.dumps(data)})

    def _load_chain_total_investment(self, chain: str) -> Generator[None, None, float]:
        """Load total investment value for a specific chain from kv_store."""
        key = f"{chain}_total_investment"
        cached_data = yield from self._read_kv(keys=(key,))
        if cached_data and cached_data.get(key):
            try:
                return float(cached_data.get(key))
            except (ValueError, TypeError):
                self.context.logger.warning(
                    f"Failed to parse cached {chain} total investment"
                )
                return 0.0
        return 0.0

    def _save_chain_total_investment(
        self, chain: str, total: float
    ) -> Generator[None, None, None]:
        """Save total investment value for a specific chain to kv_store."""
        key = f"{chain}_total_investment"
        yield from self._write_kv({key: str(total)})

    def _load_funding_events_data(self) -> Generator[None, None, Dict]:
        """Load unified transfer data from kv_store."""
        cached_data = yield from self._read_kv(keys=("funding_events",))
        if cached_data and cached_data.get("funding_events"):
            try:
                return json.loads(cached_data.get("funding_events"))
            except json.JSONDecodeError:
                self.context.logger.warning(
                    "Failed to parse cached unified transfer data"
                )
                return {}
        return {}

    def _track_eth_transfers_and_reversions(
        self,
        safe_address: str,
        chain: str,
    ) -> Generator[None, None, Dict[str, Any]]:
        """Track ETH transfers to safe address and handle reversion logic."""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            # Get all transfers until date
            all_incoming_transfers = {}
            all_outgoing_transfers = {}
            if chain == Chain.OPTIMISM.value:
                all_incoming_transfers = (
                    yield from self._fetch_all_transfers_until_date_optimism(
                        safe_address, current_date
                    )
                ) or {}
                all_outgoing_transfers = (
                    yield from self._fetch_outgoing_transfers_until_date_optimism(
                        safe_address, current_date
                    )
                ) or {}
            elif chain == Chain.MODE.value:
                # Use new Mode-specific tracking function
                transfers = self._track_eth_transfers_mode(safe_address, current_date)
                all_incoming_transfers = transfers.get("incoming", {})
                all_outgoing_transfers = transfers.get("outgoing", {})
            else:
                self.context.logger.warning(f"Unsupported chain: {chain}")
                return {}

            if not all_incoming_transfers:
                self.context.logger.warning(f"No transfers found for {chain} chain")
                return {}

            master_safe_address = yield from self.get_master_safe_address()
            if not master_safe_address:
                self.context.logger.error("No master safe address found")
                return {}

            self.context.logger.info(f"Master safe address: {master_safe_address}")

            # Track ETH transfers
            eth_transfers = []
            initial_funding = None
            reversion_transfers = []
            historical_reversion_value = 0.0
            reversion_date = None

            # Sort transfers by timestamp
            sorted_incoming_transfers = []
            for _, transfers in all_incoming_transfers.items():
                for transfer in transfers:
                    if isinstance(transfer, dict) and "timestamp" in transfer:
                        sorted_incoming_transfers.append(transfer)

            sorted_incoming_transfers.sort(key=lambda x: x["timestamp"])

            sorted_outgoing_transfers = []
            for _, transfers in all_outgoing_transfers.items():
                for transfer in transfers:
                    if isinstance(transfer, dict) and "timestamp" in transfer:
                        sorted_outgoing_transfers.append(transfer)

            sorted_outgoing_transfers.sort(key=lambda x: x["timestamp"])

            # Process transfers
            for transfer in sorted_incoming_transfers:
                # Check if it's an ETH transfer
                if transfer.get("symbol") == "ETH":
                    # If this is the first transfer, store it as initial funding
                    if not initial_funding:
                        initial_funding = {
                            "amount": transfer.get("amount", 0),
                            "from_address": transfer.get("from_address"),
                            "timestamp": transfer.get("timestamp"),
                        }
                        eth_transfers.append(transfer)
                    # If it's from the same address as initial funding
                    elif (
                        transfer.get("from_address", "").lower()
                        == master_safe_address.lower()
                    ):
                        eth_transfers.append(transfer)

            for transfer in sorted_outgoing_transfers:
                if transfer.get("symbol") == "ETH":
                    if (
                        transfer.get("to_address", "").lower()
                        == master_safe_address.lower()
                        and transfer.get("from_address", "").lower()
                        == safe_address.lower()
                    ):
                        reversion_transfers.append(transfer)

            # Get current ETH balance
            chain = self.params.target_investment_chains[0]
            account = self.params.safe_contract_addresses.get(chain)
            current_eth_balance = yield from self._get_native_balance(chain, account)

            # Determine if reversion is needed
            reversion_amount = 0

            if initial_funding and len(eth_transfers) > 1:
                # If there are additional transfers after initial funding
                if len(reversion_transfers) == 0:
                    # No reversion has happened yet, revert the last transfer amount
                    last_transfer = eth_transfers[-1]
                    reversion_amount = float(last_transfer.get("amount", 0))

                    # Get the date of the last transfer that needs reversion
                    try:
                        # Handle ISO format timestamp
                        timestamp = last_transfer.get("timestamp", "")
                        if timestamp.endswith("Z"):
                            # Convert ISO format to datetime
                            tx_datetime = datetime.fromisoformat(
                                timestamp.replace("Z", "+00:00")
                            )
                            reversion_date = tx_datetime.strftime("%d-%m-%Y")
                        else:
                            # Try parsing as Unix timestamp
                            reversion_date = datetime.fromtimestamp(
                                int(timestamp)
                            ).strftime("%d-%m-%Y")
                    except (ValueError, TypeError) as e:
                        self.context.logger.warning(f"Error parsing timestamp: {e}")
                        # Use current date as fallback
                        reversion_date = current_date

                    if current_eth_balance < reversion_amount:
                        reversion_amount = current_eth_balance
                        self.context.logger.info(
                            f"Current ETH balance is {current_eth_balance} which is less than the reversion amount {reversion_amount} indicating that some ETH has already been used"
                        )

                    self.context.logger.info(
                        f"Found additional ETH transfer of {reversion_amount} that needs reversion from date: {reversion_date}"
                    )
                else:
                    # Reversion has already happened, set amount to 0
                    reversion_amount = 0
                    self.context.logger.info(
                        "Additional ETH transfer has already been reverted"
                    )

            if len(reversion_transfers) > 0:
                historical_reversion_value = (
                    yield from self._calculate_total_reversion_value(
                        eth_transfers, reversion_transfers
                    )
                )

            return {
                "reversion_amount": reversion_amount,
                "master_safe_address": master_safe_address,
                "historical_reversion_value": historical_reversion_value,
                "reversion_date": reversion_date,
            }

        except Exception as e:
            self.context.logger.error(f"Error tracking ETH transfers: {str(e)}")
            return {
                "reversion_amount": 0,
                "master_safe_address": None,
                "historical_reversion_value": 0.0,
                "reversion_date": None,
            }

    def _calculate_total_reversion_value(
        self, eth_transfers: List[Dict], reversion_transfers: List[Dict]
    ) -> Generator[None, None, float]:
        """Calculate the total reversion value from the reversion transfers."""
        reversion_amount = 0.0
        reversion_date = None
        reversion_value = 0.0
        last_transfer = eth_transfers[-1]
        current_date = datetime.now().strftime("%d-%m-%Y")

        try:
            # Handle ISO format timestamp
            timestamp = last_transfer.get("timestamp", "")
            if timestamp.endswith("Z"):
                # Convert ISO format to datetime
                tx_datetime = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                reversion_date = tx_datetime.strftime("%d-%m-%Y")
            else:
                # Try parsing as Unix timestamp
                reversion_date = datetime.fromtimestamp(int(timestamp)).strftime(
                    "%d-%m-%Y"
                )
        except (ValueError, TypeError) as e:
            self.context.logger.warning(f"Error parsing timestamp: {e}")
            # Use current date as fallback
            reversion_date = current_date

        for index, transfer in enumerate(reversion_transfers):
            if index == 0:
                eth_price = yield from self._fetch_historical_eth_price(reversion_date)
            else:
                eth_price = yield from self._fetch_historical_eth_price(current_date)
            if eth_price:
                reversion_amount = transfer.get("amount", 0)
                reversion_value += reversion_amount * eth_price

        return reversion_value

    def _fetch_outgoing_transfers_until_date_optimism(
        self,
        address: str,
        current_date: str,
    ) -> Generator[None, None, Dict]:
        """Fetch all outgoing transfers from the safe address on Mode until a specific date."""
        all_transfers = {}

        if not address:
            self.context.logger.warning(
                "No address provided for fetching Optimism outgoing transfers"
            )
            return all_transfers

        try:
            # Use SafeGlobal API for Optimism transfers
            base_url = "https://safe-transaction-optimism.safe.global/api/v1"
            transfers_url = f"{base_url}/safes/{address}/transfers/"

            processed_count = 0
            seen_tx_ids = set()
            while True:
                success, response_json = yield from self._request_with_retries(
                    endpoint=transfers_url,
                    headers={"Accept": "application/json"},
                    rate_limited_code=429,
                    rate_limited_callback=self.coingecko.rate_limited_status_callback,
                    retry_wait=self.params.sleep_time,
                )

                if not success:
                    self.context.logger.error("Failed to fetch Optimism transfers")
                    break

                transfers = response_json.get("results", [])

                if not transfers:
                    break

                for transfer in transfers:
                    # Parse timestamp
                    timestamp = transfer.get("executionDate")
                    if not timestamp:
                        continue

                    # Handle ISO format timestamp
                    try:
                        tx_datetime = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )
                        tx_date = tx_datetime.strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        self.context.logger.warning(
                            f"Invalid timestamp format: {timestamp}"
                        )
                        continue

                    if tx_date > current_date:
                        continue

                    # Only process outgoing transfers (where from address is equal to our safe address)
                    if transfer.get("from").lower() == address.lower():
                        transfer_type = transfer.get("type", "")

                        # Deduplicate per tx hash + type
                        unique_id = f"{transfer.get('transactionHash', '')}:{transfer_type}:{transfer.get('value', '')}"
                        if unique_id in seen_tx_ids:
                            continue
                        seen_tx_ids.add(unique_id)

                        if transfer_type == "ETHER_TRANSFER":
                            try:
                                value_wei = int(transfer.get("value", "0") or "0")
                                amount_eth = value_wei / 10**18

                                if amount_eth <= 0:
                                    continue
                            except (ValueError, TypeError):
                                continue

                            transfer_data = {
                                "from_address": address,
                                "to_address": transfer.get("to"),
                                "amount": amount_eth,
                                "token_address": ZERO_ADDRESS,
                                "symbol": "ETH",
                                "timestamp": timestamp,
                                "tx_hash": transfer.get("transactionHash", ""),
                                "type": "eth",
                            }

                            if tx_date not in all_transfers:
                                all_transfers[tx_date] = []
                            all_transfers[tx_date].append(transfer_data)
                            processed_count += 1
                            continue
                    else:
                        continue

                self.context.logger.info(
                    f"Completed Optimism outgoing transfers: {processed_count} found"
                )

                # Advance pagination if available
                cursor = response_json.get("next")
                if not cursor:
                    return all_transfers
                transfers_url = cursor

            return all_transfers

        except Exception as e:
            self.context.logger.error(
                f"Error fetching Optimism outgoing transfers: {e}"
            )
            return {}

    def _track_eth_transfers_mode(
        self,
        safe_address: str,
        current_date: str,
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """Fetch and organize ETH transfers for Mode chain using the Mode explorer API."""
        try:
            all_transfers = {"incoming": {}, "outgoing": {}}

            # Use Mode internal transactions API
            base_url = "https://explorer.mode.network/api"
            params = {
                "module": "account",
                "action": "txlistinternal",
                "address": safe_address,
                "startblock": 0,
                "endblock": 99999999,
                "sort": "asc",
            }

            response = requests.get(
                base_url,
                params=params,
                headers={"Accept": "application/json"},
                timeout=30,
                verify=False,  # nosec B501
            )

            if response.status_code != 200:
                self.context.logger.error(
                    f"Failed to fetch Mode ETH transfers: {response.status_code}"
                )
                return all_transfers

            response_data = response.json()
            if response_data.get("status") != "1":
                self.context.logger.error(
                    f"Error in Mode API response: {response_data.get('message', 'Unknown error')}"
                )
                return all_transfers

            transactions = response_data.get("result", [])
            processed_count = 0

            for tx in transactions:
                # Skip if no timestamp or value is 0
                if not tx.get("timeStamp") or int(tx.get("value", "0")) <= 0:
                    continue

                try:
                    timestamp = tx.get("timeStamp")
                    # Convert timestamp to date for comparison with current_date
                    tx_datetime = datetime.fromtimestamp(int(timestamp))
                    tx_date = tx_datetime.strftime("%Y-%m-%d")

                    if tx_date > current_date:
                        continue

                    # Convert value from wei to ETH
                    value_wei = int(tx.get("value", "0"))
                    amount_eth = value_wei / 10**18

                    # Check if safe_address is in 'to' or 'from' field
                    to_address = tx.get("to", "").lower()
                    from_address = tx.get("from", "").lower()
                    safe_address_lower = safe_address.lower()

                    transfer_data = {
                        "from_address": from_address,
                        "to_address": to_address,
                        "amount": amount_eth,
                        "token_address": ZERO_ADDRESS,
                        "symbol": "ETH",
                        "timestamp": timestamp,
                        "tx_hash": tx.get("hash", ""),
                        "type": "eth",
                    }

                    # Categorize transfer based on safe_address position
                    if to_address == safe_address_lower:
                        # Incoming transfer - safe_address is recipient
                        if timestamp not in all_transfers["incoming"]:
                            all_transfers["incoming"][timestamp] = []
                        all_transfers["incoming"][timestamp].append(transfer_data)
                        processed_count += 1
                    elif from_address == safe_address_lower:
                        # Outgoing transfer - safe_address is sender
                        if timestamp not in all_transfers["outgoing"]:
                            all_transfers["outgoing"][timestamp] = []
                        all_transfers["outgoing"][timestamp].append(transfer_data)
                        processed_count += 1

                except (ValueError, TypeError) as e:
                    self.context.logger.warning(f"Error processing transaction: {e}")
                    continue

            self.context.logger.info(
                f"Completed Mode ETH transfers: {processed_count} transactions processed"
            )
            return all_transfers

        except Exception as e:
            self.context.logger.error(f"Error tracking Mode ETH transfers: {e}")
            return {"incoming": {}, "outgoing": {}}

    def _track_erc20_transfers_mode(
        self,
        safe_address: str,
        final_timestamp: int,
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """Fetch and organize ERC20 token transfers for Mode chain using the Mode explorer API with pagination."""
        try:
            all_transfers = {"outgoing": {}}

            # Use Mode Explorer API (same as _fetch_token_transfers_mode)
            base_url = "https://explorer-mode-mainnet-0.t.conduit.xyz/api/v2"
            endpoint = f"{base_url}/addresses/{safe_address}/token-transfers"
            params = {"filter": "from"}  # Only fetch outgoing transfers

            has_more_pages = True
            processed_count = 0

            while has_more_pages:
                response = requests.get(
                    endpoint,
                    params=params,
                    headers={"Accept": "application/json"},
                    timeout=30,
                    verify=False,  # nosec B501
                )

                if response.status_code != 200:
                    self.context.logger.error(
                        f"Failed to fetch Mode ERC20 transfers: {response.status_code}"
                    )
                    return all_transfers

                response_data = response.json()
                transactions = response_data.get("items", [])

                if not transactions:
                    break

                for tx in transactions:
                    # Skip if no timestamp or value is 0
                    if (
                        not tx.get("timestamp")
                        or int(tx.get("total", {}).get("value", "0")) <= 0
                    ):
                        continue

                    try:
                        # Use the same timestamp parsing as _fetch_token_transfers_mode
                        tx_datetime = self._get_datetime_from_timestamp(
                            tx.get("timestamp")
                        )
                        if not tx_datetime:
                            continue

                        tx_date = tx_datetime.strftime("%Y-%m-%d")
                        current_date = datetime.fromtimestamp(final_timestamp).strftime(
                            "%Y-%m-%d"
                        )
                        if tx_date > current_date:
                            continue

                        # Get token information (Mode Explorer API format)
                        token = tx.get("token", {})
                        token_address = token.get("address", "")
                        token_symbol = token.get("symbol", "Unknown")
                        token_decimals = int(token.get("decimals", 18))

                        # Convert value from raw units to token units
                        total = tx.get("total", {})
                        value_raw = int(total.get("value", "0"))
                        amount = value_raw / (10**token_decimals)

                        # Get addresses (Mode Explorer API format)
                        from_address = tx.get("from", {})
                        to_address = tx.get("to", {})
                        safe_address_lower = safe_address.lower()

                        should_include = self._should_include_transfer_mode(
                            to_address, tx, is_eth_transfer=False
                        )
                        if not should_include:
                            continue

                        if token_symbol == "USDC":  # nosec B105
                            transfer_data = {
                                "from_address": from_address.get("hash", ""),
                                "to_address": to_address.get("hash", ""),
                                "amount": amount,
                                "token_address": token_address,
                                "symbol": token_symbol,
                                "timestamp": tx.get("timestamp", ""),
                                "tx_hash": tx.get("transaction_hash", ""),
                                "type": "token",
                            }

                            # Since we're filtering for outgoing transfers, safe_address is always the sender
                            if (
                                from_address.get("hash", "").lower()
                                == safe_address_lower
                            ):
                                # Outgoing transfer - safe_address is sender
                                if tx_date not in all_transfers["outgoing"]:
                                    all_transfers["outgoing"][tx_date] = []
                                all_transfers["outgoing"][tx_date].append(transfer_data)
                                processed_count += 1

                    except (ValueError, TypeError) as e:
                        self.context.logger.error(f"Error processing transaction: {e}")
                        continue

                next_page_params = response_data.get("next_page_params")
                if next_page_params:
                    # Update params for next page
                    params.update(
                        {
                            "block_number": next_page_params.get("block_number"),
                            "index": next_page_params.get("index"),
                        }
                    )
                    has_more_pages = True
                else:
                    has_more_pages = False

            self.context.logger.info(
                f"Completed Mode ERC20 transfers tracking: {processed_count} outgoing transfers found"
            )
            return all_transfers

        except Exception as e:
            self.context.logger.error(f"Error tracking Mode ERC20 transfers: {e}")
            return {"outgoing": {}}

    def get_master_safe_address(self) -> Generator[None, None, Optional[str]]:
        """Get the master safe address by checking service staking state."""
        # Get service_id from params
        service_id = self.params.on_chain_service_id
        if service_id is None:
            self.context.logger.error("No service ID configured")
            return None

        if not self.params.target_investment_chains:
            self.context.logger.error("No target investment chains configured")
            return None

        operating_chain = self.params.target_investment_chains[0]

        staking_token_address = self.params.staking_token_contract_address
        staking_chain = self.params.staking_chain

        if staking_token_address and staking_chain:
            if self.service_staking_state == StakingState.UNSTAKED:
                yield from self._get_service_staking_state(chain=staking_chain)

            if self.service_staking_state != StakingState.UNSTAKED:
                service_info = yield from self._get_service_info(staking_chain)
                if service_info and len(service_info) >= 2:
                    master_safe_address = service_info[
                        1
                    ]  # owner field from service info struct
                    self.context.logger.info(
                        f"Master safe address: {master_safe_address}"
                    )
                    is_valid_address = yield from self.check_is_valid_safe_address(
                        master_safe_address, staking_chain
                    )
                    if is_valid_address:
                        return master_safe_address
                    else:
                        return None
                else:
                    self.context.logger.error(
                        "Failed to get service info from staking contract"
                    )
                    return None

        service_registry_address = self.params.service_registry_contract_addresses.get(
            operating_chain
        )
        if not service_registry_address:
            self.context.logger.error(
                f"No service registry address configured for operating chain {operating_chain}"
            )
            return None

        # Get service owner from service registry
        service_owner_result = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=service_registry_address,
            contract_public_id=ServiceRegistryContract.contract_id,
            contract_callable="get_service_owner",
            data_key="service_owner",
            service_id=service_id,
            chain_id=operating_chain,
        )

        if service_owner_result:
            master_safe_address = service_owner_result
            self.context.logger.info(f"Master safe address: {master_safe_address}")
            is_valid_address = yield from self.check_is_valid_safe_address(
                master_safe_address, operating_chain
            )
            if is_valid_address:
                return master_safe_address
            else:
                return None
        else:
            self.context.logger.error(
                "Failed to get service owner from service registry"
            )
            return None

    def _read_investing_paused(self) -> Generator[None, None, bool]:
        """Read investing_paused flag from KV store."""
        try:
            result = yield from self._read_kv(("investing_paused",))
            if result is None:
                self.context.logger.warning(
                    "No response from KV store for investing_paused flag"
                )
                return False

            investing_paused_value = result.get("investing_paused")
            if investing_paused_value is None:
                self.context.logger.warning(
                    "investing_paused value is None in KV store"
                )
                return False

            return investing_paused_value.lower() == "true"
        except Exception as e:
            self.context.logger.error(f"Error reading investing_paused flag: {str(e)}")
            return False

    def check_is_valid_safe_address(
        self, safe_address: str, operating_chain: str
    ) -> Generator[None, None, bool]:
        """Checks if an address is a GnosisSafe Contract"""
        try:
            res = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=safe_address,
                contract_public_id=GnosisSafeContract.contract_id,
                contract_callable="get_owners",
                data_key="owners",
                chain_id=operating_chain,
            )

            if res:
                return True

            return False

        except Exception:
            self.context.logger.info("Not a GnosisSafe")
            return False

    def _get_velodrome_pending_rewards(
        self, position: Dict, chain: str, user_address: str
    ) -> Generator[None, None, Decimal]:
        """Get pending VELO rewards for a staked Velodrome position."""
        try:
            pool_address = position.get("pool_address")
            is_cl_pool = position.get("is_cl_pool", False)

            if not pool_address:
                self.context.logger.warning(
                    "No pool address found for Velodrome position"
                )
                return Decimal(0)

            # Get the Velodrome pool behaviour to access reward methods
            pool = self.pools.get("velodrome")
            if not pool:
                self.context.logger.warning("Velodrome pool behaviour not found")
                return Decimal(0)

            # Get pending rewards based on pool type
            if is_cl_pool:
                # For CL pools, we need the gauge address
                gauge_address = yield from pool.get_gauge_address(
                    self, pool_address, chain=chain
                )
                if not gauge_address:
                    self.context.logger.warning(
                        f"No gauge found for CL pool {pool_address}"
                    )
                    return Decimal(0)
                positions_data = position.get("positions", [])
                token_ids = [pos["token_id"] for pos in positions_data]
                pending_rewards = 0
                for token_id in token_ids:
                    rewards = yield from pool.get_cl_pending_rewards(
                        self,
                        account=user_address,
                        gauge_address=gauge_address,
                        chain=chain,
                        token_id=token_id,
                    )
                    if rewards:
                        pending_rewards += rewards
            else:
                # For regular pools
                pending_rewards = yield from pool.get_pending_rewards(
                    self, lp_token=pool_address, user_address=user_address, chain=chain
                )

            if pending_rewards and pending_rewards > 0:
                # Convert from wei to VELO (18 decimals)
                velo_rewards = Decimal(pending_rewards) / Decimal(10**18)
                self.context.logger.info(
                    f"Found VELO rewards: {velo_rewards} for position {pool_address}"
                )
                return velo_rewards

            return Decimal(0)

        except Exception as e:
            self.context.logger.error(
                f"Error getting Velodrome pending rewards: {str(e)}"
            )
            return Decimal(0)

    def _get_velo_token_address(self, chain: str) -> Optional[str]:
        """Get the VELO token address for the specified chain from params."""
        velo_addresses = self.params.velo_token_contract_addresses
        return velo_addresses.get(chain)

    def _validate_velodrome_v2_pool_addresses(self) -> Generator[None, None, None]:
        """Validate Velodrome v2 pool addresses for all positions."""
        for position in self.current_positions:
            if (
                position.get("dex_type") == "velodrome"
                and not position.get("is_cl_pool", False)
                and position.get("is_stable", False)
            ):  # Only validate if isStable is true
                validation_success = (
                    yield from self._validate_velodrome_v2_pool_address(position)
                )
                if validation_success:
                    self.context.logger.info(
                        f"Pool address validation completed for position: {position.get('pool_address')}"
                    )

        # Store updated positions after validation
        self.store_current_positions()

    def _validate_velodrome_v2_pool_address(
        self, position: Dict[str, Any]
    ) -> Generator[None, None, bool]:
        """Validate and correct Velodrome v2 pool address from transaction logs."""
        try:
            # Skip if already updated or missing required data
            if position.get("isUpdated", False):
                return True

            tx_hash = position.get("enter_tx_hash")
            chain = position.get("chain")
            stored_pool_address = position.get("pool_address")

            if not all([tx_hash, chain, stored_pool_address]):
                self.context.logger.warning(
                    f"Missing required data for pool validation: tx_hash={tx_hash}, chain={chain}, pool_address={stored_pool_address}"
                )
                return False

            self.context.logger.info(
                f"Validating Velodrome v2 pool address for tx: {tx_hash}"
            )

            # Use existing get_transaction_receipt function
            response = yield from self.get_transaction_receipt(
                tx_digest=tx_hash,
                chain_id=chain,
            )

            if not response:
                self.context.logger.error(
                    f"Error fetching tx receipt! Response: {response}"
                )
                return False

            logs = response.get("logs", [])

            # Look for LP token mint events (Transfer from zero address)
            actual_pool_address = None

            for log in logs:
                topics = log.get("topics", [])
                if len(topics) >= 3 and topics[0] == TRANSFER_EVENT_SIGNATURE:
                    # Check if this is a mint (from zero address)
                    from_address = topics[1]
                    if from_address == ZERO_ADDRESS_PADDED:
                        # This is an LP token mint, the contract address is the pool
                        actual_pool_address = log.get("address")
                        break

            if not actual_pool_address:
                self.context.logger.warning(
                    f"Could not find LP token mint event in transaction {tx_hash}"
                )
                return False

            # Normalize addresses for comparison
            actual_pool_address = actual_pool_address.lower()
            stored_pool_address = stored_pool_address.lower()

            if actual_pool_address != stored_pool_address:
                self.context.logger.info(
                    f"Pool address mismatch detected! Stored: {stored_pool_address}, Actual: {actual_pool_address}"
                )
                # Update the position with the correct pool address
                position["pool_address"] = actual_pool_address
                position["is_updated"] = True
                self.context.logger.info(
                    f"Updated pool address to: {actual_pool_address}"
                )
                return True
            else:
                self.context.logger.info(
                    "Pool address validation passed - no update needed"
                )
                position["is_updated"] = True
                return True

        except Exception as e:
            self.context.logger.error(
                f"Error validating Velodrome v2 pool address: {str(e)}"
            )
            return False

    def _is_airdrop_transfer(self, tx: Dict) -> bool:
        """Check if a transfer is an airdrop transfer."""
        if not self.params.airdrop_started or not self.params.airdrop_contract_address:
            return False

        from_address = tx.get("from", {})
        token = tx.get("token", {})
        symbol = token.get("symbol", "Unknown")

        # Check for USDC airdrop transfers on MODE chain
        usdc_address = self._get_usdc_address("mode")
        return (
            symbol.upper() == "USDC"
            and token.get("address", "").lower() == usdc_address.lower()
            and from_address.get("hash", "").lower()
            == self.params.airdrop_contract_address.lower()
        )

    def _update_agent_performance_metrics(self):
        """Update agent performance metrics with portfolio balance and ROI."""
        try:
            # Read existing agent performance data or initialize
            self.read_agent_performance()

            # Get portfolio balance and ROI from current portfolio data
            portfolio_balance = 0.0
            total_roi_percentage = 0.0
            partial_roi_percentage = 0.0

            if self.portfolio_data:
                # Use portfolio_value instead of total_value_usd
                portfolio_balance = float(
                    self.portfolio_data.get("portfolio_value", 0.0)
                )

                # Use pre-calculated ROI percentages from portfolio data
                total_roi_percentage = float(self.portfolio_data.get("total_roi", 0.0))
                partial_roi_percentage = float(
                    self.portfolio_data.get("partial_roi", 0.0)
                )

            # Update metrics
            self.agent_performance["metrics"] = [
                {
                    "name": "Portfolio Balance",
                    "is_primary": True,
                    "description": "Total value of all assets including liquidity pools and safe balance",
                    "value": f"${portfolio_balance:.2f}",
                },
                {
                    "name": "Total ROI",
                    "is_primary": False,
                    "description": f"Total return on investment including staking rewards. Partial ROI (trading only): <b>{partial_roi_percentage:.1f}%</b>",
                    "value": f"{total_roi_percentage:.2f}%",
                },
            ]

            # Update timestamp and store
            self.update_agent_performance_timestamp()
            self.store_agent_performance()

            self.context.logger.info(
                f"Updated agent performance: Balance=${portfolio_balance:.2f}, Total ROI={total_roi_percentage:.1f}%, Partial ROI={partial_roi_percentage:.1f}%"
            )

        except Exception as e:
            self.context.logger.error(
                f"Error updating agent performance metrics: {str(e)}"
            )

    def _fetch_portfolio_from_subgraph(
        self, chain: str
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """
        Fetch complete portfolio data from the Optimus subgraph.

        Returns portfolio data structure matching the current portfolio_data format,
        or None if subgraph is unavailable or returns no data.
        """
        try:
            # Get subgraph endpoint for the chain
            subgraph_endpoint = self.params.optimus_subgraph_endpoints.get(chain)

            if not subgraph_endpoint:
                self.context.logger.warning(
                    f"No Optimus subgraph endpoint configured for chain {chain}"
                )
                return None

            # Get service safe address for this chain
            safe_address = self.params.safe_contract_addresses.get(chain)
            if not safe_address:
                self.context.logger.error(f"No safe address found for chain {chain}")
                return None

            # GraphQL query to fetch portfolio data
            query = """
            query GetPortfolio($serviceId: Bytes!) {
                service(id: $serviceId) {
                    id
                    balances {
                        token
                        symbol
                        decimals
                        balance
                        balanceUSD
                    }
                    positions {
                        id
                        protocol
                        pool
                        isActive
                        usdCurrent
                        token0
                        token0Symbol
                        token1
                        token1Symbol
                        amount0
                        amount1
                        liquidity
                        tokenId
                        tickLower
                        tickUpper
                        tickSpacing
                        fee
                    }
                }
                agentPortfolio(id: $serviceId) {
                    finalValue
                    initialValue
                    positionsValue
                    uninvestedValue
                    projectedRoi
                    roi
                    apr
                    projectedAPR
                    totalPositions
                    totalClosedPositions
                    lastUpdated
                }
                fundingBalance(id: $serviceId) {
                    totalOutUsd
                }
            }
            """

            variables = {"serviceId": safe_address.lower()}

            # Make GraphQL request
            success, response = yield from self._request_with_retries(
                endpoint=subgraph_endpoint,
                method="POST",
                body={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"},
                rate_limited_code=429,
                rate_limited_callback=lambda: self.context.logger.warning(
                    "Subgraph rate limited"
                ),
                retry_wait=self.params.sleep_time,
            )

            if not success or not response:
                self.context.logger.error(
                    f"Failed to fetch portfolio from subgraph for {chain}"
                )
                return None

            data = response.get("data", {})
            service = data.get("service")
            portfolio = data.get("agentPortfolio")
            funding_balance = data.get("fundingBalance")

            if not service or not portfolio:
                self.context.logger.warning(
                    f"No portfolio data found in subgraph for {safe_address}"
                )
                return None

            # Get withdrawal value from funding balance
            withdrawals_value = 0.0
            if funding_balance:
                withdrawals_value = float(funding_balance.get("totalOutUsd", 0))

            # Build allocations from positions
            allocations = []
            for position in service.get("positions", []):
                if position.get("isActive"):
                    allocation = yield from self._build_allocation_from_position(
                        position, chain, safe_address
                    )
                    if allocation:
                        allocations.append(allocation)

            # Build portfolio breakdown from token balances
            portfolio_breakdown = []
            for balance in service.get("balances", []):
                portfolio_breakdown.append(
                    {
                        "asset": balance.get("symbol"),
                        "address": balance.get("token"),
                        "balance": float(balance.get("balance", 0)),
                        "price": float(balance.get("balanceUSD", 0))
                        / float(balance.get("balance", 1))
                        if float(balance.get("balance", 0)) > 0
                        else 0.0,
                        "value_usd": float(balance.get("balanceUSD", 0)),
                        "ratio": 0.0,  # Will be calculated later
                    }
                )

            # Get agent hash
            agent_config = os.environ.get("AEA_AGENT", "")
            agent_hash = agent_config.split(":")[-1] if agent_config else "Not found"

            # Build final portfolio data structure
            portfolio_data = {
                "portfolio_value": float(portfolio.get("finalValue", 0)),
                "value_in_pools": float(portfolio.get("positionsValue", 0)),
                "value_in_safe": float(portfolio.get("uninvestedValue", 0)),
                "value_in_withdrawals": withdrawals_value,
                "initial_investment": float(portfolio.get("initialValue", 0)),
                "airdropped_rewards": 0.0,  # Will be added from local calculation
                "volume": None,  # Not tracked in subgraph yet
                "total_roi": float(portfolio.get("roi", 0)),
                "partial_roi": float(portfolio.get("projectedRoi", 0)),
                "agent_hash": agent_hash,
                "allocations": allocations,
                "portfolio_breakdown": portfolio_breakdown,
                "address": safe_address,
                "last_updated": int(portfolio.get("lastUpdated", 0)),
            }

            self.context.logger.info(
                f"Successfully fetched portfolio from subgraph for {chain}"
            )

            return portfolio_data

        except Exception as e:
            self.context.logger.error(
                f"Error fetching portfolio from subgraph: {str(e)}"
            )
            return None

    def _build_allocation_from_position(
        self, position: Dict[str, Any], chain: str, safe_address: str
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """
        Build an allocation object from a subgraph position.

        Args:
            position: Position data from subgraph
            chain: Chain name
            safe_address: Safe address for this service

        Returns:
            Allocation dict matching the current format, or None if invalid
        """
        try:
            protocol = position.get("protocol", "")
            pool_address = position.get("pool", "")

            if not protocol or not pool_address:
                return None

            # Map protocol names to dex types
            protocol_to_dex = {
                "velodrome-cl": "velodrome",
                "velodrome-v2": "velodrome",
                "uniswap-v3": "uniswapV3",
                "balancer": "balancerPool",
                "sturdy": "sturdy",
            }

            dex_type = protocol_to_dex.get(protocol)
            if not dex_type:
                self.context.logger.warning(f"Unknown protocol: {protocol}")
                return None

            # Get token symbols
            assets = []
            if position.get("token0Symbol"):
                assets.append(position.get("token0Symbol"))
            if position.get("token1Symbol"):
                assets.append(position.get("token1Symbol"))

            # Build allocation object
            allocation = {
                "chain": chain,
                "type": dex_type,
                "id": pool_address,
                "assets": assets,
                "apr": 0.0,  # APR not tracked in subgraph yet
                "details": f"{protocol} Pool",
                "ratio": 0.0,  # Will be calculated later
                "address": safe_address,
            }

            # Add tick ranges for concentrated liquidity positions
            if protocol in ["velodrome-cl", "uniswap-v3"]:
                if (
                    position.get("tokenId")
                    and position.get("tickLower") is not None
                    and position.get("tickUpper") is not None
                ):
                    # Note: We don't have current tick from subgraph, would need additional query
                    # For now, just add the static position info
                    allocation["tick_ranges"] = [
                        {
                            "token_id": int(position.get("tokenId", 0)),
                            "tick_lower": int(position.get("tickLower", 0)),
                            "tick_upper": int(position.get("tickUpper", 0)),
                            "current_tick": 0,  # Placeholder - needs pool query
                            "in_range": False,  # Placeholder - needs pool query
                        }
                    ]

            return allocation

        except Exception as e:
            self.context.logger.error(
                f"Error building allocation from position: {str(e)}"
            )
            return None

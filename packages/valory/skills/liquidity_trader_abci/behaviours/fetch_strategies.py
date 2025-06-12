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
    DexType,
    LiquidityTraderBaseBehaviour,
    PORTFOLIO_UPDATE_INTERVAL,
    PositionStatus,
    TradingType,
    WHITELISTED_ASSETS,
    ZERO_ADDRESS,
)
from packages.valory.skills.liquidity_trader_abci.states.fetch_strategies import (
    FetchStrategiesPayload,
    FetchStrategiesRound,
)
from packages.valory.skills.liquidity_trader_abci.utils.tick_math import (
    LiquidityAmounts,
    TickMath,
)


class FetchStrategiesBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that gets the balances of the assets of agent safes."""

    matching_round: Type[AbstractRound] = FetchStrategiesRound
    strategies = None

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            agent_config = os.environ.get("AEA_AGENT", "")
            agent_hash = agent_config.split(":")[-1] if agent_config else "Not found"
            self.context.logger.info(f"Agent hash: {agent_hash}")

            if self.current_positions:
                self.context.logger.info(
                    f"Current Positions - {self.current_positions}"
                )

            sender = self.context.agent_address

            if not self.assets:
                self.assets = self.params.initial_assets
                self.store_assets()

            self.read_assets()

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

            if not trading_type:
                trading_type = TradingType.BALANCED.value

            self.context.logger.info(
                f"Reading values from kv store... Selected protocols: {serialized_protocols}, Trading type: {trading_type}"
            )
            self.shared_state.trading_type = trading_type
            self.shared_state.selected_protocols = selected_protocols

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

            # Update the amounts of all open positions
            if self.synchronized_data.period_count == 0:
                self.context.logger.info("Updating position amounts for period 0")
                yield from self.update_position_amounts()
                self.context.logger.info(
                    "Checking and updating zero liquidity positions"
                )
                self.check_and_update_zero_liquidity_positions()

            self.context.logger.info(f"Current Positions: {self.current_positions}")

            # Check if we need to recalculate the portfolio
            self.context.logger.info("Checking if portfolio recalculation is needed")
            if self.should_recalculate_portfolio(self.portfolio_data):
                self.context.logger.info("Recalculating user share values")
                yield from self.calculate_user_share_values()
                # Store the updated portfolio data
                self.context.logger.info("Storing updated portfolio data")
                self.store_portfolio_data()

            payload = FetchStrategiesPayload(
                sender=sender,
                content=json.dumps(
                    {
                        "selected_protocols": serialized_protocols,
                        "trading_type": trading_type,
                    },
                    sort_keys=True,
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

    def should_recalculate_portfolio(self, last_portfolio_data: Dict) -> bool:
        """Determine if the portfolio should be recalculated."""
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
        """Calculate the value of shares for the user based on open pools."""
        total_user_share_value_usd = Decimal(0)
        allocations = []
        individual_shares = []
        portfolio_breakdown = []

        # Map DEX types to their handler functions
        dex_handlers = {
            DexType.BALANCER.value: self._handle_balancer_position,
            DexType.UNISWAP_V3.value: self._handle_uniswap_position,
            DexType.STURDY.value: self._handle_sturdy_position,
            DexType.VELODROME.value: self._handle_velodrome_position,
        }

        # Process open positions
        for position in (
            p
            for p in self.current_positions
            if p.get("status") == PositionStatus.OPEN.value
        ):
            try:
                dex_type = position.get("dex_type")
                chain = position.get("chain")

                if not dex_type or not chain:
                    self.context.logger.error("Missing dex_type or chain")
                    continue

                handler = dex_handlers.get(dex_type)
                if not handler:
                    self.context.logger.error(f"Unsupported DEX type: {dex_type}")
                    continue

                # Get position details and balances using the appropriate handler
                result = yield from handler(position, chain)
                if not result:
                    continue

                user_balances, details, token_info = result
                user_share = yield from self._calculate_position_value(
                    position, chain, user_balances, token_info, portfolio_breakdown
                )

                if user_share > 0:
                    total_user_share_value_usd += user_share
                    pool_address = (
                        position.get("pool_id")
                        if dex_type == DexType.BALANCER.value
                        else position.get("pool_address")
                    )

                    individual_shares.append(
                        (
                            user_share,
                            dex_type,
                            chain,
                            pool_address,
                            list(token_info.values()),  # token symbols
                            position.get("apr", 0.0),
                            details,
                            self.params.safe_contract_addresses.get(chain),
                            user_balances,
                        )
                    )

            except Exception as e:
                self.context.logger.error(f"Error processing position: {str(e)}")
                continue

        # Calculate safe balances value
        total_safe_value_usd = yield from self._calculate_safe_balances_value(
            portfolio_breakdown
        )

        # Calculate final portfolio metrics
        yield from self._update_portfolio_metrics(
            total_user_share_value_usd,
            individual_shares,
            portfolio_breakdown,
            allocations,
        )

        # Calculate initial investment value
        initial_investment = (
            yield from self.calculate_initial_investment_value_from_funding_events()
        )
        # Calculate total volume (total initial investment including closed positions)
        volume = yield from self._calculate_total_volume()

        self.portfolio_data = self._create_portfolio_data(
            total_user_share_value_usd,
            total_safe_value_usd,
            initial_investment,
            volume,
            allocations,
            portfolio_breakdown,
        )

    def _update_portfolio_metrics(
        self,
        total_user_share_value_usd: Decimal,
        individual_shares: List[Tuple],
        portfolio_breakdown: List[Dict],
        allocations: List[Dict],
    ) -> Generator[None, None, None]:
        """Update portfolio metrics including ratios for both breakdown and allocations."""
        # First update portfolio breakdown ratios
        self._update_portfolio_breakdown_ratios(
            portfolio_breakdown, total_user_share_value_usd
        )

        # Then calculate allocation ratios if total value is positive
        if total_user_share_value_usd > 0:
            yield from self._update_allocation_ratios(
                individual_shares, total_user_share_value_usd, allocations
            )

    def _update_portfolio_breakdown_ratios(
        self, portfolio_breakdown: List[Dict], total_value: Decimal
    ) -> None:
        """Calculate ratios for portfolio breakdown entries."""
        # Handle empty portfolio breakdown
        if not portfolio_breakdown:
            return

        # Calculate total ratio first, safely handling zero or negative total_value
        if total_value > 0:
            total_ratio = sum(
                Decimal(str(entry["value_usd"])) / total_value
                for entry in portfolio_breakdown
            )
        else:
            total_ratio = Decimal(0)

        # Update each entry with its ratio
        for entry in portfolio_breakdown:
            if total_value > 0 and total_ratio > 0:
                entry["ratio"] = round(
                    Decimal(str(entry["value_usd"])) / total_value / total_ratio, 6
                )
            else:
                entry["ratio"] = 0.0

            # Convert values to float for JSON serialization
            entry["value_usd"] = float(entry["value_usd"])
            entry["balance"] = float(entry["balance"])
            entry["price"] = float(entry["price"])

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

    def _update_allocation_ratios(
        self,
        individual_shares: List[Tuple],
        total_value: Decimal,
        allocations: List[Dict],
    ) -> Generator[None, None, None]:
        """Calculate and update allocation ratios."""
        if total_value <= 0:
            return

        # Calculate total ratio for allocations
        total_ratio = sum(
            float(user_share / total_value) * 100
            for user_share, *_ in individual_shares
        )

        # Process each share and create allocation entry
        for (
            user_share,
            dex_type,
            chain,
            pool_id,
            assets,
            apr,
            details,
            user_address,
            _,
        ) in individual_shares:
            ratio = (
                round(float(user_share / total_value) * 100 * 100 / total_ratio, 2)
                if total_ratio > 0
                else 0.0
            )

            # Get tick ranges for concentrated liquidity positions
            position = next(
                (
                    p
                    for p in self.current_positions
                    if (p.get("pool_address") == pool_id or p.get("pool_id") == pool_id)
                ),
                None,
            )

            tick_ranges = []
            if position:
                tick_ranges = yield from self._get_tick_ranges(position, chain)

            # UI supports only camel case names, but our strategies have different name for dex
            dex_type_mapping = {
                DexType.UNISWAP_V3.value: "uniswapV3",
                DexType.STURDY.value: "sturdy",
                DexType.VELODROME.value: "velodrome",
                DexType.BALANCER.value: "balancerPool",
            }

            allocation = {
                "chain": chain,
                "type": dex_type_mapping.get(dex_type, dex_type),
                "id": pool_id,
                "assets": assets,
                "apr": round(float(apr), 2),
                "details": details,
                "ratio": float(ratio),
                "address": user_address,
            }

            # Only add tick_ranges if they exist
            if tick_ranges:
                allocation["tick_ranges"] = tick_ranges

            allocations.append(allocation)

    def _create_portfolio_data(
        self,
        total_pools_value: Decimal,
        total_safe_value: Decimal,
        initial_investment: float,
        volume: float,
        allocations: List[Dict],
        portfolio_breakdown: List[Dict],
    ) -> Dict:
        """Create the final portfolio data structure."""

        # Get agent_hash from environment
        try:
            # Get agent_hash from environment
            agent_config = os.environ.get("AEA_AGENT", "")
            agent_hash = agent_config.split(":")[-1] if agent_config else "Not found"

            # Calculate total portfolio value
            total_portfolio_value = total_pools_value + total_safe_value

            allocation_assets = set()
            for allocation in allocations:
                try:
                    for asset in allocation["assets"]:
                        # Add both the asset symbol and address to handle either format
                        allocation_assets.add(asset.get("symbol", ""))
                        allocation_assets.add(asset.get("address", ""))
                except (KeyError, TypeError) as e:
                    self.context.logger.error(
                        f"Error processing allocation assets: {str(e)}"
                    )
                    continue

            # Then filter portfolio_breakdown to only include assets from allocations
            filtered_portfolio_breakdown = []
            for entry in portfolio_breakdown:
                try:
                    if (
                        entry["asset"] in allocation_assets
                        or entry["address"] in allocation_assets
                    ):
                        filtered_portfolio_breakdown.append(
                            {
                                "asset": entry["asset"],
                                "address": entry["address"],
                                "balance": float(entry["balance"]),
                                "price": float(entry["price"]),
                                "value_usd": float(entry["value_usd"]),
                                "ratio": float(entry["ratio"]),
                            }
                        )
                except (KeyError, ValueError, TypeError) as e:
                    self.context.logger.error(
                        f"Error processing portfolio breakdown entry: {str(e)}"
                    )
                    continue

            # Process allocations with error handling
            processed_allocations = []
            for allocation in allocations:
                try:
                    processed_allocation = {
                        "chain": allocation["chain"],
                        "type": allocation["type"],
                        "id": allocation["id"],
                        "assets": allocation["assets"],
                        "apr": float(allocation["apr"]),
                        "details": allocation["details"],
                        "ratio": float(allocation["ratio"]),
                        "address": allocation["address"],
                    }
                    if "tick_ranges" in allocation:
                        processed_allocation["tick_ranges"] = allocation["tick_ranges"]
                    processed_allocations.append(processed_allocation)
                except (KeyError, ValueError, TypeError) as e:
                    self.context.logger.error(f"Error processing allocation: {str(e)}")
                    continue

            # Create and return the final portfolio data structure
            return {
                "portfolio_value": float(total_portfolio_value),
                "value_in_pools": float(total_pools_value),
                "value_in_safe": float(total_safe_value),
                "initial_investment": float(initial_investment)
                if initial_investment is not None
                else None,
                "volume": float(volume) if volume is not None else None,
                "agent_hash": agent_hash,
                "allocations": processed_allocations,
                "portfolio_breakdown": filtered_portfolio_breakdown,
                "address": self.params.safe_contract_addresses.get(
                    self.params.target_investment_chains[0]
                ),
                "last_updated": int(self._get_current_timestamp()),
            }
        except Exception as e:
            self.context.logger.error(f"Error creating portfolio data: {str(e)}")
            # Return a minimal valid response in case of error
            return {
                "portfolio_value": 0.0,
                "value_in_pools": 0.0,
                "value_in_safe": 0.0,
                "initial_investment": None,
                "volume": None,
                "agent_hash": "Error",
                "allocations": [],
                "portfolio_breakdown": [],
                "address": None,
                "last_updated": int(self._get_current_timestamp()),
            }

    def _handle_balancer_position(
        self, position: Dict, chain: str
    ) -> Generator[Tuple[Dict, str, Dict[str, str]], None, None]:
        """Handle Balancer position processing."""
        self.context.logger.info(
            f"Calculating Balancer position for pool {position.get('pool_id')}"
        )
        user_address = self.params.safe_contract_addresses.get(chain)
        pool_address = position.get("pool_address")
        pool_id = position.get("pool_id")

        user_balances = yield from self.get_user_share_value_balancer(
            user_address, pool_id, pool_address, chain
        )
        details = yield from self._get_balancer_pool_name(pool_address, chain)
        token_info = {
            position.get("token0"): position.get("token0_symbol"),
            position.get("token1"): position.get("token1_symbol"),
        }

        return user_balances, details, token_info

    def _handle_uniswap_position(
        self, position: Dict, chain: str
    ) -> Generator[Tuple[Dict, str, Dict[str, str]], None, None]:
        """Handle Uniswap V3 position processing."""
        pool_address = position.get("pool_address")
        token_id = position.get("token_id")
        self.context.logger.info(
            f"Calculating Uniswap V3 position for pool {pool_address} with token ID {token_id}"
        )

        user_balances = yield from self.get_user_share_value_uniswap(
            pool_address, token_id, chain, position
        )
        details = f"Uniswap V3 Pool - {position.get('token0_symbol')}/{position.get('token1_symbol')}"
        token_info = {
            position.get("token0"): position.get("token0_symbol"),
            position.get("token1"): position.get("token1_symbol"),
        }

        return user_balances, details, token_info

    def _handle_sturdy_position(
        self, position: Dict, chain: str
    ) -> Generator[Tuple[Dict, str, Dict[str, str]], None, None]:
        """Handle Sturdy position processing."""
        self.context.logger.info(
            f"Calculating Sturdy position for aggregator {position.get('pool_address')}"
        )
        user_address = self.params.safe_contract_addresses.get(chain)
        aggregator_address = position.get("pool_address")
        asset_address = position.get("token0")

        user_balances = yield from self.get_user_share_value_sturdy(
            user_address, aggregator_address, asset_address, chain
        )
        details = yield from self._get_aggregator_name(aggregator_address, chain)
        token_info = {position.get("token0"): position.get("token0_symbol")}

        return user_balances, details, token_info

    def _handle_velodrome_position(
        self, position: Dict, chain: str
    ) -> Generator[Tuple[Dict, str, Dict[str, str]], None, None]:
        """Handle Velodrome position processing."""
        self.context.logger.info(
            f"Calculating Velodrome position for pool {position.get('pool_address')} with token ID {position.get('token_id')}"
        )
        user_address = self.params.safe_contract_addresses.get(chain)
        pool_address = position.get("pool_address")
        token_id = position.get("token_id")

        user_balances = yield from self.get_user_share_value_velodrome(
            user_address, pool_address, token_id, chain, position
        )
        details = "Velodrome " + ("CL Pool" if position.get("is_cl_pool") else "Pool")
        token_info = {
            position.get("token0"): position.get("token0_symbol"),
            position.get("token1"): position.get("token1_symbol"),
        }

        return user_balances, details, token_info

    def _calculate_position_value(
        self,
        position: Dict,
        chain: str,
        user_balances: Dict,
        token_info: Dict[str, str],
        portfolio_breakdown: List,
    ) -> Generator[Decimal, None, None]:
        """Calculate total value of a position and update portfolio breakdown."""
        user_share = Decimal(0)

        for token_address, token_symbol in token_info.items():
            asset_balance = user_balances.get(token_address)
            if asset_balance is None:
                self.context.logger.error(f"Could not find balance for {token_symbol}")
                continue

            asset_price = yield from self._fetch_token_price(token_address, chain)
            if asset_price is None:
                self.context.logger.error(f"Could not fetch price for {token_symbol}")
                continue

            asset_price = Decimal(str(asset_price))
            asset_value_usd = asset_balance * asset_price
            user_share += asset_value_usd

            # Update portfolio breakdown
            existing_asset = next(
                (
                    entry
                    for entry in portfolio_breakdown
                    if entry["address"] == token_address
                ),
                None,
            )

            if existing_asset:
                existing_asset.update(
                    {
                        "balance": float(asset_balance),
                        "value_usd": float(asset_value_usd),
                    }
                )
            else:
                portfolio_breakdown.append(
                    {
                        "asset": token_symbol,
                        "address": token_address,
                        "balance": float(asset_balance),
                        "price": float(asset_price),
                        "value_usd": float(asset_value_usd),
                    }
                )

        return user_share

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
        get_position_callable: str = "get_position",
        position_data_key: str = "data",
        slot0_contract_id: Any = None,
    ) -> Generator[None, None, Dict[str, Decimal]]:
        """Calculate concentrated liquidity position value.

        Calculate the value of a concentrated liquidity position by fetching
        position details and computing token amounts.

        :param pool_address: Address of the pool contract.
        :param chain: Chain identifier.
        :param position: Position data dictionary.
        :param token0_address: Address of token0.
        :param token1_address: Address of token1.
        :param position_manager_address: Address of position manager contract.
        :param contract_id: Contract identifier.
        :param get_position_callable: Name of the position getter function.
        :param position_data_key: Key for position data in response.
        :param slot0_contract_id: Optional contract ID for slot0 calls.
        :yield: Steps in the contract interaction process.
        :return: Dictionary mapping token addresses to their quantities.
        """
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
            position.get("positions", [])
            if isinstance(position.get("positions", []), list)
            else [position]
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
            amount0, amount1 = self._calculate_position_amounts(
                position_details, current_tick, sqrt_price_x96, pos
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
    ) -> Optional[Tuple[int, int]]:
        """Calculate token amounts for a position based on whether it's in range or not.

        Determines the token amounts for a liquidity position by checking if the
        current tick is within the position's range and calculating accordingly.

        :param position_details: Position details from the contract.
        :param current_tick: Current tick from the pool.
        :param sqrt_price_x96: Current sqrt price from the pool.
        :param position: Position data from our system.
        :return: Tuple of (amount0, amount1) representing token amounts.
        """
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

        # Log position details
        self.context.logger.info(
            f"For position, liquidity range is [{tick_lower}, {tick_upper}] "
            f"and current tick is {current_tick}"
        )

        # Check if current tick is within the provided tick range
        if tick_lower <= current_tick <= tick_upper:
            # In range, use getAmountsForLiquidity
            sqrtA = TickMath.getSqrtRatioAtTick(tick_lower)
            sqrtB = TickMath.getSqrtRatioAtTick(tick_upper)

            # Calculate amounts using getAmountsForLiquidity
            amount0, amount1 = LiquidityAmounts.getAmountsForLiquidity(
                sqrt_price_x96, sqrtA, sqrtB, liquidity
            )

            self.context.logger.info(
                f"Position is in range. Current tick: {current_tick}, "
                f"Range: [{tick_lower}, {tick_upper})"
            )
        else:
            # Out of range, return invested amounts + tokensOwed
            amount0 = position.get("amount0", 0)
            amount1 = position.get("amount1", 0)

            # Add uncollected fees (tokensOwed)
            amount0 += tokens_owed0
            amount1 += tokens_owed1

            self.context.logger.info(
                f"Position is out of range. Current tick: {current_tick}, "
                f"Range: [{tick_lower}, {tick_upper})"
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

    def _calculate_safe_balances_value(
        self, portfolio_breakdown: List[Dict]
    ) -> Generator[Decimal, None, None]:
        """Calculate the USD value of funds in the safe across all chains."""
        total_safe_value = Decimal(0)

        for chain in self.params.target_investment_chains:
            safe_address = self.params.safe_contract_addresses.get(chain)
            if not safe_address:
                self.context.logger.warning(f"No safe address found for chain {chain}")
                continue

            chain_assets = self.assets.get(chain, {})
            if not chain_assets:
                self.context.logger.warning(f"No assets found for chain {chain}")
                continue

            self.context.logger.info(f"Calculating safe balances for chain {chain}")

            for token_address, token_symbol in chain_assets.items():
                # Handle ETH (zero address) separately using get_eth_remaining_amount
                if token_address == ZERO_ADDRESS:
                    eth_balance_wei = yield from self.get_eth_remaining_amount()
                    self.context.logger.info(
                        f"Token balance for {token_symbol} is {eth_balance_wei}."
                    )
                    adjusted_balance = Decimal(str(eth_balance_wei)) / Decimal(
                        10**18
                    )  # ETH has 18 decimals

                    if adjusted_balance <= 0:
                        continue

                    # Get ETH price
                    token_price = yield from self._fetch_zero_address_price()
                    if token_price is None:
                        self.context.logger.warning(
                            f"Could not fetch price for {token_symbol}"
                        )
                        continue

                    token_price = Decimal(str(token_price))
                    token_value_usd = adjusted_balance * token_price
                    total_safe_value += token_value_usd

                    self.context.logger.info(
                        f"Safe balance - {token_symbol}: {adjusted_balance} (${token_value_usd})"
                    )

                else:
                    # Handle ERC20 tokens
                    token_balance = yield from self.contract_interact(
                        performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                        contract_address=token_address,
                        contract_public_id=ERC20.contract_id,
                        contract_callable="check_balance",
                        data_key="token",
                        account=safe_address,
                        chain_id=chain,
                    )
                    self.context.logger.info(
                        f"Token balance for {token_symbol} is {token_balance}."
                    )

                    if token_balance is None or token_balance == 0:
                        continue

                    # Get token decimals
                    token_decimals = yield from self._get_token_decimals(
                        chain, token_address
                    )
                    if token_decimals is None:
                        continue

                    # Adjust balance for decimals
                    adjusted_balance = Decimal(str(token_balance)) / Decimal(
                        10**token_decimals
                    )

                    if adjusted_balance <= 0:
                        continue

                    # Get token price
                    token_price = yield from self._fetch_token_price(
                        token_address, chain
                    )
                    if token_price is None:
                        self.context.logger.warning(
                            f"Could not fetch price for {token_symbol}"
                        )
                        continue

                    token_price = Decimal(str(token_price))
                    token_value_usd = adjusted_balance * token_price
                    total_safe_value += token_value_usd

                    self.context.logger.info(
                        f"Safe balance - {token_symbol}: {adjusted_balance} (${token_value_usd})"
                    )

                # Add to portfolio breakdown if not already present
                existing_asset = next(
                    (
                        entry
                        for entry in portfolio_breakdown
                        if entry["address"] == token_address
                    ),
                    None,
                )

                if existing_asset:
                    # Update existing entry by adding safe balance
                    existing_asset["balance"] = float(
                        Decimal(str(existing_asset["balance"])) + adjusted_balance
                    )
                    existing_asset["value_usd"] = float(
                        Decimal(str(existing_asset["value_usd"])) + token_value_usd
                    )
                else:
                    # Add new entry for safe balance
                    portfolio_breakdown.append(
                        {
                            "asset": token_symbol,
                            "address": token_address,
                            "balance": float(adjusted_balance),
                            "price": float(token_price),
                            "value_usd": float(token_value_usd),
                        }
                    )

        self.context.logger.info(f"Total safe value: ${total_safe_value}")
        return total_safe_value

    def _get_aggregator_name(
        self, aggregator_address: str, chain: str
    ) -> Generator[None, None, Optional[str]]:
        """Get the name of the Sturdy Aggregator."""
        aggreator_name = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=aggregator_address,
            contract_public_id=YearnV3VaultContract.contract_id,
            contract_callable="name",
            data_key="name",
            chain_id=chain,
        )
        return aggreator_name

    def _calculate_total_volume(self) -> Generator[None, None, Optional[float]]:
        """Calculate the total volume (total initial investment including closed positions)."""
        total_volume = 0.0

        # Load cached investment values from KV store
        cached_values = yield from self._read_kv(keys=("initial_investment_values",))
        if cached_values and cached_values.get("initial_investment_values"):
            try:
                self.initial_investment_values_per_pool = json.loads(
                    cached_values.get("initial_investment_values")
                )
                self.context.logger.info(
                    f"Loaded {len(self.initial_investment_values_per_pool)} cached position values from KV store"
                )
            except json.JSONDecodeError:
                self.context.logger.warning(
                    "Failed to parse cached investment values from KV store"
                )

        # Process all positions (both open and closed)
        for position in self.current_positions:
            # Create a unique key for this position
            pool_id = position.get("pool_address", position.get("pool_id"))
            tx_hash = position.get("tx_hash")
            position_key = f"{pool_id}_{tx_hash}"

            # Check if we already calculated the value for this position
            if position_key in self.initial_investment_values_per_pool:
                position_value = self.initial_investment_values_per_pool[position_key]
                self.context.logger.info(
                    f"Using cached position value: {position_value} for {position_key}"
                )
                total_volume += position_value
                continue

            # Get token addresses and amounts
            token0 = position.get("token0")
            token1 = position.get("token1")
            amount0 = position.get("amount0")
            amount1 = position.get("amount1")
            timestamp = position.get("timestamp") or position.get("enter_timestamp")
            chain = position.get("chain")

            if None in (token0, amount0, timestamp, chain):
                self.context.logger.error(
                    "Missing token0, amount0, timestamp, or chain in position data."
                )
                continue

            # Get token decimals
            token0_decimals = yield from self._get_token_decimals(chain, token0)
            if not token0_decimals:
                continue

            # Calculate adjusted amount for token0
            initial_amount0 = Decimal(str(amount0)) / Decimal(10**token0_decimals)

            # Calculate adjusted amount for token1 if it exists
            initial_amount1 = None
            if token1 is not None and amount1 is not None:
                token1_decimals = yield from self._get_token_decimals(chain, token1)
                if not token1_decimals:
                    continue
                initial_amount1 = Decimal(str(amount1)) / Decimal(10**token1_decimals)

            date_str = datetime.utcfromtimestamp(timestamp).strftime("%d-%m-%Y")

            tokens = [[position.get("token0_symbol"), token0]]
            if token1 is not None:
                tokens.append([position.get("token1_symbol"), token1])

            historical_prices = yield from self._fetch_historical_token_prices(
                tokens, date_str, chain
            )

            if not historical_prices:
                self.context.logger.error("Failed to fetch historical token prices.")
                continue

            # Calculate value for token0
            initial_price0 = historical_prices.get(token0)
            if initial_price0 is None:
                self.context.logger.error("Historical price not found for token0.")
                continue

            position_value = float(initial_amount0 * Decimal(str(initial_price0)))

            # Add value for token1 if it exists
            if token1 is not None and initial_amount1 is not None:
                initial_price1 = historical_prices.get(token1)
                if initial_price1 is None:
                    self.context.logger.error("Historical price not found for token1.")
                    continue
                position_value += float(initial_amount1 * Decimal(str(initial_price1)))

            # Cache the calculated value
            self.initial_investment_values_per_pool[position_key] = position_value

            # Save the updated cache to KV store
            yield from self._write_kv(
                {
                    "initial_investment_values": json.dumps(
                        self.initial_investment_values_per_pool
                    )
                }
            )

            total_volume += position_value
            self.context.logger.info(
                f"Position value for volume calculation: {position_value}"
            )

        self.context.logger.info(
            f"Total volume (including closed positions): {total_volume}"
        )
        return total_volume if total_volume > 0 else None

    def _get_balancer_pool_name(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[str]]:
        """Get the name of the Balancer Pool."""
        pool_name = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=WeightedPoolContract.contract_id,
            contract_callable="get_name",
            data_key="name",
            chain_id=chain,
        )
        return pool_name

    def check_and_update_zero_liquidity_positions(self) -> None:
        """Check for positions with zero liquidity and mark them as closed."""
        if not self.current_positions:
            return

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
                    return (yield self._load_chain_total_investment(chain))

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
            if chain == "mode":
                self.context.logger.info("Using Mode-specific transfer fetching")
                all_transfers = self._fetch_all_transfers_until_date_mode(
                    safe_address, current_date, fetch_till_date
                )
            elif chain == "optimism":
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
                all_transfers, chain
            )
            total_investment += chain_investment

        timestamp = int(self._get_current_timestamp())
        yield from self._write_kv(
            {"last_initial_value_calculated_timestamp": str(timestamp)}
        )
        self.context.logger.info(
            f"Total initial investment from all chains: ${total_investment}"
        )
        return total_investment if total_investment > 0 else None

    def _calculate_chain_investment_value(
        self, all_transfers: Dict, chain: str
    ) -> Generator[None, None, float]:
        """Calculate investment value for a specific chain and update stored total."""
        # Load existing total investment for this chain
        existing_total = yield from self._load_chain_total_investment(chain)
        if not existing_total:
            last_calculated_date = "1970-01-01"
        else:
            chain_events = self.funding_events.get(chain, {})
            if not chain_events:
                last_calculated_date = "1970-01-01"
            else:
                try:
                    last_calculated_date = sorted(list(chain_events.keys()))[-1]
                except IndexError:
                    last_calculated_date = "1970-01-01"

        # Only calculate value for new transfers (all_transfers contains only new dates)
        new_investment = 0.0

        for date, transfers in all_transfers.items():
            current_date = datetime.strptime(date, "%Y-%m-%d")
            last_date = datetime.strptime(last_calculated_date, "%Y-%m-%d")
            if current_date <= last_date:
                continue

            for transfer in transfers:
                try:
                    # Get token price for the transfer date
                    token_symbol = transfer.get("symbol", "Unknown")
                    amount = transfer.get("delta", transfer.get("amount", 0))

                    if amount <= 0:
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
        updated_total = existing_total + new_investment
        yield from self._save_chain_total_investment(chain, updated_total)

        self.context.logger.info(f"New {chain} investment: ${new_investment}")
        self.context.logger.info(
            f"Total {chain} investment (updated): ${updated_total}"
        )

        return updated_total

    def _fetch_all_transfers_until_date_mode(
        self, address: str, end_date: str, fetch_till_date: bool
    ) -> Dict:
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

            self._fetch_token_transfers_mode(
                address, end_datetime, all_transfers_by_date, fetch_till_date
            )

            # Fetch ETH transfers
            self.context.logger.info("Fetching Mode ETH transfers...")
            self._fetch_eth_transfers_mode(
                address, end_datetime, all_transfers_by_date, fetch_till_date
            )

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
            return {}

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

    def _fetch_eth_transfers(
        self,
        address: str,
        end_datetime: datetime,
        all_transfers_by_date: dict,
        existing_data: dict,
    ) -> Generator[None, None, None]:
        """Fetch ETH transfers from Mode blockchain explorer."""
        base_url = "https://explorer-mode-mainnet-0.t.conduit.xyz/api/v2"
        processed_count = 0

        endpoint = f"{base_url}/addresses/{address}/transactions"
        success, response_json = yield from self._request_with_retries(
            endpoint=endpoint,
            headers={"Accept": "application/json"},
            rate_limited_code=429,
            rate_limited_callback=self.coingecko.rate_limited_status_callback,
            retry_wait=self.params.sleep_time,
        )

        if not success:
            self.context.logger.error("Failed to fetch ETH transfers")
            return None

        eth_transactions = response_json.get("items", [])
        if not eth_transactions:
            return None

        for tx in eth_transactions:
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
                    from_address, tx, is_eth_transfer=True
                ):
                    value_wei = int(tx.get("value", "0"))
                    amount_eth = value_wei / 10**18

                    transfer_data = {
                        "from_address": from_address.get("hash", ""),
                        "amount": amount_eth,
                        "token_address": "",
                        "symbol": "ETH",
                        "timestamp": tx.get("timestamp", ""),
                        "tx_hash": tx.get("hash", ""),
                        "type": "eth",
                    }

                    all_transfers_by_date[tx_date].append(transfer_data)
                    processed_count += 1

        self.context.logger.info(f"Completed ETH transfers: {processed_count} found")

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
    ) -> None:
        """
        Fetch token transfers from Mode blockchain explorer for a specific date or all transfers till that date.

        Args:
            address: The address to fetch transfers for
            target_date: The specific date to fetch transfers for (format: "YYYY-MM-DD")
            all_transfers_by_date: Dictionary to store transfers by date
            fetch_all_till_date: If True, fetch all transfers up to target_date. If False, fetch only target_date transfers
        """
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
                verify=False,  # nosec B501
                timeout=30,
            )

            if not response.status_code == 200:
                self.context.logger.error("Failed to fetch Mode token transfers")
                return None

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

    def _fetch_eth_transfers_mode(
        self,
        address: str,
        target_date: str,
        all_transfers_by_date: dict,
        fetch_till_date: bool,
    ) -> None:
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
            except (IndexError, ValueError):
                latest_datetime = datetime(1970, 1, 1)
        else:
            latest_datetime = datetime(1970, 1, 1)

        while has_more_pages:
            response = requests.get(
                endpoint,
                params=params,
                headers={"Accept": "application/json"},
                verify=False,  # nosec B501
                timeout=30,
            )

            if not response.status_code == 200:
                self.context.logger.error("Failed to fetch Mode coin balance history")
                return

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

                # Check for next page
                cursor = response_json.get("next")
                if not cursor:
                    break

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
                return True

            # If it has code, check if it's a GnosisSafe
            safe_check_url = f"https://safe-transaction-optimism.safe.global/api/v1/safes/{from_address}/"
            success, _ = yield from self._request_with_retries(
                endpoint=safe_check_url,
                headers={"Accept": "application/json"},
                rate_limited_code=429,
                rate_limited_callback=self.coingecko.rate_limited_status_callback,
                retry_wait=self.params.sleep_time,
            )

            if success:
                return True

            self.context.logger.info(
                f"Excluding transfer from contract: {from_address}"
            )
            return False

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

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

"""This module contains the behaviour for evaluating opportunities and forming actions for the 'liquidity_trader_abci' skill."""

import asyncio
import json
import traceback
from concurrent.futures import Future
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

from aea.protocols.base import Message
from aea.protocols.dialogue.base import Dialogue
from eth_utils import to_checksum_address

from packages.valory.protocols.ipfs import IpfsMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    Action,
    COIN_ID_MAPPING,
    DexType,
    HTTP_OK,
    LiquidityTraderBaseBehaviour,
    METRICS_UPDATE_INTERVAL,
    MIN_TIME_IN_POSITION,
    OLAS_ADDRESSES,
    PositionStatus,
    REWARD_TOKEN_ADDRESSES,
    THRESHOLDS,
    WHITELISTED_ASSETS,
    ZERO_ADDRESS,
    execute_strategy,
)
from packages.valory.skills.liquidity_trader_abci.io_.loader import (
    ComponentPackageLoader,
)
from packages.valory.skills.liquidity_trader_abci.models import SharedState
from packages.valory.skills.liquidity_trader_abci.payloads import (
    EvaluateStrategyPayload,
)
from packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy import (
    EvaluateStrategyRound,
)


class EvaluateStrategyBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that finds the opportunity and builds actions."""

    matching_round: Type[AbstractRound] = EvaluateStrategyRound
    selected_opportunities = None
    position_to_exit = None
    trading_opportunities = []

    def async_act(self) -> Generator:
        """Execute the behaviour's async action."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            # Check minimum hold period
            # Check if no current positions and uninvested ETH, prepare swap to USDC
            actions = yield from self.check_and_prepare_non_whitelisted_swaps()
            if actions:
                yield from self.send_actions(actions)

            should_hold = self.check_minimum_hold_period()
            if should_hold:
                yield from self.send_actions()
                return

            # Check for funds
            are_funds_available = self.check_funds()
            if not are_funds_available:
                yield from self.send_actions()
                return

            # Fetch trading opportunities
            yield from self.fetch_all_trading_opportunities()

            # Update metrics for open positions
            self.update_position_metrics()

            # Execute strategy and prepare actions
            actions = yield from self.prepare_strategy_actions()

            # Send final actions
            yield from self.send_actions(actions)

    def send_actions(self, actions: Optional[List[Any]] = None) -> Generator:
        """Send actions and complete the round."""
        if actions is None:
            actions = []

        payload = EvaluateStrategyPayload(
            sender=self.context.agent_address, actions=json.dumps(actions)
        )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def validate_and_prepare_velodrome_inputs(
        self, tick_bands, current_price, tick_spacing=1
    ):
        """Validates inputs and prepares data for Velodrome CL position analysis."""

        # Input validation
        if not tick_bands:
            self.context.logger.error("No tick bands provided")
            return None

        if current_price <= 0:
            self.context.logger.error(
                f"Invalid price: {current_price}. Price must be positive."
            )
            return None

        # Convert current price to tick
        try:
            import math

            current_tick = int(math.log(current_price) / math.log(1.0001))
        except (ValueError, TypeError) as e:
            self.context.logger.error(
                f"Error converting price {current_price} to tick: {str(e)}"
            )
            return None

        # Validate tick spacing alignment
        warnings = []
        for band in tick_bands:
            if (
                band.get("tick_lower") % tick_spacing != 0
                or band.get("tick_upper") % tick_spacing != 0
            ):
                warnings.append(
                    f"Tick range [{band.get('tick_lower')}, {band.get('tick_upper')}] "
                    f"not aligned with tick spacing {tick_spacing}"
                )

        # Filter out zero allocation bands
        valid_bands = [band for band in tick_bands if band.get("allocation", 0) > 0]
        if not valid_bands:
            self.context.logger.error("No bands with positive allocation")
            return None

        # Additional validation for each band
        validated_bands = []
        for band in valid_bands:
            tick_lower = band.get("tick_lower")
            tick_upper = band.get("tick_upper")
            allocation = band.get("allocation")

            # Check if band is valid
            if tick_lower >= tick_upper:
                warnings.append(
                    f"Invalid band: tick_lower ({tick_lower}) >= tick_upper ({tick_upper})"
                )
                continue

            validated_bands.append(
                {
                    "tick_lower": tick_lower,
                    "tick_upper": tick_upper,
                    "allocation": allocation,
                }
            )

        if not validated_bands:
            self.context.logger.error("No valid bands after validation")
            return None

        return {
            "validated_bands": validated_bands,
            "current_price": current_price,
            "current_tick": current_tick,
            "tick_spacing": tick_spacing,
            "warnings": warnings,
        }

    def calculate_velodrome_token_ratios(self, validated_data):
        """Calculates token ratios and requirements for Velodrome CL positions."""

        if not validated_data:
            return None

        validated_bands = validated_data["validated_bands"]
        current_price = validated_data["current_price"]
        current_tick = validated_data["current_tick"]
        warnings = validated_data["warnings"].copy()

        # Process each band to calculate token ratios
        position_requirements = []
        total_weighted_token0 = 0
        total_weighted_token1 = 0
        total_allocation = 0

        for band in validated_bands:
            tick_lower = band["tick_lower"]
            tick_upper = band["tick_upper"]
            allocation = band["allocation"]

            # Convert ticks to prices
            lower_bound_price = 1.0001**tick_lower
            upper_bound_price = 1.0001**tick_upper

            # Calculate token ratios based on current price position
            if current_price <= lower_bound_price:
                # Price below range - need 100% token0
                token0_ratio = 1.0
                token1_ratio = 0.0
                status = "BELOW_RANGE"
            elif current_price >= upper_bound_price:
                # Price above range - need 100% token1
                token0_ratio = 0.0
                token1_ratio = 1.0
                status = "ABOVE_RANGE"
            else:
                # Price in range - calculate using interpolation formula
                try:
                    token1_ratio = min(
                        max(
                            (current_price - lower_bound_price)
                            / (upper_bound_price - lower_bound_price),
                            0,
                        ),
                        1,
                    )
                    token0_ratio = 1.0 - token1_ratio
                    status = "IN_RANGE"
                except Exception as e:
                    warnings.append(
                        f"Error calculating ratios for band [{tick_lower}, {tick_upper}]: {str(e)}"
                    )
                    # Default to 50/50 in case of calculation error
                    token0_ratio = 0.5
                    token1_ratio = 0.5
                    status = "ERROR"

            # Track the weighted token ratios
            total_weighted_token0 += token0_ratio * allocation
            total_weighted_token1 += token1_ratio * allocation
            total_allocation += allocation

            position_requirements.append(
                {
                    "tick_range": [tick_lower, tick_upper],
                    "current_tick": current_tick,
                    "status": status,
                    "allocation": float(allocation),
                    "token0_ratio": token0_ratio,
                    "token1_ratio": token1_ratio,
                }
            )

        # Calculate overall ratios
        overall_token0_ratio = (
            total_weighted_token0 / total_allocation if total_allocation > 0 else 0
        )
        overall_token1_ratio = (
            total_weighted_token1 / total_allocation if total_allocation > 0 else 0
        )

        # Generate recommendations
        all_same_status = all(
            pos["status"] == position_requirements[0]["status"]
            for pos in position_requirements
        )

        if all_same_status:
            if position_requirements[0]["status"] == "BELOW_RANGE":
                recommendation = "Provide 100% token0, 0% token1 for all positions"
            elif position_requirements[0]["status"] == "ABOVE_RANGE":
                recommendation = "Provide 0% token0, 100% token1 for all positions"
            else:
                recommendation = f"Provide {overall_token0_ratio*100:.2f}% token0, {overall_token1_ratio*100:.2f}% token1 for all positions"
        else:
            recommendation = f"Mixed position requirements. Overall: {overall_token0_ratio*100:.2f}% token0, {overall_token1_ratio*100:.2f}% token1"

        # Log any warnings
        for warning in warnings:
            self.context.logger.warning(warning)

        self.context.logger.info(
            f"Position analysis complete - Current tick: {current_tick}, "
            f"Token0 ratio: {overall_token0_ratio:.4f}, Token1 ratio: {overall_token1_ratio:.4f}"
        )

        return {
            "position_requirements": position_requirements,
            "current_price": current_price,
            "current_tick": current_tick,
            "overall_token0_ratio": overall_token0_ratio,
            "overall_token1_ratio": overall_token1_ratio,
            "recommendation": recommendation,
            "warnings": warnings,
        }

    def calculate_velodrome_cl_token_requirements(
        self, tick_bands, current_price, tick_spacing=1
    ):
        """Determines token requirements for Velodrome CL positions based on current price."""
        # Step 1: Validate and prepare inputs
        validated_data = self.validate_and_prepare_velodrome_inputs(
            tick_bands, current_price, tick_spacing
        )

        if not validated_data:
            return None

        # Step 2: Calculate token ratios and generate recommendations
        return self.calculate_velodrome_token_ratios(validated_data)

    def get_velodrome_position_requirements(
        self,
    ) -> Generator[None, None, Dict[str, Any]]:
        """Generator function to determine token requirements for Velodrome CL positions."""
        results = {}
        max_ration = 1.0
        for opportunity in self.selected_opportunities:
            if opportunity.get("dex_type") == "velodrome" and opportunity.get(
                "is_cl_pool"
            ):
                try:
                    # Get the necessary parameters
                    self.context.logger.info(
                        f"Analyzing Velodrome CL pool: {opportunity.get('pool_address')}"
                    )

                    chain = opportunity["chain"]
                    self.context.logger.info(f"chain: {chain}")
                    pool_address = opportunity["pool_address"]
                    kwargs = {
                        "chain": chain,
                        "pool_address": pool_address,
                        "is_stable": opportunity["is_stable"],
                    }

                    pool = self.pools.get(opportunity["dex_type"])

                    # Get tick spacing for the pool
                    tick_spacing = yield from pool._get_tick_spacing_velodrome(
                        self, pool_address, chain
                    )
                    if not tick_spacing:
                        self.context.logger.error(
                            f"Failed to get tick spacing for pool {pool_address}"
                        )
                        continue

                    # Calculate tick bands and get current price
                    tick_bands = (
                        yield from pool._calculate_tick_lower_and_upper_velodrome(
                            self, **kwargs
                        )
                    )
                    self.context.logger.info(f"tick_bands : {tick_bands}")
                    if not tick_bands:
                        self.context.logger.error(
                            f"Failed to calculate tick bands for pool {pool_address}"
                        )
                        continue

                    current_price = yield from pool._get_current_pool_price(
                        self, pool_address, chain
                    )
                    if current_price is None:
                        self.context.logger.error(
                            f"Failed to get current price for pool {pool_address}"
                        )
                        continue

                    # Calculate token requirements
                    requirements = self.calculate_velodrome_cl_token_requirements(
                        tick_bands, current_price, tick_spacing
                    )
                    if not requirements:
                        self.context.logger.error(
                            "Failed to calculate token requirements"
                        )
                        continue

                    token0_symbol = opportunity.get("token0_symbol", "token0")
                    token1_symbol = opportunity.get("token1_symbol", "token1")

                    self.context.logger.info(
                        f"Velodrome position requirements for {token0_symbol}/{token1_symbol}: "
                        f"{requirements['recommendation']}"
                    )

                    # Extract token ratios as percentages
                    token0_ratio = float(requirements["overall_token0_ratio"])
                    token1_ratio = float(requirements["overall_token1_ratio"])

                    # Add percentage values to the opportunity
                    opportunity["token0_percentage"] = token0_ratio * 100
                    opportunity["token1_percentage"] = token1_ratio * 100

                    self.context.logger.info(
                        f"Token allocation percentages: {opportunity['token0_percentage']:.2f}% {token0_symbol}, "
                        f"{opportunity['token1_percentage']:.2f}% {token1_symbol}"
                    )

                    # Store these requirements
                    opportunity["token_requirements"] = requirements
                    # IMPORTANT: Add tick_spacing and tick_bands to the opportunity
                    opportunity["tick_spacing"] = tick_spacing
                    opportunity["tick_ranges"] = tick_bands

                    # Get available balances
                    token0 = opportunity["token0"]
                    token1 = opportunity["token1"]

                    token0_balance = (
                        yield from self._get_token_balance(
                            chain,
                            self.params.safe_contract_addresses.get(chain),
                            token0,
                        )
                        or 0
                    )

                    token1_balance = (
                        yield from self._get_token_balance(
                            chain,
                            self.params.safe_contract_addresses.get(chain),
                            token1,
                        )
                        or 0
                    )

                    # Apply relative_funds_percentage if specified
                    relative_funds_percentage = opportunity.get(
                        "relative_funds_percentage", 1.0
                    )
                    token0_balance = int(token0_balance * relative_funds_percentage)
                    token1_balance = int(token1_balance * relative_funds_percentage)

                    # Update max_amounts_in based on the requirements and actual balances
                    # Using the weighted ratios from all the bands
                    if requirements["overall_token0_ratio"] > max_ration:
                        # Only need token0
                        opportunity["max_amounts_in"] = [token0_balance, 0]
                        self.context.logger.info(
                            f"Using only token0: {token0_balance} {token0_symbol}"
                        )
                    elif requirements["overall_token1_ratio"] > max_ration:
                        # Only need token1
                        opportunity["max_amounts_in"] = [0, token1_balance]
                        self.context.logger.info(
                            f"Using only token1: {token1_balance} {token1_symbol}"
                        )
                    else:
                        # Need both tokens in specific ratio
                        # Find which token is limiting based on the required ratio
                        max_amount0 = token0_balance
                        max_amount1 = token1_balance

                        # Check if either ratio is zero to avoid division by zero
                        if (
                            requirements["overall_token0_ratio"] <= 0
                            or requirements["overall_token1_ratio"] <= 0
                        ):
                            self.context.logger.warning(
                                "One of the token ratios is zero, using default 50/50 split"
                            )
                            # Fall back to 50/50 if we hit this edge case
                            max_amount0 = int(token0_balance * 0.5)
                            max_amount1 = int(token1_balance * 0.5)
                        else:
                            # Calculate what amount of token1 we would need given our token0
                            required_token1 = int(
                                max_amount0
                                * requirements["overall_token1_ratio"]
                                / requirements["overall_token0_ratio"]
                            )

                            # If required token1 is more than we have, scale both tokens down
                            if required_token1 > max_amount1 and required_token1 > 0:
                                scale_factor = max_amount1 / required_token1
                                max_amount0 = int(max_amount0 * scale_factor)
                                max_amount1 = required_token1
                            elif required_token1 < max_amount1:
                                # If we have excess token1, calculate how much token0 we need
                                # to maintain the ratio
                                required_token0 = int(
                                    max_amount1
                                    * requirements["overall_token0_ratio"]
                                    / requirements["overall_token1_ratio"]
                                )

                                if required_token0 < max_amount0:
                                    # We have excess of both tokens, so use the calculated amounts
                                    max_amount0 = required_token0
                                    max_amount1 = max_amount1
                                else:
                                    # We have excess token1 but not enough token0
                                    scale_factor = max_amount0 / required_token0
                                    max_amount1 = int(max_amount1 * scale_factor)

                        opportunity["max_amounts_in"] = [max_amount0, max_amount1]
                        self.context.logger.info(
                            f"Using both tokens: {max_amount0} {token0_symbol} ({requirements['overall_token0_ratio']*100:.2f}%), "
                            f"{max_amount1} {token1_symbol} ({requirements['overall_token1_ratio']*100:.2f}%)"
                        )

                    # Store results for this pool
                    results[pool_address] = requirements

                except Exception as e:
                    self.context.logger.error(
                        f"Error analyzing Velodrome position: {str(e)}"
                    )

                    self.context.logger.error(traceback.format_exc())

        self.context.logger.info("Velodrome position analysis complete")
        return results

    def check_minimum_hold_period(self) -> bool:
        """Check if any position is still in minimum hold period."""
        if not self.current_positions:
            return False

        open_position = next(
            (
                pos
                for pos in self.current_positions
                if pos.get("status") == PositionStatus.OPEN.value
            ),
            None,
        )

        if not open_position:
            return False

        timestamp = (
            open_position.get("timestamp") or open_position.get("enter_timestamp") or 0
        )
        time_in_position = int(self._get_current_timestamp()) - timestamp

        try:
            if time_in_position < MIN_TIME_IN_POSITION:
                remaining_time = MIN_TIME_IN_POSITION - time_in_position
                days, hours = divmod(remaining_time, 24 * 3600)
                hours //= 3600
                self.context.logger.info(
                    f"Position {open_position.get('pool_address')} is still in minimum hold period. "
                    f"Waiting for {days} days and {hours} hours before closing it."
                )
                return True
        except Exception as e:
            self.context.logger.error(f"Error checking minimum hold period: {str(e)}")
            return False

        return False

    def check_funds(self) -> bool:
        """Check if there are any funds available."""

        if not self.current_positions:
            has_funds = any(
                asset.get("balance", 0) > 0
                for position in self.synchronized_data.positions
                for asset in position.get("assets", [])
            )
            return has_funds

        return True

    def update_position_metrics(self) -> None:
        """Update metrics for all open positions."""
        if not self.current_positions:
            return

        current_timestamp = self._get_current_timestamp()
        open_positions = [
            pos
            for pos in self.current_positions
            if pos.get("status") == PositionStatus.OPEN.value
        ]

        for position in open_positions:
            dex_type = position.get("dex_type")
            strategy = self.params.dex_type_to_strategy.get(dex_type)

            if not strategy:
                self.context.logger.error(f"No strategy found for dex type {dex_type}")
                continue

            last_metrics_update = position.get("last_metrics_update", 0)
            time_since_update = current_timestamp - last_metrics_update

            if time_since_update >= METRICS_UPDATE_INTERVAL:
                self.context.logger.info(
                    f"Recalculating metrics for position {position.get('pool_address')} - "
                    f"last update was {time_since_update / 3600:.2f} hours ago"
                )
                if metrics := self.get_returns_metrics_for_opportunity(
                    position, strategy
                ):
                    metrics["last_metrics_update"] = current_timestamp
                    position.update(metrics)
            else:
                self.context.logger.info(
                    f"Skipping metrics calculation for position {position.get('pool_address')} - "
                    f"last update was {time_since_update / 3600:.2f} hours ago"
                )

        self.store_current_positions()

    def check_and_prepare_non_whitelisted_swaps(
        self,
    ) -> Generator[None, None, Optional[List[Any]]]:
        """Check all funds in safe and swap non-whitelisted assets to USDC if value > $5."""
        try:
            actions = []
            target_chain = self.params.target_investment_chains[0]
            usdc_address = self._get_usdc_address(target_chain)

            if not usdc_address:
                self.context.logger.warning(
                    f"Could not get USDC address for {target_chain}"
                )
                return []

            # Get all positions to check assets in safe
            for position in self.synchronized_data.positions:
                chain = position.get("chain")
                whitelisted_tokens = [
                    addr.lower() for addr in WHITELISTED_ASSETS.get(chain, {}).keys()
                ]
                olas_token_address = OLAS_ADDRESSES.get(chain, "").lower()
                for asset in position.get("assets", []):
                    asset_address = asset.get("address", "").lower()
                    asset_symbol = asset.get("asset_symbol")
                    balance = asset.get("balance", 0)

                    # Skip if no balance or asset is whitelisted
                    if (
                        balance <= 0
                        or asset_address in whitelisted_tokens
                        or asset_address == olas_token_address
                    ):
                        continue

                    self.context.logger.info("Preparing swap action to USDC.")

                    actions.append(
                        {
                            "action": Action.FIND_BRIDGE_ROUTE.value,
                            "from_chain": chain,
                            "to_chain": chain,  # Same chain swap
                            "from_token": asset_address,
                            "from_token_symbol": asset_symbol,
                            "to_token": usdc_address,
                            "to_token_symbol": "USDC",
                            "funds_percentage": 1.0,  # Use all available balance
                        }
                    )

                    self.context.logger.info(
                        f"Prepared {asset_symbol} to USDC swap on {chain}: "
                        f"{asset_symbol} -> USDC"
                    )

            return actions

        except Exception as e:
            self.context.logger.error(
                f"Error in check_and_prepare_non_whitelisted_swaps: {str(e)}"
            )
            return []

    def _get_usdc_address(self, chain: str) -> Optional[str]:
        """Get USDC token address for the specified chain."""
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

    def prepare_strategy_actions(self) -> Generator[None, None, Optional[List[Any]]]:
        """Execute strategy and prepare actions."""
        if not self.trading_opportunities:
            self.context.logger.info("No trading opportunities found")
            return []

        yield from self.execute_hyper_strategy()
        actions = (
            yield from self.get_order_of_transactions()
            if self.selected_opportunities is not None
            else []
        )

        self.context.logger.info(
            f"Actions: {actions}" if actions else "No actions prepared"
        )

        return actions

    def execute_hyper_strategy(self) -> Generator[None, None, None]:
        """Executes hyper strategy"""
        hyper_strategy = self.params.selected_hyper_strategy
        composite_score = None
        try:
            db_data = yield from self._read_kv(keys=("composite_score",))
            composite_score = db_data.get("composite_score") if db_data else None
            self.context.logger.info(
                f"Retrieved composite score from KV store: {composite_score}"
            )
        except Exception as e:
            self.context.logger.warning(
                f"Failed to read composite score from KV store: {str(e)}"
            )
        # Fall back to default thresholds if no composite score found
        if composite_score is None:
            composite_score = THRESHOLDS.get(self.synchronized_data.trading_type, {})
            self.context.logger.info(
                f"Using default threshold for {self.synchronized_data.trading_type}: {composite_score}"
            )
        # Ensure composite_score is a float
        try:
            composite_score = float(composite_score)
        except (ValueError, TypeError):
            composite_score = 0.0
        kwargs = {
            "strategy": hyper_strategy,
            "trading_opportunities": self.trading_opportunities,
            "current_positions": [
                pos
                for pos in self.current_positions
                if pos.get("status") == PositionStatus.OPEN.value
            ],
            "max_pools": self.params.max_pools,
            "composite_score_threshold": composite_score,
        }
        self.context.logger.info(f"kwargs: {kwargs}")
        self.context.logger.info(f"Evaluating hyper strategy: {hyper_strategy}")
        result = self.execute_strategy(**kwargs)
        self.selected_opportunities = result.get("optimal_strategies")
        self.position_to_exit = result.get("position_to_exit")

        logs = result.get("logs", [])
        if logs:
            for log in logs:
                self.context.logger.info(log)

        reasoning = result.get("reasoning")
        if reasoning:
            self.shared_state.agent_reasoning = reasoning
        self.context.logger.info(f"Agent Reasoning: {reasoning}")

        if self.selected_opportunities is not None:
            self.context.logger.info(
                f"Selected opportunities: {self.selected_opportunities}"
            )
            for selected_opportunity in self.selected_opportunities:
                # Convert token addresses to checksum addresses if they are present
                # Dynamically handle multiple tokens
                token_keys = [
                    key
                    for key in selected_opportunity.keys()
                    if key.startswith("token")
                    and not key.endswith("_symbol")
                    and isinstance(selected_opportunity[key], str)
                ]
                for token_key in token_keys:
                    selected_opportunity[token_key] = to_checksum_address(
                        selected_opportunity[token_key]
                    )
                    self.context.logger.info(
                        f"selected_opportunity[token_key] : {selected_opportunity[token_key]}"
                    )

    def get_result(self, future: Future) -> Generator[None, None, Optional[Any]]:
        """Get the completed futures"""
        while True:
            if not future.done():
                yield
                continue
            try:
                result = future.result()
                return result
            except Exception as e:
                self.context.logger.error(
                    f"Exception occurred while executing strategy: {e}",
                )
                return None

    def fetch_all_trading_opportunities(self) -> Generator[None, None, None]:
        """Fetches all the trading opportunities using asyncio for concurrency."""
        self.trading_opportunities.clear()
        yield from self.download_strategies()
        strategies = self.synchronized_data.selected_protocols.copy()

        tried_strategies: Set[str] = set()
        self.context.logger.info(f"Selected Strategies: {strategies}")

        # Collect strategy kwargs
        strategy_kwargs_list = []
        for next_strategy in strategies:
            self.context.logger.info(f"Preparing strategy: {next_strategy}")
            kwargs: Dict[str, Any] = self.params.strategies_kwargs.get(
                next_strategy, {}
            )
            kwargs.update(
                {
                    "strategy": next_strategy,
                    "chains": self.params.target_investment_chains,
                    "protocols": self.params.available_protocols,
                    "chain_to_chain_id_mapping": self.params.chain_to_chain_id_mapping,
                    "current_positions": (
                        [
                            to_checksum_address(pos.get("pool_address"))
                            for pos in self.current_positions
                            if pos.get("status") == PositionStatus.OPEN.value
                            and pos.get("pool_address")
                        ]
                        if self.current_positions
                        else []
                    ),
                    "coingecko_api_key": self.coingecko.api_key,
                    "whitelisted_assets": WHITELISTED_ASSETS,
                    "get_metrics": False,
                    "coin_id_mapping": COIN_ID_MAPPING,
                }
            )
            strategy_kwargs_list.append((next_strategy, kwargs))
            self.context.logger.info(f"Strategy kwargs for {next_strategy}: {kwargs}")

        strategies_executables = self.shared_state.strategies_executables

        async def async_execute_strategy(
            strategy_name, strategies_executables, **kwargs
        ):
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: execute_strategy(
                        strategy_name, strategies_executables, **kwargs
                    ),
                )
            except TypeError as e:
                # Handle missing arguments error
                return {
                    "error": [
                        f"Strategy {strategy_name} missing required argument: {str(e)}"
                    ],
                    "result": [],
                }
            except Exception as e:
                # Handle any other unexpected errors
                return {
                    "error": [
                        f"Unexpected error in strategy {strategy_name}: {str(e)}"
                    ],
                    "result": [],
                }

        async def run_all_strategies():
            tasks = []
            results = []

            for strategy_name, kwargs in strategy_kwargs_list:
                try:
                    kwargs_without_strategy = {
                        k: v for k, v in kwargs.items() if k != "strategy"
                    }
                    tasks.append(
                        async_execute_strategy(
                            strategy_name,
                            strategies_executables,
                            **kwargs_without_strategy,
                        )
                    )
                except Exception as e:
                    self.context.logger.error(
                        f"Error setting up strategy {strategy_name}: {str(e)}"
                    )
                    results.append(
                        {"error": [f"Strategy setup error: {str(e)}"], "result": []}
                    )

            if tasks:
                try:
                    strategy_results = await asyncio.gather(
                        *tasks, return_exceptions=True
                    )
                    for result in strategy_results:
                        if isinstance(result, Exception):
                            results.append(
                                {
                                    "error": [
                                        f"Strategy execution error: {str(result)}"
                                    ],
                                    "result": [],
                                }
                            )
                        else:
                            results.append(result)
                except Exception as e:
                    self.context.logger.error(
                        f"Error running strategies in parallel: {str(e)}"
                    )
                    results.append(
                        {"error": [f"Parallel execution error: {str(e)}"], "result": []}
                    )

            return results

        # Main execution loop
        try:
            future = asyncio.ensure_future(run_all_strategies())
            while not future.done():
                yield  # Yield control to the agent loop
            results = future.result()

            for next_strategy, result in zip(strategies, results):
                tried_strategies.add(next_strategy)
                if not result:
                    self.context.logger.warning(
                        f"No result returned from strategy {next_strategy}"
                    )
                    continue

                if "error" in result:
                    errors = result.get("error", [])
                    if len(errors) > 0:
                        for error in errors:
                            self.context.logger.error(
                                f"Error in strategy {next_strategy}: {error}"
                            )
                        # Continue to next strategy if this one had errors
                        continue

                opportunities = result.get("result", [])
                if opportunities:
                    self.context.logger.info(
                        f"Opportunities found using {next_strategy} strategy"
                    )
                    valid_opportunities = []
                    for opportunity in opportunities:
                        try:
                            if isinstance(opportunity, dict):
                                self.context.logger.info(
                                    f"Opportunity: {opportunity.get('pool_address', 'N/A')}, "
                                    f"Chain: {opportunity.get('chain', 'N/A')}, "
                                    f"Token0: {opportunity.get('token0_symbol', 'N/A')}, "
                                    f"Token1: {opportunity.get('token1_symbol', 'N/A')}"
                                )
                                valid_opportunities.append(opportunity)
                            else:
                                self.context.logger.error(
                                    f"Invalid opportunity format from {next_strategy} strategy. "
                                    f"Expected dict, got {type(opportunity).__name__}: {opportunity}"
                                )
                        except Exception as e:
                            self.context.logger.error(
                                f"Error processing opportunity from {next_strategy}: {str(e)}"
                            )

                    if valid_opportunities:
                        self.trading_opportunities.extend(valid_opportunities)
                else:
                    self.context.logger.warning(
                        f"No opportunity found using {next_strategy} strategy"
                    )

        except Exception as e:
            self.context.logger.error(
                f"Critical error in strategy evaluation: {str(e)}"
            )
            # Ensure we don't lose the error state
            self.trading_opportunities = []

    def download_next_strategy(self) -> None:
        """Download the strategies one by one."""
        if self._inflight_strategy_req is not None:
            # there already is a req in flight
            return
        if len(self.shared_state.strategy_to_filehash) == 0:
            # no strategies pending to be fetched
            return

        for strategy, file_hash in self.shared_state.strategy_to_filehash.items():
            self.context.logger.info(f"Fetching {strategy} strategy...")
            ipfs_msg, message = self._build_ipfs_get_file_req(file_hash)
            self._inflight_strategy_req = strategy
            self.send_message(ipfs_msg, message, self._handle_get_strategy)
            return

    def get_returns_metrics_for_opportunity(
        self, position: Dict[str, Any], strategy: str
    ) -> Optional[Dict[str, Any]]:
        """Get and update metrics for the current pool ."""

        kwargs: Dict[str, Any] = self.params.strategies_kwargs.get(strategy, {})

        kwargs.update(
            {
                "strategy": strategy,
                "get_metrics": True,
                "position": position,
                "coingecko_api_key": self.coingecko.api_key,
                "chains": self.params.target_investment_chains,
                "apr_threshold": self.params.apr_threshold,
                "protocols": self.params.available_protocols,
                "chain_to_chain_id_mapping": self.params.chain_to_chain_id_mapping,
                "current_positions": self.current_positions,
                "whitelisted_assets": WHITELISTED_ASSETS,
                "coin_id_mapping": COIN_ID_MAPPING,
            }
        )

        # Execute the strategy to calculate metrics
        metrics = self.execute_strategy(**kwargs)
        if not metrics:
            return None
        elif "error" in metrics:
            self.context.logger.error(
                f"Failed to calculate metrics for the current position {position.get('pool_address')} : {metrics.get('error')}"
            )
            return None
        else:
            self.context.logger.info(
                f"Calculated position metrics for {position.get('pool_address')} : {metrics}"
            )
            return metrics

    def download_strategies(self) -> Generator:
        """Download all the strategies, if not yet downloaded."""
        while len(self.shared_state.strategy_to_filehash) > 0:
            self.download_next_strategy()
            yield from self.sleep(self.params.sleep_time)

    def execute_strategy(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Execute the strategy and return the results."""

        strategy = kwargs.pop("strategy", None)
        if strategy is None:
            self.context.logger.error(f"No trading strategy was given in {kwargs=}!")
            return None

        strategy = self.strategy_exec(strategy)
        if strategy is None:
            self.context.logger.error(f"No executable was found for {strategy=}!")
            return None

        strategy_exec, callable_method = strategy
        if callable_method in globals():
            del globals()[callable_method]

        exec(strategy_exec, globals())  # pylint: disable=W0122  # nosec
        method = globals().get(callable_method, None)
        if method is None:
            self.context.logger.error(
                f"No {callable_method!r} method was found in {strategy} executable."
            )
            return None

        return method(*args, **kwargs)

    def strategy_exec(self, strategy: str) -> Optional[Tuple[str, str]]:
        """Get the executable strategy file's content."""
        return self.shared_state.strategies_executables.get(strategy, None)

    def send_message(
        self, msg: Message, dialogue: Dialogue, callback: Callable
    ) -> None:
        """Send a message."""
        self.context.outbox.put_message(message=msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        self.shared_state.req_to_callback[nonce] = callback
        self.shared_state.in_flight_req = True

    def _handle_get_strategy(self, message: IpfsMessage, _: Dialogue) -> None:
        """Handle get strategy response."""
        strategy_req = self._inflight_strategy_req
        if strategy_req is None:
            self.context.logger.error(f"No strategy request to handle for {message=}.")
            return

        # store the executable and remove the hash from the mapping because we have downloaded it
        _component_yaml, strategy_exec, callable_method = ComponentPackageLoader.load(
            message.files
        )

        self.shared_state.strategies_executables[strategy_req] = (
            strategy_exec,
            callable_method,
        )
        self.shared_state.strategy_to_filehash.pop(strategy_req)
        self._inflight_strategy_req = None

    def _merge_duplicate_bridge_swap_actions(
        self, actions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify and merge duplicate bridge swap actions"""
        try:
            if not actions:
                return actions

            # Find all bridge swap actions
            bridge_swap_actions = [
                (i, action)
                for i, action in enumerate(actions)
                if action.get("action") == Action.FIND_BRIDGE_ROUTE.value
            ]

            if len(bridge_swap_actions) <= 1:
                return actions  # No duplicates possible with 0 or 1 bridge actions

            # Group bridge swap actions by their key attributes
            action_groups = {}
            for idx, action in bridge_swap_actions:
                try:
                    # Create a key based on the attributes that define a duplicate
                    from_chain = action.get("from_chain", "")
                    to_chain = action.get("to_chain", "")
                    from_token = action.get("from_token", "")
                    to_token = action.get("to_token", "")

                    # Generate a unique identifier for logging and debugging
                    action_id = (
                        f"{from_chain}_{to_chain}_{from_token[:8]}_{to_token[:8]}"
                    )
                    action["bridge_action_id"] = action_id

                    key = (from_chain, to_chain, from_token, to_token)

                    if key not in action_groups:
                        action_groups[key] = []
                    action_groups[key].append((idx, action))
                except Exception as e:
                    self.context.logger.error(
                        f"Error processing bridge swap action: {e}. Action: {action}"
                    )

            # If no groups have more than one action, there are no duplicates
            if all(len(group) <= 1 for group in action_groups.values()):
                return actions

            # Process groups with duplicates
            indices_to_remove = []
            for _key, group in action_groups.items():
                if len(group) > 1:
                    try:
                        # Keep the first action and update its funds_percentage
                        base_idx, base_action = group[0]
                        total_percentage = base_action.get("funds_percentage", 0)
                        action_ids = [base_action.get("bridge_action_id", "unknown")]

                        # Sum up percentages from duplicates and mark them for removal
                        for idx, action in group[1:]:
                            action_id = action.get("bridge_action_id", "unknown")
                            action_ids.append(action_id)
                            percentage = action.get("funds_percentage", 0)
                            total_percentage += percentage
                            indices_to_remove.append(idx)

                        # Update the base action with the combined percentage
                        actions[base_idx]["funds_percentage"] = total_percentage
                        actions[base_idx]["merged_from"] = action_ids

                        self.context.logger.info(
                            f"Merged {len(group)} duplicate bridge swap actions (IDs: {', '.join(action_ids)}) from "
                            f"{base_action.get('from_token_symbol', 'unknown')} on {base_action.get('from_chain', 'unknown')} "
                            f"to {base_action.get('to_token_symbol', 'unknown')} on {base_action.get('to_chain', 'unknown')} "
                            f"with combined percentage {total_percentage}"
                        )
                    except Exception as e:
                        self.context.logger.error(
                            f"Error merging duplicate bridge swap actions: {e}. Group: {group}"
                        )

            # Create a new list without the removed actions
            if indices_to_remove:
                return [
                    action
                    for i, action in enumerate(actions)
                    if i not in indices_to_remove
                ]

            return actions
        except Exception as e:
            self.context.logger.error(
                f"Error in _merge_duplicate_bridge_swap_actions: {e}"
            )
            # In case of error, return the original actions to avoid breaking the workflow
            return actions

    def _handle_velodrome_token_allocation(
        self, actions, enter_pool_action, available_tokens
    ) -> List[Dict[str, Any]]:
        """Handle Velodrome positions that require 100% allocation to one token."""
        # Only process if it's a Velodrome position with token requirements
        self.context.logger.info(f"action inside the velodrome: {actions}")
        if (
            enter_pool_action.get("dex_type") == "velodrome"
            and "token_requirements" in enter_pool_action
        ):
            token_requirements = enter_pool_action.get("token_requirements", {})

            # Extract token ratios, handling potential NumPy types
            try:
                overall_token0_ratio = float(
                    token_requirements.get("overall_token0_ratio", 0.5)
                )
                overall_token1_ratio = float(
                    token_requirements.get("overall_token1_ratio", 0.5)
                )
            except (TypeError, ValueError):
                # Fall back to checking recommendation text
                recommendation = token_requirements.get("recommendation", "")
                if "100% token0" in recommendation:
                    overall_token0_ratio = 1.0
                    overall_token1_ratio = 0.0
                elif "100% token1" in recommendation:
                    overall_token0_ratio = 0.0
                    overall_token1_ratio = 1.0
                else:
                    overall_token0_ratio = 0.5
                    overall_token1_ratio = 0.5

            self.context.logger.info(
                f"Velodrome position requirements: token0_ratio={overall_token0_ratio}, token1_ratio={overall_token1_ratio}"
            )

            # Check if all funds should go to one token (using 0.99 threshold to handle floating point)
            if (overall_token0_ratio >= 0.99 and overall_token1_ratio <= 0.01) or (
                overall_token0_ratio <= 0.01 and overall_token1_ratio >= 0.99
            ):
                # Determine which token gets 100%
                is_token0_full = overall_token0_ratio >= 0.99
                target_token = (
                    enter_pool_action.get("token0")
                    if is_token0_full
                    else enter_pool_action.get("token1")
                )
                target_symbol = (
                    enter_pool_action.get("token0_symbol")
                    if is_token0_full
                    else enter_pool_action.get("token1_symbol")
                )

                self.context.logger.info(
                    f"Extreme allocation detected: 100% to {target_symbol}"
                )

                # Track if we found any bridge routes to modify
                bridge_routes_found = False

                # Check and modify existing FindBridgeRoute actions
                for action in actions:
                    if action.get("action") == "FindBridgeRoute" and action.get(
                        "to_chain"
                    ) == enter_pool_action.get("chain"):
                        bridge_routes_found = True

                        # Redirect all bridge routes to the target token
                        if action.get("to_token") != target_token:
                            self.context.logger.info(
                                f"Redirecting bridge route to {target_symbol}"
                            )
                            action["to_token"] = target_token
                            action["to_token_symbol"] = target_symbol

                # If no FindBridgeRoute actions were found, add one
                if not bridge_routes_found and available_tokens:
                    self.context.logger.info("No bridge routes found, adding a new one")

                    source_token = None
                    for token in available_tokens:
                        if token.get("token") != target_token:
                            source_token = token
                            break

                    if source_token:
                        # Use the first available token as source
                        new_bridge_route = {
                            "action": "FindBridgeRoute",
                            "from_chain": source_token.get("chain"),
                            "to_chain": enter_pool_action.get("chain"),
                            "from_token": source_token.get("token"),
                            "from_token_symbol": source_token.get("token_symbol"),
                            "to_token": target_token,
                            "to_token_symbol": target_symbol,
                            "funds_percentage": 1.0,  # Use 100% allocation
                        }

                        self.context.logger.info(
                            f"Added new bridge route: {source_token.get('token_symbol')} -> {target_symbol}"
                        )

                        # Add to the beginning of actions
                        actions.insert(0, new_bridge_route)

        self.context.logger.info(f"action at the end of velodrome: {actions}")
        return actions

    def _apply_investment_cap_to_actions(
        self, actions
    ) -> Generator[None, None, List[Dict[str, Any]]]:
        """Apply investment caps to actions based on current positions."""
        if not self.current_positions:
            return actions

        self.context.logger.info(f"action inside the investment: {actions}")

        yield from self.sleep(20)
        self.context.logger.info(
            f"self.current_positions in order transaction: {self.current_positions}"
        )

        # Calculate total invested amount
        invested_amount = 0
        invested_positions = False

        for position in self.current_positions:
            if position.get("status") == "open":
                self.context.logger.info("Calculating value for open position")
                invested_positions = True
                retries = 3
                delay = 10  # initial delay in seconds
                V_initial = None

                # Retry loop for calculating position value
                while retries > 0:
                    V_initial = yield from self.calculate_initial_investment_value(
                        position
                    )
                    if V_initial is not None:
                        break
                    else:
                        self.context.logger.warning(
                            "V_initial is None (possible rate limit). Retrying after delay..."
                        )
                        yield from self.sleep(delay)
                        retries -= 1
                        delay *= 2  # exponential backoff

                self.context.logger.info(f"V_initial amount: {V_initial}")
                if V_initial:
                    invested_amount += V_initial
                    self.context.logger.info(
                        f"Accumulated invested_amount: {invested_amount}"
                    )
                yield from self.sleep(10)  # Additional sleep if needed

        self.context.logger.info(f"Final invested_amount: {invested_amount}")

        # Define investment thresholds
        THRESHOLD_INVESTED_AMOUNT = 950
        global_cap = 1000

        # Apply investment cap logic
        if invested_amount >= THRESHOLD_INVESTED_AMOUNT or (
            invested_amount == 0 and invested_positions
        ):
            self.context.logger.info("Investment threshold reached, limiting actions")
            exit_pool_found = False

            # Check if there's an exit pool action
            for action in actions[:]:
                if action.get("action") == "ExitPool":
                    exit_pool_found = True
                    # Remove any enter pool actions if we're exiting
                    for sub_item in actions[:]:
                        if sub_item.get("action") == "EnterPool":
                            actions.remove(sub_item)
                    break

            # If no exit action and we're over threshold, clear all actions
            if not exit_pool_found:
                self.context.logger.info(
                    "No exit pool action found and investment threshold reached. Clearing all actions."
                )
                actions = []  # Clear actions when invested amount exceeds threshold

        # If we have some investment but under threshold, adjust enter amounts
        elif invested_amount > 0:
            self.context.logger.info(
                "Under investment threshold, adjusting enter pool amounts"
            )
            for action in actions:
                if action.get("action") == "EnterPool":
                    action["invested_amount"] = global_cap - invested_amount
                    self.context.logger.info(
                        f"Set invested_amount to {action['invested_amount']} for enter pool action"
                    )

        self.context.logger.info(f"action inside the investment at the end: {actions}")
        return actions

    def get_order_of_transactions(
        self,
    ) -> Generator[None, None, Optional[List[Dict[str, Any]]]]:
        """Get the order of transactions to perform based on the current pool status and token balances."""
        actions = []
        tokens = []
        # Process rewards
        # if self._can_claim_rewards():
        #     yield from self._process_rewards(actions)
        if not self.selected_opportunities:
            return actions

        for opportunity in self.selected_opportunities:
            if opportunity.get("dex_type") == "velodrome":
                result2 = yield from self.get_velodrome_position_requirements()
                self.context.logger.info(f"result2: {result2}")

        # Prepare tokens for exit or investment
        available_tokens = yield from self._prepare_tokens_for_investment()
        if available_tokens is None:
            return actions
        tokens.extend(available_tokens)

        if self.position_to_exit:
            dex_type = self.position_to_exit.get("dex_type")
            num_of_tokens_required = 1 if dex_type == DexType.STURDY.value else 2
            exit_pool_action = self._build_exit_pool_action(
                tokens, num_of_tokens_required
            )
            if not exit_pool_action:
                self.context.logger.error("Error building exit pool action")
                return None
            actions.append(exit_pool_action)
        # Build actions based on selected opportunities
        for opportunity in self.selected_opportunities:
            bridge_swap_actions = self._build_bridge_swap_actions(opportunity, tokens)
            if bridge_swap_actions is None:
                self.context.logger.info("Error preparing bridge swap actions")
                return None
            if bridge_swap_actions:
                actions.extend(bridge_swap_actions)

            enter_pool_action = self._build_enter_pool_action(opportunity)
            if not enter_pool_action:
                self.context.logger.error("Error building enter pool action")
                return None
            actions.append(enter_pool_action)

        # Check for Velodrome positions with 100% allocation to one token
        if (
            enter_pool_action.get("dex_type") == "velodrome"
            and "token_requirements" in enter_pool_action
        ):
            # After building all enter_pool_action and bridge_swap_actions
            actions = self._handle_velodrome_token_allocation(
                actions, enter_pool_action, available_tokens
            )
            self.context.logger.info(
                f"action after velodrome into the function: {actions}"
            )

        if self.current_positions:
            self.context.logger.info(
                f"action before the investment into the function: {actions}"
            )
            actions = yield from self._apply_investment_cap_to_actions(actions)
            self.context.logger.info(f"action into the function: {actions}")

        try:
            # Merge duplicate bridge swap actions
            merged_actions = self._merge_duplicate_bridge_swap_actions(actions)
            self.context.logger.info(f"merged_actions: {merged_actions}")
            return merged_actions
        except Exception as e:
            self.context.logger.error(f"Error while merging bridge swap actions: {e}")
            # Return original actions if merging fails to avoid breaking the workflow
            return actions

    def _process_rewards(self, actions: List[Dict[str, Any]]) -> Generator:
        """Process reward claims and add actions."""
        allowed_chains = self.params.target_investment_chains
        for chain in allowed_chains:
            chain_id = self.params.chain_to_chain_id_mapping.get(chain)
            safe_address = self.params.safe_contract_addresses.get(chain)
            rewards = yield from self._get_rewards(chain_id, safe_address)
            if not rewards:
                continue
            action = self._build_claim_reward_action(rewards, chain)
            actions.append(action)

    def _prepare_tokens_for_investment(
        self,
    ) -> Generator[None, None, Optional[List[Dict[str, Any]]]]:
        """Prepare tokens for exit or investment, and append exit actions if needed."""
        tokens = []

        if self.position_to_exit:
            dex_type = self.position_to_exit.get("dex_type")
            num_of_tokens_required = 1 if dex_type == DexType.STURDY.value else 2
            tokens = self._build_tokens_from_position(
                self.position_to_exit, num_of_tokens_required
            )
            if not tokens or len(tokens) < num_of_tokens_required:
                self.context.logger.error(
                    f"{num_of_tokens_required} tokens required to exit pool, provided: {tokens}"
                )
                return None

        available_tokens = yield from self._get_available_tokens()
        if available_tokens:
            tokens.extend(available_tokens)

        if not tokens:
            self.context.logger.error("No tokens available for investment")
            return None  # Not enough tokens

        return tokens

    def _build_tokens_from_position(
        self, position: Dict[str, Any], num_tokens: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Build token entries from position based on number of tokens required."""
        chain = position.get("chain")
        if num_tokens == 1:
            return [
                {
                    "chain": chain,
                    "token": position.get("token0"),
                    "token_symbol": position.get("token0_symbol"),
                }
            ]
        elif num_tokens == 2:
            return [
                {
                    "chain": chain,
                    "token": position.get("token0"),
                    "token_symbol": position.get("token0_symbol"),
                },
                {
                    "chain": chain,
                    "token": position.get("token1"),
                    "token_symbol": position.get("token1_symbol"),
                },
            ]
        else:
            return None

    def _get_available_tokens(
        self,
    ) -> Generator[None, None, Optional[List[Dict[str, Any]]]]:
        """Get tokens with the highest balances."""
        token_balances = []

        for position in self.synchronized_data.positions:
            chain = position.get("chain")
            for asset in position.get("assets", []):
                asset_address = asset.get("address")
                asset_symbol = asset.get("asset_symbol")
                balance = asset.get("balance", 0)

                if chain and asset_address:
                    investable_balance = yield from self._get_investable_balance(
                        chain, asset_address, balance
                    )

                    if investable_balance > 0:
                        token_balances.append(
                            {
                                "chain": chain,
                                "token": asset_address,
                                "token_symbol": asset_symbol,
                                "balance": investable_balance,
                            }
                        )

        # Sort tokens by balance in descending order
        token_balances.sort(key=lambda x: x["balance"], reverse=True)
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
        self.context.logger.info(f"Available token balances: {token_balances}")

        self.context.logger.info(
            f"Filtering tokens with a minimum investment value of: {self.params.min_investment_amount}"
        )
        token_balances = [
            token
            for token in token_balances
            if token["value"] >= self.params.min_investment_amount
        ]

        self.context.logger.info(
            f"Tokens selected for bridging/swapping: {token_balances}"
        )
        return token_balances

    def _get_investable_balance(
        self, chain: str, token_address: str, total_balance: int
    ) -> Generator[None, None, int]:
        """Get the portion of token balance available for investment (total - reserved rewards)"""

        # Check if this is a reward token
        reward_addresses = REWARD_TOKEN_ADDRESSES.get(chain, {})
        if to_checksum_address(token_address) not in reward_addresses.keys():
            return total_balance

        # Check if it's also whitelisted (like OLAS)
        whitelisted_addresses = WHITELISTED_ASSETS.get(chain, {})
        if to_checksum_address(token_address) not in whitelisted_addresses.keys():
            self.context.logger.info(
                f"Pure reward token {token_address} on {chain} - not for investment"
            )
            return 0  # Pure reward token, not for investment

        # This is a whitelisted reward token - subtract accumulated rewards
        accumulated_rewards = yield from self.get_accumulated_rewards_for_token(
            chain, token_address
        )
        investable_balance = max(0, total_balance - accumulated_rewards)

        self.context.logger.info(
            f"Token {token_address} on {chain}: "
            f"Total={total_balance}, Reserved={accumulated_rewards}, Investable={investable_balance}"
        )

        return investable_balance

    def _build_exit_pool_action(
        self, tokens: List[Dict[str, Any]], num_of_tokens_required: int
    ) -> Optional[Dict[str, Any]]:
        """Build action for exiting the current pool."""
        if not self.position_to_exit:
            self.context.logger.error("No pool present")
            return None

        if len(tokens) < num_of_tokens_required:
            self.context.logger.error(
                f"Insufficient tokens provided for exit action. Required atleast {num_of_tokens_required}, provided: {tokens}"
            )
            return None

        exit_pool_action = {
            "action": (
                Action.WITHDRAW.value
                if self.position_to_exit.get("dex_type") == DexType.STURDY.value
                else Action.EXIT_POOL.value
            ),
            "dex_type": self.position_to_exit.get("dex_type"),
            "chain": self.position_to_exit.get("chain"),
            "assets": [token.get("token") for token in tokens],
            "pool_address": self.position_to_exit.get("pool_address"),
            "pool_type": self.position_to_exit.get("pool_type"),
            "is_stable": self.position_to_exit.get("is_stable"),
            "is_cl_pool": self.position_to_exit.get("is_cl_pool"),
        }

        # Handle Velodrome CL pools with multiple positions
        if (
            self.position_to_exit.get("dex_type") == DexType.VELODROME.value
            and self.position_to_exit.get("is_cl_pool")
            and "positions" in self.position_to_exit
        ):
            # Extract token IDs from all positions
            token_ids = [
                pos["token_id"] for pos in self.position_to_exit.get("positions", [])
            ]
            liquidities = [
                pos["liquidity"] for pos in self.position_to_exit.get("positions", [])
            ]
            if token_ids and liquidities:
                self.context.logger.info(
                    f"Exiting Velodrome CL pool with {len(token_ids)} positions. "
                    f"Token IDs: {token_ids}"
                )
                exit_pool_action["token_ids"] = token_ids
                exit_pool_action["liquidities"] = liquidities
        # For single position case (backward compatibility)
        elif "token_id" in self.position_to_exit:
            exit_pool_action["token_id"] = self.position_to_exit.get("token_id")
            exit_pool_action["liquidity"] = self.position_to_exit.get("liquidity")

        return exit_pool_action

    def _get_required_tokens(
        self, opportunity: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """Get the list of required tokens for the opportunity."""
        required_tokens = []
        dest_token0_address = opportunity.get("token0")
        dest_token0_symbol = opportunity.get("token0_symbol")

        if dest_token0_address and dest_token0_symbol:
            required_tokens.append((dest_token0_address, dest_token0_symbol))

        # For non-STURDY dex types, we need two tokens
        if opportunity.get("dex_type") != DexType.STURDY.value:
            dest_token1_address = opportunity.get("token1")
            dest_token1_symbol = opportunity.get("token1_symbol")

            if dest_token1_address and dest_token1_symbol:
                required_tokens.append((dest_token1_address, dest_token1_symbol))

        return required_tokens

    def _group_tokens_by_chain(
        self, tokens: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group tokens by chain."""
        tokens_by_chain = {}
        for token in tokens:
            chain = token.get("chain")
            if chain not in tokens_by_chain:
                tokens_by_chain[chain] = []
            tokens_by_chain[chain].append(token)
        return tokens_by_chain

    def _identify_missing_tokens(
        self,
        required_tokens: List[Tuple[str, str]],
        available_tokens: Dict[str, Dict[str, Any]],
        dest_chain: str,
    ) -> List[Tuple[str, str]]:
        """Identify which tokens we need but don't have on the destination chain."""
        tokens_we_need = []
        for req_token_addr, req_token_symbol in required_tokens:
            if req_token_addr in available_tokens:
                self.context.logger.info(
                    f"Token {req_token_symbol} ({req_token_addr}) already available on {dest_chain}"
                )
            else:
                tokens_we_need.append((req_token_addr, req_token_symbol))
        return tokens_we_need

    def _handle_all_tokens_available(
        self,
        tokens: List[Dict[str, Any]],
        required_tokens: List[Tuple[str, str]],
        dest_chain: str,
        relative_funds_percentage: float,
    ) -> List[Dict[str, Any]]:
        """Handle the case where we have all required tokens on the destination chain."""
        bridge_swap_actions = []
        if required_tokens:
            # Get tokens from other chains
            other_chain_tokens = [
                token for token in tokens if token.get("chain") != dest_chain
            ]

            # If we have tokens from other chains and required tokens
            if other_chain_tokens and required_tokens:
                # Distribute tokens from other chains evenly among all required tokens
                for idx, token in enumerate(other_chain_tokens):
                    # Get the destination token for this source token
                    dest_token_address, dest_token_symbol = required_tokens[
                        idx % len(required_tokens)
                    ]

                    # Calculate percentage based on total required tokens
                    token_percentage = relative_funds_percentage / len(required_tokens)

                    # Add bridge action
                    self._add_bridge_swap_action(
                        bridge_swap_actions,
                        token,
                        dest_chain,
                        dest_token_address,
                        dest_token_symbol,
                        token_percentage,
                    )

        return bridge_swap_actions

    def _handle_some_tokens_available(
        self,
        tokens: List[Dict[str, Any]],
        required_tokens: List[Tuple[str, str]],
        tokens_we_need: List[Tuple[str, str]],
        dest_chain: str,
        relative_funds_percentage: float,
    ) -> List[Dict[str, Any]]:
        """Handle the case where we have some but not all required tokens."""
        bridge_swap_actions = []

        # First, handle tokens from other chains
        other_chain_tokens = [
            token for token in tokens if token.get("chain") != dest_chain
        ]
        for idx, token in enumerate(other_chain_tokens):
            dest_token_address, dest_token_symbol = tokens_we_need[
                idx % len(tokens_we_need)
            ]
            self._add_bridge_swap_action(
                bridge_swap_actions,
                token,
                dest_chain,
                dest_token_address,
                dest_token_symbol,
                relative_funds_percentage,
            )

        # Then, handle tokens on the destination chain that need to be swapped
        dest_chain_tokens = [
            token for token in tokens if token.get("chain") == dest_chain
        ]
        for token in dest_chain_tokens:
            # Skip if this token is already one of the required tokens
            if any(token.get("token") == req_token for req_token, _ in required_tokens):
                continue

            # Swap to the first missing token
            dest_token_address, dest_token_symbol = tokens_we_need[0]
            self._add_bridge_swap_action(
                bridge_swap_actions,
                token,
                dest_chain,
                dest_token_address,
                dest_token_symbol,
                relative_funds_percentage,
            )

        # If no actions created yet, use available required tokens to get missing ones
        if not bridge_swap_actions:
            available_tokens_on_dest_list = [
                token
                for token in dest_chain_tokens
                if any(
                    token.get("token") == req_token for req_token, _ in required_tokens
                )
            ]

            if available_tokens_on_dest_list and tokens_we_need:
                source_token = available_tokens_on_dest_list[0]
                for dest_token_address, dest_token_symbol in tokens_we_need:
                    # Skip if this is the same token (no need to swap)
                    if source_token.get("token") == dest_token_address:
                        continue

                    # Calculate percentage based on total required tokens
                    token_percentage = relative_funds_percentage / len(required_tokens)

                    self._add_bridge_swap_action(
                        bridge_swap_actions,
                        source_token,
                        dest_chain,
                        dest_token_address,
                        dest_token_symbol,
                        token_percentage,
                    )

        return bridge_swap_actions

    def _handle_all_tokens_needed(
        self,
        tokens: List[Dict[str, Any]],
        required_tokens: List[Tuple[str, str]],
        dest_chain: str,
        relative_funds_percentage: float,
    ) -> List[Dict[str, Any]]:
        """Handle the case where we need all tokens."""
        bridge_swap_actions = []

        # Handle single source token case
        if len(tokens) == 1:
            token = tokens[0]
            for dest_token_address, dest_token_symbol in required_tokens:
                # Skip if same token on same chain
                if (
                    token.get("chain") == dest_chain
                    and token.get("token") == dest_token_address
                ):
                    continue

                token_percentage = relative_funds_percentage / len(required_tokens)

                self._add_bridge_swap_action(
                    bridge_swap_actions,
                    token,
                    dest_chain,
                    dest_token_address,
                    dest_token_symbol,
                    token_percentage,
                )
        else:
            # Multiple source tokens case
            tokens.sort(key=lambda x: x["token"])
            dest_tokens = sorted(required_tokens, key=lambda x: x[0])

            for idx, token in enumerate(tokens):
                dest_token_address, dest_token_symbol = dest_tokens[
                    idx % len(dest_tokens)
                ]

                # Skip if same token on same chain
                if (
                    token.get("chain") == dest_chain
                    and token.get("token") == dest_token_address
                ):
                    continue

                token_percentage = relative_funds_percentage / len(dest_tokens)

                self._add_bridge_swap_action(
                    bridge_swap_actions,
                    token,
                    dest_chain,
                    dest_token_address,
                    dest_token_symbol,
                    token_percentage,
                )

        return bridge_swap_actions

    def _build_bridge_swap_actions(
        self, opportunity: Dict[str, Any], tokens: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build bridge and swap actions for the given tokens."""
        # Validate input
        if not opportunity:
            self.context.logger.error("No pool present.")
            return None

        # Extract key opportunity details
        dest_chain = opportunity.get("chain")
        relative_funds_percentage = opportunity.get("relative_funds_percentage")

        if not dest_chain or not opportunity.get("token0"):
            self.context.logger.error(f"Incomplete data in opportunity {opportunity}")
            return None

        # Get required tokens for this opportunity
        required_tokens = self._get_required_tokens(opportunity)
        if not required_tokens:
            self.context.logger.error("No required tokens identified")
            return None

        # Group tokens by chain and identify what we have/need
        tokens_by_chain = self._group_tokens_by_chain(tokens)
        dest_chain_tokens = tokens_by_chain.get(dest_chain, [])
        available_tokens_on_dest = {
            token.get("token"): token for token in dest_chain_tokens
        }
        tokens_we_need = self._identify_missing_tokens(
            required_tokens, available_tokens_on_dest, dest_chain
        )

        # Handle different scenarios based on what tokens we have/need
        if not tokens_we_need:
            # We have all required tokens, just bridge from other chains
            return self._handle_all_tokens_available(
                tokens, required_tokens, dest_chain, relative_funds_percentage
            )
        elif len(tokens_we_need) < len(required_tokens):
            # We have some tokens but not all
            return self._handle_some_tokens_available(
                tokens,
                required_tokens,
                tokens_we_need,
                dest_chain,
                relative_funds_percentage,
            )
        else:
            # We need all tokens
            return self._handle_all_tokens_needed(
                tokens, required_tokens, dest_chain, relative_funds_percentage
            )

    def _add_bridge_swap_action(
        self,
        actions: List[Dict[str, Any]],
        token: Dict[str, Any],
        dest_chain: str,
        dest_token_address: str,
        dest_token_symbol: str,
        relative_funds_percentage: float,
    ) -> None:
        """Helper function to add a bridge swap action."""
        source_token_chain = token.get("chain")
        source_token_address = token.get("token")
        source_token_symbol = token.get("token_symbol")

        if (
            not source_token_chain
            or not source_token_address
            or not source_token_symbol
        ):
            self.context.logger.error(f"Incomplete data in tokens {token}")
            return

        # Only add bridge/swap action if:
        # 1. We need to bridge to a different chain, OR
        # 2. We need to swap to a different token on the same chain
        if source_token_chain != dest_chain:
            # Need to bridge to different chain
            self.context.logger.info(
                f"Adding bridge action from {source_token_symbol} on {source_token_chain} "
                f"to {dest_token_symbol} on {dest_chain}"
            )
            actions.append(
                {
                    "action": Action.FIND_BRIDGE_ROUTE.value,
                    "from_chain": source_token_chain,
                    "to_chain": dest_chain,
                    "from_token": source_token_address,
                    "from_token_symbol": source_token_symbol,
                    "to_token": dest_token_address,
                    "to_token_symbol": dest_token_symbol,
                    "funds_percentage": relative_funds_percentage,
                }
            )
        elif source_token_address != dest_token_address:
            # Same chain but different token, need to swap
            self.context.logger.info(
                f"Adding swap action from {source_token_symbol} to {dest_token_symbol} "
                f"on chain {dest_chain}"
            )
            actions.append(
                {
                    "action": Action.FIND_BRIDGE_ROUTE.value,
                    "from_chain": source_token_chain,
                    "to_chain": dest_chain,
                    "from_token": source_token_address,
                    "from_token_symbol": source_token_symbol,
                    "to_token": dest_token_address,
                    "to_token_symbol": dest_token_symbol,
                    "funds_percentage": relative_funds_percentage,
                }
            )
        else:
            # Same token on same chain, no action needed
            self.context.logger.info(
                f"No bridge/swap needed for {source_token_symbol} on {source_token_chain} "
                f"as it's already the required token"
            )

    def _build_enter_pool_action(self, opportunity) -> Dict[str, Any]:
        """Build action for entering the pool with the highest APR."""
        if not opportunity:
            self.context.logger.error("No pool present.")
            return None

        action_details = {
            **opportunity,
            "action": (
                Action.DEPOSIT.value
                if opportunity.get("dex_type") == DexType.STURDY.value
                else Action.ENTER_POOL.value
            ),
        }
        return action_details

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

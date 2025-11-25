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
import math
import os
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
from packages.valory.skills.liquidity_trader_abci.states.base import Event
from packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy import (
    EvaluateStrategyRound,
)


MIN_SWAP_VALUE_USD = 0.5


class EvaluateStrategyBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that finds the opportunity and builds actions."""

    matching_round: Type[AbstractRound] = EvaluateStrategyRound
    selected_opportunities = None
    position_to_exit = None
    trading_opportunities = []
    positions_eligible_for_exit = []

    def async_act(self) -> Generator:
        """Execute the behaviour's async action."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            # Check if investing is paused due to withdrawal (read from KV store)
            investing_paused = yield from self._read_investing_paused()
            if investing_paused:
                self.context.logger.info(
                    "Investing paused due to withdrawal request. Transitioning to WithdrawFunds round."
                )
                payload = EvaluateStrategyPayload(
                    sender=self.context.agent_address,
                    actions=json.dumps(
                        {
                            "event": Event.WITHDRAWAL_INITIATED.value,
                            "updates": {},
                        },
                        sort_keys=True,
                        ensure_ascii=True,
                    ),
                )
                yield from self.send_a2a_transaction(payload)
                yield from self.wait_until_round_end()
                self.set_done()
                return

            # Check minimum hold period
            # Check if no current positions and uninvested ETH, prepare swap to USDC
            actions = yield from self.check_and_prepare_non_whitelisted_swaps()
            if actions:
                yield from self.send_actions(actions)

            (
                should_proceed,
                positions_eligible_for_exit,
            ) = yield from self._apply_tip_filters_to_exit_decisions()
            if not should_proceed:
                yield from self.send_actions()
                return

            if positions_eligible_for_exit:
                self.positions_eligible_for_exit = positions_eligible_for_exit

            # Check for funds
            are_funds_available = self.check_funds()
            if not are_funds_available:
                yield from self.send_actions()
                return

            # Check if we have a cached Velodrome CL pool opportunity to reuse
            cached_opportunity_result = (
                yield from self._check_and_use_cached_cl_opportunity()
            )

            if cached_opportunity_result:
                # We have a valid cached opportunity, use it directly
                self.context.logger.info(
                    "Using cached Velodrome CL pool opportunity, skipping opportunity fetching and strategy evaluation"
                )
                actions = cached_opportunity_result
            else:
                # No valid cache, proceed with normal flow
                # Fetch trading opportunities
                yield from self.fetch_all_trading_opportunities()

                # Execute strategy and prepare actions
                actions = yield from self.prepare_strategy_actions()

                # Push opportunity data to MirrorDB
                yield from self._push_opportunity_metrics_to_mirrordb()

            # Send final actions
            yield from self.send_actions(actions)

    def send_actions(self, actions: Optional[List[Any]] = None) -> Generator:
        """Send actions and complete the round."""
        if actions is None:
            actions = []

        payload = EvaluateStrategyPayload(
            sender=self.context.agent_address,
            actions=json.dumps(actions, ensure_ascii=True),
        )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

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

    def calculate_velodrome_token_ratios(
        self, validated_data, chain=None
    ) -> Generator[None, None, Dict[str, Any]]:
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

            # Get sqrt ratios at tick bounds using Velodrome Slipstream Helper contract
            sqrt_ratio_a_x96 = yield from self.get_velodrome_sqrt_ratio_at_tick(
                chain, tick_lower
            )
            sqrt_ratio_b_x96 = yield from self.get_velodrome_sqrt_ratio_at_tick(
                chain, tick_upper
            )

            if (
                "sqrt_price_x96" in validated_data
                and validated_data["sqrt_price_x96"] is not None
            ):
                sqrt_price_x96 = validated_data["sqrt_price_x96"]
            else:
                # Fallback: convert current price to sqrt_price_x96 format (less accurate)
                sqrt_price_x96 = int(math.sqrt(current_price) * (2**96))
                self.context.logger.warning(
                    f"Using converted sqrt_price_x96 from current_price: {sqrt_price_x96}"
                )

            # Calculate token ratios using Uniswap V3 math
            try:
                if sqrt_price_x96 < sqrt_ratio_a_x96:
                    # Price below range - need 100% token0
                    token0_ratio = 1.0
                    token1_ratio = 0.0
                    status = "BELOW_RANGE"
                elif sqrt_price_x96 > sqrt_ratio_b_x96:
                    # Price above range - need 100% token1
                    token0_ratio = 0.0
                    token1_ratio = 1.0
                    status = "ABOVE_RANGE"
                else:
                    # Price in range
                    # Use a unit of liquidity to determine ratios
                    unit_liquidity = 10**30
                    # Calculate amounts for this liquidity using Velodrome Slipstream Helper contract
                    (
                        amount0,
                        amount1,
                    ) = yield from self.get_velodrome_amounts_for_liquidity(
                        chain,
                        sqrt_price_x96,
                        sqrt_ratio_a_x96,
                        sqrt_ratio_b_x96,
                        unit_liquidity,
                    )
                    # Calculate from actual amounts
                    total_amount = amount0 + amount1
                    if total_amount > 0:
                        # Round to 5 decimal places for precision
                        token0_ratio = round(amount0 / total_amount, 5)
                        token1_ratio = round(amount1 / total_amount, 5)
                    else:
                        # Fallback if calculation fails
                        token0_ratio = 0.5
                        token1_ratio = 0.5
                    status = "IN_RANGE"

            except Exception as e:
                warnings.append(
                    f"Error calculating ratios for band [{tick_lower}, {tick_upper}]: {str(e)}"
                )
                # Fallback: use price-based estimation
                if sqrt_price_x96 < sqrt_ratio_a_x96:
                    token0_ratio = 1.0
                    token1_ratio = 0.0
                elif sqrt_price_x96 > sqrt_ratio_b_x96:
                    token0_ratio = 0.0
                    token1_ratio = 1.0
                else:
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

        # Generate recommendations based on individual band requirements
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
                # Calculate aggregate ratios for recommendation
                (
                    agg_token0_ratio,
                    agg_token1_ratio,
                ) = self._calculate_aggregate_token_ratios(position_requirements)
                recommendation = f"Provide {agg_token0_ratio*100:.2f}% token0, {agg_token1_ratio*100:.2f}% token1 for all positions"
        else:
            recommendation = (
                f"Mixed position requirements across {len(position_requirements)} bands"
            )

        # Log any warnings
        for warning in warnings:
            self.context.logger.warning(warning)

        self.context.logger.info(
            f"Position analysis complete - Current tick: {current_tick}, "
            f"Bands: {len(position_requirements)}"
        )

        return {
            "position_requirements": position_requirements,
            "current_price": current_price,
            "current_tick": current_tick,
            "recommendation": recommendation,
            "warnings": warnings,
        }

    def calculate_velodrome_cl_token_requirements(
        self, tick_bands, current_price, tick_spacing=1, sqrt_price_x96=None, chain=None
    ) -> Generator[None, None, Dict[str, Any]]:
        """Determines token requirements for Velodrome CL positions based on current price."""
        # Step 1: Validate and prepare inputs
        validated_data = self.validate_and_prepare_velodrome_inputs(
            tick_bands, current_price, tick_spacing
        )

        if not validated_data:
            return None

        # Add sqrt_price_x96 to validated_data if provided
        if sqrt_price_x96 is not None:
            validated_data["sqrt_price_x96"] = sqrt_price_x96

        # Step 2: Calculate token ratios and generate recommendations
        velodrome_token_ratios = yield from self.calculate_velodrome_token_ratios(
            validated_data, chain
        )
        return velodrome_token_ratios

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
                    chain = opportunity["chain"]
                    pool_address = opportunity["pool_address"]

                    self.context.logger.info(
                        f"Calculating fresh data for Velodrome CL pool {pool_address} on {chain}"
                    )

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
                    tick_bands = opportunity.get("tick_bands", [])
                    self.context.logger.info(f"tick_bands : {tick_bands}")
                    if not tick_bands:
                        self.context.logger.error(
                            f"Failed to calculate tick bands for pool {pool_address}"
                        )
                        continue

                    # Extract percent_in_bounds from the first position (all positions have the same value)
                    percent_in_bounds = (
                        tick_bands[0].get("percent_in_bounds", 0.0)
                        if tick_bands
                        else 0.0
                    )
                    self.context.logger.info(f"percent_in_bounds : {percent_in_bounds}")

                    current_price = yield from pool._get_current_pool_price(
                        self, pool_address, chain
                    )
                    if current_price is None:
                        self.context.logger.error(
                            f"Failed to get current price for pool {pool_address}"
                        )
                        continue

                    # Get sqrt_price_x96 for accurate Uniswap V3 math calculations
                    sqrt_price_x96 = yield from pool._get_sqrt_price_x96(
                        self, chain, pool_address
                    )
                    if sqrt_price_x96 is None:
                        self.context.logger.warning(
                            f"Failed to get sqrt_price_x96 for pool {pool_address}, will use converted value"
                        )

                    # Extract EMA and std_dev from the first position (all positions have the same values)
                    ema = tick_bands[0].get("ema") if tick_bands else None
                    std_dev = tick_bands[0].get("std_dev") if tick_bands else None
                    current_ema = (
                        tick_bands[0].get("current_ema") if tick_bands else None
                    )
                    current_std_dev = (
                        tick_bands[0].get("current_std_dev") if tick_bands else None
                    )
                    band_multipliers = (
                        tick_bands[0].get("band_multipliers") if tick_bands else None
                    )

                    # Calculate token requirements to get ratios and current_tick
                    requirements = (
                        yield from self.calculate_velodrome_cl_token_requirements(
                            tick_bands,
                            current_price,
                            tick_spacing,
                            sqrt_price_x96,
                            chain,
                        )
                    )

                    if not requirements:
                        self.context.logger.error(
                            "Failed to calculate token requirements"
                        )
                        continue

                    token0 = opportunity.get("token0")
                    token1 = opportunity.get("token1")
                    token0_symbol = opportunity.get("token0_symbol", "token0")
                    token1_symbol = opportunity.get("token1_symbol", "token1")

                    cache_kwargs = {
                        "chain": chain,
                        "pool_address": pool_address,
                        "tick_spacing": tick_spacing,
                        "tick_bands": tick_bands,
                        "current_price": current_price,
                        "current_tick": requirements.get("current_tick"),
                        "percent_in_bounds": percent_in_bounds,
                        "token0": token0,
                        "token1": token1,
                        "token0_symbol": token0_symbol,
                        "token1_symbol": token1_symbol,
                        "token_requirements": requirements,
                    }

                    # Add optional metadata if available
                    if ema is not None:
                        cache_kwargs["ema"] = ema
                    if std_dev is not None:
                        cache_kwargs["std_dev"] = std_dev
                    if current_ema is not None:
                        cache_kwargs["current_ema"] = current_ema
                    if current_std_dev is not None:
                        cache_kwargs["current_std_dev"] = current_std_dev
                    if band_multipliers is not None:
                        cache_kwargs["band_multipliers"] = band_multipliers

                    yield from self._cache_cl_pool_data(**cache_kwargs)

                    self.context.logger.info(
                        f"Velodrome position requirements for {token0_symbol}/{token1_symbol}: "
                        f"{requirements['recommendation']}"
                    )

                    # Calculate aggregate token ratios from individual band requirements
                    position_requirements = requirements.get(
                        "position_requirements", []
                    )
                    token0_ratio, token1_ratio = self._calculate_aggregate_token_ratios(
                        position_requirements
                    )

                    # Merge individual band ratios into tick_bands for use in _enter_cl_pool
                    for i, band in enumerate(tick_bands):
                        if i < len(position_requirements):
                            band["token0_ratio"] = position_requirements[i][
                                "token0_ratio"
                            ]
                            band["token1_ratio"] = position_requirements[i][
                                "token1_ratio"
                            ]
                        else:
                            # Fallback if position_requirements is shorter than tick_bands
                            band["token0_ratio"] = 0.5
                            band["token1_ratio"] = 0.5

                    # Add percentage values to the opportunity
                    opportunity["token0_percentage"] = token0_ratio * 100
                    opportunity["token1_percentage"] = token1_ratio * 100

                    self.context.logger.info(
                        f"Token allocation percentages: {opportunity['token0_percentage']:.2f}% {token0_symbol}, "
                        f"{opportunity['token1_percentage']:.2f}% {token1_symbol}"
                    )

                    # Store these requirements
                    opportunity["token_requirements"] = requirements
                    # IMPORTANT: Add tick_spacing with individual ratios and tick_bands to the opportunity
                    opportunity["tick_spacing"] = tick_spacing
                    opportunity["tick_ranges"] = tick_bands
                    # IMPORTANT: Store percent_in_bounds for TiP calculations
                    opportunity["percent_in_bounds"] = percent_in_bounds

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
                    # Calculate aggregate ratios from individual band requirements
                    position_requirements = requirements.get(
                        "position_requirements", []
                    )
                    (
                        aggregate_token0_ratio,
                        aggregate_token1_ratio,
                    ) = self._calculate_aggregate_token_ratios(position_requirements)

                    if aggregate_token0_ratio >= max_ration:
                        # Only need token0
                        opportunity["max_amounts_in"] = [token0_balance, 0]
                        self.context.logger.info(
                            f"Using only token0: {token0_balance} {token0_symbol}"
                        )
                    elif aggregate_token1_ratio >= max_ration:
                        # Only need token1
                        opportunity["max_amounts_in"] = [0, token1_balance]
                        self.context.logger.info(
                            f"Using only token1: {token1_balance} {token1_symbol}"
                        )
                    else:
                        max_amounts_in, log_message = self._calculate_max_amounts_in(
                            token0_balance=token0_balance,
                            token1_balance=token1_balance,
                            aggregate_token0_ratio=aggregate_token0_ratio,
                            aggregate_token1_ratio=aggregate_token1_ratio,
                            token0_symbol=token0_symbol,
                            token1_symbol=token1_symbol,
                        )

                        opportunity["max_amounts_in"] = max_amounts_in
                        self.context.logger.info(log_message)

                    # Store results for this pool
                    results[pool_address] = requirements

                except Exception as e:
                    self.context.logger.error(
                        f"Error analyzing Velodrome position: {str(e)}"
                    )

                    self.context.logger.error(traceback.format_exc())

        self.context.logger.info("Velodrome position analysis complete")
        return results

    def _check_tip_exit_conditions(
        self, position: Dict
    ) -> Generator[None, None, Tuple[bool, str]]:
        """Check if position can be exited based on TiP conditions

        New conditions:
        - TiP = min(MPT, Position Yield < SVy, 21 days)
        - Trailing stop-loss: Position Yield < SVy
        Where:
        - Position Yield = current yield per day (from position APR)
        - S = stoploss threshold multiplier (configurable, default 0.6)
        - Vy = yield per day at entry (from entry APR)
        """
        try:
            entry_cost = position.get("entry_cost", 0)

            # Legacy positions (no entry_cost) use 3-week rule
            if entry_cost == 0:
                if not position.get("enter_timestamp"):
                    return True, "No TiP data - legacy position"

                days_elapsed = self._calculate_days_since_entry(
                    position["enter_timestamp"]
                )
                if days_elapsed >= MIN_TIME_IN_POSITION:
                    return True, f"Legacy position: {days_elapsed:.1f} >= 21 days"

                remaining_days = MIN_TIME_IN_POSITION - days_elapsed
                return (
                    False,
                    f"Legacy position must hold {remaining_days:.1f} more days",
                )

            # Check trailing stop-loss condition: Position Yield < SVy
            # Get entry APR with backward compatibility
            entry_apr = position.get("entry_apr")
            if entry_apr is None:
                entry_apr = position.get("apr")  # Fallback for legacy positions

            # Get current APR
            current_apr = position.get("apr")

            # If both entry_apr and apr are missing, mark as open to exit
            if entry_apr is None and current_apr is None:
                return True, "No APR data - marking as open to exit"

            # If current APR is missing, mark as open to exit
            if current_apr is None:
                return True, "No current APR - marking as open to exit"

            # Calculate yields per day
            if entry_apr is not None:
                entry_yield_per_day = (
                    entry_apr / 100
                ) / 365  # Convert % to decimal, then to daily
            else:
                # Fallback: use current APR as entry APR (for very old positions)
                entry_yield_per_day = (current_apr / 100) / 365

            current_yield_per_day = (current_apr / 100) / 365

            # Get stoploss threshold multiplier (S)
            S = self.params.stoploss_threshold_multiplier

            # Check trailing stop-loss: Position Yield < SVy
            if current_yield_per_day < (S * entry_yield_per_day):
                return (
                    True,
                    f"Trailing stop-loss triggered: current yield {current_yield_per_day:.6f} < {S} * entry yield {entry_yield_per_day:.6f}",
                )

            # Check position value condition: CurrentValueRatio < MinReqPositionValue
            # This check is for mean-reverting assets to prevent false exits during volatility
            # Exit if current value is below minimum required
            min_req_value = self._calculate_min_req_position_value(position, S)
            if min_req_value is not None:
                current_value_ratio = yield from self._calculate_current_value_ratio(
                    position, position.get("chain")
                )
                if current_value_ratio is not None:
                    if current_value_ratio < min_req_value:
                        return (
                            True,
                            f"Position value check: CurrentValueRatio {current_value_ratio:.6f} < MinReqPositionValue {min_req_value:.6f}",
                        )

            # Check opportunity cost: Position Yield < SVby
            vby = self._get_best_available_opportunity_yield()
            if vby is not None:
                if current_yield_per_day < (S * vby):
                    return (
                        True,
                        f"Opportunity cost check: current yield {current_yield_per_day:.6f} < {S} * best opportunity yield {vby:.6f}",
                    )

            # For new positions, check traditional conditions:
            # 1. Minimum time requirement
            # 2. Cost recovery through yield

            cost_recovered = position.get("cost_recovered", False)
            minimum_time_met = self._check_minimum_time_met(position)

            days_elapsed = self._calculate_days_since_entry(position["enter_timestamp"])
            min_hold_days = position.get("min_hold_days", 0)

            # Check if both conditions are satisfied
            if cost_recovered and minimum_time_met:
                return (
                    True,
                    f"Both conditions met: costs recovered AND minimum time ({days_elapsed:.1f} >= {min_hold_days:.1f} days)",
                )

            # Check 21-day global temporal cap
            if days_elapsed >= MIN_TIME_IN_POSITION:
                return (
                    True,
                    f"21-day global temporal cap met: {days_elapsed:.1f} >= {MIN_TIME_IN_POSITION} days",
                )

            # Determine what's still needed
            missing_conditions = []
            if not cost_recovered:
                yield_usd = position.get("yield_usd", 0)
                missing_conditions.append(
                    f"costs not recovered (${yield_usd:.2f}/${entry_cost:.2f})"
                )

            if not minimum_time_met:
                remaining_days = min_hold_days - days_elapsed
                missing_conditions.append(
                    f"minimum time not met ({remaining_days:.1f} more days needed)"
                )

            return False, f"Cannot exit: {' AND '.join(missing_conditions)}"

        except Exception as e:
            self.context.logger.error(f"Error checking TiP exit conditions: {e}")
            return True, "Error in TiP check - allowing exit"

    def _calculate_position_yield_per_day(self, position: Dict) -> Optional[float]:
        """Calculate position yield per day from position APR

        Args:
            position: Position dictionary containing APR data

        Returns:
            Yield per day as decimal (e.g., 0.000548 for 20% APR), or None if APR missing
        """
        apr = position.get("apr")
        if apr is None:
            return None

        # APR is stored as percentage (e.g., 20.0 for 20%)
        # Convert to decimal and divide by 365 to get daily yield
        apr_decimal = apr / 100
        yield_per_day = apr_decimal / 365
        return yield_per_day

    def _get_position_token_balances(
        self, position: Dict, chain: str
    ) -> Generator[None, None, Dict[str, float]]:
        """Get current token balances for a position.

        This is a helper method to get token balances for CurrentValueRatio calculation.
        It tries to use _get_current_token_balances if available, otherwise falls back
        to getting balances directly from the safe.
        """
        # Try to use _get_current_token_balances if available (from FetchStrategiesBehaviour)
        if hasattr(self, "_get_current_token_balances"):
            balances = yield from self._get_current_token_balances(position, chain)
            if balances:
                # Convert Decimal to float
                return {k: float(v) for k, v in balances.items()}

        # Fallback: get balances directly from safe
        token0_address = position.get("token0")
        token1_address = position.get("token1")
        safe_address = self.params.safe_contract_addresses.get(chain)

        if not safe_address:
            self.context.logger.error(f"No safe address for chain {chain}")
            return {}

        balances = {}
        if token0_address:
            balance0 = yield from self._get_token_balance(
                chain, safe_address, token0_address
            )
            if balance0 is not None:
                token0_decimals = yield from self._get_token_decimals(
                    chain, token0_address
                )
                if token0_decimals:
                    balances[token0_address] = balance0 / (10**token0_decimals)

        if token1_address:
            balance1 = yield from self._get_token_balance(
                chain, safe_address, token1_address
            )
            if balance1 is not None:
                token1_decimals = yield from self._get_token_decimals(
                    chain, token1_address
                )
                if token1_decimals:
                    balances[token1_address] = balance1 / (10**token1_decimals)

        return balances

    def _calculate_current_value_ratio(
        self, position: Dict, chain: str
    ) -> Generator[None, None, Optional[float]]:
        """Calculate CurrentValueRatio using SMA prices

        Formula: CurrentValueRatio = (Q1*SMA1 + Q0*SMA0 + Qr*SMAr) / (Q1*SMA1 + Q0*SMA0)
        Where:
        - Qn = quantity of token n (decimal-adjusted)
        - SMAn = SMA price of token n
        - Qr, SMAr = reserve token (if applicable, currently not used)

        Args:
            position: Position dictionary
            chain: Chain name

        Returns:
            CurrentValueRatio as float, or None if unable to calculate
        """
        try:
            # Get token addresses
            token0_address = position.get("token0")
            token1_address = position.get("token1")

            if not token0_address or not token1_address:
                self.context.logger.error(
                    "Missing token addresses for CurrentValueRatio calculation"
                )
                return None

            # Get current token balances (quantities)
            user_balances = yield from self._get_position_token_balances(
                position, chain
            )

            if not user_balances:
                self.context.logger.error(
                    "Could not get current token balances for CurrentValueRatio"
                )
                return None

            # Get quantities (decimal-adjusted)
            Q0 = user_balances.get(token0_address, 0)
            Q1 = user_balances.get(token1_address, 0)

            if Q0 == 0 and Q1 == 0:
                self.context.logger.warning(
                    "Both token quantities are zero for CurrentValueRatio"
                )
                return None

            # Fetch SMA prices
            SMA0 = yield from self._fetch_token_prices_sma(token0_address, chain)
            SMA1 = yield from self._fetch_token_prices_sma(token1_address, chain)

            if SMA0 is None or SMA1 is None:
                self.context.logger.warning(
                    "Could not fetch SMA prices for CurrentValueRatio calculation"
                )
                return None

            # Calculate numerator: Q1*SMA1 + Q0*SMA0 + Qr*SMAr
            # For now, Qr*SMAr = 0 (reserve token not implemented)
            numerator = (Q1 * SMA1) + (Q0 * SMA0)

            # Calculate denominator: Q1*SMA1 + Q0*SMA0
            denominator = (Q1 * SMA1) + (Q0 * SMA0)

            if denominator == 0:
                self.context.logger.warning(
                    "Denominator is zero for CurrentValueRatio calculation"
                )
                return None

            # Calculate ratio
            current_value_ratio = numerator / denominator

            self.context.logger.info(
                f"CurrentValueRatio: {current_value_ratio:.6f} "
                f"(Q0={Q0:.6f}, Q1={Q1:.6f}, SMA0=${SMA0:.6f}, SMA1=${SMA1:.6f})"
            )

            return current_value_ratio

        except Exception as e:
            self.context.logger.error(f"Error calculating CurrentValueRatio: {e}")
            return None

    def _calculate_min_req_position_value(
        self, position: Dict, S: float
    ) -> Optional[float]:
        """Calculate MinReqPositionValue

        Formula: MinReqPositionValue = S * Vy * t/T + 1
        Where:
        - S = stoploss multiplier
        - Vy = yield per day (decimal)
        - t = time position held (in minutes)
        - T = time in year (in minutes: 365 * 24 * 60)

        Args:
            position: Position dictionary
            S: Stoploss threshold multiplier

        Returns:
            MinReqPositionValue as float, or None if unable to calculate
        """
        try:
            # Get entry APR with backward compatibility
            entry_apr = position.get("entry_apr")
            if entry_apr is None:
                entry_apr = position.get("apr")  # Fallback for legacy positions

            if entry_apr is None:
                self.context.logger.warning(
                    "No entry APR available for MinReqPositionValue calculation"
                )
                return None

            # Calculate Vy (yield per day as decimal)
            Vy = (entry_apr / 100) / 365

            # Get time position held (in minutes)
            enter_timestamp = position.get("enter_timestamp")
            if not enter_timestamp:
                self.context.logger.warning(
                    "No enter_timestamp for MinReqPositionValue calculation"
                )
                return None

            current_timestamp = int(self._get_current_timestamp())
            time_held_seconds = current_timestamp - enter_timestamp
            time_held_minutes = time_held_seconds / 60

            # T = time in year (in minutes)
            T_minutes = 365 * 24 * 60

            # Calculate MinReqPositionValue = S * Vy * t/T + 1
            min_req_value = (S * Vy * time_held_minutes / T_minutes) + 1

            self.context.logger.info(
                f"MinReqPositionValue: {min_req_value:.6f} "
                f"(S={S}, Vy={Vy:.6f}, t={time_held_minutes:.1f} min, T={T_minutes} min)"
            )

            return min_req_value

        except Exception as e:
            self.context.logger.error(f"Error calculating MinReqPositionValue: {e}")
            return None

    def _get_best_available_opportunity_yield(self) -> Optional[float]:
        """Get effective yield of the best available opportunity (Vby) with SMA

        Applies short look-back SMA to advertised yields to dampen transient spikes.
        This implements the opportunity cost check.

        Returns:
            Effective yield per day (as decimal) of best opportunity, or None if no opportunities
        """
        try:
            if not self.trading_opportunities:
                return None

            # Sort opportunities by APR (highest first)
            sorted_opportunities = sorted(
                self.trading_opportunities, key=lambda x: x.get("apr", 0), reverse=True
            )

            if not sorted_opportunities:
                return None

            # Get the best opportunity's APR
            best_apr = sorted_opportunities[0].get("apr")
            if best_apr is None or best_apr <= 0:
                return None

            # For now, we use the advertised APR directly
            # TODO: Implement SMA for advertised yields (requires storing historical APR values)
            # This would involve:
            # 1. Storing historical APR values for each opportunity
            # 2. Calculating SMA over a short look-back period (e.g., 3-7 days)
            # 3. Using SMA-adjusted APR instead of raw APR

            # Convert APR percentage to daily yield (decimal)
            apr_decimal = best_apr / 100
            vby = apr_decimal / 365

            self.context.logger.info(
                f"Best available opportunity yield (Vby): {vby:.6f} "
                f"(from APR: {best_apr:.2f}%)"
            )

            return vby

        except Exception as e:
            self.context.logger.error(
                f"Error getting best available opportunity yield: {e}"
            )
            return None

    def _apply_tip_filters_to_exit_decisions(
        self,
    ) -> Generator[None, None, Tuple[bool, Optional[List[Dict]]]]:
        """Filter positions that can be exited based on TiP"""
        try:
            eligible_for_exit = []
            blocked_by_tip = []

            if not self.current_positions:
                self.context.logger.info("No current positions to check for TiP.")
                return True, []

            for position in self.current_positions:
                if position.get("status") == PositionStatus.OPEN.value:
                    can_exit, reason = yield from self._check_tip_exit_conditions(
                        position
                    )

                    if can_exit:
                        eligible_for_exit.append(position)
                        # Log exit conditions (trailing stop-loss, opportunity cost, etc.)
                        self.context.logger.info(
                            f"Position {position.get('pool_address', 'unknown')} eligible for exit: {reason}"
                        )
                    else:
                        blocked_by_tip.append((position, reason))
                        self.context.logger.info(f"TiP blocking exit: {reason}")

            if blocked_by_tip and not eligible_for_exit:
                self.context.logger.info("All positions blocked by TiP conditions")

            # Return True if there are eligible positions or all positions are closed
            should_proceed = bool(eligible_for_exit) or not any(
                p.get("status") == PositionStatus.OPEN.value
                for p in self.current_positions
            )

            return should_proceed, eligible_for_exit

        except Exception as e:
            self.context.logger.error(f"Error applying TiP filters: {e}")
            # Return all open positions if TiP filtering fails
            return True, [
                p
                for p in self.current_positions
                if p.get("status") == PositionStatus.OPEN.value
            ]

    def check_funds(self) -> bool:
        """Check if there are any funds available."""

        if not any(
            p.get("status") == PositionStatus.OPEN.value for p in self.current_positions
        ):
            has_funds = any(
                asset.get("balance", 0) > 0
                for position in self.synchronized_data.positions
                for asset in position.get("assets", [])
            )
            return has_funds

        return True

    def update_position_metrics(self) -> None:
        """Update metrics for all open positions."""
        if not self.positions_eligible_for_exit:
            return

        current_timestamp = self._get_current_timestamp()
        open_positions = [
            pos
            for pos in self.positions_eligible_for_exit
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

                    # Use base class method to build swap action
                    swap_action = self._build_swap_to_usdc_action(
                        chain=chain,
                        from_token_address=asset_address,
                        from_token_symbol=asset_symbol,
                        funds_percentage=1.0,  # Use all available balance
                        description=f"Swap non-whitelisted {asset_symbol} to USDC",
                    )

                    if swap_action:
                        actions.append(swap_action)
                        self.context.logger.info(
                            f"Prepared {asset_symbol} to USDC swap on {chain}: "
                            f"{asset_symbol} -> USDC"
                        )
                    else:
                        self.context.logger.error(
                            f"Failed to create swap action for {asset_symbol}"
                        )

            return actions

        except Exception as e:
            self.context.logger.error(
                f"Error in check_and_prepare_non_whitelisted_swaps: {str(e)}"
            )
            return []

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
                for pos in self.positions_eligible_for_exit
                if pos.get("status") == PositionStatus.OPEN.value
            ],
            "max_pools": self.params.max_pools,
            "composite_score_threshold": composite_score,
        }
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
                    f"Evaluating opportunity- Dex:{selected_opportunity.get('dex_type')} Pool:{selected_opportunity.get('pool_address')} Concentrated Liquidity:{selected_opportunity.get('is_cl_pool')}"
                )

            # Track final selected opportunities
            yield from self._track_opportunities(
                self.selected_opportunities, "final_selection"
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

    async def _async_execute_strategy(
        self, strategy_name: str, strategies_executables: Dict, **kwargs
    ) -> Dict:
        """Execute a single strategy asynchronously."""
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
                "error": [f"Unexpected error in strategy {strategy_name}: {str(e)}"],
                "result": [],
            }

    async def _run_all_strategies(
        self, strategy_kwargs_list: List, strategies_executables: Dict
    ) -> List[Dict]:
        """Run all strategies in parallel."""
        tasks = []
        results = []

        for strategy_name, kwargs in strategy_kwargs_list:
            try:
                kwargs_without_strategy = {
                    k: v for k, v in kwargs.items() if k != "strategy"
                }
                tasks.append(
                    self._async_execute_strategy(
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
                strategy_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in strategy_results:
                    if isinstance(result, Exception):
                        results.append(
                            {
                                "error": [f"Strategy execution error: {str(result)}"],
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
                            for pos in self.positions_eligible_for_exit
                            if pos.get("status") == PositionStatus.OPEN.value
                            and pos.get("pool_address")
                        ]
                        if self.positions_eligible_for_exit
                        else []
                    ),
                    "coingecko_api_key": self.coingecko.api_key,
                    "whitelisted_assets": WHITELISTED_ASSETS,
                    "get_metrics": False,
                    "coin_id_mapping": COIN_ID_MAPPING,
                    "x402_signer": self.eoa_account
                    if self.coingecko.use_x402
                    else None,
                    "x402_proxy": self.coingecko.coingecko_x402_server_base_url
                    if self.coingecko.use_x402
                    else None,
                }
            )
            strategy_kwargs_list.append((next_strategy, kwargs))

        strategies_executables = self.shared_state.strategies_executables

        # Main execution loop
        try:
            future = asyncio.ensure_future(
                self._run_all_strategies(strategy_kwargs_list, strategies_executables)
            )
            while not future.done():
                yield  # Yield control to the agent loop
            results = future.result()

            all_raw_opportunities = []
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
                                # Add strategy source to opportunity
                                opportunity["strategy_source"] = next_strategy
                                self.context.logger.info(
                                    f"Opportunity: {opportunity.get('pool_address', 'N/A')}, "
                                    f"Chain: {opportunity.get('chain', 'N/A')}, "
                                    f"Token0: {opportunity.get('token0_symbol', 'N/A')}, "
                                    f"Token1: {opportunity.get('token1_symbol', 'N/A')}"
                                )
                                valid_opportunities.append(opportunity)
                                all_raw_opportunities.append(opportunity)
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

            # Track raw opportunities with all metrics
            if all_raw_opportunities:
                yield from self._track_opportunities(
                    all_raw_opportunities, "raw_with_metrics"
                )

            # Track opportunities after basic filtering (whitelist, token count, etc.)
            # This would be the pools that pass initial filtering but before composite scoring
            basic_filtered_opportunities = []
            for opp in all_raw_opportunities:
                # Basic filtering criteria that we can replicate here
                if (
                    opp.get("token_count", 0) >= 2
                    and opp.get("tvl", 0) > 0
                    and opp.get("pool_address")
                    not in [
                        pos.get("pool_address")
                        for pos in self.positions_eligible_for_exit
                        if pos.get("status") == PositionStatus.OPEN.value
                    ]
                ):
                    basic_filtered_opportunities.append(opp)

            if basic_filtered_opportunities:
                yield from self._track_opportunities(
                    basic_filtered_opportunities, "basic_filtered"
                )

            # Track opportunities after composite filtering (APR + TVL scoring)
            # This would be the top N pools after composite scoring
            composite_filtered_opportunities = []
            for opp in all_raw_opportunities:
                # Composite filtering criteria (simplified version)
                if (
                    opp.get("token_count", 0) >= 2
                    and opp.get("tvl", 0) >= 1000
                    and opp.get("apr", 0) > 0  # min_tvl_threshold
                    and opp.get("pool_address")
                    not in [
                        pos.get("pool_address")
                        for pos in self.positions_eligible_for_exit
                        if pos.get("status") == PositionStatus.OPEN.value
                    ]
                ):
                    composite_filtered_opportunities.append(opp)

            # Sort by composite score if available, otherwise by APR
            composite_filtered_opportunities.sort(
                key=lambda x: x.get("composite_score", 0) or x.get("apr", 0),
                reverse=True,
            )

            # Take top N (simulating the composite pre-filter)
            top_n = 10  # This should match the top_n parameter used in strategies
            composite_filtered_opportunities = composite_filtered_opportunities[:top_n]

            if composite_filtered_opportunities:
                yield from self._track_opportunities(
                    composite_filtered_opportunities, "composite_filtered"
                )

        except Exception as e:
            self.context.logger.error(
                f"Critical error in strategy evaluation: {str(e)}"
            )
            # Ensure we don't lose the error state
            self.trading_opportunities = []

    def _track_opportunities(
        self, opportunities: List[Dict], stage: str
    ) -> Generator[None, None, None]:
        """Track opportunities at different stages for monitoring."""
        try:
            # Read existing tracking data
            existing_data = yield from self._read_kv(keys=("opportunity_tracking",))

            # Handle case where opportunity_tracking key doesn't exist or is None
            if not existing_data or not existing_data.get("opportunity_tracking"):
                tracking_data = {}
            else:
                try:
                    tracking_data = json.loads(
                        existing_data.get("opportunity_tracking", "{}")
                    )
                except (json.JSONDecodeError, TypeError):
                    # If JSON parsing fails, start with empty dict
                    tracking_data = {}

            # Add current round's data
            round_key = f"round_{self.synchronized_data.period_count}"
            if round_key not in tracking_data:
                tracking_data[round_key] = {}

            # Format opportunities for storage
            formatted_opportunities = []
            for opp in opportunities:
                formatted_opp = self._format_opportunity_for_tracking(opp, stage)
                formatted_opportunities.append(formatted_opp)

            # Add stage-specific metadata
            stage_metadata = {
                "timestamp": self._get_current_timestamp(),
                "opportunities": formatted_opportunities,
                "count": len(formatted_opportunities),
                "strategies_used": list(
                    set(opp.get("strategy_source", "unknown") for opp in opportunities)
                ),
            }

            # Add filtering criteria for each stage
            if stage == "raw_with_metrics":
                stage_metadata[
                    "filtering_criteria"
                ] = "All pools with calculated metrics"
            elif stage == "basic_filtered":
                stage_metadata[
                    "filtering_criteria"
                ] = "Token count >= 2, TVL > 0, not current position"
            elif stage == "composite_filtered":
                stage_metadata[
                    "filtering_criteria"
                ] = "Token count >= 2, TVL >= 1000, APR > 0, top 10 by composite score"
            elif stage == "final_selection":
                stage_metadata[
                    "filtering_criteria"
                ] = "Selected by hyper-strategy after risk assessment"

            tracking_data[round_key][stage] = stage_metadata

            # Write back to KV store
            yield from self._write_kv(
                {"opportunity_tracking": json.dumps(tracking_data)}
            )
            self.context.logger.info(
                f"Tracked {len(formatted_opportunities)} opportunities at stage '{stage!r}'"
            )

        except Exception as e:
            self.context.logger.error(f"Error tracking opportunities: {str(e)}")

    def _format_opportunity_for_tracking(self, opportunity: Dict, stage: str) -> Dict:
        """Format opportunity data for tracking."""
        return {
            # Basic pool info
            "pool_address": opportunity.get("pool_address"),
            "pool_id": opportunity.get("pool_id"),
            "dex_type": opportunity.get("dex_type"),
            "chain": opportunity.get("chain"),
            "pool_type": opportunity.get("pool_type", "lp"),
            "is_stable": opportunity.get("is_stable", False),
            "is_cl_pool": opportunity.get("is_cl_pool", False),
            "token_count": opportunity.get("token_count", 2),
            # Token details
            "token0_address": opportunity.get("token0"),
            "token0_symbol": opportunity.get("token0_symbol"),
            "token1_address": opportunity.get("token1"),
            "token1_symbol": opportunity.get("token1_symbol"),
            # Market metrics
            "apr": opportunity.get("apr"),
            "tvl": opportunity.get("tvl"),
            "daily_volume": opportunity.get("daily_volume"),
            # Evaluation metrics
            "sharpe_ratio": opportunity.get("sharpe_ratio"),
            "il_risk_score": opportunity.get("il_risk_score"),
            "depth_score": opportunity.get("depth_score"),
            "composite_score": opportunity.get("composite_score"),
            # Metadata
            "stage": stage,
            "strategy_source": opportunity.get("strategy_source"),
            "timestamp": self._get_current_timestamp(),
        }

    def _push_opportunity_metrics_to_mirrordb(self) -> Generator[None, None, None]:
        """Push opportunity data to MirrorDB with complete agent information."""
        try:
            # Read opportunity data from KV store
            opportunity_data = yield from self._read_kv(keys=("opportunity_tracking",))
            if not opportunity_data or not opportunity_data.get("opportunity_tracking"):
                self.context.logger.info("No opportunity tracking data to push")
                return

            opportunity_tracking_value = opportunity_data.get("opportunity_tracking")
            if opportunity_tracking_value is None:
                tracking_data = {}
            else:
                try:
                    tracking_data = json.loads(opportunity_tracking_value)
                except (json.JSONDecodeError, TypeError):
                    tracking_data = {}

            # Get or create agent registry data (using base behavior helper)
            agent_registry = yield from self._get_or_create_agent_registry()
            if not agent_registry:
                self.context.logger.error("Failed to get or create agent registry")
                return
            agent_id = agent_registry["agent_id"]

            # Get agent type data (same pattern as APR population)
            sender = self.context.agent_address
            agent_type = yield from self._get_or_create_agent_type(sender)
            if not agent_type:
                self.context.logger.error("Failed to get or create agent type")
                return

            # Get or create attribute definition for opportunities
            attr_def_data = yield from self._read_kv(keys=("opportunity_attr_def",))
            if not attr_def_data:
                # Create opportunity attribute definition
                attr_def = yield from self._create_opportunity_attr_def(
                    agent_id, agent_type
                )
                if attr_def:
                    yield from self._write_kv(
                        {"opportunity_attr_def": json.dumps(attr_def)}
                    )
                else:
                    self.context.logger.error(
                        "Failed to create opportunity attribute definition"
                    )
                    return
            else:
                # Handle case where opportunity_attr_def value is None or empty
                opportunity_attr_def_value = attr_def_data.get("opportunity_attr_def")
                if opportunity_attr_def_value is None:
                    self.context.logger.warning(
                        "Opportunity attribute definition data is None, creating new one"
                    )
                    attr_def = yield from self._create_opportunity_attr_def(
                        agent_id, agent_type
                    )
                    if attr_def:
                        yield from self._write_kv(
                            {"opportunity_attr_def": json.dumps(attr_def)}
                        )
                    else:
                        self.context.logger.error(
                            "Failed to create opportunity attribute definition"
                        )
                        return
                else:
                    try:
                        attr_def = json.loads(opportunity_attr_def_value)
                    except (json.JSONDecodeError, TypeError) as e:
                        self.context.logger.warning(
                            f"Error parsing opportunity attribute definition: {e}, creating new one"
                        )
                        attr_def = yield from self._create_opportunity_attr_def(
                            agent_id, agent_type
                        )
                        if attr_def:
                            yield from self._write_kv(
                                {"opportunity_attr_def": json.dumps(attr_def)}
                            )
                        else:
                            self.context.logger.error(
                                "Failed to create opportunity attribute definition"
                            )
                            return

            # Prepare comprehensive opportunity data for MirrorDB
            opportunity_metrics = {
                # Agent Information
                "agent_hash": os.environ.get("AEA_AGENT", "").split(":")[-1]
                if os.environ.get("AEA_AGENT")
                else "unknown",
                "agent_name": agent_registry.get("agent_name", "unknown"),
                "agent_address": agent_registry.get("agent_address", "unknown"),
                "agent_type": agent_type.get("type_name", "unknown"),
                "agent_id": agent_id,
                # Round Information
                "round": self.synchronized_data.period_count,
                "timestamp": self._get_current_timestamp(),
                # Trading Configuration
                "trading_type": self.shared_state.trading_type,
                "selected_protocols": self.shared_state.selected_protocols,
                "target_chains": self.params.target_investment_chains,
                # Current State
                "current_positions_count": len(self.current_positions),
                "portfolio_value": self.portfolio_data.get("portfolio_value", 0)
                if hasattr(self, "portfolio_data")
                else 0,
                # Opportunity Data
                "opportunity_data": tracking_data,
                "final_selection": self.selected_opportunities
                if self.selected_opportunities
                else [],
                # Performance Context
                "composite_score_threshold": getattr(
                    self.params, "composite_score_threshold", None
                ),
                "max_pools": getattr(self.params, "max_pools", 1),
            }

            # Push to MirrorDB
            agent_attr = yield from self.create_agent_attribute(
                agent_id, attr_def["attr_def_id"], opportunity_metrics
            )

            if agent_attr:
                # Clean up KV store after successful push
                yield from self._write_kv({"opportunity_tracking": "{}"})
                self.context.logger.info(
                    "Successfully pushed opportunity data to MirrorDB and cleaned KV store"
                )
            else:
                self.context.logger.error("Failed to push opportunity data to MirrorDB")

        except Exception as e:
            self.context.logger.error(
                f"Error pushing opportunity metrics to MirrorDB: {str(e)}"
            )

    def _create_opportunity_attr_def(
        self, agent_id: str, agent_type: Dict
    ) -> Generator[None, None, Optional[Dict]]:
        """Create opportunity attribute definition in MirrorDB."""
        try:
            if not agent_type:
                self.context.logger.error("Agent type is empty or None")
                return None

            if "type_id" not in agent_type:
                self.context.logger.error(
                    f"Agent type missing type_id. Agent type data: {agent_type}"
                )
                return None

            type_id = agent_type["type_id"]
            self.context.logger.info(
                f"Creating opportunity attribute definition with type_id: {type_id}"
            )

            # Create attribute definition
            attr_def = yield from self.create_attribute_definition(
                type_id,
                "opportunity_metrics",
                "json",
                True,
                "{}",
                agent_id,
            )

            return attr_def

        except Exception as e:
            self.context.logger.error(
                f"Error creating opportunity attribute definition: {str(e)}"
            )
            self.context.logger.error(f"Agent type data: {agent_type}")
            return None

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
                "current_positions": self.positions_eligible_for_exit,
                "whitelisted_assets": WHITELISTED_ASSETS,
                "coin_id_mapping": COIN_ID_MAPPING,
                "x402_signer": self.eoa_account if self.coingecko.use_x402 else None,
                "x402_proxy": self.coingecko.coingecko_x402_server_base_url
                if self.coingecko.use_x402
                else None,
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
        """Identify and merge duplicate bridge swap actions, and remove redundant same-chain same-token actions"""
        try:
            if not actions:
                return actions

            # Find all bridge swap actions
            bridge_swap_actions = [
                (i, action)
                for i, action in enumerate(actions)
                if action.get("action") == Action.FIND_BRIDGE_ROUTE.value
            ]

            if len(bridge_swap_actions) == 0:
                return actions  # No bridge actions to process

            # First, filter out redundant same-chain same-token actions
            indices_to_remove_redundant = []
            for idx, action in bridge_swap_actions:
                try:
                    from_chain = action.get("from_chain", "")
                    to_chain = action.get("to_chain", "")
                    from_token = action.get("from_token", "")
                    to_token = action.get("to_token", "")

                    # Check if this is a redundant action (same chain and same token)
                    if (
                        from_chain == to_chain
                        and from_token.lower() == to_token.lower()
                        and from_chain
                        and to_chain
                        and from_token
                        and to_token
                    ):
                        indices_to_remove_redundant.append(idx)

                        self.context.logger.info(
                            f"Removing redundant bridge swap action: "
                            f"{action.get('from_token_symbol', 'unknown')} on {from_chain} "
                            f"to {action.get('to_token_symbol', 'unknown')} on {to_chain} "
                            f"(same chain and same token address)"
                        )
                except Exception as e:
                    self.context.logger.error(
                        f"Error checking for redundant bridge swap action: {e}. Action: {action}"
                    )

            # Remove redundant actions first
            if indices_to_remove_redundant:
                actions = [
                    action
                    for i, action in enumerate(actions)
                    if i not in indices_to_remove_redundant
                ]

                # Update bridge_swap_actions list after removing redundant actions
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

            # Calculate aggregate token ratios from individual band requirements
            position_requirements = token_requirements.get("position_requirements", [])
            if position_requirements:
                (
                    aggregate_token0_ratio,
                    aggregate_token1_ratio,
                ) = self._calculate_aggregate_token_ratios(position_requirements)
            else:
                # Fall back to checking recommendation text
                recommendation = token_requirements.get("recommendation", "")
                if "100% token0" in recommendation:
                    aggregate_token0_ratio = 1.0
                    aggregate_token1_ratio = 0.0
                elif "100% token1" in recommendation:
                    aggregate_token0_ratio = 0.0
                    aggregate_token1_ratio = 1.0
                else:
                    aggregate_token0_ratio = 0.5
                    aggregate_token1_ratio = 0.5

            self.context.logger.info(
                f"Velodrome position requirements: token0_ratio={aggregate_token0_ratio}, token1_ratio={aggregate_token1_ratio}"
            )

            # Check if all funds should go to one token (using 0.99 threshold to handle floating point)
            if (aggregate_token0_ratio >= 0.99 and aggregate_token1_ratio <= 0.01) or (
                aggregate_token0_ratio <= 0.01 and aggregate_token1_ratio >= 0.99
            ):
                # Determine which token gets 100%
                is_token0_full = aggregate_token0_ratio >= 0.99
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

                # Calculate the full slice for one-sided allocation
                try:
                    full_slice = float(
                        enter_pool_action.get("relative_funds_percentage", 1.0) or 1.0
                    )
                except (ValueError, TypeError):
                    full_slice = 1.0

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

                        # Use the full relative slice for one-sided allocation
                        action["funds_percentage"] = full_slice

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
                            "funds_percentage": full_slice,  # Use full allocation
                        }

                        self.context.logger.info(
                            f"Added new bridge route: {source_token.get('token_symbol')} -> {target_symbol}"
                        )

                        # Find the position to insert after exit pool action
                        insert_position = 0
                        for i, action in enumerate(actions):
                            if action.get("action") == "ExitPool":
                                insert_position = i + 1

                        # Insert after the last exit pool action (or at the beginning if no exit pool actions)
                        actions.insert(insert_position, new_bridge_route)

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

        # ==================== POSITION EXIT WITH STAKING ====================
        if self.position_to_exit:
            # Step 1: Claim staking rewards before exit (if position has staking)
            if self._has_staking_metadata(self.position_to_exit):
                # Step 2: Unstake LP tokens before exit
                unstake_action = self._build_unstake_lp_tokens_action(
                    self.position_to_exit
                )
                if unstake_action:
                    actions.append(unstake_action)
                    self.context.logger.info(
                        "Added unstake LP tokens action before exit"
                    )

            # Step 3: Exit the pool
            dex_type = self.position_to_exit.get("dex_type")
            num_of_tokens_required = 1 if dex_type == DexType.STURDY.value else 2
            exit_pool_action = self._build_exit_pool_action(
                tokens, num_of_tokens_required
            )
            if not exit_pool_action:
                self.context.logger.error("Error building exit pool action")
                return None
            actions.append(exit_pool_action)

        # ==================== POSITION ENTRY WITH STAKING ====================
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

            # Cache the enter pool action if this is a Velodrome CL pool
            if opportunity.get("dex_type") == "velodrome" and opportunity.get(
                "is_cl_pool"
            ):
                yield from self._cache_enter_pool_action_for_cl_pool(
                    opportunity, enter_pool_action
                )

            # Initialize entry costs for new position when actions are formed
            yield from self._initialize_entry_costs_for_new_position(enter_pool_action)

            actions.append(enter_pool_action)

            # Add staking action after entering pool (if applicable)
            if self._should_add_staking_actions(opportunity):
                stake_action = self._build_stake_lp_tokens_action(opportunity)
                if stake_action:
                    actions.append(stake_action)
                    self.context.logger.info(
                        "Added stake LP tokens action after pool entry"
                    )

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
        """Get tokens with the highest balances, filtering out reward tokens."""
        token_balances = []

        for position in self.synchronized_data.positions:
            chain = position.get("chain")
            for asset in position.get("assets", []):
                asset_address = asset.get("address")
                asset_symbol = asset.get("asset_symbol")
                balance = asset.get("balance", 0)

                if chain and asset_address:
                    # Filter out reward tokens from investment consideration
                    reward_addresses = REWARD_TOKEN_ADDRESSES.get(chain, {})
                    if asset_address.lower() in [
                        addr.lower() for addr in reward_addresses.keys()
                    ]:
                        self.context.logger.info(
                            f"Filtering out reward token {asset_symbol} ({asset_address}) - not for investment"
                        )
                        continue

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
        """Get the portion of token balance available for investment (total - reserved rewards - airdrop rewards)"""

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

        # This is a whitelisted reward token - subtract accumulated rewards and airdrop rewards
        accumulated_rewards = yield from self.get_accumulated_rewards_for_token(
            chain, token_address
        )

        # Subtract airdrop rewards for USDC on MODE chain
        airdrop_rewards = 0
        if (
            chain == "mode"
            and self.context.params.airdrop_started
            and token_address.lower() == self._get_usdc_address(chain).lower()
        ):
            airdrop_rewards = yield from self._get_total_airdrop_rewards(chain)
            self.context.logger.info(
                f"USDC airdrop rewards to exclude from investment: {airdrop_rewards} wei"
            )

        investable_balance = max(
            0, total_balance - accumulated_rewards - airdrop_rewards
        )

        self.context.logger.info(
            f"Token {token_address} on {chain}: "
            f"Total={total_balance}, Reserved={accumulated_rewards}, Airdrop={airdrop_rewards}, Investable={investable_balance}"
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

        # Use base class method to build exit action
        exit_pool_action = self._build_exit_pool_action_base(
            self.position_to_exit, tokens
        )
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
        target_ratios_by_token: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Handle the case where we have all required tokens on the destination chain with optimal allocation."""
        bridge_swap_actions = []

        if not required_tokens:
            return bridge_swap_actions

        # Filter tokens by chain
        dest_chain_tokens = [
            token for token in tokens if token.get("chain") == dest_chain
        ]
        other_chain_tokens = [
            token for token in tokens if token.get("chain") != dest_chain
        ]

        # Create mapping of available tokens on destination chain
        available_tokens_map = {
            token.get("token"): token for token in dest_chain_tokens
        }

        # Step 1: Handle tokens from other chains first
        for token in other_chain_tokens:
            # Distribute to required tokens based on target ratios
            dest_token_address, dest_token_symbol = required_tokens[
                0
            ]  # Use first required token
            token_percentage = relative_funds_percentage * target_ratios_by_token.get(
                dest_token_address, 1.0 / max(1, len(required_tokens))
            )
            self._add_bridge_swap_action(
                bridge_swap_actions,
                token,
                dest_chain,
                dest_token_address,
                dest_token_symbol,
                token_percentage,
            )

        # Step 2: Handle optimal allocation for tokens on destination chain
        try:
            if dest_chain_tokens:
                # Calculate total available value on destination chain
                total_dest_value = sum(
                    float(token.get("value", 0)) for token in dest_chain_tokens
                )

                if total_dest_value > 0:
                    # Calculate target values for each required token
                    target_values = {}
                    for req_token_addr, _req_token_symbol in required_tokens:
                        target_ratio = target_ratios_by_token.get(
                            req_token_addr, 1.0 / len(required_tokens)
                        )
                        target_values[req_token_addr] = total_dest_value * target_ratio

                    # Identify unnecessary tokens (not in required tokens)
                    unnecessary_tokens = []
                    for token in dest_chain_tokens:
                        token_addr = token.get("token")
                        if not any(
                            req_addr == token_addr for req_addr, _ in required_tokens
                        ):
                            unnecessary_tokens.append(token)

                    # Step 2a: Convert all unnecessary tokens to required tokens (distributed by target ratios)
                    if unnecessary_tokens:
                        for token in unnecessary_tokens:
                            # Distribute the unnecessary token across all required tokens based on target ratios
                            for (
                                target_token_addr,
                                target_token_symbol,
                            ) in required_tokens:
                                target_ratio = target_ratios_by_token.get(
                                    target_token_addr, 1.0 / len(required_tokens)
                                )
                                self._add_bridge_swap_action(
                                    bridge_swap_actions,
                                    token,
                                    dest_chain,
                                    target_token_addr,
                                    target_token_symbol,
                                    target_ratio,  # Convert based on target ratio
                                )

                    # Step 2b: Rebalance existing required tokens to achieve target ratios
                    for token in dest_chain_tokens:
                        token_addr = token.get("token")
                        token_value = float(token.get("value", 0))

                        # Skip if this token is not required
                        if not any(
                            req_addr == token_addr for req_addr, _ in required_tokens
                        ):
                            continue

                        # Find the target value for this token
                        target_value = target_values.get(token_addr, 0)

                        if token_value > target_value:
                            # This token has surplus, need to swap excess
                            surplus = token_value - target_value
                            # Find a token that needs more value
                            for other_req_addr, other_req_symbol in required_tokens:
                                if other_req_addr != token_addr:
                                    other_token = available_tokens_map.get(
                                        other_req_addr
                                    )
                                    if other_token:
                                        other_value = float(other_token.get("value", 0))
                                        other_target = target_values.get(
                                            other_req_addr, 0
                                        )
                                        if other_value < other_target:
                                            # Swap surplus to this token
                                            swap_fraction = min(
                                                1.0, surplus / token_value
                                            )
                                            self._add_bridge_swap_action(
                                                bridge_swap_actions,
                                                other_token,
                                                dest_chain,
                                                other_req_addr,
                                                other_req_symbol,
                                                swap_fraction,
                                            )
                                            break
        except Exception as e:
            self.context.logger.error(f"Error during on-chain rebalance planning: {e}")

        return bridge_swap_actions

    def _handle_some_tokens_available(
        self,
        tokens: List[Dict[str, Any]],
        required_tokens: List[Tuple[str, str]],
        tokens_we_need: List[Tuple[str, str]],
        dest_chain: str,
        relative_funds_percentage: float,
        target_ratios_by_token: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Handle the case where we have some but not all required tokens with optimal allocation."""
        bridge_swap_actions = []

        # Filter tokens by chain
        dest_chain_tokens = [
            token for token in tokens if token.get("chain") == dest_chain
        ]
        other_chain_tokens = [
            token for token in tokens if token.get("chain") != dest_chain
        ]

        # Create mapping of available tokens on destination chain
        available_tokens_map = {
            token.get("token"): token for token in dest_chain_tokens
        }

        # Step 1: Handle tokens from other chains first
        for token in other_chain_tokens:
            # Prioritize missing tokens first
            if tokens_we_need:
                dest_token_address, dest_token_symbol = tokens_we_need[0]
                token_percentage = (
                    relative_funds_percentage
                    * target_ratios_by_token.get(
                        dest_token_address, 1.0 / max(1, len(required_tokens))
                    )
                )
                self._add_bridge_swap_action(
                    bridge_swap_actions,
                    token,
                    dest_chain,
                    dest_token_address,
                    dest_token_symbol,
                    token_percentage,
                )

        # Step 2: Handle optimal allocation for tokens on destination chain
        if dest_chain_tokens:
            # Calculate total available value on destination chain
            total_dest_value = sum(
                float(token.get("value", 0)) for token in dest_chain_tokens
            )

            if total_dest_value > 0:
                # Calculate target values for each required token
                target_values = {}
                for req_token_addr, _req_token_symbol in required_tokens:
                    target_ratio = target_ratios_by_token.get(
                        req_token_addr, 1.0 / len(required_tokens)
                    )
                    target_values[req_token_addr] = total_dest_value * target_ratio

                # Identify unnecessary tokens (not in required tokens)
                unnecessary_tokens = []
                for token in dest_chain_tokens:
                    token_addr = token.get("token")
                    if not any(
                        req_addr == token_addr for req_addr, _ in required_tokens
                    ):
                        unnecessary_tokens.append(token)

                # Step 2a: Convert all unnecessary tokens to missing required tokens (distributed by target ratios)
                if unnecessary_tokens and tokens_we_need:
                    for token in unnecessary_tokens:
                        # Distribute the unnecessary token across missing required tokens based on target ratios
                        for target_token_addr, target_token_symbol in tokens_we_need:
                            target_ratio = target_ratios_by_token.get(
                                target_token_addr, 1.0 / len(tokens_we_need)
                            )
                            self._add_bridge_swap_action(
                                bridge_swap_actions,
                                token,
                                dest_chain,
                                target_token_addr,
                                target_token_symbol,
                                target_ratio,  # Convert based on target ratio
                            )

                # Step 2b: Handle case where we have some required tokens but need others
                # Only convert existing required tokens if we still need more to reach target ratios
                # This should only happen if unnecessary tokens weren't enough to reach target ratios
                available_required_tokens = []
                for token in dest_chain_tokens:
                    token_addr = token.get("token")
                    if any(req_addr == token_addr for req_addr, _ in required_tokens):
                        available_required_tokens.append((token_addr, token))

                # Only convert existing required tokens if we have no unnecessary tokens to convert
                if (
                    available_required_tokens
                    and tokens_we_need
                    and not unnecessary_tokens
                ):
                    # Use the first available required token to get the first missing one
                    source_token_addr, source_token = available_required_tokens[0]
                    target_token_addr, target_token_symbol = tokens_we_need[0]

                    # Skip if source and target are the same token
                    if source_token_addr != target_token_addr:
                        target_ratio = target_ratios_by_token.get(
                            target_token_addr, 1.0 / len(required_tokens)
                        )

                        self._add_bridge_swap_action(
                            bridge_swap_actions,
                            source_token,
                            dest_chain,
                            target_token_addr,
                            target_token_symbol,
                            relative_funds_percentage * target_ratio,
                        )

                # Step 2c: Rebalance existing required tokens to achieve target ratios
                for token in dest_chain_tokens:
                    token_addr = token.get("token")
                    token_value = float(token.get("value", 0))

                    # Skip if this token is not required
                    if not any(
                        req_addr == token_addr for req_addr, _ in required_tokens
                    ):
                        continue

                    # Find the target value for this token
                    target_value = target_values.get(token_addr, 0)

                    if token_value > target_value:
                        # This token has surplus, need to swap excess
                        surplus = token_value - target_value
                        # Process all surplus amounts for precise CL pool ratios
                        # Find a token that needs more value
                        for other_req_addr, other_req_symbol in required_tokens:
                            if other_req_addr != token_addr:
                                other_token = available_tokens_map.get(other_req_addr)
                                if other_token:
                                    other_value = float(other_token.get("value", 0))
                                    other_target = target_values.get(other_req_addr, 0)
                                    if other_value < other_target:
                                        # Swap surplus to this token
                                        swap_fraction = min(1.0, surplus / token_value)
                                        self._add_bridge_swap_action(
                                            bridge_swap_actions,
                                            token,
                                            dest_chain,
                                            other_req_addr,
                                            other_req_symbol,
                                            swap_fraction,
                                        )
                                        break

        return bridge_swap_actions

    def _handle_all_tokens_needed(
        self,
        tokens: List[Dict[str, Any]],
        required_tokens: List[Tuple[str, str]],
        dest_chain: str,
        relative_funds_percentage: float,
        target_ratios_by_token: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Handle the case where we need all tokens with optimal capital allocation."""
        bridge_swap_actions = []

        # Filter tokens on destination chain
        dest_chain_tokens = [
            token for token in tokens if token.get("chain") == dest_chain
        ]
        other_chain_tokens = [
            token for token in tokens if token.get("chain") != dest_chain
        ]

        # Create mapping of available tokens on destination chain
        available_tokens_map = {
            token.get("token"): token for token in dest_chain_tokens
        }

        # Identify which required tokens we already have on destination chain
        available_required_tokens = []
        missing_required_tokens = []

        for req_token_addr, req_token_symbol in required_tokens:
            if req_token_addr in available_tokens_map:
                available_required_tokens.append((req_token_addr, req_token_symbol))
            else:
                missing_required_tokens.append((req_token_addr, req_token_symbol))

        # Step 1: Handle tokens from other chains first
        for token in other_chain_tokens:
            # Distribute to missing tokens first, then available ones
            target_tokens = (
                missing_required_tokens
                if missing_required_tokens
                else available_required_tokens
            )
            if target_tokens:
                dest_token_address, dest_token_symbol = target_tokens[
                    0
                ]  # Use first missing token
                token_percentage = (
                    relative_funds_percentage
                    * target_ratios_by_token.get(
                        dest_token_address, 1.0 / max(1, len(required_tokens))
                    )
                )
                self._add_bridge_swap_action(
                    bridge_swap_actions,
                    token,
                    dest_chain,
                    dest_token_address,
                    dest_token_symbol,
                    token_percentage,
                )

        # Step 2: Handle optimal allocation for tokens on destination chain
        if dest_chain_tokens:
            # Calculate total available value on destination chain
            total_dest_value = sum(
                float(token.get("value", 0)) for token in dest_chain_tokens
            )

            if total_dest_value > 0:
                # Calculate target values for each required token
                target_values = {}
                for req_token_addr, _req_token_symbol in required_tokens:
                    target_ratio = target_ratios_by_token.get(
                        req_token_addr, 1.0 / len(required_tokens)
                    )
                    target_values[req_token_addr] = total_dest_value * target_ratio

                # Identify unnecessary tokens (not in required tokens)
                unnecessary_tokens = []
                for token in dest_chain_tokens:
                    token_addr = token.get("token")
                    if not any(
                        req_addr == token_addr for req_addr, _ in required_tokens
                    ):
                        unnecessary_tokens.append(token)

                # Step 2a: Convert all unnecessary tokens to required tokens (distributed by target ratios)
                if unnecessary_tokens and missing_required_tokens:
                    for token in unnecessary_tokens:
                        # Distribute the unnecessary token across missing required tokens based on target ratios
                        for (
                            target_token_addr,
                            target_token_symbol,
                        ) in missing_required_tokens:
                            target_ratio = target_ratios_by_token.get(
                                target_token_addr, 1.0 / len(missing_required_tokens)
                            )
                            self._add_bridge_swap_action(
                                bridge_swap_actions,
                                token,
                                dest_chain,
                                target_token_addr,
                                target_token_symbol,
                                target_ratio,  # Convert based on target ratio
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

        # Calculate aggregate ratios directly from individual band requirements
        token_requirements = opportunity.get("token_requirements", {}) or {}
        position_requirements = token_requirements.get("position_requirements", [])

        # Default to 50/50 if not a CL pool or position requirements are missing
        is_cl_pool = bool(opportunity.get("is_cl_pool", False))
        if (not is_cl_pool) or (not position_requirements):
            aggregate_token0_ratio = 0.5
            aggregate_token1_ratio = 0.5
        else:
            (
                aggregate_token0_ratio,
                aggregate_token1_ratio,
            ) = self._calculate_aggregate_token_ratios(position_requirements)

        target_ratios_by_token = {
            opportunity.get("token0"): aggregate_token0_ratio,
            opportunity.get("token1"): aggregate_token1_ratio,
        }

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
                tokens,
                required_tokens,
                dest_chain,
                relative_funds_percentage,
                target_ratios_by_token,
            )
        elif len(tokens_we_need) < len(required_tokens):
            # We have some tokens but not all
            return self._handle_some_tokens_available(
                tokens,
                required_tokens,
                tokens_we_need,
                dest_chain,
                relative_funds_percentage,
                target_ratios_by_token,
            )
        else:
            # We need all tokens
            return self._handle_all_tokens_needed(
                tokens,
                required_tokens,
                dest_chain,
                relative_funds_percentage,
                target_ratios_by_token,
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
                    "source_initial_balance": token.get("balance", 0),
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
                    "source_initial_balance": token.get("balance", 0),
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
            # Add TiP-required data for cost calculation
            "opportunity_apr": opportunity.get("apr", 0),  # Percentage
            "percent_in_bounds": opportunity.get(
                "percent_in_bounds", 0.0
            ),  # For CL pools
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

    def _initialize_entry_costs_for_new_position(
        self, enter_pool_action: Dict[str, Any]
    ) -> Generator[None, None, None]:
        """Initialize entry costs for a new position when actions are formed in EvaluateStrategy"""
        try:
            chain = enter_pool_action.get("chain")
            pool_address = enter_pool_action.get("pool_address")

            if chain and pool_address:
                # Initialize entry costs in KV store
                yield from self._initialize_position_entry_costs(chain, pool_address)

                self.context.logger.info(
                    f"Initialized entry costs for new position: {pool_address}"
                )
            else:
                self.context.logger.warning(
                    "Missing chain or pool_address for entry cost initialization"
                )

        except Exception as e:
            self.context.logger.error(
                f"Error initializing entry costs for new position: {e}"
            )

    def _initialize_position_entry_costs(
        self, chain: str, position_id: str
    ) -> Generator[None, None, None]:
        """Initialize entry costs for a new position"""
        try:
            yield from self._store_entry_costs(chain, position_id, 0.0)
            self.context.logger.info(
                f"Initialized entry costs for position: {chain}_{position_id}"
            )
        except Exception as e:
            self.context.logger.error(f"Error initializing position entry costs: {e}")

    # ==================== STAKING HELPER METHODS ====================

    def _build_stake_lp_tokens_action(
        self, opportunity: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build action for staking LP tokens after liquidity provision."""
        try:
            chain = opportunity.get("chain")
            pool_address = opportunity.get("pool_address")
            dex_type = opportunity.get("dex_type")
            is_cl_pool = opportunity.get("is_cl_pool", False)

            # Only create staking actions for Velodrome pools
            if dex_type != "velodrome":
                self.context.logger.info(
                    f"Skipping staking for non-Velodrome pool: {dex_type}"
                )
                return None

            if not all([chain, pool_address]):
                self.context.logger.error(
                    "Missing required parameters for staking action"
                )
                return None

            # Determine staking action type based on pool type
            if is_cl_pool:
                # For CL pools, we need recipient and token_id information
                safe_address = self.params.safe_contract_addresses.get(chain)
                if not safe_address:
                    self.context.logger.error(
                        f"No safe address found for chain {chain}"
                    )
                    return None

                stake_action = {
                    "action": Action.STAKE_LP_TOKENS.value,
                    "dex_type": dex_type,
                    "chain": chain,
                    "pool_address": pool_address,
                    "is_cl_pool": True,
                    "recipient": safe_address,
                    "description": f"Stake CL LP tokens for {pool_address}",
                }
            else:
                # For regular pools
                stake_action = {
                    "action": Action.STAKE_LP_TOKENS.value,
                    "dex_type": dex_type,
                    "chain": chain,
                    "pool_address": pool_address,
                    "is_cl_pool": False,
                    "description": f"Stake LP tokens for {pool_address}",
                }

            self.context.logger.info(f"Created stake LP tokens action: {stake_action}")
            return stake_action

        except Exception as e:
            self.context.logger.error(
                f"Error building stake LP tokens action: {str(e)}"
            )
            return None

    def _build_claim_staking_rewards_action(
        self, position: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build action for claiming staking rewards before position exit."""
        try:
            chain = position.get("chain")
            pool_address = position.get("pool_address")
            dex_type = position.get("dex_type")
            is_cl_pool = position.get("is_cl_pool", False)
            gauge_address = position.get("gauge_address")

            # Only create claim actions for Velodrome pools with staking metadata
            if dex_type != "velodrome":
                self.context.logger.info(
                    f"Skipping reward claim for non-Velodrome pool: {dex_type}"
                )
                return None

            if not all([chain, pool_address]):
                self.context.logger.error(
                    "Missing required parameters for claim rewards action"
                )
                return None

            safe_address = self.params.safe_contract_addresses.get(chain)
            if not safe_address:
                self.context.logger.error(f"No safe address found for chain {chain}")
                return None

            claim_action = {
                "action": Action.CLAIM_STAKING_REWARDS.value,
                "dex_type": dex_type,
                "chain": chain,
                "pool_address": pool_address,
                "is_cl_pool": is_cl_pool,
                "account": safe_address,
                "description": f"Claim staking rewards for {pool_address}",
            }

            # Add gauge address if available
            if gauge_address:
                claim_action["gauge_address"] = gauge_address

            self.context.logger.info(
                f"Created claim staking rewards action: {claim_action}"
            )
            return claim_action

        except Exception as e:
            self.context.logger.error(
                f"Error building claim staking rewards action: {str(e)}"
            )
            return None

    def _get_gauge_address_for_position(
        self, position: Dict[str, Any]
    ) -> Generator[None, None, Optional[str]]:
        """Get gauge address for a position using voter contract."""
        try:
            chain = position.get("chain")
            pool_address = position.get("pool_address")

            if not all([chain, pool_address]):
                self.context.logger.error(
                    "Missing chain or pool_address for gauge lookup"
                )
                return None

            # Get voter contract address
            voter_address = self.params.velodrome_voter_contract_addresses.get(
                chain, ""
            )
            if not voter_address:
                self.context.logger.error(
                    f"No voter contract address found for chain {chain}"
                )
                return None

            # Use the Velodrome pool behaviour to get gauge address
            pool_behaviour = self.pools.get("velodrome")
            if not pool_behaviour:
                self.context.logger.error("Velodrome pool behaviour not found")
                return None

            # Get gauge address using the pool behaviour method
            gauge_address = yield from pool_behaviour.get_gauge_address(
                self, lp_token=pool_address, chain=chain
            )

            if gauge_address:
                self.context.logger.info(
                    f"Found gauge address {gauge_address} for pool {pool_address}"
                )
                return gauge_address
            else:
                self.context.logger.warning(f"No gauge found for pool {pool_address}")
                return None

        except Exception as e:
            self.context.logger.error(f"Error getting gauge address: {str(e)}")
            return None

    def _should_add_staking_actions(self, opportunity: Dict[str, Any]) -> bool:
        """Determine if staking actions should be added for this opportunity."""
        dex_type = opportunity.get("dex_type")

        # Only add staking for Velodrome pools
        if dex_type != "velodrome":
            return False

        # Check if chain has voter contract configured
        chain = opportunity.get("chain")
        voter_address = self.params.velodrome_voter_contract_addresses.get(chain, "")

        return bool(voter_address)

    def _calculate_aggregate_token_ratios(
        self, position_requirements: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """Calculate aggregate token ratios from individual band requirements.

        :param position_requirements: List of position requirements with allocation, token0_ratio, token1_ratio
        :return: Tuple of (aggregate_token0_ratio, aggregate_token1_ratio)
        """
        if not position_requirements:
            return 0.5, 0.5

        # Calculate aggregate ratios from individual band requirements
        total_token0_weight = sum(
            band.get("allocation", 0) * band.get("token0_ratio", 0.5)
            for band in position_requirements
        )
        total_token1_weight = sum(
            band.get("allocation", 0) * band.get("token1_ratio", 0.5)
            for band in position_requirements
        )

        # Normalize to get aggregate ratios
        total_weight = total_token0_weight + total_token1_weight
        if total_weight > 0:
            aggregate_token0_ratio = total_token0_weight / total_weight
            aggregate_token1_ratio = total_token1_weight / total_weight
        else:
            aggregate_token0_ratio = 0.5
            aggregate_token1_ratio = 0.5

        return aggregate_token0_ratio, aggregate_token1_ratio

    def _calculate_max_amounts_in(
        self,
        token0_balance: int,
        token1_balance: int,
        aggregate_token0_ratio: float,
        aggregate_token1_ratio: float,
        token0_symbol: str = "token0",
        token1_symbol: str = "token1",
    ) -> Tuple[List[int], str]:
        """Calculate max amounts in for pool entry based on available balances and target ratios.

        :param token0_balance: Available balance of token0
        :param token1_balance: Available balance of token1
        :param aggregate_token0_ratio: Target ratio for token0 (0.0 to 1.0)
        :param aggregate_token1_ratio: Target ratio for token1 (0.0 to 1.0)
        :param token0_symbol: Symbol for token0 (default: "token0")
        :param token1_symbol: Symbol for token1 (default: "token1")
        :return: Tuple of (max_amounts_in, limiting_token_symbol)
        """
        # Calculate total investment amount (sum of both tokens)
        total_investment = token0_balance + token1_balance

        # Calculate what the ideal ratio would require
        ideal_token0_for_all_tokens = int(total_investment * aggregate_token0_ratio)
        ideal_token1_for_all_tokens = int(total_investment * aggregate_token1_ratio)

        # Check which token is limiting
        if ideal_token0_for_all_tokens > token0_balance:
            # Token0 is limiting - scale down to match available token0
            scale_factor = token0_balance / ideal_token0_for_all_tokens
            max_amount0 = token0_balance
            max_amount1 = int(ideal_token1_for_all_tokens * scale_factor)
        elif ideal_token1_for_all_tokens > token1_balance:
            # Token1 is limiting - scale down to match available token1
            scale_factor = token1_balance / ideal_token1_for_all_tokens
            max_amount0 = int(ideal_token0_for_all_tokens * scale_factor)
            max_amount1 = token1_balance
        else:
            # We have enough of both tokens for the ideal ratio
            max_amount0 = ideal_token0_for_all_tokens
            max_amount1 = ideal_token1_for_all_tokens

        max_amounts_in = [max_amount0, max_amount1]
        log_message = (
            f"Using both tokens: {max_amount0} {token0_symbol} ({aggregate_token0_ratio*100:.2f}%), "
            f"{max_amount1} {token1_symbol} ({aggregate_token1_ratio*100:.2f}%)"
        )

        return max_amounts_in, log_message

    def _cache_enter_pool_action_for_cl_pool(
        self, opportunity: Dict[str, Any], enter_pool_action: Dict[str, Any]
    ) -> Generator[None, None, None]:
        """Cache the enter pool action for a Velodrome CL pool."""
        try:
            chain = opportunity.get("chain")
            if not chain:
                return

            # Read existing cache
            cached_data = yield from self._get_cached_cl_pool_data(chain)

            if not cached_data:
                self.context.logger.warning(
                    f"No existing cache found for chain {chain}, cannot cache enter pool action"
                )
                return

            # Update the cache with the enter pool action
            cached_data["enter_pool_action"] = enter_pool_action

            # Write back to KV store
            kv_key = f"velodrome_cl_pool_{chain}"
            yield from self._write_kv(
                {kv_key: json.dumps(cached_data, ensure_ascii=True)}
            )

            self.context.logger.info(
                f"Cached enter pool action for CL pool {cached_data.get('pool_address')} on {chain}"
            )

        except Exception as e:
            self.context.logger.error(f"Error caching enter pool action: {str(e)}")

    def _has_open_positions(self) -> bool:
        """Check if there are any open positions in current_pool.json"""
        try:
            if not self.current_positions:
                return False

            # Check if any position has status "open"
            for position in self.current_positions:
                if position.get("status") == "open":
                    return True

            return False
        except Exception as e:
            self.context.logger.error(f"Error checking open positions: {str(e)}")
            return False  # Default to False to allow cache usage

    def _check_and_use_cached_cl_opportunity(
        self,
    ) -> Generator[None, None, Optional[List[Dict[str, Any]]]]:
        """Check if we have a valid cached CL pool opportunity and reconstruct actions if so."""
        try:
            # Check if there are any open positions - if so, bypass cache
            # because we need to prepare exit pool actions
            if self._has_open_positions():
                self.context.logger.info(
                    "Open positions detected - bypassing cache to prepare exit pool actions"
                )
                return None

            # Check all target chains for cached opportunities
            for chain in self.params.target_investment_chains:
                cached_data = yield from self._get_cached_cl_pool_data(chain)

                if not cached_data:
                    continue

                # Validate cache
                should_use_cache = self._should_use_cached_cl_data(cached_data)

                if not should_use_cache:
                    self.context.logger.info(
                        f"Cached CL pool data for chain {chain} has expired"
                    )
                    continue

                # We have valid cached data, update round tracking
                yield from self._update_cl_pool_round_tracking(chain, cached_data)

                # Reconstruct actions from cached data
                actions = yield from self._reconstruct_actions_from_cached_cl_pool(
                    cached_data, chain
                )

                if actions:
                    self.context.logger.info(
                        f"Successfully reconstructed actions from cached CL pool on {chain}"
                    )
                    return actions

            # No valid cached opportunities found
            return None

        except Exception as e:
            self.context.logger.error(f"Error checking cached CL opportunity: {str(e)}")
            return None

    def _reconstruct_actions_from_cached_cl_pool(
        self, cached_data: Dict[str, Any], chain: str
    ) -> Generator[None, None, Optional[List[Dict[str, Any]]]]:
        """Reconstruct actions from cached CL pool data.

        NOTE: We do NOT recalculate swap actions because:
        - Swaps from previous round already executed successfully
        - We already have the right tokens on the target chain
        - We just need to retry the enter pool action with current balances

        :param cached_data: Cached pool data containing opportunity information
        :param chain: Chain identifier for the pool
        :yield: None: Generator yield for async operations
        :return: Optional list of reconstructed actions or None if reconstruction fails
        """
        try:
            actions = []

            # Get the cached enter pool action
            cached_enter_action = cached_data.get("enter_pool_action")
            if not cached_enter_action:
                self.context.logger.warning(
                    "No cached enter pool action found, cannot reconstruct"
                )
                return None

            # Get current token balances
            safe_address = self.params.safe_contract_addresses.get(chain)
            if not safe_address:
                self.context.logger.error(f"No safe address for chain {chain}")
                return None

            token0 = cached_data.get("token0")
            token1 = cached_data.get("token1")

            # Update the cached enter pool action with current balances
            updated_enter_action = cached_enter_action.copy()

            # Recalculate max_amounts_in based on current token balances
            # (tokens should already be on the right chain from previous swaps)
            token0_balance = (
                yield from self._get_token_balance(chain, safe_address, token0) or 0
            )
            token1_balance = (
                yield from self._get_token_balance(chain, safe_address, token1) or 0
            )

            self.context.logger.info(
                f"Current balances for cached pool: {token0_balance} {cached_data.get('token0_symbol')}, "
                f"{token1_balance} {cached_data.get('token1_symbol')}"
            )

            # Calculate aggregate ratios from individual band requirements
            position_requirements = cached_data.get("position_requirements", [])
            (
                aggregate_token0_ratio,
                aggregate_token1_ratio,
            ) = self._calculate_aggregate_token_ratios(position_requirements)

            # Calculate max_amounts_in based on aggregate ratios and current balances
            if aggregate_token0_ratio > 0.99:
                updated_enter_action["max_amounts_in"] = [token0_balance, 0]
            elif aggregate_token1_ratio > 0.99:
                updated_enter_action["max_amounts_in"] = [0, token1_balance]
            else:
                max_amounts_in, log_message = self._calculate_max_amounts_in(
                    token0_balance=token0_balance,
                    token1_balance=token1_balance,
                    aggregate_token0_ratio=aggregate_token0_ratio,
                    aggregate_token1_ratio=aggregate_token1_ratio,
                    token0_symbol=cached_data.get("token0_symbol", "token0"),
                    token1_symbol=cached_data.get("token1_symbol", "token1"),
                )

                updated_enter_action["max_amounts_in"] = max_amounts_in
                self.context.logger.info(log_message)

            self.context.logger.info(
                f"Updated max_amounts_in: {updated_enter_action['max_amounts_in']}"
            )

            actions.append(updated_enter_action)

            # Initialize entry costs for the position
            yield from self._initialize_entry_costs_for_new_position(
                updated_enter_action
            )

            # Add staking action if applicable
            temp_opportunity = {
                "chain": chain,
                "pool_address": cached_data.get("pool_address"),
                "token0": token0,
                "token1": token1,
                "dex_type": "velodrome",
                "is_cl_pool": True,
            }

            if self._should_add_staking_actions(temp_opportunity):
                stake_action = self._build_stake_lp_tokens_action(temp_opportunity)
                if stake_action:
                    actions.append(stake_action)

            return actions if actions else None

        except Exception as e:
            self.context.logger.error(
                f"Error reconstructing actions from cache: {str(e)}"
            )
            return None

    def _get_cached_cl_pool_data(
        self, chain: str
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Retrieve cached CL pool data from KV store."""
        kv_key = f"velodrome_cl_pool_{chain}"

        db_data = yield from self._read_kv(keys=(kv_key,))

        if db_data and db_data.get(kv_key):
            try:
                cached_data = json.loads(db_data[kv_key])

                # Check if cache was invalidated
                if cached_data.get("invalidated"):
                    self.context.logger.info(
                        f"Cache for chain {chain} was invalidated, treating as no cache"
                    )
                    return None

                self.context.logger.info(
                    f"Retrieved cached CL pool data for chain {chain}: pool={cached_data.get('pool_address')}"
                )
                return cached_data
            except (json.JSONDecodeError, TypeError) as e:
                self.context.logger.error(f"Error parsing cached CL pool data: {e}")
                return None

        return None

    def _should_use_cached_cl_data(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cached CL pool data is still valid."""
        current_timestamp = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()

        pool_finalization_timestamp = cached_data.get("pool_finalization_timestamp", 0)

        # Check time condition: more than 1 day (86400 seconds) has passed
        time_elapsed = current_timestamp - pool_finalization_timestamp
        time_expired = time_elapsed > 86400  # 1 day in seconds

        if time_expired:
            self.context.logger.info(
                f"Cache expired by time: {time_elapsed / 3600:.2f} hours elapsed (> 24 hours)"
            )
            return False

        self.context.logger.info(
            f"Cache valid: {time_elapsed / 3600:.2f} hours elapsed"
        )
        return True

    def _update_cl_pool_round_tracking(
        self, chain: str, cached_data: Dict[str, Any]
    ) -> Generator[None, None, None]:
        """Update round tracking for cached CL pool."""
        current_timestamp = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()

        last_round_timestamp = cached_data.get("last_round_timestamp", 0)
        round_count = cached_data.get("round_count", 0)

        # Increment round count if current timestamp > last_round_timestamp
        if current_timestamp > last_round_timestamp:
            round_count += 1
            cached_data["round_count"] = round_count
            cached_data["last_round_timestamp"] = current_timestamp

            self.context.logger.info(
                f"Updated CL pool tracking: round_count={round_count}, timestamp={current_timestamp}"
            )

            # Save updated data back to KV store
            kv_key = f"velodrome_cl_pool_{chain}"
            yield from self._write_kv(
                {kv_key: json.dumps(cached_data, ensure_ascii=True)}
            )

    def _cache_cl_pool_data(
        self,
        chain: str,
        pool_address: str,
        tick_spacing: int,
        tick_bands: List[Dict[str, Any]],
        current_price: float,
        percent_in_bounds: float,
        current_tick: Optional[int] = None,
        ema: Optional[List[float]] = None,
        std_dev: Optional[List[float]] = None,
        current_ema: Optional[float] = None,
        current_std_dev: Optional[float] = None,
        band_multipliers: Optional[List[float]] = None,
        token0: Optional[str] = None,
        token1: Optional[str] = None,
        token0_symbol: Optional[str] = None,
        token1_symbol: Optional[str] = None,
        token_requirements: Optional[Dict[str, Any]] = None,
        enter_pool_action: Optional[Dict[str, Any]] = None,
    ) -> Generator[None, None, None]:
        """Cache CL pool calculation results to KV store."""
        current_timestamp = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()

        cache_data = {
            "pool_address": pool_address,
            "chain": chain,
            "tick_spacing": tick_spacing,
            "tick_bands": tick_bands,
            "current_price": current_price,
            "percent_in_bounds": percent_in_bounds,
            "pool_finalization_timestamp": current_timestamp,
            "last_round_timestamp": current_timestamp,
            "round_count": 1,  # First round
        }

        # Add optional data if provided
        if current_tick is not None:
            cache_data["current_tick"] = current_tick
        if ema is not None:
            cache_data["ema"] = ema
        if std_dev is not None:
            cache_data["std_dev"] = std_dev
        if current_ema is not None:
            cache_data["current_ema"] = current_ema
        if current_std_dev is not None:
            cache_data["current_std_dev"] = current_std_dev
        if band_multipliers is not None:
            cache_data["band_multipliers"] = band_multipliers
        if token0 is not None:
            cache_data["token0"] = token0
        if token1 is not None:
            cache_data["token1"] = token1
        if token0_symbol is not None:
            cache_data["token0_symbol"] = token0_symbol
        if token1_symbol is not None:
            cache_data["token1_symbol"] = token1_symbol
        if token_requirements is not None:
            cache_data["token_requirements"] = token_requirements
        if enter_pool_action is not None:
            cache_data["enter_pool_action"] = enter_pool_action

        kv_key = f"velodrome_cl_pool_{chain}"

        yield from self._write_kv({kv_key: json.dumps(cache_data, ensure_ascii=True)})

        self.context.logger.info(
            f"Cached CL pool data for {pool_address} on chain {chain} "
            f"(tick_spacing={tick_spacing}, bands={len(tick_bands)}, price={current_price}, "
            f"current_tick={current_tick}, "
            f"enter_action_cached={enter_pool_action is not None})"
        )

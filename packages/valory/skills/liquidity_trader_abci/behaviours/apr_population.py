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

"""This module contains the behaviour for writing apr related data to database for the 'liquidity_trader_abci' skill."""

import decimal
import os
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Generator, Optional, Type

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    APR_UPDATE_INTERVAL,
    LiquidityTraderBaseBehaviour,
    PositionStatus,
)
from packages.valory.skills.liquidity_trader_abci.payloads import APRPopulationPayload
from packages.valory.skills.liquidity_trader_abci.states.apr_population import (
    APRPopulationRound,
)


class APRPopulationBehaviour(LiquidityTraderBaseBehaviour):
    """Behavior for calculating and storing APR data in MirrorDB."""

    matching_round: Type[AbstractRound] = APRPopulationRound
    _initial_value = None
    _final_value = None

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            payload_context = "APR Population"

            try:
                # Check if we should calculate APR
                should_calculate = yield from self._should_calculate_apr()
                if should_calculate:
                    # Get or create required resources
                    agent_type = yield from self._get_or_create_agent_type(sender)
                    type_id = agent_type["type_id"]
                    self.context.logger.info(f"Using agent type: {agent_type}")

                    agent_registry = yield from self._get_or_create_agent_registry()
                    agent_id = agent_registry["agent_id"]
                    self.context.logger.info(f"Using agent: {agent_id}")

                    attr_def = yield from self._get_or_create_attr_def(
                        type_id, agent_id
                    )
                    attr_def_id = attr_def["attr_def_id"]
                    self.context.logger.info(f"Using attribute definition: {attr_def}")

                    # Calculate and store APR
                    yield from self._calculate_and_store_apr(agent_id, attr_def_id)

            except Exception as e:
                self.context.logger.error(f"Error in APRPopulationBehaviour: {str(e)}")
                payload_context = "APR Population Error"

            payload = APRPopulationPayload(sender=sender, context=payload_context)

            with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
                yield from self.send_a2a_transaction(payload)
                yield from self.wait_until_round_end()

            self.set_done()

    def _calculate_and_store_apr(
        self, agent_id: str, attr_def_id: str
    ) -> Generator[None, None, None]:
        """Calculate and store APR data."""
        # Get portfolio value
        portfolio_value = self.portfolio_data.get("portfolio_value", 0)

        # Create portfolio snapshot and calculate APR
        portfolio_snapshot = self._create_portfolio_snapshot()
        actual_apr_data = yield from self.calculate_actual_apr(portfolio_value)

        if not actual_apr_data:
            return

        total_actual_apr = actual_apr_data.get("total_actual_apr")
        adjusted_apr = actual_apr_data.get("adjusted_apr")

        if not total_actual_apr:
            return

        # Get agent_hash from environment
        agent_config = os.environ.get("AEA_AGENT", "")
        agent_hash = agent_config.split(":")[-1] if agent_config else "Not found"

        # Store APR data
        timestamp = self._get_current_timestamp()
        enhanced_data = {
            "apr": float(total_actual_apr),
            "adjusted_apr": float(adjusted_apr),
            "timestamp": timestamp,
            "portfolio_snapshot": portfolio_snapshot,
            "calculation_metrics": self._get_apr_calculation_metrics(),
            "first_investment_timestamp": self.current_positions[0].get("timestamp")
            if self.current_positions
            else None,
            "agent_hash": agent_hash,
            "volume": self.portfolio_data.get("volume"),
            "trading_type": self.shared_state.trading_type,
            "selected_protocols": self.shared_state.selected_protocols,
        }

        agent_attr = yield from self.create_agent_attribute(
            agent_id,
            attr_def_id,
            enhanced_data,
        )
        # Update the last calculation time in DB
        yield from self._write_kv({"last_apr_calculation": str(int(timestamp))})

        self.context.logger.info(f"Stored APR data: {agent_attr}")

    def _should_calculate_apr(self) -> Generator[bool, None, None]:
        """Check if enough time has passed since last APR calculation or if any investment has been made."""

        # Get last calculation time from DB
        data = yield from self._read_kv(keys=("last_apr_calculation",))
        last_calculation_time = None

        if data and data.get("last_apr_calculation"):
            try:
                last_calculation_time = int(data.get("last_apr_calculation"))
            except (ValueError, TypeError):
                self.context.logger.warning("Invalid last APR calculation time in DB")

        current_time = int(self._get_current_timestamp())

        # Check if any new positions have been opened since last calculation
        if self.current_positions:
            for position in self.current_positions:
                if position.get("status") == PositionStatus.OPEN.value:
                    position_timestamp = position.get(
                        "enter_timestamp"
                    ) or position.get("timestamp")
                    if position_timestamp and last_calculation_time:
                        if int(position_timestamp) > last_calculation_time:
                            self.context.logger.info(
                                "New position opened since last APR calculation"
                            )
                            return True

        if (
            not last_calculation_time
            or (current_time - last_calculation_time) >= APR_UPDATE_INTERVAL
        ):
            return True

        self.context.logger.info(
            f"Skipping APR calculation. Only {(current_time - last_calculation_time) // 60} minutes "
            f"passed since last calculation"
        )
        return False

    def _create_portfolio_snapshot(self) -> Dict[str, Any]:
        """Create a structured snapshot of the current portfolio state."""

        snapshot = {
            "portfolio": self._convert_decimals(self.portfolio_data),
            "positons": self._convert_decimals(self.current_positions),
        }

        return snapshot

    def _to_decimal(self, value) -> Optional[Decimal]:
        """Safely convert a value to Decimal, handling None, float, int, and Decimal inputs."""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (ValueError, TypeError, decimal.InvalidOperation):
            self.context.logger.warning(f"Could not convert {value} to Decimal")
            return None

    def _convert_decimals(self, data: Any) -> Any:
        """Recursively convert Decimal objects to float in a data structure."""
        if isinstance(data, dict):
            return {key: self._convert_decimals(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_decimals(item) for item in data]
        elif isinstance(data, Decimal):
            return float(data)
        else:
            return data

    def _get_apr_calculation_metrics(self) -> Dict[str, Any]:
        """Extract and structure the key metrics used in APR calculation with robust handling."""

        initial_decimal = self._to_decimal(self._initial_value)
        final_decimal = self._to_decimal(self._final_value)

        metrics = {
            "initial_value": float(initial_decimal)
            if initial_decimal is not None
            else None,
            "final_value": float(final_decimal) if final_decimal is not None else None,
            "f_i_ratio": None,
            "first_investment_timestamp": None,
            "time_ratio": None,
        }

        # Calculate value change percentage with zero division protection
        if (
            initial_decimal is not None
            and final_decimal is not None
            and initial_decimal > 0
        ):
            f_i_ratio_decimal = (final_decimal / initial_decimal) - Decimal("1")
            metrics["f_i_ratio"] = float(f_i_ratio_decimal.quantize(Decimal("0.0001")))

        # Calculate time metrics with improved accuracy
        first_investment_timestamp = self._get_first_investment_timestamp()
        if first_investment_timestamp:
            metrics["first_investment_timestamp"] = first_investment_timestamp

            current_timestamp = self._get_current_timestamp()
            hours_raw = (current_timestamp - int(first_investment_timestamp)) / 3600
            hours_decimal = Decimal(str(hours_raw))

            # Use minimum threshold of 1 minute (0.0167 hours) consistent with _calculate_apr
            MIN_HOURS = Decimal("0.0167")  # 1 minute

            if hours_decimal < MIN_HOURS:
                hours = MIN_HOURS
                metrics["volatility_warning"] = "VERY_HIGH"
            elif hours_decimal < Decimal("1"):
                hours = hours_decimal
                metrics["volatility_warning"] = "HIGH"
            elif hours_decimal < Decimal("24"):
                hours = hours_decimal
                metrics["volatility_warning"] = "MEDIUM"
            else:
                hours = hours_decimal
                metrics["volatility_warning"] = "LOW"

            hours_in_year = Decimal("8760")
            time_ratio = hours_in_year / hours
            metrics["time_ratio"] = float(time_ratio.quantize(Decimal("0.0001")))
            metrics["actual_hours"] = float(hours_raw)
            metrics["calculation_hours"] = float(hours)

        return metrics

    def sign_message(self, message) -> Generator[None, None, Optional[str]]:
        """Sign a message."""
        message_bytes = message.encode("utf-8")
        signature = yield from self.get_signature(message_bytes)
        if signature:
            signature_hex = signature[2:]
            return signature_hex
        return None

    # =========================================================================
    # APR Calculation methods
    # =========================================================================

    # =========================================================================
    # APR Calculation methods
    # =========================================================================

    def calculate_actual_apr(
        self, portfolio_value: float
    ) -> Generator[None, None, Dict[str, float]]:
        """Calculate the actual APR for the portfolio based on current positions."""
        result = {}

        if not self._has_valid_portfolio_data():
            return None

        self._final_value = portfolio_value
        current_timestamp = self._get_current_timestamp()

        # Use the stored initial investment value if available
        initial_value = self.get_stored_initial_investment()
        if not initial_value:
            self.context.logger.error("No current investment")
            return None

        self._initial_value = initial_value
        self.context.logger.info(f"Using initial investment value: {initial_value}")

        first_investment_timestamp = self._get_first_investment_timestamp()
        self._calculate_apr(current_timestamp, first_investment_timestamp, result)

        yield from self._adjust_apr_for_eth_price(result, first_investment_timestamp)

        return result

    def get_stored_initial_investment(self) -> Optional[float]:
        """Get the initial investment value from the portfolio data."""
        if not self.portfolio_data:
            return None

        initial_investment = self.portfolio_data.get("initial_investment")
        if initial_investment is not None:
            self.context.logger.info(
                f"Found stored initial investment: {initial_investment}"
            )
            return float(initial_investment)

        return None

    def _has_valid_portfolio_data(self) -> bool:
        if (
            not self.portfolio_data
            or not hasattr(self, "portfolio_data")
            or "portfolio_value" not in self.portfolio_data
        ):
            self.context.logger.info("Missing required data for APR calculation")
            return False
        return True

    def _get_first_investment_timestamp(self) -> Optional[int]:
        """Get the first available timestamp from positions."""
        for position in self.current_positions:
            timestamp = position.get("timestamp") or position.get("enter_timestamp")
            if timestamp:
                return timestamp

        # quite a few early positions don't have any timestamp
        return int(self._get_current_timestamp())

    def _calculate_apr(
        self,
        current_timestamp: int,
        first_investment_timestamp: int,
        result,
    ):
        """Calculate APR with robust error handling and precision."""

        # Convert to Decimal for precise calculation
        final_value_decimal = self._to_decimal(self._final_value)
        initial_value_decimal = self._to_decimal(self._initial_value)

        # Comprehensive validation
        if final_value_decimal is None or final_value_decimal <= 0:
            self.context.logger.warning("Final value is zero, negative, or invalid")
            return 0.0

        if initial_value_decimal is None or initial_value_decimal <= 0:
            self.context.logger.error(
                "Initial value is zero, negative, or invalid - cannot calculate APR"
            )
            return 0.0

        # Perform calculation in Decimal precision
        f_i_ratio = (final_value_decimal / initial_value_decimal) - Decimal("1")

        # Calculate actual hours with improved accuracy
        hours_raw = (current_timestamp - int(first_investment_timestamp)) / 3600
        hours_decimal = Decimal(str(hours_raw))

        # Use minimum threshold of 1 minute (0.0167 hours) with volatility warning
        MIN_HOURS = Decimal("0.0167")  # 1 minute

        if hours_decimal < MIN_HOURS:
            self.context.logger.warning(
                f"Very short investment period: {float(hours_decimal * 60):.1f} minutes. "
                f"APR calculation may be highly volatile and should be interpreted with caution."
            )
            hours = MIN_HOURS
            volatility_warning = True
        elif hours_decimal < Decimal("1"):
            self.context.logger.info(
                f"Short investment period: {float(hours_decimal * 60):.1f} minutes. "
                f"APR calculation may show high volatility."
            )
            hours = hours_decimal
            volatility_warning = True
        else:
            hours = hours_decimal
            volatility_warning = False

        self.context.logger.info(
            f"Hours since investment: {float(hours)} (raw: {hours_raw})"
        )
        if volatility_warning:
            self.context.logger.info(
                "APR calculation includes volatility warning due to short time period"
            )

        hours_in_year = Decimal("8760")
        time_ratio = hours_in_year / hours

        apr_decimal = f_i_ratio * time_ratio * Decimal("100")

        # Handle negative APR case
        if apr_decimal < 0:
            apr_decimal = (
                final_value_decimal / initial_value_decimal - Decimal("1")
            ) * Decimal("100")

        # Convert to float for storage, with proper rounding
        apr = float(apr_decimal.quantize(Decimal("0.01")))

        self.context.logger.info(f"Calculated APR: {apr}")
        if apr:
            result["total_actual_apr"] = apr

    def _adjust_apr_for_eth_price(
        self, result: Dict[str, float], first_investment_timestamp: int
    ) -> Generator[None, None, None]:
        """Adjust APR for ETH price changes with robust type handling and zero division protection."""

        date_str = datetime.utcfromtimestamp(first_investment_timestamp).strftime(
            "%d-%m-%Y"
        )

        current_eth_price = yield from self._fetch_zero_address_price()
        start_eth_price = yield from self._fetch_historical_token_price(
            coingecko_id="ethereum", date_str=date_str
        )

        if (
            current_eth_price is not None
            and start_eth_price is not None
            and result.get("total_actual_apr") is not None
        ):
            # Convert all values to Decimal for precise calculation
            current_price_decimal = self._to_decimal(current_eth_price)
            start_price_decimal = self._to_decimal(start_eth_price)
            total_apr_decimal = self._to_decimal(result["total_actual_apr"])

            # Validate all conversions succeeded
            if all(
                val is not None
                for val in [
                    current_price_decimal,
                    start_price_decimal,
                    total_apr_decimal,
                ]
            ):
                # Check for zero division in price calculation
                if start_price_decimal <= 0:
                    self.context.logger.warning(
                        "Start ETH price is zero or negative, skipping adjustment"
                    )
                    return

                # Perform calculation in Decimal precision
                adjustment_factor = Decimal("1") - (
                    current_price_decimal / start_price_decimal
                )
                adjusted_apr_decimal = total_apr_decimal + (
                    adjustment_factor * Decimal("100")
                )

                # Store results with proper rounding
                result["adjusted_apr"] = float(
                    adjusted_apr_decimal.quantize(Decimal("0.01"))
                )
                result["adjustment_factor"] = float(
                    adjustment_factor.quantize(Decimal("0.0001"))
                )
                result["current_price"] = float(current_price_decimal)
                result["initial_price"] = float(start_price_decimal)

                self.context.logger.info(f"Adjusted APR: {result['adjusted_apr']}%")
            else:
                self.context.logger.warning(
                    "Could not convert price data to Decimal, skipping adjustment"
                )

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

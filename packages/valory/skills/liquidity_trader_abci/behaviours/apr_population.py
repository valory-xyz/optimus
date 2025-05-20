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

import json
import os
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Generator, Optional, Type

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    AGENT_TYPE,
    APR_UPDATE_INTERVAL,
    LiquidityTraderBaseBehaviour,
    METRICS_NAME,
    METRICS_TYPE,
)
from packages.valory.skills.liquidity_trader_abci.states.apr_population import (
    APRPopulationPayload,
    APRPopulationRound,
)


class APRPopulationBehaviour(LiquidityTraderBaseBehaviour):
    """Behavior for calculating and storing APR data in MirrorDB."""

    matching_round: Type[AbstractRound] = APRPopulationRound
    _initial_value = None
    _final_value = None

    def async_act(self) -> Generator:
        """Execute the APR population behavior."""
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

                    agent_registry = yield from self._get_or_create_agent_registry(
                        sender, type_id
                    )
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

    def _get_or_create_agent_registry(
        self, eth_address: str, type_id: str
    ) -> Generator[Dict[str, Any], None, None]:
        """Get or create agent registry entry."""
        data = yield from self._read_kv(keys=("agent_registry",))
        if not data or not data.get("agent_registry"):
            agent_registry = yield from self.get_agent_registry_by_address(eth_address)
            if not agent_registry:
                agent_name = self.generate_name(eth_address)
                self.context.logger.info(f"agent_name : {agent_name}")
                agent_registry = yield from self.create_agent_registry(
                    agent_name, type_id, eth_address
                )
                if not agent_registry:
                    raise Exception("Failed to create agent registry.")
                yield from self._write_kv(
                    {"agent_registry": json.dumps(agent_registry)}
                )
            return agent_registry

        return json.loads(data["agent_registry"])

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
        agent_config = os.environ.get('AEA_AGENT', '')
        agent_hash = agent_config.split(':')[-1] if agent_config else 'Not found'

        # Store APR data
        timestamp = self._get_current_timestamp()
        enhanced_data = {
            "apr": float(total_actual_apr),
            "adjusted_apr": float(adjusted_apr),
            "timestamp": timestamp,
            "portfolio_snapshot": portfolio_snapshot,
            "calculation_metrics": self._get_apr_calculation_metrics(),
            "first_investment_timestamp": self.current_positions[0].get("timestamp"),
            "agent_hash": agent_hash,
            "volume": self.portfolio_data.get("volume")
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

        if len(self.current_positions) == 0:
            self.context.logger.info("No investments made by agent yet")
            return False

        # Get last calculation time from DB
        data = yield from self._read_kv(keys=("last_apr_calculation",))
        last_calculation_time = None

        if data and data.get("last_apr_calculation"):
            try:
                last_calculation_time = int(data.get("last_apr_calculation"))
            except (ValueError, TypeError):
                self.context.logger.warning("Invalid last APR calculation time in DB")

        current_time = int(self._get_current_timestamp())

        # If no last calculation or 30 minutes have passed
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
        """Extract and structure the key metrics used in APR calculation."""
        metrics = {
            "initial_value": (
                float(self._initial_value) if self._initial_value else None
            ),
            "final_value": float(self._final_value) if self._final_value else None,
            "f_i_ratio": None,
            "first_investment_timestamp": None,
            "time_ratio": None,
        }

        # Calculate value change percentage if we have both values
        if metrics["initial_value"] is not None and metrics["final_value"] is not None:
            metrics["f_i_ratio"] = (
                float(metrics["final_value"]) / float(metrics["initial_value"])
            ) - 1

        # Calculate time period and annualization factor
        first_investment_timestamp = self._get_first_investment_timestamp()
        if first_investment_timestamp:
            metrics["first_investment_timestamp"] = first_investment_timestamp

            current_timestamp = self._get_current_timestamp()
            hours = max(1, (current_timestamp - int(first_investment_timestamp)) / 3600)
            hours_in_year = 8760
            time_ratio = hours_in_year / hours
            metrics["time_ratio"] = float(time_ratio)

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
    # Agent Type API methods
    # =========================================================================

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

    # =========================================================================
    # Attribute Definition API methods
    # =========================================================================

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

    # =========================================================================
    # Agent Registry API methods
    # =========================================================================

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

    # =========================================================================
    # Agent Attribute API methods
    # =========================================================================

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
            # Fall back to calculating it if not available
            initial_value = yield from self.calculate_initial_investment()
            if not initial_value:
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
            not self.current_positions
            or not hasattr(self, "portfolio_data")
            or "portfolio_value" not in self.portfolio_data
        ):
            self.context.logger.info("Missing required data for APR calculation")
            return False
        return True

    def _get_first_investment_timestamp(self) -> Optional[int]:
        first_investment_timestamp = self.current_positions[0].get("timestamp")
        return first_investment_timestamp

    def _calculate_apr(
        self,
        current_timestamp: int,
        first_investment_timestamp: int,
        result,
    ):
        if self._final_value <= 0:
            self.context.logger.warning("Final value is zero or negative")
            return 0.0

        f_i_ratio = (self._final_value / self._initial_value) - 1
        hours = max(1, (current_timestamp - int(first_investment_timestamp)) / 3600)
        self.context.logger.info(f"Hours since investment: {hours}")

        hours_in_year = 8760
        time_ratio = hours_in_year / hours

        apr = float(f_i_ratio * time_ratio * 100)
        if apr < 0:
            apr = round(float((self._final_value / self._initial_value) - 1) * 100, 2)

        self.context.logger.info(f"Calculated APR: {apr}")
        if apr:
            result["total_actual_apr"] = float(round(apr, 2))

    def _adjust_apr_for_eth_price(
        self, result: Dict[str, float], first_investment_timestamp: int
    ) -> Generator[None, None, None]:
        date_str = datetime.utcfromtimestamp(first_investment_timestamp).strftime(
            "%d-%m-%Y"
        )

        current_eth_price = yield from self._fetch_zero_address_price()
        start_eth_price = yield from self._fetch_historical_token_price(
            coingecko_id="ethereum", date_str=date_str
        )

        if current_eth_price is not None and start_eth_price is not None:
            adjustment_factor = Decimal("1") - (
                Decimal(str(current_eth_price)) / Decimal(str(start_eth_price))
            )
            result["adjusted_apr"] = round(
                float(result["total_actual_apr"])
                + float(adjustment_factor * Decimal("100")),
                2,
            )
            result["adjustment_factor"] = float(adjustment_factor)
            result["current_price"] = current_eth_price
            result["initial_price"] = start_eth_price
            self.context.logger.info(f"Adjusted APR: {result['adjusted_apr']}%")

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

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
from datetime import datetime
from decimal import Decimal
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Type
)
from packages.valory.protocols.contract_api import ContractApiMessage

from packages.valory.skills.liquidity_trader_abci.rounds import (
    APRPopulationPayload,
    APRPopulationRound,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    LiquidityTraderBaseBehaviour,
    PositionStatus,
    METRICS_NAME,
    METRICS_TYPE,
    AGENT_TYPE
)
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.contracts.erc20.contract import ERC20

class APRPopulationBehaviour(LiquidityTraderBaseBehaviour):
    """Behavior for calculating and storing APR data in MirrorDB."""

    matching_round: Type[AbstractRound] = APRPopulationRound
    _initial_value = None
    _final_value = None

    def async_act(self) -> Generator:
        """
        Execute the APR population behavior.

        This behavior:
        1. Retrieves or creates necessary resources (agent type, agent registry, attribute definition)
        2. Calculates APR for positions
        3. Stores APR data in MirrorDB
        4. Reads APR data for comparison with other agents
        5. Completes the round
        """
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            try:
                # Get configuration
                eth_address = sender

                # Step 1: Get or create agent type for "Modius"
                data = yield from self._read_kv(keys=("agent_type",))
                if data:
                    agent_type = data.get("agent_type")
                    if agent_type:
                        agent_type = json.loads(agent_type)
                    else:
                        # Check external DB
                        agent_type = yield from self.get_agent_type_by_name(AGENT_TYPE)
                        if not agent_type:
                            agent_type = yield from self.create_agent_type(
                                AGENT_TYPE,
                                "An agent for DeFi liquidity management and APR tracking",
                            )
                            if not agent_type:
                                raise Exception("Failed to create agent type.")
                            yield from self._write_kv(
                                {"agent_type": json.dumps(agent_type)}
                            )

                type_id = agent_type.get("type_id")
                self.context.logger.info(f"Using agent type: {agent_type}")

                # Step 2: Get or create agent registry entry
                data = yield from self._read_kv(keys=("agent_registry",))
                if data:
                    agent_registry = data.get("agent_registry", "{}")
                    if agent_registry:
                        agent_registry = json.loads(agent_registry)
                    else:
                        agent_registry = yield from self.get_agent_registry_by_address(
                            eth_address
                        )
                        if not agent_registry:
                            agent_name = self.generate_name(sender)
                            self.context.logger.info(f"agent_name : {agent_name}")
                            agent_registry = yield from self.create_agent_registry(
                                agent_name, type_id, eth_address
                            )
                            if not agent_registry:
                                raise Exception("Failed to create agent registry.")
                            yield from self._write_kv(
                                {"agent_registry": json.dumps(agent_registry)}
                            )

                agent_id = agent_registry.get("agent_id")
                self.context.logger.info(f"Using agent: {agent_id}")

                # Step 3: Get or create APR attribute definition
                data = yield from self._read_kv(keys=("attr_def",))
                if data:
                    attr_def = data.get("attr_def", "{}")
                    if attr_def:
                        attr_def = json.loads(attr_def)
                    else:
                        # Check external DB
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
                                raise Exception(
                                    "Failed to create attribute definition."
                                )
                            yield from self._write_kv(
                                {"attr_def": json.dumps(attr_def)}
                            )

                attr_def_id = attr_def.get("attr_def_id")
                self.context.logger.info(f"Using attribute definition: {attr_def}")

                # Step 4: Calculate APR for positions
                portfolio_value = 0
                data = yield from self._read_kv(keys=("portfolio_value",))
                self.context.logger.info(f"data{data}")
                if data and data["portfolio_value"]:
                    self.context.logger.info(f"data{data}")
                    portfolio_value = float(data.get("portfolio_value", "0"))

                if not math.isclose(
                    portfolio_value,
                    self.portfolio_data["portfolio_value"],
                    rel_tol=1e-9,
                ):
                    # Create portfolio snapshot for debugging
                    portfolio_snapshot = self._create_portfolio_snapshot()

                    # Calculate APR and related metrics
                    actual_apr_data = yield from self.calculate_actual_apr(
                        agent_id, attr_def_id
                    )
                    if actual_apr_data:
                        total_actual_apr = actual_apr_data.get("total_actual_apr", None)
                        adjusted_apr = actual_apr_data.get("adjusted_apr", None)

                        if total_actual_apr:
                            # Step 5: Store enhanced APR data in MirrorDB
                            timestamp = int(
                                self.round_sequence.last_round_transition_timestamp.timestamp()
                            )

                            # Create enhanced data payload with portfolio metrics
                            enhanced_data = {
                                "apr": float(total_actual_apr),
                                "adjusted_apr": float(adjusted_apr),
                                "timestamp": timestamp,
                                "portfolio_snapshot": portfolio_snapshot,
                                "calculation_metrics": self._get_apr_calculation_metrics(),
                            }

                            agent_attr = yield from self.create_agent_attribute(
                                agent_id,
                                attr_def_id,
                                enhanced_data,
                            )
                            self.context.logger.info(f"Stored APR data: {agent_attr}")

                # Prepare payload for consensus
                payload = APRPopulationPayload(sender=sender, context="APR Population")

            except Exception as e:
                self.context.logger.error(f"Error in APRPopulationBehaviour: {str(e)}")
                # Create a payload even in case of error to continue the protocol
                payload = APRPopulationPayload(
                    sender=sender, context="APR Population Error"
                )
            payload = APRPopulationPayload(sender=sender, context="APR Population")

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    # =========================================================================
    # Utility methods
    # =========================================================================

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
            "last_investment_timestamp": None,
            "time_ratio": None,
        }

        # Calculate value change percentage if we have both values
        if metrics["initial_value"] is not None and metrics["final_value"] is not None:
            metrics["f_i_ratio"] = (
                float(metrics["final_value"]) / float(metrics["initial_value"])
            ) - 1

        # Calculate time period and annualization factor
        last_investment_timestamp = self._get_last_investment_timestamp()
        if last_investment_timestamp:
            metrics["last_investment_timestamp"] = last_investment_timestamp

            current_timestamp = self._get_current_timestamp()
            hours = max(1, (current_timestamp - int(last_investment_timestamp)) / 3600)
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

    def get_token_id_from_symbol_cached(
        self, symbol, token_name, coin_list
    ) -> Optional[str]:
        """Retrieve the CoinGecko token ID using the token's symbol and name."""
        # Try to find coins matching the symbol.
        candidates = [
            coin for coin in coin_list if coin["symbol"].lower() == symbol.lower()
        ]
        if not candidates:
            return None

        # If single candidate, return it.
        if len(candidates) == 1:
            return candidates[0]["id"]

        # If multiple candidates, match by name if possible.
        normalized_token_name = token_name.replace(" ", "").lower()
        for coin in candidates:
            coin_name = coin["name"].replace(" ", "").lower()
            if coin_name == normalized_token_name or coin_name == symbol.lower():
                return coin["id"]
        return None

    def get_token_id_from_symbol(
        self, token_address, symbol, coin_list, chain_name
    ) -> Generator[None, None, Optional[str]]:
        """Retrieve the CoinGecko token ID using the token's address, symbol, and chain name."""
        token_name = yield from self._fetch_token_name_from_contract(
            chain_name, token_address
        )
        if not token_name:
            matching_coins = [
                coin for coin in coin_list if coin["symbol"].lower() == symbol.lower()
            ]
            return matching_coins[0]["id"] if len(matching_coins) == 1 else None

        return self.get_token_id_from_symbol_cached(symbol, token_name, coin_list)

    def fetch_coin_list(self) -> Generator[None, None, Optional[List[Any]]]:
        """Fetches the list of coins from CoinGecko API only once."""
        url = "https://api.coingecko.com/api/v3/coins/list"
        response = yield from self.get_http_response("GET", url, None, None)

        try:
            response_json = json.loads(response.body)
            return response_json
        except json.decoder.JSONDecodeError as e:
            self.context.logger.error(f"Failed to fetch coin list: {e}")
            return None

    def _fetch_historical_token_prices(
        self, tokens: List[List[str]], date_str: str, chain: str
    ) -> Generator[None, None, Dict[str, float]]:
        """Fetch historical token prices for a specific date."""
        historical_prices = {}

        coin_list = yield from self.fetch_coin_list()
        if not coin_list:
            self.context.logger.error("Failed to fetch the coin list from CoinGecko.")
            return historical_prices

        for token_symbol, token_address in tokens:
            # Get CoinGecko ID.
            coingecko_id = yield from self.get_token_id_from_symbol(
                token_address, token_symbol, coin_list, chain
            )
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

    def calculate_actual_apr(
        self, portfolio_value: float
    ) -> Generator[None, None, Dict[str, float]]:
        """Calculate the actual APR for the portfolio based on current positions."""
        result = {}

        if not self._has_valid_portfolio_data():
            return None

        self._final_value = portfolio_value
        current_timestamp = self._get_current_timestamp()

        initial_value = yield from self._calculate_initial_value()
        if not initial_value:
            return None

        self._initial_value = initial_value

        time_since_investment = self._get_last_investment_timestamp()
        self._calculate_apr(current_timestamp, time_since_investment, result)

        yield from self._adjust_apr_for_eth_price(result, time_since_investment)

        return result

    def _has_valid_portfolio_data(self) -> bool:
        if (
            not self.current_positions
            or not hasattr(self, "portfolio_data")
            or "portfolio_value" not in self.portfolio_data
        ):
            self.context.logger.info("Missing required data for APR calculation")
            return False
        return True

    def _calculate_initial_value(
        self,
    ) -> Generator[None, None, Optional[float]]:
        initial_value = 0.0
        for position in self.current_positions:
            if position.get("status") == PositionStatus.OPEN.value:
                position_value = yield from self.calculate_initial_investment_value(
                    position
                )
                self.context.logger.info(f"Position value: {position_value}")
                if position_value is not None:
                    initial_value += float(position_value)
                else:
                    self.context.logger.warning(
                        f"Skipping position with null value: {position.get('id', 'unknown')}"
                    )

        self.context.logger.info(f"Total initial value: {initial_value}")
        if initial_value <= 0:
            self.context.logger.warning("Initial value is zero or negative")
            return None

        return initial_value

    def _get_last_investment_timestamp(self) -> Optional[int]:
        open_positions = [
            position
            for position in self.current_positions
            if position.get("status") == PositionStatus.OPEN.value
        ]
        if not open_positions:
            self.context.logger.warning(
                "No open positions found for timestamp retrieval"
            )
            return None

        last_open_position = open_positions[-1]
        time_since_investment = last_open_position.get("timestamp")

        return time_since_investment

    def _calculate_apr(
        self,
        current_timestamp: int,
        time_since_investment: int,
        result,
    ):
        if self._final_value <= 0:
            self.context.logger.warning("Final value is zero or negative")
            return 0.0

        f_i_ratio = (self._final_value / self._initial_value) - 1
        hours = max(1, (current_timestamp - int(time_since_investment)) / 3600)
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
        self, result: Dict[str, float], time_since_investment: int
    ) -> Generator[None, None, None]:
        date_str = datetime.utcfromtimestamp(time_since_investment).strftime("%d-%m-%Y")

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

    def _store_last_apr_timestamp(
        self, current_timestamp: int
    ) -> Generator[None, None, None]:
        yield from self._write_kv(
            {"last_apr_stored_timestamp": json.dumps(int(current_timestamp))}
        )

    def _is_apr_calculation_needed(
        self, current_timestamp: int, agent_id: int, attr_def_id: int
    ) -> Generator[None, None, bool]:
        """Check if APR calculation is needed based on the time gap."""
        # Get timestamp from the first position or from stored data
        last_apr_stored_timestamp = None
        data = yield from self._read_kv(keys=("last_apr_stored_timestamp",))
        if (
            data
            and "last_apr_stored_timestamp" in data
            and data["last_apr_stored_timestamp"] not in (None, "{}", "")
        ):
            try:
                last_apr_stored_timestamp = int(data.get("last_apr_stored_timestamp"))
                self.context.logger.info(
                    f"Using stored timestamp: {last_apr_stored_timestamp}"
                )
            except (ValueError, TypeError):
                self.context.logger.warning(
                    f"Invalid stored timestamp format: {data.get('timestamp')}"
                )
                last_apr_stored_timestamp = None
        else:
            time_data = yield from self._fetch_last_apr_data(agent_id, attr_def_id)
            # Fallback to position timestamp if stored timestamp is invalid
            if time_data:
                last_apr_stored_timestamp = time_data["timestamp"]
                self.context.logger.info(f"timestamp : {last_apr_stored_timestamp}")
            else:
                last_apr_stored_timestamp = self._get_last_investment_timestamp()

        if not last_apr_stored_timestamp:
            return False

        if (
            last_apr_stored_timestamp
            and (current_timestamp - last_apr_stored_timestamp) < 7200
        ):
            return False

        return True

    def calculate_initial_investment_value(
        self, position: Dict[str, Any]
    ) -> Generator[None, None, Optional[float]]:
        """Calculate the initial investment value based on the initial transaction."""

        chain = position.get("chain")
        token0 = position.get("token0")
        token1 = position.get("token1")
        amount0 = position.get("amount0")
        amount1 = position.get("amount1")
        timestamp = position.get("timestamp")

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

    def _fetch_last_apr_data(
        self, agent_id, attr_def_id
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Fetch the last stored APR data."""
        try:
            response = yield from self._call_mirrordb(
                method="read_",
                method_name="read_agent_attribute_by_agent_and_def",
                endpoint=f"api/agents/{agent_id}/attributes/",
            )
            # Assuming response is a list of attributes, sort by timestamp and get the latest
            if response and isinstance(response, list):
                filtered_response = [
                    entry for entry in response if entry["attr_def_id"] == attr_def_id
                ]
                # Sort the filtered list by timestamp in descending order
                filtered_response.sort(
                    key=lambda x: x["json_value"]["timestamp"], reverse=True
                )
                return filtered_response[0]["json_value"] if response else None
            return None
        except Exception as e:
            self.context.logger.error(f"Error fetching last APR data: {e}")
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


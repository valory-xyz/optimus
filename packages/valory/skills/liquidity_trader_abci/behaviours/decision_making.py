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

"""This module contains the behaviour for executing trades for the 'liquidity_trader_abci' skill."""

import json
from decimal import Decimal
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, cast

from eth_abi import decode
from eth_utils import keccak, to_bytes, to_checksum_address, to_hex

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
from packages.valory.contracts.sturdy_yearn_v3_vault.contract import (
    YearnV3VaultContract,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    Action,
    Decision,
    DexType,
    ERC20_DECIMALS,
    ETHER_VALUE,
    HTTP_NOT_FOUND,
    HTTP_OK,
    INTEGRATOR,
    LiquidityTraderBaseBehaviour,
    MAX_RETRIES_FOR_ROUTES,
    MAX_STEP_COST_RATIO,
    PositionStatus,
    SAFE_TX_GAS,
    SLEEP_TIME,
    SwapStatus,
    WAITING_PERIOD_FOR_BALANCE_TO_REFLECT,
    ZERO_ADDRESS,
)
from packages.valory.skills.liquidity_trader_abci.models import SharedState
from packages.valory.skills.liquidity_trader_abci.payloads import DecisionMakingPayload
from packages.valory.skills.liquidity_trader_abci.rounds import Event
from packages.valory.skills.liquidity_trader_abci.states.decision_making import (
    DecisionMakingRound,
)
from packages.valory.skills.liquidity_trader_abci.states.evaluate_strategy import (
    EvaluateStrategyRound,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
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
            if (
                self.synchronized_data.last_action == Action.ENTER_POOL.value
                or self.synchronized_data.last_action == Action.DEPOSIT.value
            ):
                yield from self._post_execute_enter_pool(
                    actions, last_executed_action_index
                )
            if (
                self.synchronized_data.last_action == Action.EXIT_POOL.value
                or self.synchronized_data.last_action == Action.WITHDRAW.value
            ):
                yield from self._post_execute_exit_pool(
                    actions, last_executed_action_index
                )
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
            res = self._update_assets_after_swap(actions, last_executed_action_index)
            return res

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

        # Add tokens to assets
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
        keys_to_extract = [
            "chain",
            "pool_address",
            "dex_type",
            "token0",
            "token1",
            "token0_symbol",
            "token1_symbol",
            "apr",
            "pool_type",
            "whitelistedSilos",
            "pool_id",
            "is_stable",
            "is_cl_pool",
        ]

        # Create the current_position dictionary with only the desired information
        current_position = {
            key: action[key] for key in keys_to_extract if key in action
        }

        if action.get("dex_type") == DexType.UNISWAP_V3.value:
            (
                token_id,
                liquidity,
                amount0,
                amount1,
                timestamp,
            ) = yield from self._get_data_from_mint_tx_receipt(
                self.synchronized_data.final_tx_hash, action.get("chain")
            )
            current_position["token_id"] = token_id
            current_position["liquidity"] = liquidity
            current_position["amount0"] = amount0
            current_position["amount1"] = amount1
            current_position["enter_timestamp"] = timestamp
            current_position["status"] = PositionStatus.OPEN.value
            current_position["enter_tx_hash"] = self.synchronized_data.final_tx_hash
            self.current_positions.append(current_position)

        elif action.get("dex_type") == DexType.VELODROME.value:
            is_cl_pool = action.get("is_cl_pool", False)
            if is_cl_pool:
                # For Velodrome CL pools, we need to handle multiple positions
                # First, check if there are multiple positions in the transaction
                positions_data = yield from self._get_all_positions_from_tx_receipt(
                    self.synchronized_data.final_tx_hash, action.get("chain")
                )

                if positions_data and len(positions_data) > 0:
                    # We have multiple positions for the same pool
                    # Create a single entry with nested positions
                    current_position["status"] = PositionStatus.OPEN.value
                    current_position[
                        "enter_tx_hash"
                    ] = self.synchronized_data.final_tx_hash
                    current_position["enter_timestamp"] = positions_data[0][
                        4
                    ]  # Use timestamp from first position

                    # Calculate total amounts across all positions
                    total_amount0 = sum(
                        position_data[2] for position_data in positions_data
                    )
                    total_amount1 = sum(
                        position_data[3] for position_data in positions_data
                    )
                    current_position["amount0"] = total_amount0
                    current_position["amount1"] = total_amount1

                    # Store individual position details in a nested structure
                    positions = []
                    for position_data in positions_data:
                        token_id, liquidity, amount0, amount1, _ = position_data
                        positions.append(
                            {
                                "token_id": token_id,
                                "liquidity": liquidity,
                                "amount0": amount0,
                                "amount1": amount1,
                            }
                        )

                    # Add the positions list to the current_position
                    current_position["positions"] = positions

                    # Add to current_positions
                    self.current_positions.append(current_position)

                    self.context.logger.info(
                        f"Added Velodrome CL pool with {len(positions)} positions to pool {current_position['pool_address']}"
                    )
                else:
                    # Fallback to single position handling
                    (
                        token_id,
                        liquidity,
                        amount0,
                        amount1,
                        timestamp,
                    ) = yield from self._get_data_from_mint_tx_receipt(
                        self.synchronized_data.final_tx_hash, action.get("chain")
                    )
                    current_position["enter_timestamp"] = timestamp
                    current_position["status"] = PositionStatus.OPEN.value
                    current_position[
                        "enter_tx_hash"
                    ] = self.synchronized_data.final_tx_hash

                    # For consistency, also add a positions list with a single entry
                    current_position["positions"] = [
                        {
                            "token_id": token_id,
                            "liquidity": liquidity,
                            "amount0": amount0,
                            "amount1": amount1,
                        }
                    ]

                    self.current_positions.append(current_position)
            else:
                # For non-CL Velodrome pools
                (
                    amount0,
                    amount1,
                    timestamp,
                ) = yield from self._get_data_from_velodrome_mint_event(
                    self.synchronized_data.final_tx_hash, action.get("chain")
                )
                current_position["amount0"] = amount0
                current_position["amount1"] = amount1
                current_position["enter_timestamp"] = timestamp
                current_position["status"] = PositionStatus.OPEN.value
                current_position["enter_tx_hash"] = self.synchronized_data.final_tx_hash
                self.current_positions.append(current_position)

        elif action.get("dex_type") == DexType.BALANCER.value:
            (
                amount0,
                amount1,
                timestamp,
            ) = yield from self._get_data_from_join_pool_tx_receipt(
                self.synchronized_data.final_tx_hash, action.get("chain")
            )
            current_position["amount0"] = amount0
            current_position["amount1"] = amount1
            current_position["enter_timestamp"] = timestamp
            current_position["status"] = PositionStatus.OPEN.value
            current_position["enter_tx_hash"] = self.synchronized_data.final_tx_hash
            self.current_positions.append(current_position)

        elif action.get("dex_type") == DexType.STURDY.value:
            (
                amount,
                shares,
                timestamp,
            ) = yield from self._get_data_from_deposit_tx_receipt(
                self.synchronized_data.final_tx_hash, action.get("chain")
            )
            current_position["amount0"] = amount
            current_position["shares"] = shares
            current_position["enter_timestamp"] = timestamp
            current_position["status"] = PositionStatus.OPEN.value
            current_position["enter_tx_hash"] = self.synchronized_data.final_tx_hash
            self.current_positions.append(current_position)

        self.store_current_positions()
        self.context.logger.info(
            f"Enter pool was successful! Updated current positions for pool {current_position['pool_address']}"
        )

    def _post_execute_exit_pool(self, actions, last_executed_action_index):
        """Handle exiting a pool."""
        action = actions[last_executed_action_index]
        pool_address = action.get("pool_address")
        is_cl_pool = action.get("is_cl_pool", False)
        dex_type = action.get("dex_type")

        # Find all positions with the matching pool address and update their status
        for position in self.current_positions:
            if position.get("pool_address") == pool_address:
                position["status"] = PositionStatus.CLOSED.value
                position["exit_tx_hash"] = self.synchronized_data.final_tx_hash
                position["exit_timestamp"] = int(self._get_current_timestamp())

                # For Velodrome CL pools, log all the positions that were exited
                if (
                    dex_type == DexType.VELODROME.value
                    and is_cl_pool
                    and "positions" in position
                ):
                    position_ids = [
                        pos["token_id"] for pos in position.get("positions", [])
                    ]
                    self.context.logger.info(
                        f"Closed Velodrome CL pool with {len(position_ids)} positions. "
                        f"Token IDs: {position_ids}"
                    )
                else:
                    self.context.logger.info(f"Closing position: {position}")

        self.store_current_positions()
        self.context.logger.info("Exit was successful! Updated positions.")
        # When we exit the pool, it may take time to reflect the balance of our assets in the safe
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
                if is_profitable is None:
                    self.context.logger.error(
                        "Error calculating profitability of route. Switching to next route.."
                    )
                if is_profitable is False:
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
        next_action_details = actions[current_action_index]
        action_name = next_action_details.get("action")

        if not action_name:
            self.context.logger.error(f"Invalid action: {next_action_details}")
            return Event.DONE.value, {}

        next_action = Action(action_name)
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
            if not routes:
                self.context.logger.error("Error fetching routes")
                return Event.DONE.value, {}

            if self.synchronized_data.max_allowed_steps_in_a_route:
                routes = [
                    route
                    for route in routes
                    if len(route.get("steps", []))
                    <= self.synchronized_data.max_allowed_steps_in_a_route
                ]
                if not routes:
                    self.context.logger.error(
                        f"Needed routes with equal to or less than {self.synchronized_data.max_allowed_steps_in_a_route} steps, none found!"
                    )
                    return Event.DONE.value, {}

            serialized_routes = json.dumps(routes)

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

        elif next_action == Action.DEPOSIT:
            tx_hash, chain_id, safe_address = yield from self.get_deposit_tx_hash(
                next_action_details, positions
            )
            last_action = Action.DEPOSIT.value

        elif next_action == Action.WITHDRAW:
            tx_hash, chain_id, safe_address = yield from self.get_withdraw_tx_hash(
                next_action_details
            )
            last_action = Action.WITHDRAW.value

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

    def _calculate_investment_amounts_from_dollar_cap(
        self, action, chain, assets
    ) -> Generator[None, None, Optional[List[int]]]:
        """Calculate token amounts to invest based on a dollar value cap."""
        if not action.get("invested_amount", 0):
            self.context.logger.error(
                f"invested_amount not defined or zero: {action.get('invested_amount')}"
            )
            return None

        invested_amount = action.get("invested_amount")
        token0 = action.get("token0")
        token1 = action.get("token1")

        # Fetch token decimals
        token0_decimals = yield from self._get_token_decimals(chain, assets[0])
        token1_decimals = yield from self._get_token_decimals(chain, assets[1])

        if token0_decimals is None or token1_decimals is None:
            self.context.logger.error("Failed to get token decimals")
            return None

        # Configure retry parameters
        sleep_time = SLEEP_TIME
        # Number of API call retries

        # Fetch token0 price with retry handling
        token0_price = yield from self._fetch_token_price(token0, chain)
        if token0_price is None:
            self.context.logger.error(f"Failed to fetch price for token: {token0}")
            return None

        yield from self.sleep(sleep_time)  # Sleep to respect API rate limits

        # Fetch token1 price with retry handling
        token1_price = yield from self._fetch_token_price(token1, chain)
        if token1_price is None:
            self.context.logger.error(f"Failed to fetch price for token: {token1}")
            return None

        # Handle funds percentage to determine allocation per strategy
        relative_funds_percentage = action.get("relative_funds_percentage", 1.0)
        if not relative_funds_percentage:
            self.context.logger.error(
                f"relative_funds_percentage not defined: {relative_funds_percentage}"
            )
            return None

        # Calculate how much to allocate to this strategy based on its relative percentage
        allocated_fund_per_strategy = invested_amount / relative_funds_percentage

        # Calculate token amounts (default 50/50 split)
        token0_amount = (allocated_fund_per_strategy / 2) / token0_price
        token1_amount = (allocated_fund_per_strategy / 2) / token1_price

        # Check for valid amounts
        if token0_amount <= 0 or token1_amount <= 0:
            self.context.logger.error(
                f"Invalid token amounts: token0_amount={token0_amount}, token1_amount={token1_amount}"
            )
            return None

        self.context.logger.info(
            f"Token amounts calculated from dollar cap - token0: {token0_amount}, token1: {token1_amount}"
        )

        # Convert to token units with proper decimal precision
        max_amounts = [
            int(
                Decimal(str(token0_amount)) * (Decimal(10) ** Decimal(token0_decimals))
            ),
            int(
                Decimal(str(token1_amount)) * (Decimal(10) ** Decimal(token1_decimals))
            ),
        ]

        self.context.logger.info(
            f"Calculated max amounts from dollar cap: {max_amounts}"
        )
        return max_amounts

    def _calculate_velodrome_investment_amounts(
        self, action, chain, assets, positions, max_amounts
    ) -> Generator[None, None, Optional[List[int]]]:
        """Calculate investment amounts for Velodrome positions based on token percentages."""
        # Get token balances
        token0_balance = self._get_balance(chain, assets[0], positions)
        token1_balance = self._get_balance(chain, assets[1], positions)

        # Use token percentages if provided
        token0_percentage = action.get("token0_percentage")
        token1_percentage = action.get("token1_percentage")

        # Validate required data
        if (
            token0_balance is None
            or token1_balance is None
            or token0_percentage is None
            or token1_percentage is None
        ):
            self.context.logger.error(
                f"Missing required data: token0_balance={token0_balance}, "
                f"token1_balance={token1_balance}, token0_percentage={token0_percentage}, "
                f"token1_percentage={token1_percentage}"
            )
            yield None
            return None

        # Convert percentages to floats if needed
        try:
            token0_percentage = float(token0_percentage)
            token1_percentage = float(token1_percentage)
        except (ValueError, TypeError):
            self.context.logger.error(
                f"Invalid percentage values: token0_percentage={token0_percentage}, "
                f"token1_percentage={token1_percentage}"
            )
            return None

        # Calculate max amounts based on percentages
        max_amounts_in = [
            int(token0_balance * token0_percentage / 100),
            int(token1_balance * token1_percentage / 100),
        ]

        # Apply additional caps if provided
        if len(max_amounts) >= 2:
            max_amounts_in = [
                min(max_amounts_in[0], max_amounts[0]),
                min(max_amounts_in[1], max_amounts[1]),
            ]

        self.context.logger.info(
            f"Calculated Velodrome investment amounts: {max_amounts_in} "
            f"based on percentages: {token0_percentage}%, {token1_percentage}%"
        )

        # Validate the calculated amounts
        if any(amount < 0 for amount in max_amounts_in):
            self.context.logger.error(
                f"Invalid negative amounts calculated: {max_amounts_in}"
            )
            return None

        return max_amounts_in

    def _get_token_balances_and_calculate_amounts(
        self,
        chain: str,
        assets: List[str],
        positions: List[Dict[str, Any]],
        relative_funds_percentage: float = 1.0,
        max_investment_amounts: Optional[List[int]] = None,
    ) -> Generator[
        None, None, Tuple[Optional[List[int]], Optional[int], Optional[int]]
    ]:
        """Helper function to get token balances and calculate investment amounts."""
        try:
            # Get token balances
            token0_balance = self._get_balance(chain, assets[0], positions)
            token1_balance = self._get_balance(chain, assets[1], positions)

            if token0_balance is None or token1_balance is None:
                self.context.logger.error("Balance for one or more tokens is None")
                return None, None, None

            # Calculate max_amounts_in based on whether max_investment_amounts are provided
            if max_investment_amounts:
                # Convert max_investment_amounts to proper decimal representation
                token0_decimals = yield from self._get_token_decimals(chain, assets[0])
                token1_decimals = yield from self._get_token_decimals(chain, assets[1])

                if token0_decimals is None or token1_decimals is None:
                    self.context.logger.error("Failed to get token decimals")
                    return None, None, None

                max_investment_amounts = [
                    int(
                        Decimal(str(max_investment_amounts[0]))
                        * (Decimal(10) ** Decimal(token0_decimals))
                    ),
                    int(
                        Decimal(str(max_investment_amounts[1]))
                        * (Decimal(10) ** Decimal(token1_decimals))
                    ),
                ]

            # Standard calculation without ETH handling
            max_amounts_in = [
                int(token0_balance * relative_funds_percentage),
                int(token1_balance * relative_funds_percentage),
            ]

            # Ensure that allocated amounts do not exceed available balances
            max_amounts_in = [
                min(max_amounts_in[0], token0_balance),
                min(max_amounts_in[1], token1_balance),
            ]

            # Apply max_investment_amounts constraint if provided
            if max_investment_amounts:
                max_amounts_in = [
                    min(max_amounts_in[i], max_investment_amounts[i])
                    for i in range(len(max_amounts_in))
                ]

            self.context.logger.info(
                f"Calculated amounts - Token0: {max_amounts_in[0]}, Token1: {max_amounts_in[1]}"
            )

            return max_amounts_in, token0_balance, token1_balance

        except Exception as e:
            self.context.logger.error(
                f"Error calculating token balances and amounts: {str(e)}"
            )
            return None, None, None

    def _get_token_balances(
        self, chain: str, assets: List[str], positions: List[Dict[str, Any]]
    ) -> Tuple[Optional[int], Optional[int]]:
        """Simple helper to get token balances for two assets."""
        try:
            token0_balance = self._get_balance(chain, assets[0], positions)
            token1_balance = self._get_balance(chain, assets[1], positions)

            if token0_balance is None or token1_balance is None:
                self.context.logger.error("Balance for one or more tokens is None")
                return None, None

            return token0_balance, token1_balance

        except Exception as e:
            self.context.logger.error(f"Error getting token balances: {str(e)}")
            return None, None

    def get_enter_pool_tx_hash(
        self, positions, action
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Get enter pool tx hash"""
        self.context.logger.info(f"Processing enter pool action: {action}")

        # Extract common parameters
        dex_type = action.get("dex_type")
        chain = action.get("chain")
        assets = [action.get("token0"), action.get("token1")]
        pool_address = action.get("pool_address")
        safe_address = self.params.safe_contract_addresses.get(chain)
        pool_type = action.get("pool_type")
        is_stable = action.get("is_stable")
        is_cl_pool = action.get("is_cl_pool")
        max_investment_amounts = action.get("max_investment_amounts")

        # Validate essential parameters
        if not all([dex_type, chain, assets, pool_address, safe_address]):
            self.context.logger.error(f"Missing required parameters: {action}")
            return None, None, None

        if len(assets) < 2:
            self.context.logger.error(f"2 assets required, provided: {assets}")
            return None, None, None

        # Get corresponding pool handler
        pool = self.pools.get(dex_type)
        if not pool:
            self.context.logger.error(f"Unknown dex type: {dex_type}")
            return None, None, None

        # Calculate investment amounts based on dex type
        max_amounts_in = []
        max_amounts = []

        # Calculate investment amounts based on dollar cap if needed
        if self.current_positions and action.get("invested_amount", 0) != 0:
            max_amounts = yield from self._calculate_investment_amounts_from_dollar_cap(
                action, chain, assets
            )
            self.context.logger.info(
                f"Calculated max amounts from dollar cap: {max_amounts}"
            )
            if max_amounts is None:
                return None, None, None

        # Handle Velodrome-specific parameters
        if dex_type == DexType.VELODROME.value:
            if is_cl_pool:
                # For CL pools, use the specialized method that requires token percentages
                max_amounts_in = (
                    yield from self._calculate_velodrome_investment_amounts(
                        action, chain, assets, positions, max_amounts
                    )
                )
                if max_amounts_in is None:
                    return None, None, None
                self.context.logger.info(
                    f"max_amounts_in for velodrome CL pool: {max_amounts_in}"
                )
            else:
                # For stable/volatile pools, use the standard method
                relative_funds_percentage = action.get("relative_funds_percentage")
                if not relative_funds_percentage:
                    self.context.logger.error(
                        f"relative_funds_percentage not defined: {relative_funds_percentage}"
                    )
                    return None, None, None
                # Use the helper function to get balances and calculate amounts
                (
                    max_amounts_in,
                    token0_balance,
                    token1_balance,
                ) = yield from self._get_token_balances_and_calculate_amounts(
                    chain=chain,
                    assets=assets,
                    positions=positions,
                    relative_funds_percentage=relative_funds_percentage,
                    max_investment_amounts=max_investment_amounts,
                )

                if max_amounts_in is None:
                    return None, None, None

                if len(max_amounts) >= 2:
                    max_amounts_in = [
                        min(max_amounts_in[0], max_amounts[0]),
                        min(max_amounts_in[1], max_amounts[1]),
                    ]

                self.context.logger.info(
                    f"Adjusted max amounts in after comparing with max investment amounts: {max_amounts_in}"
                )

                if any(amount == 0 or amount is None for amount in max_amounts_in):
                    self.context.logger.error(
                        f"Insufficient balance for entering pool: {max_amounts_in}"
                    )
                    return None, None, None
        else:
            relative_funds_percentage = action.get("relative_funds_percentage")
            if not relative_funds_percentage:
                self.context.logger.error(
                    f"relative_funds_percentage not define: {relative_funds_percentage}"
                )
                return None, None, None
            # Use the helper function to get balances and calculate amounts
            (
                max_amounts_in,
                token0_balance,
                token1_balance,
            ) = yield from self._get_token_balances_and_calculate_amounts(
                chain=chain,
                assets=assets,
                positions=positions,
                relative_funds_percentage=relative_funds_percentage,
                max_investment_amounts=max_investment_amounts,
            )

            if max_amounts_in is None:
                return None, None, None

            if len(max_amounts) >= 2:
                max_amounts_in = [
                    min(max_amounts_in[0], max_amounts[0]),
                    min(max_amounts_in[1], max_amounts[1]),
                ]

            self.context.logger.info(
                f"Adjusted max amounts in after comparing with max investment amounts: {max_amounts_in}"
            )

            if any(amount == 0 or amount is None for amount in max_amounts_in):
                self.context.logger.error(
                    f"Insufficient balance for entering pool: {max_amounts_in}"
                )
                return None, None, None

        if dex_type != DexType.VELODROME.value and any(
            amount <= 0 or amount is None for amount in max_amounts_in
        ):
            self.context.logger.error(
                f"Insufficient balance for entering pool: {max_amounts_in}"
            )
            return None, None, None

        # Prepare kwargs for pool.enter
        kwargs = {
            "pool_address": pool_address,
            "safe_address": safe_address,
            "assets": assets,
            "chain": chain,
            "max_amounts_in": max_amounts_in,
            "pool_type": pool_type,
            "is_stable": is_stable,
            "is_cl_pool": is_cl_pool,
        }

        # Add Velodrome-specific parameters if applicable
        if dex_type == DexType.VELODROME.value:
            tick_spacing = action.get("tick_spacing")
            tick_ranges = action.get("tick_ranges")
            if tick_spacing:
                kwargs["tick_spacing"] = tick_spacing
            if tick_ranges:
                kwargs["tick_ranges"] = tick_ranges
        # Add pool_fee for non-Velodrome pools
        elif action.get("pool_fee") is not None:
            kwargs["pool_fee"] = action.get("pool_fee")

        result = yield from pool.enter(self, **kwargs)
        if not result or len(result) < 2:
            return None, None, None

        # Unpack the result
        tx_hash_or_hashes, contract_address = result[:2]
        if not tx_hash_or_hashes or not contract_address:
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

        # Handle the case where tx_hash_or_hashes is a list (from enter_cl_pool)
        if isinstance(tx_hash_or_hashes, list):
            # Add each transaction hash as a separate entry in the multisend
            for tx_hash in tx_hash_or_hashes:
                multi_send_txs.append(
                    {
                        "operation": MultiSendOperation.CALL,
                        "to": contract_address,
                        "value": value,
                        "data": tx_hash,
                    }
                )
        else:
            # Handle the case where tx_hash_or_hashes is a single tx_hash
            multi_send_txs.append(
                {
                    "operation": MultiSendOperation.CALL,
                    "to": contract_address,
                    "value": value,
                    "data": tx_hash_or_hashes,
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
        pool_address = action.get("pool_address")
        token_id = action.get("token_id")
        liquidity = action.get("liquidity")
        pool_type = action.get("pool_type")
        is_stable = action.get("is_stable")
        is_cl_pool = action.get("is_cl_pool")
        safe_address = self.params.safe_contract_addresses.get(action.get("chain"))

        pool = self.pools.get(dex_type)
        if not pool:
            self.context.logger.error(f"Unknown dex type: {dex_type}")
            return None, None, None

        # Prepare common kwargs for all pool types
        exit_pool_kwargs = {
            "pool_address": pool_address,
            "safe_address": safe_address,
            "chain": chain,
            "pool_type": pool_type,
        }

        # Add specific parameters based on pool type
        if dex_type == DexType.VELODROME.value:
            exit_pool_kwargs["is_stable"] = is_stable
            exit_pool_kwargs["is_cl_pool"] = is_cl_pool

            # For Velodrome CL pools, we need to handle multiple positions
            if is_cl_pool:
                # Extract token IDs from all nested positions
                token_ids = action.get("token_ids")
                liquidities = action.get("liquidities")
                if token_ids and liquidities:
                    self.context.logger.info(
                        f"Exiting Velodrome CL pool with {len(token_ids)} positions. "
                        f"Token IDs: {token_ids}"
                    )
                    exit_pool_kwargs["token_ids"] = token_ids
                    exit_pool_kwargs["liquidities"] = liquidities
            else:
                # For non-CL pools, we need assets
                if not assets or len(assets) < 2:
                    self.context.logger.error(
                        f"2 assets required for Velodrome Stable/Volatile pools, provided: {assets}"
                    )
                    return None, None, None
                exit_pool_kwargs["assets"] = assets
                exit_pool_kwargs["liquidity"] = liquidity

        elif dex_type == DexType.BALANCER.value:
            if not assets or len(assets) < 2:
                self.context.logger.error(
                    f"2 assets required for Balancer pools, provided: {assets}"
                )
                return None, None, None
            exit_pool_kwargs["assets"] = assets
        elif dex_type == DexType.UNISWAP_V3.value:
            exit_pool_kwargs["token_id"] = token_id
            exit_pool_kwargs["liquidity"] = liquidity
        else:
            self.context.logger.error(f"Unknown dex type: {dex_type}")
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

    def get_deposit_tx_hash(
        self, action, positions
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Get deposit tx hash"""
        chain = action.get("chain")
        asset = action.get("token0")
        amount = 0
        relative_funds_percentage = action.get("relative_funds_percentage")
        if not relative_funds_percentage:
            self.context.logger.error(
                f"relative_funds_percentage not define: {relative_funds_percentage}"
            )
            return None, None, None

        token_balance = self._get_balance(chain, asset, positions)
        amount = int(min(token_balance * relative_funds_percentage, token_balance))

        safe_address = self.params.safe_contract_addresses.get(chain)
        receiver = safe_address
        contract_address = action.get("pool_address")

        if not asset or not amount or not receiver:
            self.context.logger.error(f"Missing information in action: {action}")
            return None, None, None

        # Approve the asset
        approval_tx_payload = yield from self.get_approval_tx_hash(
            token_address=asset,
            amount=amount,
            spender=contract_address,
            chain=chain,
        )
        if not approval_tx_payload:
            self.context.logger.error("Error preparing approval tx payload")
            return None, None, None

        # Prepare the deposit transaction
        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=contract_address,
            contract_public_id=YearnV3VaultContract.contract_id,
            contract_callable="deposit",
            data_key="tx_hash",
            assets=amount,
            receiver=receiver,
            chain_id=chain,
        )
        if not tx_hash:
            return None, None, None

        multisend_txs = [
            approval_tx_payload,
            {
                "operation": MultiSendOperation.CALL,
                "to": contract_address,
                "value": 0,
                "data": tx_hash,
            },
        ]

        multisend_address = self.params.multisend_contract_addresses[chain]
        multisend_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=multisend_address,
            contract_public_id=MultiSendContract.contract_id,
            contract_callable="get_tx_data",
            data_key="data",
            multi_send_txs=multisend_txs,
            chain_id=chain,
        )

        safe_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
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

    def get_withdraw_tx_hash(
        self, action
    ) -> Generator[None, None, Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Get withdraw tx hash"""
        chain = action.get("chain")
        safe_address = self.params.safe_contract_addresses.get(chain)
        receiver = safe_address
        owner = safe_address
        contract_address = action.get("pool_address")

        if not receiver or not owner:
            self.context.logger.error(f"Missing information in action: {action}")
            return None, None, None

        # Get the maximum withdrawable amount
        amount = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=contract_address,
            contract_public_id=YearnV3VaultContract.contract_id,
            contract_callable="max_withdraw",
            owner=owner,
            data_key="amount",
            chain_id=chain,
        )
        if not amount:
            self.context.logger.error("Error fetching max withdraw amount")
            return None, None, None

        # Prepare the withdraw transaction
        tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=contract_address,
            contract_public_id=YearnV3VaultContract.contract_id,
            contract_callable="withdraw",
            data_key="tx_hash",
            assets=amount,
            receiver=receiver,
            owner=owner,
            chain_id=chain,
        )
        if not tx_hash:
            return None, None, None

        safe_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=safe_address,
            contract_public_id=GnosisSafeContract.contract_id,
            contract_callable="get_raw_safe_transaction_hash",
            data_key="tx_hash",
            to_address=contract_address,
            value=ETHER_VALUE,
            data=tx_hash,
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
            "amount": tx_info.get(
                "amount", 0
            ),  # Store the amount for later use in _update_assets_after_swap
        }
        return bridge_and_swap_action

    def check_if_route_is_profitable(
        self, route: Dict[str, Any]
    ) -> Generator[None, None, Tuple[Optional[bool], Optional[float], Optional[float]]]:
        """Checks if the entire route is profitable"""
        step_transactions = yield from self._get_step_transactions_data(route)
        if not step_transactions:
            return None, None, None

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
        amount = tx_info.get("amount")

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
            if tx_info is None:
                self.context.logger.error("Error fetching step transaction data")
                return None
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
            try:
                response_data = json.loads(response.body)
                self.context.logger.error(
                    f"[LiFi API Error Message] Error encountered: {response_data['message']}"
                )
            except (ValueError, TypeError) as e:
                self.context.logger.error(
                    f"Could not parse error response from API: {e}\nResponse body: {response.body}"
                )
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

        self.context.logger.info(
            f"Attempting swap/bridge from {from_chain} (chain ID: {from_chain_id}) to {to_chain} (chain ID: {to_chain_id})"
        )
        self.context.logger.info(
            f"Token swap: {from_token_symbol} ({from_token_address}) -> {to_token_symbol} ({to_token_address})"
        )
        self.context.logger.info(
            f"From address: {from_address} | To address: {to_address}"
        )

        available_amount = self._get_balance(from_chain, from_token_address, positions)

        # Calculate the amount to swap based on the available amount and funds percentage
        amount = min(
            available_amount, int(available_amount * action.get("funds_percentage", 1))
        )

        action["amount"] = amount

        self.context.logger.info(
            f"Available balance: {available_amount} | Amount to swap: {amount} {from_token_symbol}"
        )
        if amount <= 0:
            self.context.logger.error(
                f"Not enough balance for {from_token_symbol} on chain {from_chain}"
            )
            return None

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
            "fromAmount": str(amount),
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
            try:
                response = json.loads(routes_response.body)
                self.context.logger.error(
                    f"[LiFi API Error Message] Error encountered: {response.get('message', 'Unknown error')}"
                )
            except (ValueError, TypeError):
                error_msg = f"API returned status code {routes_response.status_code} with non-JSON response. "
                if hasattr(routes_response, "body"):
                    error_msg += f"Response body: {routes_response.body}"
                else:
                    error_msg += "Response body is missing"
                self.context.logger.error(error_msg)
            return None

        # Check if response body is empty or None
        if not routes_response.body:
            self.context.logger.error("API returned empty response body")
            return None

        try:
            routes_response = json.loads(routes_response.body)
        except (ValueError, TypeError) as e:
            self.context.logger.error(
                f"Could not parse response from api, "
                f"the following error was encountered {type(e).__name__}: {e}. "
                f"Response body: {routes_response.body[:500]}..."  # Log first 500 chars for debugging
            )
            return None

        routes = routes_response.get("routes", [])
        if not routes:
            self.context.logger.error(
                "[LiFi API Error Message] No routes available for this pair"
            )
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
            # Handle 404 errors (project not found) by continuing execution
            if response.status_code == 404:
                self.context.logger.warning(
                    f"Tenderly simulation failed with 404 (project not found) from url {api_url}. "
                    f"Error Message: {response.body}. Continuing execution without simulation."
                )
                return True

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

    def _get_all_positions_from_tx_receipt(
        self, tx_hash: str, chain: str
    ) -> Generator[None, None, Optional[List[Tuple[int, int, int, int, int]]]]:
        """Extract data for all positions from a transaction receipt."""
        response = yield from self.get_transaction_receipt(
            tx_digest=tx_hash,
            chain_id=chain,
        )
        if not response:
            self.context.logger.error(
                f"Error fetching tx receipt! Response: {response}"
            )
            return None

        # Define the event signature and calculate its hash
        event_signature = "IncreaseLiquidity(uint256,uint128,uint256,uint256)"
        event_signature_hash = keccak(text=event_signature)
        event_signature_hex = to_hex(event_signature_hash)[2:]

        # Extract logs from the response
        logs = response.get("logs", [])

        # Find all logs that match the IncreaseLiquidity event
        matching_logs = [
            log
            for log in logs
            if log.get("topics", [])
            and log.get("topics", [])[0][2:] == event_signature_hex
        ]

        if not matching_logs:
            self.context.logger.error("No logs found for IncreaseLiquidity event")
            return None

        positions_data = []

        # Get the timestamp from the block (same for all positions in this tx)
        block_number = response.get("blockNumber")
        if block_number is None:
            self.context.logger.error("Block number not found in transaction receipt.")
            return None

        block = yield from self.get_block(
            block_number=block_number,
            chain_id=chain,
        )

        if block is None:
            self.context.logger.error(f"Failed to fetch block {block_number}")
            return None

        timestamp = block.get("timestamp")
        if timestamp is None:
            self.context.logger.error("Timestamp not found in block data.")
            return None

        # Process each matching log
        for log in matching_logs:
            try:
                # Decode indexed parameter (tokenId)
                token_id_topic = log.get("topics", [])[1]
                if not token_id_topic:
                    self.context.logger.error(
                        f"Token ID topic is missing from log {log}"
                    )
                    continue

                # Convert hex to bytes and decode
                token_id_bytes = bytes.fromhex(token_id_topic[2:])
                token_id = decode(["uint256"], token_id_bytes)[0]

                # Decode non-indexed parameters (liquidity, amount0, amount1) from the data field
                data_hex = log.get("data")
                if not data_hex:
                    self.context.logger.error(f"Data field is empty in log {log}")
                    continue

                data_bytes = bytes.fromhex(data_hex[2:])
                decoded_data = decode(["uint128", "uint256", "uint256"], data_bytes)
                liquidity = decoded_data[0]
                amount0 = decoded_data[1]
                amount1 = decoded_data[2]

                # Add this position's data to the list
                positions_data.append(
                    (token_id, liquidity, amount0, amount1, timestamp)
                )

                self.context.logger.info(
                    f"Found position: token_id={token_id}, liquidity={liquidity}, "
                    f"amount0={amount0}, amount1={amount1}"
                )

            except Exception as e:
                self.context.logger.error(f"Error decoding data from mint event: {e}")
                continue

        if not positions_data:
            self.context.logger.error("Failed to extract any position data from logs")
            return None

        return positions_data

    def _get_data_from_mint_tx_receipt(
        self, tx_hash: str, chain: str
    ) -> Generator[
        None,
        None,
        Tuple[
            Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]
        ],
    ]:
        """Extract data from mint transaction receipt for the first position found."""
        response = yield from self.get_transaction_receipt(
            tx_digest=tx_hash,
            chain_id=chain,
        )
        if not response:
            self.context.logger.error(
                f"Error fetching tx receipt! Response: {response}"
            )
            return None, None, None, None, None

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
            return None, None, None, None, None

        # Decode indexed parameter (tokenId)
        try:
            # Decode indexed parameter (tokenId)
            token_id_topic = log.get("topics", [])[1]
            if not token_id_topic:
                self.context.logger.error(f"Token ID topic is missing from log {log}")
                return None, None, None, None, None
            # Convert hex to bytes and decode
            token_id_bytes = bytes.fromhex(token_id_topic[2:])
            token_id = decode(["uint256"], token_id_bytes)[0]

            # Decode non-indexed parameters (liquidity, amount0, amount1) from the data field
            data_hex = log.get("data")
            if not data_hex:
                self.context.logger.error(f"Data field is empty in log {log}")
                return None, None, None, None, None

            data_bytes = bytes.fromhex(data_hex[2:])
            decoded_data = decode(["uint128", "uint256", "uint256"], data_bytes)
            liquidity = decoded_data[0]
            amount0 = decoded_data[1]
            amount1 = decoded_data[2]

            # Get the timestamp from the block
            block_number = response.get("blockNumber")
            if block_number is None:
                self.context.logger.error(
                    "Block number not found in transaction receipt."
                )
                return None, None, None, None, None

            block = yield from self.get_block(
                block_number=block_number,
                chain_id=chain,
            )

            if block is None:
                self.context.logger.error(f"Failed to fetch block {block_number}")
                return None, None, None, None, None

            timestamp = block.get("timestamp")
            if timestamp is None:
                self.context.logger.error("Timestamp not found in block data.")
                return None, None, None, None, None

            return token_id, liquidity, amount0, amount1, timestamp

        except Exception as e:
            self.context.logger.error(f"Error decoding data from mint event: {e}")
            return None, None, None, None, None

    def get_block(
        self,
        block_number: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[None, None, Optional[Dict]]:
        """Get block data from the ledger API."""
        if block_number is None:
            block_identifier = "latest"

        ledger_api_response = yield from self.get_ledger_api_response(
            performative=LedgerApiMessage.Performative.GET_STATE,  # type: ignore
            ledger_callable="get_block",
            block_identifier=block_number,
            **kwargs,
        )
        if ledger_api_response.performative != LedgerApiMessage.Performative.STATE:
            self.context.logger.error(
                f"Failed to fetch block {block_identifier}: {ledger_api_response}"
            )
            return None

        block = ledger_api_response.state.body
        return block

    def _get_data_from_join_pool_tx_receipt(
        self, tx_hash: str, chain: str
    ) -> Generator[None, None, Tuple[Optional[int], Optional[int], Optional[int]]]:
        """Extract data from join pool transaction receipt."""
        response = yield from self.get_transaction_receipt(
            tx_digest=tx_hash,
            chain_id=chain,
        )
        if not response:
            self.context.logger.error(
                f"Error fetching tx receipt for join pool! Response: {response}"
            )
            return None, None, None

        # Define the event signature and calculate its hash
        event_signature = (
            "PoolBalanceChanged(bytes32,address,address[],int256[],uint256[])"
        )
        event_signature_hash = keccak(text=event_signature).hex()

        # Extract logs from the response
        logs = response.get("logs", [])

        # Initialize variables
        amount0 = None
        amount1 = None

        # Iterate over logs to find the PoolBalanceChanged event
        for log in logs:
            topics = log.get("topics", [])
            if not topics:
                continue

            # Check if the first topic matches the event signature hash
            if topics[0].lower() == "0x" + event_signature_hash.lower():
                # Decode the event data manually
                try:
                    # Decode non-indexed parameters (tokens, deltas, protocolFeeAmounts)
                    data_hex = log.get("data")
                    if not data_hex:
                        self.context.logger.error("Data field is empty in log")
                        continue

                    data_bytes = bytes.fromhex(data_hex[2:])

                    # Define the types of the non-indexed parameters
                    data_types = [
                        "address[]",  # tokens
                        "int256[]",  # deltas
                        "uint256[]",  # protocolFeeAmounts
                    ]

                    decoded_data = decode(data_types, data_bytes)

                    tokens = decoded_data[0]
                    deltas = decoded_data[1]

                    # Assuming the pool has two tokens
                    if len(tokens) >= 2 and len(deltas) >= 2:
                        # The deltas represent the amounts; take absolute values for deposits
                        amount0 = abs(deltas[0])
                        amount1 = abs(deltas[1])
                    else:
                        self.context.logger.error(
                            "Unexpected number of tokens/deltas in event"
                        )
                        continue

                    # Break after finding the first matching event
                    break
                except Exception as e:
                    self.context.logger.error(
                        f"Error decoding PoolBalanceChanged event: {e}"
                    )
                    continue

        if amount0 is None or amount1 is None:
            self.context.logger.error("No amounts found in PoolBalanceChanged event")
            return None, None, None

        # Get the timestamp from the block
        block_number = response.get("blockNumber")
        if block_number is None:
            self.context.logger.error("Block number not found in transaction receipt.")
            return None, None, None

        block = yield from self.get_block(
            block_number=block_number,
            chain_id=chain,
        )

        if block is None:
            self.context.logger.error(f"Failed to fetch block {block_number}")
            return None, None, None

        timestamp = block.get("timestamp")
        if timestamp is None:
            self.context.logger.error("Timestamp not found in block data.")
            return None, None, None

        return amount0, amount1, timestamp

    def _add_token_to_assets(self, chain, token, symbol):
        # Read current assets
        token = to_checksum_address(token)
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

    def _get_data_from_velodrome_mint_event(
        self, tx_hash: str, chain: str
    ) -> Generator[None, None, Tuple[Optional[int], Optional[int], Optional[int]]]:
        """Extract data from Velodrome Mint event for non-CL pools."""
        response = yield from self.get_transaction_receipt(
            tx_digest=tx_hash,
            chain_id=chain,
        )
        if not response:
            self.context.logger.error(
                f"Error fetching tx receipt for Velodrome Mint event! Response: {response}"
            )
            return None, None, None

        # Define the event signature and calculate its hash
        # Mint (index_topic_1 address sender, uint256 amount0, uint256 amount1)
        event_signature = "Mint(address,uint256,uint256)"
        event_signature_hash = keccak(text=event_signature)
        event_signature_hex = to_hex(event_signature_hash)[2:]

        # Extract logs from the response
        logs = response.get("logs", [])

        # Find the log that matches the Mint event
        log = next(
            (
                log
                for log in logs
                if log.get("topics", [])
                and log.get("topics", [])[0][2:] == event_signature_hex
            ),
            None,
        )

        if not log:
            # If the standard signature is not found, try the alternative Mint signature
            event_signature = "Mint(address,address,uint256,uint256)"
            event_signature_hash = keccak(text=event_signature)
            event_signature_hex = to_hex(event_signature_hash)[2:]

            # Extract logs from the response
            logs = response.get("logs", [])

            # Find the log that matches the Mint event
            log = next(
                (
                    log
                    for log in logs
                    if log.get("topics", [])
                    and log.get("topics", [])[0][2:] == event_signature_hex
                ),
                None,
            )

            if not log:
                self.context.logger.error("No logs found for Velodrome Mint event")
                return None, None, None

        try:
            # Decode indexed parameter (sender address)
            sender_topic = log.get("topics", [])[1]
            if not sender_topic:
                self.context.logger.error(
                    f"Sender address topic is missing from log {log}"
                )
                return None, None, None

            # Decode non-indexed parameters (amount0, amount1) from the data field
            data_hex = log.get("data")
            if not data_hex:
                self.context.logger.error(f"Data field is empty in log {log}")
                return None, None, None

            data_bytes = bytes.fromhex(data_hex[2:])
            decoded_data = decode(["uint256", "uint256"], data_bytes)
            amount0 = decoded_data[0]
            amount1 = decoded_data[1]

            # Get the timestamp from the block
            block_number = response.get("blockNumber")
            if block_number is None:
                self.context.logger.error(
                    "Block number not found in transaction receipt."
                )
                return None, None, None

            block = yield from self.get_block(
                block_number=block_number,
                chain_id=chain,
            )

            if block is None:
                self.context.logger.error(f"Failed to fetch block {block_number}")
                return None, None, None

            timestamp = block.get("timestamp")
            if timestamp is None:
                self.context.logger.error("Timestamp not found in block data.")
                return None, None, None

            return amount0, amount1, timestamp

        except Exception as e:
            self.context.logger.error(
                f"Error decoding data from Velodrome Mint event: {e}"
            )
            return None, None, None

    def _get_data_from_deposit_tx_receipt(
        self, tx_hash: str, chain: str
    ) -> Generator[None, None, Tuple[int, int, int]]:
        """Extract amount, shares, and timestamp from a deposit transaction receipt."""

        # Fetch the transaction receipt
        receipt = yield from self.get_transaction_receipt(
            tx_digest=tx_hash,
            chain_id=chain,
        )
        if receipt is None:
            self.context.logger.error(
                f"Failed to fetch transaction receipt for {tx_hash}"
            )
            return None, None, None

        event_signature = "Deposit(address,address,uint256,uint256)"
        event_signature_hash = keccak(event_signature.encode()).hex()

        for log in receipt["logs"]:
            if log["topics"][0].lower() == "0x" + event_signature_hash.lower():
                # Decode non-indexed parameters (uint256 values) from the data field
                data_hex = log["data"]
                if len(data_hex) != 2 + 64 * 2:
                    self.context.logger.error("Unexpected data length in log")
                    continue

                assets_hex = data_hex[2:66]
                shares_hex = data_hex[66:130]

                assets = int(assets_hex, 16)
                shares = int(shares_hex, 16)

                # Get the timestamp from the block
                block_number = receipt.get("blockNumber")
                if block_number is None:
                    self.context.logger.error(
                        "Block number not found in transaction receipt."
                    )
                    return None, None, None

                block = yield from self.get_block(
                    block_number=block_number,
                    chain_id=chain,
                )

                if block is None:
                    self.context.logger.error(f"Failed to fetch block {block_number}")
                    return None, None, None

                timestamp = block.get("timestamp")
                if timestamp is None:
                    self.context.logger.error("Timestamp not found in block data.")
                    return None, None, None

                return assets, shares, timestamp

        self.context.logger.error("Deposit event not found in transaction receipt")
        return None, None, None

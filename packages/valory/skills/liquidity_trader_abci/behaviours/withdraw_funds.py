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

"""This module contains the WithdrawFundsBehaviour of LiquidityTraderAbciApp."""

import json
import time
from typing import Any, Dict, Generator, List, Optional

from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    Action,
    LiquidityTraderBaseBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.payloads import WithdrawFundsPayload
from packages.valory.skills.liquidity_trader_abci.states.withdraw_funds import (
    WithdrawFundsRound,
)


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


class WithdrawFundsBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that handles withdrawal operations."""

    matching_round = WithdrawFundsRound

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            self.context.logger.info("=== WithdrawFundsBehaviour started ===")

            # Check if investing is paused due to withdrawal (read from KV store)
            investing_paused = yield from self._read_investing_paused()
            self.context.logger.info(f"Investing paused flag: {investing_paused}")

            if not investing_paused:
                # No withdrawal requested, transition to normal flow
                self.context.logger.info(
                    "No withdrawal requested, transitioning to normal flow"
                )
                payload = WithdrawFundsPayload(
                    sender=self.context.agent_address, withdrawal_actions=json.dumps([])
                )
                yield from self.send_a2a_transaction(payload)
                yield from self.wait_until_round_end()
                self.set_done()
                return

            # Get current positions and portfolio data
            self.context.logger.info("Getting current positions...")
            positions = self.current_positions
            portfolio_data = self.portfolio_data
            self.context.logger.info(f"Found {len(positions)} positions")

            # Log position details for debugging
            for i, pos in enumerate(positions):
                self.context.logger.info(
                    f"Position {i+1}: {pos.get('pool_address', 'N/A')} - {pos.get('dex_type', 'N/A')} - Status: {pos.get('status', 'N/A')}"
                )
                self.context.logger.info(f"Position {i+1} full data: {pos}")

            # Read withdrawal data to get target address
            self.context.logger.info("Reading withdrawal data...")
            withdrawal_data = yield from self._read_withdrawal_data()
            self.context.logger.info(
                f"Withdrawal data result: {'Success' if withdrawal_data else 'Failed/None'}"
            )

            if not withdrawal_data:
                self.context.logger.error("No withdrawal data found")
                yield from self._update_withdrawal_status(
                    "FAILED", "Withdrawal failed due to error."
                )
                yield from self._reset_withdrawal_flags()
                payload = WithdrawFundsPayload(
                    sender=self.context.agent_address, withdrawal_actions=json.dumps([])
                )
                yield from self.send_a2a_transaction(payload)
                yield from self.wait_until_round_end()
                self.set_done()
                return

            target_address = withdrawal_data.get("withdrawal_target_address", "")
            self.context.logger.info(f"Target address: {target_address}")

            if not target_address:
                self.context.logger.error("No target address found in withdrawal data")
                yield from self._update_withdrawal_status(
                    "FAILED", "Withdrawal failed due to error."
                )
                yield from self._reset_withdrawal_flags()
                payload = WithdrawFundsPayload(
                    sender=self.context.agent_address, withdrawal_actions=json.dumps([])
                )
                yield from self.send_a2a_transaction(payload)
                yield from self.wait_until_round_end()
                self.set_done()
                return

            # Prepare withdrawal actions based on current state
            self.context.logger.info("Preparing withdrawal actions...")
            yield from self._update_withdrawal_status(
                "WITHDRAWING", "Withdrawal Initiated. Preparing your funds..."
            )
            withdrawal_actions = yield from self._prepare_withdrawal_actions(
                positions, target_address, portfolio_data
            )
            self.context.logger.info(
                f"Prepared {len(withdrawal_actions)} withdrawal actions"
            )

            if not withdrawal_actions:
                # No actions to execute, but this shouldn't happen in normal flow
                # If we have no actions, it means something went wrong
                self.context.logger.warning(
                    "No withdrawal actions prepared - this indicates an issue"
                )
                yield from self._update_withdrawal_status(
                    "FAILED", "No withdrawal actions could be prepared"
                )
                yield from self._reset_withdrawal_flags()
                payload = WithdrawFundsPayload(
                    sender=self.context.agent_address, withdrawal_actions=json.dumps([])
                )
                yield from self.send_a2a_transaction(payload)
                yield from self.wait_until_round_end()
                self.set_done()
                return

            # Send withdrawal actions to be executed in subsequent rounds
            payload = WithdrawFundsPayload(
                sender=self.context.agent_address,
                withdrawal_actions=json.dumps(withdrawal_actions),
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()

    def _read_withdrawal_data(self) -> Generator[None, None, Optional[Dict[str, str]]]:
        """Read withdrawal data from KV store."""
        try:
            self.context.logger.info(
                "Attempting to read withdrawal data from KV store..."
            )
            result = yield from self._read_kv(
                (
                    "withdrawal_id",
                    "withdrawal_status",
                    "withdrawal_target_address",
                    "withdrawal_message",
                    "withdrawal_requested_at",
                    "withdrawal_completed_at",
                    "withdrawal_estimated_value_usd",
                    "withdrawal_chain",
                    "withdrawal_safe_address",
                    "withdrawal_transaction_hashes",
                    "withdrawal_current_step",
                )
            )
            self.context.logger.info(f"KV store read result: {result}")
            if result:
                self.context.logger.info(
                    f"Found withdrawal data with keys: {list(result.keys())}"
                )
            else:
                self.context.logger.warning("No withdrawal data found in KV store")
            return result
        except Exception as e:
            self.context.logger.error(f"Error reading withdrawal data: {str(e)}")
            self.context.logger.error(f"Error type: {type(e).__name__}")
            return None

    def _get_portfolio_data(self) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get current portfolio data by fetching fresh data from blockchain."""
        # Update portfolio data with fresh blockchain data instead of reading from static file
        self.context.logger.info("Fetching fresh portfolio data from blockchain...")
        yield from self.update_portfolio_after_action()

        if not self.portfolio_data:
            self.context.logger.error(
                "Portfolio data is empty after updating from blockchain."
            )
            return None
        self.context.logger.info(
            f"Using fresh portfolio data with keys: {list(self.portfolio_data.keys())}"
        )
        return self.portfolio_data

    def _validate_and_update_position_statuses(
        self, positions: List[Dict[str, Any]]
    ) -> Generator[None, None, List[Dict[str, Any]]]:
        """Validate position statuses against on-chain state and update local data."""
        updated_positions = []

        for position in positions:
            status = position.get("status")

            # Only check positions that are marked as OPEN
            if status == "OPEN" or status == "open":
                self.context.logger.info(
                    f"Validating position status for pool: {position.get('pool_address')} ({position.get('dex_type')})"
                )

                # Check on-chain balance
                pool_address = position.get("pool_address")
                chain = position.get("chain")
                dex_type = position.get("dex_type")
                safe_address = self.context.params.safe_contract_addresses.get(chain)

                if not safe_address:
                    self.context.logger.error(
                        f"No safe address found for chain: {chain}"
                    )
                    updated_positions.append(position)
                    continue

                # Get on-chain balance using pool-specific method
                balance = yield from self._get_pool_balance(
                    pool_address, safe_address, chain, dex_type, position
                )

                if balance == 0:
                    # Position is actually closed on-chain, update local status
                    self.context.logger.info(
                        f"Position {pool_address} ({dex_type}) has zero balance on-chain. Updating status to CLOSED."
                    )
                    position["status"] = "CLOSED"
                    position["exit_timestamp"] = int(self._get_current_timestamp())
                    position["exit_tx_hash"] = "auto-closed"  # Mark as auto-closed
                else:
                    self.context.logger.info(
                        f"Position {pool_address} ({dex_type}) has balance {balance} on-chain. Status remains OPEN."
                    )

            updated_positions.append(position)

        # Update the current positions in the base behaviour
        self.current_positions = updated_positions
        self.store_current_positions()

        return updated_positions

    def _get_pool_balance(
        self,
        pool_address: str,
        safe_address: str,
        chain: str,
        dex_type: str,
        position: Dict[str, Any] = None,
    ) -> Generator[None, None, int]:
        """Get the pool balance for the safe address based on pool type."""
        try:
            # Route to appropriate handler based on DEX type
            if dex_type == "balancer":
                balance_response = yield from self._get_balancer_pool_balance(
                    pool_address, safe_address, chain
                )
            elif dex_type == "velodrome":
                balance_response = yield from self._get_velodrome_pool_balance(
                    pool_address, safe_address, chain, position
                )
            elif dex_type == "uniswap_v3":
                balance_response = yield from self._get_uniswap_v3_pool_balance(
                    pool_address, safe_address, chain, position
                )
            elif dex_type == "sturdy":
                balance_response = yield from self._get_sturdy_pool_balance(
                    pool_address, safe_address, chain
                )
            else:
                self.context.logger.error(f"Unsupported DEX type: {dex_type}")
                return 0

            if balance_response is not None:
                return int(balance_response)
            else:
                return 0

        except Exception as e:
            self.context.logger.error(
                f"Error getting pool balance for {pool_address} ({dex_type}): {str(e)}"
            )
            return 0

    def _get_balancer_pool_balance(
        self, pool_address: str, safe_address: str, chain: str
    ) -> Generator[None, None, Optional[int]]:
        """Get balance for Balancer pools using WeightedPoolContract."""
        balance_response = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=pool_address,
            contract_public_id="valory/balancer_weighted_pool:0.1.0",
            contract_callable="get_balance",
            data_key="balance",
            account=safe_address,
            chain_id=chain,
        )
        return balance_response

    def _get_velodrome_pool_balance(
        self,
        pool_address: str,
        safe_address: str,
        chain: str,
        position: Dict[str, Any] = None,
    ) -> Generator[None, None, Optional[int]]:
        """Get balance for Velodrome pools (both regular and CL pools)."""
        is_cl_pool = position.get("is_cl_pool", False) if position else False

        if is_cl_pool:
            balance_response = yield from self._get_velodrome_cl_pool_balance(
                pool_address, safe_address, chain, position
            )
            return balance_response
        else:
            balance_response = yield from self._get_velodrome_regular_pool_balance(
                pool_address, safe_address, chain
            )
            return balance_response

    def _get_velodrome_regular_pool_balance(
        self, pool_address: str, safe_address: str, chain: str
    ) -> Generator[None, None, Optional[int]]:
        """Get balance for regular Velodrome pools using VelodromePoolContract."""
        balance_response = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=pool_address,
            contract_public_id="valory/velodrome_pool:0.1.0",
            contract_callable="get_balance",
            data_key="balance",
            account=safe_address,
            chain_id=chain,
        )
        return balance_response

    def _get_velodrome_cl_pool_balance(
        self,
        pool_address: str,
        safe_address: str,
        chain: str,
        position: Dict[str, Any] = None,
    ) -> Generator[None, None, Optional[int]]:
        """Get balance for Velodrome CL pools using NonFungiblePositionManager."""
        position_manager_address = (
            self.params.velodrome_non_fungible_position_manager_contract_addresses.get(
                chain
            )
        )
        if not position_manager_address:
            self.context.logger.error(
                f"No Velodrome position manager address found for chain {chain}"
            )
            return 0

        positions = position.get("positions", []) if position else []
        if not positions:
            # If we don't have specific positions, check if the safe owns any positions
            balance_response = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_STATE,
                contract_address=position_manager_address,
                contract_public_id="valory/velodrome_non_fungible_position_manager:0.1.0",
                contract_callable="balanceOf",
                data_key="balance",
                owner=safe_address,
                chain_id=chain,
            )
            return balance_response
        else:
            # Check each position to see if any still have liquidity
            total_liquidity = yield from self._check_velodrome_cl_positions_liquidity(
                position_manager_address, safe_address, chain, positions
            )
            return total_liquidity

    def _check_velodrome_cl_positions_liquidity(
        self,
        position_manager_address: str,
        safe_address: str,
        chain: str,
        positions: List[Dict[str, Any]],
    ) -> Generator[None, None, int]:
        """Check liquidity for multiple Velodrome CL positions."""
        total_liquidity = 0

        for pos in positions:
            token_id = pos.get("token_id")
            if not token_id:
                continue

            # Check if the safe is still the owner of this specific position
            owner_response = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_STATE,
                contract_address=position_manager_address,
                contract_public_id="valory/velodrome_non_fungible_position_manager:0.1.0",
                contract_callable="ownerOf",
                data_key="owner",
                token_id=token_id,
                chain_id=chain,
            )

            if owner_response and owner_response.lower() == safe_address.lower():
                # Safe still owns the position, now check if it has liquidity
                position_data = yield from self.contract_interact(
                    performative=ContractApiMessage.Performative.GET_STATE,
                    contract_address=position_manager_address,
                    contract_public_id="valory/velodrome_non_fungible_position_manager:0.1.0",
                    contract_callable="get_position",
                    data_key="data",
                    token_id=token_id,
                    chain_id=chain,
                )

                if position_data and position_data.get("liquidity", 0) > 0:
                    total_liquidity += 1

        return total_liquidity

    def _get_uniswap_v3_pool_balance(
        self,
        pool_address: str,
        safe_address: str,
        chain: str,
        position: Dict[str, Any] = None,
    ) -> Generator[None, None, Optional[int]]:
        """Get balance for Uniswap V3 pools using NonFungiblePositionManager."""
        position_manager_address = (
            self.params.uniswap_position_manager_contract_addresses.get(chain)
        )
        if not position_manager_address:
            self.context.logger.error(
                f"No position manager address found for chain {chain}"
            )
            return 0

        token_id = position.get("token_id") if position else None
        if not token_id:
            # If we don't have a specific token_id, check if the safe owns any positions
            balance_response = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_STATE,
                contract_address=position_manager_address,
                contract_public_id="valory/uniswap_v3_non_fungible_position_manager:0.1.0",
                contract_callable="balanceOf",
                data_key="balance",
                owner=safe_address,
                chain_id=chain,
            )
            return balance_response
        else:
            # Check if the safe is still the owner of this specific position
            liquidity_check = yield from self._check_uniswap_v3_position_liquidity(
                position_manager_address, safe_address, chain, token_id
            )
            return liquidity_check

    def _check_uniswap_v3_position_liquidity(
        self,
        position_manager_address: str,
        safe_address: str,
        chain: str,
        token_id: int,
    ) -> Generator[None, None, int]:
        """Check liquidity for a specific Uniswap V3 position."""
        # Check if the safe is still the owner of this specific position
        owner_response = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=position_manager_address,
            contract_public_id="valory/uniswap_v3_non_fungible_position_manager:0.1.0",
            contract_callable="ownerOf",
            data_key="owner",
            token_id=token_id,
            chain_id=chain,
        )

        if owner_response and owner_response.lower() == safe_address.lower():
            # Safe still owns the position, now check if it has liquidity
            position_data = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_STATE,
                contract_address=position_manager_address,
                contract_public_id="valory/uniswap_v3_non_fungible_position_manager:0.1.0",
                contract_callable="get_position",
                data_key="data",
                token_id=token_id,
                chain_id=chain,
            )

            if position_data and position_data.get("liquidity", 0) > 0:
                return 1  # Position exists and has liquidity
            else:
                return 0  # Position exists but has no liquidity
        else:
            return 0  # Safe no longer owns the position

    def _get_sturdy_pool_balance(
        self, pool_address: str, safe_address: str, chain: str
    ) -> Generator[None, None, Optional[int]]:
        """Get balance for Sturdy pools using YearnV3VaultContract."""
        balance_response = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=pool_address,
            contract_public_id="valory/sturdy_yearn_v3_vault:0.1.0",
            contract_callable="balance_of",
            data_key="amount",
            owner=safe_address,
            chain_id=chain,
        )
        return balance_response

    def _prepare_withdrawal_actions(
        self,
        positions: List[Dict[str, Any]],
        target_address: str,
        portfolio_data: Dict[str, Any],
    ) -> Generator[None, None, List[Dict[str, Any]]]:
        """Prepare all withdrawal actions in order: exit pools -> swap to USDC -> transfer USDC."""
        actions = []

        self.context.logger.info("=== PREPARING ALL WITHDRAWAL ACTIONS AT ONCE ===")

        # Validate and update position statuses before preparing actions
        self.context.logger.info(
            "Validating position statuses against on-chain state..."
        )

        # Get fresh positions data for action preparation
        self.context.logger.info(
            "Getting fresh positions data for action preparation..."
        )

        # Get fresh positions data for action preparation
        self.context.logger.info(
            "Getting fresh positions data for action preparation..."
        )

        # Step 1: Check for open positions and create exit actions
        self.context.logger.info("=== STEP 1: CHECKING FOR OPEN POSITIONS ===")
        exit_actions = self._prepare_exit_pool_actions(positions)
        if exit_actions:
            self.context.logger.info(
                f"Found {len(exit_actions)} open positions to exit"
            )
            actions.extend(exit_actions)
            # yield from self._update_withdrawal_status("WITHDRAWING", "Funds prepared. Exiting pools...")
        else:
            self.context.logger.info("No open positions found")

        # Step 2: Create swap actions for all non-USDC assets
        self.context.logger.info("=== STEP 2: PREPARING SWAP ACTIONS ===")
        swap_actions = self._prepare_swap_to_usdc_actions_standard(portfolio_data)
        if swap_actions:
            self.context.logger.info(
                f"Found {len(swap_actions)} assets to swap to USDC"
            )
            actions.extend(swap_actions)
            # yield from self._update_withdrawal_status("WITHDRAWING", "All active investment positions are closed. Converting assets...")
        else:
            self.context.logger.info("No assets to swap to USDC")

        # Step 3: Create transfer action for USDC
        self.context.logger.info("=== STEP 3: PREPARING TRANSFER ACTION ===")
        transfer_actions = self._prepare_transfer_usdc_actions_standard(target_address)
        if transfer_actions:
            self.context.logger.info("Found USDC to transfer")
            actions.extend(transfer_actions)
            # yield from self._update_withdrawal_status("WITHDRAWING", "Successfully converted assets to USDC. Transfering to user wallet...")
        else:
            self.context.logger.info("No USDC to transfer - withdrawal complete")
            yield from self._update_withdrawal_status(
                "COMPLETED", "Withdrawal complete!"
            )
            yield from self._reset_withdrawal_flags()

        self.context.logger.info("=== FINAL ACTIONS SUMMARY ===")
        self.context.logger.info(f"Total actions prepared: {len(actions)}")
        self.context.logger.info("Action breakdown:")
        self.context.logger.info(
            f"  - Exit actions: {len([a for a in actions if a.get('action') == Action.EXIT_POOL.value])}"
        )
        self.context.logger.info(
            f"  - Swap actions: {len([a for a in actions if a.get('action') == Action.FIND_BRIDGE_ROUTE.value])}"
        )
        self.context.logger.info(
            f"  - Transfer actions: {len([a for a in actions if a.get('action') == Action.WITHDRAW.value])}"
        )
        self.context.logger.info(f"Final actions list: {actions}")

        return actions

    def _prepare_exit_pool_actions(
        self, positions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare exit pool actions in standard action format.

        :param positions: current positions
        :return: list of exit pool actions in standard format
        """
        actions = []

        self.context.logger.info(
            f"Preparing exit actions for {len(positions)} positions"
        )
        self.context.logger.info(f"Positions type: {type(positions)}")
        self.context.logger.info(f"Positions data: {positions}")

        for i, position in enumerate(positions):
            self.context.logger.info(f"Processing position {i+1}: {position}")
            self.context.logger.info(f"Position {i+1} keys: {list(position.keys())}")
            status = position.get("status")
            self.context.logger.info(
                f"Position {i+1} status: {status} (type: {type(status)})"
            )

            if status == "OPEN" or status == "open":
                self.context.logger.info(f"Creating exit action for position {i+1}")

                # Use base class method to build exit action
                action = self._build_exit_pool_action_base(position)
                if action:
                    # Add description for withdrawal context
                    action[
                        "description"
                    ] = f"Exit {position.get('token0_symbol')}/{position.get('token1_symbol')} pool for withdrawal"

                    # Add required assets field
                    action["assets"] = [
                        position.get("token0"),
                        position.get("token1"),
                    ]

                    # Add missing fields that are required for proper transaction preparation
                    if position.get("pool_fee") is not None:
                        action["pool_fee"] = position.get("pool_fee")
                    if position.get("tick_spacing") is not None:
                        action["tick_spacing"] = position.get("tick_spacing")
                    if position.get("tick_ranges") is not None:
                        action["tick_ranges"] = position.get("tick_ranges")

                    self.context.logger.info(f"Created exit action: {action}")
                    actions.append(action)
                else:
                    self.context.logger.error(
                        f"Failed to create exit action for position {i+1}"
                    )
            else:
                self.context.logger.info(
                    f"Skipping position {i+1} with status: {status}"
                )

        self.context.logger.info(f"Total exit actions prepared: {len(actions)}")
        return actions

    def _prepare_swap_to_usdc_actions_standard(
        self, portfolio_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Prepare swap actions to convert all assets to USDC using positions data."""
        self.context.logger.info("=== SWAP DEBUGGING ===")
        actions = []

        # Get the configured chain
        chain = self.context.params.target_investment_chains[0]
        portfolio_breakdown = portfolio_data.get("portfolio_breakdown", [])

        # Process positions data to find assets with balances
        for asset in portfolio_breakdown:
            token_symbol = asset.get("asset")
            self.context.logger.info(f"--- Processing asset: {token_symbol} ---")

            token_address = asset.get("address")
            value_usd = asset.get("value_usd", 0)

            self.context.logger.info(
                f"Token: {token_symbol}, Address: {token_address}, Value: {value_usd}"
            )

            # Skip if it's already USDC or OLAS by address
            usdc_address = self._get_usdc_address(chain)
            olas_address = (
                self._get_olas_address(chain)
                if hasattr(self, "_get_olas_address")
                else None
            )
            if (
                token_address
                and usdc_address
                and token_address.lower() == usdc_address.lower()
            ):
                self.context.logger.info("Skipping USDC - it's USDC (by address)")
                continue
            if (
                token_address
                and olas_address
                and token_address.lower() == olas_address.lower()
            ):
                self.context.logger.info(
                    "Skipping OLAS - do not swap OLAS during withdrawal (by address)"
                )
                continue

            # Skip if balance is too small
            if value_usd <= 1:  # nosec B105
                self.context.logger.info(
                    f"Skipping {token_symbol} - balance too small: {value_usd}"
                )
                continue

            self.context.logger.info(
                f"Creating swap action for {token_symbol} with balance {value_usd}"
            )

            # Use base class method to build swap action
            swap_action = self._build_swap_to_usdc_action(
                chain=chain,
                from_token_address=token_address,
                from_token_symbol=token_symbol,
                funds_percentage=1.0,  # Use 100% of available balance
                description=f"Swap {asset} to USDC for withdrawal",
            )

            if swap_action:
                self.context.logger.info(f"Created swap action: {swap_action}")
                actions.append(swap_action)
            else:
                self.context.logger.error(
                    f"Failed to create swap action for {token_symbol}"
                )

        self.context.logger.info("=== SWAP DEBUGGING COMPLETE ===")
        self.context.logger.info(f"Total swap actions created: {len(actions)}")
        self.context.logger.info(f"Swap actions: {actions}")

        return actions

    def _prepare_transfer_usdc_actions_standard(
        self, target_address: str
    ) -> List[Dict[str, Any]]:
        """Prepare actions to transfer all USDC to user wallet using positions data."""
        actions = []

        self.context.logger.info("=== TRANSFER DEBUGGING ===")
        self.context.logger.info(f"Target address: {target_address}")

        # Find USDC balance from positions data
        usdc_address = self._get_usdc_address(
            self.context.params.target_investment_chains[0]
        )
        self.context.logger.info(f"USDC address exists: {bool(usdc_address)}")

        self.context.logger.info("Creating transfer action for USDC...")

        # Create transfer action using dynamic amount calculation
        action = {
            "action": Action.WITHDRAW.value,  # Standard action key
            "chain": self.context.params.target_investment_chains[0],
            "from_address": self.context.params.safe_contract_addresses.get(
                self.context.params.target_investment_chains[0]
            ),
            "to_address": target_address,
            "token_address": usdc_address,
            "token_symbol": "USDC",
            "funds_percentage": 1.0,  # Use 100% of available USDC balance
            "description": f"Transfer USDC to {target_address}",
        }

        self.context.logger.info(f"Created transfer action: {action}")
        actions.append(action)
        self.context.logger.info(f"Actions list after append: {actions}")

        self.context.logger.info("=== TRANSFER DEBUGGING COMPLETE ===")
        self.context.logger.info(f"Total transfer actions created: {len(actions)}")
        self.context.logger.info(f"Transfer actions: {actions}")
        self.context.logger.info(f"Returning actions list: {actions}")

        return actions

    def _update_withdrawal_status(
        self, status: str, message: str
    ) -> Generator[None, None, None]:
        """Update withdrawal status in KV store."""
        try:
            update_data = {"withdrawal_status": status, "withdrawal_message": message}

            if status == "COMPLETED":
                update_data["withdrawal_completed_at"] = str(int(time.time()))
                update_data["investing_paused"] = "false"
            if status == "FAILED":
                update_data["investing_paused"] = "false"

            yield from self._write_kv(update_data)
            self.context.logger.info(
                f"Withdrawal status updated to {status}: {message}"
            )
        except Exception as e:
            self.context.logger.error(f"Error updating withdrawal status: {str(e)}")

    def _reset_withdrawal_flags(self) -> Generator[None, None, None]:
        """Reset withdrawal flags when withdrawal is completed"""
        try:
            reset_data = {"investing_paused": "false"}
            yield from self._write_kv(reset_data)
            self.context.logger.info("Withdrawal flags reset successfully")
        except Exception as e:
            self.context.logger.error(f"Error resetting withdrawal flags: {str(e)}")

    def _read_investing_paused(self) -> Generator[None, None, bool]:
        """Read investing_paused flag from KV store."""
        try:
            self.context.logger.info("Reading investing_paused flag from KV store...")
            result = yield from self._read_kv(("investing_paused",))
            self.context.logger.info(f"KV store result for investing_paused: {result}")
            investing_paused = result.get("investing_paused", "false").lower() == "true"
            self.context.logger.info(
                f"Parsed investing_paused value: {investing_paused}"
            )
            return investing_paused
        except Exception as e:
            self.context.logger.error(f"Error reading investing_paused flag: {str(e)}")
            self.context.logger.error(f"Error type: {type(e).__name__}")
            return False

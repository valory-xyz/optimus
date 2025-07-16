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
from typing import Any, Dict, Generator, List, Optional, Tuple

from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    LiquidityTraderBaseBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.payloads import WithdrawFundsPayload
from packages.valory.skills.liquidity_trader_abci.states.withdraw_funds import (
    WithdrawFundsRound,
)


class WithdrawFundsBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that handles withdrawal operations."""

    matching_round = WithdrawFundsRound

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            # Check if investing is paused due to withdrawal (read from KV store)
            investing_paused = yield from self._read_investing_paused()
            if not investing_paused:
                # No withdrawal requested, transition to normal flow
                payload = WithdrawFundsPayload(
                    sender=self.context.agent_address,
                    withdrawal_actions=json.dumps([])
                )
                yield from self.send_a2a_transaction(payload)
                yield from self.wait_until_round_end()
                self.set_done()
                return

            # Get current positions and portfolio data
            positions = self.synchronized_data.positions
            portfolio_data = yield from self._get_portfolio_data()
            
            if not portfolio_data:
                # No portfolio data available
                yield from self._update_withdrawal_status("FAILED", "Withdrawal failed due to error.")
                payload = WithdrawFundsPayload(
                    sender=self.context.agent_address,
                    withdrawal_actions=json.dumps([])
                )
                yield from self.send_a2a_transaction(payload)
                yield from self.wait_until_round_end()
                self.set_done()
                return

            # Prepare withdrawal actions based on current state
            withdrawal_actions = yield from self._prepare_withdrawal_actions(
                positions, portfolio_data
            )
            
            if not withdrawal_actions:
                # No actions to execute, mark withdrawal as completed
                yield from self._update_withdrawal_status("COMPLETED", "Withdrawal complete!")
                yield from self._reset_withdrawal_flags()
                payload = WithdrawFundsPayload(
                    sender=self.context.agent_address,
                    withdrawal_actions=json.dumps([])
                )
                yield from self.send_a2a_transaction(payload)
                yield from self.wait_until_round_end()
                self.set_done()
                return

            # Update withdrawal status to WITHDRAWING when we start preparing actions
            yield from self._update_withdrawal_status("WITHDRAWING", "Funds prepared. Exiting pools...")

            # Send withdrawal actions
            payload = WithdrawFundsPayload(
                sender=self.context.agent_address,
                withdrawal_actions=json.dumps(withdrawal_actions)
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()

    def _read_withdrawal_data(self) -> Generator[None, None, Optional[Dict[str, str]]]:
        """
        Read withdrawal data from KV store.
        
        :return: withdrawal data or None if not found
        """
        try:
            result = yield from self._read_kv(("withdrawal_id", "withdrawal_status", "withdrawal_target_address", 
                                              "withdrawal_message", "withdrawal_requested_at", "withdrawal_completed_at",
                                              "withdrawal_estimated_value_usd", "withdrawal_chain", "withdrawal_safe_address",
                                              "withdrawal_transaction_hashes", "withdrawal_current_step"))
            return result
        except Exception as e:
            self.context.logger.error(f"Error reading withdrawal data: {str(e)}")
            return None

    def _get_portfolio_data(self) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """
        Get current portfolio data.
        
        :return: portfolio data or None if not found
        """
        try:
            portfolio_data_filepath = (
                self.context.params.store_path / self.context.params.portfolio_info_filename
            )
            
            with open(portfolio_data_filepath, "r", encoding="utf-8") as file:
                portfolio_data = json.load(file)
            
            return portfolio_data
        except Exception as e:
            self.context.logger.error(f"Error reading portfolio data: {str(e)}")
            return None

    def _prepare_withdrawal_actions(
        self, 
        positions: List[Dict[str, Any]], 
        portfolio_data: Dict[str, Any]
    ) -> Generator[None, None, List[Dict[str, Any]]]:
        """
        Prepare withdrawal actions based on current portfolio state.
        
        :param positions: current positions
        :param portfolio_data: portfolio data
        :return: list of withdrawal actions
        """
        actions = []
        
        # Read withdrawal data from KV store
        withdrawal_data = yield from self._read_withdrawal_data()
        if not withdrawal_data:
            self.context.logger.error("No withdrawal data found")
            return actions
        
        target_address = withdrawal_data.get("withdrawal_target_address", "")
        if not target_address:
            self.context.logger.error("No target address found in withdrawal data")
            return actions

        # Get current withdrawal step
        current_step = withdrawal_data.get("withdrawal_current_step", "EXIT_POOLS")
        
        try:
            if current_step == "EXIT_POOLS":
                # Step 1: Exit from all pools
                exit_actions = yield from self._prepare_exit_pool_actions(positions)
                if exit_actions:
                    actions.extend(exit_actions)
                    yield from self._update_withdrawal_status("WITHDRAWING", "Funds prepared. Exiting pools...")
                else:
                    # No pools to exit, move to next step
                    current_step = "SWAP_TO_USDC"
                    yield from self._update_withdrawal_step(current_step)
                    
            if current_step == "SWAP_TO_USDC":
                # Step 2: Swap all assets to USDC
                swap_actions = yield from self._prepare_swap_to_usdc_actions(portfolio_data)
                if swap_actions:
                    actions.extend(swap_actions)
                    yield from self._update_withdrawal_status("WITHDRAWING", "All active investment positions are closed. Converting assets...")
                else:
                    # No swaps needed, move to next step
                    current_step = "TRANSFER_USDC"
                    yield from self._update_withdrawal_step(current_step)
                    
            if current_step == "TRANSFER_USDC":
                # Step 3: Transfer all USDC to user wallet
                transfer_actions = yield from self._prepare_transfer_usdc_actions(
                    target_address, portfolio_data
                )
                if transfer_actions:
                    actions.extend(transfer_actions)
                    yield from self._update_withdrawal_status("WITHDRAWING", "Successfully converted assets to USDC. Transfering to user wallet...")
                else:
                    # No USDC to transfer
                    yield from self._update_withdrawal_status("COMPLETED", "Withdrawal complete!")
            
            self.context.logger.info(f"Prepared {len(actions)} withdrawal actions for step: {current_step}")
            
        except Exception as e:
            self.context.logger.error(f"Error preparing withdrawal actions: {str(e)}")
            yield from self._update_withdrawal_status("FAILED", "Withdrawal failed due to error.")
        
        return actions

    def _prepare_exit_pool_actions(self, positions: List[Dict[str, Any]]) -> Generator[None, None, List[Dict[str, Any]]]:
        """
        Prepare actions to exit from all liquidity pools.
        
        :param positions: current positions
        :return: list of exit pool actions
        """
        actions = []
        
        for position in positions:
            if position.get("status") == "OPEN":
                # Create exit pool action
                action = {
                    "action_type": "EXIT_POOL",
                    "chain": position.get("chain", self.context.params.target_investment_chains[0]),
                    "pool_address": position.get("pool_address"),
                    "dex_type": position.get("dex_type"),
                    "token0": position.get("token0"),
                    "token1": position.get("token1"),
                    "token0_symbol": position.get("token0_symbol"),
                    "token1_symbol": position.get("token1_symbol"),
                    "pool_type": position.get("pool_type"),
                    "description": f"Exit {position.get('token0_symbol')}/{position.get('token1_symbol')} pool for withdrawal"
                }
                
                # Add position-specific data
                if position.get("token_id"):
                    action["token_id"] = position.get("token_id")
                if position.get("liquidity"):
                    action["liquidity"] = position.get("liquidity")
                
                actions.append(action)
        
        return actions

    def _prepare_swap_to_usdc_actions(self, portfolio_data: Dict[str, Any]) -> Generator[None, None, List[Dict[str, Any]]]:
        """
        Prepare actions to swap all assets to USDC.
        
        :param portfolio_data: portfolio data
        :return: list of swap actions
        """
        actions = []
        portfolio_breakdown = portfolio_data.get("portfolio_breakdown", [])
        
        for asset in portfolio_breakdown:
            asset_symbol = asset.get("asset", "")
            balance = asset.get("balance", 0)
            asset_address = asset.get("address", "")
            
            # Skip USDC and zero balances
            if asset_symbol == "USDC" or balance <= 0:
                continue
            
            # Create swap action to USDC
            action = {
                "action_type": "SWAP",
                "chain": self.context.params.target_investment_chains[0],
                "from_token": asset_address,
                "from_token_symbol": asset_symbol,
                "to_token": self._get_usdc_address(self.context.params.target_investment_chains[0]),
                "to_token_symbol": "USDC",
                "amount": balance,
                "description": f"Swap {balance} {asset_symbol} to USDC for withdrawal"
            }
            
            actions.append(action)
        
        return actions

    def _prepare_transfer_usdc_actions(
        self, target_address: str, portfolio_data: Dict[str, Any]
    ) -> Generator[None, None, List[Dict[str, Any]]]:
        """
        Prepare actions to transfer all USDC to user wallet.
        
        :param target_address: user's target address
        :param portfolio_data: portfolio data
        :return: list of transfer actions
        """
        actions = []
        portfolio_breakdown = portfolio_data.get("portfolio_breakdown", [])
        
        # Find USDC balance
        usdc_balance = 0
        usdc_address = ""
        
        for asset in portfolio_breakdown:
            if asset.get("asset") == "USDC":
                usdc_balance = asset.get("balance", 0)
                usdc_address = asset.get("address", "")
                break
        
        if usdc_balance > 0 and usdc_address:
            # Create transfer action
            action = {
                "action_type": "TRANSFER",
                "chain": self.context.params.target_investment_chains[0],
                "from_address": self.context.params.safe_contract_addresses.get(
                    self.context.params.target_investment_chains[0]
                ),
                "to_address": target_address,
                "token_address": usdc_address,
                "token_symbol": "USDC",
                "amount": usdc_balance,
                "description": f"Transfer {usdc_balance} USDC to {target_address}"
            }
            
            actions.append(action)
        
        return actions

    def _get_usdc_address(self, chain: str) -> str:
        """
        Get USDC address for the given chain.
        
        :param chain: blockchain chain
        :return: USDC contract address
        """
        usdc_addresses = {
            "optimism": "0x7F5c764cBc14f9669B88837ca1490cCa17c31607",
            "base": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "mode": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # Using Base USDC for Mode
            "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
        }
        return usdc_addresses.get(chain, usdc_addresses["optimism"])

    def _update_withdrawal_status(self, status: str, message: str) -> Generator[None, None, None]:
        """
        Update withdrawal status in KV store.
        
        :param status: new status
        :param message: status message
        """
        try:
            update_data = {
                "withdrawal_status": status,
                "withdrawal_message": message
            }
            
            if status == "COMPLETED":
                import time
                update_data["withdrawal_completed_at"] = str(int(time.time()))
            
            yield from self._write_kv(update_data)
            self.context.logger.info(f"Withdrawal status updated to {status}: {message}")
        except Exception as e:
            self.context.logger.error(f"Error updating withdrawal status: {str(e)}")

    def _update_withdrawal_step(self, step: str) -> Generator[None, None, None]:
        """
        Update withdrawal current step in KV store.
        
        :param step: new step
        """
        try:
            yield from self._write_kv({"withdrawal_current_step": step})
            self.context.logger.info(f"Withdrawal step updated to: {step}")
        except Exception as e:
            self.context.logger.error(f"Error updating withdrawal step: {str(e)}")

    def _reset_withdrawal_flags(self) -> Generator[None, None, None]:
        """
        Reset withdrawal flags when withdrawal is completed.
        Keep the status as COMPLETED for historical record.
        """
        try:
            reset_data = {
                "investing_paused": "false"
                # Note: We don't reset withdrawal_status to keep it as COMPLETED
            }
            yield from self._write_kv(reset_data)
            self.context.logger.info("Withdrawal flags reset successfully")
        except Exception as e:
            self.context.logger.error(f"Error resetting withdrawal flags: {str(e)}")

    def _read_investing_paused(self) -> Generator[None, None, bool]:
        """Read investing_paused flag from KV store."""
        try:
            result = yield from self._read_kv(("investing_paused",))
            return result.get("investing_paused", "false").lower() == "true"
        except Exception as e:
            self.context.logger.error(f"Error reading investing_paused flag: {str(e)}")
            return False 
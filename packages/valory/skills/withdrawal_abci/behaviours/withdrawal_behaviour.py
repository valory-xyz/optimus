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

"""This module contains the withdrawal behaviour for the 'withdrawal_abci' skill."""

import json
import time
from typing import Dict, Generator, List, Optional, Any

from eth_utils import to_bytes
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    LiquidityTraderBaseBehaviour,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.liquidity_trader_abci.behaviours.base import DexType, ETHER_VALUE, SAFE_TX_GAS, ZERO_ADDRESS
from packages.valory.contracts.gnosis_safe.contract import GnosisSafeContract
from packages.valory.contracts.gnosis_safe.contract import SafeOperation

# Import decision making functions to reuse existing logic
from packages.valory.skills.liquidity_trader_abci.behaviours.decision_making import DecisionMakingBehaviour
from packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy import EvaluateStrategyBehaviour


class WithdrawalBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that handles withdrawal requests independently."""

    matching_round = None

    def __init__(self, **kwargs):
        """Initialize the withdrawal behaviour."""
        super().__init__(**kwargs)
        self.withdrawal_id: Optional[str] = None
        self.target_address: Optional[str] = None
        self.operating_chain: Optional[str] = None
        self.safe_address: Optional[str] = None
        self.tx_hashes: List[str] = []
        self.position_to_exit: Optional[Dict[str, Any]] = None

    def setup(self) -> None:
        """Set up the behaviour."""
        self.context.logger.info("Withdrawal behaviour setup complete")

    def act(self) -> Generator:
        """Execute the withdrawal behaviour."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            # Check if there's a pending withdrawal request
            withdrawal_data = yield from self._read_kv(keys=("withdrawal_request",))
            
            if not withdrawal_data or not withdrawal_data.get("withdrawal_request"):
                # No withdrawal request, sleep and check again
                yield from self.sleep(5.0)
                return

            # Process the withdrawal request
            yield from self._process_withdrawal_request(withdrawal_data["withdrawal_request"])

    def _process_withdrawal_request(self, request_data: Dict[str, Any]) -> Generator:
        """Process a withdrawal request."""
        try:
            # Extract request data
            self.withdrawal_id = request_data.get("withdrawal_id")
            self.target_address = request_data.get("target_address")
            self.operating_chain = self.params.target_investment_chains[0]
            self.safe_address = self.params.safe_contract_addresses.get(self.operating_chain)

            self.context.logger.info(f"Processing withdrawal request: {self.withdrawal_id}")

            # Update status to withdrawing
            yield from self._update_withdrawal_status("withdrawing", "Withdraw initiated. Preparing your funds...")

            # Step 1: Get current positions and prepare exit actions
            yield from self._prepare_exit_actions()

            # Step 2: Execute exit actions
            yield from self._execute_exit_actions()

            # Step 3: Convert all tokens to USDC using LiFi route optimization
            yield from self._convert_tokens_to_usdc()

            # Step 4: Transfer USDC to target address
            yield from self._transfer_to_target()

            # Step 5: Complete withdrawal
            yield from self._complete_withdrawal()

        except Exception as e:
            self.context.logger.error(f"Error processing withdrawal: {e}")
            yield from self._update_withdrawal_status("failed", f"Withdrawal failed: {str(e)}")

    def _prepare_exit_actions(self) -> Generator:
        """Prepare exit actions for all current positions."""
        self.context.logger.info("Preparing exit actions for current positions")

        # Get current positions
        current_positions = yield from self._read_kv(keys=("current_positions",))
        if not current_positions:
            self.context.logger.info("No current positions to exit")
            return

        positions = current_positions.get("current_positions", [])
        if not positions:
            self.context.logger.info("No current positions to exit")
            return

        # Prepare exit actions for each position
        exit_actions = []
        for position in positions:
            action = yield from self._create_exit_action(position)
            if action:
                exit_actions.append(action)

        # Store exit actions
        yield from self._write_kv({
            "withdrawal_exit_actions": json.dumps(exit_actions),
            "withdrawal_action_index": "0"
        })

        self.context.logger.info(f"Prepared {len(exit_actions)} exit actions")

    def _create_exit_action(self, position: Dict[str, Any]) -> Generator[Optional[Dict], None, None]:
        """Create an exit action for a position by reusing existing logic from evaluate strategy."""
        try:
            self.position_to_exit = position
            
            dex_type = position.get("dex_type")
            num_of_tokens_required = 1 if dex_type == DexType.STURDY.value else 2
            
            tokens = EvaluateStrategyBehaviour._build_tokens_from_position(
                self, position, num_of_tokens_required
            )
            
            if not tokens or len(tokens) < num_of_tokens_required:
                self.context.logger.error(
                    f"Failed to extract {num_of_tokens_required} tokens from position: {position}"
                )
                return None
            
            exit_action = EvaluateStrategyBehaviour._build_exit_pool_action(
                self, tokens, num_of_tokens_required
            )
            
            if exit_action:
                self.context.logger.info(f"Created exit action for {position.get('protocol', 'unknown')} position")
                return exit_action
            else:
                self.context.logger.error(f"Failed to create exit action for position: {position}")
                return None
                
        except Exception as e:
            self.context.logger.error(f"Error creating exit action: {e}")
            return None

    def _execute_exit_actions(self) -> Generator:
        """Execute exit actions for all positions."""
        self.context.logger.info("Executing exit actions")

        # Get exit actions
        actions_data = yield from self._read_kv(keys=("withdrawal_exit_actions", "withdrawal_action_index"))
        if not actions_data:
            self.context.logger.info("No exit actions to execute")
            return

        actions = json.loads(actions_data.get("withdrawal_exit_actions", "[]"))
        current_index = int(actions_data.get("withdrawal_action_index", "0"))

        if current_index >= len(actions):
            self.context.logger.info("All exit actions completed")
            yield from self._update_withdrawal_status("withdrawing", "All active investment positions are closed. Converting assets...")
            return

        # Execute current action
        action = actions[current_index]
        tx_hash = yield from self._execute_action(action)
        
        if tx_hash:
            self.tx_hashes.append(tx_hash)
            self.context.logger.info(f"Exit action completed with tx: {tx_hash}")
            
            # Update action index
            yield from self._write_kv({
                "withdrawal_action_index": str(current_index + 1)
            })
            
            # Wait for transaction confirmation
            confirmed = yield from self._wait_for_transaction_confirmation(tx_hash, self.operating_chain)
            if not confirmed:
                self.context.logger.error(f"Exit action transaction not confirmed: {tx_hash}")
                yield from self._update_withdrawal_status("failed", f"Exit action failed: transaction not confirmed")
                return
        else:
            self.context.logger.error(f"Exit action failed: {action}")
            yield from self._update_withdrawal_status("failed", f"Exit action failed")
            return

        # Continue with next action
        yield from self._execute_exit_actions()

    def _convert_tokens_to_usdc(self) -> Generator:
        """Convert all tokens to USDC using LiFi route optimization (same as normal agent flow)."""
        self.context.logger.info("Converting tokens to USDC using LiFi route optimization")
        
        # Update status to show conversion in progress
        yield from self._update_withdrawal_status("withdrawing", "Successfully converted funds to USDC. Transfering funds to user wallet...")

        # Get USDC address for the operating chain
        usdc_addresses = json.loads(self.params.usdc_addresses)
        usdc_address = usdc_addresses.get(self.operating_chain)
        if not usdc_address:
            self.context.logger.error(f"No USDC address configured for chain {self.operating_chain}")
            return

        
        safe_address = self.params.safe_contract_addresses.get(self.operating_chain)
        initial_usdc_balance = yield from self._get_token_balance(self.operating_chain, safe_address, usdc_address)
        self.context.logger.info(f"Initial USDC balance: {initial_usdc_balance}")

        # Get token balances
        balances = yield from self._get_token_balances()
        
        if not balances:
            self.context.logger.info("No token balances found for conversion")
            return

        # Get current positions for balance calculation
        current_positions = yield from self._read_kv(keys=("current_positions",))
        positions = current_positions.get("current_positions", []) if current_positions else []

        # Convert each token to USDC using LiFi route optimization
        successful_swaps = 0
        for token_address, balance in balances.items():
            if token_address.lower() != usdc_address.lower() and balance > 0:
                # Check if balance is significant enough to swap
                min_swap_amount = 1000  # Minimum amount in wei to consider swapping
                if balance < min_swap_amount:
                    self.context.logger.info(f"Skipping swap for {token_address}: balance {balance} below minimum {min_swap_amount}")
                    continue

                token_symbol = yield from self._get_token_symbol(self.operating_chain, token_address)
                token_symbol = token_symbol or "UNKNOWN"

                self.context.logger.info(f"Finding optimal route for {token_symbol} -> USDC")

                routes = yield from self._fetch_routes_withdrawal(positions, {
                    "from_chain": self.operating_chain,
                    "to_chain": self.operating_chain,
                    "from_token": token_address,
                    "from_token_symbol": token_symbol,
                    "to_token": usdc_address,
                    "to_token_symbol": "USDC",
                    "amount": balance,
                    "funds_percentage": 1.0
                })

                if not routes:
                    self.context.logger.error(f"No routes found for {token_symbol} -> USDC")
                    continue

                best_route = routes[0]
                tx_hash = yield from self._execute_route_withdrawal(best_route, positions)
                
                if tx_hash:
                    
                    yield from self.sleep(5.0)  # Wait for balance to reflect
                    new_usdc_balance = yield from self._get_token_balance(self.operating_chain, safe_address, usdc_address)
                    
                    if new_usdc_balance and new_usdc_balance > initial_usdc_balance:
                        usdc_increase = new_usdc_balance - initial_usdc_balance
                        self.context.logger.info(f"✅ USDC balance increased by {usdc_increase} after swap")
                        initial_usdc_balance = new_usdc_balance  # Update for next swap
                        successful_swaps += 1
                        self.tx_hashes.append(tx_hash)
                    else:
                        self.context.logger.error(f"❌ USDC balance did not increase after swap. Expected increase, got: {new_usdc_balance} (was: {initial_usdc_balance})")
                else:
                    self.context.logger.error(f"Swap failed for {token_symbol} -> USDC")

        self.context.logger.info(f"Successfully converted {successful_swaps} tokens to USDC using LiFi optimization")

    def _fetch_routes_withdrawal(self, positions: List[Dict[str, Any]], action: Dict[str, Any]) -> Generator[None, None, Optional[List[Any]]]:
        """Fetch routes using LiFi API (reusing logic from DecisionMakingBehaviour)."""
        return (yield from DecisionMakingBehaviour.fetch_routes(self, positions, action))

    def _execute_route_withdrawal(self, route: Dict[str, Any], positions: List[Dict[str, Any]]) -> Generator[None, None, Optional[str]]:
        """Execute a single route step (reusing logic from DecisionMakingBehaviour)."""
        try:
            is_profitable, total_fee, total_gas_cost = yield from DecisionMakingBehaviour.check_if_route_is_profitable(self, route)
            
            if not is_profitable:
                self.context.logger.error("Route is not profitable, skipping")
                return None

            steps = route.get("steps", [])
            if not steps:
                self.context.logger.error("No steps found in route")
                return None

            step = steps[0]  
            
            step_profitable, step_data = yield from DecisionMakingBehaviour.check_step_costs(
                self, step, total_fee, total_gas_cost, 0, len(steps)
            )
            
            if not step_profitable:
                self.context.logger.error("Step is not profitable, skipping")
                return None

            bridge_swap_action = yield from DecisionMakingBehaviour.prepare_bridge_swap_action(
                self, positions, step_data, total_fee, total_gas_cost
            )
            
            if not bridge_swap_action:
                self.context.logger.error("Failed to prepare bridge swap action")
                return None

            # Execute the action using existing safe transaction logic
            tx_hash = yield from self._execute_safe_transaction(
                to_address=self.params.multisend_contract_addresses.get(self.operating_chain),
                data=bytes.fromhex(bridge_swap_action["payload"]),
                chain=self.operating_chain
            )
            
            if not tx_hash:
                self.context.logger.error("Failed to execute safe transaction")
                return None

            self.context.logger.info(f"Waiting for transaction confirmation: {tx_hash}")
            success = yield from self._wait_for_transaction_confirmation(tx_hash, self.operating_chain)
            if not success:
                self.context.logger.error(f"Transaction failed: {tx_hash}")
                return None

            self.context.logger.info("Waiting for LiFi to reflect the transaction...")
            yield from self.sleep(self.params.waiting_period_for_status_check)

            swap_status = yield from self._check_lifi_swap_status(tx_hash)
            if swap_status != "DONE":
                self.context.logger.error(f"Swap failed with status: {swap_status}")
                return None

            self.context.logger.info(f"Swap successful! Transaction: {tx_hash}")
            return tx_hash

        except Exception as e:
            self.context.logger.error(f"Error executing route: {e}")
            return None



    def _check_lifi_swap_status(self, tx_hash: str) -> Generator[None, None, Optional[str]]:
        """Check LiFi swap status (reusing logic from DecisionMakingBehaviour)."""
        try:
            status, sub_status = yield from DecisionMakingBehaviour.get_swap_status(self, tx_hash)
            
            if status is None or sub_status is None:
                self.context.logger.error("Could not get swap status from LiFi API")
                return None

            self.context.logger.info(f"LiFi swap status: {status}, sub-status: {sub_status}")
            return status

        except Exception as e:
            self.context.logger.error(f"Error checking LiFi swap status: {e}")
            return None

    def _execute_action(self, action: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        """Execute a single action and return transaction hash."""
        try:
            action_type = action.get("action")
            
            if action_type == "exit_pool":
                return (yield from self._execute_exit_action(action))
            elif action_type == "transfer":
                return (yield from self._execute_transfer_action(action))
            else:
                self.context.logger.error(f"Unknown action type: {action_type}")
                return None
                
        except Exception as e:
            self.context.logger.error(f"Error executing action {action}: {e}")
            return None

    def _execute_exit_action(self, action: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        """Execute pool exit action using the exact same logic as normal agent flow."""
        try:
            payload_string, chain, safe_address = yield from DecisionMakingBehaviour.get_exit_pool_tx_hash(
                self, action
            )
            
            if not payload_string:
                self.context.logger.error(f"Failed to get exit pool tx hash for action: {action}")
                return None
                
            tx_hash = yield from self._submit_transaction(payload_string, chain)
            
            if tx_hash:
                self.context.logger.info(f"Exit action completed with tx: {tx_hash}")
                return tx_hash
            else:
                self.context.logger.error(f"Exit action failed: {action}")
                return None
                
        except Exception as e:
            self.context.logger.error(f"Error executing exit action: {e}")
            return None

    def _execute_transfer_action(self, action: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        """Execute transfer action using safe transaction."""
        try:
            chain = action.get("chain")
            token_address = action.get("token_address")
            to_address = action.get("to_address")
            amount = action.get("amount")
            safe_address = self.params.safe_contract_addresses.get(chain)

            if not all([chain, token_address, to_address, amount, safe_address]):
                self.context.logger.error(f"Missing required parameters for transfer: {action}")
                return None

            # Create ERC20 transfer transaction
            tx_hash = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=token_address,
                contract_public_id="valory/erc20:0.1.0",
                contract_callable="transfer",
                data_key="tx_hash",
                to=to_address,
                amount=amount,
                chain_id=chain,
            )
            if not tx_hash:
                self.context.logger.error("Failed to get transfer transaction data")
                return None

            # Create safe transaction
            safe_tx_hash = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=safe_address,
                contract_public_id=GnosisSafeContract.contract_id,
                contract_callable="get_raw_safe_transaction_hash",
                data_key="tx_hash",
                to_address=token_address,
                value=ETHER_VALUE,
                data=tx_hash,
                operation=SafeOperation.CALL.value,
                safe_tx_gas=SAFE_TX_GAS,
                chain_id=chain,
            )
            if not safe_tx_hash:
                self.context.logger.error("Failed to get safe transaction hash")
                return None

            safe_tx_hash = safe_tx_hash[2:]
            self.context.logger.info(f"Safe transaction hash: {safe_tx_hash}")

            # Create payload for transaction
            payload_string = self._hash_payload_to_hex(
                safe_tx_hash=safe_tx_hash,
                ether_value=ETHER_VALUE,
                safe_tx_gas=SAFE_TX_GAS,
                operation=SafeOperation.CALL.value,
                to_address=token_address,
                data=tx_hash,
            )

            # Get signature
            signature = self._get_signature(safe_address)
            if not signature:
                self.context.logger.error("Failed to get signature")
                return None

            # Submit transaction
            tx_hash = yield from self._submit_transaction(payload_string, chain)
            if not tx_hash:
                self.context.logger.error("Failed to submit transfer transaction")
                return None

            # Wait for confirmation
            success = yield from self._wait_for_transaction_confirmation(tx_hash, chain)
            if not success:
                self.context.logger.error(f"Transfer transaction failed: {tx_hash}")
                return None

            self.context.logger.info(f"Transfer transaction successful: {tx_hash}")
            return tx_hash

        except Exception as e:
            self.context.logger.error(f"Error executing transfer action: {e}")
            return None

    def _get_signature(self, owner: str) -> str:
        """Get signature for safe transaction (reuse existing function)."""
        # Reuse the exact same deterministic signature generation from DecisionMakingBehaviour
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
    
    def _hash_payload_to_hex(
    safe_tx_hash: str,
    ether_value: int,
    safe_tx_gas: int,
    to_address: str,
    data: bytes,
    ) -> str:
        """Hash the payload to hex format."""
        import hashlib
        
        # Create payload string
        payload = f"{safe_tx_hash}{ether_value}{safe_tx_gas}{to_address}{data.hex()}"
        
        # Hash the payload
        payload_hash = hashlib.sha256(payload.encode()).hexdigest()
        
        return payload_hash

    def _execute_safe_transaction(
        self, to_address: str, data: bytes, chain: str
    ) -> Generator[None, None, Optional[str]]:
        """Execute safe transaction and return transaction hash."""
        try:
            safe_address = self.params.safe_contract_addresses.get(chain)
            
            # Prepare safe transaction hash
            safe_tx_hash = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=safe_address,
                contract_public_id=GnosisSafeContract.contract_id,
                contract_callable="get_raw_safe_transaction_hash",
                data_key="tx_hash",
                to_address=to_address,
                value=ETHER_VALUE,
                data=data,
                operation=SafeOperation.DELEGATE_CALL.value,
                safe_tx_gas=SAFE_TX_GAS,
                chain_id=chain,
            )

            if not safe_tx_hash:
                self.context.logger.error("Error preparing safe transaction")
                return None

            # Create transaction payload
            safe_tx_hash = safe_tx_hash[2:]
            tx_params = dict(
                ether_value=ETHER_VALUE,
                safe_tx_gas=SAFE_TX_GAS,
                operation=SafeOperation.DELEGATE_CALL.value,
                to_address=to_address,
                data=data,
                safe_tx_hash=safe_tx_hash,
            )
            payload_string = self._hash_payload_to_hex(**tx_params)
            
            # Send transaction
            tx_hash = yield from self.send_transaction(payload_string, chain)
            return tx_hash
            
        except Exception as e:
            self.context.logger.error(f"Error executing safe transaction: {e}")
            return None

    def _get_token_symbol(self, chain: str, token_address: str) -> Generator[Optional[str], None, None]:
        """Get token symbol."""
        try:
            if token_address == ZERO_ADDRESS:
                return "ETH"  

            symbol = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=token_address,
                contract_public_id="valory/erc20:0.1.0",
                contract_callable="symbol",
                data_key="symbol",
                chain_id=chain,
            )
            return symbol
            
        except Exception as e:
            self.context.logger.error(f"Error getting token symbol: {e}")
            return None

    def _get_token_balances(self) -> Generator[Dict[str, int], None, None]:
        """Get current token balances using real infrastructure."""
        try:
            balances = {}
            chain = self.operating_chain
            safe_address = self.params.safe_contract_addresses.get(chain)

            if not safe_address:
                self.context.logger.error(f"No safe address found for chain {chain}")
                return {}

            # Get native balance
            native_balance = yield from self._get_native_balance(chain, safe_address)
            if native_balance:
                balances[ZERO_ADDRESS.lower()] = native_balance

            # Get ERC20 token balances for all assets
            for asset_address, asset_symbol in self.params.initial_assets.get(chain, {}).items():
                if asset_address != ZERO_ADDRESS:
                    token_balance = yield from self._get_token_balance(chain, safe_address, asset_address)
                    if token_balance:
                        balances[asset_address.lower()] = token_balance

            self.context.logger.info(f"Retrieved balances: {balances}")
            return balances
            
        except Exception as e:
            self.context.logger.error(f"Error getting token balances: {e}")
            return {}

    def _get_native_balance(self, chain: str, account: str) -> Generator[Optional[int], None, None]:
        """Get native balance using existing infrastructure."""
        try:
            ledger_api_response = yield from self.get_ledger_api_response(
                performative=LedgerApiMessage.Performative.GET_STATE,
                ledger_callable="get_balance",
                block_identifier="latest",
                account=account,
                chain_id=chain,
            )

            if ledger_api_response.performative != LedgerApiMessage.Performative.STATE:
                self.context.logger.error(f"Could not get native balance: {ledger_api_response}")
                return None

            return int(ledger_api_response.state.body["get_balance_result"])
            
        except Exception as e:
            self.context.logger.error(f"Error getting native balance: {e}")
            return None

    def _get_token_balance(self, chain: str, account: str, asset_address: str) -> Generator[Optional[int], None, None]:
        """Get token balance using existing infrastructure."""
        try:
            balance = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=asset_address,
                contract_public_id="valory/erc20:0.1.0",
                contract_callable="check_balance",
                data_key="token",
                account=account,
                chain_id=chain,
            )
            return balance
            
        except Exception as e:
            self.context.logger.error(f"Error getting token balance: {e}")
            return None

    def _submit_transaction(self, signed_tx_data: str, chain: str) -> Generator[Optional[str], None, None]:
        """Submit a signed transaction to the blockchain using existing ledger API infrastructure."""
        try:
            self.context.logger.info(f"Submitting transaction to chain {chain}")
            
            # Send the signed transaction using ledger API
            ledger_response = yield from self.get_ledger_api_response(
                performative=LedgerApiMessage.Performative.SEND_SIGNED_TRANSACTION,
                signed_transaction=signed_tx_data,
                chain_id=chain,
            )
            
            if ledger_response.performative == LedgerApiMessage.Performative.TRANSACTION_DIGEST:
                tx_hash = ledger_response.transaction_digest.body.get("tx_hash")
                self.context.logger.info(f"Transaction submitted successfully: {tx_hash}")
                return tx_hash
            else:
                self.context.logger.error(f"Failed to submit transaction: {ledger_response}")
                return None
                
        except Exception as e:
            self.context.logger.error(f"Error submitting transaction: {e}")
            return None

    def _wait_for_transaction_confirmation(self, tx_hash: str, chain: str, max_attempts: int = 30) -> Generator[bool, None, None]:
        """Wait for transaction confirmation using existing ledger API infrastructure."""
        try:
            self.context.logger.info(f"Waiting for transaction confirmation: {tx_hash}")
            
            for attempt in range(max_attempts):
                # Wait a bit before checking
                yield from self.sleep(2.0)
                
                # Check transaction status
                ledger_response = yield from self.get_ledger_api_response(
                    performative=LedgerApiMessage.Performative.GET_STATE,
                    ledger_callable="get_transaction_receipt",
                    tx_hash=tx_hash,
                    chain_id=chain,
                )
                
                if ledger_response.performative == LedgerApiMessage.Performative.STATE:
                    receipt = ledger_response.state.body.get("get_transaction_receipt_result")
                    if receipt and receipt.get("status") == 1:  # 1 = success
                        self.context.logger.info(f"Transaction confirmed: {tx_hash}")
                        return True
                    elif receipt and receipt.get("status") == 0:  # 0 = failed
                        self.context.logger.error(f"Transaction failed: {tx_hash}")
                        return False
                
                self.context.logger.info(f"Transaction pending, attempt {attempt + 1}/{max_attempts}")
            
            self.context.logger.warning(f"Transaction confirmation timeout: {tx_hash}")
            return False
            
        except Exception as e:
            self.context.logger.error(f"Error checking transaction confirmation: {e}")
            return False


    def _transfer_to_target(self) -> Generator:
        """Transfer USDC to target address."""
        self.context.logger.info(f"Transferring USDC to {self.target_address}")
        
        # Update status to show transfer in progress
        yield from self._update_withdrawal_status("withdrawing", "Successfully converted funds to USDC. Transfering funds to user wallet...")

        # Get USDC address for the operating chain
        usdc_addresses = json.loads(self.params.usdc_addresses)
        usdc_address = usdc_addresses.get(self.operating_chain)
        if not usdc_address:
            self.context.logger.error(f"No USDC address configured for chain {self.operating_chain}")
            return

        # Get USDC balance
        balances = yield from self._get_token_balances()
        usdc_balance = balances.get(usdc_address.lower(), 0)

        if usdc_balance <= 0:
            self.context.logger.warning("No USDC balance to transfer")
            return

        # Create transfer action
        transfer_action = {
            "action": "Transfer",
            "chain": self.operating_chain,
            "token_address": usdc_address,
            "to_address": self.target_address,
            "amount": usdc_balance
        }

        # Execute transfer
        tx_hash = yield from self._execute_action(transfer_action)
        if tx_hash:
            self.tx_hashes.append(tx_hash)
            self.context.logger.info(f"Transfer completed with tx: {tx_hash}")

    def _complete_withdrawal(self) -> Generator:
        """Complete the withdrawal process."""
        self.context.logger.info("Completing withdrawal process")

        # Update final status
        yield from self._update_withdrawal_status("completed", "Withdrawal complete.")

        # CRITICAL: Reset investing_paused flag to allow normal trading to resume
        yield from self._write_kv({
            "investing_paused": "false"
        })

        # Clear withdrawal request
        yield from self._write_kv({
            "withdrawal_request": "",
            "withdrawal_exit_actions": "",
            "withdrawal_action_index": "",
            "withdrawal_tx_hashes": ""
        })

        # Generate transaction link
        if self.tx_hashes:
            final_tx_hash = self.tx_hashes[-1]
            tx_link = self._generate_transaction_link(final_tx_hash)
            yield from self._write_kv({
                "withdrawal_tx_link": tx_link
            })

        self.context.logger.info(f"Withdrawal {self.withdrawal_id} completed successfully. Normal trading resumed.")

    def _update_withdrawal_status(self, status: str, message: str) -> Generator:
        """Update withdrawal status in KV store."""
        yield from self._write_kv({
            "withdrawal_status": status,
            "withdrawal_message": message,
            "withdrawal_updated_at": str(int(time.time()))
        })

    def _generate_transaction_link(self, tx_hash: str) -> str:
        """Generate blockchain explorer link for transaction."""
        try:
            # Get chain ID mapping from configuration
            chain_to_chain_id = json.loads(self.params.chain_to_chain_id_mapping)
            chain_id = chain_to_chain_id.get(self.operating_chain)
            
            if not chain_id:
                self.context.logger.warning(f"No chain ID found for chain: {self.operating_chain}")
                return f"https://etherscan.io/tx/{tx_hash}"  # Fallback
            
            # Define blockchain explorer URLs based on chain ID
            explorer_urls = {
                "1": "https://etherscan.io/tx/",      # Ethereum Mainnet
                "10": "https://optimistic.etherscan.io/tx/",  # Optimism
                "34443": "https://explorer.mode.network/tx/",  # Mode Network

            }
            
            explorer_url = explorer_urls.get(chain_id)
            if explorer_url:
                return f"{explorer_url}{tx_hash}"
            else:
                self.context.logger.warning(f"No explorer URL found for chain ID: {chain_id}")
                return f"https://etherscan.io/tx/{tx_hash}"  # Fallback
                
        except Exception as e:
            self.context.logger.error(f"Error generating transaction link: {e}")
            return f"https://etherscan.io/tx/{tx_hash}"  # Fallback



    def teardown(self) -> None:
        """Tear down the behaviour."""
        self.context.logger.info("Withdrawal behaviour teardown complete") 
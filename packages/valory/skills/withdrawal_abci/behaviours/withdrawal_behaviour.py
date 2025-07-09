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
import uuid
from typing import Dict, Generator, List, Optional, Any

from aea.skills.base import BaseBehaviour
from packages.valory.protocols.kv_store import KvStoreMessage
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    LiquidityTraderBaseBehaviour,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.liquidity_trader_abci.models import ZERO_ADDRESS
from packages.valory.skills.liquidity_trader_abci.behaviours.base import DexType, ETHER_VALUE, SAFE_TX_GAS, ERC20_DECIMALS, INTEGRATOR, HTTP_OK, ZERO_ADDRESS
from packages.valory.contracts.gnosis_safe.contract import GnosisSafeContract
from packages.valory.contracts.gnosis_safe.contract import SafeOperation
from packages.valory.skills.liquidity_trader_abci.payloads import hash_payload_to_hex
from packages.valory.contracts.balancer_vault.contract import BalancerVaultContract
from packages.valory.contracts.uniswap_v3_non_fungible_position_manager.contract import UniswapV3NonFungiblePositionManagerContract
from packages.valory.contracts.velodrome_pool.contract import VelodromePoolContract
from packages.valory.contracts.uniswap_v3_router.contract import UniswapV3RouterContract
from packages.valory.contracts.erc20.contract import ERC20

# Import decision making functions to reuse existing logic
from packages.valory.skills.liquidity_trader_abci.behaviours.decision_making import DecisionMakingBehaviour
from packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy import EvaluateStrategyBehaviour


class WithdrawalBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that handles withdrawal requests independently."""

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

            # Step 3: Convert all tokens to USDC
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
            # Reuse the existing _build_exit_pool_action logic from EvaluateStrategyBehaviour
            # We need to adapt it for our use case where we have a position and want to exit it
            
            # Set the position_to_exit (same as evaluate strategy does)
            self.position_to_exit = position
            
            # Extract actual tokens from the position using existing logic
            dex_type = position.get("dex_type")
            num_of_tokens_required = 1 if dex_type == DexType.STURDY.value else 2
            
            # Use the existing _build_tokens_from_position method to get actual tokens
            tokens = EvaluateStrategyBehaviour._build_tokens_from_position(
                self, position, num_of_tokens_required
            )
            
            if not tokens or len(tokens) < num_of_tokens_required:
                self.context.logger.error(
                    f"Failed to extract {num_of_tokens_required} tokens from position: {position}"
                )
                return None
            
            # Reuse the existing logic with actual tokens
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
        self.context.logger.info(f"Executing exit action {current_index + 1}/{len(actions)}: {action}")

        # Execute the action using existing infrastructure
        tx_hash = yield from self._execute_action(action)
        
        if tx_hash:
            self.tx_hashes.append(tx_hash)
            self.context.logger.info(f"Exit action completed with tx: {tx_hash}")
        else:
            self.context.logger.error(f"Exit action {current_index + 1} failed")
            # Continue anyway to try other actions

        # Update action index
        yield from self._write_kv({
            "withdrawal_action_index": str(current_index + 1),
            "withdrawal_tx_hashes": json.dumps(self.tx_hashes)
        })

        # Continue with next action
        yield from self._execute_exit_actions()

    def _convert_tokens_to_usdc(self) -> Generator:
        """Convert all tokens to USDC using real infrastructure."""
        self.context.logger.info("Converting tokens to USDC")

        # Get token balances
        balances = yield from self._get_token_balances()
        
        if not balances:
            self.context.logger.info("No token balances found for conversion")
            return

        # Get USDC address for the operating chain
        usdc_addresses = json.loads(self.params.usdc_addresses)
        usdc_address = usdc_addresses.get(self.operating_chain)
        if not usdc_address:
            self.context.logger.error(f"No USDC address configured for chain {self.operating_chain}")
            return

        # Create swap actions for each token (except USDC)
        swap_actions = []
        for token_address, balance in balances.items():
            if token_address.lower() != usdc_address.lower() and balance > 0:
                # Check if balance is significant enough to swap
                min_swap_amount = 1000  # Minimum amount in wei to consider swapping
                if balance < min_swap_amount:
                    self.context.logger.info(f"Skipping swap for {token_address}: balance {balance} below minimum {min_swap_amount}")
                    continue
                    
                swap_action = {
                    "action": "Swap",
                    "dex_type": "LiFi",  
                    "chain": self.operating_chain,
                    "token_in": token_address,
                    "token_out": usdc_address,
                    "amount": balance
                }
                swap_actions.append(swap_action)

        self.context.logger.info(f"Prepared {len(swap_actions)} swap actions to USDC")

        # Execute swap actions
        successful_swaps = 0
        for action in swap_actions:
            self.context.logger.info(f"Executing swap: {action['token_in']} -> USDC")
            tx_hash = yield from self._execute_action(action)
            if tx_hash:
                self.tx_hashes.append(tx_hash)
                successful_swaps += 1
                self.context.logger.info(f"Swap completed with tx: {tx_hash}")
            else:
                self.context.logger.error(f"Swap failed for {action['token_in']} -> USDC")

        self.context.logger.info(f"Successfully converted {successful_swaps}/{len(swap_actions)} tokens to USDC")

    def _execute_action(self, action: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        """Execute a single action and return transaction hash."""
        try:
            action_type = action.get("action")
            
            if action_type == "exit_pool":
                return (yield from self._execute_exit_action(action))
            elif action_type == "swap":
                return (yield from self._execute_swap_action(action))
            elif action_type == "transfer":
                return (yield from self._execute_transfer_action(action))
            else:
                self.context.logger.error(f"Unknown action type: {action_type}")
                return None
                
        except Exception as e:
            self.context.logger.error(f"Error executing action {action}: {e}")
            return None

    def _execute_exit_action(self, action: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        """Execute pool exit action with gas simulation."""
        try:
            chain = action.get("chain")
            pool_address = action.get("pool_address")
            pool_type = action.get("pool_type")
            
            self.context.logger.info(f"Executing exit action for {pool_type} pool {pool_address} on {chain}")
            
            # Get exit transaction data
            if pool_type == "balancer":
                tx_data = yield from self._get_balancer_exit_tx_data(action)
            elif pool_type == "uniswap":
                tx_data = yield from self._get_uniswap_exit_tx_data(action)
            elif pool_type == "velodrome":
                tx_data = yield from self._get_velodrome_exit_tx_data(action)
            else:
                self.context.logger.error(f"Unsupported pool type: {pool_type}")
                return None
                
            if not tx_data:
                return None
                
            # Simulate transaction to check gas sufficiency
            is_ok = yield from self._simulate_transaction(
                to_address=pool_address,
                data=tx_data,
                token="0x0000000000000000000000000000000000000000",  # ETH
                amount=0,
                chain=chain,
            )
            
            if not is_ok:
                self.context.logger.error("Transaction simulation failed - insufficient gas or other error")
                yield from self._update_withdrawal_status(
                    "failed",
                    "Transaction simulation failed - insufficient gas or other error. Please ensure the safe has sufficient ETH for gas."
                )
                return None
                
            # Execute the transaction
            tx_hash = yield from self._execute_safe_transaction(
                to_address=pool_address,
                data=tx_data,
                chain=chain
            )
            
            return tx_hash
            
        except Exception as e:
            self.context.logger.error(f"Error executing exit action: {e}")
            return None

    def _execute_swap_action(self, action: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        """Execute swap action with gas simulation."""
        try:
            chain = action.get("chain")
            from_token = action.get("from_token")
            to_token = action.get("to_token")
            amount = action.get("amount")
            
            self.context.logger.info(f"Executing swap action: {amount} {from_token} -> {to_token} on {chain}")
            
            # Get swap transaction data
            tx_data = yield from self._get_swap_tx_data(action)
            if not tx_data:
                return None
                
            # Simulate transaction to check gas sufficiency
            is_ok = yield from self._simulate_transaction(
                to_address=action.get("router_address"),
                data=tx_data,
                token=from_token,
                amount=amount,
                chain=chain,
            )
            
            if not is_ok:
                self.context.logger.error("Transaction simulation failed - insufficient gas or other error")
                yield from self._update_withdrawal_status(
                    "failed",
                    "Transaction simulation failed - insufficient gas or other error. Please ensure the safe has sufficient ETH for gas."
                )
                return None
                
            # Execute the transaction
            tx_hash = yield from self._execute_safe_transaction(
                to_address=action.get("router_address"),
                data=tx_data,
                chain=chain
            )
            
            return tx_hash
            
        except Exception as e:
            self.context.logger.error(f"Error executing swap action: {e}")
            return None

    def _execute_transfer_action(self, action: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        """Execute transfer action with gas simulation."""
        try:
            chain = action.get("chain")
            to_address = action.get("to_address")
            amount = action.get("amount")
            token = action.get("token")
            
            self.context.logger.info(f"Executing transfer action: {amount} {token} to {to_address} on {chain}")
            
            # Get transfer transaction data
            tx_data = yield from self._get_transfer_tx_data(action)
            if not tx_data:
                return None
                
            # Simulate transaction to check gas sufficiency
            is_ok = yield from self._simulate_transaction(
                to_address=to_address,
                data=tx_data,
                token=token,
                amount=amount,
                chain=chain,
            )
            
            if not is_ok:
                self.context.logger.error("Transaction simulation failed - insufficient gas or other error")
                yield from self._update_withdrawal_status(
                    "failed",
                    "Transaction simulation failed - insufficient gas or other error. Please ensure the safe has sufficient ETH for gas."
                )
                return None
                
            # Execute the transaction
            tx_hash = yield from self._execute_safe_transaction(
                to_address=to_address,
                data=tx_data,
                chain=chain
            )
            
            return tx_hash
            
        except Exception as e:
            self.context.logger.error(f"Error executing transfer action: {e}")
            return None

    def _simulate_transaction(
        self,
        to_address: str,
        data: bytes,
        token: str,
        amount: int,
        chain: str,
        **kwargs: Any,
    ) -> Generator[None, None, bool]:
        """Simulate transaction using Tenderly to check gas sufficiency (reuse existing function)."""
        # Reuse the existing simulation function from DecisionMakingBehaviour
        return (yield from DecisionMakingBehaviour._simulate_transaction(
            self, to_address, data, token, amount, chain, **kwargs
        ))

    def _get_signature(self, owner: str) -> str:
        """Get signature for safe transaction (reuse existing function)."""
        # Reuse the existing signature function from DecisionMakingBehaviour
        return DecisionMakingBehaviour._get_signature(self, owner)

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
            payload_string = hash_payload_to_hex(**tx_params)
            
            # Send transaction
            tx_hash = yield from self.send_transaction(payload_string, chain)
            return tx_hash
            
        except Exception as e:
            self.context.logger.error(f"Error executing safe transaction: {e}")
            return None

    def _get_token_decimals(self, chain: str, token_address: str) -> Generator[Optional[int], None, None]:
        """Get token decimals."""
        try:
            if token_address == ZERO_ADDRESS:
                return 18  # ETH has 18 decimals

            decimals = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=token_address,
                contract_public_id="valory/erc20:0.1.0",
                contract_callable="decimals",
                data_key="decimals",
                chain_id=chain,
            )
            return decimals
            
        except Exception as e:
            self.context.logger.error(f"Error getting token decimals: {e}")
            return None

    def _get_token_symbol(self, chain: str, token_address: str) -> Generator[Optional[str], None, None]:
        """Get token symbol."""
        try:
            if token_address == ZERO_ADDRESS:
                return "ETH"  # ETH symbol

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
                to_address=to_address,
                amount=amount,
                chain_id=chain,
            )

            if tx_hash:
                # Build safe transaction
                safe_tx_hash = yield from self.contract_interact(
                    performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                    contract_address=safe_address,
                    contract_public_id="valory/gnosis_safe:0.1.0",
                    contract_callable="get_raw_safe_transaction_hash",
                    data_key="tx_hash",
                    to_address=token_address,
                    value=0,
                    data=tx_hash,
                    operation=SafeOperation.CALL.value,
                    safe_tx_gas=SAFE_TX_GAS,
                    chain_id=chain,
                )

                if not safe_tx_hash:
                    self.context.logger.error("Failed to create safe transaction for transfer")
                    return None

                safe_tx_hash = safe_tx_hash[2:]
                self._last_safe_tx_hash = safe_tx_hash  # Store for signature generation

                # Build payload
                payload_string = hash_payload_to_hex(
                    safe_tx_hash=safe_tx_hash,
                    ether_value=0,
                    safe_tx_gas=SAFE_TX_GAS,
                    operation=SafeOperation.CALL.value,
                    to_address=token_address,
                    data=tx_hash,
                )
                
                # Get signature for the transaction
                signature = self._get_signature(safe_address)
                if not signature:
                    self.context.logger.error("Failed to generate signature for transfer transaction")
                    return None

                # Submit the transaction to the blockchain
                tx_receipt = yield from self._submit_transaction(payload_string, chain)
                if not tx_receipt:
                    self.context.logger.error("Failed to submit transfer transaction")
                    return None
                
                # Wait for transaction confirmation
                confirmed = yield from self._wait_for_transaction_confirmation(tx_receipt, chain)
                if not confirmed:
                    self.context.logger.error("Transfer transaction failed or timed out")
                    return None
                
                self.context.logger.info(f"Transfer transaction executed successfully: {tx_receipt}")
                return tx_receipt

            return None
            
        except Exception as e:
            self.context.logger.error(f"Error executing transfer action: {e}")
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
                "8453": "https://basescan.org/tx/",   # Base
                "137": "https://polygonscan.com/tx/", # Polygon
                "42161": "https://arbiscan.io/tx/",   # Arbitrum
                "56": "https://bscscan.com/tx/",      # BSC
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

    def _get_balancer_exit_tx_data(self, action: Dict[str, Any]) -> Generator[None, None, Optional[bytes]]:
        """Get Balancer pool exit transaction data."""
        try:
            chain = action.get("chain")
            pool_address = action.get("pool_address")
            pool_id = action.get("pool_id")
            
            # Get exit transaction data from Balancer contract
            tx_data = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=pool_address,
                contract_public_id=BalancerVaultContract.contract_id,
                contract_callable="exit_pool",
                data_key="data",
                pool_id=pool_id,
                sender=self.params.safe_contract_addresses.get(chain),
                recipient=self.params.safe_contract_addresses.get(chain),
                request={
                    "assets": action.get("assets", []),
                    "min_amounts_out": action.get("min_amounts_out", []),
                    "user_data": b"",
                    "to_internal_balance": False
                },
                chain_id=chain,
            )
            
            return tx_data
            
        except Exception as e:
            self.context.logger.error(f"Error getting Balancer exit tx data: {e}")
            return None

    def _get_uniswap_exit_tx_data(self, action: Dict[str, Any]) -> Generator[None, None, Optional[bytes]]:
        """Get Uniswap pool exit transaction data."""
        try:
            chain = action.get("chain")
            pool_address = action.get("pool_address")
            token_id = action.get("token_id")
            
            # Get exit transaction data from Uniswap contract
            tx_data = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=pool_address,
                contract_public_id=UniswapV3NonFungiblePositionManagerContract.contract_id,
                contract_callable="decrease_liquidity",
                data_key="data",
                token_id=token_id,
                liquidity=action.get("liquidity", 0),
                amount0_min=action.get("amount0_min", 0),
                amount1_min=action.get("amount1_min", 0),
                deadline=action.get("deadline", 0),
                chain_id=chain,
            )
            
            return tx_data
            
        except Exception as e:
            self.context.logger.error(f"Error getting Uniswap exit tx data: {e}")
            return None

    def _get_velodrome_exit_tx_data(self, action: Dict[str, Any]) -> Generator[None, None, Optional[bytes]]:
        """Get Velodrome pool exit transaction data."""
        try:
            chain = action.get("chain")
            pool_address = action.get("pool_address")
            
            # Get exit transaction data from Velodrome contract
            tx_data = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=pool_address,
                contract_public_id=VelodromePoolContract.contract_id,
                contract_callable="remove_liquidity",
                data_key="data",
                lp_amount=action.get("lp_amount", 0),
                min_token0=action.get("min_token0", 0),
                min_token1=action.get("min_token1", 0),
                to=self.params.safe_contract_addresses.get(chain),
                deadline=action.get("deadline", 0),
                chain_id=chain,
            )
            
            return tx_data
            
        except Exception as e:
            self.context.logger.error(f"Error getting Velodrome exit tx data: {e}")
            return None

    def _get_swap_tx_data(self, action: Dict[str, Any]) -> Generator[None, None, Optional[bytes]]:
        """Get swap transaction data using Uniswap router."""
        try:
            chain = action.get("chain")
            router_address = action.get("router_address")
            from_token = action.get("from_token")
            to_token = action.get("to_token")
            amount = action.get("amount")
            
            # Get swap transaction data from Uniswap router
            tx_data = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=router_address,
                contract_public_id=UniswapV3RouterContract.contract_id,
                contract_callable="exact_input_single",
                data_key="data",
                params={
                    "token_in": from_token,
                    "token_out": to_token,
                    "fee": action.get("fee", 3000),
                    "recipient": self.params.safe_contract_addresses.get(chain),
                    "deadline": action.get("deadline", 0),
                    "amount_in": amount,
                    "amount_out_minimum": action.get("amount_out_minimum", 0),
                    "sqrt_price_limit_x96": 0
                },
                chain_id=chain,
            )
            
            return tx_data
            
        except Exception as e:
            self.context.logger.error(f"Error getting swap tx data: {e}")
            return None

    def _get_transfer_tx_data(self, action: Dict[str, Any]) -> Generator[None, None, Optional[bytes]]:
        """Get transfer transaction data."""
        try:
            chain = action.get("chain")
            to_address = action.get("to_address")
            amount = action.get("amount")
            token = action.get("token")
            
            if token == "0x0000000000000000000000000000000000000000":
                # ETH transfer - empty data
                return b""
            else:
                # ERC20 transfer
                tx_data = yield from self.contract_interact(
                    performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                    contract_address=token,
                    contract_public_id=ERC20.contract_id,
                    contract_callable="transfer",
                    data_key="data",
                    to=to_address,
                    amount=amount,
                    chain_id=chain,
                )
                
                return tx_data
            
        except Exception as e:
            self.context.logger.error(f"Error getting transfer tx data: {e}")
            return None

    def teardown(self) -> None:
        """Tear down the behaviour."""
        self.context.logger.info("Withdrawal behaviour teardown complete") 
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

"""This module contains the handlers for the 'withdrawal_abci' skill."""

import json
import time
import uuid
from typing import Dict, Any, Optional, Generator

from aea.protocols.base import Message
from aea.skills.base import Handler

from packages.valory.protocols.http import HttpMessage
from packages.valory.protocols.kv_store import KvStoreMessage, KvStoreDialogues, KvStoreDialogue
from packages.valory.connections.kv_store.connection import KV_STORE_CONNECTION_PUBLIC_ID


class WithdrawalHttpHandler(Handler):
    """Handler for HTTP requests related to withdrawal."""

    SUPPORTED_PROTOCOL = HttpMessage.protocol_id

    def setup(self) -> None:
        """Set up the handler."""
        self.context.logger.info("Withdrawal HTTP handler setup complete")

    def handle(self, message: Message) -> None:
        """Handle the HTTP message."""
        if message.performative == HttpMessage.Performative.REQUEST:
            self._handle_http_request(message)
        else:
            self.context.logger.warning(f"Unexpected HTTP message: {message}")

    def _handle_http_request(self, message: HttpMessage) -> None:
        """Handle HTTP request."""
        try:
            method = message.method
            url = message.url
            body = message.body.decode() if message.body else "{}"
            
            self.context.logger.info(f"Handling HTTP request: {method} {url}")

            if method == "GET" and url == "/withdrawal/amount":
                response = self._handle_get_withdrawal_amount()
            elif method == "POST" and url == "/withdrawal/initiate":
                response = self._handle_initiate_withdrawal(json.loads(body))
            elif method == "GET" and url == "/withdrawal/status":
                response = self._handle_get_withdrawal_status()
            else:
                response = {
                    "error": "Endpoint not found",
                    "status_code": 404
                }

            self._send_http_response(message, response)

        except Exception as e:
            self.context.logger.error(f"Error handling HTTP request: {e}")
            error_response = {
                "error": str(e),
                "status_code": 500
            }
            self._send_http_response(message, error_response)

    def _handle_get_withdrawal_amount(self) -> Dict[str, Any]:
        """Handle GET /withdrawal/amount request."""
        try:
            # Get portfolio data from KV store
            portfolio_data = self._read_kv_sync(keys=("portfolio_data",))
            
            if not portfolio_data or not portfolio_data.get("portfolio_data"):
                return {
                    "error": "Portfolio data not available",
                    "status_code": 404
                }

            portfolio = json.loads(portfolio_data["portfolio_data"])
            
            # Extract withdrawal amount data
            total_value = portfolio.get("total_value_usd", 0)
            operating_chain = self.context.params.target_investment_chains[0]
            safe_address = self.context.params.safe_contract_addresses.get(operating_chain)
            
            # Get asset breakdown
            asset_breakdown = portfolio.get("asset_breakdown", [])

            return {
                "amount": int(total_value * 10**6),  # Convert to USDC decimals
                "total_value_usd": float(total_value),
                "chain": operating_chain,
                "safe_address": safe_address,
                "asset_breakdown": asset_breakdown,
                "status_code": 200
            }

        except Exception as e:
            self.context.logger.error(f"Error getting withdrawal amount: {e}")
            return {
                "error": str(e),
                "status_code": 500
            }

    def _handle_initiate_withdrawal(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle POST /withdrawal/initiate request."""
        try:
            # Validate request
            target_address = request_data.get("target_address")
            if not target_address:
                return {
                    "error": "target_address is required",
                    "status_code": 400
                }

            # Check if withdrawal is already in progress
            existing_withdrawal = self._read_kv_sync(keys=("withdrawal_status",))
            if existing_withdrawal and existing_withdrawal.get("withdrawal_status") in ["pending", "withdrawing"]:
                return {
                    "error": "Withdrawal already in progress",
                    "status_code": 409
                }

            # Generate withdrawal ID
            withdrawal_id = str(uuid.uuid4())
            
            # Get portfolio data
            portfolio_data = self._read_kv_sync(keys=("portfolio_data",))
            portfolio = json.loads(portfolio_data.get("portfolio_data", "{}")) if portfolio_data else {}
            total_value = portfolio.get("total_value_usd", 0)
            operating_chain = self.context.params.target_investment_chains[0]
            safe_address = self.context.params.safe_contract_addresses.get(operating_chain)

            # Create withdrawal request
            withdrawal_request = {
                "withdrawal_id": withdrawal_id,
                "target_address": target_address,
                "total_value_usd": total_value,
                "operating_chain": operating_chain,
                "safe_address": safe_address,
                "requested_at": int(time.time())
            }

            # CRITICAL: Set investing_paused immediately to prevent new investments
            self._write_kv_sync({
                "investing_paused": "true"
            })
            
            # Store withdrawal request in KV store
            withdrawal_data = {
                "withdrawal_request": json.dumps(withdrawal_request),
                "withdrawal_status": "withdrawing",  # Start execution immediately
                "withdrawal_message": "Withdraw initiated. Preparing your funds...",
                "withdrawal_id": withdrawal_id,
                "withdrawal_target_address": target_address,
                "withdrawal_requested_at": str(int(time.time())),
                "withdrawal_portfolio_value": str(total_value),
                "withdrawal_chain": operating_chain,
                "withdrawal_safe_address": safe_address
            }
            
            self._write_kv_sync(withdrawal_data)

            self.context.logger.info(f"Withdrawal initiated and execution started: {withdrawal_id} -> {target_address}")

            return {
                "withdrawal_id": withdrawal_id,
                "status": "withdrawing",
                "message": "Withdraw initiated. Preparing your funds...",
                "target_address": target_address,
                "total_value_usd": total_value,
                "status_code": 200
            }

        except Exception as e:
            self.context.logger.error(f"Error initiating withdrawal: {e}")
            return {
                "error": str(e),
                "status_code": 500
            }

    def _handle_get_withdrawal_status(self) -> Dict[str, Any]:
        """Handle GET /withdrawal/status request."""
        try:
            # Get withdrawal data from KV store
            withdrawal_data = self._read_kv_sync(keys=(
                "withdrawal_id", "withdrawal_status", "withdrawal_message", 
                "withdrawal_target_address", "withdrawal_chain", "withdrawal_safe_address",
                "withdrawal_tx_hashes", "withdrawal_requested_at", "withdrawal_completed_at",
                "withdrawal_portfolio_value", "withdrawal_tx_link"
            ))
            
            withdrawal_id = withdrawal_data.get("withdrawal_id", "")
            status = withdrawal_data.get("withdrawal_status", "unknown")
            message = withdrawal_data.get("withdrawal_message", "No withdrawal found")
            target_address = withdrawal_data.get("withdrawal_target_address", "")
            chain = withdrawal_data.get("withdrawal_chain", "")
            safe_address = withdrawal_data.get("withdrawal_safe_address", "")
            tx_link = withdrawal_data.get("withdrawal_tx_link", "")
            requested_at = withdrawal_data.get("withdrawal_requested_at", "")
            completed_at = withdrawal_data.get("withdrawal_completed_at", "")
            portfolio_value = withdrawal_data.get("withdrawal_portfolio_value", "")
            tx_hashes = withdrawal_data.get("withdrawal_tx_hashes", "[]")

            # Provide better default messages for each status
            if status == "unknown":
                message = "No withdrawal found with this ID"
            elif status == "pending":
                message = "Withdraw request received. Waiting for agent to start processing."
            elif status == "withdrawing":
                message = "Withdraw initiated. Preparing your funds..."
            elif status == "completed":
                message = "Withdrawal complete."
            elif status == "failed":
                message = "Withdrawal failed. Please contact support."

            response_data = {
                "status": status,
                "message": message,
                "target_address": target_address,
                "chain": chain,
                "safe_address": safe_address,
                "status_code": 200
            }

            # Add timestamps if available
            if requested_at:
                response_data["requested_at"] = requested_at
            if completed_at:
                response_data["completed_at"] = completed_at

            # Add portfolio value if available
            if portfolio_value:
                try:
                    value = float(portfolio_value)
                    response_data["estimated_value_usd"] = round(value, 2)
                except (ValueError, TypeError):
                    pass

            # Add transaction details if available
            if tx_hashes and tx_hashes != "[]":
                try:
                    tx_hashes_list = json.loads(tx_hashes)
                    response_data["transaction_hashes"] = tx_hashes_list
                    response_data["transaction_count"] = len(tx_hashes_list)
                    
                    # Add transaction link if available
                    if tx_link:
                        response_data["transaction_link"] = tx_link
                        
                except json.JSONDecodeError:
                    pass

            return response_data

        except Exception as e:
            self.context.logger.error(f"Error getting withdrawal status: {e}")
            return {
                "error": str(e),
                "status_code": 500
            }

    def _read_kv_sync(self, keys: tuple) -> Dict[str, Any]:
        """Synchronous KV store read operation."""
        try:
            # This is a simplified synchronous read - in production you'd want proper async handling
            # For now, we'll use a simple approach that works with the current setup
            result = {}
            for key in keys:
                # Try to get from shared state first (fallback)
                value = self.context.shared_state.get(key, "")
                if value:
                    result[key] = value
            return result
        except Exception as e:
            self.context.logger.error(f"Error reading from KV store: {e}")
            return {}

    def _write_kv_sync(self, data: Dict[str, str]) -> None:
        """Synchronous KV store write operation."""
        try:
            # This is a simplified synchronous write - in production you'd want proper async handling
            # For now, we'll use a simple approach that works with the current setup
            for key, value in data.items():
                self.context.shared_state.set(key, value)
        except Exception as e:
            self.context.logger.error(f"Error writing to KV store: {e}")

    def _send_http_response(self, original_message: HttpMessage, response_data: Dict[str, Any]) -> None:
        """Send HTTP response."""
        try:
            status_code = response_data.get("status_code", 200)
            body = json.dumps(response_data).encode("utf-8")
            
            response = HttpMessage(
                performative=HttpMessage.Performative.RESPONSE,
                dialogue_reference=original_message.dialogue_reference,
                target=original_message.message_id,
                message_id=original_message.message_id + 1,
                version=original_message.version,
                headers="Content-Type: application/json",
                status_code=status_code,
                status_text="OK" if status_code == 200 else "Error",
                body=body,
            )
            
            self.context.outbox.put_message(message=response)
            
        except Exception as e:
            self.context.logger.error(f"Error sending HTTP response: {e}")

    def teardown(self) -> None:
        """Tear down the handler."""
        self.context.logger.info("Withdrawal HTTP handler teardown complete") 
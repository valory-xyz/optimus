# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2022-2023 Valory AG
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

"""This class contains a wrapper for Velodrome Voter contract interface."""

import logging
from typing import Any

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi


PUBLIC_ID = PublicId.from_str("valory/velodrome_voter:0.1.0")

_logger = logging.getLogger(__name__)


class VelodromeVoterContract(Contract):
    """The Velodrome Voter contract."""

    contract_id = PUBLIC_ID

    @classmethod
    def gauges(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        pool_address: str,
    ) -> JSONLike:
        """Get the gauge address for a given pool."""
        _logger.debug(f"Getting gauge for pool: {pool_address}")
        
        checksumed_pool = ledger_api.api.to_checksum_address(pool_address)
        contract_instance = cls.get_instance(ledger_api, contract_address)
        gauge_address = contract_instance.functions.gauges(checksumed_pool).call()
        _logger.debug(f"Gauge address for pool {pool_address}: {gauge_address}")
        return dict(gauge=gauge_address)

    @classmethod
    def is_gauge(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        gauge_address: str,
    ) -> JSONLike:
        """Check if an address is a valid gauge."""
        _logger.debug(f"Validating gauge address: {gauge_address}")
        
        checksumed_gauge = ledger_api.api.to_checksum_address(gauge_address)
        contract_instance = cls.get_instance(ledger_api, contract_address)
        is_valid = contract_instance.functions.isGauge(checksumed_gauge).call()
        _logger.debug(f"Gauge {gauge_address} is valid: {is_valid}")
        return dict(is_gauge=is_valid)

    @classmethod
    def pool_for_gauge(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        gauge_address: str,
    ) -> JSONLike:
        """Get the pool address for a given gauge."""
        _logger.debug(f"Getting pool for gauge: {gauge_address}")
        
        checksumed_gauge = ledger_api.api.to_checksum_address(gauge_address)
        contract_instance = cls.get_instance(ledger_api, contract_address)
        pool_address = contract_instance.functions.poolForGauge(checksumed_gauge).call()
        _logger.debug(f"Pool address for gauge {gauge_address}: {pool_address}")
        return dict(pool=pool_address)

    @classmethod
    def is_alive(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        gauge_address: str,
    ) -> JSONLike:
        """Check if a gauge is alive (active)."""
        _logger.debug(f"Checking if gauge is alive: {gauge_address}")
        
        checksumed_gauge = ledger_api.api.to_checksum_address(gauge_address)
        contract_instance = cls.get_instance(ledger_api, contract_address)
        is_alive = contract_instance.functions.isAlive(checksumed_gauge).call()
        _logger.debug(f"Gauge {gauge_address} is alive: {is_alive}")
        return dict(is_alive=is_alive)

    @classmethod
    def validate_gauge_address(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        gauge_address: str,
    ) -> JSONLike:
        """Validate a gauge address by checking if it's a valid gauge and is alive."""
        _logger.debug(f"Validating gauge address: {gauge_address}")
        
        # Check if it's a valid gauge
        is_gauge_result = cls.is_gauge(ledger_api, contract_address, gauge_address)
        if "error" in is_gauge_result:
            return is_gauge_result
        
        if not is_gauge_result.get("is_gauge", False):
            error_msg = f"Address {gauge_address} is not a valid gauge"
            _logger.error(error_msg)
            return dict(error=error_msg, is_valid=False)
        
        # Check if it's alive
        is_alive_result = cls.is_alive(ledger_api, contract_address, gauge_address)
        if "error" in is_alive_result:
            return is_alive_result
        
        is_alive = is_alive_result.get("is_alive", False)
        is_valid = is_alive  # A gauge is considered valid if it's alive
        
        _logger.debug(f"Gauge {gauge_address} validation result: {is_valid}")
        return dict(
            is_valid=is_valid,
            is_gauge=True,
            is_alive=is_alive
        )

    @classmethod
    def length(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get the total number of pools."""
        _logger.debug("Getting total number of pools")
        
        contract_instance = cls.get_instance(ledger_api, contract_address)
        total_pools = contract_instance.functions.length().call()
        _logger.debug(f"Total number of pools: {total_pools}")
        return dict(length=total_pools)

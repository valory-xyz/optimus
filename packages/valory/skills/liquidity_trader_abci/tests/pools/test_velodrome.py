# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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

"""Test the pools/velodrome.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import sys
from enum import Enum

from packages.valory.skills.liquidity_trader_abci.pools.velodrome import (
    API_CACHE_SIZE,
    DEFAULT_DAYS,
    INT_MAX,
    MAX_TICK,
    MAX_WAIT_TIME,
    MIN_TICK,
    PRICE_CHECK_INTERVAL,
    PRICE_VOLATILITY_THRESHOLD,
    TICK_TO_PERCENTAGE_FACTOR,
    WAITING_PERIOD,
    ZERO_ADDRESS,
    AllocationStatus,
    VelodromePoolBehaviour,
)


def test_import() -> None:
    """Test that the velodrome module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.pools.velodrome  # noqa


class TestAllocationStatus:
    """Test AllocationStatus enum."""

    def test_is_enum(self) -> None:
        """Test AllocationStatus is an Enum."""
        assert issubclass(AllocationStatus, Enum)

    def test_values(self) -> None:
        """Test AllocationStatus values."""
        assert AllocationStatus.READY.value == "ready"
        assert AllocationStatus.WAITING.value == "waiting"
        assert AllocationStatus.TIMEOUT.value == "timeout"
        assert AllocationStatus.FAILED.value == "failed"

    def test_member_count(self) -> None:
        """Test AllocationStatus has 4 members."""
        assert len(AllocationStatus) == 4


class TestConstants:
    """Test module-level constants."""

    def test_price_volatility_threshold(self) -> None:
        """Test PRICE_VOLATILITY_THRESHOLD constant."""
        assert PRICE_VOLATILITY_THRESHOLD == 0.02

    def test_default_days(self) -> None:
        """Test DEFAULT_DAYS constant."""
        assert DEFAULT_DAYS == 30

    def test_api_cache_size(self) -> None:
        """Test API_CACHE_SIZE constant."""
        assert API_CACHE_SIZE == 128

    def test_waiting_period(self) -> None:
        """Test WAITING_PERIOD constant."""
        assert WAITING_PERIOD == 5

    def test_max_wait_time(self) -> None:
        """Test MAX_WAIT_TIME constant."""
        assert MAX_WAIT_TIME == 600

    def test_price_check_interval(self) -> None:
        """Test PRICE_CHECK_INTERVAL constant."""
        assert PRICE_CHECK_INTERVAL == 5

    def test_tick_to_percentage_factor(self) -> None:
        """Test TICK_TO_PERCENTAGE_FACTOR constant."""
        assert TICK_TO_PERCENTAGE_FACTOR == 10000

    def test_zero_address(self) -> None:
        """Test ZERO_ADDRESS constant."""
        assert ZERO_ADDRESS == "0x0000000000000000000000000000000000000000"

    def test_min_tick(self) -> None:
        """Test MIN_TICK constant."""
        assert MIN_TICK == -887272

    def test_max_tick(self) -> None:
        """Test MAX_TICK constant."""
        assert MAX_TICK == 887272

    def test_int_max(self) -> None:
        """Test INT_MAX constant."""
        assert INT_MAX == sys.maxsize

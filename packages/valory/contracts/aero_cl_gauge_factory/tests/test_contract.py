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

"""Tests for the AeroCLGaugeFactoryContract contract."""

from unittest.mock import MagicMock, patch

from packages.valory.contracts.aero_cl_gauge_factory.contract import (
    AeroCLGaugeFactoryContract,
)

FACTORY = "0xf437f62053ACB121665b8fDBfeA8F214FB8E589d"
GAUGE = "0x2222222222222222222222222222222222222222"


class TestEffectivePenaltyConfig:
    """Tests for the effective_penalty_config method."""

    def test_returns_named_penalty_config(self) -> None:
        """Test that the PenaltyConfig tuple is returned as named fields."""
        ledger_api = MagicMock()
        ledger_api.api.to_checksum_address.return_value = GAUGE
        instance = MagicMock()
        instance.functions.effectivePenaltyConfig.return_value.call.return_value = (
            7,
            1_000_000,
        )

        with patch.object(
            AeroCLGaugeFactoryContract, "get_instance", return_value=instance
        ):
            result = AeroCLGaugeFactoryContract.effective_penalty_config(
                ledger_api, FACTORY, GAUGE.lower()
            )

        assert result == {
            "penalty_config": {"min_stake_blocks": 7, "penalty_rate": 1_000_000}
        }
        instance.functions.effectivePenaltyConfig.assert_called_once_with(GAUGE)

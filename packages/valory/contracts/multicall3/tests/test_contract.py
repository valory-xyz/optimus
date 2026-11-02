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

"""Tests for the Multicall3Contract contract."""

from unittest.mock import MagicMock, patch

from packages.valory.contracts.multicall3.contract import Multicall3Contract

MULTICALL3 = "0xcA11bde05977b3631167028862bE2a173976CA11"


def test_aggregate3_reorders_calls_and_returns_results_in_order() -> None:
    """Calls go out as (target, allowFailure, callData); results come back as (success, bytes)."""
    ledger_api = MagicMock()
    ledger_api.api.to_checksum_address.side_effect = lambda address: address.upper()
    instance = MagicMock()
    instance.functions.aggregate3.return_value.call.return_value = [
        (True, b"\x01"),
        (False, b""),
    ]

    with patch.object(Multicall3Contract, "get_instance", return_value=instance):
        result = Multicall3Contract.aggregate3(
            ledger_api, MULTICALL3, [("0xa", b"\xaa", True), ("0xb", b"\xbb", False)]
        )

    instance.functions.aggregate3.assert_called_once_with(
        [("0XA", True, b"\xaa"), ("0XB", False, b"\xbb")]
    )
    assert result == {"results": [(True, b"\x01"), (False, b"")]}

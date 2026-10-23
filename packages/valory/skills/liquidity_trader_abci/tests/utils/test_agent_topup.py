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

"""Tests for utils/agent_topup.py."""

# pylint: skip-file

from typing import Any, Dict, Optional

import pytest

from packages.valory.skills.funds_manager.models import FundRequirements
from packages.valory.skills.liquidity_trader_abci.utils.agent_topup import (
    TOPUP_GAS_FLOOR,
    executed_withdrawal_count,
    native_requirement,
    paid_call_chain,
    signer_pays_topup_gas,
    withdrawal_agent_need,
    withdrawal_gas_deficit,
    x402_agent_need_extra,
)

AGENT = "0xagent"
SAFE = "0xsafe"
NATIVE = "0x0000000000000000000000000000000000000000"
USDC = "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85"


def _funds(
    balance: Optional[int] = 100, deficit: Optional[int] = 300
) -> FundRequirements:
    """Status with an Agent Signer native entry and a non-native token."""
    token: Dict[str, Any] = {
        "topup": 400,
        "threshold": 200,
        "is_native": True,
        "balance": balance,
        "deficit": deficit,
    }
    return FundRequirements.model_validate(
        {
            "optimism": {
                "accounts": {
                    AGENT: {
                        "tokens": {
                            USDC: {"topup": 1, "threshold": 1, "is_native": False},
                            NATIVE: token,
                        }
                    },
                    SAFE: {
                        "tokens": {
                            USDC: {"topup": 1, "threshold": 1, "is_native": False}
                        }
                    },
                }
            }
        }
    )


class TestSignerPaysTopupGas:
    """Tests for signer_pays_topup_gas."""

    @pytest.mark.parametrize(
        "balance, pays",
        [
            (TOPUP_GAS_FLOOR, True),
            (TOPUP_GAS_FLOOR - 1, False),
            (None, None),
        ],
    )
    def test_pays_from_the_floor(
        self, balance: Optional[int], pays: Optional[bool]
    ) -> None:
        """The Agent Signer pays once it holds the floor."""
        assert signer_pays_topup_gas(_funds(balance=balance), "optimism", AGENT) is pays

    def test_the_gas_of_one_tx_leaves_no_margin(self) -> None:
        """A signer holding about one execTransaction of gas does not pay."""
        assert signer_pays_topup_gas(_funds(balance=10**12), "optimism", AGENT) is False

    def test_no_native_entry_is_unknown(self) -> None:
        """An Agent Signer without a native entry is not known to pay or not."""
        assert signer_pays_topup_gas(_funds(), "base", AGENT) is None


class TestPaidCallChain:
    """Tests for paid_call_chain."""

    def test_facilitator_pays_on_the_mech_chain(self) -> None:
        """The facilitator pre-deposit sits on the mech chain."""
        assert paid_call_chain(True, "Base", ["optimism"]) == "base"

    def test_direct_payment_uses_the_trading_chain(self) -> None:
        """Off the facilitator the EOA pays on the first trading chain."""
        assert paid_call_chain(False, "base", ["optimism", "mode"]) == "optimism"


class TestNativeRequirement:
    """Tests for native_requirement."""

    def test_returns_the_native_token_and_its_address(self) -> None:
        """The native entry is found among the account's tokens."""
        native = native_requirement(_funds(), "optimism", AGENT)
        assert native is not None
        address, token = native
        assert address == NATIVE
        assert token.balance == 100

    @pytest.mark.parametrize(
        "chain, account", [("base", AGENT), ("optimism", "0xother"), ("optimism", SAFE)]
    )
    def test_none_without_a_native_entry(self, chain: str, account: str) -> None:
        """An unknown chain, unknown account or no native token gives None."""
        assert native_requirement(_funds(), chain, account) is None


class TestX402AgentNeedExtra:
    """Tests for x402_agent_need_extra."""

    def test_adds_what_the_swap_needs_beyond_the_deficit(self) -> None:
        """The need becomes x402 - balance when that is above the deficit."""
        extra = x402_agent_need_extra(_funds(), "optimism", AGENT, 1000)
        assert extra == {"optimism": 600}

    def test_nothing_when_the_deficit_already_covers_it(self) -> None:
        """The need stays the deficit when x402 - balance is not above it."""
        assert x402_agent_need_extra(_funds(), "optimism", AGENT, 400) == {}

    def test_nothing_without_an_x402_need(self) -> None:
        """No stored x402 need adds nothing."""
        assert x402_agent_need_extra(_funds(), "optimism", AGENT, 0) == {}

    def test_nothing_on_a_chain_without_requirements(self) -> None:
        """A paid-call chain outside fund_requirements adds nothing."""
        assert x402_agent_need_extra(_funds(), "base", AGENT, 1000) == {}

    @pytest.mark.parametrize("balance, deficit", [(None, None), (100, None)])
    def test_nothing_when_the_balance_was_not_read(
        self, balance: Optional[int], deficit: Optional[int]
    ) -> None:
        """An unread balance adds nothing."""
        funds = _funds(balance=balance, deficit=deficit)
        assert x402_agent_need_extra(funds, "optimism", AGENT, 1000) == {}


class TestWithdrawalGasDeficit:
    """Tests for withdrawal_gas_deficit."""

    def test_gas_for_each_action_plus_buffer_minus_balance(self) -> None:
        """Two actions need 2.4e12 wei."""
        assert withdrawal_gas_deficit(2, 400_000_000_000) == 2_000_000_000_000

    @pytest.mark.parametrize("balance", [2_400_000_000_000, 10**18])
    def test_zero_when_the_balance_covers_it(self, balance: int) -> None:
        """No shortfall once the balance covers the gas."""
        assert withdrawal_gas_deficit(2, balance) == 0

    def test_zero_without_actions(self) -> None:
        """No action left needs no gas."""
        assert withdrawal_gas_deficit(0, 0) == 0


class TestExecutedWithdrawalCount:
    """Tests for executed_withdrawal_count."""

    @pytest.mark.parametrize(
        "withdrawal_data, count",
        [
            ({"withdrawal_transaction_hashes": '["0x1", "0x2"]'}, 2),
            ({"withdrawal_transaction_hashes": None}, 0),
            ({}, 0),
            (None, None),
            ({"withdrawal_transaction_hashes": "x"}, None),
            ({"withdrawal_transaction_hashes": "1"}, None),
        ],
    )
    def test_counts_or_none(
        self, withdrawal_data: Optional[Dict[str, Any]], count: Optional[int]
    ) -> None:
        """Nothing stored counts 0; a failed read or an unparsable value is None."""
        assert executed_withdrawal_count(withdrawal_data) == count


class TestWithdrawalAgentNeed:
    """Tests for withdrawal_agent_need."""

    def test_gas_for_the_actions_left(self) -> None:
        """The need is the gas for the actions not executed yet, less the balance."""
        need = withdrawal_agent_need(_funds(), "optimism", AGENT, {}, 3, 1)
        assert need == 2_400_000_000_000 - 100

    def test_nothing_left_needs_nothing(self) -> None:
        """Once every action executed nothing is needed."""
        assert withdrawal_agent_need(_funds(), "optimism", AGENT, {}, 2, 3) == 0

    def test_unprepared_actions_are_unknown(self) -> None:
        """Before the actions are prepared the need is unknown."""
        assert withdrawal_agent_need(_funds(), "optimism", AGENT, {}, 0, 0) is None

    def test_uncounted_executed_actions_are_unknown(self) -> None:
        """With a deficit, an unknown executed count leaves the need unknown."""
        assert withdrawal_agent_need(_funds(), "optimism", AGENT, {}, 1, None) is None

    @pytest.mark.parametrize(
        "funds, chain",
        [
            (_funds(deficit=0), "optimism"),
            (_funds(balance=None, deficit=None), "optimism"),
            (_funds(), "base"),
        ],
    )
    def test_no_deficit_needs_nothing(self, funds: Any, chain: str) -> None:
        """No deficit, no read balance or no native entry: nothing is needed."""
        assert withdrawal_agent_need(funds, chain, AGENT, {}, 1, None) == 0

    def test_the_x402_extra_counts_as_a_deficit(self) -> None:
        """An x402 extra alone opens the withdrawal gas."""
        need = withdrawal_agent_need(
            _funds(deficit=0), "optimism", AGENT, {"optimism": 1}, 1, 0
        )
        assert need == 1_200_000_000_000 - 100

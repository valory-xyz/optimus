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

"""The optimus inputs to the funds_manager Agent Signer top-up rule.

``AgentTopUpRound`` and the ``/funds-status`` handler both apply the rule, and
must derive its inputs the same way or the handler reports a shortfall the
round would never move.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from packages.valory.skills.funds_manager.models import (
    FundRequirements,
    TokenRequirement,
)

ESTIMATED_GAS_PER_TX = 1000000000000  # 0.000001 ETH in wei
# Ten times one Safe ``execTransaction`` on Optimism or Base, so a fee spike
# does not leave the top-up stuck in settlement.
TOPUP_GAS_FLOOR = 10000000000000
WITHDRAWAL_GAS_BUFFER = 1.2


def paid_call_chain(
    use_mech_facilitator: bool, mech_chain: str, target_investment_chains: List[str]
) -> str:
    """Return the chain the agent pays for API calls on.

    :param use_mech_facilitator: whether paid calls go through the facilitator.
    :param mech_chain: the chain the facilitator pre-deposit is held on.
    :param target_investment_chains: the chains the agent trades on.
    :return: the chain name.
    """
    if use_mech_facilitator:
        return mech_chain.lower()
    return target_investment_chains[0]


def native_requirement(
    funds: FundRequirements, chain: str, account: str
) -> Optional[Tuple[str, TokenRequirement]]:
    """Return the native token address and requirement of an account on a chain.

    :param funds: the status returned by ``get_funds_status()``.
    :param chain: the chain name.
    :param account: the account address.
    :return: ``(token address, requirement)``, or ``None`` if there is none.
    """
    chain_requirements = funds.root.get(chain)
    if chain_requirements is None:
        return None
    account_requirements = chain_requirements.accounts.get(account)
    if account_requirements is None:
        return None
    return next(
        (
            (address, token)
            for address, token in account_requirements.tokens.items()
            if token.is_native
        ),
        None,
    )


def x402_agent_need_extra(
    funds: FundRequirements, chain: str, agent_address: str, x402_eth_deficit: int
) -> Dict[str, int]:
    """Return the wei the x402 ETH need adds to the Agent Signer deficit.

    ``x402_eth_deficit`` is a level the agent EOA must hold, not a shortfall.

    :param funds: the status returned by ``get_funds_status()``.
    :param chain: the chain the paid calls are funded on.
    :param agent_address: the agent EOA.
    :param x402_eth_deficit: the ETH the paid-call swaps need, in wei.
    :return: ``{chain: extra}``, or ``{}`` when it adds nothing.
    """
    native = native_requirement(funds, chain, agent_address)
    if native is None:
        return {}
    _, token = native
    if token.balance is None or token.deficit is None:
        return {}
    extra = x402_eth_deficit - token.balance - token.deficit
    return {chain: extra} if extra > 0 else {}


def signer_pays_topup_gas(
    funds: FundRequirements, chain: str, agent_address: str
) -> Optional[bool]:
    """Return whether the Agent Signer holds the gas for one top-up ``execTransaction``.

    :param funds: the status returned by ``get_funds_status()``.
    :param chain: the chain name.
    :param agent_address: the agent EOA.
    :return: whether the Agent Signer native balance reaches the floor, or
        ``None`` when that balance is unknown.
    """
    native = native_requirement(funds, chain, agent_address)
    if native is None or native[1].balance is None:
        return None
    return native[1].balance >= TOPUP_GAS_FLOOR


def withdrawal_gas_deficit(num_actions: int, balance: int) -> int:
    """Return the wei the agent EOA lacks to pay for the withdrawal actions left.

    :param num_actions: the withdrawal actions still to execute.
    :param balance: the agent EOA native balance, in wei.
    :return: the shortfall in wei, ``0`` when the balance covers it.
    """
    total_gas_needed = int(ESTIMATED_GAS_PER_TX * num_actions * WITHDRAWAL_GAS_BUFFER)
    return max(0, total_gas_needed - balance)


def executed_withdrawal_count(
    withdrawal_data: Optional[Dict[str, Any]],
) -> Optional[int]:
    """Return how many withdrawal transactions have executed.

    :param withdrawal_data: the KV store read of ``withdrawal_transaction_hashes``,
        or ``None`` when the read failed.
    :return: the count, or ``None`` when it cannot be known.
    """
    if withdrawal_data is None:
        return None
    try:
        return len(
            json.loads(withdrawal_data.get("withdrawal_transaction_hashes") or "[]")
        )
    except (TypeError, ValueError):
        return None


def withdrawal_agent_need(  # pylint: disable=too-many-arguments
    funds: FundRequirements,
    chain: str,
    agent_address: str,
    agent_need_extra: Dict[str, int],
    num_actions: int,
    executed_count: Optional[int],
) -> Optional[int]:
    """Return the gas the Agent Signer lacks for the withdrawal actions left.

    :param funds: the status returned by ``get_funds_status()``.
    :param chain: the chain the withdrawal runs on.
    :param agent_address: the agent EOA.
    :param agent_need_extra: the x402 extra added to the Agent Signer need.
    :param num_actions: the withdrawal actions prepared.
    :param executed_count: the withdrawal transactions executed, or ``None``.
    :return: the need in wei, or ``None`` when the actions are not prepared yet
        or the executed ones cannot be counted.
    """
    if num_actions == 0:
        return None
    native = native_requirement(funds, chain, agent_address)
    if native is None:
        return 0
    _, signer = native
    deficit = (signer.deficit or 0) + agent_need_extra.get(chain, 0)
    if signer.balance is None or deficit <= 0:
        return 0
    if executed_count is None:
        return None
    return withdrawal_gas_deficit(max(0, num_actions - executed_count), signer.balance)

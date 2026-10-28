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

"""Agent Signer top-up inputs shared by AgentTopUpRound and /funds-status."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from packages.valory.skills.funds_manager.models import (
    FundRequirements,
    TokenRequirement,
)

ESTIMATED_GAS_COST_PER_TX_WEI = 1000000000000  # 0.000001 ETH
# 10x ESTIMATED_GAS_COST_PER_TX_WEI: headroom for fee spikes during settlement.
TOPUP_GAS_FLOOR = 10000000000000


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


def executed_withdrawal_count(
    withdrawal_data: Optional[Dict[str, Any]], logger: logging.Logger
) -> Optional[int]:
    """Return how many withdrawal transactions have executed.

    :param withdrawal_data: the KV store read of ``withdrawal_transaction_hashes``,
        or ``None`` when the read failed.
    :param logger: the logger to report an unreadable count on.
    :return: the count, or ``None`` when it cannot be known.
    """
    if withdrawal_data is None:
        logger.warning("Executed withdrawal count unknown: the KV store read failed.")
        return None
    raw = withdrawal_data.get("withdrawal_transaction_hashes") or "[]"
    try:
        hashes = json.loads(raw)
    except (TypeError, ValueError) as exc:
        logger.warning(
            f"Executed withdrawal count unknown: withdrawal_transaction_hashes "
            f"is not JSON ({exc})."
        )
        return None
    if not isinstance(hashes, list):
        logger.warning(
            f"Executed withdrawal count unknown: withdrawal_transaction_hashes "
            f"is a {type(hashes).__name__}, not a list."
        )
        return None
    return len(hashes)


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
    :return: the need in wei; ``0`` when the Agent Signer has no native
        requirement, no deficit, or an unread balance; ``None`` when the actions
        are not prepared yet or the executed ones cannot be counted.
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
    actions_left = max(0, num_actions - executed_count)
    gas_needed = ESTIMATED_GAS_COST_PER_TX_WEI * actions_left * 12 // 10
    return max(0, gas_needed - signer.balance)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2026 Valory AG
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


"""This package contains LLM prompts for Optimus ABCI."""

import enum
import pickle  # nosec
import typing

from pydantic import BaseModel


class ProtocolName(enum.Enum):
    """Available protocol names."""

    BALANCER_POOL = "balancerPool"
    UNISWAP_V3 = "uniswapV3"
    VELODROME = "velodrome"
    STURDY = "sturdy"


class TradingType(enum.Enum):
    """Trading type."""

    RISKY = "risky"
    BALANCED = "balanced"


class Intent(enum.Enum):
    """What the user asked the chat for."""

    QUERY = "query"
    UPDATE = "update"


# The max loss percentage the LLM may return, inclusive bounds.
MAX_LOSS_PERCENTAGE_RANGE: typing.Tuple[float, float] = (1.0, 30.0)

# The max loss percentage in force before the user ever set one.
DEFAULT_MAX_LOSS_PERCENTAGE: typing.Dict[str, float] = {
    TradingType.BALANCED.value: 10.0,
    TradingType.RISKY.value: 20.0,
}


class StrategyConfig(BaseModel):
    """Strategy configuration response."""

    intent: Intent
    selected_protocols: typing.List[str]
    trading_type: TradingType
    max_loss_percentage: float
    activity_goal: typing.Optional[int] = None
    reasoning: str


def build_strategy_config_schema() -> dict:
    """Build a schema for the StrategyConfig."""
    return {"class": pickle.dumps(StrategyConfig).hex(), "is_list": False}


# Ultra-minimal prompt for maximum speed (keeping reasoning)
STRATEGY_PROMPT = """"{user_prompt}" Current: {previous_protocols},{previous_type},{previous_threshold}% Protocols: balancerPool,velodrome,sturdy Risk: 1-5% conservative,6-10% balanced,11-15% growth,16-30% aggressive Daily goal: {activity_goal} rounds, {activity_goal_progress} done this epoch. A round is one cycle of my main loop: I refresh my positions, look at the opportunities for your funds and decide whether to act; I may decide to hold. intent: update if the user asks to change something, else query (keep Current) activity_goal: the new number of rounds if the user asks to change the goal, else null; if they only talk about the goal keep Current protocols, type and risk JSON: intent, selected_protocols, trading_type, max_loss_percentage, activity_goal, reasoning"""

# The chat on the agent's main page: a change is only proposed, and the
# agent's recent activity lets it explain what it did.
CHAT_PROMPT = """You are the user's autonomous liquidity agent. You invest the user's funds in liquidity pools on {chain} and chat with the user on your main page.

Current configuration:
- Protocols: {previous_protocols} (available: balancerPool, velodrome, sturdy)
- Trading type: {previous_type}
- Max loss percentage: {previous_threshold}% (allowed 1-30: 1-5% conservative, 6-10% balanced, 11-15% growth, 16-30% aggressive)
- Daily activity target: {activity_goal} rounds, {activity_goal_progress} done this epoch. A round is one cycle of your main loop: you refresh your positions, look at the opportunities for the user's funds and decide whether to act; you may decide to hold.

Your recent activity and chat, oldest first:
{recent_activity}

Answer the user's message in "reasoning", in plain words. Use the recent activity to explain what you did and why, such as your current pool choice; do not invent activity that is not listed. The user can change the protocols, the trading type, the max loss percentage and the daily activity target.
intent: update if the user asks to change one of these settings, else query. For a query, return the current values unchanged. For an update, return the values the user asked for, and the max loss percentage exactly as asked even outside 1-30. A change is not applied yet: the user is asked to confirm it first, so describe it as a proposal, never as done, and do not ask for the confirmation yourself.
activity_goal: the new number of rounds if the user asks to change the daily activity target, else null.

User message: "{user_prompt}"
JSON: intent, selected_protocols, trading_type, max_loss_percentage, activity_goal, reasoning"""

CONFIRM_PROMPT = """The agent proposed this change to its configuration and asked the user to confirm it:
{proposal}

The user replied: "{user_reply}"

Decide whether the reply confirms this exact change. Set "confirmed" to true only for a clear yes to this change. A refusal, a question, a different request, a different value or anything unclear is false.
"""


class ConfirmationResponse(BaseModel):
    """Whether the user's reply confirms the proposed change."""

    confirmed: bool


def build_confirmation_schema() -> dict:
    """Build a schema for the ConfirmationResponse."""
    return {"class": pickle.dumps(ConfirmationResponse).hex(), "is_list": False}

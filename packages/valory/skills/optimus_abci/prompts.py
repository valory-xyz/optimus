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


class StrategyConfig(BaseModel):
    """Strategy configuration response."""

    selected_protocols: typing.List[str]
    trading_type: TradingType
    max_loss_percentage: float
    reasoning: str


def build_strategy_config_schema() -> dict:
    """Build a schema for the StrategyConfig."""
    return {"class": pickle.dumps(StrategyConfig).hex(), "is_list": False}


# Ultra-minimal prompt for maximum speed (keeping reasoning)
STRATEGY_PROMPT = """"{user_prompt}" Current: {previous_protocols},{previous_type},{previous_threshold}% Protocols: balancerPool,uniswapV3,velodrome,sturdy Risk: 1-5% conservative,6-10% balanced,11-15% growth,16-30% aggressive JSON: selected_protocols, trading_type, max_loss_percentage, reasoning"""

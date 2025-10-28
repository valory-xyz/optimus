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


# Optimized prompt - essential info only, structured for speed
STRATEGY_PROMPT = """Parse trading instruction for protocols and risk tolerance.

User: "{user_prompt}"
Current: protocols={previous_protocols}, type={previous_type}, threshold={previous_threshold}%

Valid protocols: balancerPool, uniswapV3, velodrome, sturdy

Protocol selection rules:
- "use X" → select only X
- "remove X" → remove X from current
- "add X" → add X to current
- explicit list → use exact list
- no mention → keep current

Risk tolerance (1-30%, default 10):
- safe/conservative: 1-5%
- moderate/balanced: 6-10%
- growth/higher returns: 11-15%
- aggressive/maximize: 16-25%
- very aggressive/maximum: 26-30%

REQUIRED OUTPUT: You must provide ALL four fields:
1. selected_protocols (array of protocol names)
2. trading_type (balanced or risky)
3. max_loss_percentage (number 1-30)
4. reasoning (explanation string)"""

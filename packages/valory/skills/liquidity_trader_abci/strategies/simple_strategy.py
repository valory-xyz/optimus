"""This package contains the implemenatation of the SimpleStrategyBehaviour class."""

from abc import ABC
from typing import Any, Dict, Generator, List, Optional, Tuple

from packages.valory.skills.liquidity_trader_abci.strategy_behaviour import (
    StrategyBehaviour,
)


class SimpleStrategyBehaviour(StrategyBehaviour, ABC):
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the simple strategy behaviour."""
        super().__init__(**kwargs)

    def get_decision(self, **kwargs: Any) -> bool:
        pool_apr = kwargs.get("pool_apr")
        """Get decision"""

        if not pool_apr:
            self.context.logger.error("Pool APR cannot be None")
            return False

        if not self._is_apr_threshold_exceeded(pool_apr):
            return False

        # TO-DO: Decide on the correct method/logic for maintaining the period number for the last transaction.
        # if not self._is_round_threshold_exceeded():  # noqa: E800
        #     self.context.logger.info("Round threshold not exceeded")  # noqa: E800
        #     return False  # noqa: E800

        return True

    def _is_apr_threshold_exceeded(self, pool_apr) -> bool:
        """Check if the highest APR exceeds the threshold"""
        if pool_apr > self.params.apr_threshold:
            return True
        else:
            self.context.logger.info(
                f"APR of selected pool that is {pool_apr} does not exceed APR threshold {self.params.apr_threshold}."
            )
            return False

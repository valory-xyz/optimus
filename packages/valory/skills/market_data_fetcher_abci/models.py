# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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

"""This module contains the shared state for the abci skill of MarketDataFetcherAbciApp."""

from datetime import datetime
from time import time
from typing import Any, Dict, List, Optional

from aea.skills.base import Model

from packages.valory.skills.abstract_round_abci.models import BaseParams
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.abstract_round_abci.models import TypeCheckMixin
from packages.valory.skills.market_data_fetcher_abci.rounds import (
    MarketDataFetcherAbciApp,
)


MINUTE_UNIX = 60


def format_whitelist(token_whitelist: List) -> List:
    """Load the token whitelist into its proper format"""
    fixed_whitelist = []
    for element in token_whitelist:
        token_config = {}
        for i in element.split("&"):
            key, value = i.split("=")
            token_config[key] = value
        fixed_whitelist.append(token_config)
    return fixed_whitelist


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = MarketDataFetcherAbciApp


class Params(BaseParams):
    """Parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""
        self.token_symbol_whitelist: List[Dict] = format_whitelist(
            self._ensure("token_symbol_whitelist", kwargs, List[str])
        )
        self.ledger_ids = self._ensure("ledger_ids", kwargs, List[str])
        self.exchange_ids = self._ensure("exchange_ids", kwargs, Dict[str, List[str]])
        super().__init__(*args, **kwargs)


class CoingeckoRateLimiter:
    """Keeps track of the rate limiting for Coingecko."""

    def __init__(self, limit: int, credits_: int) -> None:
        """Initialize the Coingecko rate limiter."""
        self._limit = self._remaining_limit = limit
        self._credits = self._remaining_credits = credits_
        self._last_request_time = time()

    @property
    def limit(self) -> int:
        """Get the limit per minute."""
        return self._limit

    @property
    def credits(self) -> int:
        """Get the requests' cap per month."""
        return self._credits

    @property
    def remaining_limit(self) -> int:
        """Get the remaining limit per minute."""
        return self._remaining_limit

    @property
    def remaining_credits(self) -> int:
        """Get the remaining requests' cap per month."""
        return self._remaining_credits

    @property
    def last_request_time(self) -> float:
        """Get the timestamp of the last request."""
        return self._last_request_time

    @property
    def rate_limited(self) -> bool:
        """Check whether we are rate limited."""
        return self.remaining_limit == 0

    @property
    def no_credits(self) -> bool:
        """Check whether all the credits have been spent."""
        return self.remaining_credits == 0

    @property
    def cannot_request(self) -> bool:
        """Check whether we cannot perform a request."""
        return self.rate_limited or self.no_credits

    @property
    def credits_reset_timestamp(self) -> int:
        """Get the UNIX timestamp in which the Coingecko credits reset."""
        current_date = datetime.now()
        first_day_of_next_month = datetime(current_date.year, current_date.month + 1, 1)
        return int(first_day_of_next_month.timestamp())

    @property
    def can_reset_credits(self) -> bool:
        """Check whether the Coingecko credits can be reset."""
        return self.last_request_time >= self.credits_reset_timestamp

    def _update_limits(self) -> None:
        """Update the remaining limits and the credits if necessary."""
        time_passed = time() - self.last_request_time
        limit_increase = int(time_passed / MINUTE_UNIX) * self.limit
        self._remaining_limit = min(self.limit, self.remaining_limit + limit_increase)
        if self.can_reset_credits:
            self._remaining_credits = self.credits

    def _burn_credit(self) -> None:
        """Use one credit."""
        self._remaining_limit -= 1
        self._remaining_credits -= 1
        self._last_request_time = time()

    def check_and_burn(self) -> bool:
        """Check whether we can perform a new request, and if yes, update the remaining limit and credits."""
        self._update_limits()
        if self.cannot_request:
            return False
        self._burn_credit()
        return True


class Coingecko(Model, TypeCheckMixin):
    """Coingecko configuration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Coingecko object."""
        self.endpoint: str = self._ensure("endpoint", kwargs, str)
        self.api_key: Optional[str] = self._ensure("api_key", kwargs, Optional[str])
        self.prices_field: str = self._ensure("prices_field", kwargs, str)
        self.volumes_field: str = self._ensure("volumes_field", kwargs, str)
        self.rate_limited_code: int = self._ensure("rate_limited_code", kwargs, int)
        limit: int = self._ensure("requests_per_minute", kwargs, int)
        credits_: int = self._ensure("credits", kwargs, int)
        self.rate_limiter = CoingeckoRateLimiter(limit, credits_)
        super().__init__(*args, **kwargs)

    def rate_limited_status_callback(self) -> None:
        """Callback when a rate-limited status is returned from the API."""
        self.context.logger.error(
            "Unexpected rate-limited status code was received from the Coingecko API! "
            "Setting the limit to 0 on the local rate limiter to partially address the issue. "
            "Please check whether the `Coingecko` overrides are set corresponding to the API's rules."
        )
        self.rate_limiter._remaining_limit = 0
        self.rate_limiter._last_request_time = time()


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool

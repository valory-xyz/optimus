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

"""Shared fixtures for customs tests."""

import pytest


@pytest.fixture(autouse=True)
def _clear_coingecko_price_caches():
    """Clear CoinGecko price caches before each test to prevent cross-test pollution."""
    from packages.valory.customs.balancer_pools_search.balancer_pools_search import (
        COINGECKO_PRICE_CACHE as bal_cache,
    )
    from packages.valory.customs.uniswap_pools_search.uniswap_pools_search import (
        COINGECKO_PRICE_CACHE as uni_cache,
    )
    from packages.valory.customs.velodrome_pools_search.velodrome_pools_search import (
        COINGECKO_PRICE_CACHE as velo_cache,
    )

    bal_cache.clear()
    uni_cache.clear()
    velo_cache.clear()
    yield
    bal_cache.clear()
    uni_cache.clear()
    velo_cache.clear()

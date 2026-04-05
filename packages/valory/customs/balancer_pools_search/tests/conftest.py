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
    """Reset CoinGecko price caches before each test.

    Strategies use module-level COINGECKO_PRICE_CACHE dicts that can be
    reassigned by run(price_cache=shared_dict).  Importing the name only
    captures the reference at import time, so we reset the attribute on
    the module directly to guarantee isolation regardless of reassignment.
    """
    import packages.valory.customs.balancer_pools_search.balancer_pools_search as bal
    import packages.valory.customs.uniswap_pools_search.uniswap_pools_search as uni
    import packages.valory.customs.velodrome_pools_search.velodrome_pools_search as velo

    bal.COINGECKO_PRICE_CACHE = {}
    uni.COINGECKO_PRICE_CACHE = {}
    velo.COINGECKO_PRICE_CACHE = {}
    yield
    bal.COINGECKO_PRICE_CACHE = {}
    uni.COINGECKO_PRICE_CACHE = {}
    velo.COINGECKO_PRICE_CACHE = {}

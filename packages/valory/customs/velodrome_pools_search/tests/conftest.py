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

"""Fixtures for velodrome_pools_search tests."""

import pytest

import packages.valory.customs.velodrome_pools_search.velodrome_pools_search as velo


@pytest.fixture(autouse=True)
def _clear_coingecko_price_cache():
    """Reset the module-level COINGECKO_PRICE_CACHE between tests.

    `run(price_cache=shared_dict)` can rebind the module attribute, so
    reset via setattr to keep isolation regardless of reassignment.
    """
    velo.COINGECKO_PRICE_CACHE = {}
    yield
    velo.COINGECKO_PRICE_CACHE = {}

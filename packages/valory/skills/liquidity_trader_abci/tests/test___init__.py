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

"""Test the __init__.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from aea.configurations.base import PublicId

from packages.valory.skills.liquidity_trader_abci import PUBLIC_ID


def test_import() -> None:
    """Test that the liquidity_trader_abci package can be imported."""
    import packages.valory.skills.liquidity_trader_abci  # noqa


def test_public_id() -> None:
    """Test PUBLIC_ID is correctly defined."""
    assert isinstance(PUBLIC_ID, PublicId)
    assert PUBLIC_ID.author == "valory"
    assert PUBLIC_ID.name == "liquidity_trader_abci"
    assert PUBLIC_ID.version == "0.1.0"

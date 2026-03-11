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

"""Test the utils/protocol_validation.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from packages.valory.skills.liquidity_trader_abci.utils.protocol_validation import (
    validate_and_fix_protocols,
)


def test_import() -> None:
    """Test that the protocol_validation module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.utils.protocol_validation  # noqa


class TestValidateAndFixProtocols:
    """Test validate_and_fix_protocols function."""

    def test_all_valid_protocols(self) -> None:
        """Test with all valid protocols."""
        result = validate_and_fix_protocols(
            selected_protocols=["balancerPool", "velodrome"],
            target_investment_chains=["ethereum", "optimism"],
            available_strategies={
                "ethereum": ["balancer_pools_search"],
                "optimism": ["velodrome_pools_search"],
            },
        )
        assert result == ["balancerPool", "velodrome"]

    def test_invalid_protocol_with_valid_remaining(self) -> None:
        """Test with one invalid protocol but valid ones remaining."""
        result = validate_and_fix_protocols(
            selected_protocols=["balancerPool", "invalid_protocol"],
            target_investment_chains=["ethereum"],
            available_strategies={"ethereum": ["balancer_pools_search"]},
        )
        assert result == ["balancerPool"]

    def test_all_invalid_protocols_returns_defaults(self) -> None:
        """Test with all invalid protocols returns defaults."""
        result = validate_and_fix_protocols(
            selected_protocols=["invalid1", "invalid2"],
            target_investment_chains=["ethereum"],
            available_strategies={"ethereum": ["balancer_pools_search"]},
        )
        assert result == ["balancerPool"]

    def test_chain_incompatible_protocol_removed(self) -> None:
        """Test protocol not available on target chains is removed."""
        result = validate_and_fix_protocols(
            selected_protocols=["balancerPool", "velodrome"],
            target_investment_chains=["ethereum"],
            available_strategies={"ethereum": ["balancer_pools_search"]},
        )
        assert result == ["balancerPool"]

    def test_chain_incompatible_preserved_when_previously_selected(self) -> None:
        """Test grandfathering: protocol preserved if previously selected."""
        result = validate_and_fix_protocols(
            selected_protocols=["balancerPool", "velodrome"],
            target_investment_chains=["ethereum"],
            available_strategies={"ethereum": ["balancer_pools_search"]},
            previous_protocols=["velodrome"],
        )
        assert "velodrome" in result
        assert "balancerPool" in result

    def test_empty_selected_protocols(self) -> None:
        """Test with empty selected protocols."""
        result = validate_and_fix_protocols(
            selected_protocols=[],
            target_investment_chains=["ethereum"],
            available_strategies={"ethereum": ["balancer_pools_search"]},
        )
        assert result == []

    def test_no_previous_protocols(self) -> None:
        """Test with None previous_protocols."""
        result = validate_and_fix_protocols(
            selected_protocols=["balancerPool"],
            target_investment_chains=["ethereum"],
            available_strategies={"ethereum": ["balancer_pools_search"]},
            previous_protocols=None,
        )
        assert result == ["balancerPool"]

    def test_all_chain_incompatible_no_previous(self) -> None:
        """Test all protocols are chain-incompatible with no previous."""
        result = validate_and_fix_protocols(
            selected_protocols=["velodrome"],
            target_investment_chains=["ethereum"],
            available_strategies={"ethereum": ["balancer_pools_search"]},
        )
        assert result == ["balancerPool"]

    def test_uniswapV3_grandfathering(self) -> None:
        """Test uniswapV3 is kept for grandfathering."""
        result = validate_and_fix_protocols(
            selected_protocols=["uniswapV3"],
            target_investment_chains=["ethereum"],
            available_strategies={"ethereum": ["uniswap_pools_search"]},
        )
        assert result == ["uniswapV3"]

    def test_sturdy_protocol(self) -> None:
        """Test sturdy protocol validation."""
        result = validate_and_fix_protocols(
            selected_protocols=["sturdy"],
            target_investment_chains=["ethereum"],
            available_strategies={"ethereum": ["asset_lending"]},
        )
        assert result == ["sturdy"]

    def test_chain_not_in_available_strategies(self) -> None:
        """Test when target chain is not in available_strategies."""
        result = validate_and_fix_protocols(
            selected_protocols=["balancerPool"],
            target_investment_chains=["unknown_chain"],
            available_strategies={"ethereum": ["balancer_pools_search"]},
        )
        # balancerPool is incompatible (not on unknown_chain), returns defaults from available chains
        # But unknown_chain has no strategies, so defaults come from ethereum
        assert result == []

    def test_multiple_defaults_from_multiple_chains(self) -> None:
        """Test defaults from multiple chains when all selected are invalid."""
        result = validate_and_fix_protocols(
            selected_protocols=["invalid"],
            target_investment_chains=["ethereum", "optimism"],
            available_strategies={
                "ethereum": ["balancer_pools_search"],
                "optimism": ["velodrome_pools_search"],
            },
        )
        assert "balancerPool" in result
        assert "velodrome" in result

    def test_duplicate_default_protocol_across_chains(self) -> None:
        """Test that a protocol available on multiple chains is not duplicated in defaults."""
        result = validate_and_fix_protocols(
            selected_protocols=["invalid"],
            target_investment_chains=["ethereum", "optimism"],
            available_strategies={
                "ethereum": ["balancer_pools_search"],
                "optimism": ["balancer_pools_search", "velodrome_pools_search"],
            },
        )
        # balancerPool should appear only once even though it's on both chains
        assert result.count("balancerPool") == 1
        assert "velodrome" in result

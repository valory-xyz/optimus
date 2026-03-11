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

"""Test the prompts.py module of the optimus_abci skill."""

# pylint: skip-file

import pickle  # nosec

from packages.valory.skills.optimus_abci.prompts import (
    STRATEGY_PROMPT,
    ProtocolName,
    StrategyConfig,
    TradingType,
    build_strategy_config_schema,
)


def test_import() -> None:
    """Test that the prompts module can be imported."""
    import packages.valory.skills.optimus_abci.prompts  # noqa


class TestProtocolName:
    """Test ProtocolName enum."""

    def test_balancer_pool(self) -> None:
        """Test BALANCER_POOL value."""
        assert ProtocolName.BALANCER_POOL.value == "balancerPool"

    def test_uniswap_v3(self) -> None:
        """Test UNISWAP_V3 value."""
        assert ProtocolName.UNISWAP_V3.value == "uniswapV3"

    def test_velodrome(self) -> None:
        """Test VELODROME value."""
        assert ProtocolName.VELODROME.value == "velodrome"

    def test_sturdy(self) -> None:
        """Test STURDY value."""
        assert ProtocolName.STURDY.value == "sturdy"

    def test_all_members(self) -> None:
        """Test all enum members exist."""
        members = [e.value for e in ProtocolName]
        assert len(members) == 4
        assert "balancerPool" in members
        assert "uniswapV3" in members
        assert "velodrome" in members
        assert "sturdy" in members


class TestTradingType:
    """Test TradingType enum."""

    def test_risky(self) -> None:
        """Test RISKY value."""
        assert TradingType.RISKY.value == "risky"

    def test_balanced(self) -> None:
        """Test BALANCED value."""
        assert TradingType.BALANCED.value == "balanced"

    def test_all_members(self) -> None:
        """Test all enum members exist."""
        members = [e.value for e in TradingType]
        assert len(members) == 2
        assert "risky" in members
        assert "balanced" in members


class TestStrategyConfig:
    """Test StrategyConfig pydantic model."""

    def test_creation(self) -> None:
        """Test creating a StrategyConfig instance."""
        config = StrategyConfig(
            selected_protocols=["balancerPool", "velodrome"],
            trading_type=TradingType.BALANCED,
            max_loss_percentage=5.0,
            reasoning="test reasoning",
        )
        assert config.selected_protocols == ["balancerPool", "velodrome"]
        assert config.trading_type == TradingType.BALANCED
        assert config.max_loss_percentage == 5.0
        assert config.reasoning == "test reasoning"

    def test_risky_trading_type(self) -> None:
        """Test creating a StrategyConfig with risky trading type."""
        config = StrategyConfig(
            selected_protocols=["sturdy"],
            trading_type=TradingType.RISKY,
            max_loss_percentage=15.0,
            reasoning="aggressive strategy",
        )
        assert config.trading_type == TradingType.RISKY
        assert config.max_loss_percentage == 15.0

    def test_empty_protocols(self) -> None:
        """Test creating a StrategyConfig with empty protocols list."""
        config = StrategyConfig(
            selected_protocols=[],
            trading_type=TradingType.BALANCED,
            max_loss_percentage=5.0,
            reasoning="no protocols selected",
        )
        assert config.selected_protocols == []


class TestBuildStrategyConfigSchema:
    """Test build_strategy_config_schema function."""

    def test_returns_dict(self) -> None:
        """Test that build_strategy_config_schema returns a dict."""
        schema = build_strategy_config_schema()
        assert isinstance(schema, dict)

    def test_has_class_key(self) -> None:
        """Test that the schema has a 'class' key."""
        schema = build_strategy_config_schema()
        assert "class" in schema

    def test_has_is_list_key(self) -> None:
        """Test that the schema has an 'is_list' key."""
        schema = build_strategy_config_schema()
        assert "is_list" in schema

    def test_is_list_is_false(self) -> None:
        """Test that is_list is False."""
        schema = build_strategy_config_schema()
        assert schema["is_list"] is False

    def test_class_is_hex_encoded_pickle(self) -> None:
        """Test that the class value is a hex-encoded pickle of StrategyConfig."""
        schema = build_strategy_config_schema()
        class_value = schema["class"]
        # Verify it's a valid hex string
        assert isinstance(class_value, str)
        # Verify we can decode it back
        decoded = pickle.loads(bytes.fromhex(class_value))  # nosec
        assert decoded is StrategyConfig


class TestStrategyPrompt:
    """Test the STRATEGY_PROMPT constant."""

    def test_prompt_is_string(self) -> None:
        """Test that STRATEGY_PROMPT is a string."""
        assert isinstance(STRATEGY_PROMPT, str)

    def test_prompt_has_placeholders(self) -> None:
        """Test that STRATEGY_PROMPT has the expected placeholders."""
        assert "{user_prompt}" in STRATEGY_PROMPT
        assert "{previous_protocols}" in STRATEGY_PROMPT
        assert "{previous_type}" in STRATEGY_PROMPT
        assert "{previous_threshold}" in STRATEGY_PROMPT

    def test_prompt_formatting(self) -> None:
        """Test that STRATEGY_PROMPT can be formatted."""
        formatted = STRATEGY_PROMPT.format(
            user_prompt="invest conservatively",
            previous_protocols=["balancerPool"],
            previous_type="balanced",
            previous_threshold=5,
        )
        assert "invest conservatively" in formatted
        assert "balanced" in formatted

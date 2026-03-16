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

"""Test the models.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import json
import tempfile
from time import time
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.skills.liquidity_trader_abci.models import (
    Coingecko,
    CoingeckoRateLimiter,
    Params,
    SharedState,
)


def test_import() -> None:
    """Test that the models module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.models  # noqa


class TestSharedState:
    """Test SharedState class."""

    def test_initialization(self) -> None:
        """Test SharedState initialization."""
        mock_context = MagicMock()
        state = SharedState(name="state", skill_context=mock_context)
        assert state.in_flight_req is False
        assert state.strategy_to_filehash == {}
        assert state.strategies_executables == {}
        assert state.trading_type == ""
        assert state.selected_protocols == []
        assert state.request_count == 0
        assert state.request_queue == []
        assert state.req_to_callback == {}
        assert state.agent_reasoning == ""

    def test_setup_success(self) -> None:
        """Test setup with valid strategies."""
        mock_context = MagicMock()
        state = SharedState(name="state", skill_context=mock_context)

        mock_context.params.file_hash_to_strategies = {
            "hash1": "strategy_a",
            "hash2": "strategy_b",
        }
        mock_context.params.available_strategies = {
            "chain1": ["strategy_a", "strategy_b"]
        }

        with patch.object(type(state).__bases__[0], "setup"):
            state.setup()

        assert state.strategy_to_filehash == {
            "strategy_a": "hash1",
            "strategy_b": "hash2",
        }

    def test_setup_missing_strategy_raises(self) -> None:
        """Test setup raises ValueError when strategy is missing from executables."""
        mock_context = MagicMock()
        state = SharedState(name="state", skill_context=mock_context)

        mock_context.params.file_hash_to_strategies = {"hash1": "strategy_a"}
        mock_context.params.available_strategies = {
            "chain1": ["strategy_a", "strategy_missing"]
        }

        with patch.object(type(state).__bases__[0], "setup"):
            with pytest.raises(ValueError, match="is not in the strategies"):
                state.setup()


class TestCoingeckoRateLimiter:
    """Test CoingeckoRateLimiter class."""

    def test_initialization(self) -> None:
        """Test initialization."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        assert limiter.limit == 30
        assert limiter.credits == 10000
        assert limiter.remaining_limit == 30
        assert limiter.remaining_credits == 10000

    def test_rate_limited_false(self) -> None:
        """Test rate_limited is False when remaining limit > 0."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        assert limiter.rate_limited is False

    def test_rate_limited_true(self) -> None:
        """Test rate_limited is True when remaining limit == 0."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        limiter._remaining_limit = 0
        assert limiter.rate_limited is True

    def test_no_credits_false(self) -> None:
        """Test no_credits is False when remaining credits > 0."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        assert limiter.no_credits is False

    def test_no_credits_true(self) -> None:
        """Test no_credits is True when remaining credits == 0."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        limiter._remaining_credits = 0
        assert limiter.no_credits is True

    def test_cannot_request_rate_limited(self) -> None:
        """Test cannot_request when rate limited."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        limiter._remaining_limit = 0
        assert limiter.cannot_request is True

    def test_cannot_request_no_credits(self) -> None:
        """Test cannot_request when no credits."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        limiter._remaining_credits = 0
        assert limiter.cannot_request is True

    def test_cannot_request_false(self) -> None:
        """Test cannot_request is False when both available."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        assert limiter.cannot_request is False

    def test_credits_reset_timestamp(self) -> None:
        """Test credits_reset_timestamp returns a valid UNIX timestamp."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        ts = limiter.credits_reset_timestamp
        assert isinstance(ts, int)
        assert ts > 0

    def test_can_reset_credits(self) -> None:
        """Test can_reset_credits when last request time is past reset."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        # Set last_request_time far in the future to pass credits_reset_timestamp
        limiter._last_request_time = limiter.credits_reset_timestamp + 1000
        assert limiter.can_reset_credits is True

    def test_can_reset_credits_false(self) -> None:
        """Test can_reset_credits is False normally."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        assert limiter.can_reset_credits is False

    def test_last_request_time(self) -> None:
        """Test last_request_time property."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        assert isinstance(limiter.last_request_time, float)

    def test_check_and_burn_success(self) -> None:
        """Test check_and_burn returns True and decreases limits."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        result = limiter.check_and_burn()
        assert result is True
        assert limiter.remaining_limit == 29
        assert limiter.remaining_credits == 9999

    def test_check_and_burn_rate_limited(self) -> None:
        """Test check_and_burn returns False when rate limited."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        limiter._remaining_limit = 0
        result = limiter.check_and_burn()
        assert result is False

    def test_check_and_burn_no_credits(self) -> None:
        """Test check_and_burn returns False when no credits."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        limiter._remaining_credits = 0
        result = limiter.check_and_burn()
        assert result is False

    def test_update_limits_after_time_passed(self) -> None:
        """Test _update_limits increases limit after time passed."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        limiter._remaining_limit = 10
        # Set last request time to 2 minutes ago
        limiter._last_request_time = time() - 120
        limiter._update_limits()
        # Should have increased by at least limit (capped at limit)
        assert limiter.remaining_limit == 30

    def test_update_limits_respects_cap(self) -> None:
        """Test _update_limits caps at limit."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        limiter._remaining_limit = 29
        limiter._last_request_time = time() - 120
        limiter._update_limits()
        assert limiter.remaining_limit == 30

    def test_update_limits_resets_credits_when_possible(self) -> None:
        """Test _update_limits resets credits when can_reset_credits is True."""
        limiter = CoingeckoRateLimiter(limit=30, credits_=10000)
        limiter._remaining_credits = 100
        # Set last request time far in the future to pass credits reset
        limiter._last_request_time = limiter.credits_reset_timestamp + 1000
        limiter._update_limits()
        assert limiter.remaining_credits == 10000


class TestCoingecko:
    """Test Coingecko class."""

    def _make_kwargs(self) -> dict:
        """Create kwargs for Coingecko initialization."""
        return {
            "token_price_endpoint": "/api/v3/simple/token_price",
            "coin_price_endpoint": "/api/v3/simple/price",
            "api_key": "test_key",
            "rate_limited_code": 429,
            "historical_price_endpoint": "/api/v3/coins/{coin}/history",
            "historical_market_data_endpoint": "/api/v3/coins/{coin}/market_chart",
            "chain_to_platform_id_mapping": json.dumps({"ethereum": "ethereum"}),
            "requests_per_minute": 30,
            "credits": 10000,
            "use_x402": False,
            "network_selector": "optimism",
            "coingecko_server_base_url": "https://api.coingecko.com",
            "coingecko_x402_server_base_url": "https://x402.example.com/{chain}",
            "coin_from_address_endpoint": "/api/v3/coins/{platform}/contract/{address}",
        }

    def test_initialization(self) -> None:
        """Test Coingecko initialization."""
        mock_context = MagicMock()
        kwargs = self._make_kwargs()
        cg = Coingecko(name="coingecko", skill_context=mock_context, **kwargs)
        assert cg.token_price_endpoint == "/api/v3/simple/token_price"
        assert cg.coin_price_endpoint == "/api/v3/simple/price"
        assert cg.api_key == "test_key"
        assert cg.rate_limited_code == 429
        assert cg.use_x402 is False
        assert cg.network_selector == "optimism"
        assert isinstance(cg.rate_limiter, CoingeckoRateLimiter)
        assert cg.chain_to_platform_id_mapping == {"ethereum": "ethereum"}

    def test_initialization_no_api_key(self) -> None:
        """Test Coingecko initialization without api_key."""
        mock_context = MagicMock()
        kwargs = self._make_kwargs()
        del kwargs["api_key"]
        cg = Coingecko(name="coingecko", skill_context=mock_context, **kwargs)
        assert cg.api_key == ""

    def test_rate_limited_status_callback(self) -> None:
        """Test rate_limited_status_callback sets limit to 0."""
        mock_context = MagicMock()
        kwargs = self._make_kwargs()
        cg = Coingecko(name="coingecko", skill_context=mock_context, **kwargs)
        cg.rate_limited_status_callback()
        assert cg.rate_limiter._remaining_limit == 0
        mock_context.logger.error.assert_called_once()

    def test_request_success_non_x402(self) -> None:
        """Test request method without x402."""
        mock_context = MagicMock()
        mock_context.params.request_timeout = 20.0
        kwargs = self._make_kwargs()
        cg = Coingecko(name="coingecko", skill_context=mock_context, **kwargs)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"price": 1.0}

        with patch(
            "packages.valory.skills.liquidity_trader_abci.models.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.get.return_value = mock_response
            mock_session_cls.return_value = mock_session

            success, data = cg.request("/test", {}, None)

        assert success is True
        assert data == {"price": 1.0}
        mock_session.get.assert_called_once_with(
            "https://api.coingecko.com/test", headers={}, timeout=20.0
        )

    def test_request_success_x402(self) -> None:
        """Test request method with x402."""
        mock_context = MagicMock()
        kwargs = self._make_kwargs()
        kwargs["use_x402"] = True
        cg = Coingecko(name="coingecko", skill_context=mock_context, **kwargs)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"price": 1.0}

        with patch(
            "packages.valory.skills.liquidity_trader_abci.models.x402_requests"
        ) as mock_x402:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.get.return_value = mock_response
            mock_x402.return_value = mock_session

            success, data = cg.request("/test", {}, MagicMock())

        assert success is True
        assert data == {"price": 1.0}

    def test_request_failure_status_code(self) -> None:
        """Test request method with non-OK status code."""
        mock_context = MagicMock()
        kwargs = self._make_kwargs()
        cg = Coingecko(name="coingecko", skill_context=mock_context, **kwargs)

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": "rate limited"}

        with patch(
            "packages.valory.skills.liquidity_trader_abci.models.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.get.return_value = mock_response
            mock_session_cls.return_value = mock_session

            success, data = cg.request("/test", {}, None)

        assert success is False
        assert data == {"error": "rate limited"}

    def test_request_exception(self) -> None:
        """Test request method when exception occurs."""
        mock_context = MagicMock()
        kwargs = self._make_kwargs()
        cg = Coingecko(name="coingecko", skill_context=mock_context, **kwargs)

        with patch(
            "packages.valory.skills.liquidity_trader_abci.models.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.get.side_effect = Exception("connection error")
            mock_session_cls.return_value = mock_session

            success, data = cg.request("/test", {}, None)

        assert success is False
        assert "exception" in data


class TestParams:
    """Test Params class."""

    def _make_kwargs(self, tmpdir: str) -> dict:
        """Create kwargs for Params initialization (Params-specific fields only)."""
        return {
            "initial_assets": json.dumps({}),
            "safe_contract_addresses": json.dumps({}),
            "merkl_fetch_campaigns_args": json.dumps({}),
            "allowed_chains": ["ethereum"],
            "gas_reserve": json.dumps({"ethereum": "0.01"}),
            "apr_threshold": 5,
            "round_threshold": 10,
            "min_balance_multiplier": 2,
            "multisend_contract_addresses": json.dumps({}),
            "lifi_advance_routes_url": "https://lifi.example.com",
            "lifi_fetch_step_transaction_url": "https://lifi.example.com/step",
            "lifi_check_status_url": "https://lifi.example.com/status",
            "slippage_for_swap": 0.03,
            "slippage_tolerance": 0.01,
            "allowed_dexs": ["uniswap"],
            "balancer_vault_contract_addresses": json.dumps({}),
            "balancer_queries_contract_addresses": json.dumps({}),
            "uniswap_position_manager_contract_addresses": json.dumps({}),
            "chain_to_chain_key_mapping": json.dumps({}),
            "max_num_of_retries": 3,
            "waiting_period_for_status_check": 10,
            "reward_claiming_time_period": 86400,
            "merkl_distributor_contract_addresses": json.dumps({}),
            "intermediate_tokens": json.dumps({}),
            "lifi_fetch_tools_url": "https://lifi.example.com/tools",
            "merkl_user_rewards_url": "https://merkl.example.com/rewards",
            "chain_to_chain_id_mapping": json.dumps({}),
            "staking_token_contract_address": "0xstaking",
            "staking_activity_checker_contract_address": "0xchecker",
            "staking_threshold_period": 86400,
            "store_path": tmpdir,
            "assets_info_filename": "assets.json",
            "pool_info_filename": "pools.json",
            "portfolio_info_filename": "portfolio.json",
            "gas_cost_info_filename": "gas_costs.json",
            "whitelisted_assets_filename": "whitelisted.json",
            "funding_events_filename": "funding.json",
            "agent_performance_filename": "performance.json",
            "min_investment_amount": 100,
            "max_fee_percentage": 0.1,
            "max_gas_percentage": 0.05,
            "balancer_graphql_endpoints": json.dumps({}),
            "target_investment_chains": ["ethereum"],
            "staking_chain": "ethereum",
            "file_hash_to_strategies": json.dumps({}),
            "strategies_kwargs": json.dumps({}),
            "available_protocols": ["balancerPool"],
            "selected_hyper_strategy": "default",
            "dex_type_to_strategy": json.dumps({}),
            "default_acceptance_time": 3600,
            "max_pools": 5,
            "profit_threshold": 100,
            "loss_threshold": -50,
            "pnl_check_interval": 3600,
            "available_strategies": json.dumps({"ethereum": ["strategy_a"]}),
            "cleanup_freq": 10,
            "genai_api_key": "test_key",
            "genai_model": "test_model",
            "velodrome_router_contract_addresses": json.dumps({}),
            "velodrome_non_fungible_position_manager_contract_addresses": json.dumps(
                {}
            ),
            "service_registry_contract_addresses": json.dumps({}),
            "staking_subgraph_endpoints": json.dumps({}),
            "velodrome_slipstream_helper_contract_addresses": json.dumps({}),
            "velodrome_voter_contract_addresses": json.dumps({}),
            "velodrome_rewards_sugar_contract_addresses": json.dumps({}),
            "velo_token_contract_addresses": json.dumps({}),
            "safe_api_base_url": "https://safe.example.com",
            "safe_api_timeout": 30,
            "mode_explorer_api_base_url": "https://mode.example.com",
            "mode_api_timeout": 30,
            "airdrop_started": False,
            "airdrop_contract_address": "0xairdrop",
            "use_x402": False,
            "x402_payment_requirements": json.dumps({}),
            "optimism_ledger_rpc": "https://optimism.rpc.example.com",
            "lifi_quote_to_amount_url": "https://lifi.example.com/quote",
            "request_timeout": 20.0,
            "skill_context": MagicMock(),
        }

    def test_initialization(self) -> None:
        """Test Params initialization with super().__init__ mocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = self._make_kwargs(tmpdir)
            params = object.__new__(Params)
            with patch.object(Params.__bases__[0], "__init__", return_value=None):
                params.__init__(**kwargs)
            assert params.apr_threshold == 5
            assert params.round_threshold == 10
            assert params.allowed_chains == ["ethereum"]
            assert params.slippage_for_swap == 0.03
            assert params.stoploss_threshold_multiplier == 0.43
            assert params.min_investment_amount == 100
            assert params.genai_api_key == "test_key"
            assert params.genai_model == "test_model"
            assert params.use_x402 is False

    def test_initialization_with_stoploss_multiplier(self) -> None:
        """Test Params initialization with custom stoploss_threshold_multiplier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = self._make_kwargs(tmpdir)
            kwargs["stoploss_threshold_multiplier"] = 0.5
            params = object.__new__(Params)
            with patch.object(Params.__bases__[0], "__init__", return_value=None):
                params.__init__(**kwargs)
            assert params.stoploss_threshold_multiplier == 0.5

    def test_initialization_with_tenderly_params(self) -> None:
        """Test Params initialization with tenderly params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = self._make_kwargs(tmpdir)
            kwargs["tenderly_bundle_simulation_url"] = "https://tenderly.example.com"
            kwargs["tenderly_access_key"] = "key123"
            kwargs["tenderly_account_slug"] = "account"
            kwargs["tenderly_project_slug"] = "project"
            params = object.__new__(Params)
            with patch.object(Params.__bases__[0], "__init__", return_value=None):
                params.__init__(**kwargs)
            assert (
                params.tenderly_bundle_simulation_url == "https://tenderly.example.com"
            )
            assert params.tenderly_access_key == "key123"

    def test_initialization_without_tenderly_params(self) -> None:
        """Test Params initialization without tenderly params gives defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = self._make_kwargs(tmpdir)
            params = object.__new__(Params)
            with patch.object(Params.__bases__[0], "__init__", return_value=None):
                params.__init__(**kwargs)
            assert params.tenderly_bundle_simulation_url == ""
            assert params.tenderly_access_key == ""
            assert params.tenderly_account_slug == ""
            assert params.tenderly_project_slug == ""

    def test_get_store_path_valid(self) -> None:
        """Test get_store_path with a valid path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = object.__new__(Params)
            mock_context = MagicMock()
            result = params.get_store_path(
                {"store_path": tmpdir, "skill_context": mock_context}
            )
            from pathlib import Path

            assert result == Path(tmpdir)

    def test_get_store_path_invalid(self) -> None:
        """Test get_store_path raises for invalid path."""
        params = object.__new__(Params)
        mock_context = MagicMock()
        with pytest.raises(ValueError, match="is not a directory"):
            params.get_store_path(
                {
                    "store_path": "/nonexistent/path/that/does/not/exist",
                    "skill_context": mock_context,
                }
            )

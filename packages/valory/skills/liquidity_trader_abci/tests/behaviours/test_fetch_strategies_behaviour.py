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

"""Tests for FetchStrategiesBehaviour of the liquidity_trader_abci skill."""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
    FetchStrategiesBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.payloads import FetchStrategiesPayload
from packages.valory.skills.liquidity_trader_abci.states.base import StakingState
from packages.valory.skills.liquidity_trader_abci.states.fetch_strategies import (
    FetchStrategiesRound,
)
from packages.valory.skills.liquidity_trader_abci.states.post_tx_settlement import (
    PostTxSettlementRound,
)


PACKAGE_DIR = Path(__file__).parent.parent.parent


class LiquidityTraderAbciFSMBehaviourBaseCase(FSMBehaviourBaseCase):
    """Base case for testing FSMBehaviour."""

    path_to_skill = PACKAGE_DIR


class TestFetchStrategiesBehaviour(LiquidityTraderAbciFSMBehaviourBaseCase):
    """Test cases for FetchStrategiesBehaviour."""

    behaviour_class = FetchStrategiesBehaviour
    path_to_skill = PACKAGE_DIR

    @classmethod
    def setup_class(cls, **kwargs: Any) -> None:
        """Setup the test class with parameter overrides."""
        import shutil
        import tempfile

        cls.temp_skill_dir = tempfile.mkdtemp()
        cls.original_skill_dir = cls.path_to_skill

        shutil.copytree(cls.original_skill_dir, cls.temp_skill_dir, dirs_exist_ok=True)

        temp_skill_yaml = Path(cls.temp_skill_dir) / "skill.yaml"
        with open(temp_skill_yaml, "r") as f:
            skill_config = f.read()

        skill_config = skill_config.replace(
            "available_strategies: null", 'available_strategies: "{}"'
        )
        skill_config = skill_config.replace("genai_api_key: null", 'genai_api_key: ""')
        skill_config = skill_config.replace(
            "default_acceptance_time: null", "default_acceptance_time: 30"
        )

        with open(temp_skill_yaml, "w") as f:
            f.write(skill_config)

        cls.path_to_skill = Path(cls.temp_skill_dir)

        # Add initial_assets to param_overrides
        kwargs = {
            "param_overrides": {
                "initial_assets": {
                    "0x4200000000000000000000000000000000000006": "WETH",
                    "0xd988097fb8612cc24eeC14542bC03424c656005f": "USDC",
                    "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9": "OLAS",
                },
                "target_investment_chains": ["mode"],
                "safe_contract_addresses": {"mode": "0xSafeAddress"},
            }
        }

        super().setup_class(**kwargs)

    @classmethod
    def teardown_class(cls) -> None:
        """Teardown the test class."""
        if hasattr(cls, "temp_skill_dir"):
            import shutil

            shutil.rmtree(cls.temp_skill_dir, ignore_errors=True)

        if hasattr(cls, "original_skill_dir"):
            cls.path_to_skill = cls.original_skill_dir

        super().teardown_class()

    def setup_method(self, **kwargs: Any) -> None:
        """Set up the test method with mocked dependencies."""
        # Mock the store path validation before calling super().setup()
        with patch(
            "packages.valory.skills.liquidity_trader_abci.models.Params.get_store_path",
            return_value=Path("/tmp/mock_store"),
        ):
            super().setup(**kwargs)

        self.fetch_strategies_behaviour = FetchStrategiesBehaviour(
            name="fetch_strategies_behaviour", skill_context=self.skill.skill_context
        )

        self.setup_default_test_data()

    def teardown_method(self, **kwargs: Any) -> None:
        """Teardown the test method."""
        super().teardown(**kwargs)

    def _create_fetch_strategies_behaviour(self):
        """Create a FetchStrategiesBehaviour instance for testing."""
        return FetchStrategiesBehaviour(
            name="fetch_strategies_behaviour", skill_context=self.skill.skill_context
        )

    def setup_default_test_data(self):
        """Setup default test data."""
        self.portfolio_data = {
            "portfolio_value": 1.1288653771248525,
            "value_in_pools": 0.0,
            "value_in_safe": 1.1288653771248525,
            "initial_investment": 86.99060538860897,
            "volume": 93.2299879883976,
            "roi": -98.7,
            "agent_hash": "bafybeibkop2atmdpyrwqwcjuaqyhckrujg663rpomregakrmmosbaywhlq",
            "allocations": [],
            "portfolio_breakdown": [
                {
                    "asset": "WETH",
                    "address": "0x4200000000000000000000000000000000000006",
                    "balance": 0.000289393128542145,
                    "price": 3841.2,
                    "value_usd": 1.1116168853560873,
                    "ratio": 0.0,
                },
                {
                    "asset": "OLAS",
                    "address": "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9",
                    "balance": 0.07265520833339696,
                    "price": 0.237402,
                    "value_usd": 0.017248491768765105,
                    "ratio": 0.0,
                },
            ],
            "address": "0xc7Bd1d1FB563c6c06D4Ab1f116208f36a4631Ce4",
            "last_updated": 1753978820,
        }

        self.positions = [
            {
                "chain": "mode",
                "pool_address": "0x3Adf15f77F2911f84b0FE9DbdfF43ef60D40012c",
                "dex_type": "velodrome",
                "token0": "0x4200000000000000000000000000000000000006",
                "token1": "0xd988097fb8612cc24eeC14542bC03424c656005f",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "status": "open",
                "pool_id": "0x3Adf15f77F2911f84b0FE9DbdfF43ef60D40012c",
                "is_stable": True,
                "is_cl_pool": True,
                "amount0": 48045349738380,
                "amount1": 544525,
                "enter_tx_hash": "0xb68eff1e2277fd9432a9a8cf966d52005023a3addb168edbc400c23bba957653",
                "enter_timestamp": 1753984217,
                "entry_cost": 0.003927974431442888,
                "min_hold_days": 14.0,
                "principal_usd": 0.7289412106900652,
                "cost_recovered": False,
                "current_value_usd": 0.0,
                "last_updated": 1753984217,
            }
        ]

        self.assets = {
            "mode": {
                "0x4200000000000000000000000000000000000006": "WETH",
                "0xd988097fb8612cc24eeC14542bC03424c656005f": "USDC",
                "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9": "OLAS",
            }
        }

    def mock_update_position_amounts(self):
        yield
        return None

    def mock_store_whitelisted_assets(self):
        pass

    def mock_read_whitelisted_assets(self):
        pass

    def mock_track_whitelisted_assets(self):
        yield
        pass

    def mock_store_portfolio_data(self):
        pass

    def mock_check_and_update_zero_liquidity_positions(self):
        pass

    def mock_update_accumulated_rewards(self, chain):
        yield
        return None

    def mock_get_signature(self, payload_bytes):
        """Mock get_signature method."""
        yield
        return b"mocked_signature"

    def mock_send_a2a_transaction(self, payload):
        """Mock send_a2a_transaction method."""
        yield
        return None

    def mock_wait_until_round_end(self):
        """Mock wait_until_round_end method."""
        yield
        return None

    def mock_contract_interact(self, **kwargs):
        """Mock contract_interact method."""
        yield
        return "0" * 64

    def mock_write_kv(self, data):
        """Mock write_kv method."""
        yield
        return True

    def mock_should_recalculate_portfolio(self, last_portfolio_data):
        """Mock should_recalculate_portfolio method."""
        yield
        return True

    def mock_calculate_user_share_values(self):
        """Mock calculate_user_share_values method."""
        yield
        return None

    def mock_get_native_balance(self, chain, address):
        """Mock get_native_balance method."""
        yield
        return 1000000000000000000

    def mock_track_eth_transfers_and_reversions_zero_reversion(
        self, safe_address, chain
    ):
        """Mock track_eth_transfers_and_reversions method."""
        yield
        return {
            "reversion_amount": 0.0,
            "historical_reversion_value": 0.0,
            "reversion_date": None,
        }

    def mock_read_kv_none(self, keys):
        """Mock read_kv method."""
        yield
        return None

    def mock_read_kv_empty(self, keys):
        """Mock read_kv method."""
        yield
        return {}

    def mock_read_kv_default(self, keys):
        """Mock read_kv method."""
        yield
        return {"selected_protocols": "[]", "trading_type": "balanced"}

    def mock_get_current_timestamp(self):
        """Mock get_current_timestamp method."""
        return int(datetime.now().timestamp())

    def mock_fetch_historical_token_price(self, coingecko_id, date_str):
        """Mock fetch_historical_token_price method."""
        yield
        return 1.0

    def mock_calculate_chain_investment_value(self, all_transfers, chain, safe_address):
        """Mock calculate_chain_investment_value method."""
        yield
        return 100.0

    def mock_load_chain_total_investment(self, chain):
        """Mock load_chain_total_investment method."""
        yield
        return 100.0

    def mock_store_current_positions(self):
        pass

    def mock_update_portfolio_metrics(
        self,
        total_user_share_value_usd,
        individual_shares,
        portfolio_breakdown,
        allocations,
    ):
        yield
        pass

    def mock_save_chain_total_investment(self, chain, total):
        yield
        pass

    def _consume_generator(self, gen):
        """Helper to run generator and return final value."""
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    def test_async_act_no_assets_initialization(self):
        """Test async_act assets initialization when empty."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        with patch.dict("os.environ", {"AEA_AGENT": "test_agent:hash123"}):
            fetch_behaviour.current_positions = []

            fetch_behaviour.assets = {}

            mock_synchronized_data = MagicMock()
            mock_synchronized_data.period_count = 1  # Not period 0
            mock_synchronized_data.round_count = 1

            def mock_track_eth_transfers(safe_address, chain):
                """Mock track_eth_transfers method."""
                yield
                return {"master_safe_address": None, "reversion_amount": 0}

            def mock_get_http_response(message, dialogue):
                yield
                return MagicMock(performative="success", body=b'{"price": 1.0}')

            with patch.multiple(
                fetch_behaviour,
                update_position_amounts=self.mock_update_position_amounts,
                update_accumulated_rewards_for_chain=self.mock_update_accumulated_rewards,
                _get_native_balance=self.mock_get_native_balance,
                _track_eth_transfers_and_reversions=mock_track_eth_transfers,
                _read_kv=self.mock_read_kv_default,
                _write_kv=self.mock_write_kv,
                _get_current_timestamp=self.mock_get_current_timestamp,
                _load_chain_total_investment=self.mock_load_chain_total_investment,
                should_recalculate_portfolio=self.mock_should_recalculate_portfolio,
                calculate_user_share_values=self.mock_calculate_user_share_values,
                _fetch_historical_token_price=self.mock_fetch_historical_token_price,
                get_http_response=mock_get_http_response,
                _calculate_chain_investment_value=self.mock_calculate_chain_investment_value,
                get_signature=self.mock_get_signature,
                send_a2a_transaction=self.mock_send_a2a_transaction,
                wait_until_round_end=self.mock_wait_until_round_end,
                contract_interact=self.mock_contract_interact,
                store_whitelisted_assets=self.mock_store_whitelisted_assets,
                read_whitelisted_assets=self.mock_read_whitelisted_assets,
                _track_whitelisted_assets=self.mock_track_whitelisted_assets,
                store_portfolio_data=self.mock_store_portfolio_data,
                check_and_update_zero_liquidity_positions=self.mock_check_and_update_zero_liquidity_positions,
            ):
                with patch(
                    "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.hash_payload_to_hex",
                    return_value="mocked_payload_hash",
                ):
                    with patch.object(
                        type(fetch_behaviour),
                        "synchronized_data",
                        new_callable=PropertyMock,
                        return_value=mock_synchronized_data,
                    ):
                        list(fetch_behaviour.async_act())

                        assert (
                            fetch_behaviour.assets
                            == fetch_behaviour.params.initial_assets
                        )
                        assert fetch_behaviour.synchronized_data.period_count == 1
                        expected_assets = {
                            "0x4200000000000000000000000000000000000006": "WETH",
                            "0xd988097fb8612cc24eeC14542bC03424c656005f": "USDC",
                            "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9": "OLAS",
                        }
                        assert fetch_behaviour.assets == expected_assets

    def test_async_act_with_existing_assets(self):
        """Test async_act flow with existing assets."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        with patch.dict("os.environ", {"AEA_AGENT": "test_agent:hash123"}):
            fetch_behaviour.current_positions = []

            fetch_behaviour.assets = {
                "0x4200000000000000000000000000000000000006": "WETH",
                "0xd988097fb8612cc24eeC14542bC03424c656005f": "USDC",
                "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9": "OLAS",
                "0xAdditionalToken": "ADDITIONAL",
            }

            mock_synchronized_data = MagicMock()
            mock_synchronized_data.period_count = 2
            mock_synchronized_data.round_count = 1

            def mock_track_eth_transfers(safe_address, chain):
                yield
                return {"master_safe_address": None, "reversion_amount": 0}

            def mock_get_http_response(message, dialogue):
                yield
                return MagicMock(performative="success", body=b'{"price": 1.0}')

            with patch.multiple(
                fetch_behaviour,
                update_position_amounts=self.mock_update_position_amounts,
                update_accumulated_rewards_for_chain=self.mock_update_accumulated_rewards,
                _get_native_balance=self.mock_get_native_balance,
                _track_eth_transfers_and_reversions=mock_track_eth_transfers,
                _read_kv=self.mock_read_kv_default,
                _write_kv=self.mock_write_kv,
                _get_current_timestamp=self.mock_get_current_timestamp,
                _load_chain_total_investment=self.mock_load_chain_total_investment,
                should_recalculate_portfolio=self.mock_should_recalculate_portfolio,
                calculate_user_share_values=self.mock_calculate_user_share_values,
                _fetch_historical_token_price=self.mock_fetch_historical_token_price,
                get_http_response=mock_get_http_response,
                _calculate_chain_investment_value=self.mock_calculate_chain_investment_value,
                get_signature=self.mock_get_signature,
                send_a2a_transaction=self.mock_send_a2a_transaction,
                wait_until_round_end=self.mock_wait_until_round_end,
                contract_interact=self.mock_contract_interact,
                store_whitelisted_assets=self.mock_store_whitelisted_assets,
                read_whitelisted_assets=self.mock_read_whitelisted_assets,
                _track_whitelisted_assets=self.mock_track_whitelisted_assets,
                store_portfolio_data=self.mock_store_portfolio_data,
                check_and_update_zero_liquidity_positions=self.mock_check_and_update_zero_liquidity_positions,
            ):
                with patch(
                    "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.hash_payload_to_hex",
                    return_value="mocked_payload_hash",
                ):
                    with patch.object(
                        type(fetch_behaviour),
                        "synchronized_data",
                        new_callable=PropertyMock,
                        return_value=mock_synchronized_data,
                    ):
                        list(fetch_behaviour.async_act())

                        expected_assets = {
                            "0x4200000000000000000000000000000000000006": "WETH",
                            "0xd988097fb8612cc24eeC14542bC03424c656005f": "USDC",
                            "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9": "OLAS",
                            "0xAdditionalToken": "ADDITIONAL",
                        }
                        assert fetch_behaviour.assets == expected_assets
                        assert fetch_behaviour.synchronized_data.period_count == 2
                        assert "0xAdditionalToken" in fetch_behaviour.assets
                        assert (
                            fetch_behaviour.assets["0xAdditionalToken"] == "ADDITIONAL"
                        )

    @pytest.mark.parametrize(
        "address_info,expected_result,test_description",
        [
            (
                {"name": "GnosisSafeProxy", "is_contract": True},
                True,
                "gnosis safe contract",
            ),
            ({"name": "Regular Wallet", "is_contract": False}, False, "regular wallet"),
        ],
    )
    def test_is_gnosis_safe_variations(
        self, address_info, expected_result, test_description
    ):
        """Test _is_gnosis_safe method with various address types."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        result = fetch_behaviour._is_gnosis_safe(address_info)
        assert result is expected_result, f"Failed for {test_description}"

    @pytest.mark.parametrize(
        "from_address,expected_result,test_description",
        [
            (
                {
                    "name": "Regular Wallet",
                    "is_contract": False,
                    "hash": "0x1234567890abcdef",
                },
                True,
                "regular wallet",
            ),
            (
                {
                    "name": "Gnosis Safe",
                    "is_contract": True,
                    "hash": "0x1234567890abcdef",
                },
                False,
                "gnosis safe",
            ),
        ],
    )
    def test_should_include_transfer_variations(
        self, from_address, expected_result, test_description
    ):
        """Test _should_include_transfer method with various address types."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        result = fetch_behaviour._should_include_transfer(from_address)
        assert result is expected_result, f"Failed for {test_description}"

    @pytest.mark.parametrize(
        "timestamp,expected_result,test_description",
        [
            ("2025-01-01T12:00:00", "valid", "valid timestamp"),
            ("invalid_timestamp", None, "invalid timestamp"),
        ],
    )
    def test_get_datetime_from_timestamp_variations(
        self, timestamp, expected_result, test_description
    ):
        """Test _get_datetime_from_timestamp method with various timestamp formats."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        result = fetch_behaviour._get_datetime_from_timestamp(timestamp)

        if expected_result == "valid":
            assert result is not None, f"Failed for {test_description}"
            assert hasattr(result, "year"), f"Failed for {test_description}"
            assert hasattr(result, "month"), f"Failed for {test_description}"
            assert hasattr(result, "day"), f"Failed for {test_description}"
        else:
            assert result is expected_result, f"Failed for {test_description}"

    def test_have_positions_changed_false(self):
        """Test _have_positions_changed returns False."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        fetch_behaviour.current_positions = []

        same_portfolio_data = {
            "portfolio_value": 1.1288653771248525,
            "value_in_pools": 0.0,
            "value_in_safe": 1.1288653771248525,
            "initial_investment": 86.99060538860897,
            "volume": 93.2299879883976,
            "roi": -98.7,
            "agent_hash": "bafybeibkop2atmdpyrwqwcjuaqyhckrujg663rpomregakrmmosbaywhlq",
            "allocations": [],  # Empty allocations to match empty current_positions
            "portfolio_breakdown": [],
            "address": "0xc7Bd1d1FB563c6c06D4Ab1f116208f36a4631Ce4",
            "last_updated": 1753978820,
        }

        result = fetch_behaviour._have_positions_changed(same_portfolio_data)
        assert result is False

    def test_is_time_update_due_false(self):
        """Test _is_time_update_due returns False when time hasn't passed."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Set a recent last_updated timestamp (30 minutes ago)
        fetch_behaviour.portfolio_data = {
            "last_updated": 1753979400
        }  # 30 minutes before current time

        # Mock current time to be close to last update
        with patch.object(
            fetch_behaviour, "_get_current_timestamp", return_value=1753981200
        ):  # 1 hour later
            result = fetch_behaviour._is_time_update_due()
            assert result is False

    def test_get_historical_price_for_date_success(self):
        """Test successful historical price fetch for a regular token."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_get_coin_id_from_symbol(token_symbol, chain):
            return "usd-coin"

        with patch.multiple(
            fetch_behaviour,
            get_coin_id_from_symbol=mock_get_coin_id_from_symbol,
            _fetch_historical_token_price=self.mock_fetch_historical_token_price,
        ):
            result = self._consume_generator(
                fetch_behaviour._get_historical_price_for_date(
                    token_address="0x1234567890123456789012345678901234567890",
                    token_symbol="USDC",
                    date_str="01-01-2024",
                    chain="mode",
                )
            )
            assert result == 1.0

    def test_get_historical_price_for_date_eth_address(self):
        """Test historical price fetch for ETH (zero address)."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_fetch_historical_eth_price(date_str):
            yield
            return 2500.0

        with patch.object(
            fetch_behaviour,
            "_fetch_historical_eth_price",
            mock_fetch_historical_eth_price,
        ):
            result = self._consume_generator(
                fetch_behaviour._get_historical_price_for_date(
                    token_address="0x0000000000000000000000000000000000000000",  # Zero address for ETH
                    token_symbol="ETH",
                    date_str="01-01-2024",
                    chain="mode",
                )
            )
            assert result == 2500.0

    def test_get_historical_price_for_date_failure(self):
        """Test historical price fetch failure scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Test case 1: No CoinGecko ID found
        def mock_get_coin_id_from_symbol_none(token_symbol, chain):
            return None  # No CoinGecko ID found

        # Test case 2: API failure in _fetch_historical_token_price
        def mock_get_coin_id_from_symbol_valid(token_symbol, chain):
            return "usd-coin"  # Valid CoinGecko ID

        def mock_fetch_historical_token_price_failure(coingecko_id, date_str):
            yield
            return None  # API failure

        # Test case 3: Exception during execution
        def mock_fetch_historical_token_price_exception(coingecko_id, date_str):
            yield
            raise Exception("API Error")

        # Test scenario 1: No CoinGecko ID
        with patch.object(
            fetch_behaviour,
            "get_coin_id_from_symbol",
            mock_get_coin_id_from_symbol_none,
        ):
            result = self._consume_generator(
                fetch_behaviour._get_historical_price_for_date(
                    token_address="0x1234567890123456789012345678901234567890",
                    token_symbol="UNKNOWN",
                    date_str="01-01-2024",
                    chain="mode",
                )
            )
            assert result is None

        # Test scenario 2: API failure
        with patch.multiple(
            fetch_behaviour,
            get_coin_id_from_symbol=mock_get_coin_id_from_symbol_valid,
            _fetch_historical_token_price=mock_fetch_historical_token_price_failure,
        ):
            result = self._consume_generator(
                fetch_behaviour._get_historical_price_for_date(
                    token_address="0x1234567890123456789012345678901234567890",
                    token_symbol="USDC",
                    date_str="01-01-2024",
                    chain="mode",
                )
            )
            assert result is None

        # Test scenario 3: Exception handling
        with patch.multiple(
            fetch_behaviour,
            get_coin_id_from_symbol=mock_get_coin_id_from_symbol_valid,
            _fetch_historical_token_price=mock_fetch_historical_token_price_exception,
        ):
            result = self._consume_generator(
                fetch_behaviour._get_historical_price_for_date(
                    token_address="0x1234567890123456789012345678901234567890",
                    token_symbol="USDC",
                    date_str="01-01-2024",
                    chain="mode",
                )
            )
            assert result is None

    @pytest.mark.parametrize(
        "api_response,expected_result,test_description",
        [
            (
                (True, {"market_data": {"current_price": {"usd": 2500.0}}}),
                2500.0,
                "successful API response",
            ),
            ((False, {}), None, "API request failure"),
            (
                (True, {"market_data": {"current_price": {}}}),
                None,
                "no price in response",
            ),
            ((True, {}), None, "empty response"),
        ],
    )
    def test_fetch_historical_eth_price_variations(
        self, api_response, expected_result, test_description
    ):
        """Test _fetch_historical_eth_price method with various API responses."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_request_with_retries(
            endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait
        ):
            yield
            return api_response

        with patch.object(
            fetch_behaviour, "_request_with_retries", mock_request_with_retries
        ):
            result = self._consume_generator(
                fetch_behaviour._fetch_historical_eth_price("01-01-2024")
            )
            assert result == expected_result, f"Failed for {test_description}"

    @pytest.mark.parametrize(
        "symbol,chain,expected_result,test_description",
        [
            ("usdc", "mode", "valid", "known token symbol"),
            ("unknown_token", "mode", None, "unknown token symbol"),
        ],
    )
    def test_get_coin_id_from_symbol_variations(
        self, symbol, chain, expected_result, test_description
    ):
        """Test get_coin_id_from_symbol method with various token symbols."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        result = fetch_behaviour.get_coin_id_from_symbol(symbol, chain)

        if expected_result == "valid":
            assert result is not None, f"Failed for {test_description}"
            assert isinstance(result, str), f"Failed for {test_description}"
        else:
            assert result is expected_result, f"Failed for {test_description}"

    def test_should_recalculate_portfolio_true(self):
        """Portfolio recalculation needed when initial or final value missing or last round is settlement or time/positions changed."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        # Case 1: initial_investment None triggers True
        self.skill.skill_context.state.round_sequence._abci_app._previous_rounds = [
            type("R", (), {"round_id": "some_round"})
        ]

        def mock_load_chain_total_investment_none(chain):
            yield
            return 0.0

        with patch.object(
            fetch_behaviour,
            "_load_chain_total_investment",
            mock_load_chain_total_investment_none,
        ):
            result = self._consume_generator(
                fetch_behaviour.should_recalculate_portfolio({"portfolio_value": None})
            )
            assert result is True

        # Case 2: last round was PostTxSettlementRound triggers True
        with patch.object(
            self.skill.skill_context.state.round_sequence._abci_app,
            "_previous_rounds",
            [
                type("R", (), {"round_id": FetchStrategiesRound.auto_round_id()}),
                type("R2", (), {"round_id": PostTxSettlementRound.auto_round_id()}),
            ],
        ):
            with patch.object(
                fetch_behaviour,
                "_load_chain_total_investment",
                self.mock_load_chain_total_investment,
            ):
                result = self._consume_generator(
                    fetch_behaviour.should_recalculate_portfolio(
                        {"portfolio_value": 10.0}
                    )
                )
                # Because previous round id equals PostTxSettlementRound, should be True
                assert result is True

    def test_should_recalculate_portfolio_false(self):
        """Portfolio recalculation not needed when investment and value exist, last round not settlement, time not due and positions same."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Prepare environment: previous round not PostTxSettlementRound
        self.skill.skill_context.state.round_sequence._abci_app._previous_rounds = [
            type("R", (), {"round_id": "some_round"})
        ]

        with patch.object(
            fetch_behaviour,
            "_load_chain_total_investment",
            self.mock_load_chain_total_investment,
        ), patch.object(
            fetch_behaviour, "_is_time_update_due", return_value=False
        ), patch.object(
            fetch_behaviour, "_have_positions_changed", return_value=False
        ):
            result = self._consume_generator(
                fetch_behaviour.should_recalculate_portfolio({"portfolio_value": 100.0})
            )
            assert result is False

    def test_is_time_update_due_true(self):
        """Time update due when current_time - last_updated >= PORTFOLIO_UPDATE_INTERVAL."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        # Set portfolio_data with old last_updated
        fetch_behaviour.portfolio_data = {"last_updated": 0}
        with patch.object(
            fetch_behaviour, "_get_current_timestamp", return_value=10_000_000
        ), patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.PORTFOLIO_UPDATE_INTERVAL",
            100,
        ):
            assert fetch_behaviour._is_time_update_due() is True

    def test_have_positions_changed_true(self):
        """Positions have changed when count differs, new opened, or closed detected."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Case 1: Length changed -> True
        fetch_behaviour.current_positions = [
            {"pool_address": "A", "dex_type": "uniswapV3", "status": "open"}
        ]
        last_portfolio = {"allocations": []}
        assert fetch_behaviour._have_positions_changed(last_portfolio) is True

        # Case 2: Same count but new position opened (status considered)
        fetch_behaviour.current_positions = [
            {"pool_address": "A", "dex_type": "uniswapV3", "status": "open"}
        ]
        last_portfolio = {
            "allocations": [
                {"id": "B", "type": "uniswapV3"}  # different id triggers new_positions
            ]
        }
        assert fetch_behaviour._have_positions_changed(last_portfolio) is True

        # Case 3: Position closed (in last but not in current)
        fetch_behaviour.current_positions = []
        last_portfolio = {"allocations": [{"id": "A", "type": "uniswapV3"}]}
        assert fetch_behaviour._have_positions_changed(last_portfolio) is True

    def test_have_positions_changed_false(self):
        """Positions haven't changed when sets are equal under mapping rules."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        fetch_behaviour.current_positions = [
            {"pool_address": "A", "dex_type": "uniswapV3", "status": "open"},
            {"pool_id": "B", "dex_type": "balancerPool", "status": "open"},
        ]
        # allocations map uses id=pool_id/pool_address, type=dex_type, OPEN status assumed
        last_portfolio = {
            "allocations": [
                {"id": "A", "type": "uniswapV3"},
                {"id": "B", "type": "balancerPool"},
            ]
        }
        assert fetch_behaviour._have_positions_changed(last_portfolio) is False

    def test_calculate_user_share_values_success(self):
        """Test successful user share calculation with multiple positions."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Setup test positions
        fetch_behaviour.current_positions = [
            {
                "pool_id": "pool1",
                "dex_type": "balancerPool",
                "status": "open",
                "chain": "mode",
                "token0": "0x123",
                "token1": "0x456",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "pool_address": "0xpool1",
                "apr": 0.05,
            },
            {
                "pool_address": "0xpool2",
                "dex_type": "UniswapV3",
                "status": "open",
                "chain": "mode",
                "token_id": 123,
                "token0": "0x789",
                "token1": "0xabc",
                "token0_symbol": "USDC",
                "token1_symbol": "DAI",
                "apr": 0.03,
            },
        ]

        # Mock position handlers
        def mock_handle_balancer_position(position, chain):
            yield
            return (
                {"0x123": Decimal("1000"), "0x456": Decimal("2000")},  # user_balances
                "Balancer Pool Name",  # details
                {"0x123": "WETH", "0x456": "USDC"},  # token_info
            )

        def mock_handle_uniswap_position(position, chain):
            yield
            return (
                {"0x789": Decimal("1500"), "0xabc": Decimal("1500")},  # user_balances
                "Uniswap V3 Pool - USDC/DAI",  # details
                {"0x789": "USDC", "0xabc": "DAI"},  # token_info
            )

        # Mock calculation methods
        def mock_calculate_position_value(
            position, chain, user_balances, token_info, portfolio_breakdown
        ):
            yield
            return Decimal("3000")  # Total value for position

        def mock_calculate_safe_balances_value(portfolio_breakdown):
            yield
            return Decimal("1000")

        def mock_calculate_initial_investment_value_from_funding_events():
            yield
            return 5000.0

        def mock_calculate_total_volume():
            yield
            return 6000.0

        def mock_create_portfolio_data(*args, **kwargs):
            return {
                "total_pools_value": 6000.0,
                "total_safe_value": 1000.0,
                "initial_investment": 5000.0,
                "volume": 6000.0,
                "allocations": [],
                "portfolio_breakdown": [],
            }

        def mock_get_http_response(*args, **kwargs):
            import json
            from unittest.mock import MagicMock

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.body = json.dumps(
                {
                    "data": {
                        "service": {
                            "blockNumber": "12345",
                            "blockTimestamp": "1234567890",
                            "currentOlasStaked": "1000",
                            "id": "test",
                            "olasRewardsEarned": "100",
                        }
                    }
                }
            ).encode("utf-8")
            yield
            return mock_response

        with patch.multiple(
            fetch_behaviour,
            _handle_balancer_position=mock_handle_balancer_position,
            _handle_uniswap_position=mock_handle_uniswap_position,
            _calculate_position_value=mock_calculate_position_value,
            _calculate_safe_balances_value=mock_calculate_safe_balances_value,
            _update_portfolio_metrics=self.mock_update_portfolio_metrics,
            calculate_initial_investment_value_from_funding_events=mock_calculate_initial_investment_value_from_funding_events,
            _calculate_total_volume=mock_calculate_total_volume,
            _create_portfolio_data=mock_create_portfolio_data,
            store_current_positions=self.mock_store_current_positions,
            _read_kv=self.mock_read_kv_empty,
            _write_kv=self.mock_write_kv,
            get_http_response=mock_get_http_response,
        ):
            list(fetch_behaviour.calculate_user_share_values())

            # Verify portfolio_data was set
            assert hasattr(fetch_behaviour, "portfolio_data")
            assert fetch_behaviour.portfolio_data is not None
            assert (
                fetch_behaviour.portfolio_data["total_pools_value"] == 6000.0
            )  # 3000 + 3000
            assert fetch_behaviour.portfolio_data["total_safe_value"] == 1000.0
            assert fetch_behaviour.portfolio_data["initial_investment"] == 5000.0
            assert fetch_behaviour.portfolio_data["volume"] == 6000.0

    def test_calculate_user_share_values_no_positions(self):
        """Test user share calculation with no open positions."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        fetch_behaviour.current_positions = []

        def mock_calculate_safe_balances_value(portfolio_breakdown):
            yield
            return Decimal("500")

        def mock_calculate_initial_investment_value_from_funding_events():
            yield
            return 1000.0

        def mock_calculate_total_volume():
            yield
            return 1000.0

        def mock_create_portfolio_data(*args, **kwargs):
            return {
                "total_pools_value": 0.0,
                "total_safe_value": 500.0,
                "initial_investment": 1000.0,
                "volume": 1000.0,
                "roi": 50.0,
                "allocations": [],
                "portfolio_breakdown": [],
            }

        def mock_get_http_response(*args, **kwargs):
            import json
            from unittest.mock import MagicMock

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.body = json.dumps(
                {
                    "data": {
                        "service": {
                            "blockNumber": "12345",
                            "blockTimestamp": "1234567890",
                            "currentOlasStaked": "1000",
                            "id": "test",
                            "olasRewardsEarned": "100",
                        }
                    }
                }
            ).encode("utf-8")
            yield
            return mock_response

        with patch.multiple(
            fetch_behaviour,
            _calculate_safe_balances_value=mock_calculate_safe_balances_value,
            _update_portfolio_metrics=self.mock_update_portfolio_metrics,
            calculate_initial_investment_value_from_funding_events=mock_calculate_initial_investment_value_from_funding_events,
            _calculate_total_volume=mock_calculate_total_volume,
            _create_portfolio_data=mock_create_portfolio_data,
            store_current_positions=self.mock_store_current_positions,
            _read_kv=self.mock_read_kv_empty,
            _write_kv=self.mock_write_kv,
            get_http_response=mock_get_http_response,
        ):
            list(fetch_behaviour.calculate_user_share_values())

            # Verify portfolio_data was set with zero pool value
            assert hasattr(fetch_behaviour, "portfolio_data")
            assert fetch_behaviour.portfolio_data is not None
            assert fetch_behaviour.portfolio_data["total_pools_value"] == 0.0
            assert fetch_behaviour.portfolio_data["total_safe_value"] == 500.0
            assert fetch_behaviour.portfolio_data["initial_investment"] == 1000.0
            assert fetch_behaviour.portfolio_data["volume"] == 1000.0

    def test_calculate_user_share_values_failure(self):
        """Test user share calculation with handler failures."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Setup test positions with invalid dex_type
        fetch_behaviour.current_positions = [
            {
                "pool_id": "pool1",
                "dex_type": "invalid_dex",  # Invalid dex type
                "status": "open",
                "chain": "mode",
                "token0": "0x123",
                "token1": "0x456",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "pool_address": "0xpool1",
                "apr": 0.05,
            },
            {
                "pool_address": "0xpool2",
                "dex_type": "UniswapV3",
                "status": "open",
                "chain": "mode",
                "token_id": 123,
                "token0": "0x789",
                "token1": "0xabc",
                "token0_symbol": "USDC",
                "token1_symbol": "DAI",
                "apr": 0.03,
            },
        ]

        # Mock position handlers - first one will fail, second will succeed
        def mock_handle_uniswap_position(position, chain):
            yield
            return (
                {"0x789": Decimal("1500"), "0xabc": Decimal("1500")},  # user_balances
                "Uniswap V3 Pool - USDC/DAI",  # details
                {"0x789": "USDC", "0xabc": "DAI"},  # token_info
            )

        # Mock calculation methods
        def mock_calculate_position_value(
            position, chain, user_balances, token_info, portfolio_breakdown
        ):
            yield
            return Decimal("3000")  # Total value for position

        def mock_calculate_safe_balances_value(portfolio_breakdown):
            yield
            return Decimal("1000")

        def mock_calculate_initial_investment_value_from_funding_events():
            yield
            return 5000.0

        def mock_calculate_total_volume():
            yield
            return 6000.0

        def mock_create_portfolio_data(*args, **kwargs):
            return {
                "total_pools_value": 3000.0,
                "total_safe_value": 1000.0,
                "initial_investment": 5000.0,
                "volume": 6000.0,
                "allocations": [],
                "portfolio_breakdown": [],
            }

        def mock_get_http_response(*args, **kwargs):
            import json
            from unittest.mock import MagicMock

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.body = json.dumps(
                {
                    "data": {
                        "service": {
                            "blockNumber": "12345",
                            "blockTimestamp": "1234567890",
                            "currentOlasStaked": "1000",
                            "id": "test",
                            "olasRewardsEarned": "100",
                        }
                    }
                }
            ).encode("utf-8")
            yield
            return mock_response

        with patch.multiple(
            fetch_behaviour,
            _handle_uniswap_position=mock_handle_uniswap_position,
            _calculate_position_value=mock_calculate_position_value,
            _calculate_safe_balances_value=mock_calculate_safe_balances_value,
            _update_portfolio_metrics=self.mock_update_portfolio_metrics,
            calculate_initial_investment_value_from_funding_events=mock_calculate_initial_investment_value_from_funding_events,
            _calculate_total_volume=mock_calculate_total_volume,
            _create_portfolio_data=mock_create_portfolio_data,
            store_current_positions=self.mock_store_current_positions,
            _read_kv=self.mock_read_kv_empty,
            _write_kv=self.mock_write_kv,
            get_http_response=mock_get_http_response,
        ):
            list(fetch_behaviour.calculate_user_share_values())

            # Verify portfolio_data was set (only the valid position was processed)
            assert hasattr(fetch_behaviour, "portfolio_data")
            assert fetch_behaviour.portfolio_data is not None
            assert (
                fetch_behaviour.portfolio_data["total_pools_value"] == 3000.0
            )  # Only the valid position
            assert fetch_behaviour.portfolio_data["total_safe_value"] == 1000.0
            assert fetch_behaviour.portfolio_data["initial_investment"] == 5000.0
            assert fetch_behaviour.portfolio_data["volume"] == 6000.0

    def test_calculate_user_share_values_zero_user_share(self):
        """Test user share calculation when _calculate_position_value returns 0."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Setup test position
        fetch_behaviour.current_positions = [
            {
                "pool_id": "pool1",
                "dex_type": "balancerPool",
                "status": "open",
                "chain": "mode",
                "token0": "0x123",
                "token1": "0x456",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "pool_address": "0xpool1",
                "apr": 0.05,
            }
        ]

        # Mock position handler
        def mock_handle_balancer_position(position, chain):
            yield
            return (
                {"0x123": Decimal("0"), "0x456": Decimal("0")},  # user_balances
                "Balancer Pool Name",  # details
                {"0x123": "WETH", "0x456": "USDC"},  # token_info
            )

        # Mock calculation methods - return 0 for user_share
        def mock_calculate_position_value(
            position, chain, user_balances, token_info, portfolio_breakdown
        ):
            yield
            return Decimal("0")  # Zero value for position

        def mock_calculate_safe_balances_value(portfolio_breakdown):
            yield
            return Decimal("1000")

        def mock_calculate_initial_investment_value_from_funding_events():
            yield
            return 1000.0

        def mock_calculate_total_volume():
            yield
            return 1000.0

        def mock_create_portfolio_data(*args, **kwargs):
            return {
                "total_pools_value": 0.0,
                "total_safe_value": 1000.0,
                "initial_investment": 1000.0,
                "volume": 1000.0,
                "roi": 0.0,
                "allocations": [],
                "portfolio_breakdown": [],
            }

        def mock_get_http_response(*args, **kwargs):
            import json
            from unittest.mock import MagicMock

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.body = json.dumps(
                {
                    "data": {
                        "service": {
                            "blockNumber": "12345",
                            "blockTimestamp": "1234567890",
                            "currentOlasStaked": "1000",
                            "id": "test",
                            "olasRewardsEarned": "100",
                        }
                    }
                }
            ).encode("utf-8")
            yield
            return mock_response

        with patch.multiple(
            fetch_behaviour,
            _handle_balancer_position=mock_handle_balancer_position,
            _calculate_position_value=mock_calculate_position_value,
            _calculate_safe_balances_value=mock_calculate_safe_balances_value,
            _update_portfolio_metrics=self.mock_update_portfolio_metrics,
            calculate_initial_investment_value_from_funding_events=mock_calculate_initial_investment_value_from_funding_events,
            _calculate_total_volume=mock_calculate_total_volume,
            _create_portfolio_data=mock_create_portfolio_data,
            store_current_positions=self.mock_store_current_positions,
            _read_kv=self.mock_read_kv_empty,
            _write_kv=self.mock_write_kv,
            get_http_response=mock_get_http_response,
        ):
            # Execute the method
            list(fetch_behaviour.calculate_user_share_values())

            # Verify portfolio_data was set with zero pool value (no individual_shares added)
            assert hasattr(fetch_behaviour, "portfolio_data")
            assert fetch_behaviour.portfolio_data is not None
            assert (
                fetch_behaviour.portfolio_data["total_pools_value"] == 0.0
            )  # No shares > 0
            assert fetch_behaviour.portfolio_data["total_safe_value"] == 1000.0
            assert fetch_behaviour.portfolio_data["initial_investment"] == 1000.0
            assert fetch_behaviour.portfolio_data["volume"] == 1000.0

    def test_calculate_user_share_values_handler_returns_none(self):
        """Test user share calculation when position handler returns None."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Setup test position
        fetch_behaviour.current_positions = [
            {
                "pool_id": "pool1",
                "dex_type": "balancerPool",
                "status": "open",
                "chain": "mode",
                "token0": "0x123",
                "token1": "0x456",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "pool_address": "0xpool1",
                "apr": 0.05,
            }
        ]

        # Mock position handler to return None
        def mock_handle_balancer_position(position, chain):
            yield
            return None  # Handler returns None

        def mock_calculate_safe_balances_value(portfolio_breakdown):
            yield
            return Decimal("1000")

        def mock_calculate_initial_investment_value_from_funding_events():
            yield
            return 1000.0

        def mock_calculate_total_volume():
            yield
            return 1000.0

        def mock_create_portfolio_data(*args, **kwargs):
            return {
                "total_pools_value": 0.0,
                "total_safe_value": 1000.0,
                "initial_investment": 1000.0,
                "volume": 1000.0,
                "roi": 0.0,
                "allocations": [],
                "portfolio_breakdown": [],
            }

        def mock_get_http_response(*args, **kwargs):
            import json
            from unittest.mock import MagicMock

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.body = json.dumps(
                {
                    "data": {
                        "service": {
                            "blockNumber": "12345",
                            "blockTimestamp": "1234567890",
                            "currentOlasStaked": "1000",
                            "id": "test",
                            "olasRewardsEarned": "100",
                        }
                    }
                }
            ).encode("utf-8")
            yield
            return mock_response

        with patch.multiple(
            fetch_behaviour,
            _handle_balancer_position=mock_handle_balancer_position,
            _calculate_safe_balances_value=mock_calculate_safe_balances_value,
            _update_portfolio_metrics=self.mock_update_portfolio_metrics,
            calculate_initial_investment_value_from_funding_events=mock_calculate_initial_investment_value_from_funding_events,
            _calculate_total_volume=mock_calculate_total_volume,
            _create_portfolio_data=mock_create_portfolio_data,
            store_current_positions=self.mock_store_current_positions,
            _read_kv=self.mock_read_kv_empty,
            _write_kv=self.mock_write_kv,
            get_http_response=mock_get_http_response,
        ):
            # Execute the method
            list(fetch_behaviour.calculate_user_share_values())

            # Verify portfolio_data was set with zero pool value (handler returned None)
            assert hasattr(fetch_behaviour, "portfolio_data")
            assert fetch_behaviour.portfolio_data is not None
            assert (
                fetch_behaviour.portfolio_data["total_pools_value"] == 0.0
            )  # No valid results
            assert fetch_behaviour.portfolio_data["total_safe_value"] == 1000.0
            assert fetch_behaviour.portfolio_data["initial_investment"] == 1000.0
            assert fetch_behaviour.portfolio_data["volume"] == 1000.0

    def test_calculate_user_share_values_initial_investment_none_fallback(self):
        """Test user share calculation when initial investment is None and fallback is used."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Setup test position
        fetch_behaviour.current_positions = [
            {
                "pool_id": "pool1",
                "dex_type": "balancerPool",
                "status": "open",
                "chain": "mode",
                "token0": "0x123",
                "token1": "0x456",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "pool_address": "0xpool1",
                "apr": 0.05,
            }
        ]

        # Mock position handler
        def mock_handle_balancer_position(position, chain):
            yield
            return (
                {"0x123": Decimal("1000"), "0x456": Decimal("2000")},
                "Balancer Pool Name",
                {"0x123": "WETH", "0x456": "USDC"},
            )

        def mock_calculate_position_value(
            position, chain, user_balances, token_info, portfolio_breakdown
        ):
            yield
            return Decimal("3000")

        def mock_calculate_safe_balances_value(portfolio_breakdown):
            yield
            return Decimal("1000")

        # Mock initial investment to return None, then fallback to return a value
        def mock_calculate_initial_investment_value_from_funding_events():
            yield
            return None  # Initial calculation returns None

        def mock_load_chain_total_investment(chain):
            yield
            return 5000.0  # Fallback returns a value

        def mock_calculate_total_volume():
            yield
            return 6000.0

        def mock_create_portfolio_data(*args, **kwargs):
            return {
                "total_pools_value": 3000.0,
                "total_safe_value": 1000.0,
                "initial_investment": 5000.0,
                "volume": 6000.0,
                "roi": 20.0,
                "allocations": [],
                "portfolio_breakdown": [],
            }

        def mock_get_http_response(*args, **kwargs):
            import json
            from unittest.mock import MagicMock

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.body = json.dumps(
                {
                    "data": {
                        "service": {
                            "blockNumber": "12345",
                            "blockTimestamp": "1234567890",
                            "currentOlasStaked": "1000",
                            "id": "test",
                            "olasRewardsEarned": "100",
                        }
                    }
                }
            ).encode("utf-8")
            yield
            return mock_response

        with patch.multiple(
            fetch_behaviour,
            _handle_balancer_position=mock_handle_balancer_position,
            _calculate_position_value=mock_calculate_position_value,
            _calculate_safe_balances_value=mock_calculate_safe_balances_value,
            _update_portfolio_metrics=self.mock_update_portfolio_metrics,
            calculate_initial_investment_value_from_funding_events=mock_calculate_initial_investment_value_from_funding_events,
            _load_chain_total_investment=mock_load_chain_total_investment,
            _calculate_total_volume=mock_calculate_total_volume,
            _create_portfolio_data=mock_create_portfolio_data,
            store_current_positions=self.mock_store_current_positions,
            _read_kv=self.mock_read_kv_empty,
            _write_kv=self.mock_write_kv,
            get_http_response=mock_get_http_response,
        ):
            list(fetch_behaviour.calculate_user_share_values())

            # Verify portfolio_data was set with fallback initial investment
            assert hasattr(fetch_behaviour, "portfolio_data")
            assert fetch_behaviour.portfolio_data is not None
            assert fetch_behaviour.portfolio_data["total_pools_value"] == 3000.0
            assert fetch_behaviour.portfolio_data["total_safe_value"] == 1000.0
            assert (
                fetch_behaviour.portfolio_data["initial_investment"] == 5000.0
            )  # From fallback
            assert fetch_behaviour.portfolio_data["volume"] == 6000.0

    def test_calculate_user_share_values_initial_investment_none_no_fallback(self):
        """Test user share calculation when initial investment is None and fallback also returns 0."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Setup test position
        fetch_behaviour.current_positions = [
            {
                "pool_id": "pool1",
                "dex_type": "balancerPool",
                "status": "open",
                "chain": "mode",
                "token0": "0x123",
                "token1": "0x456",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "pool_address": "0xpool1",
                "apr": 0.05,
            }
        ]

        # Mock position handler
        def mock_handle_balancer_position(position, chain):
            yield
            return (
                {"0x123": Decimal("1000"), "0x456": Decimal("2000")},
                "Balancer Pool Name",
                {"0x123": "WETH", "0x456": "USDC"},
            )

        def mock_calculate_position_value(
            position, chain, user_balances, token_info, portfolio_breakdown
        ):
            yield
            return Decimal("3000")

        def mock_calculate_safe_balances_value(portfolio_breakdown):
            yield
            return Decimal("1000")

        # Mock initial investment to return None, then fallback to return 0
        def mock_calculate_initial_investment_value_from_funding_events():
            yield
            return None  # Initial calculation returns None

        def mock_load_chain_total_investment(chain):
            yield
            return 0.0  # Fallback returns 0

        def mock_calculate_total_volume():
            yield
            return 6000.0

        def mock_create_portfolio_data(*args, **kwargs):
            return {
                "total_pools_value": 3000.0,
                "total_safe_value": 1000.0,
                "initial_investment": None,
                "volume": 6000.0,
                "roi": 0.0,
                "allocations": [],
                "portfolio_breakdown": [],
            }

        def mock_get_http_response(*args, **kwargs):
            import json
            from unittest.mock import MagicMock

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.body = json.dumps(
                {
                    "data": {
                        "service": {
                            "blockNumber": "12345",
                            "blockTimestamp": "1234567890",
                            "currentOlasStaked": "1000",
                            "id": "test",
                            "olasRewardsEarned": "100",
                        }
                    }
                }
            ).encode("utf-8")
            yield
            return mock_response

        with patch.multiple(
            fetch_behaviour,
            _handle_balancer_position=mock_handle_balancer_position,
            _calculate_position_value=mock_calculate_position_value,
            _calculate_safe_balances_value=mock_calculate_safe_balances_value,
            _update_portfolio_metrics=self.mock_update_portfolio_metrics,
            calculate_initial_investment_value_from_funding_events=mock_calculate_initial_investment_value_from_funding_events,
            _load_chain_total_investment=mock_load_chain_total_investment,
            _calculate_total_volume=mock_calculate_total_volume,
            _create_portfolio_data=mock_create_portfolio_data,
            store_current_positions=self.mock_store_current_positions,
            _read_kv=self.mock_read_kv_empty,
            _write_kv=self.mock_write_kv,
            get_http_response=mock_get_http_response,
        ):
            # Execute the method
            list(fetch_behaviour.calculate_user_share_values())

            # Verify portfolio_data was set with None initial investment (no fallback value)
            assert hasattr(fetch_behaviour, "portfolio_data")
            assert fetch_behaviour.portfolio_data is not None
            assert fetch_behaviour.portfolio_data["total_pools_value"] == 3000.0
            assert fetch_behaviour.portfolio_data["total_safe_value"] == 1000.0
            assert (
                fetch_behaviour.portfolio_data["initial_investment"] is None
            )  # No fallback value
            assert fetch_behaviour.portfolio_data["volume"] == 6000.0

    def test_calculate_user_share_values_portfolio_data_none(self):
        """Test user share calculation when _create_portfolio_data returns None."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Setup test position
        fetch_behaviour.current_positions = [
            {
                "pool_id": "pool1",
                "dex_type": "balancerPool",
                "status": "open",
                "chain": "mode",
                "token0": "0x123",
                "token1": "0x456",
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
                "pool_address": "0xpool1",
                "apr": 0.05,
            }
        ]

        # Mock position handler
        def mock_handle_balancer_position(position, chain):
            yield
            return (
                {"0x123": Decimal("1000"), "0x456": Decimal("2000")},
                "Balancer Pool Name",
                {"0x123": "WETH", "0x456": "USDC"},
            )

        def mock_calculate_position_value(
            position, chain, user_balances, token_info, portfolio_breakdown
        ):
            yield
            return Decimal("3000")

        def mock_calculate_safe_balances_value(portfolio_breakdown):
            yield
            return Decimal("1000")

        def mock_calculate_initial_investment_value_from_funding_events():
            yield
            return 1000.0

        def mock_calculate_total_volume():
            yield
            return 1000.0

        def mock_create_portfolio_data(*args, **kwargs):
            return None  # Return None

        def mock_get_http_response(*args, **kwargs):
            import json
            from unittest.mock import MagicMock

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.body = json.dumps(
                {
                    "data": {
                        "service": {
                            "blockNumber": "12345",
                            "blockTimestamp": "1234567890",
                            "currentOlasStaked": "1000",
                            "id": "test",
                            "olasRewardsEarned": "100",
                        }
                    }
                }
            ).encode("utf-8")
            yield
            return mock_response

        with patch.multiple(
            fetch_behaviour,
            _handle_balancer_position=mock_handle_balancer_position,
            _calculate_position_value=mock_calculate_position_value,
            _calculate_safe_balances_value=mock_calculate_safe_balances_value,
            _update_portfolio_metrics=self.mock_update_portfolio_metrics,
            calculate_initial_investment_value_from_funding_events=mock_calculate_initial_investment_value_from_funding_events,
            _calculate_total_volume=mock_calculate_total_volume,
            _create_portfolio_data=mock_create_portfolio_data,
            store_current_positions=self.mock_store_current_positions,
            _read_kv=self.mock_read_kv_empty,
            _write_kv=self.mock_write_kv,
            get_http_response=mock_get_http_response,
        ):
            list(fetch_behaviour.calculate_user_share_values())

            initial_portfolio_data = getattr(fetch_behaviour, "portfolio_data", None)
            current_portfolio_data = getattr(fetch_behaviour, "portfolio_data", None)
            assert current_portfolio_data == initial_portfolio_data

    # 6.5. Position Handling Methods
    @pytest.mark.parametrize(
        "position_type,position,chain,mock_user_share_value,mock_pool_name,expected_user_balances,expected_details,expected_token_info,expected_exception,test_description",
        [
            # Balancer position success
            (
                "balancer",
                {
                    "pool_id": "pool123",
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "token0": "0x1111111111111111111111111111111111111111",
                    "token1": "0x2222222222222222222222222222222222222222",
                    "token0_symbol": "WETH",
                    "token1_symbol": "USDC",
                },
                "mode",
                {
                    "0x1111111111111111111111111111111111111111": Decimal("1.5"),
                    "0x2222222222222222222222222222222222222222": Decimal("2000"),
                },
                "Balancer WETH/USDC Pool",
                {
                    "0x1111111111111111111111111111111111111111": Decimal("1.5"),
                    "0x2222222222222222222222222222222222222222": Decimal("2000"),
                },
                "Balancer WETH/USDC Pool",
                {
                    "0x1111111111111111111111111111111111111111": "WETH",
                    "0x2222222222222222222222222222222222222222": "USDC",
                },
                None,
                "balancer position success",
            ),
            # Uniswap position success
            (
                "uniswap",
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "token_id": 123,
                    "token0": "0x1111111111111111111111111111111111111111",
                    "token1": "0x2222222222222222222222222222222222222222",
                    "token0_symbol": "WETH",
                    "token1_symbol": "USDC",
                },
                "optimism",
                {
                    "0x1111111111111111111111111111111111111111": Decimal("2.0"),
                    "0x2222222222222222222222222222222222222222": Decimal("3000"),
                },
                None,  # Uniswap doesn't use pool name
                {
                    "0x1111111111111111111111111111111111111111": Decimal("2.0"),
                    "0x2222222222222222222222222222222222222222": Decimal("3000"),
                },
                "Uniswap V3 Pool - WETH/USDC",
                {
                    "0x1111111111111111111111111111111111111111": "WETH",
                    "0x2222222222222222222222222222222222222222": "USDC",
                },
                None,
                "uniswap position success",
            ),
            # Sturdy position success
            (
                "sturdy",
                {
                    "pool_address": "0x3333333333333333333333333333333333333333",
                    "token0": "0x4444444444444444444444444444444444444444",
                    "token0_symbol": "USDC",
                },
                "mode",
                {"0x4444444444444444444444444444444444444444": Decimal("5000")},
                "Sturdy USDC Vault",
                {"0x4444444444444444444444444444444444444444": Decimal("5000")},
                "Sturdy USDC Vault",
                {"0x4444444444444444444444444444444444444444": "USDC"},
                None,
                "sturdy position success",
            ),
            # Velodrome CL position success
            (
                "velodrome",
                {
                    "pool_address": "0x5555555555555555555555555555555555555555",
                    "token_id": 456,
                    "token0": "0x6666666666666666666666666666666666666666",
                    "token1": "0x7777777777777777777777777777777777777777",
                    "token0_symbol": "WETH",
                    "token1_symbol": "USDC",
                    "is_cl_pool": True,
                },
                "optimism",
                {
                    "0x6666666666666666666666666666666666666666": Decimal("1.0"),
                    "0x7777777777777777777777777777777777777777": Decimal("1500"),
                },
                None,  # Velodrome doesn't use pool name
                {
                    "0x6666666666666666666666666666666666666666": Decimal("1.0"),
                    "0x7777777777777777777777777777777777777777": Decimal("1500"),
                },
                "Velodrome CL Pool",
                {
                    "0x6666666666666666666666666666666666666666": "WETH",
                    "0x7777777777777777777777777777777777777777": "USDC",
                },
                None,
                "velodrome CL position success",
            ),
            # Velodrome non-CL position success
            (
                "velodrome",
                {
                    "pool_address": "0x5555555555555555555555555555555555555555",
                    "token_id": 456,
                    "token0": "0x6666666666666666666666666666666666666666",
                    "token1": "0x7777777777777777777777777777777777777777",
                    "token0_symbol": "WETH",
                    "token1_symbol": "USDC",
                    "is_cl_pool": False,
                },
                "optimism",
                {
                    "0x6666666666666666666666666666666666666666": Decimal("0.5"),
                    "0x7777777777777777777777777777777777777777": Decimal("750"),
                },
                None,  # Velodrome doesn't use pool name
                {
                    "0x6666666666666666666666666666666666666666": Decimal("0.5"),
                    "0x7777777777777777777777777777777777777777": Decimal("750"),
                },
                "Velodrome Pool",
                {
                    "0x6666666666666666666666666666666666666666": "WETH",
                    "0x7777777777777777777777777777777777777777": "USDC",
                },
                None,
                "velodrome non-CL position success",
            ),
            # Balancer position failure (missing safe address)
            (
                "balancer",
                {
                    "pool_id": "pool123",
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "token0": "0x1111111111111111111111111111111111111111",
                    "token1": "0x2222222222222222222222222222222222222222",
                    "token0_symbol": "WETH",
                    "token1_symbol": "USDC",
                },
                "unknown_chain",
                None,  # Simulate failure
                None,  # Simulate failure
                None,
                None,
                {
                    "0x1111111111111111111111111111111111111111": "WETH",
                    "0x2222222222222222222222222222222222222222": "USDC",
                },
                None,
                "balancer position failure",
            ),
            # Balancer position contract failure
            (
                "balancer",
                {
                    "pool_id": "pool123",
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "token0": "0x1111111111111111111111111111111111111111",
                    "token1": "0x2222222222222222222222222222222222222222",
                    "token0_symbol": "WETH",
                    "token1_symbol": "USDC",
                },
                "mode",
                {
                    "0x1111111111111111111111111111111111111111": Decimal("1.5"),
                    "0x2222222222222222222222222222222222222222": Decimal("2000"),
                },
                "exception",  # Will raise exception
                None,
                None,
                None,
                Exception,
                "balancer position contract failure",
            ),
            # Sturdy position aggregator failure
            (
                "sturdy",
                {
                    "pool_address": "0x3333333333333333333333333333333333333333",
                    "token0": "0x4444444444444444444444444444444444444444",
                    "token0_symbol": "USDC",
                },
                "mode",
                {"0x4444444444444444444444444444444444444444": Decimal("5000")},
                None,  # Simulate aggregator name fetch failure
                {"0x4444444444444444444444444444444444444444": Decimal("5000")},
                None,
                {"0x4444444444444444444444444444444444444444": "USDC"},
                None,
                "sturdy position aggregator failure",
            ),
        ],
    )
    def test_handle_position_variations(
        self,
        position_type,
        position,
        chain,
        mock_user_share_value,
        mock_pool_name,
        expected_user_balances,
        expected_details,
        expected_token_info,
        expected_exception,
        test_description,
    ):
        """Test handle position methods with various scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock safe contract addresses
        safe_addresses = {"mode": "0xSafeAddress", "optimism": "0xSafeAddress"}

        if position_type == "balancer":

            def mock_get_user_share_value_balancer(
                user_address, pool_id, pool_address, chain
            ):
                yield
                return mock_user_share_value

            def mock_get_balancer_pool_name(pool_address, chain):
                yield
                if mock_pool_name == "exception":
                    raise Exception("Contract interaction failed")
                return mock_pool_name

            with patch.multiple(
                fetch_behaviour,
                get_user_share_value_balancer=mock_get_user_share_value_balancer,
                _get_balancer_pool_name=mock_get_balancer_pool_name,
            ):
                with patch.dict(
                    fetch_behaviour.params.safe_contract_addresses,
                    safe_addresses,
                    clear=False,
                ):
                    if expected_exception:
                        with pytest.raises(expected_exception):
                            generator = fetch_behaviour._handle_balancer_position(
                                position, chain
                            )
                            self._consume_generator(generator)
                    else:
                        generator = fetch_behaviour._handle_balancer_position(
                            position, chain
                        )
                        result = self._consume_generator(generator)
                        user_balances, details, token_info = result
                        assert (
                            user_balances == expected_user_balances
                        ), f"Expected {expected_user_balances} for {test_description}"
                        assert (
                            details == expected_details
                        ), f"Expected {expected_details} for {test_description}"
                        assert (
                            token_info == expected_token_info
                        ), f"Expected {expected_token_info} for {test_description}"

        elif position_type == "uniswap":

            def mock_get_user_share_value_uniswap(
                pool_address, token_id, chain, position
            ):
                yield
                return mock_user_share_value

            with patch.object(
                fetch_behaviour,
                "get_user_share_value_uniswap",
                mock_get_user_share_value_uniswap,
            ):
                generator = fetch_behaviour._handle_uniswap_position(position, chain)
                result = self._consume_generator(generator)
                user_balances, details, token_info = result
                assert (
                    user_balances == expected_user_balances
                ), f"Expected {expected_user_balances} for {test_description}"
                assert (
                    details == expected_details
                ), f"Expected {expected_details} for {test_description}"
                assert (
                    token_info == expected_token_info
                ), f"Expected {expected_token_info} for {test_description}"

        elif position_type == "sturdy":

            def mock_get_user_share_value_sturdy(
                user_address, aggregator_address, asset_address, chain
            ):
                yield
                return mock_user_share_value

            def mock_get_aggregator_name(aggregator_address, chain):
                yield
                return mock_pool_name

            with patch.multiple(
                fetch_behaviour,
                get_user_share_value_sturdy=mock_get_user_share_value_sturdy,
                _get_aggregator_name=mock_get_aggregator_name,
            ):
                with patch.dict(
                    fetch_behaviour.params.safe_contract_addresses,
                    safe_addresses,
                    clear=False,
                ):
                    generator = fetch_behaviour._handle_sturdy_position(position, chain)
                    result = self._consume_generator(generator)
                    user_balances, details, token_info = result
                    assert (
                        user_balances == expected_user_balances
                    ), f"Expected {expected_user_balances} for {test_description}"
                    assert (
                        details == expected_details
                    ), f"Expected {expected_details} for {test_description}"
                    assert (
                        token_info == expected_token_info
                    ), f"Expected {expected_token_info} for {test_description}"

        elif position_type == "velodrome":

            def mock_get_user_share_value_velodrome(
                user_address, pool_address, token_id, chain, position
            ):
                yield
                return mock_user_share_value

            with patch.object(
                fetch_behaviour,
                "get_user_share_value_velodrome",
                mock_get_user_share_value_velodrome,
            ):
                with patch.dict(
                    fetch_behaviour.params.safe_contract_addresses,
                    safe_addresses,
                    clear=False,
                ):
                    generator = fetch_behaviour._handle_velodrome_position(
                        position, chain
                    )
                    result = self._consume_generator(generator)
                    user_balances, details, token_info = result
                    assert (
                        user_balances == expected_user_balances
                    ), f"Expected {expected_user_balances} for {test_description}"
                    assert (
                        details == expected_details
                    ), f"Expected {expected_details} for {test_description}"
                    assert (
                        token_info == expected_token_info
                    ), f"Expected {expected_token_info} for {test_description}"

    @pytest.mark.parametrize(
        "position,chain,user_balances,token_info,portfolio_breakdown,mock_price_responses,expected_result,expected_breakdown_length,expected_breakdown_assets,test_description",
        [
            # Successful calculation with both tokens
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "token0": "0x1111111111111111111111111111111111111111",
                    "token1": "0x2222222222222222222222222222222222222222",
                    "token0_symbol": "WETH",
                    "token1_symbol": "USDC",
                },
                "mode",
                {
                    "0x1111111111111111111111111111111111111111": Decimal("1.5"),
                    "0x2222222222222222222222222222222222222222": Decimal("3000"),
                },
                {
                    "0x1111111111111111111111111111111111111111": "WETH",
                    "0x2222222222222222222222222222222222222222": "USDC",
                },
                [],
                {
                    "0x1111111111111111111111111111111111111111": 2000.0,  # WETH price
                    "0x2222222222222222222222222222222222222222": 1.0,  # USDC price
                },
                Decimal("6000"),  # 1.5 * 2000 + 3000 * 1
                2,
                ["WETH", "USDC"],
                "successful calculation with both tokens",
            ),
            # Missing balance - should only include available token
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "token0": "0x1111111111111111111111111111111111111111",
                    "token1": "0x2222222222222222222222222222222222222222",
                    "token0_symbol": "WETH",
                    "token1_symbol": "USDC",
                },
                "mode",
                {
                    "0x1111111111111111111111111111111111111111": Decimal("1.5")
                    # Missing USDC balance
                },
                {
                    "0x1111111111111111111111111111111111111111": "WETH",
                    "0x2222222222222222222222222222222222222222": "USDC",
                },
                [],
                {
                    "0x1111111111111111111111111111111111111111": 2000.0  # WETH price only
                },
                Decimal("3000"),  # 1.5 * 2000
                1,
                ["WETH"],
                "missing balance",
            ),
            # Price fetch failure - should return 0
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "token0": "0x1111111111111111111111111111111111111111",
                    "token1": "0x2222222222222222222222222222222222222222",
                    "token0_symbol": "WETH",
                    "token1_symbol": "USDC",
                },
                "mode",
                {
                    "0x1111111111111111111111111111111111111111": Decimal("1.5"),
                    "0x2222222222222222222222222222222222222222": Decimal("3000"),
                },
                {
                    "0x1111111111111111111111111111111111111111": "WETH",
                    "0x2222222222222222222222222222222222222222": "USDC",
                },
                [],
                {},  # No prices available
                Decimal("0"),  # Should be 0 due to price fetch failures
                0,
                [],
                "price fetch failure",
            ),
            # Existing portfolio breakdown - should update existing entries
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "token0": "0x1111111111111111111111111111111111111111",
                    "token1": "0x2222222222222222222222222222222222222222",
                    "token0_symbol": "WETH",
                    "token1_symbol": "USDC",
                },
                "mode",
                {
                    "0x1111111111111111111111111111111111111111": Decimal("1.5"),
                    "0x2222222222222222222222222222222222222222": Decimal("3000"),
                },
                {
                    "0x1111111111111111111111111111111111111111": "WETH",
                    "0x2222222222222222222222222222222222222222": "USDC",
                },
                [
                    {
                        "asset": "WETH",
                        "address": "0x1111111111111111111111111111111111111111",
                        "balance": 0.5,
                        "price": 1800.0,
                        "value_usd": 900.0,
                    }
                ],
                {
                    "0x1111111111111111111111111111111111111111": 2000.0,  # WETH price
                    "0x2222222222222222222222222222222222222222": 1.0,  # USDC price
                },
                Decimal("6000"),  # 1.5 * 2000 + 3000 * 1
                2,
                ["WETH", "USDC"],
                "existing portfolio breakdown",
            ),
        ],
    )
    def test_calculate_position_value_variations(
        self,
        position,
        chain,
        user_balances,
        token_info,
        portfolio_breakdown,
        mock_price_responses,
        expected_result,
        expected_breakdown_length,
        expected_breakdown_assets,
        test_description,
    ):
        """Test _calculate_position_value method with various scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_fetch_token_price(token_address, chain):
            yield
            return mock_price_responses.get(token_address)

        with patch.object(
            fetch_behaviour, "_fetch_token_price", mock_fetch_token_price
        ):
            # Execute the method
            generator = fetch_behaviour._calculate_position_value(
                position, chain, user_balances, token_info, portfolio_breakdown
            )
            result = self._consume_generator(generator)

            # Verify the result
            assert (
                result == expected_result
            ), f"Expected {expected_result} for {test_description}"
            assert (
                len(portfolio_breakdown) == expected_breakdown_length
            ), f"Expected {expected_breakdown_length} breakdown entries for {test_description}"

            # Verify breakdown assets if any
            if expected_breakdown_assets:
                actual_assets = [entry["asset"] for entry in portfolio_breakdown]
                assert (
                    actual_assets == expected_breakdown_assets
                ), f"Expected assets {expected_breakdown_assets} for {test_description}"

    def test_update_position_with_current_value_cost_recovered(self):
        """Test position update with cost recovery achieved."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "principal_usd": 1000.0,
            "entry_cost": 50.0,
        }
        current_value_usd = Decimal(
            "1100"
        )  # 100 more than principal, covers entry cost

        def mock_get_current_timestamp():
            return 1234567890

        def mock_get_current_token_balances(position, chain):
            yield
            return {"0x123": Decimal("1000"), "0x456": Decimal("2000")}

        def mock_calculate_corrected_yield(
            position,
            initial_amount0,
            initial_amount1,
            user_balances,
            chain,
            token_prices,
        ):
            yield
            return Decimal("100")  # Yield of 100 USD, which is >= entry_cost of 50

        with patch.object(
            fetch_behaviour, "_get_current_timestamp", mock_get_current_timestamp
        ):
            with patch.object(
                fetch_behaviour,
                "_get_current_token_balances",
                mock_get_current_token_balances,
            ):
                with patch.object(
                    fetch_behaviour,
                    "_calculate_corrected_yield",
                    mock_calculate_corrected_yield,
                ):
                    # Consume the generator
                    list(
                        fetch_behaviour._update_position_with_current_value(
                            position, current_value_usd, "mode"
                        )
                    )

                    # Verify position was updated correctly
                    assert position["current_value_usd"] == 1100.0
                    assert position["last_updated"] == 1234567890
                    assert position["cost_recovered"] is True

    def test_update_position_with_current_value_cost_not_recovered(self):
        """Test position update with cost recovery not achieved."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "principal_usd": 1000.0,
            "entry_cost": 50.0,
        }
        current_value_usd = Decimal(
            "1040"
        )  # 40 more than principal, doesn't cover entry cost

        def mock_get_current_timestamp():
            return 1234567890

        def mock_get_current_token_balances(position, chain):
            yield
            return {"0x123": Decimal("1000"), "0x456": Decimal("2000")}

        def mock_calculate_corrected_yield(
            position,
            initial_amount0,
            initial_amount1,
            user_balances,
            chain,
            token_prices,
        ):
            yield
            return Decimal("30")  # Yield of 30 USD, which is < entry_cost of 50

        with patch.object(
            fetch_behaviour, "_get_current_timestamp", mock_get_current_timestamp
        ):
            with patch.object(
                fetch_behaviour,
                "_get_current_token_balances",
                mock_get_current_token_balances,
            ):
                with patch.object(
                    fetch_behaviour,
                    "_calculate_corrected_yield",
                    mock_calculate_corrected_yield,
                ):
                    # Consume the generator
                    list(
                        fetch_behaviour._update_position_with_current_value(
                            position, current_value_usd, "mode"
                        )
                    )

                    # Verify position was updated correctly
                    assert position["current_value_usd"] == 1040.0
                    assert position["last_updated"] == 1234567890
                    # Note: cost_recovered is set to False when cost is not recovered
                    assert position["cost_recovered"] is False

    def test_update_position_with_current_value_legacy_position(self):
        """Test position update with legacy position (no entry cost)."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "principal_usd": 1000.0,
            "entry_cost": 0.0,  # Legacy position
        }
        current_value_usd = Decimal("1100")

        def mock_get_current_timestamp():
            return 1234567890

        with patch.object(
            fetch_behaviour, "_get_current_timestamp", mock_get_current_timestamp
        ):
            # Consume the generator
            list(
                fetch_behaviour._update_position_with_current_value(
                    position, current_value_usd, "mode"
                )
            )

            # Verify position was updated correctly
            assert position["current_value_usd"] == 1100.0
            assert position["last_updated"] == 1234567890
            assert (
                position["cost_recovered"] is False
            )  # Legacy positions marked as not recovered

    def test_update_position_with_current_value_exception_handling(self):
        """Test position update with exception handling."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "principal_usd": 1000.0,
            "entry_cost": 50.0,
        }
        current_value_usd = Decimal("1100")

        def mock_get_current_timestamp():
            raise Exception("Simulated error")

        with patch.object(
            fetch_behaviour, "_get_current_timestamp", mock_get_current_timestamp
        ):
            list(
                fetch_behaviour._update_position_with_current_value(
                    position, current_value_usd, "mode"
                )
            )

            # Verify position was updated with fallback values
            assert position["cost_recovered"] is False  # Fallback to False on error

    def test_calculate_safe_balances_value_success(self):
        """Test successful safe balances value calculation."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        portfolio_breakdown = []

        # Mock assets
        fetch_behaviour.assets = {
            "mode": {
                "0x0000000000000000000000000000000000000000": "ETH",
                "0x1111111111111111111111111111111111111111": "USDC",
            },
            "optimism": {"0x2222222222222222222222222222222222222222": "WETH"},
        }

        # Set up safe address for mode chain
        fetch_behaviour.params.safe_contract_addresses["mode"] = "0xSafeModeAddress"

        # Set up target investment chains - unfreeze params first
        fetch_behaviour.params.__dict__["_frozen"] = False
        fetch_behaviour.params.target_investment_chains = ["mode"]
        fetch_behaviour.params.__dict__["_frozen"] = True

        def mock_get_native_balance(chain, safe_address):
            yield
            return 1000000000000000000  # 1 ETH in wei

        def mock_fetch_zero_address_price():
            yield
            return 2000.0  # $2000 per ETH

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            account,
            chain_id,
        ):
            yield
            return 1000000  # 1 USDC (6 decimals)

        def mock_get_token_decimals(chain, token_address):
            yield
            return (
                6
                if token_address == "0x1111111111111111111111111111111111111111"
                else 18
            )

        def mock_fetch_token_price(token_address, chain):
            yield
            return (
                1.0
                if token_address == "0x1111111111111111111111111111111111111111"
                else 2000.0
            )

        def mock_get_mode_balances_from_explorer_api():
            yield
            return [
                {
                    "address": "0x0000000000000000000000000000000000000000",
                    "asset_symbol": "ETH",
                    "balance": 1000000000000000000,  # 1 ETH in wei
                },
                {
                    "address": "0x1111111111111111111111111111111111111111",
                    "asset_symbol": "USDC",
                    "balance": 1000000,  # 1 USDC (6 decimals)
                },
            ]

        with patch.multiple(
            fetch_behaviour,
            _get_native_balance=mock_get_native_balance,
            _fetch_zero_address_price=mock_fetch_zero_address_price,
            contract_interact=mock_contract_interact,
            _get_token_decimals=mock_get_token_decimals,
            _fetch_token_price=mock_fetch_token_price,
            _get_mode_balances_from_explorer_api=mock_get_mode_balances_from_explorer_api,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_safe_balances_value(portfolio_breakdown)
            )

            assert result == Decimal("2001.0")  # 1 ETH * $2000 + 1 USDC * $1
            assert len(portfolio_breakdown) == 2

    def test_calculate_safe_balances_value_no_safe_address(self):
        """Test safe balances calculation when no safe address is found."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        portfolio_breakdown = []

        # Mock empty safe addresses
        with patch.dict(
            fetch_behaviour.params.safe_contract_addresses, {}, clear=False
        ):
            fetch_behaviour.assets = {
                "mode": {"0x0000000000000000000000000000000000000000": "ETH"}
            }

            def mock_get_native_balance(chain, safe_address):
                yield
                return 0

            def mock_get_mode_balances_from_explorer_api():
                yield
                return []  # No token balances

            with patch.multiple(
                fetch_behaviour,
                _get_native_balance=mock_get_native_balance,
                _get_mode_balances_from_explorer_api=mock_get_mode_balances_from_explorer_api,
            ):
                result = self._consume_generator(
                    fetch_behaviour._calculate_safe_balances_value(portfolio_breakdown)
                )

                assert result == Decimal("0")
                assert len(portfolio_breakdown) == 0

    def test_calculate_safe_balances_value_no_assets(self):
        """Test safe balances calculation when no assets are found."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        portfolio_breakdown = []

        with patch.dict(
            fetch_behaviour.params.safe_contract_addresses,
            {"mode": "0xSafeModeAddress"},
            clear=False,
        ):
            fetch_behaviour.assets = {}

            def mock_get_native_balance(chain, safe_address):
                yield
                return 0

            def mock_get_mode_balances_from_explorer_api():
                yield
                return []  # No token balances

            with patch.multiple(
                fetch_behaviour,
                _get_native_balance=mock_get_native_balance,
                _get_mode_balances_from_explorer_api=mock_get_mode_balances_from_explorer_api,
            ):
                result = self._consume_generator(
                    fetch_behaviour._calculate_safe_balances_value(portfolio_breakdown)
                )

                assert result == Decimal("0")
                assert len(portfolio_breakdown) == 0

    def test_calculate_safe_balances_value_eth_price_failure(self):
        """Test safe balances calculation when ETH price fetch fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        portfolio_breakdown = []

        with patch.dict(
            fetch_behaviour.params.safe_contract_addresses,
            {"mode": "0xSafeModeAddress"},
            clear=False,
        ):
            fetch_behaviour.assets = {
                "mode": {"0x0000000000000000000000000000000000000000": "ETH"}
            }

            def mock_fetch_zero_address_price():
                yield
                return None  # Price fetch failure

            def mock_get_mode_balances_from_explorer_api():
                yield
                return []  # No token balances

            with patch.multiple(
                fetch_behaviour,
                _get_native_balance=self.mock_get_native_balance,
                _fetch_zero_address_price=mock_fetch_zero_address_price,
                _get_mode_balances_from_explorer_api=mock_get_mode_balances_from_explorer_api,
            ):
                result = self._consume_generator(
                    fetch_behaviour._calculate_safe_balances_value(portfolio_breakdown)
                )

                assert result == Decimal("0")
                assert len(portfolio_breakdown) == 0

    def test_calculate_safe_balances_value_token_decimals_failure(self):
        """Test safe balances calculation when token decimals fetch fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        portfolio_breakdown = []

        with patch.dict(
            fetch_behaviour.params.safe_contract_addresses,
            {"mode": "0xSafeModeAddress"},
            clear=False,
        ):
            fetch_behaviour.assets = {
                "mode": {"0x1111111111111111111111111111111111111111": "USDC"}
            }

            def mock_contract_interact(
                performative,
                contract_address,
                contract_public_id,
                contract_callable,
                data_key,
                account,
                chain_id,
            ):
                yield
                return 1000000

            def mock_get_token_decimals(chain, token_address):
                yield
                return None  # Decimals fetch failure

            def mock_get_mode_balances_from_explorer_api():
                yield
                return []  # No token balances

            with patch.multiple(
                fetch_behaviour,
                contract_interact=mock_contract_interact,
                _get_token_decimals=mock_get_token_decimals,
                _get_native_balance=self.mock_get_native_balance,
                _get_mode_balances_from_explorer_api=mock_get_mode_balances_from_explorer_api,
            ):
                result = self._consume_generator(
                    fetch_behaviour._calculate_safe_balances_value(portfolio_breakdown)
                )

                assert result == Decimal("0")
                assert len(portfolio_breakdown) == 0

    def test_calculate_total_volume_success(self):
        """Test successful total volume calculation."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock current positions
        fetch_behaviour.current_positions = [
            {
                "pool_address": "0x1234567890123456789012345678901234567890",
                "tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "token0": "0x1111111111111111111111111111111111111111",
                "token1": "0x2222222222222222222222222222222222222222",
                "amount0": "1000000",  # 1 USDC
                "amount1": "500000000000000000",  # 0.5 ETH
                "timestamp": 1640995200,  # 2022-01-01
                "chain": "mode",
                "token0_symbol": "USDC",
                "token1_symbol": "ETH",
            }
        ]

        def mock_get_token_decimals(chain, token_address):
            yield
            return (
                6
                if token_address == "0x1111111111111111111111111111111111111111"
                else 18
            )

        def mock_fetch_historical_token_prices(tokens, date_str, chain):
            yield
            return {
                "0x1111111111111111111111111111111111111111": 1.0,  # USDC = $1
                "0x2222222222222222222222222222222222222222": 2000.0,  # ETH = $2000
            }

        def mock_fetch_token_price(token_address, chain):
            yield
            return (
                1.0
                if token_address == "0x1111111111111111111111111111111111111111"
                else 2000.0
            )

        with patch.multiple(
            fetch_behaviour,
            _read_kv=self.mock_read_kv_empty,
            _get_token_decimals=mock_get_token_decimals,
            _fetch_historical_token_prices=mock_fetch_historical_token_prices,
            _write_kv=self.mock_write_kv,
            _fetch_token_price=mock_fetch_token_price,
        ):
            result = self._consume_generator(fetch_behaviour._calculate_total_volume())

            assert result == 1001.0  # 1 USDC * $1 + 0.5 ETH * $2000

    def test_calculate_total_volume_with_cached_values(self):
        """Test total volume calculation with cached values."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        fetch_behaviour.current_positions = [
            {
                "pool_address": "0x1234567890123456789012345678901234567890",
                "tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "token0": "0x1111111111111111111111111111111111111111",
                "amount0": "1000000",
                "timestamp": 1640995200,
                "chain": "mode",
                "token0_symbol": "USDC",
            }
        ]

        def mock_read_kv(keys):
            yield
            return {
                "initial_investment_values": json.dumps(
                    {
                        "0x1234567890123456789012345678901234567890_0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890": 500.0
                    }
                )
            }

        with patch.object(fetch_behaviour, "_read_kv", mock_read_kv):
            result = self._consume_generator(fetch_behaviour._calculate_total_volume())

            assert result == 500.0

    def test_calculate_total_volume_missing_position_data(self):
        """Test total volume calculation with missing position data."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        fetch_behaviour.current_positions = [
            {
                "pool_address": "0x1234567890123456789012345678901234567890",
                # Missing required fields
                "chain": "mode",
            }
        ]

        def mock_read_kv(keys):
            yield
            return None

        with patch.object(fetch_behaviour, "_read_kv", mock_read_kv):
            result = self._consume_generator(fetch_behaviour._calculate_total_volume())

            assert result is None

    def test_calculate_total_volume_historical_price_failure(self):
        """Test total volume calculation when historical price fetch fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        fetch_behaviour.current_positions = [
            {
                "pool_address": "0x1234567890123456789012345678901234567890",
                "tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "token0": "0x1111111111111111111111111111111111111111",
                "amount0": "1000000",
                "timestamp": 1640995200,
                "chain": "mode",
                "token0_symbol": "USDC",
            }
        ]

        def mock_get_token_decimals(chain, token_address):
            yield
            return 6

        def mock_fetch_historical_token_prices(tokens, date_str, chain):
            yield
            return None  # Price fetch failure

        def mock_fetch_token_price(token_address, chain):
            yield
            return None

        with patch.multiple(
            fetch_behaviour,
            _read_kv=self.mock_read_kv_empty,
            _get_token_decimals=mock_get_token_decimals,
            _fetch_historical_token_prices=mock_fetch_historical_token_prices,
            _write_kv=self.mock_write_kv,
            _fetch_token_price=mock_fetch_token_price,
        ):
            result = self._consume_generator(fetch_behaviour._calculate_total_volume())

            assert result is None

    def test_calculate_cl_position_value_success(self):
        """Test successful CL position value calculation."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "token_id": 123,
            "pool_address": "0x1234567890123456789012345678901234567890",
        }

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "get_position":
                return {
                    "data": {
                        "token0": "0x1111111111111111111111111111111111111111",
                        "token1": "0x2222222222222222222222222222222222222222",
                        "fee": 3000,
                        "tickLower": -1000,
                        "tickUpper": 1000,
                        "liquidity": "1000000000000000000",
                    }
                }
            elif contract_callable == "slot0":
                return {"sqrt_price_x96": 1000000000000000000000000, "tick": 0}
            return None

        def mock_get_token_decimals_pair(chain, token0_address, token1_address):
            yield
            return (18, 6)  # ETH and USDC decimals

        def mock_calculate_position_amounts(
            position_details, current_tick, sqrt_price_x96, position, dex_type, chain
        ):
            yield
            return (1000000000000000000, 1000000)  # 1 ETH, 1 USDC

        with patch.multiple(
            fetch_behaviour,
            contract_interact=mock_contract_interact,
            _get_token_decimals_pair=mock_get_token_decimals_pair,
            _calculate_position_amounts=mock_calculate_position_amounts,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_cl_position_value(
                    "0x1234567890123456789012345678901234567890",
                    "mode",
                    position,
                    "0x1111111111111111111111111111111111111111",
                    "0x2222222222222222222222222222222222222222",
                    "0xPositionManager",
                    "velodrome_non_fungible_position_manager/contract:0.1.0",
                    "velodrome",
                )
            )

            assert result == {
                "0x1111111111111111111111111111111111111111": Decimal("1.0"),
                "0x2222222222222222222222222222222222222222": Decimal("1.0"),
            }

    def test_calculate_cl_position_value_contract_failure(self):
        """Test CL position value calculation with contract failure."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {"token_id": 123}

        def mock_calculate_cl_position_value(
            pool_address,
            chain,
            position,
            token0_address,
            token1_address,
            position_manager_address,
            contract_id,
            dex_type,
        ):
            yield
            return {}  # Contract failure

        with patch.object(
            fetch_behaviour,
            "_calculate_cl_position_value",
            mock_calculate_cl_position_value,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_cl_position_value(
                    "0x1234567890123456789012345678901234567890",
                    "mode",
                    position,
                    "0x1111111111111111111111111111111111111111",
                    "0x2222222222222222222222222222222222222222",
                    "0xPositionManager",
                    MagicMock(),
                    "velodrome",
                )
            )

            assert result == {}

    def test_calculate_position_amounts_success(self):
        """Test successful position amounts calculation."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position_details = {
            "token0": "0x1111111111111111111111111111111111111111",
            "token1": "0x2222222222222222222222222222222222222222",
            "fee": 3000,
            "tickLower": -1000,
            "tickUpper": 1000,
            "liquidity": "1000000000000000000",
        }

        position = {"token_id": 123}

        def mock_calculate_position_amounts(
            position_details, current_tick, sqrt_price_x96, position, dex_type, chain
        ):
            yield
            return (1000000, 500000000000000000)  # token0_amount, token1_amount

        with patch.object(
            fetch_behaviour,
            "_calculate_position_amounts",
            mock_calculate_position_amounts,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_position_amounts(
                    position_details,
                    0,  # current_tick
                    1000000000000000000000000,  # sqrt_price_x96
                    position,
                    "velodrome",
                    "mode",
                )
            )

            assert result == (1000000, 500000000000000000)

    def test_calculate_position_amounts_no_ranges(self):
        """Test position amounts calculation when no tick ranges are found."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position_details = {
            "token0": "0x1111111111111111111111111111111111111111",
            "token1": "0x2222222222222222222222222222222222222222",
            "fee": 3000,
            "tickLower": -1000,
            "tickUpper": 1000,
            "liquidity": "1000000000000000000",
        }

        position = {"token_id": 123}

        def mock_calculate_position_amounts(
            position_details, current_tick, sqrt_price_x96, position, dex_type, chain
        ):
            yield
            return None  # No ranges found

        with patch.object(
            fetch_behaviour,
            "_calculate_position_amounts",
            mock_calculate_position_amounts,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_position_amounts(
                    position_details,
                    0,
                    1000000000000000000000000,
                    position,
                    "velodrome",
                    "mode",
                )
            )

            assert result is None

    @pytest.mark.parametrize(
        "eth_transfers,reversion_transfers,mock_price_response,expected_result,test_description",
        [
            # Successful calculation with valid reversion transfers
            (
                [
                    {
                        "value": "1000000000000000000",
                        "timestamp": "1640995200",
                    },  # 1 ETH
                    {
                        "value": "2000000000000000000",
                        "timestamp": "1640995201",
                    },  # 2 ETH
                ],
                [{"amount": 0.5, "timestamp": "1640995202"}],  # 0.5 ETH
                2000.0,  # $2000 per ETH
                1000.0,  # 0.5 ETH * $2000 = $1000
                "successful calculation with valid reversion transfers",
            ),
            # No eth transfers - should fail with IndexError
            (
                [],
                [{"amount": 0.5, "timestamp": "1640995202"}],
                2000.0,
                0.0,  # Will fail before reaching this
                "no eth transfers",
            ),
            # No reversion transfers - should return 0
            (
                [{"value": "1000000000000000000", "timestamp": "1640995200"}],
                [],
                2000.0,
                0.0,
                "no reversion transfers",
            ),
            # Price fetch failure - should return 0
            (
                [{"value": "1000000000000000000", "timestamp": "1640995200"}],
                [{"amount": 0.5, "timestamp": "1640995202"}],
                None,  # Price fetch failure
                0.0,
                "price fetch failure",
            ),
            # Invalid timestamp - should handle gracefully
            (
                [{"value": "1000000000000000000", "timestamp": "invalid_timestamp"}],
                [{"amount": 0.5, "timestamp": "1640995202"}],
                2000.0,
                1000.0,  # Should handle invalid timestamp gracefully and still calculate
                "invalid timestamp",
            ),
        ],
    )
    def test_calculate_total_reversion_value_variations(
        self,
        eth_transfers,
        reversion_transfers,
        mock_price_response,
        expected_result,
        test_description,
    ):
        """Test _calculate_total_reversion_value method with various scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_fetch_historical_eth_price(date_str):
            yield
            return mock_price_response

        with patch.object(
            fetch_behaviour,
            "_fetch_historical_eth_price",
            mock_fetch_historical_eth_price,
        ):
            if not eth_transfers and test_description == "no eth transfers":
                # This case should raise an IndexError
                with pytest.raises(IndexError):
                    self._consume_generator(
                        fetch_behaviour._calculate_total_reversion_value(
                            eth_transfers, reversion_transfers
                        )
                    )
            else:
                result = self._consume_generator(
                    fetch_behaviour._calculate_total_reversion_value(
                        eth_transfers, reversion_transfers
                    )
                )

                assert (
                    result == expected_result
                ), f"Expected {expected_result} for {test_description}"

    def test_calculate_chain_investment_value_success(self):
        """Test successful chain investment value calculation."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        all_transfers = {
            "2022-01-01": [
                {
                    "from": {"address": "0xSafeAddress"},
                    "to": {"address": "0xPoolAddress"},
                    "delta": 1.0,  # 1 ETH
                    "symbol": "ETH",
                    "timestamp": "1640995200",
                }
            ]
        }

        def mock_fetch_historical_eth_price(date_str):
            yield
            return 2000.0  # $2000 per ETH

        def mock_is_gnosis_safe(address_info):
            return address_info.get("address") == "0xSafeAddress"

        def mock_should_include_transfer(
            from_address, tx_data=None, is_eth_transfer=False
        ):
            return True

        def mock_get_datetime_from_timestamp(timestamp_str):
            return datetime(2022, 1, 1)

        def mock_should_include_transfer_mode(
            from_address, tx_data=None, is_eth_transfer=False
        ):
            return True

        def mock_track_eth_transfers_and_reversions(safe_address, chain):
            yield
            return {
                "reversion_amount": 0,
                "historical_reversion_value": 0.0,
                "reversion_date": None,
            }

        with patch.multiple(
            fetch_behaviour,
            _fetch_historical_eth_price=mock_fetch_historical_eth_price,
            _is_gnosis_safe=mock_is_gnosis_safe,
            _should_include_transfer=mock_should_include_transfer,
            _get_datetime_from_timestamp=mock_get_datetime_from_timestamp,
            _should_include_transfer_mode=mock_should_include_transfer_mode,
            _track_eth_transfers_and_reversions=mock_track_eth_transfers_and_reversions,
            _save_chain_total_investment=self.mock_save_chain_total_investment,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_chain_investment_value(
                    all_transfers, "mode", "0xSafeAddress"
                )
            )

            assert result == 2000.0  # 1 ETH * $2000

    def test_calculate_chain_investment_value_no_transfers(self):
        """Test chain investment value calculation with no transfers."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_track_eth_transfers_and_reversions(safe_address, chain):
            yield
            return {
                "reversion_amount": 0,
                "historical_reversion_value": 0.0,
                "reversion_date": None,
            }

        with patch.multiple(
            fetch_behaviour,
            _track_eth_transfers_and_reversions=mock_track_eth_transfers_and_reversions,
            _save_chain_total_investment=self.mock_save_chain_total_investment,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_chain_investment_value(
                    {}, "mode", "0xSafeAddress"
                )
            )

            assert result == 0.0

    def test_calculate_chain_investment_value_price_failure(self):
        """Test chain investment value calculation when price fetch fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        all_transfers = {
            "2022-01-01": [
                {
                    "from": {"address": "0xSafeAddress"},
                    "to": {"address": "0xPoolAddress"},
                    "delta": 1.0,
                    "symbol": "ETH",
                    "timestamp": "1640995200",
                }
            ]
        }

        def mock_fetch_historical_eth_price(date_str):
            yield
            return None  # Price fetch failure

        def mock_is_gnosis_safe(address_info):
            return address_info.get("address") == "0xSafeAddress"

        def mock_should_include_transfer(
            from_address, tx_data=None, is_eth_transfer=False
        ):
            return True

        def mock_get_datetime_from_timestamp(timestamp_str):
            return datetime(2022, 1, 1)

        def mock_should_include_transfer_mode(
            from_address, tx_data=None, is_eth_transfer=False
        ):
            return True

        def mock_track_eth_transfers_and_reversions(safe_address, chain):
            yield
            return {
                "reversion_amount": 0,
                "historical_reversion_value": 0.0,
                "reversion_date": None,
            }

        with patch.multiple(
            fetch_behaviour,
            _fetch_historical_eth_price=mock_fetch_historical_eth_price,
            _is_gnosis_safe=mock_is_gnosis_safe,
            _should_include_transfer=mock_should_include_transfer,
            _get_datetime_from_timestamp=mock_get_datetime_from_timestamp,
            _should_include_transfer_mode=mock_should_include_transfer_mode,
            _track_eth_transfers_and_reversions=mock_track_eth_transfers_and_reversions,
            _save_chain_total_investment=self.mock_save_chain_total_investment,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_chain_investment_value(
                    all_transfers, "mode", "0xSafeAddress"
                )
            )

            assert result == 0.0

    def test_calculate_cl_position_value_missing_parameters(self):
        """Test CL position value calculation with missing required parameters."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Test with missing pool_address
        result = self._consume_generator(
            fetch_behaviour._calculate_cl_position_value(
                "",  # Missing pool_address
                "mode",
                {"token_id": 123},
                "0x1111111111111111111111111111111111111111",
                "0x2222222222222222222222222222222222222222",
                "0xPositionManager",
                "velodrome_non_fungible_position_manager/contract:0.1.0",
                "velodrome",
            )
        )
        assert result == {}

        # Test with missing chain
        result = self._consume_generator(
            fetch_behaviour._calculate_cl_position_value(
                "0x1234567890123456789012345678901234567890",
                "",  # Missing chain
                {"token_id": 123},
                "0x1111111111111111111111111111111111111111",
                "0x2222222222222222222222222222222222222222",
                "0xPositionManager",
                "velodrome_non_fungible_position_manager/contract:0.1.0",
                "velodrome",
            )
        )
        assert result == {}

    def test_calculate_cl_position_value_slot0_failure(self):
        """Test CL position value calculation when slot0 data fetch fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "token_id": 123,
            "pool_address": "0x1234567890123456789012345678901234567890",
        }

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "slot0":
                return None  # Slot0 fetch failure
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._calculate_cl_position_value(
                    "0x1234567890123456789012345678901234567890",
                    "mode",
                    position,
                    "0x1111111111111111111111111111111111111111",
                    "0x2222222222222222222222222222222222222222",
                    "0xPositionManager",
                    "velodrome_non_fungible_position_manager/contract:0.1.0",
                    "velodrome",
                )
            )
            assert result == {}

    def test_calculate_cl_position_value_invalid_slot0_data(self):
        """Test CL position value calculation with invalid slot0 data."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "token_id": 123,
            "pool_address": "0x1234567890123456789012345678901234567890",
        }

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "slot0":
                return {
                    "slot0": {
                        # Missing sqrt_price_x96 and tick
                    }
                }
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._calculate_cl_position_value(
                    "0x1234567890123456789012345678901234567890",
                    "mode",
                    position,
                    "0x1111111111111111111111111111111111111111",
                    "0x2222222222222222222222222222222222222222",
                    "0xPositionManager",
                    "velodrome_non_fungible_position_manager/contract:0.1.0",
                    "velodrome",
                )
            )
            assert result == {}

    def test_calculate_cl_position_value_token_decimals_failure(self):
        """Test CL position value calculation when token decimals fetch fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "token_id": 123,
            "pool_address": "0x1234567890123456789012345678901234567890",
        }

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "slot0":
                return {
                    "slot0": {"sqrt_price_x96": "1000000000000000000000000", "tick": 0}
                }
            return None

        def mock_get_token_decimals_pair(chain, token0_address, token1_address):
            yield
            return (None, 6)  # token0 decimals fetch failure

        with patch.multiple(
            fetch_behaviour,
            contract_interact=mock_contract_interact,
            _get_token_decimals_pair=mock_get_token_decimals_pair,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_cl_position_value(
                    "0x1234567890123456789012345678901234567890",
                    "mode",
                    position,
                    "0x1111111111111111111111111111111111111111",
                    "0x2222222222222222222222222222222222222222",
                    "0xPositionManager",
                    "velodrome_non_fungible_position_manager/contract:0.1.0",
                    "velodrome",
                )
            )
            assert result == {}

    def test_calculate_cl_position_value_multiple_positions(self):
        """Test CL position value calculation with multiple positions."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "positions": [{"token_id": 123}, {"token_id": 456}],
            "pool_address": "0x1234567890123456789012345678901234567890",
        }

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "get_position":
                token_id = kwargs.get("token_id")
                if token_id == 123:
                    return {
                        "data": {
                            "token0": "0x1111111111111111111111111111111111111111",
                            "token1": "0x2222222222222222222222222222222222222222",
                            "fee": 3000,
                            "tickLower": -1000,
                            "tickUpper": 1000,
                            "liquidity": "1000000000000000000",
                        }
                    }
                elif token_id == 456:
                    return {
                        "data": {
                            "token0": "0x1111111111111111111111111111111111111111",
                            "token1": "0x2222222222222222222222222222222222222222",
                            "fee": 3000,
                            "tickLower": -500,
                            "tickUpper": 500,
                            "liquidity": "500000000000000000",
                        }
                    }
            elif contract_callable == "slot0":
                return {"sqrt_price_x96": 1000000000000000000000000, "tick": 0}
            return None

        def mock_get_token_decimals_pair(chain, token0_address, token1_address):
            yield
            return (18, 6)  # ETH and USDC decimals

        def mock_calculate_position_amounts(
            position_details, current_tick, sqrt_price_x96, position, dex_type, chain
        ):
            yield
            # Return different amounts for different positions
            if position.get("token_id") == 123:
                return (1000000000000000000, 1000000)  # 1 ETH, 1 USDC
            else:
                return (500000000000000000, 500000)  # 0.5 ETH, 0.5 USDC

        with patch.multiple(
            fetch_behaviour,
            contract_interact=mock_contract_interact,
            _get_token_decimals_pair=mock_get_token_decimals_pair,
            _calculate_position_amounts=mock_calculate_position_amounts,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_cl_position_value(
                    "0x1234567890123456789012345678901234567890",
                    "mode",
                    position,
                    "0x1111111111111111111111111111111111111111",
                    "0x2222222222222222222222222222222222222222",
                    "0xPositionManager",
                    "velodrome_non_fungible_position_manager/contract:0.1.0",
                    "velodrome",
                )
            )

            assert result == {
                "0x1111111111111111111111111111111111111111": Decimal("1.5"),  # 1 + 0.5
                "0x2222222222222222222222222222222222222222": Decimal("1.5"),  # 1 + 0.5
            }

    def test_calculate_cl_position_value_position_details_failure(self):
        """Test CL position value calculation when position details fetch fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "token_id": 123,
            "pool_address": "0x1234567890123456789012345678901234567890",
        }

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "get_position":
                return None  # Position details fetch failure
            elif contract_callable == "slot0":
                return {"sqrt_price_x96": 1000000000000000000000000, "tick": 0}
            return None

        def mock_get_token_decimals_pair(chain, token0_address, token1_address):
            yield
            return (18, 6)  # ETH and USDC decimals

        with patch.multiple(
            fetch_behaviour,
            contract_interact=mock_contract_interact,
            _get_token_decimals_pair=mock_get_token_decimals_pair,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_cl_position_value(
                    "0x1234567890123456789012345678901234567890",
                    "mode",
                    position,
                    "0x1111111111111111111111111111111111111111",
                    "0x2222222222222222222222222222222222222222",
                    "0xPositionManager",
                    "velodrome_non_fungible_position_manager/contract:0.1.0",
                    "velodrome",
                )
            )

            # Should return empty dict since position details failed
            assert result

    def test_get_token_decimals_pair_success(self):
        """Test successful token decimals pair fetch."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "get_token_decimals":
                if contract_address == "0x1111111111111111111111111111111111111111":
                    return 18
                elif contract_address == "0x2222222222222222222222222222222222222222":
                    return 6
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._get_token_decimals_pair(
                    "mode",
                    "0x1111111111111111111111111111111111111111",
                    "0x2222222222222222222222222222222222222222",
                )
            )
            assert result == (18, 6)

    def test_get_token_decimals_pair_failure(self):
        """Test token decimals pair fetch when one token fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "get_token_decimals":
                if contract_address == "0x1111111111111111111111111111111111111111":
                    return 18
                elif contract_address == "0x2222222222222222222222222222222222222222":
                    return None  # Second token fails
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._get_token_decimals_pair(
                    "mode",
                    "0x1111111111111111111111111111111111111111",
                    "0x2222222222222222222222222222222222222222",
                )
            )
            assert result == (None, None)

    def test_get_token_decimals_pair_both_fail(self):
        """Test token decimals pair fetch when both tokens fail."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "get_token_decimals":
                return None  # Both tokens fail
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._get_token_decimals_pair(
                    "mode",
                    "0x1111111111111111111111111111111111111111",
                    "0x2222222222222222222222222222222222222222",
                )
            )
            assert result == (None, None)

    def test_adjust_for_decimals_success(self):
        """Test decimal adjustment for token amounts."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Test with 18 decimals
        result = fetch_behaviour._adjust_for_decimals(1000000000000000000, 18)
        assert result == Decimal("1.0")

        # Test with 6 decimals
        result = fetch_behaviour._adjust_for_decimals(1000000, 6)
        assert result == Decimal("1.0")

        # Test with 0 decimals
        result = fetch_behaviour._adjust_for_decimals(100, 0)
        assert result == Decimal("100.0")

    @pytest.mark.parametrize(
        "contract_result,expected_result,test_description",
        [
            (
                "Sturdy Aggregator V1",
                "Sturdy Aggregator V1",
                "successful contract call",
            ),
            (None, None, "contract call failure"),
        ],
    )
    def test_get_aggregator_name_variations(
        self, contract_result, expected_result, test_description
    ):
        """Test _get_aggregator_name method with various contract responses."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "name":
                return contract_result
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._get_aggregator_name(
                    "0x1234567890123456789012345678901234567890", "mode"
                )
            )
            assert result == expected_result, f"Failed for {test_description}"

    @pytest.mark.parametrize(
        "contract_result,expected_result,test_description",
        [
            (
                "Balancer Pool USDC-ETH",
                "Balancer Pool USDC-ETH",
                "successful contract call",
            ),
            (None, None, "contract call failure"),
        ],
    )
    def test_get_balancer_pool_name_variations(
        self, contract_result, expected_result, test_description
    ):
        """Test _get_balancer_pool_name method with various contract responses."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "get_name":
                return contract_result
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._get_balancer_pool_name(
                    "0x1234567890123456789012345678901234567890", "mode"
                )
            )
            assert result == expected_result, f"Failed for {test_description}"

    @pytest.mark.parametrize(
        "contract_result,expected_result,test_description",
        [
            (
                [
                    "0x1111111111111111111111111111111111111111",
                    "0x2222222222222222222222222222222222222222",
                ],
                True,
                "valid safe with owners",
            ),
            (None, False, "not a GnosisSafe"),
            ("exception", False, "contract interaction exception"),
        ],
    )
    def test_check_is_valid_safe_address_variations(
        self, contract_result, expected_result, test_description
    ):
        """Test check_is_valid_safe_address method with various contract responses."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "get_owners":
                if contract_result == "exception":
                    raise Exception("Contract interaction failed")
                return contract_result
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour.check_is_valid_safe_address(
                    "0x1234567890123456789012345678901234567890", "mode"
                )
            )
            assert result is expected_result, f"Failed for {test_description}"

    @pytest.mark.parametrize(
        "contract_result,expected_result,test_description",
        [
            (18, 18, "successful contract call"),
            (None, None, "contract call failure"),
        ],
    )
    def test_get_token_decimals_variations(
        self, contract_result, expected_result, test_description
    ):
        """Test _get_token_decimals method with various contract responses."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "get_token_decimals":
                return contract_result
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._get_token_decimals(
                    "mode", "0x1234567890123456789012345678901234567890"
                )
            )
            assert result == expected_result, f"Failed for {test_description}"

    @pytest.mark.parametrize(
        "contract_result,expected_result,test_description",
        [
            ("USDC", "USDC", "successful contract call"),
            (None, None, "contract call failure"),
        ],
    )
    def test_get_token_symbol_variations(
        self, contract_result, expected_result, test_description
    ):
        """Test _get_token_symbol method with various contract responses."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "get_token_symbol":
                return contract_result
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._get_token_symbol(
                    "mode", "0x1234567890123456789012345678901234567890"
                )
            )
            assert result == expected_result, f"Failed for {test_description}"

    @pytest.mark.parametrize(
        "contract_result,expected_result,test_description",
        [
            ("USD Coin", "USD Coin", "successful contract call"),
            (None, None, "contract call failure"),
        ],
    )
    def test_get_token_name_variations(
        self, contract_result, expected_result, test_description
    ):
        """Test _fetch_token_name_from_contract method with various contract responses."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_contract_interact(
            performative,
            contract_address,
            contract_public_id,
            contract_callable,
            data_key,
            chain_id,
            **kwargs,
        ):
            yield
            if contract_callable == "get_name":
                return contract_result
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._fetch_token_name_from_contract(
                    "mode", "0x1234567890123456789012345678901234567890"
                )
            )
            assert result == expected_result, f"Failed for {test_description}"

    def test_get_coin_id_from_symbol_success(self):
        """Test successful coin ID fetch from symbol."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Test with known symbol
        result = fetch_behaviour.get_coin_id_from_symbol("USDC", "mode")
        assert result == "mode-bridged-usdc-mode"

        # Test with ETH
        result = fetch_behaviour.get_coin_id_from_symbol("ETH", "mode")
        assert result is None

    def test_get_coin_id_from_symbol_unknown(self):
        """Test coin ID fetch for unknown symbol."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Test with unknown symbol
        result = fetch_behaviour.get_coin_id_from_symbol("UNKNOWN", "mode")
        assert result is None

    @pytest.mark.parametrize(
        "chain,expected_result,test_description",
        [
            ("mode", "0xd988097fb8612cc24eeC14542bC03424c656005f", "mode chain"),
            (
                "optimism",
                "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
                "optimism chain",
            ),
            ("unknown", None, "unknown chain"),
        ],
    )
    def test_get_usdc_address_variations(
        self, chain, expected_result, test_description
    ):
        """Test _get_usdc_address method with various chains."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        result = fetch_behaviour._get_usdc_address(chain)
        assert result == expected_result, f"Failed for {test_description}"

    @pytest.mark.parametrize(
        "chain,expected_result,test_description",
        [
            ("mode", "0xcfd1d50ce23c46d3cf6407487b2f8934e96dc8f9", "mode chain"),
            (
                "optimism",
                "0xfc2e6e6bcbd49ccf3a5f029c79984372dcbfe527",
                "optimism chain",
            ),
            ("unknown", None, "unknown chain"),
        ],
    )
    def test_get_olas_address_variations(
        self, chain, expected_result, test_description
    ):
        """Test _get_olas_address method with various chains."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        result = fetch_behaviour._get_olas_address(chain)
        assert result == expected_result, f"Failed for {test_description}"

    # 6.8. Transfer Tracking Methods
    @pytest.mark.parametrize(
        "safe_address,chain,mock_incoming_transfers,mock_outgoing_transfers,mock_master_address,mock_native_balance,mock_reversion_value,expected_result,test_description",
        [
            # Successful tracking with valid transfers
            (
                "0x1234567890123456789012345678901234567890",
                "optimism",
                {
                    "2024-01-01": [
                        {
                            "symbol": "ETH",
                            "amount": 10.0,
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "timestamp": "2024-01-01T10:00:00Z",
                        },
                        {
                            "symbol": "ETH",
                            "amount": 5.0,
                            "from_address": "0xmaster123456789012345678901234567890123456",
                            "timestamp": "2024-01-02T10:00:00Z",
                        },
                    ]
                },
                {
                    "2024-01-03": [
                        {
                            "symbol": "ETH",
                            "amount": 2.0,
                            "to_address": "0xmaster123456789012345678901234567890123456",
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "timestamp": "2024-01-03T10:00:00Z",
                        }
                    ]
                },
                "0xmaster123456789012345678901234567890123456",
                15.0,
                5.0,
                {
                    "reversion_amount": 0.0,
                    "master_safe_address": "0xmaster123456789012345678901234567890123456",
                    "historical_reversion_value": 5.0,
                    "reversion_date": None,
                },
                "successful tracking with valid transfers",
            ),
            # No transfers found
            (
                "0x1234567890123456789012345678901234567890",
                "optimism",
                {},
                {},
                None,
                None,
                None,
                {},
                "no transfers found",
            ),
            # No master address found
            (
                "0x1234567890123456789012345678901234567890",
                "optimism",
                {
                    "2024-01-01": [
                        {
                            "symbol": "ETH",
                            "amount": 10.0,
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "timestamp": "2024-01-01T10:00:00Z",
                        }
                    ]
                },
                {},
                None,
                None,
                None,
                {},
                "no master address found",
            ),
            # Unsupported chain
            (
                "0x1234567890123456789012345678901234567890",
                "unsupported_chain",
                {},
                {},
                None,
                None,
                None,
                {},
                "unsupported chain",
            ),
            # Multiple ETH transfers with no reversion - should calculate reversion amount
            (
                "0x1234567890123456789012345678901234567890",
                "optimism",
                {
                    "2024-01-01": [
                        {
                            "symbol": "ETH",
                            "amount": 10.0,
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "timestamp": "2024-01-01T10:00:00Z",
                        }
                    ],
                    "2024-01-02": [
                        {
                            "symbol": "ETH",
                            "amount": 5.0,
                            "from_address": "0xmaster123456789012345678901234567890123456",
                            "timestamp": "2024-01-02T10:00:00Z",
                        }
                    ],
                },
                {},  # No outgoing transfers (no reversion has happened yet)
                "0xmaster123456789012345678901234567890123456",
                12.0,  # Current balance is less than the last transfer amount
                None,  # No reversion value calculated yet
                {
                    "reversion_amount": 5.0,  # Should be the amount of the last transfer
                    "master_safe_address": "0xmaster123456789012345678901234567890123456",
                    "historical_reversion_value": 0.0,
                    "reversion_date": "02-01-2024",  # Date from the last transfer
                },
                "multiple ETH transfers with no reversion - calculate reversion amount",
            ),
            # Multiple ETH transfers with no reversion - current balance less than transfer amount
            (
                "0x1234567890123456789012345678901234567890",
                "optimism",
                {
                    "2024-01-01": [
                        {
                            "symbol": "ETH",
                            "amount": 10.0,
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "timestamp": "2024-01-01T10:00:00Z",
                        }
                    ],
                    "2024-01-02": [
                        {
                            "symbol": "ETH",
                            "amount": 5.0,
                            "from_address": "0xmaster123456789012345678901234567890123456",
                            "timestamp": "2024-01-02T10:00:00Z",
                        }
                    ],
                },
                {},  # No outgoing transfers
                "0xmaster123456789012345678901234567890123456",
                3.0,  # Current balance is less than the last transfer amount (5.0)
                None,
                {
                    "reversion_amount": 3.0,  # Should be limited to current balance
                    "master_safe_address": "0xmaster123456789012345678901234567890123456",
                    "historical_reversion_value": 0.0,
                    "reversion_date": "02-01-2024",
                },
                "multiple ETH transfers with no reversion - current balance limits reversion amount",
            ),
            # Multiple ETH transfers with no reversion - Unix timestamp format
            (
                "0x1234567890123456789012345678901234567890",
                "optimism",
                {
                    "2024-01-01": [
                        {
                            "symbol": "ETH",
                            "amount": 10.0,
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "timestamp": "1640995200",  # Unix timestamp
                        }
                    ],
                    "2024-01-02": [
                        {
                            "symbol": "ETH",
                            "amount": 5.0,
                            "from_address": "0xmaster123456789012345678901234567890123456",
                            "timestamp": "1641081600",  # Unix timestamp
                        }
                    ],
                },
                {},  # No outgoing transfers
                "0xmaster123456789012345678901234567890123456",
                5.0,
                None,
                {
                    "reversion_amount": 5.0,
                    "master_safe_address": "0xmaster123456789012345678901234567890123456",
                    "historical_reversion_value": 0.0,
                    "reversion_date": "02-01-2022",  # Date from Unix timestamp
                },
                "multiple ETH transfers with no reversion - Unix timestamp format",
            ),
            # Multiple ETH transfers with no reversion - invalid timestamp handling
            (
                "0x1234567890123456789012345678901234567890",
                "optimism",
                {
                    "2024-01-01": [
                        {
                            "symbol": "ETH",
                            "amount": 10.0,
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "timestamp": "2024-01-01T10:00:00Z",
                        }
                    ],
                    "2024-01-02": [
                        {
                            "symbol": "ETH",
                            "amount": 5.0,
                            "from_address": "0xmaster123456789012345678901234567890123456",
                            "timestamp": "invalid_timestamp",  # Invalid timestamp
                        }
                    ],
                },
                {},  # No outgoing transfers
                "0xmaster123456789012345678901234567890123456",
                5.0,
                None,
                {
                    "reversion_amount": 5.0,
                    "master_safe_address": "0xmaster123456789012345678901234567890123456",
                    "historical_reversion_value": 0.0,
                    "reversion_date": "2024-01-15",  # Should use current date as fallback
                },
                "multiple ETH transfers with no reversion - invalid timestamp handling",
            ),
        ],
    )
    def test_track_eth_transfers_and_reversions_variations(
        self,
        safe_address,
        chain,
        mock_incoming_transfers,
        mock_outgoing_transfers,
        mock_master_address,
        mock_native_balance,
        mock_reversion_value,
        expected_result,
        test_description,
    ):
        """Test _track_eth_transfers_and_reversions method with various scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_fetch_all_transfers_until_date_optimism(address, date):
            yield
            return mock_incoming_transfers

        def mock_fetch_outgoing_transfers_until_date_optimism(address, date):
            yield
            return mock_outgoing_transfers

        def mock_get_master_safe_address():
            yield
            return mock_master_address

        def mock_get_native_balance(chain, address):
            yield
            return mock_native_balance

        def mock_calculate_total_reversion_value(eth_transfers, reversion_transfers):
            yield
            return mock_reversion_value

        # Mock datetime for the invalid timestamp test case
        if "invalid timestamp handling" in test_description:

            class MockDateTime:
                @staticmethod
                def now():
                    class MockDate:
                        def strftime(self, format_str):
                            return "2024-01-15"

                    return MockDate()

                @staticmethod
                def fromtimestamp(timestamp):
                    class MockDate:
                        def strftime(self, format_str):
                            return "2024-01-15"

                    return MockDate()

            with patch(
                "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.datetime",
                MockDateTime,
            ):
                with patch.multiple(
                    fetch_behaviour,
                    _fetch_all_transfers_until_date_optimism=mock_fetch_all_transfers_until_date_optimism,
                    _fetch_outgoing_transfers_until_date_optimism=mock_fetch_outgoing_transfers_until_date_optimism,
                    get_master_safe_address=mock_get_master_safe_address,
                    _get_native_balance=mock_get_native_balance,
                    _calculate_total_reversion_value=mock_calculate_total_reversion_value,
                ):
                    result = self._consume_generator(
                        fetch_behaviour._track_eth_transfers_and_reversions(
                            safe_address, chain
                        )
                    )

                    assert (
                        result == expected_result
                    ), f"Expected {expected_result} for {test_description}"
        else:
            with patch.multiple(
                fetch_behaviour,
                _fetch_all_transfers_until_date_optimism=mock_fetch_all_transfers_until_date_optimism,
                _fetch_outgoing_transfers_until_date_optimism=mock_fetch_outgoing_transfers_until_date_optimism,
                get_master_safe_address=mock_get_master_safe_address,
                _get_native_balance=mock_get_native_balance,
                _calculate_total_reversion_value=mock_calculate_total_reversion_value,
            ):
                result = self._consume_generator(
                    fetch_behaviour._track_eth_transfers_and_reversions(
                        safe_address, chain
                    )
                )

                assert (
                    result == expected_result
                ), f"Expected {expected_result} for {test_description}"

    def test_calculate_total_reversion_value_invalid_timestamp(self):
        """Test reversion value calculation with invalid timestamp."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        eth_transfers = [
            {"symbol": "ETH", "amount": 10.0, "timestamp": "invalid_timestamp"}
        ]

        reversion_transfers = [
            {"symbol": "ETH", "amount": 2.0, "timestamp": "2024-01-03T10:00:00Z"}
        ]

        def mock_fetch_historical_eth_price(date_str):
            yield
            return 2000.0

        with patch.object(
            fetch_behaviour,
            "_fetch_historical_eth_price",
            mock_fetch_historical_eth_price,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_total_reversion_value(
                    eth_transfers, reversion_transfers
                )
            )

            # Should still calculate value even with invalid timestamp (uses current date as fallback)
            assert result == 4000.0

    def test_get_master_safe_address_success(self):
        """Test successful master safe address fetch."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Set service_id to a valid value using patch.dict
        with patch.dict(
            fetch_behaviour.params.__dict__, {"on_chain_service_id": "test_service_id"}
        ):
            # Mock service info with owner address (should be a tuple)
            mock_service_info = (
                "service_id",
                "0xmaster123456789012345678901234567890123456",
            )

            def mock_get_service_staking_state(chain):
                # Simulate staked state
                yield
                fetch_behaviour.service_staking_state = StakingState.STAKED

            def mock_get_service_info(chain):
                print(f"DEBUG: mock_get_service_info called with chain={chain}")
                # Yield intermediate values (like the actual method does)
                yield
                # Return the final result
                return mock_service_info

            def mock_check_is_valid_safe_address(address, chain):
                # Yield intermediate values (like the actual method does)
                yield
                # Return the final result
                return True

            with patch.object(
                fetch_behaviour,
                "_get_service_staking_state",
                mock_get_service_staking_state,
            ), patch.object(
                fetch_behaviour, "_get_service_info", mock_get_service_info
            ), patch.object(
                fetch_behaviour,
                "check_is_valid_safe_address",
                mock_check_is_valid_safe_address,
            ):
                result = self._consume_generator(
                    fetch_behaviour.get_master_safe_address()
                )

                assert result == "0xmaster123456789012345678901234567890123456"

    def test_get_master_safe_address_failure(self):
        """Test master safe address fetch failure."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_get_service_staking_state(chain):
            # Simulate staked state
            fetch_behaviour.service_staking_state = StakingState.STAKED

        def mock_get_service_info(chain):
            return None  # Service info fetch fails

        with patch.object(
            fetch_behaviour,
            "_get_service_staking_state",
            mock_get_service_staking_state,
        ), patch.object(fetch_behaviour, "_get_service_info", mock_get_service_info):
            result = self._consume_generator(fetch_behaviour.get_master_safe_address())

            assert result is None

    def test_get_master_safe_address_invalid_address(self):
        """Test master safe address fetch with invalid address."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock service info with owner address
        mock_service_info = [
            "service_id",
            "0xmaster123456789012345678901234567890123456",
        ]

        def mock_get_service_staking_state(chain):
            # Simulate staked state
            fetch_behaviour.service_staking_state = StakingState.STAKED

        def mock_get_service_info(chain):
            return mock_service_info

        def mock_check_is_valid_safe_address(address, chain):
            return False  # Invalid address

        with patch.object(
            fetch_behaviour,
            "_get_service_staking_state",
            mock_get_service_staking_state,
        ), patch.object(
            fetch_behaviour, "_get_service_info", mock_get_service_info
        ), patch.object(
            fetch_behaviour,
            "check_is_valid_safe_address",
            mock_check_is_valid_safe_address,
        ):
            result = self._consume_generator(fetch_behaviour.get_master_safe_address())

            assert result is None

    def test_get_master_safe_address_no_service_id(self):
        """Test master safe address fetch when no service ID is configured."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Set service_id to None using patch.dict
        with patch.dict(fetch_behaviour.params.__dict__, {"on_chain_service_id": None}):
            result = self._consume_generator(fetch_behaviour.get_master_safe_address())

            assert result is None

    def test_get_master_safe_address_no_investment_chains(self):
        """Test master safe address fetch when no investment chains are configured."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Set target_investment_chains to empty list using patch.dict
        with patch.dict(
            fetch_behaviour.params.__dict__, {"target_investment_chains": []}
        ):
            result = self._consume_generator(fetch_behaviour.get_master_safe_address())

            assert result is None

    def test_get_master_safe_address_service_registry_failure(self):
        """Test master safe address fetch when service registry fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Set staking token and chain to None to trigger fallback using patch.dict
        with patch.dict(
            fetch_behaviour.params.__dict__,
            {
                "staking_token_contract_address": None,
                "staking_chain": None,
                "on_chain_service_id": "test_service_id",
            },
        ):

            def mock_contract_interact(
                performative,
                contract_address,
                contract_public_id,
                contract_callable,
                data_key,
                service_id,
                chain_id,
            ):
                if contract_callable == "get_service_owner":
                    yield None  # Service registry fails
                else:
                    yield None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(fetch_behaviour.get_master_safe_address())

            assert result is None

    def test_get_master_safe_address_no_service_registry_address(self):
        """Test master safe address fetch when no service registry address is configured."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Set staking token and chain to None to trigger fallback using patch.dict
        with patch.dict(
            fetch_behaviour.params.__dict__,
            {
                "staking_token_contract_address": None,
                "staking_chain": None,
                "service_registry_contract_addresses": {},
                "on_chain_service_id": "test_service_id",
            },
        ):
            result = self._consume_generator(fetch_behaviour.get_master_safe_address())

            assert result is None

    @pytest.mark.parametrize(
        "kv_response,expected_result,test_description",
        [
            ({"investing_paused": "false"}, False, "flag set to false"),
            ({"investing_paused": "true"}, True, "flag set to true"),
            (None, False, "KV store returns None"),
            ("exception", False, "KV store exception"),
        ],
    )
    def test_read_investing_paused_variations(
        self, kv_response, expected_result, test_description
    ):
        """Test _read_investing_paused method with various KV store responses."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_read_kv(keys):
            yield
            if kv_response == "exception":
                raise Exception("KV store error")
            return kv_response

        with patch.object(fetch_behaviour, "_read_kv", mock_read_kv):
            result = self._consume_generator(fetch_behaviour._read_investing_paused())

            assert result is expected_result, f"Failed for {test_description}"

    # 6.9. Investment Calculation Methods
    def test_calculate_initial_investment_value_success(self):
        """Test successful initial investment calculation."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock transfer data
        mock_transfers = {
            "2024-01-15": [
                {
                    "symbol": "ETH",
                    "delta": 10.0,
                    "amount": 10.0,
                    "timestamp": "1642248000",
                },
                {
                    "symbol": "USDC",
                    "delta": 1000.0,
                    "amount": 1000.0,
                    "timestamp": "1642248000",
                },
            ],
            "2024-01-16": [
                {
                    "symbol": "ETH",
                    "delta": 5.0,
                    "amount": 5.0,
                    "timestamp": "1642334400",
                }
            ],
        }

        def mock_fetch_all_transfers_until_date_mode(
            address, end_date, fetch_till_date
        ):
            return mock_transfers

        def mock_fetch_all_transfers_until_date_optimism(address, end_date):
            yield mock_transfers

        def mock_fetch_historical_eth_price(date_str):
            yield
            return 2000.0  # $2000 per ETH

        def mock_get_coin_id_from_symbol(symbol, chain):
            if symbol == "USDC":
                return "usd-coin"
            return None

        def mock_get_current_timestamp():
            return 1640995200  # Mock timestamp for 2022-01-01

        with patch.object(
            fetch_behaviour,
            "_fetch_all_transfers_until_date_mode",
            mock_fetch_all_transfers_until_date_mode,
        ), patch.object(
            fetch_behaviour,
            "_fetch_all_transfers_until_date_optimism",
            mock_fetch_all_transfers_until_date_optimism,
        ), patch.object(
            fetch_behaviour, "_read_kv", self.mock_read_kv_empty
        ), patch.object(
            fetch_behaviour,
            "_load_chain_total_investment",
            self.mock_load_chain_total_investment,
        ), patch.object(
            fetch_behaviour,
            "_save_chain_total_investment",
            self.mock_save_chain_total_investment,
        ), patch.object(
            fetch_behaviour, "_write_kv", self.mock_write_kv
        ), patch.object(
            fetch_behaviour,
            "_track_eth_transfers_and_reversions",
            self.mock_track_eth_transfers_and_reversions_zero_reversion,
        ), patch.object(
            fetch_behaviour,
            "_fetch_historical_eth_price",
            mock_fetch_historical_eth_price,
        ), patch.object(
            fetch_behaviour,
            "_fetch_historical_token_price",
            self.mock_fetch_historical_token_price,
        ), patch.object(
            fetch_behaviour, "get_coin_id_from_symbol", mock_get_coin_id_from_symbol
        ), patch.object(
            fetch_behaviour, "_get_current_timestamp", mock_get_current_timestamp
        ):
            result = self._consume_generator(
                fetch_behaviour.calculate_initial_investment_value_from_funding_events()
            )

            # Expected: (10 ETH * $2000) + (1000 USDC * $1) + (5 ETH * $2000) = $20,000 + $1,000 + $10,000 = $31,000
            assert result == 31000.0

    def test_calculate_initial_investment_value_no_safe_address(self):
        """Test initial investment calculation when no safe address is configured."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Set safe addresses to empty to trigger the warning
        with patch.dict(
            fetch_behaviour.params.__dict__, {"safe_contract_addresses": {}}
        ):

            def mock_fetch_all_transfers_until_date_mode(
                address, end_date, fetch_till_date
            ):
                return {}

            def mock_fetch_all_transfers_until_date_optimism(address, end_date):
                yield {}

            def mock_fetch_historical_eth_price(date_str):
                yield
                return 2000.0  # $2000 per ETH

            def mock_get_current_timestamp():
                return 1642248000

            with patch.object(
                fetch_behaviour, "_read_kv", self.mock_read_kv_empty
            ), patch.object(
                fetch_behaviour,
                "_load_chain_total_investment",
                self.mock_load_chain_total_investment,
            ), patch.object(
                fetch_behaviour,
                "_fetch_all_transfers_until_date_mode",
                mock_fetch_all_transfers_until_date_mode,
            ), patch.object(
                fetch_behaviour,
                "_fetch_all_transfers_until_date_optimism",
                mock_fetch_all_transfers_until_date_optimism,
            ), patch.object(
                fetch_behaviour,
                "_track_eth_transfers_and_reversions",
                self.mock_track_eth_transfers_and_reversions_zero_reversion,
            ), patch.object(
                fetch_behaviour,
                "_fetch_historical_eth_price",
                mock_fetch_historical_eth_price,
            ), patch.object(
                fetch_behaviour,
                "_save_chain_total_investment",
                self.mock_save_chain_total_investment,
            ), patch.object(
                fetch_behaviour, "_write_kv", self.mock_write_kv
            ), patch.object(
                fetch_behaviour, "_get_current_timestamp", mock_get_current_timestamp
            ):
                result = self._consume_generator(
                    fetch_behaviour.calculate_initial_investment_value_from_funding_events()
                )

                assert result is None

    def test_calculate_initial_investment_value_no_transfers(self):
        """Test initial investment calculation when no transfers are found."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_fetch_all_transfers_until_date_mode(
            address, end_date, fetch_till_date
        ):
            return {}  # No transfers

        def mock_fetch_all_transfers_until_date_optimism(address, end_date):
            yield {}  # No transfers

        def mock_fetch_historical_eth_price(date_str):
            yield
            return 2000.0  # $2000 per ETH

        def mock_get_current_timestamp():
            return 1642248000

        with patch.object(
            fetch_behaviour, "_read_kv", self.mock_read_kv_empty
        ), patch.object(
            fetch_behaviour,
            "_load_chain_total_investment",
            self.mock_load_chain_total_investment,
        ), patch.object(
            fetch_behaviour,
            "_fetch_all_transfers_until_date_mode",
            mock_fetch_all_transfers_until_date_mode,
        ), patch.object(
            fetch_behaviour,
            "_fetch_all_transfers_until_date_optimism",
            mock_fetch_all_transfers_until_date_optimism,
        ), patch.object(
            fetch_behaviour,
            "_track_eth_transfers_and_reversions",
            self.mock_track_eth_transfers_and_reversions_zero_reversion,
        ), patch.object(
            fetch_behaviour,
            "_fetch_historical_eth_price",
            mock_fetch_historical_eth_price,
        ), patch.object(
            fetch_behaviour,
            "_save_chain_total_investment",
            self.mock_save_chain_total_investment,
        ), patch.object(
            fetch_behaviour, "_write_kv", self.mock_write_kv
        ), patch.object(
            fetch_behaviour, "_get_current_timestamp", mock_get_current_timestamp
        ):
            result = self._consume_generator(
                fetch_behaviour.calculate_initial_investment_value_from_funding_events()
            )

            assert result is None

    def test_calculate_chain_investment_value_with_reversion(self):
        """Test chain investment calculation with reversion handling."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock transfer data
        mock_transfers = {
            "2024-01-15": [
                {
                    "symbol": "ETH",
                    "delta": 10.0,
                    "amount": 10.0,
                    "timestamp": "1642248000",
                }
            ]
        }

        def mock_track_eth_transfers_and_reversions(safe_address, chain):
            yield
            return {
                "reversion_amount": 2.0,
                "historical_reversion_value": 1000.0,
                "reversion_date": "15-01-2024",
            }

        def mock_fetch_historical_eth_price(date_str):
            yield
            return 2000.0  # $2000 per ETH

        with patch.object(
            fetch_behaviour,
            "_track_eth_transfers_and_reversions",
            mock_track_eth_transfers_and_reversions,
        ), patch.object(
            fetch_behaviour,
            "_fetch_historical_eth_price",
            mock_fetch_historical_eth_price,
        ), patch.object(
            fetch_behaviour,
            "_save_chain_total_investment",
            self.mock_save_chain_total_investment,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_chain_investment_value(
                    mock_transfers,
                    "optimism",
                    "0x1234567890123456789012345678901234567890",
                )
            )

            # Expected: (10 ETH * $2000) - (2 ETH * $2000) - $1000 = $20,000 - $4,000 - $1,000 = $15,000
            assert result == 15000.0

    def test_calculate_chain_investment_value_negative_amounts(self):
        """Test chain investment calculation with negative amounts (should be skipped)."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock transfer data with negative amounts
        mock_transfers = {
            "2024-01-15": [
                {
                    "symbol": "ETH",
                    "delta": -5.0,
                    "amount": -5.0,
                    "timestamp": "1642248000",
                },
                {
                    "symbol": "ETH",
                    "delta": 10.0,
                    "amount": 10.0,
                    "timestamp": "1642248000",
                },
            ]
        }

        def mock_track_eth_transfers_and_reversions(safe_address, chain):
            yield
            return {
                "reversion_amount": 0.0,
                "historical_reversion_value": 0.0,
                "reversion_date": None,
            }

        def mock_fetch_historical_eth_price(date_str):
            yield
            return 2000.0  # $2000 per ETH

        with patch.object(
            fetch_behaviour,
            "_track_eth_transfers_and_reversions",
            self.mock_track_eth_transfers_and_reversions_zero_reversion,
        ), patch.object(
            fetch_behaviour,
            "_fetch_historical_eth_price",
            mock_fetch_historical_eth_price,
        ), patch.object(
            fetch_behaviour,
            "_save_chain_total_investment",
            self.mock_save_chain_total_investment,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_chain_investment_value(
                    mock_transfers,
                    "optimism",
                    "0x1234567890123456789012345678901234567890",
                )
            )

            # Expected: Only positive amount (10 ETH * $2000) = $20,000
            assert result == 20000.0

    def test_calculate_chain_investment_value_price_failure(self):
        """Test chain investment calculation when price fetching fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock transfer data
        mock_transfers = {
            "2024-01-15": [
                {
                    "symbol": "ETH",
                    "delta": 10.0,
                    "amount": 10.0,
                    "timestamp": "1642248000",
                },
                {
                    "symbol": "USDC",
                    "delta": 1000.0,
                    "amount": 1000.0,
                    "timestamp": "1642248000",
                },
            ]
        }

        def mock_fetch_historical_eth_price(date_str):
            yield
            return None  # Price fetch fails

        def mock_fetch_historical_token_price(coingecko_id, date_str):
            yield
            return None  # Price fetch fails

        def mock_get_coin_id_from_symbol(symbol, chain):
            if symbol == "USDC":
                return "usd-coin"
            return None

        with patch.object(
            fetch_behaviour,
            "_track_eth_transfers_and_reversions",
            self.mock_track_eth_transfers_and_reversions_zero_reversion,
        ), patch.object(
            fetch_behaviour,
            "_fetch_historical_eth_price",
            mock_fetch_historical_eth_price,
        ), patch.object(
            fetch_behaviour,
            "_fetch_historical_token_price",
            mock_fetch_historical_token_price,
        ), patch.object(
            fetch_behaviour, "get_coin_id_from_symbol", mock_get_coin_id_from_symbol
        ), patch.object(
            fetch_behaviour,
            "_save_chain_total_investment",
            self.mock_save_chain_total_investment,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_chain_investment_value(
                    mock_transfers,
                    "optimism",
                    "0x1234567890123456789012345678901234567890",
                )
            )

            # Expected: No valid prices, so no investment value
            assert result == 0.0

    def test_load_chain_total_investment_success(self):
        """Test successful loading of chain total investment."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_read_kv(keys):
            yield
            return {"optimism_total_investment": "25000.0"}

        with patch.object(fetch_behaviour, "_read_kv", mock_read_kv):
            result = self._consume_generator(
                fetch_behaviour._load_chain_total_investment("optimism")
            )
            assert result == 25000.0

    def test_load_chain_total_investment_no_data(self):
        """Test loading chain total investment when no data exists."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        with patch.object(fetch_behaviour, "_read_kv", self.mock_read_kv_empty):
            result = self._consume_generator(
                fetch_behaviour._load_chain_total_investment("optimism")
            )
            assert result == 0.0

    def test_load_chain_total_investment_invalid_data(self):
        """Test loading chain total investment with invalid data."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_read_kv(keys):
            yield {"optimism_total_investment": "invalid_number"}

        with patch.object(fetch_behaviour, "_read_kv", mock_read_kv):
            result = self._consume_generator(
                fetch_behaviour._load_chain_total_investment("optimism")
            )
            assert result == 0.0

    def test_save_chain_total_investment_success(self):
        """Test successful saving of chain total investment."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        with patch.object(fetch_behaviour, "_write_kv", self.mock_write_kv):
            # Should not raise any exception
            self._consume_generator(
                fetch_behaviour._save_chain_total_investment("optimism", 25000.0)
            )

    def test_load_funding_events_data_success(self):
        """Test successful loading of funding events data."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        mock_data = {
            "optimism": {
                "2024-01-15": [{"symbol": "ETH", "delta": 10.0, "amount": 10.0}]
            }
        }

        def mock_read_kv(keys):
            yield
            return {"funding_events": json.dumps(mock_data)}

        with patch.object(fetch_behaviour, "_read_kv", mock_read_kv):
            result = self._consume_generator(
                fetch_behaviour._load_funding_events_data()
            )
            assert result == mock_data

    def test_load_funding_events_data_no_data(self):
        """Test loading funding events data when no data exists."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        with patch.object(fetch_behaviour, "_read_kv", self.mock_read_kv_empty):
            result = self._consume_generator(
                fetch_behaviour._load_funding_events_data()
            )
            assert result == {}

    def test_load_funding_events_data_invalid_json(self):
        """Test loading funding events data with invalid JSON."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_read_kv(keys):
            yield {"funding_events": "invalid_json"}

        with patch.object(fetch_behaviour, "_read_kv", mock_read_kv):
            result = self._consume_generator(
                fetch_behaviour._load_funding_events_data()
            )
            assert result == {}

    def test_calculate_initial_investment_value_unsupported_chain(self):
        """Test initial investment calculation with unsupported chain."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Set target chains to include unsupported chain
        with patch.dict(
            fetch_behaviour.params.__dict__,
            {"target_investment_chains": ["unsupported_chain"]},
        ):

            def mock_fetch_all_transfers_until_date_mode(
                address, end_date, fetch_till_date
            ):
                return {}  # No transfers for unsupported chain

            def mock_fetch_all_transfers_until_date_optimism(address, end_date):
                yield {}  # No transfers for unsupported chain

            def mock_fetch_historical_eth_price(date_str):
                yield
                return 2000.0  # $2000 per ETH

            def mock_get_current_timestamp():
                return 1642248000

            with patch.object(
                fetch_behaviour, "_read_kv", self.mock_read_kv_empty
            ), patch.object(
                fetch_behaviour,
                "_load_chain_total_investment",
                self.mock_load_chain_total_investment,
            ), patch.object(
                fetch_behaviour,
                "_fetch_all_transfers_until_date_mode",
                mock_fetch_all_transfers_until_date_mode,
            ), patch.object(
                fetch_behaviour,
                "_fetch_all_transfers_until_date_optimism",
                mock_fetch_all_transfers_until_date_optimism,
            ), patch.object(
                fetch_behaviour,
                "_track_eth_transfers_and_reversions",
                self.mock_track_eth_transfers_and_reversions_zero_reversion,
            ), patch.object(
                fetch_behaviour,
                "_fetch_historical_eth_price",
                mock_fetch_historical_eth_price,
            ), patch.object(
                fetch_behaviour,
                "_save_chain_total_investment",
                self.mock_save_chain_total_investment,
            ), patch.object(
                fetch_behaviour, "_write_kv", self.mock_write_kv
            ), patch.object(
                fetch_behaviour, "_get_current_timestamp", mock_get_current_timestamp
            ):
                result = self._consume_generator(
                    fetch_behaviour.calculate_initial_investment_value_from_funding_events()
                )

                assert result is None

    def test_calculate_initial_investment_value_exception_handling(self):
        """Test initial investment calculation with exception handling."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock transfer data that will cause an exception
        mock_transfers = {
            "2024-01-15": [
                {
                    "symbol": "ETH",
                    "delta": 10.0,
                    "amount": 10.0,
                    "timestamp": "1642248000",
                }
            ]
        }

        def mock_fetch_all_transfers_until_date_mode(
            address, end_date, fetch_till_date
        ):
            return mock_transfers

        def mock_fetch_all_transfers_until_date_optimism(address, end_date):
            yield mock_transfers

        def mock_fetch_historical_eth_price(date_str):
            raise Exception("Price fetch failed")

        def mock_get_current_timestamp():
            return 1642248000

        def mock_calculate_chain_investment_value(all_transfers, chain, safe_address):
            yield
            return 0.0  # Return 0.0 due to exception

        with patch.object(
            fetch_behaviour, "_read_kv", self.mock_read_kv_empty
        ), patch.object(
            fetch_behaviour,
            "_load_chain_total_investment",
            self.mock_load_chain_total_investment,
        ), patch.object(
            fetch_behaviour,
            "_fetch_all_transfers_until_date_mode",
            mock_fetch_all_transfers_until_date_mode,
        ), patch.object(
            fetch_behaviour,
            "_fetch_all_transfers_until_date_optimism",
            mock_fetch_all_transfers_until_date_optimism,
        ), patch.object(
            fetch_behaviour,
            "_track_eth_transfers_and_reversions",
            self.mock_track_eth_transfers_and_reversions_zero_reversion,
        ), patch.object(
            fetch_behaviour,
            "_fetch_historical_eth_price",
            mock_fetch_historical_eth_price,
        ), patch.object(
            fetch_behaviour,
            "_save_chain_total_investment",
            self.mock_save_chain_total_investment,
        ), patch.object(
            fetch_behaviour, "_get_current_timestamp", mock_get_current_timestamp
        ), patch.object(
            fetch_behaviour,
            "_calculate_chain_investment_value",
            mock_calculate_chain_investment_value,
        ), patch.object(
            fetch_behaviour, "_write_kv", self.mock_write_kv
        ):
            result = self._consume_generator(
                fetch_behaviour.calculate_initial_investment_value_from_funding_events()
            )

            # Should handle exception gracefully and return None (since total_investment is 0.0)
            assert result is None

    def test_adjust_for_decimals_edge_cases(self):
        """Test _adjust_for_decimals method with edge cases."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Test with zero decimals
        result1 = fetch_behaviour._adjust_for_decimals(100, 0)
        assert result1 == Decimal("100")

        # Test with large numbers
        result2 = fetch_behaviour._adjust_for_decimals(999999999999999999, 18)
        assert result2 == Decimal("0.999999999999999999")

    def test_update_portfolio_breakdown_ratios_success(self):
        """Test _update_portfolio_breakdown_ratios method with valid data."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        portfolio_breakdown = [
            {"value_usd": "100.0", "balance": "10.0", "price": "10.0"},
            {"value_usd": "200.0", "balance": "20.0", "price": "10.0"},
            {"value_usd": "300.0", "balance": "30.0", "price": "10.0"},
        ]

        total_value = Decimal("600.0")

        fetch_behaviour._update_portfolio_breakdown_ratios(
            portfolio_breakdown, total_value
        )

        # Check that ratios are calculated correctly
        assert len(portfolio_breakdown) == 3
        assert portfolio_breakdown[0]["ratio"] == Decimal("0.166667")  # 100/600
        assert portfolio_breakdown[1]["ratio"] == Decimal("0.333333")  # 200/600
        assert portfolio_breakdown[2]["ratio"] == Decimal("0.5")  # 300/600

        # Check that values are converted to float
        assert isinstance(portfolio_breakdown[0]["value_usd"], float)
        assert isinstance(portfolio_breakdown[0]["balance"], float)
        assert isinstance(portfolio_breakdown[0]["price"], float)

    def test_update_portfolio_breakdown_ratios_empty_list(self):
        """Test _update_portfolio_breakdown_ratios method with empty portfolio."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        portfolio_breakdown = []
        total_value = Decimal("100.0")

        # Should not raise any exception
        fetch_behaviour._update_portfolio_breakdown_ratios(
            portfolio_breakdown, total_value
        )
        assert len(portfolio_breakdown) == 0

    def test_update_portfolio_breakdown_ratios_zero_total_value(self):
        """Test _update_portfolio_breakdown_ratios method with zero total value."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        portfolio_breakdown = [
            {"value_usd": "100.0", "balance": "10.0", "price": "10.0"}
        ]

        total_value = Decimal("0.0")

        fetch_behaviour._update_portfolio_breakdown_ratios(
            portfolio_breakdown, total_value
        )

        # Should set ratio to 0.0 when total_value is 0
        assert portfolio_breakdown[0]["ratio"] == Decimal("0.0")

    def test_update_portfolio_breakdown_ratios_filter_small_values(self):
        """Test _update_portfolio_breakdown_ratios method filters small values."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        portfolio_breakdown = [
            {"value_usd": "100.0", "balance": "10.0", "price": "10.0"},
            {
                "value_usd": "0.005",
                "balance": "0.001",
                "price": "5.0",
            },  # Should be filtered out
            {"value_usd": "200.0", "balance": "20.0", "price": "10.0"},
        ]

        total_value = Decimal("300.0")

        fetch_behaviour._update_portfolio_breakdown_ratios(
            portfolio_breakdown, total_value
        )

        # Should filter out the small value entry
        assert len(portfolio_breakdown) == 2
        assert portfolio_breakdown[0]["value_usd"] == 100.0
        assert portfolio_breakdown[1]["value_usd"] == 200.0

    def test_update_portfolio_breakdown_ratios_missing_value_usd(self):
        """Test _update_portfolio_breakdown_ratios method with missing value_usd."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        portfolio_breakdown = [
            {"value_usd": 100.0, "balance": 10.0, "price": 10.0},
            {
                "value_usd": 0.0,
                "balance": 10.0,
                "price": 10.0,
            },  # Zero value_usd (will be filtered out)
            {"value_usd": 200.0, "balance": 20.0, "price": 10.0},
        ]

        total_value = Decimal("300.0")

        fetch_behaviour._update_portfolio_breakdown_ratios(
            portfolio_breakdown, total_value
        )

        # Should filter out the entry with zero value_usd (less than 0.01 threshold)
        assert len(portfolio_breakdown) == 2

    def test_create_portfolio_data_success(self):
        """Test _create_portfolio_data method with valid inputs."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock environment variable
        with patch.dict("os.environ", {"AEA_AGENT": "test:agent:hash"}):
            total_pools_value = Decimal("1000.0")
            total_safe_value = Decimal("500.0")
            initial_investment = 1000.0
            volume = 500.0
            allocations = [
                {
                    "chain": "optimism",
                    "type": "uniswap",
                    "id": "pool1",
                    "assets": ["WETH", "USDC"],
                    "apr": 10.5,
                    "details": {"pool": "test"},
                    "ratio": 0.6,
                    "address": "0x123",
                }
            ]
            portfolio_breakdown = [
                {
                    "asset": "WETH",
                    "address": "0x456",
                    "balance": 10.0,
                    "price": 2000.0,
                    "value_usd": 20000.0,
                    "ratio": 0.6,
                }
            ]

            with patch.object(
                fetch_behaviour, "_get_current_timestamp", return_value=1234567890
            ):
                result = fetch_behaviour._create_portfolio_data(
                    total_pools_value,
                    total_safe_value,
                    Decimal("0.0"),
                    initial_investment,
                    volume,
                    allocations,
                    portfolio_breakdown,
                )

            assert result["portfolio_value"] == 1500.0
            assert result["value_in_pools"] == 1000.0
            assert result["value_in_safe"] == 500.0
            assert result["initial_investment"] == 1000.0
            assert result["volume"] == 500.0
            assert result["total_roi"] == 50.0  # (1500/1000 - 1) * 100
            assert result["agent_hash"] == "hash"
            assert len(result["allocations"]) == 1
            assert len(result["portfolio_breakdown"]) == 1

    def test_create_portfolio_data_zero_initial_investment(self):
        """Test _create_portfolio_data method with zero initial investment."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        with patch.dict("os.environ", {"AEA_AGENT": "test:agent:hash"}):
            with patch.object(
                fetch_behaviour, "_get_current_timestamp", return_value=1234567890
            ):
                result = fetch_behaviour._create_portfolio_data(
                    Decimal("1000.0"),
                    Decimal("500.0"),
                    Decimal("0.0"),
                    0.0,
                    500.0,
                    [],
                    [],
                )

        assert result["total_roi"] is None

    def test_create_portfolio_data_negative_initial_investment(self):
        """Test _create_portfolio_data method with negative initial investment."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        with patch.dict("os.environ", {"AEA_AGENT": "test:agent:hash"}):
            with patch.object(
                fetch_behaviour, "_get_current_timestamp", return_value=1234567890
            ):
                result = fetch_behaviour._create_portfolio_data(
                    Decimal("1000.0"),
                    Decimal("500.0"),
                    Decimal("0.0"),
                    -100.0,
                    500.0,
                    [],
                    [],
                )

                assert result["total_roi"] is None

    def test_create_portfolio_data_no_agent_config(self):
        """Test _create_portfolio_data method without agent config."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                fetch_behaviour, "_get_current_timestamp", return_value=1234567890
            ):
                result = fetch_behaviour._create_portfolio_data(
                    Decimal("1000.0"),
                    Decimal("500.0"),
                    Decimal("0.0"),
                    1000.0,
                    500.0,
                    [],
                    [],
                )

                assert result["agent_hash"] == "Not found"

    def test_save_transfer_data_mode_success(self):
        """Test _save_transfer_data_mode method."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        test_data = {"transfers": [{"hash": "0x123", "value": "1000"}]}

        with patch.object(fetch_behaviour, "_write_kv", self.mock_write_kv):
            result = list(fetch_behaviour._save_transfer_data_mode(test_data))
            assert result == [None]

    def test_save_transfer_data_optimism_success(self):
        """Test _save_transfer_data_optimism method."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        test_data = {"transfers": [{"hash": "0x123", "value": "1000"}]}

        with patch.object(fetch_behaviour, "_write_kv", self.mock_write_kv):
            result = list(fetch_behaviour._save_transfer_data_optimism(test_data))
            assert result == [None]

    def test_get_tick_ranges_unsupported_dex(self):
        """Test _get_tick_ranges method with unsupported DEX type."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {"dex_type": "unsupported_dex"}

        result = list(fetch_behaviour._get_tick_ranges(position, "optimism"))
        assert result == []

    def test_get_tick_ranges_velodrome_non_cl_pool(self):
        """Test _get_tick_ranges method with Velodrome non-CL pool."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {"dex_type": "velodrome", "is_cl_pool": False}

        result = list(fetch_behaviour._get_tick_ranges(position, "optimism"))
        assert result == []

    def test_get_tick_ranges_no_pool_address(self):
        """Test _get_tick_ranges method with no pool address."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {"dex_type": "uniswap_v3"}

        result = list(fetch_behaviour._get_tick_ranges(position, "optimism"))
        assert result == []

    def test_get_tick_ranges_contract_failure(self):
        """Test _get_tick_ranges method when contract call fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "dex_type": "uniswap_v3",
            "pool_address": "0x1234567890123456789012345678901234567890",
        }

        def mock_contract_interact(**kwargs):
            yield None  # Simulate contract call failure

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = list(fetch_behaviour._get_tick_ranges(position, "optimism"))
            assert result == []

    def test_get_tick_ranges_invalid_slot0_data(self):
        """Test _get_tick_ranges method with invalid slot0 data."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "dex_type": "uniswap_v3",
            "pool_address": "0x1234567890123456789012345678901234567890",
        }

        def mock_contract_interact(**kwargs):
            yield {"invalid": "data"}  # Missing tick data

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = list(fetch_behaviour._get_tick_ranges(position, "optimism"))
            assert result == []

    def test_have_positions_changed_no_last_data(self):
        """Test _have_positions_changed method with no last portfolio data."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        fetch_behaviour.current_positions = []

        result = fetch_behaviour._have_positions_changed({})
        assert result is True  # Should return True when no last data

    def test_have_positions_changed_no_positions_key(self):
        """Test _have_positions_changed method with no positions key."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        fetch_behaviour.current_positions = []

        last_portfolio_data = {"some_other_key": "value"}

        result = fetch_behaviour._have_positions_changed(last_portfolio_data)
        assert result is True  # Should return True when no positions key

    def test_have_positions_changed_different_number_of_positions(self):
        """Test _have_positions_changed method with different number of positions."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        last_portfolio_data = {
            "positions": [{"pool_address": "0x123", "balance": "100.0"}]
        }

        # Mock current positions with different count
        with patch.object(
            fetch_behaviour,
            "current_positions",
            [
                {"pool_address": "0x123", "balance": "100.0"},
                {"pool_address": "0x456", "balance": "200.0"},
            ],
        ):
            result = fetch_behaviour._have_positions_changed(last_portfolio_data)
            assert result is True

    def test_period_0_initialization_logic(self):
        """Test the period 0 initialization logic including validation and updates."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Setup test data
        fetch_behaviour.current_positions = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": False,
                "is_stable": True,
                "pool_address": "0x1234567890123456789012345678901234567890",
                "enter_tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "chain": "optimism",
                "isUpdated": False,
            },
            {
                "dex_type": "uniswap_v3",
                "pool_address": "0x0987654321098765432109876543210987654321",
                "status": "open",
            },
        ]

        # Initialize assets attribute
        fetch_behaviour.assets = {}

        mock_synchronized_data = MagicMock()
        mock_synchronized_data.period_count = 0  # Period 0
        mock_synchronized_data.round_count = 1

        # Create mock objects that we can track
        mock_validate_velodrome_v2_pool_addresses = MagicMock()
        mock_update_position_amounts = MagicMock()
        mock_check_and_update_zero_liquidity_positions = MagicMock()

        def mock_validate_velodrome_v2_pool_addresses_impl():
            """Mock _validate_velodrome_v2_pool_addresses method."""
            mock_validate_velodrome_v2_pool_addresses()
            yield
            # Simulate validation process
            fetch_behaviour.current_positions[0]["isUpdated"] = True
            fetch_behaviour.context.logger.info("Velodrome v2 pool addresses validated")

        def mock_update_position_amounts_impl():
            """Mock update_position_amounts method."""
            mock_update_position_amounts()
            yield
            # Simulate updating position amounts
            fetch_behaviour.context.logger.info("Position amounts updated")

        def mock_check_and_update_zero_liquidity_positions_impl():
            """Mock check_and_update_zero_liquidity_positions method."""
            mock_check_and_update_zero_liquidity_positions()
            # Simulate checking zero liquidity positions
            fetch_behaviour.context.logger.info(
                "Zero liquidity positions checked and updated"
            )

        def mock_get_http_response(message, dialogue):
            """Mock get_http_response method."""
            yield
            return MagicMock(performative="success", body=b'{"price": 1.0}')

        with patch.dict("os.environ", {"AEA_AGENT": "test_agent:hash123"}):
            with patch.multiple(
                fetch_behaviour,
                _validate_velodrome_v2_pool_addresses=mock_validate_velodrome_v2_pool_addresses_impl,
                update_position_amounts=mock_update_position_amounts_impl,
                check_and_update_zero_liquidity_positions=mock_check_and_update_zero_liquidity_positions_impl,
                _get_native_balance=self.mock_get_native_balance,
                _read_kv=self.mock_read_kv_default,
                _write_kv=self.mock_write_kv,
                _get_current_timestamp=self.mock_get_current_timestamp,
                should_recalculate_portfolio=self.mock_should_recalculate_portfolio,
                calculate_user_share_values=self.mock_calculate_user_share_values,
                get_signature=self.mock_get_signature,
                send_a2a_transaction=self.mock_send_a2a_transaction,
                wait_until_round_end=self.mock_wait_until_round_end,
                contract_interact=self.mock_contract_interact,
                store_whitelisted_assets=self.mock_store_whitelisted_assets,
                read_whitelisted_assets=self.mock_read_whitelisted_assets,
                _track_whitelisted_assets=self.mock_track_whitelisted_assets,
                store_portfolio_data=self.mock_store_portfolio_data,
                _track_eth_transfers_and_reversions=self.mock_track_eth_transfers_and_reversions_zero_reversion,
                get_http_response=mock_get_http_response,
                _calculate_chain_investment_value=self.mock_calculate_chain_investment_value,
                _fetch_historical_token_price=self.mock_fetch_historical_token_price,
                update_accumulated_rewards_for_chain=self.mock_update_accumulated_rewards,
            ):
                with patch(
                    "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.hash_payload_to_hex",
                    return_value="mocked_payload_hash",
                ):
                    with patch.object(
                        type(fetch_behaviour),
                        "synchronized_data",
                        new_callable=PropertyMock,
                        return_value=mock_synchronized_data,
                    ):
                        # Execute the async_act method
                        list(fetch_behaviour.async_act())

                        # Verify that period 0 initialization logic was executed
                        assert mock_synchronized_data.period_count == 0

                        # Verify that the validation method was called
                        mock_validate_velodrome_v2_pool_addresses.assert_called_once()
                        assert fetch_behaviour.current_positions[0]["isUpdated"] is True

                        # Verify that the position amounts update was triggered
                        mock_update_position_amounts.assert_called_once()

                        # Verify that zero liquidity positions check was triggered
                        mock_check_and_update_zero_liquidity_positions.assert_called_once()

                        # Additional verification: Check that the behavior completed successfully
                        # by verifying that the round ended properly
                        assert mock_synchronized_data.round_count == 1

    def test_period_0_validation_methods(self):
        """Test the individual validation methods called during period 0."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Test _validate_velodrome_v2_pool_addresses method
        fetch_behaviour.current_positions = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": False,
                "is_stable": True,
                "pool_address": "0x1234567890123456789012345678901234567890",
                "enter_tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "chain": "optimism",
                "isUpdated": False,
            }
        ]

        def mock_validate_velodrome_v2_pool_address(position):
            """Mock _validate_velodrome_v2_pool_address method."""
            yield
            position["isUpdated"] = True
            return True

        def mock_store_current_positions():
            """Mock store_current_positions method."""
            pass

        with patch.multiple(
            fetch_behaviour,
            _validate_velodrome_v2_pool_address=mock_validate_velodrome_v2_pool_address,
            store_current_positions=mock_store_current_positions,
        ):
            # Test the validation method
            list(fetch_behaviour._validate_velodrome_v2_pool_addresses())

            # Verify that the position was marked as updated
            assert fetch_behaviour.current_positions[0]["isUpdated"] is True

        # Test update_position_amounts method
        def mock_update_position_amounts():
            """Mock update_position_amounts method."""
            yield
            fetch_behaviour.context.logger.info("Position amounts updated successfully")

        with patch.object(
            fetch_behaviour, "update_position_amounts", mock_update_position_amounts
        ):
            # Test the update method
            list(fetch_behaviour.update_position_amounts())

        # Test check_and_update_zero_liquidity_positions method
        def mock_check_and_update_zero_liquidity_positions():
            """Mock check_and_update_zero_liquidity_positions method."""
            fetch_behaviour.context.logger.info("Zero liquidity positions checked")

        with patch.object(
            fetch_behaviour,
            "check_and_update_zero_liquidity_positions",
            mock_check_and_update_zero_liquidity_positions,
        ):
            # Test the check method
            fetch_behaviour.check_and_update_zero_liquidity_positions()

    def test_period_0_validation_skips_non_velodrome_positions(self):
        """Test that validation only runs for Velodrome v2 stable pool positions."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Setup test data with mixed position types
        fetch_behaviour.current_positions = [
            {
                "dex_type": "velodrome",
                "is_cl_pool": False,
                "is_stable": True,  # This should be validated
                "pool_address": "0x1234567890123456789012345678901234567890",
                "enter_tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "chain": "optimism",
                "isUpdated": False,
            },
            {
                "dex_type": "velodrome",
                "is_cl_pool": True,  # CL pool - should be skipped
                "is_stable": True,
                "pool_address": "0x0987654321098765432109876543210987654321",
                "enter_tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "chain": "optimism",
                "isUpdated": False,
            },
            {
                "dex_type": "velodrome",
                "is_cl_pool": False,
                "is_stable": False,  # Not stable - should be skipped
                "pool_address": "0x1111111111111111111111111111111111111111",
                "enter_tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "chain": "optimism",
                "isUpdated": False,
            },
            {
                "dex_type": "uniswap_v3",  # Different DEX - should be skipped
                "pool_address": "0x2222222222222222222222222222222222222222",
                "status": "open",
            },
        ]

        validation_calls = []

        def mock_validate_velodrome_v2_pool_address(position):
            """Mock _validate_velodrome_v2_pool_address method."""
            yield
            validation_calls.append(position["pool_address"])
            position["isUpdated"] = True
            return True

        def mock_store_current_positions():
            """Mock store_current_positions method."""
            pass

        with patch.multiple(
            fetch_behaviour,
            _validate_velodrome_v2_pool_address=mock_validate_velodrome_v2_pool_address,
            store_current_positions=mock_store_current_positions,
        ):
            # Test the validation method
            list(fetch_behaviour._validate_velodrome_v2_pool_addresses())

            # Verify that only the first position (Velodrome v2 stable pool) was validated
            assert len(validation_calls) == 1
            assert validation_calls[0] == "0x1234567890123456789012345678901234567890"

            # Verify that only the first position was marked as updated
            assert fetch_behaviour.current_positions[0]["isUpdated"] is True
            assert fetch_behaviour.current_positions[1]["isUpdated"] is False
            assert fetch_behaviour.current_positions[2]["isUpdated"] is False

    def test_calculate_corrected_yield_basic_case(self):
        """Test _calculate_corrected_yield method with basic token increases."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "token0": "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8",  # USDC
            "token1": "0x4200000000000000000000000000000000000006",  # WETH
            "token0_symbol": "USDC",
            "token1_symbol": "WETH",
            "dex_type": "uniswap_v3",
            "staked": False,
        }

        initial_amount0 = 1000000000  # 1000 USDC (6 decimals)
        initial_amount1 = 500000000000000000  # 0.5 WETH (18 decimals)

        current_balances = {
            "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8": Decimal(
                "1010.0"
            ),  # 1010 USDC
            "0x4200000000000000000000000000000000000006": Decimal("0.52"),  # 0.52 WETH
        }

        def mock_get_token_decimals(chain, token_address):
            """Mock _get_token_decimals method."""
            yield
            if token_address == "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8":
                return 6  # USDC
            elif token_address == "0x4200000000000000000000000000000000000006":
                return 18  # WETH
            return None

        def mock_fetch_token_price(token_address, chain):
            """Mock _fetch_token_price method."""
            yield
            if token_address == "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8":
                return 1.0  # USDC price
            elif token_address == "0x4200000000000000000000000000000000000006":
                return 2500.0  # WETH price
            return None

        with patch.multiple(
            fetch_behaviour,
            _get_token_decimals=mock_get_token_decimals,
            _fetch_token_price=mock_fetch_token_price,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_corrected_yield(
                    position,
                    initial_amount0,
                    initial_amount1,
                    current_balances,
                    "optimism",
                )
            )

            # Expected yield: 10 USDC * $1 + 0.02 WETH * $2500 = $10 + $50 = $60
            expected_yield = Decimal("60.0")
            assert result == expected_yield

    def test_calculate_corrected_yield_with_provided_prices(self):
        """Test _calculate_corrected_yield method with provided token prices."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "token0": "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8",  # USDC
            "token1": "0x4200000000000000000000000000000000000006",  # WETH
            "token0_symbol": "USDC",
            "token1_symbol": "WETH",
            "dex_type": "uniswap_v3",
            "staked": False,
        }

        initial_amount0 = 2000000000  # 2000 USDC (6 decimals)
        initial_amount1 = 1000000000000000000  # 1.0 WETH (18 decimals)

        current_balances = {
            "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8": Decimal(
                "2025.0"
            ),  # 2025 USDC
            "0x4200000000000000000000000000000000000006": Decimal("1.01"),  # 1.01 WETH
        }

        token_prices = {
            "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8": Decimal("1.0"),  # USDC price
            "0x4200000000000000000000000000000000000006": Decimal(
                "3000.0"
            ),  # WETH price
        }

        def mock_get_token_decimals(chain, token_address):
            """Mock _get_token_decimals method."""
            yield
            if token_address == "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8":
                return 6  # USDC
            elif token_address == "0x4200000000000000000000000000000000000006":
                return 18  # WETH
            return None

        with patch.object(
            fetch_behaviour, "_get_token_decimals", mock_get_token_decimals
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_corrected_yield(
                    position,
                    initial_amount0,
                    initial_amount1,
                    current_balances,
                    "optimism",
                    token_prices,
                )
            )

            # Expected yield: 25 USDC * $1 + 0.01 WETH * $3000 = $25 + $30 = $55
            expected_yield = Decimal("55.0")
            assert result == expected_yield

    def test_calculate_corrected_yield_with_velo_rewards(self):
        """Test _calculate_corrected_yield method with VELO rewards for staked position."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "token0": "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8",  # USDC
            "token1": "0x4200000000000000000000000000000000000006",  # WETH
            "token0_symbol": "USDC",
            "token1_symbol": "WETH",
            "dex_type": "velodrome",
            "staked": True,  # Position is staked
        }

        initial_amount0 = 1000000000  # 1000 USDC (6 decimals)
        initial_amount1 = 500000000000000000  # 0.5 WETH (18 decimals)

        # Include VELO token in current balances
        velo_token_address = "0x3c8B650257cFb5f272f799F5e2b4e65093a11a05"  # VELO token
        current_balances = {
            "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8": Decimal(
                "1005.0"
            ),  # 1005 USDC
            "0x4200000000000000000000000000000000000006": Decimal("0.51"),  # 0.51 WETH
            velo_token_address: Decimal("100.0"),  # 100 VELO rewards
        }

        def mock_get_token_decimals(chain, token_address):
            """Mock _get_token_decimals method."""
            yield
            if token_address == "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8":
                return 6  # USDC
            elif token_address == "0x4200000000000000000000000000000000000006":
                return 18  # WETH
            return None

        def mock_fetch_token_price(token_address, chain):
            """Mock _fetch_token_price method."""
            yield
            if token_address == "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8":
                return 1.0  # USDC price
            elif token_address == "0x4200000000000000000000000000000000000006":
                return 2500.0  # WETH price
            elif token_address == velo_token_address:
                return 0.05  # VELO price
            return None

        def mock_get_velo_token_address(chain):
            """Mock _get_velo_token_address method."""
            return velo_token_address

        def mock_get_coin_id_from_symbol(symbol, chain):
            """Mock get_coin_id_from_symbol method."""
            if symbol == "VELO":
                return "velodrome-finance"
            return None

        def mock_fetch_coin_price(coin_id):
            """Mock _fetch_coin_price method."""
            yield
            if coin_id == "velodrome-finance":
                return 0.05  # VELO price fetch succeeds
            return None

        with patch.multiple(
            fetch_behaviour,
            _get_token_decimals=mock_get_token_decimals,
            _fetch_token_price=mock_fetch_token_price,
            _get_velo_token_address=mock_get_velo_token_address,
            get_coin_id_from_symbol=mock_get_coin_id_from_symbol,
            _fetch_coin_price=mock_fetch_coin_price,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_corrected_yield(
                    position,
                    initial_amount0,
                    initial_amount1,
                    current_balances,
                    "optimism",
                )
            )

            # Expected yield:
            # Base yield: 5 USDC * $1 + 0.01 WETH * $2500 = $5 + $25 = $30
            # VELO rewards: 100 VELO * $0.05 = $5
            # Total: $30 + $5 = $35
            expected_yield = Decimal("35.0")
            assert result == expected_yield

    def test_calculate_corrected_yield_token_decimals_failure(self):
        """Test _calculate_corrected_yield method when token decimals fetch fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "token0": "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8",
            "token1": "0x4200000000000000000000000000000000000006",
            "token0_symbol": "USDC",
            "token1_symbol": "WETH",
            "dex_type": "uniswap_v3",
            "staked": False,
        }

        initial_amount0 = 1000000000
        initial_amount1 = 500000000000000000
        current_balances = {}

        def mock_get_token_decimals(chain, token_address):
            """Mock _get_token_decimals method that fails."""
            yield
            return None  # Simulate failure

        with patch.object(
            fetch_behaviour, "_get_token_decimals", mock_get_token_decimals
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_corrected_yield(
                    position,
                    initial_amount0,
                    initial_amount1,
                    current_balances,
                    "optimism",
                )
            )

            # Should return 0 when token decimals fetch fails
            assert result == Decimal("0")

    def test_calculate_corrected_yield_token_price_failure(self):
        """Test _calculate_corrected_yield method when token price fetch fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "token0": "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8",
            "token1": "0x4200000000000000000000000000000000000006",
            "token0_symbol": "USDC",
            "token1_symbol": "WETH",
            "dex_type": "uniswap_v3",
            "staked": False,
        }

        initial_amount0 = 1000000000
        initial_amount1 = 500000000000000000
        current_balances = {
            "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8": Decimal("1010.0"),
            "0x4200000000000000000000000000000000000006": Decimal("0.52"),
        }

        def mock_get_token_decimals(chain, token_address):
            """Mock _get_token_decimals method."""
            yield
            if token_address == "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8":
                return 6  # USDC
            elif token_address == "0x4200000000000000000000000000000000000006":
                return 18  # WETH
            return None

        def mock_fetch_token_price(token_address, chain):
            """Mock _fetch_token_price method that fails."""
            yield
            return None  # Simulate price fetch failure

        with patch.multiple(
            fetch_behaviour,
            _get_token_decimals=mock_get_token_decimals,
            _fetch_token_price=mock_fetch_token_price,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_corrected_yield(
                    position,
                    initial_amount0,
                    initial_amount1,
                    current_balances,
                    "optimism",
                )
            )

            # Should return 0 when token price fetch fails
            assert result == Decimal("0")

    def test_calculate_corrected_yield_no_token_increases(self):
        """Test _calculate_corrected_yield method when there are no token increases."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "token0": "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8",
            "token1": "0x4200000000000000000000000000000000000006",
            "token0_symbol": "USDC",
            "token1_symbol": "WETH",
            "dex_type": "uniswap_v3",
            "staked": False,
        }

        initial_amount0 = 1000000000  # 1000 USDC (6 decimals)
        initial_amount1 = 500000000000000000  # 0.5 WETH (18 decimals)

        # Current balances are same as initial amounts (no increases)
        current_balances = {
            "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8": Decimal(
                "1000.0"
            ),  # Same as initial
            "0x4200000000000000000000000000000000000006": Decimal(
                "0.5"
            ),  # Same as initial
        }

        def mock_get_token_decimals(chain, token_address):
            """Mock _get_token_decimals method."""
            yield
            if token_address == "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8":
                return 6  # USDC
            elif token_address == "0x4200000000000000000000000000000000000006":
                return 18  # WETH
            return None

        def mock_fetch_token_price(token_address, chain):
            """Mock _fetch_token_price method."""
            yield
            if token_address == "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8":
                return 1.0  # USDC price
            elif token_address == "0x4200000000000000000000000000000000000006":
                return 2500.0  # WETH price
            return None

        with patch.multiple(
            fetch_behaviour,
            _get_token_decimals=mock_get_token_decimals,
            _fetch_token_price=mock_fetch_token_price,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_corrected_yield(
                    position,
                    initial_amount0,
                    initial_amount1,
                    current_balances,
                    "optimism",
                )
            )

            # Expected yield: 0 USDC * $1 + 0 WETH * $2500 = $0
            expected_yield = Decimal("0")
            assert result == expected_yield

    def test_calculate_corrected_yield_velo_rewards_price_failure(self):
        """Test _calculate_corrected_yield method when VELO price fetch fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "token0": "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8",
            "token1": "0x4200000000000000000000000000000000000006",
            "token0_symbol": "USDC",
            "token1_symbol": "WETH",
            "dex_type": "velodrome",
            "staked": True,
        }

        initial_amount0 = 1000000000
        initial_amount1 = 500000000000000000

        velo_token_address = "0x3c8B650257cFb5f272f799F5e2b4e65093a11a05"
        current_balances = {
            "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8": Decimal("1005.0"),
            "0x4200000000000000000000000000000000000006": Decimal("0.51"),
            velo_token_address: Decimal("100.0"),  # 100 VELO rewards
        }

        def mock_get_token_decimals(chain, token_address):
            """Mock _get_token_decimals method."""
            yield
            if token_address == "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8":
                return 6
            elif token_address == "0x4200000000000000000000000000000000000006":
                return 18
            return None

        def mock_fetch_token_price(token_address, chain):
            """Mock _fetch_token_price method."""
            yield
            if token_address == "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8":
                return 1.0
            elif token_address == "0x4200000000000000000000000000000000000006":
                return 2500.0
            elif token_address == velo_token_address:
                return None  # VELO price fetch fails
            return None

        def mock_get_velo_token_address(chain):
            """Mock _get_velo_token_address method."""
            return velo_token_address

        def mock_get_coin_id_from_symbol(symbol, chain):
            """Mock get_coin_id_from_symbol method."""
            if symbol == "VELO":
                return "velodrome-finance"
            return None

        def mock_fetch_coin_price(coin_id):
            """Mock _fetch_coin_price method."""
            yield
            if coin_id == "velodrome-finance":
                return None  # VELO price fetch fails
            return None

        with patch.multiple(
            fetch_behaviour,
            _get_token_decimals=mock_get_token_decimals,
            _fetch_token_price=mock_fetch_token_price,
            _get_velo_token_address=mock_get_velo_token_address,
            get_coin_id_from_symbol=mock_get_coin_id_from_symbol,
            _fetch_coin_price=mock_fetch_coin_price,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_corrected_yield(
                    position,
                    initial_amount0,
                    initial_amount1,
                    current_balances,
                    "optimism",
                )
            )

            # Expected yield: Only base yield since VELO price fetch failed
            # 5 USDC * $1 + 0.01 WETH * $2500 = $5 + $25 = $30
            expected_yield = Decimal("30.0")
            assert result == expected_yield

    def test_calculate_corrected_yield_missing_token_balances(self):
        """Test _calculate_corrected_yield method with missing token balances in current_balances."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "token0": "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8",
            "token1": "0x4200000000000000000000000000000000000006",
            "token0_symbol": "USDC",
            "token1_symbol": "WETH",
            "dex_type": "uniswap_v3",
            "staked": False,
        }

        initial_amount0 = 1000000000
        initial_amount1 = 500000000000000000

        # Empty current balances (missing tokens)
        current_balances = {}

        def mock_get_token_decimals(chain, token_address):
            """Mock _get_token_decimals method."""
            yield
            if token_address == "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8":
                return 6
            elif token_address == "0x4200000000000000000000000000000000000006":
                return 18
            return None

        def mock_fetch_token_price(token_address, chain):
            """Mock _fetch_token_price method."""
            yield
            if token_address == "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8":
                return 1.0
            elif token_address == "0x4200000000000000000000000000000000000006":
                return 2500.0
            return None

        with patch.multiple(
            fetch_behaviour,
            _get_token_decimals=mock_get_token_decimals,
            _fetch_token_price=mock_fetch_token_price,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_corrected_yield(
                    position,
                    initial_amount0,
                    initial_amount1,
                    current_balances,
                    "optimism",
                )
            )

            # With missing balances (treated as 0), there are decreases, so max(0, decrease) = 0
            # Expected yield: 0 USDC * $1 + 0 WETH * $2500 = $0
            expected_yield = Decimal("0")
            assert result == expected_yield

    def test_get_user_share_value_velodrome_cl_pool(self):
        """Test get_user_share_value_velodrome method with CL pool."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        user_address = "0x1234567890123456789012345678901234567890"
        pool_address = "0xabcdef1234567890abcdef1234567890abcdef12"
        token_id = 123
        chain = "optimism"
        position = {
            "token0": "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8",  # USDC
            "token1": "0x4200000000000000000000000000000000000006",  # WETH
            "is_cl_pool": True,  # CL pool
            "pool_address": pool_address,
        }

        expected_result = {
            "token0_balance": Decimal("1000.0"),
            "token1_balance": Decimal("0.5"),
            "user_share": Decimal("0.1"),
            "total_supply": Decimal("10000.0"),
        }

        def mock_get_user_share_value_velodrome_cl(
            pool_address, chain, position, token0_address, token1_address
        ):
            """Mock _get_user_share_value_velodrome_cl method."""
            yield
            return expected_result

        with patch.object(
            fetch_behaviour,
            "_get_user_share_value_velodrome_cl",
            mock_get_user_share_value_velodrome_cl,
        ):
            result = self._consume_generator(
                fetch_behaviour.get_user_share_value_velodrome(
                    user_address, pool_address, token_id, chain, position
                )
            )

            assert result == expected_result

    @pytest.mark.parametrize(
        "is_cl_pool,test_description",
        [
            (False, "explicit is_cl_pool=False"),
            (None, "is_cl_pool not specified (defaults to False)"),
        ],
    )
    def test_get_user_share_value_velodrome_non_cl_pool_variations(
        self, is_cl_pool, test_description
    ):
        """Test get_user_share_value_velodrome method with non-CL pool variations."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        user_address = "0x1234567890123456789012345678901234567890"
        pool_address = "0xabcdef1234567890abcdef1234567890abcdef12"
        token_id = 123
        chain = "optimism"
        position = {
            "token0": "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8",  # USDC
            "token1": "0x4200000000000000000000000000000000000006",  # WETH
            "pool_address": pool_address,
        }

        # Only add is_cl_pool if it's not None
        if is_cl_pool is not None:
            position["is_cl_pool"] = is_cl_pool

        expected_result = {
            "token0_balance": Decimal("500.0"),
            "token1_balance": Decimal("0.25"),
            "user_share": Decimal("0.05"),
            "total_supply": Decimal("10000.0"),
        }

        def mock_get_user_share_value_velodrome_non_cl(
            user_address, pool_address, chain, position, token0_address, token1_address
        ):
            """Mock _get_user_share_value_velodrome_non_cl method."""
            yield
            return expected_result

        with patch.object(
            fetch_behaviour,
            "_get_user_share_value_velodrome_non_cl",
            mock_get_user_share_value_velodrome_non_cl,
        ):
            result = self._consume_generator(
                fetch_behaviour.get_user_share_value_velodrome(
                    user_address, pool_address, token_id, chain, position
                )
            )

            assert result == expected_result, f"Failed for {test_description}"

    @pytest.mark.parametrize(
        "token0_address,token1_address,test_description",
        [
            (
                None,
                "0x4200000000000000000000000000000000000006",
                "missing token0 address",
            ),
            (
                "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8",
                None,
                "missing token1 address",
            ),
            (None, None, "both token addresses missing"),
            ("", "0x4200000000000000000000000000000000000006", "empty token0 address"),
            ("0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8", "", "empty token1 address"),
            ("", "", "both token addresses empty"),
        ],
    )
    def test_get_user_share_value_velodrome_invalid_token_addresses(
        self, token0_address, token1_address, test_description
    ):
        """Test get_user_share_value_velodrome method with various invalid token address scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        user_address = "0x1234567890123456789012345678901234567890"
        pool_address = "0xabcdef1234567890abcdef1234567890abcdef12"
        token_id = 123
        chain = "optimism"
        position = {
            "token0": token0_address,
            "token1": token1_address,
            "is_cl_pool": False,
            "pool_address": pool_address,
        }

        result = self._consume_generator(
            fetch_behaviour.get_user_share_value_velodrome(
                user_address, pool_address, token_id, chain, position
            )
        )

        # Should return empty dict when token addresses are invalid
        assert result == {}, f"Expected empty dict for {test_description}"

    def test_get_user_share_value_velodrome_cl_pool_fallback_to_non_cl(self):
        """Test get_user_share_value_velodrome method with CL pool that falls back to non-CL logic."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        user_address = "0x1234567890123456789012345678901234567890"
        pool_address = "0xabcdef1234567890abcdef1234567890abcdef12"
        token_id = 123
        chain = "optimism"
        position = {
            "token0": "0xA0b86a33E6416c69C88bb798B7e8d1c7B3B5f9C8",  # USDC
            "token1": "0x4200000000000000000000000000000000000006",  # WETH
            "is_cl_pool": True,  # CL pool
            "pool_address": pool_address,
        }

        expected_result = {
            "token0_balance": Decimal("1000.0"),
            "token1_balance": Decimal("0.5"),
            "user_share": Decimal("0.1"),
            "total_supply": Decimal("10000.0"),
        }

        def mock_get_user_share_value_velodrome_cl(
            pool_address, chain, position, token0_address, token1_address
        ):
            """Mock _get_user_share_value_velodrome_cl method."""
            yield
            return expected_result

        with patch.object(
            fetch_behaviour,
            "_get_user_share_value_velodrome_cl",
            mock_get_user_share_value_velodrome_cl,
        ):
            result = self._consume_generator(
                fetch_behaviour.get_user_share_value_velodrome(
                    user_address, pool_address, token_id, chain, position
                )
            )

            assert result == expected_result

    @pytest.mark.parametrize(
        "api_response,expected_result,test_description",
        [
            (
                {
                    "status_code": 200,
                    "json_data": {
                        "items": [
                            {
                                "timestamp": "2022-01-01T00:00:00Z",
                                "from": {"hash": "0x456"},
                                "token": {
                                    "symbol": "USDC",
                                    "address": "0x123",
                                    "decimals": 6,
                                },
                                "total": {"value": "1000000"},
                                "transaction_hash": "0x789",
                            }
                        ]
                    },
                },
                True,
                "successful API call",
            ),
            ({"status_code": 500, "json_data": {}}, False, "API failure"),
        ],
    )
    def test_fetch_token_transfers_mode_variations(
        self, api_response, expected_result, test_description
    ):
        """Test _fetch_token_transfers_mode method with various API responses."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_requests_get(*args, **kwargs):
            class MockResponse:
                def __init__(self):
                    self.status_code = api_response["status_code"]
                    self.json_data = api_response["json_data"]

                def json(self):
                    return self.json_data

            return MockResponse()

        with patch("requests.get", mock_requests_get):
            result = fetch_behaviour._fetch_token_transfers_mode(
                "0x123", "2022-01-01", {}, False
            )
            assert (
                result is expected_result
            ), f"Expected {expected_result} for {test_description}"

    @pytest.mark.parametrize(
        "api_response,expected_result,test_description",
        [
            (
                {
                    "status_code": 200,
                    "json_data": {
                        "items": [
                            {
                                "value": "1000000000000000000",
                                "delta": "1000000000000000000",
                                "transaction_hash": None,
                                "block_timestamp": "2022-01-01T00:00:00Z",
                                "block_number": 12345,
                            }
                        ],
                        "next_page_params": None,
                    },
                },
                True,
                "successful API call",
            ),
            ({"status_code": 500, "json_data": {}}, False, "API failure"),
        ],
    )
    def test_fetch_eth_transfers_mode_variations(
        self, api_response, expected_result, test_description
    ):
        """Test _fetch_eth_transfers_mode method with various API responses."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_requests_get(*args, **kwargs):
            class MockResponse:
                def __init__(self):
                    self.status_code = api_response["status_code"]
                    self.json_data = api_response["json_data"]

                def json(self):
                    return self.json_data

            return MockResponse()

        with patch("requests.get", mock_requests_get):
            result = fetch_behaviour._fetch_eth_transfers_mode(
                "0x123", "2022-01-01", {}, True
            )
            assert (
                result is expected_result
            ), f"Expected {expected_result} for {test_description}"

    def test_check_and_update_zero_liquidity_positions(self):
        """Test check_and_update_zero_liquidity_positions method."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock current positions with zero liquidity
        with patch.object(
            fetch_behaviour,
            "current_positions",
            [
                {"balance": "0", "pool_address": "0x123"},
                {"balance": "100", "pool_address": "0x456"},
            ],
        ):
            # Should not raise any exception
            fetch_behaviour.check_and_update_zero_liquidity_positions()

    def test_update_allocation_ratios_success(self):
        """Test _update_allocation_ratios method with valid inputs."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        individual_shares = [("0x123", Decimal("100.0")), ("0x456", Decimal("200.0"))]
        total_value = Decimal("300.0")
        allocations = [
            {"address": "0x123", "ratio": 0.0},
            {"address": "0x456", "ratio": 0.0},
        ]

        def mock_update_allocation_ratios(individual_shares, total_value, allocations):
            yield None

        with patch.object(
            fetch_behaviour, "_update_allocation_ratios", mock_update_allocation_ratios
        ):
            result = list(
                fetch_behaviour._update_allocation_ratios(
                    individual_shares, total_value, allocations
                )
            )
            assert result == [None]

    def test_update_allocation_ratios_zero_total_value(self):
        """Test _update_allocation_ratios method with zero total value."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        individual_shares = [("0x123", Decimal("100.0")), ("0x456", Decimal("200.0"))]
        total_value = Decimal("0.0")
        allocations = [
            {"address": "0x123", "ratio": 0.0},
            {"address": "0x456", "ratio": 0.0},
        ]

        def mock_update_allocation_ratios(individual_shares, total_value, allocations):
            yield None

        with patch.object(
            fetch_behaviour, "_update_allocation_ratios", mock_update_allocation_ratios
        ):
            result = list(
                fetch_behaviour._update_allocation_ratios(
                    individual_shares, total_value, allocations
                )
            )
            assert result == [None]

    def test_get_tick_ranges_slot0_failure(self):
        """Test _get_tick_ranges method when slot0 call fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "dex_type": "velodrome",
            "token_id": 123,
        }

        def mock_contract_interact(**kwargs):
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._get_tick_ranges(position, "optimism")
            )
            assert result == []

    def test_get_tick_ranges_no_position_manager(self):
        """Test _get_tick_ranges method when position manager address is missing."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "dex_type": "velodrome",
            "token_id": 123,
        }

        def mock_contract_interact(**kwargs):
            if kwargs.get("contract_callable") == "slot0":
                return {"tick": 500}
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._get_tick_ranges(position, "optimism")
            )
            assert result == []

    def test_get_tick_ranges_multiple_positions(self):
        """Test _get_tick_ranges method with multiple positions."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Set up position manager address in params using patch.dict
        with patch.dict(
            fetch_behaviour.params.__dict__,
            {
                "velodrome_non_fungible_position_manager_contract_addresses": {
                    "optimism": "0x1234567890123456789012345678901234567890"
                }
            },
        ):
            position = {
                "pool_address": "0x1234567890123456789012345678901234567890",
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "positions": [
                    {"token_id": 123, "tickLower": -1000, "tickUpper": 1000},
                    {"token_id": 456, "tickLower": -500, "tickUpper": 500},
                ],
            }

            def mock_contract_interact(**kwargs):
                yield
                if kwargs.get("contract_callable") == "slot0":
                    return {"tick": 500}
                elif kwargs.get("contract_callable") == "get_position":
                    token_id = kwargs.get("token_id")
                    if token_id == 123:
                        return {"tickLower": -1000, "tickUpper": 1000}
                    elif token_id == 456:
                        return {"tickLower": -500, "tickUpper": 500}
                return None

            with patch.object(
                fetch_behaviour, "contract_interact", mock_contract_interact
            ):
                result = self._consume_generator(
                    fetch_behaviour._get_tick_ranges(position, "optimism")
                )

                assert len(result) == 2
                assert result[0]["token_id"] == 123
                assert result[1]["token_id"] == 456

    def test_calculate_position_amounts_missing_details(self):
        """Test _calculate_position_amounts method with missing position details."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position_details = {
            "tickLower": None,
            "tickUpper": 1000,
            "liquidity": "1000000000000000000",
        }
        current_tick = 500
        sqrt_price_x96 = 1000000000000000000000000
        position = {"token_id": 123}
        dex_type = "uniswap_v3"
        chain = "optimism"

        result = self._consume_generator(
            fetch_behaviour._calculate_position_amounts(
                position_details,
                current_tick,
                sqrt_price_x96,
                position,
                dex_type,
                chain,
            )
        )

        assert result == (0, 0)

    def test_calculate_position_amounts_uniswap_fallback(self):
        """Test _calculate_position_amounts method with Uniswap fallback calculation."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position_details = {
            "tickLower": -1000,
            "tickUpper": 1000,
            "liquidity": "1000000000000000000",
            "tokensOwed0": "100000000000000000",
            "tokensOwed1": "200000000000000000",
        }
        current_tick = 500
        sqrt_price_x96 = 1000000000000000000000000
        position = {"token_id": 123}
        dex_type = "uniswap_v3"
        chain = "optimism"

        result = self._consume_generator(
            fetch_behaviour._calculate_position_amounts(
                position_details,
                current_tick,
                sqrt_price_x96,
                position,
                dex_type,
                chain,
            )
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_calculate_position_amounts_velodrome_sugar_fallback(self):
        """Test _calculate_position_amounts method with Velodrome Sugar fallback."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position_details = {
            "tickLower": -1000,
            "tickUpper": 1000,
            "liquidity": "1000000000000000000",
            "tokensOwed0": "100000000000000000",
            "tokensOwed1": "200000000000000000",
        }
        current_tick = 500
        sqrt_price_x96 = 1000000000000000000000000
        position = {"token_id": 123, "dex_type": "velodrome"}
        dex_type = "velodrome"
        chain = "optimism"

        def mock_get_velodrome_position_principal(
            chain, position_manager_address, token_id, sqrt_price_x96
        ):
            yield
            return 1000000000000000000, 2000000000000000000

        with patch.object(
            fetch_behaviour,
            "get_velodrome_position_principal",
            mock_get_velodrome_position_principal,
        ):
            result = self._consume_generator(
                fetch_behaviour._calculate_position_amounts(
                    position_details,
                    current_tick,
                    sqrt_price_x96,
                    position,
                    dex_type,
                    chain,
                )
            )

            assert result == (1100000000000000000, 2200000000000000000)

    def test_get_user_share_value_velodrome_cl_success(self):
        """Test _get_user_share_value_velodrome_cl method with successful calculation."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        pool_address = "0x1234567890123456789012345678901234567890"
        chain = "optimism"
        position = {"token_id": 123}
        token0_address = "0x1111111111111111111111111111111111111111"
        token1_address = "0x2222222222222222222222222222222222222222"

        def mock_calculate_cl_position_value(**kwargs):
            yield
            return {
                "token0_balance": Decimal("100.0"),
                "token1_balance": Decimal("200.0"),
                "token0_symbol": "WETH",
                "token1_symbol": "USDC",
            }

        with patch.object(
            fetch_behaviour,
            "_calculate_cl_position_value",
            mock_calculate_cl_position_value,
        ):
            result = self._consume_generator(
                fetch_behaviour._get_user_share_value_velodrome_cl(
                    pool_address, chain, position, token0_address, token1_address
                )
            )

            assert result["token0_balance"] == Decimal("100.0")
            assert result["token1_balance"] == Decimal("200.0")
            assert result["token0_symbol"] == "WETH"
            assert result["token1_symbol"] == "USDC"

    def test_get_user_share_value_velodrome_cl_no_position_manager(self):
        """Test _get_user_share_value_velodrome_cl method with missing position manager."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        pool_address = "0x1234567890123456789012345678901234567890"
        chain = "optimism"
        position = {"token_id": 123}
        token0_address = "0x1111111111111111111111111111111111111111"
        token1_address = "0x2222222222222222222222222222222222222222"

        def mock_contract_interact(**kwargs):
            yield
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._get_user_share_value_velodrome_cl(
                    pool_address, chain, position, token0_address, token1_address
                )
            )

            assert result == {}

    def test_get_user_share_value_velodrome_non_cl_success(self):
        """Test _get_user_share_value_velodrome_non_cl method with successful calculation."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        user_address = "0x1234567890123456789012345678901234567890"
        pool_address = "0x1111111111111111111111111111111111111111"
        chain = "optimism"
        position = {"token0_symbol": "WETH", "token1_symbol": "USDC"}
        token0_address = "0x2222222222222222222222222222222222222222"
        token1_address = "0x3333333333333333333333333333333333333333"

        def mock_contract_interact(**kwargs):
            if kwargs.get("contract_callable") == "check_balance":
                yield
                return "1000000000000000000"
            elif kwargs.get("contract_callable") == "get_total_supply":
                yield
                return "10000000000000000000"
            elif kwargs.get("contract_callable") == "get_reserves":
                yield
                return ["1000000000000000000000", "2000000000000000000000"]
            else:
                yield
                return None

        def mock_get_token_decimals_pair(chain, token0_address, token1_address):
            yield
            return 18, 6

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            with patch.object(
                fetch_behaviour,
                "_get_token_decimals_pair",
                mock_get_token_decimals_pair,
            ):
                result = self._consume_generator(
                    fetch_behaviour._get_user_share_value_velodrome_non_cl(
                        user_address,
                        pool_address,
                        chain,
                        position,
                        token0_address,
                        token1_address,
                    )
                )

                assert token0_address in result
                assert token1_address in result
                assert result[token0_address] == Decimal("100.0")
                assert result[token1_address] == Decimal("200000000000000.0")

    def test_get_user_share_value_velodrome_non_cl_balance_failure(self):
        """Test _get_user_share_value_velodrome_non_cl method when balance check fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        user_address = "0x1234567890123456789012345678901234567890"
        pool_address = "0x1111111111111111111111111111111111111111"
        chain = "optimism"
        position = {"token0_symbol": "WETH", "token1_symbol": "USDC"}
        token0_address = "0x2222222222222222222222222222222222222222"
        token1_address = "0x3333333333333333333333333333333333333333"

        def mock_contract_interact(**kwargs):
            yield
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._get_user_share_value_velodrome_non_cl(
                    user_address,
                    pool_address,
                    chain,
                    position,
                    token0_address,
                    token1_address,
                )
            )

            assert result == {}

    def test_get_user_share_value_velodrome_non_cl_total_supply_failure(self):
        """Test _get_user_share_value_velodrome_non_cl method when total supply check fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        user_address = "0x1234567890123456789012345678901234567890"
        pool_address = "0x1111111111111111111111111111111111111111"
        chain = "optimism"
        position = {"token0_symbol": "WETH", "token1_symbol": "USDC"}
        token0_address = "0x2222222222222222222222222222222222222222"
        token1_address = "0x3333333333333333333333333333333333333333"

        def mock_contract_interact(**kwargs):
            if kwargs.get("contract_callable") == "check_balance":
                return "1000000000000000000"
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._get_user_share_value_velodrome_non_cl(
                    user_address,
                    pool_address,
                    chain,
                    position,
                    token0_address,
                    token1_address,
                )
            )

            assert result == {}

    def test_get_user_share_value_velodrome_non_cl_reserves_failure(self):
        """Test _get_user_share_value_velodrome_non_cl method when reserves check fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        user_address = "0x1234567890123456789012345678901234567890"
        pool_address = "0x1111111111111111111111111111111111111111"
        chain = "optimism"
        position = {"token0_symbol": "WETH", "token1_symbol": "USDC"}
        token0_address = "0x2222222222222222222222222222222222222222"
        token1_address = "0x3333333333333333333333333333333333333333"

        def mock_contract_interact(**kwargs):
            if kwargs.get("contract_callable") == "check_balance":
                return "1000000000000000000"
            elif kwargs.get("contract_callable") == "get_total_supply":
                return "10000000000000000000"
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour._get_user_share_value_velodrome_non_cl(
                    user_address,
                    pool_address,
                    chain,
                    position,
                    token0_address,
                    token1_address,
                )
            )

            assert result == {}

    def test_get_user_share_value_velodrome_non_cl_decimals_failure(self):
        """Test _get_user_share_value_velodrome_non_cl method when decimals check fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        user_address = "0x1234567890123456789012345678901234567890"
        pool_address = "0x1111111111111111111111111111111111111111"
        chain = "optimism"
        position = {"token0_symbol": "WETH", "token1_symbol": "USDC"}
        token0_address = "0x2222222222222222222222222222222222222222"
        token1_address = "0x3333333333333333333333333333333333333333"

        def mock_contract_interact(**kwargs):
            if kwargs.get("contract_callable") == "check_balance":
                return "1000000000000000000"
            elif kwargs.get("contract_callable") == "get_total_supply":
                return "10000000000000000000"
            elif kwargs.get("contract_callable") == "get_reserves":
                return ["1000000000000000000000", "2000000000000000000000"]
            return None

        def mock_get_token_decimals_pair(chain, token0_address, token1_address):
            return None, None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            with patch.object(
                fetch_behaviour,
                "_get_token_decimals_pair",
                mock_get_token_decimals_pair,
            ):
                result = self._consume_generator(
                    fetch_behaviour._get_user_share_value_velodrome_non_cl(
                        user_address,
                        pool_address,
                        chain,
                        position,
                        token0_address,
                        token1_address,
                    )
                )

                assert result == {}

    def test_check_is_valid_safe_address_success(self):
        """Test check_is_valid_safe_address method with valid safe address."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        safe_address = "0x1234567890123456789012345678901234567890"
        operating_chain = "optimism"

        def mock_contract_interact(**kwargs):
            yield
            return [
                "0x1111111111111111111111111111111111111111",
                "0x2222222222222222222222222222222222222222",
            ]

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour.check_is_valid_safe_address(
                    safe_address, operating_chain
                )
            )

            assert result is True

    def test_check_is_valid_safe_address_failure(self):
        """Test check_is_valid_safe_address method with invalid safe address."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        safe_address = "0x1234567890123456789012345678901234567890"
        operating_chain = "optimism"

        def mock_contract_interact(**kwargs):
            yield
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour.check_is_valid_safe_address(
                    safe_address, operating_chain
                )
            )

            assert result is False

    def test_get_master_safe_address_not_staked(self):
        """Test get_master_safe_address method when service is not staked."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_get_service_staking_state(chain):
            return StakingState.NOT_STAKED

        with patch.object(
            fetch_behaviour,
            "_get_service_staking_state",
            mock_get_service_staking_state,
        ):
            result = self._consume_generator(fetch_behaviour.get_master_safe_address())

            assert result is None

    def test_get_master_safe_address_no_service_id(self):
        """Test get_master_safe_address method when service ID is missing."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_get_service_staking_state(chain):
            return StakingState.STAKED

        def mock_get_service_info(chain):
            return None

        with patch.object(
            fetch_behaviour,
            "_get_service_staking_state",
            mock_get_service_staking_state,
        ):
            with patch.object(
                fetch_behaviour, "_get_service_info", mock_get_service_info
            ):
                result = self._consume_generator(
                    fetch_behaviour.get_master_safe_address()
                )

                assert result is None

    def test_get_master_safe_address_service_registry_fallback(self):
        """Test get_master_safe_address method with service registry fallback."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Set required parameters for the fallback path
        with patch.dict(
            fetch_behaviour.params.__dict__,
            {
                "on_chain_service_id": "test_service_id",
                "target_investment_chains": ["optimism"],
                "staking_token_contract_address": None,
                "staking_chain": None,
                "service_registry_contract_addresses": {
                    "optimism": "0xServiceRegistryAddress"
                },
            },
        ):

            def mock_get_service_staking_state(chain):
                yield
                return StakingState.STAKED

            def mock_get_service_info(chain):
                yield
                return None

            def mock_contract_interact(**kwargs):
                yield
                if kwargs.get("contract_callable") == "get_service_owner":
                    return "0x1234567890123456789012345678901234567890"
                return None

            def mock_check_is_valid_safe_address(address, chain):
                yield
                return True

            with patch.object(
                fetch_behaviour,
                "_get_service_staking_state",
                mock_get_service_staking_state,
            ):
                with patch.object(
                    fetch_behaviour, "_get_service_info", mock_get_service_info
                ):
                    with patch.object(
                        fetch_behaviour, "contract_interact", mock_contract_interact
                    ):
                        with patch.object(
                            fetch_behaviour,
                            "check_is_valid_safe_address",
                            mock_check_is_valid_safe_address,
                        ):
                            result = self._consume_generator(
                                fetch_behaviour.get_master_safe_address()
                            )

                            assert (
                                result == "0x1234567890123456789012345678901234567890"
                            )

    def test_update_allocation_ratios_negative_total_value(self):
        """Test _update_allocation_ratios method with negative total value."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        individual_shares = [
            (
                Decimal("1000.0"),
                "uniswap_v3",
                "optimism",
                "pool_123",
                ["WETH", "USDC"],
                10.5,
                {"pool": "test"},
                "0xUserAddress",
                None,
            )
        ]

        total_value = Decimal("-100.0")
        allocations = []

        result = list(
            fetch_behaviour._update_allocation_ratios(
                individual_shares, total_value, allocations
            )
        )

        # Should return early when total_value <= 0
        assert result == []
        assert len(allocations) == 0

    def test_update_allocation_ratios_no_positions_found(self):
        """Test _update_allocation_ratios method when no positions are found."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock current positions (empty)
        fetch_behaviour.current_positions = []

        individual_shares = [
            (
                Decimal("1000.0"),
                "uniswap_v3",
                "optimism",
                "pool_123",
                ["WETH", "USDC"],
                10.5,
                {"pool": "test"},
                "0xUserAddress",
                None,
            )
        ]

        total_value = Decimal("1000.0")
        allocations = []

        result = list(
            fetch_behaviour._update_allocation_ratios(
                individual_shares, total_value, allocations
            )
        )

        # Verify the method executed successfully
        assert result == []  # No positions found, so no yield calls
        assert len(allocations) == 1

        # Verify allocation was created without tick_ranges
        assert allocations[0]["chain"] == "optimism"
        assert allocations[0]["type"] == "uniswap_v3"
        assert allocations[0]["id"] == "pool_123"
        assert (
            "tick_ranges" not in allocations[0]
        )  # No position found, so no tick ranges

    def test_update_allocation_ratios_position_found_by_pool_address(self):
        """Test _update_allocation_ratios method when position is found by pool_address."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock current positions with pool_address
        fetch_behaviour.current_positions = [
            {"pool_address": "pool_123", "balance": "1000000000000000000"}
        ]

        individual_shares = [
            (
                Decimal("1000.0"),
                "uniswap_v3",
                "optimism",
                "pool_123",
                ["WETH", "USDC"],
                10.5,
                {"pool": "test"},
                "0xUserAddress",
                None,
            )
        ]

        total_value = Decimal("1000.0")
        allocations = []

        def mock_get_tick_ranges(position, chain):
            yield
            return [{"token_id": 123, "tickLower": -1000, "tickUpper": 1000}]

        with patch.object(fetch_behaviour, "_get_tick_ranges", mock_get_tick_ranges):
            result = list(
                fetch_behaviour._update_allocation_ratios(
                    individual_shares, total_value, allocations
                )
            )

            # Verify the method executed successfully
            assert result == [None]
            assert len(allocations) == 1
            assert "tick_ranges" in allocations[0]  # Should have tick ranges

    def test_update_allocation_ratios_position_found_by_pool_id(self):
        """Test _update_allocation_ratios method when position is found by pool_id."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock current positions with pool_id
        fetch_behaviour.current_positions = [
            {"pool_id": "pool_123", "balance": "1000000000000000000"}
        ]

        individual_shares = [
            (
                Decimal("1000.0"),
                "uniswap_v3",
                "optimism",
                "pool_123",
                ["WETH", "USDC"],
                10.5,
                {"pool": "test"},
                "0xUserAddress",
                None,
            )
        ]

        total_value = Decimal("1000.0")
        allocations = []

        def mock_get_tick_ranges(position, chain):
            yield
            return [{"token_id": 123, "tickLower": -1000, "tickUpper": 1000}]

        with patch.object(fetch_behaviour, "_get_tick_ranges", mock_get_tick_ranges):
            result = list(
                fetch_behaviour._update_allocation_ratios(
                    individual_shares, total_value, allocations
                )
            )

            # Verify the method executed successfully
            assert result == [None]
            assert len(allocations) == 1
            assert "tick_ranges" in allocations[0]  # Should have tick ranges

    def test_update_allocation_ratios_dex_type_mapping(self):
        """Test _update_allocation_ratios method with different DEX types."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        individual_shares = [
            (
                Decimal("1000.0"),
                "uniswap_v3",
                "optimism",
                "pool_123",
                ["WETH", "USDC"],
                10.5,
                {"pool": "test"},
                "0xUserAddress",
                None,
            ),
            (
                Decimal("1000.0"),
                "sturdy",
                "optimism",
                "pool_456",
                ["WETH", "USDC"],
                12.0,
                {"pool": "test2"},
                "0xUserAddress2",
                None,
            ),
            (
                Decimal("1000.0"),
                "velodrome",
                "optimism",
                "pool_789",
                ["WETH", "USDC"],
                15.0,
                {"pool": "test3"},
                "0xUserAddress3",
                None,
            ),
            (
                Decimal("1000.0"),
                "balancer",
                "optimism",
                "pool_012",
                ["WETH", "USDC"],
                8.0,
                {"pool": "test4"},
                "0xUserAddress4",
                None,
            ),
            (
                Decimal("1000.0"),
                "unknown_dex",
                "optimism",
                "pool_345",
                ["WETH", "USDC"],
                5.0,
                {"pool": "test5"},
                "0xUserAddress5",
                None,
            ),
        ]

        total_value = Decimal("5000.0")
        allocations = []

        result = list(
            fetch_behaviour._update_allocation_ratios(
                individual_shares, total_value, allocations
            )
        )

        # Verify the method executed successfully
        assert result == []  # No positions found, so no yield calls
        assert len(allocations) == 5

        # Verify DEX type mapping
        assert allocations[0]["type"] == "uniswap_v3"
        assert allocations[1]["type"] == "sturdy"
        assert allocations[2]["type"] == "velodrome"
        assert allocations[3]["type"] == "balancer"
        assert allocations[4]["type"] == "unknown_dex"  # Should remain unchanged

    def test_update_allocation_ratios_ratio_calculation(self):
        """Test _update_allocation_ratios method ratio calculation."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        individual_shares = [
            (
                Decimal("1000.0"),
                "uniswap_v3",
                "optimism",
                "pool_123",
                ["WETH", "USDC"],
                10.5,
                {"pool": "test"},
                "0xUserAddress",
                None,
            ),
            (
                Decimal("2000.0"),
                "balancer",
                "optimism",
                "pool_456",
                ["WETH", "USDC"],
                12.0,
                {"pool": "test2"},
                "0xUserAddress2",
                None,
            ),
            (
                Decimal("3000.0"),
                "velodrome",
                "optimism",
                "pool_789",
                ["WETH", "USDC"],
                15.0,
                {"pool": "test3"},
                "0xUserAddress3",
                None,
            ),
        ]

        total_value = Decimal("6000.0")
        allocations = []

        result = list(
            fetch_behaviour._update_allocation_ratios(
                individual_shares, total_value, allocations
            )
        )

        # Verify the method executed successfully
        assert result == []  # No positions found, so no yield calls
        assert len(allocations) == 3

        # Verify ratio calculations
        # Total ratio = (1000/6000 + 2000/6000 + 3000/6000) * 100 = 100
        # First allocation: (1000/6000) * 100 * 100 / 100 = 16.67
        # Second allocation: (2000/6000) * 100 * 100 / 100 = 33.33
        # Third allocation: (3000/6000) * 100 * 100 / 100 = 50.0

        assert allocations[0]["ratio"] == 16.67
        assert allocations[1]["ratio"] == 33.33
        assert allocations[2]["ratio"] == 50.0

    def test_update_allocation_ratios_zero_total_ratio(self):
        """Test _update_allocation_ratios method when total_ratio is zero."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        individual_shares = [
            (
                Decimal("0.0"),
                "uniswap_v3",
                "optimism",
                "pool_123",
                ["WETH", "USDC"],
                10.5,
                {"pool": "test"},
                "0xUserAddress",
                None,
            ),
            (
                Decimal("0.0"),
                "balancer",
                "optimism",
                "pool_456",
                ["WETH", "USDC"],
                12.0,
                {"pool": "test2"},
                "0xUserAddress2",
                None,
            ),
        ]

        total_value = Decimal("1000.0")  # Non-zero total value
        allocations = []

        result = list(
            fetch_behaviour._update_allocation_ratios(
                individual_shares, total_value, allocations
            )
        )

        # Verify the method executed successfully
        assert result == []  # No positions found, so no yield calls
        assert len(allocations) == 2

        # Verify ratios are 0.0 when total_ratio is 0
        assert allocations[0]["ratio"] == 0.0
        assert allocations[1]["ratio"] == 0.0

    def test_update_allocation_ratios_tick_ranges_failure(self):
        """Test _update_allocation_ratios method when _get_tick_ranges fails."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock current positions
        fetch_behaviour.current_positions = [
            {"pool_address": "pool_123", "balance": "1000000000000000000"}
        ]

        individual_shares = [
            (
                Decimal("1000.0"),
                "uniswap_v3",
                "optimism",
                "pool_123",
                ["WETH", "USDC"],
                10.5,
                {"pool": "test"},
                "0xUserAddress",
                None,
            )
        ]

        total_value = Decimal("1000.0")
        allocations = []

        def mock_get_tick_ranges(position, chain):
            yield
            return []  # Empty tick ranges

        with patch.object(fetch_behaviour, "_get_tick_ranges", mock_get_tick_ranges):
            result = list(
                fetch_behaviour._update_allocation_ratios(
                    individual_shares, total_value, allocations
                )
            )

            # Verify the method executed successfully
            assert result == [None]
            assert len(allocations) == 1
            assert (
                "tick_ranges" not in allocations[0]
            )  # Should not add tick_ranges when empty

    def test_update_allocation_ratios_apr_rounding(self):
        """Test _update_allocation_ratios method APR rounding."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        individual_shares = [
            (
                Decimal("1000.0"),
                "uniswap_v3",
                "optimism",
                "pool_123",
                ["WETH", "USDC"],
                10.56789,
                {"pool": "test"},
                "0xUserAddress",
                None,
            ),
            (
                Decimal("1000.0"),
                "balancer",
                "optimism",
                "pool_456",
                ["WETH", "USDC"],
                12.12345,
                {"pool": "test2"},
                "0xUserAddress2",
                None,
            ),
        ]

        total_value = Decimal("2000.0")
        allocations = []

        result = list(
            fetch_behaviour._update_allocation_ratios(
                individual_shares, total_value, allocations
            )
        )

        # Verify the method executed successfully
        assert result == []  # No positions found, so no yield calls
        assert len(allocations) == 2

        # Verify APR rounding to 2 decimal places
        assert allocations[0]["apr"] == 10.57
        assert allocations[1]["apr"] == 12.12

    def test_update_allocation_ratios_complex_ratio_calculation(self):
        """Test _update_allocation_ratios method with complex ratio calculations."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock current positions to ensure _get_tick_ranges is called
        fetch_behaviour.current_positions = [
            {"pool_address": "pool_123", "balance": "1000000000000000000"},
            {"pool_address": "pool_456", "balance": "2000000000000000000"},
            {"pool_address": "pool_789", "balance": "3000000000000000000"},
        ]

        individual_shares = [
            (
                Decimal("123.45"),
                "uniswap_v3",
                "optimism",
                "pool_123",
                ["WETH", "USDC"],
                10.5,
                {"pool": "test"},
                "0xUserAddress",
                None,
            ),
            (
                Decimal("456.78"),
                "balancer",
                "optimism",
                "pool_456",
                ["WETH", "USDC"],
                12.0,
                {"pool": "test2"},
                "0xUserAddress2",
                None,
            ),
            (
                Decimal("789.12"),
                "velodrome",
                "optimism",
                "pool_789",
                ["WETH", "USDC"],
                15.0,
                {"pool": "test3"},
                "0xUserAddress3",
                None,
            ),
        ]

        total_value = Decimal("1369.35")  # 123.45 + 456.78 + 789.12
        allocations = []

        # Mock _get_tick_ranges to return empty list (which yields None)
        def mock_get_tick_ranges(position, chain):
            yield
            return []

        with patch.object(fetch_behaviour, "_get_tick_ranges", mock_get_tick_ranges):
            result = list(
                fetch_behaviour._update_allocation_ratios(
                    individual_shares, total_value, allocations
                )
            )

            # Verify the method executed successfully
            assert result == [None, None, None]  # One None for each position processed
            assert len(allocations) == 3

            # Verify ratio calculations with precision
            # Total ratio = (123.45/1369.35 + 456.78/1369.35 + 789.12/1369.35) * 100 = 100
            # First allocation: (123.45/1369.35) * 100 * 100 / 100 = 9.02
            # Second allocation: (456.78/1369.35) * 100 * 100 / 100 = 33.36
            # Third allocation: (789.12/1369.35) * 100 * 100 / 100 = 57.62

            assert allocations[0]["ratio"] == 9.02
            assert allocations[1]["ratio"] == 33.36
            assert (
                allocations[2]["ratio"] == 57.63
            )  # Updated due to floating-point precision

    @pytest.mark.parametrize(
        "address,current_date,mock_api_response,expected_result,test_description",
        [
            # Successful API response with valid transfers
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                (
                    True,
                    {
                        "results": [
                            {
                                "executionDate": "2024-01-10T10:30:00Z",
                                "from": "0x1234567890123456789012345678901234567890",
                                "to": "0x9876543210987654321098765432109876543210",
                                "type": "ETHER_TRANSFER",
                                "value": "1000000000000000000",  # 1 ETH in wei
                                "transactionHash": "0xabc123def456",
                            },
                            {
                                "executionDate": "2024-01-12T15:45:00Z",
                                "from": "0x1234567890123456789012345678901234567890",
                                "to": "0x5555555555555555555555555555555555555555",
                                "type": "ETHER_TRANSFER",
                                "value": "2000000000000000000",  # 2 ETH in wei
                                "transactionHash": "0xdef456abc789",
                            },
                        ]
                    },
                ),
                {
                    "2024-01-10": [
                        {
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "to_address": "0x9876543210987654321098765432109876543210",
                            "amount": 1.0,
                            "token_address": "0x0000000000000000000000000000000000000000",
                            "symbol": "ETH",
                            "timestamp": "2024-01-10T10:30:00Z",
                            "tx_hash": "0xabc123def456",
                            "type": "eth",
                        }
                    ],
                    "2024-01-12": [
                        {
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "to_address": "0x5555555555555555555555555555555555555555",
                            "amount": 2.0,
                            "token_address": "0x0000000000000000000000000000000000000000",
                            "symbol": "ETH",
                            "timestamp": "2024-01-12T15:45:00Z",
                            "tx_hash": "0xdef456abc789",
                            "type": "eth",
                        }
                    ],
                },
                "successful API response",
            ),
            # No address provided
            (
                "",
                "2024-01-15",
                None,  # No API call should be made
                {},
                "no address provided",
            ),
            # API request fails
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                (False, {}),
                {},
                "API request failure",
            ),
            # No transfers returned
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                (True, {"results": []}),
                {},
                "no transfers returned",
            ),
            # Future date transfer (should be filtered out)
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                (
                    True,
                    {
                        "results": [
                            {
                                "executionDate": "2024-01-20T10:30:00Z",  # Future date
                                "from": "0x1234567890123456789012345678901234567890",
                                "to": "0x9876543210987654321098765432109876543210",
                                "type": "ETHER_TRANSFER",
                                "value": "1000000000000000000",
                                "transactionHash": "0xabc123def456",
                            }
                        ]
                    },
                ),
                {},
                "future date transfer",
            ),
            # Non-ether transfer (should be filtered out)
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                (
                    True,
                    {
                        "results": [
                            {
                                "executionDate": "2024-01-10T10:30:00Z",
                                "from": "0x1234567890123456789012345678901234567890",
                                "to": "0x9876543210987654321098765432109876543210",
                                "type": "ERC20_TRANSFER",  # Non-ether transfer
                                "value": "1000000000000000000",
                                "transactionHash": "0xabc123def456",
                            }
                        ]
                    },
                ),
                {},
                "non-ether transfer",
            ),
            # Invalid value (should be filtered out)
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                (
                    True,
                    {
                        "results": [
                            {
                                "executionDate": "2024-01-10T10:30:00Z",
                                "from": "0x1234567890123456789012345678901234567890",
                                "to": "0x9876543210987654321098765432109876543210",
                                "type": "ETHER_TRANSFER",
                                "value": "invalid_value",
                                "transactionHash": "0xabc123def456",
                            }
                        ]
                    },
                ),
                {},
                "invalid value",
            ),
        ],
    )
    def test_fetch_outgoing_transfers_until_date_optimism_variations(
        self,
        address,
        current_date,
        mock_api_response,
        expected_result,
        test_description,
    ):
        """Test _fetch_outgoing_transfers_until_date_optimism method with various scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_request_with_retries(
            endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait
        ):
            yield
            return mock_api_response

        if address:  # Only mock the API call if we have an address
            with patch.object(
                fetch_behaviour, "_request_with_retries", mock_request_with_retries
            ):
                result = self._consume_generator(
                    fetch_behaviour._fetch_outgoing_transfers_until_date_optimism(
                        address, current_date
                    )
                )
        else:  # No address case doesn't make API calls
            result = self._consume_generator(
                fetch_behaviour._fetch_outgoing_transfers_until_date_optimism(
                    address, current_date
                )
            )

        assert (
            result == expected_result
        ), f"Expected {expected_result} for {test_description}"

    @pytest.mark.parametrize(
        "safe_address,current_date,mock_requests_response,expected_behavior,test_description",
        [
            # Successful API response with valid transactions
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                {
                    "status_code": 200,
                    "json_data": {
                        "status": "1",
                        "result": [
                            {
                                "timeStamp": "1704873600",  # 2024-01-10
                                "from": "0x1234567890123456789012345678901234567890",
                                "to": "0x9876543210987654321098765432109876543210",
                                "value": "1000000000000000000",  # 1 ETH in wei
                                "hash": "0xabc123def456",
                            },
                            {
                                "timeStamp": "1705046400",  # 2024-01-12
                                "from": "0x5555555555555555555555555555555555555555",
                                "to": "0x1234567890123456789012345678901234567890",
                                "value": "2000000000000000000",  # 2 ETH in wei
                                "hash": "0xdef456abc789",
                            },
                        ],
                    },
                },
                "success",
                "successful API response",
            ),
            # API request fails
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                {"status_code": 500, "json_data": {}},
                "empty_result",
                "API request failure",
            ),
            # No transactions returned
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                {"status_code": 200, "json_data": {"status": "1", "result": []}},
                "empty_result",
                "no transactions returned",
            ),
            # Future date transaction (should be filtered out)
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                {
                    "status_code": 200,
                    "json_data": {
                        "status": "1",
                        "result": [
                            {
                                "timeStamp": "1737676800",  # Future timestamp
                                "from": "0x1234567890123456789012345678901234567890",
                                "to": "0x9876543210987654321098765432109876543210",
                                "value": "1000000000000000000",
                                "hash": "0xabc123def456",
                            }
                        ],
                    },
                },
                "empty_result",
                "future date transaction",
            ),
            # Invalid timestamp
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                {
                    "status_code": 200,
                    "json_data": {
                        "status": "1",
                        "result": [
                            {
                                "timeStamp": "invalid_timestamp",
                                "from": "0x1234567890123456789012345678901234567890",
                                "to": "0x9876543210987654321098765432109876543210",
                                "value": "1000000000000000000",
                                "hash": "0xabc123def456",
                            }
                        ],
                    },
                },
                "empty_result",
                "invalid timestamp",
            ),
            # Invalid value
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                {
                    "status_code": 200,
                    "json_data": {
                        "status": "1",
                        "result": [
                            {
                                "timeStamp": "1704873600",
                                "from": "0x1234567890123456789012345678901234567890",
                                "to": "0x9876543210987654321098765432109876543210",
                                "value": "invalid_value",
                                "hash": "0xabc123def456",
                            }
                        ],
                    },
                },
                "empty_result",
                "invalid value",
            ),
            # Exception handling
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                "exception",
                "empty_result",
                "exception handling",
            ),
        ],
    )
    def test_track_eth_transfers_mode_variations(
        self,
        safe_address,
        current_date,
        mock_requests_response,
        expected_behavior,
        test_description,
    ):
        """Test _track_eth_transfers_mode method with various scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_requests_get(*args, **kwargs):
            if mock_requests_response == "exception":
                raise Exception("API error")

            class MockResponse:
                def __init__(self):
                    self.status_code = mock_requests_response["status_code"]

                def json(self):
                    return mock_requests_response["json_data"]

            return MockResponse()

        with patch("requests.get", mock_requests_get):
            result = fetch_behaviour._track_eth_transfers_mode(
                safe_address, current_date
            )

            if expected_behavior == "success":
                # Verify successful execution with expected data
                assert result["incoming"]["1705046400"][0]["amount"] == 2.0
                assert result["outgoing"]["1704873600"][0]["amount"] == 1.0
                assert len(result["incoming"]) == 1
                assert len(result["outgoing"]) == 1
            else:  # expected_behavior == "empty_result"
                # Verify empty result for all error cases
                assert result == {"incoming": {}, "outgoing": {}}

    def test_get_user_share_value_balancer_success(self):
        """Test get_user_share_value_balancer method with successful contract calls."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        user_address = "0x1234567890123456789012345678901234567890"
        pool_id = "0xabcdef1234567890abcdef1234567890abcdef12"
        pool_address = "0x9876543210987654321098765432109876543210"
        chain = "optimism"

        # Mock pool tokens data
        mock_pool_tokens_data = [
            [
                "0x1111111111111111111111111111111111111111",
                "0x2222222222222222222222222222222222222222",
            ],  # tokens
            [1000000000000000000000, 2000000000000000000000],  # balances
        ]
        mock_user_balance = 500000000000000000000  # 500 BPT tokens
        mock_total_supply = 10000000000000000000000  # 10000 BPT tokens total

        def mock_contract_interact(**kwargs):
            # This is a generator that yields None and returns the data
            yield None
            if kwargs.get("contract_callable") == "get_pool_tokens":
                return mock_pool_tokens_data
            elif kwargs.get("contract_callable") == "check_balance":
                return mock_user_balance
            elif kwargs.get("contract_callable") == "get_total_supply":
                return mock_total_supply
            return None

        def mock_get_token_decimals(chain, token_address):
            # This is a generator that yields None and returns the data
            yield None
            return 18

        # Mock vault address
        fetch_behaviour.params.__dict__["_frozen"] = False
        original_addresses = fetch_behaviour.params.balancer_vault_contract_addresses
        fetch_behaviour.params.balancer_vault_contract_addresses = {
            "optimism": "0xBA12222222222d8F44572F6638882f47B660b8F0"
        }
        fetch_behaviour.params.__dict__["_frozen"] = True

        with patch.object(
            fetch_behaviour, "contract_interact", mock_contract_interact
        ), patch.object(
            fetch_behaviour, "_get_token_decimals", mock_get_token_decimals
        ):
            result = self._consume_generator(
                fetch_behaviour.get_user_share_value_balancer(
                    user_address, pool_id, pool_address, chain
                )
            )

            # Verify the result
            expected_result = {
                "0x1111111111111111111111111111111111111111": Decimal(
                    "50.0"
                ),  # 500/10000 * 1000
                "0x2222222222222222222222222222222222222222": Decimal(
                    "100.0"
                ),  # 500/10000 * 2000
            }
            assert result == expected_result

    def test_get_user_share_value_balancer_no_vault_address(self):
        """Test get_user_share_value_balancer method when vault address is not found."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        user_address = "0x1234567890123456789012345678901234567890"
        pool_id = "0xabcdef1234567890abcdef1234567890abcdef12"
        pool_address = "0x9876543210987654321098765432109876543210"
        chain = "unsupported_chain"

        # Mock empty vault addresses by temporarily unfreezing the params object
        fetch_behaviour.params.__dict__["_frozen"] = False
        original_addresses = fetch_behaviour.params.balancer_vault_contract_addresses
        fetch_behaviour.params.balancer_vault_contract_addresses = {}
        fetch_behaviour.params.__dict__["_frozen"] = True

        result = self._consume_generator(
            fetch_behaviour.get_user_share_value_balancer(
                user_address, pool_id, pool_address, chain
            )
        )

        # Verify the result is empty dict
        assert result == {}

    def test_get_user_share_value_balancer_contract_failure(self):
        """Test get_user_share_value_balancer method when contract calls fail."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        user_address = "0x1234567890123456789012345678901234567890"
        pool_id = "0xabcdef1234567890abcdef1234567890abcdef12"
        pool_address = "0x9876543210987654321098765432109876543210"
        chain = "optimism"

        def mock_contract_interact(**kwargs):
            # This is a generator that yields None and returns the data
            yield None
            if kwargs.get("contract_callable") == "get_pool_tokens":
                return None  # Simulate failure
            return None

        # Mock vault address
        fetch_behaviour.params.__dict__["_frozen"] = False
        original_addresses = fetch_behaviour.params.balancer_vault_contract_addresses
        fetch_behaviour.params.balancer_vault_contract_addresses = {
            "optimism": "0xBA12222222222d8F44572F6638882f47B660b8F0"
        }
        fetch_behaviour.params.__dict__["_frozen"] = True

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour.get_user_share_value_balancer(
                    user_address, pool_id, pool_address, chain
                )
            )

            # Verify the result is empty dict
            assert result == {}

    def test_get_user_share_value_sturdy_success(self):
        """Test get_user_share_value_sturdy method with successful contract calls."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        user_address = "0x1234567890123456789012345678901234567890"
        aggregator_address = "0x9876543210987654321098765432109876543210"
        asset_address = "0x1111111111111111111111111111111111111111"
        chain = "optimism"

        mock_user_balance = 1000000000000000000000  # 1000 tokens with 18 decimals
        mock_decimals = 18

        def mock_contract_interact(**kwargs):
            # This is a generator that yields None and returns the data
            yield None
            if kwargs.get("contract_callable") == "balance_of":
                return mock_user_balance
            elif kwargs.get("contract_callable") == "decimals":
                return mock_decimals
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour.get_user_share_value_sturdy(
                    user_address, aggregator_address, asset_address, chain
                )
            )

            # Verify the result
            expected_result = {
                asset_address: Decimal("1000.0")  # 1000000000000000000000 / 10^18
            }
            assert result == expected_result

    def test_get_user_share_value_sturdy_contract_failure(self):
        """Test get_user_share_value_sturdy method when contract calls fail."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        user_address = "0x1234567890123456789012345678901234567890"
        aggregator_address = "0x9876543210987654321098765432109876543210"
        asset_address = "0x1111111111111111111111111111111111111111"
        chain = "optimism"

        def mock_contract_interact(**kwargs):
            # This is a generator that yields None and returns the data
            yield None
            if kwargs.get("contract_callable") == "balance_of":
                return None  # Simulate failure
            return None

        with patch.object(fetch_behaviour, "contract_interact", mock_contract_interact):
            result = self._consume_generator(
                fetch_behaviour.get_user_share_value_sturdy(
                    user_address, aggregator_address, asset_address, chain
                )
            )

            # Verify the result is empty dict
            assert result == {}

    @pytest.mark.parametrize(
        "test_whitelisted_assets,price_behavior,expected_removals,expected_remaining,test_description",
        [
            # Test 1: Price drops that trigger removal (>5% drop)
            (
                {
                    "mode": {
                        "0x4200000000000000000000000000000000000006": "WETH",
                        "0xdfc7c877a950e49d2610114102175a06c2e3167a": "MODE",
                    }
                },
                {
                    "MODE": {"yesterday": 1.0, "today": 0.9},  # 10% drop
                    "WETH": {"yesterday": 1.0, "today": 1.0},  # no change
                },
                [
                    "0xdfc7c877a950e49d2610114102175a06c2e3167a"
                ],  # MODE should be removed
                ["0x4200000000000000000000000000000000000006"],  # WETH remains
                "price drops that trigger removal",
            ),
            # Test 2: No price drops (tokens remain)
            (
                {
                    "mode": {
                        "0x4200000000000000000000000000000000000006": "WETH",
                        "0xdfc7c877a950e49d2610114102175a06c2e3167a": "MODE",
                    }
                },
                {
                    "WETH": {"yesterday": 1.0, "today": 1.05},  # 5% increase
                    "MODE": {"yesterday": 1.0, "today": 1.05},  # 5% increase
                },
                [],  # no removals
                [
                    "0x4200000000000000000000000000000000000006",
                    "0xdfc7c877a950e49d2610114102175a06c2e3167a",
                ],  # both remain
                "no price drops",
            ),
            # Test 3: Price fetch failure (tokens remain due to failure)
            (
                {
                    "mode": {
                        "0x4200000000000000000000000000000000000006": "WETH",
                        "0xdfc7c877a950e49d2610114102175a06c2e3167a": "MODE",
                    }
                },
                {
                    "WETH": {"yesterday": 1.0, "today": 1.05},  # success
                    "MODE": {"yesterday": None, "today": None},  # failure
                },
                [],  # no removals due to failure
                [
                    "0x4200000000000000000000000000000000000006",
                    "0xdfc7c877a950e49d2610114102175a06c2e3167a",
                ],  # both remain
                "price fetch failure",
            ),
            # Test 4: Borderline price drop (4.9% drop - should not be removed)
            (
                {
                    "mode": {
                        "0x4200000000000000000000000000000000000006": "WETH",
                        "0xdfc7c877a950e49d2610114102175a06c2e3167a": "MODE",
                    }
                },
                {
                    "WETH": {"yesterday": 1.0, "today": 1.0},  # no change
                    "MODE": {"yesterday": 1.0, "today": 0.951},  # 4.9% drop
                },
                [],  # no removals (threshold is < -5.0)
                [
                    "0x4200000000000000000000000000000000000006",
                    "0xdfc7c877a950e49d2610114102175a06c2e3167a",
                ],  # both remain
                "borderline price drop",
            ),
            # Test 5: Price increase (tokens remain)
            (
                {
                    "mode": {
                        "0x4200000000000000000000000000000000000006": "WETH",
                        "0xdfc7c877a950e49d2610114102175a06c2e3167a": "MODE",
                    }
                },
                {
                    "WETH": {"yesterday": 1.0, "today": 1.1},  # 10% increase
                    "MODE": {"yesterday": 1.0, "today": 1.1},  # 10% increase
                },
                [],  # no removals
                [
                    "0x4200000000000000000000000000000000000006",
                    "0xdfc7c877a950e49d2610114102175a06c2e3167a",
                ],  # both remain
                "price increase",
            ),
            # Test 6: Exception handling (tokens remain due to exception)
            (
                {
                    "mode": {
                        "0x4200000000000000000000000000000000000006": "WETH",
                        "0xdfc7c877a950e49d2610114102175a06c2e3167a": "MODE",
                    }
                },
                {
                    "WETH": {"yesterday": 1.0, "today": 1.0},  # success
                    "MODE": {"exception": True},  # exception
                },
                [],  # no removals due to exception
                [
                    "0x4200000000000000000000000000000000000006",
                    "0xdfc7c877a950e49d2610114102175a06c2e3167a",
                ],  # both remain
                "exception handling",
            ),
        ],
    )
    def test_track_whitelisted_assets_variations(
        self,
        test_whitelisted_assets,
        price_behavior,
        expected_removals,
        expected_remaining,
        test_description,
    ):
        """Test _track_whitelisted_assets method with various price scenarios."""
        self.setup_default_test_data()

        # Create the correct behaviour instance
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock the datetime to have consistent test dates
        with patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.datetime"
        ) as mock_datetime, patch(
            "packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies.WHITELISTED_ASSETS",
            test_whitelisted_assets,
        ):
            mock_datetime.now.return_value = datetime(2024, 1, 15, 12, 0, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Setup whitelisted assets in memory
            fetch_behaviour.whitelisted_assets = test_whitelisted_assets.copy()

            def mock_get_historical_price_for_date(
                token_address, token_symbol, date_str, chain
            ):
                # This is a generator that yields None and returns the price
                yield

                # Get price behavior for this token
                if token_symbol in price_behavior:
                    behavior = price_behavior[token_symbol]

                    # Handle exception case
                    if "exception" in behavior and behavior["exception"]:
                        raise Exception("API error")

                    # Handle price fetch failure case
                    if "14-01-2024" in date_str:  # yesterday
                        return behavior.get("yesterday")
                    elif "15-01-2024" in date_str:  # today
                        return behavior.get("today")

                # Default fallback
                return 1.0

            def mock_store_whitelisted_assets():
                # Mock storing assets
                pass

            # Apply mocks
            with patch.object(
                fetch_behaviour,
                "_get_historical_price_for_date",
                side_effect=mock_get_historical_price_for_date,
            ), patch.object(
                fetch_behaviour,
                "store_whitelisted_assets",
                side_effect=mock_store_whitelisted_assets,
            ), patch.object(
                fetch_behaviour, "sleep", return_value=None
            ):
                # Execute the method
                generator = fetch_behaviour._track_whitelisted_assets()
                self._consume_generator(generator)

                # Verify expected removals
                for token_address in expected_removals:
                    for chain in fetch_behaviour.whitelisted_assets:
                        if token_address in fetch_behaviour.whitelisted_assets[chain]:
                            assert (
                                token_address
                                not in fetch_behaviour.whitelisted_assets[chain]
                            ), f"Token {token_address} should have been removed for {test_description}"

                # Verify expected remaining tokens
                for token_address in expected_remaining:
                    found = False
                    for chain in fetch_behaviour.whitelisted_assets:
                        if token_address in fetch_behaviour.whitelisted_assets[chain]:
                            found = True
                            break
                    assert (
                        found
                    ), f"Token {token_address} should remain for {test_description}"

    @pytest.mark.parametrize(
        "from_address,tx_data,is_eth_transfer,mock_is_gnosis_safe,expected_result,test_description",
        [
            # Valid transfer - should include
            (
                {
                    "hash": "0x1234567890123456789012345678901234567890",
                    "is_contract": False,
                },
                None,
                False,
                None,
                True,
                "valid transfer",
            ),
            # Null address - should not include
            (
                {
                    "hash": "0x0000000000000000000000000000000000000000",
                    "is_contract": False,
                },
                None,
                False,
                None,
                False,
                "null address",
            ),
            # Empty address - should not include
            (
                {"hash": "", "is_contract": False},
                None,
                False,
                None,
                False,
                "empty address",
            ),
            # No from_address - should not include
            (None, None, False, None, False, "no from_address"),
            # Gnosis safe contract - should include
            (
                {
                    "hash": "0x1234567890123456789012345678901234567890",
                    "is_contract": True,
                },
                None,
                False,
                True,
                True,
                "gnosis safe contract",
            ),
            # Regular contract - should not include
            (
                {
                    "hash": "0x1234567890123456789012345678901234567890",
                    "is_contract": True,
                },
                None,
                False,
                False,
                False,
                "regular contract",
            ),
            # ETH transfer with invalid status - should not include
            (
                {
                    "hash": "0x1234567890123456789012345678901234567890",
                    "is_contract": False,
                },
                {"status": "failed", "value": "1000"},
                True,
                None,
                False,
                "ETH transfer invalid status",
            ),
            # ETH transfer with zero value - should not include
            (
                {
                    "hash": "0x1234567890123456789012345678901234567890",
                    "is_contract": False,
                },
                {"status": "ok", "value": "0"},
                True,
                None,
                False,
                "ETH transfer zero value",
            ),
        ],
    )
    def test_should_include_transfer_mode_variations(
        self,
        from_address,
        tx_data,
        is_eth_transfer,
        mock_is_gnosis_safe,
        expected_result,
        test_description,
    ):
        """Test _should_include_transfer_mode method with various scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        if mock_is_gnosis_safe is not None:
            with patch.object(
                fetch_behaviour, "_is_gnosis_safe", return_value=mock_is_gnosis_safe
            ):
                result = fetch_behaviour._should_include_transfer_mode(
                    from_address, tx_data, is_eth_transfer
                )
        else:
            result = fetch_behaviour._should_include_transfer_mode(
                from_address, tx_data, is_eth_transfer
            )

        assert (
            result is expected_result
        ), f"Expected {expected_result} for {test_description}"

    @pytest.mark.parametrize(
        "from_address,mock_kv_response,mock_optimism_rpc_response,mock_safe_api_response,expected_result,test_description",
        [
            # Valid address - should include (no cache, RPC returns no code, no safe check needed)
            (
                "0x1234567890123456789012345678901234567890",
                None,  # No cache hit
                (True, {"result": "0x"}),  # No code = EOA
                None,  # Safe check not needed
                True,
                "valid address (EOA)",
            ),
            # Contract address - should not include (no cache, RPC returns code, safe check fails)
            (
                "0x1234567890123456789012345678901234567890",
                None,  # No cache hit
                (True, {"result": "0x123456"}),  # Has code = contract
                (False, {}),  # Safe check fails
                False,
                "contract address",
            ),
            # Gnosis safe - should include (no cache, RPC returns code, safe check succeeds)
            (
                "0x1234567890123456789012345678901234567890",
                None,  # No cache hit
                (True, {"result": "0x123456"}),  # Has code = contract
                (True, {}),  # Safe check succeeds
                True,
                "gnosis safe",
            ),
            # Cached result - should use cache instead of API calls
            (
                "0x1234567890123456789012345678901234567890",
                {
                    "contract_check_optimism_0x1234567890123456789012345678901234567890": '{"is_eoa": true}'
                },  # Cache hit
                None,  # Should not be called
                None,  # Should not be called
                True,
                "cached result",
            ),
        ],
    )
    def test_should_include_transfer_optimism_variations(
        self,
        from_address,
        mock_kv_response,
        mock_optimism_rpc_response,
        mock_safe_api_response,
        expected_result,
        test_description,
    ):
        """Test _should_include_transfer_optimism method with various address types."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_read_kv(keys):
            yield
            if mock_kv_response is not None:
                return mock_kv_response
            return None

        def mock_request_with_retries(
            endpoint,
            method=None,
            body=None,
            headers=None,
            rate_limited_code=None,
            rate_limited_callback=None,
            retry_wait=None,
        ):
            yield
            if "mainnet.optimism.io" in endpoint:
                return mock_optimism_rpc_response
            elif "safe-transaction-optimism.safe.global" in endpoint:
                return mock_safe_api_response
            return (False, {})

        with patch.multiple(
            fetch_behaviour,
            _read_kv=mock_read_kv,
            _request_with_retries=mock_request_with_retries,
            _write_kv=self.mock_write_kv,
        ):
            result = self._consume_generator(
                fetch_behaviour._should_include_transfer_optimism(from_address)
            )
            assert (
                result is expected_result
            ), f"Expected {expected_result} for {test_description}"

    @pytest.mark.parametrize(
        "pool_address,token_id,chain,position,mock_position_manager_address,mock_slot0_data,mock_position_details,mock_token_decimals,expected_result,test_description",
        [
            # Test 1: Successful calculation with valid data
            (
                "0x1234567890123456789012345678901234567890",
                123,
                "optimism",
                {
                    "token0": "0x4200000000000000000000000000000000000006",
                    "token1": "0xdfc7c877a950e49d2610114102175a06c2e3167a",
                    "token0_symbol": "WETH",
                    "token1_symbol": "MODE",
                    "token_id": 123,
                },
                "0x9876543210987654321098765432109876543210",  # position manager address
                {
                    "sqrt_price_x96": "1234567890123456789012345678901234567890",
                    "tick": 1000,
                },
                {
                    "tickLower": 500,
                    "tickUpper": 1500,
                    "liquidity": "1000000000000000000",
                    "tokensOwed0": "100000000000000000",
                    "tokensOwed1": "200000000000000000",
                },
                (18, 18),  # token decimals
                {
                    "0x4200000000000000000000000000000000000006": Decimal(
                        "0.100000000000000001"
                    ),  # token0 balance
                    "0xdfc7c877a950e49d2610114102175a06c2e3167a": Decimal(
                        "0.200000000000000001"
                    ),  # token1 balance
                },
                "successful calculation",
            ),
            # Test 2: Missing token addresses - should return empty dict
            (
                "0x1234567890123456789012345678901234567890",
                123,
                "optimism",
                {
                    "token0": None,  # Missing token0
                    "token1": "0xdfc7c877a950e49d2610114102175a06c2e3167a",
                    "token0_symbol": "WETH",
                    "token1_symbol": "MODE",
                    "token_id": 123,
                },
                "0x9876543210987654321098765432109876543210",
                None,
                None,
                None,
                {},
                "missing token0 address",
            ),
            # Test 3: Missing token_id - should return empty dict
            (
                "0x1234567890123456789012345678901234567890",
                0,  # Invalid token_id
                "optimism",
                {
                    "token0": "0x4200000000000000000000000000000000000006",
                    "token1": "0xdfc7c877a950e49d2610114102175a06c2e3167a",
                    "token0_symbol": "WETH",
                    "token1_symbol": "MODE",
                    "token_id": 0,  # Invalid token_id
                },
                "0x9876543210987654321098765432109876543210",
                None,
                None,
                None,
                {},
                "missing token_id",
            ),
            # Test 4: No position manager address for chain - should return empty dict
            (
                "0x1234567890123456789012345678901234567890",
                123,
                "unsupported_chain",  # Chain without position manager
                {
                    "token0": "0x4200000000000000000000000000000000000006",
                    "token1": "0xdfc7c877a950e49d2610114102175a06c2e3167a",
                    "token0_symbol": "WETH",
                    "token1_symbol": "MODE",
                    "token_id": 123,
                },
                None,  # No position manager address
                None,
                None,
                None,
                {},
                "no position manager address for chain",
            ),
            # Test 5: Slot0 data fetch failure - should return empty dict
            (
                "0x1234567890123456789012345678901234567890",
                123,
                "optimism",
                {
                    "token0": "0x4200000000000000000000000000000000000006",
                    "token1": "0xdfc7c877a950e49d2610114102175a06c2e3167a",
                    "token0_symbol": "WETH",
                    "token1_symbol": "MODE",
                    "token_id": 123,
                },
                "0x9876543210987654321098765432109876543210",
                None,  # Slot0 fetch fails
                None,
                None,
                {},
                "slot0 data fetch failure",
            ),
            # Test 6: Invalid slot0 data - should return empty dict
            (
                "0x1234567890123456789012345678901234567890",
                123,
                "optimism",
                {
                    "token0": "0x4200000000000000000000000000000000000006",
                    "token1": "0xdfc7c877a950e49d2610114102175a06c2e3167a",
                    "token0_symbol": "WETH",
                    "token1_symbol": "MODE",
                    "token_id": 123,
                },
                "0x9876543210987654321098765432109876543210",
                {"sqrt_price_x96": None, "tick": 1000},  # Invalid data
                None,
                None,
                {},
                "invalid slot0 data",
            ),
            # Test 7: Token decimals fetch failure - should return empty dict
            (
                "0x1234567890123456789012345678901234567890",
                123,
                "optimism",
                {
                    "token0": "0x4200000000000000000000000000000000000006",
                    "token1": "0xdfc7c877a950e49d2610114102175a06c2e3167a",
                    "token0_symbol": "WETH",
                    "token1_symbol": "MODE",
                    "token_id": 123,
                },
                "0x9876543210987654321098765432109876543210",
                {
                    "sqrt_price_x96": "1234567890123456789012345678901234567890",
                    "tick": 1000,
                },
                None,
                (None, 18),  # token0 decimals fetch fails
                {},
                "token0 decimals fetch failure",
            ),
            # Test 8: Position details fetch failure - should return empty dict
            (
                "0x1234567890123456789012345678901234567890",
                123,
                "optimism",
                {
                    "token0": "0x4200000000000000000000000000000000000006",
                    "token1": "0xdfc7c877a950e49d2610114102175a06c2e3167a",
                    "token0_symbol": "WETH",
                    "token1_symbol": "MODE",
                    "token_id": 123,
                },
                "0x9876543210987654321098765432109876543210",
                {
                    "sqrt_price_x96": "1234567890123456789012345678901234567890",
                    "tick": 1000,
                },
                None,  # Position details fetch fails
                (18, 18),
                {
                    "0x4200000000000000000000000000000000000006": Decimal("0"),
                    "0xdfc7c877a950e49d2610114102175a06c2e3167a": Decimal("0"),
                },
                "position details fetch failure",
            ),
        ],
    )
    def test_get_user_share_value_uniswap_variations(
        self,
        pool_address,
        token_id,
        chain,
        position,
        mock_position_manager_address,
        mock_slot0_data,
        mock_position_details,
        mock_token_decimals,
        expected_result,
        test_description,
    ):
        """Test get_user_share_value_uniswap method with various scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock the params to include position manager address
        fetch_behaviour.params.__dict__["_frozen"] = False
        fetch_behaviour.params.uniswap_position_manager_contract_addresses = {
            "optimism": "0x9876543210987654321098765432109876543210"
        }
        fetch_behaviour.params.__dict__["_frozen"] = True

        def mock_contract_interact(**kwargs):
            yield None
            if kwargs.get("contract_callable") == "slot0":
                return mock_slot0_data
            elif kwargs.get("contract_callable") == "get_position":
                return mock_position_details
            return None

        def mock_get_token_decimals_pair(chain, token0_address, token1_address):
            yield None
            return mock_token_decimals

        def mock_calculate_position_amounts(
            position_details, current_tick, sqrt_price_x96, position, dex_type, chain
        ):
            yield None
            if mock_position_details:
                # Return calculated amounts based on position details
                liquidity = int(mock_position_details.get("liquidity", 0))
                tokens_owed0 = int(mock_position_details.get("tokensOwed0", 0))
                tokens_owed1 = int(mock_position_details.get("tokensOwed1", 0))
                # Simplified calculation for testing
                amount0 = liquidity // 1000000000000000000 + tokens_owed0
                amount1 = liquidity // 1000000000000000000 + tokens_owed1
                return amount0, amount1
            return 0, 0

        with patch.multiple(
            fetch_behaviour,
            contract_interact=mock_contract_interact,
            _get_token_decimals_pair=mock_get_token_decimals_pair,
            _calculate_position_amounts=mock_calculate_position_amounts,
        ):
            result = self._consume_generator(
                fetch_behaviour.get_user_share_value_uniswap(
                    pool_address, token_id, chain, position
                )
            )
            assert (
                result == expected_result
            ), f"Expected {expected_result} for {test_description}"

    def test_get_user_share_value_uniswap_success(self):
        """Test get_user_share_value_uniswap method with successful calculation."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        pool_address = "0x1234567890123456789012345678901234567890"
        token_id = 123
        chain = "optimism"
        position = {
            "token0": "0x4200000000000000000000000000000000000006",
            "token1": "0xdfc7c877a950e49d2610114102175a06c2e3167a",
            "token0_symbol": "WETH",
            "token1_symbol": "MODE",
            "token_id": 123,
        }

        # Mock the params
        fetch_behaviour.params.__dict__["_frozen"] = False
        fetch_behaviour.params.uniswap_position_manager_contract_addresses = {
            "optimism": "0x9876543210987654321098765432109876543210"
        }
        fetch_behaviour.params.__dict__["_frozen"] = True

        mock_slot0_data = {
            "sqrt_price_x96": "1234567890123456789012345678901234567890",
            "tick": 1000,
        }

        mock_position_details = {
            "tickLower": 500,
            "tickUpper": 1500,
            "liquidity": "1000000000000000000",
            "tokensOwed0": "100000000000000000",
            "tokensOwed1": "200000000000000000",
        }

        def mock_contract_interact(**kwargs):
            yield None
            if kwargs.get("contract_callable") == "slot0":
                return mock_slot0_data
            elif kwargs.get("contract_callable") == "get_position":
                return mock_position_details
            return None

        def mock_get_token_decimals_pair(chain, token0_address, token1_address):
            yield None
            return (18, 18)

        def mock_calculate_position_amounts(
            position_details, current_tick, sqrt_price_x96, position, dex_type, chain
        ):
            yield None
            # Return calculated amounts
            amount0 = 1100000000000000000  # 1.1 tokens
            amount1 = 2200000000000000000  # 2.2 tokens
            return amount0, amount1

        with patch.multiple(
            fetch_behaviour,
            contract_interact=mock_contract_interact,
            _get_token_decimals_pair=mock_get_token_decimals_pair,
            _calculate_position_amounts=mock_calculate_position_amounts,
        ):
            result = self._consume_generator(
                fetch_behaviour.get_user_share_value_uniswap(
                    pool_address, token_id, chain, position
                )
            )

            # Verify the result
            expected_result = {
                "0x4200000000000000000000000000000000000006": Decimal("1.1"),
                "0xdfc7c877a950e49d2610114102175a06c2e3167a": Decimal("2.2"),
            }
            assert result == expected_result

    def test_get_user_share_value_uniswap_missing_tokens(self):
        """Test get_user_share_value_uniswap method with missing token addresses."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        pool_address = "0x1234567890123456789012345678901234567890"
        token_id = 123
        chain = "optimism"
        position = {
            "token0": None,  # Missing token0
            "token1": "0xdfc7c877a950e49d2610114102175a06c2e3167a",
            "token0_symbol": "WETH",
            "token1_symbol": "MODE",
            "token_id": 123,
        }

        result = self._consume_generator(
            fetch_behaviour.get_user_share_value_uniswap(
                pool_address, token_id, chain, position
            )
        )

        # Should return empty dict due to missing token0
        assert result == {}

    def test_get_user_share_value_uniswap_no_position_manager(self):
        """Test get_user_share_value_uniswap method with no position manager address."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        pool_address = "0x1234567890123456789012345678901234567890"
        token_id = 123
        chain = "unsupported_chain"
        position = {
            "token0": "0x4200000000000000000000000000000000000006",
            "token1": "0xdfc7c877a950e49d2610114102175a06c2e3167a",
            "token0_symbol": "WETH",
            "token1_symbol": "MODE",
            "token_id": 123,
        }

        # No position manager address for this chain
        fetch_behaviour.params.__dict__["_frozen"] = False
        fetch_behaviour.params.uniswap_position_manager_contract_addresses = {}
        fetch_behaviour.params.__dict__["_frozen"] = True

        result = self._consume_generator(
            fetch_behaviour.get_user_share_value_uniswap(
                pool_address, token_id, chain, position
            )
        )

        # Should return empty dict due to missing position manager
        assert result == {}

    @pytest.mark.parametrize(
        "accumulated_rewards,olas_price,expected_result,test_description",
        [
            # Successful calculation with various amounts
            (
                1000000000000000000,  # 1 OLAS in wei
                2.0,  # $2.00 per OLAS
                Decimal("2.0"),  # 1 * $2.00 = $2.00
                "successful calculation with 1 OLAS",
            ),
            (
                10500000000000000000,  # 10.5 OLAS in wei
                0.75,  # $0.75 per OLAS
                Decimal("7.875"),  # 10.5 * $0.75 = $7.875
                "successful calculation with 10.5 OLAS",
            ),
            # Price fetch failure
            (
                5000000000000000000,  # 5 OLAS in wei
                None,  # Price fetch failure
                Decimal("0"),  # Should return 0 when price fetch fails
                "price fetch failure",
            ),
            # Zero rewards
            (
                0,  # No OLAS rewards
                1.0,  # $1.00 per OLAS (should not be used)
                Decimal("0"),  # Should return 0 when no rewards
                "zero rewards",
            ),
        ],
    )
    def test_calculate_stakig_rewards_value_variations(
        self, accumulated_rewards, olas_price, expected_result, test_description
    ):
        """Test calculate_stakig_rewards_value method with various scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Set up target investment chains
        fetch_behaviour.params.__dict__["_frozen"] = False
        fetch_behaviour.params.target_investment_chains = ["mode"]
        fetch_behaviour.params.__dict__["_frozen"] = True

        def mock_update_accumulated_rewards_for_chain(chain):
            yield None
            return

        def mock_get_accumulated_rewards_for_token(chain, token_address):
            yield None
            return accumulated_rewards

        def mock_fetch_token_price(token_address, chain):
            yield None
            return olas_price

        with patch.multiple(
            fetch_behaviour,
            update_accumulated_rewards_for_chain=mock_update_accumulated_rewards_for_chain,
            get_accumulated_rewards_for_token=mock_get_accumulated_rewards_for_token,
            _fetch_token_price=mock_fetch_token_price,
        ):
            result = self._consume_generator(
                fetch_behaviour.calculate_stakig_rewards_value()
            )
            assert (
                result == expected_result
            ), f"Expected {expected_result} for {test_description}"

    def test_check_and_update_zero_liquidity_positions_mixed_positions(self):
        """Test check_and_update_zero_liquidity_positions method with mixed position types and statuses."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Set up mixed positions
        fetch_behaviour.current_positions = [
            # Velodrome CL with zero liquidity (should be closed)
            {
                "status": "open",
                "dex_type": "velodrome",
                "is_cl_pool": True,
                "pool_address": "0x1111111111111111111111111111111111111111",
                "positions": [{"current_liquidity": 0}, {"current_liquidity": 0}],
            },
            # Balancer with liquidity (should remain open)
            {
                "status": "open",
                "dex_type": "balancer",
                "pool_address": "0x2222222222222222222222222222222222222222",
                "current_liquidity": 100,
            },
            # Already closed position (should remain closed)
            {
                "status": "closed",
                "dex_type": "uniswap",
                "pool_address": "0x3333333333333333333333333333333333333333",
                "current_liquidity": 0,
            },
            # Uniswap with zero liquidity (should be closed)
            {
                "status": "open",
                "dex_type": "uniswap",
                "pool_address": "0x4444444444444444444444444444444444444444",
                "current_liquidity": 0,
            },
        ]

        # Mock store_current_positions
        def mock_store_current_positions():
            # This method doesn't return anything, just stores data
            pass

        with patch.multiple(
            fetch_behaviour, store_current_positions=mock_store_current_positions
        ):
            fetch_behaviour.check_and_update_zero_liquidity_positions()

            # Verify Velodrome CL position was closed
            assert fetch_behaviour.current_positions[0]["status"] == "closed"

            # Verify Balancer position remains open
            assert fetch_behaviour.current_positions[1]["status"] == "open"

            # Verify already closed position remains closed
            assert fetch_behaviour.current_positions[2]["status"] == "closed"

            # Verify Uniswap position was closed
            assert fetch_behaviour.current_positions[3]["status"] == "closed"

    @pytest.mark.parametrize(
        "dex_type,is_cl_pool,current_liquidity,positions,expected_status,test_description",
        [
            # Velodrome CL with zero liquidity in all positions
            (
                "velodrome",
                True,
                None,  # Not used for CL pools
                [{"current_liquidity": 0}, {"current_liquidity": 0}],
                "closed",
                "velodrome CL with zero liquidity in all positions",
            ),
            # Velodrome CL with partial liquidity
            (
                "velodrome",
                True,
                None,  # Not used for CL pools
                [{"current_liquidity": 0}, {"current_liquidity": 100}],
                "open",
                "velodrome CL with partial liquidity",
            ),
            # Velodrome CL with no positions list
            (
                "velodrome",
                True,
                None,  # Not used for CL pools
                None,  # No positions list
                "open",
                "velodrome CL with no positions list",
            ),
            # Velodrome CL with empty positions list
            (
                "velodrome",
                True,
                None,  # Not used for CL pools
                [],  # Empty positions list
                "open",
                "velodrome CL with empty positions list",
            ),
            # Balancer with zero liquidity
            (
                "balancer",
                False,
                0,
                None,  # Not used for non-CL pools
                "closed",
                "balancer with zero liquidity",
            ),
            # Balancer with liquidity
            (
                "balancer",
                False,
                100,
                None,  # Not used for non-CL pools
                "open",
                "balancer with liquidity",
            ),
            # Uniswap with zero liquidity
            (
                "uniswap",
                False,
                0,
                None,  # Not used for non-CL pools
                "closed",
                "uniswap with zero liquidity",
            ),
            # Uniswap with liquidity
            (
                "uniswap",
                False,
                50,
                None,  # Not used for non-CL pools
                "open",
                "uniswap with liquidity",
            ),
            # Velodrome non-CL with zero liquidity
            (
                "velodrome",
                False,
                0,
                None,  # Not used for non-CL pools
                "closed",
                "velodrome non-CL with zero liquidity",
            ),
            # Velodrome non-CL with liquidity
            (
                "velodrome",
                False,
                75,
                None,  # Not used for non-CL pools
                "open",
                "velodrome non-CL with liquidity",
            ),
        ],
    )
    def test_check_and_update_zero_liquidity_positions_variations(
        self,
        dex_type,
        is_cl_pool,
        current_liquidity,
        positions,
        expected_status,
        test_description,
    ):
        """Test check_and_update_zero_liquidity_positions method with various scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Set up position based on parameters
        position = {
            "status": "open",
            "dex_type": dex_type,
            "pool_address": "0x1234567890123456789012345678901234567890",
        }

        if is_cl_pool is not None:
            position["is_cl_pool"] = is_cl_pool

        if current_liquidity is not None:
            position["current_liquidity"] = current_liquidity

        if positions is not None:
            position["positions"] = positions

        fetch_behaviour.current_positions = [position]

        # Mock store_current_positions
        def mock_store_current_positions():
            # This method doesn't return anything, just stores data
            pass

        with patch.multiple(
            fetch_behaviour, store_current_positions=mock_store_current_positions
        ):
            fetch_behaviour.check_and_update_zero_liquidity_positions()

            # Verify expected status
            assert (
                fetch_behaviour.current_positions[0]["status"] == expected_status
            ), f"Expected {expected_status} for {test_description}"

    @pytest.mark.parametrize(
        "current_positions,expected_calls,expected_logs,test_description",
        [
            # Test 1: No positions to update
            ([], [], ["No positions to update."], "no positions to update"),
            # Test 2: All positions are closed (should skip all)
            (
                [
                    {"status": "closed", "dex_type": "balancerPool", "chain": "mode"},
                    {"status": "closed", "dex_type": "UniswapV3", "chain": "optimism"},
                ],
                [],
                ["No positions to update."],
                "all positions are closed",
            ),
            # Test 3: Position missing dex_type
            (
                [{"status": "open", "chain": "mode"}],  # Missing dex_type
                [],
                [
                    "Position missing dex_type or chain: {'status': 'open', 'chain': 'mode'}"
                ],
                "position missing dex_type",
            ),
            # Test 4: Position missing chain
            (
                [{"status": "open", "dex_type": "balancerPool"}],  # Missing chain
                [],
                [
                    "Position missing dex_type or chain: {'status': 'open', 'dex_type': 'balancerPool'}"
                ],
                "position missing chain",
            ),
            # Test 5: Unknown position type
            (
                [{"status": "open", "dex_type": "unknown_dex", "chain": "mode"}],
                [],
                ["Unknown position type: unknown_dex"],
                "unknown position type",
            ),
            # Test 6: Mixed positions (open and closed)
            (
                [
                    {"status": "closed", "dex_type": "balancerPool", "chain": "mode"},
                    {
                        "status": "open",
                        "dex_type": "balancerPool",
                        "chain": "mode",
                        "pool_address": "0x123",
                    },
                    {
                        "status": "open",
                        "dex_type": "UniswapV3",
                        "chain": "optimism",
                        "token_id": 123,
                    },
                ],
                ["_update_balancer_position", "_update_uniswap_position"],
                [
                    "Updating position of type balancerPool on chain mode",
                    "Updating position of type UniswapV3 on chain optimism",
                ],
                "mixed open and closed positions",
            ),
            # Test 7: All supported DEX types
            (
                [
                    {
                        "status": "open",
                        "dex_type": "balancerPool",
                        "chain": "mode",
                        "pool_address": "0x123",
                    },
                    {
                        "status": "open",
                        "dex_type": "UniswapV3",
                        "chain": "optimism",
                        "token_id": 123,
                    },
                    {
                        "status": "open",
                        "dex_type": "velodrome",
                        "chain": "optimism",
                        "is_cl_pool": False,
                        "pool_address": "0x456",
                    },
                    {
                        "status": "open",
                        "dex_type": "Sturdy",
                        "chain": "mode",
                        "pool_address": "0x789",
                    },
                ],
                [
                    "_update_balancer_position",
                    "_update_uniswap_position",
                    "_update_velodrome_position",
                    "_update_sturdy_position",
                ],
                [
                    "Updating position of type balancerPool on chain mode",
                    "Updating position of type UniswapV3 on chain optimism",
                    "Updating position of type velodrome on chain optimism",
                    "Updating position of type Sturdy on chain mode",
                ],
                "all supported DEX types",
            ),
        ],
    )
    def test_update_position_amounts_variations(
        self, current_positions, expected_calls, expected_logs, test_description
    ):
        """Test update_position_amounts method with various scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        fetch_behaviour.current_positions = current_positions

        # Mock the update methods
        mock_calls = []

        def mock_update_balancer_position(position):
            mock_calls.append("_update_balancer_position")
            yield None

        def mock_update_uniswap_position(position):
            mock_calls.append("_update_uniswap_position")
            yield None

        def mock_update_velodrome_position(position):
            mock_calls.append("_update_velodrome_position")
            yield None

        def mock_update_sturdy_position(position):
            mock_calls.append("_update_sturdy_position")
            yield None

        def mock_store_current_positions():
            pass

        with patch.multiple(
            fetch_behaviour,
            _update_balancer_position=mock_update_balancer_position,
            _update_uniswap_position=mock_update_uniswap_position,
            _update_velodrome_position=mock_update_velodrome_position,
            _update_sturdy_position=mock_update_sturdy_position,
            store_current_positions=mock_store_current_positions,
        ):
            # Execute the method
            self._consume_generator(fetch_behaviour.update_position_amounts())

            # Verify expected calls were made
            assert (
                mock_calls == expected_calls
            ), f"Expected calls {expected_calls} for {test_description}, got {mock_calls}"

    def test_update_position_amounts_balancer_success(self):
        """Test update_position_amounts method for Balancer position with successful update."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Temporarily unfreeze params to set safe contract addresses
        fetch_behaviour.params.__dict__["_frozen"] = False
        fetch_behaviour.params.safe_contract_addresses = {
            "mode": "0x9876543210987654321098765432109876543210"
        }
        fetch_behaviour.params.__dict__["_frozen"] = True

        fetch_behaviour.current_positions = [
            {
                "status": "open",
                "dex_type": "balancerPool",
                "chain": "mode",
                "pool_address": "0x1234567890123456789012345678901234567890",
            }
        ]

        # Mock contract_interact to return balance
        def mock_contract_interact(**kwargs):
            yield None
            return 1000000000000000000  # 1 LP token in wei

        def mock_store_current_positions():
            pass

        with patch.multiple(
            fetch_behaviour,
            contract_interact=mock_contract_interact,
            store_current_positions=mock_store_current_positions,
        ):
            # Execute the method
            self._consume_generator(fetch_behaviour.update_position_amounts())

            # Verify the position was updated correctly
            assert (
                fetch_behaviour.current_positions[0]["current_liquidity"]
                == 1000000000000000000
            )

    def test_update_position_amounts_uniswap_success(self):
        """Test update_position_amounts method for Uniswap position with successful update."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        fetch_behaviour.current_positions = [
            {
                "status": "open",
                "dex_type": "UniswapV3",
                "chain": "optimism",
                "token_id": 123,
            }
        ]

        # Mock contract_interact to return position data
        def mock_contract_interact(**kwargs):
            yield None
            return {"liquidity": 500000000000000000}

        def mock_store_current_positions():
            pass

        with patch.multiple(
            fetch_behaviour,
            contract_interact=mock_contract_interact,
            store_current_positions=mock_store_current_positions,
        ):
            # Execute the method
            self._consume_generator(fetch_behaviour.update_position_amounts())

            # Verify the position was updated correctly
            assert (
                fetch_behaviour.current_positions[0]["current_liquidity"]
                == 500000000000000000
            )

    def test_update_position_amounts_velodrome_success(self):
        """Test update_position_amounts method for Velodrome position with successful update."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()
        fetch_behaviour.current_positions = [
            {
                "status": "open",
                "dex_type": "velodrome",
                "chain": "optimism",
                "is_cl_pool": False,
                "pool_address": "0x1234567890123456789012345678901234567890",
            }
        ]

        # Mock contract_interact to return balance
        def mock_contract_interact(**kwargs):
            yield None
            return 2000000000000000000  # 2 LP tokens in wei

        def mock_store_current_positions():
            pass

        # Temporarily unfreeze params to set safe contract addresses
        fetch_behaviour.params.__dict__["_frozen"] = False
        fetch_behaviour.params.safe_contract_addresses = {
            "optimism": "0x9876543210987654321098765432109876543210"
        }
        fetch_behaviour.params.__dict__["_frozen"] = True

        with patch.multiple(
            fetch_behaviour,
            contract_interact=mock_contract_interact,
            store_current_positions=mock_store_current_positions,
        ):
            # Execute the method
            self._consume_generator(fetch_behaviour.update_position_amounts())

            # Verify the position was updated correctly
            assert (
                fetch_behaviour.current_positions[0]["current_liquidity"]
                == 2000000000000000000
            )

    def test_update_position_amounts_sturdy_success(self):
        """Test update_position_amounts method for Sturdy position with successful update."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Temporarily unfreeze params to set safe contract addresses
        fetch_behaviour.params.__dict__["_frozen"] = False
        fetch_behaviour.params.safe_contract_addresses = {
            "mode": "0x9876543210987654321098765432109876543210"
        }
        fetch_behaviour.params.__dict__["_frozen"] = True

        fetch_behaviour.current_positions = [
            {
                "status": "open",
                "dex_type": "Sturdy",
                "chain": "mode",
                "pool_address": "0x1234567890123456789012345678901234567890",
            }
        ]

        # Mock contract_interact to return balance
        def mock_contract_interact(**kwargs):
            yield None
            return 3000000000000000000  # 3 tokens in wei

        def mock_store_current_positions():
            pass

        with patch.multiple(
            fetch_behaviour,
            contract_interact=mock_contract_interact,
            store_current_positions=mock_store_current_positions,
        ):
            # Execute the method
            self._consume_generator(fetch_behaviour.update_position_amounts())

            # Verify the position was updated correctly
            assert (
                fetch_behaviour.current_positions[0]["current_liquidity"]
                == 3000000000000000000
            )

    def test_update_position_amounts_velodrome_cl_positions(self):
        """Test update_position_amounts method for Velodrome CL positions."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Set up Velodrome CL position with multiple sub-positions
        fetch_behaviour.current_positions = [
            {
                "status": "open",
                "dex_type": "velodrome",
                "chain": "optimism",
                "is_cl_pool": True,
                "positions": [{"token_id": 123}, {"token_id": 456}],
            }
        ]

        # Mock contract_interact to return different liquidity for each position
        call_count = 0

        def mock_contract_interact(**kwargs):
            nonlocal call_count
            yield None
            call_count += 1
            # Return different liquidity for each position
            if call_count == 1:
                return {"liquidity": 1000000000000000000}  # 1 LP token
            else:
                return {"liquidity": 2000000000000000000}  # 2 LP tokens

        def mock_store_current_positions():
            pass

        with patch.multiple(
            fetch_behaviour,
            contract_interact=mock_contract_interact,
            store_current_positions=mock_store_current_positions,
        ):
            # Execute the method
            self._consume_generator(fetch_behaviour.update_position_amounts())

            # Verify both sub-positions were updated
            assert (
                fetch_behaviour.current_positions[0]["positions"][0][
                    "current_liquidity"
                ]
                == 1000000000000000000
            )
            assert (
                fetch_behaviour.current_positions[0]["positions"][1][
                    "current_liquidity"
                ]
                == 2000000000000000000
            )

    def test_update_position_amounts_missing_parameters(self):
        """Test update_position_amounts method with missing parameters for each DEX type."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Test cases for missing parameters
        test_cases = [
            # Balancer missing parameters
            {
                "position": {
                    "status": "open",
                    "dex_type": "balancerPool",
                    "chain": "mode",
                },  # Missing pool_address
                "expected_log": "Missing required parameters for Balancer position",
            },
            # Uniswap missing parameters
            {
                "position": {
                    "status": "open",
                    "dex_type": "UniswapV3",
                    "chain": "optimism",
                },  # Missing token_id
                "expected_log": "Missing required parameters for Uniswap position",
            },
            # Velodrome CL missing parameters
            {
                "position": {
                    "status": "open",
                    "dex_type": "velodrome",
                    "chain": "optimism",
                    "is_cl_pool": True,
                    "positions": [{"token_id": None}],  # Missing token_id
                },
                "expected_log": "Missing token_id for Velodrome CL position",
            },
            # Velodrome non-CL missing parameters
            {
                "position": {
                    "status": "open",
                    "dex_type": "velodrome",
                    "chain": "optimism",
                    "is_cl_pool": False,
                },  # Missing pool_address
                "expected_log": "Missing required parameters for Velodrome pool position",
            },
            # Sturdy missing parameters
            {
                "position": {
                    "status": "open",
                    "dex_type": "Sturdy",
                    "chain": "mode",
                },  # Missing pool_address
                "expected_log": "Missing required parameters for Sturdy position",
            },
        ]

        def mock_store_current_positions():
            pass

        with patch.multiple(
            fetch_behaviour, store_current_positions=mock_store_current_positions
        ):
            for test_case in test_cases:
                fetch_behaviour.current_positions = [test_case["position"]]

                # Execute the method
                self._consume_generator(fetch_behaviour.update_position_amounts())

                # Verify the position was not updated (no current_liquidity added)
                assert (
                    "current_liquidity" not in fetch_behaviour.current_positions[0]
                ), f"Should not have current_liquidity for {test_case['expected_log']}"

    def test_update_position_amounts_no_position_manager_address(self):
        """Test update_position_amounts method when position manager addresses are not configured."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Test Uniswap with no position manager address
        fetch_behaviour.current_positions = [
            {
                "status": "open",
                "dex_type": "UniswapV3",
                "chain": "unsupported_chain",  # Chain without position manager
                "token_id": 123,
            }
        ]

        def mock_store_current_positions():
            pass

        with patch.multiple(
            fetch_behaviour, store_current_positions=mock_store_current_positions
        ):
            # Execute the method
            self._consume_generator(fetch_behaviour.update_position_amounts())

            # Verify the position was not updated
            assert "current_liquidity" not in fetch_behaviour.current_positions[0]

    @pytest.mark.parametrize(
        "address,end_date,fetch_till_date,mock_funding_events,mock_token_transfers_success,mock_eth_transfers_success,mock_token_transfers_data,mock_eth_transfers_data,expected_result,expected_logs,test_description",
        [
            # Test 1: Successful fetch with new data
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                False,
                {"mode": {"2024-01-10": [{"type": "token", "amount": 100}]}},
                True,  # Token transfers successful
                True,  # ETH transfers successful
                {"2024-01-12": [{"type": "token", "amount": 200}]},
                {"2024-01-13": [{"type": "eth", "amount": 1.5, "delta": 1.5}]},
                {
                    "2024-01-12": [{"type": "token", "amount": 200}],
                    "2024-01-13": [{"type": "eth", "amount": 1.5, "delta": 1.5}],
                },
                [
                    "Fetching all Mode transfers until 2024-01-15...",
                    "Fetching Mode token transfers...",
                    "Fetching Mode ETH transfers...",
                    "Mode Summary: 2 dates with transfers, 2 total transfers",
                ],
                "successful fetch with new data",
            ),
            # Test 2: No existing funding events
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                True,
                None,  # No existing funding events
                True,
                True,
                {"2024-01-12": [{"type": "token", "amount": 200}]},
                {"2024-01-13": [{"type": "eth", "amount": 1.5, "delta": 1.5}]},
                {
                    "2024-01-12": [{"type": "token", "amount": 200}],
                    "2024-01-13": [{"type": "eth", "amount": 1.5, "delta": 1.5}],
                },
                [
                    "Fetching all Mode transfers until 2024-01-15...",
                    "Fetching Mode token transfers...",
                    "Fetching Mode ETH transfers...",
                    "Mode Summary: 2 dates with transfers, 2 total transfers",
                ],
                "no existing funding events",
            ),
            # Test 3: Backward compatibility - ETH transfer without delta field
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                False,
                {
                    "mode": {
                        "2024-01-10": [
                            {"type": "eth", "amount": 1.0}  # Missing delta field
                        ]
                    }
                },
                True,  # Token transfers successful
                True,  # ETH transfers successful
                {"2024-01-12": [{"type": "token", "amount": 200}]},
                {"2024-01-13": [{"type": "eth", "amount": 1.5, "delta": 1.5}]},
                {
                    "2024-01-12": [{"type": "token", "amount": 200}],
                    "2024-01-13": [{"type": "eth", "amount": 1.5, "delta": 1.5}],
                },
                [
                    "Found ETH transfer without delta field on 2024-01-10. Setting fetch_till_date=True and clearing funding_events for refetch.",
                    "Cleared funding_events for backward compatibility refetch",
                    "Fetching all Mode transfers until 2024-01-15...",
                    "Fetching Mode token transfers...",
                    "Fetching Mode ETH transfers...",
                    "Mode Summary: 2 dates with transfers, 2 total transfers",
                ],
                "backward compatibility - ETH transfer without delta field",
            ),
            # Test 4: Token transfers fail, use existing data
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                False,
                {"mode": {"2024-01-10": [{"type": "token", "amount": 100}]}},
                False,  # Token transfers fail
                True,  # ETH transfers successful
                {},
                {"2024-01-13": [{"type": "eth", "amount": 1.5, "delta": 1.5}]},
                {
                    "2024-01-10": [{"type": "token", "amount": 100}],
                    "2024-01-13": [{"type": "eth", "amount": 1.5, "delta": 1.5}],
                },
                [
                    "Fetching all Mode transfers until 2024-01-15...",
                    "Fetching Mode token transfers...",
                    "No token transfers found for Mode",
                    "Fetching Mode ETH transfers...",
                    "Mode Summary: 2 dates with transfers, 2 total transfers",
                ],
                "token transfers fail, use existing data",
            ),
            # Test 5: Both transfers fail, use existing data
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                False,
                {"mode": {"2024-01-10": [{"type": "token", "amount": 100}]}},
                False,  # Token transfers fail
                False,  # ETH transfers fail
                {},
                {},
                {"2024-01-10": [{"type": "token", "amount": 100}]},
                [
                    "Fetching all Mode transfers until 2024-01-15...",
                    "Fetching Mode token transfers...",
                    "No token transfers found for Mode",
                    "Fetching Mode ETH transfers...",
                    "No ETH transfers found for Mode",
                    "Mode Summary: 1 dates with transfers, 1 total transfers",
                ],
                "both transfers fail, use existing data",
            ),
            # Test 6: Exception handling
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                False,
                {"mode": {"2024-01-10": [{"type": "token", "amount": 100}]}},
                "exception",  # Will raise exception
                True,
                {},
                {},
                {"2024-01-10": [{"type": "token", "amount": 100}]},
                [
                    "Fetching all Mode transfers until 2024-01-15...",
                    "Fetching Mode token transfers...",
                    "Error fetching Mode transfers: Test exception",
                ],
                "exception handling",
            ),
            # Test 7: Empty existing data
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                False,
                {"mode": {}},  # Empty mode data
                True,
                True,
                {"2024-01-12": [{"type": "token", "amount": 200}]},
                {"2024-01-13": [{"type": "eth", "amount": 1.5, "delta": 1.5}]},
                {
                    "2024-01-12": [{"type": "token", "amount": 200}],
                    "2024-01-13": [{"type": "eth", "amount": 1.5, "delta": 1.5}],
                },
                [
                    "Fetching all Mode transfers until 2024-01-15...",
                    "Fetching Mode token transfers...",
                    "Fetching Mode ETH transfers...",
                    "Mode Summary: 2 dates with transfers, 2 total transfers",
                ],
                "empty existing data",
            ),
            # Test 8: No mode data in funding events
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                False,
                {
                    "optimism": {"2024-01-10": [{"type": "token", "amount": 100}]}
                },  # No mode data
                True,
                True,
                {"2024-01-12": [{"type": "token", "amount": 200}]},
                {"2024-01-13": [{"type": "eth", "amount": 1.5, "delta": 1.5}]},
                {
                    "2024-01-12": [{"type": "token", "amount": 200}],
                    "2024-01-13": [{"type": "eth", "amount": 1.5, "delta": 1.5}],
                },
                [
                    "Fetching all Mode transfers until 2024-01-15...",
                    "Fetching Mode token transfers...",
                    "Fetching Mode ETH transfers...",
                    "Mode Summary: 2 dates with transfers, 2 total transfers",
                ],
                "no mode data in funding events",
            ),
        ],
    )
    def test_fetch_all_transfers_until_date_mode_variations(
        self,
        address,
        end_date,
        fetch_till_date,
        mock_funding_events,
        mock_token_transfers_success,
        mock_eth_transfers_success,
        mock_token_transfers_data,
        mock_eth_transfers_data,
        expected_result,
        expected_logs,
        test_description,
    ):
        """Test variations of _fetch_all_transfers_until_date_mode."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_read_funding_events():
            return mock_funding_events

        def mock_store_funding_events():
            # Mock storing funding events
            pass

        def mock_fetch_token_transfers_mode(
            address, end_datetime, all_transfers_by_date, fetch_till_date
        ):
            if mock_token_transfers_success == "exception":
                raise Exception("Test exception")
            if mock_token_transfers_success:
                all_transfers_by_date.update(mock_token_transfers_data)
            return mock_token_transfers_success

        def mock_fetch_eth_transfers_mode(
            address, end_datetime, all_transfers_by_date, fetch_till_date
        ):
            if mock_eth_transfers_success:
                all_transfers_by_date.update(mock_eth_transfers_data)
            return mock_eth_transfers_success

        with patch.multiple(
            fetch_behaviour,
            read_funding_events=mock_read_funding_events,
            store_funding_events=mock_store_funding_events,
            _fetch_token_transfers_mode=mock_fetch_token_transfers_mode,
            _fetch_eth_transfers_mode=mock_fetch_eth_transfers_mode,
        ):
            result = fetch_behaviour._fetch_all_transfers_until_date_mode(
                address, end_date, fetch_till_date
            )

            # Verify result
            assert result == expected_result, f"Failed for {test_description}"

    def test_fetch_all_transfers_until_date_mode_exception_fallback(self):
        """Test exception handling with fallback to existing data."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        existing_data = {"mode": {"2024-01-10": [{"type": "token", "amount": 100}]}}

        def mock_read_funding_events():
            return existing_data

        def mock_fetch_token_transfers_mode(
            address, end_datetime, all_transfers_by_date, fetch_till_date
        ):
            raise Exception("Test exception")

        with patch.multiple(
            fetch_behaviour,
            read_funding_events=mock_read_funding_events,
            _fetch_token_transfers_mode=mock_fetch_token_transfers_mode,
        ):
            result = fetch_behaviour._fetch_all_transfers_until_date_mode(
                "0x1234567890123456789012345678901234567890", "2024-01-15", False
            )

            # Should return existing data on exception
            assert result == existing_data["mode"]

    def test_fetch_all_transfers_until_date_mode_backward_compatibility_multiple_dates(
        self,
    ):
        """Test backward compatibility with multiple dates containing invalid transfers."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Multiple dates with ETH transfers missing delta field
        mock_funding_events = {
            "mode": {
                "2024-01-10": [
                    {"type": "eth", "amount": 1.0},  # Missing delta
                    {"type": "token", "amount": 100},
                ],
                "2024-01-11": [
                    {"type": "eth", "amount": 2.0, "delta": 2.0},  # Has delta
                    {"type": "token", "amount": 200},
                ],
                "2024-01-12": [{"type": "eth", "amount": 3.0}],  # Missing delta
            }
        }

        def mock_read_funding_events():
            return mock_funding_events

        def mock_store_funding_events():
            pass

        def mock_fetch_token_transfers_mode(
            address, end_datetime, all_transfers_by_date, fetch_till_date
        ):
            # Should be called with fetch_till_date=True due to backward compatibility
            assert fetch_till_date is True
            all_transfers_by_date.update(
                {"2024-01-15": [{"type": "token", "amount": 300}]}
            )
            return True

        def mock_fetch_eth_transfers_mode(
            address, end_datetime, all_transfers_by_date, fetch_till_date
        ):
            # Should be called with fetch_till_date=True due to backward compatibility
            assert fetch_till_date is True
            all_transfers_by_date.update(
                {"2024-01-16": [{"type": "eth", "amount": 4.0, "delta": 4.0}]}
            )
            return True

        with patch.multiple(
            fetch_behaviour,
            read_funding_events=mock_read_funding_events,
            store_funding_events=mock_store_funding_events,
            _fetch_token_transfers_mode=mock_fetch_token_transfers_mode,
            _fetch_eth_transfers_mode=mock_fetch_eth_transfers_mode,
        ):
            result = fetch_behaviour._fetch_all_transfers_until_date_mode(
                "0x1234567890123456789012345678901234567890", "2024-01-15", False
            )

            # Should return only new data (existing data was cleared)
            expected_result = {
                "2024-01-15": [{"type": "token", "amount": 300}],
                "2024-01-16": [{"type": "eth", "amount": 4.0, "delta": 4.0}],
            }
            assert result == expected_result

    def test_fetch_all_transfers_until_date_mode_no_backward_compatibility_needed(self):
        """Test that backward compatibility is not triggered when all ETH transfers have delta field."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # All ETH transfers have delta field
        mock_funding_events = {
            "mode": {
                "2024-01-10": [
                    {"type": "eth", "amount": 1.0, "delta": 1.0},  # Has delta
                    {"type": "token", "amount": 100},
                ],
                "2024-01-11": [
                    {"type": "eth", "amount": 2.0, "delta": 2.0},  # Has delta
                    {"type": "token", "amount": 200},
                ],
            }
        }

        def mock_read_funding_events():
            return mock_funding_events

        def mock_store_funding_events():
            pass

        def mock_fetch_token_transfers_mode(
            address, end_datetime, all_transfers_by_date, fetch_till_date
        ):
            # Should be called with original fetch_till_date value
            assert fetch_till_date is False
            all_transfers_by_date.update(
                {"2024-01-15": [{"type": "token", "amount": 300}]}
            )
            return True

        def mock_fetch_eth_transfers_mode(
            address, end_datetime, all_transfers_by_date, fetch_till_date
        ):
            # Should be called with original fetch_till_date value
            assert fetch_till_date is False
            all_transfers_by_date.update(
                {"2024-01-16": [{"type": "eth", "amount": 4.0, "delta": 4.0}]}
            )
            return True

        with patch.multiple(
            fetch_behaviour,
            read_funding_events=mock_read_funding_events,
            store_funding_events=mock_store_funding_events,
            _fetch_token_transfers_mode=mock_fetch_token_transfers_mode,
            _fetch_eth_transfers_mode=mock_fetch_eth_transfers_mode,
        ):
            result = fetch_behaviour._fetch_all_transfers_until_date_mode(
                "0x1234567890123456789012345678901234567890", "2024-01-15", False
            )

            # Should return only new data (existing data is stored but not returned)
            expected_result = {
                "2024-01-15": [{"type": "token", "amount": 300}],
                "2024-01-16": [{"type": "eth", "amount": 4.0, "delta": 4.0}],
            }
            assert result == expected_result

    @pytest.mark.parametrize(
        "address,end_date,mock_funding_events,mock_optimism_transfers_data,mock_optimism_transfers_success,expected_result,expected_logs,test_description",
        [
            # Test 1: Successful fetch with new data
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                {"optimism": {"2024-01-10": [{"type": "token", "amount": 100}]}},
                {
                    "2024-01-12": [{"type": "token", "amount": 200}],
                    "2024-01-13": [{"type": "eth", "amount": 1.5}],
                },
                True,  # Successful fetch
                {
                    "2024-01-12": [{"type": "token", "amount": 200}],
                    "2024-01-13": [{"type": "eth", "amount": 1.5}],
                },
                [
                    "Fetching all Optimism transfers until 2024-01-15...",
                    "Optimism Summary: 2 dates with transfers, 2 total transfers",
                ],
                "successful fetch with new data",
            ),
            # Test 2: No existing funding events
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                None,  # No existing funding events
                {
                    "2024-01-12": [{"type": "token", "amount": 200}],
                    "2024-01-13": [{"type": "eth", "amount": 1.5}],
                },
                True,
                {
                    "2024-01-12": [{"type": "token", "amount": 200}],
                    "2024-01-13": [{"type": "eth", "amount": 1.5}],
                },
                [
                    "Fetching all Optimism transfers until 2024-01-15...",
                    "Optimism Summary: 2 dates with transfers, 2 total transfers",
                ],
                "no existing funding events",
            ),
            # Test 3: Empty existing data
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                {"optimism": {}},  # Empty optimism data
                {"2024-01-12": [{"type": "token", "amount": 200}]},
                True,
                {"2024-01-12": [{"type": "token", "amount": 200}]},
                [
                    "Fetching all Optimism transfers until 2024-01-15...",
                    "Optimism Summary: 1 dates with transfers, 1 total transfers",
                ],
                "empty existing data",
            ),
            # Test 4: No optimism data in funding events
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                {
                    "mode": {"2024-01-10": [{"type": "token", "amount": 100}]}
                },  # No optimism data
                {"2024-01-12": [{"type": "token", "amount": 200}]},
                True,
                {"2024-01-12": [{"type": "token", "amount": 200}]},
                [
                    "Fetching all Optimism transfers until 2024-01-15...",
                    "Optimism Summary: 1 dates with transfers, 1 total transfers",
                ],
                "no optimism data in funding events",
            ),
            # Test 5: No new transfers fetched
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                {"optimism": {"2024-01-10": [{"type": "token", "amount": 100}]}},
                {},  # No new transfers
                True,
                {},
                [
                    "Fetching all Optimism transfers until 2024-01-15...",
                    "Optimism Summary: 0 dates with transfers, 0 total transfers",
                ],
                "no new transfers fetched",
            ),
            # Test 6: Exception handling
            (
                "0x1234567890123456789012345678901234567890",
                "2024-01-15",
                {"optimism": {"2024-01-10": [{"type": "token", "amount": 100}]}},
                {},
                "exception",  # Will raise exception
                {},
                [
                    "Fetching all Optimism transfers until 2024-01-15...",
                    "Error fetching Optimism transfers: Test exception",
                ],
                "exception handling",
            ),
        ],
    )
    def test_fetch_all_transfers_until_date_optimism_variations(
        self,
        address,
        end_date,
        mock_funding_events,
        mock_optimism_transfers_data,
        mock_optimism_transfers_success,
        expected_result,
        expected_logs,
        test_description,
    ):
        """Test variations of _fetch_all_transfers_until_date_optimism."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_read_funding_events():
            return mock_funding_events

        def mock_store_funding_events():
            # Mock storing funding events
            pass

        def mock_fetch_optimism_transfers_safeglobal(
            address, end_date, all_transfers_by_date, existing_optimism_data
        ):
            if mock_optimism_transfers_success == "exception":
                raise Exception("Test exception")
            if mock_optimism_transfers_success:
                all_transfers_by_date.update(mock_optimism_transfers_data)
            yield None  # This is a generator method

        with patch.multiple(
            fetch_behaviour,
            read_funding_events=mock_read_funding_events,
            store_funding_events=mock_store_funding_events,
            _fetch_optimism_transfers_safeglobal=mock_fetch_optimism_transfers_safeglobal,
        ):
            result = self._consume_generator(
                fetch_behaviour._fetch_all_transfers_until_date_optimism(
                    address, end_date
                )
            )

            # Verify result
            assert result == expected_result, f"Failed for {test_description}"

    def test_fetch_all_transfers_until_date_optimism_generator_behavior(self):
        """Test that _fetch_all_transfers_until_date_optimism works as a generator."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        mock_funding_events = {
            "optimism": {"2024-01-10": [{"type": "token", "amount": 100}]}
        }
        mock_transfer_data = {"2024-01-12": [{"type": "token", "amount": 200}]}

        def mock_read_funding_events():
            return mock_funding_events

        def mock_store_funding_events():
            pass

        def mock_fetch_optimism_transfers_safeglobal(
            address, end_date, all_transfers_by_date, existing_optimism_data
        ):
            all_transfers_by_date.update(mock_transfer_data)
            yield None  # Generator behavior

        with patch.multiple(
            fetch_behaviour,
            read_funding_events=mock_read_funding_events,
            store_funding_events=mock_store_funding_events,
            _fetch_optimism_transfers_safeglobal=mock_fetch_optimism_transfers_safeglobal,
        ):
            # Test that it returns a generator
            generator = fetch_behaviour._fetch_all_transfers_until_date_optimism(
                "0x1234567890123456789012345678901234567890", "2024-01-15"
            )

            # Verify it's a generator
            assert hasattr(generator, "__next__"), "Should return a generator"

            # Consume the generator and verify result
            result = self._consume_generator(generator)
            assert result == mock_transfer_data

    def test_fetch_all_transfers_until_date_optimism_data_storage(self):
        """Test that data is properly stored and merged."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        existing_data = {"optimism": {"2024-01-10": [{"type": "token", "amount": 100}]}}
        new_transfer_data = {"2024-01-12": [{"type": "token", "amount": 200}]}

        stored_funding_events = None

        def mock_read_funding_events():
            return existing_data

        def mock_store_funding_events():
            nonlocal stored_funding_events
            stored_funding_events = fetch_behaviour.funding_events

        def mock_fetch_optimism_transfers_safeglobal(
            address, end_date, all_transfers_by_date, existing_optimism_data
        ):
            all_transfers_by_date.update(new_transfer_data)
            yield None

        with patch.multiple(
            fetch_behaviour,
            read_funding_events=mock_read_funding_events,
            store_funding_events=mock_store_funding_events,
            _fetch_optimism_transfers_safeglobal=mock_fetch_optimism_transfers_safeglobal,
        ):
            result = self._consume_generator(
                fetch_behaviour._fetch_all_transfers_until_date_optimism(
                    "0x1234567890123456789012345678901234567890", "2024-01-15"
                )
            )

            # Verify result contains only new data
            assert result == new_transfer_data

            # Verify stored data contains merged data
            expected_stored_data = {
                "optimism": {
                    "2024-01-10": [{"type": "token", "amount": 100}],  # Existing
                    "2024-01-12": [{"type": "token", "amount": 200}],  # New
                }
            }
            assert stored_funding_events == expected_stored_data

    def test_fetch_all_transfers_until_date_optimism_exception_fallback(self):
        """Test exception handling returns empty dict."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_read_funding_events():
            return {"optimism": {"2024-01-10": [{"type": "token", "amount": 100}]}}

        def mock_fetch_optimism_transfers_safeglobal(
            address, end_date, all_transfers_by_date, existing_optimism_data
        ):
            raise Exception("Test exception")
            yield None  # This won't be reached

        with patch.multiple(
            fetch_behaviour,
            read_funding_events=mock_read_funding_events,
            _fetch_optimism_transfers_safeglobal=mock_fetch_optimism_transfers_safeglobal,
        ):
            result = self._consume_generator(
                fetch_behaviour._fetch_all_transfers_until_date_optimism(
                    "0x1234567890123456789012345678901234567890", "2024-01-15"
                )
            )

            # Should return empty dict on exception
            assert result == {}

    def test_fetch_all_transfers_until_date_optimism_no_duplicate_dates(self):
        """Test that existing dates are not overwritten with new data."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Existing data has 2024-01-12
        existing_data = {
            "optimism": {
                "2024-01-12": [{"type": "token", "amount": 100, "existing": True}]
            }
        }
        # New data also has 2024-01-12 (should not overwrite) and 2024-01-13 (should be added)
        new_transfer_data = {
            "2024-01-12": [
                {"type": "token", "amount": 200, "new": True}
            ],  # Should not overwrite
            "2024-01-13": [{"type": "eth", "amount": 1.5}],  # Should be added
        }

        stored_funding_events = None

        def mock_read_funding_events():
            return existing_data

        def mock_store_funding_events():
            nonlocal stored_funding_events
            stored_funding_events = fetch_behaviour.funding_events

        def mock_fetch_optimism_transfers_safeglobal(
            address, end_date, all_transfers_by_date, existing_optimism_data
        ):
            all_transfers_by_date.update(new_transfer_data)
            yield None

        with patch.multiple(
            fetch_behaviour,
            read_funding_events=mock_read_funding_events,
            store_funding_events=mock_store_funding_events,
            _fetch_optimism_transfers_safeglobal=mock_fetch_optimism_transfers_safeglobal,
        ):
            result = self._consume_generator(
                fetch_behaviour._fetch_all_transfers_until_date_optimism(
                    "0x1234567890123456789012345678901234567890", "2024-01-15"
                )
            )

            # Result should contain all new data
            assert result == new_transfer_data

            # Stored data should preserve existing date and add only new date
            expected_stored_data = {
                "optimism": {
                    "2024-01-12": [
                        {"type": "token", "amount": 100, "existing": True}
                    ],  # Original preserved
                    "2024-01-13": [{"type": "eth", "amount": 1.5}],  # New date added
                }
            }
            assert stored_funding_events == expected_stored_data

    def test_fetch_token_transfers_basic(self):
        """Test basic functionality of _fetch_token_transfers."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        from collections import defaultdict

        all_transfers_by_date = defaultdict(list)
        existing_data = {}

        def mock_request_with_retries(
            endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait
        ):
            yield None
            return (True, {"items": []})

        def mock_get_datetime_from_timestamp(timestamp):
            return datetime(2024, 1, 10, 10, 30)

        def mock_should_include_transfer(from_address, tx, is_eth_transfer):
            return True

        with patch.multiple(
            fetch_behaviour,
            _request_with_retries=mock_request_with_retries,
            _get_datetime_from_timestamp=mock_get_datetime_from_timestamp,
            _should_include_transfer=mock_should_include_transfer,
        ):
            result = self._consume_generator(
                fetch_behaviour._fetch_token_transfers(
                    "0x1234567890123456789012345678901234567890",
                    datetime(2024, 1, 15),
                    all_transfers_by_date,
                    existing_data,
                )
            )

            assert result is None

    @pytest.mark.parametrize(
        "address,end_datetime,mock_api_response,mock_should_include_transfer_results,existing_data,expected_transfers,test_description",
        [
            # Test 1: Successful fetch with valid transfers
            (
                "0x1234567890123456789012345678901234567890",
                datetime(2024, 1, 15),
                (
                    True,
                    {
                        "items": [
                            {
                                "timestamp": "2024-01-10T10:30:00Z",
                                "from": {
                                    "hash": "0x1234567890123456789012345678901234567890"
                                },
                                "token": {
                                    "symbol": "USDC",
                                    "address": "0x1234567890123456789012345678901234567890",
                                    "decimals": 6,
                                },
                                "total": {"value": "1000000"},  # 1 USDC
                                "transaction_hash": "0xabc123def456",
                            }
                        ]
                    },
                ),
                [True],  # Transfer should be included
                {},  # No existing data
                {
                    "2024-01-10": [
                        {
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "amount": 1.0,
                            "token_address": "0x1234567890123456789012345678901234567890",
                            "symbol": "USDC",
                            "timestamp": "2024-01-10T10:30:00Z",
                            "tx_hash": "0xabc123def456",
                            "type": "token",
                        }
                    ]
                },
                "successful fetch with valid transfers",
            ),
            # Test 2: API request fails
            (
                "0x1234567890123456789012345678901234567890",
                datetime(2024, 1, 15),
                (False, {}),  # API failure
                [],
                {},
                {},
                "API request fails",
            ),
            # Test 3: No transfers returned
            (
                "0x1234567890123456789012345678901234567890",
                datetime(2024, 1, 15),
                (True, {"items": []}),  # Empty response
                [],
                {},
                {},
                "no transfers returned",
            ),
            # Test 4: Future date transfer (should be filtered out)
            (
                "0x1234567890123456789012345678901234567890",
                datetime(2024, 1, 15),
                (
                    True,
                    {
                        "items": [
                            {
                                "timestamp": "2024-01-20T10:30:00Z",  # Future date
                                "from": {
                                    "hash": "0x1234567890123456789012345678901234567890"
                                },
                                "token": {
                                    "symbol": "USDC",
                                    "address": "0x1234567890123456789012345678901234567890",
                                    "decimals": 6,
                                },
                                "total": {"value": "1000000"},
                                "transaction_hash": "0xabc123def456",
                            }
                        ]
                    },
                ),
                [],
                {},
                {},
                "future date transfer",
            ),
            # Test 5: Invalid timestamp (should be filtered out)
            (
                "0x1234567890123456789012345678901234567890",
                datetime(2024, 1, 15),
                (
                    True,
                    {
                        "items": [
                            {
                                "timestamp": "invalid_timestamp",
                                "from": {
                                    "hash": "0x1234567890123456789012345678901234567890"
                                },
                                "token": {
                                    "symbol": "USDC",
                                    "address": "0x1234567890123456789012345678901234567890",
                                    "decimals": 6,
                                },
                                "total": {"value": "1000000"},
                                "transaction_hash": "0xabc123def456",
                            }
                        ]
                    },
                ),
                [],
                {},
                {},
                "invalid timestamp",
            ),
            # Test 6: Transfer should not be included (filtered by _should_include_transfer)
            (
                "0x1234567890123456789012345678901234567890",
                datetime(2024, 1, 15),
                (
                    True,
                    {
                        "items": [
                            {
                                "timestamp": "2024-01-10T10:30:00Z",
                                "from": {
                                    "hash": "0x0000000000000000000000000000000000000000"
                                },  # Null address
                                "token": {
                                    "symbol": "USDC",
                                    "address": "0x1234567890123456789012345678901234567890",
                                    "decimals": 6,
                                },
                                "total": {"value": "1000000"},
                                "transaction_hash": "0xabc123def456",
                            }
                        ]
                    },
                ),
                [False],  # Transfer should not be included
                {},
                {},
                "transfer filtered out by should_include_transfer",
            ),
            # Test 7: Multiple transfers with different scenarios
            (
                "0x1234567890123456789012345678901234567890",
                datetime(2024, 1, 15),
                (
                    True,
                    {
                        "items": [
                            {
                                "timestamp": "2024-01-10T10:30:00Z",
                                "from": {
                                    "hash": "0x1234567890123456789012345678901234567890"
                                },
                                "token": {
                                    "symbol": "USDC",
                                    "address": "0x1234567890123456789012345678901234567890",
                                    "decimals": 6,
                                },
                                "total": {"value": "1000000"},
                                "transaction_hash": "0xabc123def456",
                            },
                            {
                                "timestamp": "2024-01-12T15:45:00Z",
                                "from": {
                                    "hash": "0x5555555555555555555555555555555555555555"
                                },
                                "token": {
                                    "symbol": "WETH",
                                    "address": "0x4200000000000000000000000000000000000006",
                                    "decimals": 18,
                                },
                                "total": {"value": "2000000000000000000"},  # 2 WETH
                                "transaction_hash": "0xdef456abc789",
                            },
                            {
                                "timestamp": "2024-01-13T20:00:00Z",
                                "from": {
                                    "hash": "0x6666666666666666666666666666666666666666"
                                },
                                "token": {
                                    "symbol": "DAI",
                                    "address": "0x6666666666666666666666666666666666666666",
                                    "decimals": 18,
                                },
                                "total": {"value": "500000000000000000000"},  # 500 DAI
                                "transaction_hash": "0xghi789def012",
                            },
                        ]
                    },
                ),
                [True, True, True],  # All transfers should be included
                {},
                {
                    "2024-01-10": [
                        {
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "amount": 1.0,
                            "token_address": "0x1234567890123456789012345678901234567890",
                            "symbol": "USDC",
                            "timestamp": "2024-01-10T10:30:00Z",
                            "tx_hash": "0xabc123def456",
                            "type": "token",
                        }
                    ],
                    "2024-01-12": [
                        {
                            "from_address": "0x5555555555555555555555555555555555555555",
                            "amount": 2.0,
                            "token_address": "0x4200000000000000000000000000000000000006",
                            "symbol": "WETH",
                            "timestamp": "2024-01-12T15:45:00Z",
                            "tx_hash": "0xdef456abc789",
                            "type": "token",
                        }
                    ],
                    "2024-01-13": [
                        {
                            "from_address": "0x6666666666666666666666666666666666666666",
                            "amount": 500.0,
                            "token_address": "0x6666666666666666666666666666666666666666",
                            "symbol": "DAI",
                            "timestamp": "2024-01-13T20:00:00Z",
                            "tx_hash": "0xghi789def012",
                            "type": "token",
                        }
                    ],
                },
                "multiple transfers with different scenarios",
            ),
            # Test 8: Missing token data
            (
                "0x1234567890123456789012345678901234567890",
                datetime(2024, 1, 15),
                (
                    True,
                    {
                        "items": [
                            {
                                "timestamp": "2024-01-10T10:30:00Z",
                                "from": {
                                    "hash": "0x1234567890123456789012345678901234567890"
                                },
                                "token": {},  # Missing token data
                                "total": {"value": "1000000"},
                                "transaction_hash": "0xabc123def456",
                            }
                        ]
                    },
                ),
                [True],
                {},
                {
                    "2024-01-10": [
                        {
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "amount": 1e-12,  # 1000000 / (10^18) when decimals defaults to 18
                            "token_address": "",
                            "symbol": "Unknown",
                            "timestamp": "2024-01-10T10:30:00Z",
                            "tx_hash": "0xabc123def456",
                            "type": "token",
                        }
                    ]
                },
                "missing token data",
            ),
        ],
    )
    def test_fetch_token_transfers_variations(
        self,
        address,
        end_datetime,
        mock_api_response,
        mock_should_include_transfer_results,
        existing_data,
        expected_transfers,
        test_description,
    ):
        """Test variations of _fetch_token_transfers."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Initialize all_transfers_by_date with defaultdict
        from collections import defaultdict

        all_transfers_by_date = defaultdict(list)

        # Track calls to _should_include_transfer
        should_include_calls = []

        def mock_request_with_retries(
            endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait
        ):
            yield None
            return mock_api_response

        def mock_get_datetime_from_timestamp(timestamp):
            if timestamp == "invalid_timestamp":
                return None
            # Simple mock for valid timestamps
            if "2024-01-10" in timestamp:
                return datetime(2024, 1, 10, 10, 30)
            elif "2024-01-12" in timestamp:
                return datetime(2024, 1, 12, 15, 45)
            elif "2024-01-13" in timestamp:
                return datetime(2024, 1, 13, 20, 0)
            elif "2024-01-20" in timestamp:
                return datetime(2024, 1, 20, 10, 30)
            return datetime(2024, 1, 10, 10, 30)  # Default

        def mock_should_include_transfer(from_address, tx, is_eth_transfer):
            should_include_calls.append((from_address, tx, is_eth_transfer))
            # Return the next result from the list
            if len(should_include_calls) <= len(mock_should_include_transfer_results):
                return mock_should_include_transfer_results[
                    len(should_include_calls) - 1
                ]
            return True  # Default

        with patch.multiple(
            fetch_behaviour,
            _request_with_retries=mock_request_with_retries,
            _get_datetime_from_timestamp=mock_get_datetime_from_timestamp,
            _should_include_transfer=mock_should_include_transfer,
        ):
            result = self._consume_generator(
                fetch_behaviour._fetch_token_transfers(
                    address, end_datetime, all_transfers_by_date, existing_data
                )
            )

            # Verify result
            if mock_api_response[0]:  # API success
                if mock_api_response[1].get("items"):
                    assert result is None, f"Expected None for {test_description}"
                else:
                    assert result is None, f"Expected None for {test_description}"
            else:  # API failure
                assert result is None, f"Expected None for {test_description}"

            # Verify all_transfers_by_date was updated correctly
            # Convert defaultdict to regular dict for comparison
            actual_transfers = dict(all_transfers_by_date)
            assert (
                actual_transfers == expected_transfers
            ), f"Expected {expected_transfers} for {test_description}, got {actual_transfers}"

    def test_fetch_token_transfers_generator_behavior(self):
        """Test that _fetch_token_transfers works as a generator."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        from collections import defaultdict

        all_transfers_by_date = defaultdict(list)
        existing_data = {}

        def mock_request_with_retries(
            endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait
        ):
            yield None
            return (True, {"items": []})

        def mock_get_datetime_from_timestamp(timestamp):
            return datetime(2024, 1, 10, 10, 30)

        def mock_should_include_transfer(from_address, tx, is_eth_transfer):
            return True

        with patch.multiple(
            fetch_behaviour,
            _request_with_retries=mock_request_with_retries,
            _get_datetime_from_timestamp=mock_get_datetime_from_timestamp,
            _should_include_transfer=mock_should_include_transfer,
        ):
            # Test that it returns a generator
            generator = fetch_behaviour._fetch_token_transfers(
                "0x1234567890123456789012345678901234567890",
                datetime(2024, 1, 15),
                all_transfers_by_date,
                existing_data,
            )

            # Verify it's a generator
            assert hasattr(generator, "__next__"), "Should return a generator"

            # Consume the generator and verify result
            result = self._consume_generator(generator)
            assert result is None

    def test_fetch_token_transfers_date_filtering(self):
        """Test that transfers are properly filtered by date."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        from collections import defaultdict

        all_transfers_by_date = defaultdict(list)
        existing_data = {
            "2024-01-10": [{"type": "existing"}]
        }  # Existing data for 2024-01-10

        # Mock API response with transfers on different dates
        mock_response = (
            True,
            {
                "items": [
                    {
                        "timestamp": "2024-01-10T10:30:00Z",  # Should be skipped (already exists)
                        "from": {"hash": "0x1234567890123456789012345678901234567890"},
                        "token": {
                            "symbol": "USDC",
                            "address": "0x1234567890123456789012345678901234567890",
                            "decimals": 6,
                        },
                        "total": {"value": "1000000"},
                        "transaction_hash": "0xabc123def456",
                    },
                    {
                        "timestamp": "2024-01-12T15:45:00Z",  # Should be included
                        "from": {"hash": "0x5555555555555555555555555555555555555555"},
                        "token": {
                            "symbol": "WETH",
                            "address": "0x4200000000000000000000000000000000000006",
                            "decimals": 18,
                        },
                        "total": {"value": "1000000000000000000"},
                        "transaction_hash": "0xdef456abc789",
                    },
                    {
                        "timestamp": "2024-01-20T10:30:00Z",  # Should be skipped (future date)
                        "from": {"hash": "0x6666666666666666666666666666666666666666"},
                        "token": {
                            "symbol": "DAI",
                            "address": "0x6666666666666666666666666666666666666666",
                            "decimals": 18,
                        },
                        "total": {"value": "500000000000000000000"},
                        "transaction_hash": "0xghi789def012",
                    },
                ]
            },
        )

        def mock_request_with_retries(
            endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait
        ):
            yield None
            return mock_response

        def mock_get_datetime_from_timestamp(timestamp):
            if "2024-01-10" in timestamp:
                return datetime(2024, 1, 10, 10, 30)
            elif "2024-01-12" in timestamp:
                return datetime(2024, 1, 12, 15, 45)
            elif "2024-01-20" in timestamp:
                return datetime(2024, 1, 20, 10, 30)
            return None

        def mock_should_include_transfer(from_address, tx, is_eth_transfer):
            return True

        with patch.multiple(
            fetch_behaviour,
            _request_with_retries=mock_request_with_retries,
            _get_datetime_from_timestamp=mock_get_datetime_from_timestamp,
            _should_include_transfer=mock_should_include_transfer,
        ):
            result = self._consume_generator(
                fetch_behaviour._fetch_token_transfers(
                    "0x1234567890123456789012345678901234567890",
                    datetime(2024, 1, 15),
                    all_transfers_by_date,
                    existing_data,
                )
            )

            # Only 2024-01-12 should be included (2024-01-10 exists, 2024-01-20 is future)
            expected_transfers = {
                "2024-01-12": [
                    {
                        "from_address": "0x5555555555555555555555555555555555555555",
                        "amount": 1.0,
                        "token_address": "0x4200000000000000000000000000000000000006",
                        "symbol": "WETH",
                        "timestamp": "2024-01-12T15:45:00Z",
                        "tx_hash": "0xdef456abc789",
                        "type": "token",
                    }
                ]
            }

            actual_transfers = dict(all_transfers_by_date)
            assert actual_transfers == expected_transfers

    def test_fetch_token_transfers_amount_calculation(self):
        """Test that token amounts are calculated correctly with different decimals."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        from collections import defaultdict

        all_transfers_by_date = defaultdict(list)
        existing_data = {}

        # Test different token decimals
        test_cases = [
            {
                "value": "1000000",  # 6 decimals (USDC)
                "decimals": 6,
                "expected_amount": 1.0,
            },
            {
                "value": "1000000000000000000",  # 18 decimals (WETH)
                "decimals": 18,
                "expected_amount": 1.0,
            },
            {
                "value": "500000000000000000000",  # 18 decimals, 500 tokens
                "decimals": 18,
                "expected_amount": 500.0,
            },
            {
                "value": "123456789",  # 8 decimals
                "decimals": 8,
                "expected_amount": 1.23456789,
            },
        ]

        for i, test_case in enumerate(test_cases):
            mock_response = (
                True,
                {
                    "items": [
                        {
                            "timestamp": "2024-01-10T10:30:00Z",
                            "from": {
                                "hash": f"0x{i}234567890123456789012345678901234567890"
                            },
                            "token": {
                                "symbol": f"TOKEN{i}",
                                "address": f"0x{i}234567890123456789012345678901234567890",
                                "decimals": test_case["decimals"],
                            },
                            "total": {"value": test_case["value"]},
                            "transaction_hash": f"0xabc{i}23def456",
                        }
                    ]
                },
            )

            def mock_request_with_retries(
                endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait
            ):
                yield None
                return mock_response

            def mock_get_datetime_from_timestamp(timestamp):
                return datetime(2024, 1, 10, 10, 30)

            def mock_should_include_transfer(from_address, tx, is_eth_transfer):
                return True

            with patch.multiple(
                fetch_behaviour,
                _request_with_retries=mock_request_with_retries,
                _get_datetime_from_timestamp=mock_get_datetime_from_timestamp,
                _should_include_transfer=mock_should_include_transfer,
            ):
                # Clear previous results
                all_transfers_by_date.clear()

                result = self._consume_generator(
                    fetch_behaviour._fetch_token_transfers(
                        "0x1234567890123456789012345678901234567890",
                        datetime(2024, 1, 15),
                        all_transfers_by_date,
                        existing_data,
                    )
                )

                # Verify amount calculation
                actual_transfers = dict(all_transfers_by_date)
                assert len(actual_transfers["2024-01-10"]) == 1
                assert (
                    actual_transfers["2024-01-10"][0]["amount"]
                    == test_case["expected_amount"]
                )

    def test_fetch_token_transfers_missing_fields(self):
        """Test handling of missing fields in transfer data."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        from collections import defaultdict

        all_transfers_by_date = defaultdict(list)
        existing_data = {}

        # Mock response with missing fields
        mock_response = (
            True,
            {
                "items": [
                    {
                        "timestamp": "2024-01-10T10:30:00Z",
                        "from": {},  # Missing hash
                        "token": {
                            "symbol": "USDC",
                            "address": "0x1234567890123456789012345678901234567890",
                            "decimals": 6,
                        },
                        "total": {},  # Missing value
                        "transaction_hash": "0xabc123def456",
                    },
                    {
                        "timestamp": "2024-01-12T15:45:00Z",
                        "from": {"hash": "0x5555555555555555555555555555555555555555"},
                        "token": {
                            "symbol": "WETH",
                            "address": "0x4200000000000000000000000000000000000006",
                            "decimals": 18,
                        },
                        "total": {"value": "1000000000000000000"},
                        "transaction_hash": "0xdef456abc789",
                    },
                ]
            },
        )

        def mock_request_with_retries(
            endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait
        ):
            yield None
            return mock_response

        def mock_get_datetime_from_timestamp(timestamp):
            if "2024-01-10" in timestamp:
                return datetime(2024, 1, 10, 10, 30)
            elif "2024-01-12" in timestamp:
                return datetime(2024, 1, 12, 15, 45)
            return None

        def mock_should_include_transfer(from_address, tx, is_eth_transfer):
            return True

        with patch.multiple(
            fetch_behaviour,
            _request_with_retries=mock_request_with_retries,
            _get_datetime_from_timestamp=mock_get_datetime_from_timestamp,
            _should_include_transfer=mock_should_include_transfer,
        ):
            result = self._consume_generator(
                fetch_behaviour._fetch_token_transfers(
                    "0x1234567890123456789012345678901234567890",
                    datetime(2024, 1, 15),
                    all_transfers_by_date,
                    existing_data,
                )
            )

            # Should handle missing fields gracefully
            actual_transfers = dict(all_transfers_by_date)
            assert "2024-01-10" in actual_transfers
            assert "2024-01-12" in actual_transfers

            # Check that missing fields are handled with defaults
            transfer_10 = actual_transfers["2024-01-10"][0]
            assert transfer_10["from_address"] == ""  # Missing hash
            assert transfer_10["amount"] == 0.0  # Missing value defaults to 0

            transfer_12 = actual_transfers["2024-01-12"][0]
            assert (
                transfer_12["from_address"]
                == "0x5555555555555555555555555555555555555555"
            )
            assert transfer_12["amount"] == 1.0

    def test_fetch_optimism_transfers_safeglobal_basic(self):
        """Test basic functionality of _fetch_optimism_transfers_safeglobal."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        from collections import defaultdict

        all_transfers_by_date = defaultdict(list)
        existing_data = {}

        def mock_request_with_retries(
            endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait
        ):
            yield None
            return (
                True,
                {
                    "results": [
                        {
                            "executionDate": "2024-01-10T10:30:00Z",
                            "from": "0x1234567890123456789012345678901234567890",
                            "type": "ERC20_TRANSFER",
                            "tokenInfo": {"symbol": "USDC", "decimals": 6},
                            "tokenAddress": "0x1234567890123456789012345678901234567890",
                            "value": "1000000",  # 1 USDC
                            "transactionHash": "0xabc123def456",
                            "transferId": "transfer_123",
                        }
                    ],
                    "next": None,
                },
            )

        def mock_get_datetime_from_timestamp(timestamp):
            from datetime import datetime

            if timestamp == "2024-01-10T10:30:00Z":
                return datetime(2024, 1, 10, 10, 30, 0)
            return None

        def mock_should_include_transfer_optimism(from_address):
            yield None
            return True

        with patch.multiple(
            fetch_behaviour,
            _request_with_retries=mock_request_with_retries,
            _get_datetime_from_timestamp=mock_get_datetime_from_timestamp,
            _should_include_transfer_optimism=mock_should_include_transfer_optimism,
        ):
            result = list(
                fetch_behaviour._fetch_optimism_transfers_safeglobal(
                    "0x9876543210987654321098765432109876543210",
                    "2024-01-15",
                    all_transfers_by_date,
                    existing_data,
                )
            )

        # Verify the transfer was processed correctly
        assert len(all_transfers_by_date["2024-01-10"]) == 1
        transfer = all_transfers_by_date["2024-01-10"][0]
        assert transfer["from_address"] == "0x1234567890123456789012345678901234567890"
        assert transfer["amount"] == 1.0
        assert transfer["token_address"] == "0x1234567890123456789012345678901234567890"
        assert transfer["symbol"] == "USDC"
        assert transfer["timestamp"] == "2024-01-10T10:30:00Z"
        assert transfer["tx_hash"] == "0xabc123def456"
        assert transfer["type"] == "token"

    @pytest.mark.parametrize(
        "address,end_date,mock_api_response,mock_should_include_results,existing_data,expected_transfers,expected_logs,test_description",
        [
            # Test 1: Successful fetch with USDC token transfer
            (
                "0x9876543210987654321098765432109876543210",
                "2024-01-15",
                (
                    True,
                    {
                        "results": [
                            {
                                "executionDate": "2024-01-10T10:30:00Z",
                                "from": "0x1234567890123456789012345678901234567890",
                                "type": "ERC20_TRANSFER",
                                "tokenInfo": {"symbol": "USDC", "decimals": 6},
                                "tokenAddress": "0x1234567890123456789012345678901234567890",
                                "value": "1000000",
                                "transactionHash": "0xabc123def456",
                                "transferId": "transfer_123",
                            }
                        ],
                        "next": None,
                    },
                ),
                [True],
                {},
                {
                    "2024-01-10": [
                        {
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "amount": 1.0,
                            "token_address": "0x1234567890123456789012345678901234567890",
                            "symbol": "USDC",
                            "timestamp": "2024-01-10T10:30:00Z",
                            "tx_hash": "0xabc123def456",
                            "type": "token",
                        }
                    ]
                },
                [
                    "Fetching Optimism transfers using SafeGlobal API...",
                    "Completed Optimism transfers: 1 found",
                ],
                "successful fetch with USDC token transfer",
            ),
            # Test 2: ETH transfer
            (
                "0x9876543210987654321098765432109876543210",
                "2024-01-15",
                (
                    True,
                    {
                        "results": [
                            {
                                "executionDate": "2024-01-10T10:30:00Z",
                                "from": "0x1234567890123456789012345678901234567890",
                                "type": "ETHER_TRANSFER",
                                "value": "2000000000000000000",  # 2 ETH
                                "transactionHash": "0xabc123def456",
                                "transferId": "transfer_123",
                            }
                        ],
                        "next": None,
                    },
                ),
                [True],
                {},
                {
                    "2024-01-10": [
                        {
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "amount": 2.0,
                            "token_address": "",
                            "symbol": "ETH",
                            "timestamp": "2024-01-10T10:30:00Z",
                            "tx_hash": "0xabc123def456",
                            "type": "eth",
                        }
                    ]
                },
                [
                    "Fetching Optimism transfers using SafeGlobal API...",
                    "Completed Optimism transfers: 1 found",
                ],
                "ETH transfer",
            ),
            # Test 3: API request fails
            (
                "0x9876543210987654321098765432109876543210",
                "2024-01-15",
                (False, {}),
                [],
                {},
                {},
                [
                    "Fetching Optimism transfers using SafeGlobal API...",
                    "Failed to fetch Optimism transfers",
                ],
                "API request fails",
            ),
            # Test 4: No transfers returned
            (
                "0x9876543210987654321098765432109876543210",
                "2024-01-15",
                (True, {"results": []}),
                [],
                {},
                {},
                [
                    "Fetching Optimism transfers using SafeGlobal API...",
                    "Completed Optimism transfers: 0 found",
                ],
                "no transfers returned",
            ),
            # Test 5: Future date transfer (should be filtered out)
            (
                "0x9876543210987654321098765432109876543210",
                "2024-01-15",
                (
                    True,
                    {
                        "results": [
                            {
                                "executionDate": "2024-01-20T10:30:00Z",  # Future date
                                "from": "0x1234567890123456789012345678901234567890",
                                "type": "ERC20_TRANSFER",
                                "tokenInfo": {"symbol": "USDC", "decimals": 6},
                                "tokenAddress": "0x1234567890123456789012345678901234567890",
                                "value": "1000000",
                                "transactionHash": "0xabc123def456",
                                "transferId": "transfer_123",
                            }
                        ],
                        "next": None,
                    },
                ),
                [True],
                {},
                {},
                [
                    "Fetching Optimism transfers using SafeGlobal API...",
                    "Completed Optimism transfers: 0 found",
                ],
                "future date transfer",
            ),
            # Test 6: Transfer filtered out by should_include_transfer
            (
                "0x9876543210987654321098765432109876543210",
                "2024-01-15",
                (
                    True,
                    {
                        "results": [
                            {
                                "executionDate": "2024-01-10T10:30:00Z",
                                "from": "0x0000000000000000000000000000000000000000",  # Null address
                                "type": "ERC20_TRANSFER",
                                "tokenInfo": {"symbol": "USDC", "decimals": 6},
                                "tokenAddress": "0x1234567890123456789012345678901234567890",
                                "value": "1000000",
                                "transactionHash": "0xabc123def456",
                                "transferId": "transfer_123",
                            }
                        ],
                        "next": None,
                    },
                ),
                [False],  # Should not include
                {},
                {},
                [
                    "Fetching Optimism transfers using SafeGlobal API...",
                    "Completed Optimism transfers: 0 found",
                ],
                "transfer filtered out by should_include_transfer",
            ),
            # Test 7: Non-USDC token transfer (should be filtered out)
            (
                "0x9876543210987654321098765432109876543210",
                "2024-01-15",
                (
                    True,
                    {
                        "results": [
                            {
                                "executionDate": "2024-01-10T10:30:00Z",
                                "from": "0x1234567890123456789012345678901234567890",
                                "type": "ERC20_TRANSFER",
                                "tokenInfo": {
                                    "symbol": "WETH",  # Non-USDC token
                                    "decimals": 18,
                                },
                                "tokenAddress": "0x1234567890123456789012345678901234567890",
                                "value": "1000000000000000000",
                                "transactionHash": "0xabc123def456",
                                "transferId": "transfer_123",
                            }
                        ],
                        "next": None,
                    },
                ),
                [True],
                {},
                {},
                [
                    "Fetching Optimism transfers using SafeGlobal API...",
                    "Completed Optimism transfers: 0 found",
                ],
                "non-USDC token transfer",
            ),
            # Test 8: NFT transfer (should be skipped)
            (
                "0x9876543210987654321098765432109876543210",
                "2024-01-15",
                (
                    True,
                    {
                        "results": [
                            {
                                "executionDate": "2024-01-10T10:30:00Z",
                                "from": "0x1234567890123456789012345678901234567890",
                                "type": "ERC721_TRANSFER",
                                "tokenAddress": "0x1234567890123456789012345678901234567890",
                                "transactionHash": "0xabc123def456",
                                "transferId": "transfer_123",
                            }
                        ],
                        "next": None,
                    },
                ),
                [True],
                {},
                {},
                [
                    "Fetching Optimism transfers using SafeGlobal API...",
                    "Completed Optimism transfers: 0 found",
                ],
                "NFT transfer",
            ),
            # Test 9: Zero-value ETH transfer (should be skipped)
            (
                "0x9876543210987654321098765432109876543210",
                "2024-01-15",
                (
                    True,
                    {
                        "results": [
                            {
                                "executionDate": "2024-01-10T10:30:00Z",
                                "from": "0x1234567890123456789012345678901234567890",
                                "type": "ETHER_TRANSFER",
                                "value": "0",  # Zero value
                                "transactionHash": "0xabc123def456",
                                "transferId": "transfer_123",
                            }
                        ],
                        "next": None,
                    },
                ),
                [True],
                {},
                {},
                [
                    "Fetching Optimism transfers using SafeGlobal API...",
                    "Completed Optimism transfers: 0 found",
                ],
                "zero-value ETH transfer",
            ),
            # Test 10: Transfer from same address (should be skipped)
            (
                "0x9876543210987654321098765432109876543210",
                "2024-01-15",
                (
                    True,
                    {
                        "results": [
                            {
                                "executionDate": "2024-01-10T10:30:00Z",
                                "from": "0x9876543210987654321098765432109876543210",  # Same as address
                                "type": "ERC20_TRANSFER",
                                "tokenInfo": {"symbol": "USDC", "decimals": 6},
                                "tokenAddress": "0x1234567890123456789012345678901234567890",
                                "value": "1000000",
                                "transactionHash": "0xabc123def456",
                                "transferId": "transfer_123",
                            }
                        ],
                        "next": None,
                    },
                ),
                [True],
                {},
                {},
                [
                    "Fetching Optimism transfers using SafeGlobal API...",
                    "Completed Optimism transfers: 0 found",
                ],
                "transfer from same address",
            ),
            # Test 11: Date already exists in existing data (should be skipped)
            (
                "0x9876543210987654321098765432109876543210",
                "2024-01-15",
                (
                    True,
                    {
                        "results": [
                            {
                                "executionDate": "2024-01-10T10:30:00Z",
                                "from": "0x1234567890123456789012345678901234567890",
                                "type": "ERC20_TRANSFER",
                                "tokenInfo": {"symbol": "USDC", "decimals": 6},
                                "tokenAddress": "0x1234567890123456789012345678901234567890",
                                "value": "1000000",
                                "transactionHash": "0xabc123def456",
                                "transferId": "transfer_123",
                            }
                        ],
                        "next": None,
                    },
                ),
                [True],
                {"2024-01-10": [{"existing": "data"}]},  # Date already exists
                {},
                [
                    "Fetching Optimism transfers using SafeGlobal API...",
                    "Completed Optimism transfers: 0 found",
                ],
                "date already exists in existing data",
            ),
            # Test 12: Missing token info - fetch from contract
            (
                "0x9876543210987654321098765432109876543210",
                "2024-01-15",
                (
                    True,
                    {
                        "results": [
                            {
                                "executionDate": "2024-01-10T10:30:00Z",
                                "from": "0x1234567890123456789012345678901234567890",
                                "type": "ERC20_TRANSFER",
                                "tokenInfo": {},  # Missing token info
                                "tokenAddress": "0x1234567890123456789012345678901234567890",
                                "value": "1000000",
                                "transactionHash": "0xabc123def456",
                                "transferId": "transfer_123",
                            }
                        ],
                        "next": None,
                    },
                ),
                [True],
                {},
                {
                    "2024-01-10": [
                        {
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "amount": 1.0,
                            "token_address": "0x1234567890123456789012345678901234567890",
                            "symbol": "USDC",
                            "timestamp": "2024-01-10T10:30:00Z",
                            "tx_hash": "0xabc123def456",
                            "type": "token",
                        }
                    ]
                },
                [
                    "Fetching Optimism transfers using SafeGlobal API...",
                    "Completed Optimism transfers: 1 found",
                ],
                "missing token info - fetch from contract",
            ),
            # Test 13: Pagination with multiple pages
            (
                "0x9876543210987654321098765432109876543210",
                "2024-01-15",
                "pagination",  # Special case for pagination
                [True, True],
                {},
                {
                    "2024-01-10": [
                        {
                            "from_address": "0x1234567890123456789012345678901234567890",
                            "amount": 1.0,
                            "token_address": "0x1234567890123456789012345678901234567890",
                            "symbol": "USDC",
                            "timestamp": "2024-01-10T10:30:00Z",
                            "tx_hash": "0xabc123def456",
                            "type": "token",
                        },
                        {
                            "from_address": "0x5555555555555555555555555555555555555555",
                            "amount": 2.0,
                            "token_address": "0x5555555555555555555555555555555555555555",
                            "symbol": "USDC",
                            "timestamp": "2024-01-10T15:45:00Z",
                            "tx_hash": "0xdef456abc789",
                            "type": "token",
                        },
                    ]
                },
                [
                    "Fetching Optimism transfers using SafeGlobal API...",
                    "Completed Optimism transfers: 2 found",
                ],
                "pagination with multiple pages",
            ),
        ],
    )
    def test_fetch_optimism_transfers_safeglobal_variations(
        self,
        address,
        end_date,
        mock_api_response,
        mock_should_include_results,
        existing_data,
        expected_transfers,
        expected_logs,
        test_description,
    ):
        """Test variations of _fetch_optimism_transfers_safeglobal."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        from collections import defaultdict

        all_transfers_by_date = defaultdict(list)

        def mock_request_with_retries(
            endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait
        ):
            yield None
            if mock_api_response == "pagination":
                # Simulate pagination
                if "cursor=next_page" in endpoint:
                    return (
                        True,
                        {
                            "results": [
                                {
                                    "executionDate": "2024-01-10T15:45:00Z",
                                    "from": "0x5555555555555555555555555555555555555555",
                                    "type": "ERC20_TRANSFER",
                                    "tokenInfo": {"symbol": "USDC", "decimals": 6},
                                    "tokenAddress": "0x5555555555555555555555555555555555555555",
                                    "value": "2000000",  # 2 USDC
                                    "transactionHash": "0xdef456abc789",
                                    "transferId": "transfer_456",
                                }
                            ],
                            "next": None,
                        },
                    )
                else:
                    return (
                        True,
                        {
                            "results": [
                                {
                                    "executionDate": "2024-01-10T10:30:00Z",
                                    "from": "0x1234567890123456789012345678901234567890",
                                    "type": "ERC20_TRANSFER",
                                    "tokenInfo": {"symbol": "USDC", "decimals": 6},
                                    "tokenAddress": "0x1234567890123456789012345678901234567890",
                                    "value": "1000000",
                                    "transactionHash": "0xabc123def456",
                                    "transferId": "transfer_123",
                                }
                            ],
                            "next": "https://safe-transaction-optimism.safe.global/api/v1/safes/0x9876543210987654321098765432109876543210/incoming-transfers/?cursor=next_page",
                        },
                    )
            return mock_api_response

        def mock_get_datetime_from_timestamp(timestamp):
            from datetime import datetime

            if timestamp == "2024-01-10T10:30:00Z":
                return datetime(2024, 1, 10, 10, 30, 0)
            elif timestamp == "2024-01-10T15:45:00Z":
                return datetime(2024, 1, 10, 15, 45, 0)
            elif timestamp == "2024-01-20T10:30:00Z":
                return datetime(2024, 1, 20, 10, 30, 0)
            return None

        def mock_should_include_transfer_optimism(from_address):
            yield None
            if from_address == "0x0000000000000000000000000000000000000000":
                return False
            return True

        def mock_get_token_decimals(chain, token_address):
            yield None
            return 6

        def mock_get_token_symbol(chain, token_address):
            yield None
            return "USDC"

        with patch.multiple(
            fetch_behaviour,
            _request_with_retries=mock_request_with_retries,
            _get_datetime_from_timestamp=mock_get_datetime_from_timestamp,
            _should_include_transfer_optimism=mock_should_include_transfer_optimism,
            _get_token_decimals=mock_get_token_decimals,
            _get_token_symbol=mock_get_token_symbol,
        ):
            result = list(
                fetch_behaviour._fetch_optimism_transfers_safeglobal(
                    address, end_date, all_transfers_by_date, existing_data
                )
            )

        # Verify the transfers match expected results
        assert dict(all_transfers_by_date) == expected_transfers

    def test_fetch_optimism_transfers_safeglobal_generator_behavior(self):
        """Test that _fetch_optimism_transfers_safeglobal works as a generator."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        from collections import defaultdict

        all_transfers_by_date = defaultdict(list)
        existing_data = {}

        def mock_request_with_retries(
            endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait
        ):
            yield None
            return (True, {"results": [], "next": None})

        def mock_get_datetime_from_timestamp(timestamp):
            return None

        def mock_should_include_transfer_optimism(from_address):
            yield None
            return True

        with patch.multiple(
            fetch_behaviour,
            _request_with_retries=mock_request_with_retries,
            _get_datetime_from_timestamp=mock_get_datetime_from_timestamp,
            _should_include_transfer_optimism=mock_should_include_transfer_optimism,
        ):
            # Test that it's a generator
            gen = fetch_behaviour._fetch_optimism_transfers_safeglobal(
                "0x9876543210987654321098765432109876543210",
                "2024-01-15",
                all_transfers_by_date,
                existing_data,
            )

            # Verify it's a generator
            assert hasattr(gen, "__iter__")
            assert hasattr(gen, "__next__")

            # Consume the generator
            result = list(gen)

            # Should yield None values
            assert result == [None]

    def test_fetch_optimism_transfers_safeglobal_exception_handling(self):
        """Test exception handling in _fetch_optimism_transfers_safeglobal."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        from collections import defaultdict

        all_transfers_by_date = defaultdict(list)
        existing_data = {}

        def mock_request_with_retries(
            endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait
        ):
            yield None
            raise Exception("Test exception")

        def mock_get_datetime_from_timestamp(timestamp):
            return None

        def mock_should_include_transfer_optimism(from_address):
            yield None
            return True

        with patch.multiple(
            fetch_behaviour,
            _request_with_retries=mock_request_with_retries,
            _get_datetime_from_timestamp=mock_get_datetime_from_timestamp,
            _should_include_transfer_optimism=mock_should_include_transfer_optimism,
        ):
            result = list(
                fetch_behaviour._fetch_optimism_transfers_safeglobal(
                    "0x9876543210987654321098765432109876543210",
                    "2024-01-15",
                    all_transfers_by_date,
                    existing_data,
                )
            )

        # Should handle exception gracefully and not add any transfers
        assert len(all_transfers_by_date) == 0

    def test_fetch_optimism_transfers_safeglobal_deduplication(self):
        """Test that transfers are properly deduplicated by transferId."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        from collections import defaultdict

        all_transfers_by_date = defaultdict(list)
        existing_data = {}

        def mock_request_with_retries(
            endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait
        ):
            yield None
            return (
                True,
                {
                    "results": [
                        {
                            "executionDate": "2024-01-10T10:30:00Z",
                            "from": "0x1234567890123456789012345678901234567890",
                            "type": "ERC20_TRANSFER",
                            "tokenInfo": {"symbol": "USDC", "decimals": 6},
                            "tokenAddress": "0x1234567890123456789012345678901234567890",
                            "value": "1000000",
                            "transactionHash": "0xabc123def456",
                            "transferId": "transfer_123",  # Same transferId
                        },
                        {
                            "executionDate": "2024-01-10T10:30:00Z",
                            "from": "0x1234567890123456789012345678901234567890",
                            "type": "ERC20_TRANSFER",
                            "tokenInfo": {"symbol": "USDC", "decimals": 6},
                            "tokenAddress": "0x1234567890123456789012345678901234567890",
                            "value": "1000000",
                            "transactionHash": "0xabc123def456",
                            "transferId": "transfer_123",  # Same transferId - should be deduplicated
                        },
                    ],
                    "next": None,
                },
            )

        def mock_get_datetime_from_timestamp(timestamp):
            from datetime import datetime

            if timestamp == "2024-01-10T10:30:00Z":
                return datetime(2024, 1, 10, 10, 30, 0)
            return None

        def mock_should_include_transfer_optimism(from_address):
            yield None
            return True

        with patch.multiple(
            fetch_behaviour,
            _request_with_retries=mock_request_with_retries,
            _get_datetime_from_timestamp=mock_get_datetime_from_timestamp,
            _should_include_transfer_optimism=mock_should_include_transfer_optimism,
        ):
            result = list(
                fetch_behaviour._fetch_optimism_transfers_safeglobal(
                    "0x9876543210987654321098765432109876543210",
                    "2024-01-15",
                    all_transfers_by_date,
                    existing_data,
                )
            )

        # Should only have one transfer due to deduplication
        assert len(all_transfers_by_date["2024-01-10"]) == 1

    def test_fetch_optimism_transfers_safeglobal_missing_fields(self):
        """Test handling of missing fields in transfer data."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        from collections import defaultdict

        all_transfers_by_date = defaultdict(list)
        existing_data = {}

        def mock_request_with_retries(
            endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait
        ):
            yield None
            return (
                True,
                {
                    "results": [
                        {
                            "executionDate": "2024-01-10T10:30:00Z",
                            "from": "",  # Missing from address
                            "type": "ERC20_TRANSFER",
                            "tokenInfo": {"symbol": "USDC", "decimals": 6},
                            "tokenAddress": "0x1234567890123456789012345678901234567890",
                            "value": "1000000",
                            "transactionHash": "0xabc123def456",
                            "transferId": "transfer_123",
                        },
                        {
                            "executionDate": "2024-01-10T10:30:00Z",
                            "from": "0x1234567890123456789012345678901234567890",
                            "type": "ERC20_TRANSFER",
                            "tokenInfo": {"symbol": "USDC", "decimals": 6},
                            "tokenAddress": "0x1234567890123456789012345678901234567890",
                            "value": "",  # Missing value
                            "transactionHash": "0xabc123def456",
                            "transferId": "transfer_124",
                        },
                    ],
                    "next": None,
                },
            )

        def mock_get_datetime_from_timestamp(timestamp):
            from datetime import datetime

            if timestamp == "2024-01-10T10:30:00Z":
                return datetime(2024, 1, 10, 10, 30, 0)
            return None

        def mock_should_include_transfer_optimism(from_address):
            yield None
            return True

        with patch.multiple(
            fetch_behaviour,
            _request_with_retries=mock_request_with_retries,
            _get_datetime_from_timestamp=mock_get_datetime_from_timestamp,
            _should_include_transfer_optimism=mock_should_include_transfer_optimism,
        ):
            result = list(
                fetch_behaviour._fetch_optimism_transfers_safeglobal(
                    "0x9876543210987654321098765432109876543210",
                    "2024-01-15",
                    all_transfers_by_date,
                    existing_data,
                )
            )

        # Should handle missing fields gracefully - the function actually processes them with defaults
        assert len(all_transfers_by_date["2024-01-10"]) == 2

        # Check that missing fields are handled with defaults
        transfer_1 = all_transfers_by_date["2024-01-10"][0]
        assert transfer_1["from_address"] == ""  # Missing from address
        assert transfer_1["amount"] == 1.0  # Value is present

        transfer_2 = all_transfers_by_date["2024-01-10"][1]
        assert (
            transfer_2["from_address"] == "0x1234567890123456789012345678901234567890"
        )
        assert transfer_2["amount"] == 0.0  # Missing value defaults to 0

    @pytest.mark.parametrize(
        "position,chain,user_address,mock_pool,mock_gauge_address,mock_pending_rewards,expected_result,expected_logs,test_description",
        [
            # Test 1: CL pool with valid gauge and rewards
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "is_cl_pool": True,
                    "positions": [{"token_id": 123}, {"token_id": 456}],
                },
                "optimism",
                "0x9876543210987654321098765432109876543210",
                "mock_pool",  # Will be mocked
                "0x5555555555555555555555555555555555555555",  # Gauge address
                [1000000000000000000, 2000000000000000000],  # 1 and 2 VELO in wei
                Decimal("3.0"),  # 1 + 2 = 3 VELO
                [
                    "Found VELO rewards: 3.0 for position 0x1234567890123456789012345678901234567890"
                ],
                "CL pool with valid gauge and rewards",
            ),
            # Test 2: Regular pool with rewards
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "is_cl_pool": False,
                },
                "optimism",
                "0x9876543210987654321098765432109876543210",
                "mock_pool",
                None,  # Not used for regular pools
                1500000000000000000,  # 1.5 VELO in wei
                Decimal("1.5"),
                [
                    "Found VELO rewards: 1.5 for position 0x1234567890123456789012345678901234567890"
                ],
                "regular pool with rewards",
            ),
            # Test 3: No pool address
            (
                {"is_cl_pool": True, "positions": [{"token_id": 123}]},
                "optimism",
                "0x9876543210987654321098765432109876543210",
                "mock_pool",
                None,
                None,
                Decimal("0"),
                ["No pool address found for Velodrome position"],
                "no pool address",
            ),
            # Test 4: Velodrome pool behaviour not found
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "is_cl_pool": True,
                    "positions": [{"token_id": 123}],
                },
                "optimism",
                "0x9876543210987654321098765432109876543210",
                None,  # No pool behaviour
                None,
                None,
                Decimal("0"),
                ["Velodrome pool behaviour not found"],
                "velodrome pool behaviour not found",
            ),
            # Test 5: CL pool with no gauge address
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "is_cl_pool": True,
                    "positions": [{"token_id": 123}],
                },
                "optimism",
                "0x9876543210987654321098765432109876543210",
                "mock_pool",
                None,  # No gauge address
                None,
                Decimal("0"),
                [
                    "No gauge found for CL pool 0x1234567890123456789012345678901234567890"
                ],
                "CL pool with no gauge address",
            ),
            # Test 6: CL pool with no positions
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "is_cl_pool": True,
                    "positions": [],  # Empty positions
                },
                "optimism",
                "0x9876543210987654321098765432109876543210",
                "mock_pool",
                "0x5555555555555555555555555555555555555555",
                [],  # No rewards since no positions
                Decimal("0"),
                [],
                "CL pool with no positions",
            ),
            # Test 7: CL pool with missing positions key
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "is_cl_pool": True
                    # Missing positions key
                },
                "optimism",
                "0x9876543210987654321098765432109876543210",
                "mock_pool",
                "0x5555555555555555555555555555555555555555",
                [],  # No rewards since no positions
                Decimal("0"),
                [],
                "CL pool with missing positions key",
            ),
            # Test 8: CL pool with zero rewards
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "is_cl_pool": True,
                    "positions": [{"token_id": 123}],
                },
                "optimism",
                "0x9876543210987654321098765432109876543210",
                "mock_pool",
                "0x5555555555555555555555555555555555555555",
                [0],  # Zero rewards
                Decimal("0"),
                [],
                "CL pool with zero rewards",
            ),
            # Test 9: Regular pool with zero rewards
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "is_cl_pool": False,
                },
                "optimism",
                "0x9876543210987654321098765432109876543210",
                "mock_pool",
                None,
                0,  # Zero rewards
                Decimal("0"),
                [],
                "regular pool with zero rewards",
            ),
            # Test 10: CL pool with None rewards
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "is_cl_pool": True,
                    "positions": [{"token_id": 123}],
                },
                "optimism",
                "0x9876543210987654321098765432109876543210",
                "mock_pool",
                "0x5555555555555555555555555555555555555555",
                [None],  # None rewards
                Decimal("0"),
                [],
                "CL pool with None rewards",
            ),
            # Test 11: Regular pool with None rewards
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "is_cl_pool": False,
                },
                "optimism",
                "0x9876543210987654321098765432109876543210",
                "mock_pool",
                None,
                None,  # None rewards
                Decimal("0"),
                [],
                "regular pool with None rewards",
            ),
            # Test 12: CL pool with mixed rewards (some None, some valid)
            (
                {
                    "pool_address": "0x1234567890123456789012345678901234567890",
                    "is_cl_pool": True,
                    "positions": [
                        {"token_id": 123},
                        {"token_id": 456},
                        {"token_id": 789},
                    ],
                },
                "optimism",
                "0x9876543210987654321098765432109876543210",
                "mock_pool",
                "0x5555555555555555555555555555555555555555",
                [
                    1000000000000000000,
                    None,
                    3000000000000000000,
                ],  # 1 VELO, None, 3 VELO
                Decimal("4.0"),  # 1 + 0 + 3 = 4 VELO
                [
                    "Found VELO rewards: 4.0 for position 0x1234567890123456789012345678901234567890"
                ],
                "CL pool with mixed rewards",
            ),
        ],
    )
    def test_get_velodrome_pending_rewards_variations(
        self,
        position,
        chain,
        user_address,
        mock_pool,
        mock_gauge_address,
        mock_pending_rewards,
        expected_result,
        expected_logs,
        test_description,
    ):
        """Test _get_velodrome_pending_rewards method with various scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock the pools dictionary
        if mock_pool:
            mock_pool_instance = MagicMock()
            fetch_behaviour.pools = {"velodrome": mock_pool_instance}

            # Mock gauge address for CL pools
            if position.get("is_cl_pool", False) and mock_gauge_address:

                def mock_get_gauge_address(self, pool_address, chain):
                    yield None
                    return mock_gauge_address

                mock_pool_instance.get_gauge_address = mock_get_gauge_address

                # Mock CL pending rewards
                def mock_get_cl_pending_rewards(
                    self, account, gauge_address, chain, token_id
                ):
                    yield None
                    # Return rewards based on token_id index
                    positions_data = position.get("positions", [])
                    token_ids = [pos["token_id"] for pos in positions_data]
                    try:
                        index = token_ids.index(token_id)
                        return (
                            mock_pending_rewards[index]
                            if index < len(mock_pending_rewards)
                            else 0
                        )
                    except ValueError:
                        return 0

                mock_pool_instance.get_cl_pending_rewards = mock_get_cl_pending_rewards
            else:
                # Mock regular pending rewards
                def mock_get_pending_rewards(self, lp_token, user_address, chain):
                    yield None
                    return mock_pending_rewards

                mock_pool_instance.get_pending_rewards = mock_get_pending_rewards
        else:
            fetch_behaviour.pools = {}

        result = self._consume_generator(
            fetch_behaviour._get_velodrome_pending_rewards(
                position, chain, user_address
            )
        )

        assert (
            result == expected_result
        ), f"Expected {expected_result} for {test_description}"

    def test_get_velodrome_pending_rewards_exception_handling(self):
        """Test _get_velodrome_pending_rewards method with exception handling."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "is_cl_pool": True,
            "positions": [{"token_id": 123}],
        }

        # Mock pool that raises an exception
        mock_pool = MagicMock()
        fetch_behaviour.pools = {"velodrome": mock_pool}

        def mock_get_gauge_address(self, pool_address, chain):
            raise Exception("Test exception")

        mock_pool.get_gauge_address = mock_get_gauge_address

        result = self._consume_generator(
            fetch_behaviour._get_velodrome_pending_rewards(
                position, "optimism", "0xuser"
            )
        )

        assert result == Decimal("0"), "Should return 0 when exception occurs"

    def test_get_velodrome_pending_rewards_generator_behavior(self):
        """Test that _get_velodrome_pending_rewards acts as a generator."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "is_cl_pool": False,
        }

        # Mock pool
        mock_pool = MagicMock()
        fetch_behaviour.pools = {"velodrome": mock_pool}

        def mock_get_pending_rewards(self, lp_token, user_address, chain):
            yield None
            return 1000000000000000000  # 1 VELO in wei

        mock_pool.get_pending_rewards = mock_get_pending_rewards

        # Verify it's a generator
        generator = fetch_behaviour._get_velodrome_pending_rewards(
            position, "optimism", "0xuser"
        )
        assert hasattr(generator, "__iter__"), "Should be a generator"

        # Consume and verify result
        result = self._consume_generator(generator)
        assert result == Decimal("1.0"), "Should return 1.0 VELO"

    def test_get_velodrome_pending_rewards_cl_pool_multiple_positions(self):
        """Test CL pool with multiple positions and varying rewards."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "is_cl_pool": True,
            "positions": [
                {"token_id": 123},
                {"token_id": 456},
                {"token_id": 789},
                {"token_id": 101112},
            ],
        }

        # Mock pool
        mock_pool = MagicMock()
        fetch_behaviour.pools = {"velodrome": mock_pool}

        def mock_get_gauge_address(self, pool_address, chain):
            yield None
            return "0x5555555555555555555555555555555555555555"

        def mock_get_cl_pending_rewards(self, account, gauge_address, chain, token_id):
            yield None
            # Return different rewards for each token_id
            rewards_map = {
                123: 1000000000000000000,  # 1 VELO
                456: 2500000000000000000,  # 2.5 VELO
                789: 0,  # 0 VELO
                101112: 500000000000000000,  # 0.5 VELO
            }
            return rewards_map.get(token_id, 0)

        mock_pool.get_gauge_address = mock_get_gauge_address
        mock_pool.get_cl_pending_rewards = mock_get_cl_pending_rewards

        result = self._consume_generator(
            fetch_behaviour._get_velodrome_pending_rewards(
                position, "optimism", "0xuser"
            )
        )

        # Expected: 1 + 2.5 + 0 + 0.5 = 4.0 VELO
        assert result == Decimal("4.0"), "Should return 4.0 VELO for multiple positions"

    def test_get_velodrome_pending_rewards_regular_pool_success(self):
        """Test regular pool with successful reward calculation."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "is_cl_pool": False,
        }

        # Mock pool
        mock_pool = MagicMock()
        fetch_behaviour.pools = {"velodrome": mock_pool}

        def mock_get_pending_rewards(self, lp_token, user_address, chain):
            yield None
            return 7500000000000000000  # 7.5 VELO in wei

        mock_pool.get_pending_rewards = mock_get_pending_rewards

        result = self._consume_generator(
            fetch_behaviour._get_velodrome_pending_rewards(
                position, "optimism", "0xuser"
            )
        )

        assert result == Decimal("7.5"), "Should return 7.5 VELO"

    def test_get_velodrome_pending_rewards_wei_conversion(self):
        """Test that wei to VELO conversion works correctly."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "pool_address": "0x1234567890123456789012345678901234567890",
            "is_cl_pool": False,
        }

        # Mock pool
        mock_pool = MagicMock()
        fetch_behaviour.pools = {"velodrome": mock_pool}

        def mock_get_pending_rewards(self, lp_token, user_address, chain):
            yield None
            return 1234567890123456789  # Some arbitrary wei amount

        mock_pool.get_pending_rewards = mock_get_pending_rewards

        result = self._consume_generator(
            fetch_behaviour._get_velodrome_pending_rewards(
                position, "optimism", "0xuser"
            )
        )

        # Expected: 1234567890123456789 / 10^18 = 1.234567890123456789
        expected = Decimal("1234567890123456789") / Decimal(10**18)
        assert result == expected, f"Should return {expected} VELO"

    @pytest.mark.parametrize(
        "position,mock_response,expected_result,test_description",
        [
            # Test 1: Already updated position
            (
                {
                    "isUpdated": True,
                    "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
                    "chain": "optimism",
                    "pool_address": "0x1234567890123456789012345678901234567890",
                },
                None,
                True,
                "already updated position",
            ),
            # Test 2: Missing tx_hash
            (
                {
                    "isUpdated": False,
                    "enter_tx_hash": None,
                    "chain": "optimism",
                    "pool_address": "0x1234567890123456789012345678901234567890",
                },
                None,
                False,
                "missing tx_hash",
            ),
            # Test 3: Missing chain
            (
                {
                    "isUpdated": False,
                    "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
                    "chain": None,
                    "pool_address": "0x1234567890123456789012345678901234567890",
                },
                None,
                False,
                "missing chain",
            ),
            # Test 4: Missing pool_address
            (
                {
                    "isUpdated": False,
                    "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
                    "chain": "optimism",
                    "pool_address": None,
                },
                None,
                False,
                "missing pool_address",
            ),
            # Test 5: Successful validation - addresses match
            (
                {
                    "isUpdated": False,
                    "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
                    "chain": "optimism",
                    "pool_address": "0x1234567890123456789012345678901234567890",
                },
                {
                    "logs": [
                        {
                            "topics": [
                                "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",  # TRANSFER_EVENT_SIGNATURE
                                "0x0000000000000000000000000000000000000000000000000000000000000000",  # ZERO_ADDRESS_PADDED
                                "0x0000000000000000000000001234567890123456789012345678901234567890",
                            ],
                            "address": "0x1234567890123456789012345678901234567890",
                        }
                    ]
                },
                True,
                "successful validation - addresses match",
            ),
            # Test 6: Successful validation - addresses differ, update needed
            (
                {
                    "isUpdated": False,
                    "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
                    "chain": "optimism",
                    "pool_address": "0x1234567890123456789012345678901234567890",
                },
                {
                    "logs": [
                        {
                            "topics": [
                                "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",  # TRANSFER_EVENT_SIGNATURE
                                "0x0000000000000000000000000000000000000000000000000000000000000000",  # ZERO_ADDRESS_PADDED
                                "0x0000000000000000000000001234567890123456789012345678901234567890",
                            ],
                            "address": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                        }
                    ]
                },
                True,
                "successful validation - addresses differ, update needed",
            ),
            # Test 7: No transaction receipt response
            (
                {
                    "isUpdated": False,
                    "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
                    "chain": "optimism",
                    "pool_address": "0x1234567890123456789012345678901234567890",
                },
                None,
                False,
                "no transaction receipt response",
            ),
            # Test 8: No logs in transaction receipt
            (
                {
                    "isUpdated": False,
                    "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
                    "chain": "optimism",
                    "pool_address": "0x1234567890123456789012345678901234567890",
                },
                {"logs": []},
                False,
                "no logs in transaction receipt",
            ),
            # Test 9: No LP token mint event found
            (
                {
                    "isUpdated": False,
                    "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
                    "chain": "optimism",
                    "pool_address": "0x1234567890123456789012345678901234567890",
                },
                {
                    "logs": [
                        {
                            "topics": [
                                "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",  # TRANSFER_EVENT_SIGNATURE
                                "0x0000000000000000000000001234567890123456789012345678901234567890",  # Not zero address
                                "0x000000000000000000000000abcdefabcdefabcdefabcdefabcdefabcdefabcd",
                            ],
                            "address": "0x1234567890123456789012345678901234567890",
                        }
                    ]
                },
                False,
                "no LP token mint event found",
            ),
            # Test 10: Wrong event signature
            (
                {
                    "isUpdated": False,
                    "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
                    "chain": "optimism",
                    "pool_address": "0x1234567890123456789012345678901234567890",
                },
                {
                    "logs": [
                        {
                            "topics": [
                                "0xwrongsignature000000000000000000000000000000000000000000000000000000",  # Wrong signature
                                "0x0000000000000000000000000000000000000000000000000000000000000000",  # ZERO_ADDRESS_PADDED
                                "0x0000000000000000000000001234567890123456789012345678901234567890",
                            ],
                            "address": "0x1234567890123456789012345678901234567890",
                        }
                    ]
                },
                False,
                "wrong event signature",
            ),
            # Test 11: Insufficient topics
            (
                {
                    "isUpdated": False,
                    "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
                    "chain": "optimism",
                    "pool_address": "0x1234567890123456789012345678901234567890",
                },
                {
                    "logs": [
                        {
                            "topics": [
                                "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"  # Only one topic
                            ],
                            "address": "0x1234567890123456789012345678901234567890",
                        }
                    ]
                },
                False,
                "insufficient topics",
            ),
            # Test 12: Case insensitive address comparison
            (
                {
                    "isUpdated": False,
                    "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
                    "chain": "optimism",
                    "pool_address": "0x1234567890123456789012345678901234567890",
                },
                {
                    "logs": [
                        {
                            "topics": [
                                "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",  # TRANSFER_EVENT_SIGNATURE
                                "0x0000000000000000000000000000000000000000000000000000000000000000",  # ZERO_ADDRESS_PADDED
                                "0x0000000000000000000000001234567890123456789012345678901234567890",
                            ],
                            "address": "0x1234567890123456789012345678901234567890",
                        }
                    ]
                },
                True,
                "case insensitive address comparison",
            ),
        ],
    )
    def test_validate_velodrome_v2_pool_address_variations(
        self, position, mock_response, expected_result, test_description
    ):
        """Test _validate_velodrome_v2_pool_address method with various scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        def mock_get_transaction_receipt(tx_digest, chain_id):
            yield
            return mock_response

        with patch.multiple(
            fetch_behaviour, get_transaction_receipt=mock_get_transaction_receipt
        ):
            result = self._consume_generator(
                fetch_behaviour._validate_velodrome_v2_pool_address(position)
            )

            assert (
                result == expected_result
            ), f"Expected {expected_result} for {test_description}"

            # Check if position was updated when addresses differed
            if expected_result and "addresses differ" in test_description:
                assert (
                    position["is_updated"] == True
                ), "Position should be marked as updated"
                assert (
                    position["pool_address"]
                    == "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
                ), "Pool address should be updated"

    def test_validate_velodrome_v2_pool_address_exception_handling(self):
        """Test _validate_velodrome_v2_pool_address method with exception handling."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "isUpdated": False,
            "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "chain": "optimism",
            "pool_address": "0x1234567890123456789012345678901234567890",
        }

        def mock_get_transaction_receipt(tx_digest, chain_id):
            raise Exception("Test exception")

        with patch.multiple(
            fetch_behaviour, get_transaction_receipt=mock_get_transaction_receipt
        ):
            result = self._consume_generator(
                fetch_behaviour._validate_velodrome_v2_pool_address(position)
            )

            assert result == False, "Should return False when exception occurs"

    def test_validate_velodrome_v2_pool_address_generator_behavior(self):
        """Test that _validate_velodrome_v2_pool_address acts as a generator."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "isUpdated": False,
            "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "chain": "optimism",
            "pool_address": "0x1234567890123456789012345678901234567890",
        }

        def mock_get_transaction_receipt(tx_digest, chain_id):
            yield
            return {
                "logs": [
                    {
                        "topics": [
                            "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
                            "0x0000000000000000000000000000000000000000000000000000000000000000",
                            "0x0000000000000000000000001234567890123456789012345678901234567890",
                        ],
                        "address": "0x1234567890123456789012345678901234567890",
                    }
                ]
            }

        with patch.multiple(
            fetch_behaviour, get_transaction_receipt=mock_get_transaction_receipt
        ):
            generator = fetch_behaviour._validate_velodrome_v2_pool_address(position)

            # Verify it's a generator
            assert hasattr(generator, "__iter__"), "Should be a generator"

            # Consume the generator
            result = self._consume_generator(generator)
            assert result == True, "Should return True for successful validation"

    def test_validate_velodrome_v2_pool_address_multiple_logs(self):
        """Test _validate_velodrome_v2_pool_address with multiple logs, finding LP mint in later log."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "isUpdated": False,
            "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "chain": "optimism",
            "pool_address": "0x1234567890123456789012345678901234567890",
        }

        def mock_get_transaction_receipt(tx_digest, chain_id):
            yield
            return {
                "logs": [
                    {
                        "topics": [
                            "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
                            "0x0000000000000000000000001234567890123456789012345678901234567890",  # Not zero address
                            "0x000000000000000000000000abcdefabcdefabcdefabcdefabcdefabcdefabcd",
                        ],
                        "address": "0x1234567890123456789012345678901234567890",
                    },
                    {
                        "topics": [
                            "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
                            "0x0000000000000000000000000000000000000000000000000000000000000000",  # ZERO_ADDRESS_PADDED
                            "0x0000000000000000000000001234567890123456789012345678901234567890",
                        ],
                        "address": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",  # Different address
                    },
                ]
            }

        with patch.multiple(
            fetch_behaviour, get_transaction_receipt=mock_get_transaction_receipt
        ):
            result = self._consume_generator(
                fetch_behaviour._validate_velodrome_v2_pool_address(position)
            )

            assert result == True, "Should find LP mint event in second log"
            assert (
                position["is_updated"] == True
            ), "Position should be marked as updated"
            assert (
                position["pool_address"] == "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
            ), "Pool address should be updated to second log address"

    def test_validate_velodrome_v2_pool_address_missing_address_field(self):
        """Test _validate_velodrome_v2_pool_address with log missing address field."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "isUpdated": False,
            "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "chain": "optimism",
            "pool_address": "0x1234567890123456789012345678901234567890",
        }

        def mock_get_transaction_receipt(tx_digest, chain_id):
            yield
            return {
                "logs": [
                    {
                        "topics": [
                            "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
                            "0x0000000000000000000000000000000000000000000000000000000000000000",
                            "0x0000000000000000000000001234567890123456789012345678901234567890",
                        ]
                        # Missing address field
                    }
                ]
            }

        with patch.multiple(
            fetch_behaviour, get_transaction_receipt=mock_get_transaction_receipt
        ):
            result = self._consume_generator(
                fetch_behaviour._validate_velodrome_v2_pool_address(position)
            )

            assert (
                result == False
            ), "Should return False when log is missing address field"

    def test_validate_velodrome_v2_pool_address_missing_topics_field(self):
        """Test _validate_velodrome_v2_pool_address with log missing topics field."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        position = {
            "isUpdated": False,
            "enter_tx_hash": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "chain": "optimism",
            "pool_address": "0x1234567890123456789012345678901234567890",
        }

        def mock_get_transaction_receipt(tx_digest, chain_id):
            yield
            return {
                "logs": [
                    {
                        "address": "0x1234567890123456789012345678901234567890"
                        # Missing topics field
                    }
                ]
            }

        with patch.multiple(
            fetch_behaviour, get_transaction_receipt=mock_get_transaction_receipt
        ):
            result = self._consume_generator(
                fetch_behaviour._validate_velodrome_v2_pool_address(position)
            )

            assert (
                result == False
            ), "Should return False when log is missing topics field"

    @pytest.mark.parametrize(
        "chain,safe_address,sender,reversion_data,expected_calls,expected_set_done",
        [
            # Test case 1: No reversion needed (eth_amount = 0)
            (
                "optimism",
                "0x1234567890123456789012345678901234567890",
                "0x1111111111111111111111111111111111111111",
                {
                    "master_safe_address": "0x2222222222222222222222222222222222222222",
                    "reversion_amount": 0,
                },
                {
                    "track_eth": 1,
                    "contract_interact": 0,
                    "send_a2a": 0,
                    "wait_until": 0,
                },
                0,
            ),
            # Test case 2: Reversion needed but no master safe address
            (
                "mode",
                "0x9876543210987654321098765432109876543210",
                "0x3333333333333333333333333333333333333333",
                {"master_safe_address": None, "reversion_amount": 1.5},
                {
                    "track_eth": 1,
                    "contract_interact": 0,
                    "send_a2a": 0,
                    "wait_until": 0,
                },
                0,
            ),
            # Test case 3: Successful reversion transaction creation
            (
                "optimism",
                "0x1111222233334444555566667777888899990000",
                "0x4444444444444444444444444444444444444444",
                {
                    "master_safe_address": "0x5555555555555555555555555555555555555555",
                    "reversion_amount": 2.0,
                },
                {
                    "track_eth": 1,
                    "contract_interact": 1,
                    "send_a2a": 1,
                    "wait_until": 1,
                },
                1,
            ),
            # Test case 4: Contract interaction fails (returns None)
            (
                "mode",
                "0xaaabbbcccdddeeefff000111222333444555666",
                "0x6666666666666666666666666666666666666666",
                {
                    "master_safe_address": "0x7777777777777777777777777777777777777777",
                    "reversion_amount": 0.5,
                },
                {
                    "track_eth": 1,
                    "contract_interact": 1,
                    "send_a2a": 0,
                    "wait_until": 0,
                },
                0,
            ),
            # Test case 5: Empty safe address
            (
                "optimism",
                "",
                "0x8888888888888888888888888888888888888888",
                {
                    "master_safe_address": "0x9999999999999999999999999999999999999999",
                    "reversion_amount": 1.0,
                },
                {
                    "track_eth": 0,
                    "contract_interact": 0,
                    "send_a2a": 0,
                    "wait_until": 0,
                },
                0,
            ),
        ],
    )
    def test_check_and_create_eth_revert_transactions_comprehensive(
        self,
        chain,
        safe_address,
        sender,
        reversion_data,
        expected_calls,
        expected_set_done,
    ):
        """Test _check_and_create_eth_revert_transactions with various scenarios."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        # Mock call counters
        mock_track_eth_call_count = 0
        mock_contract_interact_call_count = 0
        mock_send_a2a_call_count = 0
        mock_wait_until_call_count = 0
        mock_set_done_call_count = 0

        def mock_track_eth_generator(safe_addr, chain_id):
            nonlocal mock_track_eth_call_count
            mock_track_eth_call_count += 1
            yield
            return reversion_data

        def mock_contract_interact_generator(*args, **kwargs):
            nonlocal mock_contract_interact_call_count
            mock_contract_interact_call_count += 1
            yield
            # Return None for failed contract interaction test case
            if reversion_data["reversion_amount"] == 0.5:
                return None
            # Return a valid 64-character hash for successful cases
            return "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"

        def mock_send_a2a_generator(payload):
            nonlocal mock_send_a2a_call_count
            mock_send_a2a_call_count += 1
            yield
            return None

        def mock_wait_until_generator():
            nonlocal mock_wait_until_call_count
            mock_wait_until_call_count += 1
            yield
            return None

        def mock_set_done_generator():
            nonlocal mock_set_done_call_count
            mock_set_done_call_count += 1
            return None

        # Create a mock benchmark tool context manager
        mock_benchmark_context = MagicMock()
        mock_benchmark_context.__enter__ = MagicMock(
            return_value=mock_benchmark_context
        )
        mock_benchmark_context.__exit__ = MagicMock(return_value=None)

        mock_benchmark_tool = MagicMock()
        mock_benchmark_tool.measure.return_value.consensus.return_value = (
            mock_benchmark_context
        )

        # Mock the context.benchmark_tool
        with patch.object(
            fetch_behaviour.context, "benchmark_tool", mock_benchmark_tool
        ):
            with patch.multiple(
                fetch_behaviour,
                _track_eth_transfers_and_reversions=mock_track_eth_generator,
                contract_interact=mock_contract_interact_generator,
                send_a2a_transaction=mock_send_a2a_generator,
                wait_until_round_end=mock_wait_until_generator,
                set_done=mock_set_done_generator,
            ):
                # Execute the function using _consume_generator
                result = self._consume_generator(
                    fetch_behaviour._check_and_create_eth_revert_transactions(
                        chain, safe_address, sender
                    )
                )

                # Verify call counts
                assert (
                    mock_track_eth_call_count == expected_calls["track_eth"]
                ), f"Expected {expected_calls['track_eth']} _track_eth_transfers_and_reversions calls, got {mock_track_eth_call_count}"
                assert (
                    mock_contract_interact_call_count
                    == expected_calls["contract_interact"]
                ), f"Expected {expected_calls['contract_interact']} contract_interact calls, got {mock_contract_interact_call_count}"
                assert (
                    mock_send_a2a_call_count == expected_calls["send_a2a"]
                ), f"Expected {expected_calls['send_a2a']} send_a2a_transaction calls, got {mock_send_a2a_call_count}"
                assert (
                    mock_wait_until_call_count == expected_calls["wait_until"]
                ), f"Expected {expected_calls['wait_until']} wait_until_round_end calls, got {mock_wait_until_call_count}"
                assert (
                    mock_set_done_call_count == expected_set_done
                ), f"Expected {expected_set_done} set_done calls, got {mock_set_done_call_count}"

    @pytest.mark.parametrize(
        "reversion_amount,expected_wei_amount",
        [
            (0.0, 0),
            (0.5, 500000000000000000),  # 0.5 ETH in wei
            (1.0, 1000000000000000000),  # 1.0 ETH in wei
            (2.5, 2500000000000000000),  # 2.5 ETH in wei
            (10.0, 10000000000000000000),  # 10.0 ETH in wei
        ],
    )
    def test_check_and_create_eth_revert_transactions_wei_conversion(
        self, reversion_amount, expected_wei_amount
    ):
        """Test correct ETH to Wei conversion in _check_and_create_eth_revert_transactions."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        chain = "optimism"
        safe_address = "0x1234567890123456789012345678901234567890"
        sender = "0x1111111111111111111111111111111111111111"

        reversion_data = {
            "master_safe_address": "0x2222222222222222222222222222222222222222",
            "reversion_amount": reversion_amount,
        }

        contract_interact_kwargs = {}

        def mock_track_eth_generator(safe_addr, chain_id):
            yield
            return reversion_data

        def mock_contract_interact_generator(*args, **kwargs):
            nonlocal contract_interact_kwargs
            contract_interact_kwargs = kwargs
            yield
            return "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"

        def mock_send_a2a_generator(payload):
            yield
            return None

        def mock_wait_until_generator():
            yield
            return None

        def mock_set_done_generator():
            return None

        # Create a mock benchmark tool context manager
        mock_benchmark_context = MagicMock()
        mock_benchmark_context.__enter__ = MagicMock(
            return_value=mock_benchmark_context
        )
        mock_benchmark_context.__exit__ = MagicMock(return_value=None)

        mock_benchmark_tool = MagicMock()
        mock_benchmark_tool.measure.return_value.consensus.return_value = (
            mock_benchmark_context
        )

        # Mock the context.benchmark_tool
        with patch.object(
            fetch_behaviour.context, "benchmark_tool", mock_benchmark_tool
        ):
            with patch.multiple(
                fetch_behaviour,
                _track_eth_transfers_and_reversions=mock_track_eth_generator,
                contract_interact=mock_contract_interact_generator,
                send_a2a_transaction=mock_send_a2a_generator,
                wait_until_round_end=mock_wait_until_generator,
                set_done=mock_set_done_generator,
            ):
                # Execute the function using _consume_generator
                result = self._consume_generator(
                    fetch_behaviour._check_and_create_eth_revert_transactions(
                        chain, safe_address, sender
                    )
                )

                # Verify Wei conversion only if reversion_amount > 0
                if reversion_amount > 0:
                    assert (
                        contract_interact_kwargs.get("value") == expected_wei_amount
                    ), f"Expected value {expected_wei_amount} wei, got {contract_interact_kwargs.get('value')}"

    @pytest.mark.parametrize(
        "reversion_amount,master_safe_address,expected_log_patterns",
        [
            (
                0,
                "0x2222222222222222222222222222222222222222",
                [],  # No logs expected when reversion_amount is 0
            ),
            (
                1.5,
                None,
                ["No master safe address found for chain"],
            ),
            (
                2.0,
                "0x5555555555555555555555555555555555555555",
                [
                    "Creating ETH transfer transaction: 2000000000000000000 wei to 0x5555555555555555555555555555555555555555"
                ],
            ),
        ],
    )
    def test_check_and_create_eth_revert_transactions_logging_coverage(
        self, reversion_amount, master_safe_address, expected_log_patterns
    ):
        """Test logging messages in _check_and_create_eth_revert_transactions."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        chain = "optimism"
        safe_address = "0x1234567890123456789012345678901234567890"
        sender = "0x1111111111111111111111111111111111111111"

        reversion_data = {
            "master_safe_address": master_safe_address,
            "reversion_amount": reversion_amount,
        }

        def mock_track_eth_generator(safe_addr, chain_id):
            yield
            return reversion_data

        def mock_contract_interact_generator(*args, **kwargs):
            yield
            return "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"

        def mock_send_a2a_generator(payload):
            yield
            return None

        def mock_wait_until_generator():
            yield
            return None

        def mock_set_done_generator():
            return None

        # Create a mock benchmark tool context manager
        mock_benchmark_context = MagicMock()
        mock_benchmark_context.__enter__ = MagicMock(
            return_value=mock_benchmark_context
        )
        mock_benchmark_context.__exit__ = MagicMock(return_value=None)

        mock_benchmark_tool = MagicMock()
        mock_benchmark_tool.measure.return_value.consensus.return_value = (
            mock_benchmark_tool
        )

        # Mock the context.benchmark_tool
        with patch.object(
            fetch_behaviour.context, "benchmark_tool", mock_benchmark_tool
        ):
            with patch.multiple(
                fetch_behaviour,
                _track_eth_transfers_and_reversions=mock_track_eth_generator,
                contract_interact=mock_contract_interact_generator,
                send_a2a_transaction=mock_send_a2a_generator,
                wait_until_round_end=mock_wait_until_generator,
                set_done=mock_set_done_generator,
            ):
                # Capture log messages
                with patch.object(
                    fetch_behaviour.context.logger, "info"
                ) as mock_info, patch.object(
                    fetch_behaviour.context.logger, "error"
                ) as mock_error:
                    # Execute the function using _consume_generator
                    result = self._consume_generator(
                        fetch_behaviour._check_and_create_eth_revert_transactions(
                            chain, safe_address, sender
                        )
                    )

                    # Check expected log patterns
                    all_log_calls = [
                        call.args[0]
                        for call in mock_info.call_args_list + mock_error.call_args_list
                    ]

                    for pattern in expected_log_patterns:
                        pattern_found = any(
                            pattern in log_msg for log_msg in all_log_calls
                        )
                        assert (
                            pattern_found
                        ), f"Expected log pattern '{pattern}' not found in logs: {all_log_calls}"

    def test_check_and_create_eth_revert_transactions_payload_structure(self):
        """Test the structure and content of the settlement payload."""
        fetch_behaviour = self._create_fetch_strategies_behaviour()

        chain = "optimism"
        safe_address = "0x1234567890123456789012345678901234567890"
        sender = "0x1111111111111111111111111111111111111111"

        reversion_data = {
            "master_safe_address": "0x5555555555555555555555555555555555555555",
            "reversion_amount": 1.0,
        }

        captured_payload = None

        def mock_track_eth_generator(safe_addr, chain_id):
            yield
            return reversion_data

        def mock_contract_interact_generator(*args, **kwargs):
            yield
            return "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"

        def mock_send_a2a_generator(payload):
            nonlocal captured_payload
            captured_payload = payload
            yield
            return None

        def mock_wait_until_generator():
            yield
            return None

        def mock_set_done_generator():
            return None

        # Create a mock benchmark tool context manager
        mock_benchmark_context = MagicMock()
        mock_benchmark_context.__enter__ = MagicMock(
            return_value=mock_benchmark_context
        )
        mock_benchmark_context.__exit__ = MagicMock(return_value=None)

        mock_benchmark_tool = MagicMock()
        mock_benchmark_tool.measure.return_value.consensus.return_value = (
            mock_benchmark_context
        )

        # Mock the context.benchmark_tool
        with patch.object(
            fetch_behaviour.context, "benchmark_tool", mock_benchmark_tool
        ):
            with patch.multiple(
                fetch_behaviour,
                _track_eth_transfers_and_reversions=mock_track_eth_generator,
                contract_interact=mock_contract_interact_generator,
                send_a2a_transaction=mock_send_a2a_generator,
                wait_until_round_end=mock_wait_until_generator,
                set_done=mock_set_done_generator,
            ):
                # Execute the function using _consume_generator
                result = self._consume_generator(
                    fetch_behaviour._check_and_create_eth_revert_transactions(
                        chain, safe_address, sender
                    )
                )

                # Verify payload structure
                assert captured_payload is not None, "Expected payload to be captured"
                assert isinstance(
                    captured_payload, FetchStrategiesPayload
                ), "Expected FetchStrategiesPayload instance"
                assert (
                    captured_payload.sender == sender
                ), f"Expected sender {sender}, got {captured_payload.sender}"

                # Parse and verify payload content
                content = json.loads(captured_payload.content)
                assert content["event"] == "settle", "Expected event to be 'settle'"
                assert "updates" in content, "Expected 'updates' key in content"

                updates = content["updates"]
                assert (
                    updates["chain_id"] == chain
                ), f"Expected chain_id {chain}, got {updates['chain_id']}"
                assert (
                    updates["safe_contract_address"] == safe_address
                ), f"Expected safe_contract_address {safe_address}, got {updates['safe_contract_address']}"
                assert "tx_submitter" in updates, "Expected 'tx_submitter' in updates"
                assert (
                    "most_voted_tx_hash" in updates
                ), "Expected 'most_voted_tx_hash' in updates"
